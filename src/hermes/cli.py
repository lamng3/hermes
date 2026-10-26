"""Command line: ask, plan, generate, systems, data, fewshot, run, sweep, report, serve."""

import argparse
import sys
import json
from pathlib import Path

from hermes import ask, benchmarks, systems
from hermes.context import Context, Example
from hermes.graph.knowledge import KnowledgeGraph


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hermes")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Write one SPARQL query for a question.")
    source = generate.add_mutually_exclusive_group(required=True)
    source.add_argument("--ontology", help="RDF file (schema, and data if any).")
    source.add_argument("--graph", help="Graph name from data/graphs.yaml.")
    source.add_argument("--endpoint", help="SPARQL endpoint URL.")
    generate.add_argument("--question", required=True)
    generate.add_argument("--context", help="Text file of extra context for the question.")
    generate.add_argument("--examples", help="JSONL file of objects with nl and sparql fields.")
    generate.add_argument("--system", default="training-free",
                          help="Registered system to run (default: training-free).")

    ask_ = commands.add_parser("ask", help="Answer a question over a knowledge graph.")
    graph_of = ask_.add_mutually_exclusive_group(required=True)
    graph_of.add_argument("--graph", help="Graph name from data/graphs.yaml.")
    graph_of.add_argument("--files", nargs="+", help="RDF files: data, and the ontology.")
    graph_of.add_argument("--endpoint", help="SPARQL endpoint URL.")
    ask_.add_argument("--ontology", nargs="*", default=[], help="Ontology files, with --files or --endpoint.")
    ask_.add_argument("question")
    ask_.add_argument("--calls", type=int, default=3, help="Model calls allowed (default 3).")
    ask_.add_argument("--tokens", type=int, help="Model tokens allowed.")
    ask_.add_argument("--usd", type=float, help="Dollars allowed, for priced models.")
    ask_.add_argument("--rows", type=int, help="Graph rows the plan may fetch.")
    ask_.add_argument("--executor", choices=("plan", "engine"), default="plan")
    ask_.add_argument("--rows-shown", type=int, default=20)
    ask_.add_argument("--quiet", action="store_true", help="Print only the answer.")

    plan = commands.add_parser("plan", help="Translate a SPARQL query to relational algebra, "
                                            "optimize it with the graph's statistics, and run it.")
    where = plan.add_mutually_exclusive_group(required=True)
    where.add_argument("--graph", help="Graph name from data/graphs.yaml.")
    where.add_argument("--files", nargs="+", help="RDF files: ontology and data.")
    plan.add_argument("--query", help="SPARQL text (or use --file).")
    plan.add_argument("--file", help="A file holding the SPARQL query.")
    plan.add_argument("--rows", type=int, default=20, help="Result rows to print.")

    commands.add_parser("systems", help="List the registered systems.")

    data = commands.add_parser("data", help="Benchmarks: list, fetch, verify.")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_commands.add_parser("list", help="Downloaded and known benchmarks.")
    fetch = data_commands.add_parser("fetch", help="Download and normalize benchmarks.")
    fetch.add_argument("names", nargs="+")
    verify = data_commands.add_parser("verify", help="Check gold queries against the local graph.")
    verify.add_argument("name")

    run = commands.add_parser("run", help="Run an experiment config.")
    run.add_argument("configs", nargs="+", help="YAML configs under configs/.")
    run.add_argument("--limit", type=int, help="Only the first N items.")
    run.add_argument("--resume", help="Run id to resume.")
    run.add_argument("--results", default="results")

    sweep = commands.add_parser("sweep", help="Run a grid of benchmarks, systems, and models.")
    sweep.add_argument("file", help="Sweep YAML under configs/sweeps/.")
    sweep.add_argument("--only", help="Run only configs whose name contains this text.")
    sweep.add_argument("--list", action="store_true", help="List the runs and exit.")
    sweep.add_argument("--limit", type=int, help="Only the first N items of each run.")
    sweep.add_argument("--fresh", action="store_true",
                       help="Start new runs with new model calls, even where finished runs exist; "
                            "old runs are kept.")
    sweep.add_argument("--results", default="results")

    report = commands.add_parser("report", help="Leaderboard of runs and official peers.")
    report.add_argument("--benchmark")
    report.add_argument("--results", default="results")
    report.add_argument("--write", action="store_true", help="Also write results/LEADERBOARD.md.")

    serve = commands.add_parser("serve", help="Serve a system behind the Text2SPARQL API.")
    serve.add_argument("--config", help="Experiment config to take the system and LLM from.")
    serve.add_argument("--system", default="training-free")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.command == "ask":
        return _ask(args)
    if args.command == "plan":
        return _plan(args)
    if args.command == "systems":
        print("\n".join(systems.available()))
    elif args.command == "generate":
        if args.graph:
            kg = benchmarks.graph(args.graph)
        elif args.endpoint:
            kg = KnowledgeGraph.from_endpoint(args.endpoint)
        else:
            kg = args.ontology
        result = ask(args.question, kg, _load_context(args.context, args.examples), args.system)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "data":
        return _data(args)
    elif args.command == "run":
        from hermes.experiments import Progress, Run, load_config

        for path in args.configs:
            config = load_config(path)
            if args.limit:
                config["limit"] = args.limit
            Run(config, args.results, args.resume, observers=[Progress()]).execute()
    elif args.command == "sweep":
        from hermes.experiments import (
            Progress, Run, expand_sweep, latest_finished, latest_unfinished,
        )

        configs = [c for c in expand_sweep(args.file) if not args.only or args.only in c["name"]]
        for config in configs:
            if args.limit:
                config["limit"] = args.limit
            done = None if args.fresh else latest_finished(args.results, config)
            if done:
                print(f"{config['benchmark']}/{config['name']} (done: {done})")
                continue
            resume = None if args.fresh else latest_unfinished(args.results, config)
            if args.fresh:
                config["llm"]["refresh_cache"] = True
            print(f"{config['benchmark']}/{config['name']}"
                  + (f" (resuming {resume})" if resume else ""))
            if args.list:
                continue
            try:
                Run(config, args.results, resume, observers=[Progress()]).execute()
            except Exception as exc:  # one failed config must not stop the sweep
                print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    elif args.command == "report":
        from hermes.experiments import leaderboard

        table = leaderboard(args.results, args.benchmark)
        print(table)
        if args.write:
            Path(args.results, "LEADERBOARD.md").write_text(
                "# Leaderboard\n\nGenerated by `hermes report --write`.\n\n" + table,
                encoding="utf-8",
            )
    elif args.command == "serve":
        from hermes import llm as llms
        from hermes.experiments import _takes_llm, load_config
        from hermes.serve import serve as start

        config = load_config(args.config) if args.config else {"system": args.system,
                                                                "params": {}, "llm": {}}
        params = dict(config["params"])
        if _takes_llm(config["system"]):
            params["llm"] = llms.create(**config["llm"])
        start(systems.create(config["system"], **params), args.host, args.port)
    return 0


def _ask(args):
    from hermes import Budget, Hermes, planner
    from hermes import llm as llms
    from hermes.pipeline.events import Printer
    from hermes.pipeline.fewshot import store_path

    if args.graph:
        kg, name = benchmarks.graph(args.graph), args.graph
    elif args.endpoint:
        kg, name = KnowledgeGraph.from_endpoint(args.endpoint, schema=args.ontology), args.endpoint
    else:
        kg = KnowledgeGraph.from_files(*args.ontology, *args.files, schema=args.ontology or ())
        name = kg.name
    budget = Budget(llm_calls=args.calls, tokens=args.tokens, usd=args.usd, rows=args.rows)
    hermes = Hermes(kg, llm=llms.from_env(), budget=budget, fewshots=store_path(name),
                    executor=args.executor, observers=[] if args.quiet else [Printer()])
    answer = hermes.ask(args.question)
    print(answer.sparql or "(no query)")
    if answer.plan is not None and not args.quiet:
        query = answer.plan.query
        print("\nPlan\n" + planner.tree(query.root, query.namespaces))
    total = answer.spend.total
    print(f"\nSpent: {total.llm_calls:.0f} model calls, {total.tokens:,.0f} tokens"
          + (f", ${total.usd:.4f}" if total.usd else "")
          + f", {total.queries:.0f} graph queries, {total.rows:,.0f} rows · {answer.seconds:.1f} s")
    if answer.error:
        print(f"No answer: {answer.error}")
        return 1
    if answer.boolean is not None:
        print(f"Answer: {answer.boolean}")
        return 0
    print(f"{len(answer.rows)} rows")
    names = answer.plan.query.namespaces
    print("  ".join("?" + str(v) for v in answer.variables))
    for row in answer.rows[: args.rows_shown]:
        print("  ".join(planner.expression(row[v], names) if row.get(v) is not None else "-"
                        for v in answer.variables))
    return 0


def _plan(args):
    from hermes import planner

    sparql = args.query or Path(args.file).read_text(encoding="utf-8")
    kg = benchmarks.graph(args.graph) if args.graph else KnowledgeGraph.from_files(*args.files)
    optimized = planner.plan(sparql, kg)
    written, query, estimate = optimized.written, optimized.query, optimized.estimator
    run = planner.PlanExecutor(kg)
    before, after = run(written), run(query)
    engine = planner.EngineExecutor(kg)(written)

    def notes(trace):
        return lambda node: (f"   est {estimate(node).rows:,.1f}"
                             + (f" · got {trace.rows[id(node)]:,}" if id(node) in trace.rows else ""))

    print("As written\n" + planner.tree(written.root, written.namespaces, notes(before.trace)))
    print("\nRewrites")
    print("\n".join(f"- {r.text}" for r in optimized.rewrites) or "- none")
    print("\nOptimized\n" + planner.tree(query.root, query.namespaces, notes(after.trace)))
    print("\n" + planner.formula(query.root, query.namespaces))
    print(f"\n{'':22}{'as written':>12}{'optimized':>12}")
    for name, value in (("estimated C_out", lambda q, t: f"{estimate(q.root).cost:,.0f}"),
                        ("store queries", lambda q, t: f"{t.queries:,}"),
                        ("rows fetched", lambda q, t: f"{t.fetched:,}"),
                        ("intermediate rows", lambda q, t: f"{t.intermediate:,}"),
                        ("milliseconds", lambda q, t: f"{t.seconds * 1000:,.1f}")):
        print(f"{name:22}{value(written, before.trace):>12}{value(query, after.trace):>12}")
    same = planner.same_answers(after, engine)
    print(f"\nSame answers as the engine: {'yes' if same else 'NO'}")
    if after.form == "ASK":
        print(f"Answer: {after.boolean}")
        return 0 if same else 1
    print(f"{len(after.rows)} rows" + (f" (first {args.rows})" if len(after.rows) > args.rows else ""))
    print("  ".join("?" + str(v) for v in after.variables))
    for row in after.rows[: args.rows]:
        print("  ".join(planner.expression(row[v], query.namespaces) if row.get(v) is not None
                        else "-" for v in after.variables))
    return 0 if same else 1


def _data(args):
    from hermes.benchmarks import sources

    if args.data_command == "list":
        have = set(benchmarks.available())
        for name, source in sources.SOURCES.items():
            mark = "x" if name in have else " "
            print(f"[{mark}] {name:10} {source['title']}  (graph: {source['graph']})")
    elif args.data_command == "fetch":
        for name in args.names:
            print(f"{name}: {sources.fetch(name)}")
    elif args.data_command == "verify":
        from hermes.experiments import reproduces, verify

        report = verify(args.name)
        for item, value in report["items"].items():
            if not reproduces(value):
                print(f"{item}: F1 {value.get('set_F', 0):.2f} {value.get('error', '')}")
        print(f"{report['agree']}/{report['questions']} gold queries reproduce the gold answers.")
    return 0


def _load_context(context_path, examples_path):
    text = ""
    if context_path:
        text = Path(context_path).read_text(encoding="utf-8")
    examples = []
    if examples_path:
        for line in Path(examples_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                Example(
                    nl=row.get("nl") or row.get("nl_text") or "",
                    sparql=row.get("sparql") or row.get("sparql_query") or "",
                )
            )
    return Context(text=text, examples=examples)
