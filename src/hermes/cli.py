"""Command line entry for one-shot generation."""

import argparse
import json
from pathlib import Path

from hermes import ask, systems
from hermes.generate import Context, Example


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hermes")
    subcommands = parser.add_subparsers(dest="command", required=True)
    generate_command = subcommands.add_parser(
        "generate", help="Write one SPARQL query for a question and an ontology."
    )
    generate_command.add_argument("--ontology", required=True, help="Turtle file.")
    generate_command.add_argument("--question", required=True)
    generate_command.add_argument(
        "--context", help="Text file of extra context for the question."
    )
    generate_command.add_argument(
        "--examples",
        help="JSONL file of example objects with nl and sparql fields.",
    )
    generate_command.add_argument(
        "--system",
        default="training-free",
        help="Registered system to run (default: training-free).",
    )
    subcommands.add_parser("systems", help="List the registered systems.")
    args = parser.parse_args(argv)
    if args.command == "systems":
        print("\n".join(systems.available()))
        return 0
    result = ask(
        args.question,
        args.ontology,
        context=_load_context(args.context, args.examples),
        system=args.system,
    )
    print(json.dumps(result, indent=2))
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
