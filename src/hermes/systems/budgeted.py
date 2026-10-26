"""The budgeted planner: parse, bind, check, price, select, within a budget.

NL-to-SPARQL compiled like a database compiles SQL, except that the meaning
is not fixed by the input, so the planner must choose which query to send:

1. parse: the model reads the question into a goal and a plan of algebra
   actions whose unresolved terms are phrases in brackets (hermes.binding);
2. bind: each phrase gets ranked properties, classes or entities, and the
   best joint assignments become candidate plans;
3. check: hermes.verifier rejects ill-typed candidates before they run;
4. price: hermes.pricing estimates each one's rows and cost;
5. select: a learned estimate of correctness (hermes.selection), minus the
   price, picks the next candidate to run; running it updates the estimate.

The budget caps model calls (one parse each, the rest re-parses with what
was learned), candidate executions, and estimated graph seconds. The planner
stops as soon as a candidate's estimate passes `accept`, and reports its
confidence, so answers below `abstain` can be withheld.
"""

import json
import math
import time
from dataclasses import dataclass, replace

from hermes import llm as llms
from hermes.algebra import Explorer, Plan, Triple, parse_action, render
from hermes.binding import (
    assignments, catalog_for, collapse_labels, drop_unbound_type, mentioned, open_unknown, slots,
    substitute,
)
from hermes.context import as_context
from hermes.knowledge import as_knowledge_graph
from hermes.pricing import Pricer
from hermes.selection import Scorer
from hermes.systems import register
from hermes.verifier import KINDS, Verifier, parse_goal

SYSTEM = (
    "You read questions as SPARQL query plans over a knowledge graph. "
    "Reply with exactly one JSON object."
)

EXAMPLE = {
    "goal": {"answer": ["?c"], "kind": "entity", "type": "[city]", "many": True},
    "actions": [{"op": "triple", "pattern": "?c [country] [Norway]"},
                {"op": "triple", "pattern": "?c [hosted] ?g"},
                {"op": "triple", "pattern": "?g [season] [Winter]"},
                {"op": "project", "vars": ["?c"]}],
}
INSTRUCTIONS = """\
Read the question as a query plan. Reply with one JSON object. An example from
another graph, for "Which cities in Norway hosted the Winter Olympics?":
""" + json.dumps(EXAMPLE) + """

Write a term from the lists below when you are sure of it. Otherwise write the
words of the question in brackets, [like this], and they are matched to the
graph: in predicate position a bracket becomes a property, after `a` a class,
anywhere else an entity or a value. Name entities in brackets, [Norway]: not
as a quoted string, and not as an IRI you guess. The goal type is a class, never
a property; leave it empty if unsure.
Actions: triple {pattern}, filter {expr}, optional {patterns}, union {branches},
minus {patterns}, bind {expr, var}, group {vars}, aggregate {expr, alias},
having {expr}, order {expr}, limit {n}, offset {n}, project {vars}, distinct, ask.
Goal kind is entity, literal, number or boolean; many is false when the
question expects one answer; type is the class of the answer, if any."""


@dataclass
class Spent:
    llm_calls: int = 0
    executions: int = 0
    seconds: float = 0.0  # estimated graph seconds of executed candidates

    def to_dict(self):
        return {"llm_calls": self.llm_calls, "executions": self.executions,
                "seconds": round(self.seconds, 3)}


@dataclass
class Candidate:
    reading: int
    plan: Plan
    goal: object
    sparql: str
    features: dict
    estimate: object
    probability: float = 0.0
    rows: int | None = None
    sample: list = None

    @property
    def executed(self):
        return self.rows is not None

    def to_dict(self):
        return {"reading": self.reading, "sparql": self.sparql, "features": self.features,
                "estimate": self.estimate.to_dict(), "p": round(self.probability, 4),
                "rows": self.rows}


@register("budgeted")
class Budgeted:
    def __init__(self, llm=None, llm_calls=3, executions=8, graph_seconds=None,
                 max_query_seconds=None, per_slot=3, candidates=12, accept=0.75, abstain=0.3,
                 price_weight=1.0, weights=None, sample_size=5, max_entities=10):
        self.llm = llm
        self.llm_calls = llm_calls
        self.executions = executions
        self.graph_seconds = graph_seconds
        self.max_query_seconds = max_query_seconds
        self.per_slot = per_slot
        self.candidates = candidates
        self.accept = accept
        self.abstain = abstain
        self.price_weight = price_weight
        self.scorer = Scorer.load(weights)
        self.sample_size = sample_size
        self.max_entities = max_entities

    def __call__(self, question, kg, context=None):
        llm = self.llm or llms.from_env()
        session = self.start(question, kg, context)
        while (text := session.next_prompt()) is not None:
            session.take(llm.complete(text, SYSTEM))
        return {**session.result(), "model": llm.model, "provider": llm.provider}

    def start(self, question, kg, context=None, explain=False):
        """A planning session for one question, driven one model reply at a time.

        With explain, the session records what every phase did in `events`,
        which the step-by-step demo shows.
        """
        return Session(self, question, kg, context, explain)

    def bind(self, reading, number, catalog, explorer, pricer, pool, log=None):
        """Candidate plans of one reading: bound, verified, priced and scored."""
        try:
            parse_goal(reading.get("goal") or {})
        except ValueError as exc:
            return [], f"invalid goal: {exc}"
        given = reading
        reading = drop_unbound_type(reading, catalog)
        reading = open_unknown(reading, lambda iri: mentioned(explorer.kg, iri), explorer.expand)
        choices = {slot: catalog.rank(slot, self.per_slot) for slot in slots(reading)}
        if log is not None:
            log({"phase": "repair", "reading": number, "changes": _changes(given, reading)})
            log({"phase": "bind", "reading": number, "slots": [
                {"phrase": slot.phrase, "kind": slot.kind,
                 "candidates": [{"term": c.term, "score": c.score, "label": c.label}
                                for c in found]}
                for slot, found in choices.items()]})
        unbound = [slot.phrase for slot, found in choices.items() if not found]
        if unbound:
            return [], "no term matches " + ", ".join(f"[{p}]" for p in unbound)
        seen = {c.sparql for c in pool}
        found, rejected, errors, checked = [], 0, [], []
        for assignment, weight in assignments(choices, self.candidates):
            bound = substitute(reading, assignment)
            variants = [(bound, weight, "binding")] + [
                (variant, weight * _COLLAPSE * score, "label collapse")
                for variant, score in collapse_labels(bound, catalog.resolve)]
            for variant, score, origin in variants:
                entry = {"origin": origin, "weight": round(score, 4)} if log else None
                candidate = self.candidate(variant, score, max(len(assignment), 1),
                                           number, explorer, pricer, errors, entry)
                if candidate is None:
                    rejected += 1
                elif candidate.sparql not in seen:
                    seen.add(candidate.sparql)
                    found.append(candidate)
                elif entry is not None:
                    entry["duplicate"] = True
                if entry is not None:
                    checked.append(entry)
        if log is not None:
            log({"phase": "check", "reading": number, "candidates": checked})
        note = f"{len(found)} new candidates, {rejected} rejected"
        if not found and errors:
            note += f" (last: {errors[-1][:200]})"
        return found, note

    def candidate(self, bound, weight, slots_bound, number, explorer, pricer, errors,
                  entry=None):
        """One bound reading as a verified, priced and scored candidate, or None.

        With an `entry` dict, it is filled with what the verifier and pricing said.
        """
        entry = {} if entry is None else entry
        try:
            goal = parse_goal(bound["goal"])
            plan = Plan()
            for data in bound["actions"]:
                plan = parse_action(data).apply(plan)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(str(exc))
            entry.update(ok=False, errors=[str(exc)])
            return None
        if plan.empty:
            errors.append("the plan has no triple pattern")
            entry.update(ok=False, errors=["the plan has no triple pattern"])
            return None
        judgment = explorer.verifier.check(plan, goal)
        if judgment.ok and goal.type and any(" : " + goal.type in g for g in judgment.goals):
            # Close an open type goal the way the goal states it: a type triple.
            typed = Plan()
            for variable in goal.answer:
                typed = Triple(f"{variable} a {goal.type}").apply(typed)
            typed = replace(plan, triples=typed.triples + plan.triples)
            closed = explorer.verifier.check(typed, goal)
            if closed.ok and len(closed.goals) < len(judgment.goals):
                plan, judgment = typed, closed
                entry["closed_type"] = goal.type
        entry.update(
            sparql=render(plan, explorer.header), ok=judgment.ok, errors=list(judgment.errors),
            warnings=list(judgment.warnings), goals=list(judgment.goals),
            context={v: sorted(explorer.verifier.short(t) for t in types)
                     for v, types in sorted(judgment.context.items())})
        if not judgment.ok:
            explorer.rejected += 1
            errors.append("; ".join(judgment.errors))
            return None
        estimate = pricer.estimate(plan)
        features = {
            "binding": _log(weight) / slots_bound,
            "warnings": len(judgment.warnings),
            "open_goals": len(judgment.goals),
            "estimated_empty": float(estimate.rows < 0.5),
            "estimated_excess": float(not goal.many and estimate.rows > 1.5),
        }
        candidate = Candidate(number, plan, goal, render(plan, explorer.header), features,
                              estimate)
        candidate.probability = self.scorer.probability(features)
        entry.update(estimate=estimate.to_dict(), features=dict(features),
                     p=round(candidate.probability, 4))
        return candidate

    def select(self, pool, spent, explorer, pricer, trace, log=None):
        """Run candidates, best value for price first, until one is accepted.

        While model calls remain, each reading gets an even share of the
        executions: its candidates are alike, so when its best ones come back
        empty a new reading is the better buy.
        """
        share = math.ceil(self.executions / max(self.llm_calls, 1))
        last = spent.llm_calls >= self.llm_calls
        while spent.executions < self.executions:
            ran = {}
            for c in pool:
                ran[c.reading] = ran.get(c.reading, 0) + c.executed
            options = [c for c in pool if not c.executed and self.affordable(c, spent)
                       and (last or ran.get(c.reading, 0) < share)]
            if not options:
                if log is not None:
                    waiting = [c for c in pool if not c.executed]
                    why = ("no candidate left to run" if not waiting else
                           "this reading's share of executions is spent" if not last else
                           "the remaining candidates cost more than the budget allows")
                    log({"phase": "pause", "why": why, "spent": spent.to_dict()})
                return None
            # Rank by a price that depends only on the plan, so runs reproduce.
            candidate = max(options, key=lambda c: c.probability
                            - self.price_weight * pricer.nominal(c.estimate))
            started = time.perf_counter()
            observation = explorer.observe(candidate.plan, "run")
            seconds = time.perf_counter() - started
            spent.executions += 1
            spent.seconds += candidate.estimate.seconds or 0.0
            candidate.rows = observation.count if observation.ok else 0
            candidate.sample = observation.sample
            pricer.observe(candidate.estimate, seconds, candidate.rows if observation.ok else None)
            candidate.features.update(
                executed=1.0,
                empty=float(candidate.rows == 0),
                excess=float(not candidate.goal.many and candidate.rows > 1),
            )
            before = candidate.probability
            candidate.probability = self.scorer.probability(candidate.features)
            trace.append({"step": "run", "sparql": candidate.sparql,
                          "observation": observation.summary(),
                          "p": round(candidate.probability, 3)})
            accepted = candidate.probability >= self.accept
            if log is not None:
                log({"phase": "run", "reading": candidate.reading, "sparql": candidate.sparql,
                     "estimate": candidate.estimate.to_dict(), "seconds": round(seconds, 4),
                     "ok": observation.ok, "error": observation.error,
                     "rows": candidate.rows, "sample": observation.sample,
                     "p_before": round(before, 4), "p": round(candidate.probability, 4),
                     "features": dict(candidate.features), "accept": self.accept,
                     "accepted": accepted, "spent": spent.to_dict(),
                     "waiting": sum(1 for c in pool if not c.executed)})
            if accepted:
                return candidate
        return None

    def affordable(self, candidate, spent):
        seconds = candidate.estimate.seconds or 0.0
        if self.max_query_seconds is not None and seconds > self.max_query_seconds:
            return False
        if self.graph_seconds is not None and spent.seconds + seconds > self.graph_seconds:
            return False
        return True


class Session:
    """The state of planning one question: what was read, bound, run and spent."""

    def __init__(self, planner, question, kg, context=None, explain=False):
        question = (question or "").strip()
        if not question:
            raise ValueError("A question is required.")
        self.planner = planner
        self.question = question
        self.context = as_context(context)
        self.kg = kg = as_knowledge_graph(kg)
        self.explorer = Explorer(kg, sample_size=planner.sample_size)
        self.pricer = Pricer.for_kg(kg, self.explorer.expand)
        self.explorer.verifier = Verifier.for_kg(kg, self.explorer.prefixes,
                                                 self.pricer.statistics)
        self.catalog = catalog_for(kg, self.explorer, self.pricer.statistics)
        self.entities = self.explorer.linker.link(question)[: planner.max_entities]
        self.spent, self.pool, self.trace, self.readings = Spent(), [], [], []
        self.accepted = None
        self.events = [] if explain else None

    @property
    def done(self):
        return self.accepted is not None or self.spent.llm_calls >= self.planner.llm_calls

    def next_prompt(self):
        """The prompt for the next parse, or None when the session is done."""
        if self.done:
            return None
        return prompt(self.question, self.kg.vocabulary, self.explorer, self.entities,
                      self.context, feedback(self.readings, self.pool))

    def take(self, reply):
        """Bind, check, price and run what one model reply reads the question as."""
        text = self.next_prompt()
        self.spent.llm_calls += 1
        number = len(self.readings)
        try:
            reading = parse_reading(reply)
        except ValueError as exc:
            self.readings.append({"error": f"invalid: {exc}"})
            self.trace.append({"step": "parse", "observation": f"invalid: {exc}"})
            self.log({"phase": "parse", "reading": number, "prompt": text, "reply": reply,
                      "error": str(exc), "spent": self.spent.to_dict()})
            return
        self.log({"phase": "parse", "reading": number, "prompt": text, "reply": reply,
                  "parsed": reading, "spent": self.spent.to_dict()})
        log = self.log if self.events is not None else None
        found, note = self.planner.bind(reading, number, self.catalog, self.explorer,
                                        self.pricer, self.pool, log)
        self.readings.append({**reading, "note": note})
        self.trace.append({"step": "parse", "reading": reading, "observation": note})
        self.pool.extend(found)
        self.accepted = self.planner.select(self.pool, self.spent, self.explorer, self.pricer,
                                            self.trace, log)

    def result(self):
        chosen = self.accepted or best(self.pool)
        result = {
            # No reading bound to a well-typed plan: abstain, and keep the trace.
            "sparql": chosen.sparql if chosen else "",
            "confidence": round(chosen.probability, 4) if chosen else 0.0,
            "abstain": chosen is None or chosen.probability < self.planner.abstain,
            "readings": len(self.readings),
            "candidates": [c.to_dict() for c in self.pool],
            "spent": self.spent.to_dict(),
            "verifier_rejections": self.explorer.rejected,
            "qerror": [round(e, 2) for e in self.pricer.errors],
            "trace": self.trace,
        }
        self.log({"phase": "answer", "sparql": result["sparql"],
                  "confidence": result["confidence"], "abstain": result["abstain"],
                  "accepted": self.accepted is not None, "threshold": self.planner.abstain,
                  "spent": self.spent.to_dict()})
        return result

    def log(self, event):
        if self.events is not None:
            self.events.append(event)


def best(pool):
    """The most probable candidate, preferring one that ran and returned rows."""
    if not pool:
        return None
    return max(pool, key=lambda c: (bool(c.rows), c.executed, c.probability))


def parse_reading(text):
    from hermes.algebra import _first_json_object

    data = _first_json_object(text)
    if not isinstance(data.get("actions"), list) or not data["actions"]:
        raise ValueError("the reading needs a list of actions")
    if not all(isinstance(action, dict) for action in data["actions"]):
        raise ValueError("every action must be a JSON object")
    if data["actions"] == EXAMPLE["actions"]:
        raise ValueError("the reading copies the example; read this question")
    goal = dict(data.get("goal")) if isinstance(data.get("goal"), dict) else {}
    if goal.get("kind") not in KINDS:
        goal["kind"] = "entity"
    if not goal.get("answer"):
        projected = [a.get("vars") for a in data["actions"] if a.get("op") == "project"]
        if projected and isinstance(projected[-1], list):
            goal["answer"] = projected[-1]
    return {"goal": goal, "actions": data["actions"]}


def feedback(readings, pool):
    """What earlier readings led to, for the next parse."""
    if not readings:
        return ""
    lines = []
    for number, reading in enumerate(readings):
        if "error" in reading:
            lines.append(f"Reading {number + 1}: {reading['error']}")
            continue
        lines.append(f"Reading {number + 1}: {json.dumps(reading['actions'], ensure_ascii=False)}"
                     f"\n  -> {reading['note']}")
        for candidate in [c for c in pool if c.reading == number and c.executed][:3]:
            sample = "; ".join(", ".join(f"{k}={v}" for k, v in row.items())
                               for row in (candidate.sample or [])[:3])
            lines.append(f"  ran: {candidate.rows} rows{(' | ' + sample) if sample else ''}")
    return ("Earlier readings did not give a confident answer. Read the question "
            "differently: other terms, other structure.\n" + "\n".join(lines))


def prompt(question, vocabulary, explorer, entities, context, earlier=""):
    prefixes = "\n".join(f"PREFIX {p}: <{ns}>" for p, ns in explorer.prefixes)
    classes = "\n".join(f"- {item}" for item in vocabulary["classes"]) or "- (none)"
    properties = "\n".join(f"- {item}" for item in vocabulary["properties"]) or "- (none)"
    linked = "\n".join(f"- {explorer.short(iri)}" for iri in entities) or "- (none found)"
    sections = [
        INSTRUCTIONS,
        f"Prefixes:\n{prefixes}",
        f"Classes:\n{classes}",
        f"Properties:\n{properties}",
        f"Entities whose label appears in the question:\n{linked}",
    ]
    if context.text.strip():
        sections.append(f"Context:\n{context.text.strip()}")
    if context.examples:
        sections.append(
            "Example questions:\n\n"
            + "\n\n".join(f"Question: {e.nl}\nQuery:\n{e.sparql}" for e in context.examples)
        )
    if earlier:
        sections.append(earlier)
    sections.append(f"Question: {question}\n\nReading:")
    return "\n\n".join(sections)


_COLLAPSE = 0.9  # a variant where a name written as a literal became its entity


def _changes(given, repaired):
    """What binding's repairs changed in a reading, and why."""
    changes = []
    before, after = (given.get("goal") or {}).get("type"), (repaired.get("goal") or {}).get("type")
    if before != after:
        changes.append({"before": f"type {before}", "after": "no type",
                        "why": "the goal type names no class in the ontology"})
    for old, new in zip(given.get("actions", []), repaired.get("actions", [])):
        for key in ("pattern", "patterns", "branches"):
            if key in old and old[key] != new.get(key):
                why = ("a quoted value may name an entity: bind it, the literal first"
                       if "[=" in json.dumps(new[key]) else
                       "a guessed IRI the graph lacks becomes a phrase to bind")
                changes.append({"before": old[key], "after": new[key], "why": why})
    return changes


def _log(value):
    return math.log(max(value, 1e-6))
