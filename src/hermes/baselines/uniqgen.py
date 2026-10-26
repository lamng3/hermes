"""UniQGen: constraint-guided query generation with Chase and Backchase.

Wang, Jedema, Pandey, Krishnan, Lehmann and Wu, "Graph Query Generation with
Constraint-guided Large Language Agents", arXiv:2605.00845.

The pipeline follows the paper's three stages, rendered to SPARQL over any
graph store (local files or an endpoint):

1. Query-centric fact extraction: link question n-grams to graph nodes by
   label, then collect their k-hop neighbourhood.
2. CTable construction (agent a1): one LLM call proposes triple patterns and
   filters, then each constraint gets an uncertainty score u = n / m from its
   match count in the graph. Constraints with no match are pruned.
3. Chase and Backchase (agent a2): a beam search drops constraints until the
   query covers the reference answers (the universal query), then rebuilds a
   query from single constraints until it returns only reference answers (the
   minimal query).

Two choices differ from the paper. The query generator renders the conjunctive
query deterministically instead of with an LLM, and the evaluator scores
candidates by comparing answer sets with the reference set instead of asking an
LLM. That leaves two LLM calls per question, the cost the paper reports: one for
the CTable and one for the reference answers. Non-monotone constraints such as
aggregates, ORDER BY and LIMIT are not handled.
"""

import json
import re
from dataclasses import dataclass, field


from hermes import llm as llms
from hermes.context import as_context
from hermes.graph.knowledge import as_knowledge_graph
from hermes.graph.linking import linker_for, normalize
from hermes.systems import register
from hermes.sparql.validate import parse_error

_VARIABLE = re.compile(r"\?[A-Za-z_]\w*")
CORE_PREFIXES = (
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
)
_SYSTEM = (
    "You help answer questions over a knowledge graph. "
    "Follow the requested output format exactly."
)


@dataclass(frozen=True)
class Constraint:
    """One CTable row: a SPARQL triple pattern or a FILTER."""

    text: str
    u: float = field(default=1.0, compare=False)
    order: int = field(default=0, compare=False)

    @property
    def is_filter(self):
        return self.text.upper().startswith("FILTER")

    @property
    def variables(self):
        return frozenset(_VARIABLE.findall(self.text))


@dataclass
class Candidate:
    constraints: frozenset
    sparql: str | None
    results: set
    recall: float
    precision: float
    p: float = 0.0

    @property
    def f1(self):
        total = self.recall + self.precision
        return 2 * self.recall * self.precision / total if total else 0.0


@register("uniqgen")
class UniQGen:
    """A Hermes system. Call it with (question, kg, context).

    Candidate queries run against `kg`, so it must hold instance data, in a
    local file or behind an endpoint. `oracle` picks the reference answers:
    None asks the LLM (prompt P1), "gold" reads `context.answers` as in the
    paper's ablation, and a callable (question, context) -> names is used as is.
    """

    def __init__(
        self,
        beam_width=5,
        max_depth=5,
        threshold=1.0,
        alpha=0.5,
        hops=2,
        match_cap=1000,
        max_facts=200,
        oracle=None,
        llm=None,
    ):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.threshold = threshold
        self.alpha = alpha
        self.hops = hops
        self.match_cap = match_cap
        self.max_facts = max_facts
        self.oracle = oracle
        self.llm = llm

    def __call__(self, question, kg, context=None):
        question = (question or "").strip()
        if not question:
            raise ValueError("A question is required.")
        llm = self.llm or llms.from_env()
        context = as_context(context)
        kg = as_knowledge_graph(kg)
        start = kg.executions
        vocabulary = kg.vocabulary
        graph = GraphView(kg)
        entities = graph.linker.link(question)
        facts = graph.subgraph(entities, self.hops, self.max_facts)
        prompt = ctable_prompt(question, vocabulary, graph, entities, facts, context)
        try:
            answer, rows = parse_ctable(llm.complete(prompt, system=_SYSTEM))
        except ValueError:
            reminder = prompt + "\n\nReply with the JSON object only, even if unsure."
            answer, rows = parse_ctable(llm.complete(reminder, system=_SYSTEM))
        ctable, answer = self.score(graph, answer, rows)
        reference = self.reference(llm, question, context)
        grounded = graph.ground(reference)
        model, provider = llm.model, llm.provider
        result = {
            "answer": answer,
            "entities": entities,
            "ctable": [{"constraint": c.text, "u": round(c.u, 3)} for c in ctable],
            "reference": reference,
            "grounded": sorted(grounded),
            "model": model,
            "provider": provider,
        }
        if not grounded:
            result["sparql"] = graph.render(answer, ctable)
            result["note"] = (
                "No reference answer was found in the graph, "
                "so Chase and Backchase did not run."
            )
            return result
        universal = self.chase(graph, answer, ctable, grounded)
        minimal = self.backchase(graph, answer, universal, grounded)
        chosen = minimal if minimal.sparql and minimal.f1 >= universal.f1 else universal
        result.update(
            sparql=chosen.sparql,
            universal=universal.sparql,
            minimal=minimal.sparql,
            executions=kg.executions - start,
        )
        return result

    def reference(self, llm, question, context):
        if self.oracle == "gold":
            return list(context.answers)
        if callable(self.oracle):
            return list(self.oracle(question, context))
        return parse_answers(llm.complete(oracle_prompt(question), system=_SYSTEM))

    def score(self, graph, answer, rows):
        """Algorithm 2, lines 8-13: u_i = n_i / min(max n, cap); drop n_i = 0."""
        rows = [row for row in rows if graph.parses(row)]
        triples = [row for row in rows if not row.upper().startswith("FILTER")]
        counts = []
        for row in rows:
            if row.upper().startswith("FILTER"):
                needed = set(_VARIABLE.findall(row))
                support = [t for t in triples if needed & set(_VARIABLE.findall(t))]
                counts.append(graph.count(support + [row]))
            else:
                counts.append(graph.count([row]))
        cap = min(max(counts, default=0), self.match_cap)
        ctable = [
            Constraint(row, min(n, cap) / cap, order)
            for order, (row, n) in enumerate(zip(rows, counts, strict=True))
            if n > 0
        ]
        if not any(not c.is_filter for c in ctable):
            raise ValueError("No constraint in the CTable matches the graph.")
        bound = set().union(*(c.variables for c in ctable if not c.is_filter))
        if not bound:
            raise ValueError("No constraint in the CTable binds a variable to answer with.")
        if answer not in bound:
            answer = sorted(bound)[0]
        return ctable, answer

    def chase(self, graph, answer, ctable, grounded):
        """Algorithm 1: relax top-down until the query covers the reference."""
        beam = [frozenset(ctable)]
        seen = set(beam)
        best = None
        for _depth in range(self.max_depth + 1):
            if not beam:
                break
            labels = []
            for constraints in beam:
                candidate = self._candidate(graph, answer, constraints, grounded)
                candidate.p = candidate.recall
                labels.append(candidate)
            step = max(labels, key=lambda c: (c.p, len(c.constraints)))
            if best is None or (step.p, len(step.constraints)) > (
                best.p,
                len(best.constraints),
            ):
                best = step
            if step.p >= self.threshold:
                break
            children = {}
            for label in labels:
                for constraint in label.constraints:
                    child = label.constraints - {constraint}
                    if child in seen or not _renders(child, answer):
                        continue
                    score = self._score(constraint, label)
                    children[child] = max(children.get(child, -1.0), score)
            seen.update(children)
            beam = self._top(children)
        return best

    def backchase(self, graph, answer, universal, grounded):
        """Algorithm 3: rebuild bottom-up until only reference answers remain."""
        floor = universal.recall
        pool = sorted(universal.constraints, key=lambda c: c.order)
        beam = [frozenset([c]) for c in pool]
        seen = set(beam)
        best = None
        for _depth in range(self.max_depth + 1):
            if not beam:
                break
            labels = []
            for constraints in beam:
                candidate = self._candidate(graph, answer, constraints, grounded)
                complete_enough = candidate.sparql and candidate.recall >= floor
                candidate.p = candidate.precision if complete_enough else 0.0
                labels.append(candidate)
            step = max(labels, key=lambda c: (c.p, -len(c.constraints)))
            if best is None or (step.p, -len(step.constraints)) > (
                best.p,
                -len(best.constraints),
            ):
                best = step
            if step.p >= self.threshold:
                break
            children = {}
            for label in labels:
                for constraint in pool:
                    if constraint in label.constraints:
                        continue
                    child = label.constraints | {constraint}
                    if child in seen:
                        continue
                    score = self._score(constraint, label)
                    children[child] = max(children.get(child, -1.0), score)
            seen.update(children)
            beam = self._top(children)
        return best

    def _score(self, constraint, parent):
        """Equation 1: s'(c, Q) = alpha (1 - c.u) + (1 - alpha) Q.p."""
        return self.alpha * (1 - constraint.u) + (1 - self.alpha) * parent.p

    def _top(self, children):
        ranked = sorted(
            children.items(),
            key=lambda item: (-item[1], sorted(c.order for c in item[0])),
        )
        return [constraints for constraints, _score in ranked[: self.beam_width]]

    def _candidate(self, graph, answer, constraints, grounded):
        if not _renders(constraints, answer):
            return Candidate(constraints, None, set(), 0.0, 0.0)
        sparql = graph.render(answer, constraints)
        results = graph.answers(sparql, answer)
        recall, precision = measure(results, grounded)
        return Candidate(constraints, sparql, results, recall, precision)


def measure(results, grounded):
    """Relative completeness and soundness of an answer set w.r.t. G(A)."""
    if not grounded:
        return 0.0, 0.0
    covered = sum(1 for keys in grounded.values() if keys & results)
    reference = set().union(*grounded.values())
    precision = len(results & reference) / len(results) if results else 0.0
    return covered / len(grounded), precision


def _renders(constraints, answer):
    return any(not c.is_filter and answer in c.variables for c in constraints)


class GraphView:
    """The graph operations UniQGen needs, over any KnowledgeGraph."""

    def __init__(self, kg):
        self.kg = kg
        self.linker = linker_for(kg)
        known = {prefix for prefix, _namespace in kg.vocabulary["prefixes"]}
        self.prefixes = list(kg.vocabulary["prefixes"]) + [
            pair for pair in CORE_PREFIXES if pair[0] not in known
        ]
        self.header = "".join(f"PREFIX {p}: <{ns}>\n" for p, ns in self.prefixes)

    def subgraph(self, entities, hops, limit):
        """Triples within `hops` of the entities, as SPARQL JSON terms."""
        frontier = list(entities)
        visited = set(frontier)
        facts = []
        known = set()
        for _hop in range(hops):
            reached = []
            for node in frontier:
                found = []
                out = self.kg.try_query(f"SELECT ?p ?o WHERE {{ <{node}> ?p ?o }} LIMIT {limit}")
                for row in out["results"]["bindings"]:
                    found.append(({"type": "uri", "value": node}, row["p"], row["o"]))
                inc = self.kg.try_query(f"SELECT ?s ?p WHERE {{ ?s ?p <{node}> }} LIMIT {limit}")
                for row in inc["results"]["bindings"]:
                    found.append((row["s"], row["p"], {"type": "uri", "value": node}))
                for fact in sorted(found, key=_fact_key):
                    key = _fact_key(fact)
                    if key in known:
                        continue
                    known.add(key)
                    facts.append(fact)
                    if len(facts) >= limit:
                        return facts
                    for term in (fact[0], fact[2]):
                        iri = term["value"]
                        if term["type"] == "uri" and iri not in visited:
                            visited.add(iri)
                            reached.append(iri)
            frontier = reached
        return facts

    def term(self, term):
        """Render a SPARQL JSON term (or an IRI string) in Turtle syntax."""
        if isinstance(term, str):
            term = {"type": "uri", "value": term}
        value = term["value"]
        if term["type"] == "uri":
            for prefix, namespace in self.prefixes:
                local = value[len(namespace):]
                if value.startswith(namespace) and re.fullmatch(r"[A-Za-z_][\w-]*", local):
                    return f"{prefix}:{local}"
            return f"<{value}>"
        if term["type"] == "bnode":
            return f"_:{value}"
        text = json.dumps(value, ensure_ascii=False)
        if "xml:lang" in term:
            return f"{text}@{term['xml:lang']}"
        if "datatype" in term:
            return f"{text}^^{self.term(term['datatype'])}"
        return text

    def parses(self, row):
        return parse_error(f"{self.header}SELECT * WHERE {{ {_statement(row)} }}") is None

    def count(self, rows):
        body = "\n  ".join(_statement(row) for row in rows)
        sparql = f"{self.header}SELECT (COUNT(*) AS ?n) WHERE {{\n  {body}\n}}"
        rows = self.kg.try_query(sparql)["results"]["bindings"]
        try:
            return int(rows[0]["n"]["value"])
        except (IndexError, KeyError, ValueError):
            return 0

    def render(self, answer, constraints):
        ordered = sorted(constraints, key=lambda c: (c.is_filter, c.order))
        body = "\n  ".join(_statement(c.text) for c in ordered)
        return f"{self.header}SELECT DISTINCT {answer} WHERE {{\n  {body}\n}}"

    def answers(self, sparql, answer):
        name = answer.lstrip("?")
        rows = self.kg.try_query(sparql)["results"]["bindings"]
        return {_key(row[name]) for row in rows if name in row}

    def ground(self, reference):
        """Map each reference answer to the graph keys it can stand for: G(A)."""
        grounded = {}
        for name in reference:
            name = str(name)
            if re.match(r"https?://", name):
                keys = {name} if self._mentioned(f"<{name}>") else set()
            else:
                keys = set(self.linker.lookup(name))
                escaped = json.dumps(name, ensure_ascii=False)
                condition = f"FILTER(isLiteral(?literal) && str(?literal) = {escaped})"
                if self._mentioned("?literal", condition):
                    keys.add(f"literal:{normalize(name)}")
            if keys:
                grounded[name] = keys
        return grounded

    def _mentioned(self, term, condition=""):
        query = f"ASK {{ {{ ?s ?p {term} }} UNION {{ {term} ?p ?o }} {condition} }}"
        if term.startswith("?"):
            query = f"ASK {{ ?s ?p {term} {condition} }}"
        return bool(self.kg.try_query(query).get("boolean"))


def _fact_key(fact):
    return tuple((term["type"], term["value"]) for term in fact)


def ctable_prompt(question, vocabulary, graph, entities, facts, context):
    classes = "\n".join(f"- {item}" for item in vocabulary["classes"]) or "- (none)"
    properties = "\n".join(f"- {item}" for item in vocabulary["properties"]) or "- (none)"
    prefixes = "\n".join(f"PREFIX {p}: <{ns}>" for p, ns in graph.prefixes)
    linked = "\n".join(f"- {graph.term(iri)}" for iri in entities) or "- (none)"
    triples = (
        "\n".join(" ".join(graph.term(t) for t in fact) + " ." for fact in facts)
        or "(none)"
    )
    sections = [
        (
            "I am generating a SPARQL query for a natural language question over "
            "the knowledge graph below. Build a constraint table for the question. "
            "Each constraint is one SPARQL triple pattern, such as "
            "?city ex:hosted ?games, or one filter, such as FILTER(?year > 1896). "
            "List every constraint the query might use: relations that connect the "
            "entities, type conditions, and value filters. Also add constraints the "
            "question implies but does not state. Use only the prefixes, terms and "
            "entities shown. Use a variable for each unknown entity or value, and the "
            "same variable wherever the same thing appears. "
            'Return only JSON: {"answer": "?variable", "constraints": ["...", "..."]}, '
            "where answer is the variable the question asks for."
        ),
        f"Prefixes:\n{prefixes}",
        f"Classes:\n{classes}",
        f"Properties:\n{properties}",
        f"Entities in the question:\n{linked}",
        f"Facts near those entities:\n{triples}",
    ]
    if context.text.strip():
        sections.append(f"Context:\n{context.text.strip()}")
    if context.examples:
        sections.append(
            "Example questions:\n\n"
            + "\n\n".join(
                f"Question: {e.nl.strip()}\nQuery:\n{e.sparql.strip()}"
                for e in context.examples
            )
        )
    sections.append(f"Question: {question}")
    return "\n\n".join(sections)


def oracle_prompt(question):
    """Prompt P1 from the paper's Appendix IV-D."""
    return (
        f'Given the question: "{question}", output a semicolon-separated list of '
        "answer entities. Rules: (i) only output entities you are confident exist "
        "in the knowledge graph; (ii) if uncertain, output UNKNOWN; (iii) do not "
        "include explanations or extra entities."
    )


def parse_ctable(content):
    text = content or ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The model did not return a constraint table.")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"The constraint table is not valid JSON: {exc}") from exc
    rows = []
    for row in data.get("constraints") or []:
        row = str(row).strip().rstrip(".").strip()
        if row and row not in rows:
            rows.append(row)
    return str(data.get("answer") or "").strip(), rows


def parse_answers(content):
    answers = []
    for part in re.split(r"[;\n]", content or ""):
        part = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part).strip().strip("\"'")
        if part and part.upper() != "UNKNOWN" and part not in answers:
            answers.append(part)
    return answers


def _statement(row):
    row = row.strip().rstrip(".").strip()
    return row if row.upper().startswith("FILTER") else f"{row} ."


def _key(term):
    if term["type"] == "literal" or term["type"] == "typed-literal":
        return f"literal:{normalize(term['value'])}"
    return term["value"]
