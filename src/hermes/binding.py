"""Binding: resolve the phrases of a parsed question to graph terms.

The parser writes a plan whose unresolved terms are phrases in brackets,
`?c [hosted] ?g`. Binding gives each slot ranked candidates, the way a
database binder resolves names against its catalog: a slot in predicate
position becomes a property, one after `a` (or a goal's type) a class, and
any other an entity or, failing that, a literal. Properties and classes come
from the ontology's labels and local names and from the terms the graph
actually uses, so a graph that strays from its ontology still binds; entities
come from the entity linker.
"""

import difflib
import heapq
import math
import re
from dataclasses import dataclass

from hermes.linking import LABELS, local_name, normalize
from hermes.verifier import _TOKEN, TYPE, split_triples

SLOT = re.compile(r"\[([^\[\]]+)\]")
_PLACEHOLDER = "?hermes_slot{}"
_LITERAL_SCORE = 0.3
_INVERSE = 0.7  # a property read backwards
_VALUE_SCORE = 0.9  # a literal the model wrote itself
_COLLAPSE = 0.9  # a name written as a literal, replaced by the entity it names
_CORE = (  # the RDF, RDFS, OWL and XSD vocabularies describe schemas, not data
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2001/XMLSchema#",
)
_TERMS = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?term ?kind ?label WHERE {
  VALUES (?type ?kind) { (owl:Class "class") (rdfs:Class "class")
    (owl:ObjectProperty "property") (owl:DatatypeProperty "property") (rdf:Property "property") }
  ?term a ?type
  OPTIONAL { ?term rdfs:label ?label }
  FILTER(isIRI(?term))
}
"""


@dataclass(frozen=True)
class Slot:
    phrase: str
    kind: str  # property, class or entity


@dataclass(frozen=True)
class Candidate:
    term: str  # as written in SPARQL: a prefixed name, an <IRI> or a literal
    score: float
    label: str = ""


class Catalog:
    """Properties, classes and entities a phrase can bind to, with their names."""

    def __init__(self, kg, explorer, statistics=None):
        self.kg = kg
        self.explorer = explorer
        self.names = {"property": {}, "class": {}}
        for row in kg.schema.try_query(_TERMS)["results"]["bindings"]:
            names = self.names[row["kind"]["value"]].setdefault(row["term"]["value"], set())
            if "label" in row:
                names.add(row["label"]["value"])
        if statistics is not None:
            # Terms the data uses, declared or not.
            for iri in statistics.predicates:
                self.names["property"].setdefault(iri, set())
            for iri in statistics.classes:
                self.names["class"].setdefault(iri, set())
        for terms in self.names.values():
            for iri in [iri for iri in terms if iri.startswith(_CORE)]:
                del terms[iri]
        for terms in self.names.values():
            for iri, names in terms.items():
                names.add(local_name(iri))

    def rank(self, slot, limit=3):
        phrase = slot.phrase
        if slot.kind == "value":
            return self._values(phrase[1:], limit)
        exact = self._exact(phrase, slot.kind)
        if _guessed_iri(phrase):
            phrase = _words(local_name(self.explorer.expand(phrase)))
        if slot.kind == "entity":
            found = self._entities(phrase, limit)
        else:
            found = self._terms(phrase, slot.kind, limit)
        if exact:
            found = [exact] + [c for c in found if c.term != exact.term]
        return found

    def resolve(self, name, limit=2):
        """Entities a name may denote, as (term, score): exact labels first."""
        exact = sorted(self.explorer.linker.lookup(name))
        if exact:
            return [(self.explorer.short(iri), 1.0) for iri in exact[:limit]]
        return [(c.term, c.score) for c in self._entities(name, limit) if c.label != "literal"]

    def _exact(self, phrase, kind):
        """A term written as such, [pv:memberOf] or [rdf:type], if the graph knows it."""
        if kind == "property" and phrase in ("a", "rdf:type"):
            return Candidate("a", 1.0, "type")
        if not _guessed_iri(phrase):
            return None
        iri = self.explorer.expand(phrase)
        known = (iri in self.names.get(kind, {}) or (kind == "property" and iri in LABELS)
                 or (kind == "entity" and mentioned(self.kg, iri)))
        return Candidate(self.explorer.short(iri), 1.0, phrase) if known else None

    def _terms(self, phrase, kind, limit):
        scored = []
        for iri, names in self.names[kind].items():
            best = max(similarity(phrase, name) for name in names)
            if best > 0.3:
                scored.append((best, iri, max(names, key=len)))
        scored.sort(key=lambda row: (-row[0], row[1]))
        found = [Candidate(self.explorer.short(iri), round(score, 3), label)
                 for score, iri, label in scored[:limit]]
        if kind == "property":
            # The question may name the relation from the other end: "X's manager".
            found += [Candidate("^" + c.term, round(c.score * _INVERSE, 3), c.label)
                      for c in found[:2]]
            found.sort(key=lambda c: -c.score)
        return found

    def _entities(self, phrase, limit):
        found = []
        for rank, (iri, label) in enumerate(self.explorer.linker.search(phrase, limit=limit)):
            score = max(similarity(phrase, label), 0.4) * (0.9 ** rank)
            found.append(Candidate(self.explorer.short(iri), round(score, 3), label))
        # Failing an entity, the phrase may be a value: a number or a string.
        found.append(Candidate(_literal(phrase), _LITERAL_SCORE, "literal"))
        return found

    def _values(self, text, limit):
        """A value the model wrote as a literal: kept first, with entities it may name."""
        found = [Candidate(_literal(text), _VALUE_SCORE, "literal")]
        return found + [c for c in self._entities(text, limit) if c.label != "literal"]


def slots(reading):
    """Slots of a reading's actions and goal, in order of appearance."""
    found = []
    goal_type = (reading.get("goal") or {}).get("type") or ""
    for match in SLOT.finditer(goal_type):
        found.append(Slot(match.group(1).strip(), "class"))
    for action in reading.get("actions", []):
        for text, in_pattern in _texts(action):
            found.extend(_slots_in(text, in_pattern))
    unique = []
    for slot in found:
        if slot not in unique:
            unique.append(slot)
    return unique


def substitute(reading, assignment):
    """The reading with every slot replaced by its bound term."""

    def fill(text, in_pattern):
        def term(match):
            kind = _kind_at(text, match, in_pattern)
            return assignment[Slot(match.group(1).strip(), kind)].term
        filled = SLOT.sub(term, text)
        triples, _complete = split_triples(filled)
        if in_pattern and len(triples) == 1 and triples[0][1].startswith("^"):
            subject, predicate, value = triples[0]
            if len(_TOKEN.findall(filled.rstrip(" ."))) == 3 and value[:1] not in "\"'":
                return f"{value} {predicate[1:]} {subject}"  # a plain triple, turned around
        return filled

    goal = dict(reading.get("goal") or {})
    if goal.get("type"):
        goal["type"] = SLOT.sub(lambda m: assignment[Slot(m.group(1).strip(), "class")].term,
                                goal["type"])
    actions = []
    for action in reading.get("actions", []):
        filled = dict(action)
        for key, value in action.items():
            in_pattern = key in ("pattern", "patterns", "branches")
            filled[key] = _map_strings(value, lambda text: fill(text, in_pattern))
        actions.append(filled)
    return {"goal": goal, "actions": actions}


def drop_unbound_type(reading, catalog):
    """A goal type that names no class says nothing; drop it rather than the reading."""
    goal = dict(reading.get("goal") or {})
    phrases = [m.group(1).strip() for m in SLOT.finditer(goal.get("type") or "")]
    if phrases and any(not catalog.rank(Slot(p, "class")) for p in phrases):
        goal["type"] = ""
        return {**reading, "goal": goal}
    return reading


def open_unknown(reading, exists, expand):
    """Turn what a model guessed in subject and object position into phrases.

    A guessed IRI the graph lacks, `prodi:MsBrant`, becomes `[Ms Brant]`; a
    quoted value, `"network expert"`, becomes `[=network expert]`, which binds
    to the literal first and to the entities it may name after it.
    """
    def repair(text):
        plain, phrases = _placeholders(text)
        for subject, predicate, value in split_triples(plain)[0]:
            terms = [subject] if predicate in ("a", "rdf:type") else [subject, value]
            for token in terms:
                if _guessed_iri(token) and not exists(expand(token)):
                    plain = re.sub(rf"(?<!\S){re.escape(token)}(?!\S)",
                                   f"[{_words(local_name(expand(token)))}]", plain)
            quoted = re.fullmatch(r'"((?:[^"\\\[\]]|\\.)*)"(?:@[\w-]+)?', value)
            if quoted and predicate not in ("a", "rdf:type"):
                # A value written as a literal may name an entity: bind it, literal first.
                plain = plain.replace(value, f"[={quoted.group(1)}]", 1)
        for index, phrase in enumerate(phrases):
            plain = plain.replace(_PLACEHOLDER.format(index), f"[{phrase}]")
        return plain

    actions = []
    for action in reading.get("actions", []):
        fixed = dict(action)
        for key in ("pattern", "patterns", "branches"):
            if key in action:
                fixed[key] = _map_strings(action[key], repair)
        actions.append(fixed)
    return {**reading, "actions": actions}


def collapse_labels(bound, resolve, limit=2):
    """Variants where `?v <name> "Some Name"` becomes the entity of that name.

    Models often write an entity as its label; when the label names an
    entity, the variable can be that entity and the label triple dropped.
    Returns (variant, score) pairs; `resolve(name, limit)` gives the entities.
    """
    answers = set((bound.get("goal") or {}).get("answer") or [])
    variants = []
    for index, action in enumerate(bound["actions"]):
        if action.get("op") != "triple" or not isinstance(action.get("pattern"), str):
            continue
        triples, _complete = split_triples(action["pattern"])
        if len(triples) != 1:
            continue
        variable, _predicate, value = triples[0]
        match = re.fullmatch(r'"((?:[^"\\]|\\.)*)"(?:@[\w-]+)?', value)
        if not variable.startswith("?") or not match or variable in answers:
            continue
        projected = any(variable in (a.get("vars") or []) for a in bound["actions"])
        if projected:
            continue
        for entity, score in resolve(match.group(1), limit):
            pattern = re.compile(re.escape(variable) + r"(?!\w)")
            actions = [{key: value if key == "op" else
                        _map_strings(value, lambda text: pattern.sub(entity, text))
                        for key, value in a.items()}
                       for i, a in enumerate(bound["actions"]) if i != index]
            variants.append(({**bound, "actions": actions}, score))
    return variants


def assignments(choices, limit):
    """Best-first joint assignments: slot -> Candidate, by the product of scores."""
    keys = list(choices)
    if any(not choices[key] for key in keys):
        return []
    if not keys:
        return [({}, 1.0)]

    def weight(index):
        return math.prod(choices[k][i].score for k, i in zip(keys, index))

    start = (0,) * len(keys)
    heap, seen, found = [(-weight(start), start)], {start}, []
    while heap and len(found) < limit:
        negative, index = heapq.heappop(heap)
        found.append(({k: choices[k][i] for k, i in zip(keys, index)}, -negative))
        for position in range(len(keys)):
            step = list(index)
            step[position] += 1
            step = tuple(step)
            if step[position] < len(choices[keys[position]]) and step not in seen:
                seen.add(step)
                heapq.heappush(heap, (-weight(step), step))
    return found


def similarity(phrase, name):
    """How well a phrase names a term: stem overlap blended with spelling."""
    one, other = normalize(phrase), normalize(name)
    if not one or not other:
        return 0.0
    if one == other:
        return 1.0
    a, b = {_stem(w) for w in one.split()}, {_stem(w) for w in other.split()}
    overlap = len(a & b) / len(a | b)
    spelling = difflib.SequenceMatcher(None, one, other).ratio()
    return 0.6 * overlap + 0.4 * spelling


def _stem(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _texts(action):
    for key, value in action.items():
        if key == "op":
            continue
        in_pattern = key in ("pattern", "patterns", "branches")
        for text in _strings(value):
            yield text, in_pattern


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _map_strings(value, function):
    if isinstance(value, str):
        return function(value)
    if isinstance(value, (list, tuple)):
        return [_map_strings(item, function) for item in value]
    return value


def _slots_in(text, in_pattern):
    return [Slot(m.group(1).strip(), _kind_at(text, m, in_pattern)) for m in SLOT.finditer(text)]


def _placeholders(text):
    phrases = [m.group(1).strip() for m in SLOT.finditer(text)]
    plain = text
    for index, m in reversed(list(enumerate(SLOT.finditer(text)))):
        plain = plain[: m.start()] + _PLACEHOLDER.format(index) + plain[m.end():]
    return plain, phrases


def _literal(text):
    text = text.strip()
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def mentioned(kg, iri):
    """Does the graph mention this IRI at all?"""
    query = f"ASK {{ {{ <{iri}> ?p ?o }} UNION {{ ?s ?p <{iri}> }} }}"
    return bool(kg.try_query(query).get("boolean"))


def _guessed_iri(token):
    if token.startswith("<") and token.endswith(">"):
        return True
    return bool(re.fullmatch(r"[A-Za-z][\w.-]*:[^\s\[\]\"']+", token))


def _words(local):
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local)
    return " ".join(re.sub(r"[^\w]+|_", " ", spaced).split())


def _kind_at(text, match, in_pattern):
    """property, class, entity or value, from where the slot sits in a triple pattern."""
    if match.group(1).strip().startswith("="):
        return "value"
    if not in_pattern:
        return "entity"
    marked, mine = [], None
    for index, other in enumerate(SLOT.finditer(text)):
        marked.append(other)
        if other.start() == match.start():
            mine = _PLACEHOLDER.format(index)
    plain = text
    for index, other in reversed(list(enumerate(marked))):
        plain = plain[: other.start()] + _PLACEHOLDER.format(index) + plain[other.end():]
    for _subject, predicate, value in split_triples(plain)[0]:
        if predicate == mine:
            return "property"
        if value == mine and predicate in ("a", "rdf:type", f"<{TYPE}>"):
            return "class"
    return "entity"
