"""Research stages that sit between the draft query and validation.

Each stage takes the draft and returns it, possibly rewritten. Replace a
function in STAGES when that line of work is ready. The training-free system
always walks this list. Only guardrails does anything yet.
"""

import re

_CORE = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}
_DECLARED = re.compile(r"PREFIX\s+([A-Za-z][\w.-]*)?\s*:", re.IGNORECASE)
_USED = re.compile(r"(?<![\w<?$\"'])([A-Za-z][\w.-]*):(?=[A-Za-z_\d])")


def guardrails(sparql, *, ontology, context):
    """Declare known prefixes that the query uses but does not declare.

    Rejecting terms outside the ontology vocabulary is still to do.
    """
    known = dict(_CORE)
    known.update(dict(ontology.get("prefixes", [])))
    declared = {name.lower() for name in _DECLARED.findall(sparql) if name}
    body = re.sub(r"<[^>]*>|\"[^\"]*\"|'[^']*'", " ", sparql)
    missing = []
    for name in _USED.findall(body):
        if name.lower() not in declared and name in known and name not in missing:
            missing.append(name)
    header = "".join(f"PREFIX {name}: <{known[name]}>\n" for name in missing)
    return header + sparql


def query_guided(sparql, *, ontology, context):
    """Steer a draft using a target query shape or a previous SPARQL query."""
    return sparql


def context_guided(sparql, *, ontology, context):
    """Retrieve and place example questions, beyond stuffing them into the prompt."""
    return sparql


def chase_backchase(sparql, *, ontology, context):
    """Reformulate the query with ontology constraints, then minimize it.

    This stage does not implement the algorithm.
    """
    return sparql


STAGES = (guardrails, query_guided, context_guided, chase_backchase)


def apply_stages(sparql, ontology, context):
    for stage in STAGES:
        sparql = stage(sparql, ontology=ontology, context=context)
    return sparql
