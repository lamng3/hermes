"""Research stages that sit between the draft query and validation.

Each function returns the draft unchanged. Replace a function in STAGES
when that line of work is ready. generate() always walks this list.
"""


def guardrails(sparql, *, ontology, context):
    """Reject or rewrite queries that leave the ontology vocabulary."""
    return sparql


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
