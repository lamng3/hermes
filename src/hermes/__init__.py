"""Hermes: natural language to SPARQL over ontologies."""

from hermes import systems
from hermes.context import Context, Example
from hermes.knowledge import KnowledgeGraph, as_knowledge_graph
from hermes.systems import algebra_agent, training_free, uniqgen  # noqa: F401  (registers)
from hermes.systems.algebra_agent import AlgebraAgent
from hermes.systems.training_free import TrainingFree
from hermes.systems.uniqgen import UniQGen

DEFAULT_SYSTEM = "training-free"


def ask(question, kg, context=None, system=DEFAULT_SYSTEM, **params):
    """Run the named system on a KnowledgeGraph or an RDF file path."""
    return systems.create(system, **params)(question, as_knowledge_graph(kg), context)


__all__ = [
    "AlgebraAgent",
    "Context",
    "Example",
    "KnowledgeGraph",
    "TrainingFree",
    "UniQGen",
    "ask",
    "systems",
]
