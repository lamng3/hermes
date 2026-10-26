"""Hermes: natural language to SPARQL over ontologies."""

from hermes import systems
from hermes.context import Context, Example
from hermes.graph.knowledge import KnowledgeGraph, as_knowledge_graph
from hermes import baselines  # noqa: F401  (registers the baseline systems)
from hermes.baselines.algebra_agent import AlgebraAgent
from hermes.baselines.budgeted import Budgeted
from hermes.baselines.prover import Prover
from hermes.baselines.training_free import TrainingFree
from hermes.baselines.uniqgen import UniQGen

DEFAULT_SYSTEM = "training-free"


def ask(question, kg, context=None, system=DEFAULT_SYSTEM, **params):
    """Run the named system on a KnowledgeGraph or an RDF file path."""
    return systems.create(system, **params)(question, as_knowledge_graph(kg), context)


__all__ = [
    "AlgebraAgent",
    "Budgeted",
    "Context",
    "Example",
    "KnowledgeGraph",
    "Prover",
    "TrainingFree",
    "UniQGen",
    "ask",
    "systems",
]
