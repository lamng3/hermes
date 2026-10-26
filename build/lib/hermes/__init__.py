"""Hermes: ontology-based natural-language querying over knowledge graphs.

    from hermes import Hermes, KnowledgeGraph, Budget
    kg = KnowledgeGraph.from_files("ontology.ttl", "data.ttl", schema=["ontology.ttl"])
    answer = Hermes(kg, budget=Budget(llm_calls=3)).ask("Which cities hosted the Winter Olympics?")
    print(answer.sparql, answer.rows)

The pipeline is in hermes.pipeline, the query planner in hermes.planner, and
query pricing and budgets in hermes.pricing. The systems Hermes is compared
with are in hermes.baselines.
"""

from hermes import baselines  # noqa: F401  (registers the baseline systems)
from hermes import systems
from hermes.context import Context, Example
from hermes.graph.knowledge import KnowledgeGraph, as_knowledge_graph
from hermes.pipeline import Answer, Hermes
from hermes.pipeline import system as _system  # noqa: F401  (registers "hermes")
from hermes.pricing.budget import Budget, BudgetExceeded, Price

DEFAULT_SYSTEM = "hermes"


def ask(question, kg, context=None, system=DEFAULT_SYSTEM, **params):
    """Run the named system on a KnowledgeGraph or an RDF file path."""
    return systems.create(system, **params)(question, as_knowledge_graph(kg), context)


__all__ = [
    "Answer", "Budget", "BudgetExceeded", "Context", "Example", "Hermes", "KnowledgeGraph",
    "Price", "ask", "systems",
]
