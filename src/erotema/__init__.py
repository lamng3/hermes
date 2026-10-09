"""Erotema: natural language to SPARQL, with pluggable systems."""

from erotema import systems
from erotema.generate import Context, Example, generate

DEFAULT_SYSTEM = "training-free"

systems.register(DEFAULT_SYSTEM)(generate)


def ask(question, ontology, context=None, system=DEFAULT_SYSTEM):
    """Run the named system and return its result dict."""
    return systems.get(system)(question, ontology, context)


__all__ = ["Context", "Example", "ask", "generate", "systems"]
