"""The training-free system: one draft from the vocabulary, then fixed stages.

The draft is prompted from the ontology vocabulary and optional context, passed
through the stages in hermes.stages, parse-checked, and repaired once.
"""

from hermes import llm as llms
from hermes.context import as_context
from hermes.knowledge import as_knowledge_graph
from hermes.stages import apply_stages
from hermes.systems import register
from hermes.validate import parse_error, prepare_draft, repair_prompt


@register("training-free")
class TrainingFree:
    def __init__(self, llm=None):
        self.llm = llm

    def __call__(self, question, kg, context=None):
        question = (question or "").strip()
        if not question:
            raise ValueError("A question is required.")
        llm = self.llm or llms.from_env()
        context = as_context(context)
        vocabulary = as_knowledge_graph(kg).vocabulary
        content = llm.complete(build_prompt(question, vocabulary, context))
        sparql = apply_stages(prepare_draft(content), vocabulary, context)
        error = parse_error(sparql)
        if error:
            content = llm.complete(repair_prompt(sparql, error))
            sparql = apply_stages(prepare_draft(content), vocabulary, context)
            error = parse_error(sparql)
            if error:
                raise ValueError(f"The model returned SPARQL that does not parse: {error}")
        return {"sparql": sparql, "model": llm.model, "provider": llm.provider}


def build_prompt(question, vocabulary, context=None):
    context = as_context(context)
    classes = "\n".join(f"- {item}" for item in vocabulary["classes"]) or "- (none found)"
    properties = (
        "\n".join(f"- {item}" for item in vocabulary["properties"]) or "- (none found)"
    )
    prefixes = "\n".join(
        f"PREFIX {prefix}: <{namespace}>" for prefix, namespace in vocabulary["prefixes"]
    )
    sections = [
        (
            "Write one SPARQL query for this competency question. "
            "Use only the prefixes and terms below. "
            "Match the question to the labels in quotes. "
            "Put triple patterns inside WHERE { }. "
            "Put ORDER BY and LIMIT after that block. "
            "Do not put SELECT inside FILTER. "
            "Do not copy the shape example terms. "
            "Return only the query."
        ),
        (
            "Shape:\n"
            "PREFIX ex: <http://example.org/>\n"
            "SELECT ?item ?value\n"
            "WHERE {\n"
            "  ?item a ex:Record ;\n"
            "        ex:relatedItem ?other ;\n"
            "        ex:amount ?value .\n"
            "}\n"
            "ORDER BY DESC(?value)"
        ),
        f"Prefixes:\n{prefixes or '(none)'}",
        f"Classes:\n{classes}",
        f"Properties:\n{properties}",
    ]
    if context.text.strip():
        sections.append(f"Context:\n{context.text.strip()}")
    if context.examples:
        rendered = "\n\n".join(
            f"Question: {example.nl.strip()}\nQuery:\n{example.sparql.strip()}"
            for example in context.examples
            if example.nl.strip() or example.sparql.strip()
        )
        if rendered:
            sections.append(
                "Example questions. Follow their query style when it fits.\n\n"
                + rendered
            )
    sections.append(f"Question: {question.strip()}")
    return "\n\n".join(sections)

