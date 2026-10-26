"""The prompt builder: instructions, the ontology slice, examples, and the question."""

INSTRUCTIONS = """\
Write one SPARQL query that answers the question over the knowledge graph described below.
Use only the prefixes, classes, properties, values and entities listed; write the IRIs of
the entities as listed. Prefer the properties whose subject and object classes fit the
question. Return only the query, in a ```sparql block."""

REPAIR = """\
The query below does not work: fix it. Keep what it asks for; change only what the
problems name. Return only the fixed query, in a ```sparql block."""


class PromptBuilder:
    """Builder: add the parts a prompt needs, in any order; `build` lays them out."""

    def __init__(self):
        self.parts = {}

    def instructions(self, text=INSTRUCTIONS):
        self.parts["instructions"] = text
        return self

    def prefixes(self, header):
        self.parts["prefixes"] = "Prefixes:\n" + header.rstrip()
        return self

    def ontology(self, text):
        self.parts["ontology"] = text.rstrip()
        return self

    def examples(self, examples):
        if examples:
            self.parts["examples"] = "Examples over this graph:\n\n" + "\n\n".join(
                f"Question: {e.question}\n```sparql\n{e.sparql.strip()}\n```" for e in examples)
        return self

    def question(self, text):
        self.parts["question"] = f"Question: {text.strip()}"
        return self

    def problems(self, sparql, problems):
        self.parts["instructions"] = REPAIR
        self.parts["problems"] = ("Query:\n```sparql\n" + sparql.strip() + "\n```\nProblems:\n"
                                  + "\n".join(f"- {p}" for p in problems))
        return self

    def build(self):
        order = ("instructions", "prefixes", "ontology", "examples", "problems", "question")
        return "\n\n".join(self.parts[key] for key in order if key in self.parts)
