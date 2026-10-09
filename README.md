# Hermes

Hermes carries a question across to the graph. It is a Python library for
turning natural-language questions into SPARQL queries over an ontology. Named
for the Greek messenger and interpreter, it is a framework, not a single method. Every approach is a *system* behind one interface, so
you can import the package, run one system, and swap in another without
changing your code. Systems register by name; the one that ships today is
`training-free`, which needs no query-pair training.

Install name: `hermes-sparql`. Import name: `hermes`.

[Documentation](https://lamng3.github.io/hermes-docs/)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Quick start

A question, a Turtle ontology, and optional context go in. One SPARQL query
comes out.

```bash
git clone https://github.com/lamng3/hermes.git
cd hermes
pip install -e .
hermes generate \
  --ontology GeoOutage.ttl \
  --question "Which counties have the highest number of outages?" \
  --context notes.txt \
  --examples pairs.jsonl
```

`pairs.jsonl` is one JSON object per line, with `nl` and `sparql` fields.

Generation uses a local [Ollama](https://ollama.com) model at
`http://127.0.0.1:11434`. Install one if needed (`ollama pull llama3.1`).
The client prefers `llama3.1`, then `mistral`, then `phi3`, then whichever
model is installed. Override it with `HERMES_OLLAMA_MODEL`, and the server
with `HERMES_OLLAMA_BASE_URL`.

To use an OpenAI-compatible API instead, set `HERMES_LLM_PROVIDER=openai`
and `HERMES_LLM_API_KEY`. Optional `HERMES_LLM_BASE_URL` and
`HERMES_LLM_MODEL`.

From Python:

```python
from hermes import Context, Example, ask

result = ask(
    "Which counties have the highest number of outages?",
    "GeoOutage.ttl",
    context=Context(
        text="Counties are administrative regions.",
        examples=[Example(nl="List outage records.", sparql="SELECT ?rec WHERE { ?rec a geooutageonto:OutageRecord }")],
    ),
    system="training-free",
)
print(result["sparql"])
```

## Systems

A system is any callable `(question, ontology, context) -> {"sparql": ...}`.
Register it by name, then run it with `ask`:

```python
import hermes

@hermes.systems.register("my-system")
def my_system(question, ontology, context):
    return {"sparql": "SELECT * WHERE { ?s ?p ?o } LIMIT 1"}

hermes.systems.available()   # ['my-system', 'training-free']
hermes.ask("...", "onto.ttl", system="my-system")
```

`hermes systems` lists the registered names, and `hermes generate --system NAME`
runs one from the command line.

## The training-free system

The draft is produced from the ontology vocabulary, then passed through a
fixed list of stages in `hermes.stages`. Today these stages return the
draft unchanged:

- Guardrails: reject or rewrite queries that leave the ontology vocabulary.
- Query guided: steer a draft using a target query shape or a previous SPARQL query.
- Context guided: retrieve and place example questions, beyond stuffing them into the prompt.
- Chase and backchase: reformulate with ontology constraints, then minimize.

Parse checking and one repair call run after those stages. Later research
replaces a stage in that list.

## License

[CC BY 4.0](LICENSE). Copyright (c) 2026 Lam Nguyen.
