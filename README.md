# nl2sparql

[Documentation](https://lamng3.github.io/nl2sparql-docs/)

Training-free natural language to SPARQL. A question, a Turtle ontology, and
optional context go in. One SPARQL query comes out. The model is used as a
generator. Nothing here is trained on query pairs.

```bash
pip install -e .
nl2sparql generate \
  --ontology GeoOutage.ttl \
  --question "Which counties have the highest number of outages?" \
  --context notes.txt \
  --examples pairs.jsonl
```

`pairs.jsonl` is one JSON object per line, with `nl` and `sparql` fields.

Generation uses a local [Ollama](https://ollama.com) model at
`http://127.0.0.1:11434`. Install one if needed (`ollama pull llama3.1`).
The client prefers `llama3.1`, then `mistral`, then `phi3`, then whichever
model is installed. Override it with `ONTOCHECK_OLLAMA_MODEL`, and the server
with `ONTOCHECK_OLLAMA_BASE_URL`.

To use an OpenAI-compatible API instead, set `ONTOCHECK_LLM_PROVIDER=openai`
and `ONTOCHECK_LLM_API_KEY`. Optional `ONTOCHECK_LLM_BASE_URL` and
`ONTOCHECK_LLM_MODEL`.

From Python:

```python
from nl2sparql import Context, Example, generate

result = generate(
    "Which counties have the highest number of outages?",
    "GeoOutage.ttl",
    context=Context(
        text="Counties are administrative regions.",
        examples=[Example(nl="List outage records.", sparql="SELECT ?rec WHERE { ?rec a geooutageonto:OutageRecord }")],
    ),
)
print(result["sparql"])
```

## Pipeline

The draft is produced from the ontology vocabulary, then passed through a
fixed list of stages in `nl2sparql.stages`. Today these stages return the
draft unchanged:

- Guardrails: reject or rewrite queries that leave the ontology vocabulary.
- Query guided: steer a draft using a target query shape or a previous SPARQL query.
- Context guided: retrieve and place example questions, beyond stuffing them into the prompt.
- Chase and backchase: reformulate with ontology constraints, then minimize.

Parse checking and one repair call run after those stages. Later research
replaces a stage in that list.
