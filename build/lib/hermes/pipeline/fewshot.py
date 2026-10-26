"""Few-shot examples: question and query pairs over the graph, and choosing them per question.

A store holds examples in a JSON Lines file per graph (data/fewshot/<graph>.jsonl),
each with its source: "synthetic" for pairs generated from the graph itself
(hermes.pipeline.synthetic), never from benchmark questions, so they cannot
leak a test answer. The selector ranks them by the similarity of their
questions to the new one and keeps the best `k` (two by default).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hermes.retrieval import Document, Lexical


@dataclass
class Example:
    question: str
    sparql: str
    source: str = "synthetic"
    meta: dict = field(default_factory=dict)


class FewShotStore:
    def __init__(self, examples=()):
        self.examples = list(examples)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return cls()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return cls(Example(**row) for row in rows)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(asdict(e), ensure_ascii=False) + "\n"
                                for e in self.examples), encoding="utf-8")

    def __len__(self):
        return len(self.examples)


class FewShotSelector:
    def __init__(self, store, k=2, retriever=None):
        self.store = store
        self.k = k
        self.retriever = (retriever or Lexical()).fit(
            [Document(e.question, e) for e in store.examples]) if len(store) else None

    def __call__(self, question):
        if self.retriever is None or self.k <= 0:
            return []
        return [hit.document.item for hit in self.retriever.rank(question, self.k)]


def store_path(graph_name, root="data/fewshot"):
    return Path(root) / f"{graph_name}.jsonl"
