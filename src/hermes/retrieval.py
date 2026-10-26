"""Retrieval: rank documents (ontology terms, values, examples) for a question.

A `Retriever` is fitted on documents and ranks them for a text. `Lexical` is
BM25 over words, with camelCase split, case and plurals folded; it needs
nothing and runs anywhere, including the browser. `Embeddings` ranks by
cosine similarity of vectors from any OpenAI-compatible or Ollama embeddings
endpoint, cached per text. `retriever("lexical" | "embeddings", ...)` builds
one.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import httpx

BM25_K1 = 1.2
BM25_B = 0.75
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "and", "or", "is", "are",
    "was", "were", "be", "do", "does", "did", "which", "what", "who", "whom", "whose", "how",
    "many", "much", "list", "give", "show", "me", "all", "any", "our", "we", "us", "there", "that",
    "this", "these", "those", "it", "its", "from", "as", "has", "have", "had",
}


@dataclass
class Document:
    """A thing to retrieve: `text` is what it is matched on; `item` is what it stands for."""

    text: str
    item: object = None
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    document: Document
    score: float


class Retriever:
    def fit(self, documents):
        raise NotImplementedError

    def rank(self, text, k=10):
        raise NotImplementedError


def words(text):
    """Lower-case word stems: camelCase and snake_case split, plurals folded, stopwords out."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text))
    out = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS:
            continue
        out.append(stem(word))
    return out


def stem(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


class Lexical(Retriever):
    """BM25 (Robertson and Zaragoza, 2009) over the documents' words."""

    def fit(self, documents):
        self.documents = list(documents)
        self.terms = [Counter(words(d.text)) for d in self.documents]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.average = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        frequency = Counter()
        for terms in self.terms:
            frequency.update(terms.keys())
        n = len(self.documents)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in frequency.items()}
        return self

    def rank(self, text, k=10):
        query = words(text)
        scored = []
        for index, terms in enumerate(self.terms):
            score = 0.0
            for word in query:
                tf = terms.get(word)
                if not tf:
                    continue
                norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[index] / (self.average or 1))
                score += self.idf[word] * tf * (BM25_K1 + 1) / (tf + norm)
            if score > 0:
                scored.append(Hit(self.documents[index], score))
        scored.sort(key=lambda hit: -hit.score)
        return scored[:k]


class Embeddings(Retriever):
    """Cosine similarity of embeddings from an OpenAI-compatible or Ollama endpoint."""

    def __init__(self, model="nomic-embed-text", base_url="http://127.0.0.1:11434/v1",
                 api_key="", batch=64, timeout=60.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch = batch
        self.timeout = timeout
        self.cache = {}

    def embed(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for start in range(0, len(missing), self.batch):
            chunk = missing[start:start + self.batch]
            response = httpx.post(
                f"{self.base_url}/embeddings", timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key or 'none'}"},
                json={"model": self.model, "input": chunk},
            )
            response.raise_for_status()
            for text, row in zip(chunk, response.json()["data"]):
                self.cache[text] = _unit(row["embedding"])
        return [self.cache[t] for t in texts]

    def fit(self, documents):
        self.documents = list(documents)
        self.vectors = self.embed([d.text for d in self.documents])
        return self

    def rank(self, text, k=10):
        [query] = self.embed([text])
        scored = [Hit(d, sum(a * b for a, b in zip(query, v)))
                  for d, v in zip(self.documents, self.vectors)]
        scored.sort(key=lambda hit: -hit.score)
        return scored[:k]


def _unit(vector):
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


RETRIEVERS = {"lexical": Lexical, "embeddings": Embeddings}


def retriever(kind="lexical", **options):
    """Factory: a retriever by name."""
    try:
        return RETRIEVERS[kind](**options)
    except KeyError:
        raise ValueError(f"Unknown retriever {kind!r}; use one of {', '.join(RETRIEVERS)}.") from None
