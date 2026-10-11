"""Entity linking: map question text to graph nodes.

`LabelIndex` reads every label once and matches in memory; it suits local and
small graphs. `ExactLabel` sends the question's n-grams to the store as label
literals in one VALUES query; it suits large endpoints such as DBpedia, where
scanning labels is too slow. `linker_for` picks one from the store type.
"""

import re
import threading

LABELS = (
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#altLabel",
)
_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "which", "what", "who", "whom",
    "how", "many", "much", "have", "has", "had", "with", "from", "that", "this",
    "list", "show", "give", "all", "any", "does", "did", "der", "die", "das",
    "und", "wer", "wie", "los", "las", "que", "del", "por",
}


class Linker:
    def link(self, question):
        """IRIs whose label matches an n-gram of the question exactly."""
        raise NotImplementedError

    def search(self, text, limit=10):
        """(IRI, label) pairs for a free-text mention."""
        raise NotImplementedError

    def lookup(self, name):
        """IRIs whose label is exactly this name."""
        raise NotImplementedError


class LabelIndex(Linker):
    def __init__(self, kg):
        self.kg = kg
        self._names = None
        self._labels = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._names is None:
                self._index()

    def _index(self):
        names, labels = {}, {}
        self._names, self._labels = names, labels
        predicates = " ".join(f"<{iri}>" for iri in LABELS)
        result = self.kg.try_query(
            f"SELECT ?s ?l WHERE {{ VALUES ?p {{ {predicates} }} ?s ?p ?l FILTER(isIRI(?s)) }}"
        )
        for row in result["results"]["bindings"]:
            iri, label = row["s"]["value"], row["l"]["value"]
            self._names.setdefault(normalize(label), set()).add(iri)
            self._labels.setdefault(iri, []).append(label)
        result = self.kg.try_query("SELECT DISTINCT ?s WHERE { ?s a ?type FILTER(isIRI(?s)) }")
        for row in result["results"]["bindings"]:
            iri = row["s"]["value"]
            self._names.setdefault(normalize(local_name(iri)), set()).add(iri)
            self._labels.setdefault(iri, [local_name(iri)])

    def link(self, question, longest=5):
        self._load()
        found = []
        for gram in ngrams(question, longest):
            for iri in sorted(self._names.get(normalize(gram), ())):
                if iri not in found:
                    found.append(iri)
        return found

    def search(self, text, limit=10):
        self._load()
        wanted = normalize(text).split()
        if not wanted:
            return []
        exact = self._names.get(" ".join(wanted), set())
        found = self._matching(wanted, exact)
        if not found:
            content = [word for word in wanted if len(word) >= 3 and word not in _STOPWORDS]
            found = self._matching(content, exact, partial=True) if content else []
        return found[:limit]

    def _matching(self, wanted, exact, partial=False):
        """Labels holding every wanted word, or (partial) the most of them."""
        scored = []
        for iri, labels in self._labels.items():
            for label in labels:
                words = normalize(label).split()
                hits = sum(1 for word in wanted if word in words)
                if hits == len(wanted) or (partial and hits):
                    scored.append((-hits, iri not in exact, len(words), label, iri))
        best = {}
        for row in sorted(scored):
            best.setdefault(row[-1], row)
        return [(iri, label) for *_rest, label, iri in sorted(best.values())]

    def lookup(self, name):
        self._load()
        return set(self._names.get(normalize(name), ()))


class ExactLabel(Linker):
    def __init__(self, kg, languages=("en",)):
        self.kg = kg
        self.languages = languages

    def _match(self, texts):
        literals = []
        for text in texts:
            escaped = text.replace("\\", "\\\\").replace('"', '\\"')
            literals.append(f'"{escaped}"')
            literals.extend(f'"{escaped}"@{language}' for language in self.languages)
        if not literals:
            return []
        query = (
            "SELECT DISTINCT ?s ?l WHERE { VALUES ?l { " + " ".join(literals) + " } "
            "?s <http://www.w3.org/2000/01/rdf-schema#label> ?l }"
        )
        rows = self.kg.try_query(query)["results"]["bindings"]
        return [(row["s"]["value"], row["l"]["value"]) for row in rows]

    def link(self, question, longest=5):
        grams = [gram for gram in ngrams(question, longest) if gram[:1].isupper()]
        return sorted({iri for iri, _label in self._match(grams[:200])})

    def search(self, text, limit=10):
        variants = {text, text.title(), text.capitalize()}
        return self._match(sorted(variants))[:limit]

    def lookup(self, name):
        return {iri for iri, _label in self._match([name])}


def linker_for(kg):
    if kg.store.local:
        return LabelIndex(kg)
    return ExactLabel(kg)


def ngrams(text, longest=5):
    """Original-case n-grams, longest first, skipping stopwords and short ones."""
    words = re.findall(r"[\w'.@%-]+", text)
    words = [word.strip(".'") for word in words if word.strip(".'")]
    grams = []
    for size in range(min(longest, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            gram = " ".join(words[start : start + size])
            if len(gram) < 3 or gram.lower() in _STOPWORDS:
                continue
            grams.append(gram)
    return grams


def normalize(text):
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def local_name(iri):
    value = str(iri)
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    return value.rstrip("/").rsplit("/", 1)[-1]
