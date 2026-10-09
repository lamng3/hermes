"""Extract a SPARQL query from model text and check that it parses."""

import re

from rdflib.plugins.sparql import prepareQuery

_FENCE = re.compile(r"```(?:sparql)?\s*([\s\S]*?)```", re.IGNORECASE)
_QUERY = re.compile(
    r"((?:PREFIX\s+\S+\s*<[^>]+>\s*)*(?:SELECT|ASK|CONSTRUCT|DESCRIBE)\b[\s\S]+)",
    re.IGNORECASE,
)
_MODIFIER = re.compile(
    r"^(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET)\b",
    re.IGNORECASE,
)


def extract_sparql(content):
    fenced = _FENCE.search(content or "")
    text = fenced.group(1) if fenced else (content or "")
    match = _QUERY.search(text)
    return (match.group(1) if match else text).strip()


def parse_error(sparql):
    try:
        prepareQuery(sparql)
    except Exception as exc:
        return str(exc).strip()
    return None


def repair_prompt(sparql, error):
    return (
        "Fix this SPARQL query and return only the corrected query.\n"
        "Do not put SELECT inside FILTER. "
        "Put GROUP BY, ORDER BY, and LIMIT after WHERE { }. "
        "When grouping, select the aggregate, as in (MAX(?value) AS ?maxValue).\n\n"
        f"Error: {error}\n"
        f"Query:\n{sparql}"
    )


def lift_modifiers(sparql):
    """Move solution modifiers that were written inside WHERE to after it."""
    match = re.search(r"\bWHERE\b", sparql, re.IGNORECASE)
    if not match:
        return sparql
    start = sparql.find("{", match.end())
    if start < 0:
        return sparql
    depth = 0
    end = None
    for index, char in enumerate(sparql[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end is None:
        return sparql
    kept = []
    moved = []
    inner_depth = 0
    for line in sparql[start + 1 : end].splitlines():
        if inner_depth == 0 and _MODIFIER.match(line.strip()):
            moved.append(line.strip())
        else:
            kept.append(line)
        inner_depth += line.count("{") - line.count("}")
    if not moved:
        return sparql
    body = "\n".join(kept).rstrip()
    return f"{sparql[: start + 1]}\n{body}\n}}\n{chr(10).join(moved)}{sparql[end + 1 :]}"


def prepare_draft(content):
    sparql = extract_sparql(content)
    if not sparql:
        raise ValueError("The model did not return a SPARQL query.")
    return lift_modifiers(sparql)
