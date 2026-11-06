import pytest
from rdflib import Variable

from hermes import KnowledgeGraph, planner
from hermes.planner import Join, Scan

EX = "PREFIX ex: <http://example.org/olympics#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"

SHAPES = {
    "join": "SELECT ?c ?g WHERE { ?g ex:season \"Winter\" . ?c ex:hosted ?g }",
    "filter": "SELECT ?g ?y WHERE { ?g ex:year ?y . ?g ex:season \"Summer\" FILTER(?y > 1950) }",
    "optional": "SELECT ?c ?k WHERE { ?c a ex:City OPTIONAL { ?c ex:country ?k } }",
    "union": "SELECT ?x WHERE { { ?x a ex:City } UNION { ?x a ex:Country } }",
    "minus": "SELECT ?c WHERE { ?c a ex:City MINUS { ?c ex:country ex:USA } }",
    "bind": "SELECT ?g ?d WHERE { ?g ex:year ?y BIND(?y + 4 AS ?d) }",
    "values": "SELECT ?c ?g WHERE { VALUES ?c { ex:Oslo ex:Paris } ?c ex:hosted ?g }",
    "group": "SELECT ?k (COUNT(?g) AS ?n) WHERE { ?c ex:country ?k . ?c ex:hosted ?g } "
             "GROUP BY ?k HAVING(COUNT(?g) > 1)",
    "order": "SELECT ?g ?y WHERE { ?g ex:year ?y } ORDER BY DESC(?y) LIMIT 3",
    "distinct": "SELECT DISTINCT ?k WHERE { ?c ex:country ?k . ?c ex:hosted ?g }",
    "ask": "ASK { ex:Oslo ex:hosted ?g . ?g ex:season \"Winter\" }",
    "label": "SELECT ?c ?y WHERE { ?g ex:year ?y . ?c ex:hosted ?g . ?c ex:country ?k . "
             "?k rdfs:label \"Norway\" . ?g ex:season \"Winter\" FILTER(?y > 1950) }",
    "path": "SELECT ?c WHERE { ?c ex:hosted/ex:season \"Winter\" }",
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_plans_return_the_engines_answers(olympics, name):
    optimized = planner.plan(EX + SHAPES[name], olympics)
    engine = planner.EngineExecutor(olympics)(optimized.written)
    for query in (optimized.written, optimized.query):
        result = planner.PlanExecutor(olympics)(query)
        assert planner.same_answers(result, engine), name


def test_translation_keeps_the_written_order(olympics):
    query = planner.translate(EX + "SELECT * WHERE { ?g ex:year ?y . ?c ex:hosted ?g . ?c a ex:City }")
    join = query.root.children[0]
    assert isinstance(join, Join) and isinstance(join.right, Scan)
    assert [str(s.p).rsplit("#")[-1] for s in (join.left.left, join.left.right, join.right)] == [
        "year", "hosted", "type"]


def test_selections_move_down_and_joins_start_from_the_selective_pattern(olympics):
    optimized = planner.plan(EX + SHAPES["label"], olympics)
    rules = [r.rule for r in optimized.rewrites]
    assert rules[:2] == ["push selections", "order joins"]
    assert optimized.cost_after < optimized.cost_before
    first = optimized.query.root
    while not isinstance(first, Scan):
        first = planner.algebra.children(first)[0]
    assert str(first.o) == "Norway"  # the most selective pattern is scanned first


def test_the_optimized_plan_fetches_less(olympics):
    optimized = planner.plan(EX + SHAPES["label"], olympics)
    run = planner.PlanExecutor(olympics)
    before, after = run(optimized.written).trace, run(optimized.query).trace
    assert after.fetched < before.fetched and after.intermediate < before.intermediate


def test_scans_fetch_only_what_is_used_above_them(olympics):
    optimized = planner.plan(EX + "SELECT ?c WHERE { ?c ex:hosted ?g . ?c ex:country ?k }",
                             olympics)
    assert any(r.rule == "prune scans" for r in optimized.rewrites)
    result = planner.PlanExecutor(olympics)(optimized.query)
    assert len(result.rows) == 19  # one row per hosting, multiplicities kept


def test_plans_render_as_trees_and_formulas(olympics):
    optimized = planner.plan(EX + SHAPES["filter"], olympics)
    text = planner.tree(optimized.query.root, optimized.query.namespaces)
    assert "scan (?g ex:season \"Summer\")" in text and "σ ?y > " in text
    assert planner.formula(optimized.query.root, optimized.query.namespaces).startswith("π_{?g ?y}(")


def test_unsupported_forms_are_reported():
    with pytest.raises(planner.UnsupportedQuery):
        planner.translate(EX + "CONSTRUCT { ?c ex:x ?g } WHERE { ?c ex:hosted ?g }")


def test_results_hold_rdflib_terms(olympics):
    result = planner.PlanExecutor(olympics)(planner.translate(EX + SHAPES["order"]))
    assert [int(row[Variable("y")]) for row in result.rows] == [2012, 2004, 2002]


SCOPE_DATA = """
@prefix : <http://ex.org/> .
:a1 :p :b1 ; :q :d1 ; :r 5 .
:a2 :p :b2 ; :q :d2 ; :r 0 .
:a3 :p :b3 ; :q :d3 .
:a4 :p :b4 ; :q :d4 .
:a5 :p :b5 ; :q :d5 .
:a6 :p :b6 ; :q :d6 .
:z1 :p :y1 .
:z2 :p :y2 .
:z3 :q :y3 .
:z4 :q :y4 .
"""
SCOPE = {
    "a filter on a variable bound only outside its group":
        ("{ { ?a :p ?b . ?a :q ?d . FILTER(?c > 1) } ?a :r ?c }", 0),
    "the same, with the variable bound by OPTIONAL":
        ("{ { ?a :p ?b . ?a :q ?d . FILTER(?c > 1) } { ?a :p ?b OPTIONAL { ?a :r ?c } } }", 0),
    "BOUND on a variable bound only outside its group":
        ("{ { ?a :p ?b . ?a :q ?d . FILTER(!BOUND(?c)) } ?a :r ?c }", 2),
    "a filter in the group that binds its variable":
        ("{ { ?a :p ?b . ?a :q ?d } ?a :r ?c . FILTER(?c > 1) }", 1),
}


@pytest.mark.parametrize("name", sorted(SCOPE))
def test_reordering_joins_keeps_a_filter_in_its_scope(tmp_path, name):
    data = tmp_path / "data.ttl"
    data.write_text(SCOPE_DATA)
    kg = KnowledgeGraph.from_files(str(data), schema=[])
    pattern, rows = SCOPE[name]
    sparql = f"PREFIX : <http://ex.org/>\nSELECT ?a ?c WHERE {pattern}"
    optimized = planner.plan(sparql, kg)
    engine = planner.EngineExecutor(kg)(planner.translate(sparql))
    assert len(engine.rows) == rows
    assert planner.same_answers(planner.PlanExecutor(kg)(optimized.query), engine)
