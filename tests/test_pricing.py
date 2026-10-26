import pytest

from hermes.baselines.explorer import Explorer, Plan, parse_action
from hermes.baselines.pricing import Pricer
from hermes.pricing.seconds import SecondsModel
from hermes.pricing.statistics import Statistics, qerror, statistics_for


def plan(*actions):
    result = Plan()
    for data in actions:
        result = parse_action(data).apply(result)
    return result


def triple(text):
    return {"op": "triple", "pattern": text}


@pytest.fixture
def pricer(olympics):
    return Pricer.for_kg(olympics, Explorer(olympics).expand)


def count(kg, explorer, p):
    from hermes.baselines.explorer import render

    inner = render(p)
    rows = kg.query(f"{explorer.header}SELECT (COUNT(*) AS ?n) WHERE {{ {{ {inner} }} }}")
    return int(rows["results"]["bindings"][0]["n"]["value"])


def test_local_statistics_count_predicates_and_classes(olympics):
    stats = statistics_for(olympics)
    hosted = stats.predicate("http://example.org/olympics#hosted")
    assert (hosted.triples, hosted.subjects, hosted.objects) == (19, 13, 19)
    assert stats.instances("http://example.org/olympics#OlympicGames") == 19
    assert stats.predicate("http://example.org/olympics#nothing").triples == 0
    assert statistics_for(olympics) is stats


def test_single_patterns_are_priced_exactly(pricer):
    assert pricer.estimate(plan(triple("?c ex:hosted ?g"))).rows == 19
    assert pricer.estimate(plan(triple("?g a ex:OlympicGames"))).rows == 19
    # A constant object keeps 1/distinct objects of the predicate.
    assert pricer.estimate(plan(triple('?g ex:season "Winter"'))).rows == pytest.approx(19 / 2)


def test_joins_are_estimated_within_a_small_q_error(olympics, pricer):
    explorer = Explorer(olympics)
    for p in [
        plan(triple("?c ex:hosted ?g"), triple('?g ex:season "Winter"')),
        plan(triple("?c ex:hosted ?g"), triple("?c ex:country ex:USA")),
        plan(triple("?c ex:hosted ?g"), triple("?g ex:year ?y"), triple("?c ex:country ?k")),
    ]:
        assert qerror(pricer.estimate(p).rows, count(olympics, explorer, p)) < 3


def test_cost_grows_with_work_and_cartesian_products_cost_most(olympics, pricer):
    one = pricer.estimate(plan(triple("?c ex:hosted ?g")))
    joined = pricer.estimate(plan(triple("?c ex:hosted ?g"), triple("?c ex:country ?k")))
    cartesian = pricer.estimate(plan(triple("?c ex:hosted ?g"), triple("?x ex:country ?k")))
    assert one.cost < joined.cost < cartesian.cost
    countries = statistics_for(olympics).predicate("http://example.org/olympics#country").triples
    assert cartesian.rows == pytest.approx(19 * countries)


def test_modifiers_shape_the_estimate(pricer):
    base = [triple("?c ex:hosted ?g")]
    assert pricer.estimate(plan(*base, {"op": "limit", "n": 3})).rows == 3
    assert pricer.estimate(plan(*base, {"op": "ask"})).rows == 1
    assert pricer.estimate(plan(*base, {"op": "aggregate", "expr": "COUNT(?g)",
                                        "alias": "?n"})).rows == 1
    grouped = pricer.estimate(plan(*base, {"op": "group", "vars": ["?c"]}))
    assert grouped.rows == 13
    assert pricer.estimate(plan(*base, {"op": "filter", "expr": "?g = ex:Winter1932"})).rows < 2


def test_endpoints_are_counted_lazily_and_unknown_terms_are_reported(olympics):
    stats = Statistics(olympics, eager=False)
    assert stats.predicates == {} and stats.everything() is None
    pricer = Pricer(stats, Explorer(olympics).expand)
    assert pricer.estimate(plan(triple("?c ex:hosted ?g"))).rows == 19
    assert "http://example.org/olympics#hosted" in stats.predicates
    assert pricer.estimate(plan(triple("?c ?p ?g"))).unknown == ["?p"]


def test_seconds_are_fitted_from_observed_executions():
    model = SecondsModel(intercept=1.0, slope=0.0)
    assert model.predict(100) == 1.0  # the prior, until enough points
    for cost in (10, 20, 30, 40):
        model.observe(cost, 0.01 + 0.001 * cost)
    assert model.predict(100) == pytest.approx(0.11)


def test_ranking_price_does_not_depend_on_measured_time(pricer):
    estimate = pricer.estimate(plan(triple("?c ex:hosted ?g")))
    before = pricer.nominal(estimate)
    for seconds in (5.0, 9.0, 7.0):  # a slow machine
        pricer.observe(estimate, seconds)
    assert pricer.nominal(estimate) == before
    assert pricer.seconds.predict(estimate.cost) > before
