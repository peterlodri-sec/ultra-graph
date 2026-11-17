"""Tests for zerocpu.router: integer features and the SmartTree decision order."""

from ultragraph.zerocpu import router
from ultragraph.zerocpu.router import (
    FEATURE_KEYS,
    ROUTE_BATCH,
    ROUTE_CAVEMAN,
    ROUTE_DENY,
    ROUTE_MAX_CHARS_BATCH,
    ROUTE_MAX_CHARS_HEAVY,
    ROUTE_MODEL,
    ROUTE_PLUGIN,
    explain,
    features,
    route,
    route_name,
)


def test_router_module_imports():
    assert router.__doc__


def test_feature_keys_and_order():
    feats = features("abc")
    assert tuple(feats) == FEATURE_KEYS
    assert set(feats) == set(FEATURE_KEYS)


def test_features_on_empty_prompt():
    feats = features("")
    assert feats == {
        "chars": 0,
        "words": 0,
        "lines": 0,
        "digits": 0,
        "plugins": 0,
        "marks": 0,
        "heat": 0,
    }


def test_features_counters():
    feats = features("abc 12 @tool /plugin why?!\nsecond")
    assert feats["chars"] == len("abc 12 @tool /plugin why?!\nsecond")
    assert feats["words"] == 6
    assert feats["lines"] == 2
    assert feats["digits"] == 2
    assert feats["plugins"] == 2
    assert feats["marks"] == 2


def test_heat_is_16_bit_and_deterministic():
    first = features("ultra backprop in ternary")
    second = features("ultra backprop in ternary")
    assert first == second
    assert 0 <= first["heat"] <= 0xFFFF


def test_heat_depends_on_position():
    assert features("ab")["heat"] != features("ba")["heat"]


def test_route_empty_is_deny():
    assert route(features("")) == ROUTE_DENY
    assert route_name(ROUTE_DENY) == "ROUTE_DENY"


def test_route_plugin_token_wins():
    for prompt in ("@plugin hi", "/run something", "x " * 3000 + "@heavy"):
        assert route(features(prompt)) == ROUTE_PLUGIN


def test_route_length_thresholds():
    short = "a" * ROUTE_MAX_CHARS_BATCH
    just_over = "a" * (ROUTE_MAX_CHARS_BATCH + 1)
    heavy = "a" * ROUTE_MAX_CHARS_HEAVY
    assert route(features(short)) == ROUTE_CAVEMAN
    assert route(features(just_over)) == ROUTE_BATCH
    assert route(features(heavy)) == ROUTE_MODEL


def test_route_marks_escalate_only_past_batch():
    long_enough = "b" * (ROUTE_MAX_CHARS_BATCH + 10) + "?!?!"
    assert route(features(long_enough)) == ROUTE_MODEL
    short_and_sharp = "b" * 20 + "?!?!"
    assert route(features(short_and_sharp)) == ROUTE_CAVEMAN


CORPUS = (
    "",
    "hi",
    "@plugin hi",
    "/run it",
    "why?!",
    "why?!?!",
    "a" * ROUTE_MAX_CHARS_BATCH,
    "a" * (ROUTE_MAX_CHARS_BATCH + 1),
    "a" * (ROUTE_MAX_CHARS_HEAVY - 1),
    "a" * ROUTE_MAX_CHARS_HEAVY,
    "a" * (ROUTE_MAX_CHARS_HEAVY + 500),
    "ultra backprop in ternary, 12 layers?\nsecond line!",
)


def test_explain_always_ends_on_the_real_decision():
    for prompt in CORPUS:
        feats = features(prompt)
        decision = route(feats)
        reasons = explain(feats)
        assert reasons, prompt
        assert route_name(decision) in reasons[-1], (prompt, reasons[-1])


def test_route_name_rejects_unknown():
    try:
        route_name(99)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown route code should have raised")


def test_route_tolerates_partial_feature_dicts():
    assert route({"chars": 0}) == ROUTE_DENY
    assert route({"chars": 10}) == ROUTE_CAVEMAN
