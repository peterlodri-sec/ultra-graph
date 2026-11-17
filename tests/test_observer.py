"""Tests for zerocpu.observer -- frames, guards and anchored memory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultragraph.zerocpu.observer import (  # noqa: E402
    ANCHOR_VARIANCE,
    DECAY_Q16,
    MAX_DECAY_STEPS,
    RATE_ONE,
    VARIANCE_FLOOR,
    Mahakala,
    ObserverFrame,
    Witness,
    decay,
    decay_pade,
    pade_exp_neg,
    relativity,
)


def test_rate_constants_are_q16():
    assert RATE_ONE == 65536
    assert VARIANCE_FLOOR == round(0.1 * RATE_ONE)
    assert ANCHOR_VARIANCE == 100 * RATE_ONE
    assert 0.95 * RATE_ONE < DECAY_Q16 < 0.96 * RATE_ONE


def test_frame_at_rate_one_tracks_the_world():
    frame = ObserverFrame("world")
    assert frame.local_time(0) == 0
    assert frame.local_time(7) == 7
    assert frame.observe(9) == 9
    assert frame.local_tick == 9


def test_frame_with_offset_shifts_the_labels_not_the_order():
    slow = ObserverFrame("hearth", origin_world=4)
    fast = ObserverFrame("thunderbird")
    assert slow.local_time(4) == 0
    assert fast.local_time(4) == 4
    report = relativity(slow, fast, 10)
    assert (report.local_a, report.local_b, report.delta) == (6, 10, 4)
    assert "delta=4" in str(report)


def test_frozen_frame_does_not_advance():
    frame = ObserverFrame("frozen", rate_q16=0)
    assert frame.local_time(1000) == 0
    assert frame.local_time(1) == 0


def test_frame_rate_is_floor_division():
    half = ObserverFrame("half", rate_q16=RATE_ONE // 2)
    assert half.local_time(1) == 0
    assert half.local_time(3) == 1
    assert half.local_time(-1) == -1


def test_frame_rejects_negative_rate():
    try:
        ObserverFrame("bad", rate_q16=-1)
    except ValueError as exc:
        assert "rate" in str(exc)
    else:
        raise AssertionError("negative rate must be rejected")


def test_decay_is_idempotent_and_monotone():
    value = 10 * RATE_ONE
    one = decay(value, 3)
    assert decay(value, 3) == one
    assert one < value
    assert decay(value, 0) == value
    assert decay(value, -5) == value
    assert decay(value, 6) <= one


def test_decay_stops_at_the_floor():
    assert decay(10 * RATE_ONE, MAX_DECAY_STEPS) == VARIANCE_FLOOR
    assert decay(10 * RATE_ONE, MAX_DECAY_STEPS + 1) == VARIANCE_FLOOR
    assert decay(VARIANCE_FLOOR, 10) == VARIANCE_FLOOR


def test_decay_halves_when_rate_is_one_half():
    value = 1000 << 16
    assert decay(value, 1, decay_q16=RATE_ONE // 2) == value // 2
    assert decay(value, 3, decay_q16=RATE_ONE // 2) == value // 8


def test_mahakala_matches_whole_tokens_only():
    guard = Mahakala()
    assert guard.raise_guard("Mother, father, son.") == ("mother", "father", "son")
    assert guard.raise_guard("seasonal reasoning") == ()
    assert not guard.protects("seasonal reasoning")
    assert guard.protects("my SON")


def test_mahakala_reports_in_chain_order_once():
    guard = Mahakala(guardians=("son", "mother"))
    assert guard.raise_guard("Mother and son and son") == ("son", "mother")


def test_witness_remembers_and_decays():
    memory = Witness()
    memory.remember("plain", 0, span=40)
    peak = memory.variance("plain", 0)
    assert peak == 40 * VARIANCE_FLOOR
    assert memory.variance("plain", 5) < peak
    assert memory.tick(5) == 1


def test_anchor_never_decays():
    memory = Witness()
    memory.remember("monument", 0, anchor=True)
    assert memory.variance("monument", 10**6) == ANCHOR_VARIANCE
    assert memory.anchors() == ("monument",)
    assert memory.tick(10**6) == 1


def test_anchor_outranks_any_span():
    memory = Witness()
    memory.remember("long", 0, span=10**6)
    assert memory.variance("long", 0) == 10 * RATE_ONE
    memory.remember("monument", 0, span=1, anchor=True)
    assert memory.strongest(0, 2) == [("monument", ANCHOR_VARIANCE), ("long", 10 * RATE_ONE)]


def test_plain_memory_falls_out_of_the_live_count():
    memory = Witness()
    memory.remember("fading", 0, span=5)
    assert memory.tick(0) == 1
    assert memory.tick(MAX_DECAY_STEPS) == 0
    assert memory.variance("fading", MAX_DECAY_STEPS) == VARIANCE_FLOOR


def test_witness_rejects_bad_arguments():
    memory = Witness()
    for kwargs in ({"span": -1},):
        try:
            memory.remember("x", 0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kwargs} must be rejected")
    try:
        Witness(decay_q16=RATE_ONE + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("decay above 1.0 must be rejected")


def test_missing_slot_raises():
    memory = Witness()
    memory.remember("here", 0)
    assert "here" in memory and "gone" not in memory
    assert len(memory) == 1
    try:
        memory.variance("gone", 0)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown key must raise KeyError")


def test_pade_exp_matches_the_analytic_value():
    # |e^-x - our value| stays under 2e-5 absolute across the useful range; the
    # residual is 16-bit quantization, not a float approximation.
    for x in (0, 1, 2, 3, 5, 10):
        q = x * RATE_ONE
        ours = pade_exp_neg(q)
        want = round(2.718281828459045 ** -x * RATE_ONE)
        assert abs(ours - want) <= 2 * RATE_ONE // 1000, (x, ours, want)


def test_pade_exp_is_exact_and_total():
    assert pade_exp_neg(0) == RATE_ONE
    assert pade_exp_neg(-99) == RATE_ONE
    assert pade_exp_neg(RATE_ONE) == 24109
    assert pade_exp_neg((40 << 16) + 1) == 0
    assert 0 <= pade_exp_neg(20 << 16) < RATE_ONE


def test_pade_exp_is_monotone():
    values = [pade_exp_neg(x) for x in range(0, 4 * RATE_ONE, RATE_ONE // 4)]
    assert values == sorted(values, reverse=True)
    assert values[0] == RATE_ONE


def test_decay_pade_uses_the_half_life():
    assert decay_pade(10 * RATE_ONE, 0, 10) == 10 * RATE_ONE
    assert decay_pade(10 * RATE_ONE, 10, 10) == 241090
    assert decay_pade(10 * RATE_ONE, 30, 10) < decay_pade(10 * RATE_ONE, 10, 10)
    # A decade-long gap costs the same as a short one: no loop, so no timeout.
    assert decay_pade(10 * RATE_ONE, 10**9, 10) == 0
    try:
        decay_pade(1, 1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero half-life must be rejected")


def test_pade_rejects_non_integers():
    try:
        pade_exp_neg(1.5)
    except TypeError:
        pass
    else:
        raise AssertionError("floats must not enter a deterministic path")
