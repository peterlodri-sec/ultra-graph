"""Tests for zerocpu.trit: Q-delta thresholds and A6P-15A bitplanes."""

from ultragraph.zerocpu import trit
from ultragraph.zerocpu.trit import (
    A6P_LEVELS,
    A6P_PLANES,
    MAGNITUDE_MAX,
    MAX_LANES,
    bitplanes,
    decay,
    delta_for_level,
    from_planes,
    pack_ternary,
    plane_parity,
    quantize,
    quantize_many,
    ternary_dot_packed,
    tkand,
    tknot,
    tkor,
    tnand,
    trit_of,
    unpack_ternary,
)


def test_trit_module_imports():
    assert trit.__doc__
    assert trit.A6P_NAME == "A6P-15A"


def test_delta_for_level_spans_the_full_range():
    deltas = [delta_for_level(level) for level in range(A6P_LEVELS)]
    assert deltas[0] == 1 << (A6P_LEVELS - 1)
    assert deltas[-1] == 1
    assert deltas == sorted(deltas, reverse=True)
    assert len(set(deltas)) == A6P_LEVELS


def test_delta_for_level_rejects_out_of_range():
    for bad in (-1, A6P_LEVELS, 99):
        try:
            delta_for_level(bad)
        except ValueError:
            continue
        raise AssertionError(f"level {bad} should have raised")


def test_quantize_boundaries():
    assert quantize(16384, 16384) == 1
    assert quantize(16383, 16384) == 0
    assert quantize(-16384, 16384) == -1
    assert quantize(-16383, 16384) == 0
    assert quantize(0, 16384) == 0


def test_quantize_zero_delta_keeps_silence_at_zero():
    assert quantize(0, 0) == 0
    assert quantize(1, 0) == 1
    assert quantize(-1, 0) == -1


def test_quantize_rejects_negative_delta():
    try:
        quantize(1, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative delta should have raised")


def test_quantize_many_is_quantize_per_lane():
    values = [20000, -20000, 5, -5, 0]
    assert quantize_many(values, 16384) == [1, -1, 0, 0, 0]


def test_decay_halves_and_quarters_toward_zero():
    assert decay(1000, 1) == 500
    assert decay(1000, 2) == 750
    assert decay(-1000, 1) == -500
    assert decay(-1000, 2) == -750
    assert decay(0, 3) == 0
    assert decay(7, 1) == 4


def test_decay_extremes_and_bad_shift():
    assert decay(12345, 63) == 0
    assert decay(-12345, 999) == 0
    try:
        decay(1, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative shift should have raised")


def test_bitplanes_shape_and_sign_plane():
    planes = bitplanes([1, -1, 0])
    assert len(planes) == A6P_PLANES
    assert planes[0] == 0b011  # lanes 0 and 1 have magnitude bit 0
    assert planes[A6P_LEVELS] == 0b010  # only lane 1 is negative
    assert plane_parity(planes) == 0b001


def test_bitplanes_saturate_and_pad():
    planes = bitplanes([10**9, -10**9])
    assert planes[A6P_LEVELS] == 0b10
    assert from_planes(planes)[0] == MAGNITUDE_MAX
    assert from_planes(planes)[1] == -MAGNITUDE_MAX
    short = bitplanes([3])
    padded = bitplanes([3] + [0] * (MAX_LANES - 1))
    assert short[:] == padded[:]


def test_bitplanes_rejects_too_many_lanes():
    try:
        bitplanes([0] * (MAX_LANES + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("17 lanes should have raised")


def test_bitplanes_round_trip():
    values = [-32767, -1, 0, 1, 32767, -12345, 5432, 0, 1, -1, 2, -2, 3, -3, 4, -4]
    planes = bitplanes(values)
    assert len(planes) == A6P_PLANES
    assert from_planes(planes) == values


def test_from_planes_rejects_wrong_width():
    try:
        from_planes([0] * (A6P_PLANES - 1))
    except ValueError:
        pass
    else:
        raise AssertionError("short plane list should have raised")


def test_from_planes_masks_high_bits():
    planes = [0] * A6P_PLANES
    planes[0] = 0xFFFF_FFFF
    assert from_planes(planes)[0] == 1


def test_plane_parity_is_a_16_bit_xor_fold():
    planes = [0xFFFF, 0x0001] + [0] * (A6P_PLANES - 2)
    assert plane_parity(planes) == 0xFFFE
    assert plane_parity([0] * A6P_PLANES) == 0


def test_tnand_is_kleene_negated_min():
    assert tnand(1, 1) == -1
    assert tnand(1, 0) == 0
    assert tnand(0, 0) == 0
    assert tnand(-1, 1) == 1
    assert tnand(-1, -1) == 1
    assert tnand(1, -1) == tnand(-1, 1)


def test_kleene_duals_are_consistent_with_nand():
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            assert tkand(a, b) == -tnand(a, b)
            assert tkor(a, b) == tknot(tkand(tknot(a), tknot(b)))
            assert tknot(tknot(a)) == a
            assert tnand(a, b) == tnand(b, a)


def test_kleene_rejects_non_trits():
    for call in (lambda: tnand(2, 0), lambda: tnand(0, -2), lambda: tknot(1.0)):
        try:
            call()
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("a non-trit must be rejected")


def test_trit_of_classifies_signs():
    assert [trit_of(v) for v in (-7, 0, 9)] == [-1, 0, 1]
    assert trit_of(-0) == 0


def test_pack_unpack_round_trips_eight_lanes():
    lanes = (1, 0, -1, 1, -1, 0, 1, -1)
    assert unpack_ternary(pack_ternary(lanes)) == lanes


def test_pack_unpack_round_trips_a_partial_vector():
    # Fewer than eight lanes zero-pad; the padding contributes nothing.
    lanes = (1, -1, 1)
    assert unpack_ternary(pack_ternary(lanes), len(lanes)) == lanes
    assert unpack_ternary(pack_ternary(lanes))[3:] == (0, 0, 0, 0, 0)


def test_pack_rejects_nine_lanes_and_a_non_trit():
    for bad in (lambda: pack_ternary([1] * 9), lambda: pack_ternary([2]), lambda: pack_ternary([1, -2])):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("packing must refuse a non-ternary lane")


def test_packed_dot_matches_the_naive_sum():
    import random

    rng = random.Random(7)
    for _ in range(2000):
        n = rng.randrange(0, 9)
        a = [rng.choice((-1, 0, 1)) for _ in range(n)]
        b = [rng.choice((-1, 0, 1)) for _ in range(n)]
        assert ternary_dot_packed(pack_ternary(a), pack_ternary(b)) == sum(x * y for x, y in zip(a, b))


def test_packed_dot_corners():
    assert ternary_dot_packed(0, 0) == 0
    assert ternary_dot_packed(pack_ternary([1] * 8), pack_ternary([1] * 8)) == 8
    assert ternary_dot_packed(pack_ternary([-1] * 8), pack_ternary([-1] * 8)) == 8
    assert ternary_dot_packed(pack_ternary([1] * 8), pack_ternary([-1] * 8)) == -8
    # Disjoint supports: zeros contribute nothing, not a sign flip.
    assert ternary_dot_packed(pack_ternary([1, 0, 0, 0, 0, 0, 0, 0]),
                              pack_ternary([0, -1, 0, 0, 0, 0, 0, 0])) == 0


def test_packed_dot_handles_the_mixed_sign_case_that_breaks_naive_bits():
    """`-1` is 0xFF, whose low bit is set: reading it as +1 was a real bug."""
    a = (1, -1, 1, 0, -1, 1, 1, -1)
    b = (1, 1, -1, 0, -1, 1, -1, 1)
    assert ternary_dot_packed(pack_ternary(a), pack_ternary(b)) == sum(x * y for x, y in zip(a, b))
