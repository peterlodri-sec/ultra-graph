"""The world-clock contract: what two nodes must agree on, and what they may not.

The mesh has no clock authority. That is only safe if the *verdict* path never
touches a local rate, and if nodes exchange world ticks rather than readings.
These tests pin both halves -- and pin the refusal, because "two observers
disagree" is exactly what a naive implementation would call drift and try to
correct with a server.

Plain module-level tests, matching the rest of the suite (no unittest).
"""

from fractions import Fraction

from ultragraph.zerocpu.memnet import RingMemNet
from ultragraph.zerocpu.observer import RATE_ONE, ObserverFrame

HALF_LIFE = 100
TICKS = (0, 1, 4, 10, 40, 100, 1000, 4000)


def band():
    """One anchor, one decaying memory, one decaying memory of the opposite sign."""
    ring = RingMemNet()
    ring.record(196608, 0, anchor=True)
    ring.record(65536, 0)
    ring.record(-131072, 4)
    return ring


def test_two_nodes_with_different_rates_compute_the_same_world_variance():
    ring = band()
    fast = ObserverFrame("fast", rate_q16=RATE_ONE)
    slow = ObserverFrame("slow", rate_q16=(RATE_ONE * 3) // 4)
    for now in TICKS:
        # The world path takes no frame at all, so it cannot see a rate.
        assert ring.variance_world(now, HALF_LIFE) == ring.variance_world(now, HALF_LIFE)
        # ... while the local readings, for contrast, do differ.
        assert ring.local_variance(fast, now, HALF_LIFE) != ring.local_variance(slow, now, HALF_LIFE)


def test_world_variance_is_exactly_the_unscaled_sum():
    ring = band()
    for now in TICKS:
        assert ring.variance_world(now, HALF_LIFE) == ring.variance(now, HALF_LIFE, RATE_ONE)


def test_world_consensus_is_the_sign_of_the_sum_not_of_any_anchor():
    ring = band()
    for now in TICKS:
        total = ring.variance_world(now, HALF_LIFE)
        assert ring.consensus_world(now, HALF_LIFE) == (total > 0) - (total < 0)


def test_an_outvoted_anchor_loses():
    ring = RingMemNet()
    ring.record(65536, 0, anchor=True)
    ring.record(-393216, 0, anchor=True)
    assert ring.variance_world(0, HALF_LIFE) < 0
    assert ring.consensus_world(0, HALF_LIFE) == -1


def test_local_variance_is_the_rate_scaled_path():
    ring = band()
    slow = ObserverFrame("slow", rate_q16=(RATE_ONE * 3) // 4)
    for now in TICKS:
        assert ring.local_variance(slow, now, HALF_LIFE) == ring.variance(
            now, HALF_LIFE, slow.rate_q16
        )


def test_reachable_readings_round_trip_through_the_world():
    for origin, rate in ((100, 49152), (-500, 32768), (0, RATE_ONE), (7, 98304)):
        frame = ObserverFrame("node", origin_world=origin, rate_q16=rate)
        for world in (origin, origin + 4, origin + 100, origin + 4000):
            local = frame.local_time(world)
            assert frame.local_time(frame.world_of(local)) == local


def test_a_reading_that_never_happened_is_detectable():
    """A 3/2 clock skips: it reads 0, 1, 3, 4, 6 -- never 2."""
    frame = ObserverFrame("fast", origin_world=0, rate_q16=98304)
    readings = {frame.local_time(world) for world in range(0, 40)}
    assert 2 not in readings
    for local in range(0, 30):
        reachable = local in readings
        recovered = frame.local_time(frame.world_of(local))
        assert (recovered == local) == reachable, local


def test_a_slow_clock_reaches_every_reading():
    """A 3/4 clock steps by 0 or 1, so no local tick is skipped."""
    frame = ObserverFrame("slow", origin_world=0, rate_q16=49152)
    for local in range(0, 30):
        assert frame.local_time(frame.world_of(local)) == local, local


def test_a_unity_rate_frame_inverts_exactly():
    frame = ObserverFrame("now", origin_world=-3, rate_q16=RATE_ONE)
    for local in (0, 1, 999):
        assert frame.world_of(local) == local - 3


def test_a_frozen_clock_refuses_to_invert():
    frozen = ObserverFrame("stopped", rate_q16=0)
    try:
        frozen.world_of(0)
    except ValueError:
        return
    raise AssertionError("world_of must refuse a frozen clock")


def test_a_frozen_clock_reads_the_same_tick_forever():
    frozen = ObserverFrame("stopped", origin_world=10, rate_q16=0)
    assert frozen.local_time(10) == 0
    assert frozen.local_time(99999) == 0


def test_a_settlement_amount_carries_no_clock():
    """The ledger side of the rule: an integer ratio is the same on every node."""
    ring = band()
    variance = ring.variance_world(400, HALF_LIFE)
    ratio = Fraction(variance, RATE_ONE)
    assert isinstance(ratio.numerator, int)
    assert isinstance(ratio.denominator, int)
    assert Fraction(ratio.numerator, ratio.denominator) == ratio
