"""Tests for zerocpu.memnet: mergeable associative memory."""

from ultragraph.zerocpu import memnet
from ultragraph.zerocpu.memnet import (
    MEMNET_SLOTS,
    MemNet,
    MemNetEntry,
    key_id,
)


def test_memnet_module_imports():
    assert memnet.__doc__
    assert memnet.MEMNET_VERSION == 1


def test_key_id_is_stable_fnv1a():
    assert key_id("") == 0x811C9DC5
    assert key_id("a") == 0xE40C292C
    assert key_id("fit") == key_id("fit")
    assert key_id("fit") != key_id("fit ")


def test_put_then_get():
    net = MemNet()
    entry = net.put("fit", 3)
    assert isinstance(entry, MemNetEntry)
    assert entry.slot == key_id("fit") % MEMNET_SLOTS
    assert entry.seq == 0
    assert net.get("fit") == 3
    assert net.get("missing") is None
    assert len(net) == 1
    assert "fit" in net


def test_put_overwrites_with_a_higher_sequence():
    net = MemNet()
    net.put("k", 1)
    second = net.put("k", 2)
    assert second.seq == 1
    assert net.get("k") == 2
    assert len(net) == 1


def test_put_rejects_bad_values():
    net = MemNet()
    for bad in ("", None, 1 << 40, -(1 << 40)):
        try:
            net.put("k", bad)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"value {bad!r} should have been rejected")


def test_entries_are_sorted_not_insertion_ordered():
    net = MemNet()
    for key in ("z", "a", "m", "b"):
        net.put(key, 1)
    entries = net.entries()
    assert entries == sorted(entries, key=lambda e: (e.slot, e.key_id, e.seq))


def test_merge_takes_the_higher_sequence():
    left = MemNet()
    right = MemNet()
    left.put("shared", 10)
    right.put("shared", 20)
    right.put("shared", 30)
    changed = left.merge(right)
    assert changed == 1
    assert left.get("shared") == 30


def test_merge_is_idempotent():
    left = MemNet()
    right = MemNet()
    left.put("a", 1)
    right.put("b", 2)
    assert left.merge(right) == 1
    assert left.merge(right) == 0
    assert len(left) == 2


def test_merge_is_order_independent():
    a = MemNet()
    b = MemNet()
    a.put("x", 5)
    a.put("y", 6)
    b.put("x", 5)  # same seq, different... no: same value
    b.put("z", 7)
    ab = MemNet.from_hex(a.to_hex())
    ba = MemNet.from_hex(b.to_hex())
    ab.merge(b)
    ba.merge(a)
    assert ab.to_hex() == ba.to_hex()


def test_merge_breaks_ties_by_value_so_order_cannot_matter():
    a = MemNet()
    b = MemNet()
    a.put("k", 1)
    b.put("k", 9)
    ab = MemNet.from_hex(a.to_hex())
    ba = MemNet.from_hex(b.to_hex())
    assert ab.merge(b) == 1
    assert ba.merge(a) == 0
    assert ab.get("k") == 9
    assert ba.get("k") == 9


def test_hex_round_trip():
    net = MemNet()
    net.put("fit", 0)
    net.put("parity", 0xBEEF - 0x8000)
    net.put("collapse", -1)
    text = net.to_hex()
    clone = MemNet.from_hex(text)
    assert clone.to_hex() == text
    assert clone.get("fit") == 0
    assert clone.get("parity") == 0xBEEF - 0x8000
    assert clone.get("collapse") == -1


def test_from_hex_rejects_junk():
    for junk in ("a:b", "0:0:0:0,oops", "999:0:0:0", "0:0:0:-1", "0:0:0:0,0:0:0:0"):
        try:
            MemNet.from_hex(junk)
        except ValueError:
            continue
        raise AssertionError(f"{junk!r} should have been rejected")


def test_from_hex_empty_is_empty():
    assert len(MemNet.from_hex("")) == 0
