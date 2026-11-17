"""Tests for zerocpu.mesh: the announce frame and peer merging."""

from ultragraph.zerocpu import mesh
from ultragraph.zerocpu.mesh import (
    HEADER_FIXED,
    MAGIC,
    MAX_NAME,
    PROTO_PORT,
    PROTO_VERSION,
    Peer,
    merge_peers,
    pack_announce,
    parse_announce,
)


def test_mesh_module_imports():
    assert mesh.__doc__
    assert MAGIC == b"UGRD"
    assert PROTO_VERSION == 1
    assert HEADER_FIXED == 12


def test_pack_layout_is_fixed_and_little_endian():
    frame = pack_announce("hearth", PROTO_PORT, 7)
    assert frame[:4] == MAGIC
    assert frame[4] == PROTO_VERSION
    assert frame[5:7] == PROTO_PORT.to_bytes(2, "little")
    assert frame[7:11] == (7).to_bytes(4, "little")
    assert frame[11] == len(b"hearth")
    assert frame[HEADER_FIXED:] == b"hearth"
    assert len(frame) == HEADER_FIXED + len(b"hearth")


def test_round_trip():
    peer = parse_announce(pack_announce("thunderbird", 8424, 0xDEADBEEF))
    assert peer == Peer(name="thunderbird", port=8424, seq=0xDEADBEEF)


def test_name_length_bounds():
    assert parse_announce(pack_announce("n" * MAX_NAME, 1, 0)) is not None
    for bad in ("", "n" * (MAX_NAME + 1)):
        try:
            pack_announce(bad, 1, 0)
        except ValueError:
            continue
        raise AssertionError(f"name of {len(bad)} bytes should have been rejected")


def test_pack_rejects_bad_port_and_seq():
    for port in (-1, 0x10000):
        try:
            pack_announce("ok", port, 0)
        except ValueError:
            continue
        raise AssertionError(f"port {port} should have been rejected")
    for seq in (-1, 0x1_0000_0000):
        try:
            pack_announce("ok", 1, seq)
        except ValueError:
            continue
        raise AssertionError(f"seq {seq} should have been rejected")


def test_pack_rejects_nul_in_name():
    try:
        pack_announce("a\x00b", 1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("NUL in name should have been rejected")


def test_parse_returns_none_on_truncation():
    frame = pack_announce("hearth", PROTO_PORT, 1)
    for cut in range(len(frame)):
        assert parse_announce(frame[:cut]) is None, cut


def test_parse_returns_none_on_bad_magic_or_version():
    frame = bytearray(pack_announce("hearth", PROTO_PORT, 1))
    frame[0:4] = b"XXXX"
    assert parse_announce(bytes(frame)) is None
    frame = bytearray(pack_announce("hearth", PROTO_PORT, 1))
    frame[4] = 2
    assert parse_announce(bytes(frame)) is None


def test_parse_returns_none_on_length_lie():
    frame = pack_announce("hearth", PROTO_PORT, 1)
    lying = bytearray(frame)
    lying[11] = 10  # name_len says ten bytes, frame carries six
    assert parse_announce(bytes(lying)) is None


def test_parse_returns_none_on_invalid_utf8():
    body = b"\xff\xfe"
    frame = bytearray(pack_announce("aa", 1, 0))
    frame[11] = len(body)
    assert parse_announce(bytes(frame[:HEADER_FIXED]) + body) is None


def test_parse_survives_deterministic_fuzz():
    state = 0x12345678
    for _ in range(200):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        length = state % 48
        blob = bytes((state >> (8 * (i % 4))) & 0xFF for i in range(length))
        peer = parse_announce(blob)
        assert peer is None or isinstance(peer, Peer)


def test_parse_rejects_non_bytes():
    assert parse_announce("UGRD") is None
    assert parse_announce(None) is None


def test_merge_prefers_higher_seq():
    old = Peer("hearth", PROTO_PORT, 1)
    new = Peer("hearth", PROTO_PORT, 2)
    assert merge_peers([old], new) == [new]
    assert merge_peers([new], old) == [new]


def test_merge_is_sorted_and_order_independent():
    peers = [Peer("zeta", 1, 1), Peer("alpha", 1, 1), Peer("hearth", 1, 1)]
    forward = peers[0]
    merged = []
    for peer in peers:
        merged = merge_peers(merged, peer)
    assert [peer.name for peer in merged] == ["alpha", "hearth", "zeta"]
    assert merged == merge_peers([peers[2], peers[1]], forward)


def test_merge_does_not_mutate_current():
    current = [Peer("hearth", 1, 1)]
    snapshot = list(current)
    merge_peers(current, Peer("thunderbird", 1, 1))
    assert current == snapshot


def test_merge_fills_a_missing_source_without_reordering():
    current = [Peer("hearth", 1, 1)]
    heard = Peer("hearth", 1, 1, source="10.0.0.7")
    merged = merge_peers(current, heard)
    assert merged == [Peer("hearth", 1, 1, source="10.0.0.7")]
