"""The mesh announce frame: twelve fixed header bytes plus a short name.

One datagram, no handshake, no external crate:

    "UGRD" | version u8 | port u16 | seq u32 | name_len u8 | name (utf-8)

Everything is little-endian and fixed width, so the frame is byte-stable across
Python and Rust.  ``parse_announce`` never raises: mystery bytes from the LAN
are a normal condition, not an error, and it returns ``None`` for anything that
is not exactly one well-formed frame.

The socket path lives in ``rust/zerocpu/src/bin/peers.rs``; this module is the
protocol, which is what the tests and the fixtures pin down.
"""

from __future__ import annotations

import argparse
import binascii
import json
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass

MAGIC = b"UGRD"
PROTO_VERSION = 1
PROTO_PORT = 8423
MAX_NAME = 32
MASK32 = 0xFFFFFFFF

#: "UGRD" + version + port + seq + name length = 12 bytes.
HEADER_FIXED = 12
_HEADER = struct.Struct("<4sBHIB")
assert _HEADER.size == HEADER_FIXED, _HEADER.size


@dataclass(frozen=True)
class Peer:
    """A node heard from.  ``source`` is runtime-only and never on the wire."""

    name: str
    port: int
    seq: int
    source: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "port": self.port, "seq": self.seq, "source": self.source}


def pack_announce(name: str, port: int, seq: int) -> bytes:
    """Encode one announce frame.  Raises ``ValueError`` on anything unencodable."""
    if not isinstance(name, str):
        raise TypeError(f"name must be str, got {type(name).__name__}")
    raw = name.encode("utf-8")
    if not 1 <= len(raw) <= MAX_NAME:
        raise ValueError(f"name must be 1..{MAX_NAME} bytes, got {len(raw)}")
    if b"\x00" in raw:
        raise ValueError("name must not contain NUL")
    if not isinstance(port, int) or isinstance(port, bool):
        raise TypeError(f"port must be int, got {type(port).__name__}")
    if not 0 <= port <= 0xFFFF:
        raise ValueError(f"port must be 0..65535, got {port}")
    if not isinstance(seq, int) or isinstance(seq, bool):
        raise TypeError(f"seq must be int, got {type(seq).__name__}")
    if not 0 <= seq <= MASK32:
        raise ValueError(f"seq must be 0..{MASK32}, got {seq}")
    return _HEADER.pack(MAGIC, PROTO_VERSION, port, seq, len(raw)) + raw


def parse_announce(data: bytes | bytearray | memoryview) -> Peer | None:
    """Decode one frame, or return ``None``.  Never raises on garbage."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        return None
    frame = bytes(data)
    if len(frame) < HEADER_FIXED:
        return None
    magic, version, port, seq, name_len = _HEADER.unpack_from(frame, 0)
    if magic != MAGIC or version != PROTO_VERSION:
        return None
    if not 1 <= name_len <= MAX_NAME:
        return None
    if len(frame) != HEADER_FIXED + name_len:
        return None
    raw = frame[HEADER_FIXED:]
    if b"\x00" in raw:
        return None
    try:
        name = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return Peer(name=name, port=port, seq=seq)


def merge_peers(current: Sequence[Peer], incoming: Peer) -> list[Peer]:
    """Insert ``incoming`` into ``current``; higher ``seq`` wins, sorted by name.

    A peer heard without a source keeps the first non-empty source seen, so a
    receive loop can fill it in later without reordering anything.
    """
    merged: dict[str, Peer] = {}
    for peer in current:
        existing = merged.get(peer.name)
        if existing is None or peer.seq >= existing.seq:
            merged[peer.name] = peer
    existing = merged.get(incoming.name)
    if existing is None:
        merged[incoming.name] = incoming
    elif incoming.seq > existing.seq:
        merged[incoming.name] = incoming
    elif incoming.seq == existing.seq and not existing.source and incoming.source:
        merged[incoming.name] = Peer(
            name=existing.name, port=existing.port, seq=existing.seq, source=incoming.source
        )
    return sorted(merged.values(), key=lambda peer: peer.name)


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ultragraph.zerocpu.mesh",
        description="Encode, decode and merge mesh announce frames.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    announce = sub.add_parser("announce", help="encode one frame as hex")
    announce.add_argument("name")
    announce.add_argument("port", type=int, default=PROTO_PORT, nargs="?")
    announce.add_argument("seq", type=int, default=0, nargs="?")

    parse = sub.add_parser("parse", help="decode one hex frame")
    parse.add_argument("hex")

    merge = sub.add_parser("merge", help="merge hex frames and print the peer list")
    merge.add_argument("hex", nargs="+")

    args = parser.parse_args(argv)
    if args.command == "announce":
        print(pack_announce(args.name, args.port, args.seq).hex())
        return 0
    if args.command == "parse":
        peer = parse_announce(binascii.unhexlify(args.hex))
        print(json.dumps(None if peer is None else peer.to_dict(), indent=2))
        return 0 if peer is not None else 1
    peers: list[Peer] = []
    for chunk in args.hex:
        peer = parse_announce(binascii.unhexlify(chunk))
        if peer is not None:
            peers = merge_peers(peers, peer)
    print(json.dumps([peer.to_dict() for peer in peers], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
