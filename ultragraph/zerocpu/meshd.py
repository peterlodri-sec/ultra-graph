"""The mesh socket path: listen for UGRD v1 announces, or send one.

`mesh.py` is the protocol -- the frame and the merge rule, pinned by fixtures.
This module is the socket around it, the Python twin of
`rust/zerocpu/src/bin/peers.rs`.  One datagram, no handshake, no external
crate: `socket` plus the packing already in `mesh.py`.

The default bind is **loopback**, not `0.0.0.0`.  A listener that answers the
whole LAN by default is a service; one that answers `127.0.0.1` is a test
fixture you can also point at your own interface on purpose:

    with MeshListener() as listener:            # 127.0.0.1, ephemeral port
        announce("hearth", listener.port, seq=1, host="127.0.0.1")
        peers = listener.listen(seconds=1)

Garbage on the wire is a normal condition, not an error: `poll` and `listen`
skip anything `parse_announce` refuses, so a stray packet from another tool
cannot crash a node.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from collections.abc import Sequence

from .mesh import PROTO_PORT, Peer, merge_peers, pack_announce, parse_announce

#: Bind to loopback unless the caller says otherwise.
DEFAULT_HOST = "127.0.0.1"
#: Largest datagram we are willing to read; an announce is at most 44 bytes.
RECV_BUFFER = 512


def announce(
    name: str,
    port: int = PROTO_PORT,
    seq: int = 0,
    *,
    host: str = DEFAULT_HOST,
    timeout: float = 1.0,
) -> int:
    """Send one announce frame; return the number of bytes written.

    ``port`` is the port advertised *inside* the frame (where this node would
    listen), while ``host`` is where the datagram is sent.  They are usually the
    same number and are deliberately separate arguments.
    """
    frame = pack_announce(name, port, seq)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout)
        return sock.sendto(frame, (host, port))


class MeshListener:
    """A bound UDP socket that turns announce frames into a merged peer list."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = 0) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(0.05)
        self._heard: list[Peer] = []
        self._accepted = 0
        self._rejected = 0

    @property
    def address(self) -> tuple[str, int]:
        """The bound ``(host, port)`` -- with the real port when ``port=0``."""
        host, port = self._sock.getsockname()
        return str(host), int(port)

    @property
    def port(self) -> int:
        return self.address[1]

    def heard(self) -> list[Peer]:
        """Every peer accepted so far, merged and sorted by name."""
        return list(self._heard)

    @property
    def counts(self) -> tuple[int, int]:
        """``(accepted, rejected)`` frames -- rejected includes stray packets."""
        return self._accepted, self._rejected

    def poll(self, timeout: float = 0.1) -> Peer | None:
        """Wait up to ``timeout`` seconds for one frame; ``None`` if none lands."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._sock.settimeout(remaining)
            try:
                data, source = self._sock.recvfrom(RECV_BUFFER)
            except TimeoutError:
                return None
            except OSError:
                return None
            peer = parse_announce(data)
            if peer is None:
                self._rejected += 1
                continue
            peer = Peer(name=peer.name, port=peer.port, seq=peer.seq, source=source[0])
            self._accepted += 1
            self._heard = merge_peers(self._heard, peer)
            return peer

    def listen(self, seconds: float = 1.0) -> list[Peer]:
        """Collect frames for ``seconds`` and return the merged peer list."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.poll(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        return self.heard()

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> MeshListener:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ultragraph.zerocpu.meshd",
        description="Announce on the mesh, or listen for neighbours (loopback by default).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    say = sub.add_parser("announce", help="send one announce frame")
    say.add_argument("name")
    say.add_argument("--port", type=int, default=PROTO_PORT)
    say.add_argument("--seq", type=int, default=0)
    say.add_argument("--host", default=DEFAULT_HOST)

    hear = sub.add_parser("listen", help="collect announces and print the peer list")
    hear.add_argument("--port", type=int, default=PROTO_PORT)
    hear.add_argument("--host", default=DEFAULT_HOST)
    hear.add_argument("--seconds", type=float, default=5.0)

    args = parser.parse_args(argv)
    try:
        if args.command == "announce":
            sent = announce(args.name, args.port, args.seq, host=args.host)
            print(f"announced {args.name} on {args.host}:{args.port} ({sent}B)")
            return 0
        with MeshListener(args.host, args.port) as listener:
            host, port = listener.address
            print(f"listening on {host}:{port}", file=sys.stderr)
            peers = listener.listen(args.seconds)
            accepted, rejected = listener.counts
    except OSError as err:
        print(f"meshd: {err}", file=sys.stderr)
        return 1
    print(json.dumps([peer.to_dict() for peer in peers], indent=2))
    print(f"{len(peers)} peer(s), {accepted} accepted, {rejected} rejected", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
