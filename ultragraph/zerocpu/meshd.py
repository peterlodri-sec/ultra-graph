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
import ipaddress
import json
import socket
import sys
import time
from collections.abc import Sequence

from .mesh import PROTO_PORT, Peer, merge_peers, pack_announce, parse_announce

#: Bind to loopback unless the caller says otherwise.
#: Command-line exit codes: 0 ok, 1 the socket failed, 2 the request was
#: refused by policy (binding beyond loopback without `--allow-lan`).
EXIT_OK = 0
EXIT_SOCKET = 1
EXIT_REFUSED = 2

DEFAULT_HOST = "127.0.0.1"
#: Hosts that expose the socket beyond this machine. Binding one is a deliberate
#: act, so it needs `allow_lan=True`; a typo in a config file should not silently
#: publish the mesh to the network.
LAN_HOSTS = ("0.0.0.0", "::", "")
#: Hosts that are this machine, however they are spelled.
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
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


def _is_loopback(host: str) -> bool:
    """True for a host that names this machine, however it is spelled.

    ``ipaddress`` is used rather than a prefix test so that ``127.0.0.2`` (also
    loopback) and ``127.1.2.3`` resolve correctly while ``127.0.0.1.example``
    does not.
    """
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class MeshListener:
    """A bound UDP socket that turns announce frames into a merged peer list."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = 0, *,
                 allow_lan: bool = False) -> None:
        """Bind ``host:port``.

        Loopback is the default and needs no argument. Binding a wildcard or a
        routable address requires ``allow_lan=True``: the listener accepts any
        datagram that parses as a UGRD frame, so an accidental wildcard bind is
        a mesh anyone on the network can inject peers into.
        """
        self._lan = False
        if host in LAN_HOSTS or not _is_loopback(host):
            if not allow_lan:
                raise ValueError(
                    f"refusing to bind {host!r}: that is reachable from other "
                    "machines, and a mesh listener accepts any well-formed "
                    "frame. Pass allow_lan=True to mean it."
                )
            self._lan = True
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

    @property
    def is_lan(self) -> bool:
        """True when this listener is reachable from other machines."""
        return self._lan

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
    hear.add_argument("--allow-lan", action="store_true", dest="allow_lan",
                      help="permit binding a non-loopback address (reachable "
                           "from other machines)")
    hear.add_argument("--seconds", type=float, default=5.0)

    args = parser.parse_args(argv)
    try:
        if args.command == "announce":
            sent = announce(args.name, args.port, args.seq, host=args.host)
            print(f"announced {args.name} on {args.host}:{args.port} ({sent}B)")
            return EXIT_OK
        with MeshListener(args.host, args.port, allow_lan=args.allow_lan) as listener:
            host, port = listener.address
            print(f"listening on {host}:{port}", file=sys.stderr)
            peers = listener.listen(args.seconds)
            accepted, rejected = listener.counts
    except ValueError as err:
        # A refusal is a policy decision, not a crash: say so and stop.
        print(f"meshd: {err}", file=sys.stderr)
        return EXIT_REFUSED
    except OSError as err:
        print(f"meshd: {err}", file=sys.stderr)
        return 1
    print(json.dumps([peer.to_dict() for peer in peers], indent=2))
    print(f"{len(peers)} peer(s), {accepted} accepted, {rejected} rejected", file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
