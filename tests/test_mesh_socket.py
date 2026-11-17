"""Tests for zerocpu.meshd: real UDP on loopback, no LAN required.

Every test binds 127.0.0.1 on an ephemeral port, so the suite is safe to run on
a machine with no network and cannot collide with a real node on :8423.
"""

import socket

from ultragraph.zerocpu import meshd
from ultragraph.zerocpu.mesh import pack_announce
from ultragraph.zerocpu.meshd import MeshListener, announce


def test_meshd_module_imports():
    assert meshd.__doc__
    assert meshd.DEFAULT_HOST == "127.0.0.1"


def test_a_real_frame_survives_the_round_trip():
    with MeshListener() as listener:
        # Ephemeral port unless the test asks for one: never bind :8423 here.
        assert listener.port > 0
        sent = announce("hearth", listener.port, seq=7, host="127.0.0.1")
        assert sent == len(pack_announce("hearth", listener.port, 7))
        peer = listener.poll(timeout=2.0)
        assert peer is not None
        assert (peer.name, peer.port, peer.seq) == ("hearth", listener.port, 7)
        assert peer.source == "127.0.0.1"
        assert listener.counts == (1, 0)


def test_two_nodes_merge_into_one_sorted_list():
    with MeshListener() as listener:
        announce("thunderbird", listener.port, seq=1, host="127.0.0.1")
        announce("hearth", listener.port, seq=1, host="127.0.0.1")
        peers = listener.listen(seconds=1.5)
        assert [peer.name for peer in peers] == ["hearth", "thunderbird"]
        assert listener.counts[0] == 2


def test_garbage_is_counted_and_ignored():
    with MeshListener() as listener:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as junk:
            junk.sendto(b"not a frame at all", ("127.0.0.1", listener.port))
            junk.sendto(b"UGRD\x02" + b"\x00" * 8, ("127.0.0.1", listener.port))
        announce("hearth", listener.port, seq=3, host="127.0.0.1")
        peers = listener.listen(seconds=1.5)
        assert [peer.name for peer in peers] == ["hearth"]
        accepted, rejected = listener.counts
        assert accepted == 1
        assert rejected == 2


def test_a_later_seq_wins_and_an_earlier_one_does_not():
    with MeshListener() as listener:
        announce("hearth", listener.port, seq=5, host="127.0.0.1")
        announce("hearth", listener.port, seq=2, host="127.0.0.1")
        announce("hearth", listener.port, seq=9, host="127.0.0.1")
        peers = listener.listen(seconds=1.5)
        assert len(peers) == 1
        assert peers[0].seq == 9


def test_poll_returns_none_when_nothing_lands():
    with MeshListener() as listener:
        assert listener.poll(timeout=0.15) is None
        assert listener.heard() == []
        assert listener.counts == (0, 0)


def test_the_cli_listens_and_prints_json():
    import json
    import subprocess
    import sys
    import time

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "ultragraph.zerocpu.meshd", "listen",
            "--host", "127.0.0.1", "--port", str(port), "--seconds", "1.5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # The child needs a moment to bind; re-announce instead of sleeping blind.
        for _ in range(30):
            if proc.poll() is not None:
                break
            try:
                announce("hearth", port, seq=1, host="127.0.0.1", timeout=0.2)
            except OSError:
                pass
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        out, err = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    peers = json.loads(out)
    assert [peer["name"] for peer in peers] == ["hearth"], out
    assert "1 peer(s)" in err, err
