"""Tests for zerocpu.meshd: real UDP on loopback, no LAN required.

Every test binds 127.0.0.1 on an ephemeral port, so the suite is safe to run on
a machine with no network and cannot collide with a real node on :8423.
"""

import socket
import subprocess
import sys

from ultragraph.zerocpu import meshd
from ultragraph.zerocpu.mesh import pack_announce
from ultragraph.zerocpu.meshd import MeshListener, _is_loopback, announce


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


# --------------------------------------------------------------------------
# The loopback guard: binding beyond this machine is a decision, not a typo
# --------------------------------------------------------------------------

def test_loopback_hosts_are_recognised_however_they_are_spelled():
    for host in ("127.0.0.1", "127.0.0.2", "localhost", "::1"):
        assert _is_loopback(host), host


def test_wildcard_and_routable_hosts_are_not_loopback():
    for host in ("0.0.0.0", "::", "192.168.1.4", "10.0.0.1", "example.com",
                 "127.0.0.1.example"):
        assert not _is_loopback(host), host


def test_a_wildcard_bind_is_refused_without_opting_in():
    try:
        MeshListener("0.0.0.0", 0)
    except ValueError as err:
        assert "allow_lan" in str(err)
    else:
        raise AssertionError("binding a wildcard must require allow_lan=True")


def test_a_loopback_bind_needs_no_opt_in():
    with MeshListener("127.0.0.1", 0) as listener:
        assert listener.is_lan is False
        assert listener.address[0].startswith("127.")


def test_the_cli_reports_a_refusal_as_exit_2_not_a_traceback():
    done = subprocess.run(
        [sys.executable, "-m", "ultragraph.zerocpu.meshd", "listen",
         "--port", "0", "--host", "0.0.0.0"],
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 2, done
    assert "Traceback" not in done.stderr
    assert "allow_lan" in done.stderr
