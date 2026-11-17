//! A real announce frame over a real socket -- on loopback, so CI can run it.
//!
//! `mesh_parity` pins the bytes; this pins the fact that the bytes survive a
//! `sendto`/`recvfrom` round trip and still parse and merge.  Nothing here
//! touches the LAN: the listener binds `127.0.0.1:0` and the sender writes to
//! the port the kernel handed out.

use std::net::UdpSocket;
use std::thread;
use std::time::Duration;

use zerocpu::mesh::{merge_peers, pack_announce, parse_announce};

const BUFFER: usize = 512;

#[test]
fn one_frame_survives_a_loopback_round_trip() {
    let listener = UdpSocket::bind("127.0.0.1:0").expect("bind loopback");
    listener
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("read timeout");
    let port = listener.local_addr().expect("local addr").port();

    let frame = pack_announce("hearth", 8423, 7).expect("frame");
    let sender = UdpSocket::bind("127.0.0.1:0").expect("bind sender");
    let handle = thread::spawn(move || {
        sender.send_to(&frame, ("127.0.0.1", port)).expect("send");
    });

    let mut buffer = [0u8; BUFFER];
    let (size, from) = listener.recv_from(&mut buffer).expect("recv");
    handle.join().expect("sender thread");

    let peer = parse_announce(&buffer[..size]).expect("parsed");
    assert_eq!(
        (peer.name.as_str(), peer.port, peer.seq),
        ("hearth", 8423, 7)
    );
    assert_eq!(from.ip().to_string(), "127.0.0.1");
    assert_eq!(size, buffer.len().min(size));
}

#[test]
fn two_nodes_merge_and_garbage_is_dropped() {
    let listener = UdpSocket::bind("127.0.0.1:0").expect("bind loopback");
    listener
        .set_read_timeout(Some(Duration::from_millis(250)))
        .expect("read timeout");
    let port = listener.local_addr().expect("local addr").port();
    let sender = UdpSocket::bind("127.0.0.1:0").expect("bind sender");

    sender
        .send_to(b"not a frame at all", ("127.0.0.1", port))
        .expect("junk");
    for (name, seq) in [("thunderbird", 1u32), ("hearth", 1), ("hearth", 9)] {
        let frame = pack_announce(name, 8423, seq).expect("frame");
        sender.send_to(&frame, ("127.0.0.1", port)).expect("send");
    }

    let mut peers = Vec::new();
    let mut rejected = 0;
    let mut buffer = [0u8; BUFFER];
    for _ in 0..8 {
        match listener.recv_from(&mut buffer) {
            Ok((size, from)) => match parse_announce(&buffer[..size]) {
                Some(mut peer) => {
                    peer.source = from.ip().to_string();
                    peers = merge_peers(&peers, &peer);
                }
                None => rejected += 1,
            },
            Err(_) => break,
        }
    }

    assert_eq!(rejected, 1, "the junk datagram must be refused, not merged");
    assert_eq!(peers.len(), 2, "two names, three frames");
    let hearth = peers
        .iter()
        .find(|peer| peer.name == "hearth")
        .expect("hearth");
    assert_eq!(hearth.seq, 9, "the highest seq wins");
    assert_eq!(peers[0].name, "hearth", "merged list is sorted by name");
}
