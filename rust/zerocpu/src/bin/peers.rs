//! `peers` -- announce yourself on the LAN, or listen for neighbours.
//!
//! The socket path is deliberately thin: the frame format and the merge rule
//! live in [`zerocpu::mesh`], are unit-tested there, and are pinned to the
//! Python reference by `tests/mesh_parity.rs`.  This binary is not covered by
//! CI -- it needs a network and a broadcast-capable interface.
//!
//! ```text
//! peers announce <name> [port] [seconds]
//! peers listen   [port] [seconds]
//! ```

use std::env;
use std::net::UdpSocket;
use std::thread::sleep;
use std::time::{Duration, Instant};

use zerocpu::mesh::{merge_peers, pack_announce, parse_announce, Peer, PROTO_PORT};

fn usage() -> ! {
    eprintln!("usage: peers announce <name> [port] [seconds]");
    eprintln!("       peers listen [port] [seconds]");
    std::process::exit(2)
}

fn argument(args: &[String], index: usize) -> Option<u16> {
    args.get(index).and_then(|value| value.parse().ok())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let command = args.first().cloned().unwrap_or_else(|| usage());
    match command.as_str() {
        "announce" => announce(&args),
        "listen" => listen(&args),
        _ => usage(),
    }
}

fn announce(args: &[String]) {
    let name = args.get(1).cloned().unwrap_or_else(|| usage());
    let port = argument(args, 2).unwrap_or(PROTO_PORT);
    let seconds = u64::from(argument(args, 3).unwrap_or(3));
    let frame = match pack_announce(&name, port, 0) {
        Ok(frame) => frame,
        Err(err) => {
            eprintln!("peers: {err}");
            std::process::exit(1);
        }
    };
    let socket = UdpSocket::bind("0.0.0.0:0").unwrap_or_else(|err| {
        eprintln!("peers: bind: {err}");
        std::process::exit(1);
    });
    if let Err(err) = socket.set_broadcast(true) {
        eprintln!("peers: broadcast: {err}");
        std::process::exit(1);
    }
    for _ in 0..seconds {
        match socket.send_to(&frame, ("255.255.255.255", port)) {
            Ok(sent) => println!("announced {name} on {port} ({sent}B)"),
            Err(err) => {
                eprintln!("peers: send: {err}");
                return;
            }
        }
        sleep(Duration::from_secs(1));
    }
}

fn listen(args: &[String]) {
    let wanted = argument(args, 1).unwrap_or(PROTO_PORT);
    let seconds = u64::from(argument(args, 2).unwrap_or(10));
    let mut port = wanted;
    let socket = loop {
        match UdpSocket::bind(("0.0.0.0", port)) {
            Ok(socket) => break socket,
            Err(err) if port < u16::MAX => {
                eprintln!("peers: {port} busy ({err}), trying {}", port + 1);
                port += 1;
            }
            Err(err) => {
                eprintln!("peers: bind: {err}");
                std::process::exit(1);
            }
        }
    };
    let timeout = Duration::from_millis(500);
    if let Err(err) = socket.set_read_timeout(Some(timeout)) {
        eprintln!("peers: timeout: {err}");
        std::process::exit(1);
    }
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut peers: Vec<Peer> = Vec::new();
    let mut buffer = [0u8; 128];
    while Instant::now() < deadline {
        match socket.recv_from(&mut buffer) {
            Ok((size, from)) => {
                if let Some(mut peer) = parse_announce(&buffer[..size]) {
                    peer.source = from.ip().to_string();
                    let before = peers.len();
                    peers = merge_peers(&peers, &peer);
                    if peers.len() != before || before == 0 {
                        for known in &peers {
                            println!(
                                "{} port={} seq={} source={}",
                                known.name, known.port, known.seq, known.source
                            );
                        }
                    }
                }
            }
            Err(_) => continue,
        }
    }
    println!("{} peer(s) known", peers.len());
}
