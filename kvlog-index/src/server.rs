use kvlog::encoding::MunchError;
use mio::event::Event;
use mio::net::{UnixListener, UnixStream};
use mio::{Events, Interest, Poll, Token};
use std::collections::HashMap;
use std::io;
use std::path::Path;

use crate::index::archetype::ServiceId;
use crate::index::Index;
const MAGIC: u64 = 0x8910_FC0Eu64 << 32;
const ALT_MAGIC: u64 = 0xF745_119Eu64 << 32;
const MAGIC_MASK: u64 = 0xFFFF_FFFFu64 << 32;

struct Dechunker<'a> {
    buffer: &'a mut [u8],
    head: usize,
    tail: usize,
}
enum Chunk<'a> {
    Bytes(&'a [u8]),
    BytesAlt(&'a [u8]),
    InvalidMagic,
    Partial,
    Done,
}

impl<'a> Dechunker<'a> {
    fn new(buffer: &'a mut [u8]) -> Self {
        Self { buffer, head: 0, tail: 0 }
    }
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<usize> {
        if self.tail > self.buffer.len() / 2 && self.head > 0 {
            self.buffer.copy_within(self.head..self.tail, 0);
            self.tail -= self.head;
            self.head = 0;
        }
        let len = reader.read(&mut self.buffer[self.tail..])?;
        self.tail += len;
        Ok(len)
    }
    fn next_chunk(&mut self) -> Chunk<'_> {
        let Some((header, rest)) = self.buffer[self.head..self.tail].split_first_chunk() else {
            if self.head == self.tail {
                return Chunk::Done;
            }
            return Chunk::Partial;
        };
        let len = u64::from_le_bytes(*header);
        let mut alt = false;
        if (len & MAGIC_MASK) != MAGIC {
            if (len & MAGIC_MASK) != ALT_MAGIC {
                return Chunk::InvalidMagic;
            }
            alt = true;
        }
        let len = (len & !MAGIC_MASK) as usize;
        let Some(bytes) = rest.get(..len) else {
            return Chunk::Partial;
        };
        self.head += len + 8;
        if alt {
            Chunk::BytesAlt(bytes)
        } else {
            Chunk::Bytes(bytes)
        }
    }
}

// "/tmp/libra-server.sock"
pub fn block_on(unix_socket_path: &Path, index: &mut Index, on_update: &mut dyn FnMut()) -> io::Result<()> {
    let _ = std::fs::remove_file(unix_socket_path);
    let listener = UnixListener::bind(unix_socket_path)?;
    let server = Server { index, update_signal: on_update };
    run_server(listener, server)
}

struct Server<'a> {
    index: &'a mut Index,
    update_signal: &'a mut dyn FnMut(),
}

struct Connection {
    stream: UnixStream,
    service_id: Option<ServiceId>,
}

impl<'a> Server<'a> {
    fn process_alt(&mut self, connection: &mut Connection, data: &[u8]) {
        match std::str::from_utf8(data) {
            Ok(service_name) => {
                kvlog::info!("Metadata updated for connection", service_name);
                connection.service_id = Some(ServiceId::intern(service_name));
            }
            Err(err) => {
                kvlog::error!("Failed to parse service name update", ?err, data);
            }
        }
    }
    fn process(&mut self, conn: &mut Connection, mut data: &[u8]) {
        while !data.is_empty() {
            match kvlog::encoding::munch_log_with_span(&mut data) {
                Ok((timestamp, level, span_info, fields)) => {
                    if let Err(err) = self.index.write(timestamp, level, span_info, conn.service_id, fields) {
                        kvlog::error!("Failed to write log to index", ?err)
                    }
                }
                Err(MunchError::EofOnHeader | MunchError::EofOnFields | MunchError::Eof) => {
                    kvlog::warn!("Unexpected EOF")
                }
                Err(err) => {
                    kvlog::warn!("Error munching logs", ?err);
                }
            }
        }
    }
}
const SERVER: Token = Token(0);

fn next(current: &mut Token) -> Token {
    let next = current.0;
    current.0 += 1;
    Token(next)
}

fn run_server(mut listener: UnixListener, server: Server) -> io::Result<()> {
    // Create a poll instance.
    let poll = mio::Poll::new()?;
    // Create storage for events.
    let mut events = Events::with_capacity(128);

    // Setup the TCP server socket.

    // Register the server with poll we can receive events for it.
    poll.registry().register(&mut listener, SERVER, Interest::READABLE)?;

    // Map of `Token` -> `TcpStream`.
    // todo switch to use a slab
    let mut connections = HashMap::new();
    // Unique token for each incoming connection.
    let mut unique_token = Token(SERVER.0 + 1);
    let mut controller = MioController { buffer: vec![0; 4096 * 8], poll, server };

    loop {
        if let Err(err) = controller.poll.poll(&mut events, None) {
            if interrupted(&err) {
                continue;
            }
            return Err(err);
        }

        for event in events.iter() {
            match event.token() {
                SERVER => loop {
                    // Received an event for the TCP server socket, which
                    // indicates we can accept an connection.
                    let (mut stream, _address) = match listener.accept() {
                        Ok((connection, address)) => (connection, address),
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                            // If we get a `WouldBlock` error we know our
                            // listener has no more incoming connections queued,
                            // so we can return to polling and wait for some
                            // more.
                            break;
                        }
                        Err(e) => {
                            // If it was any other kind of error, something went
                            // wrong and we terminate with an error.
                            return Err(e);
                        }
                    };

                    let token = next(&mut unique_token);
                    controller.poll.registry().register(&mut stream, token, Interest::READABLE)?;

                    connections.insert(token, Connection { stream, service_id: None });
                },
                token => {
                    // Maybe received an event for a TCP connection.
                    let done = if let Some(connection) = connections.get_mut(&token) {
                        controller.handle_connection_event(connection, event)?
                    } else {
                        // Sporadic events happen, we can safely ignore them.
                        false
                    };
                    if done {
                        if let Some(mut connection) = connections.remove(&token) {
                            controller.poll.registry().deregister(&mut connection.stream)?;
                        }
                    }
                }
            }
        }
        (controller.server.update_signal)();
    }
}

struct MioController<'a> {
    server: Server<'a>,
    poll: Poll,
    buffer: Vec<u8>,
}
impl<'a> MioController<'a> {
    fn handle_connection_event(&mut self, connection: &mut Connection, event: &Event) -> io::Result<bool> {
        if event.is_readable() {
            let mut connection_closed = false;
            let mut dechunker = Dechunker::new(&mut self.buffer);
            'reading: loop {
                match dechunker.read_from(&mut connection.stream) {
                    Ok(0) => {
                        connection_closed = true;
                        break;
                    }
                    Ok(_) => loop {
                        match dechunker.next_chunk() {
                            Chunk::BytesAlt(bytes) => {
                                self.server.process_alt(connection, bytes);
                            }
                            Chunk::Bytes(bytes) => {
                                self.server.process(connection, bytes);
                            }
                            Chunk::InvalidMagic => {
                                kvlog::error!(
                                    "Invalid magic read from connection",
                                    service_name = connection.service_id.map(|a| a.as_str())
                                );
                                break 'reading;
                            }
                            Chunk::Partial => continue 'reading,
                            Chunk::Done => break 'reading,
                        }
                    },
                    Err(ref err) if would_block(err) => break,
                    Err(ref err) if interrupted(err) => continue,
                    Err(err) => return Err(err),
                }
            }

            if connection_closed {
                println!("Connection closed");
                return Ok(true);
            }
        }

        Ok(false)
    }
}

fn would_block(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::WouldBlock
}

fn interrupted(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::Interrupted
}
