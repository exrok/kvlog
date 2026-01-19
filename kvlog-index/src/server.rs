use kvlog::encoding::MunchError;
use mio::event::Event;
use mio::net::{UnixListener, UnixStream};
use mio::{Events, Interest, Poll, Token};
use std::collections::HashMap;
use std::io;
use std::path::Path;

use crate::index::archetype::ServiceId;
use crate::index::Index;

const CHUNK_MAGIC: u64 = 0x8910_FC0Eu64 << 32;
const CHUNK_ALT_MAGIC: u64 = 0xF745_119Eu64 << 32;
const CHUNK_MAGIC_MASK: u64 = 0xFFFF_FFFFu64 << 32;
const ENTRY_MAGIC_BYTE: u8 = 0b1110_0001;

const DEFAULT_BUFFER_SIZE: usize = 32 * 1024; // 32KB initial
const MAX_ENTRY_SIZE: usize = 4 * 1024 * 1024; // 4MB max per entry

#[derive(Debug, Clone, Copy)]
enum DechunkState {
    AwaitingChunkHeader,
    StreamingEntries { remaining: u32 },
    AccumulatingEntry { entry_len: u32, chunk_remaining: u32 },
    AccumulatingAltChunk { chunk_len: u32 },
}

#[derive(Debug)]
struct EntryRef {
    start: usize,
    len: usize,
}

#[derive(Debug)]
enum DechunkResult {
    Entry(EntryRef),
    AltChunk(EntryRef),
    InvalidChunkMagic,
    InvalidEntryMagic,
    EntryTooLarge { size: u32 },
    NeedMoreData,
    Done,
}

struct StreamingDechunker {
    buffer: Vec<u8>,
    head: usize,
    tail: usize,
    state: DechunkState,
}

impl StreamingDechunker {
    fn new() -> Self {
        Self { buffer: vec![0; DEFAULT_BUFFER_SIZE], head: 0, tail: 0, state: DechunkState::AwaitingChunkHeader }
    }

    fn buffered_len(&self) -> usize {
        self.tail - self.head
    }

    fn buffered(&self) -> &[u8] {
        &self.buffer[self.head..self.tail]
    }

    fn consume(&mut self, n: usize) {
        self.head += n;
    }

    fn compact(&mut self) {
        if self.head > 0 {
            self.buffer.copy_within(self.head..self.tail, 0);
            self.tail -= self.head;
            self.head = 0;
        }
    }

    fn ensure_space(&mut self, needed: usize) -> io::Result<()> {
        if self.buffer.len() - self.tail >= needed {
            return Ok(());
        }

        self.compact();

        if self.buffer.len() - self.tail >= needed {
            return Ok(());
        }

        let required = self.tail + needed;
        if required > MAX_ENTRY_SIZE + 1024 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "entry exceeds maximum size"));
        }

        let new_size = required.next_power_of_two().min(MAX_ENTRY_SIZE + 1024);
        self.buffer.resize(new_size, 0);
        Ok(())
    }

    fn read_from<R: io::Read>(&mut self, reader: &mut R) -> io::Result<usize> {
        if self.head > self.buffer.len() / 2 {
            self.compact();
        }

        if self.tail >= self.buffer.len() {
            self.ensure_space(1024)?;
        }

        let n = reader.read(&mut self.buffer[self.tail..])?;
        self.tail += n;
        Ok(n)
    }

    fn get_data(&self, entry_ref: &EntryRef) -> &[u8] {
        &self.buffer[entry_ref.start..entry_ref.start + entry_ref.len]
    }

    fn next_item(&mut self) -> DechunkResult {
        loop {
            match self.state {
                DechunkState::AwaitingChunkHeader => {
                    if self.buffered_len() < 8 {
                        return if self.head == self.tail { DechunkResult::Done } else { DechunkResult::NeedMoreData };
                    }

                    let header = u64::from_le_bytes(self.buffered()[..8].try_into().unwrap());

                    let is_alt = match header & CHUNK_MAGIC_MASK {
                        m if m == CHUNK_MAGIC => false,
                        m if m == CHUNK_ALT_MAGIC => true,
                        _ => return DechunkResult::InvalidChunkMagic,
                    };

                    let chunk_len = (header & !CHUNK_MAGIC_MASK) as u32;
                    self.consume(8);

                    if is_alt {
                        self.state = DechunkState::AccumulatingAltChunk { chunk_len };
                    } else {
                        self.state = DechunkState::StreamingEntries { remaining: chunk_len };
                    }
                }

                DechunkState::AccumulatingAltChunk { chunk_len } => {
                    if self.buffered_len() < chunk_len as usize {
                        return DechunkResult::NeedMoreData;
                    }

                    let entry_ref = EntryRef { start: self.head, len: chunk_len as usize };
                    self.consume(chunk_len as usize);
                    self.state = DechunkState::AwaitingChunkHeader;
                    return DechunkResult::AltChunk(entry_ref);
                }

                DechunkState::StreamingEntries { remaining } => {
                    if remaining == 0 {
                        self.state = DechunkState::AwaitingChunkHeader;
                        continue;
                    }

                    if self.buffered_len() < 4 {
                        return DechunkResult::NeedMoreData;
                    }

                    let entry_header = u32::from_le_bytes(self.buffered()[..4].try_into().unwrap());

                    if (entry_header >> 24) as u8 != ENTRY_MAGIC_BYTE {
                        return DechunkResult::InvalidEntryMagic;
                    }

                    let field_len = entry_header & 0x00FF_FFFF;
                    let entry_len = 4 + field_len; // header + fields (includes timestamp+level)

                    if entry_len > MAX_ENTRY_SIZE as u32 {
                        return DechunkResult::EntryTooLarge { size: entry_len };
                    }

                    if self.buffered_len() >= entry_len as usize {
                        let entry_ref = EntryRef { start: self.head, len: entry_len as usize };
                        self.consume(entry_len as usize);
                        self.state = DechunkState::StreamingEntries { remaining: remaining - entry_len };
                        return DechunkResult::Entry(entry_ref);
                    }

                    self.state = DechunkState::AccumulatingEntry { entry_len, chunk_remaining: remaining - entry_len };
                    return DechunkResult::NeedMoreData;
                }

                DechunkState::AccumulatingEntry { entry_len, chunk_remaining } => {
                    if self.buffered_len() < entry_len as usize {
                        return DechunkResult::NeedMoreData;
                    }

                    let entry_ref = EntryRef { start: self.head, len: entry_len as usize };
                    self.consume(entry_len as usize);
                    self.state = DechunkState::StreamingEntries { remaining: chunk_remaining };
                    return DechunkResult::Entry(entry_ref);
                }
            }
        }
    }
}

pub fn block_on(unix_socket_path: &Path, index: &mut Index, on_update: &mut dyn FnMut()) -> io::Result<()> {
    let _ = std::fs::remove_file(unix_socket_path);
    let listener = UnixListener::bind(unix_socket_path)?;
    let server = Server { index, update_signal: on_update };
    run_server(listener, server)
}

pub struct Server<'a> {
    pub index: &'a mut Index,
    pub update_signal: &'a mut dyn FnMut(),
}

struct Connection {
    stream: UnixStream,
    service_id: Option<ServiceId>,
}

struct ConnectionState {
    connection: Connection,
    dechunker: StreamingDechunker,
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

    fn process_entry(&mut self, conn: &mut Connection, entry_data: &[u8]) {
        let mut data = entry_data;
        match kvlog::encoding::munch_log_with_span(&mut data) {
            Ok((timestamp, level, span_info, fields)) => {
                if let Err(err) = self.index.write(timestamp, level, span_info, conn.service_id, fields) {
                    kvlog::error!("Failed to write log to index", ?err)
                }
            }
            Err(MunchError::EofOnHeader | MunchError::EofOnFields | MunchError::Eof) => {
                kvlog::warn!("Unexpected EOF in entry")
            }
            Err(err) => {
                kvlog::warn!("Error parsing log entry", ?err);
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

pub fn run_server(mut listener: UnixListener, server: Server) -> io::Result<()> {
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
    let mut controller = MioController { poll, server };

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

                    connections.insert(
                        token,
                        ConnectionState {
                            connection: Connection { stream, service_id: None },
                            dechunker: StreamingDechunker::new(),
                        },
                    );
                },
                token => {
                    // Maybe received an event for a TCP connection.
                    let done = if let Some(conn_state) = connections.get_mut(&token) {
                        controller.handle_connection_event(conn_state, event)?
                    } else {
                        // Sporadic events happen, we can safely ignore them.
                        false
                    };
                    if done {
                        if let Some(mut conn_state) = connections.remove(&token) {
                            controller.poll.registry().deregister(&mut conn_state.connection.stream)?;
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
}
impl<'a> MioController<'a> {
    fn handle_connection_event(&mut self, conn_state: &mut ConnectionState, event: &Event) -> io::Result<bool> {
        if !event.is_readable() {
            return Ok(false);
        }

        let ConnectionState { connection, dechunker } = conn_state;

        loop {
            // First, process any complete items in the buffer
            loop {
                match dechunker.next_item() {
                    DechunkResult::Entry(entry_ref) => {
                        let entry_data = dechunker.get_data(&entry_ref);
                        self.server.process_entry(connection, entry_data);
                    }
                    DechunkResult::AltChunk(entry_ref) => {
                        let data = dechunker.get_data(&entry_ref);
                        self.server.process_alt(connection, data);
                    }
                    DechunkResult::InvalidChunkMagic => {
                        kvlog::error!(
                            "Invalid chunk magic, closing connection",
                            service_name = connection.service_id.map(|a| a.as_str())
                        );
                        return Ok(true);
                    }
                    DechunkResult::InvalidEntryMagic => {
                        kvlog::error!(
                            "Invalid entry magic, closing connection",
                            service_name = connection.service_id.map(|a| a.as_str())
                        );
                        return Ok(true);
                    }
                    DechunkResult::EntryTooLarge { size } => {
                        kvlog::error!(
                            "Entry exceeds maximum size, closing connection",
                            service_name = connection.service_id.map(|a| a.as_str()),
                            size
                        );
                        return Ok(true);
                    }
                    DechunkResult::NeedMoreData => break,
                    DechunkResult::Done => break,
                }
            }

            // Now try to read more data
            match dechunker.read_from(&mut connection.stream) {
                Ok(0) => {
                    // Connection closed by peer
                    kvlog::info!("Connection closed", service_name = connection.service_id.map(|a| a.as_str()));
                    return Ok(true);
                }
                Ok(_) => {
                    // Got data, continue processing
                    continue;
                }
                Err(ref err) if would_block(err) => {
                    // No more data available right now
                    return Ok(false);
                }
                Err(ref err) if interrupted(err) => {
                    continue;
                }
                Err(ref err) if is_connection_error(err) => {
                    kvlog::warn!("Connection error", service_name = connection.service_id.map(|a| a.as_str()), ?err);
                    return Ok(true);
                }
                Err(ref err) if err.kind() == io::ErrorKind::InvalidData => {
                    kvlog::error!(
                        "Entry exceeds maximum buffer size, closing connection",
                        service_name = connection.service_id.map(|a| a.as_str())
                    );
                    return Ok(true);
                }
                Err(err) => {
                    return Err(err);
                }
            }
        }
    }
}

fn is_connection_error(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::BrokenPipe
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::NotConnected
    )
}

fn would_block(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::WouldBlock
}

fn interrupted(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::Interrupted
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use std::io::Cursor;

    fn make_chunk(entries: &[&[u8]]) -> Vec<u8> {
        let mut chunk = Vec::new();
        let total_len: usize = entries.iter().map(|e| e.len()).sum();
        let header = CHUNK_MAGIC | (total_len as u64);
        chunk.extend_from_slice(&header.to_le_bytes());
        for entry in entries {
            chunk.extend_from_slice(entry);
        }
        chunk
    }

    fn make_entry(fields: &[u8]) -> Vec<u8> {
        // Entry format: [4-byte header][8-byte timestamp][1-byte level][fields]
        // Header: ENTRY_MAGIC_BYTE << 24 | (field_data_len)
        // field_data_len = 8 (timestamp) + 1 (level) + fields.len()
        let field_data_len = 8 + 1 + fields.len();
        let header = ((ENTRY_MAGIC_BYTE as u32) << 24) | (field_data_len as u32);
        let mut entry = Vec::new();
        entry.extend_from_slice(&header.to_le_bytes());
        entry.extend_from_slice(&1234567890u64.to_le_bytes()); // timestamp
        entry.push(1u8); // level: Info
        entry.extend_from_slice(fields);
        entry
    }

    fn make_alt_chunk(data: &[u8]) -> Vec<u8> {
        let mut chunk = Vec::new();
        let header = CHUNK_ALT_MAGIC | (data.len() as u64);
        chunk.extend_from_slice(&header.to_le_bytes());
        chunk.extend_from_slice(data);
        chunk
    }

    #[test]
    fn test_partial_chunk_header() {
        let entry = make_entry(b"test");
        let chunk = make_chunk(&[&entry]);

        let mut dechunker = StreamingDechunker::new();

        // Feed 4 bytes (partial header)
        let mut cursor = Cursor::new(&chunk[..4]);
        dechunker.read_from(&mut cursor).unwrap();
        assert!(matches!(dechunker.next_item(), DechunkResult::NeedMoreData));

        // Feed remaining 4 bytes of header + some entry data
        let mut cursor = Cursor::new(&chunk[4..12]);
        dechunker.read_from(&mut cursor).unwrap();
        assert!(matches!(dechunker.next_item(), DechunkResult::NeedMoreData));

        // Feed rest
        let mut cursor = Cursor::new(&chunk[12..]);
        dechunker.read_from(&mut cursor).unwrap();
        assert!(matches!(dechunker.next_item(), DechunkResult::Entry(_)));
    }

    #[test]
    fn test_partial_entry_header() {
        let entry = make_entry(b"test data here");
        let chunk = make_chunk(&[&entry]);

        let mut dechunker = StreamingDechunker::new();

        // Feed chunk header + 2 bytes of entry header
        let mut cursor = Cursor::new(&chunk[..10]);
        dechunker.read_from(&mut cursor).unwrap();
        assert!(matches!(dechunker.next_item(), DechunkResult::NeedMoreData));

        // Feed rest
        let mut cursor = Cursor::new(&chunk[10..]);
        dechunker.read_from(&mut cursor).unwrap();
        assert!(matches!(dechunker.next_item(), DechunkResult::Entry(_)));
    }

    #[test]
    fn test_partial_entry_body() {
        let entry = make_entry(b"longer test data that spans multiple reads");
        let chunk = make_chunk(&[&entry]);

        let mut dechunker = StreamingDechunker::new();

        // Feed chunk header + entry header + partial body
        let split_point = 20;
        let mut cursor = Cursor::new(&chunk[..split_point]);
        dechunker.read_from(&mut cursor).unwrap();
        assert!(matches!(dechunker.next_item(), DechunkResult::NeedMoreData));

        // Feed rest
        let mut cursor = Cursor::new(&chunk[split_point..]);
        dechunker.read_from(&mut cursor).unwrap();
        assert!(matches!(dechunker.next_item(), DechunkResult::Entry(_)));
    }

    #[test]
    fn test_large_entry() {
        // 64KB of fields - larger than initial 32KB buffer
        let large_fields = vec![0u8; 64 * 1024];
        let entry = make_entry(&large_fields);
        let chunk = make_chunk(&[&entry]);

        let mut dechunker = StreamingDechunker::new();
        let mut cursor = Cursor::new(&chunk);

        loop {
            let n = dechunker.read_from(&mut cursor).unwrap();
            match dechunker.next_item() {
                DechunkResult::Entry(entry_ref) => {
                    assert_eq!(entry_ref.len, entry.len());
                    break;
                }
                DechunkResult::NeedMoreData if n > 0 => continue,
                DechunkResult::Done if n == 0 => panic!("Unexpected Done"),
                other => panic!("Unexpected result: {:?}", other),
            }
        }
    }

    #[test]
    fn test_max_size_entry() {
        // Entry at 1MB (well under 4MB limit but still large)
        let max_fields = vec![0u8; 1024 * 1024];
        let entry = make_entry(&max_fields);
        let chunk = make_chunk(&[&entry]);

        let mut dechunker = StreamingDechunker::new();
        let mut cursor = Cursor::new(&chunk);

        loop {
            dechunker.read_from(&mut cursor).unwrap();
            match dechunker.next_item() {
                DechunkResult::Entry(_) => break,
                DechunkResult::NeedMoreData => continue,
                other => panic!("Unexpected: {:?}", other),
            }
        }
    }

    #[test]
    fn test_oversized_entry() {
        // Manually craft an entry with size > MAX_ENTRY_SIZE
        let fake_size = (MAX_ENTRY_SIZE + 1000) as u32;
        let fake_field_len = fake_size - 4; // subtract header
        let header = ((ENTRY_MAGIC_BYTE as u32) << 24) | fake_field_len;

        let mut chunk = Vec::new();
        let chunk_header = CHUNK_MAGIC | (fake_size as u64);
        chunk.extend_from_slice(&chunk_header.to_le_bytes());
        chunk.extend_from_slice(&header.to_le_bytes());

        let mut dechunker = StreamingDechunker::new();
        let mut cursor = Cursor::new(&chunk);
        dechunker.read_from(&mut cursor).unwrap();

        assert!(matches!(dechunker.next_item(), DechunkResult::EntryTooLarge { .. }));
    }

    #[test]
    fn test_multiple_entries() {
        let entry1 = make_entry(b"first");
        let entry2 = make_entry(b"second");
        let entry3 = make_entry(b"third");
        let chunk = make_chunk(&[&entry1, &entry2, &entry3]);

        let mut dechunker = StreamingDechunker::new();
        let mut cursor = Cursor::new(&chunk);
        dechunker.read_from(&mut cursor).unwrap();

        assert!(matches!(dechunker.next_item(), DechunkResult::Entry(_)));
        assert!(matches!(dechunker.next_item(), DechunkResult::Entry(_)));
        assert!(matches!(dechunker.next_item(), DechunkResult::Entry(_)));
        assert!(matches!(dechunker.next_item(), DechunkResult::Done));
    }

    #[test]
    fn test_multiple_chunks() {
        let entry1 = make_entry(b"chunk1");
        let entry2 = make_entry(b"chunk2");
        let chunk1 = make_chunk(&[&entry1]);
        let chunk2 = make_chunk(&[&entry2]);

        let mut data = Vec::new();
        data.extend_from_slice(&chunk1);
        data.extend_from_slice(&chunk2);

        let mut dechunker = StreamingDechunker::new();
        let mut cursor = Cursor::new(&data);
        dechunker.read_from(&mut cursor).unwrap();

        assert!(matches!(dechunker.next_item(), DechunkResult::Entry(_)));
        assert!(matches!(dechunker.next_item(), DechunkResult::Entry(_)));
        assert!(matches!(dechunker.next_item(), DechunkResult::Done));
    }

    #[test]
    fn test_alt_chunk() {
        let alt = make_alt_chunk(b"service-name");
        let entry = make_entry(b"data");
        let regular = make_chunk(&[&entry]);

        let mut data = Vec::new();
        data.extend_from_slice(&alt);
        data.extend_from_slice(&regular);

        let mut dechunker = StreamingDechunker::new();
        let mut cursor = Cursor::new(&data);
        dechunker.read_from(&mut cursor).unwrap();

        match dechunker.next_item() {
            DechunkResult::AltChunk(entry_ref) => {
                assert_eq!(dechunker.get_data(&entry_ref), b"service-name");
            }
            other => panic!("Expected AltChunk, got {:?}", other),
        }
        assert!(matches!(dechunker.next_item(), DechunkResult::Entry(_)));
    }

    #[test]
    fn test_invalid_chunk_magic() {
        let bad_data = [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];

        let mut dechunker = StreamingDechunker::new();
        let mut cursor = Cursor::new(&bad_data);
        dechunker.read_from(&mut cursor).unwrap();

        assert!(matches!(dechunker.next_item(), DechunkResult::InvalidChunkMagic));
    }

    #[test]
    fn test_byte_by_byte() {
        let entry = make_entry(b"test");
        let chunk = make_chunk(&[&entry]);

        let mut dechunker = StreamingDechunker::new();
        let mut got_entry = false;

        for &byte in chunk.iter() {
            let byte_arr = [byte];
            let mut cursor = Cursor::new(&byte_arr);
            dechunker.read_from(&mut cursor).unwrap();

            match dechunker.next_item() {
                DechunkResult::Entry(_) => {
                    got_entry = true;
                    break;
                }
                DechunkResult::NeedMoreData | DechunkResult::Done => continue,
                other => panic!("Unexpected: {:?}", other),
            }
        }

        assert!(got_entry, "Should have received an entry");
    }

    #[test]
    fn test_entry_data_integrity() {
        let test_fields = b"hello world test data";
        let entry = make_entry(test_fields);
        let chunk = make_chunk(&[&entry]);

        let mut dechunker = StreamingDechunker::new();
        let mut cursor = Cursor::new(&chunk);
        dechunker.read_from(&mut cursor).unwrap();

        match dechunker.next_item() {
            DechunkResult::Entry(entry_ref) => {
                let data = dechunker.get_data(&entry_ref);
                // Verify the entry matches what we created
                assert_eq!(data.len(), entry.len());
                assert_eq!(data, &entry[..]);
            }
            other => panic!("Expected Entry, got {:?}", other),
        }
    }
}
