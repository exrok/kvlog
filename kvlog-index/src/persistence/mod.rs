//! Optional persistence support for `kvlog-index`.
//!
//! This module contains the bucket log stream, an interning, archetype-aware
//! wire format. The stream reuses the bucket index's existing dedup tables
//! (via high water marks on `Index::new_ranges`, `new_uuids`, `archetype_map`,
//! `msg_map`, `span_table`) rather than maintaining a parallel set of hash
//! maps. Each interned identity is declared once per bucket file and
//! referenced by a small id thereafter. See `format.rs` for the wire layout.
//!
//! All persistence is opt-in via [`IndexConfig`]. The index produces byte
//! buffers which the caller forwards to whatever sink it likes (file,
//! network, off-thread queue). No IO happens on the ingest path.

pub mod encoder;
pub mod format;
pub mod replay;

pub use crate::index::{
    BucketSnapshotScratch, BucketSnapshotSlices, GlobalInternError, LoadedBucket, PersistentInterners,
    SnapshotLoadError, SnapshotWriteError, ValidationMode,
};
pub use encoder::BucketLogStreamEncoder;
pub use format::{ReadError, StreamFooter, StreamHeader, STREAM_FOOTER_LEN, STREAM_HEADER_LEN, STREAM_VERSION};

/// Runtime configuration for [`crate::index::Index`].
///
/// Persistence is disabled by default. Set `persistence_enabled` to allocate
/// the active log buffer and attach an encoder.
#[derive(Debug, Clone, Default)]
pub struct IndexConfig {
    pub persistence_enabled: bool,
}

/// A byte buffer drained out of the index, tagged with the bucket it belongs
/// to. Concatenating drained buffers in order produces a complete stream
/// file.
pub struct DrainedLogBuffer {
    pub bytes: Vec<u8>,
    pub generation: u64,
    /// True when these bytes end with the file footer. Implies the bucket
    /// has finished and no further bytes will be appended for `generation`.
    pub closed: bool,
}

#[cfg(test)]
mod integration_test;
