use std::{
    ptr::NonNull,
    sync::{atomic::AtomicUsize, Mutex},
};

use ahash::RandomState;
use hashbrown::HashTable;

use crate::index::InternedRange;

const MAX_INTERNED_TARGETS: usize = u16::MAX as usize + 1;

#[derive(Default)]
pub struct LocalIntermentCache {
    cache: HashTable<InternedRange>,
}
impl LocalIntermentCache {
    /// Intern a target name. Returns `None` if the underlying shared buffer
    /// is exhausted (cardinality cap, value too large, or backing storage
    /// full). Callers must drop the offending record rather than panic.
    pub fn intern(&mut self, buf: &SharedIntermentBuffer, mut value: &[u8]) -> Option<u16> {
        if let Err(err) = std::str::from_utf8(value) {
            let valid_up_to = err.valid_up_to();
            kvlog::error!("Invalid UTF8 Target", ?err);
            value = &value[..valid_up_to];
        }
        let hash = buf.hasher.hash_one(value);
        let map = self.cache.entry(
            hash,
            |v| value == unsafe { buf.bytes_unchecked(*v) },
            |v| buf.hasher.hash_one(unsafe { buf.bytes_unchecked(*v) }),
        );
        match map {
            hashbrown::hash_table::Entry::Occupied(entry) => Some(entry.get().data),
            hashbrown::hash_table::Entry::Vacant(entry) => {
                let range = buf.insert_precached(hash, value)?;
                entry.insert(range);
                Some(range.data)
            }
        }
    }
}
struct InnerBuffer {
    lookup: HashTable<InternedRange>,
    written: usize,
}

unsafe impl Sync for SharedIntermentBuffer {}
unsafe impl Send for SharedIntermentBuffer {}

pub struct SharedIntermentBuffer {
    hasher: RandomState,
    lut: Mutex<InnerBuffer>,
    data_capacity: usize,
    data: NonNull<u8>,
    slices: NonNull<InternedRange>,
    len: AtomicUsize,
}
// unsafe impl Sync for SharedIntermentBuffer {}

#[derive(Default)]
pub struct IntermentInsertionPoller {
    next: usize,
}

impl SharedIntermentBuffer {
    pub fn with_capacity(data_capacity: usize) -> SharedIntermentBuffer {
        let data = unsafe {
            NonNull::new(std::alloc::alloc(std::alloc::Layout::from_size_align(data_capacity, 1).unwrap())).unwrap()
        };
        let slices = unsafe {
            NonNull::new(std::alloc::alloc(std::alloc::Layout::array::<InternedRange>(MAX_INTERNED_TARGETS).unwrap())
                as *mut InternedRange)
            .unwrap()
        };
        SharedIntermentBuffer {
            hasher: RandomState::new(),
            lut: Mutex::new(InnerBuffer { lookup: HashTable::default(), written: 0 }),
            data_capacity,
            data,
            slices,
            len: AtomicUsize::new(0),
        }
    }
    /// Insert a precomputed-hashed value into the buffer.
    ///
    /// Returns `None` (instead of panicking) when:
    /// - the unique target count would exceed `MAX_INTERNED_TARGETS`
    /// - the value length exceeds `u16::MAX`
    /// - the backing byte buffer is full
    ///
    /// The first failure of each kind logs at error level.
    fn insert_precached(&self, hash: u64, value: &[u8]) -> Option<InternedRange> {
        let mut inner = self.lut.lock().unwrap();
        let inner: &mut InnerBuffer = &mut *inner;
        let id = inner.lookup.len();
        let map = inner.lookup.entry(
            hash,
            |v| value == unsafe { self.bytes_unchecked(*v) },
            |v| self.hasher.hash_one(unsafe { self.bytes_unchecked(*v) }),
        );
        match map {
            hashbrown::hash_table::Entry::Occupied(entry) => return Some(*entry.get()),
            hashbrown::hash_table::Entry::Vacant(entry) => {
                if id >= MAX_INTERNED_TARGETS {
                    kvlog::error!("Interment cache target count exceeded; dropping target", id);
                    return None;
                }
                if value.len() > (u16::MAX as usize) {
                    kvlog::error!("Interment cache value exceeded max size; dropping target", value_len = value.len());
                    return None;
                }
                if inner.written + value.len() > self.data_capacity {
                    kvlog::error!(
                        "Interment cache buffer size exceeded; dropping target",
                        written = inner.written,
                        value_len = value.len()
                    );
                    return None;
                }
                let start = inner.written;
                unsafe {
                    std::ptr::copy_nonoverlapping(value.as_ptr(), self.data.as_ptr().add(start), value.len());
                }
                inner.written += value.len();
                let len = value.len() as u16;
                let range = InternedRange { data: id as u16, offset: start as u32, len };
                entry.insert(range);
                unsafe {
                    self.slices.as_ptr().add(id as usize).write(range);
                }
                self.len.fetch_add(1, std::sync::atomic::Ordering::Release);
                return Some(range);
            }
        }
    }
    unsafe fn bytes_unchecked(&self, range: InternedRange) -> &[u8] {
        std::slice::from_raw_parts(self.data.as_ptr().add(range.offset as usize), range.len as usize)
    }
    unsafe fn data_unchecked(&self, range: InternedRange) -> &str {
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(
            self.data.as_ptr().add(range.offset as usize),
            range.len as usize,
        ))
    }
    pub fn mapper(&self) -> Mapper {
        Mapper {
            data: self.data,
            slices: unsafe {
                std::slice::from_raw_parts(self.slices.as_ptr(), self.len.load(std::sync::atomic::Ordering::Acquire))
            },
        }
    }
    pub fn poll_insertions(
        &self,
        poller: &mut IntermentInsertionPoller,
    ) -> impl Iterator<Item = (u16, &str)> + std::iter::ExactSizeIterator + '_ {
        let mut mapper = self.mapper();
        let new_next = mapper.slices.len();
        mapper.slices = mapper.slices.get(poller.next..).unwrap_or_default();
        poller.next = new_next;
        mapper.slices.iter().map(|range| (range.data, unsafe { self.data_unchecked(*range) }))
    }
    pub fn iter(&self) -> impl Iterator<Item = (u16, &str)> + '_ {
        /// some code assumes range.data = index in slice
        self.mapper().slices.iter().map(|range| (range.data, unsafe { self.data_unchecked(*range) }))
    }
}

pub struct Mapper<'a> {
    pub(crate) data: NonNull<u8>,
    pub(crate) slices: &'a [InternedRange],
}
impl<'a> Mapper<'a> {
    pub fn len(&self) -> usize {
        self.slices.len()
    }
    pub fn empty() -> Mapper<'a> {
        Mapper { data: NonNull::dangling(), slices: &[] }
    }
    pub fn iter(&self) -> impl Iterator<Item = (u16, &str)> + '_ {
        self.slices.iter().map(|range| (range.data, unsafe { self.data_unchecked(*range) }))
    }
    unsafe fn data_unchecked(&self, range: InternedRange) -> &str {
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(
            self.data.as_ptr().add(range.offset as usize),
            range.len as usize,
        ))
    }
}

impl<'a> Mapper<'a> {
    pub fn get(&self, index: u16) -> Option<&str> {
        let range = self.slices.get(index as usize)?;
        Some(unsafe { self.data_unchecked(*range) })
    }
}
impl<'a> std::ops::Index<u16> for Mapper<'a> {
    type Output = str;

    fn index(&self, index: u16) -> &Self::Output {
        let range = &self.slices[index as usize];
        unsafe { self.data_unchecked(*range) }
    }
}
