use std::{
    ptr::NonNull,
    sync::{atomic::AtomicUsize, Mutex},
};

use ahash::RandomState;
use hashbrown::HashTable;

use crate::index::InternedRange;

#[derive(Default)]
pub struct LocalIntermentCache {
    cache: HashTable<InternedRange>,
}
impl LocalIntermentCache {
    pub fn intern(&mut self, buf: &SharedIntermentBuffer, value: &[u8]) -> u16 {
        let hash = buf.hasher.hash_one(value);
        let map = self.cache.entry(
            hash,
            |v| value == unsafe { buf.bytes_unchecked(*v) },
            |v| buf.hasher.hash_one(unsafe { buf.bytes_unchecked(*v) }),
        );
        match map {
            hashbrown::hash_table::Entry::Occupied(entry) => entry.get().data,
            hashbrown::hash_table::Entry::Vacant(entry) => {
                let range = buf.insert_precached(hash, value);
                entry.insert(range);
                range.data
            }
        }
    }
}
struct InnerBuffer {
    lookup: HashTable<InternedRange>,
    written: usize,
}

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
            NonNull::new(std::alloc::alloc(
                std::alloc::Layout::from_size_align(data_capacity, 1).unwrap(),
            ))
            .unwrap()
        };
        let slices = unsafe {
            NonNull::new(std::alloc::alloc(
                std::alloc::Layout::array::<InternedRange>(u16::MAX as usize).unwrap(),
            ) as *mut InternedRange)
            .unwrap()
        };
        SharedIntermentBuffer {
            hasher: RandomState::new(),
            lut: Mutex::new(InnerBuffer {
                lookup: HashTable::default(),
                written: 0,
            }),
            data_capacity,
            data,
            slices,
            len: AtomicUsize::new(0),
        }
    }
    fn insert_precached(&self, hash: u64, value: &[u8]) -> InternedRange {
        let mut inner = self.lut.lock().unwrap();
        let inner: &mut InnerBuffer = &mut *inner;
        let id = inner.lookup.len() as u16;
        let map = inner.lookup.entry(
            hash,
            |v| value == unsafe { self.bytes_unchecked(*v) },
            |v| self.hasher.hash_one(unsafe { self.bytes_unchecked(*v) }),
        );
        match map {
            hashbrown::hash_table::Entry::Occupied(entry) => return *entry.get(),
            hashbrown::hash_table::Entry::Vacant(entry) => {
                if let Err(err) = std::str::from_utf8(value) {
                    // should generally not happen expect bad performacne
                    // we assume this want happen
                    let ve = err.valid_up_to();
                    kvlog::error!("Invalid UTF8 Target", ?err);
                    let truncated = &value[..ve];
                    let hash = self.hasher.hash_one(truncated);
                    return self.insert_precached(hash, truncated);
                }
                if (value.len() > (u16::MAX as usize)) {
                    panic!("Value exeeced max size")
                }
                if inner.written + value.len() > self.data_capacity {
                    panic!("Interment Cache Buffer Size Exceeded")
                }
                let start = inner.written;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        value.as_ptr(),
                        self.data.as_ptr().add(start),
                        value.len(),
                    );
                }
                inner.written += value.len();
                let len = value.len() as u16;
                let range = InternedRange {
                    data: id,
                    offset: start as u32,
                    len,
                };
                entry.insert(range);
                unsafe {
                    self.slices.as_ptr().add(id as usize).write(range);
                }
                self.len.fetch_add(1, std::sync::atomic::Ordering::Release);
                return range;
            }
        }
    }
    unsafe fn bytes_unchecked(&self, range: InternedRange) -> &[u8] {
        std::slice::from_raw_parts(
            self.data.as_ptr().add(range.offset as usize),
            range.len as usize,
        )
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
                std::slice::from_raw_parts(
                    self.slices.as_ptr(),
                    self.len.load(std::sync::atomic::Ordering::Acquire),
                )
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
        mapper
            .slices
            .iter()
            .map(|range| (range.data, unsafe { self.data_unchecked(*range) }))
    }
    pub fn iter(&self) -> impl Iterator<Item = (u16, &str)> + '_ {
        self.mapper()
            .slices
            .iter()
            .map(|range| (range.data, unsafe { self.data_unchecked(*range) }))
    }
}

pub struct Mapper<'a> {
    data: NonNull<u8>,
    slices: &'a [InternedRange],
}
impl<'a> Mapper<'a> {
    pub fn iter(&self) -> impl Iterator<Item = (u16, &str)> + '_ {
        self.slices
            .iter()
            .map(|range| (range.data, unsafe { self.data_unchecked(*range) }))
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
