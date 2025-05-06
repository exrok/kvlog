use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    num::{NonZero, NonZeroU16, NonZeroU8},
    sync::{atomic::AtomicU8, OnceLock},
};

use super::{filter::LevelFilter, *};

// fn lookup_service_set(index: u8, bitset: &[u64; 4]) -> bool {
//     bitset[index & 0b11] & (1u64 << (index >> 2)) != 0
// }

// fn lookup_service_set(index: u8, bitset: &[u16; 16]) -> bool {
//     (unsafe { *bitset.as_ptr().byte_add((index as usize) & (!0b11)) }) & (1u16.wrapping_shl(index as u32)) != 0
// }

fn lookup_service_set(index: u8, bitset: &[u32; 8]) -> bool {
    (unsafe { *bitset.as_ptr().byte_add((index as usize) & (!0b111)) }) & (1u32.wrapping_shl(index as u32)) != 0
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ServiceId(NonZeroU8);

impl std::fmt::Debug for ServiceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

#[repr(C)]
struct ServiceTable {
    data: UnsafeCell<[MaybeUninit<&'static str>; 256]>,
}
static GLOBAL_SERVICE_TABLE_LEN: AtomicU8 = AtomicU8::new(0);
static GLOBAL_SERVICE_TABLE: ServiceTable =
    ServiceTable { data: unsafe { UnsafeCell::new(MaybeUninit::uninit().assume_init()) } };

static GLOBAL_SERVICE_LOOKUP_TABLE: Mutex<HashTable<ServiceId>> = Mutex::new(HashTable::new());

impl ServiceId {
    pub fn known() -> impl Iterator<Item = ServiceId> + ExactSizeIterator {
        let len = GLOBAL_SERVICE_TABLE_LEN.load(std::sync::atomic::Ordering::Acquire) as u16;
        (1..len + 1).map(|i| ServiceId(unsafe { NonZeroU8::new_unchecked(i as u8) }))
    }
    pub fn as_u8(self) -> u8 {
        self.0.get()
    }
    pub fn as_str(self) -> &'static str {
        unsafe {
            (GLOBAL_SERVICE_TABLE.data.get() as *const MaybeUninit<&'static str>)
                .add(self.0.get() as usize)
                .read()
                .assume_init()
        }
    }
    pub fn try_from_str(text: &str) -> Option<ServiceId> {
        let hasher = ahash::RandomState::with_seeds(
            0xf0b5_7675_87ab_51aa,
            0xbd7e_8b67_1fd8_e6e5,
            0xebbf_7f53_c3f1_b263,
            0x95d3_4885_a166_e32c,
        );
        let hash = hasher.hash_one(text);
        let lookup = GLOBAL_SERVICE_LOOKUP_TABLE.lock().unwrap();
        if let Some(id) = lookup.find(hash, |value| value.as_str() == text) {
            return Some(*id);
        }
        None
    }
    pub fn intern(text: &str) -> ServiceId {
        let hasher = ahash::RandomState::with_seeds(
            0xf0b5_7675_87ab_51aa,
            0xbd7e_8b67_1fd8_e6e5,
            0xebbf_7f53_c3f1_b263,
            0x95d3_4885_a166_e32c,
        );
        let hash = hasher.hash_one(text);
        let mut lookup = GLOBAL_SERVICE_LOOKUP_TABLE.lock().unwrap();
        let len = lookup.len();
        match lookup.entry(hash, |value| value.as_str() == text, |value| hasher.hash_one(value.as_str())) {
            hashbrown::hash_table::Entry::Occupied(occupied_entry) => return *occupied_entry.get(),
            hashbrown::hash_table::Entry::Vacant(vacant_entry) => {
                let id = len + 1;
                if id > 255 {
                    panic!("Service ID overflow, exceeded maximum number of services 255");
                }
                let foo: Box<str> = text.into();
                let leaked: &'static str = Box::leak(foo);

                unsafe {
                    (GLOBAL_SERVICE_TABLE.data.get() as *mut _ as *mut MaybeUninit<&'static str>)
                        .add(id as usize)
                        .write(MaybeUninit::new(leaked))
                }
                GLOBAL_SERVICE_TABLE_LEN.fetch_add(1, std::sync::atomic::Ordering::Release);
                let value = unsafe { ServiceId(NonZeroU8::new_unchecked(id as u8)) };
                vacant_entry.insert(value);
                value
            }
        }
    }
}

unsafe impl Send for ServiceTable {}
unsafe impl Sync for ServiceTable {}
// static GLOBAL_SERVER_BUFFER: OnceLock<SharedIntermentBuffer> = OnceLock::new();

#[derive(PartialOrd, Ord)]
#[repr(C, align(8))]
pub struct Archetype {
    pub(crate) msg_offset: u32,
    pub(crate) msg_len: u16,
    pub(crate) target_id: u16,
    // first 4 bits is level
    // 5 bit is set when in span
    pub(crate) mask: u32,
    pub(crate) service: Option<ServiceId>,
    pub(crate) pad: u8,
    pub(crate) size: u16,
    pub field_headers: [u16; 8],
}

impl PartialEq for Archetype {
    fn eq(&self, other: &Self) -> bool {
        self.as_raw() == other.as_raw()
    }
}

impl Eq for Archetype {}

impl std::hash::Hash for Archetype {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for chunk in self.as_raw() {
            chunk.hash(state);
        }
    }
}

impl Archetype {
    pub(crate) fn service(&self) -> Option<ServiceId> {
        self.service
    }
    #[inline]
    pub(crate) fn raw_service(&self) -> u8 {
        // compiles away due to niche
        match self.service {
            Some(s) => s.0.get(),
            None => 0,
        }
    }
    #[inline]
    pub(crate) fn field_keys(&self) -> &[KeyID] {
        unsafe {
            std::slice::from_raw_parts(&self.field_headers as *const [u16; 8] as *const KeyID, self.size as usize)
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "bmi2"))]
    pub fn index_of(&self, key: KeyID) -> Option<usize> {
        let index =
            unsafe { index_of_simd_single_match(&*(&self.field_headers as *const _ as *const [u64; 2]), key.raw()) };
        if let Some(index) = index {
            return unsafe { Some(index as usize) };
        }

        for (i, header) in self.field_keys().iter().skip(8).enumerate() {
            if *header == key {
                return unsafe { Some(i) };
            }
        }
        None
        // let mask = (index.0 as u64) << 52;
        // for field in self.raw_fields() {
        //     if field.raw < mask {
        //         continue;
        //     }
        //     if field.raw ^ mask < ((1u64 << 53) - 1) {
        //         return Some(*field);
        //     }
        //     return None;
        // }
        // return None;
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "bmi2")))]
    pub fn index_of(&self, key: KeyID) -> Option<usize> {
        for (i, header) in self.field_keys().iter().enumerate() {
            if *header == key {
                return unsafe { Some(i) };
            }
        }
        None
    }
    pub(crate) fn as_raw(&self) -> &[u64; 4] {
        unsafe { &*(self as *const Archetype as *const [u64; 4]) }
    }
    pub fn level(&self) -> LogLevel {
        match self.mask & 0b1111 {
            0b0001 => LogLevel::Debug,
            0b0010 => LogLevel::Info,
            0b0100 => LogLevel::Warn,
            0b1000 => LogLevel::Error,
            _ => LogLevel::Info,
        }
    }
    pub fn contains_key(&self, key: KeyID) -> bool {
        self.field_headers.contains(&key.0)
    }
    // safetey: Bucket must be for Archetype of same generation
    pub unsafe fn print<'a>(&self, bucket: &BucketGuard<'a>) {
        unsafe {
            print!("{:?} `{}` [target: {}] [", self.level(), self.message(bucket).escape_ascii(), self.target_id);
            let mut output = false;
            for key in self.field_headers {
                if key == u16::MAX {
                    continue;
                }
                if output {
                    print!(", ");
                } else {
                    output = true;
                }
                let text = KeyID::try_raw_to_str(key).unwrap_or("UNKNOWN");
                print!("{text}");
            }
            println!("]");
        }
    }
    // safetey: Bucket must be for Archetype of same generation
    pub unsafe fn message<'a>(&self, bucket: &BucketGuard<'a>) -> &'a [u8] {
        bucket.bucket.data_unchecked(InternedRange { offset: self.msg_offset, data: 0, len: self.msg_len })
    }
    pub fn in_span(&self) -> bool {
        (self.mask & 0b10000) != 0
    }
    pub fn level_in(&self, mask_filter: LevelFilter) -> bool {
        (self.mask as u8) & 0b1111 & mask_filter.mask != 0
    }
    pub(crate) fn new(
        msg: InternedRange,
        target_id: u16,
        level: LogLevel,
        in_span: bool,
        service: Option<ServiceId>,
        fields: &[Field],
    ) -> Archetype {
        let mut mask = 1u32 << level as u8;
        if in_span {
            mask |= 1 << 4;
        }
        let mut field_headers = [u16::MAX; 8];
        for (i, field) in fields.iter().enumerate().take(8) {
            let key = field.raw_key();
            // let x1 = (key as u64).wrapping_mul(0x517cc1b727220a95) % 53;
            // let x2 = (key as u64) % 53;
            // mask |= 0b1_0_0000 << x1;
            // mask |= 0b1_0_0000 << x2;
            //todo add kind so we can do even more pre-querying???
            field_headers[i] = key;
        }
        Archetype {
            msg_offset: msg.offset,
            msg_len: msg.len,
            target_id,
            mask,
            service,
            pad: 0,
            size: fields.len().min(8) as u16,
            field_headers,
        }
    }
}
