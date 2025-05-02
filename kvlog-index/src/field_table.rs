use std::{
    alloc::{alloc, Layout},
    ptr::NonNull,
    sync::Mutex,
};

use hashbrown::HashTable;
use kvlog::encoding::{get_static_key, StaticKey};

pub struct KeyMap<T> {
    data: Vec<T>,
}

impl<T> Default for KeyMap<T> {
    fn default() -> Self {
        Self {
            data: Default::default(),
        }
    }
}

impl<T> KeyMap<T> {
    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        self.data.iter()
    }
    pub fn iter(&self) -> impl Iterator<Item = (KeyID, &T)> + '_ {
        self.data
            .iter()
            .enumerate()
            .map(|(raw, value)| (unsafe { KeyID::new(raw as u16) }, value))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (KeyID, &mut T)> + '_ {
        self.data
            .iter_mut()
            .enumerate()
            .map(|(raw, value)| (unsafe { KeyID::new(raw as u16) }, value))
    }
    pub fn clear(&mut self) {
        self.data.clear();
    }
    pub fn get(&self, key: KeyID) -> Option<&T> {
        self.data.get(key.0 as usize)
    }
    pub fn get_mut(&mut self, key: KeyID) -> Option<&mut T> {
        self.data.get_mut(key.0 as usize)
    }
}

impl<T: Default> KeyMap<T> {
    pub fn insert(&mut self, key: KeyID, value: T) {
        *self.get_mut_or_default(key) = value;
    }
    pub fn get_mut_or_default(&mut self, key: KeyID) -> &mut T {
        if self.data.len() <= key.0 as usize {
            self.data.resize_with(key.0 as usize + 1, T::default);
        }
        // Safety: From check above
        unsafe { self.data.get_unchecked_mut(key.0 as usize) }
    }
}

static mut BUFFER: NonNull<u8> = NonNull::dangling();
const BUFFER_SIZE: usize = 1024 * 64;
static mut LUT: NonNull<u32> = NonNull::dangling();
const MAX_KEY: usize = 4096;

struct Global {
    table: HashTable<KeyID>,
    buffer_used: usize,
}
static TABLE: Mutex<Global> = Mutex::new(Global {
    table: HashTable::new(),
    buffer_used: 0,
});

// fn name() {
//     unsafe { BUFFER }
// }

pub const MIN_DYN_KEY: u16 = StaticKey::NAMES.len() as u16;

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KeyID(pub u16);
impl std::fmt::Debug for KeyID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'a> PartialEq<StaticKey> for KeyID {
    fn eq(&self, other: &StaticKey) -> bool {
        self.0 == *other as u16
    }
}

impl KeyID {
    pub fn raw(self) -> u16 {
        self.0
    }
    pub unsafe fn new(value: u16) -> KeyID {
        KeyID(value)
    }
    pub fn as_str(self) -> &'static str {
        if self.0 < MIN_DYN_KEY {
            return StaticKey::from_u8(self.0 as u8).unwrap().as_str();
        }
        unsafe {
            let offset = *LUT.as_ptr().add((self.0 - MIN_DYN_KEY) as usize);
            let len = offset >> 24;
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                BUFFER.as_ptr().add((offset & 0x00FF_FFFF) as usize),
                len as usize,
            ))
        }
    }
    pub fn try_from_str(name: &str) -> Option<KeyID> {
        if let Some(value) = get_static_key(name) {
            return Some(KeyID(value as u16));
        }
        let global = TABLE.lock().unwrap();
        let hasher = ahash::RandomState::with_seeds(0, 0, 0, 0);
        let hash = hasher.hash_one(name);
        global.table.find(hash, |key| name == key.as_str()).copied()
    }
    pub fn intern(name: &str) -> KeyID {
        if let Some(value) = get_static_key(name) {
            return KeyID(value as u16);
        }
        if name.len() > 127 {
            panic!();
        }
        let mut global = TABLE.lock().unwrap();
        let global: &mut Global = &mut *global;
        let table = &mut global.table;
        let buffer_used = &mut global.buffer_used;
        if table.is_empty() {
            unsafe {
                BUFFER = NonNull::new(alloc(Layout::new::<[u8; BUFFER_SIZE]>())).unwrap();
                LUT = NonNull::new(alloc(Layout::new::<[u32; MAX_KEY]>()) as *mut u32).unwrap();
            }
        }

        let len = table.len();
        let hasher = ahash::RandomState::with_seeds(0, 0, 0, 0);
        let hash = hasher.hash_one(name);
        match table.entry(
            hash,
            |key| name == key.as_str(),
            |value| hasher.hash_one(value.as_str()),
        ) {
            hashbrown::hash_table::Entry::Occupied(entry) => *entry.get(),
            hashbrown::hash_table::Entry::Vacant(entry) => {
                let start = *buffer_used;
                if *buffer_used + name.len() >= BUFFER_SIZE {
                    panic!("GLOBAL KEY BUFFER SIZED EXCEEDED")
                }
                *buffer_used += name.len();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        name.as_ptr(),
                        BUFFER.as_ptr().add(start),
                        name.len(),
                    )
                }
                unsafe {
                    *LUT.as_ptr().add(len as usize) = start as u32 | ((name.len() as u32) << 24)
                }

                let id = KeyID((len as u16) + MIN_DYN_KEY);
                entry.insert(id);
                id
            }
        }
    }
}
