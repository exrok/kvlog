use hashbrown::HashSet;
use kvlog::encoding::Key;

use crate::field_table::KeyID;

pub enum KeySet {
    Inline(InlineSet),
    Alloc(HashSet<u16>),
}

pub enum KeySetIter<'a> {
    Inline(std::slice::Iter<'a, KeyID>),
    Alloc(hashbrown::hash_set::Iter<'a, u16>),
}

impl<'a> IntoIterator for &'a KeySet {
    type Item = KeyID;
    type IntoIter = KeySetIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        match self {
            KeySet::Inline(inline_set) => KeySetIter::Inline(inline_set.iter()),
            KeySet::Alloc(hash_set) => KeySetIter::Alloc(hash_set.iter()),
        }
    }
}

impl std::iter::Iterator for KeySetIter<'_> {
    type Item = KeyID;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            KeySetIter::Inline(iter) => iter.next().copied(),
            KeySetIter::Alloc(iter) => iter.next().map(|x| KeyID(*x)),
        }
    }
}

impl KeySet {
    pub const fn new() -> KeySet {
        KeySet::Inline(InlineSet::new())
    }
    pub fn clear(&mut self) {
        match self {
            KeySet::Inline(inline_set) => inline_set.clear(),
            KeySet::Alloc(hash_set) => hash_set.clear(),
        }
    }
    pub fn iter(&self) -> KeySetIter<'_> {
        self.into_iter()
    }
    pub fn len(&self) -> usize {
        match self {
            KeySet::Inline(inline_set) => inline_set.len,
            KeySet::Alloc(hash_set) => hash_set.len(),
        }
    }
    pub fn contains(&self, key: KeyID) -> bool {
        match self {
            KeySet::Inline(inline_set) => inline_set.contains(key),
            KeySet::Alloc(hash_set) => hash_set.contains(&key.0),
        }
    }
    pub fn insert(&mut self, key: KeyID) -> bool {
        match self {
            KeySet::Inline(inline_set) => match inline_set.try_insert(key) {
                Ok(value) => true,
                Err(_) => {
                    let mut hash_set = HashSet::with_capacity(16);
                    for item in inline_set.iter() {
                        hash_set.insert_unique_unchecked(item.0);
                    }
                    hash_set.insert_unique_unchecked(key.0);
                    *self = KeySet::Alloc(hash_set);
                    true
                }
            },
            KeySet::Alloc(hash_set) => return hash_set.insert(key.0),
        }
    }
}

// Note due to vectorization functions we can for almost zero cost, double the size of
// the buffer here, we don't because well, 8 keys is enough for our current usescases typically
// the number is less then 3
struct InlineSet {
    buffer: [u64; 2],
    len: usize,
}
#[no_mangle]
fn index_of(data: &[u16; 8], value: u16) -> Option<u8> {
    data.iter()
        .rposition(|&x| x == value) // .rposition() finds the last element satisfying the condition
        .map(|i| i as u8)
}
impl InlineSet {
    fn contains(&self, value: KeyID) -> bool {
        // On, target-cpu=x86-64-v4 (v3 is similar)
        // contains:
        //         vpbroadcastw    xmm0, esi
        //         vpcmpeqw        k0, xmm0, xmmword ptr [rdi]
        //         kortestb        k0, k0
        //         setne   al
        //         ret

        let v = value.0;
        let buffer = unsafe { &*((&self.buffer) as *const _ as *const [u16; 8]) };
        (buffer[0] == v)
            | (buffer[1] == v)
            | (buffer[2] == v)
            | (buffer[3] == v)
            | (buffer[4] == v)
            | (buffer[5] == v)
            | (buffer[6] == v)
            | (buffer[7] == v)
    }
    fn clear(&mut self) {
        *self = InlineSet::new();
    }
    // will panic if at capacity
    fn insert_unchecked(&mut self, value: KeyID) {
        let buffer = unsafe { &mut *((&mut self.buffer) as *mut _ as *mut [u16; 8]) };
        buffer[self.len as usize] = value.0;
        self.len += 1;
    }
    #[no_mangle]
    fn try_insert(&mut self, value: KeyID) -> Result<bool, ()> {
        if self.contains(value) {
            Ok(false)
        } else if self.len < 8 {
            self.insert_unchecked(value);
            Ok(true)
        } else {
            Err(())
        }
    }
    const fn new() -> InlineSet {
        InlineSet { buffer: [u64::MAX; 2], len: 0 }
    }
    fn as_slice(&self) -> &[KeyID] {
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr() as *const _ as *const KeyID, self.len) }
    }
    fn iter(&self) -> std::slice::Iter<'_, KeyID> {
        self.as_slice().into_iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn basic_insert() {
        let mut set = InlineSet::new();
        let max = KeyID(0x7ffe);
        let min = KeyID(0);
        assert!(!set.contains(max));
        assert!(!set.contains(min));
        set.try_insert(max).unwrap();
        set.try_insert(min).unwrap();
        assert!(set.contains(max));
        assert!(set.contains(min));
        for i in 34..34 + 6 {
            let i = KeyID(i);
            assert!(!set.contains(i));
            set.try_insert(i).unwrap();
            assert!(set.contains(i));
        }
        set.try_insert(max).unwrap();
        set.try_insert(min).unwrap();
        assert!(set.try_insert(KeyID(23423)).is_err())
    }
}
