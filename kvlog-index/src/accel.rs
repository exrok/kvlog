// #[cfg(target_arch = "x86")]
// use core::arch::x86;
// #[cfg(target_arch = "x86_64")]
// use core::arch::x86_64 as x86;
use std::{num::NonZeroU64, ptr::NonNull};

#[derive(Clone, Copy)]
pub struct MicroBloom {
    base: u64,
    level: u64,
}
impl MicroBloom {
    pub fn new(select_mask: u8) -> MicroBloom {
        MicroBloom {
            base: ((((!select_mask) & 0x7f) as u64) | 0xf) * 0x01_01_01_01__01_01_01_01,
            level: ((select_mask & 0xf) as u64) * 0x01_01_01_01__01_01_01_01,
        }
    }
    const H1: u64 = 0x80_80_80_80_80_80_80_80;
    pub fn scan_reverse(&self, blooms: &[u64], offset: usize, mut func: impl FnMut(usize) -> bool) -> bool {
        for (i, mid) in blooms.iter().enumerate().rev() {
            let bits = ((*mid) | self.base).wrapping_add(self.level & *mid) & Self::H1;
            let Some(mut mask) = NonZeroU64::new(bits.to_be()) else {
                continue;
            };
            let offset = offset + (i << 3);
            loop {
                let i = mask.trailing_zeros() >> 3;
                if !func(offset + (7 - (i as usize))) {
                    return false;
                }
                let Some(mask2) = NonZeroU64::new(mask.get() & (mask.get() - 1)) else {
                    break;
                };
                mask = mask2;
            }
        }
        return true;
    }
    pub fn scan_forward(&self, blooms: &[u64], offset: usize, mut func: impl FnMut(usize) -> bool) -> bool {
        for (i, mid) in blooms.iter().enumerate() {
            let bits = ((*mid) | self.base).wrapping_add(self.level & *mid) & Self::H1;
            let Some(mut mask) = NonZeroU64::new(bits) else {
                continue;
            };
            let offset = offset + (i << 3);
            loop {
                let i = mask.trailing_zeros() >> 3;
                if !func(offset + (i as usize)) {
                    return false;
                }
                let Some(mask2) = NonZeroU64::new(mask.get() & (mask.get() - 1)) else {
                    break;
                };
                mask = mask2;
            }
        }
        return true;
    }
    pub fn query_forward(
        data: &[u8],
        select_mask: u8,
        offset: usize,
        limit: usize,
        mut func: impl FnMut(usize) -> bool,
    ) {
        let mut offset = offset.min(data.len());
        let end = (offset + limit).min(data.len());
        let data = &data[offset..end];
        let (head, mid, tail) = unsafe { data.align_to::<u64>() };

        let min = select_mask & (!0xf);
        for (i, mask) in head.iter().enumerate() {
            if (mask & select_mask) > min && !func(offset + i) {
                return;
            }
        }
        offset += head.len();
        let mb = MicroBloom::new(select_mask);
        if !mb.scan_forward(mid, offset, &mut func) {
            return;
        }

        offset += mid.len() * 8;
        for (i, mask) in tail.iter().enumerate() {
            if (mask & select_mask) > min && !func(i + offset) {
                return;
            }
        }
    }
    pub fn query_reverse(
        data: &[u8],
        select_mask: u8,
        offset: usize,
        limit: usize,
        mut func: impl FnMut(usize) -> bool,
    ) {
        let mut offset = offset.min(data.len());
        let end = (offset + limit).min(data.len());
        let data = &data[offset..end];
        let (head, mid, tail) = unsafe { data.align_to::<u64>() };

        let min = select_mask & (!0xf);
        offset += head.len() + mid.len() * 8;
        for (i, mask) in tail.iter().enumerate().rev() {
            if (mask & select_mask) > min && !func(offset + i) {
                return;
            }
        }
        offset -= mid.len() * 8;
        let mb = MicroBloom::new(select_mask);
        if !mb.scan_reverse(mid, offset, &mut func) {
            return;
        }

        offset -= head.len();
        for (i, mask) in head.iter().enumerate().rev() {
            if (mask & select_mask) > min && !func(i + offset) {
                return;
            }
        }
    }
}
pub unsafe fn single_field_query_forward(
    fields: NonNull<u64>,
    offsets: NonNull<u32>,
    range: std::ops::Range<usize>,
    mut predicate: impl FnMut(u64) -> bool,
    mut map: impl FnMut(u32) -> bool,
) {
    let field_start = *offsets.as_ptr().add(range.start) as usize;
    let field_end = *offsets.as_ptr().add(range.end) as usize;
    let mut fields_slice = std::slice::from_raw_parts(fields.as_ptr().add(field_start), field_end - field_start);
    let offsetss = std::slice::from_raw_parts(offsets.as_ptr().add(range.start + 1), range.count());
    let mut offsets_walker = offsetss.iter().peekable();
    'outer: loop {
        for field in fields_slice {
            if !predicate(*field) {
                continue;
            }
            let offset = (field as *const u64).offset_from(fields.as_ptr()) as u32;
            while let Some(i) = offsets_walker.next() {
                if *i > offset {
                    let field = (i as *const u32).offset_from(offsets.as_ptr()) as u32 - 1;
                    map(field);
                    let field_start = *i as usize;
                    fields_slice =
                        std::slice::from_raw_parts(fields.as_ptr().add(field_start), field_end - field_start);
                    continue 'outer;
                }
            }
            unreachable!()
        }
        return;
    }
}

pub unsafe fn single_field_query_reverse(
    fields: NonNull<u64>,
    offsets: NonNull<u32>,
    range: std::ops::Range<usize>,
    mut predicate: impl FnMut(u64) -> bool,
    mut map: impl FnMut(u32) -> bool,
) {
    let field_start = *offsets.as_ptr().add(range.start) as usize;
    let field_end = *offsets.as_ptr().add(range.end) as usize;
    let mut fields_slice = std::slice::from_raw_parts(fields.as_ptr().add(field_start), field_end - field_start);
    let offsetss = std::slice::from_raw_parts(offsets.as_ptr().add(range.start + 1), range.count());
    let mut offsets_walker = offsetss.iter().rev().peekable();
    'outer: loop {
        for field in fields_slice.iter().rev() {
            if !predicate(*field) {
                continue;
            }
            let offset = (field as *const u64).offset_from(fields.as_ptr()) as u32;
            while let Some(i) = offsets_walker.next() {
                if *i < offset {
                    let field = (i as *const u32).offset_from(offsets.as_ptr()) as u32;
                    map(field);
                    let field_start = *i as usize;
                    fields_slice =
                        std::slice::from_raw_parts(fields.as_ptr().add(field_start), field_end - field_start);
                    continue 'outer;
                }
            }
            unreachable!()
        }
        return;
    }
}

pub fn slow_bloom_query(data: &[u8], levels: u8, extra: u8) -> impl Iterator<Item = usize> + DoubleEndedIterator + '_ {
    data.iter()
        .enumerate()
        .filter(move |(_, &mask)| ((mask & levels) != 0) && ((mask & extra) == extra))
        .map(|(i, _)| i)
}
impl U16Set {
    pub fn is_empty(&self) -> bool {
        self.bytes.iter().all(|&x| x == 0)
    }
}

impl Default for Box<U16Set> {
    #[inline]
    fn default() -> Self {
        let bytes: Box<[u64; 1024]> = vec![0; 1024].into_boxed_slice().try_into().unwrap();
        unsafe { Box::from_raw(Box::into_raw(bytes).cast()) }
    }
}
#[repr(transparent)]
pub struct U16Set {
    bytes: [u64; 1024],
}

impl U16Set {
    pub fn insert(&mut self, value: u16) {
        let index = value as usize >> 6;
        let offset = value as usize & 0b111111;
        self.bytes[index] |= 1 << offset;
    }
    pub fn contains(&self, value: u16) -> bool {
        let index = value as usize >> 6;
        let offset = value as usize & 0b111111;
        self.bytes[index] & (1 << offset) != 0
    }
}

/// Collect the indices of input in range with values in the set.
/// Inserts no more than `target_size + 8` into output.
/// Returns smallest index on checked. Indices are in sorted order.
///
/// Similar:
/// ```ignore
/// output.extend(
///     input.iter().enumerate()
///          .skip(range.start)
///          .take(range.end)
///          .filter(|(_ ,v)| set.contains(**v))
///          .map(|(i, _)| i)
///          .take(target_size)
///          .collect()
/// )
/// ```
pub fn collect_indices_of_values_in_set_forward(
    set: &U16Set,
    input: &[u16],
    input_range: std::ops::Range<usize>,
    output: &mut Vec<u32>,
    target_size: usize,
) -> usize {
    output.reserve(target_size.checked_add(8).expect("NO UB"));
    let len = output.len();
    let mut write = unsafe { output.as_mut_ptr().add(len) };
    let end = unsafe { write.add(target_size) };
    let (_, data, rest) = unsafe { input[input_range.clone()].align_to::<[u16; 8]>() };
    for i in data {
        unsafe {
            let start = ((&i[0]) as *const u16).offset_from(input.as_ptr()) as u32;
            //4 grouping helps mitigate performance loss in rust 1.78
            {
                let v0 = set.contains(i[0]) as usize;
                let v1 = set.contains(i[1]) as usize;
                let v2 = set.contains(i[2]) as usize;
                let v3 = set.contains(i[3]) as usize;
                *write = start + 0;
                write = write.add(v0);
                *write = start + 1;
                write = write.add(v1);
                *write = start + 2;
                write = write.add(v2);
                *write = start + 3;
                write = write.add(v3);
            }
            {
                let v4 = set.contains(i[4]) as usize;
                let v5 = set.contains(i[5]) as usize;
                let v6 = set.contains(i[6]) as usize;
                let v7 = set.contains(i[7]) as usize;
                *write = start + 4;
                write = write.add(v4);
                *write = start + 5;
                write = write.add(v5);
                *write = start + 6;
                write = write.add(v6);
                *write = start + 7;
                write = write.add(v7);
            }
        }
        if write >= end {
            unsafe {
                let new_len = write.offset_from(output.as_ptr()) as usize;
                output.set_len(new_len);
            }
            let offset = unsafe { ((&i[7]) as *const u16).offset_from(input.as_ptr()) as u32 };
            return (offset as usize) + 1;
        }
    }
    unsafe {
        let new_len = write.offset_from(output.as_ptr()) as usize;
        output.set_len(new_len);
    }
    for i in rest {
        if set.contains(*i) {
            let offset = unsafe { (i as *const u16).offset_from(input.as_ptr()) as u32 };
            output.push(offset);
            if output.len() >= target_size {
                return (offset as usize) + 1;
            }
        }
    }
    input_range.end
}

pub fn collect_indices_of_values_in_set_reverse(
    set: &U16Set,
    input: &[u16],
    input_range: std::ops::Range<usize>,
    output: &mut Vec<u32>,
    target_size: usize,
) -> usize {
    // Not if this where to overflow reverse with invalid layout panic;
    output.reserve(target_size.saturating_add(8));
    let len = output.len();
    let mut write = unsafe { output.as_mut_ptr().add(len) };
    let end = unsafe { write.add(target_size) };
    let mut current = input_range.end;

    let chunk_end = current;
    let chunk_start = chunk_end.saturating_sub(8);
    current = chunk_start;

    let main_length = input_range.len() & !0b111;
    let main_data: &[[u16; 8]] = unsafe {
        std::slice::from_raw_parts(input.as_ptr().add(input_range.end - main_length).cast(), main_length / 8)
    };

    for i in main_data.iter().rev() {
        unsafe {
            let start = ((&i[0]) as *const u16).offset_from(input.as_ptr()) as u32;
            // 4-grouping to mitigate performance loss
            {
                let v7 = set.contains(i[7]) as usize;
                let v6 = set.contains(i[6]) as usize;
                let v5 = set.contains(i[5]) as usize;
                let v4 = set.contains(i[4]) as usize;
                *write = start + 7;
                write = write.add(v7);
                *write = start + 6;
                write = write.add(v6);
                *write = start + 5;
                write = write.add(v5);
                *write = start + 4;
                write = write.add(v4);
            }
            {
                let v3 = set.contains(i[3]) as usize;
                let v2 = set.contains(i[2]) as usize;
                let v1 = set.contains(i[1]) as usize;
                let v0 = set.contains(i[0]) as usize;
                *write = start + 3;
                write = write.add(v3);
                *write = start + 2;
                write = write.add(v2);
                *write = start + 1;
                write = write.add(v1);
                *write = start + 0;
                write = write.add(v0);
            }
        }
        if write >= end {
            unsafe {
                let new_len = write.offset_from(output.as_ptr()) as usize;
                output.set_len(new_len);
            }
            let offset = unsafe { ((&i[0]) as *const u16).offset_from(input.as_ptr()) as u32 };
            return (offset as usize);
        }
    }

    unsafe {
        let new_len = write.offset_from(output.as_ptr()) as usize;
        output.set_len(new_len);
    }

    let rest_data: &[u16] =
        unsafe { std::slice::from_raw_parts(input.as_ptr().add(input_range.start), input_range.len() & 0b111) };

    // Process remaining items in reverse order
    for i in rest_data.iter().rev() {
        if set.contains(*i) {
            let offset = unsafe { (i as *const u16).offset_from(input.as_ptr()) as u32 };
            output.push(offset);
            if output.len() >= target_size {
                return (offset as usize) + 1;
            }
        }
    }

    input_range.start
}

#[cfg(test)]
mod test {

    use super::*;

    pub fn slow_bloom_query(
        data: &[u8],
        levels: u8,
        extra: u8,
    ) -> impl Iterator<Item = usize> + DoubleEndedIterator + '_ {
        data.iter()
            .enumerate()
            .filter(move |(_, &mask)| ((mask & levels) != 0) && ((mask & extra) == extra))
            .map(|(i, _)| i)
    }

    #[test]
    fn micro_bloom() {
        let mut rng = oorandom::Rand32::new(0xdeadbeaf);
        let data: Vec<u8> = (0..1000)
            .map(|_| {
                let x = rng.rand_u32();
                ((1 << (rng.rand_u32() & 0b11)) | x & 0b0111_0000) as u8
            })
            .collect();

        for (levels, extra) in [(0b1111, 0b0000), (0b1100, 0b0000), (0b1101, 0b0001_0000)] {
            let mut low = slow_bloom_query(&data, levels, extra);
            MicroBloom::query_forward(&data, levels | extra, 0, usize::MAX, |i| {
                println!("{:08b}", data[i]);
                assert_eq!(i, low.next().unwrap());
                true
            });
            assert!(low.next().is_none());
        }
        for (levels, extra) in [(0b1111, 0b0000), (0b1100, 0b0000), (0b1101, 0b0001_0000)] {
            let mut low = slow_bloom_query(&data, levels, extra).rev();
            MicroBloom::query_reverse(&data, levels | extra, 0, usize::MAX, |i| {
                println!("{:08b}", data[i]);
                assert_eq!(i, low.next().unwrap());
                true
            });
            assert!(low.next().is_none());
        }
    }

    struct FieldsMap {
        fields: Vec<u64>,
        offsets: Vec<u32>,
    }
    impl Default for FieldsMap {
        fn default() -> Self {
            Self { fields: Default::default(), offsets: vec![0] }
        }
    }

    impl FieldsMap {
        fn push(&mut self, data: &[u64]) {
            self.fields.extend_from_slice(data);
            self.offsets.push(self.fields.len() as u32);
        }
    }
    #[test]
    fn dex_accel_search() {
        let mut set: Box<U16Set> = Default::default();
        set.insert(4);
        set.insert(5);
        set.insert(12);
        let mut output = Vec::new();
        collect_indices_of_values_in_set_forward(
            &set,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            0..17,
            &mut output,
            10,
        );
        assert_eq!(&output, &[4, 5, 12]);
        output.clear();
        collect_indices_of_values_in_set_reverse(
            &set,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            0..17,
            &mut output,
            10,
        );
        assert_eq!(&output, &[12, 5, 4]);

        let mut set: Box<U16Set> = Default::default();
        set.insert(0);
        set.insert(1);
        output.clear();
        collect_indices_of_values_in_set_reverse(&set, &[0, 1, 2], 0..3, &mut output, 1000);
        assert_eq!(&output, &[1, 0]);
    }
}
