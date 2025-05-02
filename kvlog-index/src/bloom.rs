// use std::harh::Hasher;

use kvlog::LogLevel;

use crate::index::Field;
// use crate::util::*;

#[derive(Default)]
pub struct BloomBuilder(pub u8);

pub struct BloomQuery(u8, pub u8);

impl BloomBuilder {
    // fn merge(&mut self, value: u16) {
    //     let mut value = value % 124;
    //     let bits = if value > 63 {
    //         value -= 64 - 4;
    //         &mut self.1
    //     } else {
    //         &mut self.0
    //     };
    //     *bits |= 1u64 << value;
    // }

    pub fn insert(&mut self, _node: Field) {
        //todo optimize
        // let mut hasher = ahash::AHasher::default();
        // hasher.write_u64(node.raw);
        // let x = hasher.finish();
        // self.merge(x as u16);
        // self.merge((x >> 16) as u16);
        // self.merge((x >> 32) as u16);
        // self.merge((x >> 48) as u16);
    }

    pub fn query(self, level_mask: u8) -> BloomQuery {
        // Todo implment level filitng
        BloomQuery(level_mask, level_mask)
    }
    pub fn line(self, level: LogLevel) -> BloomLine {
        BloomLine(1u8 << (level as u8))
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct BloomLine(pub u8);

impl BloomLine {
    pub fn level(&self) -> LogLevel {
        match self.0 & 0xf {
            0x1 => LogLevel::Debug,
            0x2 => LogLevel::Info,
            0x4 => LogLevel::Warn,
            0x8 => LogLevel::Error,
            // todo consider something else
            // atleast panic in debug_assert
            _ => LogLevel::Debug,
        }
    }
    // pub fn len(&self) -> usize {
    //     (self.1 & 0xff) as usize
    // }
    pub fn matches(&self, query: &BloomQuery) -> bool {
        (self.0 & query.0) != 0
    }
}
