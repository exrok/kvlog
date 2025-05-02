use kvlog::LogLevel;

use super::LogEntry;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct LevelFilter {
    pub mask: u8,
}
impl Default for LevelFilter {
    fn default() -> Self {
        LevelFilter::all()
    }
}
impl LevelFilter {
    pub fn empty() -> LevelFilter {
        LevelFilter { mask: 0 }
    }
    pub fn all() -> LevelFilter {
        LevelFilter { mask: 0b1111 }
    }
    pub fn with(self, level: LogLevel) -> LevelFilter {
        LevelFilter {
            mask: self.mask | (1 << (level as u8)),
        }
    }
    pub fn without(self, level: LogLevel) -> LevelFilter {
        LevelFilter {
            mask: self.mask & !(1 << (level as u8)),
        }
    }
    pub fn contains(&self, level: LogLevel) -> bool {
        (self.mask & (1 << (level as u8))) != 0
    }
    pub fn matches(&self, entry: LogEntry) -> bool {
        (entry.raw_bloom() & self.mask) != 0
    }
}
