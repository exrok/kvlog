use super::LogEntry;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TimeFilter {
    pub min_utc_ns: u64,
    pub max_utc_ns: u64,
}
impl Default for TimeFilter {
    fn default() -> Self {
        Self {
            min_utc_ns: u64::MIN,
            max_utc_ns: u64::MAX,
        }
    }
}
impl TimeFilter {
    pub fn matches(&self, entry: LogEntry) -> bool {
        let ts = entry.timestamp();
        if ts < self.min_utc_ns || ts > self.max_utc_ns {
            return false;
        }
        return true;
    }
}
