use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use crate::index::{EntryCollection, GeneralFilter};

enum QueryChunk<'a> {
    Append(EntryCollection<'a>),
    Prepend(EntryCollection<'a>),
}
struct Query {
    query_id: u64,
    filters: Vec<GeneralFilter>,
    newest_first: bool,
}
struct LiveQueryShared {
    query_id: AtomicU64,
    current_query: Mutex<Query>,
}
struct LiveQuery {
    counter: u64,
    shared: Arc<LiveQueryShared>,
}
struct LiveQueryWorker {
    shared: Arc<LiveQueryShared>,
}

impl LiveQueryWorker {
    pub fn work(&mut self) {}
}
use std::convert::identity;

impl LiveQuery {
    pub fn update_query(&mut self, query: Vec<GeneralFilter>) {
        let mut shared = self.shared.current_query.lock().unwrap();
        self.counter += 1;
        *shared = Query {
            query_id: self.counter,
            filters: query,
            newest_first: false,
        };
        self.shared.query_id.store(self.counter, Ordering::Release)
    }
    pub async fn next(&self) -> QueryChunk<'_> {
        todo!();
    }
}
