use std::sync::Arc;

use kvlog::encoding::Value;

use crate::{
    index::BucketGuard,
    query::{QueryExpr, QueryParseError},
    shared_interner::Mapper,
};

use super::{KeyID, LogEntry};

#[derive(Debug, Clone)]
pub struct QueryFilter {
    pub query: Arc<QueryExpr>,
}

impl QueryFilter {
    pub fn from_str<'a, 'b>(input: &str) -> Result<QueryFilter, QueryParseError> {
        Ok(QueryFilter { query: QueryExpr::new_arc(input)? })
    }
    #[inline]
    pub fn matches(&self, entry: LogEntry, mapper: &Mapper) -> bool {
        self.query.pred().matches(entry, mapper)
    }
}
