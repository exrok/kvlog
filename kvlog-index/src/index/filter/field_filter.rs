use kvlog::encoding::Value;

use crate::query::FieldPredicate;

use super::{KeyID, LogEntry};

#[derive(Debug, Clone)]
pub struct FieldFilter {
    pub fields: Box<[(KeyID, FieldPredicate)]>,
}
impl FieldFilter {
    pub fn matches(&self, entry: LogEntry) -> bool {
        let mut predicates_iter = self.fields.iter();
        // perf: Try storing a KeyID shifted with bits sets to speed up query
        // Field::max_field_bits_with_key(*next_key)
        let Some((next_key, next_predicate)) = predicates_iter.next() else {
            return true;
        };
        let mut key = *next_key;
        let mut predicate = next_predicate;

        'outer: for field in entry.raw_fields() {
            loop {
                unsafe {
                    let field_key = field.key();
                    if field_key < key {
                        continue 'outer;
                    }
                    if field_key == key {
                        if !predicate.matches(field.value(entry.bucket)) {
                            return false;
                        }
                    } else {
                        if !predicate.matches_missing() {
                            return false;
                        }
                    }
                    let Some((next_key, next_predicate)) = predicates_iter.next() else {
                        return true;
                    };
                    key = *next_key;
                    predicate = next_predicate;
                    continue;
                }
            }
        }
        if !predicate.matches_missing() {
            return false;
        }
        for (_, predicate) in predicates_iter {
            if !predicate.matches_missing() {
                return false;
            }
        }
        true
    }
}
