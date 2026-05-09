//! Connected-component traversal of the span tree.
//!
//! The "context query" feature in libra_monitoring needs every log whose
//! span is in the connected component of a seed span: walk parents to the
//! root, then enumerate every descendant of every ancestor (including the
//! seed itself), then ask the query layer for all logs whose span is in
//! that set.
//!
//! Per-bucket data:
//! - `parent_index` gives a span its in-bucket parent slot.
//! - `first_child_index` / `next_sibling_index` give the in-bucket
//!   children (singly-linked, newest first).
//!
//! Cross-bucket: a span that is observable in multiple buckets carries
//! distinct in-bucket parent and child links per bucket. Children are not
//! replicated across buckets; only ancestors are (up to
//! [`super::MAX_CROSS_BUCKET_PARENT_DEPTH`]). The traversal therefore
//! visits every live bucket and unions the per-bucket subtree views.
//!
//! Single writer / many readers. Readers tolerate seeing a child slot
//! whose value is `>= bucket.span_count` (the splice may install a slot
//! that has not yet been published into this reader's snapshot) by
//! bound-checking on each step.

use hashbrown::HashSet;
use kvlog::SpanID;

use super::{BucketGuard, GenerationGuard, SPAN_CHILD_INDEX_NONE, SPAN_PARENT_INDEX_NONE};
use std::sync::Arc;
use std::sync::atomic::Ordering;

impl<'a> GenerationGuard<'a> {
    /// Compute the connected component of `seed` in the span forest visible
    /// across all live buckets. Returns `None` when `seed` is not present
    /// in any live bucket (eg, its bucket has been reclaimed).
    ///
    /// The component always contains `seed` if any bucket holds it.
    pub fn span_component(&self, seed: SpanID) -> Option<Arc<HashSet<SpanID>>> {
        let mut component: HashSet<SpanID> = HashSet::new();
        let mut queue: Vec<SpanID> = Vec::new();

        let mut found_in_any = false;
        for bucket in self.buckets() {
            if find_span_slot(bucket, seed).is_some() {
                found_in_any = true;
                break;
            }
        }
        if !found_in_any {
            return None;
        }

        component.insert(seed);
        queue.push(seed);

        while let Some(span_id) = queue.pop() {
            for bucket in self.buckets() {
                let Some(slot) = find_span_slot(bucket, span_id) else { continue };
                let spans = bucket.spans();
                let span = &spans[slot as usize];

                let parent_slot = span.parent_index;
                if parent_slot != SPAN_PARENT_INDEX_NONE && (parent_slot as usize) < spans.len() {
                    let parent = &spans[parent_slot as usize];
                    if component.insert(parent.id) {
                        queue.push(parent.id);
                    }
                }

                let mut child_slot = span.first_child_index.load(Ordering::Acquire);
                while child_slot != SPAN_CHILD_INDEX_NONE && (child_slot as usize) < spans.len() {
                    let child = &spans[child_slot as usize];
                    if component.insert(child.id) {
                        queue.push(child.id);
                    }
                    child_slot = child.next_sibling_index.load(Ordering::Acquire);
                }
            }
        }

        Some(Arc::new(component))
    }

    /// Extend an existing component set with any newly-arrived spans whose
    /// parent is already in the set. Live-mode delta sweep: meant to be
    /// called after each ingest wakeup with the slot range that has become
    /// visible since the last call. Returns `true` when the set changed.
    ///
    /// `bucket` is the bucket that received the new spans, and `range` is
    /// the half-open span-slot range `[old_span_count, new_span_count)` of
    /// freshly published slots.
    pub fn extend_component_from_new_spans(
        &self,
        component: &mut HashSet<SpanID>,
        bucket: &BucketGuard<'a>,
        range: std::ops::Range<u32>,
    ) -> bool {
        let spans = bucket.spans();
        let mut changed = false;
        for slot in range {
            if (slot as usize) >= spans.len() {
                break;
            }
            let span = &spans[slot as usize];
            let Some(parent_id) = span.parent else { continue };
            if component.contains(&parent_id) && component.insert(span.id) {
                changed = true;
            }
        }
        changed
    }
}

/// Linear scan of `bucket.spans()` looking for `target`. Used at component
/// setup time and once per BFS step. Bounded by `BUCKET_SPAN_RANGE_SIZE`
/// per bucket — fine because (1) it runs at most a handful of times per
/// query and (2) span fan-out is small.
fn find_span_slot(bucket: &BucketGuard, target: SpanID) -> Option<u32> {
    bucket.spans().iter().position(|sr| sr.id == target).map(|i| i as u32)
}
