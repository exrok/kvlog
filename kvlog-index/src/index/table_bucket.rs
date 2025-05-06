use std::sync::Arc;

use super::*;

//todo figure out how the global key interment fits into this.
#[repr(C)]
struct SavedBucketHeader {
    version: u64,
    uuids_interned: u64,
    slices_interned: u64,
    spans: u64,
    archetypes: u64,
    logs: u64,
    data: u64,
}

unsafe fn as_byte_slice<T: Sized>(value: &T) -> &[u8] {
    std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>())
}

impl<'a> BucketGuard<'a> {
    // uses the CStr naming convention, because this is non trival
    /// Note: If the bucket is still being written too this value might
    /// change even without rewewing the Bucket Guard
    pub fn serialized_byte_count(&self) -> usize {
        let size_from_maps = {
            let maps = self.maps();
            maps.data_len + (maps.uuid.len() * size_of::<u32>()) + (maps.general.len() * size_of::<InternedRange>())
        };
        //not the data size was add based on the maps

        #[rustfmt::skip]
        let mut per_log_bytes = self.len * (
              size_of::<BloomLine>() //bloom
            + size_of::<u16>()       //target
            + size_of::<u32>()       //span_id
            + size_of::<u64>()       //timestamp
            + size_of::<u32>()       //offset
            + size_of::<u16>()       //archetype
        );
        per_log_bytes += size_of::<u32>(); // (first offset value)

        let archetype_bytes = self.archetypes().len() * size_of::<archetype::Archetype>();
        let span_bytes = self.spans().len() * size_of::<SpanInfo>();

        /// Safety, offset is always initialize one passed the lenght
        let field_count = unsafe { self.bucket.offset.add(self.len).read() };

        let field_bytes = field_count as usize * size_of::<Field>();
        size_from_maps + per_log_bytes + span_bytes + archetype_bytes + size_of::<SavedBucketHeader>() + field_bytes
    }

    pub fn serialize(&self, mut out: &mut Vec<u8>) {
        let byte_count = self.serialized_byte_count();
        out.reserve(byte_count);
        let spans = self.spans();
        let archetypes = self.archetypes();
        let field_count = unsafe { self.bucket.offset.add(self.len).read() } as usize;
        let data_len;
        {
            let maps = self.maps();
            {
                let header = SavedBucketHeader {
                    version: 0,
                    uuids_interned: maps.uuid.len() as u64,
                    slices_interned: maps.general.len() as u64,
                    spans: spans.len() as u64,
                    archetypes: archetypes.len() as u64,
                    logs: self.len as u64,
                    data: maps.data_len as u64,
                };
                data_len = maps.data_len;
                out.extend_from_slice(unsafe { as_byte_slice(&header) });
            }
            for uuid_index in &maps.uuid {
                out.extend_from_slice(unsafe { as_byte_slice(uuid_index) });
            }
            for range in &maps.general {
                out.extend_from_slice(unsafe { as_byte_slice(range) });
            }
        }
        unsafe fn extend_native_copy<T>(out: &mut Vec<u8>, ptr: NonNull<T>, len: usize) {
            out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(ptr.as_ptr().cast::<u8>(), len * size_of::<T>())
            });
        }
        unsafe {
            extend_native_copy(&mut out, self.bucket.data, data_len);
            extend_native_copy(&mut out, self.bucket.field, field_count);
            // Serialize Spans (SpanRange)
            extend_native_copy(&mut out, self.bucket.span_data, spans.len());
            // Serialize Archetypes
            extend_native_copy(&mut out, self.bucket.archetype, archetypes.len());

            // Serialize per-log data arrays
            // Bloom lines
            extend_native_copy(&mut out, self.bucket.bloom, self.len);
            // Target IDs
            extend_native_copy(&mut out, self.bucket.target, self.len);
            // Span Indices
            extend_native_copy(&mut out, self.bucket.span_index, self.len);
            // Timestamps
            // let mut prev = 0;
            // let timestamps = unsafe { std::slice::from_raw_parts(self.bucket.timestamp.as_ptr(), self.len) };
            // for ts in timestamps {
            //     let delta = (*ts - prev);
            //     prev = *ts;
            //     out.extend_from_slice(&(delta as u32).to_ne_bytes());
            // }
            extend_native_copy(&mut out, self.bucket.timestamp, self.len);
            // Offsets (length is self.len + 1)
            extend_native_copy(&mut out, self.bucket.offset, self.len + 1);
            // Archetype Indices
            extend_native_copy(&mut out, self.bucket.archetype_index, self.len);
        }
    }
}
