use std::mem::MaybeUninit;

/// UTC timestamp with nanosecond precision.
///
/// Represents a point in time as seconds and nanoseconds since the Unix epoch
/// (1970-01-01T00:00:00Z). Supports timestamps before the epoch using negative
/// seconds values.
///
/// # Examples
///
/// ```
/// use kvlog::Timestamp;
///
/// // Create from milliseconds since Unix epoch
/// let ts = Timestamp::from_millisecond(1705670400000);
///
/// // Access components
/// let secs = ts.seconds();
/// let nanos = ts.subsec_nanos();
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp {
    pub(crate) seconds: i64,
    pub(crate) nanos: u32,
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Timestamp as std::fmt::Debug>::fmt(&self, f)
    }
}

impl std::fmt::Debug for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buffer = new_buffer();
        let text =
            Timestamp::raw_millisecond_iso_in_buffer(self.as_millisecond_clamped(), &mut buffer);
        let without_z = unsafe { text.get_unchecked(0..text.len() - 1) };
        f.write_str(without_z)?;
        write!(f, "{:06}Z", self.nanos % 1_000_000)?;
        Ok(())
    }
}

impl Timestamp {
    /// Appends a millisecond precision UTC timestamp in ISO Format to the buffer
    pub fn raw_millisecond_iso_in_buffer(ms_timestamp: i64, buffer: &mut Buffer) -> &str {
        let len = Timestamp::write_millisecond_iso(ms_timestamp, buffer);
        let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, len) };
        unsafe { std::str::from_utf8_unchecked(slice) }
    }

    /// Appends a millisecond precision UTC timestamp in ISO format to the output
    pub fn raw_millisecond_iso_in_vec(ms_timestamp: i64, output: &mut Vec<u8>) {
        output.reserve(25);
        let len = unsafe {
            Timestamp::write_millisecond_iso(
                ms_timestamp,
                &mut *(output.spare_capacity_mut().as_mut_ptr() as *mut [MaybeUninit<u8>; 25]),
            )
        };
        unsafe {
            output.set_len(output.len() + len);
        }
    }
    /// Appends the given timestamp in ISO 8601 format to the output
    /// example: '2025-04-17T09:56:52.232Z' or '1965-08-23T12:00:00.000Z'
    /// The `ms_timestamp` is given in milliseconds since UNIX epoch (1970-01-01T00:00:00Z).
    /// Handles timestamps before the epoch (negative values) using proleptic Gregorian calendar.
    fn write_millisecond_iso(ms_timestamp: i64, output: &mut [MaybeUninit<u8>; 25]) -> usize {
        let ms_timestamp = ms_timestamp.clamp(MIN_ISO_TIMESTAMP, MAX_ISO_TIMESTAMP);
        let mut ptr = output.as_mut_ptr() as *mut u8;
        let days_since_epoch = ms_timestamp.div_euclid(MS_PER_DAY);
        let ms_today = ms_timestamp.rem_euclid(MS_PER_DAY);
        let Date(year, month, day) = to_date(days_since_epoch as i32);

        unsafe {
            ptr.write(b'-');
            let off = (year < 0) as usize;
            let new_len = 24 + off;
            ptr = ptr.add(off);

            let abs_year = year.unsigned_abs();
            write_2digit_number(ptr.add(0), (abs_year / 100) as usize);
            write_2digit_number(ptr.add(2), (abs_year % 100) as usize);

            ptr.add(4).write(b'-');
            write_2digit_number(ptr.add(5), month as usize);
            ptr.add(7).write(b'-');

            write_2digit_number(ptr.add(8), day as usize);
            ptr.add(10).write(b'T');

            let h = ms_today / MS_PER_HOUR;
            let hours_today = ms_today % MS_PER_HOUR;
            write_2digit_number(ptr.add(11), h as usize);
            ptr.add(13).write(b':');

            let m = (hours_today / MS_PER_MIN) as usize;
            write_2digit_number(ptr.add(14), m);
            ptr.add(16).write(b':');

            let s = (hours_today % MS_PER_MIN) / MS_PER_SECOND;
            write_2digit_number(ptr.add(17), s as usize);

            ptr.add(19).write(b'.');

            let ms = ms_today % MS_PER_SECOND;
            ptr.add(20).write(b'0' + (ms / 100) as u8);
            write_2digit_number(ptr.add(21), (ms % 100) as usize);

            ptr.add(23).write(b'Z');
            new_len
        }
    }
    /// Returns the sub-second component in nanoseconds.
    ///
    /// The returned value is always in the range `0..1_000_000_000`.
    pub fn subsec_nanos(&self) -> u32 {
        self.nanos
    }

    /// Returns the number of whole seconds since the Unix epoch.
    ///
    /// Negative values represent timestamps before the epoch.
    pub fn seconds(&self) -> i64 {
        self.seconds
    }

    /// Creates a new timestamp, clamping nanoseconds to valid range.
    ///
    /// If `nanos` exceeds 999,999,999 it is clamped to that maximum value.
    pub fn new_clamped(seconds: i64, nanos: u32) -> Timestamp {
        let mut nanos = nanos;
        if nanos > 999_999_999 {
            nanos = 999_999_999;
        }
        Timestamp { seconds, nanos }
    }

    /// Creates a timestamp from milliseconds since the Unix epoch.
    ///
    /// Negative values represent timestamps before the epoch.
    pub fn from_millisecond(millisecond: i64) -> Timestamp {
        let seconds = millisecond.div_euclid(1000);
        let nanos = millisecond.rem_euclid(1000) * 1_000_000;
        Timestamp {
            seconds,
            nanos: nanos as u32,
        }
    }

    /// Converts this timestamp to milliseconds since the Unix epoch.
    ///
    /// Uses saturating arithmetic to avoid overflow.
    pub fn as_millisecond_clamped(&self) -> i64 {
        self.seconds.saturating_mul(1000) + (self.nanos / 1_000_000) as i64
    }
}

// Precomputed lookup table for formatting two decimal digits (00-99) quickly.

#[rustfmt::skip]
    const DEC_DIGITS_LUT: [u8; 256] = [
        b'0', b'0', b'0', b'1', b'0', b'2', b'0', b'3', b'0', b'4', b'0', b'5', b'0', b'6', b'0',
        b'7', b'0', b'8', b'0', b'9', b'1', b'0', b'1', b'1', b'1', b'2', b'1', b'3', b'1', b'4',
        b'1', b'5', b'1', b'6', b'1', b'7', b'1', b'8', b'1', b'9', b'2', b'0', b'2', b'1', b'2',
        b'2', b'2', b'3', b'2', b'4', b'2', b'5', b'2', b'6', b'2', b'7', b'2', b'8', b'2', b'9',
        b'3', b'0', b'3', b'1', b'3', b'2', b'3', b'3', b'3', b'4', b'3', b'5', b'3', b'6', b'3',
        b'7', b'3', b'8', b'3', b'9', b'4', b'0', b'4', b'1', b'4', b'2', b'4', b'3', b'4', b'4',
        b'4', b'5', b'4', b'6', b'4', b'7', b'4', b'8', b'4', b'9', b'5', b'0', b'5', b'1', b'5',
        b'2', b'5', b'3', b'5', b'4', b'5', b'5', b'5', b'6', b'5', b'7', b'5', b'8', b'5', b'9',
        b'6', b'0', b'6', b'1', b'6', b'2', b'6', b'3', b'6', b'4', b'6', b'5', b'6', b'6', b'6',
        b'7', b'6', b'8', b'6', b'9', b'7', b'0', b'7', b'1', b'7', b'2', b'7', b'3', b'7', b'4',
        b'7', b'5', b'7', b'6', b'7', b'7', b'7', b'8', b'7', b'9', b'8', b'0', b'8', b'1', b'8',
        b'2', b'8', b'3', b'8', b'4', b'8', b'5', b'8', b'6', b'8', b'7', b'8', b'8', b'8', b'9',
        b'9', b'0', b'9', b'1', b'9', b'2', b'9', b'3', b'9', b'4', b'9', b'5', b'9', b'6', b'9',
        b'7', b'9', b'8', b'9', b'9',
        // Safety Pad:
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
unsafe fn write_2digit_number(dst: *mut u8, value: usize) {
    // safety the bit mask ensures that value will not overflow
    unsafe {
        (dst as *mut [u8; 2])
            .write(*(DEC_DIGITS_LUT.as_ptr() as *const [u8; 2]).add(value & 0b0111_1111));
    }
}
pub const fn new_buffer() -> Buffer {
    let buffer = MaybeUninit::<Buffer>::uninit();
    unsafe { buffer.assume_init() }
}
pub type Buffer = [MaybeUninit<u8>; 25];

//todo use real MIN and MAX not jiffs fake ones.
pub const MIN_ISO_TIMESTAMP: i64 = -377705023201000;
pub const MAX_ISO_TIMESTAMP: i64 = 253402207200999;

const MS_PER_DAY: i64 = 86400000;
const MS_PER_HOUR: i64 = 3600000;
const MS_PER_MIN: i64 = 60000;
const MS_PER_SECOND: i64 = 1000;

pub struct Date(i32, u32, u32);
pub fn to_date(n_u: i32) -> Date {
    const S: u32 = 82;
    const K: u32 = 719468 + 146097 * S;
    const L: u32 = 400 * S;
    let n = (n_u as u32).wrapping_add(K);

    let n_1 = 4 * n + 3;
    let c = n_1 / 146097;
    let n_c = n_1 % 146097 / 4;

    let n_2 = 4 * n_c + 3;
    let p_2 = 2939745 as u64 * (n_2 as u64);
    let z = (p_2 >> 32) as u32;
    let n_y = (p_2 as u32) / (2939745 * 4);
    let y = 100 * c + z;

    let n_3 = 2141 * n_y + 197913;
    let m = n_3 >> 16;
    let d = (n_3 & 0xffff) / 2141;

    let j = n_y >= 306;
    let y_g = ((y - L) + (j as u32)) as i32;
    let m_g = if j { m - 12 } else { m };
    let d_g = d + 1;

    return Date(y_g, m_g, d_g);
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn timestamp_conversions() {
        for input in [i64::MAX, -1, -999_999, -1000_000, 0, 999_999, i64::MAX] {
            let output = Timestamp::from_millisecond(input).as_millisecond_clamped();
            assert_eq!(input, output)
        }
        for input in -1999..2000 {
            let output = Timestamp::from_millisecond(input).as_millisecond_clamped();
            assert_eq!(input, output)
        }
    }
    #[test]
    fn timestamp_fmt() {
        assert_eq!(
            Timestamp::new_clamped(0, 0).to_string(),
            "1970-01-01T00:00:00.000000000Z"
        );
        assert_eq!(
            Timestamp::new_clamped(-1, 999_999_999).to_string(),
            "1969-12-31T23:59:59.999999999Z"
        )
    }
}
