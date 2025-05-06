use jiff::{
    civil::{self, Date, DateTime, Time},
    fmt::temporal::Pieces,
    tz::{OffsetConflict, TimeZone},
    Error as JiffError, Span, Timestamp, ToSpan, Zoned,
};

pub fn parse_timestamp_to_nanosecond_range(timestamp_str: &str) -> Result<(i128, i128), FuzzyTimeParseError> {
    {
        let fuzz: FuzzyIsoTime = timestamp_str.parse()?;
        let (start, end) = fuzz.into_range().map_err(|_| FuzzyTimeParseError::InvalidTimestamp)?;
        return Ok((start.as_nanosecond(), end.as_nanosecond()));
    }
}

fn parse_timezone(mut text: &str) -> Option<TimeZone> {
    if let Some(inner) = text.strip_prefix("[") {
        if let Some(inner) = inner.strip_suffix("]") {
            text = inner;
        }
    }
    if text == "z" || text == "Z" {
        return Some(TimeZone::UTC);
    }
    let parser = jiff::fmt::temporal::DateTimeParser::new();
    parser.parse_time_zone(text).ok()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Precision {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Nanosecond(u8), // Number of sub-second digits actually present in the input string
}

#[derive(Debug)]
struct FuzzyIsoTime {
    year: i16,
    month: i8,
    day: i8,
    hour: i8,
    minute: i8,
    second: i8,
    subsec_nanosecond: i32,
    timezone: TimeZone,
    precision: Precision,
}

impl FuzzyIsoTime {
    /// Based on the precision convert the into a min and max Timestamp that would match
    /// the given time.
    fn into_range(mut self) -> Result<(jiff::Timestamp, jiff::Timestamp), jiff::Error> {
        let min = jiff::civil::date(self.year, self.month, self.day)
            .at(self.hour, self.minute, self.second, self.subsec_nanosecond)
            .to_zoned(self.timezone.clone())?;

        match self.precision {
            Precision::Year => {
                self.month = 12;
                self.day = 31;
                self.hour = 23;
                self.minute = 59;
                self.second = 59;
                self.subsec_nanosecond = 999_999_999;
            }
            Precision::Month => {
                self.day = min.date().days_in_month();
                self.hour = 23;
                self.minute = 59;
                self.second = 59;
                self.subsec_nanosecond = 999_999_999;
            }
            Precision::Day => {
                self.hour = 23;
                self.minute = 59;
                self.second = 59;
                self.subsec_nanosecond = 999_999_999;
            }
            Precision::Hour => {
                self.minute = 59;
                self.second = 59;
                self.subsec_nanosecond = 999_999_999;
            }
            Precision::Minute => {
                self.second = 59;
                self.subsec_nanosecond = 999_999_999;
            }
            Precision::Second => {
                self.subsec_nanosecond = 999_999_999;
            }
            Precision::Nanosecond(num_digits) => {
                let power = 9 - num_digits;
                let increment_nanos = 10_i32.pow(power as u32).saturating_sub(1);
                self.subsec_nanosecond = self.subsec_nanosecond.saturating_add(increment_nanos);
            }
        }

        let max = jiff::civil::date(self.year, self.month, self.day)
            .at(self.hour, self.minute, self.second, self.subsec_nanosecond)
            .to_zoned(self.timezone.clone())?;

        Ok((min.timestamp(), max.timestamp()))
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum FuzzyTimeParseError {
    InvalidTimestamp,
    InvalidTimezone,
    InvalidYear,
    InvalidMonth,
    InvalidDay,
    InvalidHour,
    InvalidMinute, // Added this variant, assuming it was an oversight
    InvalidSecond,
    InvalidSubsec,
}

// Helper to parse N digits and advance the slice
fn parse_fixed_digits(slice: &mut &str, n: usize) -> Option<i64> {
    let (digits_str, rest) = slice.split_at_checked(n)?;
    *slice = rest;
    digits_str.parse::<i64>().ok()
}

// Helper to parse 1 to max_n digits and advance the slice
fn parse_variable_digits(slice: &mut &str, max_n: usize) -> Option<(i64, usize)> {
    let mut count = 0;
    // Using slice.bytes() for potentially faster iteration over ASCII characters.
    for (idx, char_byte) in slice.bytes().enumerate() {
        if idx >= max_n || !char_byte.is_ascii_digit() {
            break;
        }
        count += 1;
    }

    if count == 0 {
        return None;
    }

    let (digits_str, rest) = slice.split_at(count);
    *slice = rest;
    match digits_str.parse::<i64>() {
        Ok(val) => Some((val, count)),
        Err(_) => None, // Should ideally not happen if only ASCII digits are taken
    }
}

impl std::str::FromStr for FuzzyIsoTime {
    type Err = FuzzyTimeParseError;

    fn from_str(mut s: &str) -> Result<Self, Self::Err> {
        let mut result = FuzzyIsoTime {
            year: 0,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            subsec_nanosecond: 0,
            timezone: TimeZone::UTC,
            precision: Precision::Year,
        };

        let year_val = parse_fixed_digits(&mut s, 4).ok_or(FuzzyTimeParseError::InvalidYear)?;
        result.year = year_val as i16;
        result.precision = Precision::Year;
        'at_offset: {
            macro_rules! finish_if_not_consume {
                ($pat: expr) => {
                    if let Some(rest) = s.strip_prefix($pat) {
                        s = rest;
                    } else {
                        break 'at_offset;
                    }
                };
            }
            // Month (-MM)
            {
                finish_if_not_consume!('-');
                let month_val = parse_fixed_digits(&mut s, 2).ok_or(FuzzyTimeParseError::InvalidMonth)?;
                result.month = month_val as i8;
                result.precision = Precision::Month;
            }

            // Day (-DD)
            {
                finish_if_not_consume!('-');
                let day_val = parse_fixed_digits(&mut s, 2).ok_or(FuzzyTimeParseError::InvalidDay)?;
                result.day = day_val as i8;
                result.precision = Precision::Day;
            }

            // Hour (tHH)
            {
                finish_if_not_consume!(['T', 't']);
                // If 'T' is present, hour must follow.
                let hour_val = parse_fixed_digits(&mut s, 2).ok_or(FuzzyTimeParseError::InvalidHour)?;
                result.hour = hour_val as i8;
                result.precision = Precision::Hour;
            }

            // Minute (:MM)
            {
                finish_if_not_consume!(':');
                let minute_val = parse_fixed_digits(&mut s, 2).ok_or(FuzzyTimeParseError::InvalidMinute)?;
                result.minute = minute_val as i8;
                result.precision = Precision::Minute;
            }

            // Second (:SS)
            {
                finish_if_not_consume!(':');
                let second_val = parse_fixed_digits(&mut s, 2).ok_or(FuzzyTimeParseError::InvalidSecond)?;
                result.second = second_val as i8;
                result.precision = Precision::Second;
            }

            // Subsecond (.fffffffff)
            {
                finish_if_not_consume!('.');

                // Must be followed by at least one digit.
                if s.is_empty() || !s.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                    return Err(FuzzyTimeParseError::InvalidSubsec);
                }

                let (subsec_raw_val, num_digits) = parse_variable_digits(&mut s, 9)
                    .expect("parse_variable_digits should succeed: already checked for first digit.");

                // Scale to nanoseconds. E.g., "123" (3 digits) is 123 * 10^(9-3) nanoseconds.
                result.subsec_nanosecond = (subsec_raw_val * 10_i64.pow(9 - num_digits as u32)) as i32;
                result.precision = Precision::Nanosecond(num_digits as u8);
            }
        }

        if !s.is_empty() {
            result.timezone = parse_timezone(s).ok_or(FuzzyTimeParseError::InvalidTimezone)?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use std::str::FromStr;

    use super::*;

    #[track_caller]
    fn assert_temporal_range_eq(input: &str, expected_min: &str, expected_max: &str) {
        let (got_min_ns, got_max_ns) = parse_timestamp_to_nanosecond_range(input).unwrap();
        let exp_min_ns = expected_min.parse::<jiff::Timestamp>().unwrap().as_nanosecond();
        let exp_max_ns = expected_max.parse::<jiff::Timestamp>().unwrap().as_nanosecond();
        if exp_min_ns != got_min_ns {
            panic!(
                "Unexpected min nanoseconds for '{}'\n Expected: {} ({})\n      Got: {} ({})",
                input,
                exp_min_ns,
                expected_min,
                got_min_ns,
                jiff::Timestamp::from_nanosecond(got_min_ns).unwrap()
            );
        }
        if exp_max_ns != got_max_ns {
            panic!(
                "Unexpected max nanoseconds for '{}'\n Expected: {} ({})\n      Got: {} ({})",
                input,
                exp_max_ns,
                expected_max,
                got_max_ns,
                jiff::Timestamp::from_nanosecond(got_max_ns).unwrap()
            );
        }
    }

    #[test]
    fn fuzzy_time() {
        let x = FuzzyIsoTime::from_str("2025-05-22T08:32:22.121342349[EST]").unwrap();
        println!("{:#?}", x);
        // laz("[America/New_York]").unwrap();
    }
    #[test]
    fn temporal_range_parsing() {
        assert_temporal_range_eq("2025", "2025-01-01T00:00:00Z", "2025-12-31T23:59:59.999999999Z");
        assert_temporal_range_eq("2025-05", "2025-05-01T00:00:00Z", "2025-05-31T23:59:59.999999999Z");
        assert_temporal_range_eq("2025-05-22", "2025-05-22T00:00:00Z", "2025-05-22T23:59:59.999999999Z");
        assert_temporal_range_eq("2025-05-22T08", "2025-05-22T08:00:00Z", "2025-05-22T08:59:59.999999999Z");
        assert_temporal_range_eq("2025-05-22T08:32", "2025-05-22T08:32:00Z", "2025-05-22T08:32:59.999999999Z");
        assert_temporal_range_eq("2025-05-22T08:32:22", "2025-05-22T08:32:22Z", "2025-05-22T08:32:22.999999999Z");
        assert_temporal_range_eq(
            "2025-05-22T08:32:22.123",
            "2025-05-22T08:32:22.123Z",
            "2025-05-22T08:32:22.123999999Z",
        );
        assert_temporal_range_eq(
            "2025-05-22T08:32:22.12356",
            "2025-05-22T08:32:22.12356Z",
            "2025-05-22T08:32:22.123569999Z",
        );
    }
}
