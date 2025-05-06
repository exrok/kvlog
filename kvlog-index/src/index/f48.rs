const HIGHEST_BIT: u64 = 1 << 63;

/// Both smaller to fit in the field and perserve order
pub type F48 = u64;
#[inline]
pub fn f64_to_f48(val: f64) -> F48 {
    let bits = val.to_bits();

    let nerb = if val.is_sign_positive() { bits ^ HIGHEST_BIT } else { !bits };
    nerb >> 16
}

#[inline]
pub fn f48_to_f64(val: F48) -> f64 {
    let val = val << 16;
    f64::from_bits(if val & HIGHEST_BIT != 0 { val ^ HIGHEST_BIT } else { !val })
}

#[cfg(test)]
mod test {
    use super::{f48_to_f64, f64_to_f48};

    #[test]
    fn impacted_float_round_trip() {
        for original in [-234.234f64, 0.0, 184.0] {
            // Note Storing as plain f32 would fail this test.
            let round_tripped = f48_to_f64(f64_to_f48(original));
            let diff = (original - round_tripped).abs();
            if diff > 0.00000001 {
                panic!("TOO LOSSY diff: {:?} input: {}", diff, original)
            }
        }
    }
    #[test]
    fn comparison() {
        for x in [-234.234f64, 0.0, 184.0, 34293949234.0] {
            for y in [-234.234f64, 0.0, 184.0, 34293949234.0] {
                let x_conv = f64_to_f48(x);
                let y_conv = f64_to_f48(y);
                assert_eq!(x > y, x_conv > y_conv);
                assert_eq!(x < y, x_conv < y_conv);
            }
        }
    }
}
