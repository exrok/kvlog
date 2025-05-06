use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RangePredicate<T> {
    pub negated: bool,
    pub min: T,
    pub max: T,
}
impl<T: Rangeable> RangePredicate<T> {
    fn contains(&self, value: T) -> bool {
        (value >= self.min && value <= self.max) ^ self.negated
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RangeMerge<T> {
    None,
    Merged(RangePredicate<T>),
    AlwaysTrue,
    AlwaysFalse,
}

pub trait Rangeable: PartialOrd + Copy + std::fmt::Debug {
    fn min_value() -> Self;
    fn max_value() -> Self;
    fn successor(val: Self) -> Option<Self>;
    fn predecessor(val: Self) -> Option<Self>;
}

impl Rangeable for i64 {
    fn min_value() -> Self {
        i64::MIN
    }
    fn max_value() -> Self {
        i64::MAX
    }
    fn successor(val: Self) -> Option<Self> {
        val.checked_add(1)
    }
    fn predecessor(val: Self) -> Option<Self> {
        val.checked_sub(1)
    }
}

impl Rangeable for u64 {
    fn min_value() -> Self {
        u64::MIN
    }
    fn max_value() -> Self {
        u64::MAX
    }
    fn successor(val: Self) -> Option<Self> {
        val.checked_add(1)
    }
    fn predecessor(val: Self) -> Option<Self> {
        val.checked_sub(1)
    }
}

impl Rangeable for f64 {
    fn min_value() -> Self {
        f64::NEG_INFINITY
    }
    fn max_value() -> Self {
        f64::INFINITY
    }
    fn successor(val: Self) -> Option<Self> {
        if val.is_finite() {
            Some(val.next_up())
        } else {
            None
        }
    }
    fn predecessor(val: Self) -> Option<Self> {
        if val.is_finite() {
            Some(val.next_down())
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Interval<T> {
    min: T,
    max: T,
}

fn to_intervals<T: Rangeable>(p: RangePredicate<T>, lo: T, hi: T, out: &mut [Interval<T>; 2]) -> usize {
    if lo > hi {
        return 0;
    }
    let mut len = 0;
    if !p.negated {
        let mn = if p.min > lo { p.min } else { lo };
        let mx = if p.max < hi { p.max } else { hi };
        if mn <= mx {
            out[0] = Interval { min: mn, max: mx };
            len = 1;
        }
    } else {
        if lo <= p.min {
            if let Some(pred) = T::predecessor(p.min) {
                let end = if pred < hi { pred } else { hi };
                if lo <= end {
                    out[len] = Interval { min: lo, max: end };
                    len += 1;
                }
            }
        }
        if p.max <= hi {
            if let Some(succ) = T::successor(p.max) {
                let start = if succ > lo { succ } else { lo };
                if start <= hi {
                    out[len] = Interval { min: start, max: hi };
                    len += 1;
                }
            }
        }
    }
    len
}

fn merge_intervals<T: Rangeable>(segs: &mut [Interval<T>; 4], count: usize) -> usize {
    if count == 0 {
        return 0;
    }

    segs[0..count].sort_by(|a, b| a.min.partial_cmp(&b.min).unwrap_or(Ordering::Equal));
    let mut write = 0;
    for read in 1..count {
        let cur = segs[read];
        let last = &mut segs[write];
        let can = if let Some(succ) = T::successor(last.max) { cur.min <= succ } else { cur.min <= last.max };
        if can {
            if cur.max > last.max {
                last.max = cur.max;
            }
        } else {
            write += 1;
            segs[write] = cur;
        }
    }
    write + 1
}

pub fn range_merge<T: Rangeable>(
    a: RangePredicate<T>,
    b: RangePredicate<T>,
    lo: T,
    hi: T,
    is_and: bool,
) -> RangeMerge<T> {
    let mut ia: [Interval<T>; 2] = [Interval { min: T::max_value(), max: T::min_value() }; 2];
    let mut ib = ia;
    let na = to_intervals(a, lo, hi, &mut ia);
    let nb = to_intervals(b, lo, hi, &mut ib);
    if is_and && (na == 0 || nb == 0) || !is_and && na + nb == 0 {
        return RangeMerge::AlwaysFalse;
    }
    if is_and && na == 1 && ia[0].min == lo && ia[0].max == hi {
        return intervals_to_merge(&mut ib, nb, lo, hi);
    }
    if is_and && nb == 1 && ib[0].min == lo && ib[0].max == hi {
        return intervals_to_merge(&mut ia, na, lo, hi);
    }
    if !is_and && (na == 1 && ia[0].min == lo && ia[0].max == hi || nb == 1 && ib[0].min == lo && ib[0].max == hi) {
        return RangeMerge::AlwaysTrue;
    }
    let mut all: [Interval<T>; 4] = [Interval { min: T::max_value(), max: T::min_value() }; 4];
    let mut c = 0;
    if is_and {
        for i in 0..na {
            for j in 0..nb {
                let l = if ia[i].min > ib[j].min { ia[i].min } else { ib[j].min };
                let h = if ia[i].max < ib[j].max { ia[i].max } else { ib[j].max };
                if l <= h {
                    all[c] = Interval { min: l, max: h };
                    c += 1;
                }
            }
        }
    } else {
        for i in 0..na {
            all[c] = ia[i];
            c += 1;
        }
        for j in 0..nb {
            all[c] = ib[j];
            c += 1;
        }
    }
    if c == 0 {
        return RangeMerge::AlwaysFalse;
    }
    let m = merge_intervals(&mut all, c);
    intervals_to_merge(&mut all, m, lo, hi)
}

fn intervals_to_merge<T: Rangeable>(segs: &mut [Interval<T>], n: usize, lo: T, hi: T) -> RangeMerge<T> {
    match n {
        0 => RangeMerge::AlwaysFalse,
        1 => {
            let s = segs[0];
            if s.min == lo && s.max == hi {
                RangeMerge::AlwaysTrue
            } else {
                RangeMerge::Merged(RangePredicate { negated: false, min: s.min, max: s.max })
            }
        }
        2 => {
            let s1 = segs[0];
            let s2 = segs[1];
            if s1.min == lo && s2.max == hi {
                if let (Some(hmin), Some(hmax)) = (T::successor(s1.max), T::predecessor(s2.min)) {
                    if hmin <= hmax {
                        return RangeMerge::Merged(RangePredicate { negated: true, min: hmin, max: hmax });
                    }
                }
            }
            RangeMerge::None
        }
        _ => RangeMerge::None,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn ordered_value_map<T: Copy>(values: &[T], mut func: impl FnMut(RangePredicate<T>, RangePredicate<T>)) {
        for (i, a_min) in values.iter().enumerate() {
            for (i, a_max) in values[i..].iter().enumerate() {
                for (i, b_min) in values.iter().enumerate() {
                    for (i, b_max) in values[i..].iter().enumerate() {
                        for negs in 0..4 {
                            func(
                                RangePredicate { negated: negs & 0b1 == 1, min: *a_min, max: *a_max },
                                RangePredicate { negated: negs > 1, min: *b_min, max: *b_max },
                            )
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn range_merge_test() {
        let sample_set = [i64::MIN, 2, 3, 4, 5, 6, i64::MAX];
        ordered_value_map(&[i64::MIN, 3, 5, 7, i64::MAX], |a, b| {
            let merged = range_merge::<i64>(a, b, i64::MIN, i64::MAX, true);
            match merged {
                RangeMerge::None => (),
                RangeMerge::Merged(merged) => {
                    for s in sample_set {
                        assert_eq!((a.contains(s) && b.contains(s)), merged.contains(s))
                    }
                }
                RangeMerge::AlwaysTrue => {
                    for s in sample_set {
                        assert!(a.contains(s) && b.contains(s))
                    }
                }
                RangeMerge::AlwaysFalse => {
                    for s in sample_set {
                        assert!(!(a.contains(s) && b.contains(s)))
                    }
                }
            }
        });
        ordered_value_map(&[i64::MIN, 3, 5, 7, i64::MAX], |a, b| {
            let merged = range_merge::<i64>(a, b, i64::MIN, i64::MAX, false);
            match merged {
                RangeMerge::None => (),
                RangeMerge::Merged(merged) => {
                    for s in sample_set {
                        assert_eq!((a.contains(s) || b.contains(s)), merged.contains(s))
                    }
                }
                RangeMerge::AlwaysTrue => {
                    for s in sample_set {
                        assert!(a.contains(s) || b.contains(s))
                    }
                }
                RangeMerge::AlwaysFalse => {
                    for s in sample_set {
                        assert!(!(a.contains(s) || b.contains(s)))
                    }
                }
            }
        });
    }
}
