use std::{cell::Cell, marker::PhantomData};

use bumpalo::Bump;
use compiler::compile;
use jsony::Jsony;
use kvlog::{encoding::Value, LogLevel};
use memchr::memmem::Finder;
mod assembler;
mod compiler;
use assembler::Assembler;

use crate::{
    index::{
        self,
        i48::{self, try_from_i64},
        Bucket, BucketGuard, Field, FieldKind,
    },
    query::{
        self, parse_query,
        query_parts::{self, FieldTest, Pred},
        QueryParseError,
    },
    shared_interner::Mapper,
};

use super::{KeyID, KeyMapCache, LogEntry};

const TRUE: u16 = u16::MAX;
const FALSE: u16 = u16::MAX ^ 1;

#[derive(Copy, Clone, Debug)]
pub enum PredKind {
    HasParentSpan,
    SpanIs,
    ParentSpanIs,
}

#[derive(Copy, Clone, Debug)]
pub enum FieldTestKind {
    TextAny,
    TextEqual,
    I64Eq,
    BytesEqual,
    IsTrue,
    NullValue,
    FiniteFloat,
    U64Eq,
    DurationRange,
    TimeRange,
}

#[derive(Copy, Clone, Debug)]
pub enum VmCompileError {
    UnsupportedPredicate(PredKind),
    UnsupportedFieldTest(FieldTestKind),
    UnsupportedMessageTest,
    EmptyCode,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub enum TextualMatch {
    Eq,
    PointerEq,
    Any,
    Contains,
    StartsWith,
    EndsWith,
    // probably length range
    Reserved1,
    // probably regex
    Reserved2,
}

impl From<u8> for TextualMatch {
    fn from(value: u8) -> TextualMatch {
        unsafe { std::mem::transmute::<u8, TextualMatch>(value) }
    }
}

#[derive(Debug, Clone, Copy, Jsony, PartialEq, PartialOrd, Eq)]
#[repr(u8)]
#[jsony(ToStr)]
enum Op {
    Eq,
    EqOr,
    EqOrMany,
    RawRange,
    FieldExists,
    TextStartsWith,
    TextContains,
    TextEndsWith,
    InServiceSet,
    InTargetSet,
    AnyFlags,
    FieldFloatRange,
    TexturalMetaField,
    FieldType,
    TimestampRange,
    SpanDurationRange,
}

// These matches really make it to VM stage and inlining them slows down more common queries
#[inline(never)]
unsafe fn meta_field_match(inst: Inst, ip: *const Inst, log: LogEntry) -> bool {
    let data = log.message();
    match TextualMatch::from(inst.alt) {
        TextualMatch::Eq => {
            let needle = unsafe { std::slice::from_raw_parts(ptr_at(ip, 0), inst.meta as usize) };
            needle == data
        }
        TextualMatch::PointerEq => ptr_at(ip, 0) == data.as_ptr(),
        TextualMatch::Any => {
            let needles =
                unsafe { std::slice::from_raw_parts::<&str>(ptr_at(ip, 0) as *const &str, inst.meta as usize) };
            for needle in needles {
                if needle.as_bytes() == data {
                    return true;
                }
            }
            return false;
        }
        TextualMatch::Contains => {
            // todo figure out best finder
            let finder = &*(ptr_at(ip, 0) as *const Finder);
            finder.find(data).is_some()
        }
        TextualMatch::StartsWith => {
            let needle = unsafe { std::slice::from_raw_parts(ptr_at(ip, 0), inst.meta as usize) };
            return memchr::arch::all::is_prefix(data, needle);
        }
        TextualMatch::EndsWith => {
            let needle = unsafe { std::slice::from_raw_parts(ptr_at(ip, 0), inst.meta as usize) };
            return memchr::arch::all::is_suffix(data, needle);
        }
        TextualMatch::Reserved1 => false,
        TextualMatch::Reserved2 => false,
    }
}

enum Data<'b> {
    Field(Field),
    Bytes(&'b [u8]),
}

#[derive(Clone, Copy)]
union DataRepr<'a> {
    raw: u64,
    float: f64,
    ptr: *const u8,
    inst: Inst<'a>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Inst<'a> {
    op: Op,
    alt: u8,
    res: [u16; 2],
    meta: u16,
    phantom: PhantomData<&'a ()>,
}
unsafe fn float_at<'a>(ptr: *const Inst<'a>, index: usize) -> f64 {
    (*unsafe { ptr.add(index + 1) as *const DataRepr }).float
}

unsafe fn field_at<'a>(ptr: *const Inst<'a>, index: usize) -> Field {
    *unsafe { ptr.add(index + 1) as *const Field }
}

unsafe fn ptr_at<'a>(ptr: *const Inst<'a>, index: usize) -> *const u8 {
    *unsafe { ptr.add(index + 1) as *const *const u8 }
}

unsafe fn repr_at<'a>(ptr: *const Inst<'a>, index: usize) -> &DataRepr<'a> {
    &*unsafe { ptr.add(index + 1) as *const DataRepr }
}
// impl<'a> std::fmt::Debug for VmFilter<'a> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         for (i, inst) in self.code.iter().enumerate() {
//             write!(f, "{i:2}: {inst} ")?;
//             match inst.res[0] {
//                 TRUE => f.write_str(" TRUE, ")?,
//                 FALSE => f.write_str("FALSE, ")?,
//                 next => write!(f, "{:5}, ", next)?,
//             }
//             match inst.res[1] {
//                 TRUE => f.write_str(" TRUE\n")?,
//                 FALSE => f.write_str("FALSE\n")?,
//                 next => write!(f, "{:5}\n", next)?,
//             }
//         }
//         Ok(())
//     }
// }
impl<'a> std::fmt::Debug for Inst<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = self.op.to_str();
        //todo some instructions might not have data be a field
        write!(f, "{op:20}: {} {}", self.alt, self.meta)
    }
}

impl<'a> Inst<'a> {
    fn new(op: Op, if_false: u16, if_true: u16) -> Inst<'a> {
        Inst { op, alt: 0, res: [if_false, if_true], meta: 0, phantom: PhantomData }
    }
}

impl<'a> Inst<'a> {
    #[unsafe(no_mangle)]
    fn negated_terminal(&self) -> bool {
        (self.res[0] & 0b1) == 1
    }
}

#[derive(Clone)]
pub struct QueryVm<'a> {
    key_cache: Box<KeyMapCache>,
    // saftey fields is also non empty
    code: Box<[DataRepr<'a>]>,
}

pub struct VmFilter {
    pub(super) bump: Bump,
    pub(super) filter: QueryVm<'static>,
}
impl VmFilter {
    #[inline]
    pub fn query<'a>(&'a mut self) -> &'a mut QueryVm<'a> {
        unsafe { std::mem::transmute::<&mut QueryVm<'static>, &'a mut QueryVm<'a>>(&mut self.filter) }
    }
}

const INV_KEY_MASK: u64 = (0xFFFF_FFFF_FFFF_FFFF >> 12);
const INV_KEY_TYPE_MASK: u64 = 0xFFFF_FFFF_FFFF_FFFF >> 16;

// fn eval(code: &[Inst], entry: LogEntry) -> bool {
//     let mut inst = &code[0];
//     println!("=== INPUT: {:?}", entry.fields());
//     let mut vm = VmState { reg: 0, entry, fields: entry.raw_fields().iter() };
//     loop {
//         let resi: bool = eval_inst(&mut vm, inst);
//         print_entry(inst, &vm, code, resi);
//         let inst_index = inst.res[resi as usize];
//         if let Some(new_inst) = code.get(inst_index as usize) {
//             inst = new_inst;
//             continue;
//         } else {
//             let result = (inst_index & 0b1) == 1;
//             println!("> OUTPUT: {result}\n");
//             return result;
//         }
//     }
// }

pub type KeyTable<'a> = &'a [u8; 8];
#[derive(Clone, Copy, Debug)]
enum LabelTarget {
    Undefined,
    True,
    False,
    PC(u16),
    Label(u16),
}

#[derive(Debug)]
pub enum BasicBlockPatch {
    None,
    Offset(u32),
    Label(u16),
}

#[derive(Clone, Copy, Debug)]
pub struct BlockIdx(u16);

#[derive(Debug)]
struct Block {
    pub terminal: Ret,
    empty: bool,
    current_label: Cell<Option<u16>>,
    pub parent: Option<BlockIdx>,
}

#[derive(Debug, Clone, Copy)]
enum Ret {
    True,
    False,
    PC(u16),
    Label(u16),
    Next(BlockIdx),
}

#[derive(Clone, Copy)]
#[repr(align(4))]
struct Branch {
    fail: u16,
    succ: u16,
}
impl Branch {
    #[no_mangle]
    fn cardinalize(self, requires: bool) -> Branch {
        match requires {
            true => self,
            false => Branch { fail: self.succ, succ: self.fail },
        }
    }
}

#[derive(Debug)]
pub enum QueryBuilderError {
    ParseError(QueryParseError),
    AlwaysTrue,
    AlwaysFalse,
    CompileError(VmCompileError),
}

impl<'a> QueryVm<'a> {
    pub fn reachable_instruction_count(&self) -> usize {
        let mut seen = vec![false; self.code.len() + 1];
        seen[0] = true;
        let mut queue = vec![0];
        let start = self.code.as_ptr() as *const Inst;
        while !queue.is_empty() {
            // find smalled queue value index
            let mut min_index = usize::MAX;
            let mut min_value = u16::MAX;
            for (i, value) in queue.iter().enumerate() {
                if *value < min_value {
                    min_value = *value;
                    min_index = i;
                }
            }
            let pop = queue.swap_remove(min_index);
            let inst = unsafe { &*start.add(pop as usize) };
            if inst.res[0] < FALSE {
                if !seen[inst.res[0] as usize] {
                    seen[inst.res[0] as usize] = true;
                    queue.push(inst.res[0]);
                }
            }
            if inst.res[1] < FALSE {
                if !seen[inst.res[1] as usize] {
                    seen[inst.res[1] as usize] = true;
                    queue.push(inst.res[1]);
                }
            }
        }
        seen.iter().filter(|&&x| x).count()
    }
    pub fn from_query(
        bump: &'a Bump,
        text: &str,
        bucket: &BucketGuard,
        targets: Mapper,
    ) -> Result<QueryVm<'a>, QueryBuilderError> {
        let preds = parse_query(bump, text).map_err(QueryBuilderError::ParseError)?;
        let output = preds.build_with_opt(bump, bucket, &bucket.maps());
        match output {
            query_parts::PredBuildResult::AlwaysTrue => Err(QueryBuilderError::AlwaysTrue),
            query_parts::PredBuildResult::AlwaysFalse => Err(QueryBuilderError::AlwaysFalse),
            query_parts::PredBuildResult::Ok(pred) => match compile(bump, pred, &bucket, targets) {
                Ok(vm) => Ok(vm),
                Err(err) => Err(QueryBuilderError::CompileError(err)),
            },
        }
    }

    pub fn compile<'y, 'k>(
        bump: &'k Bump,
        pred: Pred<'k>,
        bucket: &BucketGuard<'y>,
        targets: Mapper,
    ) -> Result<QueryVm<'k>, VmCompileError> {
        compiler::compile(bump, pred, bucket, targets)
    }

    #[inline]
    pub fn matches(&mut self, entry: LogEntry) -> bool {
        let lut = self.key_cache.lookup(entry);
        // println!("{:?}", lut);
        let mut inst = self.code.as_ptr() as *const Inst;
        // let mut vm = VmState { reg: 0, entry, fields: entry.raw_fields().iter() };

        loop {
            let resi: bool = unsafe { eval_inst_2(inst, entry, lut) };
            let inst_index = unsafe { (*inst).res[resi as usize] };
            if inst_index as usize >= self.code.len() {
                return (inst_index & 0b1) == 1;
            }
            inst = unsafe { self.code.as_ptr().add(inst_index as usize) as *const Inst };
            // if let Some(new_inst) = self.code.get(inst_index as usize) {
            //     inst = new_inst;
            //     continue;
            // } else {
            //     return (inst_index & 0b1) == 1;
            // }
        }
    }
    pub fn print(&self) {
        unsafe { print_instructions(&self.code, &self.key_cache.query) }
    }
}

unsafe fn print_inst(index: usize, ptr: *const Inst, keys: &[u16]) {
    let inst = unsafe { &*ptr };
    print!("{index:3}: {:16} -> [ ", inst.op.to_str());
    let [fail, succ] = inst.res;
    if fail == TRUE {
        print!("Yes | ")
    } else if fail == FALSE {
        print!(" No | ")
    } else {
        print!("{:3} | ", fail)
    }
    if succ == TRUE {
        print!("Yes ] ")
    } else if succ == FALSE {
        print!(" No ] ")
    } else {
        print!("{:3} ] ", succ)
    }
    match inst.op {
        Op::EqOr => {
            print!("\n  ");
            unsafe { print_field(field_at(ptr, 0)) }
            print!("  ");
            unsafe { print_field(field_at(ptr, 1)) }
        }
        Op::EqOrMany => {
            println!();
            for i in 0..inst.meta {
                print!("  ");
                unsafe { print_field(field_at(ptr, i as usize)) }
            }
        }
        Op::Eq => unsafe { print_field(field_at(ptr, 0)) },
        Op::AnyFlags => {
            println!("flags 0b{:05b}", inst.meta);
        }
        Op::TimestampRange => {
            println!("TimestampRange 0b{:05b}", inst.meta);
        }
        Op::SpanDurationRange => {
            println!("SpanDurationRange 0b{:05b}", inst.meta);
        }
        Op::TextStartsWith => {
            let key = keys[inst.alt as usize];
            let needle = std::slice::from_raw_parts(ptr_at(ptr, 0), inst.meta as usize);
            println!("{}.starts_with(\"{}\")", KeyID(key).as_str(), needle.escape_ascii());
        }
        Op::TextEndsWith => {
            let key = keys[inst.alt as usize];
            let needle = std::slice::from_raw_parts(ptr_at(ptr, 0), inst.meta as usize);
            println!("{}.ends_with(\"{}\")", KeyID(key).as_str(), needle.escape_ascii());
        }
        Op::TextContains => {
            let key = keys[inst.alt as usize];
            let finder = &*(ptr_at(ptr, 0) as *const Finder);
            println!("{}.contains(\"{}\")", KeyID(key).as_str(), finder.needle().escape_ascii());
            // let needle = std::slice::from_raw_parts(ptr_at(ptr, 0), inst.meta as usize);
            // println!("{:?}", needle.es);
        }
        Op::InServiceSet => {
            // let needle = std::slice::from_raw_parts(ptr_at(ptr, 0), inst.meta as usize);
            print!("$service in [");
            let set = &*(ptr.add(1) as *const [u64; 4]);
            let mut first = true;
            for id in 0..=255 {
                if service_set_contains(id, set) {
                    if first {
                        first = false;
                    } else {
                        print!(", ");
                    }
                    print!("{}", id);
                }
            }
            println!("]");
        }
        Op::FieldType => {
            println!("Fieldtype 0b{:b}", inst.alt);
            // let needle = std::slice::from_raw_parts(ptr_at(ptr, 0), inst.meta as usize);
            // println!("{:?}", needle.es);
        }
        Op::TexturalMetaField => {
            println!("{:?}", TextualMatch::from(inst.alt));
            // let needle = std::slice::from_raw_parts(ptr_at(ptr, 0), inst.meta as usize);
            // println!("{:?}", needle.es);
        }
        Op::FieldExists => {
            let key = keys[inst.alt as usize];
            println!("{}.exists()", KeyID(key).as_str());
            // let needle = std::slice::from_raw_parts(ptr_at(ptr, 0), inst.meta as usize);
            // println!("{:?}", needle.es);
        }
        Op::InTargetSet => {
            println!("target_in_set");
            // let needle = std::slice::from_raw_parts(ptr_at(ptr, 0), inst.meta as usize);
            // println!("{:?}", needle.es);
        }
        Op::FieldFloatRange => {
            println!("FieldFloatRange");
        }
        Op::RawRange => {
            println!("RawRange");
        }
    }
}
unsafe fn print_instructions(code: &[DataRepr], keys: &[u16]) {
    let mut seen = vec![false; code.len() + 1];
    seen[0] = true;
    let mut queue = vec![0];
    let start = code.as_ptr() as *const Inst;
    while !queue.is_empty() {
        // find smalled queue value index
        let mut min_index = usize::MAX;
        let mut min_value = u16::MAX;
        for (i, value) in queue.iter().enumerate() {
            if *value < min_value {
                min_value = *value;
                min_index = i;
            }
        }
        let pop = queue.swap_remove(min_index);
        print_inst(pop as usize, start.add(pop as usize), keys);
        let inst = *start.add(pop as usize);
        if inst.res[0] < FALSE {
            if !seen[inst.res[0] as usize] {
                seen[inst.res[0] as usize] = true;
                queue.push(inst.res[0]);
            }
        }
        if inst.res[1] < FALSE {
            if !seen[inst.res[1] as usize] {
                seen[inst.res[1] as usize] = true;
                queue.push(inst.res[1]);
            }
        }
    }
}

fn print_field(field: Field) {
    let key = KeyID::try_raw_to_str(field.raw_key()).unwrap_or("UNKNOWN");
    println!("{:>8}: {:10} 0x{:012}", key, field.kind().to_str(), field.value_mask());
}

unsafe fn eval_inst_2(ip: *const Inst, log: LogEntry, table: KeyTable) -> bool {
    let inst = unsafe { &*ip };
    match inst.op {
        Op::TimestampRange => {
            let ts = log.timestamp();
            return unsafe { (ts >= field_at(ip, 0).raw) & (ts <= field_at(ip, 1).raw) };
        }
        Op::SpanDurationRange => {
            let Some(ns) = log.span_ns_duration() else { return false };
            return unsafe { (ns >= field_at(ip, 0).raw) & (ns <= field_at(ip, 1).raw) };
        }
        Op::AnyFlags => unsafe { return log.archetype().mask & (inst.meta as u32) != 0 },
        Op::FieldType => {
            let index = *table.get_unchecked(inst.alt as usize);
            if index == u8::MAX {
                return false;
            }
            let field = *log.get_field_unchecked(index as usize);
            return inst.meta & (1u16 << (field.kind() as u8)) != 0;
        }
        Op::Eq => {
            let index = *table.get_unchecked(inst.alt as usize);
            if index == u8::MAX {
                return false;
            }
            let field = *log.get_field_unchecked(index as usize);
            let res = unsafe { field_at(ip, 0) == field };
            return res;
        }
        Op::RawRange => {
            let index = *table.get_unchecked(inst.alt as usize);
            if index == u8::MAX {
                return false;
            }
            let field = *log.get_field_unchecked(index as usize);
            let res = unsafe { (field.raw >= field_at(ip, 0).raw) & (field.raw <= field_at(ip, 1).raw) };
            return res;
        }
        Op::FieldFloatRange => {
            let index = *table.get_unchecked(inst.alt as usize);
            if index == u8::MAX {
                return false;
            }
            let field = *log.get_field_unchecked(index as usize);
            if let Some(value) = unsafe { field.as_f64(log.bucket) } {
                return unsafe { (value >= float_at(ip, 0)) & (value <= float_at(ip, 1)) };
            } else {
                return false;
            }
        }
        Op::EqOr => {
            let index = *table.get_unchecked(inst.alt as usize);
            if index == u8::MAX {
                return false;
            }
            let field = *log.get_field_unchecked(index as usize);
            // unsafe {
            //     print_field(field);
            // }
            // unsafe { print_field(field_at(ip, 0)) }
            // unsafe { print_field(field_at(ip, 1)) }
            let res = unsafe { (field_at(ip, 0) == field) | (field_at(ip, 1) == field) };
            // println!("{}", res);
            return res;
        }
        Op::EqOrMany => {
            let index = *table.get_unchecked(inst.alt as usize);
            if index == u8::MAX {
                return false;
            }
            let field = *log.get_field_unchecked(index as usize);
            for index in 0..inst.meta {
                if unsafe { field_at(ip, index as usize) } == field {
                    return true;
                }
            }
            return false;
        }
        Op::FieldExists => {
            let index = *table.get_unchecked(inst.alt as usize);
            return index != u8::MAX;
        }
        Op::TextStartsWith => {
            let index = *table.get_unchecked(inst.alt as usize);
            if index == u8::MAX {
                return false;
            }
            let field = *log.get_field_unchecked(index as usize);
            let needle = std::slice::from_raw_parts(ptr_at(ip, 0), inst.meta as usize);
            if let Some(data) = field.as_bytes(log.bucket) {
                // let res = data.starts_with(needle);
                return memchr::arch::all::is_prefix(data, needle);
            } else {
                return false;
            }
        }
        Op::TextContains => {
            let index = *table.get_unchecked(inst.alt as usize);
            if index == u8::MAX {
                return false;
            }
            let field = *log.get_field_unchecked(index as usize);
            // let needle = std::slice::from_raw_parts(ptr_at(ip, 0), inst.meta as usize);

            if let Some(data) = field.as_bytes(log.bucket) {
                let finder = &*(ptr_at(ip, 0) as *const Finder);
                // let data = unsafe { std::str::from_utf8_unchecked(data) };
                // let needle = unsafe { std::str::from_utf8_unchecked(needle) };
                return finder.find(data).is_some();
            } else {
                return false;
            }
        }
        Op::TextEndsWith => {
            let index = *table.get_unchecked(inst.alt as usize);
            if index == u8::MAX {
                return false;
            }
            let field = *log.get_field_unchecked(index as usize);
            let needle = std::slice::from_raw_parts(ptr_at(ip, 0), inst.meta as usize);
            if let Some(data) = field.as_bytes(log.bucket) {
                // let res = data.starts_with(needle);
                return memchr::arch::all::is_suffix(data, needle);
            } else {
                return false;
            }
        }
        Op::InServiceSet => {
            let service_id = log.archetype().raw_service();
            let set = &*(ip.add(1) as *const [u64; 4]);
            return service_set_contains(service_id, set);
        }
        Op::InTargetSet => {
            let target_id = log.archetype().target_id;
            if target_id > inst.meta {
                return false;
            }
            let target_id_mask = 1u64 << (target_id & 0b111111);
            let set = *(ptr_at(ip, 0) as *const u64).add((target_id >> 6) as usize);
            return set & target_id_mask != 0;
        }

        Op::TexturalMetaField => return meta_field_match(*inst, ip, log),
    }
}

// todo maybe switch the indice and shift
fn service_set_insert(index: u8, bitset: &mut [u64; 4]) {
    bitset[(index & 0b11) as usize] |= (1u64.wrapping_shl((index as u32) >> 2));
}
fn service_set_contains(index: u8, bitset: &[u64; 4]) -> bool {
    bitset[(index & 0b11) as usize] & (1u64.wrapping_shl((index as u32) >> 2)) != 0
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::index::i48::to_i64;
    use crate::index::test::{test_index, TestIndexWriter};
    use crate::index::Bucket;
    use crate::log;
    use kvlog::Encode;

    #[test]
    fn simple_expr() {
        let mut index = test_index();
        KeyID::intern("k1");
        KeyID::intern("k2");
        KeyID::intern("k3");
        let reader = index.reader().clone();
        let mut writer = TestIndexWriter::new(&mut index);
        let _ = log!(writer; msg="Initialize bucket");
        let mut bucket = reader.newest_bucket().unwrap();
        macro_rules! entry {
            ($($tt:tt)*) => {
                {
                    let w0 = log!(writer; $($tt)*);
                    bucket.renew();
                    bucket.upgrade(w0).unwrap()
                }
            };
        }

        let mut asm = Assembler::default();
        let key = KeyID::intern("k2");
        let bb = asm.block(None, Ret::False);
        asm.field_eq_or(
            bb,
            Ret::True,
            Ret::False,
            key,
            &[
                Field::new(key.raw(), FieldKind::I48, i48::from_i64(2)),
                Field::new(key.raw(), FieldKind::I48, i48::from_i64(0)),
            ],
        );
        let mut vm = asm.build().unwrap();

        assert!(vm.matches(entry!(k1 = 0, k2 = 1)));
        assert!(!vm.matches(entry!(k1 = 2, k2 = 0)));
        assert!(!vm.matches(entry!(k1 = 1, k2 = 2)));

        // assert_eq!(eval(&k1_or_k2, entry!(k1 = "hello", k2 = "nice")), true);
        // assert_eq!(eval(&k1_or_k2, entry!(k2 = true, k3 = "hello")), true);

        // !k1 && !k2
        //     #[rustfmt::skip]
        //     let not_k1_nor_k2 = [
        //         /*0*/ Inst::new(Op::LoadKeyMin, key_mask("k1"), 1, FALSE),
        //         /*1*/ Inst::new(Op::RegOrLoadKeyMin, key_mask("k2"), TRUE, FALSE),
        //     ];

        //     assert_eq!(eval(&not_k1_nor_k2, entry!(k1 = "hello", k2 = "nice")), false);
        //     assert_eq!(eval(&not_k1_nor_k2, entry!(k2 = true, k3 = "hello")), false);
        //     assert_eq!(eval(&not_k1_nor_k2, entry!(k3 = "hello")), true);

        //     // k2.starts_with("hi")
        //     #[rustfmt::skip]
        //     let k2_starts_with_hi = [
        //         /*1*/ Inst::new(Op::LoadKeyMin, key_mask("k2"), FALSE, 1),
        //         /*1*/ Inst::new_text(Op::RegTextStartsWith, b"hi", FALSE, TRUE),
        //     ];

        //     assert_eq!(eval(&k2_starts_with_hi, entry!(k1 = "hi", k2 = "bye")), false);
        //     assert_eq!(eval(&k2_starts_with_hi, entry!(k1 = "bye", k2 = "hi")), true);
        //     assert_eq!(eval(&k2_starts_with_hi, entry!(k3 = "hello")), false);
    }
}
