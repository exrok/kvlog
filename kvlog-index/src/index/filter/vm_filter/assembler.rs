use super::*;
use crate::index::Field;
use memchr::memmem::Finder;
use std::marker::PhantomData;

pub struct PC(u32);

type Label = u16;

#[derive(Default)]
pub struct Assembler<'a> {
    key_map: Vec<KeyID>,
    blocks: Vec<Block>,
    label: Vec<LabelTarget>,
    label_use_offsets: Vec<(u16, u16)>,
    pub code: Vec<DataRepr<'a>>,
}

impl<'a> Assembler<'a> {
    pub(crate) fn pre_resolve_block(&mut self, bid: BlockIdx) -> u16 {
        let block = &self.blocks[bid.0 as usize];
        if let Some(label) = block.current_label.get() {
            return label;
        }
        let terminal = block.terminal;
        drop(block);
        let label_default = match terminal {
            Ret::True => LabelTarget::True,
            Ret::False => LabelTarget::False,
            Ret::PC(pc) => LabelTarget::PC(pc),
            Ret::Label(label) => LabelTarget::Label(label),
            Ret::Next(block_idx) => LabelTarget::Label(self.pre_resolve_block(block_idx)),
        };
        let label_index = self.label.len() as u16;
        self.label.push(label_default);
        self.blocks[bid.0 as usize].current_label.set(Some(label_index));
        label_index
    }
    pub(crate) fn pre_resolve(&mut self, ret: Ret, offset: u16) -> u16 {
        match ret {
            Ret::True => TRUE,
            Ret::False => FALSE,
            Ret::PC(value) => value,
            Ret::Label(label) => {
                self.label_use_offsets.push((offset, label));
                FALSE
            }
            Ret::Next(block) => {
                let label = self.pre_resolve_block(block);
                self.label_use_offsets.push((offset, label));
                FALSE
            }
        }
    }
    pub fn block(&mut self, parent: Option<BlockIdx>, terminal: Ret) -> BlockIdx {
        let idx = self.blocks.len();
        self.blocks.push(Block { parent, current_label: Cell::new(None), empty: true, terminal });
        BlockIdx(idx as u16)
    }
    pub(crate) fn key_intern(&mut self, key: KeyID) -> u8 {
        if let Some(index) = self.key_map.iter().position(|k| *k == key) {
            index as u8
        } else {
            let index = self.key_map.len() as u8;
            self.key_map.push(key);
            index
        }
    }

    pub(crate) fn label(&mut self) -> Ret {
        let index = self.label.len() as u16;
        self.label.push(LabelTarget::Undefined);
        Ret::Label(index)
    }

    fn next_inst_pc(&self) -> PC {
        PC(self.code.len() as u32)
    }

    fn update_block_labels(&mut self, mut bb: BlockIdx) {
        loop {
            let block = &mut self.blocks[bb.0 as usize];
            if let Some(label) = block.current_label.take() {
                self.label[label as usize] = LabelTarget::PC(self.code.len() as u16);
            } else {
            }
            if block.empty {
                block.empty = false;
                if let Some(parent) = block.parent {
                    bb = parent;
                    continue;
                }
            }
            break;
        }
    }
    pub(crate) fn push_inst(&mut self, bb: BlockIdx, op: Op, alt: u8, meta: u16, fail: Ret, success: Ret) -> PC {
        println!("{:?} fail: {:?} succ: {:?}", op, fail, success);
        self.update_block_labels(bb);
        let pc = self.next_inst_pc();

        let inst = Inst {
            op,
            alt,
            res: [self.pre_resolve(fail, ((pc.0 * 4) + 1) as u16), self.pre_resolve(success, ((pc.0 * 4) + 2) as u16)],
            meta,
            phantom: PhantomData,
        };
        self.code.push(DataRepr { inst });
        pc
    }

    pub(crate) fn field_type_in_set(&mut self, bb: BlockIdx, fail: Ret, success: Ret, key: KeyID, set: u16) -> PC {
        let key = self.key_intern(key);
        let pc = self.push_inst(bb, Op::FieldType, key, set as u16, fail, success);
        pc
    }
    pub(crate) fn any_flag(&mut self, bb: BlockIdx, fail: Ret, success: Ret, flags: u16) -> PC {
        let pc = self.push_inst(bb, Op::AnyFlags, 0, flags as u16, fail, success);
        pc
    }

    pub(crate) fn msg_eq(&mut self, bb: BlockIdx, fail: Ret, success: Ret, text: &'a [u8]) -> PC {
        let pc = self.push_inst(bb, Op::TexturalMetaField, TextualMatch::Eq as u8, text.len() as u16, fail, success);
        self.code.push(DataRepr { ptr: text.as_ptr() });
        pc
    }

    pub(crate) fn msg_ptr_eq(&mut self, bb: BlockIdx, fail: Ret, success: Ret, text: &'a [u8]) -> PC {
        let pc =
            self.push_inst(bb, Op::TexturalMetaField, TextualMatch::PointerEq as u8, text.len() as u16, fail, success);
        self.code.push(DataRepr { ptr: text.as_ptr() });
        pc
    }

    pub(crate) fn msg_any(&mut self, bb: BlockIdx, fail: Ret, success: Ret, needles: &'a [&'a str]) -> PC {
        let pc =
            self.push_inst(bb, Op::TexturalMetaField, TextualMatch::Any as u8, needles.len() as u16, fail, success);
        self.code.push(DataRepr { ptr: needles.as_ptr() as *const &str as *const u8 });
        pc
    }

    pub(crate) fn msg_contains(&mut self, bb: BlockIdx, fail: Ret, success: Ret, text: &'a [u8], bump: &'a Bump) -> PC {
        let pc =
            self.push_inst(bb, Op::TexturalMetaField, TextualMatch::Contains as u8, text.len() as u16, fail, success);
        self.code.push(DataRepr { ptr: bump.alloc(memchr::memmem::Finder::new(text)) as *mut _ as *const u8 });
        pc
    }

    pub(crate) fn msg_starts_with(&mut self, bb: BlockIdx, fail: Ret, success: Ret, text: &'a [u8]) -> PC {
        let pc =
            self.push_inst(bb, Op::TexturalMetaField, TextualMatch::StartsWith as u8, text.len() as u16, fail, success);
        self.code.push(DataRepr { ptr: text.as_ptr() });
        pc
    }
    pub(crate) fn msg_ends_with(&mut self, bb: BlockIdx, fail: Ret, success: Ret, text: &'a [u8]) -> PC {
        let pc =
            self.push_inst(bb, Op::TexturalMetaField, TextualMatch::EndsWith as u8, text.len() as u16, fail, success);
        self.code.push(DataRepr { ptr: text.as_ptr() });
        pc
    }
    pub(crate) fn field_contains_text(
        &mut self,
        bb: BlockIdx,
        fail: Ret,
        success: Ret,
        key: KeyID,
        text: &'a [u8],
        bump: &'a Bump,
    ) -> PC {
        let key = self.key_intern(key);
        let pc = self.push_inst(bb, Op::TextContains, key, text.len() as u16, fail, success);
        self.code.push(DataRepr { ptr: bump.alloc(memchr::memmem::Finder::new(text)) as *mut _ as *const u8 });
        pc
    }
    pub(crate) fn field_exists(&mut self, bb: BlockIdx, fail: Ret, success: Ret, key: KeyID) -> PC {
        let key = self.key_intern(key);
        let pc = self.push_inst(bb, Op::FieldExists, key, 0, fail, success);
        pc
    }
    pub(crate) fn field_ends_with(&mut self, bb: BlockIdx, fail: Ret, success: Ret, key: KeyID, text: &'a [u8]) -> PC {
        let key = self.key_intern(key);
        let pc = self.push_inst(bb, Op::TextEndsWith, key, text.len() as u16, fail, success);
        self.code.push(DataRepr { ptr: text.as_ptr() });
        pc
    }

    pub(crate) fn field_starts_with(
        &mut self,
        bb: BlockIdx,
        fail: Ret,
        success: Ret,
        key: KeyID,
        text: &'a [u8],
    ) -> PC {
        let key = self.key_intern(key);
        let pc = self.push_inst(bb, Op::TextStartsWith, key, text.len() as u16, fail, success);
        self.code.push(DataRepr { ptr: text.as_ptr() });
        pc
    }

    pub(crate) fn span_duration_range(
        &mut self,
        bb: BlockIdx,
        fail: Ret,
        success: Ret,
        min_ns: u64,
        max_ns: u64,
    ) -> PC {
        let pc = self.next_inst_pc();
        self.push_inst(bb, Op::SpanDurationRange, 0, 0, fail, success);
        self.code.push(DataRepr { raw: min_ns });
        self.code.push(DataRepr { raw: max_ns });
        pc
    }
    pub(crate) fn timestamp_range(&mut self, bb: BlockIdx, fail: Ret, success: Ret, min_ns: u64, max_ns: u64) -> PC {
        let pc = self.next_inst_pc();
        self.push_inst(bb, Op::TimestampRange, 0, 0, fail, success);
        self.code.push(DataRepr { raw: min_ns });
        self.code.push(DataRepr { raw: max_ns });
        pc
    }
    pub(crate) fn field_float_range(
        &mut self,
        bb: BlockIdx,
        fail: Ret,
        success: Ret,
        key: KeyID,
        min: f64,
        max: f64,
    ) -> PC {
        let key = self.key_intern(key);
        let pc = self.next_inst_pc();
        self.push_inst(bb, Op::FieldFloatRange, key, 0, fail, success);
        self.code.push(DataRepr { float: min });
        self.code.push(DataRepr { float: max });
        pc
    }
    pub(crate) fn field_raw_range(
        &mut self,
        bb: BlockIdx,
        fail: Ret,
        success: Ret,
        key: KeyID,
        min: Field,
        max: Field,
    ) -> PC {
        let key = self.key_intern(key);
        let pc = self.next_inst_pc();
        self.push_inst(bb, Op::RawRange, key, 0, fail, success);
        self.code.push(DataRepr { raw: min.raw });
        self.code.push(DataRepr { raw: max.raw });
        pc
    }
    pub(crate) fn field_eq_or(&mut self, bb: BlockIdx, fail: Ret, success: Ret, key: KeyID, fields: &[Field]) -> PC {
        let key = self.key_intern(key);
        let pc = self.next_inst_pc();
        match fields {
            [a] => {
                self.push_inst(bb, Op::Eq, key, 0, fail, success);
                self.code.push(DataRepr { raw: a.raw });
            }
            [a, b] => {
                self.push_inst(bb, Op::EqOr, key, 0, fail, success);
                self.code.push(DataRepr { raw: a.raw });
                self.code.push(DataRepr { raw: b.raw });
            }
            many => {
                self.push_inst(bb, Op::EqOrMany, key, fields.len() as u16, fail, success);
                self.code.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(many.as_ptr() as *const DataRepr, many.len())
                });
            }
        }
        pc
    }

    pub(crate) fn field_eq(&mut self, bb: BlockIdx, fail: Ret, success: Ret, key: KeyID, a: Field) -> PC {
        let key = self.key_intern(key);
        let pc = self.push_inst(bb, Op::Eq, key, 0, fail, success);
        self.code.push(DataRepr { raw: a.raw });
        pc
    }

    pub(crate) fn build(mut self) -> Result<QueryVm<'a>, VmCompileError> {
        for (offset, label) in &self.label_use_offsets {
            let ptr = self.code.as_mut_ptr().cast::<u16>();
        }
        {
            let ptr = self.code.as_mut_ptr().cast::<u16>();
            for (offset, mut label_index) in self.label_use_offsets.iter().rev().copied() {
                let ret = loop {
                    break match self.label[label_index as usize] {
                        LabelTarget::Undefined => panic!("Label was left undefined"),
                        LabelTarget::True => TRUE,
                        LabelTarget::False => FALSE,
                        LabelTarget::PC(value) => value,
                        LabelTarget::Label(label) => {
                            label_index = label;
                            continue;
                        }
                    };
                };
                unsafe {
                    ptr.add(offset as usize).write(ret);
                }
            }
        }

        let key_cache = KeyMapCache::new(&self.key_map);
        if self.code.len() == 0 {
            return Err(VmCompileError::EmptyCode);
        }
        Ok(QueryVm { key_cache, code: self.code.into() })
    }
}
#[cfg(test)]
mod test {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TestNode {
        name: [u8; 6],
        ret: u16,
    }
    impl TestNode {
        fn new(name: &str, ret: u16) -> TestNode {
            if name.len() > 6 {
                panic!("Name too long: {}", name);
            }
            let mut name_bytes = [0; 6];
            name_bytes[..name.len()].copy_from_slice(name.as_bytes());
            TestNode { name: name_bytes, ret }
        }
        fn name(&self) -> &str {
            unsafe { std::str::from_utf8_unchecked(std::ffi::CStr::from_ptr(self.name.as_ptr().cast()).to_bytes()) }
        }
    }
}
