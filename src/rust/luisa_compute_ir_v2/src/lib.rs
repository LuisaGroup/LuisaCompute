#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
pub mod binding;
use std::sync::Once;

pub use binding::*;
use libloading::Library;
pub type NodeRef = *const binding::Node;

pub mod convert;
#[allow(dead_code)]
struct Lib {
    binding: IrV2BindingTable,
    lib: Library,
}
static mut LIB: Option<Lib> = None;
static LIB_INIT: Once = Once::new();
pub fn init<P: AsRef<std::path::Path>>(path: P) {
    assert!(!LIB_INIT.is_completed());
    LIB_INIT.call_once(|| unsafe {
        let lib = Library::new(path.as_ref()).unwrap();
        let load_binding = lib
            .get::<extern "C" fn() -> IrV2BindingTable>(b"lc_ir_v2_binding_table\0")
            .unwrap();
        let binding = (load_binding)();
        LIB = Some(Lib { binding, lib });
    });
}
pub fn binding_table() -> &'static IrV2BindingTable {
    unsafe { &LIB.as_ref().unwrap().binding }
}
#[macro_export]
macro_rules! call {
    ($name:ident, $($args:expr),*) => {
        unsafe {
            (binding_table().$name.unwrap())($($args),*)
        }
    };
}
impl std::fmt::Debug for Slice<::std::os::raw::c_char> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = unsafe {
            let slice = std::slice::from_raw_parts(self.data as *const u8, self.len);
            std::ffi::CStr::from_bytes_with_nul(slice).unwrap()
        };
        write!(f, "{:?}", s)
    }
}

impl<T> From<Slice<T>> for &[T] {
    fn from(s: Slice<T>) -> Self {
        unsafe { std::slice::from_raw_parts(s.data, s.len) }
    }
}
#[allow(dead_code)]
pub struct BasicBlockIter {
    bb: *const BasicBlock,
    first: NodeRef,
    last: NodeRef,
    current: NodeRef,
}
impl BasicBlockIter {
    pub fn new(bb: *const BasicBlock) -> Self {
        let first = call!(basic_block_first, bb);
        let last = call!(basic_block_last, bb);
        Self {
            bb,
            first,
            last,
            current: first,
        }
    }
}
impl Iterator for BasicBlockIter {
    type Item = NodeRef;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.last {
            None
        } else {
            let ret = self.current;
            self.current = call!(node_next, self.current);
            Some(ret)
        }
    }
}
pub fn basic_block_iter(bb: *const BasicBlock) -> BasicBlockIter {
    BasicBlockIter::new(bb)
}
pub fn basic_block_foreach<F: FnMut(NodeRef)>(bb: *const BasicBlock, mut f: F) {
    let mut iter = BasicBlockIter::new(bb);
    while let Some(node) = iter.next() {
        f(node);
    }
}
pub fn basic_block_phis(bb: *const BasicBlock) -> Vec<NodeRef> {
    BasicBlockIter::new(bb)
        .filter(|node| {
            let inst = call!(node_inst, *node);
            let tag = call!(Instruction_tag, inst);
            tag == InstructionTag::PHI
        })
        .collect::<Vec<_>>()
}
