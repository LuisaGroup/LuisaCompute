use std::alloc::Layout;

use luisa_compute_cpu_kernel_defs as defs;
use luisa_compute_ir::{ir::Type, Gc};

#[repr(C)]
pub struct BufferImpl {
    pub data: *mut u8,
    pub size: usize,
    pub align: usize,
    pub ty: Option<Gc<Type>>,
}
#[repr(C)]
pub struct BindlessArrayImpl {
    pub buffers: Vec<defs::BufferView>,
}

impl BufferImpl {
    pub(super) fn new(size: usize, align: usize) -> Self {
        let layout = Layout::from_size_align(size, align).unwrap();
        let data = unsafe { std::alloc::alloc_zeroed(layout) };
        Self {
            data,
            size,
            align,
            ty: None,
        }
    }
}

impl Drop for BufferImpl {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, self.align).unwrap();
        unsafe { std::alloc::dealloc(self.data, layout) };
    }
}
