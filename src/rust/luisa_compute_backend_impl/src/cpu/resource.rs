use std::{
    alloc::Layout,
    sync::atomic::{AtomicU64},
};

use luisa_compute_api_types::{BindlessArrayUpdateModification, BindlessArrayUpdateOperation};
use luisa_compute_cpu_kernel_defs as defs;
use parking_lot::{Condvar, Mutex};

use super::texture::TextureImpl;

pub struct EventImpl {
    pub mutex: Mutex<u64>,
    pub device: AtomicU64,
    pub on_signal: Condvar,
}
impl EventImpl {
    pub fn new() -> Self {
        Self {
            mutex: Mutex::new(0),
            device: AtomicU64::new(0),
            on_signal: Condvar::new(),
        }
    }
    pub fn signal(&self, ticket: u64) {
        let _lk = self.mutex.lock();
        self.device.fetch_max(ticket, std::sync::atomic::Ordering::SeqCst);
        self.on_signal.notify_all();
    }
    pub fn wait(&self, ticket: u64) {
        let mut lk = self.mutex.lock();
        loop {
            if self.device.load(std::sync::atomic::Ordering::SeqCst) >= ticket {
                break;
            }
            self.on_signal.wait(&mut lk);
        }
    }
    pub fn synchronize(&self, ticket: u64) {
        self.wait(ticket);
    }
    pub fn is_completed(&self, ticket: u64) -> bool {
        return self.device.load(std::sync::atomic::Ordering::SeqCst) >= ticket;
    }
}
#[repr(C)]
pub struct BufferImpl {
    pub data: *mut u8,
    pub size: usize,
    pub align: usize,
    pub ty: u64,
}
#[repr(C)]
pub struct BindlessArrayImpl {
    pub buffers: Vec<defs::BufferView>,
    pub tex2ds: Vec<defs::Texture>,
    pub tex3ds: Vec<defs::Texture>,
}

impl BindlessArrayImpl {
    pub unsafe fn update(&mut self, modifications: &[BindlessArrayUpdateModification]) {
        for m in modifications {
            let slot = m.slot;
            match m.buffer.op {
                BindlessArrayUpdateOperation::None => {}
                BindlessArrayUpdateOperation::Emplace => {
                    let buffer = &*(m.buffer.handle.0 as *mut BufferImpl);
                    let view = &mut self.buffers[slot];
                    view.data = buffer.data as *mut u8;
                    view.size = buffer.size;
                    view.data = view.data.add(m.buffer.offset);
                    view.size -= m.buffer.offset;
                    view.ty = buffer.ty;
                    self.buffers[slot] = *view;
                }
                BindlessArrayUpdateOperation::Remove => {
                    self.buffers[slot] = defs::BufferView::default();
                }
            };
            match m.tex2d.op {
                BindlessArrayUpdateOperation::None => {}
                BindlessArrayUpdateOperation::Emplace => {
                    let tex = &*(m.tex2d.handle.0 as *mut TextureImpl);
                    self.tex2ds[slot] = defs::Texture {
                        data: tex.data,
                        width: tex.size[0],
                        height: tex.size[1],
                        depth: 1,
                        storage: tex.storage as u8,
                        dimension: 2,
                        mip_levels: tex.mip_levels,
                        pixel_stride_shift: tex.pixel_stride_shift.try_into().unwrap(),
                        mip_offsets: tex.mip_offsets,
                        sampler: m.tex2d.sampler.encode(),
                    };
                }
                BindlessArrayUpdateOperation::Remove => {
                    self.tex2ds[slot] = defs::Texture::default();
                }
            };
            match m.tex3d.op {
                BindlessArrayUpdateOperation::None => {}
                BindlessArrayUpdateOperation::Emplace => {
                    let tex = &*(m.tex3d.handle.0 as *mut TextureImpl);
                    self.tex3ds[slot] = defs::Texture {
                        data: tex.data,
                        width: tex.size[0],
                        height: tex.size[1],
                        depth: tex.size[2],
                        storage: tex.storage as u8,
                        dimension: 3,
                        mip_levels: tex.mip_levels,
                        pixel_stride_shift: tex.pixel_stride_shift.try_into().unwrap(),
                        mip_offsets: tex.mip_offsets,
                        sampler: m.tex2d.sampler.encode(),
                    };
                }
                BindlessArrayUpdateOperation::Remove => {
                    self.tex3ds[slot] = defs::Texture::default();
                }
            };
        }
    }
}
impl BufferImpl {
    pub(super) fn new(size: usize, align: usize, ty: u64) -> Self {
        let layout = Layout::from_size_align(size, align).unwrap();
        let data = unsafe { std::alloc::alloc_zeroed(layout) };
        Self {
            data,
            size,
            align,
            ty,
        }
    }
}

impl Drop for BufferImpl {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, self.align).unwrap();
        unsafe { std::alloc::dealloc(self.data, layout) };
    }
}
