use std::{
    collections::VecDeque,
    sync::{atomic::AtomicUsize, Arc},
    thread::{self, JoinHandle},
};

use luisa_compute_api_types::{Argument, Sampler};
use luisa_compute_ir::{
    context::type_hash,
    ir::{Binding, Capture},
};
use parking_lot::{Condvar, Mutex};
use rayon;

use luisa_compute_cpu_kernel_defs as defs;

use super::{
    accel::{AccelImpl, MeshImpl},
    resource::{BindlessArrayImpl, BufferImpl},
    shader::ShaderImpl,
    texture::TextureImpl,
};
struct StreamContext {
    queue: Mutex<VecDeque<Arc<dyn Fn() + Send + Sync>>>,
    new_work: Condvar,
    sync: Condvar,
    work_count: AtomicUsize,
    finished_count: AtomicUsize,
}
pub(super) struct StreamImpl {
    shared_pool: Arc<rayon::ThreadPool>,
    #[allow(dead_code)]
    private_thread: Arc<JoinHandle<()>>,
    ctx: Arc<StreamContext>,
}

impl StreamImpl {
    pub(super) fn new(shared_pool: Arc<rayon::ThreadPool>) -> Self {
        let ctx = Arc::new(StreamContext {
            queue: Mutex::new(VecDeque::new()),
            new_work: Condvar::new(),
            sync: Condvar::new(),
            work_count: AtomicUsize::new(0),
            finished_count: AtomicUsize::new(0),
        });
        let private_thread = {
            let ctx = ctx.clone();
            Arc::new(thread::spawn(move || {
                let mut guard = ctx.queue.lock();
                loop {
                    while guard.is_empty() {
                        ctx.new_work.wait(&mut guard);
                    }
                    loop {
                        if guard.is_empty() {
                            break;
                        }
                        let work = guard.pop_front().unwrap();
                        drop(guard);
                        work();
                        ctx.finished_count
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        guard = ctx.queue.lock();
                        if guard.is_empty() {
                            ctx.sync.notify_one();
                            break;
                        }
                    }
                }
            }))
        };
        Self {
            shared_pool,
            private_thread,
            ctx,
        }
    }
    pub(super) fn synchronize(&self) {
        let mut guard = self.ctx.queue.lock();
        while self
            .ctx
            .work_count
            .load(std::sync::atomic::Ordering::Relaxed)
            > self
                .ctx
                .finished_count
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            self.ctx.sync.wait(&mut guard);
        }
    }
    pub(super) fn enqueue(&self, work: impl Fn() + Send + Sync + 'static) {
        let mut guard = self.ctx.queue.lock();
        guard.push_back(Arc::new(work));
        self.ctx
            .work_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.ctx.new_work.notify_one();
    }
    pub(super) fn parallel_for(
        &self,
        kernel: impl Fn(usize) + Send + Sync + 'static,
        block: usize,
        count: usize,
    ) {
        let kernel = Arc::new(kernel);
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = self.shared_pool.clone();
        let nthreads = pool.current_num_threads();
        pool.scope(|s| {
            for _ in 0..nthreads {
                s.spawn(|_| loop {
                    let index = counter.fetch_add(block, std::sync::atomic::Ordering::Relaxed);
                    if index >= count {
                        break;
                    }

                    for i in index..(index + block).min(count) {
                        kernel(i);
                    }
                })
            }
        });
    }
    pub(super) fn dispatch(&self, command_list: &[luisa_compute_api_types::Command]) {
        unsafe {
            for cmd in command_list {
                match cmd {
                    luisa_compute_api_types::Command::BufferUpload(cmd) => {
                        let buffer = &*(cmd.buffer.0 as *mut BufferImpl);
                        let _lk = buffer.lock.try_write().unwrap();
                        let offset = cmd.offset;
                        let size = cmd.size;
                        let data = cmd.data;

                        std::ptr::copy_nonoverlapping(data, buffer.data.add(offset), size);
                    }
                    luisa_compute_api_types::Command::BufferDownload(cmd) => {
                        let buffer = &*(cmd.buffer.0 as *mut BufferImpl);
                        let _lk = buffer.lock.try_read().unwrap();
                        let offset = cmd.offset;
                        let size = cmd.size;
                        let data = cmd.data;
                        std::ptr::copy_nonoverlapping(buffer.data.add(offset), data, size);
                    }
                    luisa_compute_api_types::Command::BufferCopy(cmd) => {
                        let src = &*(cmd.src.0 as *mut BufferImpl);
                        let dst = &*(cmd.dst.0 as *mut BufferImpl);
                        let _src_lk = src.lock.try_read().unwrap();
                        let _dst_lk = dst.lock.try_write().unwrap();
                        assert_ne!(src.data, dst.data);
                        let src_offset = cmd.src_offset;
                        let dst_offset = cmd.dst_offset;
                        let size = cmd.size;
                        std::ptr::copy_nonoverlapping(
                            src.data.add(src_offset),
                            dst.data.add(dst_offset),
                            size,
                        );
                    }
                    luisa_compute_api_types::Command::BufferToTextureCopy(cmd) => {
                        let buffer = &*(cmd.buffer.0 as *mut BufferImpl);
                        let texture = &*(cmd.texture.0 as *mut TextureImpl);
                        let _buffer_lk = buffer.lock.try_read().unwrap();
                        let level: u8 = cmd.texture_level.try_into().unwrap();
                        let _texture_lk = texture.locks[level as usize].try_write().unwrap();
                        let dim = texture.dimension;
                        let view = texture.view(level);
                        assert_eq!(cmd.storage, texture.storage);
                        assert!(buffer.size >= cmd.buffer_offset + view.data_size());
                        if dim == 2 {
                            assert_eq!(cmd.texture_size, view.size);
                            view.copy_from_2d(buffer.data.add(cmd.buffer_offset))
                        } else {
                            view.copy_from_3d(buffer.data.add(cmd.buffer_offset))
                        }
                    }
                    luisa_compute_api_types::Command::TextureToBufferCopy(cmd) => {
                        let buffer = &*(cmd.buffer.0 as *mut BufferImpl);
                        let texture = &*(cmd.texture.0 as *mut TextureImpl);
                        let level: u8 = cmd.texture_level.try_into().unwrap();
                        let _texture_lk = texture.locks[level as usize].try_read().unwrap();
                        let _buffer_lk = buffer.lock.try_write().unwrap();
                        let dim = texture.dimension;
                        let view = texture.view(level);
                        assert_eq!(cmd.storage, texture.storage);
                        assert!(buffer.size >= cmd.buffer_offset + view.data_size());
                        if dim == 2 {
                            assert_eq!(cmd.texture_size, view.size);
                            view.copy_to_2d(buffer.data.add(cmd.buffer_offset))
                        } else {
                            view.copy_to_3d(buffer.data.add(cmd.buffer_offset))
                        }
                    }
                    luisa_compute_api_types::Command::TextureUpload(cmd) => {
                        let texture = &*(cmd.texture.0 as *mut TextureImpl);
                        let level: u8 = cmd.level.try_into().unwrap();
                        let _lk = texture.locks[level as usize].try_write().unwrap();
                        let dim = texture.dimension;
                        let view = texture.view(level);
                        assert_eq!(cmd.storage, texture.storage);
                        if dim == 2 {
                            assert_eq!(cmd.size, view.size);
                            view.copy_from_2d(cmd.data)
                        } else {
                            view.copy_from_3d(cmd.data)
                        }
                    }
                    luisa_compute_api_types::Command::TextureDownload(cmd) => {
                        let texture = &*(cmd.texture.0 as *mut TextureImpl);
                        let level: u8 = cmd.level.try_into().unwrap();
                        let _lk = texture.locks[level as usize].try_read().unwrap();
                        let dim = texture.dimension;
                        assert_eq!(cmd.storage, texture.storage);
                        let view = texture.view(level);
                        if dim == 2 {
                            view.copy_to_2d(cmd.data)
                        } else {
                            view.copy_to_3d(cmd.data)
                        }
                    }
                    luisa_compute_api_types::Command::TextureCopy(cmd) => {
                        let src = &*(cmd.src.0 as *mut TextureImpl);
                        let dst = &*(cmd.dst.0 as *mut TextureImpl);
                        let src_level: u8 = cmd.src_level.try_into().unwrap();
                        let dst_level: u8 = cmd.dst_level.try_into().unwrap();
                        let src_view = src.view(src_level);
                        let dst_view = dst.view(dst_level);
                        let _src_lk = src.locks[src_level as usize].try_read().unwrap();
                        let _dst_lk = dst.locks[dst_level as usize].try_write().unwrap();
                        assert_eq!(cmd.storage, src.storage);
                        assert_eq!(cmd.storage, dst.storage);
                        assert_eq!(src_view.size, cmd.size);
                        assert_eq!(src_view.size, cmd.size);
                        if src_view.data == dst_view.data {
                            return;
                        }
                        std::ptr::copy_nonoverlapping(
                            src_view.data,
                            dst_view.data,
                            src_view.data_size(),
                        );
                    }
                    luisa_compute_api_types::Command::ShaderDispatch(cmd) => {
                        let shader = &*(cmd.shader.0 as *mut ShaderImpl);
                        let dispatch_size = cmd.dispatch_size;
                        let block_size = shader.block_size;

                        let blocks: [u32; 3] = [
                            ((dispatch_size[0] + block_size[0] - 1) / block_size[0]).max(1),
                            ((dispatch_size[1] + block_size[1] - 1) / block_size[1]).max(1),
                            ((dispatch_size[2] + block_size[2] - 1) / block_size[2]).max(1),
                        ];
                        let block_count =
                            blocks[0] as usize * blocks[1] as usize * blocks[2] as usize;
                        let kernel = shader.fn_ptr();
                        let mut args: Vec<defs::KernelFnArg> = Vec::new();

                        for i in 0..cmd.args_count {
                            let arg = *cmd.args.add(i);
                            args.push(convert_arg(arg));
                        }
                        let args = Arc::new(args);
                        let kernel_args = defs::KernelFnArgs {
                            captured: shader.captures.as_ptr(),
                            captured_count: shader.captures.len(),
                            args: (*args).as_ptr(),
                            dispatch_id: [0, 0, 0],
                            thread_id: [0, 0, 0],
                            dispatch_size,
                            block_id: [0, 0, 0],
                            args_count: args.len(),
                            custom_ops: shader.custom_ops.as_ptr(),
                            custom_ops_count: shader.custom_ops.len(),
                        };

                        self.parallel_for(
                            move |i| {
                                let mut args = kernel_args;
                                let block_z = i / (blocks[0] * blocks[1]) as usize;
                                let block_y =
                                    (i % (blocks[0] * blocks[1]) as usize) / blocks[0] as usize;
                                let block_x = i % blocks[0] as usize;
                                args.block_id = [block_x as u32, block_y as u32, block_z as u32];
                                let max_tx = dispatch_size[0]
                                    .min(block_size[0] * (block_x as u32 + 1))
                                    - block_size[0] * block_x as u32;
                                let max_ty = dispatch_size[1]
                                    .min(block_size[1] * (block_y as u32 + 1))
                                    - block_size[1] * block_y as u32;
                                let max_tz = dispatch_size[2]
                                    .min(block_size[2] * (block_z as u32 + 1))
                                    - block_size[2] * block_z as u32;
                                for tz in 0..max_tz {
                                    for ty in 0..max_ty {
                                        for tx in 0..max_tx {
                                            let dispatch_x = block_size[0] * block_x as u32 + tx;
                                            let dispatch_y = block_size[1] * block_y as u32 + ty;
                                            let dispatch_z = block_size[2] * block_z as u32 + tz;
                                            args.thread_id = [tx, ty, tz];
                                            args.dispatch_id = [dispatch_x, dispatch_y, dispatch_z];
                                            kernel(&args);
                                        }
                                    }
                                }
                            },
                            1,
                            block_count,
                        );
                    }
                    luisa_compute_api_types::Command::MeshBuild(mesh_build) => {
                        let mesh = &mut *(mesh_build.mesh.0 as *mut MeshImpl);
                        mesh.build(mesh_build);
                    }
                    luisa_compute_api_types::Command::AccelBuild(accel_build) => {
                        let accel = &mut *(accel_build.accel.0 as *mut AccelImpl);
                        accel.update(
                            accel_build.instance_count as usize,
                            std::slice::from_raw_parts(
                                accel_build.modifications,
                                accel_build.modifications_count,
                            ),
                        );
                    }
                    luisa_compute_api_types::Command::BindlessArrayUpdate(_) => {}
                }
            }
        }
    }
}
#[inline]
pub unsafe fn convert_arg(arg: Argument) -> defs::KernelFnArg {
    match arg {
        luisa_compute_api_types::Argument::Buffer(buffer_arg) => {
            let buffer = &*(buffer_arg.buffer.0 as *mut BufferImpl);
            let offset = buffer_arg.offset;
            let size = buffer_arg.size;
            assert!(offset + size <= buffer.size);
            defs::KernelFnArg::Buffer(defs::BufferView {
                data: buffer.data.add(offset),
                size,
                ty: buffer.ty.as_ref().map(|t| type_hash(t) as u64).unwrap_or(0),
            })
        }
        luisa_compute_api_types::Argument::Texture(t) => {
            let texture = &*(t.texture.0 as *mut TextureImpl);
            let level = t.level as usize;
            defs::KernelFnArg::Texture(
                texture.into_c_texture(Sampler {
                    address: luisa_compute_api_types::SamplerAddress::Edge,
                    filter: luisa_compute_api_types::SamplerFilter::Point,
                }),
                level as u8,
            )
        }
        luisa_compute_api_types::Argument::Uniform(_) => todo!(),
        luisa_compute_api_types::Argument::Accel(accel) => {
            let accel = &*(accel.0 as *mut AccelImpl);
            defs::KernelFnArg::Accel(defs::Accel {
                handle: accel as *const _ as *const std::ffi::c_void,
                trace_closest,
                trace_any,
                set_instance_visibility,
                set_instance_transform,
                instance_transform,
            })
        }
        luisa_compute_api_types::Argument::BindlessArray(a) => {
            let a = &*(a.0 as *mut BindlessArrayImpl);
            defs::KernelFnArg::BindlessArray(defs::BindlessArray {
                buffers: a.buffers.as_ptr(),
                buffers_count: a.buffers.len(),
                texture2ds: a.tex2ds.as_ptr(),
                texture2ds_count: a.tex2ds.len(),
                texture3ds: a.tex3ds.as_ptr(),
                texture3ds_count: a.tex3ds.len(),
            })
        }
    }
}
#[inline]
pub unsafe fn convert_capture(c: Capture) -> defs::KernelFnArg {
    match c.binding {
        Binding::Buffer(buffer_arg) => {
            let buffer = &*(buffer_arg.handle as *mut BufferImpl);
            let offset = buffer_arg.offset as usize;
            let size = buffer_arg.size;
            assert!(
                offset + size <= buffer.size,
                "offset: {}, size: {}, buffer.size: {}",
                offset,
                size,
                buffer.size
            );
            defs::KernelFnArg::Buffer(defs::BufferView {
                data: buffer.data.add(offset),
                size,
                ty: buffer.ty.as_ref().map(|t| type_hash(t) as u64).unwrap_or(0),
            })
        }
        Binding::Texture(t) => {
            let texture = &*(t.handle as *mut TextureImpl);
            let level = t.level as usize;
            defs::KernelFnArg::Texture(
                texture.into_c_texture(Sampler {
                    address: luisa_compute_api_types::SamplerAddress::Edge,
                    filter: luisa_compute_api_types::SamplerFilter::Point,
                }),
                level as u8,
            )
        }
        Binding::BindlessArray(a) => {
            let a = &*(a.handle as *mut BindlessArrayImpl);
            defs::KernelFnArg::BindlessArray(defs::BindlessArray {
                buffers: a.buffers.as_ptr(),
                buffers_count: a.buffers.len(),
                texture2ds: a.tex2ds.as_ptr(),
                texture2ds_count: a.tex2ds.len(),
                texture3ds: a.tex3ds.as_ptr(),
                texture3ds_count: a.tex3ds.len(),
            })
        }
        Binding::Accel(accel) => {
            let accel = &*(accel.handle as *mut AccelImpl);
            defs::KernelFnArg::Accel(defs::Accel {
                handle: accel as *const _ as *const std::ffi::c_void,
                trace_closest,
                trace_any,
                set_instance_visibility,
                set_instance_transform,
                instance_transform,
            })
        }
    }
}

extern "C" fn trace_closest(accel: *const std::ffi::c_void, ray: &defs::Ray) -> defs::Hit {
    unsafe {
        let accel = &*(accel as *const AccelImpl);
        accel.trace_closest(ray)
    }
}
extern "C" fn trace_any(accel: *const std::ffi::c_void, ray: &defs::Ray) -> bool {
    unsafe {
        let accel = &*(accel as *const AccelImpl);
        accel.trace_any(ray)
    }
}
extern "C" fn instance_transform(accel: *const std::ffi::c_void, instance_id: u32) -> defs::Mat4 {
    unsafe {
        let accel = &*(accel as *const AccelImpl);
        let affine = accel.instance_transform(instance_id);
        defs::Mat4([
            affine[0], affine[1], affine[2], 0.0, affine[3], affine[4], affine[5], 0.0, affine[6],
            affine[7], affine[8], 0.0, affine[9], affine[10], affine[11], 1.0,
        ])
    }
}
extern "C" fn set_instance_transform(
    accel: *const std::ffi::c_void,
    instance_id: u32,
    transform: &defs::Mat4,
) {
    unsafe {
        let accel = &*(accel as *const AccelImpl);
        let affine = [
            transform.0[0],
            transform.0[1],
            transform.0[2],
            transform.0[4],
            transform.0[5],
            transform.0[6],
            transform.0[8],
            transform.0[9],
            transform.0[10],
            transform.0[12],
            transform.0[13],
            transform.0[14],
        ];
        accel.set_instance_transform(instance_id, affine);
    }
}
extern "C" fn set_instance_visibility(
    accel: *const std::ffi::c_void,
    instance_id: u32,
    visible: bool,
) {
    unsafe {
        let accel = &*(accel as *const AccelImpl);
        accel.set_instance_visibility(instance_id, visible);
    }
}
