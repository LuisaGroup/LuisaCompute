// A Rust implementation of LuisaCompute backend.
#![allow(non_snake_case)]
use std::sync::{atomic::AtomicBool, Arc};

use self::{
    accel::{AccelImpl, GeometryImpl},
    resource::{BindlessArrayImpl, BufferImpl, EventImpl},
    stream::{convert_capture, StreamImpl},
    texture::TextureImpl,
};
use super::Backend;
use crate::{cpu::llvm::LLVM_PATH, SwapChainForCpuContext};
use crate::{cpu::shader::clang_args, panic_abort};
use api::{AccelOption, CreatedBufferInfo, CreatedResourceInfo};
use libc::c_void;
use log::debug;
use luisa_compute_api_types as api;
use luisa_compute_cpu_kernel_defs as defs;
use luisa_compute_ir::{context::type_hash, ir, CArc};
use parking_lot::{Condvar, Mutex, RwLock};
mod codegen;
use codegen::sha256_short;
mod accel;
mod llvm;
mod resource;
mod shader;
mod stream;
mod texture;
pub struct RustBackend {
    shared_pool: Arc<rayon::ThreadPool>,
    swapchain_context: RwLock<Option<Arc<SwapChainForCpuContext>>>,
}
impl RustBackend {
    pub(crate) unsafe fn set_swapchain_contex(&self, ctx: Arc<SwapChainForCpuContext>) {
        let mut self_ctx = self.swapchain_context.write();
        if self_ctx.is_some() {
            panic_abort!("swapchain context already set");
        }
        *self_ctx = Some(ctx);
    }
}
impl Backend for RustBackend {
    fn compute_warp_size(&self) -> u32 {
        1
    }
    fn native_handle(&self) -> *mut c_void {
        self as *const _ as *mut c_void
    }
    fn create_buffer(
        &self,
        ty: &CArc<ir::Type>,
        count: usize,
    ) -> luisa_compute_api_types::CreatedBufferInfo {
        let size_bytes = if ty == &ir::Type::void() {
            count
        } else {
            ty.size() * count
        };
        let alignment = if ty == &ir::Type::void() {
            16
        } else {
            ty.alignment()
        };
        let buffer = Box::new(BufferImpl::new(size_bytes, alignment, type_hash(&ty)));
        let data = buffer.data;
        let ptr = Box::into_raw(buffer);
        CreatedBufferInfo {
            resource: CreatedResourceInfo {
                handle: ptr as u64,
                native_handle: data as *mut std::ffi::c_void,
            },
            element_stride: ty.size(),
            total_size_bytes: size_bytes,
        }
    }
    fn destroy_buffer(&self, buffer: luisa_compute_api_types::Buffer) {
        unsafe {
            let ptr = buffer.0 as *mut BufferImpl;
            drop(Box::from_raw(ptr));
        }
    }

    fn create_texture(
        &self,
        format: luisa_compute_api_types::PixelFormat,
        dimension: u32,
        width: u32,
        height: u32,
        depth: u32,
        mipmap_levels: u32,
        allow_simultaneous_access: bool,
    ) -> luisa_compute_api_types::CreatedResourceInfo {
        let storage = format.storage();

        let texture = TextureImpl::new(
            dimension as u8,
            [width, height, depth],
            storage,
            mipmap_levels as u8,
            allow_simultaneous_access,
        );
        let data = texture.data;
        let ptr = Box::into_raw(Box::new(texture));
        CreatedResourceInfo {
            handle: ptr as u64,
            native_handle: data as *mut std::ffi::c_void,
        }
    }

    fn destroy_texture(&self, texture: luisa_compute_api_types::Texture) {
        unsafe {
            let texture = texture.0 as *mut TextureImpl;
            drop(Box::from_raw(texture));
        }
    }

    fn create_bindless_array(&self, size: usize) -> luisa_compute_api_types::CreatedResourceInfo {
        let bindless_array = BindlessArrayImpl {
            buffers: vec![defs::BufferView::default(); size],
            tex2ds: vec![defs::Texture::default(); size],
            tex3ds: vec![defs::Texture::default(); size],
        };
        let ptr = Box::into_raw(Box::new(bindless_array));
        CreatedResourceInfo {
            handle: ptr as u64,
            native_handle: ptr as *mut std::ffi::c_void,
        }
    }
    fn destroy_bindless_array(&self, array: luisa_compute_api_types::BindlessArray) {
        unsafe {
            let ptr = array.0 as *mut BindlessArrayImpl;
            drop(Box::from_raw(ptr));
        }
    }

    fn create_stream(&self, _tag: api::StreamTag) -> luisa_compute_api_types::CreatedResourceInfo {
        let stream = Box::into_raw(Box::new(StreamImpl::new(self.shared_pool.clone())));
        CreatedResourceInfo {
            handle: stream as u64,
            native_handle: stream as *mut std::ffi::c_void,
        }
    }

    fn destroy_stream(&self, stream: luisa_compute_api_types::Stream) {
        unsafe {
            let stream = stream.0 as *mut StreamImpl;
            drop(Box::from_raw(stream));
        }
    }

    fn synchronize_stream(&self, stream: luisa_compute_api_types::Stream) {
        unsafe {
            let stream = stream.0 as *mut StreamImpl;
            (*stream).synchronize()
        }
    }

    fn dispatch(
        &self,
        stream_: luisa_compute_api_types::Stream,
        command_list: &[luisa_compute_api_types::Command],
        callback: (extern "C" fn(*mut u8), *mut u8),
    ) {
        unsafe {
            let stream = &*(stream_.0 as *mut StreamImpl);
            let command_list = command_list.to_vec();
            let sb = stream.allocate_staging_buffers(&command_list);
            stream.enqueue(move || stream.dispatch(sb, &command_list), callback);
        }
    }

    fn create_swapchain(
        &self,
        window_handle: u64,
        _stream_handle: api::Stream,
        width: u32,
        height: u32,
        allow_hdr: bool,
        vsync: bool,
        back_buffer_size: u32,
    ) -> api::CreatedSwapchainInfo {
        let ctx = self.swapchain_context.read();
        let ctx = ctx
            .as_ref()
            .unwrap_or_else(|| panic_abort!("swapchain context is not initialized"));
        unsafe {
            let sc_ctx = (ctx.create_cpu_swapchain)(
                window_handle,
                width,
                height,
                allow_hdr,
                vsync,
                back_buffer_size,
            );
            let storage = (ctx.cpu_swapchain_storage)(sc_ctx);
            api::CreatedSwapchainInfo {
                resource: api::CreatedResourceInfo {
                    handle: sc_ctx as u64,
                    native_handle: sc_ctx as *mut std::ffi::c_void,
                },
                storage: std::mem::transmute(storage as u32),
            }
        }
    }
    fn destroy_swapchain(&self, swap_chain: api::Swapchain) {
        let ctx = self.swapchain_context.read();
        let ctx = ctx
            .as_ref()
            .unwrap_or_else(|| panic_abort!("swapchain context is not initialized"));
        unsafe {
            (ctx.destroy_cpu_swapchain)(swap_chain.0 as *mut c_void);
        }
    }
    fn present_display_in_stream(
        &self,
        stream_handle: api::Stream,
        swapchain_handle: api::Swapchain,
        image_handle: api::Texture,
    ) {
        let ctx = self.swapchain_context.read();
        let ctx = ctx
            .as_ref()
            .unwrap_or_else(|| panic_abort!("swapchain context is not initialized"));
        unsafe {
            let stream = &*(stream_handle.0 as *mut StreamImpl);
            let img = &*(image_handle.0 as *mut TextureImpl);
            let storage = (ctx.cpu_swapchain_storage)(swapchain_handle.0 as *mut c_void);
            let storage: api::PixelStorage = std::mem::transmute(storage as u32);
            assert_eq!(storage, img.storage);

            let present = ctx.cpu_swapchain_present;
            let present_completed = Arc::new(AtomicBool::new(false));
            stream.enqueue(
                {
                    let present_completed = present_completed.clone();
                    move || {
                        let pixels = img.view(0).copy_to_vec_par_2d();
                        present(
                            swapchain_handle.0 as *mut c_void,
                            pixels.as_ptr() as *const c_void,
                            pixels.len() as u64,
                        );
                        present_completed.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                },
                (empty_callback, std::ptr::null_mut()),
            );
            loop {
                if present_completed.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                std::thread::yield_now();
            }
        }
    }
    fn create_shader(
        &self,
        kernel: &luisa_compute_ir::ir::KernelModule,
        _options: &api::ShaderOption,
    ) -> luisa_compute_api_types::CreatedShaderInfo {
        // let debug =
        //     luisa_compute_ir::ir::debug::luisa_compute_ir_dump_human_readable(&kernel.module);
        // let debug = std::ffi::CString::new(debug.as_ref()).unwrap();
        // println!("{}", debug.to_str().unwrap());
        // {
        //     let debug = luisa_compute_ir::serialize::serialize_kernel_module_to_json_str(&kernel);
        //     println!("{}", debug);
        // }
        let tic = std::time::Instant::now();
        let mut gened = codegen::cpp::CpuCodeGen::run(&kernel);
        debug!(
            "Source generated in {:.3}ms",
            (std::time::Instant::now() - tic).as_secs_f64() * 1e3
        );
        let args = clang_args();
        let args = args.join(",");
        gened.source.push_str(&format!(
            "\n// clang args: {}\n// clang path: {}\n// llvm path:{}",
            args, LLVM_PATH.clang, LLVM_PATH.llvm
        ));
        let hash = sha256_short(&gened.source);
        let gened_src = gened.source.replace("##kernel_fn##", &hash);
        let mut shader = None;
        for tries in 0..2 {
            let lib_path = shader::compile(&hash, &gened_src, tries == 1).unwrap();
            let mut captures = vec![];
            let mut custom_ops = vec![];
            unsafe {
                for c in kernel.captures.as_ref() {
                    captures.push(convert_capture(*c));
                }
                for op in kernel.cpu_custom_ops.as_ref() {
                    custom_ops.push(defs::CpuCustomOp {
                        func: op.func,
                        data: op.data,
                    });
                }
            }
            shader = shader::ShaderImpl::new(
                hash.clone(),
                lib_path,
                captures,
                custom_ops,
                kernel.block_size,
                &gened.messages,
            );
            if shader.is_some() {
                break;
            }
            if tries == 0 {
                log::error!("Failed to compile kernel. Could LLVM be updated? Retrying");
            } else {
                panic_abort!("Failed to compile kernel. Aborting");
            }
        }
        let shader = Box::new(shader.unwrap());
        let shader = Box::into_raw(shader);
        luisa_compute_api_types::CreatedShaderInfo {
            resource: CreatedResourceInfo {
                handle: shader as u64,
                native_handle: shader as *mut std::ffi::c_void,
            },
            block_size: kernel.block_size,
        }
    }

    fn shader_cache_dir(
        &self,
        shader: luisa_compute_api_types::Shader,
    ) -> Option<std::path::PathBuf> {
        unsafe {
            let shader = shader.0 as *mut shader::ShaderImpl;
            Some((*shader).dir.clone())
        }
    }

    fn destroy_shader(&self, shader: luisa_compute_api_types::Shader) {
        unsafe {
            let shader = shader.0 as *mut shader::ShaderImpl;
            drop(Box::from_raw(shader));
        }
    }

    fn create_event(&self) -> luisa_compute_api_types::CreatedResourceInfo {
        let event = Box::new(EventImpl::new());
        let event = Box::into_raw(event);
        luisa_compute_api_types::CreatedResourceInfo {
            handle: event as u64,
            native_handle: event as *mut std::ffi::c_void,
        }
    }

    fn destroy_event(&self, _event: luisa_compute_api_types::Event) {
        unsafe {
            let event = _event.0 as *mut EventImpl;
            drop(Box::from_raw(event));
        }
    }

    fn signal_event(&self, event: api::Event, stream: api::Stream, value: u64) {
        unsafe {
            let event = &*(event.0 as *mut EventImpl);
            let stream = &*(stream.0 as *mut StreamImpl);
            stream.enqueue(
                move || {
                    event.signal(value);
                },
                (empty_callback, std::ptr::null_mut()),
            )
        }
    }
    fn wait_event(&self, event: luisa_compute_api_types::Event, stream: api::Stream, value: u64) {
        unsafe {
            let event = &*(event.0 as *mut EventImpl);
            let stream = &*(stream.0 as *mut StreamImpl);
            stream.enqueue(
                move || {
                    event.wait(value);
                },
                (empty_callback, std::ptr::null_mut()),
            );
        }
    }
    fn synchronize_event(&self, event: luisa_compute_api_types::Event, value: u64) {
        unsafe {
            let event = &*(event.0 as *mut EventImpl);
            event.synchronize(value);
        }
    }
    fn is_event_completed(&self, event: luisa_compute_api_types::Event, value: u64) -> bool {
        unsafe {
            let event = &*(event.0 as *mut EventImpl);
            event.is_completed(value)
        }
    }
    fn create_mesh(&self, option: AccelOption) -> api::CreatedResourceInfo {
        unsafe {
            let mesh = Box::new(GeometryImpl::new(
                option.hint,
                option.allow_compaction,
                option.allow_update,
            ));
            let mesh = Box::into_raw(mesh);
            api::CreatedResourceInfo {
                handle: mesh as u64,
                native_handle: mesh as *mut std::ffi::c_void,
            }
        }
    }
    fn create_procedural_primitive(&self, option: api::AccelOption) -> api::CreatedResourceInfo {
        unsafe {
            let mesh = Box::new(GeometryImpl::new(
                option.hint,
                option.allow_compaction,
                option.allow_update,
            ));
            let mesh = Box::into_raw(mesh);
            api::CreatedResourceInfo {
                handle: mesh as u64,
                native_handle: mesh as *mut std::ffi::c_void,
            }
        }
    }
    fn destroy_mesh(&self, mesh: api::Mesh) {
        unsafe {
            let mesh = mesh.0 as *mut GeometryImpl;
            drop(Box::from_raw(mesh));
        }
    }
    fn destroy_procedural_primitive(&self, primitive: api::ProceduralPrimitive) {
        unsafe {
            let mesh = primitive.0 as *mut GeometryImpl;
            drop(Box::from_raw(mesh));
        }
    }
    fn create_accel(&self, _option: AccelOption) -> api::CreatedResourceInfo {
        unsafe {
            let accel = Box::new(AccelImpl::new());
            let accel = Box::into_raw(accel);
            api::CreatedResourceInfo {
                handle: accel as u64,
                native_handle: accel as *mut std::ffi::c_void,
            }
        }
    }
    fn destroy_accel(&self, accel: api::Accel) {
        unsafe {
            let accel = accel.0 as *mut AccelImpl;
            drop(Box::from_raw(accel));
        }
    }
    fn query(&self, property: &str) -> Option<String> {
        match property {
            "device_name" => Some("cpu".to_string()),
            _ => None,
        }
    }
}
impl RustBackend {
    pub fn new() -> Self {
        let num_threads = match std::env::var("LUISA_NUM_THREADS") {
            Ok(s) => s.parse::<usize>().unwrap(),
            Err(_) => std::thread::available_parallelism().unwrap().get(),
        };

        RustBackend {
            shared_pool: Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .start_handler(|_| {
                        #[cfg(target_arch = "x86_64")]
                        {
                            unsafe {
                                use core::arch::x86_64::*;
                                _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
                                const _MM_DENORMALS_ZERO_MASK: u32 = 0x0040;
                                const _MM_DENORMALS_ZERO_ON: u32 = 0x0040;
                                _mm_setcsr(
                                    (_mm_getcsr() & !_MM_DENORMALS_ZERO_MASK)
                                        | (_MM_DENORMALS_ZERO_ON),
                                );
                            }
                        }
                    })
                    .build()
                    .unwrap(),
            ),
            swapchain_context: RwLock::new(None),
        }
    }
}
extern "C" fn empty_callback(_: *mut u8) {}
