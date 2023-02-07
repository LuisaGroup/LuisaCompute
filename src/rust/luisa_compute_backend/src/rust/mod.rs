// A Rust implementation of LuisaCompute backend.

use std::{future::Future, ptr::null, sync::Arc};

use self::{
    accel::{AccelImpl, MeshImpl},
    resource::{BindlessArrayImpl, BufferImpl},
    stream::{convert_capture, StreamImpl},
};
use super::Backend;
use log::info;
use luisa_compute_api_types as api;
use luisa_compute_cpu_kernel_defs as defs;
use luisa_compute_ir::{codegen::CodeGen, ir::Type, Gc};
use rayon::ThreadPool;
use sha2::{Digest, Sha256};
mod accel;
mod resource;
mod shader;
mod stream;
mod texture;
pub struct RustBackend {
    shared_pool: Arc<rayon::ThreadPool>,
}
impl RustBackend {
    #[inline]
    fn create_shader_impl(
        kernel: Gc<luisa_compute_ir::ir::KernelModule>,
    ) -> super::Result<luisa_compute_api_types::Shader> {
        // let debug =
        //     luisa_compute_ir::ir::debug::luisa_compute_ir_dump_human_readable(&kernel.module);
        // let debug = std::ffi::CString::new(debug.as_ref()).unwrap();
        // println!("{}", debug.to_str().unwrap());
        let tic = std::time::Instant::now();
        let gened_src = luisa_compute_ir::codegen::generic_cpp::CpuCodeGen::run(&kernel);
        info!(
            "kernel source generated in {:.3}s",
            (std::time::Instant::now() - tic).as_secs_f64() * 1e3
        );
        // println!("{}", gened_src);
        let lib_path = shader::compile(gened_src).unwrap();
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
        let shader = Box::new(shader::ShaderImpl::new(
            lib_path,
            captures,
            custom_ops,
            kernel.block_size,
        ));
        let shader = Box::into_raw(shader);
        Ok(luisa_compute_api_types::Shader(shader as u64))
    }
}
impl Backend for RustBackend {
    fn query(&self, property: &str) -> Option<String> {
        match property {
            "device_name" => Some("cpu".to_string()),
            _ => None,
        }
    }
    fn set_buffer_type(&self, buffer: luisa_compute_api_types::Buffer, ty: Gc<Type>) {
        unsafe {
            let buffer = &mut *(buffer.0 as *mut BufferImpl);
            buffer.ty = Some(ty);
        }
    }
    fn create_buffer(
        &self,
        size_bytes: usize,
        align: usize,
    ) -> super::Result<luisa_compute_api_types::Buffer> {
        let buffer = Box::new(BufferImpl::new(size_bytes, align));
        let ptr = Box::into_raw(buffer);
        Ok(luisa_compute_api_types::Buffer(ptr as u64))
    }

    fn destroy_buffer(&self, buffer: luisa_compute_api_types::Buffer) {
        unsafe {
            let ptr = buffer.0 as *mut BufferImpl;
            drop(Box::from_raw(ptr));
        }
    }

    fn buffer_native_handle(&self, buffer: luisa_compute_api_types::Buffer) -> *mut libc::c_void {
        unsafe {
            let buffer = &*(buffer.0 as *mut BufferImpl);
            buffer.data as *mut libc::c_void
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
    ) -> super::Result<luisa_compute_api_types::Texture> {
        todo!()
    }

    fn destroy_texture(&self, texture: luisa_compute_api_types::Texture) {
        todo!()
    }

    fn texture_native_handle(
        &self,
        texture: luisa_compute_api_types::Texture,
    ) -> *mut libc::c_void {
        todo!()
    }

    fn create_bindless_array(
        &self,
        size: usize,
    ) -> super::Result<luisa_compute_api_types::BindlessArray> {
        let bindless_array = BindlessArrayImpl {
            buffers: vec![defs::BufferView::default(); size],
        };
        let ptr = Box::into_raw(Box::new(bindless_array));
        Ok(luisa_compute_api_types::BindlessArray(ptr as u64))
    }

    fn destroy_bindless_array(&self, array: luisa_compute_api_types::BindlessArray) {
        unsafe {
            let ptr = array.0 as *mut BindlessArrayImpl;
            drop(Box::from_raw(ptr));
        }
    }

    fn emplace_buffer_in_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
        handle: luisa_compute_api_types::Buffer,
        offset_bytes: usize,
    ) {
        unsafe {
            let array = &mut *(array.0 as *mut BindlessArrayImpl);
            let buffer = &*(handle.0 as *mut BufferImpl);
            let view = &mut array.buffers[index];
            view.data = buffer.data as *mut u8;
            view.size = buffer.size;
            view.data = view.data.add(offset_bytes);
            view.size -= offset_bytes;
            view.ty = buffer.ty.map(|t| Gc::as_ptr(t) as u64).unwrap_or(0);
            array.buffers[index] = *view;
        }
    }

    fn emplace_tex2d_in_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
        handle: luisa_compute_api_types::Texture,
        sampler: luisa_compute_api_types::Sampler,
    ) {
        todo!()
    }

    fn emplace_tex3d_in_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
        handle: luisa_compute_api_types::Texture,
        sampler: luisa_compute_api_types::Sampler,
    ) {
        todo!()
    }

    fn remove_buffer_from_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
    ) {
        unsafe {
            let array = &mut *(array.0 as *mut BindlessArrayImpl);
            array.buffers[index] = defs::BufferView::default();
        }
    }

    fn remove_tex2d_from_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
    ) {
        todo!()
    }

    fn remove_tex3d_from_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
    ) {
        todo!()
    }

    fn create_stream(&self) -> super::Result<luisa_compute_api_types::Stream> {
        let stream = Box::into_raw(Box::new(StreamImpl::new(self.shared_pool.clone())));
        Ok(luisa_compute_api_types::Stream(stream as u64))
    }

    fn destroy_stream(&self, stream: luisa_compute_api_types::Stream) {
        unsafe {
            let stream = stream.0 as *mut StreamImpl;
            drop(Box::from_raw(stream));
        }
    }

    fn synchronize_stream(&self, stream: luisa_compute_api_types::Stream) -> super::Result<()> {
        unsafe {
            let stream = stream.0 as *mut StreamImpl;
            (*stream).synchronize();
            Ok(())
        }
    }

    fn stream_native_handle(&self, stream: luisa_compute_api_types::Stream) -> *mut libc::c_void {
        stream.0 as *mut libc::c_void
    }

    fn dispatch(
        &self,
        stream_: luisa_compute_api_types::Stream,
        command_list: &[luisa_compute_api_types::Command],
    ) -> super::Result<()> {
        unsafe {
            let stream = &*(stream_.0 as *mut StreamImpl);
            let command_list = command_list.to_vec();
            stream.enqueue(move || {
                let stream = &*(stream_.0 as *mut StreamImpl);
                stream.dispatch(&command_list)
            });
            Ok(())
        }
    }
    fn create_shader_async(
        &self,
        kernel: Gc<luisa_compute_ir::ir::KernelModule>,
    ) -> super::Result<luisa_compute_api_types::Shader> {
        Self::create_shader_impl(kernel)
    }
    fn create_shader(
        &self,
        kernel: Gc<luisa_compute_ir::ir::KernelModule>,
    ) -> super::Result<luisa_compute_api_types::Shader> {
        Self::create_shader_impl(kernel)
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

    fn create_event(&self) -> super::Result<luisa_compute_api_types::Event> {
        todo!()
    }

    fn destroy_event(&self, event: luisa_compute_api_types::Event) {
        todo!()
    }

    fn signal_event(&self, event: luisa_compute_api_types::Event) {
        todo!()
    }

    fn wait_event(&self, event: luisa_compute_api_types::Event) -> super::Result<()> {
        todo!()
    }

    fn synchronize_event(&self, event: luisa_compute_api_types::Event) -> super::Result<()> {
        todo!()
    }
    fn create_mesh(
        &self,
        hint: api::AccelUsageHint,
        ty: api::MeshType,
        allow_compact: bool,
        allow_update: bool,
    ) -> api::Mesh {
        // unsafe {
        //     let mesh = Box::new(MeshImpl::new(
        //         usage, v_buffer, v_offset, v_stride, v_count, t_buffer, t_offset, t_stride, t_count,
        //     ));
        //     let mesh = Box::into_raw(mesh);
        //     api::Mesh(mesh as u64)
        // }
        todo!()
    }
    fn destroy_mesh(&self, mesh: api::Mesh) {
        unsafe {
            let mesh = mesh.0 as *mut MeshImpl;
            drop(Box::from_raw(mesh));
        }
    }
    fn mesh_native_handle(&self, mesh: api::Mesh) -> *mut libc::c_void {
        mesh.0 as *mut libc::c_void
    }
    fn create_accel(
        &self,
        hint: api::AccelUsageHint,
        allow_compact: bool,
        allow_update: bool,
    ) -> api::Accel {
        unsafe {
            let accel = Box::new(AccelImpl::new());
            let accel = Box::into_raw(accel);
            api::Accel(accel as u64)
        }
    }
    fn destory_accel(&self, accel: api::Accel) {
        unsafe {
            let accel = accel.0 as *mut AccelImpl;
            drop(Box::from_raw(accel));
        }
    }
    fn accel_native_handle(&self, accel: api::Accel) -> *mut libc::c_void {
        accel.0 as *mut libc::c_void
    }
}
impl RustBackend {
    pub fn new() -> Arc<Self> {
        let num_threads = match std::env::var("LUISA_NUM_THREADS") {
            Ok(s) => s.parse::<usize>().unwrap(),
            Err(_) => std::thread::available_parallelism().unwrap().get(),
        };

        Arc::new(RustBackend {
            shared_pool: Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap(),
            ),
        })
    }
}
