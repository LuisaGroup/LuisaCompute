use std::{future::Future, path::PathBuf};

use api::PixelFormat;
use libc::c_void;
use luisa_compute_api_types as api;
use luisa_compute_ir::{
    ir::{KernelModule, Type},
    Gc,
};
pub mod remote;
pub mod rust;
#[derive(Debug)]
pub struct BackendError {}
pub type Result<T> = std::result::Result<T, BackendError>;
pub trait Backend: Sync + Send {
    fn create_buffer(&self, size_bytes: usize, align: usize) -> Result<api::Buffer>;
    fn destroy_buffer(&self, buffer: api::Buffer);
    fn buffer_native_handle(&self, buffer: api::Buffer) -> *mut c_void;
    fn create_texture(
        &self,
        format: PixelFormat,
        dimension: u32,
        width: u32,
        height: u32,
        depth: u32,
        mipmap_levels: u32,
    ) -> Result<api::Texture>;
    fn destroy_texture(&self, texture: api::Texture);
    fn texture_native_handle(&self, texture: api::Texture) -> *mut c_void;
    fn create_bindless_array(&self, size: usize) -> Result<api::BindlessArray>;
    fn destroy_bindless_array(&self, array: api::BindlessArray);
    fn emplace_buffer_in_bindless_array(
        &self,
        array: api::BindlessArray,
        index: usize,
        handle: api::Buffer,
        offset_bytes: usize,
    );
    fn emplace_tex2d_in_bindless_array(
        &self,
        array: api::BindlessArray,
        index: usize,
        handle: api::Texture,
        sampler: api::Sampler,
    );
    fn emplace_tex3d_in_bindless_array(
        &self,
        array: api::BindlessArray,
        index: usize,
        handle: api::Texture,
        sampler: api::Sampler,
    );

    fn remove_buffer_from_bindless_array(&self, array: api::BindlessArray, index: usize);
    fn remove_tex2d_from_bindless_array(&self, array: api::BindlessArray, index: usize);
    fn remove_tex3d_from_bindless_array(&self, array: api::BindlessArray, index: usize);
    fn create_stream(&self) -> Result<api::Stream>;
    fn destroy_stream(&self, stream: api::Stream);
    fn synchronize_stream(&self, stream: api::Stream) -> Result<()>;
    fn stream_native_handle(&self, stream: api::Stream) -> *mut c_void;
    fn dispatch(&self, stream: api::Stream, command_list: &[api::Command]) -> Result<()>;
    // fn create_swap_chain(
    //     &self,
    //     window_handle: u64,
    //     stream_handle: u64,
    //     width: u32,
    //     height: u32,
    //     allow_hdr: bool,
    //     back_buffer_size: u32,
    // ) -> u64;
    // fn destroy_swap_chain(&self, swap_chain: u64);
    // fn swap_chain_pixel_storage(&self, swap_chain: u64) -> api::PixelStorage;
    // fn present_display_in_stream(
    //     &self,
    //     stream_handle: u64,
    //     swapchain_handle: u64,
    //     image_handle: u64,
    // );
    fn create_shader(&self, kernel: Gc<KernelModule>) -> Result<api::Shader>;
    fn create_shader_async(&self, kernel: Gc<KernelModule>) -> Result<api::Shader>;
    fn shader_cache_dir(&self, shader: api::Shader) -> Option<PathBuf>;
    fn destroy_shader(&self, shader: api::Shader);
    fn create_event(&self) -> Result<api::Event>;
    fn destroy_event(&self, event: api::Event);
    fn signal_event(&self, event: api::Event);
    fn wait_event(&self, event: api::Event) -> Result<()>;
    fn synchronize_event(&self, event: api::Event) -> Result<()>;
    fn create_mesh(
        &self,
        hint: api::AccelUsageHint,
        ty: api::MeshType,
        allow_compact: bool,
        allow_update: bool,
    ) -> api::Mesh;
    fn destroy_mesh(&self, mesh: api::Mesh);
    fn create_accel(
        &self,
        hint: api::AccelUsageHint,
        allow_compact: bool,
        allow_update: bool,
    ) -> api::Accel;
    fn destory_accel(&self, accel: api::Accel);
    fn mesh_native_handle(&self, mesh: api::Mesh) -> *mut c_void;
    fn accel_native_handle(&self, accel: api::Accel) -> *mut c_void;
    fn query(&self, property: &str) -> Option<String>;
    fn set_buffer_type(&self, buffer: api::Buffer, ty: Gc<Type>);
}

// pub struct BackendWrapper<B: Backend> {
//     interface: LCDeviceInterface,
//     backend: B,
// }
