use std::path::PathBuf;

use api::PixelFormat;
use libc::c_void;
use luisa_compute_api_types as api;
use luisa_compute_ir::{
    ir::{self, KernelModule, Type},
    CArc,
};
pub mod remote;
pub mod rust;

#[derive(Debug)]
pub struct BackendError {}
pub type Result<T> = std::result::Result<T, BackendError>;
pub trait Backend: Sync + Send {
    fn create_buffer(&self, ty: &CArc<ir::Type>, count: usize) -> Result<api::CreatedBufferInfo>;
    fn destroy_buffer(&self, buffer: api::Buffer);
    fn create_texture(
        &self,
        format: PixelFormat,
        dimension: u32,
        width: u32,
        height: u32,
        depth: u32,
        mipmap_levels: u32,
    ) -> Result<api::CreatedResourceInfo>;
    fn destroy_texture(&self, texture: api::Texture);
    fn create_bindless_array(&self, size: usize) -> Result<api::CreatedResourceInfo>;
    fn destroy_bindless_array(&self, array: api::BindlessArray);
    fn create_stream(&self) -> Result<api::CreatedResourceInfo>;
    fn destroy_stream(&self, stream: api::Stream);
    fn synchronize_stream(&self, stream: api::Stream) -> Result<()>;
    fn dispatch(
        &self,
        stream: api::Stream,
        command_list: &[api::Command],
        callback: (fn(*mut u8), *mut u8),
    ) -> Result<()>;
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
    fn create_shader(&self, kernel: CArc<KernelModule>) -> Result<api::CreatedShaderInfo>;
    fn shader_cache_dir(&self, shader: api::Shader) -> Option<PathBuf>;
    fn destroy_shader(&self, shader: api::Shader);
    fn create_event(&self) -> Result<api::Event>;
    fn destroy_event(&self, event: api::Event);
    fn signal_event(&self, event: api::Event);
    fn wait_event(&self, event: api::Event) -> Result<()>;
    fn synchronize_event(&self, event: api::Event) -> Result<()>;
    fn create_mesh(&self, option: api::AccelOption) -> Result<api::CreatedResourceInfo>;
    fn create_procedural_primitive(
        &self,
        option: api::AccelOption,
    ) -> Result<api::CreatedResourceInfo>;
    fn destroy_mesh(&self, mesh: api::Mesh);
    fn destroy_procedural_primitive(&self, primitive: api::ProceduralPrimitive);
    fn create_accel(&self, option: api::AccelOption) -> Result<api::CreatedResourceInfo>;
    fn destroy_accel(&self, accel: api::Accel);
    fn query(&self, property: &str) -> Option<String>;
}

// pub struct BackendWrapper<B: Backend> {
//     interface: LCDeviceInterface,
//     backend: B,
// }
