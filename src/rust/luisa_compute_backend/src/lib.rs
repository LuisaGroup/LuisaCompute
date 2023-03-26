use std::path::PathBuf;

use api::PixelFormat;
use libc::c_void;
use luisa_compute_api_types as api;
use luisa_compute_ir::{
    ir::{self, KernelModule, Type},
    CArc,
};
#[cfg(feature = "remote")]
pub mod remote;
#[cfg(feature = "cpu")]
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
    fn create_stream(&self, tag: api::StreamTag) -> Result<api::CreatedResourceInfo>;
    fn destroy_stream(&self, stream: api::Stream);
    fn synchronize_stream(&self, stream: api::Stream) -> Result<()>;
    fn dispatch(
        &self,
        stream: api::Stream,
        command_list: &[api::Command],
        callback: (extern "C" fn(*mut u8), *mut u8),
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
    fn create_shader(
        &self,
        kernel: CArc<KernelModule>,
        options: &api::ShaderOption,
    ) -> Result<api::CreatedShaderInfo>;
    fn shader_cache_dir(&self, shader: api::Shader) -> Option<PathBuf>;
    fn destroy_shader(&self, shader: api::Shader);
    fn create_event(&self) -> Result<api::CreatedResourceInfo>;
    fn destroy_event(&self, event: api::Event);
    fn signal_event(&self, event: api::Event, stream:api::Stream);
    fn wait_event(&self, event: api::Event, stream:api::Stream) -> Result<()>;
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

#[no_mangle]
pub extern "C" fn lc_rs_destroy_backend(ptr: *mut c_void) {
    unsafe {
        let ptr = ptr as *mut Box<dyn Backend>;
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub extern "C" fn lc_rs_create_backend(name: *const std::ffi::c_char) -> *mut c_void {
    let name = unsafe { std::ffi::CStr::from_ptr(name) };
    let name = name.to_str().unwrap();
    let backend = match name {
        #[cfg(feature = "cpu")]
        "rust" => Box::new(rust::RustBackend::new()),
        #[cfg(feature = "remote")]
        "remote" => Box::new(remote::RemoteBackend::new()),
        _ => panic!("unknown backend"),
    };
    Box::into_raw(Box::new(backend)) as *mut c_void
}

#[no_mangle]
pub extern "C" fn lc_rs_create_buffer(
    backend: *mut c_void,
    ty: *const CArc<ir::Type>,
    count: usize,
) -> api::CreatedBufferInfo {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    let ty = unsafe { &*(ty as *const CArc<ir::Type>) };
    backend.create_buffer(ty, count).unwrap()
}

#[no_mangle]
pub extern "C" fn lc_rs_destroy_buffer(backend: *mut c_void, buffer: api::Buffer) {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.destroy_buffer(buffer)
}
#[no_mangle]
pub extern "C" fn lc_rs_create_texture(
    backend: *mut c_void,
    format: PixelFormat,
    dimension: u32,
    width: u32,
    height: u32,
    depth: u32,
    mipmap_levels: u32,
) -> api::CreatedResourceInfo {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend
        .create_texture(format, dimension, width, height, depth, mipmap_levels)
        .unwrap()
}
#[no_mangle]
pub extern "C" fn lc_rs_destroy_texture(backend: *mut c_void, texture: api::Texture) {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.destroy_texture(texture)
}
#[no_mangle]
pub extern "C" fn lc_rs_create_bindless_array(
    backend: *mut c_void,
    size: usize,
) -> api::CreatedResourceInfo {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.create_bindless_array(size).unwrap()
}
#[no_mangle]
pub extern "C" fn lc_rs_destroy_bindless_array(backend: *mut c_void, array: api::BindlessArray) {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.destroy_bindless_array(array)
}
#[no_mangle]
pub extern "C" fn lc_rs_create_stream(
    backend: *mut c_void,
    tag: api::StreamTag,
) -> api::CreatedResourceInfo {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.create_stream(tag).unwrap()
}
#[no_mangle]
pub extern "C" fn lc_rs_destroy_stream(backend: *mut c_void, stream: api::Stream) {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.destroy_stream(stream)
}
#[no_mangle]
pub extern "C" fn lc_rs_synchronize_stream(backend: *mut c_void, stream: api::Stream) -> bool {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.synchronize_stream(stream).is_ok()
}
#[repr(C)]
pub struct LCDispatchCallback {
    callback: extern "C" fn(*mut u8),
    user_data: *mut u8,
}
#[no_mangle]
pub extern "C" fn lc_rs_dispatch(
    backend: *mut c_void,
    stream: api::Stream,
    command_list: *const api::Command,
    command_count: usize,
    callback: LCDispatchCallback,
) -> bool {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    let command_list = unsafe { std::slice::from_raw_parts(command_list, command_count) };
    backend
        .dispatch(
            stream,
            command_list,
            (callback.callback, callback.user_data),
        )
        .is_ok()
}
#[no_mangle]
pub extern "C" fn lc_rs_create_shader(
    backend: *mut c_void,
    kernel: CArc<KernelModule>,
    option: &api::ShaderOption,
) -> api::CreatedShaderInfo {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.create_shader(kernel, option).unwrap()
}
#[no_mangle]
pub extern "C" fn lc_rs_destroy_shader(backend: *mut c_void, shader: api::Shader) {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.destroy_shader(shader)
}
#[no_mangle]
pub extern "C" fn lc_rs_create_event(backend: *mut c_void) -> api::CreatedResourceInfo {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.create_event().unwrap()
}
#[no_mangle]
pub extern "C" fn lc_rs_destroy_event(backend: *mut c_void, event: api::Event) {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.destroy_event(event)
}
#[no_mangle]
pub extern "C" fn lc_rs_create_accel(
    backend: *mut c_void,
    option: api::AccelOption,
) -> api::CreatedResourceInfo {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.create_accel(option).unwrap()
}
#[no_mangle]
pub extern "C" fn lc_rs_destroy_accel(backend: *mut c_void, accel: api::Accel) {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.destroy_accel(accel)
}
#[no_mangle]
pub extern "C" fn lc_rs_create_mesh(
    backend: *mut c_void,
    option: api::AccelOption,
) -> api::CreatedResourceInfo {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.create_mesh(option).unwrap()
}
#[no_mangle]
pub extern "C" fn lc_rs_destroy_mesh(backend: *mut c_void, mesh: api::Mesh) {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.destroy_mesh(mesh)
}
#[no_mangle]
pub extern "C" fn lc_rs_create_procedural_primitive(
    backend: *mut c_void,
    option: api::AccelOption,
) -> api::CreatedResourceInfo {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    backend.create_procedural_primitive(option).unwrap()
}
#[no_mangle]
pub extern "C" fn lc_rs_query(
    backend: *mut c_void,
    property: *mut std::ffi::c_char,
    result: *mut std::ffi::c_char,
    result_size: usize,
) {
    let backend = unsafe { &*(backend as *mut Box<dyn Backend>) };
    let property = unsafe { std::ffi::CStr::from_ptr(property) };
    let property = property.to_str().unwrap();
    let value = backend.query(property);
    let value = std::ffi::CString::new(value.unwrap_or("".into())).unwrap();
    let result = unsafe { std::slice::from_raw_parts_mut(result as *mut u8, result_size) };
    let value_len = value.as_bytes().len();
    assert!(value_len < result_size);
    result[..value_len].copy_from_slice(value.as_bytes());
    result[value_len..].fill(0);
}
