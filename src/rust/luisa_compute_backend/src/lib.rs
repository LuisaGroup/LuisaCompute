use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use std::ffi::{CStr, CString};

use api::{CreatedSwapchainInfo, PixelFormat};
use binding::Binding;
use libc::{c_char, c_void};
use libloading::Library;
use luisa_compute_api_types as api;
use luisa_compute_ir::{
    ir::{self, KernelModule},
    CArc,
};

pub mod binding;
pub mod cpp_proxy_backend;
#[cfg(feature = "remote")]
pub mod remote;
#[cfg(feature = "remote")]
pub mod api_message;
#[cfg(feature = "cpu")]
pub mod rust;

#[derive(Debug)]
pub struct BackendError {}

pub type Result<T> = std::result::Result<T, BackendError>;

pub struct Context {
    pub(crate) binding: Option<Arc<Binding>>,
    pub(crate) swapchain: Option<Arc<SwapChainForCpuContext>>,
    pub(crate) context: Option<api::Context>,
}

impl Context {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        unsafe {
            let lib_path = path.as_ref().to_path_buf();
            let api_dll = if cfg!(target_os = "windows") {
                lib_path.join("liblc-api.dll")
            } else if cfg!(target_os = "linux") {
                lib_path.join("liblc-api.so")
            } else {
                todo!()
            };
            let swapchain_dll = if cfg!(target_os = "windows") {
                "liblc-vulkan-swapchain.dll"
            } else if cfg!(target_os = "linux") {
                "liblc-vulkan-swapchain.so"
            } else {
                todo!()
            };
            let swapchain = lib_path.join(swapchain_dll);
            let binding = Binding::new(&api_dll).ok().map(Arc::new);
            if let Some(binding) = &binding {
                unsafe extern "C"  fn callback(info: *const c_char, msg: *const c_char) {
                    let info = CStr::from_ptr(info as *mut c_char).to_str().unwrap();
                    let msg = CStr::from_ptr(msg as *mut c_char).to_str().unwrap();
                    match info {
                        "I" => log::log!(target: "lc-cpp", log::Level::Info, "{}", msg),
                        "W" => log::log!(target: "lc-cpp", log::Level::Warn, "{}", msg),
                        "E" | "C" => log::log!(target: "lc-cpp", log::Level::Error, "{}", msg),
                        "D" => log::log!(target: "lc-cpp", log::Level::Debug, "{}", msg),
                        "T" => log::log!(target: "lc-cpp", log::Level::Trace, "{}", msg),
                        _ => panic!("unknown log level: {}", info)
                    }
                }
                (binding.luisa_compute_set_logger_callback)(callback);
            }
            let swapchain = SwapChainForCpuContext::new(swapchain)
                .ok()
                .map(|x| Arc::new(x));
            let lib_path_c_str = std::ffi::CString::new(lib_path.to_str().unwrap()).unwrap();
            let context = binding.as_ref().map(|binding| {
                (binding.luisa_compute_context_create)(lib_path_c_str.as_c_str().as_ptr())
            });
            Self {
                binding,
                swapchain,
                context,
            }
        }
    }

    pub fn create_device(&self, device: &str) -> crate::Result<Arc<dyn Backend>> {
        let backend: Arc<dyn Backend> = match device {
            "cpu" => {
                let device = rust::RustBackend::new();
                unsafe {
                    if let Some(swapchain) = &self.swapchain {
                        device.set_swapchain_contex(swapchain.clone());
                    }
                }
                device
            }
            "remote" => {
                todo!()
            }
            "cuda" | "dx" | "metal" => cpp_proxy_backend::CppProxyBackend::new(self, device),
            _ => panic!("unsupported device: {}", device),
        };
        Ok(backend)
    }
}

pub struct SwapChainForCpuContext {
    #[allow(dead_code)]
    lib: Library,
    pub create_cpu_swapchain: unsafe extern "C" fn(
        window_handle: u64,
        width: u32,
        height: u32,
        allow_hdr: bool,
        vsync: bool,
        back_buffer_size: u32,
    ) -> *mut c_void,
    pub cpu_swapchain_storage: unsafe extern "C" fn(swapchain: *mut c_void) -> u8,
    pub destroy_cpu_swapchain: unsafe extern "C" fn(swapchain: *mut c_void),
    pub cpu_swapchain_present:
    unsafe extern "C" fn(swapchain: *mut c_void, pixels: *const c_void, size: u64),
}

unsafe impl Send for SwapChainForCpuContext {}

unsafe impl Sync for SwapChainForCpuContext {}

impl SwapChainForCpuContext {
    pub unsafe fn new(libpath: impl AsRef<Path>) -> std::result::Result<Self, libloading::Error> {
        let lib = Library::new(libpath.as_ref())?;
        let create_cpu_swapchain = *lib.get(b"luisa_compute_create_cpu_swapchain\0")?;
        let destroy_cpu_swapchain = *lib.get(b"luisa_compute_destroy_cpu_swapchain\0")?;
        let cpu_swapchain_present = *lib.get(b"luisa_compute_cpu_swapchain_present\0")?;
        let cpu_swapchain_storage = *lib.get(b"luisa_compute_cpu_swapchain_storage\0")?;
        Ok(Self {
            lib,
            create_cpu_swapchain,
            destroy_cpu_swapchain,
            cpu_swapchain_present,
            cpu_swapchain_storage,
        })
    }
}

pub trait Backend: Sync + Send {
    unsafe fn set_swapchain_contex(&self, ctx: Arc<SwapChainForCpuContext>);
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
    fn create_swapchain(
        &self,
        window_handle: u64,
        stream_handle: api::Stream,
        width: u32,
        height: u32,
        allow_hdr: bool,
        vsync: bool,
        back_buffer_size: u32,
    ) -> Result<CreatedSwapchainInfo>;
    fn destroy_swapchain(&self, swap_chain: api::Swapchain);
    fn present_display_in_stream(
        &self,
        stream_handle: api::Stream,
        swapchain_handle: api::Swapchain,
        image_handle: api::Texture,
    );
    fn create_shader(
        &self,
        kernel: CArc<KernelModule>,
        options: &api::ShaderOption,
    ) -> Result<api::CreatedShaderInfo>;
    fn shader_cache_dir(&self, shader: api::Shader) -> Option<PathBuf>;
    fn destroy_shader(&self, shader: api::Shader);
    fn create_event(&self) -> Result<api::CreatedResourceInfo>;
    fn destroy_event(&self, event: api::Event);
    fn signal_event(&self, event: api::Event, stream: api::Stream);
    fn wait_event(&self, event: api::Event, stream: api::Stream) -> Result<()>;
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