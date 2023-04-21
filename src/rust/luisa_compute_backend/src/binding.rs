use luisa_compute_api_types as api;
use std::ffi::*;
use std::path::Path;
#[repr(C)]
pub struct Binding {
    #[allow(dead_code)]
    pub lib: libloading::Library,
    pub luisa_compute_set_logger_callback:
        unsafe extern "C" fn(callback: unsafe extern "C" fn(*const c_char, *const c_char)) -> (),
    pub luisa_compute_free_c_string: unsafe extern "C" fn(cs: *mut c_char) -> (),
    pub luisa_compute_context_create: unsafe extern "C" fn(exe_path: *const c_char) -> api::Context,
    pub luisa_compute_context_destroy: unsafe extern "C" fn(ctx: api::Context) -> (),
    pub luisa_compute_device_create: unsafe extern "C" fn(
        ctx: api::Context,
        name: *const c_char,
        properties: *const c_char,
    ) -> api::Device,
    pub luisa_compute_device_destroy: unsafe extern "C" fn(device: api::Device) -> (),
    pub luisa_compute_device_retain: unsafe extern "C" fn(device: api::Device) -> (),
    pub luisa_compute_device_release: unsafe extern "C" fn(device: api::Device) -> (),
    pub luisa_compute_buffer_create: unsafe extern "C" fn(
        device: api::Device,
        element: *const c_void,
        elem_count: usize,
    ) -> api::CreatedBufferInfo,
    pub luisa_compute_buffer_destroy:
        unsafe extern "C" fn(device: api::Device, buffer: api::Buffer) -> (),
    pub luisa_compute_texture_create: unsafe extern "C" fn(
        device: api::Device,
        format: api::PixelFormat,
        dim: u32,
        w: u32,
        h: u32,
        d: u32,
        mips: u32,
    ) -> api::CreatedResourceInfo,
    pub luisa_compute_texture_destroy:
        unsafe extern "C" fn(device: api::Device, texture: api::Texture) -> (),
    pub luisa_compute_stream_create: unsafe extern "C" fn(
        device: api::Device,
        stream_tag: api::StreamTag,
    ) -> api::CreatedResourceInfo,
    pub luisa_compute_stream_destroy:
        unsafe extern "C" fn(device: api::Device, stream: api::Stream) -> (),
    pub luisa_compute_stream_synchronize:
        unsafe extern "C" fn(device: api::Device, stream: api::Stream) -> (),
    pub luisa_compute_stream_dispatch: unsafe extern "C" fn(
        device: api::Device,
        stream: api::Stream,
        cmd_list: api::CommandList,
        callback: api::DispatchCallback,
        callback_ctx: *mut u8,
    ) -> (),
    pub luisa_compute_shader_create: unsafe extern "C" fn(
        device: api::Device,
        func: api::KernelModule,
        option: &api::ShaderOption,
    ) -> api::CreatedShaderInfo,
    pub luisa_compute_shader_destroy:
        unsafe extern "C" fn(device: api::Device, shader: api::Shader) -> (),
    pub luisa_compute_event_create:
        unsafe extern "C" fn(device: api::Device) -> api::CreatedResourceInfo,
    pub luisa_compute_event_destroy:
        unsafe extern "C" fn(device: api::Device, event: api::Event) -> (),
    pub luisa_compute_event_signal:
        unsafe extern "C" fn(device: api::Device, event: api::Event, stream: api::Stream) -> (),
    pub luisa_compute_event_wait:
        unsafe extern "C" fn(device: api::Device, event: api::Event, stream: api::Stream) -> (),
    pub luisa_compute_event_synchronize:
        unsafe extern "C" fn(device: api::Device, event: api::Event) -> (),
    pub luisa_compute_bindless_array_create:
        unsafe extern "C" fn(device: api::Device, n: usize) -> api::CreatedResourceInfo,
    pub luisa_compute_bindless_array_destroy:
        unsafe extern "C" fn(device: api::Device, array: api::BindlessArray) -> (),
    pub luisa_compute_mesh_create: unsafe extern "C" fn(
        device: api::Device,
        option: &api::AccelOption,
    ) -> api::CreatedResourceInfo,
    pub luisa_compute_mesh_destroy:
        unsafe extern "C" fn(device: api::Device, mesh: api::Mesh) -> (),
    pub luisa_compute_accel_create: unsafe extern "C" fn(
        device: api::Device,
        option: &api::AccelOption,
    ) -> api::CreatedResourceInfo,
    pub luisa_compute_accel_destroy:
        unsafe extern "C" fn(device: api::Device, accel: api::Accel) -> (),
    pub luisa_compute_device_query: unsafe extern "C" fn(
        device: api::Device,
        query: *const c_char,
        result: *mut c_char,
        maxlen: usize,
    ) -> usize,
    pub luisa_compute_swapchain_create: unsafe extern "C" fn(
        device: api::Device,
        window_handle: u64,
        stream_handle: api::Stream,
        width: u32,
        height: u32,
        allow_hdr: bool,
        vsync: bool,
        back_buffer_size: u32,
    ) -> api::CreatedSwapchainInfo,
    pub luisa_compute_swapchain_destroy:
        unsafe extern "C" fn(device: api::Device, swapchain: api::Swapchain) -> (),
    pub luisa_compute_swapchain_present: unsafe extern "C" fn(
        device: api::Device,
        stream: api::Stream,
        swapchain: api::Swapchain,
        image: api::Texture,
    ) -> (),
    pub luisa_compute_pixel_format_to_storage:
        unsafe extern "C" fn(format: api::PixelFormat) -> api::PixelStorage,
    pub luisa_compute_set_log_level_verbose: unsafe extern "C" fn() -> (),
    pub luisa_compute_set_log_level_info: unsafe extern "C" fn() -> (),
    pub luisa_compute_set_log_level_warning: unsafe extern "C" fn() -> (),
    pub luisa_compute_set_log_level_error: unsafe extern "C" fn() -> (),
    pub luisa_compute_log_verbose: unsafe extern "C" fn(msg: *const c_char) -> (),
    pub luisa_compute_log_info: unsafe extern "C" fn(msg: *const c_char) -> (),
    pub luisa_compute_log_warning: unsafe extern "C" fn(msg: *const c_char) -> (),
    pub luisa_compute_log_error: unsafe extern "C" fn(msg: *const c_char) -> (),
}
impl Binding {
    pub unsafe fn new(lib_path: &Path) -> Result<Self, libloading::Error> {
        let lib = libloading::Library::new(lib_path)?;
        let luisa_compute_set_logger_callback = *lib.get(b"luisa_compute_set_logger_callback")?;
        let luisa_compute_free_c_string = *lib.get(b"luisa_compute_free_c_string")?;
        let luisa_compute_context_create = *lib.get(b"luisa_compute_context_create")?;
        let luisa_compute_context_destroy = *lib.get(b"luisa_compute_context_destroy")?;
        let luisa_compute_device_create = *lib.get(b"luisa_compute_device_create")?;
        let luisa_compute_device_destroy = *lib.get(b"luisa_compute_device_destroy")?;
        let luisa_compute_device_retain = *lib.get(b"luisa_compute_device_retain")?;
        let luisa_compute_device_release = *lib.get(b"luisa_compute_device_release")?;
        let luisa_compute_buffer_create = *lib.get(b"luisa_compute_buffer_create")?;
        let luisa_compute_buffer_destroy = *lib.get(b"luisa_compute_buffer_destroy")?;
        let luisa_compute_texture_create = *lib.get(b"luisa_compute_texture_create")?;
        let luisa_compute_texture_destroy = *lib.get(b"luisa_compute_texture_destroy")?;
        let luisa_compute_stream_create = *lib.get(b"luisa_compute_stream_create")?;
        let luisa_compute_stream_destroy = *lib.get(b"luisa_compute_stream_destroy")?;
        let luisa_compute_stream_synchronize = *lib.get(b"luisa_compute_stream_synchronize")?;
        let luisa_compute_stream_dispatch = *lib.get(b"luisa_compute_stream_dispatch")?;
        let luisa_compute_shader_create = *lib.get(b"luisa_compute_shader_create")?;
        let luisa_compute_shader_destroy = *lib.get(b"luisa_compute_shader_destroy")?;
        let luisa_compute_event_create = *lib.get(b"luisa_compute_event_create")?;
        let luisa_compute_event_destroy = *lib.get(b"luisa_compute_event_destroy")?;
        let luisa_compute_event_signal = *lib.get(b"luisa_compute_event_signal")?;
        let luisa_compute_event_wait = *lib.get(b"luisa_compute_event_wait")?;
        let luisa_compute_event_synchronize = *lib.get(b"luisa_compute_event_synchronize")?;
        let luisa_compute_bindless_array_create =
            *lib.get(b"luisa_compute_bindless_array_create")?;
        let luisa_compute_bindless_array_destroy =
            *lib.get(b"luisa_compute_bindless_array_destroy")?;
        let luisa_compute_mesh_create = *lib.get(b"luisa_compute_mesh_create")?;
        let luisa_compute_mesh_destroy = *lib.get(b"luisa_compute_mesh_destroy")?;
        let luisa_compute_accel_create = *lib.get(b"luisa_compute_accel_create")?;
        let luisa_compute_accel_destroy = *lib.get(b"luisa_compute_accel_destroy")?;
        let luisa_compute_device_query = *lib.get(b"luisa_compute_device_query")?;
        let luisa_compute_swapchain_create = *lib.get(b"luisa_compute_swapchain_create")?;
        let luisa_compute_swapchain_destroy = *lib.get(b"luisa_compute_swapchain_destroy")?;
        let luisa_compute_swapchain_present = *lib.get(b"luisa_compute_swapchain_present")?;
        let luisa_compute_pixel_format_to_storage =
            *lib.get(b"luisa_compute_pixel_format_to_storage")?;
        let luisa_compute_set_log_level_verbose =
            *lib.get(b"luisa_compute_set_log_level_verbose")?;
        let luisa_compute_set_log_level_info = *lib.get(b"luisa_compute_set_log_level_info")?;
        let luisa_compute_set_log_level_warning =
            *lib.get(b"luisa_compute_set_log_level_warning")?;
        let luisa_compute_set_log_level_error = *lib.get(b"luisa_compute_set_log_level_error")?;
        let luisa_compute_log_verbose = *lib.get(b"luisa_compute_log_verbose")?;
        let luisa_compute_log_info = *lib.get(b"luisa_compute_log_info")?;
        let luisa_compute_log_warning = *lib.get(b"luisa_compute_log_warning")?;
        let luisa_compute_log_error = *lib.get(b"luisa_compute_log_error")?;
        Ok(Self {
            lib,
            luisa_compute_set_logger_callback,
            luisa_compute_free_c_string,
            luisa_compute_context_create,
            luisa_compute_context_destroy,
            luisa_compute_device_create,
            luisa_compute_device_destroy,
            luisa_compute_device_retain,
            luisa_compute_device_release,
            luisa_compute_buffer_create,
            luisa_compute_buffer_destroy,
            luisa_compute_texture_create,
            luisa_compute_texture_destroy,
            luisa_compute_stream_create,
            luisa_compute_stream_destroy,
            luisa_compute_stream_synchronize,
            luisa_compute_stream_dispatch,
            luisa_compute_shader_create,
            luisa_compute_shader_destroy,
            luisa_compute_event_create,
            luisa_compute_event_destroy,
            luisa_compute_event_signal,
            luisa_compute_event_wait,
            luisa_compute_event_synchronize,
            luisa_compute_bindless_array_create,
            luisa_compute_bindless_array_destroy,
            luisa_compute_mesh_create,
            luisa_compute_mesh_destroy,
            luisa_compute_accel_create,
            luisa_compute_accel_destroy,
            luisa_compute_device_query,
            luisa_compute_swapchain_create,
            luisa_compute_swapchain_destroy,
            luisa_compute_swapchain_present,
            luisa_compute_pixel_format_to_storage,
            luisa_compute_set_log_level_verbose,
            luisa_compute_set_log_level_info,
            luisa_compute_set_log_level_warning,
            luisa_compute_set_log_level_error,
            luisa_compute_log_verbose,
            luisa_compute_log_info,
            luisa_compute_log_warning,
            luisa_compute_log_error,
        })
    }
}
