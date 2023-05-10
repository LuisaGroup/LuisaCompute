#![allow(unused_unsafe)]

use std::ffi::CStr;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{Backend, BackendError, BackendProvider, Context, Interface};
use api::StreamTag;
use libc::c_void;
use luisa_compute_api_types as api;
use luisa_compute_api_types::BackendErrorKind;
use luisa_compute_ir::{
    ir::{KernelModule, Type},
    CArc,
};
use parking_lot::Mutex;
use std::sync::Once;

static INIT_CPP: Once = Once::new();
static mut OLD_SIGABRT_HANDLER: libc::sighandler_t = 0;
static CPP_MUTEX: Mutex<()> = Mutex::new(());

fn restore_signal_handler() {
    unsafe {
        libc::signal(libc::SIGABRT, OLD_SIGABRT_HANDLER);
        libc::signal(libc::SIGSEGV, OLD_SIGABRT_HANDLER);
    }
}

pub(crate) fn _signal_handler(signal: libc::c_int) {
    if signal == libc::SIGABRT {
        restore_signal_handler();
        panic!("std::abort() called inside LuisaCompute");
    }
    if signal == libc::SIGSEGV {
        restore_signal_handler();
        panic!("segfault inside LuisaCompute");
    }
    #[cfg(target_os = "linux")]
    if signal == libc::SIGBUS {
        restore_signal_handler();
        panic!("bus error inside LuisaCompute");
    }
}
#[macro_export]
macro_rules! catch_abort {
    ($stmts:expr) => {
        unsafe {
            #[cfg(debug_assertions)]
            {
                log::trace!("catch_abort: {}", stringify!($stmts));
            }
            let _guard = CPP_MUTEX.lock();
            OLD_SIGABRT_HANDLER =
                libc::signal(libc::SIGABRT, _signal_handler as libc::sighandler_t);
            OLD_SIGABRT_HANDLER =
                libc::signal(libc::SIGSEGV, _signal_handler as libc::sighandler_t);
            let ret = $stmts;
            restore_signal_handler();
            ret
        }
    };
}

pub fn init_cpp<P: AsRef<Path>>(_bin_path: P) {
    INIT_CPP.call_once(|| unsafe {});
}

pub struct ProxyBackend {
    pub(crate) interface: Arc<Interface>,
    pub(crate) device: api::DeviceInterface,
}

impl ProxyBackend {
    pub(crate) fn new(provider: &BackendProvider, device: &str, config: serde_json::Value) -> Self {
        let interface = &provider.interface;
        let device_c_str = std::ffi::CString::new(device).unwrap();
        let device_c_str = device_c_str.as_ptr();
        let config = std::ffi::CString::new(serde_json::to_string(&config).unwrap()).unwrap();
        let device = unsafe {
            (interface.inner.create_device)(provider.context, device_c_str, config.as_ptr())
        };
        Self {
            device,
            interface: interface.clone(),
        }
    }
}

unsafe fn map<T>(a: api::Result<T>) -> crate::Result<T> {
    match a {
        api::Result::Ok(a) => Ok(a),
        api::Result::Err(a) => Err(crate::BackendError {
            kind: match a.kind {
                api::BackendErrorKind::BackendNotFound => crate::BackendErrorKind::BackendNotFound,
                api::BackendErrorKind::KernelExecution => crate::BackendErrorKind::KernelExecution,
                api::BackendErrorKind::KernelCompilation => {
                    crate::BackendErrorKind::KernelCompilation
                }
                api::BackendErrorKind::Network => crate::BackendErrorKind::Network,
                api::BackendErrorKind::Other => crate::BackendErrorKind::Other,
            },
            message: CStr::from_ptr(a.message).to_str().unwrap().to_string(),
        }),
    }
}

impl Backend for ProxyBackend {
    #[inline]
    fn create_buffer(
        &self,
        ty: &CArc<Type>,
        count: usize,
    ) -> crate::Result<api::CreatedBufferInfo> {
        catch_abort!({
            map((self.device.create_buffer)(
                self.device.device,
                ty as *const _ as *const c_void,
                count,
            ))
        })
    }
    #[inline]
    fn destroy_buffer(&self, buffer: api::Buffer) {
        catch_abort!({
            (self.device.destroy_buffer)(self.device.device, buffer);
        })
    }
    #[inline]
    fn create_texture(
        &self,
        format: api::PixelFormat,
        dimension: u32,
        width: u32,
        height: u32,
        depth: u32,
        mipmap_levels: u32,
    ) -> crate::Result<api::CreatedResourceInfo> {
        catch_abort!({
            map((self.device.create_texture)(
                self.device.device,
                std::mem::transmute(format),
                dimension,
                width,
                height,
                depth,
                mipmap_levels,
            ))
        })
    }
    #[inline]
    fn destroy_texture(&self, texture: api::Texture) {
        catch_abort!({ (self.device.destroy_texture)(self.device.device, texture,) })
    }
    #[inline]
    fn create_bindless_array(&self, size: usize) -> crate::Result<api::CreatedResourceInfo> {
        catch_abort!({
            map((self.device.create_bindless_array)(
                self.device.device,
                size,
            ))
        })
    }
    #[inline]
    fn destroy_bindless_array(&self, array: api::BindlessArray) {
        catch_abort!({ (self.device.destroy_bindless_array)(self.device.device, array,) })
    }
    #[inline]
    fn create_stream(&self, tag: StreamTag) -> crate::Result<api::CreatedResourceInfo> {
        catch_abort!({ map((self.device.create_stream)(self.device.device, tag,)) })
    }
    #[inline]
    fn destroy_stream(&self, stream: api::Stream) {
        catch_abort!({ (self.device.destroy_stream)(self.device.device, stream) })
    }
    #[inline]
    fn synchronize_stream(&self, stream: api::Stream) -> crate::Result<()> {
        catch_abort!({
            map((self.device.synchronize_stream)(self.device.device, stream)).map(|_| ())
        })
    }
    #[inline]
    fn dispatch(
        &self,
        stream: api::Stream,
        command_list: &[api::Command],
        callback: (extern "C" fn(*mut u8), *mut u8),
    ) -> crate::Result<()> {
        catch_abort!({
            map((self.device.dispatch)(
                self.device.device,
                stream,
                api::CommandList {
                    commands: command_list.as_ptr(),
                    commands_count: command_list.len(),
                },
                callback.0,
                callback.1,
            ))
            .map(|_| ())
        })
    }
    #[inline]
    fn create_swapchain(
        &self,
        window_handle: u64,
        stream_handle: api::Stream,
        width: u32,
        height: u32,
        allow_hdr: bool,
        vsync: bool,
        back_buffer_size: u32,
    ) -> crate::Result<api::CreatedSwapchainInfo> {
        catch_abort!({
            map((self.device.create_swapchain)(
                self.device.device,
                window_handle,
                stream_handle,
                width,
                height,
                allow_hdr,
                vsync,
                back_buffer_size,
            ))
        })
    }
    #[inline]
    fn destroy_swapchain(&self, swap_chain: api::Swapchain) {
        catch_abort!({ (self.device.destroy_swapchain)(self.device.device, swap_chain,) })
    }
    #[inline]
    fn present_display_in_stream(
        &self,
        stream_handle: api::Stream,
        swapchain_handle: api::Swapchain,
        image_handle: api::Texture,
    ) {
        catch_abort!({
            (self.device.present_display_in_stream)(
                self.device.device,
                stream_handle,
                swapchain_handle,
                image_handle,
            )
        })
    }
    #[inline]
    fn create_shader(
        &self,
        kernel: &KernelModule,
        option: &api::ShaderOption,
    ) -> crate::Result<api::CreatedShaderInfo> {
        catch_abort!({
            map((self.device.create_shader)(
                self.device.device,
                api::KernelModule {
                    ptr:kernel as *const _ as u64,
                },
                option,
            ))
        })
    }
    #[inline]
    fn shader_cache_dir(&self, _shader: api::Shader) -> Option<PathBuf> {
        Some(".cache".into())
    }
    #[inline]
    fn destroy_shader(&self, shader: api::Shader) {
        catch_abort!({ (self.device.destroy_shader)(self.device.device, shader) })
    }
    #[inline]
    fn create_event(&self) -> crate::Result<api::CreatedResourceInfo> {
        catch_abort!({ map((self.device.create_event)(self.device.device)) })
    }
    #[inline]
    fn destroy_event(&self, event: api::Event) {
        catch_abort!({ (self.device.destroy_event)(self.device.device, event) })
    }
    #[inline]
    fn signal_event(&self, event: api::Event, stream: api::Stream) {
        catch_abort!({ (self.device.signal_event)(self.device.device, event, stream) })
    }
    #[inline]
    fn wait_event(&self, event: api::Event, stream: api::Stream) -> crate::Result<()> {
        catch_abort!({
            map((self.device.wait_event)(self.device.device, event, stream)).map(|_| ())
        })
    }
    #[inline]
    fn synchronize_event(&self, event: api::Event) -> crate::Result<()> {
        catch_abort!({
            map((self.device.synchronize_event)(self.device.device, event)).map(|_| ())
        })
    }
    #[inline]
    fn create_mesh(&self, option: api::AccelOption) -> crate::Result<api::CreatedResourceInfo> {
        catch_abort!({ map((self.device.create_mesh)(self.device.device, &option,)) })
    }
    #[inline]
    fn create_procedural_primitive(
        &self,
        _option: api::AccelOption,
    ) -> crate::Result<api::CreatedResourceInfo> {
        todo!()
    }
    #[inline]
    fn destroy_mesh(&self, mesh: api::Mesh) {
        catch_abort!((self.device.destroy_mesh)(self.device.device, mesh))
    }
    #[inline]
    fn destroy_procedural_primitive(&self, _primitive: api::ProceduralPrimitive) {
        todo!()
    }

    #[inline]
    fn create_accel(&self, option: api::AccelOption) -> crate::Result<api::CreatedResourceInfo> {
        catch_abort!({ map((self.device.create_accel)(self.device.device, &option,)) })
    }
    #[inline]
    fn destroy_accel(&self, accel: api::Accel) {
        catch_abort!((self.device.destroy_accel)(self.device.device, accel))
    }
    #[inline]
    fn query(&self, property: &str) -> Option<String> {
        catch_abort! {{
            let property = std::ffi::CString::new(property).unwrap();
            let property = property.as_ptr();
            let result = (self.device.query)(self.device.device, property);
            if result.is_null() {
                return None;
            }
            let result_str = std::ffi::CStr::from_ptr(result as *const i8).to_str().unwrap().to_string();
            (self.interface.inner.free_string)(result);
            Some(result_str)
        }}
    }
}
