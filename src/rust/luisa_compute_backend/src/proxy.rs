#![allow(unused_unsafe)]
use std::ffi::CStr;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{Backend, BackendError, Context, Interface, SwapChainForCpuContext};
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
}
#[macro_export]
macro_rules! catch_abort {
    ($stmts:expr) => {
        unsafe {
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
    interface: Arc<Interface>,
    device: api::Device,
}
fn default_path() -> PathBuf {
    std::env::current_exe().unwrap()
}
impl ProxyBackend {
    pub fn new(ctx: &Context, backend: &str) -> Arc<Self> {
        let backend_c_str = std::ffi::CString::new(backend).unwrap();
        let interface = ctx.interface.clone().unwrap();
        let device = catch_abort!({
            (interface.create_device)(
                ctx.context.unwrap(),
                backend_c_str.as_ptr(),
                b"{}\0".as_ptr() as *const _,
            )
        });
        Arc::new(Self { device, interface })
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
    fn create_buffer(
        &self,
        ty: &CArc<Type>,
        count: usize,
    ) -> crate::Result<api::CreatedBufferInfo> {
        catch_abort!({
            map((self.interface.create_buffer)(
                self.device,
                ty as *const _ as *const c_void,
                count,
            ))
        })
    }

    fn destroy_buffer(&self, buffer: api::Buffer) {
        catch_abort!({
            (self.interface.destroy_buffer)(self.device, buffer);
        })
    }
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
            map((self.interface.create_texture)(
                self.device,
                std::mem::transmute(format),
                dimension,
                width,
                height,
                depth,
                mipmap_levels,
            ))
        })
    }

    fn destroy_texture(&self, texture: api::Texture) {
        catch_abort!({ (self.interface.destroy_texture)(self.device, texture,) })
    }

    fn create_bindless_array(&self, size: usize) -> crate::Result<api::CreatedResourceInfo> {
        catch_abort!({
            map((self.interface.create_bindless_array)(
                self.device,
                size,
            ))
        })
    }

    fn destroy_bindless_array(&self, array: api::BindlessArray) {
        catch_abort!({ (self.interface.destroy_bindless_array)(self.device, array,) })
    }

    fn create_stream(&self, tag: StreamTag) -> crate::Result<api::CreatedResourceInfo> {
        catch_abort!({
            map((self.interface.create_stream)(
                self.device,
                tag,
            ))
        })
    }

    fn destroy_stream(&self, stream: api::Stream) {
        catch_abort!({ (self.interface.destroy_stream)(self.device, stream) })
    }

    fn synchronize_stream(&self, stream: api::Stream) -> crate::Result<()> {
        catch_abort!({
            map((self.interface.synchronize_stream)(
                self.device,
                stream,
            ))
            .map(|_| ())
        })
    }

    fn dispatch(
        &self,
        stream: api::Stream,
        command_list: &[api::Command],
        callback: (extern "C" fn(*mut u8), *mut u8),
    ) -> crate::Result<()> {
        catch_abort!({
            map((self.interface.dispatch)(
                self.device,
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

    fn create_shader(
        &self,
        kernel: CArc<KernelModule>,
        option: &api::ShaderOption,
    ) -> crate::Result<api::CreatedShaderInfo> {
        catch_abort!({
            map((self.interface.create_shader)(
                self.device,
                api::KernelModule {
                    ptr: CArc::as_ptr(&kernel) as u64,
                },
                option,
            ))
        })
    }

    fn destroy_shader(&self, shader: api::Shader) {
        catch_abort!({ (self.interface.destroy_shader)(self.device, shader) })
    }

    fn create_event(&self) -> crate::Result<api::CreatedResourceInfo> {
        catch_abort!({ map((self.interface.create_event)(self.device)) })
    }

    fn destroy_event(&self, event: api::Event) {
        catch_abort!({ (self.interface.destroy_event)(self.device, event) })
    }

    fn signal_event(&self, event: api::Event, stream: api::Stream) {
        catch_abort!({ (self.interface.signal_event)(self.device, event, stream) })
    }

    fn wait_event(&self, event: api::Event, stream: api::Stream) -> crate::Result<()> {
        catch_abort!({
            map((self.interface.wait_event)(
                self.device,
                event,
                stream,
            ))
            .map(|_| ())
        })
    }

    fn synchronize_event(&self, event: api::Event) -> crate::Result<()> {
        catch_abort!({
            map((self.interface.synchronize_event)(
                self.device,
                event,
            ))
            .map(|_| ())
        })
    }

    fn create_mesh(&self, option: api::AccelOption) -> crate::Result<api::CreatedResourceInfo> {
        catch_abort!({ map((self.interface.create_mesh)(self.device, &option,)) })
    }

    fn destroy_mesh(&self, mesh: api::Mesh) {
        catch_abort!((self.interface.destroy_mesh)(self.device, mesh))
    }

    fn create_accel(&self, option: api::AccelOption) -> crate::Result<api::CreatedResourceInfo> {
        catch_abort!({ map((self.interface.create_accel)(self.device, &option,)) })
    }

    fn destroy_accel(&self, accel: api::Accel) {
        catch_abort!((self.interface.destroy_accel)(self.device, accel))
    }

    fn query(&self, property: &str) -> Option<String> {
        catch_abort! {{
            let property = std::ffi::CString::new(property).unwrap();
            let property = property.as_ptr();
            let result = (self.interface.query)(self.device, property);
            if result.is_null() {
                return None;
            }
            let result_str = std::ffi::CStr::from_ptr(result as *const i8).to_str().unwrap().to_string();
            (self.interface.free_string)(result);
            Some(result_str)
        }}
    }

    fn shader_cache_dir(&self, _shader: api::Shader) -> Option<PathBuf> {
        todo!()
    }

    fn create_procedural_primitive(
        &self,
        _option: api::AccelOption,
    ) -> crate::Result<api::CreatedResourceInfo> {
        todo!()
    }

    fn destroy_procedural_primitive(&self, _primitive: api::ProceduralPrimitive) {
        todo!()
    }

    unsafe fn set_swapchain_contex(&self, _ctx: Arc<crate::SwapChainForCpuContext>) {}

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
            map((self.interface.create_swapchain)(
                self.device,
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

    fn destroy_swapchain(&self, swap_chain: api::Swapchain) {
        catch_abort!({ (self.interface.destroy_swapchain)(self.device, swap_chain,) })
    }

    fn present_display_in_stream(
        &self,
        stream_handle: api::Stream,
        swapchain_handle: api::Swapchain,
        image_handle: api::Texture,
    ) {
        catch_abort!({
            (self.interface.present_display_in_stream)(
                self.device,
                stream_handle,
                swapchain_handle,
                image_handle,
            )
        })
    }
}
