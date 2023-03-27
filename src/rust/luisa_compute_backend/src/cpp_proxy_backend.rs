#![allow(unused_unsafe)]
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{binding::Binding, Backend, Context, SwapChainForCpuContext};
use api::StreamTag;
use libc::c_void;
use luisa_compute_api_types as api;
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

pub struct CppProxyBackend {
    binding: Arc<Binding>,
    device: api::Device,
}
fn default_path() -> PathBuf {
    std::env::current_exe().unwrap()
}
impl CppProxyBackend {
    pub fn new(ctx: &Context, backend: &str) -> Arc<Self> {
        let backend_c_str = std::ffi::CString::new(backend).unwrap();
        let binding = ctx.binding.clone().unwrap();
        let device = catch_abort!({
            (binding.luisa_compute_device_create)(
                ctx.context.unwrap(),
                backend_c_str.as_ptr(),
                b"{}\0".as_ptr() as *const _,
            )
        });
        Arc::new(Self { device, binding })
    }
}
impl Backend for CppProxyBackend {
    fn create_buffer(
        &self,
        ty: &CArc<Type>,
        count: usize,
    ) -> crate::Result<api::CreatedBufferInfo> {
        let buffer = catch_abort!({
            (self.binding.luisa_compute_buffer_create)(
                self.device,
                ty as *const _ as *const c_void,
                count,
            )
        });
        Ok(buffer)
    }

    fn destroy_buffer(&self, buffer: api::Buffer) {
        catch_abort!({
            (self.binding.luisa_compute_buffer_destroy)(self.device, buffer);
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
        let texture = catch_abort!({
            (self.binding.luisa_compute_texture_create)(
                self.device,
                std::mem::transmute(format),
                dimension,
                width,
                height,
                depth,
                mipmap_levels,
            )
        });
        Ok(texture)
    }

    fn destroy_texture(&self, texture: api::Texture) {
        catch_abort!({ (self.binding.luisa_compute_texture_destroy)(self.device, texture,) })
    }

    fn create_bindless_array(&self, size: usize) -> crate::Result<api::CreatedResourceInfo> {
        let array =
            catch_abort!({ (self.binding.luisa_compute_bindless_array_create)(self.device, size) });
        Ok(array)
    }

    fn destroy_bindless_array(&self, array: api::BindlessArray) {
        catch_abort!({ (self.binding.luisa_compute_bindless_array_destroy)(self.device, array,) })
    }

    fn create_stream(&self, tag: StreamTag) -> crate::Result<api::CreatedResourceInfo> {
        unsafe {
            let stream =
                catch_abort!({ (self.binding.luisa_compute_stream_create)(self.device, tag) });
            Ok(stream)
        }
    }

    fn destroy_stream(&self, stream: api::Stream) {
        catch_abort!({ (self.binding.luisa_compute_stream_destroy)(self.device, stream) })
    }

    fn synchronize_stream(&self, stream: api::Stream) -> crate::Result<()> {
        catch_abort!({ (self.binding.luisa_compute_stream_synchronize)(self.device, stream,) });
        Ok(())
    }

    fn dispatch(
        &self,
        stream: api::Stream,
        command_list: &[api::Command],
        callback: (extern "C" fn(*mut u8), *mut u8),
    ) -> crate::Result<()> {
        catch_abort!({
            (self.binding.luisa_compute_stream_dispatch)(
                self.device,
                stream,
                api::CommandList {
                    commands: command_list.as_ptr(),
                    commands_count: command_list.len(),
                },
                callback.0,
                callback.1,
            );
        });
        Ok(())
    }

    fn create_shader(
        &self,
        kernel: CArc<KernelModule>,
        option: &api::ShaderOption,
    ) -> crate::Result<api::CreatedShaderInfo> {
        //  let debug =
        //     luisa_compute_ir::ir::debug::luisa_compute_ir_dump_human_readable(&kernel.module);
        // let debug = std::ffi::CString::new(debug.as_ref()).unwrap();
        // println!("{}", debug.to_str().unwrap());

        Ok(catch_abort!({
            (self.binding.luisa_compute_shader_create)(
                self.device,
                api::KernelModule {
                    ptr: CArc::as_ptr(&kernel) as u64,
                },
                option,
            )
        }))
    }

    fn destroy_shader(&self, shader: api::Shader) {
        catch_abort!({ (self.binding.luisa_compute_shader_destroy)(self.device, shader) })
    }

    fn create_event(&self) -> crate::Result<api::CreatedResourceInfo> {
        unsafe {
            let event = catch_abort!({ (self.binding.luisa_compute_event_create)(self.device) });
            Ok(event)
        }
    }

    fn destroy_event(&self, event: api::Event) {
        catch_abort!({ (self.binding.luisa_compute_event_destroy)(self.device, event) })
    }

    fn signal_event(&self, event: api::Event, stream: api::Stream) {
        catch_abort!({ (self.binding.luisa_compute_event_signal)(self.device, event, stream) })
    }

    fn wait_event(&self, event: api::Event, stream: api::Stream) -> crate::Result<()> {
        catch_abort!({
            (self.binding.luisa_compute_event_wait)(self.device, event, stream);
        });
        Ok(())
    }

    fn synchronize_event(&self, event: api::Event) -> crate::Result<()> {
        catch_abort!({
            (self.binding.luisa_compute_event_synchronize)(self.device, event);
        });
        Ok(())
    }

    fn create_mesh(&self, option: api::AccelOption) -> crate::Result<api::CreatedResourceInfo> {
        unsafe {
            let mesh =
                catch_abort!({ (self.binding.luisa_compute_mesh_create)(self.device, &option) });
            Ok(mesh)
        }
    }

    fn destroy_mesh(&self, mesh: api::Mesh) {
        catch_abort!((self.binding.luisa_compute_mesh_destroy)(self.device, mesh))
    }

    fn create_accel(&self, option: api::AccelOption) -> crate::Result<api::CreatedResourceInfo> {
        unsafe {
            let mesh =
                catch_abort!({ (self.binding.luisa_compute_accel_create)(self.device, &option) });
            Ok(std::mem::transmute(mesh))
        }
    }

    fn destroy_accel(&self, accel: api::Accel) {
        catch_abort!((self.binding.luisa_compute_accel_destroy)(
            self.device,
            accel
        ))
    }

    fn query(&self, property: &str) -> Option<String> {
        catch_abort! {{
            let property = std::ffi::CString::new(property).unwrap();
            let property = property.as_ptr();
            let str_buf = vec![0u8; 1024];
            let result_len = (self.binding.luisa_compute_device_query)(self.device, property, str_buf.as_ptr() as *mut i8, str_buf.len());
            if result_len > 0 {
                let result_str = std::ffi::CStr::from_ptr(str_buf.as_ptr() as *const i8).to_str().unwrap().to_string();
                Some(result_str)
            } else {
                None
            }
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
            let swap_chain = (self.binding.luisa_compute_swapchain_create)(
                self.device,
                window_handle,
                stream_handle,
                width,
                height,
                allow_hdr,
                vsync,
                back_buffer_size,
            );
            Ok(std::mem::transmute(swap_chain))
        })
    }

    fn destroy_swapchain(&self, swap_chain: api::Swapchain) {
        catch_abort!({ (self.binding.luisa_compute_swapchain_destroy)(self.device, swap_chain,) })
    }

    fn present_display_in_stream(
        &self,
        stream_handle: api::Stream,
        swapchain_handle: api::Swapchain,
        image_handle: api::Texture,
    ) {
        catch_abort!({
            (self.binding.luisa_compute_swapchain_present)(
                self.device,
                stream_handle,
                swapchain_handle,
                image_handle,
            )
        })
    }
}
