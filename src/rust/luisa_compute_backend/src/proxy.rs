#![allow(unused_unsafe)]

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{Backend, BackendProvider, Interface};
use api::StreamTag;
use libc::c_void;
use luisa_compute_api_types as api;
use luisa_compute_ir::{
    ir::{KernelModule, Type},
    CArc,
};
use parking_lot::Mutex;
use std::sync::Once;

// This is uselss, we should remove it
#[macro_export]
macro_rules! catch_abort {
    ($stmts:expr) => {
        unsafe {
            let ret = $stmts;
            ret
        }
    };
}

pub struct ProxyBackend {
    pub(crate) interface: Arc<Interface>,
    pub(crate) device: api::DeviceInterface,
}
impl Drop for ProxyBackend {
    fn drop(&mut self) {
        catch_abort!({ (self.device.destroy_device)(self.device) })
    }
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

impl Backend for ProxyBackend {
    fn compute_warp_size(&self) -> u32 {
        catch_abort!({ (self.device.compute_warp_size)(self.device.device) })
    }
    fn native_handle(&self) -> *mut c_void {
        catch_abort!({ (self.device.native_handle)(self.device.device) })
    }
    #[inline]
    fn create_buffer(&self, ty: &CArc<Type>, count: usize) -> api::CreatedBufferInfo {
        catch_abort!({
            (self.device.create_buffer)(self.device.device, ty as *const _ as *const c_void, count)
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
        allow_simultaneous_access: bool,
    ) -> api::CreatedResourceInfo {
        catch_abort!({
            (self.device.create_texture)(
                self.device.device,
                std::mem::transmute(format),
                dimension,
                width,
                height,
                depth,
                mipmap_levels,
                allow_simultaneous_access,
            )
        })
    }
    #[inline]
    fn destroy_texture(&self, texture: api::Texture) {
        catch_abort!({ (self.device.destroy_texture)(self.device.device, texture,) })
    }
    #[inline]
    fn create_bindless_array(&self, size: usize) -> api::CreatedResourceInfo {
        catch_abort!({ (self.device.create_bindless_array)(self.device.device, size,) })
    }
    #[inline]
    fn destroy_bindless_array(&self, array: api::BindlessArray) {
        catch_abort!({ (self.device.destroy_bindless_array)(self.device.device, array,) })
    }
    #[inline]
    fn create_stream(&self, tag: StreamTag) -> api::CreatedResourceInfo {
        catch_abort!({ (self.device.create_stream)(self.device.device, tag,) })
    }
    #[inline]
    fn destroy_stream(&self, stream: api::Stream) {
        catch_abort!({ (self.device.destroy_stream)(self.device.device, stream) })
    }
    #[inline]
    fn synchronize_stream(&self, stream: api::Stream) {
        catch_abort!({ (self.device.synchronize_stream)(self.device.device, stream) })
    }
    #[inline]
    fn dispatch(
        &self,
        stream: api::Stream,
        command_list: &[api::Command],
        callback: (extern "C" fn(*mut u8), *mut u8),
    ) {
        catch_abort!({
            (self.device.dispatch)(
                self.device.device,
                stream,
                api::CommandList {
                    commands: command_list.as_ptr(),
                    commands_count: command_list.len(),
                },
                callback.0,
                callback.1,
            )
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
    ) -> api::CreatedSwapchainInfo {
        catch_abort!({
            (self.device.create_swapchain)(
                self.device.device,
                window_handle,
                stream_handle,
                width,
                height,
                allow_hdr,
                vsync,
                back_buffer_size,
            )
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
    ) -> api::CreatedShaderInfo {
        // let debug =
        //     luisa_compute_ir::ir::debug::luisa_compute_ir_dump_human_readable(&kernel.module);
        // let debug = std::ffi::CString::new(debug.as_ref()).unwrap();
        // println!("{}", debug.to_str().unwrap());
        catch_abort!({
            (self.device.create_shader)(
                self.device.device,
                api::KernelModule {
                    ptr: kernel as *const _ as u64,
                },
                option,
            )
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
    fn create_event(&self) -> api::CreatedResourceInfo {
        catch_abort!({ (self.device.create_event)(self.device.device) })
    }
    #[inline]
    fn destroy_event(&self, event: api::Event) {
        catch_abort!({ (self.device.destroy_event)(self.device.device, event) })
    }
    #[inline]
    fn signal_event(&self, event: api::Event, stream: api::Stream, value: u64) {
        catch_abort!({ (self.device.signal_event)(self.device.device, event, stream, value) })
    }
    #[inline]
    fn wait_event(&self, event: api::Event, stream: api::Stream, value: u64) {
        catch_abort!({ (self.device.wait_event)(self.device.device, event, stream, value) })
    }
    #[inline]
    fn synchronize_event(&self, event: api::Event, value: u64) {
        catch_abort!({ (self.device.synchronize_event)(self.device.device, event, value) })
    }
    #[inline]
    fn is_event_completed(&self, event: api::Event, value: u64) -> bool {
        catch_abort!({ (self.device.is_event_completed)(self.device.device, event, value) })
    }
    #[inline]
    fn create_mesh(&self, option: api::AccelOption) -> api::CreatedResourceInfo {
        catch_abort!({ (self.device.create_mesh)(self.device.device, &option,) })
    }
    #[inline]
    fn create_procedural_primitive(&self, option: api::AccelOption) -> api::CreatedResourceInfo {
        catch_abort!({ (self.device.create_procedural_primitive)(self.device.device, &option,) })
    }
    #[inline]
    fn destroy_mesh(&self, mesh: api::Mesh) {
        catch_abort!((self.device.destroy_mesh)(self.device.device, mesh))
    }
    #[inline]
    fn destroy_procedural_primitive(&self, primitive: api::ProceduralPrimitive) {
        catch_abort!((self.device.destroy_procedural_primitive)(
            self.device.device,
            primitive
        ))
    }

    #[inline]
    fn create_accel(&self, option: api::AccelOption) -> api::CreatedResourceInfo {
        catch_abort!({ (self.device.create_accel)(self.device.device, &option,) })
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
