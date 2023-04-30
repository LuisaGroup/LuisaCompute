#[cfg(feature = "remote")]
mod remote;
#[cfg(feature = "cpu")]
mod rust;

use crate::rust::RustBackend;
use libloading::Library;
use log::{Level, LevelFilter, Metadata, Record, SetLoggerError};
use luisa_compute_api_types as api;
use luisa_compute_api_types::DeviceInterface;
use luisa_compute_backend::create_device_interface;
pub(crate) use luisa_compute_backend::Result;
pub(crate) use luisa_compute_backend::{Backend, BackendError, BackendErrorKind};
pub(crate) use luisa_compute_ir::ir;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        let cb = unsafe { LOGGER_CALLBACK.unwrap() };
        let target = record.target().to_string();
        let target = CString::new(target).unwrap();
        let level = match record.level() {
            Level::Error => "E",
            Level::Warn => "W",
            Level::Info => "I",
            Level::Debug => "D",
            Level::Trace => "T",
        };
        let level = CString::new(level).unwrap();
        let message = CString::new(record.args().to_string()).unwrap();
        let msg = api::LoggerMessage {
            target: target.as_ptr(),
            level: level.as_ptr(),
            message: message.as_ptr(),
        };
        unsafe {
            cb(msg);
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;
static mut LOGGER_CALLBACK: Option<unsafe extern "C" fn(api::LoggerMessage)> = None;
fn init() {
    log::set_logger(&LOGGER)
        .map(|()| log::set_max_level(LevelFilter::Trace))
        .unwrap();
}
extern "C" fn free_string(ptr: *mut c_char) {
    unsafe {
        if !ptr.is_null() {
            drop(CString::from_raw(ptr));
        }
    }
}
static INIT_LOGGER: std::sync::Once = std::sync::Once::new();
extern "C" fn set_logger_callback(cb: unsafe extern "C" fn(api::LoggerMessage)) {
    INIT_LOGGER.call_once(|| {
        init();
        unsafe {
            LOGGER_CALLBACK = Some(cb);
        }
    });
}
struct Context {
    path: PathBuf,
}
extern "C" fn create_context(path: *const c_char) -> api::Context {
    let path = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    let path = PathBuf::from(path);
    api::Context(Box::into_raw(Box::new(Context { path })) as u64)
}
extern "C" fn destroy_context(ctx: api::Context) {
    unsafe {
        drop(Box::from_raw(ctx.0 as *mut Context));
    }
}

unsafe extern "C" fn create_device(
    ctx: api::Context,
    device: *const c_char,
    config: *const c_char,
) -> DeviceInterface {
    let device = CStr::from_ptr(device).to_str().unwrap();
    let config = CStr::from_ptr(config).to_str().unwrap();
    let ctx = &*(ctx.0 as *const Context);
    match device {
        "cpu" => {
            #[cfg(feature = "cpu")]
            {
                let lib_path = &ctx.path;
                let swapchain_dll = if cfg!(target_os = "windows") {
                    "lc-vulkan-swapchain.dll"
                } else if cfg!(target_os = "linux") {
                    "liblc-vulkan-swapchain.so"
                } else {
                    todo!()
                };
                let swapchain = lib_path.join(swapchain_dll);
                let sw = SwapChainForCpuContext::new(swapchain_dll).unwrap();
                let swapchain = SwapChainForCpuContext::new(swapchain)
                    .map_err(|e| {
                        log::warn!("failed to load swapchain: {}", e);
                        e
                    })
                    .ok()
                    .map(|x| Arc::new(x));
                let device = RustBackend::new();
                if let Some(swapchain) = swapchain {
                    device.set_swapchain_contex(swapchain);
                }
                create_device_interface(device)
            }
            #[cfg(not(feature = "cpu"))]
            {
                panic!("cpu device is not enabled")
            }
        }
        "remote" => {
            #[cfg(feature = "remote")]
            {
                // let device = remote::RemoteBackend::new(config);
                // create_device_interface(device)
                todo!()
            }
            #[cfg(not(feature = "remote"))]
            {
                panic!("remote device is not enabled")
            }
        }
        _ => panic!("unknown device {}", device),
    }
}
#[no_mangle]
pub extern "C" fn luisa_compute_lib_interface() -> api::LibInterface {
    api::LibInterface {
        inner: std::ptr::null_mut(),
        set_logger_callback,
        create_context,
        destroy_context,
        create_device,
        free_string,
    }
}
