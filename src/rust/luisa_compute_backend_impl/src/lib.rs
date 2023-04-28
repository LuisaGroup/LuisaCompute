#[cfg(feature = "remote")]
mod remote;
#[cfg(feature = "cpu")]
mod rust;

use luisa_compute_backend::{RUSTC_CHANNEL, RUSTC_DATE, RUSTC_VERSION};
pub(crate) use luisa_compute_backend::Result;
pub(crate) use luisa_compute_backend::{Backend, BackendError, BackendErrorKind};
use luisa_compute_backend::{RustcInfo, SwapChainForCpuContext};
pub(crate) use luisa_compute_ir::ir;
use std::sync::Arc;

#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn luisa_compute_rustc_info() -> RustcInfo {
    RustcInfo {
        version: RUSTC_VERSION,
        channel: RUSTC_CHANNEL,
        date: RUSTC_DATE,
    }
}
#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn luisa_compute_create_device_rust_interface(
    device: &str,
) -> Result<Arc<dyn Backend>> {
    match device {
        "cpu" => {
            #[cfg(feature = "cpu")]
            {
                Ok(rust::RustBackend::new())
            }
            #[cfg(not(feature = "cpu"))]
            {
                Err(BackendError {
                    kind: BackendErrorKind::BackendNotFound,
                    message:
                        "cpu backend is not enabled. Try to recompile with cpu backend enabled"
                            .to_string(),
                })
            }
        }
        "remote" => {
            #[cfg(feature = "remote")]
            {
                // Ok(remote::RemoteBackend::new())
                todo!()
            }
            #[cfg(not(feature = "remote"))]
            {
                Err(BackendError{
                    kind:BackendErrorKind::BackendNotFound,
                    message:"remote backend is not enabled. Try to recompile with remote backend enabled".to_string()
                })
            }
        }
        _ => Err(BackendError {
            kind: BackendErrorKind::BackendNotFound,
            message: format!("backend {} is not defined in this shared library", device),
        }),
    }
}
