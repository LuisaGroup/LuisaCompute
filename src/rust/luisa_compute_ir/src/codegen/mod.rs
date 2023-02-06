use base64ct::Encoding;
use lazy_static::lazy_static;
use sha2::{Digest, Sha256};
use std::ffi::{c_char, CString};

use crate::{ir, CBoxedSlice};

pub mod generic_cpp;

pub trait CodeGen {
    fn run(module: &ir::KernelModule) -> String;
}

pub fn sha256(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s);
    let hash = hasher.finalize();
    format!("A{}", base64ct::Base64UrlUnpadded::encode_string(&hash))
}

lazy_static! {
    pub static ref CUDA_DEVICE_MATH: CString = {
        let device_math = include_str!("device_math.h");
        let prelude = include_str!("cuda_prelude.h");
        CString::new(format!("{}{}", prelude, device_math)).unwrap()
    };
    pub static ref CUDA_DEVICE_RESOURCE: CString =
        CString::new(include_str!("cuda_resource.h")).unwrap();
}

#[no_mangle]
pub extern "C" fn luisa_compute_cuda_header_device_math() -> *const c_char {
    return CUDA_DEVICE_MATH.as_ptr();
}

#[no_mangle]
pub extern "C" fn luisa_compute_cuda_header_device_resource() -> *const c_char {
    return CUDA_DEVICE_RESOURCE.as_ptr();
}
