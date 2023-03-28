use base64ct::Encoding;
use lazy_static::lazy_static;
use sha2::{Digest, Sha256};
use std::ffi::{c_char, CString};

use crate::{ir};

pub mod cpp;

pub trait CodeGen {
    fn run(module: &ir::KernelModule) -> String;
}

pub fn sha256(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s);
    let hash = hasher.finalize();
    format!("A{}", base64ct::Base64UrlUnpadded::encode_string(&hash))
}
