use std::env;

use cbindgen::Config;

#[path = "../write_if_different.rs"]
mod write_if_different;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    // println!("cargo:rerun-if-env-changed=LC_RS_DO_NOT_GENERATE_BINDINGS");
    println!("cargo:rerun-if-changed={}/cpp.toml", crate_dir);
    println!("cargo:rerun-if-changed={}/src", crate_dir);
    match env::var("LC_RS_DO_NOT_GENERATE_BINDINGS") {
        Ok(s) => {
            if s == "1" {
                return;
            }
        }
        Err(_) => {}
    }
    let path = format!("{}/../../../include/luisa/rust/ir.hpp", crate_dir);
    let mut content = Vec::new();
    cbindgen::Builder::new()
        .with_config(Config::from_file("cpp.toml").unwrap())
        .with_crate(&crate_dir)
        .with_language(cbindgen::Language::Cxx)
        .generate()
        .expect("Unable to generate bindings")
        .write(&mut content);
    write_if_different::write_if_different(&path, content);
}
