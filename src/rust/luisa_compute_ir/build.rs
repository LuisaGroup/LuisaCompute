use std::env;

use cbindgen::Config;

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
        },
        Err(_) => {}
    }
    let path = format!("{}/../../../include/luisa/rust/ir.hpp", crate_dir);
    cbindgen::Builder::new()
        .with_config(Config::from_file("cpp.toml").unwrap())
        .with_crate(&crate_dir)
        .with_language(cbindgen::Language::Cxx)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(path);

    // cbindgen::Builder::new()
    //     .with_config(Config::from_file("c.toml").unwrap())
    //     .with_crate(&crate_dir)
    //     .with_language(cbindgen::Language::C)
    //     .with_item_prefix("LC")
    //     .generate()
    //     .expect("Unable to generate bindings")
    //     .write_to_file("bindings.h");
}
