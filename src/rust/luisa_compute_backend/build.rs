use std::env;

use cbindgen::Config;
use version_check as rustc;
fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    cbindgen::Builder::new()
        .with_config(Config::from_file("cpp.toml").unwrap())
        .with_crate(&crate_dir)
        .with_language(cbindgen::Language::Cxx)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("bindings.hpp");

    let version = rustc::Version::read().unwrap();
    let channel = rustc::Channel::read().unwrap();
    let date = rustc::Date::read().unwrap();
    let config = format!("pub const RUSTC_VERSION: &str = \"{}\";", version);
    let config = format!("{}\npub const RUSTC_CHANNEL: &str = \"{}\";", config, channel);
    let config = format!("{}\npub const RUSTC_DATE: &str = \"{}\";", config, date);
    std::fs::write("src/rustc_version.rs", config).unwrap();
}
