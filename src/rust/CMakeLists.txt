corrosion_import_crate(MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml)

add_library(luisa-compute-rust-meta INTERFACE)
target_include_directories(luisa-compute-rust-meta INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/)