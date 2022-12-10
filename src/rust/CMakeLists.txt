corrosion_import_crate(MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml)
if (WIN32 AND NOT MINGW AND NOT MSYS)
    corrosion_set_linker(luisa_compute_ir link.exe)
    corrosion_set_linker(luisa_compute_api_types link.exe)
endif ()

add_library(luisa-compute-rust-meta INTERFACE)
target_include_directories(luisa-compute-rust-meta INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(luisa-compute-rust-meta INTERFACE luisa_compute_ir luisa_compute_api_types)
