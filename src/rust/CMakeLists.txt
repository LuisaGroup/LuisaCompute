corrosion_import_crate(MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml)
if (WIN32 AND NOT MINGW AND NOT MSYS)
    corrosion_set_linker(luisa_compute_ir link.exe)
    corrosion_set_linker(luisa_compute_api_types link.exe)
endif ()

add_library(luisa-compute-rust-meta INTERFACE)
target_include_directories(luisa-compute-rust-meta INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(luisa-compute-rust-meta INTERFACE luisa_compute_ir luisa_compute_api_types)

function(luisa_compute_install_rust target)
    install(TARGETS luisa_compute_${target} EXPORT LuisaComputeTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY luisa_compute_${target}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LuisaCompute/rust
            FILES_MATCHING REGEX ".*\\.(h|hpp)$")
endfunction()

luisa_compute_install_rust(ir)
luisa_compute_install_rust(api_types)
luisa_compute_install(rust-meta)
