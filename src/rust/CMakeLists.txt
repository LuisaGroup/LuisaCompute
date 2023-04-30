if (LUISA_COMPUTE_ENABLE_RUST)

    corrosion_import_crate(
            MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
            NO_DEFAULT_FEATURES)

    set(LUISA_CARGO_PROFILE "$<IF:$<CONFIG:Debug>,debug,release>")
    set(LUISA_RUST_OUTPUT_DIR
            "$<TARGET_FILE_DIR:luisa-compute-core>/../cargo/build/${Rust_CARGO_TARGET_CACHED}/${LUISA_CARGO_PROFILE}")
    corrosion_set_env_vars(luisa_compute_backend_impl
            "EMBREE_DLL_OUT_DIR=${LUISA_RUST_OUTPUT_DIR}/backend_support")

    if (LUISA_COMPUTE_ENABLE_CPU)
        corrosion_set_features(luisa_compute_backend_impl FEATURES cpu)
    endif ()

    if (LUISA_COMPUTE_ENABLE_REMOTE)
        corrosion_set_features(luisa_compute_backend_impl FEATURES remote)
    endif ()

    add_library(luisa-compute-rust-meta INTERFACE)
    target_include_directories(luisa-compute-rust-meta INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_link_libraries(luisa-compute-rust-meta INTERFACE luisa_compute_ir luisa_compute_api_types)
    target_compile_definitions(luisa-compute-rust-meta INTERFACE LUISA_ENABLE_RUST)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(luisa-compute-rust-meta INTERFACE -Wno-return-type-c-linkage)
    endif ()

    if (LUISA_COMPUTE_ENABLE_CPU)

        # FIXME: using the internal target is not ideal
        add_custom_target(luisa-compute-rust-copy DEPENDS
                _cargo-build_luisa_compute_backend_impl)

        add_custom_command(TARGET luisa-compute-rust-copy
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${LUISA_RUST_OUTPUT_DIR}/backend_support/"
                "$<TARGET_FILE_DIR:luisa-compute-core>/")

        install(DIRECTORY "${LUISA_RUST_OUTPUT_DIR}/backend_support/"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/"
                FILES_MATCHING REGEX ".*\\.(dll|so|dylib)$")
    endif ()

    function(luisa_compute_install_rust target)
        # FIXME: corrosion does not correctly installs the rust library so we have to manually install it here
        # corrosion_install(TARGETS luisa_compute_${target}
        #         RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        #         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        #         ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        #         PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
        install(TARGETS luisa_compute_${target} EXPORT LuisaComputeTargets)
        install(DIRECTORY luisa_compute_${target}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LuisaCompute/rust
                FILES_MATCHING REGEX ".*\\.(h|hpp)$")
        install(DIRECTORY "$<TARGET_FILE_DIR:luisa-compute-core>/"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/"
                FILES_MATCHING REGEX ".*luisa_compute_${target}.*\\.(dll|so|dylib)$")
        install(DIRECTORY "$<TARGET_LINKER_FILE_DIR:luisa-compute-core>/"
                DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
                FILES_MATCHING REGEX ".*luisa_compute_${target}.*\\.(lib|a)$")
    endfunction()
    luisa_compute_install_rust(backend_impl)
    luisa_compute_install_rust(api_types)
    luisa_compute_install_rust(ir)
    luisa_compute_install(rust-meta)
endif ()
