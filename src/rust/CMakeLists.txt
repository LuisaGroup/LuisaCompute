if (LUISA_COMPUTE_ENABLE_RUST)
    corrosion_import_crate(
            MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
            NO_DEFAULT_FEATURES
    )
    #    if (WIN32 AND NOT MINGW AND NOT MSYS)
    #        corrosion_set_linker(luisa_compute_ir link.exe)
    #        corrosion_set_linker(luisa_compute_api_types link.exe)
    #        corrosion_set_linker(luisa_compute_backend link.exe)
    #    endif ()
    #    if(LUISA_COMPUTE_COMPILED_BY_RUST)
    #        corrosion_set_features(luisa_compute_backend_impl FEATURES hack_copy_dll)
    #    endif()
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

    if (TARGET luisa_compute_backend_impl)
        target_link_libraries(luisa-compute-rust-meta INTERFACE luisa_compute_backend_impl)
        get_target_property(LUISA_CARGO_PROFILE luisa_compute_backend_impl INTERFACE_CORROSION_CARGO_PROFILE)
        message(STATUS "Cargo profile: ${LUISA_CARGO_PROFILE}")
        add_custom_target(luisa-compute-rust-copy ALL DEPENDS luisa_compute_backend_impl)
        function(luisa_compute_copy_rust_file name)
            # get last extension
            get_filename_component(name_ext "${name}" LAST_EXT)
            string(TOLOWER "${name_ext}" name_ext)
            if (name_EXT STREQUAL ".lib" OR
                    name_EXT STREQUAL ".exp" OR
                    name_EXT STREQUAL ".pdb" OR
                    name_EXT STREQUAL ".a")
                set(LUISA_RUST_DEST_DIR "$<TARGET_LINKER_FILE_DIR:luisa-compute-core>/")
            else ()
                set(LUISA_RUST_DEST_DIR "$<TARGET_FILE_DIR:luisa-compute-core>/")
            endif ()
            set(LUISA_CARGO_PROFILE "$<$<CONFIG:DEBUG>:debug>$<$<NOT:$<CONFIG:DEBUG>>:release>")
            set(LUISA_RUST_SRC_DIR "$<TARGET_FILE_DIR:luisa-compute-core>/../cargo/build/${Rust_CARGO_TARGET_CACHED}/${LUISA_CARGO_PROFILE}")
            add_custom_command(TARGET luisa-compute-rust-copy
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${LUISA_RUST_SRC_DIR}/${name}"
                    "${LUISA_RUST_DEST_DIR}/${name}"
                    COMMENT "Copying '${LUISA_RUST_SRC_DIR}/${name}' to '${LUISA_RUST_DEST_DIR}/${name}'...")
        endfunction()
        if (WIN32)
            set(LUISA_RUST_COPY_LIST
                    embree4.dll
                    embree4.lib
                    libmmd.dll
                    msvcp140.dll
                    svml_dispmd.dll
                    tbb.lib
                    tbb12.dll
                    tbbmalloc.dll
                    vcruntime140.dll
                    vcruntime140_1.dll)
        endif ()
        foreach (rust_file ${LUISA_RUST_COPY_LIST})
            luisa_compute_copy_rust_file("${rust_file}")
        endforeach ()
        add_dependencies(luisa-compute-rust-meta luisa-compute-rust-copy)
    endif ()

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
    function(luisa_compute_install_rust2 target)
        install(TARGETS luisa_compute_${target} EXPORT LuisaComputeTargets
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
        install(DIRECTORY luisa_compute_${target}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LuisaCompute/rust
                FILES_MATCHING REGEX ".*\\.(h|hpp)$")
    endfunction()
    luisa_compute_install_rust(backend_impl)
    luisa_compute_install_rust(api_types)
    luisa_compute_install_rust(ir)
    luisa_compute_install(rust-meta)
endif ()