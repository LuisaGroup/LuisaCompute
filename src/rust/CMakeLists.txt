if (LUISA_COMPUTE_ENABLE_RUST)
    # set profile based on build type
    set(LUISA_CARGO_PROFILE "$<IF:$<CONFIG:Debug>,dev,release>")
    corrosion_import_crate(
            MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
            PROFILE ${LUISA_CARGO_PROFILE}
            NO_DEFAULT_FEATURES)

    corrosion_set_env_vars(luisa_compute_ir "LC_RS_DO_NOT_GENERATE_BINDINGS=0")
    corrosion_set_env_vars(luisa_compute_api_types "LC_RS_DO_NOT_GENERATE_BINDINGS=0")
    

    add_library(luisa-compute-rust-meta INTERFACE)
    target_include_directories(luisa-compute-rust-meta INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_link_libraries(luisa-compute-rust-meta INTERFACE luisa_compute_ir luisa_compute_api_types)
    target_compile_definitions(luisa-compute-rust-meta INTERFACE LUISA_ENABLE_RUST)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(luisa-compute-rust-meta INTERFACE -Wno-return-type-c-linkage)
    endif ()

    function(luisa_compute_rust_fix_rpath target_name)
        set(target luisa_compute_${target_name})
        if (APPLE)
            add_custom_target(${target}-fix-rpath ALL DEPENDS cargo-build_${target}
                    COMMAND install_name_tool -id "@rpath/lib${target}.dylib"
                    "$<TARGET_FILE_DIR:luisa-compute-core>/lib${target}.dylib"
                    COMMAND install_name_tool -add_rpath "@loader_path"
                    "$<TARGET_FILE_DIR:luisa-compute-core>/lib${target}.dylib")
            add_dependencies(luisa-compute-rust-meta ${target}-fix-rpath)
        elseif (UNIX)
            find_program(PATCHELF_EXE patchelf)
            if (PATCHELF_EXE)
                add_custom_target(${target}-fix-rpath ALL DEPENDS cargo-build_${target}
                        COMMENT "Fixing rpath for ${target}..."
                        COMMAND ${PATCHELF_EXE} --set-rpath "$ORIGIN"
                        "$<TARGET_FILE_DIR:luisa-compute-core>/lib${target}.so"
                        VERBATIM)
                add_dependencies(luisa-compute-rust-meta ${target}-fix-rpath)
            else ()
                message(WARNING "Cannot find patchelf. The Rust backend may not work correctly.")
            endif ()
        endif ()
    endfunction()

    luisa_compute_rust_fix_rpath(ir)
    luisa_compute_rust_fix_rpath(backend_impl)

    function(luisa_compute_install_rust target)
        # FIXME: corrosion does not correctly installs the rust library so we have to manually install it here
        # corrosion_install(TARGETS luisa_compute_${target}
        #         RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        #         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        #         ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        #         PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

        install(TARGETS luisa_compute_${target} EXPORT LuisaComputeTargets)
        install(DIRECTORY "$<TARGET_FILE_DIR:luisa-compute-core>/"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/"
                FILES_MATCHING REGEX ".*luisa_compute_${target}.*\\.(dll|so|dylib)(\\.[0-9]+)?$")
        install(DIRECTORY "$<TARGET_LINKER_FILE_DIR:luisa-compute-core>/"
                DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
                FILES_MATCHING REGEX ".*luisa_compute_${target}.*\\.(lib|a)$")
    endfunction()

    luisa_compute_install_rust(api_types)
    luisa_compute_install_rust(ir)
    luisa_compute_install(rust-meta)

    # optionally enable the CPU and remote backends implemented in Rust
    if (LUISA_COMPUTE_ENABLE_CPU OR LUISA_COMPUTE_ENABLE_REMOTE)

        if (LUISA_COMPUTE_ENABLE_REMOTE)
            corrosion_set_features(luisa_compute_backend_impl FEATURES remote)
        endif ()
        if (LUISA_COMPUTE_EMBREE_ZIP_PATH STREQUAL "" OR NOT LUISA_COMPUTE_EMBREE_ZIP_PATH)
        else()
            if(NOT EXISTS ${LUISA_COMPUTE_EMBREE_ZIP_PATH})
                message(FATAL_ERROR "LUISA_COMPUTE_EMBREE_ZIP_PATH is set to ${LUISA_COMPUTE_EMBREE_ZIP_PATH} but the file does not exist.")
            endif()
            corrosion_set_env_vars(luisa_compute_backend_impl "EMBREE_ZIP_FILE=${LUISA_COMPUTE_EMBREE_ZIP_PATH}")
        endif()
        if (LUISA_COMPUTE_ENABLE_CPU)

            corrosion_set_features(luisa_compute_backend_impl FEATURES cpu)
            set(LUISA_CARGO_PROFILE "$<IF:$<CONFIG:Debug>,debug,release>")
            set(LUISA_RUST_OUTPUT_DIR
                    "$<TARGET_FILE_DIR:luisa-compute-core>/../cargo/build/${Rust_CARGO_TARGET_CACHED}/${LUISA_CARGO_PROFILE}")
            corrosion_set_env_vars(luisa_compute_backend_impl
                    "EMBREE_DLL_OUT_DIR=${LUISA_RUST_OUTPUT_DIR}/backend_support"
                    "CMAKE=${CMAKE_COMMAND}"
                    "CMAKE_GENERATOR=${CMAKE_GENERATOR}"
                    "CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}")

            # building from source is preferred for Python wheels
            if (UNIX)
                if (APPLE) # workaround clang linking
                    corrosion_set_env_vars(luisa_compute_backend_impl "EMBREE_CC=cc" "EMBREE_CXX=c++")
                endif ()
                if (SKBUILD) # force building from source for Python binding to avoid dependency issues
                    corrosion_set_env_vars(luisa_compute_backend_impl "EMBREE_FORCE_BUILD_FROM_SOURCE=1")
                endif ()
            endif ()
            add_custom_target(luisa-compute-rust-copy DEPENDS cargo-build_luisa_compute_backend_impl)
            add_custom_command(TARGET luisa-compute-rust-copy
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${LUISA_RUST_OUTPUT_DIR}/backend_support/"
                    "$<TARGET_FILE_DIR:luisa-compute-core>/")
            install(DIRECTORY "${LUISA_RUST_OUTPUT_DIR}/backend_support/"
                    DESTINATION "${CMAKE_INSTALL_BINDIR}/"
                    FILES_MATCHING REGEX ".*\\.(dll|so|dylib)(\\.[0-9]+)?$")
            add_dependencies(luisa_compute_backend_impl luisa-compute-rust-copy)
        endif ()

        luisa_compute_install_rust(backend_impl)

    else ()
        # exclude the backend implementation from the build
        set_target_properties(luisa_compute_backend_impl PROPERTIES EXCLUDE_FROM_ALL ON)
    endif ()

endif ()
