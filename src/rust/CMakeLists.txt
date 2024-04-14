if (LUISA_COMPUTE_ENABLE_RUST)

    file(GLOB_RECURSE LUISA_RUST_SOURCES
            CONFIGURE_DEPENDS
            "luisa_*/*.rs"
            "luisa_*/*.h"
            "luisa_*/*.toml")
    file(GLOB LUISA_RUST_TOP_LEVEL_SOURCES
            CONFIGURE_DEPENDS
            "*.toml"
            "*.rs")
    list(APPEND LUISA_RUST_SOURCES ${LUISA_RUST_TOP_LEVEL_SOURCES})

    set(LUISA_RUST_PROFILE "--profile;$<IF:$<CONFIG:Debug>,dev,release>")
    set(LUISA_RUST_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml")
    set(LUISA_RUST_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/$<1:rust>") # use generator expr to avoid unexpected directory creation with multi-config generators
    set(LUISA_RUST_OUTPUT_DIR "${LUISA_RUST_BUILD_DIR}/$<IF:$<CONFIG:Debug>,debug,release>")
    set(LUISA_RUST_ENV "CARGO_TARGET_DIR=${LUISA_RUST_BUILD_DIR}")

    # process features
    set(LUISA_RUST_FEATURES "")
    if (LUISA_COMPUTE_ENABLE_CPU)
        list(APPEND LUISA_RUST_FEATURES "cpu")
    endif ()
    if (LUISA_COMPUTE_ENABLE_REMOTE)
        list(APPEND LUISA_RUST_FEATURES "remote")
    endif ()
    if (LUISA_RUST_FEATURES)
        list(JOIN LUISA_RUST_FEATURES "," LUISA_RUST_FEATURES)
        set(LUISA_RUST_FEATURES "--no-default-features;--features;${LUISA_RUST_FEATURES}")
    else ()
        set(LUISA_RUST_FEATURES "--no-default-features")
    endif ()

    # handle embree
    if (LUISA_COMPUTE_ENABLE_CPU)
        if (NOT LUISA_COMPUTE_EMBREE_ZIP_PATH)
            if ($ENV{EMBREE_ZIP_FILE})
                set(LUISA_COMPUTE_EMBREE_ZIP_PATH $ENV{EMBREE_ZIP_FILE})
            elseif ($ENV{EMBREE_ZIP_PATH})
                set(LUISA_COMPUTE_EMBREE_ZIP_PATH $ENV{EMBREE_ZIP_PATH})
            endif ()
        endif ()
        if (LUISA_COMPUTE_EMBREE_ZIP_PATH)
            if (NOT EXISTS ${LUISA_COMPUTE_EMBREE_ZIP_PATH})
                message(FATAL_ERROR "LUISA_COMPUTE_EMBREE_ZIP_PATH is set to ${LUISA_COMPUTE_EMBREE_ZIP_PATH} but the file does not exist.")
            endif ()
            list(APPEND LUISA_RUST_ENV "EMBREE_ZIP_FILE=${LUISA_COMPUTE_EMBREE_ZIP_PATH}")
        endif ()
        list(APPEND LUISA_RUST_ENV
                "EMBREE_DLL_OUT_DIR=${LUISA_RUST_OUTPUT_DIR}/embree"
                "CMAKE=${CMAKE_COMMAND}"
                "CMAKE_GENERATOR=${CMAKE_GENERATOR}"
                "CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}")
        if (UNIX)
            if (APPLE) # workaround clang linking
                list(APPEND LUISA_RUST_ENV "EMBREE_CC=cc" "EMBREE_CXX=c++")
            endif ()
            if (SKBUILD) # force building from source for Python binding to avoid dependency issues
                list(APPEND LUISA_RUST_ENV "EMBREE_FORCE_BUILD_FROM_SOURCE=1")
            endif ()
        endif ()
    endif ()
    set(LUISA_RUST_BUILD_COMMAND "${CMAKE_COMMAND};-E;env;${LUISA_RUST_ENV};--;${CARGO_EXE};build;--manifest-path;${LUISA_RUST_MANIFEST};${LUISA_RUST_PROFILE};${LUISA_RUST_FEATURES}")

    if (WIN32)
        set(LUISA_RUST_STATIC_LIBS "luisa_compute_ir_static.lib;luisa_compute_api_types.lib")
        set(LUISA_RUST_SHARED_LIBS "luisa_compute_backend_impl.dll")
    else ()
        set(LUISA_RUST_STATIC_LIBS "libluisa_compute_ir_static.a;libluisa_compute_api_types.a")
        if (APPLE)
            set(LUISA_RUST_SHARED_LIBS "libluisa_compute_backend_impl.dylib")
        else ()
            set(LUISA_RUST_SHARED_LIBS "libluisa_compute_backend_impl.so")
        endif ()
    endif ()

    set(LUISA_RUST_STATIC_LIB_FILES ${LUISA_RUST_STATIC_LIBS})
    list(TRANSFORM LUISA_RUST_STATIC_LIB_FILES PREPEND "${LUISA_RUST_OUTPUT_DIR}/")
    set(LUISA_RUST_SHARED_LIB_FILES ${LUISA_RUST_SHARED_LIBS})
    list(TRANSFORM LUISA_RUST_SHARED_LIB_FILES PREPEND "${LUISA_RUST_OUTPUT_DIR}/")

    set(LUISA_RUST_LIB_FILES ${LUISA_RUST_STATIC_LIB_FILES} ${LUISA_RUST_SHARED_LIB_FILES})
    if (WIN32)
        foreach (lib ${LUISA_RUST_SHARED_LIBS})
            string(REPLACE ".dll" ".pdb" pdb ${lib})
            list(APPEND LUISA_RUST_LIB_FILES "${LUISA_RUST_OUTPUT_DIR}/${pdb}" "${LUISA_RUST_OUTPUT_DIR}/${lib}.lib")
        endforeach ()
    endif ()

    # final path of rust libs
    set(LUISA_RUST_RESULT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    add_custom_command(OUTPUT ${LUISA_RUST_LIB_FILES}
            COMMAND ${LUISA_RUST_BUILD_COMMAND}
            DEPENDS ${LUISA_RUST_SOURCES}
            COMMENT "Building Rust-dependent libraries..."
            VERBATIM)

    add_custom_target(luisa_compute_rust_build ALL
            DEPENDS ${LUISA_RUST_LIB_FILES})

    # copy the static libraries to the output directory
    foreach (lib ${LUISA_RUST_STATIC_LIBS})
        add_custom_command(OUTPUT "${LUISA_RUST_RESULT_DIR}/${lib}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${LUISA_RUST_OUTPUT_DIR}/${lib}"
                "${LUISA_RUST_RESULT_DIR}/${lib}"
                DEPENDS "${LUISA_RUST_OUTPUT_DIR}/${lib}"
                VERBATIM)
        add_custom_target(${lib} DEPENDS "${LUISA_RUST_RESULT_DIR}/${lib}")
        add_dependencies(luisa_compute_rust_build ${lib})
    endforeach ()

    # install the static libraries
    install(FILES ${LUISA_RUST_STATIC_LIB_FILES}
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/")

    # copy embree to the output directory and also install it
    if (LUISA_COMPUTE_ENABLE_CPU)
        add_custom_command(OUTPUT "${LUISA_RUST_RESULT_DIR}/embree/dummy"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${LUISA_RUST_OUTPUT_DIR}/embree/"
                "${LUISA_RUST_RESULT_DIR}/"
                COMMAND ${CMAKE_COMMAND} -E touch
                "${LUISA_RUST_RESULT_DIR}/dummy"
                # if LUISA_RUST_LIB_FILES is rebuilt, then it should be rebuilt too, it's incorrect.
                # TODO: use embree file as DEPENDS
                DEPENDS "${LUISA_RUST_LIB_FILES}"
                COMMENT "Copying embree to the output directory..."
                VERBATIM)
        add_custom_target(luisa_compute_rust_build_embree DEPENDS "${LUISA_RUST_RESULT_DIR}/embree/dummy")
        add_dependencies(luisa_compute_rust_build luisa_compute_rust_build_embree)
        install(DIRECTORY "${LUISA_RUST_OUTPUT_DIR}/embree/"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/"
                FILES_MATCHING REGEX ".*\\.(dll|so|dylib)(\\.[0-9]+)?$")
    endif ()

    if (APPLE)
        find_program(INSTALL_NAME_TOOL_EXE install_name_tool)
    elseif (UNIX)
        find_program(PATCHELF_EXE patchelf)
    endif ()

    # copy the DLLs to the output directory and patch the rpath
    foreach (lib ${LUISA_RUST_SHARED_LIBS})
        if (APPLE)
            if (INSTALL_NAME_TOOL_EXE)
                add_custom_command(OUTPUT "${LUISA_RUST_RESULT_DIR}/${lib}"
                        COMMENT "Copying ${LUISA_RUST_OUTPUT_DIR}/${lib} to ${LUISA_RUST_RESULT_DIR}/${lib} and patch it"
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LUISA_RUST_OUTPUT_DIR}/${lib}" "${LUISA_RUST_RESULT_DIR}/${lib}"
                        COMMAND ${INSTALL_NAME_TOOL_EXE} -id "@rpath/${lib}" "${LUISA_RUST_RESULT_DIR}/${lib}"
                        COMMAND ${INSTALL_NAME_TOOL_EXE} -add_rpath "@loader_path" "${LUISA_RUST_RESULT_DIR}/${lib}"
                        DEPENDS "${LUISA_RUST_OUTPUT_DIR}/${lib}"
                        VERBATIM)
            else ()
                add_custom_command(OUTPUT "${LUISA_RUST_RESULT_DIR}/${lib}"
                        COMMENT "Copying ${LUISA_RUST_OUTPUT_DIR}/${lib} to ${LUISA_RUST_RESULT_DIR}/${lib}"
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LUISA_RUST_OUTPUT_DIR}/${lib}" "${LUISA_RUST_RESULT_DIR}/${lib}"
                        DEPENDS "${LUISA_RUST_OUTPUT_DIR}/${lib}"
                        VERBATIM)
                message(WARNING "Cannot find install_name_tool. The Rust backend may not work correctly.")
            endif ()
            add_custom_target(luisa_compute_rust_build_${lib} DEPENDS
                    "${LUISA_RUST_RESULT_DIR}/${lib}")
        elseif (UNIX)
            if (PATCHELF_EXE)
                add_custom_command(OUTPUT "${LUISA_RUST_RESULT_DIR}/${lib}"
                        COMMENT "Copying ${LUISA_RUST_OUTPUT_DIR}/${lib} to ${LUISA_RUST_RESULT_DIR}/${lib} and patch it"
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LUISA_RUST_OUTPUT_DIR}/${lib}" "${LUISA_RUST_RESULT_DIR}/${lib}"
                        COMMAND ${PATCHELF_EXE} --set-rpath "$ORIGIN" "${LUISA_RUST_RESULT_DIR}/${lib}"
                        DEPENDS "${LUISA_RUST_OUTPUT_DIR}/${lib}"
                        VERBATIM)
            else ()
                add_custom_command(OUTPUT "${LUISA_RUST_RESULT_DIR}/${lib}"
                        COMMENT "Copying ${LUISA_RUST_OUTPUT_DIR}/${lib} to ${LUISA_RUST_RESULT_DIR}/${lib}"
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LUISA_RUST_OUTPUT_DIR}/${lib}" "${LUISA_RUST_RESULT_DIR}/${lib}"
                        DEPENDS "${LUISA_RUST_OUTPUT_DIR}/${lib}"
                        VERBATIM)
                message(WARNING "Cannot find patchelf. The Rust backend may not work correctly.")
            endif ()
            add_custom_target(luisa_compute_rust_build_${lib} DEPENDS
                    "${LUISA_RUST_RESULT_DIR}/${lib}")
        else () # WIN32
            add_custom_command(OUTPUT "${LUISA_RUST_RESULT_DIR}/${lib}"
                    COMMENT "Copying ${LUISA_RUST_OUTPUT_DIR}/${lib} to ${LUISA_RUST_RESULT_DIR}/${lib}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LUISA_RUST_OUTPUT_DIR}/${lib}" "${LUISA_RUST_RESULT_DIR}/${lib}"
                    DEPENDS "${LUISA_RUST_OUTPUT_DIR}/${lib}"
                    VERBATIM)
            add_custom_command(OUTPUT "${LUISA_RUST_RESULT_DIR}/${lib}.lib"
                    COMMENT "Copying ${LUISA_RUST_OUTPUT_DIR}/${lib}.lib to ${LUISA_RUST_RESULT_DIR}/${lib}.lib"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${LUISA_RUST_OUTPUT_DIR}/${lib}.lib"
                    "${LUISA_RUST_RESULT_DIR}/${lib}.lib"
                    DEPENDS "${LUISA_RUST_OUTPUT_DIR}/${lib}.lib"
                    VERBATIM)
            string(REPLACE ".dll" ".pdb" pdb ${lib})
            add_custom_command(OUTPUT "${LUISA_RUST_RESULT_DIR}/${pdb}"
                    COMMENT "Copying ${LUISA_RUST_OUTPUT_DIR}/${pdb} to ${LUISA_RUST_RESULT_DIR}/${pdb}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${LUISA_RUST_OUTPUT_DIR}/${pdb}"
                    "${LUISA_RUST_RESULT_DIR}/${pdb}"
                    DEPENDS "${LUISA_RUST_OUTPUT_DIR}/${pdb}"
                    VERBATIM)
            add_custom_target(luisa_compute_rust_build_${lib} DEPENDS
                    "${LUISA_RUST_RESULT_DIR}/${lib}"
                    "${LUISA_RUST_RESULT_DIR}/${lib}.lib"
                    "${LUISA_RUST_RESULT_DIR}/${pdb}")
        endif ()
        add_dependencies(luisa_compute_rust_build luisa_compute_rust_build_${lib})
        install(FILES "${LUISA_RUST_RESULT_DIR}/${lib}"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/")
        if (WIN32)
            string(REPLACE ".dll" ".pdb" pdb ${lib})
            install(FILES "${LUISA_RUST_OUTPUT_DIR}/${pdb}"
                    DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
            install(FILES "${LUISA_RUST_OUTPUT_DIR}/${lib}.lib"
                    DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
        endif ()
    endforeach ()

    # add an imported static target for linking
    add_library(luisa-compute-rust-meta INTERFACE)
    add_dependencies(luisa-compute-rust-meta luisa_compute_rust_build)
    target_link_directories(luisa-compute-rust-meta INTERFACE "$<BUILD_INTERFACE:${LUISA_RUST_RESULT_DIR}>")
    target_link_libraries(luisa-compute-rust-meta INTERFACE "$<BUILD_INTERFACE:${LUISA_RUST_STATIC_LIBS}>")
    if (WIN32)
        target_link_libraries(luisa-compute-rust-meta INTERFACE
                Ws2_32 Advapi32 Bcrypt Userenv ntdll Synchronization)
    endif ()
    install(TARGETS luisa-compute-rust-meta EXPORT LuisaComputeTargets)

    # add the backend impl target
    add_library(luisa_compute_backend_impl INTERFACE)
    target_link_libraries(luisa_compute_backend_impl INTERFACE luisa-compute-rust-meta)
    foreach (lib ${LUISA_RUST_SHARED_LIBS})
        if (WIN32)
            target_link_libraries(luisa_compute_backend_impl INTERFACE "$<BUILD_INTERFACE:${lib}.lib>")
        else ()
            target_link_libraries(luisa_compute_backend_impl INTERFACE "$<BUILD_INTERFACE:${lib}>")
        endif ()
    endforeach ()
    install(TARGETS luisa_compute_backend_impl EXPORT LuisaComputeTargets)
endif ()

#if (LUISA_COMPUTE_ENABLE_RUST)
#    # set profile based on build type
#    set(LUISA_CARGO_PROFILE "$<IF:$<CONFIG:Debug>,dev,release>")
#    corrosion_import_crate(
#            MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
#            PROFILE ${LUISA_CARGO_PROFILE}
#            NO_DEFAULT_FEATURES)
#
#    # interestly this is equivalent to doting nothing...
#    # corrosion_set_env_vars(luisa_compute_ir "LC_RS_GENERATE_BINDINGS=1" "LC_RS_DO_NOT_GENERATE_BINDINGS=0")
#    # corrosion_set_env_vars(luisa_compute_api_types "LC_RS_GENERATE_BINDINGS=1" "LC_RS_DO_NOT_GENERATE_BINDINGS=0")
#
#    add_library(luisa-compute-rust-meta INTERFACE)
#    target_include_directories(luisa-compute-rust-meta INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
#    target_link_libraries(luisa-compute-rust-meta INTERFACE luisa_compute_ir_static)
#    target_compile_definitions(luisa-compute-rust-meta INTERFACE LUISA_ENABLE_RUST)
#    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#        target_compile_options(luisa-compute-rust-meta INTERFACE -Wno-return-type-c-linkage)
#    endif ()
#
#    function(luisa_compute_rust_fix_rpath target_name)
#        set(target luisa_compute_${target_name})
#        if (APPLE)
#            add_custom_target(${target}-fix-rpath ALL DEPENDS cargo-build_${target}
#                    COMMAND install_name_tool -id "@rpath/lib${target}.dylib"
#                    "$<TARGET_FILE_DIR:luisa-compute-core>/lib${target}.dylib"
#                    COMMAND install_name_tool -add_rpath "@loader_path"
#                    "$<TARGET_FILE_DIR:luisa-compute-core>/lib${target}.dylib")
#            add_dependencies(luisa-compute-rust-meta ${target}-fix-rpath)
#        elseif (UNIX)
#            find_program(PATCHELF_EXE patchelf)
#            if (PATCHELF_EXE)
#                add_custom_target(${target}-fix-rpath ALL DEPENDS cargo-build_${target}
#                        COMMENT "Fixing rpath for ${target}..."
#                        COMMAND ${PATCHELF_EXE} --set-rpath "$ORIGIN"
#                        "$<TARGET_FILE_DIR:luisa-compute-core>/lib${target}.so"
#                        VERBATIM)
#                add_dependencies(luisa-compute-rust-meta ${target}-fix-rpath)
#            else ()
#                message(WARNING "Cannot find patchelf. The Rust backend may not work correctly.")
#            endif ()
#        endif ()
#    endfunction()
#
#    luisa_compute_rust_fix_rpath(backend_impl)
#
#    function(luisa_compute_install_rust target)
#        # FIXME: corrosion does not correctly installs the rust library so we have to manually install it here
#        # corrosion_install(TARGETS luisa_compute_${target}
#        #         RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#        #         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        #         ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        #         PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#
#        install(TARGETS luisa_compute_${target} EXPORT LuisaComputeTargets)
#        install(DIRECTORY "$<TARGET_FILE_DIR:luisa-compute-core>/"
#                DESTINATION "${CMAKE_INSTALL_BINDIR}/"
#                FILES_MATCHING REGEX ".*luisa_compute_${target}.*\\.(dll|so|dylib)(\\.[0-9]+)?$")
#        install(DIRECTORY "$<TARGET_LINKER_FILE_DIR:luisa-compute-core>/"
#                DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
#                FILES_MATCHING REGEX ".*luisa_compute_${target}.*\\.(lib|a)$")
#    endfunction()
#
#    # luisa_compute_install_rust(api_types)
#    luisa_compute_install_rust(ir_static)
#    luisa_compute_install(rust-meta)
#
#    # optionally enable the CPU and remote backends implemented in Rust
#    if (LUISA_COMPUTE_ENABLE_CPU OR LUISA_COMPUTE_ENABLE_REMOTE)
#
#        if (LUISA_COMPUTE_ENABLE_REMOTE)
#            corrosion_set_features(luisa_compute_backend_impl FEATURES remote)
#        endif ()
#        if (LUISA_COMPUTE_EMBREE_ZIP_PATH STREQUAL "" OR NOT LUISA_COMPUTE_EMBREE_ZIP_PATH)
#        else()
#            if(NOT EXISTS ${LUISA_COMPUTE_EMBREE_ZIP_PATH})
#                message(FATAL_ERROR "LUISA_COMPUTE_EMBREE_ZIP_PATH is set to ${LUISA_COMPUTE_EMBREE_ZIP_PATH} but the file does not exist.")
#            endif()
#            corrosion_set_env_vars(luisa_compute_backend_impl "EMBREE_ZIP_FILE=${LUISA_COMPUTE_EMBREE_ZIP_PATH}")
#        endif()
#        if (LUISA_COMPUTE_ENABLE_CPU)
#
#            corrosion_set_features(luisa_compute_backend_impl FEATURES cpu)
#            set(LUISA_CARGO_PROFILE "$<IF:$<CONFIG:Debug>,debug,release>")
#            set(LUISA_RUST_OUTPUT_DIR
#                    "$<TARGET_FILE_DIR:luisa-compute-core>/../cargo/build/${Rust_CARGO_TARGET_CACHED}/${LUISA_CARGO_PROFILE}")
#            corrosion_set_env_vars(luisa_compute_backend_impl
#                    "EMBREE_DLL_OUT_DIR=${LUISA_RUST_OUTPUT_DIR}/backend_support"
#                    "CMAKE=${CMAKE_COMMAND}"
#                    "CMAKE_GENERATOR=${CMAKE_GENERATOR}"
#                    "CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}")
#
#            # building from source is preferred for Python wheels
#            if (UNIX)
#                if (APPLE) # workaround clang linking
#                    corrosion_set_env_vars(luisa_compute_backend_impl "EMBREE_CC=cc" "EMBREE_CXX=c++")
#                endif ()
#                if (SKBUILD) # force building from source for Python binding to avoid dependency issues
#                    corrosion_set_env_vars(luisa_compute_backend_impl "EMBREE_FORCE_BUILD_FROM_SOURCE=1")
#                endif ()
#            endif ()
#            add_custom_target(luisa-compute-rust-copy DEPENDS cargo-build_luisa_compute_backend_impl)
#            add_custom_command(TARGET luisa-compute-rust-copy
#                    COMMAND ${CMAKE_COMMAND} -E copy_directory
#                    "${LUISA_RUST_OUTPUT_DIR}/backend_support/"
#                    "$<TARGET_FILE_DIR:luisa-compute-core>/")
#            install(DIRECTORY "${LUISA_RUST_OUTPUT_DIR}/backend_support/"
#                    DESTINATION "${CMAKE_INSTALL_BINDIR}/"
#                    FILES_MATCHING REGEX ".*\\.(dll|so|dylib)(\\.[0-9]+)?$")
#            add_dependencies(luisa_compute_backend_impl luisa-compute-rust-copy)
#        endif ()
#
#        luisa_compute_install_rust(backend_impl)
#
#    else ()
#        # exclude the backend implementation from the build
#        set_target_properties(luisa_compute_backend_impl PROPERTIES EXCLUDE_FROM_ALL ON)
#    endif ()
#
#endif ()
