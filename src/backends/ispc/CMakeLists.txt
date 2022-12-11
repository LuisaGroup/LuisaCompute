if (NOT CMAKE_ISPC_COMPILER)
    find_program(BUNDLE_ISPC_EXE NAMES ispc ispc.exe PATHS ispc_support NO_DEFAULT_PATH NO_CACHE)
    set(CMAKE_ISPC_COMPILER ${BUNDLE_ISPC_EXE})
    if (NOT BUNDLE_ISPC_EXE)
        find_program(SYSTEM_ISPC_EXE NAMES ispc ispc.exe NO_CACHE)
        set(CMAKE_ISPC_COMPILER ${SYSTEM_ISPC_EXE})
    endif ()
endif ()

if (CMAKE_ISPC_COMPILER)
    enable_language(ISPC)
    if (CMAKE_ISPC_COMPILER_VERSION VERSION_LESS 1.16)
        message(WARNING "ISPC version 1.16 or higher is required (found ${CMAKE_ISPC_COMPILER_VERSION})")
    else ()
        message(STATUS "Found ISPC compiler: ${CMAKE_ISPC_COMPILER} (version ${CMAKE_ISPC_COMPILER_VERSION})")
        set(ISPC_FOUND TRUE)
    endif ()
endif ()

if (ISPC_FOUND)
    message(STATUS "Build with ISPC backend")
    set(ISPC_BACKEND_SOURCES
            ispc_device.cpp ispc_device.h
            ispc_stream.cpp ispc_stream.h
            ispc_event.cpp ispc_event.h
            ispc_codegen.cpp ispc_codegen.h
            ispc_module.h
            ispc_shader.cpp ispc_shader.h
            ispc_mesh.cpp ispc_mesh.h
            ispc_accel.cpp ispc_accel.h
            ispc_texture.cpp ispc_texture.h
            ispc_bindless_array.cpp ispc_bindless_array.h)
    luisa_compute_add_backend(ispc SOURCES ${ISPC_BACKEND_SOURCES})

    add_custom_target(luisa-compute-copy-ispc-support ALL
            COMMENT "Copying ISPC support files")
    add_dependencies(luisa-compute-backend-ispc luisa-compute-copy-ispc-support)

    # copy device libs
    add_custom_command(TARGET luisa-compute-copy-ispc-support
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/ispc_support"
            "$<TARGET_FILE_DIR:luisa-compute-core>"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ispc_support")
    # copy the ISPC compiler
    add_custom_command(TARGET luisa-compute-copy-ispc-support
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_ISPC_COMPILER}"
            "$<TARGET_FILE_DIR:luisa-compute-core>/"
            DEPENDS "${CMAKE_ISPC_COMPILER}")
    install(FILES "${CMAKE_ISPC_COMPILER}" DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY ispc_support DESTINATION ${CMAKE_INSTALL_BINDIR})

    option(LUISA_COMPUTE_ENABLE_ISPC_LLVM_JIT "Enable LLVM JIT for ISPC kernels" ON)
    if (LUISA_COMPUTE_ENABLE_ISPC_LLVM_JIT)
        find_package(LLVM)
    endif ()

    if (LLVM_FOUND) # link with llvm for jit
        message(STATUS "Use LLVM JIT for ISPC kernels")
        llvm_map_components_to_libnames(
                LLVM_LIBS core executionengine support mcjit nativecodegen irreader passes analysis coroutines)
        target_sources(luisa-compute-backend-ispc PRIVATE ispc_jit_module.cpp ispc_jit_module.h)
        message(STATUS "LLVM: ${LLVM_INCLUDE_DIRS}")
        target_include_directories(luisa-compute-backend-ispc PRIVATE ${LLVM_INCLUDE_DIRS})
        target_link_libraries(luisa-compute-backend-ispc PRIVATE ${LLVM_LIBS})
        target_compile_definitions(luisa-compute-backend-ispc PRIVATE LUISA_COMPUTE_ISPC_LLVM_JIT)
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(luisa-compute-backend-ispc PRIVATE /wd4624 /wd4996) # do not complain about LLVM
        endif ()
    else () # fallback to DLL
        message(STATUS "Use runtime DLL loading for ISPC kernels. You may need to manually copy linker stuff to ispc_support and re-run build.")
        target_sources(luisa-compute-backend-ispc PRIVATE ispc_dll_module.cpp ispc_dll_module.h)
    endif ()
endif ()
