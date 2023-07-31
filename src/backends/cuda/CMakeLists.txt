find_package(CUDAToolkit 11.7)

if (CUDAToolkit_FOUND)
    message(STATUS "Build with CUDA backend: ${CUDAToolkit_VERSION}")

    set(LUISA_COMPUTE_CUDA_SOURCES
            cuda_error.h
            cuda_bindless_array.cpp cuda_bindless_array.h
            cuda_buffer.cpp cuda_buffer.h
            cuda_stream.cpp cuda_stream.h
            cuda_device.cpp cuda_device.h
            cuda_dstorage.cpp cuda_dstorage.h
            cuda_event.cpp cuda_event.h
            cuda_host_buffer_pool.cpp cuda_host_buffer_pool.h
            cuda_command_encoder.cpp cuda_command_encoder.h
            cuda_texture.cpp cuda_texture.h
            cuda_codegen_ast.cpp cuda_codegen_ast.h
            cuda_compiler.cpp cuda_compiler.h
            cuda_accel.cpp cuda_accel.h
            cuda_primitive.cpp cuda_primitive.h
            cuda_procedural_primitive.cpp cuda_procedural_primitive.h
            cuda_mesh.cpp cuda_mesh.h
            cuda_shader.cpp cuda_shader.h
            cuda_shader_metadata.cpp cuda_shader_metadata.h
            cuda_shader_native.cpp cuda_shader_native.h
            cuda_shader_optix.cpp cuda_shader_optix.h
            cuda_ext.cpp cuda_ext.h
            cuda_swapchain.cpp cuda_swapchain.h
            cuda_callback_context.h
            cuda_builtin_embedded.cpp cuda_builtin_embedded.h
            optix_api.cpp optix_api.h
            default_binary_io.cpp
            string_scratch.cpp)

    luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})

    target_link_libraries(luisa-compute-backend-cuda PRIVATE
            CUDA::cuda_driver
            luisa-compute-vulkan-swapchain)
    target_precompile_headers(luisa-compute-backend-cuda PRIVATE pch.h)

    find_library(CUDA_DEVICE_RUNTIME_LIBRARY cudadevrt ${CUDAToolkit_LIBRARY_DIR})
    add_custom_command(TARGET luisa-compute-backend-cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDA_DEVICE_RUNTIME_LIBRARY}"
            $<TARGET_FILE_DIR:luisa-compute-core>)
    install(FILES "${CUDA_DEVICE_RUNTIME_LIBRARY}" DESTINATION ${CMAKE_INSTALL_BINDIR})

    if (WIN32)
        # FIXME: trouble with nvrtc_static on Windows due to the mismatch of runtime library...
        target_link_libraries(luisa-compute-backend-cuda PRIVATE CUDA::nvrtc)
    else ()
        if (TARGET CUDA::nvrtc_static)
            target_link_libraries(luisa-compute-backend-cuda PRIVATE CUDA::nvrtc_static)
        else ()
            message(WARNING "CUDA::nvrtc_static not found. Using CUDA::nvrtc instead. This may cause DLL dependency issues.")
            target_link_libraries(luisa-compute-backend-cuda PRIVATE CUDA::nvrtc)
        endif ()
    endif ()

    if (WIN32)
        target_link_libraries(luisa-compute-backend-cuda PRIVATE cfgmgr32)
    endif ()

    # nvCOMP
    if (LUISA_COMPUTE_DOWNLOAD_NVCOMP)
        if (WIN32)
            set(NVCOMP_PLATFORM "windows")
            set(NVCOMP_EXT "zip")
        else ()
            set(NVCOMP_PLATFORM "x86_64")
            set(NVCOMP_EXT "tgz")
        endif ()
        set(NVCOMP_VERSION "2.6.1")
        set(NVCOMP_DOWNLOAD_URL "https://developer.download.nvidia.com/compute/nvcomp/${NVCOMP_VERSION}/local_installers/nvcomp_${NVCOMP_VERSION}_${NVCOMP_PLATFORM}_${CUDAToolkit_VERSION_MAJOR}.x.${NVCOMP_EXT}")
        message(STATUS "Downloading nvcomp from ${NVCOMP_DOWNLOAD_URL}")
        include(FetchContent)
        FetchContent_Declare(nvcomp URL ${NVCOMP_DOWNLOAD_URL} DOWNLOAD_EXTRACT_TIMESTAMP ON)
        FetchContent_MakeAvailable(nvcomp)
        FetchContent_GetProperties(nvcomp)
        set(nvcomp_DIR ${nvcomp_SOURCE_DIR}/lib/cmake/nvcomp)
    endif ()

    find_package(nvcomp CONFIG)
    if (NOT nvcomp_FOUND)
        message(WARNING "nvcomp not found. The CUDA backend will be built without GPU compression support. You may want to set the LUISA_COMPUTE_DOWNLOAD_NVCOMP option to ON to let LuisaCompute automatically download it.")
    else ()
        target_compile_definitions(luisa-compute-backend-cuda PRIVATE LUISA_COMPUTE_ENABLE_NVCOMP=1)
        set(NVCOMP_TARGETS
                nvcomp::nvcomp
                nvcomp::nvcomp_bitcomp
                nvcomp::nvcomp_gdeflate)
        target_link_libraries(luisa-compute-backend-cuda PRIVATE ${NVCOMP_TARGETS})
        foreach (target ${NVCOMP_TARGETS})
            add_custom_command(TARGET luisa-compute-backend-cuda POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:${target}>
                    $<TARGET_FILE_DIR:luisa-compute-core>)
            install(FILES $<TARGET_FILE:${target}> DESTINATION ${CMAKE_INSTALL_BINDIR})
        endforeach ()
    endif ()

    # CUB LuisaCompute integration
    if (LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB)
        add_subdirectory(lcub)
    endif()

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "CUDA not found. The CUDA backend will not be built.")
else ()
    message(WARNING "CUDA not found. The CUDA backend will not be built.")
endif ()
