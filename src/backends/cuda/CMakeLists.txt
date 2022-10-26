find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)

    message(STATUS "Build with CUDA backend: ${CUDAToolkit_VERSION}")
    if (NOT DEFINED OptiX_DIR)
        if (DEFINED OPTIX_DIR)
            set(OptiX_DIR "${OPTIX_DIR}")
        elseif (DEFINED Optix_DIR)
            set(OptiX_DIR "${Optix_DIR}")
        endif ()
    endif ()
    if (DEFINED OptiX_DIR)
        if (EXISTS "${OptiX_DIR}/include/optix.h")
            set(OptiX_INCLUDE_DIR "${OptiX_DIR}/include")
        else ()
            message(FATAL_ERROR "OptiX_DIR is set to '${OptiX_DIR}', but 'optix.h' is not found at '${OptiX_DIR}/include/optix.h'")
        endif ()
    else ()
        if (EXISTS optix/optix.h)
            set(OptiX_INCLUDE_DIR optix)
        else ()
            message(FATAL_ERROR "OptiX_DIR is not defined and OptiX headers are not copied to '${CMAKE_CURRENT_SOURCE_DIR}/optix'")
        endif ()
    endif ()
    set(LUISA_COMPUTE_CUDA_SOURCES
            cuda_error.h
            cuda_bindless_array.cpp cuda_bindless_array.h
            cuda_stream.cpp cuda_stream.h
            cuda_device.cpp cuda_device.h
            cuda_host_buffer_pool.cpp cuda_host_buffer_pool.h
            cuda_command_encoder.cpp cuda_command_encoder.h
            cuda_mipmap_array.cpp cuda_mipmap_array.h
            cuda_device_math_embedded.inl.h
            cuda_device_resource_embedded.inl.h
            cuda_codegen.cpp cuda_codegen.h
            cuda_compiler.cpp cuda_compiler.h
            cuda_accel.cpp cuda_accel.h
            cuda_mesh.cpp cuda_mesh.h
            cuda_shader.cpp cuda_shader.h
            cuda_callback_context.h)

    luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})
    target_link_libraries(luisa-compute-backend-cuda PRIVATE CUDA::cuda_driver CUDA::nvrtc)
    if (WIN32)
        target_link_libraries(luisa-compute-backend-cuda PRIVATE cfgmgr32)
    endif ()
    target_include_directories(luisa-compute-backend-cuda PRIVATE "${OptiX_INCLUDE_DIR}")
endif ()
