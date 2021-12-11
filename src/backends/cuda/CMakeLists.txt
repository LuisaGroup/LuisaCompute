find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)

    message(STATUS "Build with CUDA backend: ${CUDAToolkit_VERSION}")

    set(LUISA_COMPUTE_CUDA_SOURCES
            cuda_error.h
            cuda_bindless_array.cpp cuda_bindless_array.h
            cuda_stream.cpp cuda_stream.h
            cuda_device.cpp cuda_device.h
            cuda_ring_buffer.cpp cuda_ring_buffer.h
            cuda_command_encoder.cpp cuda_command_encoder.h
            cuda_mipmap_array.cpp cuda_mipmap_array.h
            cuda_device_math_embedded.inl.h
            cuda_device_resource_embedded.inl.h
            cuda_codegen.cpp cuda_codegen.h
            cuda_compiler.cpp cuda_compiler.h
            cuda_accel.cpp cuda_accel.h
            cuda_mesh.cpp cuda_mesh.h
            cuda_shader.cpp cuda_shader.h
            cuda_heap.cpp cuda_heap.h
            )

    luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})
    target_link_libraries(luisa-compute-backend-cuda PRIVATE CUDA::cuda_driver CUDA::nvrtc)
    if(WIN32)
        target_link_libraries(luisa-compute-backend-cuda PRIVATE cfgmgr32)
    endif()
    target_include_directories(luisa-compute-backend-cuda PRIVATE optix)
    set_target_properties(luisa-compute-backend-cuda PROPERTIES UNITY_BUILD OFF)
endif ()
