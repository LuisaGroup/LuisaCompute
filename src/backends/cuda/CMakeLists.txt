find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)

    message(STATUS "Build with CUDA backend: ${CUDAToolkit_VERSION}")

    set(LUISA_COMPUTE_CUDA_SOURCES
            cuda_error.h
            cuda_heap.cpp cuda_heap.h
            cuda_device.cpp cuda_device.h
            cuda_buffer.cpp cuda_buffer.h
            cuda_command_encoder.cpp cuda_command_encoder.h)

    luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})
    target_link_libraries(luisa-compute-backend-cuda PRIVATE CUDA::cuda_driver CUDA::nvrtc)
    target_compile_definitions(luisa-compute-backend-cuda PUBLIC LUISA_BACKEND_CUDA_ENABLED=1)
    set_target_properties(luisa-compute-backend-cuda PROPERTIES UNITY_BUILD OFF)

endif ()
