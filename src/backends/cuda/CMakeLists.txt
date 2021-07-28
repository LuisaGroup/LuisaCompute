find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)

    message(STATUS "Build with CUDA backend: ${CUDAToolkit_VERSION}")

    set(LUISA_COMPUTE_CUDA_SOURCES
            cuda_device.cpp cuda_device.h)

    luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})
    target_link_libraries(luisa-compute-backend-cuda PRIVATE CUDA::cuda_driver CUDA::nvrtc)
    set_target_properties(luisa-compute-backend-cuda PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS OFF
            UNITY_BUILD OFF)

endif ()
