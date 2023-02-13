set(LUISA_COMPUTE_CPU_SOURCES
        cpu_device.h cpu_device.cpp)
luisa_compute_add_backend(cpu SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})
target_link_libraries(luisa-compute-backend-cpu PRIVATE luisa-compute-rust-meta)