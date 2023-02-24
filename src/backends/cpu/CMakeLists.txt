set(LUISA_COMPUTE_CPU_SOURCES
        cpu_device.h cpu_device.cpp)
luisa_compute_add_backend(cpu SOURCES ${LUISA_COMPUTE_CPU_SOURCES})
target_link_libraries(lc-backend-cpu PRIVATE luisa-compute-rust-meta)