set(LUISA_COMPUTE_CPU_SOURCES
        ../common/rust_device_common.cpp ../common/rust_device_common.h
        cpu_device.h cpu_device.cpp)
luisa_compute_add_backend(cpu SOURCES ${LUISA_COMPUTE_CPU_SOURCES})
target_link_libraries(luisa-compute-backend-cpu PRIVATE
        luisa-compute-vulkan-swapchain
        luisa-compute-rust-meta
        luisa_compute_backend_impl)
