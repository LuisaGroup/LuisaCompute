set(LUISA_COMPUTE_REMOTE_SOURCES
        ../common/rust_device_common.cpp ../common/rust_device_common.h
        remote_device.h remote_device.cpp)
luisa_compute_add_backend(remote SOURCES ${LUISA_COMPUTE_REMOTE_SOURCES})
target_link_libraries(luisa-compute-backend-remote PRIVATE
        luisa-compute-vulkan-swapchain
        luisa-compute-rust-meta
        luisa_compute_backend_impl)

# add_dependencies(luisa-compute-backend-remote luisa-compute-rust-copy)
