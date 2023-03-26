set(LUISA_COMPUTE_REMOTE_SOURCES
        remote_device.h remote_device.cpp)
luisa_compute_add_backend(remote SOURCES ${LUISA_COMPUTE_REMOTE_SOURCES})
target_link_libraries(lc-backend-remote PRIVATE luisa-compute-device-common luisa-compute-rust-meta)