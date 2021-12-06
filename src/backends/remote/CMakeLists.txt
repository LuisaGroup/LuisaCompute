set(REMOTE_SOURCES
        remote_device.cpp remote_device.h)

message(STATUS "Build with remote backend")
luisa_compute_add_backend(remote SOURCES ${REMOTE_SOURCES})
