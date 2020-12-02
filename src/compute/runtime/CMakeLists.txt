set(LUISA_COMPUTE_RUNTIME_SOURCES
    device.cpp device.h
    kernel.cpp kernel.h)

add_library(luisa-compute-runtime SHARED ${LUISA_COMPUTE_RUNTIME_SOURCES})
target_link_libraries(luisa-compute-runtime PUBLIC luisa-core luisa-compute-include)
