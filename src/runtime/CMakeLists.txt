set(LUISA_COMPUTE_RUNTIME_SOURCES
    context.cpp context.h
    device.cpp device.h
    kernel.cpp kernel.h
    buffer.cpp buffer.h)

add_library(luisa-compute-runtime SHARED ${LUISA_COMPUTE_RUNTIME_SOURCES})
target_link_libraries(luisa-compute-runtime PUBLIC luisa-compute-core)
set_target_properties(luisa-compute-runtime PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
