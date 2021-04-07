set(LUISA_COMPUTE_RUNTIME_SOURCES
    context.cpp context.h
    device.h
    command.h command.cpp
    command_buffer.cpp command_buffer.h
    pixel_format.h
    stream.cpp stream.h)

add_library(luisa-compute-runtime SHARED ${LUISA_COMPUTE_RUNTIME_SOURCES})
target_link_libraries(luisa-compute-runtime PUBLIC luisa-compute-core)
set_target_properties(luisa-compute-runtime PROPERTIES
                      WINDOWS_EXPORT_ALL_SYMBOLS ON
                      UNITY_BUILD ON)
