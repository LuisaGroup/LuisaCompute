set(LUISA_COMPUTE_RUNTIME_SOURCES
    context.cpp context.h
    device.cpp device.h
    command.cpp command.h
    command_buffer.cpp command_buffer.h
    pixel.h
    stream.cpp stream.h
    event.cpp event.h
    buffer.h
    image.h
    volume.h
    texture.cpp texture.h)

add_library(luisa-compute-runtime SHARED ${LUISA_COMPUTE_RUNTIME_SOURCES})
target_link_libraries(luisa-compute-runtime PUBLIC luisa-compute-core)
set_target_properties(luisa-compute-runtime PROPERTIES
                      WINDOWS_EXPORT_ALL_SYMBOLS ON
                      UNITY_BUILD ON)
