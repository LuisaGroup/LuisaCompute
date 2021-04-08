set(LUISA_COMPUTE_RUNTIME_SOURCES
    context.cpp context.h
    device.h
    command.cpp command.h
    command_buffer.cpp command_buffer.h
    pixel_format.h
    stream.cpp stream.h
    event.cpp event.h
    rg_enum.h
    rg_executor.h
    rg_node.cpp rg_node.h
    rg_system.cpp rg_system.h)

add_library(luisa-compute-runtime SHARED ${LUISA_COMPUTE_RUNTIME_SOURCES})
target_link_libraries(luisa-compute-runtime PUBLIC luisa-compute-core)
set_target_properties(luisa-compute-runtime PROPERTIES
                      WINDOWS_EXPORT_ALL_SYMBOLS ON
                      UNITY_BUILD ON)
