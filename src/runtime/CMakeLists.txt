set(LUISA_COMPUTE_RUNTIME_SOURCES
    context.cpp context.h
    device.cpp device.h
    command.cpp command.h
    command_list.cpp command_list.h
    pixel.h
    stream.cpp stream.h
    event.cpp event.h
    buffer.h
    image.h
    volume.h
    texture_sampler.h
    texture_heap.cpp texture_heap.h
    shader.h)

add_library(luisa-compute-runtime SHARED ${LUISA_COMPUTE_RUNTIME_SOURCES}  )
target_link_libraries(luisa-compute-runtime PUBLIC luisa-compute-ast)
set_target_properties(luisa-compute-runtime PROPERTIES
                      WINDOWS_EXPORT_ALL_SYMBOLS ON
                      UNITY_BUILD ON)
