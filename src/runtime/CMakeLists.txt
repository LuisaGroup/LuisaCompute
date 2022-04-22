set(LUISA_COMPUTE_RUNTIME_SOURCES
        context.cpp context.h
        device.h
        command.cpp command.h
        command_list.cpp command_list.h
        command_buffer.cpp command_buffer.h
        pixel.h
        stream.cpp stream.h
        event.cpp event.h
        buffer.h
        image.h
        volume.h
        bindless_array.cpp bindless_array.h
        shader.h
        sampler.h
        resource.cpp resource.h
        mipmap.h
        command_reorder_visitor.h command_reorder_visitor.cpp
        swap_chain.h swap_chain.cpp
        resource_tracker.cpp resource_tracker.h
        command_scheduler.cpp command_scheduler.h)

add_library(luisa-compute-runtime SHARED ${LUISA_COMPUTE_RUNTIME_SOURCES})
target_link_libraries(luisa-compute-runtime PUBLIC luisa-compute-ast)
target_compile_definitions(luisa-compute-runtime PRIVATE LC_RUNTIME_EXPORT_DLL=1)
set_target_properties(luisa-compute-runtime PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})
