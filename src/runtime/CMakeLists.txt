set(LUISA_COMPUTE_RUNTIME_SOURCES
        bindless_array.cpp bindless_array.h
        buffer.h
        command.cpp command.h
        command_buffer.cpp command_buffer.h
        command_list.cpp command_list.h
        command_scheduler.cpp command_scheduler.h
        context.cpp context.h
        device.h
        event.cpp event.h
        image.h
        mipmap.h
        pixel.h
        resource.cpp resource.h
        resource_tracker.cpp resource_tracker.h
        sampler.h
        shader.cpp shader.h
        stream.cpp stream.h
        swap_chain.cpp swap_chain.h
        volume.h)

add_library(luisa-compute-runtime SHARED ${LUISA_COMPUTE_RUNTIME_SOURCES})
target_link_libraries(luisa-compute-runtime PUBLIC luisa-compute-ast luisa-compute-ir)
target_compile_definitions(luisa-compute-runtime PRIVATE LC_RUNTIME_EXPORT_DLL=1)
set_target_properties(luisa-compute-runtime PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

luisa_compute_install(runtime SOURCES ${LUISA_COMPUTE_RUNTIME_SOURCES})
