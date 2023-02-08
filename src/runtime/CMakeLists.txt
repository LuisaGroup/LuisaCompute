set(LUISA_COMPUTE_RUNTIME_SOURCES
        bindless_array.cpp bindless_array.h
        buffer.cpp buffer.h
        command.cpp command.h
        command_buffer.cpp command_buffer.h
        command_list.cpp command_list.h
        context.cpp context.h
        context_paths.h
        custom_pass.cpp custom_pass.h
        custom_struct.cpp custom_struct.h
        depth_format.h
        device_interface.cpp device_interface.h
        device.cpp device.h
        dynamic_buffer.cpp dynamic_buffer.h
        dynamic_struct.h
        event.cpp event.h
        image.cpp image.h
        mipmap.cpp mipmap.h
        pixel.cpp pixel.h
        resource.cpp resource.h
        resource_tracker.cpp resource_tracker.h
        sampler.h
        shader.h
        stream.cpp stream.h
        stream_tag.h
        swap_chain.cpp swap_chain.h
        volume.cpp volume.h)

add_library(luisa-compute-runtime SHARED ${LUISA_COMPUTE_RUNTIME_SOURCES})
target_link_libraries(luisa-compute-runtime PUBLIC luisa-compute-ast
        # luisa-compute-ir # this is so broken, disable it for now
        )
target_compile_definitions(luisa-compute-runtime PRIVATE LC_RUNTIME_EXPORT_DLL=1)
set_target_properties(luisa-compute-runtime PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

luisa_compute_install(runtime SOURCES ${LUISA_COMPUTE_RUNTIME_SOURCES})
