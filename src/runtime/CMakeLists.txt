set(LUISA_COMPUTE_RUNTIME_RTX_SOURCES
        rtx/aabb.h
        rtx/accel.cpp
        rtx/accel.h
        rtx/hit.h
        rtx/mesh.cpp
        rtx/mesh.h
        rtx/procedural_primitive.cpp
        rtx/procedural_primitive.h
        rtx/ray.h
        rtx/triangle.h)

set(LUISA_COMPUTE_RUNTIME_RASTER_SOURCES
        raster/depth_buffer.cpp
        raster/depth_buffer.h
        raster/dsl.h
        raster/raster.cpp
        raster/raster_kernel.h
        raster/raster_scene.h
        raster/raster_shader.h
        raster/raster_state.h
        raster/vertex_attribute.h
        raster/viewport.h)

set(LUISA_COMPUTE_RUNTIME_SOURCES
        bindless_array.cpp bindless_array.h
        buffer.cpp buffer.h
        buffer_arena.h
        command.h
        command_encoder.cpp command_encoder.h
        command_buffer.cpp command_buffer.h
        command_list.cpp command_list.h
        context.cpp context.h
        context_paths.h
        custom_pass.cpp custom_pass.h
        depth_format.h
        device_interface.h
        device.cpp device.h
        event.cpp event.h
        image.cpp image.h
        mipmap.cpp mipmap.h
        pixel.cpp pixel.h
        resource.cpp resource.h
        sampler.h
        shader.h
        stream.cpp stream.h
        stream_tag.h
        swap_chain.cpp swap_chain.h
        volume.cpp volume.h
        ${LUISA_COMPUTE_RUNTIME_RTX_SOURCES}
        ${LUISA_COMPUTE_RUNTIME_RASTER_SOURCES})

add_library(luisa-compute-runtime SHARED ${LUISA_COMPUTE_RUNTIME_SOURCES})
target_link_libraries(luisa-compute-runtime PUBLIC luisa-compute-ast luisa-compute-ir) # this is so broken, disable it for now

target_compile_definitions(luisa-compute-runtime PRIVATE LC_RUNTIME_EXPORT_DLL=1)
set_target_properties(luisa-compute-runtime PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

luisa_compute_install(runtime SOURCES ${LUISA_COMPUTE_RUNTIME_SOURCES})
