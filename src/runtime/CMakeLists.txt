set(LUISA_COMPUTE_RUNTIME_RTX_SOURCES
        rtx/accel.cpp
        rtx/mesh.cpp
        rtx/procedural_primitive.cpp
        rtx/motion_instance.cpp
        rtx/curve.cpp)

set(LUISA_COMPUTE_RUNTIME_RASTER_SOURCES
        raster/depth_buffer.cpp
        raster/raster.cpp)

set(LUISA_COMPUTE_RUNTIME_RHI_SOURCES
        rhi/command_encoder.cpp
        rhi/device_interface.cpp
        rhi/pixel.cpp
        rhi/resource.cpp)

set(LUISA_COMPUTE_RUNTIME_REMOTE_SOURCES
        remote/client_interface.cpp
        remote/server_interface.cpp)

set(LUISA_COMPUTE_RUNTIME_SOURCES
        bindless_array.cpp
        buffer.cpp
        byte_buffer.cpp
        command_list.cpp
        context.cpp
        device.cpp
        dispatch_buffer.cpp
        event.cpp
        image.cpp
        mipmap.cpp
        sparse_buffer.cpp
        sparse_texture.cpp
        sparse_heap.cpp
        sparse_command_list.cpp
        stream.cpp
        swapchain.cpp
        volume.cpp
        ${LUISA_COMPUTE_RUNTIME_RHI_SOURCES}
        ${LUISA_COMPUTE_RUNTIME_RTX_SOURCES}
        ${LUISA_COMPUTE_RUNTIME_RASTER_SOURCES}
        ${LUISA_COMPUTE_RUNTIME_REMOTE_SOURCES})

add_library(luisa-compute-runtime SHARED ${LUISA_COMPUTE_RUNTIME_SOURCES})
target_link_libraries(luisa-compute-runtime PUBLIC luisa-compute-ast luisa-compute-ir luisa-compute-ir-v2)
target_precompile_headers(luisa-compute-runtime PRIVATE pch.h)

target_compile_definitions(luisa-compute-runtime PRIVATE LC_RUNTIME_EXPORT_DLL=1)
set_target_properties(luisa-compute-runtime PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME lc-runtime)

luisa_compute_install(runtime SOURCES ${LUISA_COMPUTE_RUNTIME_SOURCES})
