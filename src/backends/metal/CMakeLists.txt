if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    enable_language(OBJC)
    enable_language(OBJCXX)

    message(STATUS "Build with Metal backend")

    set(METAL_SOURCES

            # common
            ../common/default_binary_io.cpp ../common/default_binary_io.h
            ../common/resource_tracker.h
            ../common/string_scratch.cpp ../common/string_scratch.h

            # metal-cpp
            metal-cpp/SingleHeader/Metal.hpp

            # metal
            metal_api.cpp metal_api.h
            metal_builtin_embedded.cpp metal_builtin_embedded.h
            metal_device.cpp metal_device.h
            metal_compiler.cpp metal_compiler.h
            metal_codegen_ast.cpp metal_codegen_ast.h
            metal_texture.cpp metal_texture.h
            metal_stream.cpp metal_stream.mm metal_stream.h
            metal_event.cpp metal_event.h
            metal_bindless_array.cpp metal_bindless_array.h
            metal_swapchain.cpp metal_swapchain.h metal_swapchain.mm
            metal_command_encoder.cpp metal_command_encoder.h
            metal_shader.cpp metal_shader.h
            metal_shader_metadata.cpp metal_shader_metadata.h
            metal_stage_buffer_pool.cpp metal_stage_buffer_pool.h
            metal_callback_context.cpp metal_callback_context.h
            metal_primitive.cpp metal_primitive.h
            metal_procedural_primitive.cpp metal_procedural_primitive.h
            metal_mesh.cpp metal_mesh.h
            metal_accel.cpp metal_accel.h
            metal_buffer.cpp metal_buffer.h
            metal_dstorage.cpp metal_dstorage.h
            metal_debug_capture.cpp metal_debug_capture.h
            metal_pinned_memory.cpp metal_pinned_memory.h)

    if (LUISA_COMPUTE_ENABLE_RUST)
        list(APPEND METAL_SOURCES metal_codegen_ir.cpp metal_codegen_ir.h)
    endif ()

    luisa_compute_add_backend(metal SOURCES ${METAL_SOURCES})

    find_library(COMPRESSION_LIB compression)
    target_link_libraries(luisa-compute-backend-metal PRIVATE
            ${COMPRESSION_LIB}
            "-framework Foundation"
            "-framework Metal"
            "-framework QuartzCore"
            "-framework AppKit")

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "Metal backend requires Apple platforms with clang.")
else ()
    message(WARNING "Metal backend requires Apple platforms with clang. The backend will be disabled.")
endif ()
