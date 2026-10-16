if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    enable_language(OBJC)
    enable_language(OBJCXX)

    message(STATUS "Build with Metal backend")

    # embed Metal built-in library sources with luisa_embed_device_lib
    file(GLOB LUISA_COMPUTE_METAL_DEVICE_LIB_SOURCES CONFIGURE_DEPENDS metal_builtin/*.metal)
    file(GLOB LUISA_COMPUTE_METAL_TEX_COMPRESS_SOURCES CONFIGURE_DEPENDS metal-tex-compress/*.patched.metal)
    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
        # A target-platform executable cannot run as part of a cross build.
        # Compile the source embedder for the macOS host instead.
        execute_process(
                COMMAND /usr/bin/xcrun --sdk macosx --show-sdk-path
                OUTPUT_VARIABLE _LUISA_COMPUTE_MACOS_SDK
                OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
        set(_LUISA_COMPUTE_METAL_HOST_EMBEDDER
                "${CMAKE_CURRENT_BINARY_DIR}/luisa_embed_device_lib_host")
        add_custom_command(
                OUTPUT "${_LUISA_COMPUTE_METAL_HOST_EMBEDDER}"
                COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CMAKE_CURRENT_BINARY_DIR}"
                COMMAND /usr/bin/xcrun --sdk macosx clang++
                -target "${CMAKE_HOST_SYSTEM_PROCESSOR}-apple-macos14.0"
                -isysroot "${_LUISA_COMPUTE_MACOS_SDK}"
                -std=c++20
                "${CMAKE_SOURCE_DIR}/utils/embed_device_lib.cpp"
                -o "${_LUISA_COMPUTE_METAL_HOST_EMBEDDER}"
                DEPENDS "${CMAKE_SOURCE_DIR}/utils/embed_device_lib.cpp"
                COMMENT "Building the host Metal source embedder..."
                VERBATIM)
        set(_LUISA_COMPUTE_METAL_EMBEDDER
                "${_LUISA_COMPUTE_METAL_HOST_EMBEDDER}")
        set(_LUISA_COMPUTE_METAL_EMBEDDER_DEPENDENCY
                "${_LUISA_COMPUTE_METAL_HOST_EMBEDDER}")
    else ()
        set(_LUISA_COMPUTE_METAL_EMBEDDER
                "$<TARGET_FILE:luisa_embed_device_lib>")
        set(_LUISA_COMPUTE_METAL_EMBEDDER_DEPENDENCY
                luisa_embed_device_lib)
    endif ()
    add_custom_target(luisa-compute-backend-metal-embed
            COMMAND ${_LUISA_COMPUTE_METAL_EMBEDDER} ${LUISA_COMPUTE_METAL_DEVICE_LIB_SOURCES} --prefix luisa_compute_ -o metal_builtin_embedded.cpp -h metal_builtin_embedded.h
            COMMAND ${_LUISA_COMPUTE_METAL_EMBEDDER} ${LUISA_COMPUTE_METAL_TEX_COMPRESS_SOURCES} --prefix luisa_compute_metal_texture_compress_ -o metal_tex_compress_embedded.cpp -h metal_tex_compress_embedded.h
            BYPRODUCTS ${CMAKE_CURRENT_LIST_DIR}/metal_builtin_embedded.cpp ${CMAKE_CURRENT_LIST_DIR}/metal_builtin_embedded.h ${CMAKE_CURRENT_LIST_DIR}/metal_tex_compress_embedded.cpp ${CMAKE_CURRENT_LIST_DIR}/metal_tex_compress_embedded.h
            DEPENDS ${_LUISA_COMPUTE_METAL_EMBEDDER_DEPENDENCY} ${CMAKE_CURRENT_LIST_FILE}
            SOURCES ${LUISA_COMPUTE_METAL_DEVICE_LIB_SOURCES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Embedding metal built-in sources..."
            VERBATIM)

    set(METAL_SOURCES

            # common
            ../common/default_binary_io.cpp ../common/default_binary_io.h
            ../common/resource_tracker.h

            # device lib
            metal_builtin_embedded.cpp metal_builtin_embedded.h
            metal_tex_compress_embedded.cpp metal_tex_compress_embedded.h

            # metal
            metal_api.cpp metal_api.h
            metal_static_backend.h
            metal_device.cpp metal_device.h
            metal_compiler.cpp metal_compiler.h
            metal_codegen_ast.cpp metal_codegen_ast.h
            tile/metal_tile.cpp tile/metal_tile_codegen.cpp tile/metal_tile_codegen.h
            metal_texture.cpp metal_texture.h
            metal_stream.cpp metal_stream.mm metal_stream.h
            metal_event.cpp metal_event.h
            metal_bindless_array.cpp metal_bindless_array.h
            metal_swapchain.cpp metal_swapchain.h metal_swapchain.mm
            metal_command_encoder.cpp metal_command_encoder.h
            metal_shader.cpp metal_shader.h
            metal_shader_metadata.cpp metal_shader_metadata.h
            metal_shader_printer.cpp metal_shader_printer.h
            metal_stage_buffer_pool.cpp metal_stage_buffer_pool.h
            metal_callback_context.cpp metal_callback_context.h
            metal_primitive.cpp metal_primitive.h
            metal_procedural_primitive.cpp metal_procedural_primitive.h
            metal_curve.cpp metal_curve.h
            metal_mesh.cpp metal_mesh.h
            metal_accel.cpp metal_accel.h
            metal_buffer.cpp metal_buffer.h
            metal_denoiser.cpp metal_denoiser.h
            metal_dstorage.cpp metal_dstorage.h
            metal_debug_capture.cpp metal_debug_capture.h
            metal_pinned_memory.cpp metal_pinned_memory.h
            metal_tex_compress.cpp metal_tex_compress.h)

    luisa_compute_add_backend(metal SOURCES ${METAL_SOURCES})
    add_dependencies(luisa-compute-backend-metal
            luisa-compute-backend-metal-embed)

    if (TARGET luisa-compute-tile-bridge-tirx)
        target_link_libraries(luisa-compute-backend-metal PRIVATE luisa-compute-tile-bridge-tirx)
        target_compile_definitions(luisa-compute-backend-metal PRIVATE LUISA_METAL_TILE_TIRX=1)
    endif ()

    if (TARGET luisa-compute-oidn-ext)
        target_link_libraries(luisa-compute-backend-metal PRIVATE luisa-compute-oidn-ext)
    endif ()

    find_library(COMPRESSION_LIB compression)
    target_link_libraries(luisa-compute-backend-metal PRIVATE
            luisa-compute-tile
            luisa-compute-xir
            luisa-compute-metal-cpp
            ${COMPRESSION_LIB}
            "-framework Foundation"
            "-framework Metal"
            "-framework QuartzCore")
    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
        target_link_libraries(luisa-compute-backend-metal PRIVATE
                "-framework UIKit")
    else ()
        target_link_libraries(luisa-compute-backend-metal PRIVATE
                "-framework AppKit")
    endif ()

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "Metal backend requires Apple platforms with clang.")
else ()
    message(WARNING "Metal backend requires Apple platforms with clang. The backend will be disabled.")
endif ()
