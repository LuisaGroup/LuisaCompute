if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    enable_language(OBJC)
    enable_language(OBJCXX)

    message(STATUS "Build with Metal 4 AIR backend")

    file(GLOB LUISA_COMPUTE_METAL4_DEVICE_LIB_SOURCES CONFIGURE_DEPENDS metal_builtin/*.metal)
    file(GLOB LUISA_COMPUTE_METAL4_TEX_COMPRESS_SOURCES CONFIGURE_DEPENDS metal-tex-compress/*.patched.metal)
    add_custom_target(luisa-compute-backend-metal4-embed
            COMMAND $<TARGET_FILE:luisa_embed_device_lib> ${LUISA_COMPUTE_METAL4_DEVICE_LIB_SOURCES} --prefix luisa_compute_ -o metal_builtin_embedded.cpp -h metal_builtin_embedded.h
            COMMAND $<TARGET_FILE:luisa_embed_device_lib> ${LUISA_COMPUTE_METAL4_TEX_COMPRESS_SOURCES} --prefix luisa_compute_metal_texture_compress_ -o metal_tex_compress_embedded.cpp -h metal_tex_compress_embedded.h
            BYPRODUCTS ${CMAKE_CURRENT_LIST_DIR}/metal_builtin_embedded.cpp ${CMAKE_CURRENT_LIST_DIR}/metal_builtin_embedded.h ${CMAKE_CURRENT_LIST_DIR}/metal_tex_compress_embedded.cpp ${CMAKE_CURRENT_LIST_DIR}/metal_tex_compress_embedded.h
            DEPENDS luisa_embed_device_lib ${CMAKE_CURRENT_LIST_FILE}
            SOURCES ${LUISA_COMPUTE_METAL4_DEVICE_LIB_SOURCES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Embedding Metal 4 built-in sources..."
            VERBATIM)

    set(METAL4_SOURCES

            # common
            ../common/default_binary_io.cpp ../common/default_binary_io.h
            ../common/resource_tracker.h

            # device lib
            metal_builtin_embedded.cpp metal_builtin_embedded.h
            metal_tex_compress_embedded.cpp metal_tex_compress_embedded.h

            # Metal runtime
            metal_api.cpp metal_api.h
            metal_device.cpp metal_device.h
            metal_compiler.cpp metal_compiler.h
            metal_texture.cpp metal_texture.h
            metal_depth_buffer.cpp metal_depth_buffer.h
            metal_metallib.cpp metal_metallib.h
            metal_raster_archive.cpp metal_raster_archive.h
            metal_raster_ext.cpp metal_raster_ext.h
            metal_raster_shader.cpp metal_raster_shader.h
            metal_stream.cpp metal_stream.h
            metal_event.cpp metal_event.h
            metal_bindless_array.cpp metal_bindless_array.h
            metal_swapchain.cpp metal_swapchain.h metal_swapchain.mm
            metal_command_encoder.cpp metal_command_encoder.h
            metal_shader.cpp metal_shader.h
            metal_shader_metadata.cpp metal_shader_metadata.h
            metal_stage_buffer_pool.cpp metal_stage_buffer_pool.h
            metal_callback_context.cpp metal_callback_context.h
            metal_acceleration_structure_build.cpp metal_acceleration_structure_build.h
            metal_primitive.cpp metal_primitive.h
            metal_procedural_primitive.cpp metal_procedural_primitive.h
            metal_curve.cpp metal_curve.h
            metal_mesh.cpp metal_mesh.h
            metal_motion_instance.cpp metal_motion_instance.h
            metal_accel.cpp metal_accel.h
            metal_accel_motion.cpp
            metal_buffer.cpp metal_buffer.h
            metal_denoiser.cpp metal_denoiser.h
            metal_dstorage.cpp metal_dstorage.h
            metal_debug_capture.cpp metal_debug_capture.h
            metal_pinned_memory.cpp metal_pinned_memory.h
            metal_tex_compress.cpp metal_tex_compress.h

            # XIR -> LLVM -> AIR
            metal_xir_pipeline.cpp metal_xir_pipeline.h
            metal_air_pipeline.cpp metal_air_pipeline.h
            llvm_codegen/metal_codegen_llvm.cpp
            llvm_codegen/metal_codegen_llvm.h
            llvm_codegen/metal_codegen_llvm_impl.h
            llvm_codegen/metal_codegen_llvm_impl.cpp
            llvm_codegen/metal_codegen_llvm_type.cpp
            llvm_codegen/metal_codegen_llvm_function.cpp
            llvm_codegen/metal_codegen_llvm_access.cpp
            llvm_codegen/metal_codegen_llvm_resource.cpp
            llvm_codegen/metal_codegen_llvm_atomic.cpp
            llvm_codegen/metal_codegen_llvm_arithmetic.cpp
            llvm_codegen/metal_codegen_llvm_raster.cpp
            llvm_codegen/metal_codegen_llvm_metadata.cpp
            llvm_codegen/metal_codegen_llvm_preflight.cpp)

    luisa_compute_add_backend(metal4 SOURCES ${METAL4_SOURCES})
    add_dependencies(luisa-compute-backend-metal4
            luisa-compute-backend-metal4-embed)

    execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-version
            OUTPUT_VARIABLE LUISA_METAL4_AIR_SDK_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE LUISA_METAL4_AIR_SDK_VERSION_RESULT)
    set(LUISA_METAL4_COMPILE_DEFINITIONS
            LUISA_METAL4_BACKEND=1)
    if (LUISA_METAL4_AIR_SDK_VERSION_RESULT EQUAL 0 AND
            NOT LUISA_METAL4_AIR_SDK_VERSION STREQUAL "")
        list(APPEND LUISA_METAL4_COMPILE_DEFINITIONS
                LUISA_METAL_AIR_SDK_VERSION="${LUISA_METAL4_AIR_SDK_VERSION}")
    endif ()
    target_compile_definitions(luisa-compute-backend-metal4 PRIVATE
            ${LUISA_METAL4_COMPILE_DEFINITIONS})

    target_link_libraries(luisa-compute-backend-metal4 PRIVATE
            luisa-compute-xir
            luisa-compute-ext-llvm-downgrade
            luisa-compute-metal-cpp)
    luisa_compute_link_llvm_into_backend(metal4 REQUIRED COMPONENTS
            core support bitreader bitwriter irreader linker passes analysis transformutils
            ipo scalaropts instcombine vectorize)

    if (TARGET luisa-compute-oidn-ext)
        target_link_libraries(luisa-compute-backend-metal4 PRIVATE
                luisa-compute-oidn-ext)
    endif ()

    find_library(COMPRESSION_LIB compression)
    target_link_libraries(luisa-compute-backend-metal4 PRIVATE
            ${COMPRESSION_LIB}
            "-framework Foundation"
            "-framework Metal"
            "-framework QuartzCore"
            "-framework AppKit")

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "Metal 4 AIR backend requires Apple platforms with clang.")
else ()
    message(WARNING "Metal 4 AIR backend requires Apple platforms with clang. The backend will be disabled.")
endif ()
