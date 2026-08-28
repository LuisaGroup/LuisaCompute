if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    enable_language(OBJC)
    enable_language(OBJCXX)

    message(STATUS "Build with Metal 4 AIR backend")

    set(LUISA_COMPUTE_METAL4_BUILD_RUNTIME ON)
    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(LUISA_METAL4_AIR_SDK iphoneos)
        set(LUISA_METAL4_TEX_COMPRESS_DEPLOYMENT_FLAG
                "-mios-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
        message(STATUS "Metal 4 iOS: build the static AIR runtime backend")
    else ()
        set(LUISA_METAL4_AIR_SDK macosx)
        set(LUISA_METAL4_TEX_COMPRESS_DEPLOYMENT_FLAG
                "-mmacosx-version-min=26.0")
    endif ()

    # Texture compression is a fixed support library rather than user shader
    # codegen. Compile it to target-specific metallibs at build time so the
    # runtime backend never contains or invokes an MSL compiler path.
    file(GLOB LUISA_COMPUTE_METAL4_TEX_COMPRESS_SOURCES
            CONFIGURE_DEPENDS metal-tex-compress/*.patched.metal)
    set(LUISA_COMPUTE_METAL4_TEX_COMPRESS_OUTPUT_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/metal_tex_compress_air")
    set(LUISA_COMPUTE_METAL4_TEX_COMPRESS_EMBEDDED_SOURCES)
    foreach (source IN LISTS LUISA_COMPUTE_METAL4_TEX_COMPRESS_SOURCES)
        get_filename_component(source_stem "${source}" NAME_WE)
        string(REPLACE ".patched" "" kernel_id "${source_stem}")
        set(air "${LUISA_COMPUTE_METAL4_TEX_COMPRESS_OUTPUT_DIR}/${kernel_id}.air")
        set(metallib "${LUISA_COMPUTE_METAL4_TEX_COMPRESS_OUTPUT_DIR}/${kernel_id}.metallib")
        set(embedded "${LUISA_COMPUTE_METAL4_TEX_COMPRESS_OUTPUT_DIR}/${kernel_id}.inc")
        set(symbol "luisa_compute_metal_texture_compress_${kernel_id}_metallib")
        add_custom_command(
                OUTPUT "${embedded}"
                BYPRODUCTS "${air}" "${metallib}"
                COMMAND ${CMAKE_COMMAND} -E make_directory
                "${LUISA_COMPUTE_METAL4_TEX_COMPRESS_OUTPUT_DIR}"
                COMMAND xcrun --sdk ${LUISA_METAL4_AIR_SDK} metal
                -std=metal4.0 -ffast-math
                ${LUISA_METAL4_TEX_COMPRESS_DEPLOYMENT_FLAG}
                -c "${source}" -o "${air}"
                COMMAND xcrun --sdk ${LUISA_METAL4_AIR_SDK} metallib
                "${air}" -o "${metallib}"
                COMMAND /usr/bin/xxd -i -n "${symbol}"
                "${metallib}" "${embedded}"
                DEPENDS "${source}"
                COMMENT "Building Metal 4 AIR texture kernel ${kernel_id}"
                VERBATIM)
        list(APPEND LUISA_COMPUTE_METAL4_TEX_COMPRESS_EMBEDDED_SOURCES
                "${embedded}")
    endforeach ()
    set_source_files_properties(
            ${LUISA_COMPUTE_METAL4_TEX_COMPRESS_EMBEDDED_SOURCES}
            PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
    add_custom_target(luisa-compute-backend-metal4-embed
            DEPENDS ${LUISA_COMPUTE_METAL4_TEX_COMPRESS_EMBEDDED_SOURCES}
            SOURCES ${LUISA_COMPUTE_METAL4_TEX_COMPRESS_SOURCES})

    set(METAL4_AIR_CODEGEN_SOURCES
            metal_metallib.cpp metal_metallib.h
            metal_raster_archive.cpp metal_raster_archive.h
            metal_xir_pipeline.cpp metal_xir_pipeline.h
            metal_air_pipeline.cpp metal_air_pipeline.h
            llvm_codegen/metal_codegen_llvm.cpp
            llvm_codegen/metal_codegen_llvm.h
            llvm_codegen/metal_codegen_llvm_builtin.cpp
            llvm_codegen/metal_codegen_llvm_builtin.h
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

    list(APPEND METAL4_AIR_CODEGEN_SOURCES
            metal_builtin_air.cpp metal_builtin_air.h)

    set(METAL4_SOURCES

            # common
            ../common/default_binary_io.cpp ../common/default_binary_io.h
            ../common/resource_tracker.h

            # precompiled texture compression AIR libraries
            ${LUISA_COMPUTE_METAL4_TEX_COMPRESS_EMBEDDED_SOURCES}

            # Metal runtime
            metal_api.cpp metal_api.h
            metal_device.cpp metal_device.h
            metal_compiler.cpp metal_compiler.h
            metal_texture.cpp metal_texture.h
            metal_depth_buffer.cpp metal_depth_buffer.h
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
            metal_tex_compress.cpp metal_tex_compress.h)

    if (LUISA_COMPUTE_METAL4_BUILD_RUNTIME)
        luisa_compute_add_backend(metal4 SOURCES ${METAL4_SOURCES})
        add_dependencies(luisa-compute-backend-metal4
                luisa-compute-backend-metal4-embed)
        target_include_directories(luisa-compute-backend-metal4 PRIVATE
                "${CMAKE_CURRENT_BINARY_DIR}")
    endif ()

    execute_process(
            COMMAND xcrun --sdk ${LUISA_METAL4_AIR_SDK} --show-sdk-version
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
    if (LUISA_COMPUTE_METAL4_BUILD_RUNTIME)
        target_compile_definitions(luisa-compute-backend-metal4 PRIVATE
                ${LUISA_METAL4_COMPILE_DEFINITIONS})
    endif ()

    add_library(luisa-compute-metal4-air-codegen STATIC
            ${METAL4_AIR_CODEGEN_SOURCES})
    set_target_properties(luisa-compute-metal4-air-codegen PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            FOLDER backends)
    target_compile_definitions(luisa-compute-metal4-air-codegen PRIVATE
            ${LUISA_METAL4_COMPILE_DEFINITIONS})
    target_include_directories(luisa-compute-metal4-air-codegen PUBLIC
            ${CMAKE_CURRENT_LIST_DIR})
    target_link_libraries(luisa-compute-metal4-air-codegen
            PUBLIC
            luisa-compute-ast
            luisa-compute-runtime
            luisa-compute-xir
            luisa-compute-ext-llvm-downgrade
            luisa-compute-metal-cpp
            "-framework Foundation")
    luisa_compute_link_llvm_into_backend(metal4-air-codegen
            TARGET luisa-compute-metal4-air-codegen
            REQUIRED COMPONENTS
            core support bitreader bitwriter irreader linker passes analysis transformutils
            ipo scalaropts instcombine vectorize)

    if (LUISA_COMPUTE_METAL4_BUILD_RUNTIME)
        target_link_libraries(luisa-compute-backend-metal4 PRIVATE
                luisa-compute-metal4-air-codegen
                luisa-compute-metal-cpp)
    endif ()

    if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        add_executable(luisa-metal4-ios-path-tracer-aot
                tools/ios_path_tracing_aot.cpp
                metal_api.cpp)
        target_compile_definitions(luisa-metal4-ios-path-tracer-aot PRIVATE
                ${LUISA_METAL4_COMPILE_DEFINITIONS})
        target_link_libraries(luisa-metal4-ios-path-tracer-aot PRIVATE
                luisa-compute-metal4-air-codegen
                luisa-compute-dsl
                luisa-compute-metal-cpp
                "-framework Foundation"
                "-framework Metal"
                "-framework QuartzCore")
    else ()
        set(LUISA_IOS_DEVELOPMENT_TEAM "" CACHE STRING
                "Apple development team used to sign the iOS AIR test app")
        set(LUISA_IOS_PATH_TRACING_SPP 8 CACHE STRING
                "Samples per pixel rendered by the iOS AIR test app")
        add_executable(luisa-metal4-ios-device-air-path-tracer MACOSX_BUNDLE
                tools/ios_path_tracer_app/main.mm
                tools/ios_path_tracing_kernel.h)
        target_compile_features(
                luisa-metal4-ios-device-air-path-tracer PRIVATE cxx_std_20)
        target_compile_options(
                luisa-metal4-ios-device-air-path-tracer PRIVATE -fblocks)
        target_include_directories(
                luisa-metal4-ios-device-air-path-tracer PRIVATE
                ${CMAKE_CURRENT_LIST_DIR}
                ${CMAKE_CURRENT_LIST_DIR}/tools)
        target_compile_definitions(
                luisa-metal4-ios-device-air-path-tracer PRIVATE
                ${LUISA_METAL4_COMPILE_DEFINITIONS}
                LUISA_IOS_ON_DEVICE_AIR=1
                LUISA_IOS_RUNTIME_DEVICE=1
                LUISA_IOS_PATH_TRACING_SPP=${LUISA_IOS_PATH_TRACING_SPP}
                LUISA_IOS_AIR_DEPLOYMENT_VERSION="${CMAKE_OSX_DEPLOYMENT_TARGET}"
                LUISA_IOS_AIR_SDK_VERSION="${LUISA_METAL4_AIR_SDK_VERSION}")
        target_link_libraries(
                luisa-metal4-ios-device-air-path-tracer PRIVATE
                luisa-compute-metal4-air-codegen
                luisa-compute-backend-metal4
                luisa-compute-dsl
                luisa-compute-metal-cpp
                "-framework Foundation"
                "-framework UIKit"
                "-framework Metal"
                "-framework QuartzCore"
                "-framework CoreGraphics"
                "-framework ImageIO")
        set_target_properties(
                luisa-metal4-ios-device-air-path-tracer PROPERTIES
                MACOSX_BUNDLE_INFO_PLIST
                "${CMAKE_CURRENT_LIST_DIR}/tools/ios_path_tracer_app/Info.plist.in"
                MACOSX_BUNDLE_BUNDLE_NAME "Luisa Metal4 Device AIR"
                MACOSX_BUNDLE_GUI_IDENTIFIER
                "org.luisa-compute.metal4-path-tracer"
                MACOSX_BUNDLE_BUNDLE_VERSION "2"
                MACOSX_BUNDLE_SHORT_VERSION_STRING "2.0"
                XCODE_GENERATE_SCHEME TRUE
                XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER
                "org.luisa-compute.metal4-path-tracer"
                XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
                XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET
                "${CMAKE_OSX_DEPLOYMENT_TARGET}"
                XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1"
                XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS "iphoneos"
                XCODE_ATTRIBUTE_SUPPORTS_MACCATALYST NO
                XCODE_ATTRIBUTE_ENABLE_BITCODE NO)
        if (LUISA_IOS_DEVELOPMENT_TEAM)
            set_target_properties(
                    luisa-metal4-ios-device-air-path-tracer PROPERTIES
                    XCODE_ATTRIBUTE_DEVELOPMENT_TEAM
                    "${LUISA_IOS_DEVELOPMENT_TEAM}")
        endif ()
    endif ()

    if (LUISA_COMPUTE_METAL4_BUILD_RUNTIME)
        if (TARGET luisa-compute-oidn-ext)
            target_link_libraries(luisa-compute-backend-metal4 PRIVATE
                    luisa-compute-oidn-ext)
        endif ()

        find_library(COMPRESSION_LIB compression)
        target_link_libraries(luisa-compute-backend-metal4 PRIVATE
                ${COMPRESSION_LIB}
                "-framework Foundation"
                "-framework Metal"
                "-framework QuartzCore")
        if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
            target_link_libraries(luisa-compute-backend-metal4 PRIVATE
                    "-framework UIKit")
        else ()
            target_link_libraries(luisa-compute-backend-metal4 PRIVATE
                    "-framework AppKit")
        endif ()
    endif ()

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "Metal 4 AIR backend requires Apple platforms with clang.")
else ()
    message(WARNING "Metal 4 AIR backend requires Apple platforms with clang. The backend will be disabled.")
endif ()
