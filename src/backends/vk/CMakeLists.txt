if (LUISA_COMPUTE_ENABLE_VULKAN)
    message(STATUS "Build with Vulkan backend")
    file(GLOB LUISA_COMPUTE_VK_SOURCES CONFIGURE_DEPENDS *.cpp *.h)
    luisa_compute_add_backend(vk SOURCES ${LUISA_COMPUTE_VK_SOURCES} ../common/default_binary_io.cpp)
    target_link_libraries(luisa-compute-backend-vk PUBLIC luisa-compute-ext-volk luisa-compute-hlsl-codegen)
    if (NOT LUISA_COMPUTE_VULKAN_ENABLE_DXC_COMPATIBILITY)
        target_compile_definitions(
                luisa-compute-backend-vk PRIVATE LC_NO_HLSL_BUILTIN=1)
        if (NOT LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV AND
                NOT LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV)
            message(FATAL_ERROR
                    "The Vulkan backend needs a native SPIR-V codegen route "
                    "when DXC compatibility is disabled.")
        endif ()
    endif ()

    # Compile backend-private kernels with the bundled, version-pinned
    # glslang, then embed the generated SPIR-V exactly like HIP embeds its
    # build-time motion-mesh fat binary. All artifacts stay in the build tree.
    if (NOT TARGET glslang-standalone)
        message(FATAL_ERROR
                "Vulkan built-in SPIR-V generation requires the bundled "
                "glslang-standalone target.")
    endif ()
    set(_vk_builtin_source_dir "${CMAKE_CURRENT_LIST_DIR}/builtin")
    set(_vk_builtin_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/builtin")
    set(_vk_builtin_layout_definitions
            "${CMAKE_CURRENT_LIST_DIR}/../common/indirect_dispatch_layout.def"
            "${_vk_builtin_source_dir}/vulkan_accel_update_layout.def")
    file(GLOB _vk_checked_in_spirv CONFIGURE_DEPENDS
            "${_vk_builtin_source_dir}/*.spv"
            "${CMAKE_CURRENT_LIST_DIR}/../common/hlsl/builtin/*.spv")
    if (_vk_checked_in_spirv)
        message(FATAL_ERROR
                "Vulkan built-in SPIR-V must be generated in the build tree, "
                "not checked into the source tree: ${_vk_checked_in_spirv}")
    endif ()

    function(_luisa_compute_compile_vulkan_builtin source_name output_name)
        set(_source "${_vk_builtin_source_dir}/${source_name}")
        set(_output "${_vk_builtin_binary_dir}/${output_name}")
        add_custom_command(
                OUTPUT "${_output}"
                COMMAND ${CMAKE_COMMAND} -E make_directory
                        "${_vk_builtin_binary_dir}"
                COMMAND $<TARGET_FILE:glslang-standalone>
                        -D -V --target-env vulkan1.2 -S comp -e main
                        "-I${CMAKE_CURRENT_LIST_DIR}/../common"
                        -o "${_output}" "${_source}"
                COMMAND $<TARGET_FILE:luisa_validate_spirv> "${_output}"
                DEPENDS glslang-standalone luisa_validate_spirv "${_source}"
                        ${_vk_builtin_layout_definitions}
                COMMENT "[Vulkan] Compiling and validating built-in ${source_name}"
                VERBATIM)
        set(_vk_builtin_spirv ${_vk_builtin_spirv} "${_output}"
                PARENT_SCOPE)
    endfunction()

    _luisa_compute_compile_vulkan_builtin(
            indirect_prepare.comp.hlsl indirect_prepare.spv)
    _luisa_compute_compile_vulkan_builtin(
            accel_process.comp.hlsl accel_process.spv)
    _luisa_compute_compile_vulkan_builtin(
            bindless_upload.comp.hlsl bindless_upload.spv)

    set(_vk_builtin_embedded_cpp
            "${_vk_builtin_binary_dir}/vulkan_builtin_spirv_embedded.cpp")
    set(_vk_builtin_embedded_h
            "${_vk_builtin_binary_dir}/vulkan_builtin_spirv_embedded.h")
    add_custom_command(
            OUTPUT "${_vk_builtin_embedded_cpp}" "${_vk_builtin_embedded_h}"
            COMMAND $<TARGET_FILE:luisa_embed_device_lib>
                    ${_vk_builtin_spirv}
                    --unsigned --preserve-ext
                    --prefix luisa_compute_vk_builtin_
                    -o "${_vk_builtin_embedded_cpp}"
                    -h "${_vk_builtin_embedded_h}"
            COMMAND ${CMAKE_COMMAND} -E touch
                    "${_vk_builtin_embedded_cpp}"
                    "${_vk_builtin_embedded_h}"
            DEPENDS luisa_embed_device_lib ${_vk_builtin_spirv}
            COMMENT "[Vulkan] Embedding built-in SPIR-V modules"
            VERBATIM)
    add_custom_target(luisa-compute-backend-vk-embed-builtins
            DEPENDS ${_vk_builtin_spirv}
                    "${_vk_builtin_embedded_cpp}"
                    "${_vk_builtin_embedded_h}")
    add_dependencies(luisa-compute-backend-vk
            luisa-compute-backend-vk-embed-builtins)
    set_source_files_properties(
            "${_vk_builtin_embedded_cpp}"
            "${_vk_builtin_embedded_h}"
            PROPERTIES GENERATED TRUE SKIP_UNITY_BUILD_INCLUSION TRUE)
    target_sources(luisa-compute-backend-vk PRIVATE
            "${_vk_builtin_embedded_cpp}"
            "${_vk_builtin_embedded_h}")
    target_include_directories(luisa-compute-backend-vk PRIVATE
            "${_vk_builtin_binary_dir}")
    if (LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV AND LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV)
        message(FATAL_ERROR "Vulkan compute shader codegen options are mutually exclusive. "
                "Use LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV for the default native SPIR-V path, "
                "or LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV for the experimental LLVM SPIR-V path.")
    endif ()
    if (LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV)
        target_link_libraries(luisa-compute-backend-vk PRIVATE luisa-compute-spirv)
        target_compile_definitions(luisa-compute-backend-vk PRIVATE LUISA_XIR_TO_SPIRV)
    endif ()
    # AST LLVM → SPIR-V codegen path
    if (LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV)
        # The backend still uses the native SPIR-V support library for its
        # disassembler/validation utilities in the LLVM configuration.
        target_link_libraries(luisa-compute-backend-vk PRIVATE
                luisa-compute-spirv
                luisa-compute-spirv-llvm)
        target_compile_definitions(luisa-compute-backend-vk PRIVATE LUISA_AST_LLVM_TO_SPIRV)
    endif ()
    if (APPLE)
        enable_language(OBJC)
        enable_language(OBJCXX)
        target_sources(luisa-compute-backend-vk PRIVATE ../common/moltenvk_surface.mm)
        target_link_libraries(luisa-compute-backend-vk PRIVATE
                "-framework Foundation"
                "-framework Metal"
                "-framework QuartzCore"
                "-framework AppKit")
    endif ()
    if (WIN32)
        target_compile_definitions(luisa-compute-backend-vk PRIVATE VK_USE_PLATFORM_WIN32_KHR)
    elseif (UNIX AND NOT APPLE)
        target_compile_definitions(luisa-compute-backend-vk PRIVATE VK_USE_PLATFORM_XCB_KHR VK_USE_PLATFORM_XLIB_KHR)
        luisa_compute_link_wayland_if_enabled(luisa-compute-backend-vk)
    endif ()

    # Install the optional DXC compatibility runtime. Native XIR/LLVM SPIR-V
    # builds do not load or deploy it when compatibility is disabled.
    if (LUISA_COMPUTE_VULKAN_ENABLE_DXC_COMPATIBILITY AND
            UNIX AND NOT APPLE AND
            (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64"))
        option(LUISA_COMPUTE_VK_SDK_DIR "The directory of Vulkan(DXC) SDK" "")
        if (LUISA_COMPUTE_VK_SDK_DIR)
            cmake_path(IS_RELATIVE LUISA_COMPUTE_VK_SDK_DIR LUISA_COMPUTE_VK_SDK_DIR_is_relative)
            if (LUISA_COMPUTE_VK_SDK_DIR_is_relative)
                set(LUISA_COMPUTE_VK_SDK_DIR
                        "${PROJECT_SOURCE_DIR}/${LUISA_COMPUTE_VK_SDK_DIR}")
            endif ()
            cmake_path(NORMAL_PATH LUISA_COMPUTE_VK_SDK_DIR OUTPUT_VARIABLE LUISA_COMPUTE_VK_SDK_DIR)
            message(STATUS "Vulkan(DXC) SDK directory: ${LUISA_COMPUTE_VK_SDK_DIR}")
        endif ()
        if (NOT LUISA_COMPUTE_VK_SDK_DIR OR NOT EXISTS "${LUISA_COMPUTE_VK_SDK_DIR}")
            set(LUISA_COMPUTE_VK_SDK_DOWNLOAD_URL "https://github.com/LuisaGroup/SDKs/releases/download/sdk/linux_dxc_2026_07_29.x86_64.zip")
            set(LUISA_COMPUTE_VK_SDK_SHA256 "4BF754C2E2692276F20BE50FEDF1E02A3FFBE1461D4EA233BB39A9C01BDFF713")
            message(STATUS "Downloading Vulkan(DXC) SDK from ${LUISA_COMPUTE_VK_SDK_DOWNLOAD_URL}")
            include(FetchContent)
            FetchContent_Declare(luisa_compute_vk_sdk
                    URL ${LUISA_COMPUTE_VK_SDK_DOWNLOAD_URL}
                    URL_HASH SHA256=${LUISA_COMPUTE_VK_SDK_SHA256})
            FetchContent_MakeAvailable(luisa_compute_vk_sdk)
            FetchContent_GetProperties(luisa_compute_vk_sdk)
            set(LUISA_COMPUTE_VK_SDK_DIR ${luisa_compute_vk_sdk_SOURCE_DIR})
        endif ()
        # check if the SDK files exist
        set(LUISA_COMPUTE_VK_SDK_FILES
                libdxcompiler.so)
        foreach (file ${LUISA_COMPUTE_VK_SDK_FILES})
            if (NOT EXISTS "${LUISA_COMPUTE_VK_SDK_DIR}/${file}")
                message(FATAL_ERROR "The required file ${file} is missing in the Vulkan(DXC) SDK directory (${LUISA_COMPUTE_VK_SDK_DIR}). \
                You can either manually download the SDK from ${LUISA_COMPUTE_VK_SDK_DOWNLOAD_URL} and extract it to the specified directory, \
                or set LUISA_COMPUTE_VK_SDK_DIR to an empty string to let the build script download the SDK for you.")
            endif ()
        endforeach ()
        # Copy DXC library to runtime output directory
        add_custom_command(TARGET luisa-compute-backend-vk POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${LUISA_COMPUTE_VK_SDK_DIR}/libdxcompiler.so"
                "$<TARGET_FILE_DIR:luisa-compute-backend-vk>")
    endif ()
endif ()
