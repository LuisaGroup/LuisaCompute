if (LUISA_COMPUTE_ENABLE_VULKAN)
    message(STATUS "Build with Vulkan backend")
    file(GLOB LUISA_COMPUTE_VK_SOURCES CONFIGURE_DEPENDS *.cpp *.h)
    luisa_compute_add_backend(vk SOURCES ${LUISA_COMPUTE_VK_SOURCES} ../common/default_binary_io.cpp)
    target_link_libraries(luisa-compute-backend-vk PUBLIC luisa-compute-ext-volk luisa-compute-hlsl-codegen)
    if (LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV)
        target_link_libraries(luisa-compute-backend-vk PRIVATE luisa-compute-spirv)
        target_compile_definitions(luisa-compute-backend-vk PRIVATE LUISA_XIR_TO_SPIRV)
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

    # install dxc
    if (UNIX AND NOT APPLE AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64"))
        option(LUISA_COMPUTE_VK_SDK_DIR "The directory of Vulkan(DXC) SDK" "")
        if (LUISA_COMPUTE_VK_SDK_DIR)
            cmake_path(IS_RELATIVE LUISA_COMPUTE_VK_SDK_DIR LUISA_COMPUTE_VK_SDK_DIR_is_relative)
            if (LUISA_COMPUTE_VK_SDK_DIR_is_relative)
                set(LUISA_COMPUTE_VK_SDK_DIR "${CMAKE_SOURCE_DIR}/${LUISA_COMPUTE_VK_SDK_DIR}")
            endif ()
            cmake_path(NORMAL_PATH LUISA_COMPUTE_VK_SDK_DIR OUTPUT_VARIABLE LUISA_COMPUTE_VK_SDK_DIR)
            message(STATUS "Vulkan(DXC) SDK directory: ${LUISA_COMPUTE_VK_SDK_DIR}")
        endif ()
        if (NOT LUISA_COMPUTE_VK_SDK_DIR OR NOT EXISTS "${LUISA_COMPUTE_VK_SDK_DIR}")
            set(LUISA_COMPUTE_VK_SDK_DOWNLOAD_URL "https://github.com/LuisaGroup/SDKs/releases/download/sdk/linux_dxc_2025_07_14.x86_64.zip")
            set(LUISA_COMPUTE_VK_SDK_SHA256 "9ba3197dcc58d0613afd183757102dd87466223a84f5ad5095c8598ca7f91adc")
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
