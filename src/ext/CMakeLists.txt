add_library(luisa-compute-ext SHARED dummy.cpp)
set_target_properties(luisa-compute-ext PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_custom_target(luisa-compute-ext-copy ALL
        COMMENT "Copy necessary files for third-party libraries")
add_dependencies(luisa-compute-ext-copy luisa-compute-ext)


set(SPDLOG_ENABLE_PCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_SHARED ON CACHE BOOL "" FORCE)
set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)
set(SPDLOG_NO_THREAD_ID ON CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_DISABLE_DEFAULT_LOGGER ON CACHE BOOL "" FORCE)
add_subdirectory(spdlog)
target_compile_definitions(spdlog_header_only INTERFACE
        FMT_CONSTEVAL=constexpr
        FMT_EXCEPTIONS=0
        FMT_HEADER_ONLY=1
        FMT_USE_NOEXCEPT=1)
target_link_libraries(luisa-compute-ext PUBLIC spdlog::spdlog_header_only)


add_library(xxhash INTERFACE)
target_include_directories(xxhash INTERFACE xxHash)
target_compile_definitions(xxhash INTERFACE XXH_INLINE_ALL)
set_target_properties(xxhash PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})
target_link_libraries(luisa-compute-ext PUBLIC xxhash)

add_library(asio INTERFACE dummy.cpp)
target_include_directories(asio INTERFACE asio/asio/include)
target_compile_definitions(asio INTERFACE
        ASIO_STANDALONE=1
        ASIO_HAS_STD_INVOKE_RESULT=1
        ASIO_NO_EXCEPTIONS=1)
if (WIN32)
    target_compile_definitions(asio INTERFACE _WIN32_WINNT=0x0A00)
    target_link_libraries(asio INTERFACE ws2_32 wsock32 bcrypt)
endif ()
target_link_libraries(luisa-compute-ext PUBLIC asio)

add_subdirectory(stb)
target_link_libraries(luisa-compute-ext PUBLIC stb)

set(JSON_MultipleHeaders ON CACHE BOOL "" FORCE)
set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
add_subdirectory(json)
target_link_libraries(luisa-compute-ext PUBLIC nlohmann_json::nlohmann_json)

if (LUISA_COMPUTE_ENABLE_GUI)
    add_subdirectory(glad)
    target_link_libraries(luisa-compute-ext PUBLIC glad)

    find_package(GLFW3 CONFIG QUIET)
    if (NOT GLFW3_FOUND)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        set(GLFW_LIBRARY_TYPE SHARED CACHE STRING "" FORCE)
        add_subdirectory(glfw)
        set_target_properties(glfw PROPERTIES
                UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})
    endif ()
    target_link_libraries(luisa-compute-ext PUBLIC glfw)

    add_subdirectory(imgui)
    target_link_libraries(luisa-compute-ext PUBLIC imgui)
endif ()

add_subdirectory(EASTL)
target_link_libraries(luisa-compute-ext PUBLIC EASTL)

if (LUISA_COMPUTE_ENABLE_PYTHON)
    add_subdirectory(pybind11)
endif ()

if (LUISA_COMPUTE_ENABLE_ISPC OR LUISA_COMPUTE_ENABLE_LLVM)
    set(EMBREE_TASKING_SYSTEM "INTERNAL" CACHE STRING "" FORCE)
    find_package(embree 3 CONFIG QUIET)
    if (embree_FOUND)
        message(STATUS "Found embree: ${embree_VERSION}")
    else ()
        set(LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION "3.13.5")
        file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/embree)
        message(STATUS "Downloading embree ${LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION}...")
        include(FetchContent)
        if (WIN32)
            FetchContent_Declare(embree_binary
                    URL "https://github.com/embree/embree/releases/download/v${LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION}/embree-${LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION}.x64.vc14.windows.zip")
        elseif (APPLE)
            FetchContent_Declare(embree_binary
                    URL "https://github.com/embree/embree/releases/download/v${LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION}/embree-${LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION}.x86_64.macosx.zip")
        else ()# Linux
            FetchContent_Declare(embree_binary
                    URL "https://github.com/embree/embree/releases/download/v${LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION}/embree-${LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION}.x86_64.linux.tar.gz")
        endif ()
        FetchContent_MakeAvailable(embree_binary)
        FetchContent_GetProperties(embree_binary)
        # patch embree
        message(STATUS "Patching embree-config.cmake...")
        file(READ "${embree_binary_SOURCE_DIR}/lib/cmake/embree-${LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION}/embree-config.cmake" EMBREE_CONFIG)
        string(REPLACE "FIND_PACKAGE(TBB REQUIRED \${EMBREE_TBB_COMPONENT})" "" EMBREE_CONFIG "${EMBREE_CONFIG}")
        file(WRITE "${embree_binary_SOURCE_DIR}/lib/cmake/embree-${LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION}/embree-config.cmake" "${EMBREE_CONFIG}")
        message(STATUS "Downloaded embree: ${embree_binary_SOURCE_DIR}")
        set(embree_ROOT ${embree_binary_SOURCE_DIR})
        find_package(embree 3 CONFIG REQUIRED)
    endif ()
    target_link_libraries(luisa-compute-ext PUBLIC embree)
    add_custom_command(TARGET luisa-compute-ext-copy
            DEPENDS "$<TARGET_FILE:embree>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:embree>" "$<TARGET_FILE_DIR:luisa-compute-ext>")
    add_custom_command(TARGET luisa-compute-ext-copy
            DEPENDS "$<TARGET_LINKER_FILE:embree>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_LINKER_FILE:embree>" "$<TARGET_FILE_DIR:luisa-compute-ext>")
endif ()

if (LUISA_COMPUTE_ENABLE_VULKAN)
    find_package(Vulkan)
    if (NOT Vulkan_FOUND)
        message(WARNING "Vulkan is not found. Vulkan backend will be disabled.")
        set(LUISA_COMPUTE_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
    else ()
        set(ENABLE_PCH OFF CACHE BOOL "" FORCE) # CMake generates -Xarch_* flags that leads to build errors on macOS
        set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
        set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
        set(SPIRV_HEADERS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
        set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
        add_subdirectory(SPIRV-Headers)
        add_subdirectory(SPIRV-Tools)
        add_subdirectory(glslang)
        set(SHADERC_SKIP_TESTS ON CACHE BOOL "" FORCE)
        set(SHADERC_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
        set(SHADERC_SKIP_COPYRIGHT_CHECK ON CACHE BOOL "" FORCE)
        set(SHADERC_ENABLE_SHARED_CRT ON CACHE BOOL "" FORCE)
        add_subdirectory(shaderc)
        set_target_properties(shaderc_combined_genfile PROPERTIES EXCLUDE_FROM_ALL ON) # Problematic on M1 Macs
        set_target_properties(glslc PROPERTIES EXCLUDE_FROM_ALL ON) # Problematic on Windows
        target_link_libraries(luisa-compute-ext PUBLIC Vulkan::Vulkan SPIRV SPIRV-Tools-shared)
        if (BUILD_SHARED_LIBS)
            target_link_libraries(luisa-compute-ext PUBLIC shaderc_shared)
        else ()
            target_link_libraries(luisa-compute-ext PUBLIC shaderc)
        endif ()
    endif ()
endif ()

if (WIN32 AND NOT MINGW AND NOT MSYS)
    set(BUILD_SHARED_LIBS OFF)
endif ()

set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
add_subdirectory(abseil-cpp)
target_link_libraries(luisa-compute-ext PUBLIC absl::base absl::flat_hash_map)
