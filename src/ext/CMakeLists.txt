add_library(luisa-compute-ext SHARED dummy.cpp)
set_target_properties(luisa-compute-ext PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_custom_target(luisa-compute-ext-copy ALL
        COMMENT "Copy necessary files for third-party libraries")
add_dependencies(luisa-compute-ext-copy luisa-compute-ext)

function(luisa_compute_install_extension target)
    cmake_parse_arguments("EXT" "" "INCLUDE" "" ${ARGN})
    install(TARGETS ${target}
            EXPORT LuisaComputeTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LuisaCompute/ext)
    if (EXT_INCLUDE)
        install(DIRECTORY ${EXT_INCLUDE}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LuisaCompute/ext
                FILES_MATCHING REGEX ".*\\.(h|hpp|hxx|hxx|inl)$")
    endif ()
endfunction()

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
target_compile_definitions(spdlog PUBLIC
        FMT_CONSTEVAL=constexpr
        FMT_EXCEPTIONS=0
        FMT_HEADER_ONLY=1
        FMT_USE_NOEXCEPT=1)
target_link_libraries(luisa-compute-ext PUBLIC spdlog::spdlog_header_only)
luisa_compute_install_extension(spdlog_header_only INCLUDE spdlog/include/spdlog)

add_library(xxhash INTERFACE)
target_include_directories(xxhash INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/xxHash>)
target_compile_definitions(xxhash INTERFACE XXH_INLINE_ALL)
set_target_properties(xxhash PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})
target_link_libraries(luisa-compute-ext PUBLIC xxhash)
luisa_compute_install_extension(xxhash INCLUDE xxHash)

add_library(asio INTERFACE dummy.cpp)
target_include_directories(asio INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/asio/asio/include>)
target_compile_definitions(asio INTERFACE
        ASIO_STANDALONE=1
        ASIO_HAS_STD_INVOKE_RESULT=1
        ASIO_NO_EXCEPTIONS=1)
if (WIN32)
    target_compile_definitions(asio INTERFACE _WIN32_WINNT=0x0A00)
    target_link_libraries(asio INTERFACE ws2_32 wsock32 bcrypt)
endif ()
target_link_libraries(luisa-compute-ext PUBLIC asio)
luisa_compute_install_extension(asio INCLUDE asio/asio/include/asio)

add_subdirectory(stb)
target_link_libraries(luisa-compute-ext PUBLIC stb)
luisa_compute_install_extension(stb INCLUDE stb)

set(JSON_MultipleHeaders ON CACHE BOOL "" FORCE)
set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
add_subdirectory(json)
target_link_libraries(luisa-compute-ext PUBLIC nlohmann_json::nlohmann_json)
luisa_compute_install_extension(nlohmann_json INCLUDE json/include/nlohmann)

if (LUISA_COMPUTE_ENABLE_GUI)
    add_subdirectory(glad)
    target_link_libraries(luisa-compute-ext PUBLIC glad)
    luisa_compute_install_extension(glad INCLUDE glad)
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
        luisa_compute_install_extension(glfw INCLUDE glfw/include)
    endif ()
    target_link_libraries(luisa-compute-ext PUBLIC glfw)

    add_subdirectory(imgui)
    target_link_libraries(luisa-compute-ext PUBLIC imgui)
    luisa_compute_install_extension(imgui INCLUDE imgui)
endif ()

add_subdirectory(EASTL)
target_link_libraries(luisa-compute-ext PUBLIC EASTL)
luisa_compute_install_extension(EASTL INCLUDE EASTL/include/EASTL)
luisa_compute_install_extension(EABase INCLUDE EASTL/packages/EABase/include/Common/EABase)

if (LUISA_COMPUTE_ENABLE_PYTHON)
    add_subdirectory(pybind11)
endif ()

if (LUISA_COMPUTE_ENABLE_ISPC OR LUISA_COMPUTE_ENABLE_LLVM)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        find_package(embree 3 CONFIG QUIET)
    else ()
        find_package(embree 3 CONFIG QUIET REQUIRED)
    endif ()
    if (embree_FOUND)
        message(STATUS "Found embree: ${embree_VERSION}")
    else ()
        include(FetchContent)
        # tbb (required by embree)
        set(LUISA_COMPUTE_TBB_DOWNLOAD_VERSION "2021.7.0")
        message(STATUS "Downloading TBB ${LUISA_COMPUTE_TBB_DOWNLOAD_VERSION}...")
        if (WIN32)
            FetchContent_Declare(tbb_binary
                    URL "https://github.com/oneapi-src/oneTBB/releases/download/v${LUISA_COMPUTE_TBB_DOWNLOAD_VERSION}/oneapi-tbb-${LUISA_COMPUTE_TBB_DOWNLOAD_VERSION}-win.zip")
        elseif (APPLE)
            FetchContent_Declare(tbb_binary
                    URL "https://github.com/oneapi-src/oneTBB/releases/download/v${LUISA_COMPUTE_TBB_DOWNLOAD_VERSION}/oneapi-tbb-${LUISA_COMPUTE_TBB_DOWNLOAD_VERSION}-mac.tgz")
        else ()# Linux
            FetchContent_Declare(tbb_binary
                    URL "https://github.com/oneapi-src/oneTBB/releases/download/v${LUISA_COMPUTE_TBB_DOWNLOAD_VERSION}/oneapi-tbb-${LUISA_COMPUTE_TBB_DOWNLOAD_VERSION}-lin.tgz")
        endif ()
        FetchContent_MakeAvailable(tbb_binary)
        FetchContent_GetProperties(tbb_binary)
        set(TBB_ROOT ${tbb_binary_SOURCE_DIR})
        find_package(TBB CONFIG REQUIRED)
        add_custom_command(TARGET luisa-compute-ext-copy
                DEPENDS "$<TARGET_FILE:TBB::tbb>"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:TBB::tbb>" "$<TARGET_FILE_DIR:luisa-compute-ext>")
        add_custom_command(TARGET luisa-compute-ext-copy
                DEPENDS "$<TARGET_LINKER_FILE:TBB::tbb>"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_LINKER_FILE:TBB::tbb>" "$<TARGET_FILE_DIR:luisa-compute-ext>")
        install(FILES "$<TARGET_FILE:TBB::tbb>" "$<TARGET_LINKER_FILE:TBB::tbb>" TYPE LIB)
        # embree
        set(LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION "3.13.5")
        message(STATUS "Downloading embree ${LUISA_COMPUTE_EMBREE_DOWNLOAD_VERSION}...")
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
    install(FILES "$<TARGET_FILE:embree>" "$<TARGET_LINKER_FILE:embree>" TYPE LIB)
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
set(ABSL_ENABLE_INSTALL ON CACHE BOOL "" FORCE)
set(CMAKE_INSTALL_INCLUDEDIR include/LuisaCompute/ext)
add_subdirectory(abseil-cpp)
target_link_libraries(luisa-compute-ext PUBLIC absl::base absl::flat_hash_map)

install(TARGETS luisa-compute-ext EXPORT LuisaComputeTargets)
