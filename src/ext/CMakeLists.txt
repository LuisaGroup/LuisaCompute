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

add_subdirectory(stb)
target_link_libraries(luisa-compute-ext PUBLIC stb)
luisa_compute_install_extension(stb INCLUDE stb)

#set(JSON_MultipleHeaders ON CACHE BOOL "" FORCE)
#set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
#add_subdirectory(json)
#target_link_libraries(luisa-compute-ext PUBLIC nlohmann_json::nlohmann_json)
#luisa_compute_install_extension(nlohmann_json INCLUDE json/include/nlohmann)

if (LUISA_COMPUTE_ENABLE_GUI)
    add_subdirectory(glad)
    target_link_libraries(luisa-compute-ext PUBLIC glad)
    luisa_compute_install_extension(glad INCLUDE glad)

    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    set(GLFW_LIBRARY_TYPE SHARED CACHE STRING "" FORCE)
    add_subdirectory(glfw)
    set_target_properties(glfw PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})
    target_link_libraries(luisa-compute-ext PUBLIC glfw)
    luisa_compute_install_extension(glfw INCLUDE glfw/include)

    add_subdirectory(imgui)
    target_link_libraries(luisa-compute-ext PUBLIC imgui)
    luisa_compute_install_extension(imgui INCLUDE imgui)
endif ()

add_subdirectory(EASTL)
target_link_libraries(luisa-compute-ext PUBLIC EASTL)
luisa_compute_install_extension(EASTL INCLUDE EASTL/include/EASTL)
luisa_compute_install_extension(EABase INCLUDE EASTL/packages/EABase/include/Common/EABase)

add_library(unordered_dense INTERFACE)
target_include_directories(unordered_dense INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/unordered_dense/include>)
target_link_libraries(luisa-compute-ext PUBLIC unordered_dense)
luisa_compute_install_extension(unordered_dense INCLUDE unordered_dense/include)

add_library(parallel-hashmap INTERFACE)
target_include_directories(parallel-hashmap INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/parallel-hashmap>)
target_link_libraries(luisa-compute-ext PUBLIC parallel-hashmap)
luisa_compute_install_extension(parallel-hashmap INCLUDE parallel-hashmap/parallel_hashmap)

if (LUISA_COMPUTE_ENABLE_PYTHON)
    add_subdirectory(pybind11)
endif ()

install(TARGETS luisa-compute-ext EXPORT LuisaComputeTargets)

# for rust bindings
add_subdirectory(corrosion)
