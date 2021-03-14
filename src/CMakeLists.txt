add_library(luisa-compute-include INTERFACE)
target_include_directories(luisa-compute-include INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(luisa-compute-include INTERFACE c_std_11 cxx_std_20)
if (MSVC)
    target_compile_options(luisa-compute-include INTERFACE /Zc:preprocessor)
    target_compile_definitions(luisa-compute-include INTERFACE
                               NOMINMAX
                               _ENABLE_EXTENDED_ALIGNED_STORAGE
                               _CRT_SECURE_NO_WARNINGS)
endif()

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(ext)
add_subdirectory(core)
add_subdirectory(runtime)
add_subdirectory(ast)
add_subdirectory(compile)
add_subdirectory(dsl)
add_subdirectory(backends)

add_library(luisa-compute INTERFACE)
target_link_libraries(luisa-compute INTERFACE
                      luisa-compute-core
                      luisa-compute-ast
                      luisa-compute-compile
                      luisa-compute-runtime
                      luisa-compute-dsl
                      luisa-compute-backends)
add_library(luisa::compute ALIAS luisa-compute)

add_subdirectory(tests)
