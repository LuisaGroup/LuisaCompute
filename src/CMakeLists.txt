add_library(luisa-compute-include INTERFACE)
target_include_directories(luisa-compute-include INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(luisa-compute-include INTERFACE c_std_11 cxx_std_20)
if (MSVC)
    target_compile_options(luisa-compute-include INTERFACE
                           /we4834 # discarding [[nodiscard]] results
                           /Zc:preprocessor)
    target_compile_definitions(luisa-compute-include INTERFACE
                               NOMINMAX
                               _ENABLE_EXTENDED_ALIGNED_STORAGE
                               _CRT_SECURE_NO_WARNINGS
                               _UNICODE)
else ()
    target_compile_options(luisa-compute-include INTERFACE
                           -Wall -Wextra -Wpedantic -Wabi
                           -Wno-unused-parameter
                           -Wno-unused-variable
                           -Wno-gnu-zero-variadic-macro-arguments
                           -Wno-dollar-in-identifier-extension
                           -Werror=unused-result)
endif ()

add_subdirectory(ext)
add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(ast)
add_subdirectory(runtime)
add_subdirectory(compile)
add_subdirectory(dsl)
add_subdirectory(rtx)
add_subdirectory(backends)
add_subdirectory(python)

add_library(luisa-compute INTERFACE)
target_link_libraries(luisa-compute INTERFACE
                      luisa-compute-core
                      luisa-compute-util
                      luisa-compute-ast
                      luisa-compute-compile
                      luisa-compute-runtime
                      luisa-compute-dsl
                      luisa-compute-rtx
                      luisa-compute-backends
                      luisa-compute-python)

add_library(luisa::compute ALIAS luisa-compute)

if (LUISA_COMPUTE_BUILD_TESTS)
    add_subdirectory(tests)
endif ()
