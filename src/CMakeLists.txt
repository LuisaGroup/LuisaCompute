add_library(luisa-compute-include INTERFACE)
target_include_directories(luisa-compute-include INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(luisa-compute-include INTERFACE c_std_11 cxx_std_20)
target_compile_definitions(luisa-compute-include INTERFACE NOMINMAX)
target_compile_options(luisa-compute-include INTERFACE $<$<CXX_COMPILER_ID:MSVC>:"/Zc:preprocessor">)
set_target_properties(luisa-compute-include PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_subdirectory(ext)
add_subdirectory(core)
add_subdirectory(ast)
add_subdirectory(runtime)
add_subdirectory(dsl)
add_subdirectory(tests)

add_library(luisa-compute INTERFACE)
target_link_libraries(luisa-compute INTERFACE
                      luisa-compute-ext
                      luisa-compute-core
                      luisa-compute-ast
                      luisa-compute-runtime
                      luisa-compute-dsl)

add_library(luisa::compute ALIAS luisa-compute)
