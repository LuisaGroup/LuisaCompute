add_library(luisa-compute-include INTERFACE)
target_include_directories(luisa-compute-include INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(luisa-compute-include INTERFACE c_std_11 cxx_std_17)
target_compile_definitions(luisa-compute-include INTERFACE NOMINMAX)

add_subdirectory(ext)
add_subdirectory(compute)
add_subdirectory(tests)
