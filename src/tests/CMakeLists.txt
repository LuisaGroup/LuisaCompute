add_executable(test main.cpp)
target_link_libraries(test PRIVATE luisa::compute)

add_executable(test_type test_type.cpp)
target_link_libraries(test_type PRIVATE luisa::compute)

add_executable(test_dsl test_dsl.cpp)
target_link_libraries(test_dsl PRIVATE luisa::compute)

add_executable(test_dsl_multithread test_dsl_multithread.cpp)
target_link_libraries(test_dsl_multithread PRIVATE luisa::compute)

add_executable(test_dsl_sugar test_dsl_sugar.cpp)
target_link_libraries(test_dsl_sugar PRIVATE luisa::compute)

add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE luisa::compute)

add_executable(test_simple test_simple.cpp)
target_link_libraries(test_simple PRIVATE luisa::compute)

add_executable(test_texture_io test_texture_io.cpp)
target_link_libraries(test_texture_io PRIVATE luisa::compute)

add_executable(test_builtin test_builtin.cpp)
target_link_libraries(test_builtin PRIVATE luisa::compute)
