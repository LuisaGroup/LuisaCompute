add_executable(test main.cpp)
target_compile_features(test PRIVATE c_std_11 cxx_std_20)
target_link_libraries(test PRIVATE luisa::compute)

add_executable(test_ast test_ast.cpp)
target_link_libraries(test_ast PRIVATE luisa::compute)

add_executable(test_dsl test_dsl.cpp)
target_link_libraries(test_dsl PRIVATE luisa::compute)

add_executable(test_dsl_multithread test_dsl_multithread.cpp)
target_link_libraries(test_dsl_multithread PRIVATE luisa::compute)

add_executable(test_dsl_sugar test_dsl_sugar.cpp)
target_link_libraries(test_dsl_sugar PRIVATE luisa::compute)
