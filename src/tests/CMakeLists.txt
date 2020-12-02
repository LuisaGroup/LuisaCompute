add_executable(test main.cpp)
target_compile_features(test PRIVATE c_std_11 cxx_std_20)
target_link_libraries(test PRIVATE luisa-ext-oiio luisa-ext-opencv)
