add_executable(test main.cpp)
target_compile_features(test PRIVATE c_std_11 cxx_std_20)
target_link_libraries(test PRIVATE luisa-ext-oiio luisa-ext-opencv)

find_package(pybind11 REQUIRED)
pybind11_add_module(test_pybind test_pybind.cpp)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
add_executable(test_pybind_embed test_pybind.cpp)
target_link_libraries(test_pybind_embed PRIVATE pybind11::pybind11 ${PYTHON_LIBRARIES})
add_dependencies(test_pybind_embed test_pybind)

add_executable(test_playground test_playground.cpp)
target_link_libraries(test_playground PRIVATE luisa-compute-core)
