add_executable(test main.cpp)
target_compile_features(test PRIVATE c_std_11 cxx_std_20)
target_link_libraries(test PRIVATE luisa::compute)

#find_package(pybind11 REQUIRED)
#pybind11_add_module(test_pybind test_pybind.cpp)
#
#find_package(PythonInterp REQUIRED)
#find_package(PythonLibs REQUIRED)
#add_executable(test_pybind_embed test_pybind.cpp)
#target_link_libraries(test_pybind_embed PRIVATE pybind11::pybind11 ${PYTHON_LIBRARIES})
#add_dependencies(test_pybind_embed test_pybind)

add_executable(test_ast test_ast.cpp)
target_link_libraries(test_ast PRIVATE luisa::compute)

add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE luisa::compute)

add_executable(test_interpreter test_interpreter.cpp)
target_link_libraries(test_interpreter PRIVATE luisa::compute)
