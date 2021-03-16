add_library(luisa-compute-python INTERFACE)

pybind11_add_module(luisa-compute-python-ast ast.cpp)
target_link_libraries(luisa-compute-python-ast PRIVATE luisa-compute-ast)
add_dependencies(luisa-compute-python luisa-compute-python-ast)
