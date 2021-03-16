add_library(luisa-compute-python INTERFACE)

pybind11_add_module(pyluisa pyluisa.cpp)
target_link_libraries(pyluisa PRIVATE luisa-compute-ast)
add_dependencies(luisa-compute-python pyluisa)
