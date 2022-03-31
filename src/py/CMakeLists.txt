pybind11_add_module(lcapi lcapi.cpp)
target_link_libraries(lcapi PRIVATE luisa::compute)
