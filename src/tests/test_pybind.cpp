//
// Created by Mike Smith on 2020/12/10.
//

#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

PYBIND11_MODULE(test_pybind, m) {
    m.def("add", [](int a, int b) {
        return a + b;
    });
}

int main() {
    
    namespace py = pybind11;
    
    py::scoped_interpreter guard{};
    
    auto fast_calc = py::module_::import("test_pybind");
    auto result = fast_calc.attr("add")(1, 2).cast<int>();
    std::cout << result << std::endl;
    
    
}
