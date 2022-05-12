#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <luisa-compute.h>
#include <dsl/builtin.h>

namespace py = pybind11;
using namespace luisa::compute;

#define LUISA_EXPORT_ARITHMETIC_OP(T) \
    m##T \
        .def("__add__", [](const Vector<T,2>&a, const Vector<T,2>&b) { return a + b; }, py::is_operator()) \
        .def("__add__", [](const Vector<T,2>&a, const T&b) { return a + b; }, py::is_operator())           \
        .def("__radd__", [](const Vector<T,2>&a, const T&b) { return a + b; }, py::is_operator())           \
        .def("__sub__", [](const Vector<T,2>&a, const Vector<T,2>&b) { return a - b; }, py::is_operator()) \
        .def("__sub__", [](const Vector<T,2>&a, const T&b) { return a - b; }, py::is_operator()) \
        .def("__rsub__", [](const Vector<T,2>&a, const T&b) { return b - a; }, py::is_operator()) \
        .def("__mul__", [](const Vector<T,2>&a, const Vector<T,2>&b) { return a * b; }, py::is_operator()) \
        .def("__mul__", [](const Vector<T,2>&a, const T&b) { return a * b; }, py::is_operator()) \
        .def("__rmul__", [](const Vector<T,2>&a, const T&b) { return a * b; }, py::is_operator()) \
        .def("__truediv__", [](const Vector<T,2>&a, const Vector<T,2>&b) { return a / b; }, py::is_operator()) \
        .def("__truediv__", [](const Vector<T,2>&a, const T&b) { return a / b; }, py::is_operator()) \
        .def("__rtruediv__", [](const Vector<T,2>&a, const T&b) { return b / a; }, py::is_operator()) \
        ;

#define LUISA_EXPORT_INT_OP(T) \
    m##T \
        .def("__mod__", [](const Vector<T,2>&a, const Vector<T,2>&b) { return a % b; }, py::is_operator()) \
        .def("__mod__", [](const Vector<T,2>&a, const T&b) { return a % b; }, py::is_operator()) \
        .def("__rmod__", [](const Vector<T,2>&a, const T&b) { return b % a; }, py::is_operator()) \
        ;

#define LUISA_EXPORT_FLOAT_OP(T) \
    m##T                    \
        .def("__pow__", [](const Vector<T,2>&a, const Vector<T,2>&b) { return luisa::pow(a, b); }, py::is_operator());

#define LUISA_EXPORT_VECTOR2(T) \
    py::class_<luisa::detail::VectorStorage<T, 2>>(m, "_vectorstorage_"#T"2"); \
    auto m##T = py::class_<Vector<T,2>, luisa::detail::VectorStorage<T, 2>>(m, #T"2") \
    	.def(py::init<T>()) \
    	.def(py::init<T,T>()) \
    	.def(py::init<Vector<T,2>>()) \
    	  .def("__repr__", [](Vector<T,2>& self){return format(#T"2({},{})", self.x, self.y);}) \
        .def("__getitem__", [](Vector<T,2>& self, size_t i){return self[i];}) \
        .def("__setitem__", [](Vector<T,2>& self, size_t i, T k){ self[i]=k; }) \
        .def("copy", [](Vector<T,2>& self){return Vector<T,2>(self);}) \
    	.def_readwrite("x", &Vector<T,2>::x) \
    	.def_readwrite("y", &Vector<T,2>::y) \
		.def_property_readonly("xx", &Vector<T,2>::xx) \
		.def_property_readonly("xy", &Vector<T,2>::xy) \
		.def_property_readonly("yx", &Vector<T,2>::yx) \
		.def_property_readonly("yy", &Vector<T,2>::yy) \
		.def_property_readonly("xxx", &Vector<T,2>::xxx) \
		.def_property_readonly("xxy", &Vector<T,2>::xxy) \
		.def_property_readonly("xyx", &Vector<T,2>::xyx) \
		.def_property_readonly("xyy", &Vector<T,2>::xyy) \
		.def_property_readonly("yxx", &Vector<T,2>::yxx) \
		.def_property_readonly("yxy", &Vector<T,2>::yxy) \
		.def_property_readonly("yyx", &Vector<T,2>::yyx) \
		.def_property_readonly("yyy", &Vector<T,2>::yyy) \
		.def_property_readonly("xxxx", &Vector<T,2>::xxxx) \
		.def_property_readonly("xxxy", &Vector<T,2>::xxxy) \
		.def_property_readonly("xxyx", &Vector<T,2>::xxyx) \
		.def_property_readonly("xxyy", &Vector<T,2>::xxyy) \
		.def_property_readonly("xyxx", &Vector<T,2>::xyxx) \
		.def_property_readonly("xyxy", &Vector<T,2>::xyxy) \
		.def_property_readonly("xyyx", &Vector<T,2>::xyyx) \
		.def_property_readonly("xyyy", &Vector<T,2>::xyyy) \
		.def_property_readonly("yxxx", &Vector<T,2>::yxxx) \
		.def_property_readonly("yxxy", &Vector<T,2>::yxxy) \
		.def_property_readonly("yxyx", &Vector<T,2>::yxyx) \
		.def_property_readonly("yxyy", &Vector<T,2>::yxyy) \
		.def_property_readonly("yyxx", &Vector<T,2>::yyxx) \
		.def_property_readonly("yyxy", &Vector<T,2>::yyxy) \
		.def_property_readonly("yyyx", &Vector<T,2>::yyyx) \
		.def_property_readonly("yyyy", &Vector<T,2>::yyyy) \
		; \
	m.def("make_"#T"2", [](T a){return make_##T##2(a);}); \
	m.def("make_"#T"2", [](T a, T b){return make_##T##2(a,b);}); \
	m.def("make_"#T"2", [](Vector<T,2> a){return make_##T##2(a);});

void export_vector2(py::module &m) {
    LUISA_EXPORT_VECTOR2(bool)
    LUISA_EXPORT_VECTOR2(uint)
    LUISA_EXPORT_VECTOR2(int)
    LUISA_EXPORT_VECTOR2(float)
    LUISA_EXPORT_ARITHMETIC_OP(uint)
    LUISA_EXPORT_ARITHMETIC_OP(int)
    LUISA_EXPORT_ARITHMETIC_OP(float)
    LUISA_EXPORT_INT_OP(uint)
    LUISA_EXPORT_INT_OP(int)
    LUISA_EXPORT_FLOAT_OP(float)
}
