#include <pybind11/pybind11.h>
#include <ast/function.h>
#include <core/logging.h>
#include <runtime/device.h>
#include <runtime/context.h>
#include <runtime/stream.h>
#include <runtime/command.h>
#include <runtime/image.h>
#include <rtx/accel.h>
#include <rtx/mesh.h>
#include <rtx/hit.h>
#include <rtx/ray.h>

namespace py = pybind11;
using namespace luisa;
using namespace luisa::compute;

#define LUISA_EXPORT_ARITHMETIC_OP(T) \
    m##T \
        .def("__add__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a + b; }, py::is_operator()) \
        .def("__add__", [](const Vector<T,3>&a, const T&b) { return a + b; }, py::is_operator())           \
        .def("__radd__", [](const Vector<T,3>&a, const T&b) { return a + b; }, py::is_operator())           \
        .def("__sub__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a - b; }, py::is_operator()) \
        .def("__sub__", [](const Vector<T,3>&a, const T&b) { return a - b; }, py::is_operator()) \
        .def("__rsub__", [](const Vector<T,3>&a, const T&b) { return b - a; }, py::is_operator()) \
        .def("__mul__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a * b; }, py::is_operator()) \
        .def("__mul__", [](const Vector<T,3>&a, const T&b) { return a * b; }, py::is_operator()) \
        .def("__rmul__", [](const Vector<T,3>&a, const T&b) { return a * b; }, py::is_operator()) \
        .def("__truediv__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a / b; }, py::is_operator()) \
        .def("__truediv__", [](const Vector<T,3>&a, const T&b) { return a / b; }, py::is_operator()) \
        .def("__rtruediv__", [](const Vector<T,3>&a, const T&b) { return b / a; }, py::is_operator()) \
        .def("__gt__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a > b; }, py::is_operator()) \
        .def("__ge__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a >= b; }, py::is_operator()) \
        .def("__lt__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a < b; }, py::is_operator()) \
        .def("__le__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a <= b; }, py::is_operator()) \
        .def("__eq__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a == b; }, py::is_operator()) \
        .def("__ne__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a != b; }, py::is_operator()) \
        ;

#define LUISA_EXPORT_BOOL_OP(T) \
    m##T                           \
        .def("__eq__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a == b; }, py::is_operator()) \
        .def("__ne__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a != b; }, py::is_operator()) \
        .def("__and__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a && b; }, py::is_operator()) \
        .def("__or__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a || b; }, py::is_operator()) \
        ;

#define LUISA_EXPORT_INT_OP(T) \
    m##T \
        .def("__mod__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a % b; }, py::is_operator()) \
        .def("__mod__", [](const Vector<T,3>&a, const T&b) { return a % b; }, py::is_operator()) \
        .def("__rmod__", [](const Vector<T,3>&a, const T&b) { return b % a; }, py::is_operator()) \
        .def("__shl__", [](const Vector<T,3>&a, const T&b) { return a << b; }, py::is_operator()) \
        .def("__shr__", [](const Vector<T,3>&a, const T&b) { return a >> b; }, py::is_operator()) \
        .def("__xor__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return a ^ b; }, py::is_operator()) \
        ;

#define LUISA_EXPORT_FLOAT_OP(T) \
    m##T                    \
        .def("__pow__", [](const Vector<T,3>&a, const Vector<T,3>&b) { return luisa::pow(a, b); }, py::is_operator());


#define LUISA_EXPORT_UNARY_FUNC(T, name) \
    m.def(#name, [](const Vector<T,3>& v) { return luisa::name(v); });

#define LUISA_EXPORT_ARITHMETIC_FUNC(T) \
    m.def("min", [](const Vector<T,3>& a, const Vector<T,3>& b) { return luisa::min(a, b); }); \
    m.def("min", [](const Vector<T,3>& a, const T& b){ return luisa::min(a, b); });            \
    m.def("min", [](const T& a, const Vector<T,3>& b){ return luisa::min(a, b); });            \
    m.def("max", [](const Vector<T,3>& a, const Vector<T,3>& b) { return luisa::max(a, b); }); \
    m.def("max", [](const Vector<T,3>& a, const T& b){ return luisa::max(a, b); });            \
    m.def("max", [](const T& a, const Vector<T,3>& b){ return luisa::max(a, b); });            \
    m.def("select", [](const Vector<T,3>& a, const Vector<T,3>& b, bool pred) { return luisa::select(a, b, pred); }); \
    m.def("clamp", [](const Vector<T,3>& v, const T& a, const T& b) { return luisa::clamp(v, a, b); }); \

#define LUISA_EXPORT_FLOAT_FUNC(T) \
    m.def("pow", [](const Vector<T,3>& a, const Vector<T,3>& b) { return luisa::pow(a, b); }); \
    m.def("atan2", [](const Vector<T,3>& a, const Vector<T,3>& b) { return luisa::atan2(a, b); }); \
    m.def("lerp", [](const Vector<T,3>& a, const Vector<T,3>& b, float t) { return luisa::lerp(a, b, t); }); \
    m.def("dot", [](const Vector<T,3>& a, const Vector<T,3>& b) { return luisa::dot(a, b); });   \
    m.def("distance", [](const Vector<T,3>& a, const Vector<T,3>& b) { return luisa::distance(a, b); });     \
    m.def("cross", [](const Vector<T,3>& a, const Vector<T,3>& b) { return luisa::cross(a, b); });        \
    m.def("rotation", [](const Vector<T,3>& a, const T& b) { return luisa::rotation(a, b); });        \
    m.def("scaling", [](const Vector<T,3>& v) { return luisa::scaling(v); });        \
    m.def("scaling", [](const T& v) { return luisa::scaling(v); });        \
    LUISA_EXPORT_UNARY_FUNC(T, acos)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, asin)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, atan)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, cos)                                                            \
    LUISA_EXPORT_UNARY_FUNC(T, sin)                                                            \
    LUISA_EXPORT_UNARY_FUNC(T, tan)                                                            \
    LUISA_EXPORT_UNARY_FUNC(T, sqrt)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, ceil)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, floor)                                                          \
    LUISA_EXPORT_UNARY_FUNC(T, round)                                                          \
    LUISA_EXPORT_UNARY_FUNC(T, exp)                                                            \
    LUISA_EXPORT_UNARY_FUNC(T, log)                                                            \
    LUISA_EXPORT_UNARY_FUNC(T, log10)                                                          \
    LUISA_EXPORT_UNARY_FUNC(T, log2)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, abs)                                                            \
    LUISA_EXPORT_UNARY_FUNC(T, radians)                                                        \
    LUISA_EXPORT_UNARY_FUNC(T, degrees)                                                        \
    LUISA_EXPORT_UNARY_FUNC(T, length)                                                         \
    LUISA_EXPORT_UNARY_FUNC(T, normalize)                                                      \
    LUISA_EXPORT_UNARY_FUNC(T, translation)

#define LUISA_EXPORT_VECTOR3(T) \
    py::class_<luisa::detail::VectorStorage<T, 3>>(m, "_vectorstorage_"#T"3"); \
    auto m##T = py::class_<Vector<T,3>, luisa::detail::VectorStorage<T, 3>>(m, #T"3") \
    	.def(py::init<T>()) \
    	.def(py::init<T,T,T>()) \
    	.def(py::init<Vector<T,3>>()) \
    	.def("__repr__", [](Vector<T,3>& self){return format(#T"3({},{},{})", self.x, self.y, self.z);}) \
        .def("__getitem__", [](Vector<T,3>& self, size_t i){return self[i];}) \
        .def("__setitem__", [](Vector<T,3>& self, size_t i, T k){ self[i]=k; }) \
        .def("copy", [](Vector<T,3>& self){return Vector<T,3>(self);}) \
    	.def_readwrite("x", &Vector<T,3>::x) \
    	.def_readwrite("y", &Vector<T,3>::y) \
    	.def_readwrite("z", &Vector<T,3>::z) \
		.def_property_readonly("xx", &Vector<T,3>::xx) \
		.def_property_readonly("xy", &Vector<T,3>::xy) \
		.def_property_readonly("xz", &Vector<T,3>::xz) \
		.def_property_readonly("yx", &Vector<T,3>::yx) \
		.def_property_readonly("yy", &Vector<T,3>::yy) \
		.def_property_readonly("yz", &Vector<T,3>::yz) \
		.def_property_readonly("zx", &Vector<T,3>::zx) \
		.def_property_readonly("zy", &Vector<T,3>::zy) \
		.def_property_readonly("zz", &Vector<T,3>::zz) \
		.def_property_readonly("xxx", &Vector<T,3>::xxx) \
		.def_property_readonly("xxy", &Vector<T,3>::xxy) \
		.def_property_readonly("xxz", &Vector<T,3>::xxz) \
		.def_property_readonly("xyx", &Vector<T,3>::xyx) \
		.def_property_readonly("xyy", &Vector<T,3>::xyy) \
		.def_property_readonly("xyz", &Vector<T,3>::xyz) \
		.def_property_readonly("xzx", &Vector<T,3>::xzx) \
		.def_property_readonly("xzy", &Vector<T,3>::xzy) \
		.def_property_readonly("xzz", &Vector<T,3>::xzz) \
		.def_property_readonly("yxx", &Vector<T,3>::yxx) \
		.def_property_readonly("yxy", &Vector<T,3>::yxy) \
		.def_property_readonly("yxz", &Vector<T,3>::yxz) \
		.def_property_readonly("yyx", &Vector<T,3>::yyx) \
		.def_property_readonly("yyy", &Vector<T,3>::yyy) \
		.def_property_readonly("yyz", &Vector<T,3>::yyz) \
		.def_property_readonly("yzx", &Vector<T,3>::yzx) \
		.def_property_readonly("yzy", &Vector<T,3>::yzy) \
		.def_property_readonly("yzz", &Vector<T,3>::yzz) \
		.def_property_readonly("zxx", &Vector<T,3>::zxx) \
		.def_property_readonly("zxy", &Vector<T,3>::zxy) \
		.def_property_readonly("zxz", &Vector<T,3>::zxz) \
		.def_property_readonly("zyx", &Vector<T,3>::zyx) \
		.def_property_readonly("zyy", &Vector<T,3>::zyy) \
		.def_property_readonly("zyz", &Vector<T,3>::zyz) \
		.def_property_readonly("zzx", &Vector<T,3>::zzx) \
		.def_property_readonly("zzy", &Vector<T,3>::zzy) \
		.def_property_readonly("zzz", &Vector<T,3>::zzz) \
		.def_property_readonly("xxxx", &Vector<T,3>::xxxx) \
		.def_property_readonly("xxxy", &Vector<T,3>::xxxy) \
		.def_property_readonly("xxxz", &Vector<T,3>::xxxz) \
		.def_property_readonly("xxyx", &Vector<T,3>::xxyx) \
		.def_property_readonly("xxyy", &Vector<T,3>::xxyy) \
		.def_property_readonly("xxyz", &Vector<T,3>::xxyz) \
		.def_property_readonly("xxzx", &Vector<T,3>::xxzx) \
		.def_property_readonly("xxzy", &Vector<T,3>::xxzy) \
		.def_property_readonly("xxzz", &Vector<T,3>::xxzz) \
		.def_property_readonly("xyxx", &Vector<T,3>::xyxx) \
		.def_property_readonly("xyxy", &Vector<T,3>::xyxy) \
		.def_property_readonly("xyxz", &Vector<T,3>::xyxz) \
		.def_property_readonly("xyyx", &Vector<T,3>::xyyx) \
		.def_property_readonly("xyyy", &Vector<T,3>::xyyy) \
		.def_property_readonly("xyyz", &Vector<T,3>::xyyz) \
		.def_property_readonly("xyzx", &Vector<T,3>::xyzx) \
		.def_property_readonly("xyzy", &Vector<T,3>::xyzy) \
		.def_property_readonly("xyzz", &Vector<T,3>::xyzz) \
		.def_property_readonly("xzxx", &Vector<T,3>::xzxx) \
		.def_property_readonly("xzxy", &Vector<T,3>::xzxy) \
		.def_property_readonly("xzxz", &Vector<T,3>::xzxz) \
		.def_property_readonly("xzyx", &Vector<T,3>::xzyx) \
		.def_property_readonly("xzyy", &Vector<T,3>::xzyy) \
		.def_property_readonly("xzyz", &Vector<T,3>::xzyz) \
		.def_property_readonly("xzzx", &Vector<T,3>::xzzx) \
		.def_property_readonly("xzzy", &Vector<T,3>::xzzy) \
		.def_property_readonly("xzzz", &Vector<T,3>::xzzz) \
		.def_property_readonly("yxxx", &Vector<T,3>::yxxx) \
		.def_property_readonly("yxxy", &Vector<T,3>::yxxy) \
		.def_property_readonly("yxxz", &Vector<T,3>::yxxz) \
		.def_property_readonly("yxyx", &Vector<T,3>::yxyx) \
		.def_property_readonly("yxyy", &Vector<T,3>::yxyy) \
		.def_property_readonly("yxyz", &Vector<T,3>::yxyz) \
		.def_property_readonly("yxzx", &Vector<T,3>::yxzx) \
		.def_property_readonly("yxzy", &Vector<T,3>::yxzy) \
		.def_property_readonly("yxzz", &Vector<T,3>::yxzz) \
		.def_property_readonly("yyxx", &Vector<T,3>::yyxx) \
		.def_property_readonly("yyxy", &Vector<T,3>::yyxy) \
		.def_property_readonly("yyxz", &Vector<T,3>::yyxz) \
		.def_property_readonly("yyyx", &Vector<T,3>::yyyx) \
		.def_property_readonly("yyyy", &Vector<T,3>::yyyy) \
		.def_property_readonly("yyyz", &Vector<T,3>::yyyz) \
		.def_property_readonly("yyzx", &Vector<T,3>::yyzx) \
		.def_property_readonly("yyzy", &Vector<T,3>::yyzy) \
		.def_property_readonly("yyzz", &Vector<T,3>::yyzz) \
		.def_property_readonly("yzxx", &Vector<T,3>::yzxx) \
		.def_property_readonly("yzxy", &Vector<T,3>::yzxy) \
		.def_property_readonly("yzxz", &Vector<T,3>::yzxz) \
		.def_property_readonly("yzyx", &Vector<T,3>::yzyx) \
		.def_property_readonly("yzyy", &Vector<T,3>::yzyy) \
		.def_property_readonly("yzyz", &Vector<T,3>::yzyz) \
		.def_property_readonly("yzzx", &Vector<T,3>::yzzx) \
		.def_property_readonly("yzzy", &Vector<T,3>::yzzy) \
		.def_property_readonly("yzzz", &Vector<T,3>::yzzz) \
		.def_property_readonly("zxxx", &Vector<T,3>::zxxx) \
		.def_property_readonly("zxxy", &Vector<T,3>::zxxy) \
		.def_property_readonly("zxxz", &Vector<T,3>::zxxz) \
		.def_property_readonly("zxyx", &Vector<T,3>::zxyx) \
		.def_property_readonly("zxyy", &Vector<T,3>::zxyy) \
		.def_property_readonly("zxyz", &Vector<T,3>::zxyz) \
		.def_property_readonly("zxzx", &Vector<T,3>::zxzx) \
		.def_property_readonly("zxzy", &Vector<T,3>::zxzy) \
		.def_property_readonly("zxzz", &Vector<T,3>::zxzz) \
		.def_property_readonly("zyxx", &Vector<T,3>::zyxx) \
		.def_property_readonly("zyxy", &Vector<T,3>::zyxy) \
		.def_property_readonly("zyxz", &Vector<T,3>::zyxz) \
		.def_property_readonly("zyyx", &Vector<T,3>::zyyx) \
		.def_property_readonly("zyyy", &Vector<T,3>::zyyy) \
		.def_property_readonly("zyyz", &Vector<T,3>::zyyz) \
		.def_property_readonly("zyzx", &Vector<T,3>::zyzx) \
		.def_property_readonly("zyzy", &Vector<T,3>::zyzy) \
		.def_property_readonly("zyzz", &Vector<T,3>::zyzz) \
		.def_property_readonly("zzxx", &Vector<T,3>::zzxx) \
		.def_property_readonly("zzxy", &Vector<T,3>::zzxy) \
		.def_property_readonly("zzxz", &Vector<T,3>::zzxz) \
		.def_property_readonly("zzyx", &Vector<T,3>::zzyx) \
		.def_property_readonly("zzyy", &Vector<T,3>::zzyy) \
		.def_property_readonly("zzyz", &Vector<T,3>::zzyz) \
		.def_property_readonly("zzzx", &Vector<T,3>::zzzx) \
		.def_property_readonly("zzzy", &Vector<T,3>::zzzy) \
		.def_property_readonly("zzzz", &Vector<T,3>::zzzz) \
		; \
	m.def("make_"#T"3", [](T a){return make_##T##3(a);}); \
	m.def("make_"#T"3", [](T a, T b, T c){return make_##T##3(a,b,c);}); \
	m.def("make_"#T"3", [](Vector<T,3> a){return make_##T##3(a);});

void export_vector3(py::module &m) {
	LUISA_EXPORT_VECTOR3(bool)
	LUISA_EXPORT_VECTOR3(uint)
	LUISA_EXPORT_VECTOR3(int)
	LUISA_EXPORT_VECTOR3(float)
    LUISA_EXPORT_ARITHMETIC_FUNC(int)
    LUISA_EXPORT_ARITHMETIC_FUNC(uint)
    LUISA_EXPORT_ARITHMETIC_FUNC(float)
    LUISA_EXPORT_FLOAT_FUNC(float)
    LUISA_EXPORT_ARITHMETIC_OP(uint)
    LUISA_EXPORT_ARITHMETIC_OP(int)
    LUISA_EXPORT_ARITHMETIC_OP(float)
    LUISA_EXPORT_INT_OP(uint)
    LUISA_EXPORT_INT_OP(int)
    LUISA_EXPORT_FLOAT_OP(float)
    LUISA_EXPORT_BOOL_OP(bool)
}
