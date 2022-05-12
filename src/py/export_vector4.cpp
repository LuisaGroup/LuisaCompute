#include <pybind11/pybind11.h>
#include <luisa-compute.h>

namespace py = pybind11;
using namespace luisa::compute;

#define LUISA_EXPORT_ARITHMETIC_OP(T) \
    m##T \
        .def("__add__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a + b; }, py::is_operator()) \
        .def("__add__", [](const Vector<T,4>&a, const T&b) { return a + b; }, py::is_operator())           \
        .def("__radd__", [](const Vector<T,4>&a, const T&b) { return a + b; }, py::is_operator())           \
        .def("__sub__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a - b; }, py::is_operator()) \
        .def("__sub__", [](const Vector<T,4>&a, const T&b) { return a - b; }, py::is_operator()) \
        .def("__rsub__", [](const Vector<T,4>&a, const T&b) { return b - a; }, py::is_operator()) \
        .def("__mul__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a * b; }, py::is_operator()) \
        .def("__mul__", [](const Vector<T,4>&a, const T&b) { return a * b; }, py::is_operator()) \
        .def("__rmul__", [](const Vector<T,4>&a, const T&b) { return a * b; }, py::is_operator()) \
        .def("__truediv__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a / b; }, py::is_operator()) \
        .def("__truediv__", [](const Vector<T,4>&a, const T&b) { return a / b; }, py::is_operator()) \
        .def("__rtruediv__", [](const Vector<T,4>&a, const T&b) { return b / a; }, py::is_operator()) \
        .def("__gt__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a > b; }, py::is_operator()) \
        .def("__ge__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a >= b; }, py::is_operator()) \
        .def("__lt__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a < b; }, py::is_operator()) \
        .def("__le__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a <= b; }, py::is_operator()) \
        .def("__eq__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a == b; }, py::is_operator()) \
        .def("__ne__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a != b; }, py::is_operator()) \
        ;

#define LUISA_EXPORT_BOOL_OP(T) \
    m##T                           \
        .def("__eq__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a == b; }, py::is_operator()) \
        .def("__ne__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a != b; }, py::is_operator()) \
        .def("__and__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a && b; }, py::is_operator()) \
        .def("__or__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a || b; }, py::is_operator()) \
        ;

#define LUISA_EXPORT_INT_OP(T) \
    m##T \
        .def("__mod__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a % b; }, py::is_operator()) \
        .def("__mod__", [](const Vector<T,4>&a, const T&b) { return a % b; }, py::is_operator()) \
        .def("__rmod__", [](const Vector<T,4>&a, const T&b) { return b % a; }, py::is_operator()) \
        .def("__shl__", [](const Vector<T,4>&a, const T&b) { return a << b; }, py::is_operator()) \
        .def("__shr__", [](const Vector<T,4>&a, const T&b) { return a >> b; }, py::is_operator()) \
        .def("__xor__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return a ^ b; }, py::is_operator()) \
        ;

#define LUISA_EXPORT_FLOAT_OP(T) \
    m##T                    \
        .def("__pow__", [](const Vector<T,4>&a, const Vector<T,4>&b) { return luisa::pow(a, b); }, py::is_operator());

#define LUISA_EXPORT_UNARY_FUNC(T, name) \
    m.def(#name, [](const Vector<T,4>& v) { return luisa::name(v); });

#define LUISA_EXPORT_ARITHMETIC_FUNC(T) \
    m.def("min", [](const Vector<T,4>& a, const Vector<T,4>& b) { return luisa::min(a, b); }); \
    m.def("min", [](const Vector<T,4>& a, const T& b){ return luisa::min(a, b); });            \
    m.def("min", [](const T& a, const Vector<T,4>& b){ return luisa::min(a, b); });            \
    m.def("max", [](const Vector<T,4>& a, const Vector<T,4>& b) { return luisa::max(a, b); }); \
    m.def("max", [](const Vector<T,4>& a, const T& b){ return luisa::max(a, b); });            \
    m.def("max", [](const T& a, const Vector<T,4>& b){ return luisa::max(a, b); });            \
    m.def("select", [](const Vector<T,4>& a, const Vector<T,4>& b, bool pred) { return luisa::select(a, b, pred); }); \
    m.def("clamp", [](const Vector<T,4>& v, const T& a, const T& b) { return luisa::clamp(v, a, b); }); \

#define LUISA_EXPORT_FLOAT_FUNC(T) \
    m.def("pow", [](const Vector<T,4>& a, const Vector<T,4>& b) { return luisa::pow(a, b); }); \
    m.def("atan2", [](const Vector<T,4>& a, const Vector<T,4>& b) { return luisa::atan2(a, b); }); \
    m.def("lerp", [](const Vector<T,4>& a, const Vector<T,4>& b, float t) { return luisa::lerp(a, b, t); }); \
    m.def("dot", [](const Vector<T,4>& a, const Vector<T,4>& b) { return luisa::dot(a, b); });   \
    m.def("distance", [](const Vector<T,4>& a, const Vector<T,4>& b) { return luisa::distance(a, b); });   \
    LUISA_EXPORT_UNARY_FUNC(T, acos)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, asin)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, atan)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, cos)                                                            \
    LUISA_EXPORT_UNARY_FUNC(T, sin)                                                            \
    LUISA_EXPORT_UNARY_FUNC(T, tan)                                                            \
    LUISA_EXPORT_UNARY_FUNC(T, sqrt)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, ceil)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, floor)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, round)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, exp)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, log)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, log10)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, log2)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, abs)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, radians)                                                           \
    LUISA_EXPORT_UNARY_FUNC(T, degrees)                                                        \
    LUISA_EXPORT_UNARY_FUNC(T, length)                                                         \
    LUISA_EXPORT_UNARY_FUNC(T, normalize)

#define LUISA_EXPORT_VECTOR4(T) \
    py::class_<luisa::detail::VectorStorage<T, 4>>(m, "_vectorstorage_"#T"4"); \
    auto m##T = py::class_<Vector<T,4>, luisa::detail::VectorStorage<T, 4>>(m, #T"4") \
    	.def(py::init<T>()) \
    	.def(py::init<T,T,T,T>()) \
    	.def(py::init<Vector<T,4>>()) \
    	.def("__repr__", [](Vector<T,4>& self){return format(#T"4({},{},{},{})", self.x, self.y, self.z, self.w);}) \
        .def("__getitem__", [](Vector<T,4>& self, size_t i){return self[i];}) \
        .def("__setitem__", [](Vector<T,4>& self, size_t i, T k){ self[i]=k; }) \
        .def("copy", [](Vector<T,4>& self){return Vector<T,4>(self);}) \
    	.def_readwrite("x", &Vector<T,4>::x) \
    	.def_readwrite("y", &Vector<T,4>::y) \
    	.def_readwrite("z", &Vector<T,4>::z) \
    	.def_readwrite("w", &Vector<T,4>::w) \
		.def_property_readonly("xx", &Vector<T,4>::xx) \
		.def_property_readonly("xy", &Vector<T,4>::xy) \
		.def_property_readonly("xz", &Vector<T,4>::xz) \
		.def_property_readonly("xw", &Vector<T,4>::xw) \
		.def_property_readonly("yx", &Vector<T,4>::yx) \
		.def_property_readonly("yy", &Vector<T,4>::yy) \
		.def_property_readonly("yz", &Vector<T,4>::yz) \
		.def_property_readonly("yw", &Vector<T,4>::yw) \
		.def_property_readonly("zx", &Vector<T,4>::zx) \
		.def_property_readonly("zy", &Vector<T,4>::zy) \
		.def_property_readonly("zz", &Vector<T,4>::zz) \
		.def_property_readonly("zw", &Vector<T,4>::zw) \
		.def_property_readonly("wx", &Vector<T,4>::wx) \
		.def_property_readonly("wy", &Vector<T,4>::wy) \
		.def_property_readonly("wz", &Vector<T,4>::wz) \
		.def_property_readonly("ww", &Vector<T,4>::ww) \
		.def_property_readonly("xxx", &Vector<T,4>::xxx) \
		.def_property_readonly("xxy", &Vector<T,4>::xxy) \
		.def_property_readonly("xxz", &Vector<T,4>::xxz) \
		.def_property_readonly("xxw", &Vector<T,4>::xxw) \
		.def_property_readonly("xyx", &Vector<T,4>::xyx) \
		.def_property_readonly("xyy", &Vector<T,4>::xyy) \
		.def_property_readonly("xyz", &Vector<T,4>::xyz) \
		.def_property_readonly("xyw", &Vector<T,4>::xyw) \
		.def_property_readonly("xzx", &Vector<T,4>::xzx) \
		.def_property_readonly("xzy", &Vector<T,4>::xzy) \
		.def_property_readonly("xzz", &Vector<T,4>::xzz) \
		.def_property_readonly("xzw", &Vector<T,4>::xzw) \
		.def_property_readonly("xwx", &Vector<T,4>::xwx) \
		.def_property_readonly("xwy", &Vector<T,4>::xwy) \
		.def_property_readonly("xwz", &Vector<T,4>::xwz) \
		.def_property_readonly("xww", &Vector<T,4>::xww) \
		.def_property_readonly("yxx", &Vector<T,4>::yxx) \
		.def_property_readonly("yxy", &Vector<T,4>::yxy) \
		.def_property_readonly("yxz", &Vector<T,4>::yxz) \
		.def_property_readonly("yxw", &Vector<T,4>::yxw) \
		.def_property_readonly("yyx", &Vector<T,4>::yyx) \
		.def_property_readonly("yyy", &Vector<T,4>::yyy) \
		.def_property_readonly("yyz", &Vector<T,4>::yyz) \
		.def_property_readonly("yyw", &Vector<T,4>::yyw) \
		.def_property_readonly("yzx", &Vector<T,4>::yzx) \
		.def_property_readonly("yzy", &Vector<T,4>::yzy) \
		.def_property_readonly("yzz", &Vector<T,4>::yzz) \
		.def_property_readonly("yzw", &Vector<T,4>::yzw) \
		.def_property_readonly("ywx", &Vector<T,4>::ywx) \
		.def_property_readonly("ywy", &Vector<T,4>::ywy) \
		.def_property_readonly("ywz", &Vector<T,4>::ywz) \
		.def_property_readonly("yww", &Vector<T,4>::yww) \
		.def_property_readonly("zxx", &Vector<T,4>::zxx) \
		.def_property_readonly("zxy", &Vector<T,4>::zxy) \
		.def_property_readonly("zxz", &Vector<T,4>::zxz) \
		.def_property_readonly("zxw", &Vector<T,4>::zxw) \
		.def_property_readonly("zyx", &Vector<T,4>::zyx) \
		.def_property_readonly("zyy", &Vector<T,4>::zyy) \
		.def_property_readonly("zyz", &Vector<T,4>::zyz) \
		.def_property_readonly("zyw", &Vector<T,4>::zyw) \
		.def_property_readonly("zzx", &Vector<T,4>::zzx) \
		.def_property_readonly("zzy", &Vector<T,4>::zzy) \
		.def_property_readonly("zzz", &Vector<T,4>::zzz) \
		.def_property_readonly("zzw", &Vector<T,4>::zzw) \
		.def_property_readonly("zwx", &Vector<T,4>::zwx) \
		.def_property_readonly("zwy", &Vector<T,4>::zwy) \
		.def_property_readonly("zwz", &Vector<T,4>::zwz) \
		.def_property_readonly("zww", &Vector<T,4>::zww) \
		.def_property_readonly("wxx", &Vector<T,4>::wxx) \
		.def_property_readonly("wxy", &Vector<T,4>::wxy) \
		.def_property_readonly("wxz", &Vector<T,4>::wxz) \
		.def_property_readonly("wxw", &Vector<T,4>::wxw) \
		.def_property_readonly("wyx", &Vector<T,4>::wyx) \
		.def_property_readonly("wyy", &Vector<T,4>::wyy) \
		.def_property_readonly("wyz", &Vector<T,4>::wyz) \
		.def_property_readonly("wyw", &Vector<T,4>::wyw) \
		.def_property_readonly("wzx", &Vector<T,4>::wzx) \
		.def_property_readonly("wzy", &Vector<T,4>::wzy) \
		.def_property_readonly("wzz", &Vector<T,4>::wzz) \
		.def_property_readonly("wzw", &Vector<T,4>::wzw) \
		.def_property_readonly("wwx", &Vector<T,4>::wwx) \
		.def_property_readonly("wwy", &Vector<T,4>::wwy) \
		.def_property_readonly("wwz", &Vector<T,4>::wwz) \
		.def_property_readonly("www", &Vector<T,4>::www) \
		.def_property_readonly("xxxx", &Vector<T,4>::xxxx) \
		.def_property_readonly("xxxy", &Vector<T,4>::xxxy) \
		.def_property_readonly("xxxz", &Vector<T,4>::xxxz) \
		.def_property_readonly("xxxw", &Vector<T,4>::xxxw) \
		.def_property_readonly("xxyx", &Vector<T,4>::xxyx) \
		.def_property_readonly("xxyy", &Vector<T,4>::xxyy) \
		.def_property_readonly("xxyz", &Vector<T,4>::xxyz) \
		.def_property_readonly("xxyw", &Vector<T,4>::xxyw) \
		.def_property_readonly("xxzx", &Vector<T,4>::xxzx) \
		.def_property_readonly("xxzy", &Vector<T,4>::xxzy) \
		.def_property_readonly("xxzz", &Vector<T,4>::xxzz) \
		.def_property_readonly("xxzw", &Vector<T,4>::xxzw) \
		.def_property_readonly("xxwx", &Vector<T,4>::xxwx) \
		.def_property_readonly("xxwy", &Vector<T,4>::xxwy) \
		.def_property_readonly("xxwz", &Vector<T,4>::xxwz) \
		.def_property_readonly("xxww", &Vector<T,4>::xxww) \
		.def_property_readonly("xyxx", &Vector<T,4>::xyxx) \
		.def_property_readonly("xyxy", &Vector<T,4>::xyxy) \
		.def_property_readonly("xyxz", &Vector<T,4>::xyxz) \
		.def_property_readonly("xyxw", &Vector<T,4>::xyxw) \
		.def_property_readonly("xyyx", &Vector<T,4>::xyyx) \
		.def_property_readonly("xyyy", &Vector<T,4>::xyyy) \
		.def_property_readonly("xyyz", &Vector<T,4>::xyyz) \
		.def_property_readonly("xyyw", &Vector<T,4>::xyyw) \
		.def_property_readonly("xyzx", &Vector<T,4>::xyzx) \
		.def_property_readonly("xyzy", &Vector<T,4>::xyzy) \
		.def_property_readonly("xyzz", &Vector<T,4>::xyzz) \
		.def_property_readonly("xyzw", &Vector<T,4>::xyzw) \
		.def_property_readonly("xywx", &Vector<T,4>::xywx) \
		.def_property_readonly("xywy", &Vector<T,4>::xywy) \
		.def_property_readonly("xywz", &Vector<T,4>::xywz) \
		.def_property_readonly("xyww", &Vector<T,4>::xyww) \
		.def_property_readonly("xzxx", &Vector<T,4>::xzxx) \
		.def_property_readonly("xzxy", &Vector<T,4>::xzxy) \
		.def_property_readonly("xzxz", &Vector<T,4>::xzxz) \
		.def_property_readonly("xzxw", &Vector<T,4>::xzxw) \
		.def_property_readonly("xzyx", &Vector<T,4>::xzyx) \
		.def_property_readonly("xzyy", &Vector<T,4>::xzyy) \
		.def_property_readonly("xzyz", &Vector<T,4>::xzyz) \
		.def_property_readonly("xzyw", &Vector<T,4>::xzyw) \
		.def_property_readonly("xzzx", &Vector<T,4>::xzzx) \
		.def_property_readonly("xzzy", &Vector<T,4>::xzzy) \
		.def_property_readonly("xzzz", &Vector<T,4>::xzzz) \
		.def_property_readonly("xzzw", &Vector<T,4>::xzzw) \
		.def_property_readonly("xzwx", &Vector<T,4>::xzwx) \
		.def_property_readonly("xzwy", &Vector<T,4>::xzwy) \
		.def_property_readonly("xzwz", &Vector<T,4>::xzwz) \
		.def_property_readonly("xzww", &Vector<T,4>::xzww) \
		.def_property_readonly("xwxx", &Vector<T,4>::xwxx) \
		.def_property_readonly("xwxy", &Vector<T,4>::xwxy) \
		.def_property_readonly("xwxz", &Vector<T,4>::xwxz) \
		.def_property_readonly("xwxw", &Vector<T,4>::xwxw) \
		.def_property_readonly("xwyx", &Vector<T,4>::xwyx) \
		.def_property_readonly("xwyy", &Vector<T,4>::xwyy) \
		.def_property_readonly("xwyz", &Vector<T,4>::xwyz) \
		.def_property_readonly("xwyw", &Vector<T,4>::xwyw) \
		.def_property_readonly("xwzx", &Vector<T,4>::xwzx) \
		.def_property_readonly("xwzy", &Vector<T,4>::xwzy) \
		.def_property_readonly("xwzz", &Vector<T,4>::xwzz) \
		.def_property_readonly("xwzw", &Vector<T,4>::xwzw) \
		.def_property_readonly("xwwx", &Vector<T,4>::xwwx) \
		.def_property_readonly("xwwy", &Vector<T,4>::xwwy) \
		.def_property_readonly("xwwz", &Vector<T,4>::xwwz) \
		.def_property_readonly("xwww", &Vector<T,4>::xwww) \
		.def_property_readonly("yxxx", &Vector<T,4>::yxxx) \
		.def_property_readonly("yxxy", &Vector<T,4>::yxxy) \
		.def_property_readonly("yxxz", &Vector<T,4>::yxxz) \
		.def_property_readonly("yxxw", &Vector<T,4>::yxxw) \
		.def_property_readonly("yxyx", &Vector<T,4>::yxyx) \
		.def_property_readonly("yxyy", &Vector<T,4>::yxyy) \
		.def_property_readonly("yxyz", &Vector<T,4>::yxyz) \
		.def_property_readonly("yxyw", &Vector<T,4>::yxyw) \
		.def_property_readonly("yxzx", &Vector<T,4>::yxzx) \
		.def_property_readonly("yxzy", &Vector<T,4>::yxzy) \
		.def_property_readonly("yxzz", &Vector<T,4>::yxzz) \
		.def_property_readonly("yxzw", &Vector<T,4>::yxzw) \
		.def_property_readonly("yxwx", &Vector<T,4>::yxwx) \
		.def_property_readonly("yxwy", &Vector<T,4>::yxwy) \
		.def_property_readonly("yxwz", &Vector<T,4>::yxwz) \
		.def_property_readonly("yxww", &Vector<T,4>::yxww) \
		.def_property_readonly("yyxx", &Vector<T,4>::yyxx) \
		.def_property_readonly("yyxy", &Vector<T,4>::yyxy) \
		.def_property_readonly("yyxz", &Vector<T,4>::yyxz) \
		.def_property_readonly("yyxw", &Vector<T,4>::yyxw) \
		.def_property_readonly("yyyx", &Vector<T,4>::yyyx) \
		.def_property_readonly("yyyy", &Vector<T,4>::yyyy) \
		.def_property_readonly("yyyz", &Vector<T,4>::yyyz) \
		.def_property_readonly("yyyw", &Vector<T,4>::yyyw) \
		.def_property_readonly("yyzx", &Vector<T,4>::yyzx) \
		.def_property_readonly("yyzy", &Vector<T,4>::yyzy) \
		.def_property_readonly("yyzz", &Vector<T,4>::yyzz) \
		.def_property_readonly("yyzw", &Vector<T,4>::yyzw) \
		.def_property_readonly("yywx", &Vector<T,4>::yywx) \
		.def_property_readonly("yywy", &Vector<T,4>::yywy) \
		.def_property_readonly("yywz", &Vector<T,4>::yywz) \
		.def_property_readonly("yyww", &Vector<T,4>::yyww) \
		.def_property_readonly("yzxx", &Vector<T,4>::yzxx) \
		.def_property_readonly("yzxy", &Vector<T,4>::yzxy) \
		.def_property_readonly("yzxz", &Vector<T,4>::yzxz) \
		.def_property_readonly("yzxw", &Vector<T,4>::yzxw) \
		.def_property_readonly("yzyx", &Vector<T,4>::yzyx) \
		.def_property_readonly("yzyy", &Vector<T,4>::yzyy) \
		.def_property_readonly("yzyz", &Vector<T,4>::yzyz) \
		.def_property_readonly("yzyw", &Vector<T,4>::yzyw) \
		.def_property_readonly("yzzx", &Vector<T,4>::yzzx) \
		.def_property_readonly("yzzy", &Vector<T,4>::yzzy) \
		.def_property_readonly("yzzz", &Vector<T,4>::yzzz) \
		.def_property_readonly("yzzw", &Vector<T,4>::yzzw) \
		.def_property_readonly("yzwx", &Vector<T,4>::yzwx) \
		.def_property_readonly("yzwy", &Vector<T,4>::yzwy) \
		.def_property_readonly("yzwz", &Vector<T,4>::yzwz) \
		.def_property_readonly("yzww", &Vector<T,4>::yzww) \
		.def_property_readonly("ywxx", &Vector<T,4>::ywxx) \
		.def_property_readonly("ywxy", &Vector<T,4>::ywxy) \
		.def_property_readonly("ywxz", &Vector<T,4>::ywxz) \
		.def_property_readonly("ywxw", &Vector<T,4>::ywxw) \
		.def_property_readonly("ywyx", &Vector<T,4>::ywyx) \
		.def_property_readonly("ywyy", &Vector<T,4>::ywyy) \
		.def_property_readonly("ywyz", &Vector<T,4>::ywyz) \
		.def_property_readonly("ywyw", &Vector<T,4>::ywyw) \
		.def_property_readonly("ywzx", &Vector<T,4>::ywzx) \
		.def_property_readonly("ywzy", &Vector<T,4>::ywzy) \
		.def_property_readonly("ywzz", &Vector<T,4>::ywzz) \
		.def_property_readonly("ywzw", &Vector<T,4>::ywzw) \
		.def_property_readonly("ywwx", &Vector<T,4>::ywwx) \
		.def_property_readonly("ywwy", &Vector<T,4>::ywwy) \
		.def_property_readonly("ywwz", &Vector<T,4>::ywwz) \
		.def_property_readonly("ywww", &Vector<T,4>::ywww) \
		.def_property_readonly("zxxx", &Vector<T,4>::zxxx) \
		.def_property_readonly("zxxy", &Vector<T,4>::zxxy) \
		.def_property_readonly("zxxz", &Vector<T,4>::zxxz) \
		.def_property_readonly("zxxw", &Vector<T,4>::zxxw) \
		.def_property_readonly("zxyx", &Vector<T,4>::zxyx) \
		.def_property_readonly("zxyy", &Vector<T,4>::zxyy) \
		.def_property_readonly("zxyz", &Vector<T,4>::zxyz) \
		.def_property_readonly("zxyw", &Vector<T,4>::zxyw) \
		.def_property_readonly("zxzx", &Vector<T,4>::zxzx) \
		.def_property_readonly("zxzy", &Vector<T,4>::zxzy) \
		.def_property_readonly("zxzz", &Vector<T,4>::zxzz) \
		.def_property_readonly("zxzw", &Vector<T,4>::zxzw) \
		.def_property_readonly("zxwx", &Vector<T,4>::zxwx) \
		.def_property_readonly("zxwy", &Vector<T,4>::zxwy) \
		.def_property_readonly("zxwz", &Vector<T,4>::zxwz) \
		.def_property_readonly("zxww", &Vector<T,4>::zxww) \
		.def_property_readonly("zyxx", &Vector<T,4>::zyxx) \
		.def_property_readonly("zyxy", &Vector<T,4>::zyxy) \
		.def_property_readonly("zyxz", &Vector<T,4>::zyxz) \
		.def_property_readonly("zyxw", &Vector<T,4>::zyxw) \
		.def_property_readonly("zyyx", &Vector<T,4>::zyyx) \
		.def_property_readonly("zyyy", &Vector<T,4>::zyyy) \
		.def_property_readonly("zyyz", &Vector<T,4>::zyyz) \
		.def_property_readonly("zyyw", &Vector<T,4>::zyyw) \
		.def_property_readonly("zyzx", &Vector<T,4>::zyzx) \
		.def_property_readonly("zyzy", &Vector<T,4>::zyzy) \
		.def_property_readonly("zyzz", &Vector<T,4>::zyzz) \
		.def_property_readonly("zyzw", &Vector<T,4>::zyzw) \
		.def_property_readonly("zywx", &Vector<T,4>::zywx) \
		.def_property_readonly("zywy", &Vector<T,4>::zywy) \
		.def_property_readonly("zywz", &Vector<T,4>::zywz) \
		.def_property_readonly("zyww", &Vector<T,4>::zyww) \
		.def_property_readonly("zzxx", &Vector<T,4>::zzxx) \
		.def_property_readonly("zzxy", &Vector<T,4>::zzxy) \
		.def_property_readonly("zzxz", &Vector<T,4>::zzxz) \
		.def_property_readonly("zzxw", &Vector<T,4>::zzxw) \
		.def_property_readonly("zzyx", &Vector<T,4>::zzyx) \
		.def_property_readonly("zzyy", &Vector<T,4>::zzyy) \
		.def_property_readonly("zzyz", &Vector<T,4>::zzyz) \
		.def_property_readonly("zzyw", &Vector<T,4>::zzyw) \
		.def_property_readonly("zzzx", &Vector<T,4>::zzzx) \
		.def_property_readonly("zzzy", &Vector<T,4>::zzzy) \
		.def_property_readonly("zzzz", &Vector<T,4>::zzzz) \
		.def_property_readonly("zzzw", &Vector<T,4>::zzzw) \
		.def_property_readonly("zzwx", &Vector<T,4>::zzwx) \
		.def_property_readonly("zzwy", &Vector<T,4>::zzwy) \
		.def_property_readonly("zzwz", &Vector<T,4>::zzwz) \
		.def_property_readonly("zzww", &Vector<T,4>::zzww) \
		.def_property_readonly("zwxx", &Vector<T,4>::zwxx) \
		.def_property_readonly("zwxy", &Vector<T,4>::zwxy) \
		.def_property_readonly("zwxz", &Vector<T,4>::zwxz) \
		.def_property_readonly("zwxw", &Vector<T,4>::zwxw) \
		.def_property_readonly("zwyx", &Vector<T,4>::zwyx) \
		.def_property_readonly("zwyy", &Vector<T,4>::zwyy) \
		.def_property_readonly("zwyz", &Vector<T,4>::zwyz) \
		.def_property_readonly("zwyw", &Vector<T,4>::zwyw) \
		.def_property_readonly("zwzx", &Vector<T,4>::zwzx) \
		.def_property_readonly("zwzy", &Vector<T,4>::zwzy) \
		.def_property_readonly("zwzz", &Vector<T,4>::zwzz) \
		.def_property_readonly("zwzw", &Vector<T,4>::zwzw) \
		.def_property_readonly("zwwx", &Vector<T,4>::zwwx) \
		.def_property_readonly("zwwy", &Vector<T,4>::zwwy) \
		.def_property_readonly("zwwz", &Vector<T,4>::zwwz) \
		.def_property_readonly("zwww", &Vector<T,4>::zwww) \
		.def_property_readonly("wxxx", &Vector<T,4>::wxxx) \
		.def_property_readonly("wxxy", &Vector<T,4>::wxxy) \
		.def_property_readonly("wxxz", &Vector<T,4>::wxxz) \
		.def_property_readonly("wxxw", &Vector<T,4>::wxxw) \
		.def_property_readonly("wxyx", &Vector<T,4>::wxyx) \
		.def_property_readonly("wxyy", &Vector<T,4>::wxyy) \
		.def_property_readonly("wxyz", &Vector<T,4>::wxyz) \
		.def_property_readonly("wxyw", &Vector<T,4>::wxyw) \
		.def_property_readonly("wxzx", &Vector<T,4>::wxzx) \
		.def_property_readonly("wxzy", &Vector<T,4>::wxzy) \
		.def_property_readonly("wxzz", &Vector<T,4>::wxzz) \
		.def_property_readonly("wxzw", &Vector<T,4>::wxzw) \
		.def_property_readonly("wxwx", &Vector<T,4>::wxwx) \
		.def_property_readonly("wxwy", &Vector<T,4>::wxwy) \
		.def_property_readonly("wxwz", &Vector<T,4>::wxwz) \
		.def_property_readonly("wxww", &Vector<T,4>::wxww) \
		.def_property_readonly("wyxx", &Vector<T,4>::wyxx) \
		.def_property_readonly("wyxy", &Vector<T,4>::wyxy) \
		.def_property_readonly("wyxz", &Vector<T,4>::wyxz) \
		.def_property_readonly("wyxw", &Vector<T,4>::wyxw) \
		.def_property_readonly("wyyx", &Vector<T,4>::wyyx) \
		.def_property_readonly("wyyy", &Vector<T,4>::wyyy) \
		.def_property_readonly("wyyz", &Vector<T,4>::wyyz) \
		.def_property_readonly("wyyw", &Vector<T,4>::wyyw) \
		.def_property_readonly("wyzx", &Vector<T,4>::wyzx) \
		.def_property_readonly("wyzy", &Vector<T,4>::wyzy) \
		.def_property_readonly("wyzz", &Vector<T,4>::wyzz) \
		.def_property_readonly("wyzw", &Vector<T,4>::wyzw) \
		.def_property_readonly("wywx", &Vector<T,4>::wywx) \
		.def_property_readonly("wywy", &Vector<T,4>::wywy) \
		.def_property_readonly("wywz", &Vector<T,4>::wywz) \
		.def_property_readonly("wyww", &Vector<T,4>::wyww) \
		.def_property_readonly("wzxx", &Vector<T,4>::wzxx) \
		.def_property_readonly("wzxy", &Vector<T,4>::wzxy) \
		.def_property_readonly("wzxz", &Vector<T,4>::wzxz) \
		.def_property_readonly("wzxw", &Vector<T,4>::wzxw) \
		.def_property_readonly("wzyx", &Vector<T,4>::wzyx) \
		.def_property_readonly("wzyy", &Vector<T,4>::wzyy) \
		.def_property_readonly("wzyz", &Vector<T,4>::wzyz) \
		.def_property_readonly("wzyw", &Vector<T,4>::wzyw) \
		.def_property_readonly("wzzx", &Vector<T,4>::wzzx) \
		.def_property_readonly("wzzy", &Vector<T,4>::wzzy) \
		.def_property_readonly("wzzz", &Vector<T,4>::wzzz) \
		.def_property_readonly("wzzw", &Vector<T,4>::wzzw) \
		.def_property_readonly("wzwx", &Vector<T,4>::wzwx) \
		.def_property_readonly("wzwy", &Vector<T,4>::wzwy) \
		.def_property_readonly("wzwz", &Vector<T,4>::wzwz) \
		.def_property_readonly("wzww", &Vector<T,4>::wzww) \
		.def_property_readonly("wwxx", &Vector<T,4>::wwxx) \
		.def_property_readonly("wwxy", &Vector<T,4>::wwxy) \
		.def_property_readonly("wwxz", &Vector<T,4>::wwxz) \
		.def_property_readonly("wwxw", &Vector<T,4>::wwxw) \
		.def_property_readonly("wwyx", &Vector<T,4>::wwyx) \
		.def_property_readonly("wwyy", &Vector<T,4>::wwyy) \
		.def_property_readonly("wwyz", &Vector<T,4>::wwyz) \
		.def_property_readonly("wwyw", &Vector<T,4>::wwyw) \
		.def_property_readonly("wwzx", &Vector<T,4>::wwzx) \
		.def_property_readonly("wwzy", &Vector<T,4>::wwzy) \
		.def_property_readonly("wwzz", &Vector<T,4>::wwzz) \
		.def_property_readonly("wwzw", &Vector<T,4>::wwzw) \
		.def_property_readonly("wwwx", &Vector<T,4>::wwwx) \
		.def_property_readonly("wwwy", &Vector<T,4>::wwwy) \
		.def_property_readonly("wwwz", &Vector<T,4>::wwwz) \
		.def_property_readonly("wwww", &Vector<T,4>::wwww) \
		; \
	m.def("make_"#T"4", [](T a){return make_##T##4(a);}); \
	m.def("make_"#T"4", [](T a, T b, T c, T d){return make_##T##4(a,b,c,d);}); \
	m.def("make_"#T"4", [](Vector<T,4> a){return make_##T##4(a);});

void export_vector4(py::module &m) {
	LUISA_EXPORT_VECTOR4(bool)
	LUISA_EXPORT_VECTOR4(uint)
	LUISA_EXPORT_VECTOR4(int)
	LUISA_EXPORT_VECTOR4(float)
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
