#include <pybind11/pybind11.h>
#include <luisa-compute.h>
namespace py = pybind11;
using namespace luisa::compute;
const auto pyref = py::return_value_policy::reference; // object lifetime is managed on C++ side


void export_matrix(py::module &m) {
    py::class_<float2x2>(m, "float2x2")
        .def("identity", [](){return float2x2();})
        .def("__getitem__", [](float2x2& self, size_t i){return &self[i];}, pyref)
        .def("__setitem__", [](float2x2& self, size_t i, float2 k){ self[i]=k; })
        .def(py::init([](float a){return make_float2x2(a);}))
        .def(py::init([](float a, float b, float c, float d){return make_float2x2(a,b,c,d);}))
        .def(py::init([](float2 a, float2 b){return make_float2x2(a,b);}))
        .def(py::init([](float2x2 a){return make_float2x2(a);}))
        .def(py::init([](float3x3 a){return make_float2x2(a);}))
        .def(py::init([](float4x4 a){return make_float2x2(a);}))
        .def("__repr__", [](float2x2& self){return format("float2x2([{},{}], [{},{}])",
            self[0].x, self[0].y, self[1].x, self[1].y);});

    py::class_<float3x3>(m, "float3x3")
        .def("identity", [](){return float3x3();})
        .def("__getitem__", [](float3x3& self, size_t i){return &self[i];}, pyref)
        .def("__setitem__", [](float3x3& self, size_t i, float3 k){ self[i]=k; })
        .def(py::init([](float a){return make_float3x3(a);}))
        .def(py::init([](
            float m00, float m01, float m02,
            float m10, float m11, float m12,
            float m20, float m21, float m22)
            {return make_float3x3(m00,m01,m02, m10,m11,m12, m20,m21,m22);}))
        .def(py::init([](float3 a, float3 b, float3 c){return make_float3x3(a,b,c);}))
        .def(py::init([](float2x2 a){return make_float3x3(a);}))
        .def(py::init([](float3x3 a){return make_float3x3(a);}))
        .def(py::init([](float4x4 a){return make_float3x3(a);}))
        .def("__repr__", [](float3x3& self){return format("float3x3([{},{},{}], [{},{},{}], [{},{},{}])",
            self[0].x, self[0].y, self[0].z, self[1].x, self[1].y, self[1].z, self[2].x, self[2].y, self[2].z);});

    py::class_<float4x4>(m, "float4x4")
        .def("identity", [](){return float4x4();})
        .def("__getitem__", [](float4x4& self, size_t i){return &self[i];}, pyref)
        .def("__setitem__", [](float4x4& self, size_t i, float4 k){ self[i]=k; })
        .def(py::init([](float a){return make_float4x4(a);}))
        .def(py::init([](
            float m00, float m01, float m02, float m03,
            float m10, float m11, float m12, float m13,
            float m20, float m21, float m22, float m23,
            float m30, float m31, float m32, float m33)
            {return make_float4x4(m00,m01,m02,m03, m10,m11,m12,m13, m20,m21,m22,m23, m30,m31,m32,m33);}))
        .def(py::init([](float4 a, float4 b, float4 c, float4 d){return make_float4x4(a,b,c,d);}))
        .def(py::init([](float2x2 a){return make_float4x4(a);}))
        .def(py::init([](float3x3 a){return make_float4x4(a);}))
        .def(py::init([](float4x4 a){return make_float4x4(a);}))
        .def("__repr__", [](float4x4& self){return format("float4x4([{},{},{},{}], [{},{},{},{}], [{},{},{},{}], [{},{},{},{}])",
            self[0].x, self[0].y, self[0].z, self[0].w, self[1].x, self[1].y, self[1].z, self[1].w,
            self[2].x, self[2].y, self[2].z, self[2].w, self[3].x, self[3].y, self[3].z, self[3].w);});

    m.def("make_float2x2", [](float a){return make_float2x2(a);});
    m.def("make_float2x2", [](float a, float b, float c, float d){return make_float2x2(a,b,c,d);});
    m.def("make_float2x2", [](float2 a, float2 b){return make_float2x2(a,b);});
    m.def("make_float2x2", [](float2x2 a){return make_float2x2(a);});
    m.def("make_float2x2", [](float3x3 a){return make_float2x2(a);});
    m.def("make_float2x2", [](float4x4 a){return make_float2x2(a);});

    m.def("make_float3x3", [](float a){return make_float3x3(a);});
    m.def("make_float3x3", [](
        float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22)
        {return make_float3x3(m00,m01,m02, m10,m11,m12, m20,m21,m22);});
    m.def("make_float3x3", [](float3 a, float3 b, float3 c){return make_float3x3(a,b,c);});
    m.def("make_float3x3", [](float2x2 a){return make_float3x3(a);});
    m.def("make_float3x3", [](float3x3 a){return make_float3x3(a);});
    m.def("make_float3x3", [](float4x4 a){return make_float3x3(a);});

    m.def("make_float4x4", [](float a){return make_float4x4(a);});
    m.def("make_float4x4", [](
        float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33)
        {return make_float4x4(m00,m01,m02,m03, m10,m11,m12,m13, m20,m21,m22,m23, m30,m31,m32,m33);});
    m.def("make_float4x4", [](float4 a, float4 b, float4 c, float4 d){return make_float4x4(a,b,c,d);});
    m.def("make_float4x4", [](float2x2 a){return make_float4x4(a);});
    m.def("make_float4x4", [](float3x3 a){return make_float4x4(a);});
    m.def("make_float4x4", [](float4x4 a){return make_float4x4(a);});

    // TODO export matrix operators
}