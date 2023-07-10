#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <luisa/ast/function_builder.h>
#include <luisa/runtime/dispatch_buffer.h>
namespace py = pybind11;
using namespace luisa;
using namespace luisa::compute;
constexpr auto pyref = py::return_value_policy::reference;
using luisa::compute::detail::FunctionBuilder;

template<typename T>
class raw_ptr {

private:
    T *_p;

public:
    [[nodiscard]] raw_ptr(T *p) noexcept : _p{p} {}
    [[nodiscard]] T *get() const noexcept { return _p; }
    [[nodiscard]] T *operator->() const noexcept { return _p; }
    [[nodiscard]] T &operator*() const noexcept { return *_p; }
    [[nodiscard]] explicit operator bool() const noexcept { return _p != nullptr; }
};

PYBIND11_DECLARE_HOLDER_TYPE(T, raw_ptr<T>, true)
void export_expr(py::module &m) {
    py::class_<Expression>(m, "Expression");
    py::class_<LiteralExpr, Expression>(m, "LiteralExpr");
    py::class_<RefExpr, Expression>(m, "RefExpr");
    py::class_<CallExpr, Expression>(m, "CallExpr");
    py::class_<UnaryExpr, Expression>(m, "UnaryExpr");
    py::class_<BinaryExpr, Expression>(m, "BinaryExpr");
    py::class_<MemberExpr, Expression>(m, "MemberExpr");
    py::class_<AccessExpr, Expression>(m, "AccessExpr");
    py::class_<CastExpr, Expression>(m, "CastExpr");
    // statement types
    py::class_<ScopeStmt>(m, "ScopeStmt")// not yet exporting base class (Statement)
        .def("__enter__", [](ScopeStmt &self) { FunctionBuilder::current()->push_scope(&self); })
        .def("__exit__", [](ScopeStmt &self, py::object &e1, py::object &e2, py::object &tb) { FunctionBuilder::current()->pop_scope(&self); });
    py::class_<IfStmt>(m, "IfStmt")
        .def("true_branch", py::overload_cast<>(&IfStmt::true_branch), pyref)// using overload_cast because there's also a const method variant
        .def("false_branch", py::overload_cast<>(&IfStmt::false_branch), pyref);
    py::class_<AutoDiffStmt>(m, "AutoDiffStmt")
        .def("body", py::overload_cast<>(&AutoDiffStmt::body), pyref);
    py::class_<SwitchStmt>(m, "SwitchStmt")
        .def("body", py::overload_cast<>(&SwitchStmt::body), pyref);
    py::class_<SwitchCaseStmt>(m, "SwitchCaseStmt")
        .def("body", py::overload_cast<>(&SwitchCaseStmt::body), pyref);
    py::class_<SwitchDefaultStmt>(m, "SwitchDefaultStmt")
        .def("body", py::overload_cast<>(&SwitchDefaultStmt::body), pyref);
    py::class_<LoopStmt>(m, "LoopStmt")
        .def("body", py::overload_cast<>(&LoopStmt::body), pyref);
    py::class_<ForStmt>(m, "ForStmt")
        .def("body", py::overload_cast<>(&ForStmt::body), pyref);
    py::class_<RayQueryStmt>(m, "RayQueryStmt")
        .def("on_triangle_candidate", py::overload_cast<>(&RayQueryStmt::on_triangle_candidate), pyref)
        .def("on_procedural_candidate", py::overload_cast<>(&RayQueryStmt::on_procedural_candidate), pyref);
    py::class_<Type, raw_ptr<Type>>(m, "Type")
        .def_static("from_", &Type::from, pyref)
        .def("size", &Type::size)
        .def("alignment", &Type::alignment)
        .def("is_scalar", &Type::is_scalar)
        .def("is_vector", &Type::is_vector)
        .def("is_matrix", &Type::is_matrix)
        .def("is_basic", &Type::is_basic)
        .def("is_array", &Type::is_array)
        .def("is_structure", &Type::is_structure)
        .def("is_buffer", &Type::is_buffer)
        .def("is_texture", &Type::is_texture)
        .def("is_bindless_array", &Type::is_bindless_array)
        .def("is_accel", &Type::is_accel)
        .def("is_custom", &Type::is_custom)
        .def("element", &Type::element, pyref)
        .def("description", &Type::description)
        .def("dimension", &Type::dimension)
        .def("is_custom_buffer", [](Type const *t) {
            return t == Type::of<IndirectDispatchBuffer>();
        })
        .def_static(
            "custom", [](luisa::string_view str) {
                return Type::custom(str);
            },
            pyref);
}
