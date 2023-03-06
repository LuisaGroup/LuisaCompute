#include <dsl/syntax.h>
#include <dsl/rtx/ray_query.h>
#include <core/logging.h>
#include <vstl/meta_lib.h>

namespace luisa::compute {

RayQuery::RayQuery(const CallExpr *func) noexcept {
    auto func_builder = detail::FunctionBuilder::current();
    _expr = func_builder->local(Type::from("LC_RayQuery"));
    func_builder->assign(_expr, func);
}

// Var<Hit> RayQuery::proceed(const Callback &triangle_callback, const Callback &prim_callback) noexcept {

//     auto func_builder = detail::FunctionBuilder::current();
//     detail::LoopStmtBuilder{} % [&]() noexcept {
//         // if(not_finished){
//         detail::IfStmtBuilder{!Expr<bool>{func_builder->call(Type::of<bool>(), CallOp::RAY_QUERY_PROCEED, {_expr})}} % [&]() noexcept {
//             break_();
//         };
//         rayquery_detail::rayquery_expr = _expr;

//         auto reset = vstd::scope_exit([&] {
//             rayquery_detail::rayquery_expr = nullptr;
//             rayquery_detail::state = rayquery_detail::State::None;
//         });
//         // if(is_triangle){
//         detail::IfStmtBuilder{Expr<bool>{func_builder->call(Type::of<bool>(), CallOp::RAY_QUERY_IS_CANDIDATE_TRIANGLE, {_expr})}} % [&]() noexcept {
//             rayquery_detail::state = rayquery_detail::State::Triangle;
//             triangle_callback(def<Hit>(func_builder->call(Type::of<Hit>(), CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT, {_expr})));
//         } / [&]() noexcept {
//             rayquery_detail::state = rayquery_detail::State::Primitive;
//             prim_callback(def<Hit>(func_builder->call(Type::of<Hit>(), CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT, {_expr})));
//         };
//     };
//     return def<Hit>(func_builder->call(Type::of<Hit>(), CallOp::RAY_QUERY_COMMITTED_HIT, {_expr}));
// }
Var<bool> RayQuery::proceed() const noexcept {
    return def<bool>(detail::FunctionBuilder::current()->call(Type::of<bool>(), CallOp::RAY_QUERY_PROCEED, {_expr}));
}
Var<Hit> RayQuery::triangle_candidate() const noexcept {
    return def<Hit>(detail::FunctionBuilder::current()->call(Type::of<Hit>(), CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT, {_expr}));
}
Var<Hit> RayQuery::procedural_candidate() const noexcept {
    return def<Hit>(detail::FunctionBuilder::current()->call(Type::of<Hit>(), CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT, {_expr}));
}
Var<bool> RayQuery::is_candidate_triangle() const noexcept {
    return def<bool>(detail::FunctionBuilder::current()->call(Type::of<bool>(), CallOp::RAY_QUERY_IS_CANDIDATE_TRIANGLE, {_expr}));
}
Var<bool> RayQuery::is_candidate_procedural() const noexcept {
    return !is_candidate_triangle();
}
void RayQuery::commit_triangle() const noexcept {
    detail::FunctionBuilder::current()->call(CallOp::RAY_QUERY_COMMIT_TRIANGLE, {_expr});
}
void RayQuery::commit_procedural(Expr<float> distance) const noexcept {
    detail::FunctionBuilder::current()->call(CallOp::RAY_QUERY_COMMIT_PROCEDURAL, {_expr, distance.expression()});
}
Var<Hit> RayQuery::committed_hit() const noexcept {
    return def<Hit>(detail::FunctionBuilder::current()->call(Type::of<Hit>(), CallOp::RAY_QUERY_COMMITTED_HIT, {_expr}));
}
}// namespace luisa::compute
