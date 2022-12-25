#include <rtx/ray_query.h>
#include <core/logging.h>
#include <vstl/meta_lib.h>

namespace luisa::compute {

namespace rayquery_detail {
enum class State : uint8_t {
    None,
    Primitive,
    Triangle
};
static thread_local State state{State::None};
static thread_local const Expression *rayquery_expr;
}// namespace rayquery_detail

RayQuery::RayQuery(const CallExpr *func) noexcept {
    auto func_builder = detail::FunctionBuilder::current();
    _expr = func_builder->local(Type::from("LC_RayQuery"));
    func_builder->assign(_expr, func);
}

#ifndef LC_DISABLE_DSL
Var<Hit> RayQuery::proceed(const Callback &triangle_callback, const Callback &prim_callback) noexcept {

    auto func_builder = detail::FunctionBuilder::current();
    detail::LoopStmtBuilder{} % [&]() noexcept {
        // if(not_finished){
        detail::IfStmtBuilder{!Expr<bool>{func_builder->call(Type::of<bool>(), CallOp::RAY_QUERY_PROCEED, {_expr})}} % [&]() noexcept {
            break_();
        };
        rayquery_detail::rayquery_expr = _expr;
        auto reset = vstd::scope_exit([&] {
            rayquery_detail::rayquery_expr = nullptr;
            rayquery_detail::state = rayquery_detail::State::None;
        });
        // if(is_triangle){
        detail::IfStmtBuilder{Expr<bool>{func_builder->call(Type::of<bool>(), CallOp::RAY_QUERY_IS_CANDIDATE_TRIANGLE, {_expr})}} % [&]() noexcept {
            rayquery_detail::state = rayquery_detail::State::Triangle;
            triangle_callback(def<Hit>(func_builder->call(Type::of<Hit>(), CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT, {_expr})));
        } / [&]() noexcept {
            rayquery_detail::state = rayquery_detail::State::Primitive;
            prim_callback(def<Hit>(func_builder->call(Type::of<Hit>(), CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT, {_expr})));
        };
    };
    return def<Hit>(func_builder->call(Type::of<Hit>(), CallOp::RAY_QUERY_COMMITTED_HIT, {_expr}));
}

void commit_triangle() noexcept {
    if (rayquery_detail::rayquery_expr == nullptr) [[unlikely]] {
        LUISA_ERROR("Commit only availiable in callback scope!");
    }
    if (rayquery_detail::state != rayquery_detail::State::Triangle) [[unlikely]] {
        LUISA_ERROR("Commit triangle only availiable in triangle callback scope!");
    }
    detail::FunctionBuilder::current()->call(CallOp::RAY_QUERY_COMMIT_TRIANGLE,
                                             {rayquery_detail::rayquery_expr});
}
void commit_primitive(Expr<float> distance) noexcept {
    if (rayquery_detail::rayquery_expr == nullptr) [[unlikely]] {
        LUISA_ERROR("Commit only availiable in callback scope!");
    }
    if (rayquery_detail::state != rayquery_detail::State::Primitive) [[unlikely]] {
        LUISA_ERROR("Commit primitive only availiable in primitive callback scope!");
    }
    detail::FunctionBuilder::current()->call(CallOp::RAY_QUERY_COMMIT_PROCEDURAL,
                                             {rayquery_detail::rayquery_expr, distance.expression()});
}
#endif

}// namespace luisa::compute
