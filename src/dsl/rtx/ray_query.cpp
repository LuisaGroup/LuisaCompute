#include <core/logging.h>
#include <dsl/syntax.h>
#include <dsl/rtx/ray_query.h>

namespace luisa::compute {

Var<CommittedHit> RayQuery::committed_hit() const noexcept {
    return def<CommittedHit>(detail::FunctionBuilder::current()->call(
        Type::of<CommittedHit>(), CallOp::RAY_QUERY_COMMITTED_HIT, {_expr}));
}

Var<TriangleHit> TriangleCandidate::hit() const noexcept {
    return def<TriangleHit>(detail::FunctionBuilder::current()->call(
        Type::of<TriangleHit>(),
        CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT,
        {_query}));
}

void TriangleCandidate::commit() const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_QUERY_COMMIT_TRIANGLE, {_query});
}

void TriangleCandidate::terminate() const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_QUERY_TERMINATE, {_query});
}

Var<ProceduralHit> ProceduralCandidate::hit() const noexcept {
    return def<ProceduralHit>(detail::FunctionBuilder::current()->call(
        Type::of<ProceduralHit>(),
        CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT,
        {_query}));
}

void ProceduralCandidate::commit(Expr<float> distance) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_QUERY_COMMIT_PROCEDURAL,
        {_query, distance.expression()});
}

void ProceduralCandidate::terminate() const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_QUERY_TERMINATE, {_query});
}

namespace detail {

[[nodiscard]] inline auto make_ray_query_object(const Expression *accel,
                                                const Expression *ray,
                                                const Expression *mask) noexcept {
    auto builder = detail::FunctionBuilder::current();
    auto local = builder->local(Type::of<RayQuery>());
    auto call = builder->call(Type::of<RayQuery>(),
                              CallOp::RAY_TRACING_QUERY_ALL,
                              {accel, ray, mask});
    builder->assign(local, call);
    return local;
}

RayQueryBuilder::RayQueryBuilder(const Expression *accel,
                                 const Expression *ray,
                                 const Expression *mask) noexcept
    : _stmt{detail::FunctionBuilder::current()->ray_query_(
          make_ray_query_object(accel, ray, mask))} {}

RayQuery RayQueryBuilder::query() &&noexcept {
    _queried = true;
    return RayQuery{_stmt->query()};
}

RayQueryBuilder RayQueryBuilder::on_triangle_candidate(
    const RayQueryBuilder::TriangleCandidateHandler &handler) &&noexcept {

    LUISA_ASSERT(_stmt != nullptr && !_triangle_handler_set && !_queried &&
                     !_inside_triangle_handler && !_inside_procedural_handler,
                 "RayQueryBuilder::on_triangle_candidate() is in an invalid state.");
    _triangle_handler_set = true;
    _inside_triangle_handler = true;
    auto builder = detail::FunctionBuilder::current();
    builder->with(_stmt->on_triangle_candidate(), [&] {
        TriangleCandidate candidate{this, _stmt->query()};
        handler(candidate);
    });
    _inside_triangle_handler = false;
    return std::move(*this);
}

RayQueryBuilder detail::RayQueryBuilder::on_procedural_candidate(
    const RayQueryBuilder::ProceduralCandidateHandler &handler) &&noexcept {

    LUISA_ASSERT(_stmt != nullptr && !_procedural_handler_set && !_queried &&
                     !_inside_triangle_handler && !_inside_procedural_handler,
                 "RayQueryBuilder::on_procedural_candidate() is in an invalid state.");
    _procedural_handler_set = true;
    _inside_procedural_handler = true;
    auto builder = detail::FunctionBuilder::current();
    builder->with(_stmt->on_procedural_candidate(), [&] {
        ProceduralCandidate candidate{this, _stmt->query()};
        handler(candidate);
    });
    _inside_procedural_handler = false;
    return std::move(*this);
}

RayQueryBuilder::RayQueryBuilder(RayQueryBuilder &&another) noexcept
    : _stmt{another._stmt},
      _queried{another._queried},
      _triangle_handler_set{another._triangle_handler_set},
      _procedural_handler_set{another._procedural_handler_set},
      _inside_triangle_handler{another._inside_triangle_handler},
      _inside_procedural_handler{another._inside_procedural_handler} { another._stmt = nullptr; }

RayQueryBuilder::~RayQueryBuilder() noexcept {
    if (_stmt != nullptr) {
        LUISA_ASSERT(_queried, "RayQueryBuilder is destructed "
                               "without calling query().");
    }
}

}// namespace detail

}// namespace luisa::compute
