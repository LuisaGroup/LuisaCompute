#include <luisa/core/logging.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/rtx/ray_query.h>

namespace luisa::compute {

Var<Ray> TriangleCandidate::ray() const noexcept {
    return def<Ray>(detail::FunctionBuilder::current()->call(
        Type::of<Ray>(), CallOp::RAY_QUERY_WORLD_SPACE_RAY, {_query}));
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

Var<Ray> ProceduralCandidate::ray() const noexcept {
    return def<Ray>(detail::FunctionBuilder::current()->call(
        Type::of<Ray>(), CallOp::RAY_QUERY_WORLD_SPACE_RAY, {_query}));
}

namespace detail {

template<bool terminate_on_first>
[[nodiscard]] inline auto make_ray_query_object(const Expression *accel,
                                                const Expression *ray,
                                                const Expression *mask) noexcept {
    auto builder = detail::FunctionBuilder::current();
    auto type = Type::of<RayQueryBase<terminate_on_first>>();
    auto local = builder->local(type);
    CallOp op = terminate_on_first ?
                    CallOp::RAY_TRACING_QUERY_ANY :
                    CallOp::RAY_TRACING_QUERY_ALL;
    auto call = builder->call(type, op, {accel, ray, mask});
    builder->assign(local, call);
    return local;
}

template<bool terminate_on_first>
RayQueryBase<terminate_on_first>::RayQueryBase(const Expression *accel,
                                               const Expression *ray,
                                               const Expression *mask) noexcept
    : _stmt{detail::FunctionBuilder::current()->ray_query_(
          make_ray_query_object<terminate_on_first>(accel, ray, mask))} {}

template<bool terminate_on_first>
RayQueryBase<terminate_on_first>
RayQueryBase<terminate_on_first>::on_triangle_candidate(
    const RayQueryBase::TriangleCandidateHandler &handler) && noexcept {

    LUISA_ASSERT(_stmt != nullptr && !_triangle_handler_set &&
                     !_inside_triangle_handler && !_inside_procedural_handler,
                 "RayQueryBase::on_triangle_candidate() is in an invalid state.");
    _triangle_handler_set = true;
    _inside_triangle_handler = true;
    auto builder = detail::FunctionBuilder::current();
    builder->with(_stmt->on_triangle_candidate(), [&] {
        TriangleCandidate candidate{_stmt->query()};
        handler(candidate);
    });
    _inside_triangle_handler = false;
    return std::move(*this);
}

template<bool terminate_on_first>
RayQueryBase<terminate_on_first>
RayQueryBase<terminate_on_first>::on_procedural_candidate(
    const RayQueryBase::ProceduralCandidateHandler &handler) && noexcept {

    LUISA_ASSERT(_stmt != nullptr && !_procedural_handler_set &&
                     !_inside_triangle_handler && !_inside_procedural_handler,
                 "RayQueryBase::on_procedural_candidate() is in an invalid state.");
    _procedural_handler_set = true;
    _inside_procedural_handler = true;
    auto builder = detail::FunctionBuilder::current();
    builder->with(_stmt->on_procedural_candidate(), [&] {
        ProceduralCandidate candidate{_stmt->query()};
        handler(candidate);
    });
    _inside_procedural_handler = false;
    return std::move(*this);
}

template<bool terminate_on_first>
Var<CommittedHit> RayQueryBase<terminate_on_first>::trace() const noexcept {
    LUISA_ASSERT(_stmt != nullptr,
                 "RayQueryBase::trace() called on moved object.");
    return def<CommittedHit>(detail::FunctionBuilder::current()->call(
        Type::of<CommittedHit>(), CallOp::RAY_QUERY_COMMITTED_HIT, {_stmt->query()}));
}

template<bool terminate_on_first>
RayQueryBase<terminate_on_first>::RayQueryBase(RayQueryBase &&another) noexcept
    : _stmt{another._stmt},
      _triangle_handler_set{another._triangle_handler_set},
      _procedural_handler_set{another._procedural_handler_set},
      _inside_triangle_handler{another._inside_triangle_handler},
      _inside_procedural_handler{another._inside_procedural_handler} { another._stmt = nullptr; }

// export the template instantiations
template class RayQueryBase<false>;
template class RayQueryBase<true>;

}// namespace detail

}// namespace luisa::compute
