#pragma once

#include <luisa/runtime/rtx/accel.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/dsl/rtx/hit.h>
#include <luisa/dsl/rtx/ray.h>

namespace luisa::compute {

struct AccelTraceOptions {
    CurveBasisSet curve_bases{CurveBasisSet::make_none()};
    UInt visibility_mask{0xffu};
};

#define LUISA_ACCEL_TRACE_DEPRECATED                                                          \
    [[deprecated(                                                                             \
        "\n\n"                                                                                \
        "Accel::trace_*(ray, vis_mask) and query_*(ray, vis_mask) are deprecated.\n"          \
        "Please use Accel::intersect_*/traverse_*(ray, const AccelTraceOptions &options) instead.\n" \
        "\n"                                                                                  \
        "Note: curve tracing is disabled by default for performance reasons. If you would\n"  \
        "      like to enable it, please specify the required curve bases in the options.\n"  \
        "\n")]]

template<>
struct LC_DSL_API Expr<Accel> {

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept;
    explicit Expr(const Accel &accel) noexcept;
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] Var<TriangleHit> trace_closest(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] Var<bool> trace_any(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] RayQueryAll query_all(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] RayQueryAny query_any(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    [[nodiscard]] Var<SurfaceHit> intersect(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] Var<bool> intersect_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAll traverse(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAny traverse_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_user_id(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_user_id(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_visibility_mask(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_visibility_mask(Expr<int> instance_id) const noexcept;
    void set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<int> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept;
    void set_instance_user_id(Expr<int> instance_id, Expr<uint> id) const noexcept;
    void set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<uint> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept;
    void set_instance_user_id(Expr<uint> instance_id, Expr<uint> id) const noexcept;
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

Expr(const Accel &) noexcept -> Expr<Accel>;

template<>
struct Var<Accel> : public Expr<Accel> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<Accel>{detail::FunctionBuilder::current()->accel()} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

using AccelVar = Var<Accel>;

namespace detail {

class LC_DSL_API AccelExprProxy {

private:
    Accel _accel;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(AccelExprProxy)

public:
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] Var<TriangleHit> trace_closest(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] Var<bool> trace_any(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] RayQueryAll query_all(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] RayQueryAny query_any(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    [[nodiscard]] Var<SurfaceHit> intersect(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] Var<bool> intersect_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAll traverse(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAny traverse_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_user_id(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_user_id(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_visibility_mask(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_visibility_mask(Expr<int> instance_id) const noexcept;
    void set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<int> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_visibility(Expr<uint> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept;
    void set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept;
    void set_instance_user_id(Expr<int> instance_id, Expr<uint> id) const noexcept;
    void set_instance_user_id(Expr<uint> instance_id, Expr<uint> id) const noexcept;
};

}// namespace detail

#undef LUISA_ACCEL_TRACE_DEPRECATED

}// namespace luisa::compute
