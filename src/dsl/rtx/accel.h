#pragma once

#include <runtime/rtx/accel.h>
#include <dsl/rtx/ray_query.h>
#include <dsl/rtx/hit.h>
#include <dsl/rtx/ray.h>

namespace luisa::compute {

template<>
struct LC_DSL_API Expr<Accel> {

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept;
    Expr(const Accel &accel) noexcept;
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] Var<Hit> trace_closest(Expr<Ray> ray, Expr<uint> vis_mask = 255u) const noexcept;
    [[nodiscard]] RayQuery trace_all(Expr<Ray> ray, Expr<uint> vis_mask = 255u) const noexcept;
    [[nodiscard]] Var<bool> trace_any(Expr<Ray> ray, Expr<uint> vis_mask = 255u) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<int> instance_id) const noexcept;
    void set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<int> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept;
    void set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<uint> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept;
};

Expr(const Accel &) -> Expr<Accel>;

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
    [[nodiscard]] Var<Hit> trace_closest(Expr<Ray> ray, Expr<uint> vis_mask = 255u) const noexcept;
    [[nodiscard]] RayQuery trace_all(Expr<Ray> ray, Expr<uint> vis_mask = 255u) const noexcept;
    [[nodiscard]] Var<bool> trace_any(Expr<Ray> ray, Expr<uint> vis_mask = 255u) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<uint> instance_id) const noexcept;
    void set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<int> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_visibility(Expr<uint> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept;
    void set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept;
};

}// namespace detail

}// namespace luisa::compute
