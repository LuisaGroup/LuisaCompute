//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <runtime/device.h>
#include <runtime/buffer.h>
#include <rtx/ray.h>
#include <rtx/hit.h>
#include <rtx/mesh.h>

namespace luisa::compute {

class Accel;

class Instance {

private:
    Accel *_geometry;
    size_t _index;

private:
    friend class Accel;
    Instance(Accel *geom, size_t index) noexcept
        : _geometry{geom}, _index{index} {}

public:
    [[nodiscard]] uint64_t mesh_handle() const noexcept;
    void set_mesh(const Mesh &mesh) noexcept;
    void set_transform(float4x4 m) noexcept;
};

class Accel : concepts::Noncopyable {

private:
    Device::Handle _device;
    uint64_t _handle;
    std::vector<uint64_t> _instance_mesh_handles;
    std::vector<float4x4> _instance_transforms;
    bool _built{false};
    bool _dirty{false};

private:
    friend class Device;
    friend class Instance;
    explicit Accel(Device::Handle device) noexcept;

    void _destroy() noexcept;
    void _mark_dirty() noexcept;
    void _mark_should_rebuild() noexcept;
    void _check_built() const noexcept;

public:
    ~Accel() noexcept;
    Accel(Accel &&) noexcept = default;
    Accel &operator=(Accel &&rhs) noexcept;
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<Hit> hits) const noexcept;
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits) const noexcept;
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<bool> hits) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *update() noexcept;
    [[nodiscard]] Command *build(AccelBuildHint mode) noexcept;
    [[nodiscard]] Instance add(const Mesh &mesh, float4x4 transform) noexcept;
    [[nodiscard]] Instance instance(size_t i) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] explicit operator bool() const noexcept { return _device != nullptr; }

    // shader functions
    [[nodiscard]] detail::Expr<Hit> trace_closest(detail::Expr<Ray> ray) const noexcept;
    [[nodiscard]] detail::Expr<bool> trace_any(detail::Expr<Ray> ray) const noexcept;
};

namespace detail {

template<>
struct Expr<Accel> {

public:
    using ValueType = TextureHeap;

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}
    explicit Expr(const Accel &accel) noexcept
        : _expression{FunctionBuilder::current()->accel_binding(accel.handle())} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto trace_closest(Expr<Ray> ray) const noexcept {
        return Expr<Hit>{FunctionBuilder::current()->call(
            Type::of<Hit>(), CallOp::TRACE_CLOSEST,
            {_expression, ray.expression()})};
    }
    [[nodiscard]] auto trace_any(Expr<Ray> ray) const noexcept {
        return Expr<bool>{FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::TRACE_ANY,
            {_expression, ray.expression()})};
    }
};

}// namespace detail

template<>
struct Var<Accel> : public detail::Expr<Accel> {
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<Accel>{
            detail::FunctionBuilder::current()->accel()} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

using AccelVar = Var<Accel>;

}// namespace luisa::compute
