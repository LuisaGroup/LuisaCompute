//
// Created by Mike Smith on 2021/6/24.
//

#include <ast/function_builder.h>
#include <runtime/shader.h>
#include <rtx/accel.h>

namespace luisa::compute {

namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const Accel &accel) noexcept {
    _encode_pending_bindings();
    if (auto t = _kernel.arguments()[_argument_index].type();
        !t->is_accel()) {
        LUISA_ERROR_WITH_LOCATION(
            "Expected {} but got accel for argument {}.",
            t->description(), _argument_index);
    }
    auto v = _kernel.arguments()[_argument_index++].uid();
    _dispatch_command()->encode_accel(v, accel.handle());
    return *this;
}

}// namespace detail

Accel Device::create_accel(AccelBuildHint hint) noexcept { return _create<Accel>(hint); }

Accel::Accel(Device::Interface *device, AccelBuildHint hint) noexcept
    : Resource{device, Resource::Tag::ACCEL, device->create_accel(hint)},
      _rebuild_observer{luisa::make_shared<RebuildObserver>()} {}

Command *Accel::update() noexcept {
    if (_rebuild_observer->requires_rebuild()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Accel #{} requires rebuild rather than update. "
            "Automatically replacing with AccelBuildCommand.",
            handle());
        return build();
    }
    return AccelUpdateCommand::create(handle());
}

Var<Hit> Accel::trace_closest(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_closest(ray);
}

Var<bool> Accel::trace_any(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_any(ray);
}

Accel &Accel::emplace_back(const Mesh &mesh, float4x4 transform, bool vis) noexcept {
    _rebuild_observer->notify();
    device()->emplace_back_instance_in_accel(handle(), mesh.handle(), transform, vis);
    _mesh_subjects.emplace_back(mesh.subject())->add(_rebuild_observer);
    return *this;
}

Accel &Accel::set_transform(size_t index, float4x4 transform) noexcept {
    if (index >= size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    }
    device()->set_instance_transform_in_accel(handle(), index, transform);
    return *this;
}

Command *Accel::build() noexcept {
    _rebuild_observer->clear();
    return AccelBuildCommand::create(handle());
}

Accel &Accel::pop_back() noexcept {
    _rebuild_observer->notify();
    _mesh_subjects.back()->remove(_rebuild_observer.get());
    _mesh_subjects.pop_back();
    device()->pop_back_instance_from_accel(handle());
    return *this;
}

Accel &Accel::set(size_t index, const Mesh &mesh, float4x4 transform, bool visible) noexcept {
    if (index >= size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    }
    _rebuild_observer->notify();
    _mesh_subjects[index]->remove(_rebuild_observer.get());
    _mesh_subjects[index] = mesh.subject();
    mesh.subject()->add(_rebuild_observer);
    device()->set_instance_in_accel(handle(), index, mesh.handle(), transform, visible);
    return *this;
}

Accel &Accel::set_visibility(size_t index, bool visible) noexcept {
    if (index >= size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    }
    device()->set_instance_visibility_in_accel(handle(), index, visible);
    return *this;
}

}// namespace luisa::compute
