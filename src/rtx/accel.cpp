//
// Created by Mike Smith on 2021/6/24.
//

#include <ast/function_builder.h>
#include <runtime/shader.h>
#include <rtx/accel.h>

namespace luisa::compute {

namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const Accel &accel) noexcept {
    _command->encode_accel(accel.handle());
    return *this;
}

}// namespace detail

Accel Device::create_accel(AccelBuildHint hint) noexcept { return _create<Accel>(hint); }

Accel::Accel(Device::Interface *device, AccelBuildHint hint) noexcept
    : Resource{device, Resource::Tag::ACCEL, device->create_accel(hint)},
      _mutex{luisa::make_unique<std::mutex>()} {}

Command *Accel::update() noexcept {
    return AccelUpdateCommand::create(handle(), _get_update_requests());
}

luisa::vector<AccelUpdateRequest> Accel::_get_update_requests() noexcept {
    std::scoped_lock lock{*_mutex};
    eastl::vector<AccelUpdateRequest> requests;
    requests.reserve(_update_requests.size());
    for (auto [_, r] : _update_requests) { requests.emplace_back(r); }
    _update_requests.clear();
    return requests;
}

Command *Accel::build() noexcept {
    if (_mesh_handles.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("No mesh found in accel.");
    }
    return AccelBuildCommand::create(
        handle(), _mesh_handles, _get_update_requests());
}

Var<Hit> Accel::trace_closest(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_closest(ray);
}

Var<bool> Accel::trace_any(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_any(ray);
}

Var<float4x4> Accel::instance_transform(Expr<int> instance_id) const noexcept {
    return Expr<Accel>{*this}.instance_transform(instance_id);
}

Var<float4x4> Accel::instance_transform(Expr<uint> instance_id) const noexcept {
    return Expr<Accel>{*this}.instance_transform(instance_id);
}

void Accel::set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept {
    Expr<Accel>{*this}.set_instance_transform(instance_id, mat);
}

void Accel::set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept {
    Expr<Accel>{*this}.set_instance_transform(instance_id, mat);
}

void Accel::set_instance_visibility(Expr<int> instance_id, Expr<bool> vis) const noexcept {
    Expr<Accel>{*this}.set_instance_visibility(instance_id, vis);
}

void Accel::set_instance_visibility(Expr<uint> instance_id, Expr<bool> vis) const noexcept {
    Expr<Accel>{*this}.set_instance_visibility(instance_id, vis);
}

void Accel::emplace_back(const Mesh &mesh, float4x4 transform, bool visible) noexcept {
    std::scoped_lock lock{*_mutex};
    auto index = static_cast<uint>(_mesh_handles.size());
    _update_requests[index] = AccelUpdateRequest::encode(index, transform, visible);
    _mesh_handles.emplace_back(mesh.handle());
}

void Accel::pop_back() noexcept {
    std::scoped_lock lock{*_mutex};
    if (auto n = _mesh_handles.size()) {
        _mesh_handles.pop_back();
        _update_requests.erase(n - 1u);
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Ignoring pop-back operation on empty accel.");
    }
}

void Accel::set(size_t index, const Mesh &mesh, float4x4 transform, bool visible) noexcept {
    std::scoped_lock lock{*_mutex};
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        _update_requests[index] = AccelUpdateRequest::encode(index, transform, visible);
        _mesh_handles[index] = mesh.handle();
    }
}

void Accel::set_transform_on_update(size_t index, float4x4 transform) noexcept {
    std::scoped_lock lock{*_mutex};
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto r = AccelUpdateRequest::encode(index, transform);
        if (auto [iter, success] = _update_requests.try_emplace(index, r);
            !success) [[unlikely]] {// already exists
            iter->second.set_transform(transform);
        }
    }
}

void Accel::set_visibility_on_update(size_t index, bool visible) noexcept {
    std::scoped_lock lock{*_mutex};
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto r = AccelUpdateRequest::encode(index, visible);
        if (auto [iter, success] = _update_requests.try_emplace(index, r);
            !success) [[unlikely]] {// already exists
            iter->second.set_visibility(visible);
        }
    }
}

}// namespace luisa::compute
