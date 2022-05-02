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

Accel Device::create_accel(AccelUsageHint hint) noexcept { return _create<Accel>(hint); }

Accel::Accel(Device::Interface *device, AccelUsageHint hint) noexcept
    : Resource{device, Resource::Tag::ACCEL, device->create_accel(hint)},
      _mutex{luisa::make_unique<std::mutex>()} {}

Command *Accel::build(Accel::BuildRequest request) noexcept {
    std::scoped_lock lock{*_mutex};
    if (_mesh_handles.empty()) { LUISA_ERROR_WITH_LOCATION(
        "Building acceleration structure without instances."); }
    // collect modifications
    luisa::vector<Accel::Modification> modifications(_modifications.size());
    std::transform(_modifications.cbegin(), _modifications.cend(), modifications.begin(),
                   [](auto &&pair) noexcept { return pair.second; });
    _modifications.clear();
    std::sort(modifications.begin(), modifications.end(),
              [](auto &&lhs, auto &&rhs) noexcept { return lhs.index < rhs.index; });
    return AccelBuildCommand::create(handle(), static_cast<uint>(_mesh_handles.size()),
                                     request, std::move(modifications));
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
    _emplace_back(mesh.handle(), transform, visible);
}

void Accel::_emplace_back(uint64_t mesh_handle, float4x4 transform, bool visible) noexcept {
    std::scoped_lock lock{*_mutex};
    auto index = static_cast<uint>(_mesh_handles.size());
    Modification modification{index};
    modification.set_mesh(mesh_handle);
    modification.set_transform(transform);
    modification.set_visibility(visible);
    _modifications[index] = modification;
    _mesh_handles.emplace_back(mesh_handle);
}

void Accel::pop_back() noexcept {
    std::scoped_lock lock{*_mutex};
    if (auto n = _mesh_handles.size()) {
        _mesh_handles.pop_back();
        _modifications.erase(n - 1u);
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
        Modification modification{static_cast<uint>(index)};
        modification.set_transform(transform);
        modification.set_visibility(visible);
        if (mesh.handle() != _mesh_handles[index]) [[likely]] {
            modification.set_mesh(mesh.handle());
            _mesh_handles[index] = mesh.handle();
        }
        _modifications[index] = modification;
    }
}

void Accel::set_transform_on_update(size_t index, float4x4 transform) noexcept {
    std::scoped_lock lock{*_mutex};
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto [iter, _] = _modifications.try_emplace(
            index, Modification{static_cast<uint>(index)});
        iter->second.set_transform(transform);
    }
}

void Accel::set_visibility_on_update(size_t index, bool visible) noexcept {
    std::scoped_lock lock{*_mutex};
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto [iter, _] = _modifications.try_emplace(
            index, Modification{static_cast<uint>(index)});
        iter->second.set_visibility(visible);
    }
}

}// namespace luisa::compute
