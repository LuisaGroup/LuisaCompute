//
// Created by Mike Smith on 2021/6/24.
//

#include <luisa/ast/function_builder.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/vstl/pdqsort.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const Accel &accel) noexcept {
    _encoder.encode_accel(accel.handle());
    return *this;
}

}// namespace detail

Accel Device::create_accel(const AccelOption &option) noexcept {
    return _create<Accel>(option);
}

Accel::Accel(DeviceInterface *device, const AccelOption &option) noexcept
    : Resource{device, Resource::Tag::ACCEL, device->create_accel(option)} {}

Accel::~Accel() noexcept {
    if (*this) { device()->destroy_accel(handle()); }
}

luisa::unique_ptr<Command> Accel::_build(Accel::BuildRequest request,
                                         bool update_instance_buffer_only) noexcept {
    if (_mesh_handles.empty()) { LUISA_ERROR_WITH_LOCATION(
        "Building acceleration structure without instances."); }
    // collect modifications
    luisa::vector<Accel::Modification> modifications(_modifications.size());
    std::transform(_modifications.cbegin(), _modifications.cend(), modifications.begin(),
                   [](auto &&pair) noexcept { return pair.second; });
    _modifications.clear();
    pdqsort(modifications.begin(), modifications.end(),
            [](auto &&lhs, auto &&rhs) noexcept { return lhs.index < rhs.index; });
    return luisa::make_unique<AccelBuildCommand>(handle(), static_cast<uint>(_mesh_handles.size()),
                                                 request, std::move(modifications),
                                                 update_instance_buffer_only);
}

void Accel::_emplace_back_handle(uint64_t mesh, float4x4 const &transform, uint8_t visibility_mask, bool opaque) noexcept {
    auto index = static_cast<uint>(_mesh_handles.size());
    Modification modification{index};
    modification.set_primitive(mesh);
    modification.set_transform(transform);
    modification.set_visibility(visibility_mask);
    modification.set_opaque(opaque);
    _modifications[index] = modification;
    _mesh_handles.emplace_back(mesh);
}

void Accel::pop_back() noexcept {
    if (auto n = _mesh_handles.size()) {
        _mesh_handles.pop_back();
        _modifications.erase(n - 1u);
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Ignoring pop-back operation on empty accel.");
    }
}

void Accel::_set_handle(size_t index, uint64_t mesh, float4x4 const &transform, uint8_t visibility_mask, bool opaque) noexcept {
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        Modification modification{static_cast<uint>(index)};
        modification.set_transform(transform);
        modification.set_visibility(visibility_mask);
        modification.set_opaque(opaque);
        if (mesh != _mesh_handles[index]) [[likely]] {
            modification.set_primitive(mesh);
            _mesh_handles[index] = mesh;
        }
        _modifications[index] = modification;
    }
}

void Accel::set_transform_on_update(size_t index, float4x4 transform) noexcept {
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

void Accel::set_opaque_on_update(size_t index, bool opaque) noexcept {
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto [iter, _] = _modifications.try_emplace(
            index, Modification{static_cast<uint>(index)});
        iter->second.set_opaque(opaque);
    }
}

void Accel::set_visibility_on_update(size_t index, uint8_t visibility_mask) noexcept {
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto [iter, _] = _modifications.try_emplace(
            index, Modification{static_cast<uint>(index)});
        iter->second.set_visibility(visibility_mask);
    }
}

}// namespace luisa::compute

