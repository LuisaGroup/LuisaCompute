#include <luisa/ast/function_builder.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/vstl/pdqsort.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const Accel &accel) noexcept {
    accel._check_is_valid();
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
    if (!_modifications.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Accel #{} destroyed with {} uncommitted modifications. "
            "Did you forget to call build()?",
            this->handle(), _modifications.size());
    }
    if (*this) { device()->destroy_accel(handle()); }
}

luisa::unique_ptr<Command> Accel::_build(Accel::BuildRequest request,
                                         bool update_instance_buffer_only) noexcept {
    _check_is_valid();
    if (_mesh_size == 0) { LUISA_ERROR_WITH_LOCATION(
        "Building acceleration structure without instances."); }
    // collect modifications
    luisa::vector<Accel::Modification> modifications(_modifications.size());
    std::transform(_modifications.cbegin(), _modifications.cend(), modifications.begin(),
                   [](auto &&pair) noexcept { return pair.second; });
    _modifications.clear();
    pdqsort(modifications.begin(), modifications.end(),
            [](auto &&lhs, auto &&rhs) noexcept { return lhs.index < rhs.index; });
    return luisa::make_unique<AccelBuildCommand>(handle(), static_cast<uint>(_mesh_size),
                                                 request, std::move(modifications),
                                                 update_instance_buffer_only);
}

void Accel::_emplace_back_handle(uint64_t mesh, float4x4 const &transform, uint8_t visibility_mask, bool opaque, uint user_id) noexcept {
    _check_is_valid();
    auto index = static_cast<uint>(_mesh_size);
    Modification modification{index};
    modification.set_primitive(mesh);
    modification.set_transform(transform);
    modification.set_visibility(visibility_mask);
    modification.set_opaque(opaque);
    modification.set_user_id(user_id);
    _modifications[index] = modification;
    _mesh_size += 1;
}

void Accel::pop_back() noexcept {
    _check_is_valid();
    if (_mesh_size > 0) {
        _mesh_size -= 1;
        _modifications.erase(_mesh_size);
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Ignoring pop-back operation on empty accel.");
    }
}

void Accel::_set_handle(size_t index, uint64_t mesh, float4x4 const &transform, uint8_t visibility_mask, bool opaque, uint user_id) noexcept {
    _check_is_valid();
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        Modification modification{static_cast<uint>(index)};
        modification.set_transform(transform);
        modification.set_visibility(visibility_mask);
        modification.set_opaque(opaque);
        modification.set_primitive(mesh);
        modification.set_user_id(user_id);
        _modifications[index] = modification;
    }
}
void Accel::_set_prim_handle(size_t index, uint64_t prim_handle) noexcept {
    _check_is_valid();
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto [iter, _] = _modifications.try_emplace(
            index, Modification{static_cast<uint>(index)});
        iter->second.set_primitive(prim_handle);
    }
}
void Accel::set_transform_on_update(size_t index, float4x4 transform) noexcept {
    _check_is_valid();
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
    _check_is_valid();
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
    _check_is_valid();
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

void Accel::set_instance_user_id_on_update(size_t index, uint user_id) noexcept {
    _check_is_valid();
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto [iter, _] = _modifications.try_emplace(
            index, Modification{static_cast<uint>(index)});
        iter->second.set_user_id(user_id);
    }
}

}// namespace luisa::compute
