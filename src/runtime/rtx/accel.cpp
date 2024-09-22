#include <luisa/core/stl/algorithm.h>
#include <luisa/ast/function_builder.h>
#include <luisa/runtime/shader.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/curve.h>
#include <luisa/runtime/rtx/procedural_primitive.h>
#include <luisa/runtime/rtx/motion_instance.h>
#include <luisa/runtime/rtx/accel.h>

namespace luisa::compute {

namespace detail {

void ShaderInvokeBase::encode(ShaderDispatchCmdEncoder &encoder, const Accel &accel) noexcept {
    accel._check_is_valid();
#ifndef NDEBUG
    if (accel.dirty()) [[unlikely]] {
        LUISA_WARNING("Dispatching shader with a dirty accel.");
    }
#endif
    encoder.encode_accel(accel.handle());
}

}// namespace detail

Accel Device::create_accel(const AccelOption &option) noexcept {
    return _create<Accel>(option);
}

Accel::Accel(DeviceInterface *device, const AccelOption &option) noexcept
    : Resource{device, Resource::Tag::ACCEL, device->create_accel(option)} {}

Accel::Accel(Accel &&rhs) noexcept
    : Resource{std::move(rhs)},
      _modifications{std::move(rhs._modifications)},
      _instance_count{rhs._instance_count} {
    rhs._instance_count = 0;
}

Accel &Accel::operator=(Accel &&rhs) noexcept {
    _move_from(std::move(rhs));
    return *this;
}

size_t Accel::size() const noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    return _instance_count;
}

bool Accel::dirty() const noexcept {
    _check_is_valid();
    std::lock_guard lck{_mtx};
    return !_modifications.empty();
}

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
    std::lock_guard lock{_mtx};
    if (_instance_count == 0) { LUISA_ERROR_WITH_LOCATION(
        "Building acceleration structure without instances."); }
    // collect modifications
    luisa::vector<Accel::Modification> modifications;
    modifications.push_back_uninitialized(_modifications.size());
    luisa::transform(_modifications.cbegin(), _modifications.cend(), modifications.begin(),
                     [](auto &&pair) noexcept -> auto && { return pair.second; });
    _modifications.clear();
    // Is sort necessary?
    // luisa::sort(modifications.begin(), modifications.end(),
    //         [](auto &&lhs, auto &&rhs) noexcept { return lhs.index < rhs.index; });
    return luisa::make_unique<AccelBuildCommand>(handle(), static_cast<uint>(_instance_count),
                                                 request, std::move(modifications),
                                                 update_instance_buffer_only);
}

void Accel::emplace_back_handle(uint64_t mesh, float4x4 const &transform, uint8_t visibility_mask, bool opaque, uint user_id) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    auto index = static_cast<uint>(_instance_count);
    Modification modification{index};
    modification.set_primitive(mesh);
    modification.set_transform(transform);
    modification.set_visibility(visibility_mask);
    modification.set_opaque(opaque);
    modification.set_user_id(user_id);
    _modifications[index] = modification;
    _instance_count += 1;
}

void Accel::pop_back() noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (_instance_count > 0) {
        _instance_count -= 1;
        _modifications.erase(_instance_count);
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Ignoring pop-back operation on empty accel.");
    }
}

void Accel::set_handle(size_t index, uint64_t mesh, float4x4 const &transform, uint8_t visibility_mask, bool opaque, uint user_id) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _instance_count) [[unlikely]] {
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

void Accel::set_prim_handle(size_t index, uint64_t prim_handle) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _instance_count) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto [iter, _] = _modifications.try_emplace(
            index, Modification{static_cast<uint>(index)});
        iter->second.set_primitive(prim_handle);
    }
}

void Accel::emplace_back(const Mesh &mesh, float4x4 transform, uint8_t visibility_mask, bool opaque, uint user_id) noexcept {
    emplace_back_handle(mesh.handle(), transform, visibility_mask, opaque, user_id);
}

void Accel::emplace_back(const Curve &curve, float4x4 transform, uint8_t visibility_mask, bool opaque, uint user_id) noexcept {
    emplace_back_handle(curve.handle(), transform, visibility_mask, opaque, user_id);
}

void Accel::emplace_back(const ProceduralPrimitive &prim, float4x4 transform, uint8_t visibility_mask, uint user_id) noexcept {
    emplace_back_handle(prim.handle(), transform, visibility_mask,
                        false /* procedural geometry is always non-opaque */, user_id);
}

void Accel::emplace_back(const MotionInstance &instance, float4x4 transform, uint8_t visibility_mask, bool opaque, uint user_id) noexcept {
    emplace_back_handle(instance.handle(), transform, visibility_mask, opaque, user_id);
}

void Accel::set(size_t index, const Mesh &mesh, float4x4 transform, uint8_t visibility_mask, bool opaque, uint user_id) noexcept {
    set_handle(index, mesh.handle(), transform, visibility_mask, opaque, user_id);
}

void Accel::set(size_t index, const Curve &curve, float4x4 transform, uint8_t visibility_mask, bool opaque, uint user_id) noexcept {
    set_handle(index, curve.handle(), transform, visibility_mask, opaque, user_id);
}

void Accel::set(size_t index, const ProceduralPrimitive &prim, float4x4 transform, uint8_t visibility_mask, uint user_id) noexcept {
    set_handle(index, prim.handle(), transform, visibility_mask, false, user_id);
}

void Accel::set(size_t index, const MotionInstance &instance, float4x4 transform, uint8_t visibility_mask, bool opaque, uint user_id) noexcept {
    set_handle(index, instance.handle(), transform, visibility_mask, opaque, user_id);
}

void Accel::set_mesh(size_t index, const Mesh &mesh) noexcept {
    set_prim_handle(index, mesh.handle());
}

void Accel::set_curve(size_t index, const Curve &curve) noexcept {
    set_prim_handle(index, curve.handle());
}

void Accel::set_procedural_primitive(size_t index, const ProceduralPrimitive &prim) noexcept {
    set_prim_handle(index, prim.handle());
}

void Accel::set_motion_instance(size_t index, const MotionInstance &instance) noexcept {
    set_prim_handle(index, instance.handle());
}

void Accel::set_transform_on_update(size_t index, float4x4 transform) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _instance_count) [[unlikely]] {
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
    std::lock_guard lock{_mtx};
    if (index >= _instance_count) [[unlikely]] {
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
    std::lock_guard lock{_mtx};
    if (index >= _instance_count) [[unlikely]] {
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
    std::lock_guard lock{_mtx};
    if (index >= _instance_count) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto [iter, _] = _modifications.try_emplace(
            index, Modification{static_cast<uint>(index)});
        iter->second.set_user_id(user_id);
    }
}

luisa::unique_ptr<Command> Accel::update_instance_buffer() noexcept {
    return _build(Accel::BuildRequest::PREFER_UPDATE, true);
}

luisa::unique_ptr<Command> Accel::build(BuildRequest request) noexcept {
    return _build(request, false);
}

const detail::AccelExprProxy *Accel::operator->() const noexcept {
    _check_is_valid();
    return reinterpret_cast<const detail::AccelExprProxy *>(this);
}

}// namespace luisa::compute
