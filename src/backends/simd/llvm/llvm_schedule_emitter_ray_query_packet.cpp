#include "llvm_schedule_emitter.h"

#include <array>

namespace luisa::compute::simd::detail {

bool ScheduleEmitter::_store_ray_query_surface_filter_ray_packet(
    const schedule::Value &ray_value, ::llvm::Value *ray,
    ::llvm::Value *safe_time, ::llvm::Value *visibility,
    uint32_t status_index) {
    if (status_index >=
        _ray_query_surface_filter_ray_packet_storage.size()) {
        return true;
    }
    auto *packet =
        _ray_query_surface_filter_ray_packet_storage[status_index];
    if (packet == nullptr) {
        _fail("surface-filter ray packet has no analyzed storage");
        return false;
    }
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *zero_i32 = ::llvm::Constant::getNullValue(i32_lanes);
    auto *origin_type = _child_type(ray_value.type, 0u);
    auto *direction_type = _child_type(ray_value.type, 2u);
    auto *origin = _extract_child(ray, ray_value.type, 0u, true);
    auto *direction = _extract_child(ray, ray_value.type, 2u, true);
    std::array<::llvm::Value *, 8u> ray_components{
        _extract_child(origin, origin_type, 0u, true),
        _extract_child(origin, origin_type, 1u, true),
        _extract_child(origin, origin_type, 2u, true),
        _extract_child(ray, ray_value.type, 1u, true),
        _extract_child(direction, direction_type, 0u, true),
        _extract_child(direction, direction_type, 1u, true),
        _extract_child(direction, direction_type, 2u, true),
        _extract_child(ray, ray_value.type, 3u, true),
    };
    auto *packet_type = ::llvm::ArrayType::get(
        i32_lanes, simd_host_accel_ray_packet_field_count);
    auto store_packet_field = [&](uint32_t field,
                                  ::llvm::Value *value) noexcept {
        auto *pointer = _builder.CreateGEP(
            packet_type, packet,
            {_builder.getInt32(0u), _builder.getInt32(field)});
        _builder.CreateMaskedStore(
            value, pointer,
            ::llvm::Align{_width * sizeof(uint32_t)},
            _active_mask);
    };
    auto float_bits = [&](::llvm::Value *value) noexcept {
        return _builder.CreateBitCast(value, i32_lanes);
    };
    auto embree_tnear_bits = [&](::llvm::Value *value) noexcept {
        auto *bits = float_bits(value);
        auto *sign = _builder.CreateAnd(
            bits, _builder.CreateVectorSplat(
                      _width, _builder.getInt32(0x80000000u)));
        auto *magnitude = _builder.CreateAnd(
            bits, _builder.CreateVectorSplat(
                      _width, _builder.getInt32(0x7fffffffu)));
        auto *clamped = _builder.CreateVectorSplat(
            _width, _builder.getInt32(0x80800000u));
        auto *negative = _builder.CreateICmpNE(
            sign, ::llvm::Constant::getNullValue(i32_lanes));
        auto *stepped = _builder.CreateSelect(
            negative,
            _builder.CreateAdd(
                bits, _builder.CreateVectorSplat(
                          _width, _builder.getInt32(1u))),
            _builder.CreateSub(
                bits, _builder.CreateVectorSplat(
                          _width, _builder.getInt32(1u))));
        auto *stepped_magnitude = _builder.CreateAnd(
            stepped, _builder.CreateVectorSplat(
                         _width, _builder.getInt32(0x7fffffffu)));
        stepped = _builder.CreateSelect(
            _builder.CreateICmpULT(
                stepped_magnitude,
                _builder.CreateVectorSplat(
                    _width, _builder.getInt32(0x00800000u))),
            clamped, stepped);
        auto *finite = _builder.CreateICmpULT(
            magnitude,
            _builder.CreateVectorSplat(
                _width, _builder.getInt32(0x7f800000u)));
        auto *nonzero = _builder.CreateICmpNE(
            magnitude, ::llvm::Constant::getNullValue(i32_lanes));
        return _builder.CreateSelect(
            finite, _builder.CreateSelect(nonzero, stepped, clamped),
            bits, "ray.query.embree.tnear.bits");
    };
    for (auto component = uint32_t{0u}; component < 7u; component++) {
        store_packet_field(
            component,
            component == 3u ?
                embree_tnear_bits(ray_components[component]) :
                float_bits(ray_components[component]));
    }
    store_packet_field(7u, float_bits(safe_time));
    store_packet_field(8u, float_bits(ray_components[7u]));
    store_packet_field(9u, visibility);
    store_packet_field(10u, _lane_ids());
    store_packet_field(11u, zero_i32);
    return true;
}

[[nodiscard]] ::llvm::Value *
ScheduleEmitter::_ray_query_surface_filter_ray_packet_for_call(
    ::llvm::Value *ray_packet, ::llvm::Value *call_packet,
    ::llvm::Value *active_mask_bits) {
    if (ray_packet == nullptr || call_packet == nullptr ||
        active_mask_bits == nullptr || _width < 2u) {
        _fail("surface-filter call packet has no analyzed storage");
        return nullptr;
    }
    auto *source = _builder.GetInsertBlock();
    auto &context = _module.getContext();
    auto *sanitize = ::llvm::BasicBlock::Create(
        context, "ray.query.surface.filter.packet.sanitize", _entry);
    auto *ready = ::llvm::BasicBlock::Create(
        context, "ray.query.surface.filter.packet.ready", _entry);
    auto lane_mask = (uint64_t{1u} << _width) - 1u;
    auto *full = _builder.CreateICmpEQ(
        _builder.CreateAnd(
            active_mask_bits, _builder.getInt64(lane_mask)),
        _builder.getInt64(lane_mask),
        "ray.query.surface.filter.packet.full");
    _builder.CreateCondBr(full, ready, sanitize);

    _builder.SetInsertPoint(sanitize);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *packet_type = ::llvm::ArrayType::get(
        i32_lanes, simd_host_accel_ray_packet_field_count);
    auto alignment = ::llvm::Align{_width * sizeof(uint32_t)};
    for (auto field = uint32_t{0u};
         field < simd_host_accel_ray_packet_field_count; field++) {
        auto indices = std::array<::llvm::Value *, 2u>{
            _builder.getInt32(0u), _builder.getInt32(field)};
        auto *source_pointer = _builder.CreateGEP(
            packet_type, ray_packet, indices);
        auto *loaded = _builder.CreateAlignedLoad(
            i32_lanes, source_pointer, alignment);
        auto safe_bits = field == 6u ? 0x3f800000u : 0u;
        auto *safe = _builder.CreateVectorSplat(
            _width, _builder.getInt32(safe_bits));
        auto *value = _builder.CreateSelect(
            _active_mask, loaded, safe,
            "ray.query.surface.filter.packet.safe");
        auto *destination_pointer = _builder.CreateGEP(
            packet_type, call_packet, indices);
        _builder.CreateAlignedStore(
            value, destination_pointer, alignment);
    }
    _builder.CreateBr(ready);
    auto *sanitize_exit = _builder.GetInsertBlock();

    _builder.SetInsertPoint(ready);
    auto *packet = _builder.CreatePHI(
        ray_packet->getType(), 2u,
        "ray.query.surface.filter.packet");
    packet->addIncoming(ray_packet, source);
    packet->addIncoming(call_packet, sanitize_exit);
    return packet;
}

}// namespace luisa::compute::simd::detail
