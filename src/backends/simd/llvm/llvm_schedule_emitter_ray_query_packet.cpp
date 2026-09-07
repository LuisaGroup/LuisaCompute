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

std::pair<::llvm::Value *, ::llvm::Value *>
ScheduleEmitter::_ray_query_output_surface_filter_ray_packet_for_call(
    ::llvm::Value *ray_packet, ::llvm::Value *call_packet,
    ::llvm::Value *active_mask_bits, uint32_t runtime_flag,
    ::llvm::Value *narrowing_eligible,
    std::string_view label) {
#if LLVM_VERSION_MAJOR < 18
    static_cast<void>(runtime_flag);
    static_cast<void>(narrowing_eligible);
    static_cast<void>(label);
    auto *packet = _ray_query_surface_filter_ray_packet_for_call(
        ray_packet, call_packet, active_mask_bits);
    return {packet, packet == nullptr ? nullptr : _builder.getInt32(_width)};
#else
    if (_width != 16u || !_enable_native_vector_compress) {
        auto *packet = _ray_query_surface_filter_ray_packet_for_call(
            ray_packet, call_packet, active_mask_bits);
        return {packet, packet == nullptr ? nullptr : _builder.getInt32(_width)};
    }
    if (ray_packet == nullptr || call_packet == nullptr ||
        active_mask_bits == nullptr) {
        _fail("output-only surface-filter call packet has no analyzed storage");
        return {nullptr, nullptr};
    }

    auto *runtime_flags = _load_launch_u32(offsetof(
        SIMDPacketLaunchConfig, reserved_runtime_flags));
    auto *narrowing_enabled = _builder.CreateICmpNE(
        _builder.CreateAnd(
            runtime_flags,
            _builder.getInt32(runtime_flag)),
        _builder.getInt32(0u),
        std::string{"ray.query."} + std::string{label} +
            ".packet.narrowing.enabled");
    narrowing_enabled = _builder.CreateAnd(
        narrowing_enabled, narrowing_eligible,
        std::string{"ray.query."} + std::string{label} +
            ".packet.narrowing.eligible");
    auto lane_mask = (uint64_t{1u} << _width) - 1u;
    auto *active_count = _builder.CreateIntrinsic(
        _builder.getInt64Ty(), ::llvm::Intrinsic::ctpop,
        {_builder.CreateAnd(
            active_mask_bits, _builder.getInt64(lane_mask))},
        nullptr, "ray.query.empty.packet.active.count");

    auto &context = _module.getContext();
    auto *narrow4 = ::llvm::BasicBlock::Create(
        context, std::string{"ray.query."} + std::string{label} + ".packet.w4",
        _entry);
    auto *narrow8 = ::llvm::BasicBlock::Create(
        context, std::string{"ray.query."} + std::string{label} + ".packet.w8",
        _entry);
    auto *check8 = ::llvm::BasicBlock::Create(
        context, std::string{"ray.query."} + std::string{label} + ".packet.check.w8",
        _entry);
    auto *wide = ::llvm::BasicBlock::Create(
        context, std::string{"ray.query."} + std::string{label} + ".packet.wide",
        _entry);
    auto *ready = ::llvm::BasicBlock::Create(
        context, std::string{"ray.query."} + std::string{label} + ".packet.ready",
        _entry);
    auto *use4 = _builder.CreateAnd(
        narrowing_enabled,
        _builder.CreateICmpULE(active_count, _builder.getInt64(4u)));
    _builder.CreateCondBr(use4, narrow4, check8);
    _builder.SetInsertPoint(check8);
    auto *use8 = _builder.CreateAnd(
        narrowing_enabled,
        _builder.CreateICmpULE(active_count, _builder.getInt64(8u)));
    _builder.CreateCondBr(use8, narrow8, wide);

    auto *wide_lane_type = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *wide_packet_type = ::llvm::ArrayType::get(
        wide_lane_type, simd_host_accel_ray_packet_field_count);
    auto *compress =
#if LLVM_VERSION_MAJOR >= 22
        ::llvm::Intrinsic::getOrInsertDeclaration(
#else
        ::llvm::Intrinsic::getDeclaration(
#endif
            &_module, ::llvm::Intrinsic::masked_compressstore,
            {wide_lane_type});
    auto emit_narrow_packet = [&](uint32_t packet_width) noexcept {
        auto *narrow_lane_type = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), packet_width);
        auto *narrow_packet_type = ::llvm::ArrayType::get(
            narrow_lane_type, simd_host_accel_ray_packet_field_count);
        auto *zero_chunk_type = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), 16u);
        auto *zero_chunk = ::llvm::Constant::getNullValue(
            zero_chunk_type);
        auto packet_word_count =
            packet_width * simd_host_accel_ray_packet_field_count;
        for (auto word = uint32_t{0u}; word < packet_word_count;
             word += 16u) {
            auto *destination = _builder.CreateGEP(
                _builder.getInt32Ty(), call_packet,
                _builder.getInt64(word));
            _builder.CreateAlignedStore(
                zero_chunk, destination, ::llvm::Align{64u});
        }
        auto *direction_z_pointer = _builder.CreateGEP(
            narrow_packet_type, call_packet,
            {_builder.getInt32(0u), _builder.getInt32(6u)});
        _builder.CreateAlignedStore(
            _builder.CreateVectorSplat(
                packet_width, _builder.getInt32(0x3f800000u)),
            direction_z_pointer,
            ::llvm::Align{packet_width * sizeof(uint32_t)});
        for (auto field = uint32_t{0u};
             field < simd_host_accel_ray_packet_field_count; field++) {
            auto indices = std::array<::llvm::Value *, 2u>{
                _builder.getInt32(0u), _builder.getInt32(field)};
            auto *source_pointer = _builder.CreateGEP(
                wide_packet_type, ray_packet, indices);
            auto *loaded = _builder.CreateAlignedLoad(
                wide_lane_type, source_pointer,
                ::llvm::Align{_width * sizeof(uint32_t)});
            auto *destination_pointer = _builder.CreateGEP(
                narrow_packet_type, call_packet, indices);
            auto *store = _builder.CreateCall(
                compress, {loaded, destination_pointer, _active_mask});
            store->addParamAttr(
                1u, ::llvm::Attribute::getWithAlignment(
                        context,
                        ::llvm::Align{
                            packet_width * sizeof(uint32_t)}));
        }
        return call_packet;
    };

    _builder.SetInsertPoint(narrow4);
    auto *packet4 = emit_narrow_packet(4u);
    auto *narrow4_exit = _builder.GetInsertBlock();
    _builder.CreateBr(ready);

    _builder.SetInsertPoint(narrow8);
    auto *packet8 = emit_narrow_packet(8u);
    auto *narrow8_exit = _builder.GetInsertBlock();
    _builder.CreateBr(ready);

    _builder.SetInsertPoint(wide);
    auto *wide_packet = _ray_query_surface_filter_ray_packet_for_call(
        ray_packet, call_packet, active_mask_bits);
    if (wide_packet == nullptr) { return {nullptr, nullptr}; }
    auto *wide_exit = _builder.GetInsertBlock();
    _builder.CreateBr(ready);

    _builder.SetInsertPoint(ready);
    auto *packet = _builder.CreatePHI(
        ray_packet->getType(), 3u,
        std::string{"ray.query."} + std::string{label} + ".packet");
    auto *packet_width = _builder.CreatePHI(
        _builder.getInt32Ty(), 3u,
        std::string{"ray.query."} + std::string{label} +
            ".packet.width");
    packet->addIncoming(packet4, narrow4_exit);
    packet_width->addIncoming(_builder.getInt32(4u), narrow4_exit);
    packet->addIncoming(packet8, narrow8_exit);
    packet_width->addIncoming(_builder.getInt32(8u), narrow8_exit);
    packet->addIncoming(wide_packet, wide_exit);
    packet_width->addIncoming(_builder.getInt32(_width), wide_exit);
    return {packet, packet_width};
#endif
}

}// namespace luisa::compute::simd::detail
