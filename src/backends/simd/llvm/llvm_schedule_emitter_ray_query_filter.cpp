#include "llvm_schedule_emitter.h"

namespace luisa::compute::simd::detail {

namespace {

[[nodiscard]] bool is_float2(const Type *type) noexcept {
    return type != nullptr && type->is_vector() &&
           type->dimension() == 2u &&
           type->element()->is_float32();
}

[[nodiscard]] bool is_surface_hit_type(const Type *type) noexcept {
    return type != nullptr && type->is_structure() &&
           type->members().size() == 4u &&
           type->members()[0u]->is_uint32() &&
           type->members()[1u]->is_uint32() &&
           is_float2(type->members()[2u]) &&
           type->members()[3u]->is_float32() &&
           type->size() == sizeof(SIMDHostRayQuerySurfaceHit);
}

}// namespace

::llvm::Value *ScheduleEmitter::_ray_query_surface_filter_read(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op ||
        instruction.operands.size() != 1u ||
        static_cast<xir::RayQueryObjectReadOp>(
            *instruction.source_op) !=
            xir::RayQueryObjectReadOp::
                RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT) {
        _fail("direct surface-filter handler has an invalid candidate read");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    if (result == nullptr ||
        result->value_class != schedule::ValueClass::varying ||
        !is_surface_hit_type(result->type) ||
        _surface_filter_ray_packet == nullptr ||
        _surface_filter_hit_packet == nullptr) {
        _fail("direct surface-filter handler candidate has an invalid type or ABI");
        return nullptr;
    }

    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *float_lanes = ::llvm::FixedVectorType::get(
        _builder.getFloatTy(), _width);
    // Embree has no W2 traversal ABI. A logical W2 provider pads to a native
    // W4 packet, so candidate fields retain a four-element physical stride
    // even though the handler consumes only lanes zero and one.
    auto physical_packet_width = _width == 2u ? 4u : _width;
    auto load_field = [&](::llvm::Value *packet,
                          uint32_t field) noexcept {
        auto *pointer = _builder.CreateGEP(
            _builder.getInt8Ty(), packet,
            _builder.getInt64(
                static_cast<uint64_t>(field) * physical_packet_width *
                sizeof(uint32_t)));
        auto *value = _builder.CreateAlignedLoad(
            i32_lanes, pointer, ::llvm::Align{alignof(uint32_t)});
        // The filter owns a full packet but Embree only defines candidate
        // fields in active physical lanes. Sanitizing immediately prevents
        // inactive garbage from reaching trapping or poison-producing ALU.
        return _builder.CreateSelect(
            _active_mask, value,
            ::llvm::Constant::getNullValue(i32_lanes));
    };
    auto load_hit_field = [&](uint32_t combined_field) noexcept {
        return load_field(
            _surface_filter_hit_packet,
            combined_field - simd_host_accel_ray_packet_field_count);
    };
    auto as_float = [&](::llvm::Value *value) noexcept {
        return _builder.CreateBitCast(value, float_lanes);
    };

    auto *bary_type = _child_type(result->type, 2u);
    auto *bary = _assemble(
        bary_type, true,
        [&](uint32_t component) {
            return as_float(load_hit_field(
                simd_host_accel_hit_u_field + component));
        });
    if (bary == nullptr) { return nullptr; }
    auto *hit = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(_data_type(result->type, true)));
    hit = _insert_child(
        hit, load_hit_field(simd_host_accel_hit_inst_field),
        result->type, 0u, true);
    hit = _insert_child(
        hit, load_hit_field(simd_host_accel_hit_prim_field),
        result->type, 1u, true);
    hit = _insert_child(hit, bary, result->type, 2u, true);
    hit = _insert_child(
        hit,
        as_float(load_field(
            _surface_filter_ray_packet,
            simd_host_accel_ray_tfar_field)),
        result->type, 3u, true);
    return hit;
}

void ScheduleEmitter::_ray_query_surface_filter_write(
    const schedule::Instruction &instruction) {
    if (!instruction.source_op ||
        instruction.operands.size() != 1u ||
        static_cast<xir::RayQueryObjectWriteOp>(
            *instruction.source_op) !=
            xir::RayQueryObjectWriteOp::
                RAY_QUERY_OBJECT_COMMIT_TRIANGLE ||
        _surface_filter_committed_mask_bits == nullptr) {
        _fail("direct surface-filter handler has an invalid candidate commit");
        return;
    }
    auto *packed = _builder.CreateBitCast(
        _active_mask,
        ::llvm::IntegerType::get(_module.getContext(), _width));
    auto *bits = _builder.CreateZExt(
        packed, _builder.getInt64Ty(),
        "surface.filter.commit.bits");
    auto *old_bits = _builder.CreateLoad(
        _builder.getInt64Ty(),
        _surface_filter_committed_mask_bits);
    _builder.CreateStore(
        _builder.CreateOr(old_bits, bits),
        _surface_filter_committed_mask_bits);
}

}// namespace luisa::compute::simd::detail
