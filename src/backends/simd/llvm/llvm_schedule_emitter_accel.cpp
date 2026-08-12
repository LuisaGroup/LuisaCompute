#include "llvm_schedule_emitter.h"

#include <array>

namespace luisa::compute::simd::detail {

namespace {

[[nodiscard]] bool is_float_array(
    const Type *type, uint32_t dimension) noexcept {
    return type != nullptr && type->is_array() &&
           type->dimension() == dimension &&
           type->element()->is_float32();
}

[[nodiscard]] bool is_ray_type(const Type *type) noexcept {
    return type != nullptr && type->is_structure() &&
           type->members().size() == 4u &&
           is_float_array(type->members()[0u], 3u) &&
           type->members()[1u]->is_float32() &&
           is_float_array(type->members()[2u], 3u) &&
           type->members()[3u]->is_float32();
}

[[nodiscard]] bool is_surface_hit_type(const Type *type) noexcept {
    return type != nullptr && type->is_structure() &&
           type->members().size() == 4u &&
           type->members()[0u]->is_uint32() &&
           type->members()[1u]->is_uint32() &&
           type->members()[2u]->is_vector() &&
           type->members()[2u]->dimension() == 2u &&
           type->members()[2u]->element()->is_float32() &&
           type->members()[3u]->is_float32();
}

}// namespace

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_accel_query(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op) {
        _fail("acceleration-structure query instruction is malformed");
        return nullptr;
    }
    if (_width > 16u) {
        _fail("Embree packet callbacks support SIMD widths up to 16");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceQueryOp>(
        *instruction.source_op);
    auto closest =
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST ||
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR;
    auto any =
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY ||
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR;
    auto motion =
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR ||
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR;
    auto expected_operand_count = motion ? 4u : 3u;
    if ((!closest && !any) ||
        instruction.operands.size() != expected_operand_count) {
        _fail("unsupported SIMD acceleration-structure query");
        return nullptr;
    }

    auto mask_operand = motion ? 3u : 2u;
    auto *result = _source.value(*instruction.result);
    auto *accel_value = _source.value(instruction.operands[0u]);
    auto *ray_value = _source.value(instruction.operands[1u]);
    auto *time_value = motion ?
                           _source.value(instruction.operands[2u]) :
                           nullptr;
    auto *mask_value = _source.value(instruction.operands[mask_operand]);
    auto *accel = _load_value(instruction.operands[0u]);
    auto *ray = ray_value == nullptr ? nullptr :
                                       _as_lane_vector(
                                           _load_value(instruction.operands[1u]), *ray_value);
    auto *time = time_value == nullptr ? nullptr :
                                         _as_lane_vector(
                                             _load_value(instruction.operands[2u]), *time_value);
    auto *visibility = mask_value == nullptr ? nullptr :
                                               _as_lane_vector(
                                                   _load_value(instruction.operands[mask_operand]), *mask_value);
    if (result == nullptr || accel_value == nullptr ||
        ray_value == nullptr || mask_value == nullptr ||
        accel == nullptr || ray == nullptr || visibility == nullptr ||
        (motion &&
         (time_value == nullptr || time == nullptr ||
          time_value->type == nullptr ||
          !time_value->type->is_float32())) ||
        accel_value->type == nullptr ||
        !accel_value->type->is_accel() ||
        !is_ray_type(ray_value->type) ||
        mask_value->type == nullptr ||
        !mask_value->type->is_uint32() ||
        (closest ? !is_surface_hit_type(result->type) :
                   !result->type->is_bool())) {
        _fail("acceleration-structure query has invalid operands or result");
        return nullptr;
    }

    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *float_lanes = ::llvm::FixedVectorType::get(
        _builder.getFloatTy(), _width);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *null_pointer =
        ::llvm::ConstantPointerNull::get(pointer_type);
    visibility = _builder.CreateZExtOrTrunc(visibility, i32_lanes);
    visibility = _builder.CreateSelect(
        _active_mask, visibility,
        ::llvm::Constant::getNullValue(i32_lanes),
        "accel.safe.visibility");

    auto *origin_type = _child_type(ray_value->type, 0u);
    auto *direction_type = _child_type(ray_value->type, 2u);
    auto *origin = _extract_child(
        ray, ray_value->type, 0u, true);
    auto *direction = _extract_child(
        ray, ray_value->type, 2u, true);
    std::array<::llvm::Value *, 8u> ray_components{
        _extract_child(origin, origin_type, 0u, true),
        _extract_child(origin, origin_type, 1u, true),
        _extract_child(origin, origin_type, 2u, true),
        _extract_child(ray, ray_value->type, 1u, true),
        _extract_child(direction, direction_type, 0u, true),
        _extract_child(direction, direction_type, 1u, true),
        _extract_child(direction, direction_type, 2u, true),
        _extract_child(ray, ray_value->type, 3u, true),
    };
    auto *zero_float = ::llvm::Constant::getNullValue(float_lanes);
    auto *one_float = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(_width),
        ::llvm::ConstantFP::get(_builder.getFloatTy(), 1.0));
    auto *ray_scratch_type = ::llvm::ArrayType::get(
        float_lanes, ray_components.size());
    auto *ray_scratch = _entry_scratch(
        ray_scratch_type,
        "accel.rays." +
            std::to_string(instruction.result->value));
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(ray_scratch_type), ray_scratch);
    for (auto component = uint32_t{0u};
         component < ray_components.size(); component++) {
        auto *fallback = component == 6u ? one_float : zero_float;
        auto *safe = _builder.CreateSelect(
            _active_mask, ray_components[component], fallback,
            "accel.safe.ray.component");
        auto *pointer = _builder.CreateGEP(
            ray_scratch_type, ray_scratch,
            {_builder.getInt32(0u), _builder.getInt32(component)});
        _builder.CreateStore(safe, pointer);
    }
    auto *mask_scratch = _entry_scratch(
        i32_lanes,
        "accel.masks." +
            std::to_string(instruction.result->value));
    _builder.CreateStore(visibility, mask_scratch);
    auto *time_scratch = static_cast<::llvm::Value *>(null_pointer);
    if (motion) {
        auto *safe_time = _builder.CreateSelect(
            _active_mask, time, zero_float,
            "accel.safe.ray.time");
        auto *scratch = _entry_scratch(
            float_lanes,
            "accel.times." +
                std::to_string(instruction.result->value));
        _builder.CreateStore(safe_time, scratch);
        time_scratch = scratch;
    }

    auto *object = _builder.CreateExtractValue(accel, {0u});
    auto callback_index = closest ? 1u : 2u;
    auto *callback = _builder.CreateExtractValue(
        accel, {callback_index});
    auto *missing_callback = _builder.CreateOr(
        _builder.CreateICmpEQ(object, null_pointer),
        _builder.CreateICmpEQ(callback, null_pointer));
    _trap_if(
        _builder.CreateAnd(
            _builder.CreateOrReduce(_active_mask), missing_callback),
        closest ? "accel.trace.closest.callback.null" :
                  "accel.trace.any.callback.null");
    auto *active_mask_bits = _bindless_callback_mask(
        result->value_class == schedule::ValueClass::varying);

    if (closest) {
        auto *ids_type = ::llvm::ArrayType::get(i32_lanes, 2u);
        auto *values_type = ::llvm::ArrayType::get(float_lanes, 3u);
        auto *ids = _entry_scratch(
            ids_type,
            "accel.closest.ids." +
                std::to_string(instruction.result->value));
        auto *values = _entry_scratch(
            values_type,
            "accel.closest.values." +
                std::to_string(instruction.result->value));
        auto *invalid_ids = ::llvm::ConstantArray::get(
            ids_type,
            {::llvm::Constant::getAllOnesValue(i32_lanes),
             ::llvm::Constant::getAllOnesValue(i32_lanes)});
        _builder.CreateStore(invalid_ids, ids);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(values_type), values);
        auto *callback_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {pointer_type, _builder.getInt32Ty(),
             _builder.getInt64Ty(), pointer_type,
             pointer_type, pointer_type, pointer_type,
             pointer_type},
            false);
        _builder.CreateCall(
            callback_type, callback,
            {object, _builder.getInt32(_width), active_mask_bits,
             ray_scratch, mask_scratch, time_scratch,
             ids, values});

        auto load_ids = [&](uint32_t component) {
            auto *pointer = _builder.CreateGEP(
                ids_type, ids,
                {_builder.getInt32(0u), _builder.getInt32(component)});
            return _builder.CreateLoad(i32_lanes, pointer);
        };
        auto load_values = [&](uint32_t component) {
            auto *pointer = _builder.CreateGEP(
                values_type, values,
                {_builder.getInt32(0u), _builder.getInt32(component)});
            return _builder.CreateLoad(float_lanes, pointer);
        };
        auto *bary_type = _child_type(result->type, 2u);
        auto *bary = _assemble(
            bary_type, true,
            [&](uint32_t component) { return load_values(component); });
        auto *hits = static_cast<::llvm::Value *>(
            ::llvm::PoisonValue::get(_data_type(result->type, true)));
        hits = _insert_child(
            hits, load_ids(0u), result->type, 0u, true);
        hits = _insert_child(
            hits, load_ids(1u), result->type, 1u, true);
        hits = _insert_child(
            hits, bary, result->type, 2u, true);
        hits = _insert_child(
            hits, load_values(2u), result->type, 3u, true);
        return result->value_class == schedule::ValueClass::varying ?
                   hits :
                   _extract_lane(
                       hits, result->type,
                       _safe_first_lane(_active_mask));
    }

    auto *occluded = _entry_scratch(
        i32_lanes,
        "accel.any.result." +
            std::to_string(instruction.result->value));
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(i32_lanes), occluded);
    auto *callback_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(),
        {pointer_type, _builder.getInt32Ty(),
         _builder.getInt64Ty(), pointer_type,
         pointer_type, pointer_type, pointer_type},
        false);
    _builder.CreateCall(
        callback_type, callback,
        {object, _builder.getInt32(_width), active_mask_bits,
         ray_scratch, mask_scratch, time_scratch, occluded});
    auto *bits = _builder.CreateICmpNE(
        _builder.CreateLoad(i32_lanes, occluded),
        ::llvm::Constant::getNullValue(i32_lanes));
    return result->value_class == schedule::ValueClass::varying ?
               bits :
               _builder.CreateExtractElement(
                   bits, _safe_first_lane(_active_mask));
}

}// namespace luisa::compute::simd::detail
