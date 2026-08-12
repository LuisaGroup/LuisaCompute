#include "llvm_schedule_emitter.h"

#include <array>

namespace luisa::compute::simd::detail {

namespace {

[[nodiscard]] bool is_ray_query_type(const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           (type == Type::custom("LC_RayQueryAll") ||
            type == Type::custom("LC_RayQueryAny"));
}

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
           type->members()[3u]->is_float32() &&
           type->size() == sizeof(SIMDHostRayQueryState::world_ray);
}

[[nodiscard]] bool is_surface_hit_type(const Type *type) noexcept {
    return type != nullptr && type->is_structure() &&
           type->members().size() == 4u &&
           type->members()[0u]->is_uint32() &&
           type->members()[1u]->is_uint32() &&
           type->members()[2u]->is_vector() &&
           type->members()[2u]->dimension() == 2u &&
           type->members()[2u]->element()->is_float32() &&
           type->members()[3u]->is_float32() &&
           type->size() == sizeof(SIMDHostRayQuerySurfaceHit);
}

[[nodiscard]] bool is_procedural_hit_type(const Type *type) noexcept {
    return type != nullptr && type->is_structure() &&
           type->members().size() == 2u &&
           type->members()[0u]->is_uint32() &&
           type->members()[1u]->is_uint32() &&
           type->size() == 8u;
}

[[nodiscard]] bool is_committed_hit_type(const Type *type) noexcept {
    return type != nullptr && type->is_structure() &&
           type->members().size() == 5u &&
           type->members()[0u]->is_uint32() &&
           type->members()[1u]->is_uint32() &&
           type->members()[2u]->is_vector() &&
           type->members()[2u]->dimension() == 2u &&
           type->members()[2u]->element()->is_float32() &&
           type->members()[3u]->is_uint32() &&
           type->members()[4u]->is_float32() &&
           type->size() == sizeof(SIMDHostRayQueryCommittedHit);
}

}// namespace

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_ray_query_create(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op || _width > 16u ||
        (_width != 1u && _width != 2u && _width != 4u &&
         _width != 8u && _width != 16u)) {
        _fail("ray-query construction is malformed or has an unsupported width");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceQueryOp>(*instruction.source_op);
    auto query_all =
        op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL ||
        op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR;
    auto query_any =
        op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY ||
        op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR;
    auto motion =
        op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR ||
        op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR;
    auto expected_operand_count = motion ? 4u : 3u;
    if ((!query_all && !query_any) ||
        instruction.operands.size() != expected_operand_count) {
        _fail("unsupported SIMD ray-query construction operation");
        return nullptr;
    }

    auto *result = _source.value(*instruction.result);
    auto *accel_value = _source.value(instruction.operands[0u]);
    auto *ray_value = _source.value(instruction.operands[1u]);
    auto *time_value = motion ?
                           _source.value(instruction.operands[2u]) :
                           nullptr;
    auto mask_operand = motion ? 3u : 2u;
    auto *mask_value = _source.value(instruction.operands[mask_operand]);
    auto *accel = _load_value(instruction.operands[0u]);
    auto *ray = ray_value == nullptr ? nullptr :
                                       _as_lane_vector(
                                           _load_value(instruction.operands[1u]),
                                           *ray_value);
    auto *time = time_value == nullptr ? nullptr :
                                         _as_lane_vector(
                                             _load_value(instruction.operands[2u]),
                                             *time_value);
    auto *visibility = mask_value == nullptr ? nullptr :
                                               _as_lane_vector(
                                                   _load_value(instruction.operands[mask_operand]),
                                                   *mask_value);
    auto *expected_query_type = Type::custom(
        query_all ? "LC_RayQueryAll" : "LC_RayQueryAny");
    if (result == nullptr || result->type != expected_query_type ||
        result->value_class != schedule::ValueClass::varying ||
        accel_value == nullptr || accel_value->type == nullptr ||
        !accel_value->type->is_accel() || accel == nullptr ||
        ray_value == nullptr || !is_ray_type(ray_value->type) ||
        ray == nullptr || mask_value == nullptr ||
        mask_value->type == nullptr || !mask_value->type->is_uint32() ||
        visibility == nullptr ||
        (motion && (time_value == nullptr ||
                    time_value->type == nullptr ||
                    !time_value->type->is_float32() || time == nullptr))) {
        _fail("ray-query construction has invalid operands or result");
        return nullptr;
    }

    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *pointer_lanes = ::llvm::FixedVectorType::get(
        pointer_type, _width);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *float_lanes = ::llvm::FixedVectorType::get(
        _builder.getFloatTy(), _width);
    auto *storage_type = ::llvm::ArrayType::get(
        _builder.getInt8Ty(),
        static_cast<uint64_t>(_width) * sizeof(SIMDHostRayQueryState));
    auto *storage = _entry_scratch(
        storage_type,
        "ray.query.state." + std::to_string(instruction.result->value));
    storage->setAlignment(::llvm::Align{alignof(SIMDHostRayQueryState)});
    auto *states = _builder.CreateGEP(
        _builder.getInt8Ty(),
        _builder.CreateVectorSplat(_width, storage),
        _lane_offsets(_lane_ids(), sizeof(SIMDHostRayQueryState)),
        "ray.query.states");
    auto field_pointers = [&](size_t offset) noexcept {
        return offset == 0u ? states :
                              _builder.CreateGEP(
                                  _builder.getInt8Ty(), states,
                                  _builder.CreateVectorSplat(
                                      _width, _builder.getInt64(offset)));
    };
    auto scatter = [&](::llvm::Value *value, size_t offset,
                       size_t alignment) noexcept {
        if (!value->getType()->isVectorTy()) {
            value = _builder.CreateVectorSplat(_width, value);
        }
        _builder.CreateMaskedScatter(
            value, field_pointers(offset),
            ::llvm::Align{alignment}, _active_mask);
    };

    auto *object = _builder.CreateExtractValue(accel, {0u});
    auto *proceed = _builder.CreateExtractValue(accel, {4u});
    auto *null_pointer = ::llvm::ConstantPointerNull::get(pointer_type);
    auto *missing_callback = _builder.CreateOr(
        _builder.CreateICmpEQ(object, null_pointer),
        _builder.CreateICmpEQ(proceed, null_pointer));
    _trap_if(
        _builder.CreateAnd(
            _builder.CreateOrReduce(_active_mask), missing_callback),
        "accel.ray.query.callback.null");
    scatter(
        object, offsetof(SIMDHostRayQueryState, accel), alignof(void *));
    scatter(
        proceed, offsetof(SIMDHostRayQueryState, proceed), alignof(void *));

    auto *zero_offsets = ::llvm::Constant::getNullValue(
        ::llvm::FixedVectorType::get(_builder.getInt64Ty(), _width));
    _scatter_data(
        states, zero_offsets, ray_value->type, ray,
        offsetof(SIMDHostRayQueryState, world_ray));
    auto *zero_float = ::llvm::Constant::getNullValue(float_lanes);
    auto *safe_time = motion ?
                          _builder.CreateSelect(
                              _active_mask, time, zero_float,
                              "ray.query.safe.time") :
                          zero_float;
    scatter(
        safe_time, offsetof(SIMDHostRayQueryState, time), alignof(float));
    auto *zero_i32 = ::llvm::Constant::getNullValue(i32_lanes);
    visibility = _builder.CreateZExtOrTrunc(visibility, i32_lanes);
    visibility = _builder.CreateSelect(
        _active_mask, visibility, zero_i32,
        "ray.query.safe.visibility");
    scatter(
        visibility, offsetof(SIMDHostRayQueryState, visibility_mask),
        alignof(uint32_t));
    scatter(
        _builder.getInt32(query_any ? 1u : 0u),
        offsetof(SIMDHostRayQueryState, terminate_on_first),
        alignof(uint32_t));
    scatter(
        zero_i32, offsetof(SIMDHostRayQueryState, cursor_valid),
        alignof(uint32_t));
    scatter(
        zero_i32, offsetof(SIMDHostRayQueryState, candidate_kind),
        alignof(uint32_t));
    scatter(
        zero_i32, offsetof(SIMDHostRayQueryState, candidate_committed),
        alignof(uint32_t));
    scatter(
        zero_i32, offsetof(SIMDHostRayQueryState, terminated),
        alignof(uint32_t));

    auto *invalid_i32 = ::llvm::Constant::getAllOnesValue(i32_lanes);
    scatter(
        invalid_i32,
        offsetof(SIMDHostRayQueryState, committed) +
            offsetof(SIMDHostRayQueryCommittedHit, inst),
        alignof(uint32_t));
    scatter(
        invalid_i32,
        offsetof(SIMDHostRayQueryState, committed) +
            offsetof(SIMDHostRayQueryCommittedHit, prim),
        alignof(uint32_t));
    scatter(
        zero_float,
        offsetof(SIMDHostRayQueryState, committed) +
            offsetof(SIMDHostRayQueryCommittedHit, bary),
        alignof(float));
    scatter(
        zero_float,
        offsetof(SIMDHostRayQueryState, committed) +
            offsetof(SIMDHostRayQueryCommittedHit, bary) + sizeof(float),
        alignof(float));
    scatter(
        zero_i32,
        offsetof(SIMDHostRayQueryState, committed) +
            offsetof(SIMDHostRayQueryCommittedHit, kind),
        alignof(uint32_t));
    scatter(
        zero_float,
        offsetof(SIMDHostRayQueryState, committed) +
            offsetof(SIMDHostRayQueryCommittedHit, t),
        alignof(float));
    return _builder.CreateBitCast(states, pointer_lanes);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_ray_query_state_handles(
    schedule::ValueId object_id) {
    auto *object = _source.value(object_id);
    auto *local = _load_value(object_id);
    if (object == nullptr || !is_ray_query_type(object->type) ||
        !_is_local_lvalue(object_id) || local == nullptr) {
        _fail("ray-query object is not a valid thread-local lvalue");
        return nullptr;
    }
    auto *states = _gather_data(
        _local_base(_builder, local),
        _local_offsets(_builder, local), object->type);
    if (states == nullptr) { return nullptr; }
    auto *null_states = ::llvm::Constant::getNullValue(states->getType());
    auto *invalid = _builder.CreateAnd(
        _active_mask,
        _builder.CreateICmpEQ(states, null_states));
    _trap_if(
        _builder.CreateOrReduce(invalid),
        "ray.query.active.state.null");
    return states;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_ray_query_read(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op ||
        instruction.operands.size() != 1u) {
        _fail("ray-query object read is malformed");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    auto *states = _ray_query_state_handles(instruction.operands[0u]);
    if (result == nullptr || result->type == nullptr ||
        result->value_class != schedule::ValueClass::varying ||
        states == nullptr) {
        _fail("ray-query object read has an invalid result");
        return nullptr;
    }
    auto *zero_offsets = ::llvm::Constant::getNullValue(
        ::llvm::FixedVectorType::get(_builder.getInt64Ty(), _width));
    auto gather = [&](const Type *type, size_t offset) noexcept {
        return _gather_data(states, zero_offsets, type, offset);
    };
    auto gather_i32 = [&](size_t offset) noexcept {
        auto *pointers = _builder.CreateGEP(
            _builder.getInt8Ty(), states,
            _builder.CreateVectorSplat(
                _width, _builder.getInt64(offset)));
        auto *lanes = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), _width);
        return _builder.CreateMaskedGather(
            lanes, pointers, ::llvm::Align{alignof(uint32_t)},
            _active_mask, ::llvm::Constant::getNullValue(lanes));
    };

    auto op = static_cast<xir::RayQueryObjectReadOp>(
        *instruction.source_op);
    switch (op) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
            if (!is_ray_type(result->type)) { break; }
            return gather(
                result->type,
                offsetof(SIMDHostRayQueryState, world_ray));
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT:
            if (!is_procedural_hit_type(result->type)) { break; }
            return gather(
                result->type,
                offsetof(SIMDHostRayQueryState, candidate));
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT:
            if (!is_surface_hit_type(result->type)) { break; }
            return gather(
                result->type,
                offsetof(SIMDHostRayQueryState, candidate));
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT:
            if (!is_committed_hit_type(result->type)) { break; }
            return gather(
                result->type,
                offsetof(SIMDHostRayQueryState, committed));
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE:
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE: {
            if (!result->type->is_bool()) { break; }
            auto kind =
                op == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE ?
                    SIMDHostRayQueryCandidateKind::surface :
                    SIMDHostRayQueryCandidateKind::procedural;
            auto *kinds = gather_i32(
                offsetof(SIMDHostRayQueryState, candidate_kind));
            return _builder.CreateICmpEQ(
                kinds,
                _builder.CreateVectorSplat(
                    _width, _builder.getInt32(
                                static_cast<uint32_t>(kind))));
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED: {
            if (!result->type->is_bool()) { break; }
            auto *terminated = gather_i32(
                offsetof(SIMDHostRayQueryState, terminated));
            return _builder.CreateICmpNE(
                terminated,
                ::llvm::Constant::getNullValue(terminated->getType()));
        }
    }
    _fail("ray-query object read has an unsupported operation or type");
    return nullptr;
}

void ScheduleEmitter::_ray_query_write(
    const schedule::Instruction &instruction) {
    if (!instruction.source_op || instruction.operands.empty()) {
        _fail("ray-query object write is malformed");
        return;
    }
    auto op = static_cast<xir::RayQueryObjectWriteOp>(
        *instruction.source_op);
    auto expected_operands =
        op == xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL ?
            2u :
            1u;
    if (instruction.operands.size() != expected_operands) {
        _fail("ray-query object write has an invalid operand count");
        return;
    }
    auto *states = _ray_query_state_handles(instruction.operands[0u]);
    if (states == nullptr) { return; }
    auto field_pointers = [&](size_t offset) noexcept {
        return offset == 0u ? states :
                              _builder.CreateGEP(
                                  _builder.getInt8Ty(), states,
                                  _builder.CreateVectorSplat(
                                      _width, _builder.getInt64(offset)));
    };
    auto gather_i32 = [&](size_t offset) noexcept {
        auto *lanes = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), _width);
        return _builder.CreateMaskedGather(
            lanes, field_pointers(offset),
            ::llvm::Align{alignof(uint32_t)}, _active_mask,
            ::llvm::Constant::getNullValue(lanes));
    };
    auto gather_float = [&](size_t offset) noexcept {
        auto *lanes = ::llvm::FixedVectorType::get(
            _builder.getFloatTy(), _width);
        return _builder.CreateMaskedGather(
            lanes, field_pointers(offset),
            ::llvm::Align{alignof(float)}, _active_mask,
            ::llvm::Constant::getNullValue(lanes));
    };
    auto scatter = [&](::llvm::Value *value, size_t offset,
                       size_t alignment, ::llvm::Value *mask) noexcept {
        if (!value->getType()->isVectorTy()) {
            value = _builder.CreateVectorSplat(_width, value);
        }
        _builder.CreateMaskedScatter(
            value, field_pointers(offset),
            ::llvm::Align{alignment}, mask);
    };

    switch (op) {
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED: {
            auto &context = _module.getContext();
            auto *pointer_type = ::llvm::PointerType::getUnqual(context);
            auto *pointer_lanes = ::llvm::FixedVectorType::get(
                pointer_type, _width);
            auto *callbacks = _builder.CreateMaskedGather(
                pointer_lanes,
                field_pointers(
                    offsetof(SIMDHostRayQueryState, proceed)),
                ::llvm::Align{alignof(void *)}, _active_mask,
                ::llvm::Constant::getNullValue(pointer_lanes));
            auto *callback = _builder.CreateExtractElement(
                callbacks, _safe_first_lane(_active_mask));
            auto *null_pointer =
                ::llvm::ConstantPointerNull::get(pointer_type);
            _trap_if(
                _builder.CreateICmpEQ(callback, null_pointer),
                "ray.query.proceed.callback.null");
            auto *callback_mismatch = _builder.CreateAnd(
                _active_mask,
                _builder.CreateICmpNE(
                    callbacks,
                    _builder.CreateVectorSplat(_width, callback)));
            _trap_if(
                _builder.CreateOrReduce(callback_mismatch),
                "ray.query.proceed.callback.mismatch");

            auto *scratch = _entry_scratch(
                pointer_lanes,
                "ray.query.packet." +
                    std::to_string(instruction.operands[0u].value));
            scratch->setAlignment(::llvm::Align{alignof(void *)});
            auto *safe_states = _builder.CreateSelect(
                _active_mask, states,
                ::llvm::Constant::getNullValue(states->getType()));
            auto *store = _builder.CreateStore(safe_states, scratch);
            store->setAlignment(::llvm::Align{alignof(void *)});
            auto *callback_type = ::llvm::FunctionType::get(
                _builder.getVoidTy(),
                {_builder.getInt32Ty(), _builder.getInt64Ty(),
                 pointer_type},
                false);
            _builder.CreateCall(
                callback_type, callback,
                {_builder.getInt32(_width),
                 _bindless_callback_mask(true), scratch});
            return;
        }
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE: {
            auto *kinds = gather_i32(
                offsetof(SIMDHostRayQueryState, candidate_kind));
            auto *surface = _builder.CreateICmpEQ(
                kinds,
                _builder.CreateVectorSplat(
                    _width,
                    _builder.getInt32(static_cast<uint32_t>(
                        SIMDHostRayQueryCandidateKind::surface))));
            auto *invalid = _builder.CreateAnd(
                _active_mask, _builder.CreateNot(surface));
            _trap_if(
                _builder.CreateOrReduce(invalid),
                "ray.query.commit.triangle.without.candidate");
            auto *commit_mask = _builder.CreateAnd(
                _active_mask, surface);
            scatter(
                _builder.getInt32(1u),
                offsetof(SIMDHostRayQueryState, candidate_committed),
                alignof(uint32_t), commit_mask);
            auto *candidate_t = gather_float(
                offsetof(SIMDHostRayQueryState, candidate) +
                offsetof(SIMDHostRayQuerySurfaceHit, t));
            scatter(
                candidate_t,
                offsetof(SIMDHostRayQueryState, world_ray) +
                    7u * sizeof(float),
                alignof(float), commit_mask);
            return;
        }
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL: {
            auto *distance_value = _source.value(instruction.operands[1u]);
            auto *distance = distance_value == nullptr ? nullptr :
                                                         _as_lane_vector(
                                                             _load_value(instruction.operands[1u]),
                                                             *distance_value);
            if (distance_value == nullptr ||
                distance_value->type == nullptr ||
                !distance_value->type->is_float32() || distance == nullptr) {
                _fail("procedural ray-query commit requires a float distance");
                return;
            }
            auto *kinds = gather_i32(
                offsetof(SIMDHostRayQueryState, candidate_kind));
            auto *procedural = _builder.CreateICmpEQ(
                kinds,
                _builder.CreateVectorSplat(
                    _width,
                    _builder.getInt32(static_cast<uint32_t>(
                        SIMDHostRayQueryCandidateKind::procedural))));
            auto *invalid_kind = _builder.CreateAnd(
                _active_mask, _builder.CreateNot(procedural));
            _trap_if(
                _builder.CreateOrReduce(invalid_kind),
                "ray.query.commit.procedural.without.candidate");
            auto *candidate_mask = _builder.CreateAnd(
                _active_mask, procedural);
            scatter(
                distance,
                offsetof(SIMDHostRayQueryState, candidate) +
                    offsetof(SIMDHostRayQuerySurfaceHit, t),
                alignof(float), candidate_mask);
            auto *t_min = gather_float(
                offsetof(SIMDHostRayQueryState, world_ray) +
                3u * sizeof(float));
            auto *t_max = gather_float(
                offsetof(SIMDHostRayQueryState, world_ray) +
                7u * sizeof(float));
            auto *in_range = _builder.CreateAnd(
                _builder.CreateFCmpOGE(distance, t_min),
                _builder.CreateFCmpOLE(distance, t_max));
            auto *commit_mask = _builder.CreateAnd(
                candidate_mask, in_range);
            scatter(
                _builder.getInt32(1u),
                offsetof(SIMDHostRayQueryState, candidate_committed),
                alignof(uint32_t), commit_mask);
            scatter(
                distance,
                offsetof(SIMDHostRayQueryState, world_ray) +
                    7u * sizeof(float),
                alignof(float), commit_mask);
            return;
        }
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE:
            scatter(
                _builder.getInt32(1u),
                offsetof(SIMDHostRayQueryState, terminated),
                alignof(uint32_t), _active_mask);
            return;
    }
    _fail("unsupported ray-query object write operation");
}

}// namespace luisa::compute::simd::detail
