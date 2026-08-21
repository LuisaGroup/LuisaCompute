#include "llvm_schedule_emitter.h"

#include <limits>

#include "../../common/env_flag.h"

namespace luisa::compute::simd::detail {

namespace {

[[nodiscard]] bool is_ray_query_type(const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           (type == Type::custom("LC_RayQueryAll") ||
            type == Type::custom("LC_RayQueryAny"));
}

[[nodiscard]] bool is_ray_query_construction(
    const schedule::Instruction &instruction) noexcept {
    if (instruction.opcode != schedule::Opcode::resource_query ||
        !instruction.result || !instruction.source_op) {
        return false;
    }
    auto op = static_cast<xir::ResourceQueryOp>(*instruction.source_op);
    return op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL ||
           op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY ||
           op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR ||
           op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR;
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
    constexpr auto invalid = std::numeric_limits<uint32_t>::max();
    auto result_id = instruction.result->value;
    auto scratch_slot =
        result_id < _ray_query_scratch_slots.size() ?
            _ray_query_scratch_slots[result_id] :
            invalid;
    if (scratch_slot == invalid ||
        scratch_slot >= _ray_query_scratch_storage.size()) {
        _fail("ray-query construction has no analyzed scratch slot");
        return nullptr;
    }
    auto *&storage = _ray_query_scratch_storage[scratch_slot];
    if (storage == nullptr) {
        auto *storage_type = ::llvm::ArrayType::get(
            _builder.getInt8Ty(),
            static_cast<uint64_t>(_width) *
                sizeof(SIMDHostRayQueryState));
        storage = _entry_scratch(
            storage_type,
            "ray.query.state.slot." +
                std::to_string(scratch_slot));
        storage->setAlignment(
            ::llvm::Align{alignof(SIMDHostRayQueryState)});
    }
    auto *full_states = _builder.CreateGEP(
        _builder.getInt8Ty(),
        _builder.CreateVectorSplat(_width, storage),
        _lane_offsets(_lane_ids(), sizeof(SIMDHostRayQueryState)),
        "ray.query.full.states");
    auto *compact_states = _builder.CreateGEP(
        _builder.getInt8Ty(),
        _builder.CreateVectorSplat(_width, storage),
        _lane_offsets(
            _lane_ids(), simd_host_ray_query_hot_state_stride),
        "ray.query.compact.states");

    auto *object = _builder.CreateExtractValue(accel, {0u});
    auto *plain_proceed = _builder.CreateExtractValue(
        accel, {_width >= 8u ? 5u : 4u});
    auto *candidate_object_ray = _builder.CreateExtractValue(accel, {6u});
    auto cache_status =
        _ray_query_status_slot(*instruction.result) != nullptr;
    auto *null_pointer = ::llvm::ConstantPointerNull::get(pointer_type);
    auto *status_proceed = static_cast<::llvm::Value *>(nullptr);
    auto *pipeline_w1 = static_cast<::llvm::Value *>(nullptr);
    auto *surface_filter_pipeline =
        static_cast<::llvm::Value *>(nullptr);
    auto *empty_surface_filter_pipeline =
        static_cast<::llvm::Value *>(nullptr);
    auto *direct_output_surface_filter_pipeline =
        static_cast<::llvm::Value *>(nullptr);
    auto status_index = cache_status ?
                            _ray_query_status_slots[instruction.result->value] :
                            std::numeric_limits<uint32_t>::max();
    auto cache_pipeline_w1 =
        _width == 1u && cache_status &&
        status_index <
            _ray_query_pipeline_callback_storage.size();
    auto output_only_eligible =
        result_id <
            _ray_query_output_only_empty_surface_filter_state.size() &&
        _ray_query_output_only_empty_surface_filter_state[result_id] != 0u;
    auto direct_output_eligible =
        result_id <
            _ray_query_direct_output_surface_filter_state.size() &&
        _ray_query_direct_output_surface_filter_state[result_id] != 0u;
    if (cache_status) {
        auto *instances = _builder.CreateExtractValue(accel, {3u});
        _trap_if(
            _builder.CreateICmpEQ(instances, null_pointer),
            "accel.ray.query.status.table.null");
        auto status_offset = _width >= 8u ?
                                 offsetof(
                                     SIMDHostAccelInstanceTable,
                                     ray_query_proceed_wide_status) :
                                 offsetof(
                                     SIMDHostAccelInstanceTable,
                                     ray_query_proceed_status);
        auto *proceed_pointer = _byte_pointer(instances, status_offset);
        auto *proceed_load = _builder.CreateLoad(
            pointer_type, proceed_pointer,
            "accel.ray.query.status.callback");
        proceed_load->setAlignment(::llvm::Align{alignof(void *)});
        status_proceed = proceed_load;
        if (cache_pipeline_w1) {
            auto *pipeline_pointer = _byte_pointer(
                instances,
                offsetof(
                    SIMDHostAccelInstanceTable,
                    ray_query_pipeline_w1));
            auto *pipeline_load = _builder.CreateLoad(
                pointer_type, pipeline_pointer,
                "accel.ray.query.pipeline.w1.callback");
            pipeline_load->setAlignment(
                ::llvm::Align{alignof(void *)});
            pipeline_w1 = pipeline_load;
        }
        if (status_index <
            _ray_query_surface_filter_pipeline_callback_storage.size()) {
            auto *pipeline_pointer = _byte_pointer(
                instances,
                offsetof(
                    SIMDHostAccelInstanceTable,
                    ray_query_surface_filter_pipeline));
            auto *pipeline_load = _builder.CreateLoad(
                pointer_type, pipeline_pointer,
                "accel.ray.query.surface.filter.pipeline.callback");
            pipeline_load->setAlignment(
                ::llvm::Align{alignof(void *)});
            surface_filter_pipeline = pipeline_load;
        }
        if (output_only_eligible &&
            status_index <
                _ray_query_empty_surface_filter_pipeline_callback_storage
                    .size()) {
            auto *pipeline_pointer = _byte_pointer(
                instances,
                offsetof(
                    SIMDHostAccelInstanceTable,
                    ray_query_empty_surface_filter_packet_pipeline));
            auto *pipeline_load = _builder.CreateLoad(
                pointer_type, pipeline_pointer,
                "accel.ray.query.empty.surface.filter.pipeline.callback");
            pipeline_load->setAlignment(
                ::llvm::Align{alignof(void *)});
            empty_surface_filter_pipeline = pipeline_load;
        }
        if (direct_output_eligible &&
            status_index <
                _ray_query_direct_output_surface_filter_pipeline_callback_storage
                    .size()) {
            auto *pipeline_pointer = _byte_pointer(
                instances,
                offsetof(
                    SIMDHostAccelInstanceTable,
                    ray_query_direct_output_surface_filter_packet_pipeline));
            auto *pipeline_load = _builder.CreateLoad(
                pointer_type, pipeline_pointer,
                "accel.ray.query.direct.output.surface.filter.pipeline.callback");
            pipeline_load->setAlignment(
                ::llvm::Align{alignof(void *)});
            direct_output_surface_filter_pipeline = pipeline_load;
        }
    }
    auto *missing_callback = _builder.CreateOr(
        _builder.CreateICmpEQ(object, null_pointer),
        _builder.CreateICmpEQ(plain_proceed, null_pointer));
    if (cache_status) {
        missing_callback = _builder.CreateOr(
            missing_callback,
            _builder.CreateICmpEQ(status_proceed, null_pointer));
    }
    if (cache_pipeline_w1) {
        missing_callback = _builder.CreateOr(
            missing_callback,
            _builder.CreateICmpEQ(pipeline_w1, null_pointer));
    }
    _trap_if(
        _builder.CreateAnd(
            _builder.CreateOrReduce(_active_mask), missing_callback),
        "accel.ray.query.callback.null");
    auto compact_eligible =
        result_id < _ray_query_compact_surface_filter_state.size() &&
        _ray_query_compact_surface_filter_state[result_id] != 0u;
    auto *use_output_only_state = static_cast<::llvm::Value *>(
        _builder.getFalse());
    if (output_only_eligible) {
        if (empty_surface_filter_pipeline == nullptr) {
            _fail("output-only ray-query state lost its packet provider");
            return nullptr;
        }
        use_output_only_state = _builder.CreateICmpNE(
            empty_surface_filter_pipeline, null_pointer,
            "ray.query.output.only.state");
    }
    auto *use_direct_output_state = static_cast<::llvm::Value *>(
        _builder.getFalse());
    if (direct_output_eligible) {
        if (direct_output_surface_filter_pipeline == nullptr) {
            _fail("direct-output ray-query state lost its packet provider");
            return nullptr;
        }
        use_direct_output_state = _builder.CreateICmpNE(
            direct_output_surface_filter_pipeline, null_pointer,
            "ray.query.direct.output.state");
    }
    auto *use_minimal_output_state = _builder.CreateOr(
        use_output_only_state, use_direct_output_state,
        "ray.query.minimal.output.state");
    auto *use_compact_state = static_cast<::llvm::Value *>(
        _builder.getFalse());
    if (compact_eligible) {
        if (surface_filter_pipeline == nullptr) {
            _fail("compact ray-query state lost its surface-filter provider");
            return nullptr;
        }
        auto *runtime_flags = _load_launch_u32(offsetof(
            SIMDPacketLaunchConfig, reserved_runtime_flags));
        auto *enabled = _builder.CreateICmpNE(
            _builder.CreateAnd(
                runtime_flags,
                _builder.getInt32(
                    simd_packet_launch_flag_compact_surface_filter_state)),
            _builder.getInt32(0u),
            "ray.query.compact.state.enabled");
        use_compact_state = _builder.CreateAnd(
            enabled,
            _builder.CreateICmpNE(
                surface_filter_pipeline, null_pointer),
            "ray.query.compact.state");
    }
    auto *states = compact_eligible ?
                       _builder.CreateSelect(
                           use_compact_state, compact_states, full_states,
                           "ray.query.states") :
                       full_states;
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
    if (cache_status) {
        auto status_slot = status_index;
        auto *old_callbacks = _builder.CreateAlignedLoad(
            pointer_lanes,
            _ray_query_status_callback_storage[status_slot],
            ::llvm::Align{alignof(void *)});
        _builder.CreateAlignedStore(
            _builder.CreateSelect(
                _active_mask,
                _builder.CreateVectorSplat(_width, status_proceed),
                old_callbacks),
            _ray_query_status_callback_storage[status_slot],
            ::llvm::Align{alignof(void *)});
        if (cache_pipeline_w1) {
            auto *old_pipelines = _builder.CreateAlignedLoad(
                pointer_lanes,
                _ray_query_pipeline_callback_storage[status_slot],
                ::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                _builder.CreateSelect(
                    _active_mask,
                    _builder.CreateVectorSplat(
                        _width, pipeline_w1),
                    old_pipelines),
                _ray_query_pipeline_callback_storage[status_slot],
                ::llvm::Align{alignof(void *)});
        }
        if (status_slot <
            _ray_query_surface_filter_pipeline_callback_storage.size()) {
            auto *old_pipelines = _builder.CreateAlignedLoad(
                pointer_lanes,
                _ray_query_surface_filter_pipeline_callback_storage[status_slot],
                ::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                _builder.CreateSelect(
                    _active_mask,
                    _builder.CreateVectorSplat(
                        _width, surface_filter_pipeline),
                    old_pipelines),
                _ray_query_surface_filter_pipeline_callback_storage[status_slot],
                ::llvm::Align{alignof(void *)});
        }
        if (status_slot <
            _ray_query_empty_surface_filter_pipeline_callback_storage.size()) {
            auto *old_pipelines = _builder.CreateAlignedLoad(
                pointer_lanes,
                _ray_query_empty_surface_filter_pipeline_callback_storage
                    [status_slot],
                ::llvm::Align{alignof(void *)});
            auto *pipeline = empty_surface_filter_pipeline == nullptr ?
                                 static_cast<::llvm::Value *>(null_pointer) :
                                 empty_surface_filter_pipeline;
            _builder.CreateAlignedStore(
                _builder.CreateSelect(
                    _active_mask,
                    _builder.CreateVectorSplat(_width, pipeline),
                    old_pipelines),
                _ray_query_empty_surface_filter_pipeline_callback_storage
                    [status_slot],
                ::llvm::Align{alignof(void *)});
        }
        if (status_slot <
            _ray_query_empty_surface_filter_accel_storage.size()) {
            auto *old_accels = _builder.CreateAlignedLoad(
                pointer_lanes,
                _ray_query_empty_surface_filter_accel_storage[status_slot],
                ::llvm::Align{alignof(void *)});
            auto *empty_accel = output_only_eligible ? object : null_pointer;
            _builder.CreateAlignedStore(
                _builder.CreateSelect(
                    _active_mask,
                    _builder.CreateVectorSplat(_width, empty_accel),
                    old_accels),
                _ray_query_empty_surface_filter_accel_storage[status_slot],
                ::llvm::Align{alignof(void *)});
        }
        if (status_slot <
            _ray_query_direct_output_surface_filter_pipeline_callback_storage
                .size()) {
            auto *old_pipelines = _builder.CreateAlignedLoad(
                pointer_lanes,
                _ray_query_direct_output_surface_filter_pipeline_callback_storage
                    [status_slot],
                ::llvm::Align{alignof(void *)});
            auto *pipeline =
                direct_output_surface_filter_pipeline == nullptr ?
                    static_cast<::llvm::Value *>(null_pointer) :
                    direct_output_surface_filter_pipeline;
            _builder.CreateAlignedStore(
                _builder.CreateSelect(
                    _active_mask,
                    _builder.CreateVectorSplat(_width, pipeline),
                    old_pipelines),
                _ray_query_direct_output_surface_filter_pipeline_callback_storage
                    [status_slot],
                ::llvm::Align{alignof(void *)});
        }
        if (status_slot <
            _ray_query_direct_output_surface_filter_accel_storage.size()) {
            auto *old_accels = _builder.CreateAlignedLoad(
                pointer_lanes,
                _ray_query_direct_output_surface_filter_accel_storage
                    [status_slot],
                ::llvm::Align{alignof(void *)});
            auto *direct_accel = direct_output_eligible ?
                                     object :
                                     null_pointer;
            _builder.CreateAlignedStore(
                _builder.CreateSelect(
                    _active_mask,
                    _builder.CreateVectorSplat(_width, direct_accel),
                    old_accels),
                _ray_query_direct_output_surface_filter_accel_storage
                    [status_slot],
                ::llvm::Align{alignof(void *)});
        }
    }

    auto *zero_offsets = ::llvm::Constant::getNullValue(
        ::llvm::FixedVectorType::get(_builder.getInt64Ty(), _width));
    auto *zero_float = ::llvm::Constant::getNullValue(float_lanes);
    auto *safe_time = motion ?
                          _builder.CreateSelect(
                              _active_mask, time, zero_float,
                              "ray.query.safe.time") :
                          zero_float;
    auto *zero_i32 = ::llvm::Constant::getNullValue(i32_lanes);
    visibility = _builder.CreateZExtOrTrunc(visibility, i32_lanes);
    visibility = _builder.CreateSelect(
        _active_mask, visibility, zero_i32,
        "ray.query.safe.visibility");
    if (!_store_ray_query_surface_filter_ray_packet(
            *ray_value, ray, safe_time, visibility, status_index)) {
        return nullptr;
    }
    auto packed_init =
        _width >= 4u &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_RAY_QUERY_PACKED_INIT");
    auto *i64_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt64Ty(), _width);
    auto *zero_i64 = ::llvm::Constant::getNullValue(i64_lanes);
    auto eager_batch_init =
        _width == 2u ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_RAY_QUERY_LAZY_BATCH_INIT");
    // The initialized fields gate every runtime read of the six metadata
    // fields below. The first scan clears them before publishing both gates,
    // so eager construction stores are redundant at accepted widths.
    if (packed_init) {
        auto *invalid_i64 =
            ::llvm::Constant::getAllOnesValue(i64_lanes);
        scatter(
            invalid_i64,
            offsetof(SIMDHostRayQueryState, committed),
            alignof(uint64_t));
        scatter(
            zero_i64,
            offsetof(SIMDHostRayQueryState, committed) +
                offsetof(SIMDHostRayQueryCommittedHit, bary),
            alignof(uint64_t));
        scatter(
            zero_i64,
            offsetof(SIMDHostRayQueryState, committed) +
                offsetof(SIMDHostRayQueryCommittedHit, kind),
            alignof(uint64_t));
    } else {
        auto *invalid_i32 =
            ::llvm::Constant::getAllOnesValue(i32_lanes);
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
                offsetof(SIMDHostRayQueryCommittedHit, bary) +
                sizeof(float),
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
    }
    auto initialize_full_state = [&]() noexcept {
        scatter(
            candidate_object_ray,
            offsetof(SIMDHostRayQueryState, candidate_object_ray),
            alignof(void *));
        if (eager_batch_init) {
            scatter(
                zero_i32,
                offsetof(SIMDHostRayQueryState, candidate_batch_count),
                alignof(uint32_t));
            scatter(
                zero_i32,
                offsetof(SIMDHostRayQueryState, candidate_batch_index),
                alignof(uint32_t));
            scatter(
                zero_i32,
                offsetof(SIMDHostRayQueryState, candidate_batch_has_more),
                alignof(uint32_t));
            scatter(
                zero_i32,
                offsetof(SIMDHostRayQueryState, procedural_batch_count),
                alignof(uint32_t));
            scatter(
                zero_i32,
                offsetof(SIMDHostRayQueryState, procedural_batch_index),
                alignof(uint32_t));
            scatter(
                zero_i32,
                offsetof(SIMDHostRayQueryState, procedural_batch_has_more),
                alignof(uint32_t));
        }
        scatter(
            zero_i32,
            offsetof(
                SIMDHostRayQueryState, candidate_batch_initialized),
            alignof(uint32_t));
        scatter(
            zero_i32,
            offsetof(
                SIMDHostRayQueryState, procedural_batch_initialized),
            alignof(uint32_t));
    };
    auto initialize_operational_state = [&]() noexcept {
        scatter(
            object, offsetof(SIMDHostRayQueryState, accel),
            alignof(void *));
        scatter(
            plain_proceed, offsetof(SIMDHostRayQueryState, proceed),
            alignof(void *));
        _scatter_data(
            states, zero_offsets, ray_value->type, ray,
            offsetof(SIMDHostRayQueryState, world_ray));
        scatter(
            safe_time, offsetof(SIMDHostRayQueryState, time),
            alignof(float));
        scatter(
            visibility,
            offsetof(SIMDHostRayQueryState, visibility_mask),
            alignof(uint32_t));
        scatter(
            _builder.getInt32(query_any ? 1u : 0u),
            offsetof(SIMDHostRayQueryState, terminate_on_first),
            alignof(uint32_t));
        scatter(
            zero_i32, offsetof(SIMDHostRayQueryState, cursor_valid),
            alignof(uint32_t));
        if (packed_init) {
            scatter(
                zero_i64,
                offsetof(SIMDHostRayQueryState, candidate_kind),
                alignof(uint32_t));
            scatter(
                zero_i64,
                offsetof(SIMDHostRayQueryState, terminated),
                alignof(uint32_t));
        } else {
            scatter(
                zero_i32,
                offsetof(SIMDHostRayQueryState, candidate_kind),
                alignof(uint32_t));
            scatter(
                zero_i32,
                offsetof(SIMDHostRayQueryState, candidate_committed),
                alignof(uint32_t));
            scatter(
                zero_i32,
                offsetof(SIMDHostRayQueryState, terminated),
                alignof(uint32_t));
            scatter(
                zero_i32,
                offsetof(
                    SIMDHostRayQueryState,
                    procedural_cursor_valid),
                alignof(uint32_t));
        }
        if (compact_eligible) {
            // The complete-state tail is needed only by the ordinary provider.
            // Retain it behind the existing scalar compact-state guard.
            auto *full_state_init = ::llvm::BasicBlock::Create(
                _module.getContext(), "ray.query.full.state.init", _entry);
            auto *compact_state_ready = ::llvm::BasicBlock::Create(
                _module.getContext(), "ray.query.compact.state.ready", _entry);
            _builder.CreateCondBr(
                use_compact_state, compact_state_ready, full_state_init);
            _builder.SetInsertPoint(full_state_init);
            initialize_full_state();
            _builder.CreateBr(compact_state_ready);
            _builder.SetInsertPoint(compact_state_ready);
        } else {
            initialize_full_state();
        }
    };
    auto initialize_output_only_state = [&]() noexcept {
        auto *t_min = _extract_child(ray, ray_value->type, 1u, true);
        auto *t_max = _extract_child(ray, ray_value->type, 3u, true);
        scatter(
            t_min,
            offsetof(SIMDHostRayQueryState, world_ray) +
                3u * sizeof(float),
            alignof(float));
        scatter(
            t_max,
            offsetof(SIMDHostRayQueryState, world_ray) +
                7u * sizeof(float),
            alignof(float));
    };
    if (output_only_eligible || direct_output_eligible) {
        auto *output_only_init = ::llvm::BasicBlock::Create(
            _module.getContext(), "ray.query.output.only.state.init", _entry);
        auto *operational_init = ::llvm::BasicBlock::Create(
            _module.getContext(), "ray.query.operational.state.init", _entry);
        auto *state_ready = ::llvm::BasicBlock::Create(
            _module.getContext(), "ray.query.state.ready", _entry);
        _builder.CreateCondBr(
            use_minimal_output_state, output_only_init, operational_init);
        _builder.SetInsertPoint(output_only_init);
        initialize_output_only_state();
        _builder.CreateBr(state_ready);
        _builder.SetInsertPoint(operational_init);
        initialize_operational_state();
        _builder.CreateBr(state_ready);
        _builder.SetInsertPoint(state_ready);
    } else {
        initialize_operational_state();
    }
    return _builder.CreateBitCast(states, pointer_lanes);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_ray_query_state_handles(
    schedule::ValueId object_id) {
    auto *object = _source.value(object_id);
    auto *handle_slot = _ray_query_state_handle_slot(object_id);
    auto *local = handle_slot == nullptr ? _load_value(object_id) : nullptr;
    if (object == nullptr || !is_ray_query_type(object->type) ||
        !_is_local_lvalue(object_id) ||
        (handle_slot == nullptr && local == nullptr)) {
        _fail("ray-query object is not a valid thread-local lvalue");
        return nullptr;
    }
    auto *states = handle_slot == nullptr ?
                       _gather_data(
                           _local_base(_builder, local),
                           _local_offsets(_builder, local), object->type) :
                       _builder.CreateAlignedLoad(
                           ::llvm::FixedVectorType::get(
                               ::llvm::PointerType::getUnqual(
                                   _module.getContext()),
                               _width),
                           handle_slot, ::llvm::Align{alignof(void *)},
                           "ray.query.cached.state.handles");
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

[[nodiscard]] ::llvm::AllocaInst *ScheduleEmitter::_ray_query_state_handle_slot(
    schedule::ValueId object_id) const noexcept {
    constexpr auto invalid = std::numeric_limits<uint32_t>::max();
    auto slot = object_id.value < _ray_query_status_slots.size() ?
                    _ray_query_status_slots[object_id.value] :
                    invalid;
    return slot == invalid ||
                   slot >= _ray_query_state_handle_storage.size() ?
               nullptr :
               _ray_query_state_handle_storage[slot];
}

[[nodiscard]] ::llvm::AllocaInst *ScheduleEmitter::_ray_query_status_slot(
    schedule::ValueId object_id) const noexcept {
    constexpr auto invalid = std::numeric_limits<uint32_t>::max();
    auto slot = object_id.value < _ray_query_status_slots.size() ?
                    _ray_query_status_slots[object_id.value] :
                    invalid;
    return slot == invalid || slot >= _ray_query_status_storage.size() ?
               nullptr :
               _ray_query_status_storage[slot];
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_ray_query_status_mask(
    schedule::ValueId object_id, uint32_t shift) {
    auto *slot = _ray_query_status_slot(object_id);
    if (slot == nullptr) { return nullptr; }
    auto *status = _builder.CreateLoad(
        _builder.getInt64Ty(), slot, "ray.query.status");
    auto *packed_mask_type = ::llvm::IntegerType::get(
        _module.getContext(), _width);
    auto unpack = [&](uint32_t field_shift) noexcept {
        auto *bits = field_shift == 0u ?
                         status :
                         _builder.CreateLShr(status, field_shift);
        return _builder.CreateBitCast(
            _builder.CreateTrunc(bits, packed_mask_type),
            _layout.mask_type());
    };
    auto *valid = unpack(simd_host_ray_query_valid_status_shift);
    _trap_if(
        _builder.CreateOrReduce(
            _builder.CreateAnd(
                _active_mask, _builder.CreateNot(valid))),
        "ray.query.status.uninitialized");
    // Match the zero passthrough of the replaced masked gather. A scratch
    // color may retain another cohort's bits outside the current definition
    // mask; those lanes must never escape as observable predicate values.
    return _builder.CreateAnd(unpack(shift), _active_mask);
}

void ScheduleEmitter::_ray_query_update_status(
    schedule::ValueId object_id, ::llvm::Value *status) {
    auto *slot = _ray_query_status_slot(object_id);
    if (slot == nullptr) { return; }
    auto *active = _bindless_callback_mask(true);
    auto *active_fields = _builder.CreateOr(
        _builder.CreateOr(
            _builder.CreateOr(
                active,
                _builder.CreateShl(
                    active,
                    simd_host_ray_query_valid_status_shift)),
            _builder.CreateShl(
                active, simd_host_ray_query_surface_status_shift)),
        _builder.CreateShl(
            active, simd_host_ray_query_procedural_status_shift));
    auto *old_status = _builder.CreateLoad(
        _builder.getInt64Ty(), slot);
    auto *merged = _builder.CreateOr(
        _builder.CreateAnd(old_status, _builder.CreateNot(active_fields)),
        _builder.CreateAnd(
            _builder.CreateOr(
                status,
                _builder.CreateShl(
                    active,
                    simd_host_ray_query_valid_status_shift)),
            active_fields));
    _builder.CreateStore(merged, slot);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_ray_query_read(
    const schedule::Instruction &instruction) {
    if (_is_surface_filter_handler_entry()) {
        return _ray_query_surface_filter_read(instruction);
    }
    if (!instruction.result || !instruction.source_op ||
        instruction.operands.size() != 1u) {
        _fail("ray-query object read is malformed");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    if (result == nullptr || result->type == nullptr ||
        result->value_class != schedule::ValueClass::varying) {
        _fail("ray-query object read has an invalid result");
        return nullptr;
    }
    auto op = static_cast<xir::RayQueryObjectReadOp>(
        *instruction.source_op);
    if (result->type->is_bool()) {
        auto shift =
            op == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED ?
                simd_host_ray_query_terminated_status_shift :
            op == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE ?
                simd_host_ray_query_surface_status_shift :
            op == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE ?
                simd_host_ray_query_procedural_status_shift :
                std::numeric_limits<uint32_t>::max();
        if (shift != std::numeric_limits<uint32_t>::max()) {
            if (auto *cached = _ray_query_status_mask(
                    instruction.operands[0u], shift)) {
                return cached;
            }
        }
    }
    auto *states = _ray_query_state_handles(instruction.operands[0u]);
    if (states == nullptr) { return nullptr; }
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

    switch (op) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
            if (!is_ray_type(result->type)) { break; }
            return gather(
                result->type,
                offsetof(SIMDHostRayQueryState, world_ray));
        case xir::RayQueryObjectReadOp::
            RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY: {
            if (!is_ray_type(result->type)) { break; }
            auto &context = _module.getContext();
            auto *pointer_type = ::llvm::PointerType::getUnqual(context);
            auto *pointer_lanes = ::llvm::FixedVectorType::get(
                pointer_type, _width);
            auto *callback_pointers = _builder.CreateGEP(
                _builder.getInt8Ty(), states,
                _builder.CreateVectorSplat(
                    _width, _builder.getInt64(offsetof(
                                SIMDHostRayQueryState,
                                candidate_object_ray))));
            auto *callbacks = _builder.CreateMaskedGather(
                pointer_lanes, callback_pointers,
                ::llvm::Align{alignof(void *)}, _active_mask,
                ::llvm::Constant::getNullValue(pointer_lanes));
            auto *callback = _builder.CreateExtractElement(
                callbacks, _safe_first_lane(_active_mask));
            auto *null_pointer =
                ::llvm::ConstantPointerNull::get(pointer_type);
            _trap_if(
                _builder.CreateICmpEQ(callback, null_pointer),
                "ray.query.object.ray.callback.null");
            auto *callback_mismatch = _builder.CreateAnd(
                _active_mask,
                _builder.CreateICmpNE(
                    callbacks,
                    _builder.CreateVectorSplat(_width, callback)));
            _trap_if(
                _builder.CreateOrReduce(callback_mismatch),
                "ray.query.object.ray.callback.mismatch");
            auto *scratch = _entry_scratch(
                pointer_lanes,
                "ray.query.object.ray.packet." +
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
            return gather(
                result->type,
                offsetof(SIMDHostRayQueryState, object_ray));
        }
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
    if (_is_surface_filter_handler_entry()) {
        _ray_query_surface_filter_write(instruction);
        return;
    }
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
            auto *status_slot = _ray_query_status_slot(
                instruction.operands[0u]);
            auto cache_status = status_slot != nullptr;
            // A cached status callback and the plain callback installed by
            // the same construction are one internal ABI pair. Every status
            // provider must call its paired plain provider and fail closed if
            // an active state's plain callback differs. The status callback
            // vector is still checked below, so a divergent construction may
            // use this path only when the whole active cohort selected the
            // same pair.
            auto trust_status_pairing =
                cache_status &&
                !luisa::compute::detail::env_flag(
                    "LUISA_SIMD_DISABLE_RAY_QUERY_STATUS_CALLBACK_PAIRING");
            auto *callbacks = static_cast<::llvm::Value *>(nullptr);
            auto *plain_callback = static_cast<::llvm::Value *>(nullptr);
            if (!trust_status_pairing) {
                callbacks = _builder.CreateMaskedGather(
                    pointer_lanes,
                    field_pointers(
                        offsetof(SIMDHostRayQueryState, proceed)),
                    ::llvm::Align{alignof(void *)}, _active_mask,
                    ::llvm::Constant::getNullValue(pointer_lanes));
                plain_callback = _builder.CreateExtractElement(
                    callbacks, _safe_first_lane(_active_mask));
            }
            auto *callback = plain_callback;
            auto *status_callbacks = static_cast<::llvm::Value *>(nullptr);
            if (cache_status) {
                auto slot = _ray_query_status_slots[instruction.operands[0u].value];
                status_callbacks = _builder.CreateAlignedLoad(
                    pointer_lanes,
                    _ray_query_status_callback_storage[slot],
                    ::llvm::Align{alignof(void *)},
                    "ray.query.status.callbacks");
                callback = _builder.CreateExtractElement(
                    status_callbacks, _safe_first_lane(_active_mask));
            }
            auto *null_pointer =
                ::llvm::ConstantPointerNull::get(pointer_type);
            _trap_if(
                _builder.CreateICmpEQ(callback, null_pointer),
                "ray.query.proceed.callback.null");
            if (!trust_status_pairing) {
                auto *callback_mismatch = _builder.CreateAnd(
                    _active_mask,
                    _builder.CreateICmpNE(
                        callbacks,
                        _builder.CreateVectorSplat(
                            _width, plain_callback)));
                _trap_if(
                    _builder.CreateOrReduce(callback_mismatch),
                    "ray.query.proceed.callback.mismatch");
            }
            if (cache_status) {
                auto *status_callback_mismatch = _builder.CreateAnd(
                    _active_mask,
                    _builder.CreateICmpNE(
                        status_callbacks,
                        _builder.CreateVectorSplat(_width, callback)));
                _trap_if(
                    _builder.CreateOrReduce(status_callback_mismatch),
                    "ray.query.proceed.status.callback.mismatch");
            }

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
                cache_status ?
                    static_cast<::llvm::Type *>(_builder.getInt64Ty()) :
                    static_cast<::llvm::Type *>(_builder.getVoidTy()),
                {_builder.getInt32Ty(), _builder.getInt64Ty(),
                 pointer_type},
                false);
            auto *call = _builder.CreateCall(
                callback_type, callback,
                {_builder.getInt32(_width),
                 _bindless_callback_mask(true), scratch});
            if (cache_status) {
                _ray_query_update_status(
                    instruction.operands[0u], call);
            }
            return;
        }
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE: {
            auto *surface = _ray_query_status_mask(
                instruction.operands[0u],
                simd_host_ray_query_surface_status_shift);
            if (surface == nullptr) {
                auto *kinds = gather_i32(
                    offsetof(SIMDHostRayQueryState, candidate_kind));
                surface = _builder.CreateICmpEQ(
                    kinds,
                    _builder.CreateVectorSplat(
                        _width,
                        _builder.getInt32(static_cast<uint32_t>(
                            SIMDHostRayQueryCandidateKind::surface))));
            }
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
            auto *procedural = _ray_query_status_mask(
                instruction.operands[0u],
                simd_host_ray_query_procedural_status_shift);
            if (procedural == nullptr) {
                auto *kinds = gather_i32(
                    offsetof(SIMDHostRayQueryState, candidate_kind));
                procedural = _builder.CreateICmpEQ(
                    kinds,
                    _builder.CreateVectorSplat(
                        _width,
                        _builder.getInt32(static_cast<uint32_t>(
                            SIMDHostRayQueryCandidateKind::procedural))));
            }
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
            if (auto *slot = _ray_query_status_slot(
                    instruction.operands[0u])) {
                auto *status = _builder.CreateLoad(
                    _builder.getInt64Ty(), slot);
                _ray_query_update_status(
                    instruction.operands[0u],
                    _builder.CreateOr(
                        status,
                        _bindless_callback_mask(true)));
            }
            return;
    }
    _fail("unsupported ray-query object write operation");
}

}// namespace luisa::compute::simd::detail
