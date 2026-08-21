#include "llvm_schedule_emitter.h"

namespace luisa::compute::simd::detail {

void ScheduleEmitter::_ray_query_pipeline(
    const schedule::Instruction &instruction) {
    if (!instruction.source_op || instruction.operands.empty() ||
        *instruction.source_op >=
            _ray_query_pipeline_handlers.size()) {
        _fail("ray-query pipeline is malformed");
        return;
    }
    auto object_id = instruction.operands.front();
    auto *status_slot = _ray_query_status_slot(object_id);
    if (status_slot == nullptr) {
        _fail("ray-query pipeline requires a proven status owner");
        return;
    }
    auto handler_pair =
        _ray_query_pipeline_handlers[*instruction.source_op];
    if (handler_pair.on_surface == nullptr ||
        handler_pair.on_procedural == nullptr) {
        _fail("ray-query pipeline has a null handler");
        return;
    }

    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *pointer_lanes = ::llvm::FixedVectorType::get(
        pointer_type, _width);
    auto status_index = _ray_query_status_slots[object_id.value];

    if (_width == 1u) {
        auto *states = _ray_query_state_handles(object_id);
        if (states == nullptr) {
            _fail("W1 ray-query pipeline requires a state owner");
            return;
        }
        if (handler_pair.on_candidate_w1 == nullptr ||
            status_index >=
                _ray_query_pipeline_callback_storage.size()) {
            _fail("W1 ray-query pipeline has no resident callback ABI");
            return;
        }
        auto *pipeline_callbacks = _builder.CreateAlignedLoad(
            pointer_lanes,
            _ray_query_pipeline_callback_storage[status_index],
            ::llvm::Align{alignof(void *)},
            "ray.query.pipeline.w1.callbacks");
        auto *pipeline_callback = _builder.CreateExtractElement(
            pipeline_callbacks, _safe_first_lane(_active_mask));
        auto *null_pointer =
            ::llvm::ConstantPointerNull::get(pointer_type);
        _trap_if(
            _builder.CreateICmpEQ(
                pipeline_callback, null_pointer),
            "ray.query.pipeline.w1.callback.null");
        auto *callback_mismatch = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpNE(
                pipeline_callbacks,
                _builder.CreateVectorSplat(
                    _width, pipeline_callback)));
        _trap_if(
            _builder.CreateOrReduce(callback_mismatch),
            "ray.query.pipeline.w1.callback.mismatch");

        auto capture_count = instruction.operands.size() - 1u;
        auto *capture_pointer = static_cast<::llvm::Value *>(
            null_pointer);
        if (capture_count != 0u) {
            auto capture_types = std::vector<::llvm::Type *>{};
            auto capture_values = std::vector<::llvm::Value *>{};
            capture_types.reserve(capture_count);
            capture_values.reserve(capture_count);
            for (auto capture_index = size_t{0u};
                 capture_index < capture_count; capture_index++) {
                auto *capture = _load_value(
                    instruction.operands[capture_index + 1u]);
                if (capture == nullptr) { return; }
                auto *expected = handler_pair.on_surface
                                     ->getFunctionType()
                                     ->getParamType(
                                         static_cast<unsigned>(
                                             capture_index + 4u));
                if (capture->getType() != expected) {
                    _fail("W1 ray-query pipeline capture type mismatch");
                    return;
                }
                capture_types.emplace_back(expected);
                capture_values.emplace_back(capture);
            }
            auto *capture_type = ::llvm::StructType::get(
                context, capture_types, false);
            auto *capture_storage = _entry_scratch(
                capture_type,
                "ray.query.pipeline.w1.captures." +
                    std::to_string(*instruction.source_op));
            capture_storage->setAlignment(::llvm::Align{1u});
            auto *captured = static_cast<::llvm::Value *>(
                ::llvm::PoisonValue::get(capture_type));
            for (auto capture_index = size_t{0u};
                 capture_index < capture_count; capture_index++) {
                captured = _builder.CreateInsertValue(
                    captured, capture_values[capture_index],
                    {static_cast<unsigned>(capture_index)});
            }
            auto *capture_store = _builder.CreateStore(
                captured, capture_storage);
            capture_store->setAlignment(::llvm::Align{1u});
            capture_pointer = capture_storage;
        }

        auto *state = _builder.CreateExtractElement(
            states, _safe_first_lane(_active_mask));
        auto *pipeline_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {pointer_type, pointer_type, pointer_type,
             pointer_type},
            false);
        _builder.CreateCall(
            pipeline_type, pipeline_callback,
            {state, capture_pointer, _launch_config,
             handler_pair.on_candidate_w1});
        auto *outer_active_bits = _bindless_callback_mask(true);
        _ray_query_update_status(object_id, outer_active_bits);
        return;
    }

    auto *null_pointer =
        ::llvm::ConstantPointerNull::get(pointer_type);
    auto *outer_active_bits = _bindless_callback_mask(true);
    auto *exit = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.exit", _entry);
    auto *surface_filter_ray_packet =
        static_cast<::llvm::Value *>(nullptr);
    auto *surface_filter_call_ray_packet =
        static_cast<::llvm::Value *>(nullptr);
    if (handler_pair.embree_surface_filter_safe &&
        status_index <
            _ray_query_surface_filter_ray_packet_storage.size()) {
        surface_filter_ray_packet =
            _ray_query_surface_filter_ray_packet_storage[status_index];
        surface_filter_call_ray_packet =
            _ray_query_surface_filter_ray_packet_call_storage[status_index];
    }
    auto *surface_filter_handler =
        static_cast<::llvm::Value *>(nullptr);
    if (handler_pair.on_surface_filter != nullptr) {
        surface_filter_handler = handler_pair.on_surface_filter;
        if (handler_pair.on_surface_filter_scheduler_oracle != nullptr) {
            if (handler_pair.on_surface_filter->getFunctionType() !=
                handler_pair.on_surface_filter_scheduler_oracle
                    ->getFunctionType()) {
                _fail("surface-filter scheduler oracle type does not match the compact handler");
                return;
            }
            auto *enabled = _load_launch_u32(offsetof(
                SIMDPacketLaunchConfig,
                enable_predicated_acyclic_surface_filter));
            surface_filter_handler = _builder.CreateSelect(
                _builder.CreateICmpNE(
                    enabled, _builder.getInt32(0u)),
                handler_pair.on_surface_filter,
                handler_pair.on_surface_filter_scheduler_oracle,
                "ray.query.surface.filter.handler");
        }
    }
    auto *direct_output_surface_filter_callback =
        static_cast<::llvm::Value *>(nullptr);
    auto *direct_output_surface_filter_accel =
        static_cast<::llvm::Value *>(nullptr);
    if (handler_pair.embree_surface_filter_safe &&
        !handler_pair.surface_handler_empty &&
        handler_pair.on_surface_filter != nullptr && _width >= 4u &&
        status_index <
            _ray_query_direct_output_surface_filter_pipeline_callback_storage
                .size() &&
        status_index <
            _ray_query_direct_output_surface_filter_accel_storage.size()) {
        auto *pipeline_callbacks = _builder.CreateMaskedLoad(
            pointer_lanes,
            _ray_query_direct_output_surface_filter_pipeline_callback_storage
                [status_index],
            ::llvm::Align{alignof(void *)},
            _active_mask,
            ::llvm::Constant::getNullValue(pointer_lanes),
            "ray.query.direct.output.surface.filter.pipeline.callbacks");
        direct_output_surface_filter_callback =
            _builder.CreateExtractElement(
                pipeline_callbacks, _safe_first_lane(_active_mask));
        auto *pipeline_callback_mismatch = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpNE(
                pipeline_callbacks,
                _builder.CreateVectorSplat(
                    _width, direct_output_surface_filter_callback)));
        _trap_if(
            _builder.CreateOrReduce(pipeline_callback_mismatch),
            "ray.query.direct.output.surface.filter.pipeline.callback.mismatch");
        auto *accels = _builder.CreateMaskedLoad(
            pointer_lanes,
            _ray_query_direct_output_surface_filter_accel_storage
                [status_index],
            ::llvm::Align{alignof(void *)},
            _active_mask,
            ::llvm::Constant::getNullValue(pointer_lanes),
            "ray.query.direct.output.surface.filter.accels");
        direct_output_surface_filter_accel =
            _builder.CreateExtractElement(
                accels, _safe_first_lane(_active_mask));
        auto *accel_mismatch = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpNE(
                accels,
                _builder.CreateVectorSplat(
                    _width, direct_output_surface_filter_accel)));
        _trap_if(
            _builder.CreateOrReduce(accel_mismatch),
            "ray.query.direct.output.surface.filter.accel.mismatch");
        _trap_if(
            _builder.CreateAnd(
                _builder.CreateICmpNE(
                    direct_output_surface_filter_callback,
                    null_pointer),
                _builder.CreateICmpEQ(
                    direct_output_surface_filter_accel,
                    null_pointer)),
            "ray.query.direct.output.surface.filter.accel.null");
    }
    if (direct_output_surface_filter_callback != nullptr) {
        auto *output_only_call = ::llvm::BasicBlock::Create(
            context, "ray.query.pipeline.direct.output", _entry);
        auto *regular = ::llvm::BasicBlock::Create(
            context, "ray.query.pipeline.direct.output.regular", _entry);
        _builder.CreateCondBr(
            _builder.CreateICmpNE(
                direct_output_surface_filter_callback,
                null_pointer),
            output_only_call, regular);

        _builder.SetInsertPoint(output_only_call);
        if (surface_filter_ray_packet == nullptr ||
            surface_filter_call_ray_packet == nullptr ||
            direct_output_surface_filter_accel == nullptr ||
            surface_filter_handler == nullptr ||
            status_index >=
                _ray_query_direct_output_surface_filter_committed_storage
                    .size() ||
            _ray_query_direct_output_surface_filter_committed_storage
                    [status_index] == nullptr) {
            _fail("direct-output surface-filter packet has no analyzed storage or handler");
            return;
        }
        auto *ray_packet =
            _ray_query_surface_filter_ray_packet_for_call(
                surface_filter_ray_packet,
                surface_filter_call_ray_packet,
                outer_active_bits);
        if (ray_packet == nullptr) { return; }
        auto *pipeline_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {_builder.getInt32Ty(), _builder.getInt64Ty(),
             pointer_type, pointer_type, pointer_type,
             _builder.getInt32Ty(), pointer_type},
            false);
        auto *query_object = _source.value(object_id);
        auto query_any =
            query_object != nullptr &&
            query_object->type == Type::custom("LC_RayQueryAny");
        _builder.CreateCall(
            pipeline_type, direct_output_surface_filter_callback,
            {_builder.getInt32(_width), outer_active_bits,
             direct_output_surface_filter_accel,
             _ray_query_direct_output_surface_filter_committed_storage
                 [status_index],
             ray_packet,
             _builder.getInt32(query_any ? 1u : 0u),
             surface_filter_handler});
        _builder.CreateBr(exit);
        _builder.SetInsertPoint(regular);
    }

    auto *states = _ray_query_state_handles(object_id);
    if (states == nullptr) {
        _fail("ray-query pipeline fallback requires a state owner");
        return;
    }

    if (status_index >= _ray_query_status_callback_storage.size()) {
        _fail("ray-query pipeline status callback slot is invalid");
        return;
    }
    auto *status_callbacks = _builder.CreateAlignedLoad(
        pointer_lanes,
        _ray_query_status_callback_storage[status_index],
        ::llvm::Align{alignof(void *)},
        "ray.query.pipeline.status.callbacks");
    auto *status_callback = _builder.CreateExtractElement(
        status_callbacks, _safe_first_lane(_active_mask));
    _trap_if(
        _builder.CreateICmpEQ(status_callback, null_pointer),
        "ray.query.pipeline.callback.null");
    auto *callback_mismatch = _builder.CreateAnd(
        _active_mask,
        _builder.CreateICmpNE(
            status_callbacks,
            _builder.CreateVectorSplat(
                _width, status_callback)));
    _trap_if(
        _builder.CreateOrReduce(callback_mismatch),
        "ray.query.pipeline.callback.mismatch");

    auto *scratch = _entry_scratch(
        pointer_lanes,
        "ray.query.pipeline.packet." +
            std::to_string(*instruction.source_op));
    scratch->setAlignment(::llvm::Align{alignof(void *)});

    auto handler_arguments = std::vector<::llvm::Value *>{
        _builder.getInt32(_width), nullptr, scratch, _launch_config};
    handler_arguments.reserve(instruction.operands.size() + 3u);
    for (auto capture_index = size_t{1u};
         capture_index < instruction.operands.size(); capture_index++) {
        auto *capture = _load_value(
            instruction.operands[capture_index]);
        if (capture == nullptr) { return; }
        handler_arguments.emplace_back(capture);
    }
    auto *surface_filter_callback =
        static_cast<::llvm::Value *>(nullptr);
    auto *empty_surface_filter_callback =
        static_cast<::llvm::Value *>(nullptr);
    auto *empty_surface_filter_accel =
        static_cast<::llvm::Value *>(nullptr);
    if (handler_pair.embree_surface_filter_safe &&
        status_index <
            _ray_query_surface_filter_pipeline_callback_storage.size()) {
        auto *pipeline_callbacks = _builder.CreateAlignedLoad(
            pointer_lanes,
            _ray_query_surface_filter_pipeline_callback_storage[status_index],
            ::llvm::Align{alignof(void *)},
            "ray.query.surface.filter.pipeline.callbacks");
        surface_filter_callback = _builder.CreateExtractElement(
            pipeline_callbacks, _safe_first_lane(_active_mask));
        auto *pipeline_callback_mismatch = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpNE(
                pipeline_callbacks,
                _builder.CreateVectorSplat(
                    _width, surface_filter_callback)));
        _trap_if(
            _builder.CreateOrReduce(pipeline_callback_mismatch),
            "ray.query.surface.filter.pipeline.callback.mismatch");
    }
    if (handler_pair.surface_handler_empty && _width >= 4u &&
        status_index <
            _ray_query_empty_surface_filter_pipeline_callback_storage.size() &&
        status_index <
            _ray_query_empty_surface_filter_accel_storage.size()) {
        auto *pipeline_callbacks = _builder.CreateAlignedLoad(
            pointer_lanes,
            _ray_query_empty_surface_filter_pipeline_callback_storage
                [status_index],
            ::llvm::Align{alignof(void *)},
            "ray.query.empty.surface.filter.pipeline.callbacks");
        empty_surface_filter_callback = _builder.CreateExtractElement(
            pipeline_callbacks, _safe_first_lane(_active_mask));
        auto *pipeline_callback_mismatch = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpNE(
                pipeline_callbacks,
                _builder.CreateVectorSplat(
                    _width, empty_surface_filter_callback)));
        _trap_if(
            _builder.CreateOrReduce(pipeline_callback_mismatch),
            "ray.query.empty.surface.filter.pipeline.callback.mismatch");
        auto *accels = _builder.CreateAlignedLoad(
            pointer_lanes,
            _ray_query_empty_surface_filter_accel_storage[status_index],
            ::llvm::Align{alignof(void *)},
            "ray.query.empty.surface.filter.accels");
        empty_surface_filter_accel = _builder.CreateExtractElement(
            accels, _safe_first_lane(_active_mask));
        auto *accel_mismatch = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpNE(
                accels,
                _builder.CreateVectorSplat(
                    _width, empty_surface_filter_accel)));
        _trap_if(
            _builder.CreateOrReduce(accel_mismatch),
            "ray.query.empty.surface.filter.accel.mismatch");
        _trap_if(
            _builder.CreateAnd(
                _builder.CreateICmpNE(
                    empty_surface_filter_callback, null_pointer),
                _builder.CreateICmpEQ(
                    empty_surface_filter_accel, null_pointer)),
            "ray.query.empty.surface.filter.accel.null");
    }
    auto state_packet_stored = false;
    auto store_state_packet = [&]() noexcept {
        auto *safe_states = _builder.CreateSelect(
            _active_mask, states,
            ::llvm::Constant::getNullValue(states->getType()));
        auto *state_store = _builder.CreateStore(
            safe_states, scratch);
        state_store->setAlignment(
            ::llvm::Align{alignof(void *)});
        state_packet_stored = true;
    };
    if (empty_surface_filter_callback != nullptr) {
        // The empty-handler ABI still receives the minimal state packet.
        store_state_packet();
    }
    auto *preheader = _builder.GetInsertBlock();
    auto *loop = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.loop", _entry);
    auto *surface_filter_call =
        surface_filter_callback == nullptr ?
            nullptr :
            ::llvm::BasicBlock::Create(
                context,
                "ray.query.pipeline.surface.filter", _entry);
    auto *surface_call = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.surface", _entry);
    auto *after_surface = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.after.surface", _entry);
    auto *procedural_call = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.procedural", _entry);
    auto *continue_loop = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.continue", _entry);
    auto *regular_preheader = preheader;
    if (empty_surface_filter_callback != nullptr) {
        auto *output_only_call = ::llvm::BasicBlock::Create(
            context, "ray.query.pipeline.output.only", _entry);
        regular_preheader = ::llvm::BasicBlock::Create(
            context, "ray.query.pipeline.regular", _entry);
        _builder.CreateCondBr(
            _builder.CreateICmpNE(
                empty_surface_filter_callback, null_pointer),
            output_only_call, regular_preheader);

        _builder.SetInsertPoint(output_only_call);
        if (surface_filter_ray_packet == nullptr ||
            surface_filter_call_ray_packet == nullptr ||
            empty_surface_filter_accel == nullptr) {
            _fail("output-only empty surface-filter packet has no analyzed storage");
            return;
        }
        auto *ray_packet =
            _ray_query_surface_filter_ray_packet_for_call(
                surface_filter_ray_packet,
                surface_filter_call_ray_packet,
                outer_active_bits);
        if (ray_packet == nullptr) { return; }
        auto *pipeline_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {_builder.getInt32Ty(), _builder.getInt64Ty(),
             pointer_type, pointer_type, pointer_type,
             _builder.getInt32Ty()},
            false);
        auto *query_object = _source.value(object_id);
        auto query_any =
            query_object != nullptr &&
            query_object->type == Type::custom("LC_RayQueryAny");
        _builder.CreateCall(
            pipeline_type, empty_surface_filter_callback,
            {_builder.getInt32(_width), outer_active_bits,
             empty_surface_filter_accel, scratch, ray_packet,
             _builder.getInt32(query_any ? 1u : 0u)});
        _builder.CreateBr(exit);
        _builder.SetInsertPoint(regular_preheader);
    }
    if (!state_packet_stored) { store_state_packet(); }
    if (surface_filter_callback == nullptr) {
        _builder.CreateBr(loop);
    } else {
        _builder.CreateCondBr(
            _builder.CreateICmpNE(
                surface_filter_callback, null_pointer),
            surface_filter_call, loop);

        _builder.SetInsertPoint(surface_filter_call);
        if (_width >= 4u) {
            if (surface_filter_ray_packet == nullptr ||
                surface_filter_call_ray_packet == nullptr) {
                _fail("surface-filter call packet has no analyzed storage");
                return;
            }
            auto *pipeline_type = ::llvm::FunctionType::get(
                _builder.getVoidTy(),
                {_builder.getInt32Ty(), _builder.getInt64Ty(),
                 pointer_type, pointer_type, pointer_type,
                 pointer_type, pointer_type},
                false);
            auto *ray_packet =
                _ray_query_surface_filter_ray_packet_for_call(
                    surface_filter_ray_packet,
                    surface_filter_call_ray_packet,
                    outer_active_bits);
            if (ray_packet == nullptr) { return; }
            if (surface_filter_handler == nullptr) {
                _fail("surface-filter packet has no direct handler");
                return;
            }
            _builder.CreateCall(
                pipeline_type, surface_filter_callback,
                {_builder.getInt32(_width), outer_active_bits,
                 scratch, ray_packet, _launch_config,
                 handler_pair.on_surface,
                 surface_filter_handler});
        } else {
            auto *pipeline_type = ::llvm::FunctionType::get(
                _builder.getVoidTy(),
                {_builder.getInt32Ty(), _builder.getInt64Ty(),
                 pointer_type, pointer_type, pointer_type},
                false);
            _builder.CreateCall(
                pipeline_type, surface_filter_callback,
                {_builder.getInt32(_width), outer_active_bits,
                 scratch, _launch_config,
                 handler_pair.on_surface});
        }
        _builder.CreateBr(exit);
    }

    _builder.SetInsertPoint(loop);
    auto *active_bits = _builder.CreatePHI(
        _builder.getInt64Ty(), 2u,
        "ray.query.pipeline.active.bits");
    active_bits->addIncoming(outer_active_bits, regular_preheader);
    auto *status_callback_type = ::llvm::FunctionType::get(
        _builder.getInt64Ty(),
        {_builder.getInt32Ty(), _builder.getInt64Ty(), pointer_type},
        false);
    auto *status = _builder.CreateCall(
        status_callback_type, status_callback,
        {_builder.getInt32(_width), active_bits, scratch},
        "ray.query.pipeline.status");
    auto lane_bits = (uint64_t{1u} << _width) - 1u;
    auto *lane_mask = _builder.getInt64(lane_bits);
    auto field = [&](uint32_t shift,
                     std::string_view name) noexcept {
        auto *bits = shift == 0u ?
                         status :
                         _builder.CreateLShr(status, shift);
        return _builder.CreateAnd(
            _builder.CreateAnd(bits, active_bits), lane_mask,
            std::string{name});
    };
    auto *terminated = field(
        simd_host_ray_query_terminated_status_shift,
        "ray.query.pipeline.terminated");
    auto *alive = _builder.CreateAnd(
        active_bits, _builder.CreateNot(terminated),
        "ray.query.pipeline.alive");
    auto *surface = field(
        simd_host_ray_query_surface_status_shift,
        "ray.query.pipeline.surface.bits");
    surface = _builder.CreateAnd(surface, alive);
    auto *procedural = field(
        simd_host_ray_query_procedural_status_shift,
        "ray.query.pipeline.procedural.bits");
    procedural = _builder.CreateAnd(procedural, alive);
    _trap_if(
        _builder.CreateICmpNE(
            _builder.CreateAnd(surface, procedural),
            _builder.getInt64(0u)),
        "ray.query.pipeline.candidate.overlap");
    auto *unclassified = _builder.CreateAnd(
        alive,
        _builder.CreateNot(
            _builder.CreateOr(surface, procedural)));
    _trap_if(
        _builder.CreateICmpNE(
            unclassified, _builder.getInt64(0u)),
        "ray.query.pipeline.unclassified");
    _builder.CreateCondBr(
        _builder.CreateICmpNE(surface, _builder.getInt64(0u)),
        surface_call, after_surface);

    _builder.SetInsertPoint(surface_call);
    handler_arguments[1u] = surface;
    _builder.CreateCall(
        handler_pair.on_surface, handler_arguments);
    _builder.CreateBr(after_surface);

    _builder.SetInsertPoint(after_surface);
    _builder.CreateCondBr(
        _builder.CreateICmpNE(
            procedural, _builder.getInt64(0u)),
        procedural_call, continue_loop);

    _builder.SetInsertPoint(procedural_call);
    handler_arguments[1u] = procedural;
    _builder.CreateCall(
        handler_pair.on_procedural, handler_arguments);
    _builder.CreateBr(continue_loop);

    _builder.SetInsertPoint(continue_loop);
    auto *has_more = _builder.CreateICmpNE(
        alive, _builder.getInt64(0u));
    _builder.CreateCondBr(has_more, loop, exit);
    active_bits->addIncoming(alive, continue_loop);

    _builder.SetInsertPoint(exit);
    // The pipeline returns only once every original active lane has reached
    // the terminal state. Publish that fact into the same sidecar consumed by
    // any following query-object predicate.
    _ray_query_update_status(object_id, outer_active_bits);
}

}// namespace luisa::compute::simd::detail
