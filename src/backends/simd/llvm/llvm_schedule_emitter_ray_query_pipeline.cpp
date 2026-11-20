#include "llvm_schedule_emitter.h"
#include <luisa/core/stl/string.h>

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
        handler_pair.on_surface_filter != nullptr && _width >= 2u &&
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
    auto *empty_surface_filter_callback =
        static_cast<::llvm::Value *>(nullptr);
    auto *empty_surface_filter_accel =
        static_cast<::llvm::Value *>(nullptr);
    if (handler_pair.embree_surface_filter_safe &&
        handler_pair.surface_handler_empty && _width >= 2u &&
        status_index <
            _ray_query_empty_surface_filter_pipeline_callback_storage.size() &&
        status_index <
            _ray_query_empty_surface_filter_accel_storage.size()) {
        auto *pipeline_callbacks = _builder.CreateMaskedLoad(
            pointer_lanes,
            _ray_query_empty_surface_filter_pipeline_callback_storage
                [status_index],
            ::llvm::Align{alignof(void *)},
            _active_mask,
            ::llvm::Constant::getNullValue(pointer_lanes),
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
        auto *accels = _builder.CreateMaskedLoad(
            pointer_lanes,
            _ray_query_empty_surface_filter_accel_storage[status_index],
            ::llvm::Align{alignof(void *)},
            _active_mask,
            ::llvm::Constant::getNullValue(pointer_lanes),
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
            status_index >= _ray_query_output_packet_storage.size() ||
            _ray_query_output_packet_storage[status_index] == nullptr) {
            _fail("direct-output surface-filter packet has no analyzed storage or handler");
            return;
        }
        auto *narrowing_eligible = static_cast<::llvm::Value *>(
            _builder.getFalse());
        if (_width == 16u &&
            handler_pair.on_surface_filter_w4 != nullptr &&
            handler_pair.on_surface_filter_w8 != nullptr) {
            // A retained scheduler oracle has only the logical-W16 ABI. Keep
            // its same-module control exact-width even if the narrowing flag
            // was supplied manually by a diagnostic caller.
            narrowing_eligible = _builder.CreateICmpEQ(
                surface_filter_handler,
                handler_pair.on_surface_filter,
                "ray.query.direct.packet.compact.handler");
        }
        auto [ray_packet, ray_packet_width] =
            _ray_query_output_surface_filter_ray_packet_for_call(
                surface_filter_ray_packet,
                surface_filter_call_ray_packet,
                outer_active_bits,
                simd_packet_launch_flag_w16_sparse_direct_output_surface_filter_packet_narrowing,
                narrowing_eligible, "direct");
        if (ray_packet == nullptr || ray_packet_width == nullptr) { return; }
        auto *physical_surface_filter_handler =
            surface_filter_handler;
        if (_width == 16u &&
            handler_pair.on_surface_filter_w4 != nullptr &&
            handler_pair.on_surface_filter_w8 != nullptr) {
            physical_surface_filter_handler = _builder.CreateSelect(
                _builder.CreateICmpEQ(
                    ray_packet_width, _builder.getInt32(4u)),
                handler_pair.on_surface_filter_w4,
                _builder.CreateSelect(
                    _builder.CreateICmpEQ(
                        ray_packet_width, _builder.getInt32(8u)),
                    handler_pair.on_surface_filter_w8,
                    surface_filter_handler),
                "ray.query.direct.packet.handler");
        }
        auto *pipeline_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {_builder.getInt32Ty(), _builder.getInt32Ty(),
             _builder.getInt64Ty(), pointer_type, pointer_type,
             pointer_type, _builder.getInt32Ty(), pointer_type},
            false);
        auto *query_object = _source.value(object_id);
        auto query_any =
            query_object != nullptr &&
            query_object->type == Type::custom("LC_RayQueryAny");
        _builder.CreateCall(
            pipeline_type, direct_output_surface_filter_callback,
            {_builder.getInt32(_width), ray_packet_width,
             outer_active_bits,
             direct_output_surface_filter_accel,
             _ray_query_output_packet_storage[status_index],
             ray_packet,
             _builder.getInt32(query_any ? 1u : 0u),
             physical_surface_filter_handler});
        _builder.CreateBr(exit);
        _builder.SetInsertPoint(regular);
    }
    if (empty_surface_filter_callback != nullptr) {
        auto *output_only_call = ::llvm::BasicBlock::Create(
            context, "ray.query.pipeline.empty.output", _entry);
        auto *regular = ::llvm::BasicBlock::Create(
            context, "ray.query.pipeline.empty.output.regular", _entry);
        _builder.CreateCondBr(
            _builder.CreateICmpNE(
                empty_surface_filter_callback, null_pointer),
            output_only_call, regular);

        _builder.SetInsertPoint(output_only_call);
        if (surface_filter_ray_packet == nullptr ||
            surface_filter_call_ray_packet == nullptr ||
            empty_surface_filter_accel == nullptr ||
            status_index >= _ray_query_output_packet_storage.size() ||
            _ray_query_output_packet_storage[status_index] == nullptr) {
            _fail("empty output-only surface-filter packet has no analyzed storage");
            return;
        }
        auto [ray_packet, ray_packet_width] =
            _ray_query_output_surface_filter_ray_packet_for_call(
                surface_filter_ray_packet,
                surface_filter_call_ray_packet,
                outer_active_bits,
                simd_packet_launch_flag_w16_sparse_empty_surface_filter_packet_narrowing,
                _builder.getTrue(), "empty");
        if (ray_packet == nullptr || ray_packet_width == nullptr) { return; }
        auto *pipeline_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {_builder.getInt32Ty(), _builder.getInt32Ty(),
             _builder.getInt64Ty(),
             pointer_type, pointer_type, pointer_type,
             _builder.getInt32Ty()},
            false);
        auto *query_object = _source.value(object_id);
        auto query_any =
            query_object != nullptr &&
            query_object->type == Type::custom("LC_RayQueryAny");
        _builder.CreateCall(
            pipeline_type, empty_surface_filter_callback,
            {_builder.getInt32(_width), ray_packet_width,
             outer_active_bits,
             empty_surface_filter_accel,
             _ray_query_output_packet_storage[status_index],
             ray_packet,
             _builder.getInt32(query_any ? 1u : 0u)});
        _builder.CreateBr(exit);
        _builder.SetInsertPoint(regular);
    }
    if (_width >= 2u && !handler_pair.embree_surface_filter_safe &&
        handler_pair.surface_exclusion_safe &&
        status_index <
            _ray_query_surface_exclusion_soa_accel_storage.size() &&
        status_index <
            _ray_query_surface_exclusion_soa_input_storage.size() &&
        status_index < _ray_query_output_packet_storage.size() &&
        status_index < _ray_query_live_dispatch_storage.size() &&
        _ray_query_surface_exclusion_soa_accel_storage[status_index] !=
            nullptr &&
        _ray_query_surface_exclusion_soa_input_storage[status_index] !=
            nullptr &&
        _ray_query_output_packet_storage[status_index] != nullptr &&
        _ray_query_live_dispatch_storage[status_index] != nullptr) {
        auto *soa_accels = _builder.CreateMaskedLoad(
            pointer_lanes,
            _ray_query_surface_exclusion_soa_accel_storage[status_index],
            ::llvm::Align{alignof(void *)},
            _active_mask,
            ::llvm::Constant::getNullValue(pointer_lanes),
            "ray.query.surface.exclusion.soa.accels");
        auto *soa_accel = _builder.CreateExtractElement(
            soa_accels, _safe_first_lane(_active_mask));
        auto *soa_accel_mismatch = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpNE(
                soa_accels,
                _builder.CreateVectorSplat(_width, soa_accel)));
        _trap_if(
            _builder.CreateOrReduce(soa_accel_mismatch),
            "ray.query.surface.exclusion.soa.accel.mismatch");
        auto *soa_call = ::llvm::BasicBlock::Create(
            context, "ray.query.pipeline.surface.exclusion.soa", _entry);
        auto *regular = ::llvm::BasicBlock::Create(
            context, "ray.query.pipeline.surface.exclusion.soa.regular",
            _entry);
        _builder.CreateCondBr(
            _builder.CreateICmpNE(soa_accel, null_pointer),
            soa_call, regular);

        _builder.SetInsertPoint(soa_call);
        auto *live_dispatches = _builder.CreateAlignedLoad(
            pointer_lanes,
            _ray_query_live_dispatch_storage[status_index],
            ::llvm::Align{alignof(void *)},
            "ray.query.surface.exclusion.soa.live.dispatches");
        auto *live_dispatch = _builder.CreateExtractElement(
            live_dispatches, _safe_first_lane(_active_mask));
        auto *live_mismatch = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpNE(
                live_dispatches,
                _builder.CreateVectorSplat(_width, live_dispatch)));
        _trap_if(
            _builder.CreateOrReduce(live_mismatch),
            "ray.query.surface.exclusion.soa.live.dispatch.mismatch");
        _trap_if(
            _builder.CreateICmpEQ(live_dispatch, null_pointer),
            "ray.query.surface.exclusion.soa.live.dispatch.null");
        auto *fn_ptr = _builder.CreateLoad(
            pointer_type,
            _byte_pointer(
                live_dispatch,
                offsetof(
                    SIMDHostAccelRayQueryLiveDispatch,
                    surface_exclusion_soa)),
            "ray.query.surface.exclusion.soa.fn");
        fn_ptr->setAlignment(::llvm::Align{alignof(void *)});
        _trap_if(
            _builder.CreateICmpEQ(fn_ptr, null_pointer),
            "ray.query.surface.exclusion.soa.fn.null");
        auto *i32_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), _width);
        std::array<::llvm::Value *, 4u> excl_ptrs{};
        for (auto k = size_t{0u}; k < 4u; k++) {
            auto cap_idx =
                handler_pair.surface_exclusion_capture_indices[k];
            ::llvm::Value *vec = nullptr;
            if (cap_idx != std::numeric_limits<uint32_t>::max()) {
                if (cap_idx == 0u ||
                    cap_idx >= instruction.operands.size()) {
                    _fail("surface-exclusion capture index is out of range");
                    return;
                }
                auto operand_id = instruction.operands[cap_idx];
                auto *op_val = _source.value(operand_id);
                if (op_val == nullptr) {
                    _fail("surface-exclusion capture value is missing");
                    return;
                }
                auto *loaded = _load_value(operand_id);
                if (loaded == nullptr) { return; }
                vec = _is_local_lvalue(operand_id) ?
                          _gather_data(
                              _local_base(_builder, loaded),
                              _local_offsets(_builder, loaded),
                              op_val->type) :
                          _as_lane_vector(loaded, *op_val);
                if (vec == nullptr) { return; }
            } else {
                vec = _builder.CreateVectorSplat(
                    _width,
                    _builder.getInt32(
                        handler_pair.surface_exclusion_constants[k]));
            }
            auto *slot = _entry_scratch(
                i32_lanes,
                "ray.query.excl.soa." +
                    std::to_string(*instruction.source_op) + "." +
                    std::to_string(k));
            slot->setAlignment(::llvm::Align{16u});
            _builder.CreateAlignedStore(vec, slot, ::llvm::Align{16u});
            excl_ptrs[k] = slot;
        }
        auto *query_object = _source.value(object_id);
        auto query_any =
            query_object != nullptr &&
            query_object->type == Type::custom("LC_RayQueryAny");
        auto flags =
            (handler_pair.surface_exclusion_terminate_on_commit ?
                 simd_host_ray_query_live_flag_terminate_on_commit :
                 0u) |
            (query_any ? simd_host_ray_query_live_flag_query_any : 0u);
        auto *fn_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {_builder.getInt32Ty(), _builder.getInt64Ty(),
             pointer_type, pointer_type, pointer_type,
             pointer_type, pointer_type, pointer_type, pointer_type,
             _builder.getInt32Ty()},
            false);
        _builder.CreateCall(
            fn_type, fn_ptr,
            {_builder.getInt32(_width), outer_active_bits,
             soa_accel,
             _ray_query_surface_exclusion_soa_input_storage[status_index],
             _ray_query_output_packet_storage[status_index],
             excl_ptrs[0u], excl_ptrs[1u], excl_ptrs[2u], excl_ptrs[3u],
             _builder.getInt32(flags)});
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
    auto *safe_states = _builder.CreateSelect(
        _active_mask, states,
        ::llvm::Constant::getNullValue(states->getType()));
    auto *state_store = _builder.CreateStore(
        safe_states, scratch);
    state_store->setAlignment(
        ::llvm::Align{alignof(void *)});
    if (_width >= 2u && !handler_pair.embree_surface_filter_safe &&
        (handler_pair.surface_exclusion_safe ||
         (handler_pair.live_embree_safe &&
          handler_pair.on_candidate_live != nullptr)) &&
        status_index < _ray_query_live_dispatch_storage.size() &&
        _ray_query_live_dispatch_storage[status_index] != nullptr) {
        auto *live_dispatches = _builder.CreateAlignedLoad(
            pointer_lanes,
            _ray_query_live_dispatch_storage[status_index],
            ::llvm::Align{alignof(void *)},
            "ray.query.live.dispatches");
        auto *live_dispatch = _builder.CreateExtractElement(
            live_dispatches, _safe_first_lane(_active_mask));
        auto *live_mismatch = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpNE(
                live_dispatches,
                _builder.CreateVectorSplat(_width, live_dispatch)));
        _trap_if(
            _builder.CreateOrReduce(live_mismatch),
            "ray.query.live.dispatch.mismatch");
        auto *live_call = ::llvm::BasicBlock::Create(
            context, "ray.query.pipeline.live", _entry);
        auto *after_live = ::llvm::BasicBlock::Create(
            context, "ray.query.pipeline.after.live", _entry);
        _builder.CreateCondBr(
            _builder.CreateICmpNE(live_dispatch, null_pointer),
            live_call, after_live);

        _builder.SetInsertPoint(live_call);
        if (handler_pair.surface_exclusion_safe) {
            auto *fn_ptr = _builder.CreateLoad(
                pointer_type,
                _byte_pointer(
                    live_dispatch,
                    offsetof(
                        SIMDHostAccelRayQueryLiveDispatch,
                        surface_exclusion)),
                "ray.query.surface.exclusion.fn");
            fn_ptr->setAlignment(::llvm::Align{alignof(void *)});
            auto *i32_lanes = ::llvm::FixedVectorType::get(
                _builder.getInt32Ty(), _width);
            std::array<::llvm::Value *, 4u> excl_ptrs{};
            for (auto k = size_t{0u}; k < 4u; k++) {
                auto cap_idx =
                    handler_pair.surface_exclusion_capture_indices[k];
                ::llvm::Value *vec = nullptr;
                if (cap_idx != std::numeric_limits<uint32_t>::max()) {
                    if (cap_idx == 0u ||
                        cap_idx >= instruction.operands.size()) {
                        _fail("surface-exclusion capture index is out of range");
                        return;
                    }
                    auto operand_id = instruction.operands[cap_idx];
                    auto *op_val = _source.value(operand_id);
                    if (op_val == nullptr) {
                        _fail("surface-exclusion capture value is missing");
                        return;
                    }
                    auto *loaded = _load_value(operand_id);
                    if (loaded == nullptr) { return; }
                    vec = _is_local_lvalue(operand_id) ?
                              _gather_data(
                                  _local_base(_builder, loaded),
                                  _local_offsets(_builder, loaded),
                                  op_val->type) :
                              _as_lane_vector(loaded, *op_val);
                    if (vec == nullptr) { return; }
                } else {
                    vec = _builder.CreateVectorSplat(
                        _width,
                        _builder.getInt32(
                            handler_pair.surface_exclusion_constants[k]));
                }
                auto *slot = _entry_scratch(
                    i32_lanes,
                    "ray.query.excl." +
                        std::to_string(*instruction.source_op) + "." +
                        std::to_string(k));
                slot->setAlignment(::llvm::Align{16u});
                _builder.CreateAlignedStore(
                    vec, slot, ::llvm::Align{16u});
                excl_ptrs[k] = slot;
            }
            auto flags =
                handler_pair.surface_exclusion_terminate_on_commit ?
                    simd_host_ray_query_live_flag_terminate_on_commit :
                    0u;
            auto *fn_type = ::llvm::FunctionType::get(
                _builder.getVoidTy(),
                {_builder.getInt32Ty(), _builder.getInt64Ty(),
                 pointer_type, pointer_type, pointer_type,
                 pointer_type, pointer_type, _builder.getInt32Ty()},
                false);
            _builder.CreateCall(
                fn_type, fn_ptr,
                {_builder.getInt32(_width), outer_active_bits, scratch,
                 excl_ptrs[0u], excl_ptrs[1u],
                 excl_ptrs[2u], excl_ptrs[3u],
                 _builder.getInt32(flags)});
            _builder.CreateBr(exit);
        } else {
            auto *fn_ptr = _builder.CreateLoad(
                pointer_type,
                _byte_pointer(
                    live_dispatch,
                    offsetof(
                        SIMDHostAccelRayQueryLiveDispatch,
                        live_pipeline)),
                "ray.query.live.pipeline.fn");
            fn_ptr->setAlignment(::llvm::Align{alignof(void *)});
            auto capture_count = instruction.operands.size() - 1u;
            auto *capture_pointer =
                static_cast<::llvm::Value *>(null_pointer);
            if (capture_count != 0u) {
                std::vector<::llvm::Type *> capture_types;
                capture_types.reserve(capture_count);
                for (auto i = size_t{0u}; i < capture_count; i++) {
                    capture_types.emplace_back(
                        handler_arguments[i + 4u]->getType());
                }
                auto *capture_type = ::llvm::StructType::get(
                    context, capture_types, false);
                auto *capture_storage = _entry_scratch(
                    capture_type,
                    "ray.query.pipeline.live.captures." +
                        std::to_string(*instruction.source_op));
                capture_storage->setAlignment(::llvm::Align{64u});
                for (auto i = size_t{0u}; i < capture_count; i++) {
                    auto *field_ptr = _builder.CreateStructGEP(
                        capture_type, capture_storage,
                        static_cast<unsigned>(i));
                    _builder.CreateStore(
                        handler_arguments[i + 4u], field_ptr);
                }
                capture_pointer = capture_storage;
            }
            auto flags =
                !handler_pair.procedural_handler_empty ?
                    simd_host_ray_query_live_flag_has_procedural :
                    0u;
            auto *fn_type = ::llvm::FunctionType::get(
                _builder.getVoidTy(),
                {_builder.getInt32Ty(), _builder.getInt64Ty(),
                 pointer_type, pointer_type, pointer_type,
                 pointer_type, _builder.getInt32Ty()},
                false);
            _builder.CreateCall(
                fn_type, fn_ptr,
                {_builder.getInt32(_width), outer_active_bits, scratch,
                 capture_pointer, _launch_config,
                 handler_pair.on_candidate_live,
                 _builder.getInt32(flags)});
            _builder.CreateBr(exit);
        }
        _builder.SetInsertPoint(after_live);
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
                     luisa::string_view name) noexcept {
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
