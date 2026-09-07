#include "llvm_schedule_emitter.h"

#include "../../common/env_flag.h"

namespace luisa::compute::simd::detail {

void ScheduleEmitter::_emit_terminator(
    const schedule::BasicBlock &block,
    bool allow_all_on_region_versioning) {
    if (auto region = _find_chained_predicated_region(block)) {
        auto *split = std::get_if<schedule::SplitTerminator>(
            &block.terminator);
        _emit_chained_predicated_region(*split, *region);
        return;
    }
    if (auto region = _find_nested_predicated_region(block)) {
        auto *split = std::get_if<schedule::SplitTerminator>(
            &block.terminator);
        _emit_nested_predicated_region(*split, *region);
        return;
    }
    if (auto diamond =
            _find_guarded_predicated_math_diamond(block)) {
        auto *split = std::get_if<schedule::SplitTerminator>(
            &block.terminator);
        _emit_guarded_predicated_math_diamond(*split, *diamond);
        return;
    }
    if (auto diamond = _find_predicated_memory_diamond(block)) {
        auto *split = std::get_if<schedule::SplitTerminator>(
            &block.terminator);
        _emit_predicated_memory_diamond(
            block, *split, *diamond, nullptr);
        return;
    }
    auto reuse_coherent_mask =
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_COHERENT_MASK_REUSE");
    static constexpr auto kDirectDivergentChildMinStateSlots = 32u;
    auto enable_direct_divergent_child =
        _width >= 4u &&
        (_result.state_slot_count >=
             kDirectDivergentChildMinStateSlots ||
         luisa::compute::detail::env_flag(
             "LUISA_SIMD_FORCE_DIRECT_DIVERGENT_CHILD")) &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_DIRECT_DIVERGENT_CHILD");
    std::visit(
        [&](const auto &control) {
            using T = std::decay_t<decltype(control)>;
            if constexpr (std::is_same_v<T, schedule::BranchTerminator>) {
                _emit_arrival(control.edge, _active_mask);
            } else if constexpr (
                std::is_same_v<T, schedule::SplitTerminator>) {
                auto *condition_value = _source.value(control.condition);
                auto *condition = _load_value(control.condition);
                if (condition == nullptr || condition_value == nullptr) {
                    return;
                }
                if (condition_value->value_class ==
                    schedule::ValueClass::varying) {
                    if (control.cohort_uniform_condition) {
                        _result.cohort_uniform_loop_branch_count++;
                        auto *safe_condition = _builder.CreateSelect(
                            _active_mask, condition, _zero_mask());
                        auto *take_true =
                            _builder.CreateOrReduce(safe_condition);
                        auto *true_path = ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "cohort.uniform.true", _entry);
                        auto *false_path = ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "cohort.uniform.false", _entry);
                        _builder.CreateCondBr(
                            take_true, true_path, false_path);
                        _builder.SetInsertPoint(true_path);
                        _emit_arrival(control.true_edge, _active_mask);
                        _builder.SetInsertPoint(false_path);
                        _emit_arrival(control.false_edge, _active_mask);
                        return;
                    }
                    auto true_region = allow_all_on_region_versioning ?
                                           _find_coherent_all_on_region(
                                               control, control.true_edge) :
                                           std::nullopt;
                    auto false_region = allow_all_on_region_versioning ?
                                            _find_coherent_all_on_region(
                                                control,
                                                control.false_edge) :
                                            std::nullopt;
                    // Clone at most one arm per split. Prefer the lower-cost
                    // arm and break ties toward false, which is the canonical
                    // miss/skip shape produced by the DSL frontend.
                    if (true_region && false_region) {
                        if (true_region->weighted_cost <
                            false_region->weighted_cost) {
                            false_region.reset();
                        } else {
                            true_region.reset();
                        }
                    }
                    if (reuse_coherent_mask) {
                        _result.coherent_mask_reuse_count += 2u;
                    }
                    auto *true_mask = _builder.CreateAnd(
                        _active_mask, condition);
                    auto *false_mask = _builder.CreateAnd(
                        _active_mask, _builder.CreateNot(condition));
                    auto *true_nonempty =
                        _builder.CreateOrReduce(true_mask);
                    auto *false_nonempty =
                        _builder.CreateOrReduce(false_mask);
                    auto *divergent = _builder.CreateAnd(
                        true_nonempty, false_nonempty);
                    auto *divergent_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.divergent", _entry);
                    auto *coherent_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.coherent", _entry);
                    _builder.CreateCondBr(
                        divergent, divergent_path,
                        coherent_path);

                    _builder.SetInsertPoint(divergent_path);
                    if (control.convergence) {
                        _declare_convergence(
                            *control.convergence,
                            _builder.getTrue());
                    }
                    auto *true_flow = _route_edge(
                        control.true_edge, true_mask);
                    auto *false_flow = _route_edge(
                        control.false_edge, false_mask);
                    if (true_flow == nullptr ||
                        false_flow == nullptr) {
                        return;
                    }
                    _resume(control.true_edge.target, true_flow);
                    if (enable_direct_divergent_child) {
                        _result.direct_divergent_child_count++;
                        // The divergent predicate proves false_flow nonempty.
                        // It was the last record pushed and therefore the
                        // first one immediately popped by the LIFO scheduler.
                        // Keep the same token/runnable state and enter it
                        // through the shared PC route while leaving true_flow
                        // at the stack top.
                        auto *predecessor = _builder.GetInsertBlock();
                        _builder.CreateStore(false_flow, _current_mask);
                        _builder.CreateBr(_scheduler_dispatch_route);
                        _scheduler_dispatch_pc->addIncoming(
                            _builder.getInt32(
                                control.false_edge.target.value),
                            predecessor);
                    } else {
                        _resume(control.false_edge.target, false_flow);
                        _builder.CreateBr(_scheduler_loop);
                    }

                    _builder.SetInsertPoint(coherent_path);
                    auto *true_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.coherent.true", _entry);
                    auto *false_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.coherent.false", _entry);
                    _builder.CreateCondBr(
                        true_nonempty, true_path, false_path);
                    auto *outer_mask = _active_mask;
                    auto *outer_seed = _seed_lane;
                    auto outer_locals = _locals;
                    auto emit_coherent_edge =
                        [&](const schedule::ControlEdge &edge,
                            ::llvm::Value *derived_mask,
                            const std::optional<CoherentAllOnRegion> &region) {
                            _active_mask = outer_mask;
                            _seed_lane = outer_seed;
                            _locals = outer_locals;
                            auto *flow_mask = reuse_coherent_mask ?
                                                  outer_mask :
                                                  derived_mask;
                            if (!region) {
                                _emit_arrival(edge, flow_mask);
                                return;
                            }
                            auto *version = ::llvm::BasicBlock::Create(
                                _module.getContext(),
                                "all.on.region.version", _entry);
                            auto *fallback = ::llvm::BasicBlock::Create(
                                _module.getContext(),
                                "all.on.region.fallback", _entry);
                            auto *all_on =
                                _builder.CreateAndReduce(outer_mask);
                            _builder.CreateCondBr(
                                all_on, version, fallback);

                            _builder.SetInsertPoint(fallback);
                            _active_mask = outer_mask;
                            _seed_lane = outer_seed;
                            _locals = outer_locals;
                            _emit_arrival(edge, flow_mask);

                            _builder.SetInsertPoint(version);
                            _active_mask = outer_mask;
                            _seed_lane = outer_seed;
                            _locals = outer_locals;
                            _emit_coherent_all_on_region(edge, *region);
                        };
                    _builder.SetInsertPoint(true_path);
                    emit_coherent_edge(
                        control.true_edge, true_mask, true_region);
                    _builder.SetInsertPoint(false_path);
                    emit_coherent_edge(
                        control.false_edge, false_mask, false_region);
                    _active_mask = outer_mask;
                    _seed_lane = outer_seed;
                    _locals = std::move(outer_locals);
                } else {
                    auto *true_path = ::llvm::BasicBlock::Create(
                        _module.getContext(), "uniform.true", _entry);
                    auto *false_path = ::llvm::BasicBlock::Create(
                        _module.getContext(), "uniform.false", _entry);
                    _builder.CreateCondBr(
                        condition, true_path, false_path);
                    _builder.SetInsertPoint(true_path);
                    _emit_arrival(control.true_edge, _active_mask);
                    _builder.SetInsertPoint(false_path);
                    _emit_arrival(control.false_edge, _active_mask);
                }
            } else if constexpr (
                std::is_same_v<T, schedule::SwitchTerminator>) {
                auto *selector_value = _source.value(control.selector);
                auto *selector = _load_value(control.selector);
                if (selector == nullptr || selector_value == nullptr) {
                    return;
                }
                if (selector_value->value_class ==
                    schedule::ValueClass::varying) {
                    if (reuse_coherent_mask) {
                        _result.coherent_mask_reuse_count +=
                            control.cases.size() + 1u;
                    }
                    auto *selector_type = ::llvm::cast<::llvm::VectorType>(
                        selector->getType());
                    auto *element_type = ::llvm::cast<::llvm::IntegerType>(
                        selector_type->getElementType());
                    std::vector<::llvm::Value *> case_masks;
                    case_masks.reserve(control.cases.size());
                    auto *remaining = _active_mask;
                    ::llvm::Value *has_previous_path = _builder.getFalse();
                    ::llvm::Value *divergent = _builder.getFalse();
                    auto record_path = [&](::llvm::Value *mask) noexcept {
                        auto *nonempty = _builder.CreateOrReduce(mask);
                        divergent = _builder.CreateOr(
                            divergent,
                            _builder.CreateAnd(
                                has_previous_path, nonempty));
                        has_previous_path = _builder.CreateOr(
                            has_previous_path, nonempty);
                    };
                    for (auto &&item : control.cases) {
                        auto *case_value = ::llvm::ConstantInt::get(
                            element_type, item.value);
                        auto *matches = _builder.CreateICmpEQ(
                            selector,
                            _builder.CreateVectorSplat(
                                _width, case_value));
                        auto *case_mask = _builder.CreateAnd(
                            remaining, matches);
                        case_masks.emplace_back(case_mask);
                        record_path(case_mask);
                        remaining = _builder.CreateAnd(
                            remaining, _builder.CreateNot(matches));
                    }
                    auto *default_mask = remaining;
                    record_path(default_mask);
                    auto *divergent_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.switch.divergent", _entry);
                    auto *coherent_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.switch.coherent", _entry);
                    _builder.CreateCondBr(
                        divergent, divergent_path,
                        coherent_path);

                    _builder.SetInsertPoint(divergent_path);
                    if (control.convergence) {
                        _declare_convergence(
                            *control.convergence,
                            _builder.getTrue());
                    }
                    for (auto i = size_t{0u};
                         i < control.cases.size(); i++) {
                        auto *flow = _route_edge(
                            control.cases[i].edge, case_masks[i]);
                        if (flow == nullptr) { return; }
                        _resume(
                            control.cases[i].edge.target, flow);
                    }
                    auto *default_flow = _route_edge(
                        control.default_edge, default_mask);
                    if (default_flow == nullptr) { return; }
                    _resume(
                        control.default_edge.target, default_flow);
                    _builder.CreateBr(_scheduler_loop);

                    _builder.SetInsertPoint(coherent_path);
                    auto *default_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.switch.coherent.default", _entry);
                    auto *seed_selector =
                        _builder.CreateExtractElement(
                            selector, _seed_lane);
                    auto *llvm_switch = _builder.CreateSwitch(
                        seed_selector, default_path,
                        control.cases.size());
                    std::vector<::llvm::BasicBlock *> case_paths;
                    case_paths.reserve(control.cases.size());
                    for (auto &&item : control.cases) {
                        auto *case_path =
                            ::llvm::BasicBlock::Create(
                                _module.getContext(),
                                "varying.switch.coherent.case", _entry);
                        case_paths.emplace_back(case_path);
                        llvm_switch->addCase(
                            ::llvm::ConstantInt::get(
                                element_type, item.value),
                            case_path);
                    }
                    for (auto i = size_t{0u};
                         i < control.cases.size(); i++) {
                        _builder.SetInsertPoint(case_paths[i]);
                        _emit_arrival(
                            control.cases[i].edge,
                            reuse_coherent_mask ?
                                _active_mask :
                                case_masks[i]);
                    }
                    _builder.SetInsertPoint(default_path);
                    _emit_arrival(
                        control.default_edge,
                        reuse_coherent_mask ?
                            _active_mask :
                            default_mask);
                } else {
                    auto *default_path = ::llvm::BasicBlock::Create(
                        _module.getContext(),
                        "uniform.switch.default", _entry);
                    std::vector<::llvm::BasicBlock *> case_paths;
                    case_paths.reserve(control.cases.size());
                    auto *llvm_switch = _builder.CreateSwitch(
                        selector, default_path, control.cases.size());
                    auto *selector_type = ::llvm::cast<
                        ::llvm::IntegerType>(selector->getType());
                    for (auto &&item : control.cases) {
                        auto *case_path = ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "uniform.switch.case", _entry);
                        case_paths.emplace_back(case_path);
                        llvm_switch->addCase(
                            ::llvm::ConstantInt::get(
                                selector_type, item.value),
                            case_path);
                    }
                    for (auto i = size_t{0u};
                         i < control.cases.size(); i++) {
                        _builder.SetInsertPoint(case_paths[i]);
                        _emit_arrival(
                            control.cases[i].edge, _active_mask);
                    }
                    _builder.SetInsertPoint(default_path);
                    _emit_arrival(
                        control.default_edge, _active_mask);
                }
            } else if constexpr (
                std::is_same_v<T, schedule::JoinTerminator>) {
                auto *point = _source.convergence(control.convergence);
                schedule::ControlEdge edge{point->target};
                edge.joins.emplace_back(control.convergence);
                edge.assignments = control.assignments;
                _emit_arrival(edge, _active_mask);
            } else if constexpr (
                std::is_same_v<
                    T, schedule::BlockBarrierTerminator>) {
                _emit_block_barrier(control);
            } else if constexpr (
                std::is_same_v<T, schedule::ReturnTerminator>) {
                if (control.value) {
                    auto *schedule_value = _source.value(*control.value);
                    auto *value = _as_lane_vector(
                        _load_value(*control.value), *schedule_value);
                    if (value != nullptr) {
                        _builder.CreateMaskedStore(
                            value, _return_buffer,
                            ::llvm::Align{schedule_value->type->alignment()},
                            _active_mask);
                    }
                }
                auto *live = _builder.CreateLoad(
                    _live_mask->getAllocatedType(), _live_mask);
                auto *new_live = _builder.CreateAnd(
                    live, _builder.CreateNot(_active_mask));
                _builder.CreateStore(new_live, _live_mask);
                auto *runnable = _builder.CreateLoad(
                    _runnable_mask->getAllocatedType(),
                    _runnable_mask);
                _builder.CreateStore(
                    _builder.CreateAnd(
                        runnable, _builder.CreateNot(_active_mask)),
                    _runnable_mask);

                auto guard_inactive_frames =
                    !luisa::compute::detail::env_flag(
                        "LUISA_SIMD_DISABLE_RETURN_FRAME_GUARD");
                ::llvm::BasicBlock *frame_cleanup_done = nullptr;
                if (guard_inactive_frames) {
                    _result.return_frame_guard_count++;
                    auto *frame_cleanup = ::llvm::BasicBlock::Create(
                        _module.getContext(),
                        "return.frame.cleanup", _entry);
                    frame_cleanup_done = ::llvm::BasicBlock::Create(
                        _module.getContext(),
                        "return.frame.cleanup.done", _entry);
                    auto *active_frames = _builder.CreateLoad(
                        _frame_active->getAllocatedType(),
                        _frame_active, "return.active.frames");
                    _builder.CreateCondBr(
                        _builder.CreateICmpNE(
                            active_frames,
                            ::llvm::Constant::getNullValue(
                                active_frames->getType()),
                            "return.frames.present"),
                        frame_cleanup, frame_cleanup_done);
                    _builder.SetInsertPoint(frame_cleanup);
                }

                // A lane that terminates is removed from every frame's
                // expected mask. Frame storage is bounded by W rather
                // than the number of static CFG convergence points.
                for (auto frame = uint32_t{0u}; frame < _width; frame++) {
                    auto *index = _builder.getInt32(frame);
                    auto *active_frames = _builder.CreateLoad(
                        _frame_active->getAllocatedType(),
                        _frame_active);
                    auto *frame_active = _frame_is_active(
                        active_frames, index);
                    auto *expected_ptr = _frame_mask_pointer(
                        _frame_expected, index);
                    auto *arrived_ptr = _frame_mask_pointer(
                        _frame_arrived, index);
                    auto *expected = _builder.CreateLoad(
                        _layout.mask_type(), expected_ptr);
                    auto *arrived = _builder.CreateLoad(
                        _layout.mask_type(), arrived_ptr);
                    auto *expected_live = _builder.CreateAnd(
                        expected, new_live);
                    auto *complete = _builder.CreateAnd(
                        frame_active,
                        _builder.CreateAndReduce(
                            _builder.CreateICmpEQ(
                                arrived, expected_live)));
                    auto *released = _builder.CreateSelect(
                        _splat(complete), arrived, _zero_mask());
                    _builder.CreateStore(
                        _builder.CreateSelect(
                            complete, _zero_mask(), expected_live),
                        expected_ptr);
                    _builder.CreateStore(
                        _builder.CreateSelect(
                            complete, _zero_mask(), arrived),
                        arrived_ptr);
                    auto *released_frames = _builder.CreateAnd(
                        active_frames,
                        _builder.CreateNot(_frame_bit(index)));
                    _builder.CreateStore(
                        _builder.CreateSelect(
                            complete, released_frames, active_frames),
                        _frame_active);

                    auto *parent = _load_frame_metadata(
                        _frame_parent_token, index);
                    auto *static_id = _load_frame_metadata(
                        _frame_static_id, index);
                    if (_convergence_targets != nullptr) {
                        auto *target = _load_convergence_target(static_id);
                        // The target entry performs the same dynamic cascade
                        // as an ordinary CFG arrival. Deferring it avoids one
                        // copy of the full frame logic per return-site frame.
                        _resume(target, released, parent);
                    }
                }
                if (guard_inactive_frames) {
                    _builder.CreateBr(frame_cleanup_done);
                    _builder.SetInsertPoint(frame_cleanup_done);
                }
                _builder.CreateBr(_scheduler_loop);
            } else if constexpr (
                std::is_same_v<T, schedule::UnreachableTerminator>) {
                _builder.CreateUnreachable();
            } else {
                _fail("unsupported Schedule IR terminator reached LLVM emission");
            }
        },
        block.terminator);
}

void ScheduleEmitter::_emit_direct_terminator(
    const schedule::BasicBlock &block,
    const std::vector<::llvm::BasicBlock *> &blocks) {
    if (auto diamond = _find_predicated_memory_diamond(block)) {
        auto *split = std::get_if<schedule::SplitTerminator>(
            &block.terminator);
        _emit_predicated_memory_diamond(
            block, *split, *diamond, &blocks);
        return;
    }
    auto scalar = [&](::llvm::Value *value) -> ::llvm::Value * {
        if (value == nullptr) { return nullptr; }
        return value->getType()->isVectorTy() ?
                   _builder.CreateExtractElement(value, uint64_t{0u}) :
                   value;
    };
    auto emit_edge = [&](const schedule::ControlEdge &edge) {
        _apply_assignments(edge.assignments, _active_mask);
        if (!_failed()) {
            _builder.CreateBr(blocks[edge.target.value]);
        }
    };
    std::visit(
        [&](const auto &control) {
            using T = std::decay_t<decltype(control)>;
            if constexpr (std::is_same_v<
                              T, schedule::BranchTerminator>) {
                emit_edge(control.edge);
            } else if constexpr (std::is_same_v<
                                     T, schedule::SplitTerminator>) {
                auto *condition = scalar(
                    _load_value(control.condition));
                if (condition == nullptr) { return; }
                auto *true_path = ::llvm::BasicBlock::Create(
                    _module.getContext(), "direct.true", _entry);
                auto *false_path = ::llvm::BasicBlock::Create(
                    _module.getContext(), "direct.false", _entry);
                _builder.CreateCondBr(
                    condition, true_path, false_path);
                _builder.SetInsertPoint(true_path);
                emit_edge(control.true_edge);
                _builder.SetInsertPoint(false_path);
                emit_edge(control.false_edge);
            } else if constexpr (std::is_same_v<
                                     T, schedule::SwitchTerminator>) {
                auto *selector = scalar(
                    _load_value(control.selector));
                if (selector == nullptr) { return; }
                auto *default_path = ::llvm::BasicBlock::Create(
                    _module.getContext(),
                    "direct.switch.default", _entry);
                std::vector<::llvm::BasicBlock *> case_paths;
                case_paths.reserve(control.cases.size());
                auto *llvm_switch = _builder.CreateSwitch(
                    selector, default_path, control.cases.size());
                auto *selector_type = ::llvm::cast<
                    ::llvm::IntegerType>(selector->getType());
                for (auto &&item : control.cases) {
                    auto *case_path = ::llvm::BasicBlock::Create(
                        _module.getContext(),
                        "direct.switch.case", _entry);
                    case_paths.emplace_back(case_path);
                    llvm_switch->addCase(
                        ::llvm::ConstantInt::get(
                            selector_type, item.value),
                        case_path);
                }
                for (auto i = size_t{0u};
                     i < control.cases.size(); i++) {
                    _builder.SetInsertPoint(case_paths[i]);
                    emit_edge(control.cases[i].edge);
                }
                _builder.SetInsertPoint(default_path);
                emit_edge(control.default_edge);
            } else if constexpr (std::is_same_v<
                                     T, schedule::JoinTerminator>) {
                auto *point = _source.convergence(
                    control.convergence);
                schedule::ControlEdge edge{point->target};
                edge.assignments = control.assignments;
                emit_edge(edge);
            } else if constexpr (std::is_same_v<
                                     T, schedule::ReturnTerminator>) {
                if (control.value) {
                    auto *schedule_value = _source.value(
                        *control.value);
                    auto *value = _as_lane_vector(
                        _load_value(*control.value),
                        *schedule_value);
                    if (value != nullptr) {
                        _builder.CreateMaskedStore(
                            value, _return_buffer,
                            ::llvm::Align{
                                schedule_value->type->alignment()},
                            _active_mask);
                    }
                }
                _builder.CreateRetVoid();
            } else if constexpr (std::is_same_v<
                                     T,
                                     schedule::UnreachableTerminator>) {
                _builder.CreateUnreachable();
            } else {
                _fail("unsupported Schedule IR terminator reached direct LLVM emission");
            }
        },
        block.terminator);
}

}// namespace luisa::compute::simd::detail
