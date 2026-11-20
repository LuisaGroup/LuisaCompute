#include "llvm_schedule_emitter.h"

#include <algorithm>
#include <unordered_set>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Support/ModRef.h>

#include "../../common/env_flag.h"
#include <luisa/core/stl/string.h>

namespace luisa::compute::simd::detail {

void ScheduleEmitter::_find_instruction_spills() {
    _spilled_instruction_values.assign(
        _source.values().size(), uint8_t{0u});
    std::vector<schedule::BlockId> emission_blocks;
    emission_blocks.reserve(_source.blocks().size());
    for (auto &&block : _source.blocks()) {
        emission_blocks.emplace_back(block.id);
    }
    std::vector<uint8_t> chained_blocks(
        _source.blocks().size(), uint8_t{0u});
    auto in_structured_loop = [&](schedule::BlockId id) noexcept {
        return _structured_early_exit_loop &&
               std::find(
                   _structured_early_exit_loop->loop->blocks.cbegin(),
                   _structured_early_exit_loop->loop->blocks.cend(), id) !=
                   _structured_early_exit_loop->loop->blocks.cend();
    };
    if (!_predicated_acyclic_control_flow) {
        for (auto &&block : _source.blocks()) {
            if (chained_blocks[block.id.value] != 0u) { continue; }
            if (auto region =
                    _find_chained_predicated_region(block)) {
                chained_blocks[block.id.value] = 1u;
                for (auto *inlined : region->inlined_blocks) {
                    emission_blocks[inlined->id.value] = block.id;
                    chained_blocks[inlined->id.value] = 1u;
                }
            }
        }
        for (auto &&block : _source.blocks()) {
            if (chained_blocks[block.id.value] != 0u) { continue; }
            if (auto diamond =
                    _find_guarded_predicated_math_diamond(
                        block, in_structured_loop(block.id))) {
                for (auto *arm : diamond->true_blocks) {
                    emission_blocks[arm->id.value] = block.id;
                }
                for (auto *arm : diamond->false_blocks) {
                    emission_blocks[arm->id.value] = block.id;
                }
            }
        }
        for (auto &&block : _source.blocks()) {
            if (chained_blocks[block.id.value] != 0u) { continue; }
            if (auto region = _find_nested_predicated_region(
                    block, in_structured_loop(block.id))) {
                emission_blocks[region->nested_split_block->id.value] =
                    block.id;
                for (auto *arm : region->nested_diamond.true_blocks) {
                    emission_blocks[arm->id.value] = block.id;
                }
                for (auto *arm : region->nested_diamond.false_blocks) {
                    emission_blocks[arm->id.value] = block.id;
                }
                emission_blocks[region->nested_merge_block->id.value] =
                    block.id;
                emission_blocks[region->other_block->id.value] = block.id;
            }
        }
    }
    if (_direct_control_flow) {
        for (auto &&block : _source.blocks()) {
            if (auto diamond =
                    _find_predicated_memory_diamond(block)) {
                if (diamond->true_block) { emission_blocks[diamond->true_block->id.value] = block.id; }
                if (diamond->false_block) { emission_blocks[diamond->false_block->id.value] = block.id; }
            }
        }
    }
    std::vector<uint8_t> is_alloca_result(_source.values().size(), 0u);
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode == schedule::Opcode::alloca &&
                instruction.result &&
                instruction.result->value < is_alloca_result.size()) {
                is_alloca_result[instruction.result->value] = 1u;
            }
        }
    }
    auto mark = [&](schedule::ValueId id,
                    schedule::BlockId use_block) noexcept {
        if (id.value < is_alloca_result.size() &&
            is_alloca_result[id.value] != 0u) {
            return;
        }
        auto *value = _source.value(id);
        if (value != nullptr &&
            value->origin == schedule::ValueOrigin::instruction &&
            value->defining_block &&
            emission_blocks[value->defining_block->value] !=
                emission_blocks[use_block.value]) {
            _spilled_instruction_values[id.value] = 1u;
        }
    };
    for (auto &&block : _source.blocks()) {
        // Predicated direct-control diamonds emit both arms in the split
        // block. Compare definition and use after that emission-block remap:
        // arm-local/outer values remain LLVM SSA, while a value genuinely
        // produced by an earlier emitted block still receives a spill slot.
        for (auto &&instruction : block.instructions) {
            for (auto operand : instruction.operands) {
                mark(operand, block.id);
            }
            if (instruction.participant_mask) {
                mark(*instruction.participant_mask, block.id);
            }
        }
        auto mark_assignments = [&](const auto &assignments) noexcept {
            for (auto assignment : assignments) {
                mark(assignment.source, block.id);
            }
        };
        auto mark_edge = [&](const schedule::ControlEdge &edge) noexcept {
            mark_assignments(edge.assignments);
        };
        std::visit(
            [&](const auto &terminator) noexcept {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    mark_edge(terminator.edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SplitTerminator>) {
                    mark(terminator.condition, block.id);
                    mark_edge(terminator.true_edge);
                    mark_edge(terminator.false_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SwitchTerminator>) {
                    mark(terminator.selector, block.id);
                    for (auto &&item : terminator.cases) {
                        mark_edge(item.edge);
                    }
                    mark_edge(terminator.default_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator> ||
                                     std::is_same_v<
                                         T, schedule::LoopBackTerminator>) {
                    mark_assignments(terminator.assignments);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::BlockBarrierTerminator>) {
                    mark_edge(terminator.resume_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::ReturnTerminator>) {
                    if (terminator.value) {
                        mark(*terminator.value, block.id);
                    }
                }
            },
            block.terminator);
    }
}

void ScheduleEmitter::_allocate_state() {
    if (!_direct_control_flow &&
        !_predicated_acyclic_control_flow) {
        for (auto &&block : _source.blocks()) {
            if (auto candidate =
                    _find_structured_early_exit_loop(block)) {
                _structured_early_exit_loop =
                    std::move(*candidate);
                break;
            }
        }
    }
    if (!_direct_control_flow &&
        !_predicated_acyclic_control_flow) {
        auto *mask_type = _layout.mask_type();
        auto *zero_mask = ::llvm::Constant::getNullValue(mask_type);
        _outline_scheduler_helpers =
            !_cooperative_block && _width >= 4u &&
            _source.blocks().size() >= 32u &&
            !luisa::compute::detail::env_flag(
                "LUISA_SIMD_DISABLE_OUTLINED_SCHEDULER_HELPERS");
        // Dynamic whole-vector frame updates spill to temporary stack slots
        // and emit vpcmpeqd/vpbroadcastd ladders on W8+ large schedules.
        // Keep scalar array metadata on W8+ (and whenever scheduler helpers
        // are outlined) while keeping the width policy A/B-testable.
        _use_scalar_frame_metadata =
            _outline_scheduler_helpers ||
            (_width >= 8u &&
             !luisa::compute::detail::env_flag(
                 "LUISA_SIMD_DISABLE_SCALAR_FRAME_METADATA"));
        _result.scalar_frame_metadata =
            _use_scalar_frame_metadata;
        auto *i32_lanes = _use_scalar_frame_metadata ?
                              static_cast<::llvm::Type *>(
                                  ::llvm::ArrayType::get(
                                      _builder.getInt32Ty(), _width)) :
                              static_cast<::llvm::Type *>(
                                  ::llvm::FixedVectorType::get(
                                      _builder.getInt32Ty(), _width));
        auto *zero_i32_lanes = ::llvm::Constant::getNullValue(i32_lanes);
        auto *frame_active_type = ::llvm::IntegerType::get(
            _module.getContext(), _width);
        auto *ready_masks = ::llvm::ArrayType::get(mask_type, _width);
        auto *ready_i32 = ::llvm::ArrayType::get(_builder.getInt32Ty(), _width);
        _frame_active_type = frame_active_type;
        _frame_metadata_type = i32_lanes;
        _ready_masks_type = ready_masks;
        _ready_i32_type = ready_i32;

        if (_outline_scheduler_helpers) {
            _sched_state_type = _get_or_create_sched_state_type();
            _sched_state = _builder.CreateAlloca(
                _sched_state_type, nullptr, "sched.state");
            _sched_state->setAlignment(::llvm::Align{64u});
            _builder.CreateStore(
                ::llvm::Constant::getNullValue(_sched_state_type),
                _sched_state);
            _live_mask = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 0u, "live.mask");
            _runnable_mask = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 1u, "runnable.mask");
            _ready_count = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 2u, "ready.count");
            _ready_masks = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 3u, "ready.mask");
            _ready_targets = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 4u, "ready.target");
            _ready_tokens = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 5u, "ready.token");
            _current_mask = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 6u, "current.mask");
            _current_token = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 7u, "current.token");
            _frame_active = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 8u, "frame.active");
            _frame_static_id = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 9u, "frame.static.id");
            _frame_parent_token = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 10u, "frame.parent.token");
            _frame_expected = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 11u, "frame.expected");
            _frame_arrived = _builder.CreateStructGEP(
                _sched_state_type, _sched_state, 12u, "frame.arrived");
        } else {
            _live_mask = _builder.CreateAlloca(
                mask_type, nullptr, "live.mask");
            _runnable_mask = _builder.CreateAlloca(
                mask_type, nullptr, "runnable.mask");
            _ready_count = _builder.CreateAlloca(
                _builder.getInt32Ty(), nullptr, "ready.count");
            _ready_masks = _builder.CreateAlloca(
                ready_masks, nullptr, "ready.mask");
            _ready_targets = _builder.CreateAlloca(
                ready_i32, nullptr, "ready.target");
            _ready_tokens = _builder.CreateAlloca(
                ready_i32, nullptr, "ready.token");
            _current_mask = _builder.CreateAlloca(
                mask_type, nullptr, "current.mask");
            _current_token = _builder.CreateAlloca(
                _builder.getInt32Ty(), nullptr, "current.token");
            _frame_active = _builder.CreateAlloca(
                frame_active_type, nullptr, "frame.active");
            _frame_static_id = _builder.CreateAlloca(
                i32_lanes, nullptr, "frame.static.id");
            _frame_parent_token = _builder.CreateAlloca(
                i32_lanes, nullptr, "frame.parent.token");
            _frame_expected = _builder.CreateAlloca(
                ready_masks, nullptr, "frame.expected");
            _frame_arrived = _builder.CreateAlloca(
                ready_masks, nullptr, "frame.arrived");
            _builder.CreateStore(zero_mask, _live_mask);
            _builder.CreateStore(zero_mask, _runnable_mask);
            _builder.CreateStore(
                _builder.getInt32(0u), _ready_count);
            _builder.CreateStore(
                ::llvm::Constant::getNullValue(ready_masks),
                _ready_masks);
            _builder.CreateStore(
                ::llvm::Constant::getNullValue(ready_i32),
                _ready_targets);
            _builder.CreateStore(
                ::llvm::Constant::getNullValue(ready_i32),
                _ready_tokens);
            _builder.CreateStore(zero_mask, _current_mask);
            _builder.CreateStore(
                _builder.getInt32(0u), _current_token);
            _builder.CreateStore(
                ::llvm::Constant::getNullValue(frame_active_type),
                _frame_active);
            _builder.CreateStore(zero_i32_lanes, _frame_static_id);
            _builder.CreateStore(zero_i32_lanes, _frame_parent_token);
            _builder.CreateStore(
                ::llvm::Constant::getNullValue(ready_masks),
                _frame_expected);
            _builder.CreateStore(
                ::llvm::Constant::getNullValue(ready_masks),
                _frame_arrived);
        }

        auto *loop_epoch_type = ::llvm::FixedVectorType::get(
            _builder.getInt64Ty(), _width);
        _cooperative_loop_epochs.reserve(
            _result.block_barrier_loop_epoch_count);
        for (auto index = size_t{0u};
             index < _result.block_barrier_loop_epoch_count; index++) {
            auto *epoch = _builder.CreateAlloca(
                loop_epoch_type, nullptr,
                "cooperative.loop.epoch." + std::to_string(index));
            _builder.CreateStore(
                ::llvm::Constant::getNullValue(loop_epoch_type), epoch);
            _cooperative_loop_epochs.emplace_back(epoch);
        }

        std::vector<::llvm::Constant *> convergence_targets;
        convergence_targets.reserve(
            _source.convergence_points().size());
        for (auto &&point : _source.convergence_points()) {
            convergence_targets.emplace_back(
                _builder.getInt32(point.target.value));
        }
        if (!convergence_targets.empty() && _width >= 4u) {
            auto *array = ::llvm::ArrayType::get(
                _builder.getInt32Ty(), convergence_targets.size());
            auto *initializer = ::llvm::ConstantArray::get(
                array, convergence_targets);
            auto *global = new ::llvm::GlobalVariable(
                _module, array, true,
                ::llvm::GlobalValue::PrivateLinkage, initializer,
                "convergence.targets");
            global->setUnnamedAddr(
                ::llvm::GlobalValue::UnnamedAddr::Global);
            global->setAlignment(::llvm::Align{4u});
            _convergence_targets = global;
        } else if (!convergence_targets.empty()) {
            _convergence_targets =
                ::llvm::ConstantVector::get(convergence_targets);
        }
        _target_convergence_depths.assign(
            _source.blocks().size(), 0u);
        for (auto &&point : _source.convergence_points()) {
            ++_target_convergence_depths[point.target.value];
        }
    }
    _find_instruction_spills();
    _local_allocations.resize(_source.values().size(), nullptr);
    _shared_memory_size = 0u;
    // A private Tile is replicated per packet lane. Bound this physical
    // allocation before choosing stack versus Runtime-owned workspace, not
    // merely the size of one logical worker's array. No lifetime coalescing
    // is assumed: every allocation receives a distinct aligned interval.
    auto private_bytes = size_t{0u};
    if (_private_stack_budget_bytes != 0u) {
        for (auto &&block : _source.blocks()) {
            for (auto &&instruction : block.instructions) {
                if (instruction.opcode != schedule::Opcode::alloca ||
                    !instruction.result || _is_shared_lvalue(*instruction.result)) { continue; }
                auto *value = _source.value(*instruction.result);
                if (value == nullptr || value->type == nullptr) {
                    _fail("private workspace allocation has an invalid type");
                    return;
                }
                auto size = _abi_size(value->type);
                auto alignment = _abi_alignment(value->type);
                if (_local_soa_kind_of(*instruction.result) == LocalSoAKind::uniform) {
                    auto cell = _local_soa_cell_of(*instruction.result);
                    alignment = std::max<size_t>(
                        alignment,
                        std::min<size_t>(
                            static_cast<size_t>(_width) * cell, size_t{64u}));
                }
                if (alignment > 64u || size > simd_max_private_workspace_bytes / _width) {
                    _fail("SIMD private workspace exceeds the runtime capacity/alignment limit");
                    return;
                }
                auto offset = _align_up(private_bytes, alignment);
                auto bytes = size * _width;
                if (offset > simd_max_private_workspace_bytes || bytes > simd_max_private_workspace_bytes - offset) {
                    _fail("SIMD private workspace exceeds the runtime capacity limit");
                    return;
                }
                private_bytes = offset + bytes;
            }
        }
        if (private_bytes > _private_stack_budget_bytes) {
            if (_cooperative_block || _is_handler_entry() || !_ray_query_pipeline_handlers.empty() || !_outlined_callables.empty()) {
                _fail("private workspace reuse requires independent non-cooperative packets without nested handlers");
                return;
            }
            _result.private_workspace_size = private_bytes;
        }
    }
    auto private_offset = size_t{0u};
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode != schedule::Opcode::alloca ||
                !instruction.result) {
                continue;
            }
            auto *value = _source.value(*instruction.result);
            auto value_size = value == nullptr ? 0u :
                                                 _abi_size(value->type);
            auto value_alignment = value == nullptr ? 1u :
                                                      _abi_alignment(value->type);
            if (value == nullptr || value->type == nullptr ||
                !_is_local_lvalue(*instruction.result) ||
                value_size == 0u) {
                _fail("thread-local allocation has an invalid data type");
                return;
            }
            if (_is_shared_lvalue(*instruction.result)) {
                auto alignment = std::max<size_t>(
                    value_alignment, 16u);
                auto offset = _align_up(
                    _shared_memory_size, alignment);
                if (offset > simd_max_shared_memory_bytes ||
                    value_size >
                        simd_max_shared_memory_bytes - offset) {
                    _fail("SIMD cooperative shared memory exceeds the runtime limit");
                    return;
                }
                _shared_memory_size = offset + value_size;
                auto *shared_address = _byte_pointer(
                    _launch_config,
                    offsetof(SIMDPacketLaunchConfig, shared_memory));
                auto *shared = _builder.CreateLoad(
                    ::llvm::PointerType::getUnqual(
                        _module.getContext()),
                    shared_address, "shared.memory");
                auto *base = offset == 0u ?
                                 shared :
                                 _builder.CreateInBoundsPtrAdd(
                                     shared,
                                     _builder.getInt64(offset));
                auto *zero_offsets = ::llvm::Constant::getNullValue(
                    ::llvm::FixedVectorType::get(
                        _builder.getInt64Ty(), _width));
                _local_allocations[instruction.result->value] =
                    _local_handle(
                        _builder.CreateVectorSplat(_width, base),
                        zero_offsets);
            } else {
                auto soa_uniform =
                    _local_soa_kind_of(*instruction.result) ==
                    LocalSoAKind::uniform;
                auto alloc_lanes = soa_uniform ? _soa_stride() : _width;
                auto byte_count = static_cast<uint64_t>(alloc_lanes) *
                                  value_size;
                auto local_alignment = value_alignment;
                if (soa_uniform) {
                    auto cell = _local_soa_cell_of(*instruction.result);
                    local_alignment = std::max<size_t>(
                        value_alignment,
                        std::min<size_t>(
                            static_cast<size_t>(alloc_lanes) * cell, size_t{64u}));
                }
                ::llvm::Value *storage = nullptr;
                if (_result.private_workspace_size != 0u) {
                    private_offset = _align_up(private_offset, local_alignment);
                    auto *address = _byte_pointer(_launch_config, offsetof(SIMDPacketLaunchConfig, private_workspace));
                    auto *workspace = _builder.CreateLoad(::llvm::PointerType::getUnqual(_module.getContext()), address, "private.workspace");
                    storage = _builder.CreateInBoundsPtrAdd(workspace, _builder.getInt64(private_offset), value->name + ".private");
                    private_offset += byte_count;
                } else {
                    auto *storage_type = ::llvm::ArrayType::get(_builder.getInt8Ty(), byte_count);
                    auto *local = _builder.CreateAlloca(storage_type, nullptr, value->name + ".local");
                    local->setAlignment(::llvm::Align{local_alignment});
                    storage = local;
                }
                if (soa_uniform) {
                    _local_allocations[instruction.result->value] = storage;
                } else {
                    auto *offsets = _lane_offsets(_lane_ids(), value_size);
                    _local_allocations[instruction.result->value] =
                        _local_handle(
                            _builder.CreateVectorSplat(_width, storage),
                            offsets);
                }
            }
        }
    }
    _result.shared_memory_size = _shared_memory_size;
    for (auto slot = size_t{0u};
         slot < _ray_query_status_storage.size(); slot++) {
        auto *storage = _builder.CreateAlloca(
            _builder.getInt64Ty(), nullptr,
            "ray.query.status.slot." + std::to_string(slot));
        storage->setAlignment(::llvm::Align{alignof(uint64_t)});
        _builder.CreateStore(_builder.getInt64(0u), storage);
        _ray_query_status_storage[slot] = storage;
        auto *callback_type = ::llvm::FixedVectorType::get(
            ::llvm::PointerType::getUnqual(_module.getContext()), _width);
        auto *callback = _builder.CreateAlloca(
            callback_type, nullptr,
            "ray.query.status.callback.slot." + std::to_string(slot));
        callback->setAlignment(::llvm::Align{alignof(void *)});
        _builder.CreateAlignedStore(
            ::llvm::Constant::getNullValue(callback_type),
            callback, ::llvm::Align{alignof(void *)});
        _ray_query_status_callback_storage[slot] = callback;
        if (slot < _ray_query_live_dispatch_storage.size()) {
            auto *live_dispatch = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.live.dispatch.slot." + std::to_string(slot));
            live_dispatch->setAlignment(::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type),
                live_dispatch, ::llvm::Align{alignof(void *)});
            _ray_query_live_dispatch_storage[slot] = live_dispatch;
        }
        if (slot < _ray_query_pipeline_callback_storage.size()) {
            auto *pipeline_callback = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.pipeline.w1.callback.slot." +
                    std::to_string(slot));
            pipeline_callback->setAlignment(
                ::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type),
                pipeline_callback,
                ::llvm::Align{alignof(void *)});
            _ray_query_pipeline_callback_storage[slot] =
                pipeline_callback;
        }
        if (slot <
            _ray_query_surface_filter_pipeline_callback_storage.size()) {
            auto *pipeline_callback = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.surface.filter.pipeline.callback.slot." +
                    std::to_string(slot));
            pipeline_callback->setAlignment(
                ::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type),
                pipeline_callback,
                ::llvm::Align{alignof(void *)});
            _ray_query_surface_filter_pipeline_callback_storage[slot] =
                pipeline_callback;
        }
        if (slot <
            _ray_query_empty_surface_filter_pipeline_callback_storage.size()) {
            auto *pipeline_callback = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.empty.surface.filter.pipeline.callback.slot." +
                    std::to_string(slot));
            pipeline_callback->setAlignment(
                ::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type),
                pipeline_callback,
                ::llvm::Align{alignof(void *)});
            _ray_query_empty_surface_filter_pipeline_callback_storage[slot] =
                pipeline_callback;
        }
        if (slot < _ray_query_empty_surface_filter_accel_storage.size()) {
            auto *accel = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.empty.surface.filter.accel.slot." +
                    std::to_string(slot));
            accel->setAlignment(::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type), accel,
                ::llvm::Align{alignof(void *)});
            _ray_query_empty_surface_filter_accel_storage[slot] = accel;
        }
        if (slot <
            _ray_query_direct_output_surface_filter_pipeline_callback_storage
                .size()) {
            auto *pipeline_callback = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.direct.output.surface.filter.pipeline.callback.slot." +
                    std::to_string(slot));
            pipeline_callback->setAlignment(
                ::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type),
                pipeline_callback,
                ::llvm::Align{alignof(void *)});
            _ray_query_direct_output_surface_filter_pipeline_callback_storage
                [slot] = pipeline_callback;
        }
        if (slot <
            _ray_query_direct_output_surface_filter_accel_storage.size()) {
            auto *accel = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.direct.output.surface.filter.accel.slot." +
                    std::to_string(slot));
            accel->setAlignment(::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type), accel,
                ::llvm::Align{alignof(void *)});
            _ray_query_direct_output_surface_filter_accel_storage[slot] =
                accel;
        }
        if (slot <
            _ray_query_surface_exclusion_soa_accel_storage.size()) {
            auto *accel = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.surface.exclusion.soa.accel.slot." +
                    std::to_string(slot));
            accel->setAlignment(::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type), accel,
                ::llvm::Align{alignof(void *)});
            _ray_query_surface_exclusion_soa_accel_storage[slot] = accel;
        }
        if (slot <
            _ray_query_surface_exclusion_soa_input_storage.size()) {
            auto *input_type = ::llvm::ArrayType::get(
                _builder.getInt8Ty(),
                sizeof(SIMDHostRayQuerySurfaceExclusionInputPacket));
            auto *input = _builder.CreateAlloca(
                input_type, nullptr,
                "ray.query.surface.exclusion.soa.input.slot." +
                    std::to_string(slot));
            input->setAlignment(::llvm::Align{
                alignof(SIMDHostRayQuerySurfaceExclusionInputPacket)});
            _ray_query_surface_exclusion_soa_input_storage[slot] = input;
        }
        if (slot < _ray_query_output_packet_storage.size()) {
            auto *committed_type = ::llvm::ArrayType::get(
                _builder.getInt8Ty(),
                sizeof(SIMDHostRayQueryOutputPacket));
            auto *committed = _builder.CreateAlloca(
                committed_type, nullptr,
                "ray.query.output.packet.slot." +
                    std::to_string(slot));
            committed->setAlignment(::llvm::Align{
                alignof(SIMDHostRayQueryOutputPacket)});
            _ray_query_output_packet_storage[slot] = committed;
        }
        if (slot <
            _ray_query_surface_filter_ray_packet_storage.size()) {
            auto *lane_type = ::llvm::FixedVectorType::get(
                _builder.getInt32Ty(), _width);
            auto *packet_type = ::llvm::ArrayType::get(
                lane_type, simd_host_accel_ray_packet_field_count);
            auto *ray_packet = _builder.CreateAlloca(
                packet_type, nullptr,
                "ray.query.surface.filter.ray.packet.slot." +
                    std::to_string(slot));
            ray_packet->setAlignment(::llvm::Align{
                _width * sizeof(uint32_t)});
            _ray_query_surface_filter_ray_packet_storage[slot] =
                ray_packet;
            auto *call_packet = _builder.CreateAlloca(
                packet_type, nullptr,
                "ray.query.surface.filter.call.ray.packet.slot." +
                    std::to_string(slot));
            call_packet->setAlignment(::llvm::Align{
                _width * sizeof(uint32_t)});
            _ray_query_surface_filter_ray_packet_call_storage[slot] =
                call_packet;
        }
        if (slot < _ray_query_state_handle_storage.size()) {
            auto *state_handles = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.state.handles.slot." + std::to_string(slot));
            state_handles->setAlignment(::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type),
                state_handles, ::llvm::Align{alignof(void *)});
            _ray_query_state_handle_storage[slot] = state_handles;
        }
    }
    _state_slots.resize(_source.values().size(), nullptr);
    for (auto &&value : _source.values()) {
        auto spill = value.id.value <
                         _spilled_instruction_values.size() &&
                     _spilled_instruction_values[value.id.value] != 0u;
        if (value.origin != schedule::ValueOrigin::state_slot &&
            !spill) {
            continue;
        }
        auto *type = _is_local_lvalue(value.id) ?
                         _local_value_llvm_type(value.id) :
                     _direct_control_flow &&
                             value.value_class ==
                                 schedule::ValueClass::cohort_uniform ?
                         _layout.expression_type(value) :
                         _layout.state_type(value);
        if (type == nullptr) {
            _fail(_layout.error());
            return;
        }
        auto *slot = _builder.CreateAlloca(
            type, nullptr,
            value.name + (spill ? ".spill" : ".slot"));
        _state_slots[value.id.value] = slot;
        _result.state_slot_count++;
        _result.spilled_instruction_count += spill;
    }
    _coalesce_state_slots();
    std::vector<::llvm::AllocaInst *> unique_state_slots;
    unique_state_slots.reserve(_result.state_slot_count);
    std::unordered_set<::llvm::AllocaInst *> seen_state_slots;
    seen_state_slots.reserve(_result.state_slot_count);
    for (auto *slot : _state_slots) {
        if (slot != nullptr && seen_state_slots.emplace(slot).second) {
            unique_state_slots.emplace_back(slot);
        }
    }
    static constexpr auto max_stores_per_init_block = size_t{128u};
    auto split_init_blocks =
        unique_state_slots.size() > max_stores_per_init_block;
    auto stores_in_current_block = size_t{0u};
    auto init_block_index = size_t{0u};
    for (auto *slot : unique_state_slots) {
        if (split_init_blocks &&
            stores_in_current_block >= max_stores_per_init_block) {
            auto *next_init = ::llvm::BasicBlock::Create(
                _module.getContext(),
                "prologue.state.init." + std::to_string(++init_block_index),
                _entry);
            _builder.CreateBr(next_init);
            _builder.SetInsertPoint(next_init);
            stores_in_current_block = 0u;
        }
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(slot->getAllocatedType()), slot);
        stores_in_current_block++;
    }
}

void ScheduleEmitter::_partition_state_residency() {
    // LLVM's default O2 pipeline promotes eligible allocas through the global
    // dispatcher. That is profitable for hot state, but a large set of rarely
    // accessed slots creates wide scheduler PHIs and ultimately physical
    // register spills. Keep hot state promotable and pin cold state to its L1
    // stack slot when cold state dominates the function's state set.
    static constexpr auto max_cold_accesses = size_t{6u};
    static constexpr auto high_pressure_state_slot_count = size_t{2048u};
    static constexpr auto high_pressure_scheduler_phi_budget = size_t{4096u};
    static constexpr auto max_high_pressure_promotable_state_slots =
        size_t{256u};
    auto count_ssa_leaves = [](auto &&self, const ::llvm::Type *type) noexcept -> size_t {
        if (type == nullptr) { return 0u; }
        if (auto *array_type = ::llvm::dyn_cast<::llvm::ArrayType>(type)) {
            return static_cast<size_t>(array_type->getNumElements()) *
                   self(self, array_type->getElementType());
        }
        if (auto *struct_type = ::llvm::dyn_cast<::llvm::StructType>(type)) {
            auto leaves = size_t{0u};
            for (auto i = 0u; i < struct_type->getNumElements(); ++i) {
                leaves += self(self, struct_type->getElementType(i));
            }
            return leaves;
        }
        return 1u;
    };
    auto count_accesses = [](const ::llvm::AllocaInst *slot) noexcept {
        auto count = size_t{0u};
        for (auto *user : slot->users()) {
            count += ::llvm::isa<::llvm::LoadInst>(user) ||
                     ::llvm::isa<::llvm::StoreInst>(user);
        }
        return count;
    };
    std::vector<::llvm::AllocaInst *> physical_slots;
    physical_slots.reserve(_state_slots.size());
    std::unordered_set<::llvm::AllocaInst *> seen_slots;
    seen_slots.reserve(_state_slots.size());
    for (auto *slot : _state_slots) {
        if (slot != nullptr && seen_slots.emplace(slot).second) {
            physical_slots.emplace_back(slot);
        }
    }
    auto slot_count = physical_slots.size();
    auto cold_count = size_t{0u};
    for (auto *slot : physical_slots) {
        cold_count += count_accesses(slot) <= max_cold_accesses;
    }
    _result.cold_state_slot_count = cold_count;
    if (slot_count == 0u ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_COLD_STATE_PARTITION")) {
        return;
    }
    const auto use_volatile_pin =
        luisa::compute::detail::env_flag("LUISA_SIMD_VOLATILE_STATE_PIN");
    auto pin_alloca_non_volatile = [&](::llvm::AllocaInst *alloca_inst) noexcept {
        if (alloca_inst == nullptr) { return; }
        auto *parent = alloca_inst->getParent();
        auto *insert_before =
            parent != nullptr ? parent->getTerminator() : nullptr;
        if (insert_before == nullptr) { return; }
        auto &ctx = _module.getContext();
        auto *ptr_ty = ::llvm::PointerType::getUnqual(ctx);
        auto *asm_fn_ty = ::llvm::FunctionType::get(
            ::llvm::Type::getVoidTy(ctx), {ptr_ty}, false);
        auto *inline_asm = ::llvm::InlineAsm::get(
            asm_fn_ty, "", "X", /*hasSideEffects=*/true);
        ::llvm::IRBuilder<> pin_builder{insert_before};
        auto *call = pin_builder.CreateCall(inline_asm, {alloca_inst});
        call->addFnAttr(::llvm::Attribute::NoUnwind);
        call->addFnAttr(::llvm::Attribute::WillReturn);
        call->setMemoryEffects(::llvm::MemoryEffects::inaccessibleMemOnly(
            ::llvm::ModRefInfo::Mod));
#if LLVM_VERSION_MAJOR >= 21
        call->addParamAttr(
            0u, ::llvm::Attribute::getWithCaptureInfo(
                    ctx, ::llvm::CaptureInfo::none()));
#else
        call->addParamAttr(0u, ::llvm::Attribute::NoCapture);
#endif
    };
    auto pin = [&](::llvm::AllocaInst *slot) noexcept {
        if (use_volatile_pin) {
            for (auto *user : slot->users()) {
                if (auto *load = ::llvm::dyn_cast<::llvm::LoadInst>(user)) {
                    load->setVolatile(true);
                } else if (auto *store =
                               ::llvm::dyn_cast<::llvm::StoreInst>(user)) {
                    store->setVolatile(true);
                }
            }
        } else {
            pin_alloca_non_volatile(slot);
        }
        _result.stack_pinned_state_slot_count++;
    };

    // Coalescing has already collapsed mutually exclusive PHI versions into
    // a much smaller physical set. Pinning a majority of that set recreates
    // the very copy/load traffic the liveness proof removed, so leave compact
    // state promotable. Very large graphics kernels are different: asking
    // LLVM SROA to promote thousands of slots through the global dispatcher
    // creates a basic-block-by-slot SSA explosion before codegen. Bound that
    // work by ranking slots by direct access count, keeping only the hottest
    // fixed-size prefix promotable and preserving all other state in its L1
    // stack slot.
    auto total_ssa_leaves = size_t{0u};
    for (auto *slot : physical_slots) {
        total_ssa_leaves += count_ssa_leaves(
            count_ssa_leaves, slot->getAllocatedType());
    }
    auto scheduler_phi_pressure =
        _result.schedule_block_count * total_ssa_leaves;
    auto high_pressure =
        slot_count >= high_pressure_state_slot_count ||
        scheduler_phi_pressure >= high_pressure_scheduler_phi_budget;
    if (high_pressure) {
        std::stable_sort(
            physical_slots.begin(), physical_slots.end(),
            [&](auto *lhs, auto *rhs) noexcept {
                auto lhs_leaves = count_ssa_leaves(
                    count_ssa_leaves, lhs->getAllocatedType());
                auto rhs_leaves = count_ssa_leaves(
                    count_ssa_leaves, rhs->getAllocatedType());
                return count_accesses(lhs) * rhs_leaves >
                       count_accesses(rhs) * lhs_leaves;
            });
        auto promotable_leaf_budget =
            (_result.schedule_block_count >= size_t{512u} &&
             !luisa::compute::detail::env_flag(
                 "LUISA_SIMD_ALLOW_HIGH_BLOCK_PHI_PROMOTION")) ?
                size_t{0u} :
                std::min(
                    max_high_pressure_promotable_state_slots,
                    high_pressure_scheduler_phi_budget /
                        std::max<size_t>(1u, _result.schedule_block_count));
        auto promoted_leaves = size_t{0u};
        for (auto *slot : physical_slots) {
            auto leaves = count_ssa_leaves(
                count_ssa_leaves, slot->getAllocatedType());
            if (leaves <= promotable_leaf_budget - promoted_leaves) {
                promoted_leaves += leaves;
            } else {
                pin(slot);
            }
        }
        // Non-escaping UniformSoA private allocas now use direct aligned vector
        // loads/stores that LLVM SROA would otherwise promote through the
        // global state-machine dispatcher into tens of thousands of cross-block
        // PHIs. Pin non-escaping UniformSoA allocas to their L1D stack slots
        // while keeping escaping allocas (and outlined-callable reference
        // parameters) non-volatile so intra-block CSE/DSE/LICM remain active.
        if (!_direct_control_flow) {
            std::vector<::llvm::Value *> worklist;
            std::unordered_set<::llvm::Value *> visited;
            std::vector<::llvm::Instruction *> mem_ops;
            for (size_t id = 0u; id < _local_allocations.size(); ++id) {
                auto *alloca_inst = ::llvm::dyn_cast_or_null<::llvm::AllocaInst>(
                    _local_allocations[id]);
                if (alloca_inst == nullptr ||
                    _local_soa_kind_of(
                        schedule::ValueId{static_cast<uint32_t>(id)}) !=
                        LocalSoAKind::uniform) {
                    continue;
                }
                worklist.clear();
                visited.clear();
                mem_ops.clear();
                worklist.push_back(alloca_inst);
                visited.insert(alloca_inst);
                auto escapes = false;
                while (!worklist.empty()) {
                    auto *v = worklist.back();
                    worklist.pop_back();
                    for (auto *u : v->users()) {
                        if (auto *gep =
                                ::llvm::dyn_cast<::llvm::GetElementPtrInst>(u)) {
                            if (visited.insert(gep).second) {
                                worklist.push_back(gep);
                            }
                        } else if (auto *load = ::llvm::dyn_cast<::llvm::LoadInst>(u)) {
                            mem_ops.push_back(load);
                        } else if (auto *store = ::llvm::dyn_cast<::llvm::StoreInst>(u)) {
                            if (store->getPointerOperand() == v) {
                                mem_ops.push_back(store);
                            } else {
                                escapes = true;
                            }
                        } else {
                            escapes = true;
                        }
                    }
                }
                if (!escapes) {
                    if (use_volatile_pin) {
                        for (auto *op : mem_ops) {
                            if (auto *load = ::llvm::dyn_cast<::llvm::LoadInst>(op)) {
                                load->setVolatile(true);
                            } else if (auto *store = ::llvm::dyn_cast<::llvm::StoreInst>(op)) {
                                store->setVolatile(true);
                            }
                        }
                    } else {
                        pin_alloca_non_volatile(alloca_inst);
                    }
                }
            }
        }
        return;
    }
    if (_result.coalesced_state_slot_count != 0u ||
        cold_count * 2u < slot_count) {
        return;
    }
    for (auto *slot : physical_slots) {
        if (count_accesses(slot) <= max_cold_accesses) { pin(slot); }
    }
}

void ScheduleEmitter::_build() {
    auto &context = _module.getContext();
    auto handler_entry = _is_handler_entry();
    auto parameter_types = std::vector<::llvm::Type *>{};
    if (handler_entry) {
        if (_is_outlined_callable_entry()) {
            std::vector<::llvm::Type *> io_fields;
            io_fields.reserve(
                _parameters.size() +
                (_outlined_return_type != nullptr ? 1u : 0u));
            for (auto *param : _parameters) {
                auto *type = _handler_parameter_type(*param);
                if (type == nullptr) { return; }
                io_fields.emplace_back(type);
            }
            if (_outlined_return_type != nullptr) {
                auto *ret_type = _data_type(_outlined_return_type, true);
                if (ret_type == nullptr) { return; }
                io_fields.emplace_back(ret_type);
            }
            _outlined_io_frame_type =
                ::llvm::StructType::get(context, io_fields, false);
            parameter_types = {
                ::llvm::Type::getInt32Ty(context),
                ::llvm::Type::getInt64Ty(context),
                ::llvm::PointerType::getUnqual(context),
                ::llvm::PointerType::getUnqual(context)};
            if (_width == 1u && _outlined_parameter_soa_stride > 1u) {
                parameter_types.emplace_back(::llvm::Type::getInt32Ty(context));
            }
        } else if (_is_surface_filter_handler_entry()) {
            parameter_types = {
                ::llvm::Type::getInt32Ty(context),
                ::llvm::Type::getInt64Ty(context),
                ::llvm::PointerType::getUnqual(context),
                ::llvm::PointerType::getUnqual(context),
                ::llvm::PointerType::getUnqual(context)};
        } else {
            parameter_types = {
                ::llvm::Type::getInt32Ty(context),
                ::llvm::Type::getInt64Ty(context),
                ::llvm::PointerType::getUnqual(context),
                ::llvm::PointerType::getUnqual(context)};
            parameter_types.reserve(3u + _parameters.size());
            for (auto index = size_t{1u};
                 index < _parameters.size(); index++) {
                auto *type = _handler_parameter_type(
                    *_parameters[index]);
                if (type == nullptr) { return; }
                parameter_types.emplace_back(type);
            }
        }
    } else {
        parameter_types = {
            ::llvm::PointerType::getUnqual(context),
            ::llvm::PointerType::getUnqual(context),
            ::llvm::PointerType::getUnqual(context),
            ::llvm::Type::getInt32Ty(context)};
    }
    auto *function_type = ::llvm::FunctionType::get(
        !handler_entry && _cooperative_block ?
            static_cast<::llvm::Type *>(
                ::llvm::PointerType::getUnqual(context)) :
            static_cast<::llvm::Type *>(
                ::llvm::Type::getVoidTy(context)),
        parameter_types,
        false);
    if (_entry_name.empty()) {
        _entry_name = _source.name().empty() ? "simd_kernel" :
                                               _source.name();
        _entry_name += ".simd_w" + std::to_string(_width);
    }
    if (_module.getFunction(_entry_name) != nullptr) {
        _fail("LLVM module already contains the requested SIMD entry name");
        return;
    }
    _entry = ::llvm::Function::Create(
        function_type,
        handler_entry ? ::llvm::GlobalValue::InternalLinkage :
                        ::llvm::GlobalValue::ExternalLinkage,
        _entry_name, _module);
    if (handler_entry) {
        _entry->setDSOLocal(true);
        if (_is_outlined_callable_entry()) {
            _entry->setCallingConv(::llvm::CallingConv::Fast);
            _entry->addFnAttr(::llvm::Attribute::NoInline);
            _entry->addFnAttr(::llvm::Attribute::NoUnwind);
        } else {
            _entry->addFnAttr(::llvm::Attribute::InlineHint);
        }
        if (_is_surface_filter_handler_entry()) {
            for (auto index : {2u, 3u}) {
                _entry->addParamAttr(index, ::llvm::Attribute::NonNull);
                _entry->addParamAttr(index, ::llvm::Attribute::ReadOnly);
            }
            _entry->addParamAttr(4u, ::llvm::Attribute::NonNull);
            _entry->addParamAttr(4u, ::llvm::Attribute::NoAlias);
        }
    }
    _result.schedule_block_count = _source.blocks().size();
    _result.convergence_point_count =
        _source.convergence_points().size();
    auto argument = _entry->arg_begin();
    if (handler_entry) {
        _active_lane_count = &*argument++;
        _active_lane_count->setName("lane_count");
        _handler_active_mask_bits = &*argument++;
        _handler_active_mask_bits->setName("active_mask_bits");
        if (_is_outlined_callable_entry()) {
            auto *pointer_type = ::llvm::PointerType::getUnqual(context);
            _argument_buffer = ::llvm::ConstantPointerNull::get(pointer_type);
            _launch_config = &*argument++;
            _launch_config->setName("launch_config");
            _outlined_io_frame = &*argument++;
            _outlined_io_frame->setName("io_frame");
            if (_width == 1u && _outlined_parameter_soa_stride > 1u) {
                _outlined_soa_lane = &*argument;
                _outlined_soa_lane->setName("soa_lane");
            }
        } else if (_is_surface_filter_handler_entry()) {
            auto *pointer_type =
                ::llvm::PointerType::getUnqual(context);
            _argument_buffer =
                ::llvm::ConstantPointerNull::get(pointer_type);
            _launch_config =
                ::llvm::ConstantPointerNull::get(pointer_type);
            _surface_filter_ray_packet = &*argument++;
            _surface_filter_ray_packet->setName("ray_packet");
            _surface_filter_hit_packet = &*argument++;
            _surface_filter_hit_packet->setName("hit_packet");
            _surface_filter_committed_mask_bits = &*argument;
            _surface_filter_committed_mask_bits->setName(
                "committed_mask_bits");
        } else {
            _argument_buffer = &*argument++;
            _argument_buffer->setName("state_pointer_lanes");
            _launch_config = &*argument++;
            _launch_config->setName("launch_config");
            for (auto index = size_t{1u};
                 index < _parameters.size(); index++) {
                argument->setName(
                    "capture." + std::to_string(index - 1u));
                ++argument;
            }
        }
        _return_buffer = ::llvm::ConstantPointerNull::get(
            ::llvm::PointerType::getUnqual(context));
    } else {
        _argument_buffer = &*argument++;
        _argument_buffer->setName("argument_buffer");
        _return_buffer = &*argument++;
        _return_buffer->setName("return_lanes");
        _launch_config = &*argument++;
        _launch_config->setName("launch_config");
        _active_lane_count = &*argument;
        _active_lane_count->setName("active_lane_count");
    }
    // The runtime owns launch_config separately from every resource and from
    // the packed descriptor buffer. The packet body only observes it. These
    // attributes let an inlined batch hoist immutable launch geometry across
    // packet iterations without changing the portable packet ABI.
    if (!handler_entry &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_PACKET_ABI_ALIAS_ATTRIBUTES")) {
        _entry->addParamAttr(0u, ::llvm::Attribute::NoAlias);
        _entry->addParamAttr(0u, ::llvm::Attribute::ReadOnly);
        _entry->addParamAttr(2u, ::llvm::Attribute::NoAlias);
        _entry->addParamAttr(2u, ::llvm::Attribute::NonNull);
        if (!_cooperative_block) {
            _entry->addParamAttr(2u, ::llvm::Attribute::ReadOnly);
        }
    }

    auto *prologue = ::llvm::BasicBlock::Create(
        context, "prologue", _entry);
    _builder.SetInsertPoint(prologue);
    if (handler_entry) {
        if (_is_outlined_callable_entry()) {
            _entry->addParamAttr(2u, ::llvm::Attribute::NoAlias);
            _entry->addParamAttr(2u, ::llvm::Attribute::NonNull);
            _entry->addParamAttr(2u, ::llvm::Attribute::ReadOnly);
            _entry->addParamAttr(3u, ::llvm::Attribute::NoAlias);
            if (_outlined_io_frame_type != nullptr &&
                _outlined_io_frame_type->getNumElements() != 0u) {
                _entry->addParamAttr(3u, ::llvm::Attribute::NonNull);
            }
            _trap_if(
                _builder.CreateICmpNE(
                    _active_lane_count, _builder.getInt32(_width)),
                "outlined.callable.width.mismatch");
            if (_width < 64u) {
                auto lane_bits = (uint64_t{1u} << _width) - 1u;
                _trap_if(
                    _builder.CreateICmpNE(
                        _builder.CreateAnd(
                            _handler_active_mask_bits,
                            _builder.getInt64(~lane_bits)),
                        _builder.getInt64(0u)),
                    "outlined.callable.mask.out.of.range");
            }
        } else {
            _entry->addParamAttr(2u, ::llvm::Attribute::NoAlias);
            _entry->addParamAttr(2u, ::llvm::Attribute::NonNull);
            _entry->addParamAttr(3u, ::llvm::Attribute::NoAlias);
            _entry->addParamAttr(3u, ::llvm::Attribute::NonNull);
            _entry->addParamAttr(3u, ::llvm::Attribute::ReadOnly);
            _trap_if(
                _builder.CreateICmpNE(
                    _active_lane_count, _builder.getInt32(_width)),
                "ray.query.handler.width.mismatch");
            auto lane_bits = (uint64_t{1u} << _width) - 1u;
            _trap_if(
                _builder.CreateICmpNE(
                    _builder.CreateAnd(
                        _handler_active_mask_bits,
                        _builder.getInt64(~lane_bits)),
                    _builder.getInt64(0u)),
                "ray.query.handler.mask.out.of.range");
        }
    }
    if (_cooperative_block) {
        _begin_cooperative_coroutine();
        if (_failed()) { return; }
    }
    _direct_control_flow =
        !_cooperative_block &&
        (_width == 1u ||
         (!luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_COHERENT_DIRECT_CFG") &&
          _can_emit_direct_control_flow()));
    _result.direct_control_flow = _direct_control_flow;
    if (!_direct_control_flow &&
        _enable_predicated_acyclic_control_flow) {
        if (auto order = _find_predicated_acyclic_order()) {
            _predicated_acyclic_control_flow = true;
            _predicated_acyclic_order = std::move(*order);
        }
    }
    _result.predicated_acyclic_control_flow =
        _predicated_acyclic_control_flow;
    _allocate_state();
    if (_failed()) { return; }
    _create_external_values();
    if (_failed()) { return; }

    auto *lane_ids = _lane_ids();
    auto *initial_mask = static_cast<::llvm::Value *>(nullptr);
    if (handler_entry) {
        auto *packed_mask_type = ::llvm::IntegerType::get(
            context, _width);
        initial_mask = _builder.CreateBitCast(
            _builder.CreateTrunc(
                _handler_active_mask_bits, packed_mask_type),
            _layout.mask_type(), "handler.active.mask");
    } else {
        auto *count = _builder.CreateVectorSplat(
            _width, _active_lane_count);
        initial_mask = _builder.CreateICmpULT(lane_ids, count);
    }
    _ensure_launch_vectors();
    // Runtime block IDs are always drawn from ceil(dispatch_size / block_size)
    // in each dimension. A statically unit-sized dimension therefore has
    // thread_id == 0 and block_id < dispatch_size for every launched block;
    // it cannot contribute a dispatch-edge tail. Avoid materializing that
    // redundant vector compare, especially for the two unit dimensions of a
    // 1D kernel. Standalone packet callers have the same valid-block contract.
    auto elide_unit_dimension_masks =
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_UNIT_DIMENSION_MASK_ELISION");
    if (handler_entry) {
        // The caller supplies the exact sparse physical-lane cohort. Dispatch
        // geometry remains available to handler expressions but must not
        // widen or narrow this semantic candidate mask.
    } else if (_enable_linear_1d_packet_tail_narrowing &&
               !_cooperative_block) {
        // The runtime-only packet wrapper narrows its final 1D packet to the
        // exact dispatch and block remainder, and skips packets with no live
        // lanes. The active-lane prefix above is therefore the complete
        // dispatch mask; rebuilding dispatch_id < dispatch_size in every
        // packet would be redundant. Cooperative wrappers deliberately issue
        // every static packet in a block so all participating packets can
        // rendezvous. They must retain the full dispatch-extent mask for a
        // partial edge block, including a mixed final packet.
        _result.linear_1d_packet_tail_narrowing_count++;
    } else {
        for (auto i = uint32_t{0u}; i < 3u; i++) {
            if (elide_unit_dimension_masks &&
                _static_block_size[i] == 1u) {
                _result.unit_dimension_mask_elision_count++;
                continue;
            }
            initial_mask = _builder.CreateAnd(
                initial_mask,
                _builder.CreateICmpULT(
                    _dispatch_id[i],
                    _builder.CreateVectorSplat(
                        _width, _dispatch_size[i])));
        }
    }
    if (_cooperative_block) {
        _initialize_cooperative_packet(initial_mask);
    }
    if (_direct_control_flow) {
        _build_direct(initial_mask);
        return;
    }
    if (_predicated_acyclic_control_flow) {
        _build_predicated_acyclic(initial_mask);
        return;
    }
    _builder.CreateStore(initial_mask, _live_mask);
    _resume(_source.entry(), initial_mask);

    _scheduler_loop = ::llvm::BasicBlock::Create(
        context, "scheduler.loop", _entry);
    auto *dispatch = ::llvm::BasicBlock::Create(
        context, "scheduler.dispatch", _entry);
    _scheduler_dispatch_route = ::llvm::BasicBlock::Create(
        context, "scheduler.dispatch.route", _entry);
    auto *done = ::llvm::BasicBlock::Create(
        context, "scheduler.done", _entry);
    auto *exit = ::llvm::BasicBlock::Create(
        context, "scheduler.exit", _entry);
    auto *stalled = ::llvm::BasicBlock::Create(
        context, "scheduler.stalled", _entry);
    auto *invalid = ::llvm::BasicBlock::Create(
        context, "scheduler.invalid", _entry);
    _schedule_blocks.reserve(_source.blocks().size());
    for (auto &&block : _source.blocks()) {
        _schedule_blocks.emplace_back(::llvm::BasicBlock::Create(
            context,
            "schedule." + std::to_string(block.id.value), _entry));
    }
    std::vector<uint8_t> structured_emitted_blocks(
        _source.blocks().size(), uint8_t{0u});
    std::vector<uint8_t> structured_absorbed_blocks(
        _source.blocks().size(), uint8_t{0u});
    if (_structured_early_exit_loop) {
        for (auto *block :
             _structured_early_exit_loop->emitted_blocks) {
            structured_emitted_blocks[block->id.value] = 1u;
        }
        for (auto *block :
             _structured_early_exit_loop->absorbed_blocks) {
            structured_absorbed_blocks[block->id.value] = 1u;
        }
    }
    _builder.CreateBr(_scheduler_loop);

    _builder.SetInsertPoint(_scheduler_loop);
    auto *ready_count = _builder.CreateLoad(
        _builder.getInt32Ty(), _ready_count);
    _builder.CreateCondBr(
        _builder.CreateICmpNE(
            ready_count, _builder.getInt32(0u)),
        dispatch, done);

    _builder.SetInsertPoint(dispatch);
    auto *ready_index = _builder.CreateSub(
        ready_count, _builder.getInt32(1u));
    _builder.CreateStore(ready_index, _ready_count);
    auto *ready_mask_ptr = _ready_element_pointer(
        _ready_masks, _ready_masks_type, ready_index);
    auto *popped_mask = _builder.CreateLoad(
        _layout.mask_type(), ready_mask_ptr);
    _builder.CreateStore(popped_mask, _current_mask);
    auto *ready_target_ptr = _ready_element_pointer(
        _ready_targets, _ready_i32_type, ready_index);
    auto *pc = _builder.CreateLoad(
        _builder.getInt32Ty(), ready_target_ptr);
    auto *ready_token_ptr = _ready_element_pointer(
        _ready_tokens, _ready_i32_type, ready_index);
    auto *token = _builder.CreateLoad(
        _builder.getInt32Ty(), ready_token_ptr);
    _builder.CreateStore(token, _current_token);
    auto *runnable = _builder.CreateLoad(
        _layout.mask_type(), _runnable_mask);
    _builder.CreateStore(
        _builder.CreateAnd(
            runnable, _builder.CreateNot(popped_mask)),
        _runnable_mask);
    _builder.CreateBr(_scheduler_dispatch_route);

    _builder.SetInsertPoint(_scheduler_dispatch_route);
    _scheduler_dispatch_pc = _builder.CreatePHI(
        _builder.getInt32Ty(), 1u, "scheduler.dispatch.pc");
    _scheduler_dispatch_pc->addIncoming(pc, dispatch);
    auto *dispatch_switch = _builder.CreateSwitch(
        _scheduler_dispatch_pc,
        invalid,
        static_cast<unsigned>(_schedule_blocks.size()));
    for (auto &&block : _source.blocks()) {
        if (structured_emitted_blocks[block.id.value] != 0u ||
            structured_absorbed_blocks[block.id.value] != 0u) {
            continue;
        }
        dispatch_switch->addCase(
            _builder.getInt32(block.id.value),
            _schedule_blocks[block.id.value]);
    }

    _builder.SetInsertPoint(invalid);
    auto *invalid_trap =
#if LLVM_VERSION_MAJOR >= 22
        ::llvm::Intrinsic::getOrInsertDeclaration(
#else
        ::llvm::Intrinsic::getDeclaration(
#endif
            &_module, ::llvm::Intrinsic::trap);
    _builder.CreateCall(invalid_trap);
    _builder.CreateUnreachable();

    _builder.SetInsertPoint(done);
    auto *live = _builder.CreateLoad(
        _layout.mask_type(), _live_mask);
    _builder.CreateCondBr(
        _builder.CreateOrReduce(live), stalled, exit);

    _builder.SetInsertPoint(stalled);
    auto *stalled_trap =
#if LLVM_VERSION_MAJOR >= 22
        ::llvm::Intrinsic::getOrInsertDeclaration(
#else
        ::llvm::Intrinsic::getDeclaration(
#endif
            &_module, ::llvm::Intrinsic::trap);
    _builder.CreateCall(stalled_trap);
    _builder.CreateUnreachable();

    _builder.SetInsertPoint(exit);
    if (_cooperative_block) {
        _finish_entry();
    } else {
        _builder.CreateRetVoid();
    }

    std::vector<uint8_t> locally_inlined_blocks(
        _source.blocks().size(), uint8_t{0u});
    auto in_structured_loop = [&](schedule::BlockId id) noexcept {
        return _structured_early_exit_loop &&
               std::find(
                   _structured_early_exit_loop->loop->blocks.cbegin(),
                   _structured_early_exit_loop->loop->blocks.cend(), id) !=
                   _structured_early_exit_loop->loop->blocks.cend();
    };
    std::vector<uint8_t> chained_blocks(
        _source.blocks().size(), uint8_t{0u});
    for (auto &&block : _source.blocks()) {
        if (chained_blocks[block.id.value] != 0u) { continue; }
        if (auto region =
                _find_chained_predicated_region(block)) {
            chained_blocks[block.id.value] = 1u;
            for (auto *inlined : region->inlined_blocks) {
                locally_inlined_blocks[inlined->id.value] = 1u;
                chained_blocks[inlined->id.value] = 1u;
            }
        }
    }
    for (auto &&block : _source.blocks()) {
        if (chained_blocks[block.id.value] != 0u) { continue; }
        if (auto diamond =
                _find_guarded_predicated_math_diamond(
                    block, in_structured_loop(block.id))) {
            for (auto *arm : diamond->true_blocks) {
                locally_inlined_blocks[arm->id.value] = 1u;
            }
            for (auto *arm : diamond->false_blocks) {
                locally_inlined_blocks[arm->id.value] = 1u;
            }
        }
    }
    for (auto &&block : _source.blocks()) {
        if (chained_blocks[block.id.value] != 0u) { continue; }
        if (auto region = _find_nested_predicated_region(
                block, in_structured_loop(block.id))) {
            locally_inlined_blocks[region->nested_split_block->id.value] = 1u;
            for (auto *arm : region->nested_diamond.true_blocks) {
                locally_inlined_blocks[arm->id.value] = 1u;
            }
            for (auto *arm : region->nested_diamond.false_blocks) {
                locally_inlined_blocks[arm->id.value] = 1u;
            }
            locally_inlined_blocks[region->nested_merge_block->id.value] = 1u;
            locally_inlined_blocks[region->other_block->id.value] = 1u;
        }
    }
    if (_structured_early_exit_loop) {
        for (auto *block :
             _structured_early_exit_loop->absorbed_blocks) {
            locally_inlined_blocks[block->id.value] = 1u;
        }
    }
    for (auto &&block : _source.blocks()) {
        if (structured_emitted_blocks[block.id.value] != 0u) {
            continue;
        }
        _builder.SetInsertPoint(
            _schedule_blocks[block.id.value]);
        if (locally_inlined_blocks[block.id.value] != 0u) {
            _builder.CreateUnreachable();
            continue;
        }
        _locals.clear();
        _active_mask = _builder.CreateLoad(
            _layout.mask_type(), _current_mask);
        if (_target_convergence_depths[block.id.value] != 0u) {
            auto *flow = _cascade_at_convergence_target(
                _builder.getInt32(block.id.value), _active_mask);
            auto *ready = ::llvm::BasicBlock::Create(
                context,
                "convergence.target." +
                    std::to_string(block.id.value) + ".ready",
                _entry);
            _builder.CreateCondBr(
                _builder.CreateOrReduce(flow),
                ready, _scheduler_loop);
            _builder.SetInsertPoint(ready);
            _active_mask = flow;
        }
        _seed_lane = _safe_first_lane(_active_mask);
        if (_structured_early_exit_loop &&
            _structured_early_exit_loop->header->id == block.id) {
            _emit_structured_early_exit_loop(
                *_structured_early_exit_loop);
            if (_failed()) { return; }
            continue;
        }
        if (auto loop = _find_predicated_loop(block)) {
            _emit_predicated_loop(*loop);
            if (_failed()) { return; }
            continue;
        }
        for (auto &&instruction : block.instructions) {
            _emit_instruction(instruction);
            if (_failed()) { return; }
        }
        _emit_terminator(block);
        if (_failed()) { return; }
    }

    _partition_state_residency();
}

}// namespace luisa::compute::simd::detail
