#include "llvm_schedule_emitter.h"

#include <cstdlib>
#include <cstring>

#include "../../common/env_flag.h"

namespace luisa::compute::simd {

namespace {

[[nodiscard]] size_t align_up_soa(size_t value, size_t alignment) noexcept {
    return alignment == 0u ? value :
                             (value + alignment - 1u) & ~(alignment - 1u);
}

[[nodiscard]] luisa::optional<uint64_t> constant_int_index_soa(
    const schedule::Function &fn, schedule::ValueId id) noexcept {
    auto *value = fn.value(id);
    if (value == nullptr ||
        value->origin != schedule::ValueOrigin::constant ||
        value->type == nullptr || !value->type->is_scalar() ||
        value->type->is_float() || value->type->size() == 0u ||
        value->type->size() > sizeof(uint64_t)) {
        return luisa::nullopt;
    }
    auto *metadata = std::get_if<schedule::ConstantValueMetadata>(
        &value->metadata);
    if (metadata == nullptr || metadata->bytes.size() < value->type->size()) {
        return luisa::nullopt;
    }
    auto result = uint64_t{0u};
    std::memcpy(&result, metadata->bytes.data(), value->type->size());
    auto bits = static_cast<uint32_t>(value->type->size() * 8u);
    if (bits < 64u) { result &= (uint64_t{1u} << bits) - 1u; }
    if (value->type->is_int() &&
        (result & (uint64_t{1u} << (bits - 1u))) != 0u) {
        return luisa::nullopt;
    }
    return result;
}

void for_each_assignment_soa(
    const schedule::BasicBlock &block,
    const luisa::function<void(schedule::EdgeAssignment)> &visit) {
    auto visit_edge = [&](const schedule::ControlEdge &edge) {
        for (auto a : edge.assignments) { visit(a); }
    };
    std::visit(
        [&](const auto &terminator) {
            using T = std::decay_t<decltype(terminator)>;
            if constexpr (std::is_same_v<T, schedule::BranchTerminator>) {
                visit_edge(terminator.edge);
            } else if constexpr (std::is_same_v<T, schedule::SplitTerminator>) {
                visit_edge(terminator.true_edge);
                visit_edge(terminator.false_edge);
            } else if constexpr (std::is_same_v<T, schedule::SwitchTerminator>) {
                for (auto &&item : terminator.cases) { visit_edge(item.edge); }
                visit_edge(terminator.default_edge);
            } else if constexpr (std::is_same_v<T, schedule::JoinTerminator> ||
                                 std::is_same_v<T, schedule::LoopBackTerminator>) {
                for (auto a : terminator.assignments) { visit(a); }
            } else if constexpr (std::is_same_v<T, schedule::BlockBarrierTerminator>) {
                visit_edge(terminator.resume_edge);
            }
        },
        block.terminator);
}

}// namespace

namespace {

[[nodiscard]] uint8_t max_soa_leaf_size_at(
    const Type *type, size_t base_offset) noexcept {
    if (type == nullptr || type->is_resource() || type->is_custom() ||
        type->is_cooperative_vector() ||
        type->is_cooperative_vector_ref() ||
        type->is_cooperative_matrix_ref()) {
        return 0u;
    }
    if (type->is_scalar()) {
        if (type->is_float8()) { return 0u; }
        auto sz = type->size();
        if (sz != 1u && sz != 2u && sz != 4u && sz != 8u) { return 0u; }
        if (base_offset % sz != 0u) { return 0u; }
        return static_cast<uint8_t>(sz);
    }
    if (type->is_vector() || type->is_matrix() || type->is_array()) {
        if (type->dimension() == 0u || type->size() == 0u) { return 0u; }
        auto *child = type->is_matrix() ?
                          Type::vector(type->element(), type->dimension()) :
                          type->element();
        auto stride = type->is_vector() ?
                          type->element()->size() :
                          type->size() / type->dimension();
        if (stride == 0u) { return 0u; }
        auto c0 = max_soa_leaf_size_at(child, base_offset);
        if (c0 == 0u) { return 0u; }
        if (type->dimension() > 1u &&
            max_soa_leaf_size_at(child, base_offset + stride) == 0u) {
            return 0u;
        }
        return c0;
    }
    if (type->is_structure()) {
        if (type->members().empty() || type->size() == 0u) { return 0u; }
        auto max_leaf = uint8_t{0u};
        auto offset = size_t{0u};
        for (auto *member : type->members()) {
            offset = align_up_soa(offset, member->alignment());
            auto m = max_soa_leaf_size_at(member, base_offset + offset);
            if (m == 0u) { return 0u; }
            if (m > max_leaf) { max_leaf = m; }
            offset += member->size();
        }
        return max_leaf;
    }
    return 0u;
}

}// namespace

[[nodiscard]] uint8_t uniform_leaf_cell_size(const Type *type) noexcept {
    return max_soa_leaf_size_at(type, 0u);
}

[[nodiscard]] uint32_t simd_sparse_w1_outlined_max_lanes() noexcept {
    if (luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_SPARSE_W1_OUTLINED")) {
        return 0u;
    }
    if (const char *raw = std::getenv("LUISA_SIMD_SPARSE_W1_MAX_LANES");
        raw != nullptr && raw[0] != '\0') {
        char *end = nullptr;
        auto parsed = std::strtoul(raw, &end, 10);
        if (end != raw) {
            return static_cast<uint32_t>(parsed);
        }
    }
    return 2u;
}

[[nodiscard]] LocalSoAAnalysis analyze_local_soa(
    const schedule::Function &function,
    uint32_t width,
    luisa::span<const uint8_t> param_soa_cell_bytes,
    luisa::span<const std::vector<uint8_t>> callee_param_soa_cell_bytes,
    bool enable_interleaved_private_arrays,
    uint32_t soa_stride) noexcept {
    constexpr auto no_root = std::numeric_limits<uint32_t>::max();
    constexpr auto ambiguous_root = std::numeric_limits<uint32_t>::max() - 1u;
    auto effective_stride = soa_stride > 1u ? soa_stride : width;
    auto value_count = function.values().size();
    LocalSoAAnalysis out{};
    out.kind.assign(value_count, LocalSoAKind::none);
    out.cell_bytes.assign(value_count, 0u);
    out.root_value.assign(value_count, no_root);
    if (!enable_interleaved_private_arrays || effective_stride <= 1u ||
        value_count == 0u) {
        return out;
    }

    struct Use {
        const schedule::Instruction *inst;
        size_t op_idx;
        uint32_t block_id;
    };
    std::vector<std::vector<Use>> users(value_count);
    std::vector<uint32_t> root(value_count, no_root);
    std::vector<uint8_t> root_cell(value_count, 0u);
    std::vector<uint8_t> root_is_alloca(value_count, 0u);
    std::vector<uint8_t> root_passed_to_outlined_call(value_count, 0u);
    std::vector<uint8_t> is_root_alias(value_count, 0u);
    std::vector<uint8_t> poisoned(value_count, 0u);
    std::vector<uint8_t> in_assignment(value_count, 0u);
    std::vector<uint8_t> returned(value_count, 0u);
    std::vector<uint8_t> used_outside_def_block(value_count, 0u);
    std::vector<uint8_t> passed_to_outlined_call(value_count, 0u);
    std::vector<const schedule::Instruction *> gep_insts;
    std::vector<uint32_t> gep_blocks;
    bool has_candidate = false;

    for (auto &&val : function.values()) {
        if (val.origin != schedule::ValueOrigin::parameter) { continue; }
        auto *md = std::get_if<schedule::ParameterValueMetadata>(&val.metadata);
        if (md == nullptr) { continue; }
        auto tag = static_cast<xir::DerivedArgumentTag>(md->argument_tag);
        if (tag == xir::DerivedArgumentTag::REFERENCE &&
            md->index < param_soa_cell_bytes.size() &&
            param_soa_cell_bytes[md->index] != 0u) {
            auto cell = uniform_leaf_cell_size(val.type);
            if (cell != 0u && cell == param_soa_cell_bytes[md->index]) {
                root[val.id.value] = val.id.value;
                root_cell[val.id.value] = cell;
                is_root_alias[val.id.value] = 1u;
                has_candidate = true;
            }
        }
    }

    for (auto &&block : function.blocks()) {
        for (auto &&inst : block.instructions) {
            if (inst.opcode == schedule::Opcode::alloca && inst.result &&
                inst.source_op &&
                static_cast<xir::AllocaOp>(*inst.source_op) ==
                    xir::AllocaOp::LOCAL) {
                auto id = inst.result->value;
                auto *val = function.value(*inst.result);
                auto cell = val == nullptr ? 0u :
                                             uniform_leaf_cell_size(val->type);
                if (cell != 0u && val->type->size() != 0u &&
                    static_cast<uint64_t>(val->type->size()) * effective_stride <
                        (uint64_t{1u} << 30u)) {
                    root[id] = id;
                    root_cell[id] = cell;
                    root_is_alloca[id] = 1u;
                    is_root_alias[id] = 1u;
                    has_candidate = true;
                }
            } else if (inst.opcode == schedule::Opcode::gep && inst.result &&
                       !inst.operands.empty()) {
                gep_insts.emplace_back(&inst);
                gep_blocks.emplace_back(block.id.value);
            }
            for (size_t i = 0u; i < inst.operands.size(); ++i) {
                auto op = inst.operands[i].value;
                if (op < value_count) {
                    users[op].emplace_back(Use{&inst, i, block.id.value});
                    auto *op_val = function.value(inst.operands[i]);
                    if (op_val != nullptr && op_val->defining_block &&
                        op_val->defining_block->value != block.id.value) {
                        used_outside_def_block[op] = 1u;
                    }
                }
            }
        }
        for_each_assignment_soa(block, [&](schedule::EdgeAssignment a) {
            if (a.source.value < value_count) {
                in_assignment[a.source.value] = 1u;
            }
            if (a.destination.value < value_count) {
                in_assignment[a.destination.value] = 1u;
            }
        });
        if (auto *ret = std::get_if<schedule::ReturnTerminator>(&block.terminator);
            ret && ret->value && ret->value->value < value_count) {
            returned[ret->value->value] = 1u;
        }
    }
    if (!has_candidate) { return out; }

    auto poison_root = [&](uint32_t r) noexcept {
        if (r < value_count) { poisoned[r] = 1u; }
    };
    auto merge_state_root = [&](uint32_t dst, uint32_t src) noexcept -> bool {
        auto src_r = root[src];
        auto &dst_r = root[dst];
        auto *dst_val = function.value(schedule::ValueId{dst});
        auto valid_slot = dst_val != nullptr &&
                          dst_val->origin == schedule::ValueOrigin::state_slot;
        if (!valid_slot || !is_root_alias[src] ||
            src_r == ambiguous_root || src_r == no_root) {
            poison_root(src_r);
            poison_root(dst_r);
            if (dst_r == ambiguous_root) { return false; }
            dst_r = ambiguous_root;
            is_root_alias[dst] = 0u;
            return true;
        }
        if (dst_r == ambiguous_root) {
            poison_root(src_r);
            return false;
        }
        if (dst_r == no_root) {
            dst_r = src_r;
            is_root_alias[dst] = 1u;
            return true;
        }
        if (dst_r != src_r) {
            poison_root(dst_r);
            poison_root(src_r);
            dst_r = ambiguous_root;
            is_root_alias[dst] = 0u;
            return true;
        }
        return false;
    };

    auto changed = true;
    while (changed) {
        changed = false;
        for (size_t gi = 0u; gi < gep_insts.size(); ++gi) {
            auto *gep = gep_insts[gi];
            auto base_r = root[gep->operands.front().value];
            auto res = gep->result->value;
            if (root[res] != base_r) {
                root[res] = base_r;
                changed = true;
            }
        }
        for (auto &&block : function.blocks()) {
            for_each_assignment_soa(block, [&](schedule::EdgeAssignment a) {
                auto src = a.source.value;
                auto dst = a.destination.value;
                if (src >= value_count || dst >= value_count) { return; }
                if (root[src] != no_root || root[dst] != no_root) {
                    changed |= merge_state_root(dst, src);
                }
            });
        }
    }

    // Validate every use of every candidate value.
    for (uint32_t v = 0u; v < value_count; ++v) {
        auto r = root[v];
        if (r >= value_count || poisoned[r]) { continue; }
        if (returned[v]) {
            poisoned[r] = 1u;
            continue;
        }
        for (auto &use : users[v]) {
            auto *inst = use.inst;
            if (inst->opcode == schedule::Opcode::gep) {
                if (use.op_idx != 0u || !inst->result) {
                    poisoned[r] = 1u;
                    break;
                }
            } else if (inst->opcode == schedule::Opcode::load ||
                       inst->opcode == schedule::Opcode::store) {
                if (use.op_idx != 0u) {
                    poisoned[r] = 1u;
                    break;
                }
            } else if (inst->opcode == schedule::Opcode::call &&
                       inst->contiguous_copy) {
                auto *val = function.value(schedule::ValueId{v});
                if (use.op_idx != 2u || !is_root_alias[v] ||
                    val == nullptr || val->type == nullptr ||
                    !val->type->is_array() ||
                    !val->type->element()->is_scalar()) {
                    poisoned[r] = 1u;
                    break;
                }
            } else if (inst->opcode == schedule::Opcode::call &&
                       inst->outlined_callable_id) {
                auto cid = *inst->outlined_callable_id;
                if (cid >= callee_param_soa_cell_bytes.size() ||
                    use.op_idx >= callee_param_soa_cell_bytes[cid].size() ||
                    callee_param_soa_cell_bytes[cid][use.op_idx] == 0u) {
                    poisoned[r] = 1u;
                    break;
                }
                passed_to_outlined_call[v] = 1u;
                root_passed_to_outlined_call[r] = 1u;
            } else {
                poisoned[r] = 1u;
                break;
            }
        }
    }

    if (width == 1u && effective_stride > 1u) {
        // In a W=1 sparse companion, keep local allocas as contiguous scalar
        // stack storage (so LLVM SROA promotes them to registers) unless they
        // are passed to a callee expecting UniformSoA storage.
        for (uint32_t v = 0u; v < value_count; ++v) {
            if (root_is_alloca[v] && !root_passed_to_outlined_call[v]) {
                poisoned[v] = 1u;
            }
        }
    }

    // Classify uniform vs varying for every non-poisoned value.
    std::vector<uint8_t> const_uniform(value_count, 0u);
    for (uint32_t v = 0u; v < value_count; ++v) {
        auto r = root[v];
        if (r < value_count && !poisoned[r] && is_root_alias[v]) {
            const_uniform[v] = 1u;
            out.kind[v] = LocalSoAKind::uniform;
        }
    }
    auto gep_changed = true;
    while (gep_changed) {
        gep_changed = false;
        for (size_t gi = 0u; gi < gep_insts.size(); ++gi) {
            auto *gep = gep_insts[gi];
            auto res = gep->result->value;
            auto r = root[res];
            if (r >= value_count || poisoned[r]) { continue; }
            auto base = gep->operands.front().value;
            if (out.kind[base] == LocalSoAKind::none) { continue; }
            auto new_const_u = uint8_t{0u};
            auto new_kind = LocalSoAKind::varying;
            if (width == 1u) {
                new_const_u = 1u;
                new_kind = LocalSoAKind::uniform;
            } else if (const_uniform[base]) {
                auto *cur_type = function.value(gep->operands.front())->type;
                auto all_const = true;
                auto all_uniform = true;
                for (size_t i = 1u; i < gep->operands.size(); ++i) {
                    if (cur_type == nullptr) {
                        all_const = all_uniform = false;
                        break;
                    }
                    auto idx_id = gep->operands[i];
                    if (cur_type->is_structure()) {
                        auto c = constant_int_index_soa(function, idx_id);
                        if (!c || *c >= cur_type->members().size()) {
                            all_const = all_uniform = false;
                            break;
                        }
                        cur_type = cur_type->members()[static_cast<size_t>(*c)];
                        continue;
                    }
                    if (!constant_int_index_soa(function, idx_id).has_value()) {
                        all_const = false;
                        auto *idx_val = function.value(idx_id);
                        auto is_u =
                            idx_val != nullptr &&
                            (idx_val->value_class ==
                                 schedule::ValueClass::warp_uniform ||
                             idx_val->value_class ==
                                 schedule::ValueClass::cohort_uniform ||
                             (gep->cohort_uniform_operand_index &&
                              *gep->cohort_uniform_operand_index == i));
                        if (!is_u) { all_uniform = false; }
                    }
                    cur_type = cur_type->is_matrix() ?
                                   Type::vector(cur_type->element(),
                                                cur_type->dimension()) :
                                   cur_type->element();
                }
                if (all_const) {
                    new_const_u = 1u;
                    new_kind = LocalSoAKind::uniform;
                } else if (all_uniform && !used_outside_def_block[res] &&
                           !in_assignment[res] && !passed_to_outlined_call[res]) {
                    auto same_block_mem_only = true;
                    for (auto &u : users[res]) {
                        if (u.block_id != gep_blocks[gi] ||
                            (u.inst->opcode != schedule::Opcode::load &&
                             u.inst->opcode != schedule::Opcode::store)) {
                            same_block_mem_only = false;
                            break;
                        }
                    }
                    if (same_block_mem_only) {
                        new_kind = LocalSoAKind::uniform;
                    }
                }
            }
            if (const_uniform[res] != new_const_u || out.kind[res] != new_kind) {
                const_uniform[res] = new_const_u;
                out.kind[res] = new_kind;
                gep_changed = true;
            }
        }
    }

    for (uint32_t v = 0u; v < value_count; ++v) {
        auto r = root[v];
        if (r < value_count && !poisoned[r] &&
            out.kind[v] != LocalSoAKind::none) {
            out.cell_bytes[v] = root_cell[r];
            out.root_value[v] = r;
        } else {
            out.kind[v] = LocalSoAKind::none;
            out.cell_bytes[v] = 0u;
            out.root_value[v] = no_root;
        }
    }
    for (uint32_t v = 0u; v < value_count; ++v) {
        if (root_is_alloca[v] && !poisoned[v] &&
            out.kind[v] == LocalSoAKind::uniform) {
            out.interleaved_allocation_count++;
        }
    }
    return out;
}

}// namespace luisa::compute::simd

namespace luisa::compute::simd::detail {

void ScheduleEmitter::_preflight_outlined_calls() {
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode != schedule::Opcode::call ||
                !instruction.outlined_callable_id) {
                continue;
            }
            if (_is_ray_query_handler_entry()) {
                _fail("ray-query handlers cannot invoke outlined callables");
                return;
            }
            if (_width > 64u) {
                _fail("outlined callable call site requires packet width at most 64");
                return;
            }
            auto id = *instruction.outlined_callable_id;
            if (id >= _outlined_callables.size()) {
                _fail("outlined call references an out-of-range callable ID");
                return;
            }
            auto &&callee = _outlined_callables[id];
            if (callee.entry == nullptr || callee.io_frame_type == nullptr ||
                callee.parameter_types.size() != instruction.operands.size() ||
                callee.parameter_argument_tags.size() !=
                    instruction.operands.size() ||
                callee.parameter_value_classes.size() !=
                    instruction.operands.size()) {
                _fail("outlined call descriptor does not match its Schedule IR operands");
                return;
            }
            if (callee.return_type == nullptr) {
                if (instruction.result) {
                    _fail("void outlined call cannot define a result value");
                    return;
                }
            } else {
                auto *result_val = instruction.result ?
                                       _source.value(*instruction.result) :
                                       nullptr;
                if (result_val == nullptr ||
                    result_val->type != callee.return_type) {
                    _fail("outlined call result type does not match its callee return type");
                    return;
                }
            }
            for (auto i = size_t{0u}; i < instruction.operands.size(); i++) {
                auto op_id = instruction.operands[i];
                auto *op_val = _source.value(op_id);
                if (op_val == nullptr ||
                    op_val->type != callee.parameter_types[i]) {
                    _fail("outlined call operand type mismatch");
                    return;
                }
                auto tag = static_cast<xir::DerivedArgumentTag>(
                    callee.parameter_argument_tags[i]);
                if (tag == xir::DerivedArgumentTag::REFERENCE) {
                    if (!_is_local_lvalue(op_id) ||
                        _is_shared_lvalue(op_id)) {
                        _fail("outlined call reference operand must be a thread-local lvalue");
                        return;
                    }
                } else if (tag == xir::DerivedArgumentTag::RESOURCE) {
                    if (_is_local_lvalue(op_id) ||
                        op_val->value_class !=
                            schedule::ValueClass::warp_uniform) {
                        _fail("outlined call resource operand must be warp-uniform");
                        return;
                    }
                } else if (tag == xir::DerivedArgumentTag::VALUE) {
                    if (_is_local_lvalue(op_id) ||
                        op_val->value_class == schedule::ValueClass::mask ||
                        op_val->value_class == schedule::ValueClass::token ||
                        (callee.parameter_value_classes[i] ==
                             schedule::ValueClass::warp_uniform &&
                         op_val->value_class !=
                             schedule::ValueClass::warp_uniform)) {
                        _fail("outlined call value operand has an incompatible class");
                        return;
                    }
                } else {
                    _fail("outlined call has an invalid parameter argument tag");
                    return;
                }
            }
        }
    }
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_outlined_call(
    const schedule::Instruction &instruction) {
    if (!instruction.outlined_callable_id ||
        *instruction.outlined_callable_id >= _outlined_callables.size()) {
        _fail("invalid outlined callable ID during LLVM emission");
        return nullptr;
    }
    auto &&callee = _outlined_callables[*instruction.outlined_callable_id];
    auto &context = _module.getContext();
    auto *active_bits = _bindless_callback_mask(true);
    auto has_io_fields = callee.io_frame_type->getNumElements() != 0u;
    auto *io_frame_alloca =
        has_io_fields ?
            _entry_scratch(
                callee.io_frame_type,
                "outlined.call.frame." +
                    std::to_string(*instruction.outlined_callable_id)) :
            nullptr;
    if (io_frame_alloca != nullptr) {
        io_frame_alloca->setAlignment(::llvm::Align{64u});
    }
    auto *io_frame_ptr =
        io_frame_alloca != nullptr ?
            static_cast<::llvm::Value *>(io_frame_alloca) :
            static_cast<::llvm::Value *>(::llvm::ConstantPointerNull::get(
                ::llvm::PointerType::getUnqual(context)));

    // Pre-load operands before branching so any uniform/local expressions stay
    // available in the dominating block while frame stores happen only when at
    // least one lane is active.
    std::vector<::llvm::Value *> loaded_operands;
    loaded_operands.reserve(instruction.operands.size());
    for (auto i = size_t{0u}; i < instruction.operands.size(); i++) {
        auto op_id = instruction.operands[i];
        auto tag = static_cast<xir::DerivedArgumentTag>(
            callee.parameter_argument_tags[i]);
        if (tag == xir::DerivedArgumentTag::REFERENCE) {
            auto expected_cell = i < callee.parameter_soa_cell_bytes.size() ?
                                     callee.parameter_soa_cell_bytes[i] :
                                     uint8_t{0u};
            auto actual_kind = _local_soa_kind_of(op_id);
            if (expected_cell != 0u) {
                if (actual_kind != LocalSoAKind::uniform) {
                    _fail("outlined call SoA reference operand layout mismatch");
                    return nullptr;
                }
            } else if (actual_kind != LocalSoAKind::none) {
                _fail("outlined call reference operand cannot use packet-interleaved private arrays");
                return nullptr;
            }
        }
        auto *raw = _load_value(op_id);
        if (raw == nullptr) { return nullptr; }
        if (tag == xir::DerivedArgumentTag::VALUE &&
            callee.parameter_value_classes[i] !=
                schedule::ValueClass::warp_uniform) {
            auto *op_val = _source.value(op_id);
            raw = _as_lane_vector(raw, *op_val);
            if (raw == nullptr) { return nullptr; }
        }
        loaded_operands.emplace_back(raw);
    }

    auto sparse_max_lanes =
        (_width > 1u && callee.entry_w1 != nullptr &&
         callee.io_frame_type_w1 != nullptr) ?
            (callee.prefer_w1_all_lanes ? _width :
                                          simd_sparse_w1_outlined_max_lanes()) :
            0u;

    auto *pre_block = _builder.GetInsertBlock();
    auto *invoke_block = ::llvm::BasicBlock::Create(
        context, "outlined.call.invoke", _entry);
    auto *done_block = ::llvm::BasicBlock::Create(
        context, "outlined.call.done", _entry);
    auto *any_active = _builder.CreateICmpNE(
        active_bits, _builder.getInt64(0u), "outlined.call.any");

    ::llvm::BasicBlock *sparse_w1_setup = nullptr;
    if (sparse_max_lanes > 0u) {
        auto *dispatch_block = ::llvm::BasicBlock::Create(
            context, "outlined.call.dispatch", _entry);
        sparse_w1_setup = ::llvm::BasicBlock::Create(
            context, "outlined.call.w1.setup", _entry);
        _builder.CreateCondBr(any_active, dispatch_block, done_block);
        _builder.SetInsertPoint(dispatch_block);
        auto *ctpop =
#if LLVM_VERSION_MAJOR >= 22
            ::llvm::Intrinsic::getOrInsertDeclaration(
#else
            ::llvm::Intrinsic::getDeclaration(
#endif
                &_module, ::llvm::Intrinsic::ctpop, {_builder.getInt64Ty()});
        auto *popcnt = _builder.CreateCall(
            ctpop, {active_bits}, "outlined.call.popcnt");
        auto *use_w1 = _builder.CreateICmpULE(
            popcnt, _builder.getInt64(sparse_max_lanes),
            "outlined.call.use.w1");
        _builder.CreateCondBr(use_w1, sparse_w1_setup, invoke_block);
    } else {
        _builder.CreateCondBr(any_active, invoke_block, done_block);
    }

    _builder.SetInsertPoint(invoke_block);
    if (io_frame_alloca != nullptr) {
        _builder.CreateLifetimeStart(io_frame_alloca);
    }
    for (auto i = size_t{0u}; i < loaded_operands.size(); i++) {
        auto *expected_type = callee.io_frame_type->getElementType(
            static_cast<unsigned>(i));
        if (loaded_operands[i]->getType() != expected_type) {
            _fail("outlined call frame operand LLVM type mismatch");
            return nullptr;
        }
        auto *field_ptr = _builder.CreateStructGEP(
            callee.io_frame_type, io_frame_alloca,
            static_cast<unsigned>(i));
        _builder.CreateStore(loaded_operands[i], field_ptr);
    }
    ::llvm::Type *ret_llvm_type = nullptr;
    ::llvm::Value *ret_field_ptr = nullptr;
    if (callee.return_type != nullptr) {
        auto ret_index = static_cast<unsigned>(loaded_operands.size());
        ret_llvm_type = callee.io_frame_type->getElementType(ret_index);
        ret_field_ptr = _builder.CreateStructGEP(
            callee.io_frame_type, io_frame_alloca, ret_index,
            "outlined.call.ret.slot");
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(ret_llvm_type), ret_field_ptr);
    }

    ::llvm::CallInst *call = nullptr;
    if (_width == 1u && _outlined_soa_lane != nullptr) {
        call = _builder.CreateCall(
            callee.entry,
            {_builder.getInt32(_width), active_bits, _launch_config,
             io_frame_ptr, _outlined_soa_lane});
    } else {
        call = _builder.CreateCall(
            callee.entry,
            {_builder.getInt32(_width), active_bits, _launch_config,
             io_frame_ptr});
    }
    call->setCallingConv(::llvm::CallingConv::Fast);

    ::llvm::Value *loaded_ret = nullptr;
    if (callee.return_type != nullptr) {
        loaded_ret = _builder.CreateLoad(
            ret_llvm_type, ret_field_ptr, "outlined.call.ret");
    }
    if (io_frame_alloca != nullptr) {
        _builder.CreateLifetimeEnd(io_frame_alloca);
    }
    auto *invoke_end_block = _builder.GetInsertBlock();
    _builder.CreateBr(done_block);

    ::llvm::BasicBlock *sparse_w1_done_block = nullptr;
    ::llvm::Value *sparse_w1_ret_value = nullptr;
    if (sparse_w1_setup != nullptr) {
        auto has_w1_io_fields =
            callee.io_frame_type_w1->getNumElements() != 0u;
        auto *io_frame_w1_alloca =
            has_w1_io_fields ?
                _entry_scratch(
                    callee.io_frame_type_w1,
                    "outlined.call.w1.frame." +
                        std::to_string(*instruction.outlined_callable_id)) :
                nullptr;
        if (io_frame_w1_alloca != nullptr) {
            io_frame_w1_alloca->setAlignment(::llvm::Align{64u});
        }
        auto *io_frame_w1_ptr =
            io_frame_w1_alloca != nullptr ?
                static_cast<::llvm::Value *>(io_frame_w1_alloca) :
                static_cast<::llvm::Value *>(::llvm::ConstantPointerNull::get(
                    ::llvm::PointerType::getUnqual(context)));

        ::llvm::AllocaInst *w1_launch_alloca = nullptr;
        ::llvm::Value *w1_launch_ptr = _launch_config;
        ::llvm::Value *orig_thread_index = nullptr;
        ::llvm::Value *w1_thread_index_ptr = nullptr;
        if (callee.uses_lane_launch_index) {
            auto *launch_bytes_type = ::llvm::ArrayType::get(
                _builder.getInt8Ty(), sizeof(SIMDPacketLaunchConfig));
            w1_launch_alloca = _entry_scratch(
                launch_bytes_type,
                "outlined.call.w1.launch." +
                    std::to_string(*instruction.outlined_callable_id));
            w1_launch_alloca->setAlignment(
                ::llvm::Align{alignof(SIMDPacketLaunchConfig)});
            w1_launch_ptr = w1_launch_alloca;
        }

        auto pack_w1_varying =
            [&](auto &&self, ::llvm::Value *scalar_val, const Type *type,
                ::llvm::Type *w1_type) -> ::llvm::Value * {
            if (_is_scalar_data(type)) {
                return _builder.CreateInsertElement(
                    ::llvm::PoisonValue::get(w1_type), scalar_val,
                    _builder.getInt32(0u));
            }
            ::llvm::Value *agg = ::llvm::PoisonValue::get(w1_type);
            for (auto c = uint32_t{0u}; c < _child_count(type); c++) {
                auto *child_scalar = _extract_child(scalar_val, type, c, false);
                auto *child_w1_type =
                    type->is_structure() ?
                        ::llvm::cast<::llvm::StructType>(w1_type)
                            ->getElementType(c) :
                        ::llvm::cast<::llvm::ArrayType>(w1_type)
                            ->getElementType();
                auto *child_w1 = self(
                    self, child_scalar, _child_type(type, c), child_w1_type);
                agg = _builder.CreateInsertValue(agg, child_w1, {c});
            }
            return agg;
        };

        auto *sparse_w1_loop = ::llvm::BasicBlock::Create(
            context, "outlined.call.w1.loop", _entry);
        sparse_w1_done_block = ::llvm::BasicBlock::Create(
            context, "outlined.call.w1.done", _entry);

        _builder.SetInsertPoint(sparse_w1_setup);
        if (io_frame_w1_alloca != nullptr) {
            _builder.CreateLifetimeStart(io_frame_w1_alloca);
        }
        if (w1_launch_alloca != nullptr) {
            _builder.CreateLifetimeStart(w1_launch_alloca);
            auto launch_align = ::llvm::Align{alignof(SIMDPacketLaunchConfig)};
            _builder.CreateMemCpy(
                w1_launch_alloca, launch_align, _launch_config, launch_align,
                sizeof(SIMDPacketLaunchConfig));
            auto *orig_ti_ptr = _builder.CreateConstInBoundsGEP1_64(
                _builder.getInt8Ty(), _launch_config,
                offsetof(SIMDPacketLaunchConfig, thread_index));
            orig_thread_index = _builder.CreateLoad(
                _builder.getInt32Ty(), orig_ti_ptr, "outlined.call.w1.orig.ti");
            w1_thread_index_ptr = _builder.CreateConstInBoundsGEP1_64(
                _builder.getInt8Ty(), w1_launch_alloca,
                offsetof(SIMDPacketLaunchConfig, thread_index));
        }

        // Store lane-invariant arguments once before the sparse lane loop.
        for (auto i = size_t{0u}; i < loaded_operands.size(); i++) {
            auto tag = static_cast<xir::DerivedArgumentTag>(
                callee.parameter_argument_tags[i]);
            auto expected_cell = i < callee.parameter_soa_cell_bytes.size() ?
                                     callee.parameter_soa_cell_bytes[i] :
                                     uint8_t{0u};
            auto is_invariant =
                tag == xir::DerivedArgumentTag::RESOURCE ||
                (tag == xir::DerivedArgumentTag::VALUE &&
                 callee.parameter_value_classes[i] ==
                     schedule::ValueClass::warp_uniform) ||
                (tag == xir::DerivedArgumentTag::REFERENCE &&
                 expected_cell != 0u);
            if (!is_invariant) { continue; }
            auto *expected_w1_type = callee.io_frame_type_w1->getElementType(
                static_cast<unsigned>(i));
            if (loaded_operands[i]->getType() != expected_w1_type) {
                _fail("outlined W=1 sparse call invariant operand type mismatch");
                return nullptr;
            }
            auto *field_ptr = _builder.CreateStructGEP(
                callee.io_frame_type_w1, io_frame_w1_alloca,
                static_cast<unsigned>(i));
            _builder.CreateStore(loaded_operands[i], field_ptr);
        }
        _builder.CreateBr(sparse_w1_loop);

        _builder.SetInsertPoint(sparse_w1_loop);
        auto *rem_bits = _builder.CreatePHI(
            _builder.getInt64Ty(), 2u, "outlined.call.w1.bits");
        rem_bits->addIncoming(active_bits, sparse_w1_setup);
        ::llvm::PHINode *ret_acc = nullptr;
        if (callee.return_type != nullptr) {
            ret_acc = _builder.CreatePHI(
                ret_llvm_type, 2u, "outlined.call.w1.ret.acc");
            ret_acc->addIncoming(
                ::llvm::Constant::getNullValue(ret_llvm_type), sparse_w1_setup);
        }
        auto *cttz =
#if LLVM_VERSION_MAJOR >= 22
            ::llvm::Intrinsic::getOrInsertDeclaration(
#else
            ::llvm::Intrinsic::getDeclaration(
#endif
                &_module, ::llvm::Intrinsic::cttz, {_builder.getInt64Ty()});
        auto *lane64 = _builder.CreateCall(
            cttz, {rem_bits, _builder.getTrue()}, "outlined.call.w1.lane64");
        auto *lane32 = _builder.CreateTrunc(
            lane64, _builder.getInt32Ty(), "outlined.call.w1.lane");

        if (w1_thread_index_ptr != nullptr) {
            auto *lane_ti = _builder.CreateAdd(orig_thread_index, lane32);
            _builder.CreateStore(lane_ti, w1_thread_index_ptr);
        }

        // Pack per-lane varying values and AoS reference handles.
        for (auto i = size_t{0u}; i < loaded_operands.size(); i++) {
            auto tag = static_cast<xir::DerivedArgumentTag>(
                callee.parameter_argument_tags[i]);
            auto expected_cell = i < callee.parameter_soa_cell_bytes.size() ?
                                     callee.parameter_soa_cell_bytes[i] :
                                     uint8_t{0u};
            auto *expected_w1_type = callee.io_frame_type_w1->getElementType(
                static_cast<unsigned>(i));
            if (tag == xir::DerivedArgumentTag::VALUE &&
                callee.parameter_value_classes[i] !=
                    schedule::ValueClass::warp_uniform) {
                auto *lane_scalar = _extract_lane(
                    loaded_operands[i], callee.parameter_types[i], lane32);
                auto *packed = pack_w1_varying(
                    pack_w1_varying, lane_scalar, callee.parameter_types[i],
                    expected_w1_type);
                auto *field_ptr = _builder.CreateStructGEP(
                    callee.io_frame_type_w1, io_frame_w1_alloca,
                    static_cast<unsigned>(i));
                _builder.CreateStore(packed, field_ptr);
            } else if (tag == xir::DerivedArgumentTag::REFERENCE &&
                       expected_cell == 0u) {
                auto *lane_base = _builder.CreateExtractElement(
                    _local_base(_builder, loaded_operands[i]), lane32);
                auto *lane_off = _builder.CreateExtractElement(
                    _local_offsets(_builder, loaded_operands[i]), lane32);
                auto *w1_handle_struct =
                    ::llvm::cast<::llvm::StructType>(expected_w1_type);
                auto *v_base = _builder.CreateInsertElement(
                    ::llvm::PoisonValue::get(
                        w1_handle_struct->getElementType(0u)),
                    lane_base, _builder.getInt32(0u));
                auto *v_off = _builder.CreateInsertElement(
                    ::llvm::PoisonValue::get(
                        w1_handle_struct->getElementType(1u)),
                    lane_off, _builder.getInt32(0u));
                ::llvm::Value *w1_handle =
                    ::llvm::PoisonValue::get(w1_handle_struct);
                w1_handle = _builder.CreateInsertValue(w1_handle, v_base, {0u});
                w1_handle = _builder.CreateInsertValue(w1_handle, v_off, {1u});
                auto *field_ptr = _builder.CreateStructGEP(
                    callee.io_frame_type_w1, io_frame_w1_alloca,
                    static_cast<unsigned>(i));
                _builder.CreateStore(w1_handle, field_ptr);
            }
        }

        ::llvm::Type *w1_ret_llvm_type = nullptr;
        ::llvm::Value *w1_ret_field_ptr = nullptr;
        if (callee.return_type != nullptr) {
            auto ret_index = static_cast<unsigned>(loaded_operands.size());
            w1_ret_llvm_type = callee.io_frame_type_w1->getElementType(ret_index);
            w1_ret_field_ptr = _builder.CreateStructGEP(
                callee.io_frame_type_w1, io_frame_w1_alloca, ret_index,
                "outlined.call.w1.ret.slot");
            _builder.CreateStore(
                ::llvm::Constant::getNullValue(w1_ret_llvm_type),
                w1_ret_field_ptr);
        }

        auto *w1_call = _builder.CreateCall(
            callee.entry_w1,
            {_builder.getInt32(1u), _builder.getInt64(1u), w1_launch_ptr,
             io_frame_w1_ptr, lane32});
        w1_call->setCallingConv(::llvm::CallingConv::Fast);

        ::llvm::Value *next_ret_acc = nullptr;
        if (callee.return_type != nullptr) {
            auto *w1_ret = _builder.CreateLoad(
                w1_ret_llvm_type, w1_ret_field_ptr, "outlined.call.w1.ret");
            auto *scalar_ret = _extract_lane(
                w1_ret, callee.return_type, _builder.getInt32(0u));
            next_ret_acc = _insert_lane(
                ret_acc, scalar_ret, callee.return_type, lane32);
            if (next_ret_acc == nullptr) { return nullptr; }
        }
        auto *next_bits = _builder.CreateAnd(
            rem_bits,
            _builder.CreateSub(rem_bits, _builder.getInt64(1u)),
            "outlined.call.w1.bits.next");
        auto *loop_end_block = _builder.GetInsertBlock();
        rem_bits->addIncoming(next_bits, loop_end_block);
        if (ret_acc != nullptr) {
            ret_acc->addIncoming(next_ret_acc, loop_end_block);
        }
        auto *has_more = _builder.CreateICmpNE(
            next_bits, _builder.getInt64(0u), "outlined.call.w1.more");
        _builder.CreateCondBr(has_more, sparse_w1_loop, sparse_w1_done_block);

        _builder.SetInsertPoint(sparse_w1_done_block);
        if (io_frame_w1_alloca != nullptr) {
            _builder.CreateLifetimeEnd(io_frame_w1_alloca);
        }
        if (w1_launch_alloca != nullptr) {
            _builder.CreateLifetimeEnd(w1_launch_alloca);
        }
        sparse_w1_ret_value = next_ret_acc;
        _builder.CreateBr(done_block);
    }

    _builder.SetInsertPoint(done_block);
    if (callee.return_type == nullptr) {
        return nullptr;
    }
    auto *phi = _builder.CreatePHI(
        ret_llvm_type, sparse_w1_done_block != nullptr ? 3u : 2u,
        "outlined.call.result");
    phi->addIncoming(loaded_ret, invoke_end_block);
    phi->addIncoming(
        ::llvm::Constant::getNullValue(ret_llvm_type), pre_block);
    if (sparse_w1_done_block != nullptr) {
        phi->addIncoming(sparse_w1_ret_value, sparse_w1_done_block);
    }
    return phi;
}

}// namespace luisa::compute::simd::detail
