#include "llvm_schedule_emitter.h"

#include <algorithm>
#include <limits>
#include <numeric>

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

void ScheduleEmitter::_analyze_ray_query_scratch() {
    constexpr auto invalid = std::numeric_limits<uint32_t>::max();
    auto value_count = _source.values().size();
    _ray_query_scratch_slots.assign(value_count, invalid);
    _ray_query_status_slots.assign(value_count, invalid);
    _ray_query_status_storage.clear();
    _ray_query_status_callback_storage.clear();
    _ray_query_state_handle_storage.clear();

    std::vector<schedule::ValueId> constructions;
    std::vector<uint32_t> construction_for_value(value_count, invalid);
    auto has_pipeline = false;
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            has_pipeline |= instruction.opcode ==
                            schedule::Opcode::ray_query_pipeline;
            if (!is_ray_query_construction(instruction)) { continue; }
            auto id = *instruction.result;
            construction_for_value[id.value] =
                static_cast<uint32_t>(constructions.size());
            constructions.emplace_back(id);
        }
    }
    _result.ray_query_count = constructions.size();

    auto assign_unique_slots = [&] {
        for (auto i = uint32_t{0u}; i < constructions.size(); i++) {
            _ray_query_scratch_slots[constructions[i].value] = i;
        }
        _ray_query_scratch_storage.assign(
            constructions.size(), nullptr);
        _result.ray_query_scratch_slot_count = constructions.size();
        _result.ray_query_scratch_bytes =
            constructions.size() * static_cast<size_t>(_width) *
            sizeof(SIMDHostRayQueryState);
    };
    if (constructions.empty()) {
        assign_unique_slots();
        return;
    }
    auto disable_coloring =
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_RAY_QUERY_SCRATCH_COLORING");

    // Resolve every query-typed local reference to the packet-local alloca
    // that owns it. Schedule edge assignments may copy an lvalue handle; a
    // merge of handles from different allocas is deliberately treated as an
    // unprovable alias below and disables coloring.
    std::vector<std::vector<uint32_t>> roots(value_count);
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode != schedule::Opcode::alloca ||
                !instruction.result) {
                continue;
            }
            auto *value = _source.value(*instruction.result);
            if (value != nullptr && is_ray_query_type(value->type)) {
                roots[instruction.result->value].emplace_back(
                    instruction.result->value);
            }
        }
    }
    auto merge_roots = [&](schedule::ValueId destination,
                           schedule::ValueId source) noexcept {
        if (destination.value >= roots.size() ||
            source.value >= roots.size()) {
            return false;
        }
        auto &to = roots[destination.value];
        auto old_size = to.size();
        for (auto root : roots[source.value]) {
            if (std::find(to.cbegin(), to.cend(), root) == to.cend()) {
                to.emplace_back(root);
            }
        }
        return to.size() != old_size;
    };
    auto changed = true;
    while (changed) {
        changed = false;
        for (auto &&block : _source.blocks()) {
            for (auto &&instruction : block.instructions) {
                auto *result = instruction.result ?
                                   _source.value(*instruction.result) :
                                   nullptr;
                if (instruction.opcode == schedule::Opcode::gep &&
                    instruction.result &&
                    !instruction.operands.empty() &&
                    result != nullptr &&
                    is_ray_query_type(result->type)) {
                    changed |= merge_roots(
                        *instruction.result,
                        instruction.operands.front());
                }
            }
            _for_each_assignment(
                block, [&](schedule::EdgeAssignment assignment) {
                    auto *destination =
                        _source.value(assignment.destination);
                    if (destination != nullptr &&
                        is_ray_query_type(destination->type) &&
                        _is_local_lvalue(assignment.destination)) {
                        changed |= merge_roots(
                            assignment.destination,
                            assignment.source);
                    }
                });
        }
    }

    auto safe = true;
    std::vector<std::vector<uint32_t>> construction_roots(
        constructions.size());
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            for (auto operand_index = size_t{0u};
                 operand_index < instruction.operands.size();
                 operand_index++) {
                auto operand = instruction.operands[operand_index];
                if (operand.value >= construction_for_value.size()) {
                    continue;
                }
                auto construction =
                    construction_for_value[operand.value];
                if (construction == invalid) { continue; }
                if (instruction.opcode != schedule::Opcode::store ||
                    operand_index != 1u ||
                    instruction.operands.empty()) {
                    safe = false;
                    continue;
                }
                auto destination = instruction.operands.front();
                if (destination.value >= roots.size() ||
                    roots[destination.value].size() != 1u) {
                    safe = false;
                    continue;
                }
                construction_roots[construction].emplace_back(
                    roots[destination.value].front());
            }

            // Copying an already materialized query pointer into another
            // local creates aliasing that variable liveness cannot express.
            if (instruction.opcode == schedule::Opcode::store &&
                instruction.operands.size() == 2u) {
                auto destination = instruction.operands[0u];
                auto source = instruction.operands[1u];
                auto *destination_value = _source.value(destination);
                auto source_construction =
                    source.value < construction_for_value.size() ?
                        construction_for_value[source.value] :
                        invalid;
                if (destination_value != nullptr &&
                    is_ray_query_type(destination_value->type) &&
                    source_construction == invalid) {
                    safe = false;
                }
            }

            if (instruction.opcode == schedule::Opcode::ray_query_read ||
                instruction.opcode == schedule::Opcode::ray_query_write ||
                instruction.opcode == schedule::Opcode::ray_query_pipeline) {
                if (instruction.operands.empty()) {
                    safe = false;
                } else {
                    auto object = instruction.operands.front();
                    safe &= object.value < roots.size() &&
                            roots[object.value].size() == 1u;
                }
            } else {
                for (auto operand_index = size_t{0u};
                     operand_index < instruction.operands.size();
                     operand_index++) {
                    auto operand = instruction.operands[operand_index];
                    auto *value = _source.value(operand);
                    if (value == nullptr ||
                        !is_ray_query_type(value->type) ||
                        !_is_local_lvalue(operand)) {
                        continue;
                    }
                    auto allowed_store_destination =
                        instruction.opcode == schedule::Opcode::store &&
                        operand_index == 0u;
                    if (!allowed_store_destination) { safe = false; }
                }
            }
        }
        _for_each_assignment(
            block, [&](schedule::EdgeAssignment assignment) {
                if (assignment.source.value <
                        construction_for_value.size() &&
                    construction_for_value[assignment.source.value] !=
                        invalid) {
                    safe = false;
                }
            });
    }
    std::vector<uint32_t> construction_root(
        constructions.size(), invalid);
    for (auto i = size_t{0u}; i < constructions.size(); i++) {
        auto &candidates = construction_roots[i];
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(
            std::unique(candidates.begin(), candidates.end()),
            candidates.end());
        if (candidates.size() != 1u) {
            safe = false;
        } else {
            construction_root[i] = candidates.front();
        }
    }
    if (!safe) {
        assign_unique_slots();
        return;
    }

    // The public query pointer changes at the store, not at construction.
    // Standard lowering may insert a harmless alloca between them. Permit
    // such instructions, but only cache status when every construction has
    // one later store in the same block and no query access/construction can
    // observe or overwrite the old owner in the gap. Scratch coloring keeps
    // its established liveness proof even when this stricter cache proof
    // fails; only the sidecar then falls back to AoS gathers.
    auto status_cache_safe = true;
    for (auto construction_index = size_t{0u};
         construction_index < constructions.size(); construction_index++) {
        auto construction = constructions[construction_index];
        const schedule::BasicBlock *owner_block = nullptr;
        auto definition_index = size_t{0u};
        auto store_index = size_t{0u};
        auto store_count = size_t{0u};
        for (auto &&block : _source.blocks()) {
            for (auto i = size_t{0u}; i < block.instructions.size(); i++) {
                auto &&candidate = block.instructions[i];
                if (candidate.result == construction) {
                    owner_block = &block;
                    definition_index = i;
                }
                if (candidate.opcode == schedule::Opcode::store &&
                    candidate.operands.size() == 2u &&
                    candidate.operands[1u] == construction) {
                    store_count++;
                    if (owner_block == &block) { store_index = i; }
                }
            }
        }
        if (owner_block == nullptr || store_count != 1u ||
            store_index <= definition_index) {
            status_cache_safe = false;
            continue;
        }
        auto root = construction_root[construction_index];
        for (auto i = definition_index + 1u; i < store_index; i++) {
            auto &&candidate = owner_block->instructions[i];
            if (is_ray_query_construction(candidate)) {
                status_cache_safe = false;
                break;
            }
            if ((candidate.opcode == schedule::Opcode::ray_query_read ||
                 candidate.opcode == schedule::Opcode::ray_query_write ||
                 candidate.opcode == schedule::Opcode::ray_query_pipeline) &&
                !candidate.operands.empty()) {
                auto object = candidate.operands.front();
                if (object.value < roots.size() &&
                    roots[object.value].size() == 1u &&
                    roots[object.value].front() == root) {
                    status_cache_safe = false;
                    break;
                }
            }
        }
    }

    // The scratch-coloring oracle deliberately restores the old independent
    // allocations. Keep the status cache fail-closed under the same oracle:
    // multiple constructions assigned to one query local no longer have a
    // unique sidecar owner in that mode.
    if (disable_coloring) {
        assign_unique_slots();
        return;
    }

    std::vector<uint32_t> variable_roots = construction_root;
    std::sort(variable_roots.begin(), variable_roots.end());
    variable_roots.erase(
        std::unique(variable_roots.begin(), variable_roots.end()),
        variable_roots.end());
    auto variable_count = variable_roots.size();
    std::vector<uint32_t> variable_for_root(value_count, invalid);
    for (auto i = uint32_t{0u}; i < variable_count; i++) {
        variable_for_root[variable_roots[i]] = i;
    }
    std::vector<uint32_t> construction_variable(
        constructions.size(), invalid);
    for (auto i = size_t{0u}; i < constructions.size(); i++) {
        construction_variable[i] =
            variable_for_root[construction_root[i]];
    }

    auto block_count = _source.blocks().size();
    using BitSet = std::vector<uint8_t>;
    std::vector<BitSet> block_use(
        block_count, BitSet(variable_count, uint8_t{0u}));
    std::vector<BitSet> block_def = block_use;
    auto query_variable = [&](schedule::ValueId reference) noexcept {
        if (reference.value >= roots.size() ||
            roots[reference.value].size() != 1u) {
            return invalid;
        }
        auto root = roots[reference.value].front();
        return root < variable_for_root.size() ?
                   variable_for_root[root] :
                   invalid;
    };
    auto construction_variable_for =
        [&](const schedule::Instruction &instruction) noexcept {
            if (!is_ray_query_construction(instruction)) {
                return invalid;
            }
            auto construction =
                construction_for_value[instruction.result->value];
            return construction_variable[construction];
        };
    auto use_variable_for =
        [&](const schedule::Instruction &instruction) noexcept {
            if ((instruction.opcode != schedule::Opcode::ray_query_read &&
                 instruction.opcode != schedule::Opcode::ray_query_write &&
                 instruction.opcode != schedule::Opcode::ray_query_pipeline) ||
                instruction.operands.empty()) {
                return invalid;
            }
            return query_variable(instruction.operands.front());
        };
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            if ((instruction.opcode == schedule::Opcode::ray_query_read ||
                 instruction.opcode == schedule::Opcode::ray_query_write ||
                 instruction.opcode == schedule::Opcode::ray_query_pipeline) &&
                use_variable_for(instruction) == invalid) {
                assign_unique_slots();
                return;
            }
        }
    }
    for (auto &&block : _source.blocks()) {
        auto &uses = block_use[block.id.value];
        auto &definitions = block_def[block.id.value];
        for (auto &&instruction : block.instructions) {
            auto use = use_variable_for(instruction);
            if (use != invalid && definitions[use] == 0u) {
                uses[use] = 1u;
            }
            auto definition = construction_variable_for(instruction);
            if (definition != invalid) {
                definitions[definition] = 1u;
            }
        }
    }

    std::vector<std::vector<uint32_t>> successors(block_count);
    auto add_successor = [&](schedule::BlockId source,
                             schedule::BlockId target) noexcept {
        auto &items = successors[source.value];
        if (std::find(items.cbegin(), items.cend(), target.value) ==
            items.cend()) {
            items.emplace_back(target.value);
        }
    };
    for (auto &&block : _source.blocks()) {
        std::visit(
            [&](const auto &terminator) {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    add_successor(block.id, terminator.edge.target);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SplitTerminator>) {
                    add_successor(
                        block.id, terminator.true_edge.target);
                    add_successor(
                        block.id, terminator.false_edge.target);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SwitchTerminator>) {
                    for (auto &&item : terminator.cases) {
                        add_successor(block.id, item.edge.target);
                    }
                    add_successor(
                        block.id, terminator.default_edge.target);
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator>) {
                    add_successor(
                        block.id,
                        _source.convergence(
                                   terminator.convergence)
                            ->target);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::LoopBackTerminator>) {
                    add_successor(
                        block.id,
                        _source.loop(terminator.loop)->header);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::BlockBarrierTerminator>) {
                    add_successor(
                        block.id, terminator.resume_edge.target);
                }
            },
            block.terminator);
    }

    std::vector<BitSet> live_in(
        block_count, BitSet(variable_count, uint8_t{0u}));
    std::vector<BitSet> live_out = live_in;
    changed = true;
    while (changed) {
        changed = false;
        for (auto block_index = block_count;
             block_index-- > 0u;) {
            BitSet next_out(variable_count, uint8_t{0u});
            for (auto successor : successors[block_index]) {
                for (auto variable = size_t{0u};
                     variable < variable_count; variable++) {
                    next_out[variable] |=
                        live_in[successor][variable];
                }
            }
            BitSet next_in(variable_count, uint8_t{0u});
            for (auto variable = size_t{0u};
                 variable < variable_count; variable++) {
                next_in[variable] =
                    block_use[block_index][variable] ||
                    (next_out[variable] &&
                     !block_def[block_index][variable]);
            }
            if (next_out != live_out[block_index] ||
                next_in != live_in[block_index]) {
                live_out[block_index] = std::move(next_out);
                live_in[block_index] = std::move(next_in);
                changed = true;
            }
        }
    }

    std::vector<BitSet> interference(
        variable_count, BitSet(variable_count, uint8_t{0u}));
    auto connect = [&](uint32_t a, uint32_t b) noexcept {
        if (a != b) {
            interference[a][b] = 1u;
            interference[b][a] = 1u;
        }
    };
    auto connect_live = [&](const BitSet &live) noexcept {
        for (auto a = uint32_t{0u}; a < variable_count; a++) {
            if (live[a] == 0u) { continue; }
            for (auto b = a + 1u; b < variable_count; b++) {
                if (live[b] != 0u) { connect(a, b); }
            }
        }
    };
    for (auto &&block : _source.blocks()) {
        auto live = live_out[block.id.value];
        connect_live(live);
        for (auto instruction_index = block.instructions.size();
             instruction_index-- > 0u;) {
            auto &&instruction =
                block.instructions[instruction_index];
            auto definition = construction_variable_for(instruction);
            if (definition != invalid) {
                for (auto other = uint32_t{0u};
                     other < variable_count; other++) {
                    if (live[other] != 0u) {
                        connect(definition, other);
                    }
                }
                live[definition] = 0u;
            }
            auto use = use_variable_for(instruction);
            if (use != invalid) {
                for (auto other = uint32_t{0u};
                     other < variable_count; other++) {
                    if (live[other] != 0u) { connect(use, other); }
                }
                live[use] = 1u;
            }
        }
    }

    std::vector<uint32_t> order(variable_count);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(
        order.begin(), order.end(), [&](uint32_t lhs, uint32_t rhs) {
            auto lhs_degree = static_cast<size_t>(std::count(
                interference[lhs].cbegin(),
                interference[lhs].cend(), uint8_t{1u}));
            auto rhs_degree = static_cast<size_t>(std::count(
                interference[rhs].cbegin(),
                interference[rhs].cend(), uint8_t{1u}));
            return lhs_degree != rhs_degree ?
                       lhs_degree > rhs_degree :
                       variable_roots[lhs] < variable_roots[rhs];
        });
    std::vector<uint32_t> colors(variable_count, invalid);
    auto color_count = uint32_t{0u};
    for (auto variable : order) {
        std::vector<uint8_t> unavailable(
            color_count, uint8_t{0u});
        for (auto other = uint32_t{0u};
             other < variable_count; other++) {
            if (interference[variable][other] != 0u &&
                colors[other] != invalid) {
                unavailable[colors[other]] = 1u;
            }
        }
        auto color = uint32_t{0u};
        while (color < unavailable.size() && unavailable[color] != 0u) {
            color++;
        }
        if (color == color_count) { color_count++; }
        colors[variable] = color;
    }
    for (auto i = size_t{0u}; i < constructions.size(); i++) {
        _ray_query_scratch_slots[constructions[i].value] =
            colors[construction_variable[i]];
    }
    if ((_width >= 4u || has_pipeline) && status_cache_safe &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_RAY_QUERY_STATUS_CACHE")) {
        std::vector<uint8_t> needs_status(
            variable_count, uint8_t{0u});
        for (auto &&block : _source.blocks()) {
            for (auto &&instruction : block.instructions) {
                if (!instruction.source_op ||
                    instruction.operands.empty()) {
                    continue;
                }
                auto uses_status = false;
                if (instruction.opcode ==
                    schedule::Opcode::ray_query_read) {
                    auto op = static_cast<xir::RayQueryObjectReadOp>(
                        *instruction.source_op);
                    uses_status =
                        op == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED ||
                        op == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE ||
                        op == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE;
                } else if (instruction.opcode ==
                           schedule::Opcode::ray_query_write) {
                    auto op = static_cast<xir::RayQueryObjectWriteOp>(
                        *instruction.source_op);
                    uses_status =
                        op == xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE ||
                        op == xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL;
                } else if (instruction.opcode ==
                           schedule::Opcode::ray_query_pipeline) {
                    uses_status = true;
                }
                auto variable = use_variable_for(instruction);
                if (uses_status && variable != invalid) {
                    needs_status[variable] = 1u;
                }
            }
        }
        std::vector<uint32_t> status_slot_for_color(
            color_count, invalid);
        auto status_slot_count = uint32_t{0u};
        for (auto variable = uint32_t{0u};
             variable < variable_count; variable++) {
            if (needs_status[variable] == 0u) { continue; }
            auto color = colors[variable];
            if (status_slot_for_color[color] == invalid) {
                status_slot_for_color[color] = status_slot_count++;
            }
        }
        for (auto i = size_t{0u}; i < constructions.size(); i++) {
            auto variable = construction_variable[i];
            if (needs_status[variable] != 0u) {
                _ray_query_status_slots[constructions[i].value] =
                    status_slot_for_color[colors[variable]];
            }
        }
        for (auto value = uint32_t{0u}; value < roots.size(); value++) {
            auto variable = query_variable(schedule::ValueId{value});
            if (variable != invalid && needs_status[variable] != 0u) {
                _ray_query_status_slots[value] =
                    status_slot_for_color[colors[variable]];
            }
        }
        _ray_query_status_storage.assign(status_slot_count, nullptr);
        _ray_query_status_callback_storage.assign(status_slot_count, nullptr);
        _result.ray_query_status_slot_count = status_slot_count;
        // The status proof already establishes one published local owner per
        // active lane and non-overlapping lifetimes per color. Reuse that
        // proof for a contiguous packet of state handles; unproven queries and
        // W1/W2 retain their authoritative local gathers.
        if (!luisa::compute::detail::env_flag(
                "LUISA_SIMD_DISABLE_RAY_QUERY_STATE_HANDLE_CACHE")) {
            _ray_query_state_handle_storage.assign(
                status_slot_count, nullptr);
            _result.ray_query_state_handle_slot_count = status_slot_count;
        }
    } else {
        std::fill(
            _ray_query_status_slots.begin(),
            _ray_query_status_slots.end(), invalid);
    }
    _ray_query_scratch_storage.assign(color_count, nullptr);
    _result.ray_query_scratch_slot_count = color_count;
    _result.ray_query_scratch_bytes =
        static_cast<size_t>(color_count) * _width *
        sizeof(SIMDHostRayQueryState);
}

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
    auto *plain_proceed = _builder.CreateExtractValue(
        accel, {_width >= 8u ? 5u : 4u});
    auto cache_status =
        _ray_query_status_slot(*instruction.result) != nullptr;
    auto *null_pointer = ::llvm::ConstantPointerNull::get(pointer_type);
    auto *status_proceed = static_cast<::llvm::Value *>(nullptr);
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
    }
    auto *missing_callback = _builder.CreateOr(
        _builder.CreateICmpEQ(object, null_pointer),
        _builder.CreateICmpEQ(plain_proceed, null_pointer));
    if (cache_status) {
        missing_callback = _builder.CreateOr(
            missing_callback,
            _builder.CreateICmpEQ(status_proceed, null_pointer));
    }
    _trap_if(
        _builder.CreateAnd(
            _builder.CreateOrReduce(_active_mask), missing_callback),
        "accel.ray.query.callback.null");
    scatter(
        object, offsetof(SIMDHostRayQueryState, accel), alignof(void *));
    scatter(
        plain_proceed, offsetof(SIMDHostRayQueryState, proceed), alignof(void *));
    if (cache_status) {
        auto status_slot = _ray_query_status_slots[instruction.result->value];
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
    }

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
    auto packed_init =
        _width >= 4u &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_RAY_QUERY_PACKED_INIT");
    auto *i64_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt64Ty(), _width);
    auto *zero_i64 = ::llvm::Constant::getNullValue(i64_lanes);
    if (packed_init) {
        scatter(
            zero_i64, offsetof(SIMDHostRayQueryState, candidate_kind),
            alignof(uint32_t));
        scatter(
            zero_i64, offsetof(SIMDHostRayQueryState, terminated),
            alignof(uint32_t));
    } else {
        scatter(
            zero_i32, offsetof(SIMDHostRayQueryState, candidate_kind),
            alignof(uint32_t));
        scatter(
            zero_i32,
            offsetof(SIMDHostRayQueryState, candidate_committed),
            alignof(uint32_t));
        scatter(
            zero_i32, offsetof(SIMDHostRayQueryState, terminated),
            alignof(uint32_t));
        scatter(
            zero_i32,
            offsetof(SIMDHostRayQueryState, procedural_cursor_valid),
            alignof(uint32_t));
    }
    auto eager_batch_init =
        _width == 2u ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_RAY_QUERY_LAZY_BATCH_INIT");
    // The initialized fields gate every runtime read of the six metadata
    // fields below. The first scan clears them before publishing both gates,
    // so eager construction stores are redundant at accepted widths.
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
        offsetof(SIMDHostRayQueryState, candidate_batch_initialized),
        alignof(uint32_t));
    scatter(
        zero_i32,
        offsetof(SIMDHostRayQueryState, procedural_batch_initialized),
        alignof(uint32_t));

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
    auto *states = _ray_query_state_handles(object_id);
    if (status_slot == nullptr || states == nullptr) {
        _fail("ray-query pipeline requires a proven status/state owner");
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
    auto *null_pointer =
        ::llvm::ConstantPointerNull::get(pointer_type);
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
    auto *safe_states = _builder.CreateSelect(
        _active_mask, states,
        ::llvm::Constant::getNullValue(states->getType()));
    auto *state_store = _builder.CreateStore(safe_states, scratch);
    state_store->setAlignment(::llvm::Align{alignof(void *)});

    auto *outer_active_bits = _bindless_callback_mask(true);
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
    auto *preheader = _builder.GetInsertBlock();
    auto *loop = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.loop", _entry);
    auto *surface_call = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.surface", _entry);
    auto *after_surface = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.after.surface", _entry);
    auto *procedural_call = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.procedural", _entry);
    auto *continue_loop = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.continue", _entry);
    auto *exit = ::llvm::BasicBlock::Create(
        context, "ray.query.pipeline.exit", _entry);
    _builder.CreateBr(loop);

    _builder.SetInsertPoint(loop);
    auto *active_bits = _builder.CreatePHI(
        _builder.getInt64Ty(), 2u,
        "ray.query.pipeline.active.bits");
    active_bits->addIncoming(outer_active_bits, preheader);
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

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_ray_query_read(
    const schedule::Instruction &instruction) {
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
