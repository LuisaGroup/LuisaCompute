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

}// namespace

void ScheduleEmitter::_analyze_ray_query_scratch() {
    constexpr auto invalid = std::numeric_limits<uint32_t>::max();
    auto value_count = _source.values().size();
    _ray_query_scratch_slots.assign(value_count, invalid);
    _ray_query_status_slots.assign(value_count, invalid);
    _ray_query_compact_surface_filter_state.assign(
        value_count, uint8_t{0u});
    _ray_query_output_only_empty_surface_filter_state.assign(
        value_count, uint8_t{0u});
    _ray_query_direct_output_surface_filter_state.assign(
        value_count, uint8_t{0u});
    _ray_query_status_storage.clear();
    _ray_query_status_callback_storage.clear();
    _ray_query_pipeline_callback_storage.clear();
    _ray_query_surface_filter_pipeline_callback_storage.clear();
    _ray_query_empty_surface_filter_pipeline_callback_storage.clear();
    _ray_query_empty_surface_filter_accel_storage.clear();
    _ray_query_direct_output_surface_filter_pipeline_callback_storage.clear();
    _ray_query_direct_output_surface_filter_accel_storage.clear();
    _ray_query_output_packet_storage.clear();
    _ray_query_surface_filter_ray_packet_storage.clear();
    _ray_query_surface_filter_ray_packet_call_storage.clear();
    _ray_query_state_handle_storage.clear();

    std::vector<schedule::ValueId> constructions;
    std::vector<uint32_t> construction_for_value(value_count, invalid);
    auto has_pipeline = false;
    auto has_surface_filter_pipeline = false;
    auto has_output_only_empty_surface_filter_pipeline = false;
    auto has_direct_output_surface_filter_pipeline = false;
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            has_pipeline |= instruction.opcode ==
                            schedule::Opcode::ray_query_pipeline;
            if (instruction.opcode ==
                    schedule::Opcode::ray_query_pipeline &&
                instruction.source_op &&
                *instruction.source_op <
                    _ray_query_pipeline_handlers.size()) {
                auto handlers =
                    _ray_query_pipeline_handlers[*instruction.source_op];
                has_surface_filter_pipeline |=
                    handlers.embree_surface_filter_safe;
                has_output_only_empty_surface_filter_pipeline |=
                    handlers.embree_surface_filter_safe &&
                    handlers.surface_handler_empty;
                has_direct_output_surface_filter_pipeline |=
                    handlers.embree_surface_filter_safe &&
                    !handlers.surface_handler_empty;
            }
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
        if (_width == 1u && has_pipeline) {
            _ray_query_pipeline_callback_storage.assign(
                status_slot_count, nullptr);
        }
        if (_width != 1u && has_surface_filter_pipeline) {
            _ray_query_surface_filter_pipeline_callback_storage.assign(
                status_slot_count, nullptr);
            if (_width >= 4u ||
                (_width == 2u &&
                 (has_output_only_empty_surface_filter_pipeline ||
                  has_direct_output_surface_filter_pipeline))) {
                _ray_query_surface_filter_ray_packet_storage.assign(
                    status_slot_count, nullptr);
                _ray_query_surface_filter_ray_packet_call_storage.assign(
                    status_slot_count, nullptr);
            }
        }
        if (_width >= 2u &&
            has_output_only_empty_surface_filter_pipeline) {
            _ray_query_empty_surface_filter_pipeline_callback_storage.assign(
                status_slot_count, nullptr);
            _ray_query_empty_surface_filter_accel_storage.assign(
                status_slot_count, nullptr);
        }
        if (_width >= 2u &&
            has_direct_output_surface_filter_pipeline) {
            _ray_query_direct_output_surface_filter_pipeline_callback_storage.assign(
                status_slot_count, nullptr);
            _ray_query_direct_output_surface_filter_accel_storage.assign(
                status_slot_count, nullptr);
        }
        if (_width >= 2u &&
            (has_output_only_empty_surface_filter_pipeline ||
             has_direct_output_surface_filter_pipeline)) {
            _ray_query_output_packet_storage.assign(
                status_slot_count, nullptr);
        }
        _result.ray_query_status_slot_count = status_slot_count;
        // The status proof already establishes one published local owner per
        // active lane and non-overlapping lifetimes per color. Reuse that
        // proof for a contiguous packet of state handles. Unproven queries,
        // W1, and ordinary W2 queries retain their authoritative local
        // gathers; output-only W2 may also use the logical ray packet below.
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

    // A nonnull surface-filter provider completes the complete query inside
    // one runtime call and touches no batch/object-ray fields. Prove that fact
    // per query variable rather than per function: a module may contain both
    // eligible and ordinary query sites sharing the same acceleration table.
    if (_width >= 2u &&
        !_ray_query_surface_filter_pipeline_callback_storage.empty()) {
        std::vector<uint8_t> eligible(
            variable_count, uint8_t{1u});
        std::vector<uint8_t> output_only_eligible(
            variable_count, uint8_t{1u});
        std::vector<uint8_t> direct_output_eligible(
            variable_count, uint8_t{1u});
        std::vector<uint32_t> pipeline_count(variable_count, 0u);
        std::vector<uint32_t> empty_pipeline_count(
            variable_count, 0u);
        std::vector<uint32_t> direct_pipeline_count(
            variable_count, 0u);
        for (auto &&block : _source.blocks()) {
            for (auto &&instruction : block.instructions) {
                auto variable = use_variable_for(instruction);
                if (variable == invalid) { continue; }
                if (instruction.opcode ==
                    schedule::Opcode::ray_query_pipeline) {
                    auto safe = instruction.source_op &&
                                *instruction.source_op <
                                    _ray_query_pipeline_handlers.size() &&
                                _ray_query_pipeline_handlers[*instruction.source_op]
                                    .embree_surface_filter_safe;
                    auto output_only_safe =
                        safe &&
                        _ray_query_pipeline_handlers[*instruction.source_op]
                            .surface_handler_empty;
                    auto direct_output_safe =
                        safe &&
                        !_ray_query_pipeline_handlers[*instruction.source_op]
                             .surface_handler_empty;
                    if (safe) {
                        pipeline_count[variable]++;
                    } else {
                        eligible[variable] = 0u;
                    }
                    if (output_only_safe) {
                        empty_pipeline_count[variable]++;
                    } else {
                        output_only_eligible[variable] = 0u;
                    }
                    if (direct_output_safe) {
                        direct_pipeline_count[variable]++;
                    } else {
                        direct_output_eligible[variable] = 0u;
                    }
                } else if (instruction.opcode ==
                           schedule::Opcode::ray_query_read) {
                    auto read_op = instruction.source_op ?
                                       static_cast<xir::RayQueryObjectReadOp>(
                                           *instruction.source_op) :
                                       xir::RayQueryObjectReadOp::
                                           RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY;
                    auto safe = instruction.source_op &&
                                read_op !=
                                    xir::RayQueryObjectReadOp::
                                        RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY;
                    if (!safe) { eligible[variable] = 0u; }
                    auto output_only_safe =
                        instruction.source_op &&
                        (read_op ==
                             xir::RayQueryObjectReadOp::
                                 RAY_QUERY_OBJECT_COMMITTED_HIT ||
                         read_op ==
                             xir::RayQueryObjectReadOp::
                                 RAY_QUERY_OBJECT_IS_TERMINATED);
                    if (!output_only_safe) {
                        output_only_eligible[variable] = 0u;
                    }
                    if (!output_only_safe) {
                        direct_output_eligible[variable] = 0u;
                    }
                } else if (instruction.opcode ==
                           schedule::Opcode::ray_query_write) {
                    // The caller-side compact form is deliberately limited to
                    // construction, one complete pipeline, and prefix reads.
                    // Candidate handler writes live in separately lowered
                    // functions and do not appear in this Schedule.
                    eligible[variable] = 0u;
                    output_only_eligible[variable] = 0u;
                    direct_output_eligible[variable] = 0u;
                }
            }
        }
        for (auto i = size_t{0u}; i < constructions.size(); i++) {
            auto variable = construction_variable[i];
            auto construction = constructions[i];
            if (variable >= variable_count) { continue; }
            auto status_slot =
                construction.value < _ray_query_status_slots.size() ?
                    _ray_query_status_slots[construction.value] :
                    invalid;
            if (_width >= 4u && eligible[variable] != 0u &&
                pipeline_count[variable] == 1u &&
                status_slot != invalid &&
                status_slot <
                    _ray_query_surface_filter_pipeline_callback_storage
                        .size()) {
                _ray_query_compact_surface_filter_state[construction.value] = 1u;
                _result.compact_surface_filter_state_count++;
            }
            if (output_only_eligible[variable] != 0u &&
                empty_pipeline_count[variable] == 1u &&
                status_slot != invalid &&
                status_slot <
                    _ray_query_empty_surface_filter_pipeline_callback_storage
                        .size() &&
                status_slot <
                    _ray_query_empty_surface_filter_accel_storage.size()) {
                _ray_query_output_only_empty_surface_filter_state
                    [construction.value] = 1u;
                _result.output_only_empty_surface_filter_state_count++;
            }
            if (direct_output_eligible[variable] != 0u &&
                direct_pipeline_count[variable] == 1u &&
                status_slot != invalid &&
                status_slot <
                    _ray_query_direct_output_surface_filter_pipeline_callback_storage
                        .size() &&
                status_slot <
                    _ray_query_direct_output_surface_filter_accel_storage
                        .size()) {
                _ray_query_direct_output_surface_filter_state
                    [construction.value] = 1u;
                _result.direct_output_surface_filter_state_count++;
            }
        }
    }
    _ray_query_scratch_storage.assign(color_count, nullptr);
    _result.ray_query_scratch_slot_count = color_count;
    _result.ray_query_scratch_bytes =
        static_cast<size_t>(color_count) * _width *
        sizeof(SIMDHostRayQueryState);
}

}// namespace luisa::compute::simd::detail
