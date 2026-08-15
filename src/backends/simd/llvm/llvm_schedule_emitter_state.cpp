#include "llvm_schedule_emitter.h"

#include <algorithm>
#include <numeric>

#include "../../common/env_flag.h"

namespace luisa::compute::simd::detail {

void ScheduleEmitter::_coalesce_state_slots() {
    // Direct control flow already exposes ordinary SSA to LLVM. The packet
    // scheduler instead materializes XIR PHIs as masked state copies. Reuse a
    // physical slot first for move-related state values whose per-lane live
    // ranges do not interfere. Value names select profitable copies from the
    // same destructured XIR register. A separate W16 high-pressure pass may
    // then color compatible non-move live ranges into the same slot. Liveness,
    // not either heuristic, is the safety proof.
    if (_direct_control_flow || _width < 2u ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_STATE_PHI_COALESCING")) {
        return;
    }

    using BitSet = std::vector<uint8_t>;
    struct StateEdge {
        schedule::BlockId target{};
        const std::vector<schedule::EdgeAssignment> *assignments{nullptr};
    };

    static constexpr auto invalid_state_index = ~uint32_t{0u};
    auto value_count = _source.values().size();
    auto block_count = _source.blocks().size();
    std::vector<uint32_t> state_indices(
        value_count, invalid_state_index);
    std::vector<schedule::ValueId> state_values;
    state_values.reserve(_result.state_slot_count);
    for (auto &&value : _source.values()) {
        if (value.origin == schedule::ValueOrigin::state_slot &&
            value.id.value < _state_slots.size() &&
            _state_slots[value.id.value] != nullptr) {
            state_indices[value.id.value] =
                static_cast<uint32_t>(state_values.size());
            state_values.emplace_back(value.id);
        }
    }
    auto state_count = state_values.size();
    auto is_state = [&](schedule::ValueId id) noexcept {
        return id.value < state_indices.size() &&
               state_indices[id.value] != invalid_state_index;
    };
    auto state_index = [&](schedule::ValueId id) noexcept {
        return state_indices[id.value];
    };

    std::vector<std::vector<StateEdge>> edges(block_count);
    auto add_edge = [&](schedule::BlockId source,
                        const schedule::ControlEdge &edge) noexcept {
        edges[source.value].emplace_back(
            StateEdge{edge.target, &edge.assignments});
    };
    for (auto &&block : _source.blocks()) {
        std::visit(
            [&](const auto &terminator) noexcept {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    add_edge(block.id, terminator.edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SplitTerminator>) {
                    add_edge(block.id, terminator.true_edge);
                    add_edge(block.id, terminator.false_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SwitchTerminator>) {
                    for (auto &&item : terminator.cases) {
                        add_edge(block.id, item.edge);
                    }
                    add_edge(block.id, terminator.default_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator>) {
                    auto *point = _source.convergence(
                        terminator.convergence);
                    edges[block.id.value].emplace_back(
                        StateEdge{point->target, &terminator.assignments});
                } else if constexpr (std::is_same_v<
                                         T, schedule::LoopBackTerminator>) {
                    auto *loop = _source.loop(terminator.loop);
                    edges[block.id.value].emplace_back(
                        StateEdge{loop->header, &terminator.assignments});
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::BlockBarrierTerminator>) {
                    add_edge(block.id, terminator.resume_edge);
                }
            },
            block.terminator);
    }

    std::vector<BitSet> block_uses(
        block_count, BitSet(state_count, uint8_t{0u}));
    auto mark_use = [&](BitSet &uses, schedule::ValueId id) noexcept {
        if (is_state(id)) { uses[state_index(id)] = 1u; }
    };
    for (auto &&block : _source.blocks()) {
        auto &uses = block_uses[block.id.value];
        for (auto &&instruction : block.instructions) {
            for (auto operand : instruction.operands) {
                mark_use(uses, operand);
            }
            if (instruction.participant_mask) {
                mark_use(uses, *instruction.participant_mask);
            }
        }
        std::visit(
            [&](const auto &terminator) noexcept {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<
                                  T, schedule::SplitTerminator>) {
                    mark_use(uses, terminator.condition);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SwitchTerminator>) {
                    mark_use(uses, terminator.selector);
                } else if constexpr (std::is_same_v<
                                         T, schedule::ReturnTerminator>) {
                    if (terminator.value) {
                        mark_use(uses, *terminator.value);
                    }
                }
            },
            block.terminator);
    }

    // Edge assignments are parallel PHI copies. For one logical lane, the
    // transfer is uses(edge) U (live_in(target) - defs(edge)). Divergent
    // cohorts may occupy different CFG locations simultaneously, but they use
    // disjoint physical lanes; this per-lane liveness is therefore exactly the
    // storage-interference relation required by masked vector state.
    std::vector<BitSet> live_in(
        block_count, BitSet(state_count, uint8_t{0u}));
    auto transfer_edge = [&](const StateEdge &edge) noexcept {
        auto live = live_in[edge.target.value];
        for (auto assignment : *edge.assignments) {
            if (is_state(assignment.destination)) {
                live[state_index(assignment.destination)] = 0u;
            }
        }
        for (auto assignment : *edge.assignments) {
            if (is_state(assignment.source)) {
                live[state_index(assignment.source)] = 1u;
            }
        }
        return live;
    };
    auto changed = true;
    while (changed) {
        changed = false;
        for (auto block_index = block_count; block_index-- > 0u;) {
            auto next = block_uses[block_index];
            for (auto &&edge : edges[block_index]) {
                auto edge_live = transfer_edge(edge);
                for (auto value_index = size_t{0u};
                     value_index < state_count; value_index++) {
                    next[value_index] |= edge_live[value_index];
                }
            }
            if (next != live_in[block_index]) {
                live_in[block_index] = std::move(next);
                changed = true;
            }
        }
    }

    std::vector<BitSet> interference(
        state_count, BitSet(state_count, uint8_t{0u}));
    auto connect = [&](schedule::ValueId lhs,
                       schedule::ValueId rhs) noexcept {
        if (lhs != rhs && is_state(lhs) && is_state(rhs)) {
            auto lhs_index = state_index(lhs);
            auto rhs_index = state_index(rhs);
            interference[lhs_index][rhs_index] = 1u;
            interference[rhs_index][lhs_index] = 1u;
        }
    };
    auto connect_live = [&](const BitSet &live) noexcept {
        for (auto lhs = size_t{0u}; lhs < state_count; lhs++) {
            if (live[lhs] == 0u) { continue; }
            for (auto rhs = lhs + 1u; rhs < state_count; rhs++) {
                if (live[rhs] != 0u) {
                    interference[lhs][rhs] = 1u;
                    interference[rhs][lhs] = 1u;
                }
            }
        }
    };
    for (auto &&block : _source.blocks()) {
        connect_live(live_in[block.id.value]);
        for (auto &&edge : edges[block.id.value]) {
            auto live_after = live_in[edge.target.value];
            connect_live(live_after);
            auto live_before = transfer_edge(edge);
            connect_live(live_before);
            // Assignments are semantically parallel but emitted in source
            // order. A destination must therefore not reuse another copy's
            // source slot: doing so could overwrite that source before its
            // own assignment reads it. The matching destination/source pair
            // remains a coalescing candidate and becomes an identity copy.
            for (auto destination_index = size_t{0u};
                 destination_index < edge.assignments->size();
                 destination_index++) {
                auto destination =
                    (*edge.assignments)[destination_index].destination;
                for (auto source_index = size_t{0u};
                     source_index < edge.assignments->size();
                     source_index++) {
                    if (destination_index == source_index) { continue; }
                    connect(destination,
                            (*edge.assignments)[source_index].source);
                }
            }
        }
    }

    std::vector<uint32_t> parents(state_count);
    std::iota(parents.begin(), parents.end(), 0u);
    auto find_root = [&](auto &&self, uint32_t value) -> uint32_t {
        if (parents[value] != value) {
            parents[value] = self(self, parents[value]);
        }
        return parents[value];
    };
    auto groups_interfere = [&](uint32_t lhs_root,
                                uint32_t rhs_root) noexcept {
        for (auto lhs = uint32_t{0u}; lhs < state_count; lhs++) {
            if (find_root(find_root, lhs) != lhs_root) { continue; }
            for (auto rhs = uint32_t{0u}; rhs < state_count; rhs++) {
                if (find_root(find_root, rhs) == rhs_root &&
                    interference[lhs][rhs] != 0u) {
                    return true;
                }
            }
        }
        return false;
    };
    for (auto &&block_edges : edges) {
        for (auto &&edge : block_edges) {
            for (auto assignment : *edge.assignments) {
                if (!is_state(assignment.destination) ||
                    !is_state(assignment.source)) {
                    continue;
                }
                auto *destination = _source.value(
                    assignment.destination);
                auto *source = _source.value(assignment.source);
                auto *destination_slot =
                    _state_slots[assignment.destination.value];
                auto *source_slot = _state_slots[assignment.source.value];
                if (destination->name.empty() ||
                    destination->name != source->name ||
                    destination->value_class != source->value_class ||
                    destination->type != source->type ||
                    _is_local_lvalue(assignment.destination) !=
                        _is_local_lvalue(assignment.source) ||
                    destination_slot->getAllocatedType() !=
                        source_slot->getAllocatedType()) {
                    continue;
                }
                auto destination_root = find_root(
                    find_root, state_index(assignment.destination));
                auto source_root = find_root(
                    find_root, state_index(assignment.source));
                if (destination_root != source_root &&
                    !groups_interfere(
                        destination_root, source_root)) {
                    parents[source_root] = destination_root;
                }
            }
        }
    }

    static constexpr auto kGeneralStateColoringMinStateSlots = size_t{32u};
    static constexpr auto kGeneralStateColoringMinSavedSlots = size_t{2u};
    auto force_general_state_coloring =
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_FORCE_GENERAL_STATE_COLORING");
    auto enable_general_state_coloring =
        (force_general_state_coloring ||
         (_width == 16u &&
          _result.state_slot_count >=
              kGeneralStateColoringMinStateSlots)) &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_GENERAL_STATE_COLORING");
    if (enable_general_state_coloring) {
        auto parents_before_general_coloring = parents;
        auto general_colored_state_slot_count = size_t{0u};
        auto representative = [&](uint32_t root) noexcept {
            for (auto index = uint32_t{0u}; index < state_count; index++) {
                if (find_root(find_root, index) == root) {
                    return index;
                }
            }
            return static_cast<uint32_t>(state_count);
        };
        auto compatible = [&](uint32_t lhs_root,
                              uint32_t rhs_root) noexcept {
            auto lhs_index = representative(lhs_root);
            auto rhs_index = representative(rhs_root);
            if (lhs_index == state_count || rhs_index == state_count) {
                return false;
            }
            auto lhs_id = state_values[lhs_index];
            auto rhs_id = state_values[rhs_index];
            auto *lhs = _source.value(lhs_id);
            auto *rhs = _source.value(rhs_id);
            auto *lhs_slot = _state_slots[lhs_id.value];
            auto *rhs_slot = _state_slots[rhs_id.value];
            return lhs != nullptr && rhs != nullptr &&
                   lhs_slot != nullptr && rhs_slot != nullptr &&
                   lhs->value_class == rhs->value_class &&
                   lhs->type == rhs->type &&
                   _is_local_lvalue(lhs_id) ==
                       _is_local_lvalue(rhs_id) &&
                   lhs_slot->getAllocatedType() ==
                       rhs_slot->getAllocatedType();
        };
        std::vector<uint32_t> roots;
        roots.reserve(state_count);
        for (auto index = uint32_t{0u}; index < state_count; index++) {
            auto root = find_root(find_root, index);
            if (root == index) { roots.emplace_back(root); }
        }
        auto degree = [&](uint32_t root) noexcept {
            auto result = size_t{0u};
            for (auto other : roots) {
                result += root != other &&
                          groups_interfere(root, other);
            }
            return result;
        };
        std::vector<std::pair<uint32_t, size_t>> ranked_roots;
        ranked_roots.reserve(roots.size());
        for (auto root : roots) {
            ranked_roots.emplace_back(root, degree(root));
        }
        std::stable_sort(
            ranked_roots.begin(), ranked_roots.end(),
            [](auto lhs, auto rhs) noexcept {
                return lhs.second != rhs.second ?
                           lhs.second > rhs.second :
                           lhs.first < rhs.first;
            });
        for (auto i = size_t{0u}; i < roots.size(); i++) {
            roots[i] = ranked_roots[i].first;
        }
        std::vector<uint32_t> colors;
        colors.reserve(roots.size());
        for (auto root : roots) {
            auto merged = false;
            for (auto &color : colors) {
                color = find_root(find_root, color);
                if (compatible(root, color) &&
                    !groups_interfere(root, color)) {
                    parents[root] = color;
                    general_colored_state_slot_count++;
                    merged = true;
                    break;
                }
            }
            if (!merged) { colors.emplace_back(root); }
        }
        if (!force_general_state_coloring &&
            general_colored_state_slot_count <
                kGeneralStateColoringMinSavedSlots) {
            parents = std::move(parents_before_general_coloring);
        } else {
            _result.general_colored_state_slot_count =
                general_colored_state_slot_count;
        }
    }

    std::vector<::llvm::AllocaInst *> physical_slots(
        state_count, nullptr);
    for (auto id : state_values) {
        auto root = find_root(find_root, state_index(id));
        auto *&physical = physical_slots[root];
        if (physical == nullptr) {
            physical = _state_slots[id.value];
        } else if (_state_slots[id.value] != physical) {
            _state_slots[id.value] = physical;
            _result.coalesced_state_slot_count++;
        }
    }
}

}// namespace luisa::compute::simd::detail
