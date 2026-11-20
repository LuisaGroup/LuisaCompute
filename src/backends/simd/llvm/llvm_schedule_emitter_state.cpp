#include "llvm_schedule_emitter.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

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
    auto is_colorable_spill = [&](const schedule::Value &value) noexcept {
        if (value.id.value >= _spilled_instruction_values.size() ||
            _spilled_instruction_values[value.id.value] == 0u) {
            return false;
        }
        if (value.value_class == schedule::ValueClass::warp_uniform ||
            value.value_class == schedule::ValueClass::token) {
            return false;
        }
        if (_is_local_lvalue(value.id) &&
            _local_soa_kind_of(value.id) != LocalSoAKind::none) {
            return false;
        }
        return true;
    };
    for (auto &&value : _source.values()) {
        if ((value.origin == schedule::ValueOrigin::state_slot ||
             is_colorable_spill(value)) &&
            value.id.value < _state_slots.size() &&
            _state_slots[value.id.value] != nullptr) {
            state_indices[value.id.value] =
                static_cast<uint32_t>(state_values.size());
            state_values.emplace_back(value.id);
        }
    }
    auto state_count = state_values.size();
    if (state_count <= 1u) { return; }
    auto word_count = (state_count + 63u) / 64u;

    auto is_state = [&](schedule::ValueId id) noexcept {
        return id.value < state_indices.size() &&
               state_indices[id.value] != invalid_state_index;
    };
    auto state_index = [&](schedule::ValueId id) noexcept {
        return state_indices[id.value];
    };
    auto set_bit = [](uint64_t *bits, size_t index) noexcept {
        bits[index >> 6u] |= (uint64_t{1u} << (index & 63u));
    };
    auto clear_bit = [](uint64_t *bits, size_t index) noexcept {
        bits[index >> 6u] &= ~(uint64_t{1u} << (index & 63u));
    };
    auto test_bit = [](const uint64_t *bits, size_t index) noexcept {
        return (bits[index >> 6u] & (uint64_t{1u} << (index & 63u))) != 0u;
    };

    std::vector<std::vector<StateEdge>> edges(block_count);
    std::vector<std::vector<uint32_t>> predecessors(block_count);
    auto add_edge = [&](schedule::BlockId source,
                        const schedule::ControlEdge &edge) noexcept {
        edges[source.value].emplace_back(
            StateEdge{edge.target, &edge.assignments});
        predecessors[edge.target.value].emplace_back(source.value);
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
                    predecessors[point->target.value].emplace_back(
                        block.id.value);
                } else if constexpr (std::is_same_v<
                                         T, schedule::LoopBackTerminator>) {
                    auto *loop = _source.loop(terminator.loop);
                    edges[block.id.value].emplace_back(
                        StateEdge{loop->header, &terminator.assignments});
                    predecessors[loop->header.value].emplace_back(
                        block.id.value);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::BlockBarrierTerminator>) {
                    add_edge(block.id, terminator.resume_edge);
                }
            },
            block.terminator);
    }

    std::vector<uint64_t> block_uses(block_count * word_count, 0ull);
    std::vector<uint64_t> block_defs(block_count * word_count, 0ull);
    auto uses_row = [&](size_t block_index) noexcept {
        return block_uses.data() + block_index * word_count;
    };
    auto defs_row = [&](size_t block_index) noexcept {
        return block_defs.data() + block_index * word_count;
    };
    auto mark_use = [&](uint64_t *uses, const uint64_t *defs,
                        schedule::ValueId id) noexcept {
        if (is_state(id)) {
            auto idx = state_index(id);
            if (!test_bit(defs, idx)) { set_bit(uses, idx); }
        }
    };
    for (auto &&block : _source.blocks()) {
        auto *uses = uses_row(block.id.value);
        auto *defs = defs_row(block.id.value);
        for (auto &&instruction : block.instructions) {
            for (auto operand : instruction.operands) {
                mark_use(uses, defs, operand);
            }
            if (instruction.participant_mask) {
                mark_use(uses, defs, *instruction.participant_mask);
            }
            if (instruction.result && is_state(*instruction.result)) {
                set_bit(defs, state_index(*instruction.result));
            }
        }
        std::visit(
            [&](const auto &terminator) noexcept {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<
                                  T, schedule::SplitTerminator>) {
                    mark_use(uses, defs, terminator.condition);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SwitchTerminator>) {
                    mark_use(uses, defs, terminator.selector);
                } else if constexpr (std::is_same_v<
                                         T, schedule::ReturnTerminator>) {
                    if (terminator.value) {
                        mark_use(uses, defs, *terminator.value);
                    }
                }
            },
            block.terminator);
    }

    // Edge assignments are parallel PHI copies. For one logical lane, the
    // transfer is uses(edge) U (live_in(target) - defs(edge)), and across a
    // block is uses(block) U (live_out(block) - defs(block)). Divergent
    // cohorts may occupy different CFG locations simultaneously, but they use
    // disjoint physical lanes; this per-lane liveness is therefore exactly the
    // storage-interference relation required by masked vector state.
    std::vector<uint64_t> live_in = block_uses;
    auto live_in_row = [&](size_t block_index) noexcept {
        return live_in.data() + block_index * word_count;
    };
    std::vector<uint64_t> scratch_edge_live(word_count, 0ull);
    std::vector<uint64_t> scratch_out_live(word_count, 0ull);
    std::vector<uint64_t> scratch_next_live(word_count, 0ull);
    auto compute_transfer_edge = [&](const StateEdge &edge,
                                     uint64_t *out) noexcept {
        auto *target_live = live_in_row(edge.target.value);
        std::copy_n(target_live, word_count, out);
        for (auto assignment : *edge.assignments) {
            if (is_state(assignment.destination)) {
                clear_bit(out, state_index(assignment.destination));
            }
        }
        for (auto assignment : *edge.assignments) {
            if (is_state(assignment.source)) {
                set_bit(out, state_index(assignment.source));
            }
        }
    };

    std::vector<uint32_t> worklist(block_count);
    std::vector<uint8_t> queued(block_count, uint8_t{1u});
    for (auto i = size_t{0u}; i < block_count; ++i) {
        worklist[i] = static_cast<uint32_t>(i);
    }
    while (!worklist.empty()) {
        auto block_index = worklist.back();
        worklist.pop_back();
        queued[block_index] = 0u;
        auto *cur_live = live_in_row(block_index);
        std::fill_n(scratch_out_live.data(), word_count, 0ull);
        for (auto &&edge : edges[block_index]) {
            if (edge.assignments->empty()) {
                auto *target_live = live_in_row(edge.target.value);
                for (auto w = size_t{0u}; w < word_count; ++w) {
                    scratch_out_live[w] |= target_live[w];
                }
            } else {
                compute_transfer_edge(edge, scratch_edge_live.data());
                for (auto w = size_t{0u}; w < word_count; ++w) {
                    scratch_out_live[w] |= scratch_edge_live[w];
                }
            }
        }
        auto *uses = uses_row(block_index);
        auto *defs = defs_row(block_index);
        auto block_changed = false;
        for (auto w = size_t{0u}; w < word_count; ++w) {
            auto next = uses[w] | (scratch_out_live[w] & ~defs[w]);
            if (next != cur_live[w]) {
                cur_live[w] = next;
                block_changed = true;
            }
        }
        if (block_changed) {
            for (auto pred : predecessors[block_index]) {
                if (queued[pred] == 0u) {
                    queued[pred] = 1u;
                    worklist.emplace_back(pred);
                }
            }
        }
    }

    std::vector<uint64_t> interference(state_count * word_count, 0ull);
    auto interference_row = [&](size_t index) noexcept {
        return interference.data() + index * word_count;
    };
    auto connect = [&](schedule::ValueId lhs,
                       schedule::ValueId rhs) noexcept {
        if (lhs != rhs && is_state(lhs) && is_state(rhs)) {
            auto lhs_index = state_index(lhs);
            auto rhs_index = state_index(rhs);
            set_bit(interference_row(lhs_index), rhs_index);
            set_bit(interference_row(rhs_index), lhs_index);
        }
    };
    auto connect_live = [&](const uint64_t *live) noexcept {
        for (auto w = size_t{0u}; w < word_count; ++w) {
            auto word = live[w];
            while (word != 0ull) {
                auto bit = static_cast<size_t>(std::countr_zero(word));
                word &= (word - 1ull);
                auto lhs = (w << 6u) + bit;
                auto *row = interference_row(lhs);
                for (auto rw = size_t{0u}; rw < word_count; ++rw) {
                    row[rw] |= live[rw];
                }
            }
        }
    };

    std::vector<uint64_t> scratch_all_sources(word_count, 0ull);
    std::vector<uint64_t> scratch_dup_sources(word_count, 0ull);
    std::vector<uint64_t> scratch_all_dests(word_count, 0ull);
    std::vector<uint64_t> scratch_dup_dests(word_count, 0ull);

    for (auto &&block : _source.blocks()) {
        connect_live(live_in_row(block.id.value));
        std::fill_n(scratch_out_live.data(), word_count, 0ull);
        for (auto &&edge : edges[block.id.value]) {
            if (edge.assignments->empty()) {
                auto *target_live = live_in_row(edge.target.value);
                for (auto w = size_t{0u}; w < word_count; ++w) {
                    scratch_out_live[w] |= target_live[w];
                }
                continue;
            }
            compute_transfer_edge(edge, scratch_edge_live.data());
            for (auto w = size_t{0u}; w < word_count; ++w) {
                scratch_out_live[w] |= scratch_edge_live[w];
            }
            connect_live(scratch_edge_live.data());
            // Keep the conservative cross-copy interference policy. Sources
            // are snapshotted before masked writes, including genuine PHI
            // cycles; relaxing this coloring constraint is a separate change.
            // Matching destination/source pairs remain identity candidates.
            auto assignment_count = edge.assignments->size();
            if (assignment_count <= 16u) {
                for (auto destination_index = size_t{0u};
                     destination_index < assignment_count;
                     destination_index++) {
                    auto destination =
                        (*edge.assignments)[destination_index].destination;
                    for (auto source_index = size_t{0u};
                         source_index < assignment_count;
                         source_index++) {
                        if (destination_index == source_index) { continue; }
                        connect(destination,
                                (*edge.assignments)[source_index].source);
                    }
                }
            } else {
                std::fill_n(scratch_all_sources.data(), word_count, 0ull);
                std::fill_n(scratch_dup_sources.data(), word_count, 0ull);
                std::fill_n(scratch_all_dests.data(), word_count, 0ull);
                std::fill_n(scratch_dup_dests.data(), word_count, 0ull);
                for (auto &&assignment : *edge.assignments) {
                    if (is_state(assignment.source)) {
                        auto s = state_index(assignment.source);
                        if (test_bit(scratch_all_sources.data(), s)) {
                            set_bit(scratch_dup_sources.data(), s);
                        } else {
                            set_bit(scratch_all_sources.data(), s);
                        }
                    }
                    if (is_state(assignment.destination)) {
                        auto d = state_index(assignment.destination);
                        if (test_bit(scratch_all_dests.data(), d)) {
                            set_bit(scratch_dup_dests.data(), d);
                        } else {
                            set_bit(scratch_all_dests.data(), d);
                        }
                    }
                }
                for (auto &&assignment : *edge.assignments) {
                    auto has_dest = is_state(assignment.destination);
                    auto has_src = is_state(assignment.source);
                    if (has_dest) {
                        auto d = state_index(assignment.destination);
                        auto *d_row = interference_row(d);
                        auto preserve_src = false;
                        size_t s = 0u;
                        if (has_src) {
                            s = state_index(assignment.source);
                            preserve_src = test_bit(d_row, s) ||
                                           test_bit(scratch_dup_sources.data(), s);
                        }
                        for (auto rw = size_t{0u}; rw < word_count; ++rw) {
                            d_row[rw] |= scratch_all_sources[rw];
                        }
                        if (has_src && !preserve_src) {
                            clear_bit(d_row, s);
                        }
                    }
                    if (has_src) {
                        auto s = state_index(assignment.source);
                        auto *s_row = interference_row(s);
                        auto preserve_dest = false;
                        size_t d = 0u;
                        if (has_dest) {
                            d = state_index(assignment.destination);
                            preserve_dest = test_bit(s_row, d) ||
                                            test_bit(scratch_dup_dests.data(), d);
                        }
                        for (auto rw = size_t{0u}; rw < word_count; ++rw) {
                            s_row[rw] |= scratch_all_dests[rw];
                        }
                        if (has_dest && !preserve_dest) {
                            clear_bit(s_row, d);
                        }
                    }
                }
            }
        }
        // Backward per-instruction interference for spilled instructions.
        auto *defs = defs_row(block.id.value);
        auto has_defs = false;
        for (auto w = size_t{0u}; w < word_count; ++w) {
            if (defs[w] != 0ull) {
                has_defs = true;
                break;
            }
        }
        if (has_defs) {
            auto mark_term_use = [&](schedule::ValueId id) noexcept {
                if (is_state(id)) {
                    set_bit(scratch_out_live.data(), state_index(id));
                }
            };
            std::visit(
                [&](const auto &terminator) noexcept {
                    using T = std::decay_t<decltype(terminator)>;
                    if constexpr (std::is_same_v<T, schedule::SplitTerminator>) {
                        mark_term_use(terminator.condition);
                    } else if constexpr (std::is_same_v<T, schedule::SwitchTerminator>) {
                        mark_term_use(terminator.selector);
                    } else if constexpr (std::is_same_v<T, schedule::ReturnTerminator>) {
                        if (terminator.value) { mark_term_use(*terminator.value); }
                    }
                },
                block.terminator);
            for (auto it = block.instructions.rbegin();
                 it != block.instructions.rend(); ++it) {
                const auto &inst = *it;
                if (inst.result && is_state(*inst.result)) {
                    auto d = state_index(*inst.result);
                    auto *d_row = interference_row(d);
                    for (auto w = size_t{0u}; w < word_count; ++w) {
                        auto word = scratch_out_live[w];
                        d_row[w] |= word;
                        while (word != 0ull) {
                            auto bit = static_cast<size_t>(std::countr_zero(word));
                            word &= (word - 1ull);
                            set_bit(interference_row((w << 6u) + bit), d);
                        }
                    }
                    clear_bit(scratch_out_live.data(), d);
                }
                for (auto operand : inst.operands) {
                    if (is_state(operand)) {
                        set_bit(scratch_out_live.data(), state_index(operand));
                    }
                }
                if (inst.participant_mask && is_state(*inst.participant_mask)) {
                    set_bit(scratch_out_live.data(),
                            state_index(*inst.participant_mask));
                }
            }
        }
    }
    for (auto i = size_t{0u}; i < state_count; ++i) {
        clear_bit(interference_row(i), i);
    }

    std::vector<uint32_t> parents(state_count);
    std::iota(parents.begin(), parents.end(), 0u);
    std::vector<uint64_t> group_masks(state_count * word_count, 0ull);
    auto group_mask_row = [&](size_t index) noexcept {
        return group_masks.data() + index * word_count;
    };
    for (auto index = uint32_t{0u}; index < state_count; index++) {
        set_bit(group_mask_row(index), index);
    }
    auto find_root = [&](auto &&self, uint32_t value) -> uint32_t {
        if (parents[value] != value) {
            parents[value] = self(self, parents[value]);
        }
        return parents[value];
    };
    auto merge_groups = [&](uint32_t destination_root,
                            uint32_t source_root) noexcept {
        parents[source_root] = destination_root;
        auto *dst_int = interference_row(destination_root);
        auto *src_int = interference_row(source_root);
        auto *dst_mask = group_mask_row(destination_root);
        auto *src_mask = group_mask_row(source_root);
        for (auto w = size_t{0u}; w < word_count; ++w) {
            dst_int[w] |= src_int[w];
            dst_mask[w] |= src_mask[w];
        }
    };
    auto groups_interfere = [&](uint32_t lhs_root,
                                uint32_t rhs_root) noexcept {
        auto *lhs_int = interference_row(lhs_root);
        auto *rhs_mask = group_mask_row(rhs_root);
        for (auto w = size_t{0u}; w < word_count; ++w) {
            if ((lhs_int[w] & rhs_mask[w]) != 0ull) {
                return true;
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
                if (destination->origin != schedule::ValueOrigin::state_slot ||
                    source->origin != schedule::ValueOrigin::state_slot) {
                    continue;
                }
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
                    merge_groups(destination_root, source_root);
                }
            }
        }
    }

    static constexpr auto kGeneralStateColoringMinStateSlots = size_t{32u};
    static constexpr auto kGeneralStateColoringLargeStateSlots = size_t{256u};
    static constexpr auto kGeneralStateColoringMinSavedSlots = size_t{2u};
    auto force_general_state_coloring =
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_FORCE_GENERAL_STATE_COLORING");
    auto enable_general_state_coloring =
        (force_general_state_coloring ||
         (_width == 16u &&
          _result.state_slot_count >=
              kGeneralStateColoringMinStateSlots) ||
         _result.state_slot_count >=
             kGeneralStateColoringLargeStateSlots) &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_GENERAL_STATE_COLORING");
    if (enable_general_state_coloring) {
        std::vector<uint32_t> roots;
        roots.reserve(state_count);
        std::vector<uint32_t> root_compact_index(
            state_count, invalid_state_index);
        for (auto index = uint32_t{0u}; index < state_count; index++) {
            auto root = find_root(find_root, index);
            if (root == index) {
                root_compact_index[root] =
                    static_cast<uint32_t>(roots.size());
                roots.emplace_back(root);
            }
        }
        auto root_count = roots.size();
        if (root_count > 1u) {
            std::vector<uint32_t> state_to_compact_root(state_count);
            for (auto index = uint32_t{0u}; index < state_count; index++) {
                state_to_compact_root[index] =
                    root_compact_index[find_root(find_root, index)];
            }
            auto root_word_count = (root_count + 63u) / 64u;
            std::vector<uint64_t> root_interference(
                root_count * root_word_count, 0ull);
            auto root_int_row = [&](size_t compact_idx) noexcept {
                return root_interference.data() +
                       compact_idx * root_word_count;
            };
            for (auto idx = size_t{0u}; idx < root_count; ++idx) {
                auto root = roots[idx];
                auto *src_row = interference_row(root);
                auto *dst_row = root_int_row(idx);
                for (auto w = size_t{0u}; w < word_count; ++w) {
                    auto word = src_row[w];
                    while (word != 0ull) {
                        auto bit = static_cast<size_t>(std::countr_zero(word));
                        word &= (word - 1ull);
                        auto neighbor_state = (w << 6u) + bit;
                        auto neighbor_compact =
                            state_to_compact_root[neighbor_state];
                        if (neighbor_compact != idx) {
                            set_bit(dst_row, neighbor_compact);
                        }
                    }
                }
            }
            std::vector<std::pair<uint32_t, size_t>> ranked_roots;
            ranked_roots.reserve(root_count);
            for (auto idx = size_t{0u}; idx < root_count; ++idx) {
                auto *row = root_int_row(idx);
                auto deg = size_t{0u};
                for (auto rw = size_t{0u}; rw < root_word_count; ++rw) {
                    deg += static_cast<size_t>(std::popcount(row[rw]));
                }
                ranked_roots.emplace_back(static_cast<uint32_t>(idx), deg);
            }
            std::stable_sort(
                ranked_roots.begin(), ranked_roots.end(),
                [&](auto lhs, auto rhs) noexcept {
                    return lhs.second != rhs.second ?
                               lhs.second > rhs.second :
                               roots[lhs.first] < roots[rhs.first];
                });

            struct CompatKey {
                const Type *type{nullptr};
                const ::llvm::Type *llvm_type{nullptr};
                uint8_t value_class{0u};
                uint8_t is_lvalue{0u};
                bool operator==(const CompatKey &) const noexcept = default;
            };
            struct CompatKeyHash {
                size_t operator()(const CompatKey &k) const noexcept {
                    auto h = std::hash<const void *>{}(k.type);
                    h ^= std::hash<const void *>{}(k.llvm_type) + 0x9e3779b9u +
                         (h << 6u) + (h >> 2u);
                    h ^= (static_cast<size_t>(k.value_class) << 1u) |
                         static_cast<size_t>(k.is_lvalue);
                    return h;
                }
            };
            struct Color {
                uint32_t representative_root{};
                std::vector<uint32_t> member_roots{};
                std::vector<uint64_t> compact_mask{};
            };
            std::unordered_map<CompatKey, std::vector<Color>, CompatKeyHash>
                bucketed_colors;
            auto general_colored_state_slot_count = size_t{0u};
            for (auto [compact_idx, _] : ranked_roots) {
                auto root = roots[compact_idx];
                auto value_id = state_values[root];
                auto *val = _source.value(value_id);
                auto *slot = _state_slots[value_id.value];
                if (val == nullptr || slot == nullptr) { continue; }
                CompatKey key{
                    .type = val->type,
                    .llvm_type = slot->getAllocatedType(),
                    .value_class = static_cast<uint8_t>(val->value_class),
                    .is_lvalue = static_cast<uint8_t>(_is_local_lvalue(value_id)),
                };
                auto &colors = bucketed_colors[key];
                auto *int_row = root_int_row(compact_idx);
                auto merged = false;
                for (auto &color : colors) {
                    auto interferes = false;
                    for (auto rw = size_t{0u}; rw < root_word_count; ++rw) {
                        if ((int_row[rw] & color.compact_mask[rw]) != 0ull) {
                            interferes = true;
                            break;
                        }
                    }
                    if (!interferes) {
                        color.member_roots.emplace_back(root);
                        set_bit(color.compact_mask.data(), compact_idx);
                        general_colored_state_slot_count++;
                        merged = true;
                        break;
                    }
                }
                if (!merged) {
                    Color created{
                        .representative_root = root,
                        .member_roots = {root},
                        .compact_mask = std::vector<uint64_t>(
                            root_word_count, 0ull),
                    };
                    set_bit(created.compact_mask.data(), compact_idx);
                    colors.emplace_back(std::move(created));
                }
            }
            if (force_general_state_coloring ||
                general_colored_state_slot_count >=
                    kGeneralStateColoringMinSavedSlots) {
                for (auto &[_, colors] : bucketed_colors) {
                    for (auto &&color : colors) {
                        for (auto index = size_t{1u};
                             index < color.member_roots.size(); index++) {
                            parents[color.member_roots[index]] =
                                color.representative_root;
                        }
                    }
                }
                _result.general_colored_state_slot_count =
                    general_colored_state_slot_count;
            }
        }
    }

    std::vector<::llvm::AllocaInst *> physical_slots(
        state_count, nullptr);
    std::vector<::llvm::AllocaInst *> dead_slots;
    for (auto id : state_values) {
        auto root = find_root(find_root, state_index(id));
        auto *&physical = physical_slots[root];
        if (physical == nullptr) {
            physical = _state_slots[id.value];
        } else if (_state_slots[id.value] != physical) {
            dead_slots.emplace_back(_state_slots[id.value]);
            _state_slots[id.value] = physical;
            _result.coalesced_state_slot_count++;
        }
    }
    if (!dead_slots.empty()) {
        std::unordered_set<::llvm::AllocaInst *> active_slots;
        active_slots.reserve(state_count);
        for (auto id : state_values) {
            if (_state_slots[id.value] != nullptr) {
                active_slots.emplace(_state_slots[id.value]);
            }
        }
        std::unordered_set<::llvm::AllocaInst *> erased_slots;
        erased_slots.reserve(dead_slots.size());
        for (auto *dead : dead_slots) {
            if (dead == nullptr || active_slots.contains(dead) ||
                !erased_slots.emplace(dead).second) {
                continue;
            }
            auto can_erase = true;
            for (auto *user : dead->users()) {
                auto *store = ::llvm::dyn_cast<::llvm::StoreInst>(user);
                if (store == nullptr || store->getPointerOperand() != dead) {
                    can_erase = false;
                    break;
                }
            }
            if (!can_erase) { continue; }
            while (!dead->user_empty()) {
                if (auto *inst = ::llvm::dyn_cast<::llvm::Instruction>(
                        *dead->user_begin())) {
                    inst->eraseFromParent();
                } else {
                    break;
                }
            }
            if (dead->user_empty()) {
                dead->eraseFromParent();
            }
        }
    }
}

}// namespace luisa::compute::simd::detail
