#include "coro_scalar_relation_liveness.h"

#include <algorithm>

#include <luisa/xir/basic_block.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>

#include "coro_semantic_graph.h"

namespace luisa::compute::xir::detail {

namespace {

using Bits = luisa::vector<uint64_t>;

[[nodiscard]] bool bit_test(
    const Bits &bits, size_t index) noexcept {
    return (bits[index / 64u] &
            (uint64_t{1u} << (index % 64u))) != 0u;
}

void bit_set(Bits &bits, size_t index) noexcept {
    bits[index / 64u] |= uint64_t{1u} << (index % 64u);
}

void bit_clear(Bits &bits, size_t index) noexcept {
    bits[index / 64u] &= ~(uint64_t{1u} << (index % 64u));
}

}// namespace

CoroScalarRelationLiveness::CoroScalarRelationLiveness(
    const CoroSemanticGraph &graph,
    luisa::span<const uint8_t> active_blocks,
    size_t lifetime_target,
    luisa::span<AllocaInst *const> slots,
    const CoroScalarSemanticUses &semantic_uses) noexcept {
    _live_in.resize(graph.block_count());
    if (slots.empty() || active_blocks.size() != graph.block_count()) {
        return;
    }
    luisa::unordered_map<AllocaInst *, size_t> slot_ids;
    slot_ids.reserve(slots.size());
    for (size_t i = 0u; i < slots.size(); ++i) {
        slot_ids.emplace(slots[i], i);
    }
    auto word_count = (slots.size() + 63u) / 64u;
    luisa::vector<Bits> uses(
        graph.block_count(), Bits(word_count, 0u));
    luisa::vector<Bits> definitions(
        graph.block_count(), Bits(word_count, 0u));
    luisa::vector<Bits> live_in_bits(
        graph.block_count(), Bits(word_count, 0u));
    luisa::vector<Bits> live_out_bits(
        graph.block_count(), Bits(word_count, 0u));

    for (size_t block_id = 0u;
         block_id < graph.block_count(); ++block_id) {
        if (active_blocks[block_id] == 0u) { continue; }
        for (auto *instruction : graph.block(block_id)->instructions()) {
            if (instruction->isa<LoadInst>()) {
                auto *variable =
                    static_cast<LoadInst *>(instruction)->variable();
                if (variable != nullptr && variable->isa<AllocaInst>()) {
                    if (auto iter = slot_ids.find(
                            static_cast<AllocaInst *>(variable));
                        iter != slot_ids.end() &&
                        !bit_test(definitions[block_id], iter->second)) {
                        bit_set(uses[block_id], iter->second);
                    }
                }
            } else if (instruction->isa<StoreInst>()) {
                auto *variable =
                    static_cast<StoreInst *>(instruction)->variable();
                if (variable != nullptr && variable->isa<AllocaInst>()) {
                    if (auto iter = slot_ids.find(
                            static_cast<AllocaInst *>(variable));
                        iter != slot_ids.end()) {
                        bit_set(definitions[block_id], iter->second);
                    }
                }
            }
            if (auto use_iter = semantic_uses.find(instruction);
                use_iter != semantic_uses.end()) {
                for (auto *slot : use_iter->second) {
                    if (auto iter = slot_ids.find(slot);
                        iter != slot_ids.end() &&
                        !bit_test(definitions[block_id], iter->second)) {
                        bit_set(uses[block_id], iter->second);
                    }
                }
            }
        }
    }

    for (;;) {
        auto changed = false;
        for (size_t reverse_id = graph.block_count();
             reverse_id != 0u; --reverse_id) {
            auto block_id = reverse_id - 1u;
            if (active_blocks[block_id] == 0u) { continue; }
            auto next_out = Bits(word_count, 0u);
            for (auto successor : graph.successors(block_id)) {
                if (successor == lifetime_target ||
                    active_blocks[successor] == 0u) {
                    continue;
                }
                for (size_t word = 0u; word < word_count; ++word) {
                    next_out[word] |= live_in_bits[successor][word];
                }
            }
            auto next_in = uses[block_id];
            for (size_t word = 0u; word < word_count; ++word) {
                next_in[word] |=
                    next_out[word] & ~definitions[block_id][word];
            }
            if (next_in != live_in_bits[block_id] ||
                next_out != live_out_bits[block_id]) {
                live_in_bits[block_id] = std::move(next_in);
                live_out_bits[block_id] = std::move(next_out);
                changed = true;
            }
        }
        if (!changed) { break; }
    }

    for (size_t block_id = 0u;
         block_id < graph.block_count(); ++block_id) {
        if (active_blocks[block_id] == 0u) { continue; }
        for (size_t i = 0u; i < slots.size(); ++i) {
            if (bit_test(live_in_bits[block_id], i)) {
                _live_in[block_id].emplace_back(slots[i]);
            }
        }
        auto live = live_out_bits[block_id];
        luisa::vector<Instruction *> instructions;
        for (auto *instruction : graph.block(block_id)->instructions()) {
            instructions.emplace_back(instruction);
        }
        for (auto iter = instructions.rbegin();
             iter != instructions.rend(); ++iter) {
            auto *instruction = *iter;
            if (instruction->isa<StoreInst>()) {
                auto *variable =
                    static_cast<StoreInst *>(instruction)->variable();
                if (variable != nullptr && variable->isa<AllocaInst>()) {
                    auto slot_iter = slot_ids.find(
                        static_cast<AllocaInst *>(variable));
                    if (slot_iter != slot_ids.end()) {
                        if (!bit_test(live, slot_iter->second)) {
                            _dead_after[instruction].emplace_back(
                                slots[slot_iter->second]);
                        }
                        bit_clear(live, slot_iter->second);
                    }
                }
            } else if (instruction->isa<LoadInst>()) {
                auto *variable =
                    static_cast<LoadInst *>(instruction)->variable();
                if (variable != nullptr && variable->isa<AllocaInst>()) {
                    auto slot_iter = slot_ids.find(
                        static_cast<AllocaInst *>(variable));
                    if (slot_iter != slot_ids.end()) {
                        if (!bit_test(live, slot_iter->second)) {
                            _dead_after[instruction].emplace_back(
                                slots[slot_iter->second]);
                        }
                        bit_set(live, slot_iter->second);
                    }
                }
            }
            if (auto use_iter = semantic_uses.find(instruction);
                use_iter != semantic_uses.end()) {
                for (auto *slot : use_iter->second) {
                    auto slot_iter = slot_ids.find(slot);
                    if (slot_iter == slot_ids.end()) { continue; }
                    if (!bit_test(live, slot_iter->second)) {
                        _dead_after[instruction].emplace_back(slot);
                    }
                    bit_set(live, slot_iter->second);
                }
            }
        }
    }
}

luisa::span<AllocaInst *const>
CoroScalarRelationLiveness::live_in(
    size_t block_id) const noexcept {
    return block_id < _live_in.size() ?
               luisa::span<AllocaInst *const>{_live_in[block_id]} :
               luisa::span<AllocaInst *const>{};
}

luisa::span<AllocaInst *const>
CoroScalarRelationLiveness::dead_after(
    Instruction *instruction) const noexcept {
    if (auto iter = _dead_after.find(instruction);
        iter != _dead_after.end()) {
        return iter->second;
    }
    return {};
}

CoroBooleanPredicateLiveness::CoroBooleanPredicateLiveness(
    const CoroSemanticGraph &graph,
    luisa::span<const uint8_t> active_blocks,
    size_t lifetime_target,
    luisa::span<Value *const> predicates,
    const CoroBooleanSemanticValues &semantic_uses,
    const CoroBooleanSemanticValues &semantic_definitions) noexcept {
    _live_in.resize(graph.block_count());
    if (predicates.empty() ||
        active_blocks.size() != graph.block_count()) {
        return;
    }
    luisa::unordered_map<Value *, size_t> predicate_ids;
    predicate_ids.reserve(predicates.size());
    for (size_t i = 0u; i < predicates.size(); ++i) {
        predicate_ids.emplace(predicates[i], i);
    }
    auto word_count = (predicates.size() + 63u) / 64u;
    luisa::vector<Bits> uses(
        graph.block_count(), Bits(word_count, 0u));
    luisa::vector<Bits> definitions(
        graph.block_count(), Bits(word_count, 0u));
    luisa::vector<Bits> live_in_bits(
        graph.block_count(), Bits(word_count, 0u));
    luisa::vector<Bits> live_out_bits(
        graph.block_count(), Bits(word_count, 0u));

    for (size_t block_id = 0u;
         block_id < graph.block_count(); ++block_id) {
        if (active_blocks[block_id] == 0u) { continue; }
        for (auto *instruction : graph.block(block_id)->instructions()) {
            if (auto iter = semantic_uses.find(instruction);
                iter != semantic_uses.end()) {
                for (auto *predicate : iter->second) {
                    if (auto id = predicate_ids.find(predicate);
                        id != predicate_ids.end() &&
                        !bit_test(definitions[block_id], id->second)) {
                        bit_set(uses[block_id], id->second);
                    }
                }
            }
            if (auto iter = semantic_definitions.find(instruction);
                iter != semantic_definitions.end()) {
                for (auto *predicate : iter->second) {
                    if (auto id = predicate_ids.find(predicate);
                        id != predicate_ids.end()) {
                        bit_set(definitions[block_id], id->second);
                    }
                }
            }
        }
    }

    for (;;) {
        auto changed = false;
        for (size_t reverse_id = graph.block_count();
             reverse_id != 0u; --reverse_id) {
            auto block_id = reverse_id - 1u;
            if (active_blocks[block_id] == 0u) { continue; }
            auto next_out = Bits(word_count, 0u);
            for (auto successor : graph.successors(block_id)) {
                if (successor == lifetime_target ||
                    active_blocks[successor] == 0u) {
                    continue;
                }
                for (size_t word = 0u; word < word_count; ++word) {
                    next_out[word] |= live_in_bits[successor][word];
                }
            }
            auto next_in = uses[block_id];
            for (size_t word = 0u; word < word_count; ++word) {
                next_in[word] |=
                    next_out[word] & ~definitions[block_id][word];
            }
            if (next_in != live_in_bits[block_id] ||
                next_out != live_out_bits[block_id]) {
                live_in_bits[block_id] = std::move(next_in);
                live_out_bits[block_id] = std::move(next_out);
                changed = true;
            }
        }
        if (!changed) { break; }
    }

    for (size_t block_id = 0u;
         block_id < graph.block_count(); ++block_id) {
        if (active_blocks[block_id] == 0u) { continue; }
        for (size_t i = 0u; i < predicates.size(); ++i) {
            if (bit_test(live_in_bits[block_id], i)) {
                _live_in[block_id].emplace_back(predicates[i]);
            }
        }
    }
}

luisa::span<Value *const>
CoroBooleanPredicateLiveness::live_in(
    size_t block_id) const noexcept {
    return block_id < _live_in.size() ?
               luisa::span<Value *const>{_live_in[block_id]} :
               luisa::span<Value *const>{};
}

}// namespace luisa::compute::xir::detail
