#include "coro_initialized_prefix.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <utility>

#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/assume.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/store.h>

#include "coro_frame_access.h"
#include "coro_guarded_scalar_relation.h"
#include "coro_scalar_relation_liveness.h"
#include "coro_semantic_graph.h"

namespace luisa::compute::xir::detail {

namespace {

struct PrefixInstructionLocation {
    size_t block_id;
    size_t ordinal;
};

using PrefixInstructionLocationMap =
    luisa::unordered_map<Instruction *, PrefixInstructionLocation>;

[[nodiscard]] PrefixInstructionLocationMap make_prefix_instruction_locations(
    FunctionDefinition *definition,
    const CoroSemanticGraph &graph) noexcept {
    PrefixInstructionLocationMap locations;
    for (auto *block : definition->basic_blocks()) {
        auto ordinal = size_t{0u};
        for (auto *instruction : block->instructions()) {
            locations.emplace(
                instruction,
                PrefixInstructionLocation{
                    graph.block_id(block), ordinal++});
        }
    }
    return locations;
}

struct ArrayUseRegion {
    bool valid{true};
    luisa::unordered_set<Value *> pointers;
    luisa::unordered_set<Instruction *> users;
    luisa::vector<BasicBlock *> blocks;
};

[[nodiscard]] ArrayUseRegion collect_array_use_region(
    AllocaInst *array, FunctionDefinition *definition,
    const CoroSemanticGraph &graph) noexcept {
    ArrayUseRegion result;
    luisa::unordered_set<BasicBlock *> seen_blocks;
    luisa::vector<Value *> worklist{array};
    while (!worklist.empty() && result.valid) {
        auto *pointer = worklist.back();
        worklist.pop_back();
        if (!result.pointers.emplace(pointer).second) { continue; }
        for (auto *use : pointer->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<Instruction>()) {
                result.valid = false;
                break;
            }
            auto *instruction = static_cast<Instruction *>(user);
            auto *block = instruction->parent_block();
            if (block == nullptr ||
                instruction->parent_function() != definition ||
                !graph.contains(block) || instruction->isa<PhiInst>()) {
                result.valid = false;
                break;
            }
            result.users.emplace(instruction);
            if (seen_blocks.emplace(block).second) {
                result.blocks.emplace_back(block);
            }
            if (instruction->isa<GEPInst>() &&
                static_cast<GEPInst *>(instruction)->base() == pointer) {
                worklist.emplace_back(instruction);
            }
        }
    }
    return result;
}

struct ActiveSlice {
    bool valid{false};
    size_t target_id{0u};
    luisa::vector<uint8_t> active;
    luisa::vector<size_t> blocks;
};

[[nodiscard]] ActiveSlice make_active_slice(
    BasicBlock *target, const ArrayUseRegion &region,
    const CoroSemanticGraph &graph) noexcept {
    ActiveSlice slice;
    if (target == nullptr || !region.valid) { return slice; }
    slice.target_id = graph.block_id(target);
    if (slice.target_id >= graph.block_count()) { return slice; }
    slice.active.assign(graph.block_count(), 0u);
    luisa::vector<size_t> worklist;
    for (auto *block : region.blocks) {
        if (!graph.dominates(target, block)) { return slice; }
        auto id = graph.block_id(block);
        if (id >= graph.block_count()) { return slice; }
        if (slice.active[id] == 0u) {
            slice.active[id] = 1u;
            worklist.emplace_back(id);
        }
    }
    if (slice.active[slice.target_id] == 0u) {
        slice.active[slice.target_id] = 1u;
        worklist.emplace_back(slice.target_id);
    }
    for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
        auto id = worklist[cursor];
        if (id == slice.target_id) { continue; }
        for (auto predecessor : graph.predecessors(id)) {
            if (!graph.dominates(target, graph.block(predecessor))) {
                return slice;
            }
            if (slice.active[predecessor] == 0u) {
                slice.active[predecessor] = 1u;
                worklist.emplace_back(predecessor);
            }
        }
    }
    std::sort(worklist.begin(), worklist.end());
    slice.blocks = std::move(worklist);
    slice.valid = true;
    return slice;
}

[[nodiscard]] bool instruction_precedes(
    Instruction *before, Instruction *after,
    const PrefixInstructionLocationMap &locations) noexcept {
    if (before == nullptr || after == nullptr) { return false; }
    auto before_iter = locations.find(before);
    auto after_iter = locations.find(after);
    return before_iter != locations.end() &&
           after_iter != locations.end() &&
           before_iter->second.block_id == after_iter->second.block_id &&
           before_iter->second.ordinal < after_iter->second.ordinal;
}

struct ScalarSlotInfo {
    bool valid{false};
    StoreInst *single_store{nullptr};
    luisa::unordered_map<LoadInst *, StoreInst *> local_reaching_stores;
};

class ScalarCopyResolver {
private:
    const PrefixInstructionLocationMap &_locations;
    const CoroSemanticGraph &_graph;
    mutable luisa::unordered_map<AllocaInst *, ScalarSlotInfo> _slots;
    mutable luisa::unordered_map<
        LoadInst *, luisa::unordered_map<Instruction *, bool>>
        _unchanged_snapshot_cache;

private:
    [[nodiscard]] bool _instruction_dominates(
        Instruction *definition, Instruction *use) const noexcept {
        if (definition == nullptr || use == nullptr) { return false; }
        if (definition->parent_block() == use->parent_block()) {
            return instruction_precedes(definition, use, _locations);
        }
        return _graph.dominates(
            definition->parent_block(), use->parent_block());
    }

    [[nodiscard]] static bool _value_depends_on_slot(
        Value *value, AllocaInst *slot) noexcept {
        luisa::unordered_set<Value *> visited;
        luisa::vector<Value *> worklist{value};
        while (!worklist.empty()) {
            auto *current = worklist.back();
            worklist.pop_back();
            if (current == nullptr || !visited.emplace(current).second) {
                continue;
            }
            if (current->isa<LoadInst>() &&
                static_cast<LoadInst *>(current)->variable() == slot) {
                return true;
            }
            if (!current->isa<Instruction>()) { continue; }
            auto *instruction = static_cast<Instruction *>(current);
            for (size_t i = 0u; i < instruction->operand_count(); ++i) {
                worklist.emplace_back(instruction->operand(i));
            }
        }
        return false;
    }

    [[nodiscard]] const ScalarSlotInfo &_slot_info(
        AllocaInst *slot) const noexcept {
        if (auto iter = _slots.find(slot); iter != _slots.end()) {
            return iter->second;
        }
        ScalarSlotInfo info;
        if (slot == nullptr || !slot->is_local() ||
            slot->type() == nullptr || !slot->type()->is_scalar()) {
            return _slots.emplace(slot, std::move(info)).first->second;
        }
        luisa::vector<LoadInst *> loads;
        luisa::vector<StoreInst *> stores;
        for (auto *use : slot->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<Instruction>()) {
                return _slots.emplace(slot, std::move(info)).first->second;
            }
            auto *instruction = static_cast<Instruction *>(user);
            if (instruction->isa<LoadInst>() &&
                static_cast<LoadInst *>(instruction)->variable() == slot) {
                loads.emplace_back(static_cast<LoadInst *>(instruction));
                continue;
            }
            if (instruction->isa<StoreInst>() &&
                static_cast<StoreInst *>(instruction)->variable() == slot) {
                stores.emplace_back(static_cast<StoreInst *>(instruction));
                continue;
            }
            return _slots.emplace(slot, std::move(info)).first->second;
        }
        if (stores.empty()) {
            return _slots.emplace(slot, std::move(info)).first->second;
        }
        if (stores.size() == 1u) {
            // Preserve the original exact single-store rule. Individual
            // substitutions below still require store < load < use within
            // one basic block.
            info.valid = true;
            info.single_store = stores.front();
            return _slots.emplace(slot, std::move(info)).first->second;
        }

        // Local reaching-definition lemma: in a basic block's total order,
        // the last direct store preceding a load executes on every path to
        // that load and kills all incoming definitions. With no indirect
        // accesses, substituting that store's SSA value is exact. This is a
        // per-load fact: loads without a local reaching store remain opaque,
        // rather than invalidating independent exact copies elsewhere.
        for (auto *load : loads) {
            auto load_iter = _locations.find(load);
            if (load_iter == _locations.end()) { continue; }
            StoreInst *reaching_store = nullptr;
            size_t reaching_ordinal = 0u;
            for (auto *store : stores) {
                auto store_iter = _locations.find(store);
                if (store_iter == _locations.end() ||
                    store_iter->second.block_id !=
                        load_iter->second.block_id ||
                    store_iter->second.ordinal >=
                        load_iter->second.ordinal) {
                    continue;
                }
                if (reaching_store == nullptr ||
                    reaching_ordinal < store_iter->second.ordinal) {
                    reaching_store = store;
                    reaching_ordinal = store_iter->second.ordinal;
                }
            }
            // A recurrence is mutable state, not a transparent copy. Reject
            // only the affected reaching definition: an unrelated local
            // overwrite in another block may still be substituted exactly.
            if (reaching_store != nullptr &&
                !_value_depends_on_slot(reaching_store->value(), slot)) {
                info.local_reaching_stores.emplace(load, reaching_store);
            }
        }
        info.valid = true;
        return _slots.emplace(slot, std::move(info)).first->second;
    }

    [[nodiscard]] StoreInst *_reaching_store(
        AllocaInst *slot, LoadInst *load) const noexcept {
        auto &&info = _slot_info(slot);
        if (!info.valid) { return nullptr; }
        if (info.single_store != nullptr) { return info.single_store; }
        if (auto iter = info.local_reaching_stores.find(load);
            iter != info.local_reaching_stores.end()) {
            return iter->second;
        }
        return nullptr;
    }

    [[nodiscard]] bool _snapshot_reaches_use_unchanged(
        LoadInst *load, AllocaInst *slot,
        Instruction *use) const noexcept {
        auto &by_use = _unchanged_snapshot_cache[load];
        if (auto iter = by_use.find(use); iter != by_use.end()) {
            return iter->second;
        }
        const auto cache_result = [&](bool result) noexcept {
            by_use.emplace(use, result);
            return result;
        };
        if (load == nullptr || slot == nullptr || use == nullptr ||
            !_instruction_dominates(load, use) ||
            !scalar_slot_has_only_direct_accesses(slot)) {
            return cache_result(false);
        }
        auto load_location = _locations.find(load);
        auto use_location = _locations.find(use);
        if (load_location == _locations.end() ||
            use_location == _locations.end()) {
            return cache_result(false);
        }

        struct WorkItem {
            size_t block;
            bool dirty;
        };
        luisa::vector<uint8_t> reached(_graph.block_count(), 0u);
        luisa::vector<WorkItem> worklist;
        auto reached_use = false;

        const auto scan_block = [&](size_t block_id, bool dirty,
                                    size_t first_ordinal,
                                    auto &&enqueue) noexcept {
            auto ordinal = size_t{0u};
            for (auto *instruction :
                 _graph.block(block_id)->instructions()) {
                if (ordinal++ < first_ordinal) { continue; }
                if (instruction == load) { dirty = false; }
                if (instruction == use) {
                    reached_use = true;
                    if (dirty) { return false; }
                }
                if (instruction->isa<StoreInst>() &&
                    static_cast<StoreInst *>(instruction)->variable() ==
                        slot) {
                    dirty = true;
                }
            }
            for (auto successor : _graph.successors(block_id)) {
                enqueue(successor, dirty);
            }
            return true;
        };
        const auto enqueue = [&](size_t block, bool dirty) noexcept {
            auto bit = static_cast<uint8_t>(dirty ? 2u : 1u);
            if ((reached[block] & bit) == 0u) {
                reached[block] |= bit;
                worklist.emplace_back(WorkItem{block, dirty});
            }
        };

        // The first dynamic state begins immediately after this load. A
        // later re-entry scans the whole block and the same load resets the
        // dirty bit, which precisely models a new snapshot instance.
        if (!scan_block(
                load_location->second.block_id, false,
                load_location->second.ordinal + 1u, enqueue)) {
            return cache_result(false);
        }
        for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
            auto item = worklist[cursor];
            if (!scan_block(item.block, item.dirty, 0u, enqueue)) {
                return cache_result(false);
            }
        }
        return cache_result(reached_use);
    }

public:
    explicit ScalarCopyResolver(
        const PrefixInstructionLocationMap &locations,
        const CoroSemanticGraph &graph) noexcept
        : _locations{locations}, _graph{graph} {}

    // A scalar local is an exact snapshot only after a proven block-local
    // reaching store and before the consuming instruction. This restriction
    // rules out loop-carried and cross-edge substitution without memory SSA.
    [[nodiscard]] Value *resolve(
        Value *value, Instruction *use,
        size_t depth = 0u) const noexcept {
        if (value == nullptr || use == nullptr || depth >= 32u ||
            !value->isa<LoadInst>()) {
            return value;
        }
        auto *load = static_cast<LoadInst *>(value);
        auto *variable = load->variable();
        if (variable == nullptr || !variable->isa<AllocaInst>()) {
            return value;
        }
        auto *slot = static_cast<AllocaInst *>(variable);
        auto *store = _reaching_store(slot, load);
        if (store == nullptr ||
            !_instruction_dominates(store, load) ||
            !_instruction_dominates(load, use)) {
            return value;
        }
        return resolve(store->value(), store, depth + 1u);
    }

    [[nodiscard]] bool scalar_slot_has_only_direct_accesses(
        AllocaInst *slot) const noexcept {
        if (slot == nullptr || !slot->is_local() ||
            slot->type() == nullptr || !slot->type()->is_scalar()) {
            return false;
        }
        for (auto *use : slot->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<Instruction>()) {
                return false;
            }
            auto *instruction = static_cast<Instruction *>(user);
            if ((instruction->isa<LoadInst>() &&
                 static_cast<LoadInst *>(instruction)->variable() == slot) ||
                (instruction->isa<StoreInst>() &&
                 static_cast<StoreInst *>(instruction)->variable() == slot)) {
                continue;
            }
            return false;
        }
        return true;
    }

    [[nodiscard]] bool snapshot_reaches_use_unchanged(
        LoadInst *load, AllocaInst *slot,
        Instruction *use) const noexcept {
        return _snapshot_reaches_use_unchanged(load, slot, use);
    }
};

[[nodiscard]] luisa::optional<uint64_t> decode_unsigned_constant(
    Value *value) noexcept {
    uint64_t decoded = 0u;
    if (value == nullptr || !value->isa<Constant>() ||
        !try_decode_constant_nonnegative_integer(value, decoded)) {
        return luisa::nullopt;
    }
    return decoded;
}

[[nodiscard]] uint64_t unsigned_type_max(
    const Type *type) noexcept {
    if (type == nullptr) { return 0u; }
    switch (type->tag()) {
        case Type::Tag::UINT8:
            return std::numeric_limits<uint8_t>::max();
        case Type::Tag::UINT16:
            return std::numeric_limits<uint16_t>::max();
        case Type::Tag::UINT32:
            return std::numeric_limits<uint32_t>::max();
        case Type::Tag::UINT64:
            return std::numeric_limits<uint64_t>::max();
        default: return 0u;
    }
}

[[nodiscard]] bool has_store_between(
    AllocaInst *slot, Instruction *before, Instruction *after,
    const PrefixInstructionLocationMap &locations) noexcept {
    if (slot == nullptr || before == nullptr || after == nullptr ||
        before->parent_block() != after->parent_block()) {
        return true;
    }
    auto before_iter = locations.find(before);
    auto after_iter = locations.find(after);
    if (before_iter == locations.end() || after_iter == locations.end() ||
        before_iter->second.ordinal >= after_iter->second.ordinal) {
        return true;
    }
    for (auto *use : slot->use_list()) {
        auto *user = use == nullptr ? nullptr : use->user();
        if (user == nullptr || !user->isa<StoreInst>()) { continue; }
        auto *store = static_cast<StoreInst *>(user);
        if (store->variable() != slot ||
            store->parent_block() != before->parent_block()) {
            continue;
        }
        auto iter = locations.find(store);
        if (iter != locations.end() &&
            before_iter->second.ordinal < iter->second.ordinal &&
            iter->second.ordinal < after_iter->second.ordinal) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool is_current_slot_snapshot(
    Value *value, AllocaInst *slot, Instruction *use,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    const auto matches = [&](Value *candidate) noexcept {
        if (candidate == nullptr || !candidate->isa<LoadInst>()) {
            return false;
        }
        auto *load = static_cast<LoadInst *>(candidate);
        return load->variable() == slot &&
               ((load->parent_block() == use->parent_block() &&
                 instruction_precedes(load, use, locations) &&
                 !has_store_between(slot, load, use, locations)) ||
                resolver.snapshot_reaches_use_unchanged(
                    load, slot, use));
    };
    // Exact scalar forwarding may replace a current load by its constant or
    // copy definition. Preserve the original load-identity proof before
    // consulting that canonical value; forwarding must not erase a true
    // memory-state relation.
    if (matches(value)) { return true; }
    return matches(resolver.resolve(value, use));
}

[[nodiscard]] AllocaInst *current_scalar_slot(
    Value *value, Instruction *use,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    const auto direct_slot = [&](Value *candidate) noexcept {
        if (candidate == nullptr || !candidate->isa<LoadInst>()) {
            return static_cast<AllocaInst *>(nullptr);
        }
        auto *load = static_cast<LoadInst *>(candidate);
        auto *variable = load->variable();
        if (variable == nullptr || !variable->isa<AllocaInst>()) {
            return static_cast<AllocaInst *>(nullptr);
        }
        auto *slot = static_cast<AllocaInst *>(variable);
        if (!resolver.scalar_slot_has_only_direct_accesses(slot)) {
            return static_cast<AllocaInst *>(nullptr);
        }
        auto unchanged =
            (load->parent_block() == use->parent_block() &&
             instruction_precedes(load, use, locations) &&
             !has_store_between(slot, load, use, locations)) ||
            resolver.snapshot_reaches_use_unchanged(load, slot, use);
        return unchanged ? slot : nullptr;
    };
    auto *fallback = direct_slot(value);
    value = resolver.resolve(value, use);
    if (auto *resolved = direct_slot(value)) { return resolved; }
    return fallback;
}

[[nodiscard]] bool same_scalar_snapshot(
    Value *lhs, Value *rhs, Instruction *use,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (depth >= 16u) { return false; }
    lhs = resolver.resolve(lhs, use);
    rhs = resolver.resolve(rhs, use);
    if (lhs == rhs) { return true; }
    auto lhs_constant = decode_unsigned_constant(lhs);
    auto rhs_constant = decode_unsigned_constant(rhs);
    if (lhs_constant && rhs_constant) {
        return *lhs_constant == *rhs_constant &&
               lhs->type() == rhs->type();
    }
    if (lhs != nullptr && rhs != nullptr &&
        lhs->isa<LoadInst>() && rhs->isa<LoadInst>()) {
        auto *lhs_load = static_cast<LoadInst *>(lhs);
        auto *rhs_load = static_cast<LoadInst *>(rhs);
        auto *variable = lhs_load->variable();
        if (variable == rhs_load->variable() && variable != nullptr &&
            variable->isa<AllocaInst>()) {
            auto *slot = static_cast<AllocaInst *>(variable);
            return is_current_slot_snapshot(
                       lhs, slot, use, resolver, locations) &&
                   is_current_slot_snapshot(
                       rhs, slot, use, resolver, locations);
        }
    }
    if (lhs == nullptr || rhs == nullptr ||
        !lhs->isa<ArithmeticInst>() || !rhs->isa<ArithmeticInst>()) {
        return false;
    }
    auto *lhs_arithmetic = static_cast<ArithmeticInst *>(lhs);
    auto *rhs_arithmetic = static_cast<ArithmeticInst *>(rhs);
    if (lhs_arithmetic->op() != rhs_arithmetic->op() ||
        lhs_arithmetic->type() != rhs_arithmetic->type() ||
        lhs_arithmetic->operand_count() != rhs_arithmetic->operand_count()) {
        return false;
    }
    for (size_t i = 0u; i < lhs_arithmetic->operand_count(); ++i) {
        if (!same_scalar_snapshot(
                lhs_arithmetic->operand(i), rhs_arithmetic->operand(i),
                use, resolver, locations, depth + 1u)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] GEPInst *top_array_gep(
    Value *pointer, AllocaInst *array) noexcept {
    while (pointer != nullptr && pointer != array &&
           pointer->isa<GEPInst>()) {
        auto *gep = static_cast<GEPInst *>(pointer);
        if (gep->base() == array) { return gep; }
        pointer = gep->base();
    }
    return nullptr;
}

struct RelationSlotCollection {
    luisa::vector<AllocaInst *> slots;
    CoroScalarSemanticUses semantic_uses;
};

[[nodiscard]] RelationSlotCollection collect_relation_slots(
    AllocaInst *array, AllocaInst *counter,
    const ArrayUseRegion &region,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    RelationSlotCollection result;
    const auto insert = [&](AllocaInst *slot) noexcept {
        if (slot != nullptr && slot != counter &&
            slot->type() == counter->type() &&
            slot->type()->is_uint() &&
            std::find(result.slots.begin(), result.slots.end(), slot) ==
                result.slots.end()) {
            result.slots.emplace_back(slot);
        }
    };
    for (auto *instruction : region.users) {
        if (!instruction->isa<LoadInst>()) { continue; }
        auto *pointer = static_cast<LoadInst *>(instruction)->variable();
        auto *element = top_array_gep(pointer, array);
        if (element != nullptr && element->index_count() != 0u) {
            auto *slot = current_scalar_slot(
                element->index(0u), instruction, resolver, locations);
            insert(slot);
            if (slot != nullptr) {
                result.semantic_uses[instruction].emplace_back(slot);
            }
        }
    }
    // Close backwards over exact scalar copies. If D is live as an array
    // index, a relation carried by S is needed until every D := S transfer
    // has executed. Unsupported RHS expressions simply add no source and
    // leave D unknown, which is conservative.
    for (size_t cursor = 0u; cursor < result.slots.size(); ++cursor) {
        auto *destination = result.slots[cursor];
        for (auto *use : destination->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<StoreInst>()) { continue; }
            auto *store = static_cast<StoreInst *>(user);
            if (store->variable() != destination) { continue; }
            auto *source = current_scalar_slot(
                store->value(), store, resolver, locations);
            insert(source);
            if (source != nullptr) {
                result.semantic_uses[store].emplace_back(source);
            }
        }
    }
    return result;
}

[[nodiscard]] AllocaInst *counter_from_full_element_store(
    StoreInst *store, AllocaInst *array,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    auto *gep = top_array_gep(store->variable(), array);
    if (gep == nullptr || store->variable() != gep ||
        gep->base() != array ||
        gep->index_count() != 1u ||
        gep->type() != array->type()->element()) {
        return nullptr;
    }
    auto *index = resolver.resolve(gep->index(0u), gep);
    if (index == nullptr || !index->isa<LoadInst>()) { return nullptr; }
    auto *variable = static_cast<LoadInst *>(index)->variable();
    if (variable == nullptr || !variable->isa<AllocaInst>()) {
        return nullptr;
    }
    auto *counter = static_cast<AllocaInst *>(variable);
    return counter->type() != nullptr && counter->type()->is_uint() &&
                   is_current_slot_snapshot(
                       index, counter, gep, resolver, locations) ?
               counter :
               nullptr;
}

struct FactLayout {
    luisa::unordered_map<size_t, size_t> atoms;
    luisa::unordered_map<Value *, size_t> pointers;
    size_t count{0u};
};

[[nodiscard]] FactLayout make_fact_layout(
    AllocaInst *array, const ArrayUseRegion &region,
    const CoroFrameAtomDomain &domain) noexcept {
    FactLayout layout;
    for (size_t i = 0u; i < domain.size(); ++i) {
        if (domain.atom(i).root == array) {
            layout.atoms.emplace(i, layout.count++);
        }
    }
    for (auto *pointer : region.pointers) {
        layout.pointers.emplace(pointer, layout.count++);
    }
    return layout;
}

using FactState = luisa::vector<uint8_t>;

void redefine_pointer(Value *pointer, FactState &facts,
                      const FactLayout &layout) noexcept {
    if (auto iter = layout.pointers.find(pointer);
        iter != layout.pointers.end()) {
        facts[iter->second] = 0u;
    }
}

void define_pointer(Value *pointer, FactState &facts,
                    const FactLayout &layout,
                    const CoroFrameAtomDomain &domain) noexcept {
    if (auto iter = layout.pointers.find(pointer);
        iter != layout.pointers.end()) {
        facts[iter->second] = 1u;
    }
    for (auto access : domain.memory_accesses(pointer)) {
        if (!access.covers_atom) { continue; }
        if (auto iter = layout.atoms.find(access.atom_index);
            iter != layout.atoms.end()) {
            facts[iter->second] = 1u;
        }
    }
}

[[nodiscard]] bool pointer_is_defined(
    Value *pointer, const FactState &facts,
    const FactLayout &layout,
    const CoroFrameAtomDomain &domain) noexcept {
    if (auto iter = layout.pointers.find(pointer);
        iter != layout.pointers.end() && facts[iter->second] != 0u) {
        return true;
    }
    auto found = false;
    for (auto access : domain.memory_accesses(pointer)) {
        if (auto iter = layout.atoms.find(access.atom_index);
            iter != layout.atoms.end()) {
            found = true;
            if (facts[iter->second] == 0u) { return false; }
        }
    }
    return found;
}

[[nodiscard]] bool static_array_element_is_defined(
    Value *index, Instruction *use, AllocaInst *array,
    const FactState &facts, const FactLayout &layout,
    const CoroFrameAtomDomain &domain,
    const ScalarCopyResolver &resolver) noexcept {
    index = resolver.resolve(index, use);
    auto decoded = decode_unsigned_constant(index);
    if (!decoded || *decoded >= array->type()->dimension()) {
        return false;
    }
    auto found = false;
    for (auto &[atom_index, fact_index] : layout.atoms) {
        auto &&atom = domain.atom(atom_index);
        if (atom.root != array) { continue; }
        if (atom.access_chain.empty() ||
            atom.access_chain.front() == *decoded) {
            found = true;
            if (facts[fact_index] == 0u) { return false; }
        }
    }
    return found;
}

[[nodiscard]] Value *strip_boolean_wrappers(
    Value *condition, bool &truth, Instruction *use,
    const ScalarCopyResolver &resolver) noexcept {
    for (auto depth = 0u; depth < 16u; ++depth) {
        condition = resolver.resolve(condition, use);
        if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
            break;
        }
        auto *arithmetic = static_cast<ArithmeticInst *>(condition);
        if (arithmetic->op() == ArithmeticOp::UNARY_BIT_NOT &&
            arithmetic->operand_count() == 1u &&
            arithmetic->operand(0u)->type() == Type::of<bool>()) {
            condition = arithmetic->operand(0u);
            truth = !truth;
            continue;
        }
        if ((arithmetic->op() == ArithmeticOp::BINARY_EQUAL ||
             arithmetic->op() == ArithmeticOp::BINARY_NOT_EQUAL ||
             arithmetic->op() == ArithmeticOp::BINARY_BIT_XOR) &&
            arithmetic->operand_count() == 2u) {
            Value *variable = nullptr;
            luisa::optional<bool> constant;
            for (auto constant_operand = 0u;
                 constant_operand < 2u; ++constant_operand) {
                auto *operand = arithmetic->operand(constant_operand);
                if (operand != nullptr && operand->isa<Constant>() &&
                    operand->type() == Type::of<bool>()) {
                    variable = arithmetic->operand(1u - constant_operand);
                    constant = static_cast<Constant *>(operand)->as<bool>();
                    break;
                }
            }
            if (variable != nullptr && constant &&
                variable->type() == Type::of<bool>()) {
                auto invert = arithmetic->op() == ArithmeticOp::BINARY_EQUAL ?
                                  !*constant :
                                  *constant;
                condition = variable;
                truth ^= invert;
                continue;
            }
        }
        break;
    }
    return condition;
}

struct MaskedScalarTest {
    AllocaInst *scalar{nullptr};
    uint64_t mask{0u};
    bool nonzero_when_condition_true{true};
};

struct ScalarZeroTest {
    AllocaInst *scalar{nullptr};
    bool zero_when_condition_true{true};
};

// Recognizes an exact Boolean test of the form `(S & M) != 0` or its
// equality/Boolean-negation dual. `S` must be an unchanged snapshot of a
// direct unsigned scalar local at the use. This is deliberately narrower
// than arbitrary known-bits analysis: unsupported arithmetic yields no fact.
[[nodiscard]] luisa::optional<MaskedScalarTest> match_masked_scalar_test(
    Value *condition, Instruction *use,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    auto underlying_truth_when_condition_true = true;
    condition = strip_boolean_wrappers(
        condition, underlying_truth_when_condition_true, use, resolver);
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        return luisa::nullopt;
    }
    auto *comparison = static_cast<ArithmeticInst *>(condition);
    if ((comparison->op() != ArithmeticOp::BINARY_EQUAL &&
         comparison->op() != ArithmeticOp::BINARY_NOT_EQUAL) ||
        comparison->operand_count() != 2u ||
        comparison->type() != Type::of<bool>()) {
        return luisa::nullopt;
    }
    Value *masked_value = nullptr;
    for (auto zero_operand = 0u; zero_operand < 2u; ++zero_operand) {
        auto zero = decode_unsigned_constant(
            resolver.resolve(
                comparison->operand(zero_operand), use));
        if (zero && *zero == 0u) {
            masked_value = comparison->operand(1u - zero_operand);
            break;
        }
    }
    masked_value = resolver.resolve(masked_value, use);
    if (masked_value == nullptr ||
        !masked_value->isa<ArithmeticInst>()) {
        return luisa::nullopt;
    }
    auto *bit_and = static_cast<ArithmeticInst *>(masked_value);
    if (bit_and->op() != ArithmeticOp::BINARY_BIT_AND ||
        bit_and->operand_count() != 2u ||
        bit_and->type() == nullptr ||
        !bit_and->type()->is_uint()) {
        return luisa::nullopt;
    }
    for (auto mask_operand = 0u; mask_operand < 2u; ++mask_operand) {
        auto mask = decode_unsigned_constant(
            resolver.resolve(
                bit_and->operand(mask_operand), use));
        if (!mask || *mask == 0u) { continue; }
        auto *scalar = current_scalar_slot(
            bit_and->operand(1u - mask_operand), use,
            resolver, locations);
        if (scalar != nullptr && scalar->type() == bit_and->type()) {
            auto comparison_true_means_nonzero =
                comparison->op() == ArithmeticOp::BINARY_NOT_EQUAL;
            return MaskedScalarTest{
                .scalar = scalar,
                .mask = *mask,
                .nonzero_when_condition_true =
                    underlying_truth_when_condition_true ==
                    comparison_true_means_nonzero};
        }
    }
    return luisa::nullopt;
}

// Recognizes an exact Boolean test S==0 or S!=0 (including Boolean wrapper
// duals) for the current snapshot of a direct unsigned scalar local.
[[nodiscard]] luisa::optional<ScalarZeroTest> match_scalar_zero_test(
    Value *condition, Instruction *use,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    auto underlying_truth_when_condition_true = true;
    condition = strip_boolean_wrappers(
        condition, underlying_truth_when_condition_true, use, resolver);
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        return luisa::nullopt;
    }
    auto *comparison = static_cast<ArithmeticInst *>(condition);
    if ((comparison->op() != ArithmeticOp::BINARY_EQUAL &&
         comparison->op() != ArithmeticOp::BINARY_NOT_EQUAL) ||
        comparison->operand_count() != 2u ||
        comparison->type() != Type::of<bool>()) {
        return luisa::nullopt;
    }
    for (auto zero_operand = 0u; zero_operand < 2u; ++zero_operand) {
        auto zero = decode_unsigned_constant(
            resolver.resolve(
                comparison->operand(zero_operand), use));
        if (!zero || *zero != 0u) { continue; }
        auto *scalar = current_scalar_slot(
            comparison->operand(1u - zero_operand), use,
            resolver, locations);
        if (scalar == nullptr || scalar->type() == nullptr ||
            !scalar->type()->is_uint()) {
            continue;
        }
        auto comparison_true_means_zero =
            comparison->op() == ArithmeticOp::BINARY_EQUAL;
        return ScalarZeroTest{
            .scalar = scalar,
            .zero_when_condition_true =
                underlying_truth_when_condition_true ==
                comparison_true_means_zero};
    }
    return luisa::nullopt;
}

struct MaskedScalarWitnessSet {
    luisa::vector<CoroMaskedScalarWitness> roots;
    luisa::vector<CoroMaskedScalarWitness> tracked;
};

[[nodiscard]] MaskedScalarWitnessSet
collect_masked_scalar_witnesses(
    const CoroSemanticGraph &graph,
    luisa::span<const uint8_t> active_blocks,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    MaskedScalarWitnessSet result;
    const auto insert_tracked = [&](AllocaInst *scalar,
                                    uint64_t mask) noexcept {
        if (std::none_of(
                result.tracked.begin(), result.tracked.end(),
                [&](auto witness) noexcept {
                    return witness.scalar == scalar &&
                           witness.mask == mask;
                })) {
            result.tracked.emplace_back(CoroMaskedScalarWitness{
                scalar, mask});
            return true;
        }
        return false;
    };
    const auto insert_root = [&](MaskedScalarTest test) noexcept {
        if (std::none_of(
                result.roots.begin(), result.roots.end(),
                [&](auto witness) noexcept {
                    return witness.scalar == test.scalar &&
                           witness.mask == test.mask;
                })) {
            result.roots.emplace_back(CoroMaskedScalarWitness{
                test.scalar, test.mask});
        }
        static_cast<void>(insert_tracked(test.scalar, test.mask));
    };
    for (size_t block_id = 0u;
         block_id < graph.block_count(); ++block_id) {
        // Only conditions in the backward slice from the candidate's array
        // uses to its proposed declaration can affect this proof. Tracking a
        // masked scalar outside that closed CFG slice cannot refine any state
        // reaching a relevant use, while adding it would multiply the BDD
        // product by an observationally irrelevant component.
        if (block_id >= active_blocks.size() ||
            active_blocks[block_id] == 0u) {
            continue;
        }
        for (auto *instruction : graph.block(block_id)->instructions()) {
            Value *condition = nullptr;
            if (instruction->isa<ConditionalBranchInst>()) {
                condition = static_cast<ConditionalBranchInst *>(
                                instruction)
                                ->condition();
            } else if (instruction->isa<AssumeInst>()) {
                condition = static_cast<AssumeInst *>(instruction)
                                ->condition();
            }
            if (condition != nullptr) {
                if (auto test = match_masked_scalar_test(
                        condition, instruction, resolver, locations)) {
                    insert_root(*test);
                }
            }
        }
    }

    // Close the observed witnesses backwards over exact scalar def-use
    // chains. Lowering is free to materialize `S = tmp | K` instead of
    // `S = S | K`; tracking only S would then mistake an unrelated copy slot
    // for an unconstrained source. The least fixed point below adds exactly
    // the unsigned scalar memory snapshots on which a tracked definition
    // depends. It creates no relation by itself: ordinary forward transfer
    // must still prove every copy/bitwise step, and unsupported definitions
    // remain fail-closed.
    for (size_t cursor = 0u; cursor < result.tracked.size(); ++cursor) {
        auto witness = result.tracked[cursor];
        for (auto *use : witness.scalar->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<StoreInst>()) { continue; }
            auto *store = static_cast<StoreInst *>(user);
            if (store->variable() != witness.scalar) { continue; }
            auto location = locations.find(store);
            if (location == locations.end() ||
                location->second.block_id >= active_blocks.size() ||
                active_blocks[location->second.block_id] == 0u) {
                continue;
            }
            luisa::unordered_set<Value *> visited;
            luisa::vector<Value *> worklist{store->value()};
            while (!worklist.empty()) {
                auto *value = worklist.back();
                worklist.pop_back();
                if (value == nullptr || !visited.emplace(value).second) {
                    continue;
                }
                if (auto *source = current_scalar_slot(
                        value, store, resolver, locations);
                    source != nullptr &&
                    source->type() == witness.scalar->type()) {
                    static_cast<void>(insert_tracked(
                        source, witness.mask));
                    // A scalar load is the memory boundary of this
                    // expression. Its own reaching definitions are visited
                    // when the newly inserted witness reaches the worklist.
                    continue;
                }
                if (!value->isa<Instruction>()) { continue; }
                auto *instruction = static_cast<Instruction *>(value);
                for (size_t operand = 0u;
                     operand < instruction->operand_count(); ++operand) {
                    worklist.emplace_back(
                        instruction->operand(operand));
                }
            }
        }
    }
    return result;
}

struct BooleanExpressionSource {
    Value *predicate{nullptr};
    bool true_when_predicate_is{true};
    luisa::optional<bool> constant;
};

[[nodiscard]] BooleanExpressionSource analyze_boolean_expression(
    Value *value, Instruction *use,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    auto true_when_base_is = true;
    for (auto depth = 0u; depth < 16u; ++depth) {
        // A direct Boolean load is the durable memory-state predicate. Do not
        // canonicalize it to its reaching RHS first: that can be an argument
        // or an older slot whose later mutation does not change this copy.
        if (auto *slot = current_scalar_slot(
                value, use, resolver, locations);
            slot != nullptr && slot->type() == Type::of<bool>()) {
            return BooleanExpressionSource{
                .predicate = slot,
                .true_when_predicate_is = true_when_base_is};
        }
        auto *resolved = resolver.resolve(value, use);
        if (resolved != value) {
            value = resolved;
            continue;
        }
        if (value != nullptr && value->isa<Constant>() &&
            value->type() == Type::of<bool>()) {
            auto base_value = static_cast<Constant *>(value)->as<bool>();
            return BooleanExpressionSource{
                .true_when_predicate_is = true_when_base_is,
                .constant = base_value == true_when_base_is};
        }
        if (value == nullptr || !value->isa<ArithmeticInst>()) {
            if (value != nullptr && value->type() == Type::of<bool>()) {
                return BooleanExpressionSource{
                    .predicate = value,
                    .true_when_predicate_is = true_when_base_is};
            }
            break;
        }
        auto *arithmetic = static_cast<ArithmeticInst *>(value);
        if (arithmetic->op() == ArithmeticOp::UNARY_BIT_NOT &&
            arithmetic->operand_count() == 1u &&
            arithmetic->operand(0u)->type() == Type::of<bool>()) {
            value = arithmetic->operand(0u);
            true_when_base_is = !true_when_base_is;
            continue;
        }
        if ((arithmetic->op() == ArithmeticOp::BINARY_EQUAL ||
             arithmetic->op() == ArithmeticOp::BINARY_NOT_EQUAL ||
             arithmetic->op() == ArithmeticOp::BINARY_BIT_XOR) &&
            arithmetic->operand_count() == 2u) {
            Value *variable = nullptr;
            luisa::optional<bool> constant;
            for (auto constant_operand = 0u;
                 constant_operand < 2u; ++constant_operand) {
                auto *operand = arithmetic->operand(constant_operand);
                if (operand != nullptr && operand->isa<Constant>() &&
                    operand->type() == Type::of<bool>()) {
                    variable = arithmetic->operand(1u - constant_operand);
                    constant = static_cast<Constant *>(operand)->as<bool>();
                    break;
                }
            }
            if (variable != nullptr && constant &&
                variable->type() == Type::of<bool>()) {
                auto invert =
                    arithmetic->op() == ArithmeticOp::BINARY_EQUAL ?
                        !*constant :
                        *constant;
                value = variable;
                true_when_base_is ^= invert;
                continue;
            }
        }
        if (value->type() == Type::of<bool>()) {
            return BooleanExpressionSource{
                .predicate = value,
                .true_when_predicate_is = true_when_base_is};
        }
        break;
    }
    return {};
}

[[nodiscard]] luisa::unordered_set<Value *>
collect_boolean_guard_slots(
    const CoroSemanticGraph &graph,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    luisa::unordered_set<Value *> result;
    luisa::vector<Value *> worklist;
    const auto insert = [&](Value *predicate) noexcept {
        if (predicate != nullptr &&
            predicate->type() == Type::of<bool>() &&
            result.emplace(predicate).second) {
            worklist.emplace_back(predicate);
        }
    };
    for (size_t block_id = 0u; block_id < graph.block_count(); ++block_id) {
        auto *terminator = graph.block(block_id)->terminator();
        if (terminator == nullptr ||
            !terminator->isa<ConditionalBranchInst>()) {
            continue;
        }
        auto *branch = static_cast<ConditionalBranchInst *>(terminator);
        insert(analyze_boolean_expression(
                   branch->condition(), branch, resolver, locations)
                   .predicate);
    }
    // A branch-visible Boolean may be copied through any number of direct
    // scalar locals. Close the guard domain backwards over those exact copy
    // definitions so an implication created for the source can be forwarded
    // to the eventually tested destination.
    for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
        auto *destination = worklist[cursor];
        if (!destination->isa<AllocaInst>()) { continue; }
        for (auto *use : destination->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<StoreInst>()) { continue; }
            auto *store = static_cast<StoreInst *>(user);
            if (store->variable() != destination) { continue; }
            insert(analyze_boolean_expression(
                       store->value(), store, resolver, locations)
                       .predicate);
        }
    }
    return result;
}

struct BooleanPredicateFlow {
    luisa::vector<Value *> predicates;
    CoroBooleanSemanticValues uses;
    CoroBooleanSemanticValues definitions;
};

[[nodiscard]] BooleanPredicateFlow collect_boolean_predicate_flow(
    const CoroSemanticGraph &graph,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations,
    const luisa::unordered_set<Value *> &boolean_guards) noexcept {
    BooleanPredicateFlow result;
    result.predicates.reserve(boolean_guards.size());
    for (auto *predicate : boolean_guards) {
        result.predicates.emplace_back(predicate);
    }
    const auto insert = [&](auto &map, Instruction *instruction,
                            Value *predicate) noexcept {
        if (predicate != nullptr &&
            boolean_guards.contains(predicate)) {
            map[instruction].emplace_back(predicate);
        }
    };
    for (size_t block_id = 0u; block_id < graph.block_count(); ++block_id) {
        for (auto *instruction : graph.block(block_id)->instructions()) {
            if (instruction->type() == Type::of<bool>() &&
                boolean_guards.contains(instruction)) {
                insert(result.definitions, instruction, instruction);
            }
            if (instruction->isa<StoreInst>()) {
                auto *store = static_cast<StoreInst *>(instruction);
                auto *pointer = store->variable();
                auto *destination =
                    pointer != nullptr && pointer->isa<AllocaInst>() ?
                        static_cast<AllocaInst *>(pointer) :
                        nullptr;
                if (destination != nullptr &&
                    destination->type() == Type::of<bool>() &&
                    boolean_guards.contains(destination)) {
                    insert(result.definitions, store, destination);
                    auto source = analyze_boolean_expression(
                        store->value(), store, resolver, locations);
                    insert(result.uses, store, source.predicate);
                }
            }
            if (instruction->isa<ConditionalBranchInst>()) {
                auto *branch =
                    static_cast<ConditionalBranchInst *>(instruction);
                auto source = analyze_boolean_expression(
                    branch->condition(), branch, resolver, locations);
                insert(result.uses, branch, source.predicate);
            }
        }
    }
    return result;
}

// Returns U only when taking the selected edge proves the unsigned relation
// C <= U for the current value of counter C. This is an edge refinement, not
// a guess about branch probability: comparisons are accepted only when the
// counter operand is an exact snapshot at the terminator and the other operand
// is a compile-time non-negative integer. For Boolean conjunction, a true
// result proves both operands; dually, a false disjunction proves both false.
[[nodiscard]] luisa::optional<size_t>
condition_implied_counter_upper_bound(
    Value *condition, bool truth, AllocaInst *counter,
    Instruction *terminator, const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (depth >= 16u) { return luisa::nullopt; }
    condition = strip_boolean_wrappers(
        condition, truth, terminator, resolver);
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        return luisa::nullopt;
    }
    auto *arithmetic = static_cast<ArithmeticInst *>(condition);
    if (arithmetic->operand_count() != 2u) {
        return luisa::nullopt;
    }

    const auto combine_implied_bounds = [&]() noexcept {
        auto lhs = condition_implied_counter_upper_bound(
            arithmetic->operand(0u), truth, counter, terminator,
            resolver, locations, depth + 1u);
        auto rhs = condition_implied_counter_upper_bound(
            arithmetic->operand(1u), truth, counter, terminator,
            resolver, locations, depth + 1u);
        if (lhs && rhs) { return luisa::optional<size_t>{std::min(*lhs, *rhs)}; }
        return lhs ? lhs : rhs;
    };
    if (truth && arithmetic->op() == ArithmeticOp::BINARY_BIT_AND &&
        arithmetic->type() == Type::of<bool>()) {
        return combine_implied_bounds();
    }
    if (!truth && arithmetic->op() == ArithmeticOp::BINARY_BIT_OR &&
        arithmetic->type() == Type::of<bool>()) {
        return combine_implied_bounds();
    }

    const auto decode_bound = [&](Value *value) noexcept
        -> luisa::optional<size_t> {
        auto decoded = decode_unsigned_constant(
            resolver.resolve(value, terminator));
        if (!decoded ||
            *decoded > std::numeric_limits<size_t>::max()) {
            return luisa::nullopt;
        }
        return static_cast<size_t>(*decoded);
    };
    const auto strict_predecessor = [](size_t bound) noexcept
        -> luisa::optional<size_t> {
        return bound == 0u ? luisa::nullopt :
                             luisa::optional<size_t>{bound - 1u};
    };

    auto *lhs = arithmetic->operand(0u);
    auto *rhs = arithmetic->operand(1u);
    if (is_current_slot_snapshot(
            lhs, counter, terminator, resolver, locations)) {
        auto bound = decode_bound(rhs);
        if (!bound) { return luisa::nullopt; }
        switch (arithmetic->op()) {
            case ArithmeticOp::BINARY_LESS:
                return truth ? strict_predecessor(*bound) :
                               luisa::nullopt;
            case ArithmeticOp::BINARY_LESS_EQUAL:
                return truth ? bound : luisa::nullopt;
            case ArithmeticOp::BINARY_GREATER:
                return truth ? luisa::nullopt : bound;
            case ArithmeticOp::BINARY_GREATER_EQUAL:
                return truth ? luisa::nullopt :
                               strict_predecessor(*bound);
            default: return luisa::nullopt;
        }
    }
    if (is_current_slot_snapshot(
            rhs, counter, terminator, resolver, locations)) {
        auto bound = decode_bound(lhs);
        if (!bound) { return luisa::nullopt; }
        switch (arithmetic->op()) {
            case ArithmeticOp::BINARY_GREATER:
                return truth ? strict_predecessor(*bound) :
                               luisa::nullopt;
            case ArithmeticOp::BINARY_GREATER_EQUAL:
                return truth ? bound : luisa::nullopt;
            case ArithmeticOp::BINARY_LESS:
                return truth ? luisa::nullopt : bound;
            case ArithmeticOp::BINARY_LESS_EQUAL:
                return truth ? luisa::nullopt :
                               strict_predecessor(*bound);
            default: return luisa::nullopt;
        }
    }
    return luisa::nullopt;
}

// Returns L only when taking the selected edge proves L <= C for the current
// unsigned counter value. This is the lower-bound dual of the analysis above;
// conjunction-true and disjunction-false edges expose every constituent fact,
// so their strongest implied lower bound is the maximum constituent bound.
[[nodiscard]] luisa::optional<size_t>
condition_implied_counter_lower_bound(
    Value *condition, bool truth, AllocaInst *counter,
    Instruction *terminator, const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (depth >= 16u) { return luisa::nullopt; }
    condition = strip_boolean_wrappers(
        condition, truth, terminator, resolver);
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        return luisa::nullopt;
    }
    auto *arithmetic = static_cast<ArithmeticInst *>(condition);
    if (arithmetic->operand_count() != 2u) {
        return luisa::nullopt;
    }

    const auto combine_implied_bounds = [&]() noexcept {
        auto lhs = condition_implied_counter_lower_bound(
            arithmetic->operand(0u), truth, counter, terminator,
            resolver, locations, depth + 1u);
        auto rhs = condition_implied_counter_lower_bound(
            arithmetic->operand(1u), truth, counter, terminator,
            resolver, locations, depth + 1u);
        if (lhs && rhs) {
            return luisa::optional<size_t>{std::max(*lhs, *rhs)};
        }
        return lhs ? lhs : rhs;
    };
    if (truth && arithmetic->op() == ArithmeticOp::BINARY_BIT_AND &&
        arithmetic->type() == Type::of<bool>()) {
        return combine_implied_bounds();
    }
    if (!truth && arithmetic->op() == ArithmeticOp::BINARY_BIT_OR &&
        arithmetic->type() == Type::of<bool>()) {
        return combine_implied_bounds();
    }

    const auto decode_bound = [&](Value *value) noexcept
        -> luisa::optional<size_t> {
        auto decoded = decode_unsigned_constant(
            resolver.resolve(value, terminator));
        if (!decoded ||
            *decoded > std::numeric_limits<size_t>::max()) {
            return luisa::nullopt;
        }
        return static_cast<size_t>(*decoded);
    };
    const auto strict_successor = [](size_t bound) noexcept
        -> luisa::optional<size_t> {
        return bound == std::numeric_limits<size_t>::max() ?
                   luisa::nullopt :
                   luisa::optional<size_t>{bound + 1u};
    };

    auto *lhs = arithmetic->operand(0u);
    auto *rhs = arithmetic->operand(1u);
    if (is_current_slot_snapshot(
            lhs, counter, terminator, resolver, locations)) {
        auto bound = decode_bound(rhs);
        if (!bound) { return luisa::nullopt; }
        switch (arithmetic->op()) {
            case ArithmeticOp::BINARY_GREATER:
                return truth ? strict_successor(*bound) :
                               luisa::nullopt;
            case ArithmeticOp::BINARY_GREATER_EQUAL:
                return truth ? bound : luisa::nullopt;
            case ArithmeticOp::BINARY_LESS:
                return truth ? luisa::nullopt : bound;
            case ArithmeticOp::BINARY_LESS_EQUAL:
                return truth ? luisa::nullopt :
                               strict_successor(*bound);
            default: return luisa::nullopt;
        }
    }
    if (is_current_slot_snapshot(
            rhs, counter, terminator, resolver, locations)) {
        auto bound = decode_bound(lhs);
        if (!bound) { return luisa::nullopt; }
        switch (arithmetic->op()) {
            case ArithmeticOp::BINARY_LESS:
                return truth ? strict_successor(*bound) :
                               luisa::nullopt;
            case ArithmeticOp::BINARY_LESS_EQUAL:
                return truth ? bound : luisa::nullopt;
            case ArithmeticOp::BINARY_GREATER:
                return truth ? luisa::nullopt : bound;
            case ArithmeticOp::BINARY_GREATER_EQUAL:
                return truth ? luisa::nullopt :
                               strict_successor(*bound);
            default: return luisa::nullopt;
        }
    }
    return luisa::nullopt;
}

[[nodiscard]] bool condition_implies_less_than_counter(
    Value *condition, bool truth, Value *index,
    AllocaInst *counter, Instruction *read,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    condition = strip_boolean_wrappers(
        condition, truth, read, resolver);
    if (!truth || condition == nullptr ||
        !condition->isa<ArithmeticInst>()) {
        return false;
    }
    auto *comparison = static_cast<ArithmeticInst *>(condition);
    if (comparison->operand_count() != 2u) { return false; }
    if (comparison->op() == ArithmeticOp::BINARY_LESS) {
        return same_scalar_snapshot(
                   comparison->operand(0u), index, read,
                   resolver, locations) &&
               is_current_slot_snapshot(
                   comparison->operand(1u), counter, read,
                   resolver, locations);
    }
    if (comparison->op() == ArithmeticOp::BINARY_GREATER) {
        return same_scalar_snapshot(
                   comparison->operand(1u), index, read,
                   resolver, locations) &&
               is_current_slot_snapshot(
                   comparison->operand(0u), counter, read,
                   resolver, locations);
    }
    return false;
}

// Returns a direct scalar slot I only when the selected edge proves the
// relation contents(I) < contents(C) for the current contents of counter C.
// This is a relation between memory states, not between two particular load
// instructions: it remains true across loads and forwarding blocks, and a
// subsequent store to either slot kills it in the transfer function below.
[[nodiscard]] AllocaInst *condition_implied_index_slot_less_than_counter(
    Value *condition, bool truth, AllocaInst *counter,
    Instruction *terminator, const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (depth >= 16u) { return nullptr; }
    condition = strip_boolean_wrappers(
        condition, truth, terminator, resolver);
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        return nullptr;
    }
    auto *arithmetic = static_cast<ArithmeticInst *>(condition);
    if (arithmetic->operand_count() != 2u) { return nullptr; }

    // For true conjunctions and false disjunctions, both operands hold with
    // the selected truth value. Finding the relation in either operand is
    // therefore sufficient. The dual cases would be disjunctive and are
    // deliberately rejected.
    auto decomposable =
        (truth && arithmetic->op() == ArithmeticOp::BINARY_BIT_AND) ||
        (!truth && arithmetic->op() == ArithmeticOp::BINARY_BIT_OR);
    if (decomposable && arithmetic->type() == Type::of<bool>()) {
        if (auto *slot = condition_implied_index_slot_less_than_counter(
                arithmetic->operand(0u), truth, counter, terminator,
                resolver, locations, depth + 1u)) {
            return slot;
        }
        return condition_implied_index_slot_less_than_counter(
            arithmetic->operand(1u), truth, counter, terminator,
            resolver, locations, depth + 1u);
    }

    auto *lhs = arithmetic->operand(0u);
    auto *rhs = arithmetic->operand(1u);
    auto *lhs_slot = current_scalar_slot(
        lhs, terminator, resolver, locations);
    auto *rhs_slot = current_scalar_slot(
        rhs, terminator, resolver, locations);
    const auto valid_index_slot = [counter](AllocaInst *slot) noexcept {
        return slot != nullptr && slot != counter &&
               slot->type() == counter->type() &&
               slot->type()->is_uint();
    };
    auto lhs_is_counter = is_current_slot_snapshot(
        lhs, counter, terminator, resolver, locations);
    auto rhs_is_counter = is_current_slot_snapshot(
        rhs, counter, terminator, resolver, locations);

    if (valid_index_slot(lhs_slot) && rhs_is_counter &&
        ((truth && arithmetic->op() == ArithmeticOp::BINARY_LESS) ||
         (!truth && arithmetic->op() ==
                        ArithmeticOp::BINARY_GREATER_EQUAL))) {
        return lhs_slot;
    }
    if (valid_index_slot(rhs_slot) && lhs_is_counter &&
        ((truth && arithmetic->op() == ArithmeticOp::BINARY_GREATER) ||
         (!truth && arithmetic->op() ==
                        ArithmeticOp::BINARY_LESS_EQUAL))) {
        return rhs_slot;
    }
    return nullptr;
}

// Returns a direct scalar slot I only when the selected edge proves
//
//                       I + 1 == C && C > 0.
//
// The positivity side condition excludes unsigned wrap and makes I the
// unique last element of the counted prefix. As with the less-than matcher,
// only conjunction-true and disjunction-false expressions may be decomposed.
// The caller establishes C > 0 independently before admitting the relation.
[[nodiscard]] AllocaInst *condition_implied_index_slot_last_before_counter(
    Value *condition, bool truth, AllocaInst *counter,
    Instruction *terminator, const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (depth >= 16u) { return nullptr; }
    condition = strip_boolean_wrappers(
        condition, truth, terminator, resolver);
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        return nullptr;
    }
    auto *arithmetic = static_cast<ArithmeticInst *>(condition);
    if (arithmetic->operand_count() != 2u) { return nullptr; }

    auto decomposable =
        (truth && arithmetic->op() == ArithmeticOp::BINARY_BIT_AND) ||
        (!truth && arithmetic->op() == ArithmeticOp::BINARY_BIT_OR);
    if (decomposable && arithmetic->type() == Type::of<bool>()) {
        if (auto *slot =
                condition_implied_index_slot_last_before_counter(
                    arithmetic->operand(0u), truth, counter, terminator,
                    resolver, locations, depth + 1u)) {
            return slot;
        }
        return condition_implied_index_slot_last_before_counter(
            arithmetic->operand(1u), truth, counter, terminator,
            resolver, locations, depth + 1u);
    }

    auto establishes_equality =
        (truth && arithmetic->op() == ArithmeticOp::BINARY_EQUAL) ||
        (!truth && arithmetic->op() == ArithmeticOp::BINARY_NOT_EQUAL);
    if (!establishes_equality) { return nullptr; }

    const auto one_before_slot = [&](Value *value) noexcept {
        value = resolver.resolve(value, terminator);
        if (value == nullptr || !value->isa<ArithmeticInst>()) {
            return static_cast<AllocaInst *>(nullptr);
        }
        auto *add = static_cast<ArithmeticInst *>(value);
        if (add->op() != ArithmeticOp::BINARY_ADD ||
            add->operand_count() != 2u ||
            add->type() != counter->type()) {
            return static_cast<AllocaInst *>(nullptr);
        }
        for (auto slot_operand = 0u; slot_operand < 2u; ++slot_operand) {
            auto one = decode_unsigned_constant(
                resolver.resolve(add->operand(1u - slot_operand), add));
            auto *slot = current_scalar_slot(
                add->operand(slot_operand), terminator,
                resolver, locations);
            if (one && *one == 1u && slot != nullptr &&
                slot != counter && slot->type() == counter->type() &&
                slot->type()->is_uint()) {
                return slot;
            }
        }
        return static_cast<AllocaInst *>(nullptr);
    };

    auto *lhs = arithmetic->operand(0u);
    auto *rhs = arithmetic->operand(1u);
    if (is_current_slot_snapshot(
            rhs, counter, terminator, resolver, locations)) {
        return one_before_slot(lhs);
    }
    if (is_current_slot_snapshot(
            lhs, counter, terminator, resolver, locations)) {
        return one_before_slot(rhs);
    }
    return nullptr;
}

[[nodiscard]] bool index_is_initialized(
    Value *index, Instruction *read, AllocaInst *array,
    AllocaInst *counter, bool prefix_defined,
    size_t counter_lower_bound,
    const CoroGuardedScalarRelationDomain &relations,
    const FactState &facts, const FactLayout &layout,
    const CoroFrameAtomDomain &domain,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (depth >= 8u) { return false; }
    if (static_array_element_is_defined(
            index, read, array, facts, layout, domain, resolver)) {
        return true;
    }
    auto *relation_slot = current_scalar_slot(
        index, read, resolver, locations);
    index = resolver.resolve(index, read);
    if (!prefix_defined || index == nullptr) {
        return false;
    }
    // Prefix(A,C) states that every i<C is initialized. The numeric abstract
    // component supplies a Must lower bound L<=C, hence any constant i<L is
    // initialized on every represented execution. This is independent of
    // the symbolic ticket domain and remains valid across CFG joins because
    // their lower-bound meet is min.
    if (auto constant = decode_unsigned_constant(index);
        constant && *constant < counter_lower_bound) {
        return true;
    }
    if (relations.knows_initialized(relation_slot)) {
        return true;
    }
    // Tail_N(A,C) means C<N implies A[C] is initialized. The load itself is
    // defined only when its index is in bounds, so an exact current-counter
    // identity discharges the implication's antecedent without assuming a
    // numeric counter upper bound.
    if (relations.knows_tail() &&
        is_current_slot_snapshot(
            index, counter, read, resolver, locations)) {
        return true;
    }
    if (!index->isa<ArithmeticInst>()) { return false; }
    auto *select = static_cast<ArithmeticInst *>(index);
    if (select->op() != ArithmeticOp::SELECT ||
        select->operand_count() != 3u ||
        select->operand(2u)->type() != Type::of<bool>()) {
        return false;
    }
    auto *condition = select->operand(2u);
    const auto arm_is_safe = [&](Value *arm, bool truth) noexcept {
        return static_array_element_is_defined(
                   arm, read, array, facts, layout, domain, resolver) ||
               condition_implies_less_than_counter(
                   condition, truth, arm, counter, read,
                   resolver, locations) ||
               index_is_initialized(
                   arm, read, array, counter, prefix_defined,
                   counter_lower_bound,
                   relations,
                   facts, layout, domain, resolver, locations,
                   depth + 1u);
    };
    // XIR SELECT is select(false_value, true_value, condition).
    return arm_is_safe(select->operand(0u), false) &&
           arm_is_safe(select->operand(1u), true);
}

[[nodiscard]] bool match_counter_increment(
    Value *value, AllocaInst *counter, StoreInst *store,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    value = resolver.resolve(value, store);
    if (value == nullptr || !value->isa<ArithmeticInst>()) {
        return false;
    }
    auto *add = static_cast<ArithmeticInst *>(value);
    if (add->op() != ArithmeticOp::BINARY_ADD ||
        add->operand_count() != 2u) {
        return false;
    }
    for (auto counter_operand = 0u;
         counter_operand < 2u; ++counter_operand) {
        auto one = decode_unsigned_constant(
            resolver.resolve(add->operand(1u - counter_operand), add));
        if (one && *one == 1u &&
            is_current_slot_snapshot(
                add->operand(counter_operand), counter, store,
                resolver, locations)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool match_counter_decrement(
    Value *value, AllocaInst *counter, StoreInst *store,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    value = resolver.resolve(value, store);
    if (value == nullptr || !value->isa<ArithmeticInst>()) {
        return false;
    }
    auto *subtract = static_cast<ArithmeticInst *>(value);
    if (subtract->op() != ArithmeticOp::BINARY_SUB ||
        subtract->operand_count() != 2u) {
        return false;
    }
    auto one = decode_unsigned_constant(
        resolver.resolve(subtract->operand(1u), subtract));
    return one && *one == 1u &&
           is_current_slot_snapshot(
               subtract->operand(0u), counter, store,
               resolver, locations);
}

enum class ScalarZeroClass : uint8_t {
    unreachable,
    zero,
    nonzero,
    unknown
};

[[nodiscard]] ScalarZeroClass merge_scalar_zero_class(
    ScalarZeroClass lhs, ScalarZeroClass rhs) noexcept {
    if (lhs == ScalarZeroClass::unreachable) { return rhs; }
    if (rhs == ScalarZeroClass::unreachable) { return lhs; }
    return lhs == rhs ? lhs : ScalarZeroClass::unknown;
}

[[nodiscard]] ScalarZeroClass transfer_scalar_zero_class(
    BasicBlock *block, AllocaInst *slot,
    ScalarZeroClass state, Instruction *stop_before,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    if (state == ScalarZeroClass::unreachable) {
        return ScalarZeroClass::unreachable;
    }
    for (auto *instruction : block->instructions()) {
        if (instruction == stop_before) { break; }
        if (!instruction->isa<StoreInst>()) { continue; }
        auto *store = static_cast<StoreInst *>(instruction);
        if (store->variable() != slot) { continue; }
        auto constant = decode_unsigned_constant(
            resolver.resolve(store->value(), store));
        if (constant) {
            state = *constant == 0u ?
                        ScalarZeroClass::zero :
                        ScalarZeroClass::nonzero;
        } else if (!is_current_slot_snapshot(
                       store->value(), slot, store, resolver,
                       locations)) {
            state = ScalarZeroClass::unknown;
        }
    }
    return state;
}

[[nodiscard]] ScalarZeroClass scalar_zero_class_at_instruction(
    AllocaInst *slot, BasicBlock *target,
    Instruction *instruction, const CoroSemanticGraph &graph,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    if (slot == nullptr || target == nullptr || instruction == nullptr ||
        !resolver.scalar_slot_has_only_direct_accesses(slot)) {
        return ScalarZeroClass::unknown;
    }
    auto target_id = graph.block_id(target);
    if (target_id >= graph.block_count()) {
        return ScalarZeroClass::unknown;
    }

    // Forward constant-class analysis over the coroutine semantic CFG. The
    // quotient lattice is
    //
    //        unknown
    //        /     \
    //      zero  nonzero
    //        \     /
    //       unreachable
    //
    // and join is concrete-set union. Stores of integer constants map every
    // reachable input to the corresponding singleton class; unsupported
    // stores map it to unknown. The finite monotone system therefore
    // converges and claims a class only when every executable predecessor
    // agrees. This is an ordinary IR fact, not an application lifetime
    // contract.
    luisa::vector<ScalarZeroClass> in_states(
        graph.block_count(), ScalarZeroClass::unreachable);
    luisa::vector<ScalarZeroClass> out_states(
        graph.block_count(), ScalarZeroClass::unreachable);
    for (;;) {
        auto changed = false;
        for (size_t block_id = 0u;
             block_id < graph.block_count(); ++block_id) {
            auto next_in = block_id == 0u ?
                               ScalarZeroClass::unknown :
                               ScalarZeroClass::unreachable;
            for (auto predecessor : graph.predecessors(block_id)) {
                next_in = merge_scalar_zero_class(
                    next_in, out_states[predecessor]);
            }
            auto next_out = transfer_scalar_zero_class(
                graph.block(block_id), slot, next_in, nullptr,
                resolver, locations);
            if (in_states[block_id] != next_in ||
                out_states[block_id] != next_out) {
                in_states[block_id] = next_in;
                out_states[block_id] = next_out;
                changed = true;
            }
        }
        if (!changed) { break; }
    }
    return transfer_scalar_zero_class(
        target, slot, in_states[target_id], instruction,
        resolver, locations);
}

[[nodiscard]] bool scalar_is_zero_at_instruction(
    AllocaInst *slot, BasicBlock *target,
    Instruction *instruction, const CoroSemanticGraph &graph,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    return scalar_zero_class_at_instruction(
               slot, target, instruction, graph,
               resolver, locations) == ScalarZeroClass::zero;
}

// Sparse forward known-zero analysis for the unsigned scalar projections
// observed by masked control predicates. A state denotes a set of concrete
// stores. Fact Z(S,M) means every represented value of S has every bit in M
// clear. Join is intersection of facts (union of concrete states), while an
// absent block state denotes the empty concrete set and is the join identity.
//
// The expression transfer uses only implications valid in Boolean algebra:
//
//   Z(x,M) || Z(y,M) => Z(x & y,M)
//   Z(x,M) && Z(y,M) => Z(x | y,M), Z(x ^ y,M)
//   Z(x,M) && Z(y,M) => Z(select(x,y,c),M)
//
// Unsupported expressions yield no fact. The tracked product is the least
// backwards closure of the observed (slot, mask) roots over these supported
// expressions, rather than all scalar locals in the function. The finite
// Boolean product and monotone transfers guarantee termination. No source or
// application lifetime annotation participates in the proof.
class MaskedScalarKnownZeroAnalysis {
private:
    struct Projection {
        AllocaInst *scalar;
        uint64_t mask;
    };
    using State = luisa::vector<uint8_t>;

    const CoroSemanticGraph &_graph;
    const ScalarCopyResolver &_resolver;
    const PrefixInstructionLocationMap &_locations;
    luisa::vector<Projection> _projections;
    luisa::unordered_map<AllocaInst *, luisa::vector<size_t>>
        _projections_by_scalar;
    luisa::vector<luisa::optional<State>> _in_states;
    luisa::vector<luisa::optional<State>> _out_states;
    luisa::optional<State> _query_state;

private:
    [[nodiscard]] bool _add_projection(
        AllocaInst *scalar, uint64_t mask) noexcept {
        if (scalar == nullptr || mask == 0u ||
            scalar->type() == nullptr || !scalar->type()->is_uint() ||
            !_resolver.scalar_slot_has_only_direct_accesses(scalar)) {
            return false;
        }
        auto &indices = _projections_by_scalar[scalar];
        if (std::any_of(
                indices.begin(), indices.end(),
                [&](size_t index) noexcept {
                    return _projections[index].mask == mask;
                })) {
            return false;
        }
        indices.emplace_back(_projections.size());
        _projections.emplace_back(Projection{scalar, mask});
        return true;
    }

    void _collect_dependencies(
        Value *value, Instruction *use, uint64_t mask,
        size_t depth = 0u) noexcept {
        if (value == nullptr || use == nullptr || depth >= 32u) { return; }
        if (auto *source = current_scalar_slot(
                value, use, _resolver, _locations);
            source != nullptr && source->type() != nullptr &&
            source->type()->is_uint()) {
            static_cast<void>(_add_projection(source, mask));
            return;
        }
        value = _resolver.resolve(value, use);
        if (value == nullptr || decode_unsigned_constant(value) ||
            !value->isa<ArithmeticInst>()) {
            return;
        }
        auto *arithmetic = static_cast<ArithmeticInst *>(value);
        switch (arithmetic->op()) {
            case ArithmeticOp::BINARY_BIT_AND:
            case ArithmeticOp::BINARY_BIT_OR:
            case ArithmeticOp::BINARY_BIT_XOR:
                if (arithmetic->operand_count() == 2u) {
                    _collect_dependencies(
                        arithmetic->operand(0u), use, mask, depth + 1u);
                    _collect_dependencies(
                        arithmetic->operand(1u), use, mask, depth + 1u);
                }
                break;
            case ArithmeticOp::SELECT:
                if (arithmetic->operand_count() == 3u) {
                    _collect_dependencies(
                        arithmetic->operand(0u), use, mask, depth + 1u);
                    _collect_dependencies(
                        arithmetic->operand(1u), use, mask, depth + 1u);
                }
                break;
            default: break;
        }
    }

    void _close_dependencies() noexcept {
        for (size_t cursor = 0u; cursor < _projections.size(); ++cursor) {
            auto projection = _projections[cursor];
            for (auto *use : projection.scalar->use_list()) {
                auto *user = use == nullptr ? nullptr : use->user();
                if (user == nullptr || !user->isa<StoreInst>()) { continue; }
                auto *store = static_cast<StoreInst *>(user);
                if (store->variable() != projection.scalar ||
                    store->parent_block() == nullptr ||
                    !_graph.contains(store->parent_block())) {
                    continue;
                }
                _collect_dependencies(
                    store->value(), store, projection.mask);
            }
        }
    }

    [[nodiscard]] bool _state_proves_zero(
        const State &state, AllocaInst *scalar,
        uint64_t mask) const noexcept {
        if (scalar == nullptr || mask == 0u) { return false; }
        auto iter = _projections_by_scalar.find(scalar);
        if (iter == _projections_by_scalar.end()) { return false; }
        return std::any_of(
            iter->second.begin(), iter->second.end(),
            [&](size_t index) noexcept {
                auto projection = _projections[index];
                return state[index] != 0u &&
                       (mask & ~projection.mask) == 0u;
            });
    }

    [[nodiscard]] bool _value_is_zero(
        Value *value, uint64_t mask, const State &state,
        Instruction *use, size_t depth = 0u) const noexcept {
        if (value == nullptr || use == nullptr || depth >= 32u) {
            return false;
        }
        if (auto *source = current_scalar_slot(
                value, use, _resolver, _locations);
            _state_proves_zero(state, source, mask)) {
            return true;
        }
        value = _resolver.resolve(value, use);
        if (auto constant = decode_unsigned_constant(value)) {
            return (*constant & mask) == 0u;
        }
        if (value == nullptr || !value->isa<ArithmeticInst>()) {
            return false;
        }
        auto *arithmetic = static_cast<ArithmeticInst *>(value);
        const auto operand_is_zero = [&](size_t operand) noexcept {
            return _value_is_zero(
                arithmetic->operand(operand), mask, state,
                use, depth + 1u);
        };
        switch (arithmetic->op()) {
            case ArithmeticOp::BINARY_BIT_AND:
                return arithmetic->operand_count() == 2u &&
                       (operand_is_zero(0u) || operand_is_zero(1u));
            case ArithmeticOp::BINARY_BIT_OR:
            case ArithmeticOp::BINARY_BIT_XOR:
                return arithmetic->operand_count() == 2u &&
                       operand_is_zero(0u) && operand_is_zero(1u);
            case ArithmeticOp::SELECT:
                return arithmetic->operand_count() == 3u &&
                       operand_is_zero(0u) && operand_is_zero(1u);
            default: return false;
        }
    }

    void _assume_zero(
        State &state, AllocaInst *scalar,
        uint64_t zero_mask) const noexcept {
        auto iter = _projections_by_scalar.find(scalar);
        if (iter == _projections_by_scalar.end()) { return; }
        for (auto index : iter->second) {
            if ((_projections[index].mask & ~zero_mask) == 0u) {
                state[index] = 1u;
            }
        }
    }

    [[nodiscard]] bool _transfer_instruction(
        Instruction *instruction, State &state) const noexcept {
        if (instruction->isa<AssumeInst>()) {
            auto *assume = static_cast<AssumeInst *>(instruction);
            if (auto test = match_masked_scalar_test(
                    assume->condition(), assume,
                    _resolver, _locations)) {
                if (test->nonzero_when_condition_true) {
                    if (_state_proves_zero(
                            state, test->scalar, test->mask)) {
                        return false;
                    }
                } else {
                    _assume_zero(state, test->scalar, test->mask);
                }
            }
            return true;
        }
        if (!instruction->isa<StoreInst>()) { return true; }
        auto *store = static_cast<StoreInst *>(instruction);
        auto *pointer = store->variable();
        auto *destination =
            pointer != nullptr && pointer->isa<AllocaInst>() ?
                static_cast<AllocaInst *>(pointer) :
                nullptr;
        auto iter = _projections_by_scalar.find(destination);
        if (iter == _projections_by_scalar.end()) { return true; }
        luisa::vector<uint8_t> next;
        next.reserve(iter->second.size());
        for (auto index : iter->second) {
            next.emplace_back(static_cast<uint8_t>(
                _value_is_zero(
                    store->value(), _projections[index].mask,
                    state, store)));
        }
        for (size_t i = 0u; i < iter->second.size(); ++i) {
            state[iter->second[i]] = next[i];
        }
        return true;
    }

    [[nodiscard]] bool _refine_edge(
        BasicBlock *predecessor, BasicBlock *successor,
        State &state) const noexcept {
        auto *terminator = predecessor == nullptr ?
                               nullptr :
                               predecessor->terminator();
        if (terminator == nullptr ||
            !terminator->isa<ConditionalBranchInst>()) {
            return true;
        }
        auto *branch = static_cast<ConditionalBranchInst *>(terminator);
        bool truth;
        if (branch->true_block() == successor &&
            branch->false_block() != successor) {
            truth = true;
        } else if (branch->false_block() == successor &&
                   branch->true_block() != successor) {
            truth = false;
        } else {
            return true;
        }
        auto test = match_masked_scalar_test(
            branch->condition(), branch, _resolver, _locations);
        if (!test) { return true; }
        auto selected_nonzero =
            truth == test->nonzero_when_condition_true;
        if (selected_nonzero) {
            return !_state_proves_zero(
                state, test->scalar, test->mask);
        }
        _assume_zero(state, test->scalar, test->mask);
        return true;
    }

    [[nodiscard]] static bool _merge_into(
        luisa::optional<State> &target,
        const State &incoming) noexcept {
        if (!target) {
            target = incoming;
            return true;
        }
        auto changed = false;
        for (size_t i = 0u; i < target->size(); ++i) {
            auto next = static_cast<uint8_t>(
                (*target)[i] & incoming[i]);
            changed |= next != (*target)[i];
            (*target)[i] = next;
        }
        return changed;
    }

    [[nodiscard]] luisa::optional<State> _state_before(
        Instruction *instruction) const noexcept {
        if (instruction == nullptr ||
            instruction->parent_block() == nullptr) {
            return luisa::nullopt;
        }
        auto location = _locations.find(instruction);
        if (location == _locations.end() ||
            location->second.block_id >= _in_states.size() ||
            !_in_states[location->second.block_id]) {
            return luisa::nullopt;
        }
        auto state = *_in_states[location->second.block_id];
        for (auto *current :
             instruction->parent_block()->instructions()) {
            if (current == instruction) { return state; }
            if (!_transfer_instruction(current, state)) {
                return luisa::nullopt;
            }
        }
        return luisa::nullopt;
    }

    void _solve(BasicBlock *target, Instruction *instruction) noexcept {
        _in_states.resize(_graph.block_count());
        _out_states.resize(_graph.block_count());
        if (_graph.block_count() == 0u) { return; }
        _in_states[0u] = State(_projections.size(), uint8_t{0u});
        luisa::vector<size_t> worklist{0u};
        luisa::vector<uint8_t> queued(
            _graph.block_count(), uint8_t{0u});
        queued[0u] = 1u;
        for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
            auto block_id = worklist[cursor];
            queued[block_id] = 0u;
            if (!_in_states[block_id]) { continue; }
            auto state = *_in_states[block_id];
            auto reachable = true;
            for (auto *current :
                 _graph.block(block_id)->instructions()) {
                if (!_transfer_instruction(current, state)) {
                    reachable = false;
                    break;
                }
            }
            if (!reachable) { continue; }
            if (_out_states[block_id] &&
                *_out_states[block_id] == state) {
                continue;
            }
            _out_states[block_id] = state;
            for (auto successor : _graph.successors(block_id)) {
                auto edge_state = state;
                if (!_refine_edge(
                        _graph.block(block_id),
                        _graph.block(successor), edge_state)) {
                    continue;
                }
                if (_merge_into(_in_states[successor], edge_state) &&
                    queued[successor] == 0u) {
                    queued[successor] = 1u;
                    worklist.emplace_back(successor);
                }
            }
        }

        if (instruction != nullptr &&
            instruction->parent_block() == target) {
            _query_state = _state_before(instruction);
        }
    }

public:
    MaskedScalarKnownZeroAnalysis(
        const CoroSemanticGraph &graph,
        const ScalarCopyResolver &resolver,
        const PrefixInstructionLocationMap &locations,
        luisa::span<const CoroMaskedScalarWitness> roots,
        BasicBlock *target, Instruction *instruction) noexcept
        : _graph{graph}, _resolver{resolver}, _locations{locations} {
        for (auto root : roots) {
            static_cast<void>(_add_projection(root.scalar, root.mask));
        }
        _close_dependencies();
        _solve(target, instruction);
    }

    [[nodiscard]] bool proves_zero(
        AllocaInst *scalar, uint64_t mask) const noexcept {
        return _query_state &&
               _state_proves_zero(*_query_state, scalar, mask);
    }

    [[nodiscard]] bool proves_zero_at(
        AllocaInst *scalar, uint64_t mask,
        Instruction *instruction) const noexcept {
        auto state = _state_before(instruction);
        return state && _state_proves_zero(*state, scalar, mask);
    }

    [[nodiscard]] bool proves_value_zero_at(
        Value *value, uint64_t mask,
        Instruction *instruction) const noexcept {
        auto state = _state_before(instruction);
        return state && _value_is_zero(
                            value, mask, *state, instruction);
    }
};

struct PrefixState {
    FactState facts;
    bool prefix_defined{false};
    size_t counter_lower_bound{0u};
    luisa::optional<size_t> counter_upper_bound;
    CoroGuardedScalarRelationDomain relations;

    PrefixState(FactState initial_facts,
                CoroBooleanSetManager &boolean_sets,
                luisa::span<const CoroMaskedScalarWitness>
                    masked_scalar_witnesses,
                luisa::span<const CoroCounterAvailabilityWitness>
                    counter_availability_witnesses,
                bool initial_prefix,
                luisa::optional<size_t> initial_upper_bound) noexcept
        : facts{std::move(initial_facts)},
          prefix_defined{initial_prefix},
          counter_upper_bound{initial_upper_bound},
          relations{boolean_sets, masked_scalar_witnesses,
                    counter_availability_witnesses} {}

    [[nodiscard]] bool operator==(
        const PrefixState &) const noexcept = default;
};

[[nodiscard]] bool merge_state(
    PrefixState &target, const PrefixState &incoming) noexcept {
    auto changed = false;
    changed |= target.relations.merge(incoming.relations);
    for (size_t i = 0u; i < target.facts.size(); ++i) {
        auto next = static_cast<uint8_t>(
            target.facts[i] & incoming.facts[i]);
        changed |= next != target.facts[i];
        target.facts[i] = next;
    }
    auto next_prefix =
        target.prefix_defined && incoming.prefix_defined;
    changed |= next_prefix != target.prefix_defined;
    target.prefix_defined = next_prefix;
    auto next_lower = std::min(
        target.counter_lower_bound,
        incoming.counter_lower_bound);
    changed |= next_lower != target.counter_lower_bound;
    target.counter_lower_bound = next_lower;
    luisa::optional<size_t> next_upper;
    if (target.counter_upper_bound && incoming.counter_upper_bound) {
        next_upper = std::max(
            *target.counter_upper_bound,
            *incoming.counter_upper_bound);
    }
    changed |= next_upper != target.counter_upper_bound;
    target.counter_upper_bound = next_upper;
    return changed;
}

[[nodiscard]] bool full_element_store_at_current_counter(
    StoreInst *store, AllocaInst *array, AllocaInst *counter,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    auto *gep = top_array_gep(store->variable(), array);
    return gep != nullptr && store->variable() == gep &&
           gep->base() == array &&
           gep->index_count() == 1u &&
           gep->type() == array->type()->element() &&
           is_current_slot_snapshot(
               gep->index(0u), counter, store, resolver, locations);
}

[[nodiscard]] luisa::vector<CoroCounterAvailabilityWitness>
collect_counter_availability_witnesses(
    AllocaInst *array, AllocaInst *counter,
    const ArrayUseRegion &region,
    const CoroSemanticGraph &graph,
    luisa::span<const uint8_t> active_blocks,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    luisa::vector<CoroCounterAvailabilityWitness> result;
    const auto insert = [&](AllocaInst *scalar) noexcept {
        if (scalar == nullptr || scalar == counter ||
            !resolver.scalar_slot_has_only_direct_accesses(scalar) ||
            std::any_of(
                result.begin(), result.end(),
                [scalar](auto witness) noexcept {
                    return witness.scalar == scalar;
                })) {
            return;
        }
        result.emplace_back(CoroCounterAvailabilityWitness{scalar});
    };

    // Discover only scalar resources whose nonzero edge directly admits a
    // full A[C] definition. This is a sparse syntactic root set for the
    // semantic invariant C>0 || R>0; it neither recognizes renderer names nor
    // assumes that every zero-tested scalar is a capacity. Once selected,
    // ordinary abstract transfer proves (or conservatively loses) the
    // invariant across all subsequent stores and joins.
    for (auto *instruction : region.users) {
        if (instruction == nullptr || !instruction->isa<StoreInst>()) {
            continue;
        }
        auto *store = static_cast<StoreInst *>(instruction);
        if (!full_element_store_at_current_counter(
                store, array, counter, resolver, locations)) {
            continue;
        }
        auto block_id = graph.block_id(store->parent_block());
        if (block_id >= graph.block_count() ||
            block_id >= active_blocks.size() ||
            active_blocks[block_id] == 0u) {
            continue;
        }
        for (auto predecessor_id : graph.predecessors(block_id)) {
            auto *predecessor = graph.block(predecessor_id);
            auto *terminator = predecessor == nullptr ?
                                   nullptr :
                                   predecessor->terminator();
            if (terminator == nullptr ||
                !terminator->isa<ConditionalBranchInst>()) {
                continue;
            }
            auto *branch = static_cast<ConditionalBranchInst *>(terminator);
            bool selected_truth;
            if (branch->true_block() == store->parent_block() &&
                branch->false_block() != store->parent_block()) {
                selected_truth = true;
            } else if (branch->false_block() == store->parent_block() &&
                       branch->true_block() != store->parent_block()) {
                selected_truth = false;
            } else {
                continue;
            }
            auto test = match_scalar_zero_test(
                branch->condition(), branch, resolver, locations);
            if (test &&
                selected_truth != test->zero_when_condition_true) {
                insert(test->scalar);
            }
        }
    }
    return result;
}

struct CandidateContext {
    AllocaInst *array;
    AllocaInst *counter;
    size_t dimension;
    const ArrayUseRegion &region;
    const FactLayout &layout;
    const CoroFrameAtomDomain &domain;
    const CoroSemanticGraph &graph;
    const ScalarCopyResolver &resolver;
    const PrefixInstructionLocationMap &locations;
    const luisa::unordered_set<Value *> &boolean_guards;
    const CoroScalarRelationLiveness &relation_liveness;
    const CoroBooleanPredicateLiveness &boolean_liveness;
    const MaskedScalarKnownZeroAnalysis &known_zero;
    luisa::span<const CoroMaskedScalarWitness>
        masked_scalar_witnesses;
    luisa::span<const CoroCounterAvailabilityWitness>
        counter_availability_witnesses;
};

[[nodiscard]] CoroMaskedScalarProjection
evaluate_masked_scalar_expression(
    Value *value, uint64_t mask, Instruction *use,
    const CoroGuardedScalarRelationDomain &relations,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (value == nullptr || use == nullptr || depth >= 32u) {
        return relations.masked_scalar_unknown_projection();
    }
    if (auto *source = current_scalar_slot(
            value, use, resolver, locations)) {
        return relations.masked_scalar_load_projection(source, mask);
    }
    value = resolver.resolve(value, use);
    if (auto constant = decode_unsigned_constant(value)) {
        return relations.masked_scalar_constant_projection(
            *constant, mask);
    }
    if (value == nullptr || !value->isa<ArithmeticInst>()) {
        return relations.masked_scalar_unknown_projection();
    }
    auto *arithmetic = static_cast<ArithmeticInst *>(value);
    const auto evaluate_operand = [&](size_t operand) noexcept {
        return evaluate_masked_scalar_expression(
            arithmetic->operand(operand), mask, use, relations,
            resolver, locations, depth + 1u);
    };
    switch (arithmetic->op()) {
        case ArithmeticOp::BINARY_BIT_AND:
            if (arithmetic->operand_count() == 2u) {
                return relations.masked_scalar_projection_intersection(
                    evaluate_operand(0u), evaluate_operand(1u));
            }
            break;
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
            if (arithmetic->operand_count() == 2u) {
                return relations.masked_scalar_projection_union(
                    evaluate_operand(0u), evaluate_operand(1u));
            }
            break;
        case ArithmeticOp::SELECT:
            if (arithmetic->operand_count() == 3u) {
                // Without coupling the selector to the Boolean set manager,
                // either data arm can be chosen. Their set union is the
                // least sound result in the current product domain.
                return relations.masked_scalar_projection_union(
                    evaluate_operand(0u), evaluate_operand(1u));
            }
            break;
        default: break;
    }
    return relations.masked_scalar_unknown_projection();
}

void transfer_masked_scalar_store(
    StoreInst *store, CoroGuardedScalarRelationDomain &relations,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    auto *pointer = store == nullptr ? nullptr : store->variable();
    auto *destination =
        pointer != nullptr && pointer->isa<AllocaInst>() ?
            static_cast<AllocaInst *>(pointer) :
            nullptr;
    if (!relations.tracks_masked_scalar(destination)) { return; }

    // Every right-hand-side projection is evaluated in the same pre-store
    // abstract state. This mirrors simultaneous assignment and prevents a
    // self-referential expression from observing partially updated masks.
    auto masks = relations.masked_scalar_masks(destination);
    luisa::vector<CoroMaskedScalarProjection> projections;
    projections.reserve(masks.size());
    for (auto mask : masks) {
        projections.emplace_back(evaluate_masked_scalar_expression(
            store->value(), mask, store, relations,
            resolver, locations));
    }
    for (size_t i = 0u; i < masks.size(); ++i) {
        relations.assign_masked_scalar_projection(
            destination, masks[i], projections[i]);
    }
}

[[nodiscard]] CoroScalarZeroProjection
evaluate_scalar_zero_expression(
    Value *value, Instruction *use,
    const CoroGuardedScalarRelationDomain &relations,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (value == nullptr || use == nullptr || depth >= 32u) {
        return relations.scalar_zero_unknown_projection();
    }
    if (auto *source = current_scalar_slot(
            value, use, resolver, locations)) {
        return relations.scalar_zero_load_projection(source);
    }
    value = resolver.resolve(value, use);
    if (auto constant = decode_unsigned_constant(value)) {
        return relations.scalar_zero_constant_projection(*constant);
    }
    if (value == nullptr || !value->isa<ArithmeticInst>()) {
        return relations.scalar_zero_unknown_projection();
    }
    auto *arithmetic = static_cast<ArithmeticInst *>(value);
    const auto evaluate_operand = [&](size_t operand) noexcept {
        return evaluate_scalar_zero_expression(
            arithmetic->operand(operand), use, relations,
            resolver, locations, depth + 1u);
    };
    switch (arithmetic->op()) {
        case ArithmeticOp::BINARY_BIT_OR:
            if (arithmetic->operand_count() == 2u) {
                // x|y==0 iff x==0 and y==0. Intersecting the two
                // over-approximations is therefore still an
                // over-approximation of the concrete zero states.
                return relations.scalar_zero_projection_intersection(
                    evaluate_operand(0u), evaluate_operand(1u));
            }
            break;
        case ArithmeticOp::SELECT:
            if (arithmetic->operand_count() == 3u) {
                // Without selector correlation either data arm may be
                // chosen; union is the least sound result in this product.
                return relations.scalar_zero_projection_union(
                    evaluate_operand(0u), evaluate_operand(1u));
            }
            break;
        default: break;
    }
    return relations.scalar_zero_unknown_projection();
}

void transfer_counter_availability_store(
    StoreInst *store, CoroGuardedScalarRelationDomain &relations,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations) noexcept {
    auto *pointer = store == nullptr ? nullptr : store->variable();
    auto *destination =
        pointer != nullptr && pointer->isa<AllocaInst>() ?
            static_cast<AllocaInst *>(pointer) :
            nullptr;
    if (!relations.tracks_counter_availability(destination)) { return; }
    auto projection = evaluate_scalar_zero_expression(
        store->value(), store, relations, resolver, locations);
    relations.assign_scalar_zero_projection(
        destination, projection);
}

[[nodiscard]] bool refine_counter_bound_on_edge(
    PrefixState &state, BasicBlock *predecessor,
    BasicBlock *successor,
    const CandidateContext &context) noexcept {
    auto *terminator = predecessor == nullptr ?
                           nullptr :
                           predecessor->terminator();
    if (terminator == nullptr ||
        !terminator->isa<ConditionalBranchInst>()) {
        return true;
    }
    auto *branch = static_cast<ConditionalBranchInst *>(terminator);
    bool truth;
    if (branch->true_block() == successor &&
        branch->false_block() != successor) {
        truth = true;
    } else if (branch->false_block() == successor &&
               branch->true_block() != successor) {
        truth = false;
    } else {
        return true;
    }

    auto condition_source = analyze_boolean_expression(
        branch->condition(), branch, context.resolver,
        context.locations);
    if (condition_source.constant &&
        *condition_source.constant != truth) {
        return false;
    }
    if (condition_source.predicate != nullptr &&
        context.boolean_guards.contains(condition_source.predicate)) {
        auto selected_slot_value =
            truth == condition_source.true_when_predicate_is;
        if (!state.relations.refine_boolean(
                condition_source.predicate, selected_slot_value)) {
            return false;
        }
    }
    auto bound = condition_implied_counter_upper_bound(
        branch->condition(), truth, context.counter, branch,
        context.resolver, context.locations);
    if (bound) {
        state.counter_upper_bound = state.counter_upper_bound ?
                                        std::min(*state.counter_upper_bound,
                                                 *bound) :
                                        *bound;
    }
    auto lower_bound = condition_implied_counter_lower_bound(
        branch->condition(), truth, context.counter, branch,
        context.resolver, context.locations);
    if (lower_bound) {
        state.counter_lower_bound = std::max(
            state.counter_lower_bound, *lower_bound);
        if (*lower_bound != 0u) {
            state.relations.add_counter_positive();
        }
    }
    if (state.counter_upper_bound &&
        state.counter_lower_bound > *state.counter_upper_bound) {
        return false;
    }

    if (auto zero_test = match_scalar_zero_test(
            branch->condition(), branch, context.resolver,
            context.locations);
        zero_test && state.relations.tracks_counter_availability(
                         zero_test->scalar)) {
        auto selected_zero =
            truth == zero_test->zero_when_condition_true;
        if (selected_zero) {
            if (!state.relations.refine_scalar_zero(
                    zero_test->scalar)) {
                return false;
            }
            if (state.relations.
                    scalar_zero_implies_counter_positive(
                        zero_test->scalar)) {
                state.counter_lower_bound = std::max(
                    state.counter_lower_bound, size_t{1u});
                state.relations.add_counter_positive();
            }
        } else {
            state.relations.assume_scalar_nonzero(
                zero_test->scalar);
        }
    }

    if (auto masked_test = match_masked_scalar_test(
            branch->condition(), branch, context.resolver,
            context.locations)) {
        auto selected_nonzero =
            truth == masked_test->nonzero_when_condition_true;
        if (selected_nonzero && context.known_zero.proves_zero_at(
                                    masked_test->scalar,
                                    masked_test->mask,
                                    branch)) {
            // MustKnownZero denotes every concrete state reaching this
            // instruction. Its intersection with S&M!=0 is empty, so the
            // selected semantic edge has no executions to propagate.
            return false;
        }
        if (!selected_nonzero) {
            state.relations.assume_masked_scalar_zero(
                masked_test->scalar, masked_test->mask);
        } else if (!state.relations.refine_masked_scalar_nonzero(
                       masked_test->scalar, masked_test->mask)) {
            return false;
        } else if (state.relations.
                       masked_nonzero_implies_counter_positive(
                           masked_test->scalar,
                           masked_test->mask)) {
            state.counter_lower_bound = std::max(
                state.counter_lower_bound, size_t{1u});
            state.relations.add_counter_positive();
        }
    }

    if (auto *slot = condition_implied_index_slot_less_than_counter(
            branch->condition(), truth, context.counter, branch,
            context.resolver, context.locations);
        slot != nullptr) {
        // Equality and strict inequality cannot both hold on a feasible
        // edge. The branch establishes the latter relation for the selected
        // edge; discard the weaker incoming representation of the former.
        state.relations.erase_index(slot);
        state.relations.add_less(slot);
    }
    if (auto *slot =
            condition_implied_index_slot_last_before_counter(
                branch->condition(), truth, context.counter, branch,
                context.resolver, context.locations);
        slot != nullptr &&
        (state.counter_lower_bound != 0u ||
         state.relations.knows_counter_positive())) {
        state.relations.add_last(slot);
    }
    return true;
}

[[nodiscard]] bool process_instruction(
    Instruction *instruction, PrefixState &state,
    const CandidateContext &context, bool &used_prefix_read,
    Instruction *&failing_read, bool validate_reads) noexcept {
    const auto finish = [&](bool result) noexcept {
        for (auto *slot :
             context.relation_liveness.dead_after(instruction)) {
            state.relations.erase_index(slot);
        }
        return result;
    };
    // An SSA instruction denotes a fresh dynamic value whenever its block is
    // re-entered. Facts keyed by that predicate therefore die at its
    // definition, just as facts keyed by a mutable scalar die at a store.
    if (instruction->type() == Type::of<bool>() &&
        context.boolean_guards.contains(instruction)) {
        state.relations.forget_boolean(instruction);
    }
    if (instruction->isa<AssumeInst>()) {
        auto *assume = static_cast<AssumeInst *>(instruction);
        auto source = analyze_boolean_expression(
            assume->condition(), assume,
            context.resolver, context.locations);
        if (source.predicate != nullptr &&
            context.boolean_guards.contains(source.predicate)) {
            static_cast<void>(state.relations.refine_boolean(
                source.predicate,
                source.true_when_predicate_is));
        }
        if (auto masked_test = match_masked_scalar_test(
                assume->condition(), assume,
                context.resolver, context.locations)) {
            if (masked_test->nonzero_when_condition_true) {
                if (context.known_zero.proves_zero_at(
                        masked_test->scalar,
                        masked_test->mask, assume)) {
                    state.relations.assume_masked_scalar_zero(
                        masked_test->scalar,
                        masked_test->mask);
                }
                if (state.relations.refine_masked_scalar_nonzero(
                        masked_test->scalar, masked_test->mask) &&
                    state.relations.
                        masked_nonzero_implies_counter_positive(
                            masked_test->scalar,
                            masked_test->mask)) {
                    state.counter_lower_bound = std::max(
                        state.counter_lower_bound, size_t{1u});
                    state.relations.add_counter_positive();
                }
            } else {
                state.relations.assume_masked_scalar_zero(
                    masked_test->scalar, masked_test->mask);
            }
        }
        if (auto zero_test = match_scalar_zero_test(
                assume->condition(), assume,
                context.resolver, context.locations);
            zero_test && state.relations.tracks_counter_availability(
                             zero_test->scalar)) {
            if (zero_test->zero_when_condition_true) {
                if (state.relations.refine_scalar_zero(
                        zero_test->scalar) &&
                    state.relations.
                        scalar_zero_implies_counter_positive(
                            zero_test->scalar)) {
                    state.counter_lower_bound = std::max(
                        state.counter_lower_bound, size_t{1u});
                    state.relations.add_counter_positive();
                }
            } else {
                state.relations.assume_scalar_nonzero(
                    zero_test->scalar);
            }
        }
    }
    if (instruction->isa<GEPInst>() &&
        context.region.pointers.contains(instruction)) {
        redefine_pointer(instruction, state.facts, context.layout);
    }

    if (instruction->isa<StoreInst>()) {
        auto *store = static_cast<StoreInst *>(instruction);
        auto *pointer = store->variable();
        auto *scalar_pointer =
            pointer != nullptr && pointer->isa<AllocaInst>() ?
                static_cast<AllocaInst *>(pointer) :
                nullptr;
        auto *source_slot = current_scalar_slot(
            store->value(), store, context.resolver,
            context.locations);
        const auto source_equals_counter =
            is_current_slot_snapshot(
                store->value(), context.counter, store,
                context.resolver, context.locations) ||
            state.relations.knows_equal(source_slot);
        const auto source_less_than_counter =
            state.relations.knows_less(source_slot);
        const auto source_is_last =
            state.relations.knows_last(source_slot);
        const auto source_constant = decode_unsigned_constant(
            context.resolver.resolve(store->value(), store));
        const auto preserves_scalar_slot =
            scalar_pointer != nullptr &&
            is_current_slot_snapshot(
                store->value(), scalar_pointer,
                store, context.resolver, context.locations);
        const auto counter_increment =
            pointer == context.counter &&
            match_counter_increment(
                store->value(), context.counter, store,
                context.resolver, context.locations);
        const auto counter_decrement =
            pointer == context.counter &&
            match_counter_decrement(
                store->value(), context.counter, store,
                context.resolver, context.locations);

        transfer_masked_scalar_store(
            store, state.relations,
            context.resolver, context.locations);
        transfer_counter_availability_store(
            store, state.relations,
            context.resolver, context.locations);

        auto boolean_source = BooleanExpressionSource{};
        luisa::optional<bool> copied_boolean_constant;
        auto tracks_boolean_destination =
            scalar_pointer != nullptr &&
            context.boolean_guards.contains(scalar_pointer);
        if (tracks_boolean_destination && !preserves_scalar_slot) {
            boolean_source = analyze_boolean_expression(
                store->value(), store, context.resolver,
                context.locations);
            copied_boolean_constant = boolean_source.constant;
        }

        // Scalar relations describe current memory contents. Assignment
        // kills facts about the destination, then generates exactly the
        // relations carried by the right-hand side. Counter updates are
        // handled below because a proven non-wrapping C := C + 1 maps
        // I == old(C) to I < new(C), while preserving every I < old(C).
        if (scalar_pointer != nullptr &&
            scalar_pointer != context.counter &&
            !preserves_scalar_slot) {
            if (scalar_pointer->type() == context.counter->type() &&
                scalar_pointer->type()->is_uint()) {
                state.relations.assign_index_copy(
                    scalar_pointer, source_slot,
                    is_current_slot_snapshot(
                        store->value(), context.counter, store,
                        context.resolver, context.locations));
                if (source_less_than_counter) {
                    state.relations.add_less(scalar_pointer);
                }
                if (source_equals_counter) {
                    state.relations.add_equal(scalar_pointer);
                }
                if (source_is_last) {
                    state.relations.add_last(scalar_pointer);
                }
                if (source_constant &&
                    (*source_constant < state.counter_lower_bound ||
                     (*source_constant == 0u &&
                      state.relations.knows_counter_positive()))) {
                    state.relations.add_less(scalar_pointer);
                }
            } else {
                state.relations.erase_index(scalar_pointer);
            }
        }
        if (tracks_boolean_destination && !preserves_scalar_slot) {
            state.relations.assign_boolean(
                scalar_pointer, boolean_source.predicate,
                boolean_source.true_when_predicate_is,
                copied_boolean_constant);
        }
        if (context.region.pointers.contains(pointer)) {
            define_pointer(
                pointer, state.facts, context.layout, context.domain);
            if (state.prefix_defined &&
                full_element_store_at_current_counter(
                    store, context.array, context.counter,
                    context.resolver, context.locations)) {
                // The abstract invariant is the bounded prefix
                //   forall i < min(C, N): initialized(A[i]).
                // On every defined execution, dereferencing A[C] proves
                // C < N. The store therefore establishes the one pending
                // element needed by C := C + 1 and also proves that update
                // cannot wrap. If an incoming numeric lower bound conflicts
                // with this memory-safety precondition, widening it to zero
                // conservatively represents the remaining defined paths.
                auto in_bounds_upper = context.dimension - 1u;
                state.counter_upper_bound =
                    state.counter_upper_bound ?
                        std::min(*state.counter_upper_bound,
                                 in_bounds_upper) :
                        luisa::optional<size_t>{in_bounds_upper};
                if (state.counter_lower_bound > in_bounds_upper) {
                    state.counter_lower_bound = 0u;
                }
                state.relations.add_tail();
            }
        }
        if (pointer == context.counter) {
            // Prefix-derived initialization is a physical memory fact and
            // survives changes to C. Materialize it before transforming or
            // killing the scalar relations that establish it.
            if (state.prefix_defined) {
                state.relations.materialize_initialized();
            }
            auto constant = decode_unsigned_constant(
                context.resolver.resolve(store->value(), store));
            auto self_assignment = is_current_slot_snapshot(
                store->value(), context.counter, store,
                context.resolver, context.locations);
            if (constant && *constant == 0u) {
                state.prefix_defined = true;
                state.relations.clear_tail();
                state.counter_lower_bound = 0u;
                state.counter_upper_bound = 0u;
                state.relations.clear_relations();
                state.relations.clear_counter_positive();
                state.relations.invalidate_counter_implications();
            } else if (self_assignment) {
                // Exact self-assignment preserves both the prefix and tail
                // facts for the same counter value.
            } else if (counter_increment &&
                       state.prefix_defined &&
                       state.relations.knows_tail()) {
                // The preceding full-element store proves C < dimension on
                // defined executions. Since dimension fits the unsigned
                // counter type, the increment cannot wrap. Thus I < old(C)
                // implies I < new(C), and every saved allocation ticket
                // I == old(C) becomes I < new(C).
                state.relations.advance_counter();
                ++state.counter_lower_bound;
                state.counter_upper_bound = context.dimension;
                state.relations.clear_tail();
            } else if (counter_decrement &&
                       state.prefix_defined &&
                       (state.counter_lower_bound != 0u ||
                        state.relations.knows_counter_positive())) {
                // L <= C with L >= 1 proves unsigned C - 1 cannot wrap.
                // Prefix_N(A, old(C)) implies both Prefix_N(A, new(C)) and
                // Tail_N(A, new(C)). A saved last-element relation
                // I+1=old(C) becomes I=new(C); a generic I<old(C) yields only
                // I<=new(C) and is therefore discarded by retreat_counter.
                --state.counter_lower_bound;
                if (state.counter_upper_bound) {
                    LUISA_DEBUG_ASSERT(
                        *state.counter_upper_bound != 0u,
                        "Contradictory counted-prefix interval.");
                    --*state.counter_upper_bound;
                }
                state.relations.retreat_counter();
                state.relations.add_tail();
                if (state.counter_lower_bound != 0u) {
                    state.relations.add_counter_positive();
                } else {
                    state.relations.clear_counter_positive();
                }
            } else {
                state.prefix_defined = false;
                state.relations.clear_tail();
                state.counter_lower_bound = 0u;
                state.counter_upper_bound = luisa::nullopt;
                state.relations.clear_relations();
                state.relations.clear_counter_positive();
                state.relations.invalidate_counter_implications();
            }
        }
    }

    if (instruction->isa<LoadInst>()) {
        // Loads do not change any fact in this transfer domain. During the
        // fixed-point phase they must therefore be ignored: guarded facts
        // can be discovered only after another predecessor arrives, so an
        // obligation that is not yet provable is not a fixed-point failure.
        // Every load is checked in a separate validation sweep below, using
        // the final incoming state of its block.
        if (!validate_reads) { return finish(true); }
        auto *load = static_cast<LoadInst *>(instruction);
        auto *pointer = load->variable();
        if (!context.region.pointers.contains(pointer) ||
            pointer_is_defined(
                pointer, state.facts, context.layout, context.domain)) {
            return finish(true);
        }
        auto *element = top_array_gep(pointer, context.array);
        if (element != nullptr && element->index_count() >= 1u &&
            index_is_initialized(
                element->index(0u), load, context.array,
                context.counter, state.prefix_defined,
                state.counter_lower_bound,
                state.relations,
                state.facts, context.layout, context.domain,
                context.resolver, context.locations)) {
            used_prefix_read = true;
            return finish(true);
        }
        failing_read = load;
        if (auto *dump = std::getenv("LUISA_CORO_DUMP_ALLOCA_SCOPE");
            dump != nullptr && luisa::string_view{dump} == "1") {
            auto *index = element != nullptr &&
                                  element->index_count() != 0u ?
                              element->index(0u) :
                              nullptr;
            auto *resolved_index = index == nullptr ?
                                       nullptr :
                                       context.resolver.resolve(index, load);
            auto *index_slot = index == nullptr ?
                                   nullptr :
                                   current_scalar_slot(
                                       index, load, context.resolver,
                                       context.locations);
            auto location = context.locations.find(load);
            auto block_id = location == context.locations.end() ?
                                ~size_t{0u} :
                                location->second.block_id;
            auto ordinal = location == context.locations.end() ?
                               ~size_t{0u} :
                               location->second.ordinal;
            XIRDebugPrinter printer;
            luisa::string read_ir;
            printer.emit_instruction(read_ir, load);
            luisa::string index_ir;
            if (resolved_index != nullptr &&
                resolved_index->isa<Instruction>()) {
                printer.emit_instruction(
                    index_ir,
                    static_cast<Instruction *>(resolved_index));
            }
            LUISA_INFO(
                "Coroutine initialized-prefix failing load: array='{}' "
                "counter='{}' block={} ordinal={} prefix={} lower={} upper={} positive={} "
                "index='{}' resolved='{}' slot='{}' less={} equal={} last={} tail={} "
                "read_ir=[{}] index_ir=[{}].",
                context.array->name().value_or("<unnamed>"),
                context.counter->name().value_or("<unnamed>"),
                block_id, ordinal, state.prefix_defined,
                state.counter_lower_bound,
                state.counter_upper_bound.value_or(~size_t{0u}),
                state.relations.knows_counter_positive(),
                index == nullptr ?
                    luisa::string_view{"<none>"} :
                    index->name().value_or("<unnamed>"),
                resolved_index == nullptr ?
                    luisa::string_view{"<none>"} :
                    resolved_index->name().value_or("<unnamed>"),
                index_slot == nullptr ?
                    luisa::string_view{"<none>"} :
                    index_slot->name().value_or("<unnamed>"),
                state.relations.knows_less(index_slot),
                state.relations.knows_equal(index_slot),
                state.relations.knows_last(index_slot),
                state.relations.knows_tail(),
                read_ir, index_ir);
        }
        return finish(false);
    }
    return finish(true);
}

[[nodiscard]] CoroInitializedPrefixProofResult prove_candidate(
    AllocaInst *array, AllocaInst *counter,
    BasicBlock *target, Instruction *insertion,
    const ArrayUseRegion &region, const ActiveSlice &slice,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain,
    const ScalarCopyResolver &resolver,
    const PrefixInstructionLocationMap &locations,
    const luisa::unordered_set<Value *> &boolean_guards) noexcept {
    CoroInitializedPrefixProofResult result;
    if (!resolver.scalar_slot_has_only_direct_accesses(counter)) {
        return result;
    }
    if (array->type()->dimension() >
        unsigned_type_max(counter->type())) {
        return result;
    }
    auto layout = make_fact_layout(array, region, domain);
    if (layout.count == 0u) { return result; }
    auto insertion_iter = locations.find(insertion);
    if (insertion_iter == locations.end() ||
        insertion_iter->second.block_id != slice.target_id) {
        return result;
    }

    // Preflight the array pointer language. Prefix reasoning is layered over
    // exact typed loads/stores only; calls, atomics, Phis, and unknown pointer
    // consumers remain the responsibility of the conservative base proof.
    for (auto *instruction : region.users) {
        auto supported =
            (instruction->isa<GEPInst>() &&
             region.pointers.contains(
                 static_cast<GEPInst *>(instruction)->base())) ||
            (instruction->isa<LoadInst>() &&
             region.pointers.contains(
                 static_cast<LoadInst *>(instruction)->variable())) ||
            (instruction->isa<StoreInst>() &&
             region.pointers.contains(
                 static_cast<StoreInst *>(instruction)->variable()));
        if (!supported) { return result; }
        auto iter = locations.find(instruction);
        if (iter == locations.end() ||
            (iter->second.block_id == slice.target_id &&
             iter->second.ordinal < insertion_iter->second.ordinal)) {
            return result;
        }
    }

    auto relation_slots = collect_relation_slots(
        array, counter, region, resolver, locations);
    CoroScalarRelationLiveness relation_liveness{
        graph, slice.active, slice.target_id,
        relation_slots.slots, relation_slots.semantic_uses};
    auto boolean_flow = collect_boolean_predicate_flow(
        graph, resolver, locations, boolean_guards);
    CoroBooleanPredicateLiveness boolean_liveness{
        graph, slice.active, slice.target_id,
        boolean_flow.predicates, boolean_flow.uses,
        boolean_flow.definitions};
    auto masked_scalar_witnesses = collect_masked_scalar_witnesses(
        graph, slice.active, resolver, locations);
    auto counter_availability_witnesses =
        collect_counter_availability_witnesses(
            array, counter, region, graph, slice.active,
            resolver, locations);
    MaskedScalarKnownZeroAnalysis initially_known_zero{
        graph, resolver, locations,
        masked_scalar_witnesses.tracked, target, insertion};

    CandidateContext context{
        .array = array,
        .counter = counter,
        .dimension = array->type()->dimension(),
        .region = region,
        .layout = layout,
        .domain = domain,
        .graph = graph,
        .resolver = resolver,
        .locations = locations,
        .boolean_guards = boolean_guards,
        .relation_liveness = relation_liveness,
        .boolean_liveness = boolean_liveness,
        .known_zero = initially_known_zero,
        .masked_scalar_witnesses = masked_scalar_witnesses.tracked,
        .counter_availability_witnesses =
            counter_availability_witnesses};

    auto starts_with_empty_prefix = scalar_is_zero_at_instruction(
        counter, target, insertion, graph, resolver, locations);

    // The scalar-relation component is a canonical Boolean set domain. For
    // every ticket it records the valuations on which I<C is unsafe. Edge
    // refinement is set intersection and CFG join is set union, so the join
    // is associative, commutative and idempotent. The remaining components
    // are ordinary finite Must facts. Starting from the first executable
    // product and only merging weaker products therefore yields a monotone,
    // terminating fixed point independent of predecessor arrival order.
    CoroBooleanSetManager boolean_sets;
    luisa::vector<luisa::optional<PrefixState>> in_states(
        graph.block_count());
    luisa::vector<luisa::optional<PrefixState>> out_states(
        graph.block_count());
    in_states[slice.target_id].emplace(
        FactState(layout.count, uint8_t{0u}), boolean_sets,
        masked_scalar_witnesses.tracked,
        counter_availability_witnesses,
        starts_with_empty_prefix,
        starts_with_empty_prefix ?
            luisa::optional<size_t>{0u} :
            luisa::nullopt);
    // The placement point may follow a dominating scalar invariant even when
    // the counted array's lifetime begins at a later C:=0. Seed every tracked
    // projection independently proved by the sparse whole-CFG MustMaskedZero
    // analysis. The proof follows lowering-created scalar copies but never
    // assumes a renderer-specific lifetime boundary.
    for (auto witness : masked_scalar_witnesses.tracked) {
        if (initially_known_zero.proves_zero(
                witness.scalar, witness.mask)) {
            in_states[slice.target_id]
                ->relations.assume_masked_scalar_zero(
                    witness.scalar, witness.mask);
        }
    }
    for (auto witness : counter_availability_witnesses) {
        if (scalar_zero_class_at_instruction(
                witness.scalar, target, insertion, graph,
                resolver, locations) == ScalarZeroClass::nonzero) {
            in_states[slice.target_id]
                ->relations.assume_scalar_nonzero(witness.scalar);
        }
    }
    luisa::vector<size_t> worklist{slice.target_id};
    luisa::vector<uint8_t> queued(graph.block_count(), 0u);
    queued[slice.target_id] = 1u;
    auto used_prefix_read = false;
    for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
        auto block_id = worklist[cursor];
        queued[block_id] = 0u;
        ++result.block_evaluation_count;
        auto state = *in_states[block_id];
        for (auto *instruction : graph.block(block_id)->instructions()) {
            auto iter = locations.find(instruction);
            if (block_id == slice.target_id &&
                iter != locations.end() &&
                iter->second.ordinal < insertion_iter->second.ordinal) {
                continue;
            }
            static_cast<void>(process_instruction(
                instruction, state, context, used_prefix_read,
                result.failing_read, false));
        }
        if (!state.relations.feasible()) {
            out_states[block_id] = state;
            continue;
        }
        if (out_states[block_id] && *out_states[block_id] == state) {
            continue;
        }
        out_states[block_id] = state;
        for (auto successor : graph.successors(block_id)) {
            if (successor == slice.target_id ||
                slice.active[successor] == 0u) {
                continue;
            }
            auto edge_state = state;
            if (!refine_counter_bound_on_edge(
                    edge_state, graph.block(block_id),
                    graph.block(successor), context)) {
                continue;
            }
            // Boolean projection must not destroy a proof that is already
            // independent of the counter. Capture Safe(I) as the stable
            // Initialized(A,I) fact before dead control predicates vanish.
            if (edge_state.prefix_defined) {
                edge_state.relations.materialize_initialized();
            }
            edge_state.relations.retain_indices(
                relation_liveness.live_in(successor));
            edge_state.relations.retain_booleans(
                boolean_liveness.live_in(successor));
            auto changed = false;
            if (!in_states[successor]) {
                in_states[successor] = std::move(edge_state);
                changed = true;
            } else {
                changed = merge_state(
                    *in_states[successor], edge_state);
            }
            if (changed && queued[successor] == 0u) {
                queued[successor] = 1u;
                worklist.emplace_back(successor);
            }
        }
    }

    // Validate the initialized-read obligations only after the abstract
    // transition system has reached its fixed point. The transfer is replayed
    // from each final block input so stores and predicate copies preceding a
    // read in the same block are observed exactly once. Unreachable blocks
    // have no input state and impose no obligation.
    used_prefix_read = false;
    result.failing_read = nullptr;
    for (auto block_id : slice.blocks) {
        if (!in_states[block_id] ||
            !in_states[block_id]->relations.feasible()) {
            continue;
        }
        auto state = *in_states[block_id];
        for (auto *instruction : graph.block(block_id)->instructions()) {
            // An assume may make the remainder of this block unreachable. Such
            // a path has no concrete initialized-read obligation.
            if (!state.relations.feasible()) { break; }
            auto iter = locations.find(instruction);
            if (block_id == slice.target_id &&
                iter != locations.end() &&
                iter->second.ordinal < insertion_iter->second.ordinal) {
                continue;
            }
            if (!process_instruction(
                    instruction, state, context, used_prefix_read,
                    result.failing_read, true)) {
                return result;
            }
        }
    }
    result.succeeded = used_prefix_read;
    if (result.succeeded) {
        result.placement_block = target;
        result.placement_instruction = insertion;
    }
    return result;
}

[[nodiscard]] luisa::vector<StoreInst *> collect_zero_stores(
    AllocaInst *slot, const ScalarCopyResolver &resolver) noexcept {
    luisa::vector<StoreInst *> stores;
    if (!resolver.scalar_slot_has_only_direct_accesses(slot)) {
        return stores;
    }
    for (auto *use : slot->use_list()) {
        auto *user = use == nullptr ? nullptr : use->user();
        if (user == nullptr || !user->isa<StoreInst>()) { continue; }
        auto *store = static_cast<StoreInst *>(user);
        if (store->variable() != slot) { continue; }
        auto constant = decode_unsigned_constant(
            resolver.resolve(store->value(), store));
        if (constant && *constant == 0u) {
            stores.emplace_back(store);
        }
    }
    return stores;
}

[[nodiscard]] bool block_dominates_array_region(
    BasicBlock *block, const ArrayUseRegion &region,
    const CoroSemanticGraph &graph) noexcept {
    return block != nullptr && std::all_of(
        region.blocks.begin(), region.blocks.end(),
        [&](BasicBlock *use_block) noexcept {
            return graph.dominates(block, use_block);
        });
}

[[nodiscard]] Instruction *find_prefix_lifetime_insertion(
    BasicBlock *target, StoreInst *reset) noexcept {
    if (target == nullptr || reset == nullptr ||
        reset->parent_block() != target) {
        return nullptr;
    }
    // The proof does not require the storage lifetime to begin at the exact
    // C:=0 transition: it may begin earlier with Prefix(A,C) unknown and then
    // observe that reset. Choose the earliest point after the last resume in
    // the reset block. This keeps the storage inside one coroutine
    // subroutine while retaining scalar snapshots and assumptions that feed
    // later guarded prefix reads. Starting immediately after C:=0 would
    // discard those real def-use relations and force an application-level
    // lifetime annotation to recover them.
    Instruction *candidate = nullptr;
    for (auto *instruction : target->instructions()) {
        if (instruction == reset) {
            return candidate != nullptr ? candidate : reset;
        }
        if (instruction->isa<CoroResumeInst>()) {
            candidate = nullptr;
        } else if (candidate == nullptr) {
            candidate = instruction;
        }
    }
    return nullptr;
}

}// namespace

CoroInitializedPrefixProofResult
prove_initialized_prefix_fresh_lifetime(
    AllocaInst *array, BasicBlock *target,
    Instruction *insertion_instruction,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain) noexcept {
    CoroInitializedPrefixProofResult result;
    if (array == nullptr || array->type() == nullptr ||
        array->type()->tag() != Type::Tag::ARRAY ||
        array->type()->dimension() == 0u || target == nullptr ||
        insertion_instruction == nullptr) {
        return result;
    }
    auto *parent_function = array->parent_function();
    auto *definition = parent_function == nullptr ?
                           nullptr :
                           parent_function->definition();
    if (definition == nullptr) { return result; }
    auto locations = make_prefix_instruction_locations(definition, graph);
    auto region = collect_array_use_region(array, definition, graph);
    ScalarCopyResolver resolver{locations, graph};
    auto boolean_guards = collect_boolean_guard_slots(
        graph, resolver, locations);

    struct CounterCandidate {
        AllocaInst *counter;
        luisa::vector<StoreInst *> zero_stores;
    };
    luisa::vector<CounterCandidate> counters;
    for (auto *instruction : region.users) {
        if (!instruction->isa<StoreInst>()) { continue; }
        auto *counter = counter_from_full_element_store(
            static_cast<StoreInst *>(instruction), array,
            resolver, locations);
        if (counter != nullptr &&
            std::none_of(
                counters.begin(), counters.end(),
                [counter](auto &&candidate) noexcept {
                    return candidate.counter == counter;
                })) {
            auto zero_stores = collect_zero_stores(counter, resolver);
            zero_stores.erase(
                std::remove_if(
                    zero_stores.begin(), zero_stores.end(),
                    [&](StoreInst *store) noexcept {
                        return !block_dominates_array_region(
                            store->parent_block(), region, graph);
                    }),
                zero_stores.end());
            // Prefix(A, C) can only be established by C = 0. Rejecting a
            // candidate with no dominating exact reset avoids both false
            // counter identities and needless whole-CFG fixed points. This
            // deliberately rejects branch-distributed resets: they require a
            // separate relational proof rather than multiplying candidates.
            if (!zero_stores.empty()) {
                counters.emplace_back(CounterCandidate{
                    counter, std::move(zero_stores)});
            }
        }
    }
    auto slice = make_active_slice(target, region, graph);
    if (!slice.valid) { return result; }
    for (auto &&counter : counters) {
        auto candidate = prove_candidate(
            array, counter.counter, target, insertion_instruction,
            region, slice, graph, domain, resolver, locations,
            boolean_guards);
        result.block_evaluation_count +=
            candidate.block_evaluation_count;
        if (candidate.succeeded) {
            candidate.block_evaluation_count =
                result.block_evaluation_count;
            return candidate;
        }
        if (candidate.failing_read != nullptr) {
            result.failing_read = candidate.failing_read;
        }
    }

    // The deepest use dominator may be an inner loop header that carries the
    // counted array between iterations. In that case, place the lifetime at
    // the deepest dominating C = 0 store instead. This is not an inferred
    // initialization: the chosen reset is an exact store, it textually
    // precedes the insertion point, its block dominates every array use, and
    // the complete prefix proof is rerun from that outer boundary.
    auto *source = array->parent_block();
    for (auto &&counter : counters) {
        StoreInst *deepest_reset = nullptr;
        for (auto *reset : counter.zero_stores) {
            auto *block = reset->parent_block();
            if (block == source ||
                !block_dominates_array_region(block, region, graph)) {
                continue;
            }
            if (deepest_reset == nullptr ||
                graph.dominates(
                    deepest_reset->parent_block(), block)) {
                deepest_reset = reset;
            }
        }
        if (deepest_reset == nullptr) { continue; }
        auto *outer_target = deepest_reset->parent_block();
        auto *outer_insertion = find_prefix_lifetime_insertion(
            outer_target, deepest_reset);
        if (outer_insertion == nullptr ||
            (outer_target == target &&
             outer_insertion == insertion_instruction)) {
            continue;
        }
        auto outer_slice = make_active_slice(
            outer_target, region, graph);
        if (!outer_slice.valid) { continue; }
        auto candidate = prove_candidate(
            array, counter.counter, outer_target, outer_insertion,
            region, outer_slice, graph, domain, resolver, locations,
            boolean_guards);
        result.block_evaluation_count +=
            candidate.block_evaluation_count;
        if (candidate.succeeded) {
            candidate.block_evaluation_count =
                result.block_evaluation_count;
            return candidate;
        }
        if (candidate.failing_read != nullptr) {
            result.failing_read = candidate.failing_read;
        }
    }
    return result;
}

}// namespace luisa::compute::xir::detail
