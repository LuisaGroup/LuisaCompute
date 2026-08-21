#include "coro_initialized_prefix.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/store.h>

#include "coro_frame_access.h"
#include "coro_semantic_graph.h"

namespace luisa::compute::xir::detail {

namespace {

struct InstructionLocation {
    size_t block_id;
    size_t ordinal;
};

using InstructionLocationMap =
    luisa::unordered_map<Instruction *, InstructionLocation>;

[[nodiscard]] InstructionLocationMap make_instruction_locations(
    FunctionDefinition *definition,
    const CoroSemanticGraph &graph) noexcept {
    InstructionLocationMap locations;
    for (auto *block : definition->basic_blocks()) {
        auto ordinal = size_t{0u};
        for (auto *instruction : block->instructions()) {
            locations.emplace(
                instruction,
                InstructionLocation{
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
    const InstructionLocationMap &locations) noexcept {
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
};

class ScalarCopyResolver {
private:
    const InstructionLocationMap &_locations;
    mutable luisa::unordered_map<AllocaInst *, ScalarSlotInfo> _slots;

private:
    [[nodiscard]] ScalarSlotInfo _slot_info(
        AllocaInst *slot) const noexcept {
        if (auto iter = _slots.find(slot); iter != _slots.end()) {
            return iter->second;
        }
        ScalarSlotInfo info;
        if (slot == nullptr || !slot->is_local() ||
            slot->type() == nullptr || !slot->type()->is_scalar()) {
            _slots.emplace(slot, info);
            return info;
        }
        info.valid = true;
        for (auto *use : slot->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<Instruction>()) {
                info.valid = false;
                break;
            }
            auto *instruction = static_cast<Instruction *>(user);
            if (instruction->isa<LoadInst>() &&
                static_cast<LoadInst *>(instruction)->variable() == slot) {
                continue;
            }
            if (instruction->isa<StoreInst>() &&
                static_cast<StoreInst *>(instruction)->variable() == slot &&
                info.single_store == nullptr) {
                info.single_store = static_cast<StoreInst *>(instruction);
                continue;
            }
            info.valid = false;
            break;
        }
        info.valid &= info.single_store != nullptr;
        _slots.emplace(slot, info);
        return info;
    }

public:
    explicit ScalarCopyResolver(
        const InstructionLocationMap &locations) noexcept
        : _locations{locations} {}

    // A single-store scalar local is an exact snapshot only within the same
    // linear block execution. This restriction rules out loop-carried and
    // cross-edge substitution without requiring memory SSA.
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
        auto info = _slot_info(slot);
        if (!info.valid || load->parent_block() != use->parent_block() ||
            info.single_store->parent_block() != load->parent_block() ||
            !instruction_precedes(info.single_store, load, _locations) ||
            !instruction_precedes(load, use, _locations)) {
            return value;
        }
        return resolve(info.single_store->value(), info.single_store,
                       depth + 1u);
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
    const InstructionLocationMap &locations) noexcept {
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
    const InstructionLocationMap &locations) noexcept {
    value = resolver.resolve(value, use);
    if (value == nullptr || !value->isa<LoadInst>()) { return false; }
    auto *load = static_cast<LoadInst *>(value);
    return load->variable() == slot &&
           load->parent_block() == use->parent_block() &&
           instruction_precedes(load, use, locations) &&
           !has_store_between(slot, load, use, locations);
}

[[nodiscard]] bool same_scalar_snapshot(
    Value *lhs, Value *rhs, Instruction *use,
    const ScalarCopyResolver &resolver,
    const InstructionLocationMap &locations,
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

[[nodiscard]] AllocaInst *counter_from_full_element_store(
    StoreInst *store, AllocaInst *array,
    const ScalarCopyResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
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

[[nodiscard]] bool condition_implies_less_than_counter(
    Value *condition, bool truth, Value *index,
    AllocaInst *counter, Instruction *read,
    const ScalarCopyResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
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

[[nodiscard]] bool index_is_initialized(
    Value *index, Instruction *read, AllocaInst *array,
    AllocaInst *counter, bool prefix_defined,
    const FactState &facts, const FactLayout &layout,
    const CoroFrameAtomDomain &domain,
    const ScalarCopyResolver &resolver,
    const InstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (depth >= 8u) { return false; }
    if (static_array_element_is_defined(
            index, read, array, facts, layout, domain, resolver)) {
        return true;
    }
    index = resolver.resolve(index, read);
    if (!prefix_defined || index == nullptr ||
        !index->isa<ArithmeticInst>()) {
        return false;
    }
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
    const InstructionLocationMap &locations) noexcept {
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

struct PrefixState {
    FactState facts;
    bool prefix_defined{false};
    bool pending_extension{false};
    luisa::optional<size_t> counter_upper_bound;

    [[nodiscard]] bool operator==(
        const PrefixState &) const noexcept = default;
};

[[nodiscard]] bool merge_state(
    PrefixState &target, const PrefixState &incoming) noexcept {
    auto changed = false;
    for (size_t i = 0u; i < target.facts.size(); ++i) {
        auto next = static_cast<uint8_t>(
            target.facts[i] & incoming.facts[i]);
        changed |= next != target.facts[i];
        target.facts[i] = next;
    }
    auto next_prefix =
        target.prefix_defined && incoming.prefix_defined;
    auto next_pending =
        target.pending_extension && incoming.pending_extension;
    changed |= next_prefix != target.prefix_defined ||
               next_pending != target.pending_extension;
    target.prefix_defined = next_prefix;
    target.pending_extension = next_pending;
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
    const InstructionLocationMap &locations) noexcept {
    auto *gep = top_array_gep(store->variable(), array);
    return gep != nullptr && store->variable() == gep &&
           gep->base() == array &&
           gep->index_count() == 1u &&
           gep->type() == array->type()->element() &&
           is_current_slot_snapshot(
               gep->index(0u), counter, store, resolver, locations);
}

struct CandidateContext {
    AllocaInst *array;
    AllocaInst *counter;
    size_t dimension;
    const ArrayUseRegion &region;
    const FactLayout &layout;
    const CoroFrameAtomDomain &domain;
    const ScalarCopyResolver &resolver;
    const InstructionLocationMap &locations;
};

[[nodiscard]] bool process_instruction(
    Instruction *instruction, PrefixState &state,
    const CandidateContext &context, bool &used_prefix_read,
    Instruction *&failing_read) noexcept {
    if (instruction->isa<GEPInst>() &&
        context.region.pointers.contains(instruction)) {
        redefine_pointer(instruction, state.facts, context.layout);
    }

    if (instruction->isa<StoreInst>()) {
        auto *store = static_cast<StoreInst *>(instruction);
        auto *pointer = store->variable();
        if (context.region.pointers.contains(pointer)) {
            define_pointer(
                pointer, state.facts, context.layout, context.domain);
            if (state.prefix_defined &&
                state.counter_upper_bound &&
                *state.counter_upper_bound < context.dimension &&
                full_element_store_at_current_counter(
                    store, context.array, context.counter,
                    context.resolver, context.locations)) {
                state.pending_extension = true;
            }
        }
        if (pointer == context.counter) {
            auto constant = decode_unsigned_constant(
                context.resolver.resolve(store->value(), store));
            if (constant && *constant == 0u) {
                state.prefix_defined = true;
                state.pending_extension = false;
                state.counter_upper_bound = 0u;
            } else if (is_current_slot_snapshot(
                           store->value(), context.counter, store,
                           context.resolver, context.locations)) {
                // Exact self-assignment preserves both the prefix and a
                // pending extension for the same counter value.
            } else if (match_counter_increment(
                           store->value(), context.counter, store,
                           context.resolver, context.locations) &&
                       state.prefix_defined &&
                       state.pending_extension &&
                       state.counter_upper_bound &&
                       *state.counter_upper_bound < context.dimension) {
                ++*state.counter_upper_bound;
                state.pending_extension = false;
            } else {
                state.prefix_defined = false;
                state.pending_extension = false;
                state.counter_upper_bound = luisa::nullopt;
            }
        }
    }

    if (instruction->isa<LoadInst>()) {
        auto *load = static_cast<LoadInst *>(instruction);
        auto *pointer = load->variable();
        if (!context.region.pointers.contains(pointer) ||
            pointer_is_defined(
                pointer, state.facts, context.layout, context.domain)) {
            return true;
        }
        auto *element = top_array_gep(pointer, context.array);
        if (element != nullptr && element->index_count() >= 1u &&
            index_is_initialized(
                element->index(0u), load, context.array,
                context.counter, state.prefix_defined,
                state.facts, context.layout, context.domain,
                context.resolver, context.locations)) {
            used_prefix_read = true;
            return true;
        }
        failing_read = load;
        return false;
    }
    return true;
}

[[nodiscard]] CoroInitializedPrefixProofResult prove_candidate(
    AllocaInst *array, AllocaInst *counter,
    BasicBlock *target, Instruction *insertion,
    const ArrayUseRegion &region, const ActiveSlice &slice,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain,
    const ScalarCopyResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
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

    CandidateContext context{
        .array = array,
        .counter = counter,
        .dimension = array->type()->dimension(),
        .region = region,
        .layout = layout,
        .domain = domain,
        .resolver = resolver,
        .locations = locations};

    luisa::vector<luisa::optional<PrefixState>> in_states(
        graph.block_count());
    in_states[slice.target_id] = PrefixState{
        .facts = FactState(layout.count, uint8_t{0u})};
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
                // The array lifetime has not started yet, but an earlier
                // counter reset in the same block can establish the empty
                // Prefix(A, C) relation at the insertion point. Process only
                // direct counter stores; no array observation is allowed here
                // by the preflight above.
                if (!instruction->isa<StoreInst>() ||
                    static_cast<StoreInst *>(instruction)->variable() !=
                        counter) {
                    continue;
                }
            }
            if (!process_instruction(
                    instruction, state, context, used_prefix_read,
                    result.failing_read)) {
                return result;
            }
        }
        for (auto successor : graph.successors(block_id)) {
            if (successor == slice.target_id ||
                slice.active[successor] == 0u) {
                continue;
            }
            auto changed = false;
            if (!in_states[successor]) {
                in_states[successor] = state;
                changed = true;
            } else {
                changed = merge_state(*in_states[successor], state);
            }
            if (changed && queued[successor] == 0u) {
                queued[successor] = 1u;
                worklist.emplace_back(successor);
            }
        }
    }
    result.succeeded = used_prefix_read;
    return result;
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
    auto locations = make_instruction_locations(definition, graph);
    auto region = collect_array_use_region(array, definition, graph);
    auto slice = make_active_slice(target, region, graph);
    if (!slice.valid) { return result; }
    ScalarCopyResolver resolver{locations};

    luisa::vector<AllocaInst *> counters;
    for (auto *instruction : region.users) {
        if (!instruction->isa<StoreInst>()) { continue; }
        auto *counter = counter_from_full_element_store(
            static_cast<StoreInst *>(instruction), array,
            resolver, locations);
        if (counter != nullptr &&
            std::find(counters.begin(), counters.end(), counter) ==
                counters.end()) {
            counters.emplace_back(counter);
        }
    }
    for (auto *counter : counters) {
        auto candidate = prove_candidate(
            array, counter, target, insertion_instruction,
            region, slice, graph, domain, resolver, locations);
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
