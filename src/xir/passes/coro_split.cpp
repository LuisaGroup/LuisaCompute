#include <algorithm>
#include <cstdlib>
#include <limits>

#include "helpers.h"
#include "coro_replayable.h"

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/op.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/special_register.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static bool verify_intermediate_xir_enabled() noexcept {
    if (auto value = std::getenv("LUISA_XIR_VERIFY_INTERMEDIATE")) {
        return luisa::string_view{value} == "1";
    }
    return false;
}

static constexpr uint32_t FRAME_FIELD_ID_X = 0u;
static constexpr uint32_t FRAME_FIELD_ID_Y = 1u;
static constexpr uint32_t FRAME_FIELD_ID_Z = 2u;
static constexpr uint32_t FRAME_FIELD_SIZE_X = 3u;
static constexpr uint32_t FRAME_FIELD_SIZE_Y = 4u;
static constexpr uint32_t FRAME_FIELD_SIZE_Z = 5u;
static constexpr uint32_t FRAME_FIELD_TOKEN = 6u;
static constexpr uint32_t FRAME_USER_FIELD_OFFSET = 7u;

static void coro_split_clone_metadata(const MetadataListMixin &source,
                           MetadataListMixin &target) noexcept {
    for (auto *metadata : source.metadata_list()) {
        target.metadata_list().push_front(metadata->clone());
    }
}

class CoroSplitValueResolver final : public InstructionCloneValueResolver {

private:
    luisa::unordered_map<const Value *, Value *> _value_map;
    luisa::unordered_map<const Value *, Value *> _entry_value_map;
    struct RematerializedValue {
        const BasicBlock *original_block;
        Value *value;
    };
    luisa::unordered_map<const Value *,
                         luisa::vector<RematerializedValue>>
        _rematerialized_values;
    luisa::unordered_map<const BasicBlock *, BasicBlock *> _block_map;
    luisa::unordered_map<const Argument *, Argument *> _arg_map;
    luisa::unordered_set<const BasicBlock *> _scope_blocks;
    XIRBuilder *_builder{nullptr};
    BasicBlock *_alloca_bb{nullptr};
    Instruction *_alloca_insertion_point{nullptr};
    Value *_frame_arg{nullptr};
    Module *_module{nullptr};
    const BasicBlock *_scope_root{nullptr};
    const BasicBlock *_current_orig_block{nullptr};
    detail::CoroReplayableValueAnalysis _replayable;

private:
    [[nodiscard]] bool _scope_dominates(const BasicBlock *def, const BasicBlock *use) const noexcept {
        if (def == nullptr || use == nullptr || _scope_root == nullptr) { return false; }
        if (def == use) { return true; }
        if (!_scope_blocks.contains(def) || !_scope_blocks.contains(use)) { return false; }
        if (use == _scope_root) { return false; }
        if (def == _scope_root) { return true; }
        luisa::unordered_set<const BasicBlock *> visited;
        luisa::vector<const BasicBlock *> work;
        visited.emplace(_scope_root);
        work.emplace_back(_scope_root);
        while (!work.empty()) {
            auto *bb = work.back();
            work.pop_back();
            auto *mut_bb = const_cast<BasicBlock *>(bb);
            mut_bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (!_scope_blocks.contains(succ) || succ == def) { return; }
                if (visited.emplace(succ).second) {
                    work.emplace_back(succ);
                }
            });
            if (visited.contains(use)) { return false; }
        }
        return true;
    }

public:
    void set_builder(XIRBuilder *b, BasicBlock *alloca_bb) noexcept {
        _builder = b;
        _alloca_bb = alloca_bb;
        _alloca_insertion_point =
            alloca_bb == nullptr ?
                nullptr :
                alloca_bb->instructions().head_sentinel();
    }

    void set_frame_arg(Module *module, Value *frame_arg) noexcept {
        _module = module;
        _frame_arg = frame_arg;
    }

    void set_scope(const BasicBlock *root, luisa::unordered_set<const BasicBlock *> blocks) noexcept {
        _scope_root = root;
        _scope_blocks = std::move(blocks);
    }

    void set_current_original_block(const BasicBlock *bb) noexcept {
        _current_orig_block = bb;
    }

    void map_block(const BasicBlock *orig, BasicBlock *cloned) noexcept {
        _block_map.emplace(orig, cloned);
    }

    void map_arg(const Argument *orig, Argument *cloned) noexcept {
        _arg_map.emplace(orig, cloned);
    }

    void map_value(const Value *orig, Value *cloned) noexcept {
        if (orig != nullptr && cloned != nullptr && orig != cloned) {
            _value_map.emplace(orig, cloned);
            if (orig->isa<Instruction>() &&
                static_cast<const Instruction *>(orig)
                        ->derived_instruction_tag() ==
                    DerivedInstructionTag::ALLOCA &&
                cloned->isa<Instruction>() &&
                static_cast<Instruction *>(cloned)->parent_block() ==
                    _alloca_bb) {
                _alloca_insertion_point =
                    static_cast<Instruction *>(cloned);
            }
        }
    }

    void map_entry_value(const Value *orig, Value *value) noexcept {
        if (orig != nullptr && value != nullptr) {
            _entry_value_map.emplace(orig, value);
        }
    }

    [[nodiscard]] bool has_value(const Value *orig) const noexcept {
        return _value_map.contains(orig);
    }

    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) { return nullptr; }
        switch (value->derived_value_tag()) {
            case DerivedValueTag::UNDEFINED:
            case DerivedValueTag::FUNCTION:
            case DerivedValueTag::CONSTANT:
                return const_cast<Value *>(value);
            case DerivedValueTag::SPECIAL_REGISTER: {
                auto *sreg = static_cast<const SpecialRegister *>(value);
                auto tag = sreg->derived_special_register_tag();
                if ((tag == DerivedSpecialRegisterTag::DISPATCH_ID ||
                     tag == DerivedSpecialRegisterTag::DISPATCH_SIZE) &&
                    _builder != nullptr && _module != nullptr && _frame_arg != nullptr) {
                    auto load_uint_field = [&](uint32_t field_index) noexcept {
                        auto *idx = static_cast<Value *>(_module->create_constant(Type::of<uint32_t>(), &field_index));
                        auto *gep = _builder->gep(Type::of<uint>(), _frame_arg, {idx});
                        return static_cast<Value *>(_builder->load(Type::of<uint>(), gep));
                    };
                    auto base = tag == DerivedSpecialRegisterTag::DISPATCH_ID ?
                                    FRAME_FIELD_ID_X :
                                    FRAME_FIELD_SIZE_X;
                    auto *x = load_uint_field(base + 0u);
                    auto *y = load_uint_field(base + 1u);
                    auto *z = load_uint_field(base + 2u);
                    return _builder->call(Type::of<uint3>(), ArithmeticOp::AGGREGATE, {x, y, z});
                }
                return const_cast<Value *>(value);
            }
            case DerivedValueTag::BASIC_BLOCK: {
                auto it = _block_map.find(static_cast<const BasicBlock *>(value));
                LUISA_DEBUG_ASSERT(it != _block_map.end(), "Block not found in resolver.");
                return it->second;
            }
            case DerivedValueTag::ARGUMENT: {
                auto it = _arg_map.find(static_cast<const Argument *>(value));
                LUISA_DEBUG_ASSERT(it != _arg_map.end(), "Argument not found in resolver.");
                return it->second;
            }
            case DerivedValueTag::INSTRUCTION:
            default: {
                const auto *inst = static_cast<const Instruction *>(value);
                auto entry_it = _entry_value_map.find(inst);
                if (entry_it != _entry_value_map.end()) {
                    auto *def_block = inst->parent_block();
                    if (def_block == nullptr || !_scope_blocks.contains(def_block) ||
                        _current_orig_block == nullptr ||
                        !_scope_dominates(def_block, _current_orig_block)) {
                        return entry_it->second;
                    }
                }
                auto it = _value_map.find(inst);
                if (it != _value_map.end()) { return it->second; }
                if (entry_it != _entry_value_map.end()) { return entry_it->second; }
                if (auto remat = _rematerialized_values.find(inst);
                    remat != _rematerialized_values.end()) {
                    for (auto &&candidate : remat->second) {
                        if (_scope_dominates(
                                candidate.original_block,
                                _current_orig_block)) {
                            return candidate.value;
                        }
                    }
                }
                if (inst->derived_instruction_tag() ==
                        DerivedInstructionTag::ALLOCA &&
                    _alloca_insertion_point != nullptr) {
                    auto *orig_alloca =
                        static_cast<const AllocaInst *>(inst);
                    XIRBuilder alloca_builder;
                    alloca_builder.set_insertion_point(
                        _alloca_insertion_point);
                    auto *cloned = alloca_builder.alloca_(
                        orig_alloca->type(), orig_alloca->op());
                    coro_split_clone_metadata(*orig_alloca, *cloned);
                    _alloca_insertion_point = cloned;
                    _value_map.emplace(inst, cloned);
                    return cloned;
                }
                if (_builder != nullptr) {
                    if (_replayable.detect(inst)) {
                        auto *cloned = inst->clone_with_metadata(
                            *_builder, *this);
                        _rematerialized_values[inst].emplace_back(
                            RematerializedValue{
                                .original_block =
                                    _current_orig_block,
                                .value = cloned});
                        return cloned;
                    }
                }
                LUISA_ASSERT(
                    false,
                    "Coro split could not resolve a cloned {} instruction. "
                    "The distilled liveness/clone-order contract was violated.",
                    to_string(inst->derived_instruction_tag()));
                return nullptr;
            }
        }
    }
};

[[nodiscard]] static const Type *create_frame_type(const CoroCfgDistillResult &result) noexcept {
    luisa::vector<const Type *> fields;
    fields.reserve(FRAME_USER_FIELD_OFFSET + result.frame_slots.size());
    auto alignment = Type::of<uint>()->alignment();
    for (auto i = 0u; i < FRAME_USER_FIELD_OFFSET; ++i) {
        fields.emplace_back(Type::of<uint>());
    }
    for (auto &slot : result.frame_slots) {
        fields.emplace_back(slot.type);
        alignment = std::max(alignment, slot.type->alignment());
    }
    return Type::structure(alignment, fields);
}

[[nodiscard]] static bool validate_frame_type(const Type *frame_type,
                                              const CoroCfgDistillResult &result) noexcept {
    if (frame_type == nullptr || !frame_type->is_structure()) { return false; }
    auto members = frame_type->members();
    if (members.size() != FRAME_USER_FIELD_OFFSET + result.frame_slots.size()) { return false; }
    for (auto i = 0u; i < FRAME_USER_FIELD_OFFSET; ++i) {
        if (members[i] != Type::of<uint>()) { return false; }
    }
    for (size_t i = 0u; i < result.frame_slots.size(); ++i) {
        if (result.frame_slots[i].type == nullptr ||
            members[FRAME_USER_FIELD_OFFSET + i] != result.frame_slots[i].type) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] static bool coro_split_validate_coroutine_tokens(FunctionDefinition *def) noexcept {
    if (def == nullptr) { return false; }
    luisa::unordered_set<uint32_t> suspend_tokens;
    luisa::unordered_set<uint32_t> resume_tokens;
    auto valid = true;
    for (auto *block : def->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_SUSPEND) {
                auto token = static_cast<CoroSuspendInst *>(inst)->token();
                valid &= token != 0u && token != TERMINAL_TOKEN && suspend_tokens.emplace(token).second;
            } else if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_RESUME) {
                auto token = static_cast<CoroResumeInst *>(inst)->token();
                valid &= token != 0u && token != TERMINAL_TOKEN && resume_tokens.emplace(token).second;
            }
        }
    }
    if (suspend_tokens.size() != resume_tokens.size()) { return false; }
    for (auto token : suspend_tokens) {
        if (!resume_tokens.contains(token)) { return false; }
    }
    return valid;
}

[[nodiscard]] static bool distilled_cfg_matches_canonical(
    const CoroCfgDistillResult &result,
    const CoroCfgDistillResult &canonical) noexcept {
    if (result.scopes.size() != canonical.scopes.size() ||
        result.edges != canonical.edges ||
        result.transition_edges.size() !=
            canonical.transition_edges.size() ||
        result.frame_values.size() != canonical.frame_values.size() ||
        result.frame_slots.size() != canonical.frame_slots.size()) {
        return false;
    }
    for (auto i = 0u; i < result.scopes.size(); ++i) {
        auto &lhs = result.scopes[i];
        auto &rhs = canonical.scopes[i];
        if (lhs.blocks != rhs.blocks ||
            lhs.suspend_points.size() != rhs.suspend_points.size() ||
            lhs.scope_id != rhs.scope_id ||
            lhs.suspend_token != rhs.suspend_token ||
            lhs.suspend_name != rhs.suspend_name ||
            lhs.trigger_token != rhs.trigger_token ||
            lhs.trigger_name != rhs.trigger_name ||
            lhs.external_values != rhs.external_values ||
            lhs.touched_values != rhs.touched_values ||
            lhs.live_in_values != rhs.live_in_values ||
            lhs.live_out_values != rhs.live_out_values ||
            lhs.external_frame_value_indices !=
                rhs.external_frame_value_indices ||
            lhs.touched_frame_value_indices !=
                rhs.touched_frame_value_indices ||
            lhs.live_in_frame_value_indices !=
                rhs.live_in_frame_value_indices ||
            lhs.live_out_frame_value_indices !=
                rhs.live_out_frame_value_indices ||
            lhs.external_variables != rhs.external_variables ||
            lhs.touched_variables != rhs.touched_variables ||
            lhs.live_in_variables != rhs.live_in_variables ||
            lhs.live_out_variables != rhs.live_out_variables ||
            lhs.is_terminal != rhs.is_terminal) {
            return false;
        }
        for (auto j = 0u; j < lhs.suspend_points.size(); ++j) {
            auto &lhs_point = lhs.suspend_points[j];
            auto &rhs_point = rhs.suspend_points[j];
            if (lhs_point.block != rhs_point.block ||
                lhs_point.token != rhs_point.token ||
                lhs_point.name != rhs_point.name) {
                return false;
            }
        }
    }
    for (auto i = 0u; i < result.transition_edges.size(); ++i) {
        auto &lhs = result.transition_edges[i];
        auto &rhs = canonical.transition_edges[i];
        if (lhs.from_scope != rhs.from_scope ||
            lhs.to_scope != rhs.to_scope ||
            lhs.token != rhs.token ||
            lhs.exit_block != rhs.exit_block ||
            lhs.is_suspend != rhs.is_suspend ||
            lhs.killed_values != rhs.killed_values ||
            lhs.touched_values != rhs.touched_values ||
            lhs.live_values != rhs.live_values ||
            lhs.store_values != rhs.store_values ||
            lhs.killed_frame_value_indices !=
                rhs.killed_frame_value_indices ||
            lhs.touched_frame_value_indices !=
                rhs.touched_frame_value_indices ||
            lhs.live_frame_value_indices !=
                rhs.live_frame_value_indices ||
            lhs.store_frame_value_indices !=
                rhs.store_frame_value_indices ||
            lhs.killed_variables != rhs.killed_variables ||
            lhs.touched_variables != rhs.touched_variables ||
            lhs.live_variables != rhs.live_variables ||
            lhs.store_variables != rhs.store_variables) {
            return false;
        }
    }
    for (auto i = 0u; i < result.frame_values.size(); ++i) {
        auto &lhs = result.frame_values[i];
        auto &rhs = canonical.frame_values[i];
        if (lhs.value != rhs.value ||
            lhs.access_chain != rhs.access_chain ||
            lhs.name != rhs.name ||
            lhs.type != rhs.type || lhs.slot != rhs.slot) {
            return false;
        }
    }
    for (auto i = 0u; i < result.frame_slots.size(); ++i) {
        auto &lhs = result.frame_slots[i];
        auto &rhs = canonical.frame_slots[i];
        if (lhs.name != rhs.name || lhs.type != rhs.type) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] static const Type *resolve_static_access_type(
    const Type *type, luisa::span<const uint32_t> access_chain) noexcept {
    for (auto index : access_chain) {
        if (type == nullptr) { return nullptr; }
        switch (type->tag()) {
            case Type::Tag::ARRAY:
            case Type::Tag::VECTOR:
                if (index >= type->dimension()) { return nullptr; }
                type = type->element();
                break;
            case Type::Tag::MATRIX:
                if (index >= type->dimension()) { return nullptr; }
                type = Type::vector(type->element(), type->dimension());
                break;
            case Type::Tag::STRUCTURE: {
                auto members = type->members();
                if (index >= members.size()) { return nullptr; }
                type = members[index];
                break;
            }
            default: return nullptr;
        }
    }
    return type;
}

[[nodiscard]] static bool static_access_path_is_prefix(
    luisa::span<const uint32_t> prefix,
    luisa::span<const uint32_t> path) noexcept {
    return prefix.size() <= path.size() &&
           std::equal(prefix.begin(), prefix.end(), path.begin());
}

[[nodiscard]] static bool validate_distilled_cfg(FunctionDefinition *def,
                                                 const CoroCfgDistillResult &result) noexcept {
    if (def == nullptr || !result.succeeded() || result.scopes.empty() ||
        result.edges.size() != result.scopes.size()) {
        return false;
    }
    if (!coro_split_validate_coroutine_tokens(def)) { return false; }
    // Distillation seals both the source CFG version and all semantic result
    // fields. This linear certificate check rejects stale or caller-mutated
    // metadata without rerunning the liveness fixed point at every consumer.
    if (!result.validation_certificate_matches(def)) {
        return false;
    }
    // Full canonical recomputation is a diagnostic oracle, not part of the
    // production split algorithm. Enable it explicitly when auditing an XIR
    // pass boundary.
    if (verify_intermediate_xir_enabled()) {
        auto canonical = coro_cfg_distill_pass_run_on_function(def);
        if (!canonical.succeeded() ||
            !distilled_cfg_matches_canonical(result, canonical)) {
            return false;
        }
    }
    luisa::unordered_set<uint32_t> triggers;
    luisa::unordered_set<uint32_t> suspends;
    for (size_t i = 0u; i < result.scopes.size(); ++i) {
        auto &scope = result.scopes[i];
        if (scope.blocks.empty() || scope.scope_id != static_cast<int>(i)) { return false; }
        auto token = scope.trigger_token;
        if ((i == 0u && token != 0u) ||
            (i != 0u && (token == 0u || token == TERMINAL_TOKEN)) ||
            !triggers.emplace(token).second) {
            return false;
        }
        if (i == 0u) {
            if (scope.blocks.front() != def->body_block()) { return false; }
        } else {
            auto found_resume = false;
            for (auto *inst : scope.blocks.front()->instructions()) {
                if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_RESUME &&
                    static_cast<CoroResumeInst *>(inst)->token() == token) {
                    found_resume = true;
                    break;
                }
            }
            if (!found_resume) { return false; }
        }
        luisa::unordered_set<BasicBlock *> scope_blocks;
        luisa::unordered_set<BasicBlock *> scope_suspend_blocks;
        for (auto *block : scope.blocks) {
            if (block == nullptr || block->parent_function() != def) { return false; }
            if (!scope_blocks.emplace(block).second) { return false; }
            if (block->is_terminated() && block->terminator()->isa<CoroSuspendInst>()) {
                scope_suspend_blocks.emplace(block);
            }
        }
        for (auto &point : scope.suspend_points) {
            if (point.block == nullptr || point.token == 0u || point.token == TERMINAL_TOKEN) {
                return false;
            }
            if (!scope_blocks.contains(point.block) ||
                scope_suspend_blocks.erase(point.block) != 1u) {
                return false;
            }
            auto *suspend = static_cast<CoroSuspendInst *>(point.block->terminator());
            if (suspend->token() != point.token) { return false; }
            suspends.emplace(point.token);
        }
        if (!scope_suspend_blocks.empty()) { return false; }
        for (auto target : result.edges[i]) {
            if (target >= result.scopes.size()) { return false; }
        }
    }
    for (size_t i = 1u; i < result.scopes.size(); ++i) {
        if (!suspends.contains(result.scopes[i].trigger_token)) { return false; }
    }
    for (auto token : suspends) {
        if (!triggers.contains(token)) { return false; }
    }
    luisa::unordered_map<const Value *,
                         luisa::vector<luisa::vector<uint32_t>>>
        value_paths;
    luisa::vector<uint8_t> occupied_slots(
        result.frame_slots.size(), 0u);
    luisa::unordered_set<luisa::string> slot_names;
    for (auto &slot : result.frame_slots) {
        if (slot.type == nullptr || slot.name.empty() ||
            !slot_names.emplace(slot.name).second) {
            return false;
        }
    }
    for (auto &frame_value : result.frame_values) {
        auto path_root_is_local_alloca =
            frame_value.access_chain.empty() ||
            (frame_value.value != nullptr &&
             frame_value.value->isa<AllocaInst>() &&
             static_cast<AllocaInst *>(frame_value.value)->is_local());
        if (frame_value.value == nullptr || frame_value.type == nullptr ||
            !path_root_is_local_alloca ||
            frame_value.slot >= result.frame_slots.size() ||
            result.frame_slots[frame_value.slot].type != frame_value.type ||
            resolve_static_access_type(
                frame_value.value->type(), frame_value.access_chain) !=
                frame_value.type) {
            return false;
        }
        value_paths[frame_value.value].emplace_back(
            frame_value.access_chain);
        occupied_slots[frame_value.slot] = 1u;
        if (frame_value.value->isa<Instruction>()) {
            auto *inst = static_cast<Instruction *>(frame_value.value);
            if (inst->parent_block() == nullptr || inst->parent_block()->parent_function() != def) {
                return false;
            }
        }
    }
    // Frame fields for one root must denote a partition: duplicate paths and
    // ancestor/descendant pairs overlap the same storage and would make spill
    // order observable. In lexicographic order every descendant interval
    // starts immediately after its prefix, so adjacent-prefix checks are both
    // necessary and sufficient (and avoid a quadratic pairwise scan).
    for (auto &[value, paths] : value_paths) {
        static_cast<void>(value);
        std::sort(paths.begin(), paths.end());
        for (size_t i = 1u; i < paths.size(); ++i) {
            if (static_access_path_is_prefix(paths[i - 1u], paths[i])) {
                return false;
            }
        }
    }
    if (std::find(occupied_slots.begin(), occupied_slots.end(), 0u) !=
        occupied_slots.end()) {
        return false;
    }
    auto valid_frame_indices = [&](luisa::span<const size_t> indices) noexcept {
        for (auto index : indices) {
            if (index >= result.frame_values.size()) { return false; }
        }
        return true;
    };
    for (auto &scope : result.scopes) {
        if (!valid_frame_indices(scope.external_frame_value_indices) ||
            !valid_frame_indices(scope.touched_frame_value_indices) ||
            !valid_frame_indices(scope.live_in_frame_value_indices) ||
            !valid_frame_indices(scope.live_out_frame_value_indices)) {
            return false;
        }
    }
    for (auto &edge : result.transition_edges) {
        if (edge.from_scope >= result.scopes.size() || edge.to_scope >= result.scopes.size() ||
            edge.token != result.scopes[edge.to_scope].trigger_token) {
            return false;
        }
        if (!valid_frame_indices(edge.killed_frame_value_indices) ||
            !valid_frame_indices(edge.touched_frame_value_indices) ||
            !valid_frame_indices(edge.live_frame_value_indices) ||
            !valid_frame_indices(edge.store_frame_value_indices)) {
            return false;
        }
    }
    return true;
}

static void coro_split_clone_instruction_metadata(
    const Instruction *source, Instruction *target) noexcept {
    if (source == nullptr || target == nullptr) { return; }
    coro_split_clone_metadata(*source, *target);
}

static void store_frame_token(XIRBuilder &b, Value *frame_arg, Module *mod, uint32_t token) noexcept {
    auto *field_token = mod->create_constant(Type::of<uint32_t>(), &FRAME_FIELD_TOKEN);
    auto *gep = b.gep(Type::of<uint>(), frame_arg, {field_token});
    auto *tok_const = mod->create_constant(Type::of<uint>(), &token);
    b.store(gep, tok_const);
}

[[nodiscard]] static bool is_memory_frame_value(const Value *value) noexcept {
    return value != nullptr && value->isa<Instruction>() &&
           static_cast<const Instruction *>(value)->derived_instruction_tag() == DerivedInstructionTag::ALLOCA;
}

[[nodiscard]] static Value *frame_value_memory_pointer(
    XIRBuilder &b, Module *mod,
    const CoroCfgDistillResult::FrameValue &frame_value,
    CoroSplitValueResolver &resolver) noexcept {
    LUISA_DEBUG_ASSERT(is_memory_frame_value(frame_value.value),
                       "Coroutine memory frame value must be a local alloca.");
    auto *root = resolver.resolve(frame_value.value);
    if (frame_value.access_chain.empty()) { return root; }
    luisa::vector<Value *> indices;
    indices.reserve(frame_value.access_chain.size());
    for (auto component : frame_value.access_chain) {
        indices.emplace_back(
            mod->create_constant(Type::of<uint32_t>(), &component));
    }
    return b.gep(frame_value.type, root, indices);
}

[[nodiscard]] static Value *frame_field_ptr(XIRBuilder &b, Module *mod, Value *frame_arg,
                                            const Type *type, size_t field_index) noexcept {
    LUISA_ASSERT(field_index <= std::numeric_limits<uint32_t>::max(),
                 "Coroutine frame field index is not representable.");
    auto i = static_cast<uint32_t>(field_index);
    auto *idx = mod->create_constant(Type::of<uint32_t>(), &i);
    return b.gep(type, frame_arg, {idx});
}

static void store_live_values_to_frame(XIRBuilder &b, Module *mod, Value *frame_arg,
                                       const CoroCfgDistillResult &result,
                                       luisa::span<const size_t> frame_value_indices,
                                       CoroSplitValueResolver &resolver) noexcept {
    for (auto frame_value_index : frame_value_indices) {
        LUISA_DEBUG_ASSERT(frame_value_index < result.frame_values.size(),
                           "Coroutine frame value index is out of range.");
        auto &frame_value = result.frame_values[frame_value_index];
        auto field_index = FRAME_USER_FIELD_OFFSET + frame_value.slot;
        auto *field = frame_field_ptr(b, mod, frame_arg, frame_value.type, field_index);
        if (is_memory_frame_value(frame_value.value)) {
            auto *pointer = frame_value_memory_pointer(
                b, mod, frame_value, resolver);
            auto *loaded = b.load(frame_value.type, pointer);
            b.store(field, loaded);
        } else {
            b.store(field, resolver.resolve(frame_value.value));
        }
    }
}

[[nodiscard]] static luisa::span<const size_t> store_values_for_suspend(
    const CoroCfgDistillResult &result, size_t scope_index, uint32_t token) noexcept {
    for (auto &edge : result.transition_edges) {
        if (edge.is_suspend && edge.from_scope == scope_index && edge.token == token) {
            return luisa::span<const size_t>{edge.store_frame_value_indices};
        }
    }
    return {};
}

[[nodiscard]] static luisa::span<const size_t> store_values_for_branch_transition(
    const CoroCfgDistillResult &result, size_t scope_index,
    const BasicBlock *exit_block, size_t target_scope) noexcept {
    for (auto &edge : result.transition_edges) {
        if (!edge.is_suspend &&
            edge.from_scope == scope_index &&
            edge.to_scope == target_scope &&
            edge.exit_block == exit_block) {
            return luisa::span<const size_t>{edge.store_frame_value_indices};
        }
    }
    return {};
}

static void load_live_values_from_frame(XIRBuilder &b, Module *mod, Value *frame_arg,
                                        const CoroCfgDistillResult &result,
                                        const CoroCfgDistillResult::Scope &scope,
                                        CoroSplitValueResolver &resolver) noexcept {
    for (auto frame_value_index : scope.live_in_frame_value_indices) {
        LUISA_DEBUG_ASSERT(frame_value_index < result.frame_values.size(),
                           "Coroutine frame value index is out of range.");
        auto &frame_value = result.frame_values[frame_value_index];
        auto field_index = FRAME_USER_FIELD_OFFSET + frame_value.slot;
        auto *field = frame_field_ptr(b, mod, frame_arg, frame_value.type, field_index);
        auto *loaded = b.load(frame_value.type, field);
        if (is_memory_frame_value(frame_value.value)) {
            auto *pointer = frame_value_memory_pointer(
                b, mod, frame_value, resolver);
            b.store(pointer, loaded);
        } else {
            resolver.map_entry_value(frame_value.value, loaded);
        }
    }
}

static void clone_scope(Module *mod, const CoroCfgDistillResult::Scope &scope,
                        CallableFunction *new_func, Value *frame_arg,
                        const CoroCfgDistillResult &result,
                        CoroSplitValueResolver &resolver) noexcept {

    luisa::unordered_set<const BasicBlock *> scope_block_set;
    for (auto *bb : scope.blocks) {
        scope_block_set.insert(bb);
    }
    resolver.set_scope(scope.blocks.front(), scope_block_set);

    luisa::unordered_map<const BasicBlock *, size_t> block_to_scope_index;
    for (size_t i = 0u; i < result.scopes.size(); ++i) {
        auto &other_scope = result.scopes[i];
        for (auto *bb : other_scope.blocks) {
            block_to_scope_index.emplace(bb, i);
        }
    }

    luisa::unordered_map<const BasicBlock *, luisa::unordered_map<const BasicBlock *, BasicBlock *>> fallback_returns;

    XIRBuilder b;

    luisa::vector<BasicBlock *> clone_order;
    clone_order.reserve(scope.blocks.size());
    luisa::unordered_set<BasicBlock *> visited_blocks;
    luisa::vector<BasicBlock *> block_worklist{scope.blocks.front()};
    while (!block_worklist.empty()) {
        auto *block = block_worklist.back();
        block_worklist.pop_back();
        if (block == nullptr || !scope_block_set.contains(block) ||
            !visited_blocks.emplace(block).second) {
            continue;
        }
        clone_order.emplace_back(block);
        luisa::vector<BasicBlock *> successors;
        block->traverse_successors(false, [&](BasicBlock *successor) noexcept {
            if (scope_block_set.contains(successor) && !visited_blocks.contains(successor)) {
                successors.emplace_back(successor);
            }
        });
        for (auto iter = successors.rbegin(); iter != successors.rend(); ++iter) {
            block_worklist.emplace_back(*iter);
        }
    }
    for (auto *block : scope.blocks) {
        if (visited_blocks.emplace(block).second) { clone_order.emplace_back(block); }
    }

    auto *first_cloned_bb = static_cast<BasicBlock *>(resolver.resolve(scope.blocks.front()));
    resolver.set_builder(&b, first_cloned_bb);

    for (auto *orig_bb : clone_order) {
        auto *cloned_bb = static_cast<BasicBlock *>(resolver.resolve(orig_bb));
        resolver.set_current_original_block(orig_bb);

        b.set_insertion_point(cloned_bb);
        for (auto *inst : orig_bb->instructions()) {
            if (inst->derived_instruction_tag() == DerivedInstructionTag::ALLOCA) {
                if (resolver.has_value(inst)) { continue; }
                auto *orig_alloca = static_cast<const AllocaInst *>(inst);
                auto *cloned_alloca = b.alloca_(orig_alloca->type(), orig_alloca->op());
                coro_split_clone_metadata(*orig_alloca, *cloned_alloca);
                resolver.map_value(inst, cloned_alloca);
            }
        }
    }

    // An alloca carried by the frame is a memory identity, not an SSA value.
    // Materialize every such identity before emitting any frame reloads. Lazy
    // creation from resolve() is unsound here: restoring the builder's old
    // insertion point can place the reload store before the newly created
    // alloca in the continuation entry block.
    b.set_insertion_point(first_cloned_bb);
    for (auto &frame_value : result.frame_values) {
        if (!is_memory_frame_value(frame_value.value) ||
            resolver.has_value(frame_value.value)) {
            continue;
        }
        auto *orig_alloca =
            static_cast<const AllocaInst *>(frame_value.value);
        auto *cloned_alloca =
            b.alloca_(orig_alloca->type(), orig_alloca->op());
        coro_split_clone_metadata(*orig_alloca, *cloned_alloca);
        resolver.map_value(orig_alloca, cloned_alloca);
    }

    b.set_insertion_point(first_cloned_bb);
    load_live_values_from_frame(
        b, mod, frame_arg, result, scope, resolver);

    auto resolve_branch_target = [&](const BasicBlock *source, BasicBlock *target) noexcept -> BasicBlock * {
        if (target == nullptr) { return nullptr; }
        if (scope_block_set.contains(target)) {
            return static_cast<BasicBlock *>(resolver.resolve(target));
        }
        auto &by_target = fallback_returns[source];
        if (auto it = by_target.find(target); it != by_target.end()) {
            return it->second;
        }
        auto *fb = new_func->create_basic_block();
        XIRBuilder fb_builder;
        fb_builder.set_insertion_point(fb);
        if (auto target_scope = block_to_scope_index.find(target);
            target_scope != block_to_scope_index.end()) {
            auto values = store_values_for_branch_transition(
                result, static_cast<size_t>(scope.scope_id), source, target_scope->second);
            store_live_values_to_frame(
                fb_builder, mod, frame_arg, result, values, resolver);
            store_frame_token(fb_builder, frame_arg, mod, result.scopes[target_scope->second].trigger_token);
        } else {
            store_live_values_to_frame(fb_builder, mod, frame_arg, result,
                                       luisa::span<const size_t>{
                                           scope.live_out_frame_value_indices},
                                       resolver);
            store_frame_token(fb_builder, frame_arg, mod, TERMINAL_TOKEN);
        }
        fb_builder.return_void();
        by_target.emplace(target, fb);
        return fb;
    };

    for (auto *orig_bb : clone_order) {
        auto *cloned_bb = static_cast<BasicBlock *>(resolver.resolve(orig_bb));
        resolver.set_current_original_block(orig_bb);
        for (auto *inst : orig_bb->instructions()) {
            auto tag = inst->derived_instruction_tag();

            switch (tag) {
                case DerivedInstructionTag::ALLOCA: {
                    break;// already cloned in pre-scan
                }
                case DerivedInstructionTag::CORO_SUSPEND: {
                    auto *s = static_cast<CoroSuspendInst *>(inst);
                    b.set_insertion_point(cloned_bb);
                    auto values = store_values_for_suspend(result, static_cast<size_t>(scope.scope_id), s->token());
                    store_live_values_to_frame(
                        b, mod, frame_arg, result, values, resolver);
                    store_frame_token(b, frame_arg, mod, s->token());
                    auto *cloned = b.return_void();
                    coro_split_clone_instruction_metadata(inst, cloned);
                    goto block_terminated;
                }
                case DerivedInstructionTag::CORO_TERMINATE: {
                    b.set_insertion_point(cloned_bb);
                    store_live_values_to_frame(b, mod, frame_arg, result,
                                               luisa::span<const size_t>{
                                                   scope.live_out_frame_value_indices},
                                               resolver);
                    store_frame_token(b, frame_arg, mod, TERMINAL_TOKEN);
                    auto *cloned = b.return_void();
                    coro_split_clone_instruction_metadata(inst, cloned);
                    goto block_terminated;
                }
                case DerivedInstructionTag::CORO_RESUME: {
                    auto *r = static_cast<CoroResumeInst *>(inst);
                    b.set_insertion_point(cloned_bb);
                    auto *cloned = b.coro_resume(r->token(), frame_arg);
                    coro_split_clone_instruction_metadata(inst, cloned);
                    resolver.map_value(inst, cloned);
                    break;
                }
                case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                    auto *cbr = static_cast<ConditionalBranchInst *>(inst);
                    auto *cond = resolver.resolve(cbr->condition());
                    auto *true_block = resolve_branch_target(orig_bb, cbr->true_block());
                    auto *false_block = resolve_branch_target(orig_bb, cbr->false_block());
                    b.set_insertion_point(cloned_bb);
                    Instruction *cloned = true_block == false_block ?
                                              static_cast<Instruction *>(b.br(true_block)) :
                                              static_cast<Instruction *>(b.cond_br(cond, true_block, false_block));
                    coro_split_clone_instruction_metadata(inst, cloned);
                    resolver.map_value(inst, cloned);
                    break;
                }
                case DerivedInstructionTag::BRANCH: {
                    auto *br = static_cast<BranchInst *>(inst);
                    auto *target = resolve_branch_target(orig_bb, br->target_block());
                    b.set_insertion_point(cloned_bb);
                    auto *cloned = b.br(target);
                    coro_split_clone_instruction_metadata(inst, cloned);
                    resolver.map_value(inst, cloned);
                    break;
                }
                case DerivedInstructionTag::INDEXED_BRANCH: {
                    auto *indexed_branch =
                        static_cast<IndexedBranchInst *>(inst);
                    auto *value =
                        resolver.resolve(indexed_branch->value());
                    auto *default_block = resolve_branch_target(
                        orig_bb, indexed_branch->default_block());
                    b.set_insertion_point(cloned_bb);
                    auto *cloned = b.indexed_branch(value);
                    cloned->set_default_block(default_block);
                    for (size_t i = 0u;
                         i < indexed_branch->case_count(); ++i) {
                        cloned->add_case(
                            indexed_branch->case_value(i),
                            resolve_branch_target(
                                orig_bb,
                                indexed_branch->case_block(i)));
                    }
                    for (auto *metadata :
                         indexed_branch->metadata_list()) {
                        cloned->metadata_list().push_front(
                            metadata->clone());
                    }
                    resolver.map_value(inst, cloned);
                    break;
                }
                default: {
                    b.set_insertion_point(cloned_bb);
                    auto *cloned = inst->clone_with_metadata(b, resolver);
                    LUISA_DEBUG_ASSERT(cloned != nullptr, "Failed to clone instruction.");
                    resolver.map_value(inst, cloned);
                    break;
                }
            }
        }
    block_terminated:;
    }
}

static void instrument_terminal_returns(Module *mod, const CoroCfgDistillResult::Scope &scope,
                                        Value *frame_arg, const CoroCfgDistillResult &result,
                                        CoroSplitValueResolver &resolver) noexcept {
    XIRBuilder b;
    for (auto *orig_bb : scope.blocks) {
        auto *cloned_bb = static_cast<BasicBlock *>(resolver.resolve(orig_bb));
        if (!cloned_bb->is_terminated()) { continue; }
        auto *term = cloned_bb->terminator();
        if (term != nullptr && term->derived_instruction_tag() == DerivedInstructionTag::RETURN) {
            auto *orig_term = orig_bb->terminator();
            bool was_suspend = (orig_term != nullptr &&
                                orig_term->derived_instruction_tag() == DerivedInstructionTag::CORO_SUSPEND);
            bool was_terminal = (orig_term != nullptr &&
                                 orig_term->derived_instruction_tag() ==
                                     DerivedInstructionTag::CORO_TERMINATE);
            b.set_insertion_point(term->prev());
            if (!was_suspend && !was_terminal) {
                store_live_values_to_frame(b, mod, frame_arg, result,
                                           luisa::span<const size_t>{
                                               scope.live_out_frame_value_indices},
                                           resolver);
                store_frame_token(b, frame_arg, mod, TERMINAL_TOKEN);
            }
        }
    }
}

[[nodiscard]] static CoroSplitInfo split_function_with_cfg_info(
    Module *mod, FunctionDefinition *def,
    const CoroCfgDistillResult &result,
    const Type *frame_type = nullptr) noexcept {
    CoroSplitInfo info;
    if (mod == nullptr || def == nullptr) { return info; }
    if (contains_structured_control_flow(def)) {
        info.structured_cfg_error_count = 1u;
        LUISA_WARNING_WITH_LOCATION(
            "Coro split rejected structured or ambiguous CFG; run "
            "destructure_cfg first. IR was left unchanged.");
        return info;
    }
    if (!validate_distilled_cfg(def, result)) {
        info.invalid_cfg_error_count = 1u;
        LUISA_WARNING_WITH_LOCATION(
            "Coro split rejected invalid coroutine tokens or distilled CFG metadata. "
            "IR was left unchanged.");
        return info;
    }
    // A coroutine whose every suspend was removed by CFG optimization still
    // has one executable entry scope. Lower that scope to a normal
    // continuation callable as the degenerate |T_live| = 0 case; returning no
    // subroutine here would make the front-end coroutine lose its entry and
    // would incorrectly require every original suspend token to survive.
    auto *actual_frame_type = frame_type ? frame_type : create_frame_type(result);
    if (!validate_frame_type(actual_frame_type, result)) {
        info.invalid_cfg_error_count = 1u;
        LUISA_WARNING_WITH_LOCATION(
            "Coro split rejected a frame type that does not match the distilled frame layout. "
            "IR was left unchanged.");
        return info;
    }
    info.subroutines.reserve(result.scopes.size());
    for (size_t i = 0; i < result.scopes.size(); ++i) {
        auto &scope = result.scopes[i];

        auto *new_func = mod->create_callable(nullptr);
        coro_split_clone_metadata(*def, *new_func);
        auto *frame_arg = new_func->create_reference_argument(actual_frame_type);

        CoroSplitValueResolver resolver;

        for (auto *orig_arg : def->arguments()) {
            auto *cloned_arg = new_func->create_argument(orig_arg->type(), orig_arg->is_lvalue());
            coro_split_clone_metadata(*orig_arg, *cloned_arg);
            resolver.map_arg(orig_arg, cloned_arg);
        }

        for (auto *orig_bb : scope.blocks) {
            auto *cloned_bb = new_func->create_basic_block();
            coro_split_clone_metadata(*orig_bb, *cloned_bb);
            resolver.map_block(orig_bb, cloned_bb);
        }

        auto *body_entry = static_cast<BasicBlock *>(resolver.resolve(scope.blocks.front()));
        resolver.set_frame_arg(mod, frame_arg);

        new_func->set_body_block(body_entry);

        clone_scope(mod, scope, new_func, frame_arg, result, resolver);

        instrument_terminal_returns(
            mod, scope, frame_arg, result, resolver);

        info.subroutines.emplace_back(CoroSplitInfo::Subroutine{
            .scope_index = i,
            .trigger_token = scope.trigger_token,
            .trigger_name = scope.trigger_name,
            .callable = new_func,
            .frame_argument = frame_arg,
        });
    }

    return info;
}

[[nodiscard]] static CoroSplitInfo split_function(Module *mod, FunctionDefinition *def) noexcept {
    auto result = coro_cfg_distill_pass_run_on_function(def);
    return split_function_with_cfg_info(mod, def, result);
}

[[nodiscard]] static bool has_coroutine_instruction(FunctionDefinition *def) noexcept {
    if (def == nullptr) { return false; }
    for (auto *block : def->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::CORO_SUSPEND:
                case DerivedInstructionTag::CORO_RESUME:
                case DerivedInstructionTag::CORO_TERMINATE:
                    return true;
                default: break;
            }
        }
    }
    return false;
}

static void append_split_info(CoroSplitInfo &dst, CoroSplitInfo src) noexcept {
    dst.structured_cfg_error_count += src.structured_cfg_error_count;
    dst.invalid_cfg_error_count += src.invalid_cfg_error_count;
    dst.subroutines.reserve(dst.subroutines.size() + src.subroutines.size());
    for (auto &subroutine : src.subroutines) {
        dst.subroutines.emplace_back(std::move(subroutine));
    }
}

struct CfgOwner {
    FunctionDefinition *definition{nullptr};
    bool valid{false};
};

[[nodiscard]] static CfgOwner find_cfg_owner(Module *module,
                                             const CoroCfgDistillResult &cfg) noexcept {
    FunctionDefinition *owner = nullptr;
    if (module == nullptr || cfg.scopes.empty()) { return {}; }
    for (auto &scope : cfg.scopes) {
        if (scope.blocks.empty()) { return {}; }
        for (auto *block : scope.blocks) {
            if (block == nullptr) { return {}; }
            auto *parent = block->parent_function();
            if (parent == nullptr || !parent->is_definition()) { return {}; }
            auto *definition = static_cast<FunctionDefinition *>(parent);
            if (definition->parent_module() != module) { return {}; }
            if (owner != nullptr && owner != definition) { return {}; }
            owner = definition;
        }
    }
    return {.definition = owner, .valid = owner != nullptr};
}

}// namespace detail

CoroSplitInfo coro_split_pass_run_on_module_info(Module *m) noexcept {
    CoroSplitInfo info;
    if (m == nullptr) { return info; }
    luisa::vector<FunctionDefinition *> defs;
    for (auto *f : m->function_list()) {
        if (f->is_definition()) {
            auto *def = static_cast<FunctionDefinition *>(f);
            if (detail::has_coroutine_instruction(def)) {
                defs.push_back(def);
            }
        }
    }
    luisa::vector<CoroCfgDistillResult> cfgs;
    cfgs.reserve(defs.size());
    for (auto *def : defs) {
        if (contains_structured_control_flow(def)) {
            ++info.structured_cfg_error_count;
            cfgs.emplace_back();
            continue;
        }
        cfgs.emplace_back(coro_cfg_distill_pass_run_on_function(def));
        if (!cfgs.back().succeeded() ||
            !detail::coro_split_validate_coroutine_tokens(def) ||
            !detail::validate_distilled_cfg(def, cfgs.back())) {
            ++info.invalid_cfg_error_count;
        }
    }
    if (!info.succeeded()) {
        if (info.structured_cfg_error_count != 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "Coro split rejected {} coroutine definition(s) with structured or "
                "ambiguous CFG; run destructure_cfg first. "
                "The module was left unchanged.",
                info.structured_cfg_error_count);
        }
        if (info.invalid_cfg_error_count != 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "Coro split rejected {} coroutine definition(s) with invalid tokens or CFG metadata. "
                "The module was left unchanged.",
                info.invalid_cfg_error_count);
        }
        return info;
    }
    for (size_t i = 0u; i < defs.size(); ++i) {
        detail::append_split_info(info, detail::split_function_with_cfg_info(m, defs[i], cfgs[i]));
    }
    return info;
}

size_t coro_split_pass_run_on_module(Module *m) noexcept {
    return coro_split_pass_run_on_module_info(m).subroutines.size();
}

size_t coro_split_pass_run_on_module_with_cfg(
    Module *m, const CoroCfgDistillResult &cfg) noexcept {
    return coro_split_pass_run_on_module_with_cfg_and_frame_info(m, cfg, nullptr)
        .subroutines.size();
}

size_t coro_split_pass_run_on_module_with_cfg_and_frame(
    Module *m, const CoroCfgDistillResult &cfg, const Type *frame_type) noexcept {
    return coro_split_pass_run_on_module_with_cfg_and_frame_info(m, cfg, frame_type)
        .subroutines.size();
}

CoroSplitInfo coro_split_pass_run_on_module_with_cfg_and_frame_info(
    Module *m, const CoroCfgDistillResult &cfg, const Type *frame_type) noexcept {
    auto owner = detail::find_cfg_owner(m, cfg);
    if (!owner.valid) {
        LUISA_WARNING_WITH_LOCATION(
            "Coro split rejected an invalid or cross-function distilled CFG. "
            "IR was left unchanged.");
        CoroSplitInfo info;
        info.invalid_cfg_error_count = 1u;
        return info;
    }
    return detail::split_function_with_cfg_info(m, owner.definition, cfg, frame_type);
}

}// namespace luisa::compute::xir
