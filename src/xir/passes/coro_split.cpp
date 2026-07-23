#include <algorithm>

#include "helpers.h"

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
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/op.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/special_register.h>

namespace luisa::compute::xir {

namespace detail {

static constexpr uint32_t FRAME_FIELD_ID_X = 0u;
static constexpr uint32_t FRAME_FIELD_ID_Y = 1u;
static constexpr uint32_t FRAME_FIELD_ID_Z = 2u;
static constexpr uint32_t FRAME_FIELD_SIZE_X = 3u;
static constexpr uint32_t FRAME_FIELD_SIZE_Y = 4u;
static constexpr uint32_t FRAME_FIELD_SIZE_Z = 5u;
static constexpr uint32_t FRAME_FIELD_TOKEN = 6u;
static constexpr uint32_t FRAME_USER_FIELD_OFFSET = 7u;

class CoroSplitValueResolver final : public InstructionCloneValueResolver {

private:
    luisa::unordered_map<const Value *, Value *> _value_map;
    luisa::unordered_map<const Value *, Value *> _entry_value_map;
    luisa::unordered_map<const BasicBlock *, BasicBlock *> _block_map;
    luisa::unordered_map<const Argument *, Argument *> _arg_map;
    luisa::unordered_set<const BasicBlock *> _scope_blocks;
    XIRBuilder *_builder{nullptr};
    BasicBlock *_alloca_bb{nullptr};
    Value *_frame_arg{nullptr};
    Module *_module{nullptr};
    const BasicBlock *_scope_root{nullptr};
    const BasicBlock *_current_orig_block{nullptr};

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
                if (inst->derived_instruction_tag() == DerivedInstructionTag::ALLOCA &&
                    _builder != nullptr && _alloca_bb != nullptr) {
                    auto *orig_alloca = static_cast<const AllocaInst *>(inst);
                    auto *prev_ip = _builder->insertion_point();
                    _builder->set_insertion_point(_alloca_bb);
                    auto *cloned = _builder->alloca_(orig_alloca->type(), orig_alloca->op());
                    auto name_opt = orig_alloca->name();
                    if (name_opt.has_value()) { cloned->set_name(name_opt.value()); }
                    _builder->set_insertion_point(prev_ip);
                    _value_map.emplace(inst, cloned);
                    return cloned;
                }
                if (_builder != nullptr) {
                    switch (inst->derived_instruction_tag()) {
                        case DerivedInstructionTag::GEP:
                        case DerivedInstructionTag::ARITHMETIC:
                        case DerivedInstructionTag::CAST:
                        case DerivedInstructionTag::RESOURCE_QUERY: {
                            auto *cloned = inst->clone_with_metadata(*_builder, *this);
                            _value_map.emplace(inst, cloned);
                            return cloned;
                        }
                        default:
                            break;
                    }
                }
                LUISA_DEBUG_ASSERT(false, "Instruction not found in resolver: {}.", to_string(inst->derived_instruction_tag()));
                return nullptr;
            }
        }
    }
};

[[nodiscard]] static const Type *create_frame_type(const CoroCfgDistillResult &result) noexcept {
    luisa::vector<const Type *> fields;
    fields.reserve(FRAME_USER_FIELD_OFFSET + result.frame_values.size());
    auto alignment = Type::of<uint>()->alignment();
    for (auto i = 0u; i < FRAME_USER_FIELD_OFFSET; ++i) {
        fields.emplace_back(Type::of<uint>());
    }
    for (auto &value : result.frame_values) {
        fields.emplace_back(value.type);
        alignment = std::max(alignment, value.type->alignment());
    }
    return Type::structure(alignment, fields);
}

[[nodiscard]] static bool validate_frame_type(const Type *frame_type,
                                              const CoroCfgDistillResult &result) noexcept {
    if (frame_type == nullptr || !frame_type->is_structure()) { return false; }
    auto members = frame_type->members();
    if (members.size() != FRAME_USER_FIELD_OFFSET + result.frame_values.size()) { return false; }
    for (auto i = 0u; i < FRAME_USER_FIELD_OFFSET; ++i) {
        if (members[i] != Type::of<uint>()) { return false; }
    }
    for (size_t i = 0u; i < result.frame_values.size(); ++i) {
        if (result.frame_values[i].type == nullptr ||
            members[FRAME_USER_FIELD_OFFSET + i] != result.frame_values[i].type) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] static bool validate_coroutine_tokens(FunctionDefinition *def) noexcept {
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

[[nodiscard]] static bool validate_distilled_cfg(FunctionDefinition *def,
                                                 const CoroCfgDistillResult &result) noexcept {
    if (def == nullptr || result.scopes.empty() || result.edges.size() != result.scopes.size()) { return false; }
    if (!validate_coroutine_tokens(def)) { return false; }
    auto canonical = coro_cfg_distill_pass_run_on_function(def);
    if (canonical.scopes.size() != result.scopes.size()) { return false; }
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
        if (scope_blocks.size() != canonical.scopes[i].blocks.size()) { return false; }
        for (auto *block : canonical.scopes[i].blocks) {
            if (!scope_blocks.contains(block)) { return false; }
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
    luisa::unordered_set<const Value *> values;
    for (auto &frame_value : result.frame_values) {
        if (frame_value.value == nullptr || frame_value.type == nullptr ||
            frame_value.value->type() != frame_value.type ||
            !values.emplace(frame_value.value).second) {
            return false;
        }
        if (frame_value.value->isa<Instruction>()) {
            auto *inst = static_cast<Instruction *>(frame_value.value);
            if (inst->parent_block() == nullptr || inst->parent_block()->parent_function() != def) {
                return false;
            }
        }
    }
    for (auto &edge : result.transition_edges) {
        if (edge.from_scope >= result.scopes.size() || edge.to_scope >= result.scopes.size() ||
            edge.token != result.scopes[edge.to_scope].trigger_token) {
            return false;
        }
        for (auto *value : edge.live_values) {
            if (!values.contains(value)) { return false; }
        }
        for (auto *value : edge.store_values) {
            if (!values.contains(value)) { return false; }
        }
    }
    return true;
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

[[nodiscard]] static Value *frame_field_ptr(XIRBuilder &b, Module *mod, Value *frame_arg,
                                            const Type *type, size_t field_index) noexcept {
    auto i = static_cast<uint32_t>(field_index);
    auto *idx = mod->create_constant(Type::of<uint32_t>(), &i);
    return b.gep(type, frame_arg, {idx});
}

static void store_live_values_to_frame(XIRBuilder &b, Module *mod, Value *frame_arg,
                                       const CoroCfgDistillResult &result,
                                       luisa::span<Value *const> values,
                                       const luisa::unordered_map<const Value *, size_t> &field_indices,
                                       CoroSplitValueResolver &resolver) noexcept {
    for (auto *value : values) {
        auto it = field_indices.find(value);
        if (it == field_indices.end()) { continue; }
        auto &frame_value = result.frame_values[it->second];
        auto field_index = FRAME_USER_FIELD_OFFSET + it->second;
        auto *field = frame_field_ptr(b, mod, frame_arg, frame_value.type, field_index);
        auto *cloned = resolver.resolve(value);
        if (is_memory_frame_value(value)) {
            auto *loaded = b.load(frame_value.type, cloned);
            b.store(field, loaded);
        } else {
            b.store(field, cloned);
        }
    }
}

[[nodiscard]] static luisa::span<Value *const> store_values_for_suspend(
    const CoroCfgDistillResult &result, size_t scope_index, uint32_t token) noexcept {
    for (auto &edge : result.transition_edges) {
        if (edge.is_suspend && edge.from_scope == scope_index && edge.token == token) {
            return luisa::span{edge.store_values};
        }
    }
    return {};
}

[[nodiscard]] static luisa::span<Value *const> store_values_for_branch_transition(
    const CoroCfgDistillResult &result, size_t scope_index,
    const BasicBlock *exit_block, size_t target_scope) noexcept {
    for (auto &edge : result.transition_edges) {
        if (!edge.is_suspend &&
            edge.from_scope == scope_index &&
            edge.to_scope == target_scope &&
            edge.exit_block == exit_block) {
            return luisa::span{edge.store_values};
        }
    }
    return {};
}

static void load_live_values_from_frame(XIRBuilder &b, Module *mod, Value *frame_arg,
                                        const CoroCfgDistillResult &result,
                                        const CoroCfgDistillResult::Scope &scope,
                                        const luisa::unordered_map<const Value *, size_t> &field_indices,
                                        CoroSplitValueResolver &resolver) noexcept {
    for (auto *value : scope.live_in_values) {
        auto it = field_indices.find(value);
        if (it == field_indices.end()) { continue; }
        auto &frame_value = result.frame_values[it->second];
        auto field_index = FRAME_USER_FIELD_OFFSET + it->second;
        auto *field = frame_field_ptr(b, mod, frame_arg, frame_value.type, field_index);
        auto *loaded = b.load(frame_value.type, field);
        if (is_memory_frame_value(value)) {
            auto *cloned = resolver.resolve(value);
            b.store(cloned, loaded);
        } else {
            resolver.map_entry_value(value, loaded);
        }
    }
}

static void clone_scope(Module *mod, const CoroCfgDistillResult::Scope &scope,
                        CallableFunction *new_func, Value *frame_arg,
                        const CoroCfgDistillResult &result,
                        const luisa::unordered_map<const Value *, size_t> &field_indices,
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
                auto name_opt = orig_alloca->name();
                if (name_opt.has_value()) { cloned_alloca->set_name(name_opt.value()); }
                resolver.map_value(inst, cloned_alloca);
            }
        }
    }

    b.set_insertion_point(first_cloned_bb);
    load_live_values_from_frame(b, mod, frame_arg, result, scope, field_indices, resolver);

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
            store_live_values_to_frame(fb_builder, mod, frame_arg, result, values, field_indices, resolver);
            store_frame_token(fb_builder, frame_arg, mod, result.scopes[target_scope->second].trigger_token);
        } else {
            store_live_values_to_frame(fb_builder, mod, frame_arg, result,
                                       luisa::span{scope.live_out_values},
                                       field_indices, resolver);
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
                    store_live_values_to_frame(b, mod, frame_arg, result, values, field_indices, resolver);
                    store_frame_token(b, frame_arg, mod, s->token());
                    b.return_void();
                    goto block_terminated;
                }
                case DerivedInstructionTag::CORO_TERMINATE: {
                    b.set_insertion_point(cloned_bb);
                    store_live_values_to_frame(b, mod, frame_arg, result,
                                               luisa::span{scope.live_out_values},
                                               field_indices, resolver);
                    store_frame_token(b, frame_arg, mod, TERMINAL_TOKEN);
                    b.return_void();
                    goto block_terminated;
                }
                case DerivedInstructionTag::CORO_RESUME: {
                    auto *r = static_cast<CoroResumeInst *>(inst);
                    b.set_insertion_point(cloned_bb);
                    auto *cloned = b.coro_resume(r->token(), frame_arg);
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
                    resolver.map_value(inst, cloned);
                    break;
                }
                case DerivedInstructionTag::BRANCH: {
                    auto *br = static_cast<BranchInst *>(inst);
                    auto *target = resolve_branch_target(orig_bb, br->target_block());
                    b.set_insertion_point(cloned_bb);
                    auto *cloned = b.br(target);
                    resolver.map_value(inst, cloned);
                    break;
                }
                case DerivedInstructionTag::SWITCH: {
                    auto *sw = static_cast<SwitchInst *>(inst);
                    auto *value = resolver.resolve(sw->value());
                    auto *default_block = resolve_branch_target(orig_bb, sw->default_block());
                    b.set_insertion_point(cloned_bb);
                    auto *cloned = b.switch_(value);
                    cloned->set_default_block(default_block);
                    for (size_t i = 0u; i < sw->case_count(); ++i) {
                        cloned->add_case(sw->case_value(i), resolve_branch_target(orig_bb, sw->case_block(i)));
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
                                        const luisa::unordered_map<const Value *, size_t> &field_indices,
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
            b.set_insertion_point(term->prev());
            if (!was_suspend) {
                store_live_values_to_frame(b, mod, frame_arg, result,
                                           luisa::span{scope.live_out_values},
                                           field_indices, resolver);
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
            "Coro split rejected structured or ambiguous CFG; run lower_switch "
            "followed by destructure_cfg first. IR was left unchanged.");
        return info;
    }
    if (!validate_distilled_cfg(def, result)) {
        info.invalid_cfg_error_count = 1u;
        LUISA_WARNING_WITH_LOCATION(
            "Coro split rejected invalid coroutine tokens or distilled CFG metadata. "
            "IR was left unchanged.");
        return info;
    }
    if (result.scopes.size() <= 1u) { return info; }

    auto *actual_frame_type = frame_type ? frame_type : create_frame_type(result);
    if (!validate_frame_type(actual_frame_type, result)) {
        info.invalid_cfg_error_count = 1u;
        LUISA_WARNING_WITH_LOCATION(
            "Coro split rejected a frame type that does not match the distilled frame layout. "
            "IR was left unchanged.");
        return info;
    }
    luisa::unordered_map<const Value *, size_t> frame_value_indices;
    for (size_t i = 0u; i < result.frame_values.size(); ++i) {
        frame_value_indices.emplace(result.frame_values[i].value, i);
    }

    info.subroutines.reserve(result.scopes.size());
    for (size_t i = 0; i < result.scopes.size(); ++i) {
        auto &scope = result.scopes[i];

        auto *new_func = mod->create_callable(nullptr);
        auto *frame_arg = new_func->create_reference_argument(actual_frame_type);

        CoroSplitValueResolver resolver;

        for (auto *orig_arg : def->arguments()) {
            auto *cloned_arg = new_func->create_argument(orig_arg->type(), orig_arg->is_lvalue());
            resolver.map_arg(orig_arg, cloned_arg);
        }

        for (auto *orig_bb : scope.blocks) {
            auto *cloned_bb = new_func->create_basic_block();
            resolver.map_block(orig_bb, cloned_bb);
        }

        auto *body_entry = static_cast<BasicBlock *>(resolver.resolve(scope.blocks.front()));
        resolver.set_frame_arg(mod, frame_arg);

        new_func->set_body_block(body_entry);

        clone_scope(mod, scope, new_func, frame_arg, result, frame_value_indices, resolver);

        instrument_terminal_returns(mod, scope, frame_arg, result, frame_value_indices, resolver);

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
        }
        cfgs.emplace_back(coro_cfg_distill_pass_run_on_function(def));
        if (!detail::validate_coroutine_tokens(def) ||
            !detail::validate_distilled_cfg(def, cfgs.back())) {
            ++info.invalid_cfg_error_count;
        }
    }
    if (!info.succeeded()) {
        if (info.structured_cfg_error_count != 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "Coro split rejected {} coroutine definition(s) with structured or "
                "ambiguous CFG; run lower_switch followed by destructure_cfg first. "
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
