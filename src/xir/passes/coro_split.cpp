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

namespace luisa::compute::xir {

namespace detail {

static constexpr uint32_t FRAME_FIELD_TOKEN = 0u;
static constexpr uint32_t FRAME_FIELD_SKIP_FLAG = 1u;
static constexpr uint32_t SKIP_FLAG_TRUE = 1u;

class CoroSplitValueResolver final : public InstructionCloneValueResolver {

private:
    luisa::unordered_map<const Value *, Value *> _value_map;
    luisa::unordered_map<const Value *, Value *> _entry_value_map;
    luisa::unordered_map<const BasicBlock *, BasicBlock *> _block_map;
    luisa::unordered_map<const Argument *, Argument *> _arg_map;
    luisa::unordered_set<const BasicBlock *> _scope_blocks;
    XIRBuilder *_builder{nullptr};
    BasicBlock *_alloca_bb{nullptr};
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
            case DerivedValueTag::SPECIAL_REGISTER:
                return const_cast<Value *>(value);
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

[[nodiscard]] static const Type *create_frame_type() noexcept {
    return Type::structure({Type::of<uint>(), Type::of<uint>()});
}

static void store_frame_token(XIRBuilder &b, Value *frame_arg, Module *mod, uint32_t token) noexcept {
    auto *field_zero = mod->create_constant_zero(Type::of<uint>());
    auto *gep = b.gep(Type::of<uint>(), frame_arg, {field_zero});
    auto *tok_const = mod->create_constant(Type::of<uint>(), &token);
    b.store(gep, tok_const);
}

static void store_skip_flag_true(XIRBuilder &b, Value *frame_arg, Module *mod) noexcept {
    auto *field_one = mod->create_constant(Type::of<uint>(), &FRAME_FIELD_SKIP_FLAG);
    auto *gep = b.gep(Type::of<uint>(), frame_arg, {field_one});
    auto *flag_true = mod->create_constant(Type::of<uint>(), &SKIP_FLAG_TRUE);
    b.store(gep, flag_true);
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
        auto field_index = FRAME_FIELD_SKIP_FLAG + 1u + it->second;
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
        if (edge.from_scope == scope_index && edge.token == token) {
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
        auto field_index = FRAME_FIELD_SKIP_FLAG + 1u + it->second;
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

static void build_skip_check_entry(Module *mod, CallableFunction *func,
                                   Value *frame_arg, BasicBlock *body_entry,
                                   bool check_token = false) noexcept {
    auto *check_block = func->create_basic_block();
    auto *ret_block = func->create_basic_block();
    func->set_body_block(check_block);

    XIRBuilder b;
    b.set_insertion_point(check_block);

    if (check_token) {
        auto *field_zero = mod->create_constant_zero(Type::of<uint>());
        auto *gep0 = b.gep(Type::of<uint>(), frame_arg, {field_zero});
        auto *loaded_token = b.load(Type::of<uint>(), gep0);
        auto *zero = mod->create_constant_zero(Type::of<uint>());
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL, {loaded_token, zero});
        b.cond_br(cond, ret_block, body_entry);
    } else {
        auto *field_one = mod->create_constant(Type::of<uint>(), &FRAME_FIELD_SKIP_FLAG);
        auto *gep = b.gep(Type::of<uint>(), frame_arg, {field_one});
        auto *loaded_flag = b.load(Type::of<uint>(), gep);
        auto *zero = mod->create_constant_zero(Type::of<uint>());
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL, {loaded_flag, zero});
        b.cond_br(cond, ret_block, body_entry);
    }

    b.set_insertion_point(ret_block);
    b.return_void();
}

static void clone_scope(Module *mod, const CoroCfgDistillResult::Scope &scope,
                        CallableFunction *new_func, Value *frame_arg,
                        const CoroCfgDistillResult &result,
                        const luisa::unordered_map<const Value *, size_t> &field_indices,
                        CoroSplitValueResolver &resolver) noexcept {

    // pre-scan: create fallback return blocks for cross-scope branch targets
    luisa::unordered_set<const BasicBlock *> scope_block_set;
    for (auto *bb : scope.blocks) {
        scope_block_set.insert(bb);
    }
    resolver.set_scope(scope.blocks.front(), scope_block_set);

    luisa::unordered_map<const BasicBlock *, BasicBlock *> fallback_returns;

    auto get_or_create_fallback = [&](const BasicBlock *target) -> BasicBlock * {
        if (scope_block_set.contains(target)) { return nullptr; }
        auto it = fallback_returns.find(target);
        if (it != fallback_returns.end()) { return it->second; }
        auto *fb = new_func->create_basic_block();
        XIRBuilder fb_builder;
        fb_builder.set_insertion_point(fb);
        fb_builder.return_void();
        fallback_returns.emplace(target, fb);
        resolver.map_block(target, fb);
        return fb;
    };

    // pre-scan all branch instructions for cross-scope targets
    for (auto *orig_bb : scope.blocks) {
        auto *term = orig_bb->terminator();
        if (term == nullptr) { continue; }
        auto tag = term->derived_instruction_tag();
        if (tag == DerivedInstructionTag::BRANCH) {
            auto *br = static_cast<const BranchInst *>(term);
            get_or_create_fallback(br->target_block());
        } else if (tag == DerivedInstructionTag::CONDITIONAL_BRANCH) {
            auto *cbr = static_cast<const ConditionalBranchInst *>(term);
            get_or_create_fallback(cbr->true_block());
            get_or_create_fallback(cbr->false_block());
        } else if (tag == DerivedInstructionTag::SWITCH) {
            auto *sw = static_cast<const SwitchInst *>(term);
            get_or_create_fallback(sw->default_block());
            for (size_t i = 0u; i < sw->case_count(); ++i) {
                get_or_create_fallback(sw->case_block(i));
            }
        }
    }

    XIRBuilder b;

    auto clone_order = scope.blocks;
    luisa::unordered_map<const BasicBlock *, size_t> block_order;
    if (!scope.blocks.empty()) {
        if (auto *parent = scope.blocks.front()->parent_function()) {
            size_t index = 0u;
            for (auto *bb : parent->basic_blocks()) {
                block_order.emplace(bb, index++);
            }
        }
    }
    std::sort(clone_order.begin(), clone_order.end(), [&](auto *lhs, auto *rhs) noexcept {
        auto li = block_order.find(lhs);
        auto ri = block_order.find(rhs);
        auto lo = li == block_order.end() ? static_cast<size_t>(-1) : li->second;
        auto ro = ri == block_order.end() ? static_cast<size_t>(-1) : ri->second;
        return lo < ro;
    });

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
                    auto *true_block = static_cast<BasicBlock *>(resolver.resolve(cbr->true_block()));
                    auto *false_block = static_cast<BasicBlock *>(resolver.resolve(cbr->false_block()));
                    b.set_insertion_point(cloned_bb);
                    Instruction *cloned = true_block == false_block ?
                                              static_cast<Instruction *>(b.br(true_block)) :
                                              static_cast<Instruction *>(b.cond_br(cond, true_block, false_block));
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

static void instrument_returns_with_skip_flag(Module *mod, const CoroCfgDistillResult::Scope &scope,
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
            }
            store_skip_flag_true(b, frame_arg, mod);
            if (!was_suspend) {
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
    if (result.scopes.size() <= 1u) { return info; }

    auto *actual_frame_type = frame_type ? frame_type : create_frame_type();
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

        if (i > 0) {
            build_skip_check_entry(mod, new_func, frame_arg, body_entry);
        } else {
            new_func->set_body_block(body_entry);
        }

        clone_scope(mod, scope, new_func, frame_arg, result, frame_value_indices, resolver);

        instrument_returns_with_skip_flag(mod, scope, frame_arg, result, frame_value_indices, resolver);

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

[[nodiscard]] static size_t split_function_with_cfg(
    Module *mod, FunctionDefinition *def,
    const CoroCfgDistillResult &result,
    const Type *frame_type = nullptr) noexcept {
    return split_function_with_cfg_info(mod, def, result, frame_type).subroutines.size();
}

[[nodiscard]] static size_t split_function(Module *mod, FunctionDefinition *def) noexcept {
    auto result = coro_cfg_distill_pass_run_on_function(def);
    return split_function_with_cfg(mod, def, result);
}

}// namespace detail

size_t coro_split_pass_run_on_module(Module *m) noexcept {
    size_t total = 0u;
    luisa::vector<FunctionDefinition *> defs;
    for (auto *f : m->function_list()) {
        if (f->is_definition()) {
            defs.push_back(static_cast<FunctionDefinition *>(f));
        }
    }
    for (auto *def : defs) {
        total += detail::split_function(m, def);
    }
    return total;
}

size_t coro_split_pass_run_on_module_with_cfg(
    Module *m, const CoroCfgDistillResult &cfg) noexcept {
    LUISA_DEBUG_ASSERT(!cfg.scopes.empty(), "CoroCfgDistillResult has no scopes.");
    for (auto &scope : cfg.scopes) {
        for (auto *bb : scope.blocks) {
            auto *parent = bb->parent_function();
            if (parent != nullptr && parent->is_definition()) {
                return detail::split_function_with_cfg(
                    m, static_cast<FunctionDefinition *>(parent), cfg);
            }
        }
    }
    return 0u;
}

size_t coro_split_pass_run_on_module_with_cfg_and_frame(
    Module *m, const CoroCfgDistillResult &cfg, const Type *frame_type) noexcept {
    LUISA_DEBUG_ASSERT(!cfg.scopes.empty(), "CoroCfgDistillResult has no scopes.");
    for (auto &scope : cfg.scopes) {
        for (auto *bb : scope.blocks) {
            auto *parent = bb->parent_function();
            if (parent != nullptr && parent->is_definition()) {
                return detail::split_function_with_cfg(
                    m, static_cast<FunctionDefinition *>(parent), cfg, frame_type);
            }
        }
    }
    return 0u;
}

CoroSplitInfo coro_split_pass_run_on_module_with_cfg_and_frame_info(
    Module *m, const CoroCfgDistillResult &cfg, const Type *frame_type) noexcept {
    LUISA_DEBUG_ASSERT(!cfg.scopes.empty(), "CoroCfgDistillResult has no scopes.");
    for (auto &scope : cfg.scopes) {
        for (auto *bb : scope.blocks) {
            auto *parent = bb->parent_function();
            if (parent != nullptr && parent->is_definition()) {
                return detail::split_function_with_cfg_info(
                    m, static_cast<FunctionDefinition *>(parent), cfg, frame_type);
            }
        }
    }
    return {};
}

}// namespace luisa::compute::xir
