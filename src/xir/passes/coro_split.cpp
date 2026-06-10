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
#include <luisa/xir/metadata/signature_constraint.h>
#include <luisa/xir/module.h>
#include <luisa/xir/op.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_split.h>

namespace luisa::compute::xir {

namespace detail {

static constexpr uint32_t TERMINAL_TOKEN = 0xFFFFFFFFu;
static constexpr uint32_t FRAME_FIELD_TOKEN = 0u;
static constexpr uint32_t FRAME_FIELD_SKIP_FLAG = 1u;
static constexpr uint32_t SKIP_FLAG_TRUE = 1u;

class CoroSplitValueResolver final : public InstructionCloneValueResolver {

private:
    luisa::unordered_map<const Value *, Value *> _value_map;
    luisa::unordered_map<const BasicBlock *, BasicBlock *> _block_map;
    luisa::unordered_map<const Argument *, Argument *> _arg_map;
    XIRBuilder *_builder{nullptr};
    BasicBlock *_alloca_bb{nullptr};

public:
    void set_builder(XIRBuilder *b, BasicBlock *alloca_bb) noexcept {
        _builder = b;
        _alloca_bb = alloca_bb;
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
                auto it = _value_map.find(inst);
                if (it != _value_map.end()) { return it->second; }
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
                LUISA_DEBUG_ASSERT(false, "Instruction not found in resolver.");
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
                        CoroSplitValueResolver &resolver) noexcept {

    // pre-scan: create fallback return blocks for cross-scope branch targets
    luisa::unordered_set<const BasicBlock *> scope_block_set;
    for (auto *bb : scope.blocks) {
        scope_block_set.insert(bb);
    }

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
        }
    }

    XIRBuilder b;

    auto *first_cloned_bb = static_cast<BasicBlock *>(resolver.resolve(scope.blocks.front()));
    resolver.set_builder(&b, first_cloned_bb);

    for (auto *orig_bb : scope.blocks) {
        auto *cloned_bb = static_cast<BasicBlock *>(resolver.resolve(orig_bb));

        b.set_insertion_point(cloned_bb);
        for (auto *inst : orig_bb->instructions()) {
            if (inst->derived_instruction_tag() == DerivedInstructionTag::ALLOCA) {
                auto *orig_alloca = static_cast<const AllocaInst *>(inst);
                auto *cloned_alloca = b.alloca_(orig_alloca->type(), orig_alloca->op());
                auto name_opt = orig_alloca->name();
                if (name_opt.has_value()) { cloned_alloca->set_name(name_opt.value()); }
                resolver.map_value(inst, cloned_alloca);
            }
        }

        for (auto *inst : orig_bb->instructions()) {
            auto tag = inst->derived_instruction_tag();

            switch (tag) {
                case DerivedInstructionTag::ALLOCA: {
                    break;// already cloned in pre-scan
                }
                case DerivedInstructionTag::CORO_SUSPEND: {
                    auto *s = static_cast<CoroSuspendInst *>(inst);
                    b.set_insertion_point(cloned_bb);
                    store_frame_token(b, frame_arg, mod, s->token());
                    b.return_void();
                    goto block_terminated;
                }
                case DerivedInstructionTag::CORO_TERMINATE: {
                    b.set_insertion_point(cloned_bb);
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
                                              Value *frame_arg, CoroSplitValueResolver &resolver,
                                              bool is_last_scope) noexcept {
    XIRBuilder b;
    for (auto *orig_bb : scope.blocks) {
        auto *cloned_bb = static_cast<BasicBlock *>(resolver.resolve(orig_bb));
        auto *term = cloned_bb->terminator();
        if (term != nullptr && term->derived_instruction_tag() == DerivedInstructionTag::RETURN) {
            b.set_insertion_point(term->prev());
            store_skip_flag_true(b, frame_arg, mod);
            if (is_last_scope) {
                store_frame_token(b, frame_arg, mod, TERMINAL_TOKEN);
            }
        }
    }
}

[[nodiscard]] static size_t split_function_with_cfg(
    Module *mod, FunctionDefinition *def,
    const CoroCfgDistillResult &result,
    const Type *frame_type = nullptr) noexcept {
    if (result.scopes.size() <= 1u) { return 0u; }

    auto *actual_frame_type = frame_type ? frame_type : create_frame_type();

    size_t created = 0u;
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

        clone_scope(mod, scope, new_func, frame_arg, resolver);

        if (i > 0) {
            bool is_last = (i == result.scopes.size() - 1u);
            instrument_returns_with_skip_flag(mod, scope, frame_arg, resolver, is_last);
        }

        static_cast<void>(new_func->create_metadata<SignatureConstraintMD>());

        created++;
    }

    return created;
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

}// namespace luisa::compute::xir
