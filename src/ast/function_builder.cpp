//
// Created by Mike Smith on 2020/12/2.
//

#include "function_builder.h"

namespace luisa::compute {

std::vector<FunctionBuilder *> &FunctionBuilder::_function_stack() noexcept {
    static thread_local std::vector<FunctionBuilder *> stack;
    return stack;
}

void FunctionBuilder::_push(FunctionBuilder *func) noexcept {
    _function_stack().emplace_back(func);
}

FunctionBuilder *FunctionBuilder::_pop() noexcept {
    if (_function_stack().empty()) { LUISA_ERROR_WITH_LOCATION("Invalid pop on empty function stack."); }
    auto f = _function_stack().back();
    _function_stack().pop_back();
    return f;
}
FunctionBuilder *FunctionBuilder::current() noexcept {
    if (_function_stack().empty()) { LUISA_ERROR_WITH_LOCATION("Function stack is empty."); }
    return _function_stack().back();
}

void FunctionBuilder::_append(const Statement *statement) noexcept {
    if (_scope_stack.empty()) { LUISA_ERROR_WITH_LOCATION("Scope stack is empty."); }
    _scope_stack.back()->append(statement);
}

void FunctionBuilder::break_() noexcept {
    _append(_arena.create<BreakStmt>());
}

void FunctionBuilder::continue_() noexcept {
    _append(_arena.create<ContinueStmt>());
}

void FunctionBuilder::return_(const Expression *expr) noexcept {
    if (_ret != nullptr) { LUISA_ERROR_WITH_LOCATION("Multiple non-void return statements are not allowed."); }
    _ret = expr ? expr->type() : nullptr;
    _append(_arena.create<ReturnStmt>(expr));
}

void FunctionBuilder::if_(const Expression *cond, const ScopeStmt *true_branch, const ScopeStmt *false_branch) noexcept {
    _append(_arena.create<IfStmt>(cond, true_branch, false_branch));
}

void FunctionBuilder::while_(const Expression *cond, const ScopeStmt *body) noexcept {
    _append(_arena.create<WhileStmt>(cond, body));
}

void FunctionBuilder::void_(const Expression *expr) noexcept {
    _append(_arena.create<ExprStmt>(expr));
}

void FunctionBuilder::switch_(const Expression *expr, const ScopeStmt *body) noexcept {
    _append(_arena.create<SwitchStmt>(expr, body));
}

void FunctionBuilder::case_(const Expression *expr, const ScopeStmt *body) noexcept {
    _append(_arena.create<SwitchCaseStmt>(expr, body));
}

void FunctionBuilder::default_(const ScopeStmt *body) noexcept {
    _append(_arena.create<SwitchDefaultStmt>(body));
}

void FunctionBuilder::assign(AssignOp op, const Expression *lhs, const Expression *rhs) noexcept {
    _append(_arena.create<AssignStmt>(op, lhs, rhs));
}

const LiteralExpr *FunctionBuilder::literal(const Type *type, LiteralExpr::Value value) noexcept {
    return _arena.create<LiteralExpr>(type, value);
}

const RefExpr *FunctionBuilder::local(const Type *type, std::span<const Expression *> init) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    ArenaVector initializer{_arena, init};
    _append(_arena.create<DeclareStmt>(v, initializer));
    return _ref(v);
}

const RefExpr *FunctionBuilder::local(const Type *type, std::initializer_list<const Expression *> init) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    ArenaVector initializer{_arena, init};
    _append(_arena.create<DeclareStmt>(v, initializer));
    return _ref(v);
}

const RefExpr *FunctionBuilder::shared(const Type *type) noexcept {
    return _ref(_shared_variables.emplace_back(
        Variable{type, Variable::Tag::SHARED, _next_variable_uid()}));
}

uint32_t FunctionBuilder::_next_variable_uid() noexcept {
    auto uid = static_cast<uint32_t>(_variable_usages.size());
    _variable_usages.emplace_back(Usage::NONE);
    return uid;
}

const RefExpr *FunctionBuilder::thread_id() noexcept { return _builtin(Variable::Tag::THREAD_ID); }
const RefExpr *FunctionBuilder::block_id() noexcept { return _builtin(Variable::Tag::BLOCK_ID); }
const RefExpr *FunctionBuilder::dispatch_id() noexcept { return _builtin(Variable::Tag::DISPATCH_ID); }

const RefExpr *FunctionBuilder::_builtin(Variable::Tag tag) noexcept {
    if (auto iter = std::find_if(
            _builtin_variables.cbegin(),
            _builtin_variables.cend(),
            [tag](auto &&v) noexcept { return v.tag() == tag; });
        iter != _builtin_variables.cend()) {
        return _ref(*iter);
    }
    Variable v{Type::of<uint3>(), tag, _next_variable_uid()};
    _builtin_variables.emplace_back(v);
    return _ref(v);
}

const RefExpr *FunctionBuilder::uniform(const Type *type) noexcept {
    Variable v{type, Variable::Tag::UNIFORM, _next_variable_uid()};
    _arguments.emplace_back(v);
    return _ref(v);
}

const RefExpr *FunctionBuilder::buffer(const Type *type) noexcept {
    Variable v{type, Variable::Tag::BUFFER, _next_variable_uid()};
    _arguments.emplace_back(v);
    return _ref(v);
}

const RefExpr *FunctionBuilder::buffer_binding(const Type *element_type, uint64_t handle, size_t offset_bytes) noexcept {
    if (auto iter = std::find_if(
            _captured_buffers.cbegin(),
            _captured_buffers.cend(),
            [handle](auto &&binding) { return binding.handle == handle; });
        iter != _captured_buffers.cend()) {
        if (iter->offset_bytes != offset_bytes) {
            LUISA_ERROR_WITH_LOCATION(
                "Aliasing in implicitly captured buffer (handle = {}, original offset = {}, requested offset = {}).",
                handle, iter->offset_bytes, offset_bytes);
        }
        auto v = iter->variable;
        if (*v.type() != *element_type) {
            LUISA_ERROR_WITH_LOCATION(
                "Aliasing in implicitly captured buffer (handle = {}, original type = {}, requested type = {}).",
                handle, v.type()->description(), element_type->description());
        }
        return _ref(v);
    }
    Variable v{element_type, Variable::Tag::BUFFER, _next_variable_uid()};
    _captured_buffers.emplace_back(BufferBinding{v, handle, offset_bytes});
    return _ref(v);
}

const UnaryExpr *FunctionBuilder::unary(const Type *type, UnaryOp op, const Expression *expr) noexcept {
    return _arena.create<UnaryExpr>(type, op, expr);
}

const BinaryExpr *FunctionBuilder::binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
    return _arena.create<BinaryExpr>(type, op, lhs, rhs);
}

const MemberExpr *FunctionBuilder::member(const Type *type, const Expression *self, size_t member_index) noexcept {
    return _arena.create<MemberExpr>(type, self, member_index);
}

const AccessExpr *FunctionBuilder::access(const Type *type, const Expression *range, const Expression *index) noexcept {
    return _arena.create<AccessExpr>(type, range, index);
}

const CallExpr *FunctionBuilder::call(const Type *type, std::string_view func, std::initializer_list<const Expression *> args) noexcept {
    ArenaString func_name{_arena, func};
    ArenaVector func_args{_arena, args};
    auto expr = _arena.create<CallExpr>(type, func_name, func_args);
    if (expr->is_builtin()) {
        if (auto iter = std::find(_used_builtin_callables.cbegin(), _used_builtin_callables.cend(), func_name);
            iter == _used_builtin_callables.cend()) { _used_builtin_callables.emplace_back(func_name); }
    } else {
        if (auto iter = std::find(_used_custom_callables.cbegin(), _used_custom_callables.cend(), expr->uid());
            iter == _used_custom_callables.cend()) { _used_custom_callables.emplace_back(expr->uid()); }
    }
    return expr;
}

const CastExpr *FunctionBuilder::cast(const Type *type, CastOp op, const Expression *expr) noexcept {
    return _arena.create<CastExpr>(type, op, expr);
}

const RefExpr *FunctionBuilder::_ref(Variable v) noexcept {
    return _arena.create<RefExpr>(v);
}

std::vector<std::unique_ptr<FunctionBuilder>> &FunctionBuilder::_function_registry() noexcept {
    static std::vector<std::unique_ptr<FunctionBuilder>> registry;
    return registry;
}

Function FunctionBuilder::callable(uint32_t uid) noexcept {
    auto f = [uid] {
        std::scoped_lock lock{_function_registry_mutex()};
        const auto &registry = _function_registry();
        if (uid >= registry.size()) { LUISA_ERROR_WITH_LOCATION("Invalid custom callable function with uid {}.", uid); }
        return registry[uid].get();
    }();
    if (f->tag() != Function::Tag::CALLABLE) { LUISA_ERROR_WITH_LOCATION("Requested function (with uid = {}) is not a callable function.", uid); }
    return Function{f};
}

Function FunctionBuilder::kernel(uint32_t uid) noexcept {
    auto f = [uid] {
        std::scoped_lock lock{_function_registry_mutex()};
        const auto &registry = _function_registry();
        if (uid >= registry.size()) { LUISA_ERROR_WITH_LOCATION("Invalid kernel function with uid {}.", uid); }
        return registry[uid].get();
    }();
    if (f->tag() != Function::Tag::KERNEL) { LUISA_ERROR_WITH_LOCATION("Requested function (with uid = {}) is not a kernel function.", uid); }
    return Function{f};
}

spin_mutex &FunctionBuilder::_function_registry_mutex() noexcept {
    static spin_mutex mutex;
    return mutex;
}

ScopeStmt *FunctionBuilder::scope() noexcept {
    return _arena.create<ScopeStmt>(ArenaVector<const Statement *>(_arena));
}

const ConstantExpr *FunctionBuilder::constant(const Type *type, uint64_t hash) noexcept {
    if (!type->is_array()) { LUISA_ERROR_WITH_LOCATION("Constant data must be array."); }
    _captured_constants.emplace_back(ConstantBinding{type, hash});
    return _arena.create<ConstantExpr>(type, hash);
}

void FunctionBuilder::push_scope(ScopeStmt *s) noexcept {
    _scope_stack.emplace_back(s);
}

void FunctionBuilder::pop_scope(const ScopeStmt *s) noexcept {
    if (_scope_stack.empty() || _scope_stack.back() != s) {
        LUISA_ERROR_WITH_LOCATION("Invalid scope stack pop.");
    }
    _scope_stack.pop_back();
}

void FunctionBuilder::for_(const Statement *init, const Expression *condition, const Statement *update, const ScopeStmt *body) noexcept {
    _append(_arena.create<ForStmt>(init, condition, update, body));
}

void FunctionBuilder::mark_variable_usage(uint32_t uid, Usage usage) noexcept {
    auto old_usage = _variable_usages[uid];
    _variable_usages[uid] = static_cast<Usage>(old_usage | usage);
}

}// namespace luisa::compute
