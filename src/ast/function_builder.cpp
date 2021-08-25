//
// Created by Mike Smith on 2020/12/2.
//

#include <ast/function_builder.h>

namespace luisa::compute::detail {

std::vector<FunctionBuilder *> &FunctionBuilder::_function_stack() noexcept {
    static thread_local std::vector<FunctionBuilder *> stack;
    return stack;
}

void FunctionBuilder::push(FunctionBuilder *func) noexcept {
    if (func->tag() == Function::Tag::KERNEL && !_function_stack().empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Kernel definitions cannot be nested.");
    }
    _function_stack().emplace_back(func);
}

void FunctionBuilder::pop(const FunctionBuilder *func) noexcept {
    if (_function_stack().empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid pop on empty function stack.");
    }
    auto f = _function_stack().back();
    _function_stack().pop_back();
    if (f != func) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Invalid function on stack top."); }
    if (f->tag() == Function::Tag::CALLABLE
        && !(f->builtin_variables().empty()
             && f->shared_variables().empty()
             && f->captured_buffers().empty()
             && f->captured_textures().empty()
             && f->captured_heaps().empty())) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Custom callables may not have builtin, "
            "shared or captured variables.");
    }
}

FunctionBuilder *FunctionBuilder::current() noexcept {
    return _function_stack().empty() ? nullptr : _function_stack().back();
}

void FunctionBuilder::_append(const Statement *statement) noexcept {
    if (_scope_stack.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Scope stack is empty.");
    }
    _scope_stack.back()->append(statement);
}

void FunctionBuilder::break_() noexcept {
    _append(_arena->create<BreakStmt>());
}

void FunctionBuilder::continue_() noexcept {
    _append(_arena->create<ContinueStmt>());
}

void FunctionBuilder::return_(const Expression *expr) noexcept {
    if (_ret != nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Multiple non-void return statements are not allowed.");
    }
    _ret = expr ? expr->type() : nullptr;
    _append(_arena->create<ReturnStmt>(expr));
}

void FunctionBuilder::if_(const Expression *cond, const ScopeStmt *true_branch, const ScopeStmt *false_branch) noexcept {
    _append(_arena->create<IfStmt>(cond, true_branch, false_branch));
}

void FunctionBuilder::loop_(const ScopeStmt *body) noexcept {
    _append(_arena->create<LoopStmt>(body));
}

void FunctionBuilder::_void_expr(const Expression *expr) noexcept {
    _append(_arena->create<ExprStmt>(expr));
}

void FunctionBuilder::switch_(const Expression *expr, const ScopeStmt *body) noexcept {
    _append(_arena->create<SwitchStmt>(expr, body));
}

void FunctionBuilder::case_(const Expression *expr, const ScopeStmt *body) noexcept {
    _append(_arena->create<SwitchCaseStmt>(expr, body));
}

void FunctionBuilder::default_(const ScopeStmt *body) noexcept {
    _append(_arena->create<SwitchDefaultStmt>(body));
}

void FunctionBuilder::assign(AssignOp op, const Expression *lhs, const Expression *rhs) noexcept {
    _append(_arena->create<AssignStmt>(op, lhs, rhs));
}

const LiteralExpr *FunctionBuilder::literal(const Type *type, LiteralExpr::Value value) noexcept {
    return _arena->create<LiteralExpr>(type, value);
}

const RefExpr *FunctionBuilder::local(const Type *type) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    _append(_arena->create<DeclareStmt>(v));
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
const RefExpr *FunctionBuilder::dispatch_size() noexcept { return _builtin(Variable::Tag::DISPATCH_SIZE); }

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

const RefExpr *FunctionBuilder::argument(const Type *type) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
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
        if (iter->offset_bytes != offset_bytes) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Aliasing in implicitly captured buffer "
                "(handle = {}, original offset = {}, requested offset = {}).",
                handle, iter->offset_bytes, offset_bytes);
        }
        auto v = iter->variable;
        if (*v.type() != *element_type) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Aliasing in implicitly captured buffer "
                "(handle = {}, original type = {}, requested type = {}).",
                handle, v.type()->description(), element_type->description());
        }
        return _ref(v);
    }
    Variable v{element_type, Variable::Tag::BUFFER, _next_variable_uid()};
    _captured_buffers.emplace_back(BufferBinding{v, handle, offset_bytes});
    return _ref(v);
}

const UnaryExpr *FunctionBuilder::unary(const Type *type, UnaryOp op, const Expression *expr) noexcept {
    return _arena->create<UnaryExpr>(type, op, expr);
}

const BinaryExpr *FunctionBuilder::binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
    return _arena->create<BinaryExpr>(type, op, lhs, rhs);
}

const MemberExpr *FunctionBuilder::member(const Type *type, const Expression *self, size_t member_index) noexcept {
    return _arena->create<MemberExpr>(type, self, member_index);
}

const MemberExpr *FunctionBuilder::swizzle(const Type *type, const Expression *self, size_t swizzle_size, uint64_t swizzle_code) noexcept {
    return _arena->create<MemberExpr>(type, self, swizzle_size, swizzle_code);
}

const AccessExpr *FunctionBuilder::access(const Type *type, const Expression *range, const Expression *index) noexcept {
    return _arena->create<AccessExpr>(type, range, index);
}

const CastExpr *FunctionBuilder::cast(const Type *type, CastOp op, const Expression *expr) noexcept {
    return _arena->create<CastExpr>(type, op, expr);
}

const RefExpr *FunctionBuilder::_ref(Variable v) noexcept {
    return _arena->create<RefExpr>(v);
}

ScopeStmt *FunctionBuilder::scope() noexcept {
    return _arena->create<ScopeStmt>(ArenaVector<const Statement *>(*_arena));
}

const ConstantExpr *FunctionBuilder::constant(const Type *type, ConstantData data) noexcept {
    if (!type->is_array()) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Constant data must be array."); }
    _captured_constants.emplace_back(ConstantBinding{type, data});
    return _arena->create<ConstantExpr>(type, data);
}

void FunctionBuilder::push_scope(ScopeStmt *s) noexcept {
    _scope_stack.emplace_back(s);
}

void FunctionBuilder::pop_scope(const ScopeStmt *s) noexcept {
    if (_scope_stack.empty() || _scope_stack.back() != s) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid scope stack pop.");
    }
    _scope_stack.pop_back();
}

void FunctionBuilder::for_(const Expression *var, const Expression *condition, const Expression *update, const ScopeStmt *body) noexcept {
    _append(_arena->create<ForStmt>(var, condition, update, body));
}

void FunctionBuilder::mark_variable_usage(uint32_t uid, Usage usage) noexcept {
    auto old_usage = to_underlying(_variable_usages[uid]);
    auto u = static_cast<Usage>(old_usage | to_underlying(usage));
    _variable_usages[uid] = u;
}

FunctionBuilder::FunctionBuilder(Arena *arena, FunctionBuilder::Tag tag) noexcept
    : _arena{arena},
      _body{ArenaVector<const Statement *>(*arena)},
      _scope_stack{*arena},
      _builtin_variables{*arena},
      _shared_variables{*arena},
      _captured_constants{*arena},
      _captured_buffers{*arena},
      _captured_textures{*arena},
      _captured_heaps{*arena},
      _captured_accels{*arena},
      _arguments{*arena},
      _used_custom_callables{*arena},
      _variable_usages{*arena, 128u},
      _hash{0ul},
      _tag{tag} {}

const RefExpr *FunctionBuilder::texture(const Type *type) noexcept {
    Variable v{type, Variable::Tag::TEXTURE, _next_variable_uid()};
    _arguments.emplace_back(v);
    return _ref(v);
}

const RefExpr *FunctionBuilder::texture_binding(const Type *type, uint64_t handle) noexcept {
    if (auto iter = std::find_if(
            _captured_textures.cbegin(),
            _captured_textures.cend(),
            [handle](auto &&binding) { return binding.handle == handle; });
        iter != _captured_textures.cend()) {
        auto v = iter->variable;
        if (*v.type() != *type) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Aliasing in implicitly captured texture "
                "(handle = {}, original type = {}, requested type = {}).",
                handle, v.type()->description(), type->description());
        }
        return _ref(v);
    }
    Variable v{type, Variable::Tag::TEXTURE, _next_variable_uid()};
    _captured_textures.emplace_back(TextureBinding{v, handle});
    return _ref(v);
}

const CallExpr *FunctionBuilder::call(const Type *type, CallOp call_op, std::initializer_list<const Expression *> args) noexcept {
    if (call_op == CallOp::CUSTOM) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Custom functions are not allowed to "
            "be called with enum CallOp.");
    }
    _used_builtin_callables.mark(call_op);
    ArenaVector func_args{*_arena, args};
    return _arena->create<CallExpr>(type, call_op, func_args);
}

const CallExpr *FunctionBuilder::call(const Type *type, Function custom, std::initializer_list<const Expression *> args) noexcept {
    if (custom.tag() != Function::Tag::CALLABLE) {
        LUISA_ERROR_WITH_LOCATION("Calling non-callable function in device code.");
    }
    ArenaVector func_args{*_arena, args};
    auto expr = _arena->create<CallExpr>(type, custom, func_args);
    if (auto iter = std::find(_used_custom_callables.cbegin(), _used_custom_callables.cend(), custom);
        iter == _used_custom_callables.cend()) {
        _used_custom_callables.emplace_back(custom);
    }
    return expr;
}

void FunctionBuilder::call(CallOp call_op, std::initializer_list<const Expression *> args) noexcept {
    _void_expr(call(nullptr, call_op, args));
}

void FunctionBuilder::call(Function custom, std::initializer_list<const Expression *> args) noexcept {
    _void_expr(call(nullptr, custom, args));
}

void FunctionBuilder::_compute_hash() noexcept {
    // TODO: compute hash based on the AST
    _hash = std::hash<uint64_t>{}(reinterpret_cast<uint64_t>(this));
}

const RefExpr *FunctionBuilder::heap_binding(uint64_t handle) noexcept {
    if (auto iter = std::find_if(
            _captured_heaps.cbegin(),
            _captured_heaps.cend(),
            [handle](auto &&binding) { return binding.handle == handle; });
        iter != _captured_heaps.cend()) {
        return _ref(iter->variable);
    }
    Variable v{Type::of<Heap>(), Variable::Tag::HEAP, _next_variable_uid()};
    _captured_heaps.emplace_back(HeapBinding{v, handle});
    return _ref(v);
}

const RefExpr *FunctionBuilder::heap() noexcept {
    Variable v{Type::of<Heap>(), Variable::Tag::HEAP, _next_variable_uid()};
    _arguments.emplace_back(v);
    return _ref(v);
}

const RefExpr *FunctionBuilder::accel_binding(uint64_t handle) noexcept {
    _raytracing = true;
    if (auto iter = std::find_if(
            _captured_accels.cbegin(),
            _captured_accels.cend(),
            [handle](auto &&binding) { return binding.handle == handle; });
        iter != _captured_accels.cend()) {
        return _ref(iter->variable);
    }
    Variable v{Type::of<Accel>(), Variable::Tag::ACCEL, _next_variable_uid()};
    _captured_accels.emplace_back(AccelBinding{v, handle});
    return _ref(v);
}

const RefExpr *FunctionBuilder::accel() noexcept {
    _raytracing = true;
    Variable v{Type::of<Accel>(), Variable::Tag::ACCEL, _next_variable_uid()};
    _arguments.emplace_back(v);
    return _ref(v);
}

const CallExpr *FunctionBuilder::call(const Type *type, CallOp call_op, std::span<const Expression *const> args) noexcept {
    if (call_op == CallOp::CUSTOM) [[unlikely]] {
      LUISA_ERROR_WITH_LOCATION(
          "Custom functions are not allowed to "
          "be called with enum CallOp.");
    }
    _used_builtin_callables.mark(call_op);
    ArenaVector func_args{*_arena, args};
    return _arena->create<CallExpr>(type, call_op, func_args);
}

const CallExpr *FunctionBuilder::call(const Type *type, Function custom, std::span<const Expression *const> args) noexcept {
    if (custom.tag() != Function::Tag::CALLABLE) {
        LUISA_ERROR_WITH_LOCATION("Calling non-callable function in device code.");
    }
    ArenaVector func_args{*_arena, args};
    auto expr = _arena->create<CallExpr>(type, custom, func_args);
    if (auto iter = std::find(_used_custom_callables.cbegin(), _used_custom_callables.cend(), custom);
    iter == _used_custom_callables.cend()) {
        _used_custom_callables.emplace_back(custom);
    }
    return expr;
}

void FunctionBuilder::call(CallOp call_op, std::span<const Expression *const> args) noexcept {
    _void_expr(call(nullptr, call_op, args));
}

void FunctionBuilder::call(Function custom, std::span<const Expression *const> args) noexcept {
    _void_expr(call(nullptr, custom, args));
}

const RefExpr *FunctionBuilder::reference(const Type *type) noexcept {
    Variable v{type, Variable::Tag::REFERENCE, _next_variable_uid()};
    _arguments.emplace_back(v);
    return _ref(v);
}

void FunctionBuilder::comment_(std::string_view comment) noexcept {
    ArenaString s{*_arena, comment};
    _append(_arena->create<CommentStmt>(s));
}

}// namespace luisa::compute::detail
