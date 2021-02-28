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
    _scope_stack.back().emplace_back(statement);
}

void FunctionBuilder::break_() noexcept {
    _append(_arena.create<BreakStmt>());
}

void FunctionBuilder::continue_() noexcept {
    _append(_arena.create<ContinueStmt>());
}

void FunctionBuilder::return_(const Expression *expr) noexcept {
    _append(_arena.create<ReturnStmt>(expr));
}

void FunctionBuilder::if_(const Expression *cond, const Statement *true_branch) noexcept {
    _append(_arena.create<IfStmt>(cond, true_branch));
}

void FunctionBuilder::if_(const Expression *cond, const Statement *true_branch, const Statement *false_branch) noexcept {
    _append(_arena.create<IfStmt>(cond, true_branch, false_branch));
}

void FunctionBuilder::while_(const Expression *cond, const Statement *body) noexcept {
    _append(_arena.create<WhileStmt>(cond, body));
}

void FunctionBuilder::void_(const Expression *expr) noexcept {
    _append(_arena.create<ExprStmt>(expr));
}

void FunctionBuilder::switch_(const Expression *expr, const Statement *body) noexcept {
    _append(_arena.create<SwitchStmt>(expr, body));
}

void FunctionBuilder::case_(const Expression *expr, const Statement *body) noexcept {
    _append(_arena.create<SwitchCaseStmt>(expr, body));
}

void FunctionBuilder::default_(const Statement *body) noexcept {
    _append(_arena.create<SwitchDefaultStmt>(body));
}

void FunctionBuilder::assign(AssignOp op, const Expression *lhs, const Expression *rhs) noexcept {
    _append(_arena.create<AssignStmt>(op, lhs, rhs));
}

const Expression *FunctionBuilder::_literal(const Type *type, LiteralExpr::Value value) noexcept {
    return _arena.create<LiteralExpr>(type, std::move(value));
}

const Expression *FunctionBuilder::_constant(const Type *type, const void *data) noexcept {
    Variable v{type, Variable::Tag::CONSTANT, _next_variable_uid()};
    _constant_variables.emplace_back(ConstantData{v, data});
    return _ref(v);
}

const Expression *FunctionBuilder::local(const Type *type, std::span<const Expression *> init) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    ArenaVector initializer{_arena, init};
    _append(_arena.create<DeclareStmt>(v, initializer));
    return _ref(v);
}

const Expression *FunctionBuilder::local(const Type *type, std::initializer_list<const Expression *> init) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    ArenaVector initializer{_arena, init};
    _append(_arena.create<DeclareStmt>(v, initializer));
    return _ref(v);
}

const Expression *FunctionBuilder::shared(const Type *type) noexcept {
    return _ref(_shared_variables.emplace_back(
        Variable{type, Variable::Tag::SHARED, _next_variable_uid()}));
}

uint32_t FunctionBuilder::_next_variable_uid() noexcept { return ++_variable_counter; }

const Expression *FunctionBuilder::thread_id() noexcept { return _builtin(Variable::Tag::THREAD_ID); }
const Expression *FunctionBuilder::block_id() noexcept { return _builtin(Variable::Tag::BLOCK_ID); }
const Expression *FunctionBuilder::dispatch_id() noexcept { return _builtin(Variable::Tag::DISPATCH_ID); }

const Expression *FunctionBuilder::_builtin(Variable::Tag tag) noexcept {
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

const Expression *FunctionBuilder::uniform(const Type *type) noexcept {
    Variable v{type, Variable::Tag::UNIFORM, _next_variable_uid()};
    _arguments.emplace_back(v);
    return _ref(v);
}

const Expression *FunctionBuilder::buffer(const Type *type) noexcept {
    Variable v{type, Variable::Tag::BUFFER, _next_variable_uid()};
    _arguments.emplace_back(v);
    return _ref(v);
}

const Expression *FunctionBuilder::_uniform_binding(const Type *type, const void *data) noexcept {
    if (auto iter = std::find_if(
            _captured_uniforms.cbegin(),
            _captured_uniforms.cend(),
            [data](auto &&binding) { return binding.data == data; });
        iter != _captured_uniforms.cend()) {
        auto v = iter->variable;
        if (*v.type() != *type) {
            LUISA_ERROR_WITH_LOCATION(
                "Pointer aliasing in implicitly captured uniform data (original type = {}, requested type = {}).",
                v.type()->description(), type->description());
        }
        return _ref(v);
    }
    Variable v{type, Variable::Tag::UNIFORM, _next_variable_uid()};
    _captured_uniforms.emplace_back(UniformBinding{v, data});
    return _ref(v);
}

const Expression *FunctionBuilder::_buffer_binding(const Type *type, uint64_t handle, size_t offset_bytes) noexcept {
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
        if (*v.type() != *type) {
            LUISA_ERROR_WITH_LOCATION(
                "Aliasing in implicitly captured buffer (handle = {}, original type = {}, requested type = {}).",
                handle, v.type()->description(), type->description());
        }
        return _ref(v);
    }
    Variable v{type, Variable::Tag::BUFFER, _next_variable_uid()};
    _captured_buffers.emplace_back(BufferBinding{v, handle, offset_bytes});
    return _ref(v);
}

const Expression *FunctionBuilder::unary(const Type *type, UnaryOp op, const Expression *expr) noexcept {
    return _arena.create<UnaryExpr>(type, op, expr);
}

const Expression *FunctionBuilder::binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
    return _arena.create<BinaryExpr>(type, op, lhs, rhs);
}

const Expression *FunctionBuilder::member(const Type *type, const Expression *self, size_t member_index) noexcept {
    return _arena.create<MemberExpr>(type, self, member_index);
}

const Expression *FunctionBuilder::access(const Type *type, const Expression *range, const Expression *index) noexcept {
    return _arena.create<AccessExpr>(type, range, index);
}

const Expression *FunctionBuilder::call(const Type *type, std::string_view func, std::span<const Expression *> args) noexcept {
    ArenaString func_name{_arena, func};
    ArenaVector func_args{_arena, args};
    return _arena.create<CallExpr>(type, func_name, func_args);
}

const Expression *FunctionBuilder::call(const Type *type, std::string_view func, std::initializer_list<const Expression *> args) noexcept {
    ArenaString func_name{_arena, func};
    ArenaVector func_args{_arena, args};
    return _arena.create<CallExpr>(type, func_name, func_args);
}

const Expression *FunctionBuilder::cast(const Type *type, CastOp op, const Expression *expr) noexcept {
    return _arena.create<CastExpr>(type, op, expr);
}

const Expression *FunctionBuilder::_ref(Variable v) noexcept {
    return _arena.create<RefExpr>(v);
}

}// namespace luisa::compute
