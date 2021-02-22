//
// Created by Mike Smith on 2020/12/2.
//

#include "function.h"

namespace luisa::compute {

std::vector<Function *> &Function::_function_stack() noexcept {
    static thread_local std::vector<Function *> stack;
    return stack;
}

void Function::_push(Function *func) noexcept {
    _function_stack().emplace_back(func);
}

Function *Function::_pop() noexcept {
    if (_function_stack().empty()) { LUISA_ERROR_WITH_LOCATION("Invalid pop on empty function stack."); }
    auto f = _function_stack().back();
    _function_stack().pop_back();
    return f;
}
Function *Function::current() noexcept {
    if (_function_stack().empty()) { LUISA_ERROR_WITH_LOCATION("Function stack is empty."); }
    return _function_stack().back();
}

void Function::_add(const Statement *statement) noexcept {
    if (_scope_stack.empty()) { LUISA_ERROR_WITH_LOCATION("Scope stack is empty."); }
    _scope_stack.back().emplace_back(statement);
}

void Function::break_() noexcept {
    _add(_arena.create<BreakStmt>());
}

void Function::continue_() noexcept {
    _add(_arena.create<ContinueStmt>());
}

void Function::return_(const Expression *expr) noexcept {
    _add(_arena.create<ReturnStmt>(expr));
}

void Function::if_(const Expression *cond, const Statement *true_branch) noexcept {
    _add(_arena.create<IfStmt>(cond, true_branch));
}

void Function::if_(const Expression *cond, const Statement *true_branch, const Statement *false_branch) noexcept {
    _add(_arena.create<IfStmt>(cond, true_branch, false_branch));
}

void Function::while_(const Expression *cond, const Statement *body) noexcept {
    _add(_arena.create<WhileStmt>(cond, body));
}

void Function::void_(const Expression *expr) noexcept {
    _add(_arena.create<ExprStmt>(expr));
}

void Function::switch_(const Expression *expr, const Statement *body) noexcept {
    _add(_arena.create<SwitchStmt>(expr, body));
}

void Function::case_(const Expression *expr, const Statement *body) noexcept {
    _add(_arena.create<SwitchCaseStmt>(expr, body));
}

void Function::default_(const Statement *body) noexcept {
    _add(_arena.create<SwitchDefaultStmt>(body));
}

void Function::assign(AssignOp op, const Expression *lhs, const Expression *rhs) noexcept {
    _add(_arena.create<AssignStmt>(op, lhs, rhs));
}

const Expression *Function::_value(const Type *type, ValueExpr::Value value) noexcept {
    return _arena.create<ValueExpr>(type, std::move(value));
}

Variable Function::_constant(const Type *type, const void *data) noexcept {
    Variable v{type, Variable::Tag::CONSTANT, _next_variable_uid()};
    _constant_variables.emplace_back(ConstantData{v, data});
    return v;
}

Variable Function::local(const Type *type, std::span<const Expression *> init) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    ArenaVector initializer{_arena, init};
    _add(_arena.create<DeclareStmt>(v, initializer));
    return v;
}

Variable Function::local(const Type *type, std::initializer_list<const Expression *> init) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    ArenaVector initializer{_arena, init};
    _add(_arena.create<DeclareStmt>(v, initializer));
    return v;
}

Variable Function::shared(const Type *type) noexcept {
    return _shared_variables.emplace_back(
        Variable{type, Variable::Tag::SHARED, _next_variable_uid()});
}

uint32_t Function::_next_variable_uid() noexcept { return ++_variable_counter; }

Variable Function::thread_id() noexcept { return _builtin(Variable::Tag::THREAD_ID); }
Variable Function::block_id() noexcept { return _builtin(Variable::Tag::BLOCK_ID); }
Variable Function::dispatch_id() noexcept { return _builtin(Variable::Tag::DISPATCH_ID); }

Variable Function::_builtin(Variable::Tag tag) noexcept {
    if (auto iter = std::find_if(
            _builtin_variables.cbegin(),
            _builtin_variables.cend(),
            [tag](auto &&v) noexcept { return v.tag() == tag; });
        iter != _builtin_variables.cend()) {
        return *iter;
    }
    Variable v{Type::of<uint3>(), tag, _next_variable_uid()};
    _builtin_variables.emplace_back(v);
    return v;
}

Variable Function::uniform(const Type *type) noexcept {
    Variable v{type, Variable::Tag::UNIFORM, _next_variable_uid()};
    _arguments.emplace_back(v);
    return v;
}

Variable Function::buffer(const Type *type) noexcept {
    Variable v{type, Variable::Tag::BUFFER, _next_variable_uid()};
    _arguments.emplace_back(v);
    return v;
}

Variable Function::_uniform_binding(const Type *type, const void *data) noexcept {
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
        return v;
    }
    Variable v{type, Variable::Tag::UNIFORM, _next_variable_uid()};
    _captured_uniforms.emplace_back(UniformBinding{v, data});
    return v;
}

Variable Function::_buffer_binding(const Type *type, uint64_t handle, size_t offset_bytes) noexcept {
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
        return v;
    }
    Variable v{type, Variable::Tag::BUFFER, _next_variable_uid()};
    _captured_buffers.emplace_back(BufferBinding{v, handle, offset_bytes});
    return v;
}

}// namespace luisa::compute
