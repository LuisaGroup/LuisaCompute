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

const Statement *Function::scope(const std::function<void()> &body) noexcept {
    auto s = _arena.create<ScopeStmt>(ArenaVector<const Statement *>{_arena});
    _scope_stack.emplace_back(s);
    body();
    if (_scope_stack.empty() || _scope_stack.back() != s) { LUISA_ERROR_WITH_LOCATION("Invalid scope when pop."); }
    _scope_stack.pop_back();
    return s;
}

ScopeStmt *Function::_current_scope() noexcept {
    if (_scope_stack.empty()) { LUISA_ERROR_WITH_LOCATION("Scope stack is empty."); }
    return _scope_stack.back();
}

void Function::break_() noexcept {
    _current_scope()->statements().emplace_back(_arena.create<BreakStmt>());
}

void Function::continue_() noexcept {
    _current_scope()->statements().emplace_back(_arena.create<ContinueStmt>());
}

void Function::return_(const Expression *expr) noexcept {
    _current_scope()->statements().emplace_back(_arena.create<ReturnStmt>(expr));
}

void Function::if_(const Expression *cond, const Statement *true_branch) noexcept {
    _current_scope()->statements().emplace_back(_arena.create<IfStmt>(cond, true_branch));
}

void Function::if_(const Expression *cond, const Statement *true_branch, const Statement *false_branch) noexcept {
    _current_scope()->statements().emplace_back(_arena.create<IfStmt>(cond, true_branch, false_branch));
}

void Function::while_(const Expression *cond, const Statement *body) noexcept {
    _current_scope()->statements().emplace_back(_arena.create<WhileStmt>(cond, body));
}

void Function::void_(const Expression *expr) noexcept {
    _current_scope()->statements().emplace_back(_arena.create<ExprStmt>(expr));
}

void Function::switch_(const Expression *expr, const Statement *body) noexcept {
    _current_scope()->statements().emplace_back(_arena.create<SwitchStmt>(expr, body));
}

void Function::case_(const Expression *expr, const Statement *body) noexcept {
    _current_scope()->statements().emplace_back(_arena.create<SwitchCaseStmt>(expr, body));
}

void Function::default_(const Statement *body) noexcept {
    _current_scope()->statements().emplace_back(_arena.create<SwitchDefaultStmt>(body));
}

void Function::assign(AssignOp op, const Expression *lhs, const Expression *rhs) noexcept {
    _current_scope()->statements().emplace_back(_arena.create<AssignStmt>(op, lhs, rhs));
}

void Function::define(const std::function<void()> &def) noexcept {
    if (_defined) { LUISA_ERROR_WITH_LOCATION("Multiple definition."); }
    _push(this);
    def();
    if (_pop() != this) { LUISA_ERROR_WITH_LOCATION("Invalid function on stack top."); }
    _defined = true;
}

}// namespace luisa::compute
