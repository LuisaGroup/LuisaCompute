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

const Expression *Function::_value(const Type *type, ValueExpr::Value value) noexcept {
//    return _arena.create<ValueExpr>(type, std::move(value));
    return nullptr;  // TODO...
}

Variable Function::_constant(const Type *type, const void *data) noexcept {
    return Variable(nullptr, Variable::Tag::DISPATCH_ID, 0);
}

}// namespace luisa::compute
