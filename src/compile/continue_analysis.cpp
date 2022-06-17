//
// Created by Mike Smith on 2022/3/8.
//

#include <compile/continue_analysis.h>

namespace luisa::compute {

void ContinueAnalysis::visit(const BreakStmt *stmt) {}
void ContinueAnalysis::visit(const ContinueStmt *stmt) {
    LUISA_ASSERT(!_scope_stack.empty(), "Invalid continue statement.");
    auto success = _continue_scopes.try_emplace(stmt, _scope_stack.back()).second;
    LUISA_ASSERT(success, "Continue statement already visited.");
}

void ContinueAnalysis::visit(const ReturnStmt *stmt) {}

void ContinueAnalysis::visit(const ScopeStmt *stmt) {
    for (auto s : stmt->statements()) { s->accept(*this); }
}

void ContinueAnalysis::visit(const IfStmt *stmt) {
    stmt->true_branch()->accept(*this);
    stmt->false_branch()->accept(*this);
}

void ContinueAnalysis::visit(const LoopStmt *stmt) {
    _scope_stack.emplace_back(stmt->body());
    stmt->body()->accept(*this);
    _scope_stack.pop_back();
}

void ContinueAnalysis::visit(const ExprStmt *stmt) {}

void ContinueAnalysis::visit(const SwitchStmt *stmt) {
    stmt->body()->accept(*this);
}

void ContinueAnalysis::visit(const SwitchCaseStmt *stmt) {
    stmt->body()->accept(*this);
}

void ContinueAnalysis::visit(const SwitchDefaultStmt *stmt) {
    stmt->body()->accept(*this);
}

void ContinueAnalysis::visit(const AssignStmt *stmt) {}

void ContinueAnalysis::visit(const ForStmt *stmt) {
    _scope_stack.emplace_back(stmt->body());
    stmt->body()->accept(*this);
    _scope_stack.pop_back();
}

void ContinueAnalysis::visit(const CommentStmt *stmt) {}

void ContinueAnalysis::analyze(Function f) noexcept {
    LUISA_ASSERT(_scope_stack.empty() && _continue_scopes.empty(), "Not clear.");
    f.body()->accept(*this);
}

}// namespace luisa::compute