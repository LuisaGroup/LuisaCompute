//
// Created by Mike Smith on 2022/3/4.
//

#include <compile/definition_analysis.h>

namespace luisa::compute {

void DefinitionAnalysis::visit(const BreakStmt *stmt) {}
void DefinitionAnalysis::visit(const ContinueStmt *stmt) {}

void DefinitionAnalysis::visit(const ReturnStmt *stmt) {
    if (auto expr = stmt->expression()) {
        _require_definition(expr);
    }
}

void DefinitionAnalysis::visit(const ScopeStmt *stmt) {
    _scope_stack.emplace_back(stmt);
    for (auto s : stmt->statements()) { s->accept(*this); }
    _scope_stack.back().finalize();
    auto variables = std::move(_scope_stack.back().variables());
    _scope_stack.pop_back();
    _scoped_variables.emplace(stmt, std::move(variables));
}

void DefinitionAnalysis::visit(const IfStmt *stmt) {
    _require_definition(stmt->condition());
    stmt->true_branch()->accept(*this);
    stmt->false_branch()->accept(*this);
}

void DefinitionAnalysis::visit(const LoopStmt *stmt) {
    stmt->body()->accept(*this);
}

void DefinitionAnalysis::visit(const ExprStmt *stmt) {
    _require_definition(stmt->expression());
}

void DefinitionAnalysis::visit(const SwitchStmt *stmt) {
    _require_definition(stmt->expression());
    stmt->body()->accept(*this);
}

void DefinitionAnalysis::visit(const SwitchCaseStmt *stmt) {
    _require_definition(stmt->expression());
    stmt->body()->accept(*this);
}

void DefinitionAnalysis::visit(const SwitchDefaultStmt *stmt) {
    stmt->body()->accept(*this);
}

void DefinitionAnalysis::visit(const AssignStmt *stmt) {
    _require_definition(stmt->lhs());
    _require_definition(stmt->rhs());
}

void DefinitionAnalysis::visit(const ForStmt *stmt) {
    _require_definition(stmt->variable());
    _require_definition(stmt->condition());
    _require_definition(stmt->step());
    stmt->body()->accept(*this);
}

void DefinitionAnalysis::visit(const CommentStmt *stmt) {}

void DefinitionAnalysis::analyze(Function f) noexcept {
    // initialize states
    for (auto a : f.arguments()) {
        _arguments.emplace(a.uid());
    }
    // perform analysis
    f.body()->accept(*this);
    if (!_scope_stack.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Non-empty stack (size = {}) after "
            "ISPCVariableDefinition analysis.",
            _scope_stack.size());
        _scope_stack.clear();
    }
}

void DefinitionAnalysis::reset() noexcept {
    _scoped_variables.clear();
    _arguments.clear();
}

void DefinitionAnalysis::_require_definition(const Expression *expr) noexcept {
    expr->accept(*this);
}

void DefinitionAnalysis::visit(const UnaryExpr *expr) {
    expr->operand()->accept(*this);
}

void DefinitionAnalysis::visit(const BinaryExpr *expr) {
    expr->lhs()->accept(*this);
    expr->rhs()->accept(*this);
}

void DefinitionAnalysis::visit(const MemberExpr *expr) {
    expr->self()->accept(*this);
}

void DefinitionAnalysis::visit(const AccessExpr *expr) {
    expr->range()->accept(*this);
    expr->index()->accept(*this);
}

void DefinitionAnalysis::visit(const LiteralExpr *expr) {}

void DefinitionAnalysis::visit(const RefExpr *expr) {
    if (auto v = expr->variable();
        v.tag() == Variable::Tag::LOCAL &&
        _arguments.find(v.uid()) == _arguments.cend()) {
        _scope_stack.back().reference(v);
        auto scope = _scope_stack.back().scope();
        for (auto it = _scope_stack.rbegin() + 1u;
             it != _scope_stack.rend(); it++) {
            it->propagate(v, scope);
            scope = it->scope();
        }
    }
}

void DefinitionAnalysis::visit(const ConstantExpr *expr) {}

void DefinitionAnalysis::visit(const CallExpr *expr) {
    for (auto arg : expr->arguments()) {
        arg->accept(*this);
    }
}

void DefinitionAnalysis::visit(const CastExpr *expr) {
    expr->expression()->accept(*this);
}

void DefinitionAnalysis::ScopeRecord::reference(Variable v) noexcept {
    _variables.emplace(v);
}

void DefinitionAnalysis::ScopeRecord::propagate(Variable v, const ScopeStmt *scope) {
    auto iter = _propagated.emplace(v, ScopeSet{}).first;
    iter->second.emplace(scope);
}

void DefinitionAnalysis::ScopeRecord::finalize() noexcept {
    for (auto &&[v, s] : _propagated) {
        if (s.size() > 1u) {
            _variables.emplace(v);
        }
    }
}

}// namespace luisa::compute
