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
    // add to parent record
    if (!_scope_stack.empty()) {
        _scope_stack.back().add(stmt);
    }
    // propagate basic usages
    _scope_stack.emplace_back(stmt);
    for (auto s : stmt->statements()) { s->accept(*this); }
    auto record = std::move(_scope_stack.back());
    _scope_stack.pop_back();
    // gather child scope usages
    luisa::unordered_map<Variable, size_t, VariableHash> counters;
    for (auto s : record.children()) {
        for (auto &&v : _scoped_variables.at(s)) {
            counters.try_emplace(v, 0u).first->second++;
        }
    }
    for (auto &&[v, count] : counters) {
        if (count > 1u) { record.def(v); }
    }
    for (auto child : record.children()) {
        auto &&vs = _scoped_variables.at(child);
        for (auto v : record.variables()) {
            vs.erase(v);
        }
    }
    _scoped_variables.emplace(
        stmt, std::move(record.variables()));
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
    for (auto s : stmt->body()->statements()) {
        s->accept(*this);
    }
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

void DefinitionAnalysis::visit(const MetaStmt *stmt) {
    stmt->scope()->accept(*this);
}

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
}

void DefinitionAnalysis::visit(const LiteralExpr *expr) {}

void DefinitionAnalysis::visit(const RefExpr *expr) {
    if (auto v = expr->variable();
        v.tag() == Variable::Tag::LOCAL &&
        _arguments.find(v.uid()) == _arguments.cend()) {
        _scope_stack.back().def(v);
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

}// namespace luisa::compute
