//
// Created by Mike Smith on 2022/3/4.
//

#include <backends/ispc/ispc_ast_analysis.h>

namespace luisa::compute::ispc {

void ISPCVariableDefinitionAnalysis::visit(const BreakStmt *stmt) {}
void ISPCVariableDefinitionAnalysis::visit(const ContinueStmt *stmt) {}
void ISPCVariableDefinitionAnalysis::visit(const ReturnStmt *stmt) {}

void ISPCVariableDefinitionAnalysis::visit(const ScopeStmt *stmt) {
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
    auto variables = std::move(record.variables());
    luisa::unordered_map<Variable, size_t, VariableHash> counters;
    for (auto s : record.children()) {
        for (auto &&v : _scoped_variables.at(s)) {
            counters.try_emplace(v, 0u).first->second++;
        }
    }
    for (auto &&[v, count] : counters) {
        if (count > 1u) {
            variables.emplace(v);
        }
    }
    _scoped_variables.emplace(stmt, std::move(variables));
}

void ISPCVariableDefinitionAnalysis::visit(const IfStmt *stmt) {
    stmt->true_branch()->accept(*this);
    stmt->false_branch()->accept(*this);
}

void ISPCVariableDefinitionAnalysis::visit(const LoopStmt *stmt) {
    stmt->body()->accept(*this);
}

void ISPCVariableDefinitionAnalysis::visit(const ExprStmt *stmt) {}

void ISPCVariableDefinitionAnalysis::visit(const SwitchStmt *stmt) {
    for (auto s : stmt->body()->statements()) {
        s->accept(*this);
    }
}

void ISPCVariableDefinitionAnalysis::visit(const SwitchCaseStmt *stmt) {
    stmt->body()->accept(*this);
}

void ISPCVariableDefinitionAnalysis::visit(const SwitchDefaultStmt *stmt) {
    stmt->body()->accept(*this);
}

void ISPCVariableDefinitionAnalysis::visit(const AssignStmt *stmt) {
    _define(stmt->lhs());
}

void ISPCVariableDefinitionAnalysis::visit(const ForStmt *stmt) {
    stmt->body()->accept(*this);
}

void ISPCVariableDefinitionAnalysis::visit(const CommentStmt *stmt) {}

void ISPCVariableDefinitionAnalysis::visit(const MetaStmt *stmt) {
    stmt->scope()->accept(*this);
}

void ISPCVariableDefinitionAnalysis::analyze(Function f) noexcept {
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

void ISPCVariableDefinitionAnalysis::reset() noexcept {
    _scoped_variables.clear();
    _arguments.clear();
}

void ISPCVariableDefinitionAnalysis::_define(const Expression *expr) noexcept {
    if (auto ref = dynamic_cast<const RefExpr *>(expr)) {
        if (auto v = ref->variable();
            v.tag() == Variable::Tag::LOCAL &&
            _arguments.find(v.uid()) == _arguments.cend()) {
            _scope_stack.back().def(v);
        }
    } else if (auto member = dynamic_cast<const MemberExpr *>(expr)) {
        _define(member->self());
    } else if (auto access = dynamic_cast<const AccessExpr *>(expr)) {
        _define(access->range());
    }
}

}// namespace luisa::compute::ispc
