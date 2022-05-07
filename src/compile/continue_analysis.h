//
// Created by Mike Smith on 2022/3/8.
//

#pragma once

#include <core/stl.h>
#include <ast/interface.h>

namespace luisa::compute {

class ContinueAnalysis final : public StmtVisitor {

private:
    luisa::unordered_map<const ContinueStmt *, const ScopeStmt *, PointerHash> _continue_scopes;
    luisa::vector<const ScopeStmt *> _scope_stack;

public:
    void visit(const BreakStmt *stmt) override;
    void visit(const ContinueStmt *stmt) override;
    void visit(const ReturnStmt *stmt) override;
    void visit(const ScopeStmt *stmt) override;
    void visit(const IfStmt *stmt) override;
    void visit(const LoopStmt *stmt) override;
    void visit(const ExprStmt *stmt) override;
    void visit(const SwitchStmt *stmt) override;
    void visit(const SwitchCaseStmt *stmt) override;
    void visit(const SwitchDefaultStmt *stmt) override;
    void visit(const AssignStmt *stmt) override;
    void visit(const ForStmt *stmt) override;
    void visit(const CommentStmt *stmt) override;
    void visit(const MetaStmt *stmt) override;
    void reset() noexcept { _continue_scopes.clear(); }
    void analyze(Function f) noexcept;
    [[nodiscard]] auto &continue_scopes() const noexcept { return _continue_scopes; }
};

}// namespace luisa::compute