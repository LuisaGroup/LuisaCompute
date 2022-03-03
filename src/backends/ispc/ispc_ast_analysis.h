//
// Created by Mike Smith on 2022/3/4.
//

#pragma once

#include <core/stl.h>
#include <ast/interface.h>

namespace luisa::compute::ispc {

class ISPCVariableDefinitionAnalysis final : public StmtVisitor, public ExprVisitor {

public:
    struct VariableHash {
        [[nodiscard]] auto operator()(Variable v) const noexcept { return v.hash(); }
    };

    using VariableSet = luisa::unordered_set<Variable, VariableHash>;
    using ScopedVariableMap = luisa::unordered_map<const ScopeStmt *, VariableSet>;

    class ScopeRecord {

    private:
        const ScopeStmt *_scope;
        VariableSet _variables;
        luisa::vector<const ScopeStmt *> _children;

    public:
        explicit ScopeRecord(const ScopeStmt *scope) noexcept : _scope{scope} {}
        void def(Variable v) noexcept { _variables.emplace(v); }
        void add(const ScopeStmt *s) noexcept { _children.emplace_back(s); }
        [[nodiscard]] auto scope() const noexcept { return _scope; }
        [[nodiscard]] auto &variables() noexcept { return _variables; }
        [[nodiscard]] auto &variables() const noexcept { return _variables; }
        [[nodiscard]] auto children() const noexcept { return luisa::span{_children}; }
    };

private:
    luisa::unordered_set<uint> _arguments;
    luisa::vector<ScopeRecord> _scope_stack;
    ScopedVariableMap _scoped_variables;

private:
    void _define(const Expression *expr) noexcept;

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

public:
    void analyze(Function f) noexcept;
    void reset() noexcept;
    [[nodiscard]] auto &scoped_variables() const noexcept {
        return _scoped_variables;
    }
    void visit(const UnaryExpr *expr) override;
    void visit(const BinaryExpr *expr) override;
    void visit(const MemberExpr *expr) override;
    void visit(const AccessExpr *expr) override;
    void visit(const LiteralExpr *expr) override;
    void visit(const RefExpr *expr) override;
    void visit(const ConstantExpr *expr) override;
    void visit(const CallExpr *expr) override;
    void visit(const CastExpr *expr) override;
};

}// namespace luisa::compute::ispc