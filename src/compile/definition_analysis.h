//
// Created by Mike on 2022/3/4.
//

#pragma once

#include <core/stl.h>
#include <ast/interface.h>

namespace luisa::compute {

class DefinitionAnalysis final : public StmtVisitor, public ExprVisitor {

public:
    using VariableSet = luisa::unordered_set<Variable>;
    using ScopeSet = luisa::unordered_set<const ScopeStmt *, pointer_hash<ScopeStmt>>;
    using VariableScopeMap = luisa::unordered_map<Variable, ScopeSet>;
    using ScopedVariableMap = luisa::unordered_map<const ScopeStmt *, VariableSet, pointer_hash<ScopeStmt>>;

    class ScopeRecord {

    private:
        const ScopeStmt *_scope;
        VariableSet _variables;
        VariableScopeMap _propagated;

    public:
        explicit ScopeRecord(const ScopeStmt *scope) noexcept : _scope{scope} {}
        void reference(Variable v) noexcept;
        void propagate(Variable v, const ScopeStmt *scope);
        void finalize() noexcept;
        [[nodiscard]] auto scope() const noexcept { return _scope; }
        [[nodiscard]] auto &variables() noexcept { return _variables; }
        [[nodiscard]] auto &variables() const noexcept { return _variables; }
    };

private:
    luisa::unordered_set<uint> _arguments;
    luisa::vector<ScopeRecord> _scope_stack;
    ScopedVariableMap _scoped_variables;

private:
    void _require_definition(const Expression *expr) noexcept;

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

}// namespace luisa::compute