//
// Created by Mike Smith on 2021/3/5.
//

#pragma once

#include <string>
#include <ast/function.h>
#include <ast/statement.h>
#include <ast/expression.h>

namespace luisa::compute {

class Codegen {

public:
    class Scratch {

    private:
        std::string _buffer;

    public:
        Scratch() noexcept;
        Scratch &operator<<(bool x) noexcept;
        Scratch &operator<<(float x) noexcept;
        Scratch &operator<<(int x) noexcept;
        Scratch &operator<<(uint x) noexcept;
        Scratch &operator<<(size_t x) noexcept;
        Scratch &operator<<(std::string_view s) noexcept;
        Scratch &operator<<(const char *s) noexcept;
        Scratch &operator<<(const std::string &s) noexcept;
        [[nodiscard]] std::string_view view() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] size_t size() const noexcept;
        void pop_back() noexcept;
        void clear() noexcept;
    };

protected:
    Scratch &_scratch;

public:
    explicit Codegen(Scratch &scratch) noexcept : _scratch{scratch} {}
    virtual void emit(Function f) = 0;
};

class CppCodegen : public Codegen, private TypeVisitor, private ExprVisitor, private StmtVisitor {

private:
    std::vector<uint32_t> _generated;
    uint32_t _function_uid{0u};
    uint32_t _indent{0u};

private:
    void visit(const UnaryExpr *expr) override;
    void visit(const BinaryExpr *expr) override;
    void visit(const MemberExpr *expr) override;
    void visit(const AccessExpr *expr) override;
    void visit(const LiteralExpr *expr) override;
    void visit(const RefExpr *expr) override;
    void visit(const CallExpr *expr) override;
    void visit(const CastExpr *expr) override;
    void visit(const BreakStmt *stmt) override;
    void visit(const ContinueStmt *stmt) override;
    void visit(const ReturnStmt *stmt) override;
    void visit(const ScopeStmt *stmt) override;
    void visit(const DeclareStmt *stmt) override;
    void visit(const IfStmt *stmt) override;
    void visit(const WhileStmt *stmt) override;
    void visit(const Type *type) noexcept override;
    void visit(const ExprStmt *stmt) override;
    void visit(const SwitchStmt *stmt) override;
    void visit(const SwitchCaseStmt *stmt) override;
    void visit(const SwitchDefaultStmt *stmt) override;
    void visit(const AssignStmt *stmt) override;

private:
    virtual void _emit_type_decl() noexcept;
    virtual void _emit_variable_decl(Variable v) noexcept;
    virtual void _emit_type_name(const Type *type) noexcept;
    virtual void _emit_function(Function f) noexcept;
    virtual void _emit_variable_name(Variable v) noexcept;
    virtual void _emit_indent() noexcept;
    virtual void _emit_statements(std::span<const Statement *const> stmts) noexcept;
    virtual void _emit_constant(Function::ConstantData c) noexcept;

public:
    explicit CppCodegen(Codegen::Scratch &scratch) noexcept : Codegen{scratch} {}
    void emit(Function f) override;
};

}// namespace luisa::compute
