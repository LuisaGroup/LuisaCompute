//
// Created by Mike on 2021/11/8.
//

#pragma once

#include <ast/function.h>
#include <ast/statement.h>
#include <ast/expression.h>
#include <compile/codegen.h>
#include <backends/ispc/ispc_accel.h>
#include <backends/ispc/ispc_bindless_array.h>
#include <backends/ispc/ispc_ast_analysis.h>

namespace luisa::compute::ispc {

/**
 * @brief Device code generator
 * 
 */
class ISPCCodegen final : public Codegen, private TypeVisitor, private ExprVisitor, private StmtVisitor {

public:
    static constexpr auto accel_handle_size = sizeof(ISPCAccel::Handle);
    static constexpr auto buffer_handle_size = sizeof(const void *);
    static constexpr auto texture_handle_size = 16;
    static constexpr auto bindless_array_handle_size = sizeof(ISPCBindlessArray::Handle);

private:
    Function _function;
    luisa::vector<Function> _generated_functions;
    luisa::vector<uint64_t> _generated_constants;
    uint32_t _indent{0u};
    ISPCVariableDefinitionAnalysis _definition_analysis;
    ISPCVariableDefinitionAnalysis::VariableSet _defined_variables;
    ISPCVariableDefinitionAnalysis::VariableSet _scope_defined_variables;

private:
    void visit(const Type *type) noexcept override;
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
    void visit(const IfStmt *stmt) override;
    void visit(const LoopStmt *stmt) override;
    void visit(const ExprStmt *stmt) override;
    void visit(const SwitchStmt *stmt) override;
    void visit(const SwitchCaseStmt *stmt) override;
    void visit(const SwitchDefaultStmt *stmt) override;
    void visit(const AssignStmt *stmt) override;
    void visit(const ForStmt *stmt) override;
    void visit(const ConstantExpr *expr) override;
    void visit(const CommentStmt *stmt) override;
    void visit(const MetaStmt *stmt) override;

private:
    void _emit_type_decl() noexcept;
    void _emit_variable_decl(Variable v, bool force_const) noexcept;
    void _emit_type_name(const Type *type) noexcept;
    void _emit_function(Function f) noexcept;
    void _emit_variable_name(Variable v) noexcept;
    void _emit_indent() noexcept;
    void _emit_statements(luisa::span<const Statement *const> stmts) noexcept;
    void _emit_constant(Function::Constant c) noexcept;
    void _emit_variable_declarations(const MetaStmt *meta) noexcept;
    void _emit_scoped_variables(const ScopeStmt *scope) noexcept;

public:
    /**
     * @brief Construct a new ISPCCodegen object
     * 
     * @param scratch generated code
     */
    explicit ISPCCodegen(Codegen::Scratch &scratch) noexcept : Codegen{scratch} {}
    /**
     * @brief Emit a function
     * 
     * @param f function
     */
    void emit(Function f) override;
};

}// namespace luisa::compute::ispc
