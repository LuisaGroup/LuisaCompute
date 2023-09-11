#pragma once

#include <luisa/ast/function.h>
#include <luisa/ast/statement.h>
#include <luisa/ast/expression.h>
#include "../common/string_scratch.h"

namespace luisa::compute::metal {

class MetalConstantPrinter;

class MetalCodegenAST final : private ExprVisitor, private StmtVisitor {

    friend class MetalConstantPrinter;

private:
    StringScratch &_scratch;
    Function _function;
    uint _indention{0u};
    const Type *_ray_type;
    const Type *_triangle_hit_type;
    const Type *_procedural_hit_type;
    const Type *_committed_hit_type;
    const Type *_ray_query_all_type;
    const Type *_ray_query_any_type;
    const Type *_indirect_dispatch_buffer_type;

private:
    void visit(const UnaryExpr *expr) noexcept override;
    void visit(const BinaryExpr *expr) noexcept override;
    void visit(const MemberExpr *expr) noexcept override;
    void visit(const AccessExpr *expr) noexcept override;
    void visit(const LiteralExpr *expr) noexcept override;
    void visit(const RefExpr *expr) noexcept override;
    void visit(const CallExpr *expr) noexcept override;
    void visit(const CastExpr *expr) noexcept override;
    void visit(const TypeIDExpr *expr) noexcept override;
    void visit(const StringIDExpr *expr) noexcept override;
    void visit(const BreakStmt *stmt) noexcept override;
    void visit(const ContinueStmt *stmt) noexcept override;
    void visit(const ReturnStmt *stmt) noexcept override;
    void visit(const ScopeStmt *stmt) noexcept override;
    void visit(const IfStmt *stmt) noexcept override;
    void visit(const LoopStmt *stmt) noexcept override;
    void visit(const ExprStmt *stmt) noexcept override;
    void visit(const SwitchStmt *stmt) noexcept override;
    void visit(const SwitchCaseStmt *stmt) noexcept override;
    void visit(const SwitchDefaultStmt *stmt) noexcept override;
    void visit(const AutoDiffStmt *stmt) noexcept override;
    void visit(const AssignStmt *stmt) noexcept override;
    void visit(const ForStmt *stmt) noexcept override;
    void visit(const ConstantExpr *expr) noexcept override;
    void visit(const CommentStmt *stmt) noexcept override;
    void visit(const RayQueryStmt *stmt) noexcept override;
    void visit(const CpuCustomOpExpr *expr) noexcept override;
    void visit(const GpuCustomOpExpr *expr) noexcept override;

private:
    void _emit_type_decls(Function kernel) noexcept;
    void _emit_type_name(const Type *type, Usage usage = Usage::READ_WRITE) noexcept;
    void _emit_variable_name(Variable v) noexcept;
    void _emit_function() noexcept;
    void _emit_constant(const Function::Constant &c) noexcept;
    void _emit_indention() noexcept;
    void _emit_access_chain(luisa::span<const Expression *const> chain) noexcept;

public:
    explicit MetalCodegenAST(StringScratch &scratch) noexcept;
    void emit(Function kernel, luisa::string_view native_include) noexcept;
    [[nodiscard]] static size_t type_size_bytes(const Type *type) noexcept;
};

}// namespace luisa::compute::metal

