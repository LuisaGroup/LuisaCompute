#pragma once

#include <luisa/ast/function.h>
#include <luisa/ast/statement.h>
#include <luisa/ast/expression.h>
#include <luisa/core/string_scratch.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::metal {

class MetalConstantPrinter;

class MetalCodegenAST final : private ExprVisitor, private StmtVisitor {

    friend class MetalConstantPrinter;

private:
    StringScratch &_scratch;
    Function _function;
    uint _indention{0u};
    struct ReferenceTemporary {
        const Expression *expression;
        luisa::string name;
    };
    struct LocalVariableEmission {
        const ScopeStmt *scope{nullptr};
        const AssignStmt *initializer{nullptr};
    };
    luisa::vector<ReferenceTemporary> _reference_temporaries;
    uint _next_reference_temporary{0u};
    luisa::unordered_map<uint, LocalVariableEmission> _local_variable_emissions;
    luisa::unordered_map<const ScopeStmt *, luisa::vector<Variable>> _scope_local_variables;
    luisa::unordered_map<const AssignStmt *, Variable> _local_variable_initializers;
    luisa::unordered_set<Variable> _gradient_variables;
    luisa::unordered_map<uint64_t, luisa::unordered_set<uint32_t>> _sampled_texture_variables;
    luisa::vector<uint8_t> _argument_sampled;
    bool _uses_printing{false};
    const Type *_ray_type;
    const Type *_triangle_hit_type;
    const Type *_procedural_hit_type;
    const Type *_committed_hit_type;
    const Type *_ray_query_all_type;
    const Type *_ray_query_any_type;
    const Type *_indirect_dispatch_buffer_type;
    luisa::vector<std::pair<luisa::string, const Type *>> _print_formats;

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
    void visit(const ConstantExpr *expr) noexcept override;
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
    void visit(const CommentStmt *stmt) noexcept override;
    void visit(const RayQueryStmt *stmt) noexcept override;
    void visit(const SuspendStmt *stmt) noexcept override;

    void visit(const PrintStmt *stmt) noexcept override;
    void visit(const CpuCustomOpExpr *expr) noexcept override;
    void visit(const GpuCustomOpExpr *expr) noexcept override;

private:
    void _emit_type_decls(Function kernel) noexcept;
    void _emit_type_name(const Type *type, Usage usage = Usage::READ_WRITE,
                         bool sampled = false) noexcept;
    void _emit_variable_name(Variable v, bool sampled = false) noexcept;
    void _analyze_local_variables() noexcept;
    void _emit_scope_local_variables(const ScopeStmt *scope) noexcept;
    void _emit_assignment_lhs(const AssignStmt *stmt) noexcept;
    void _emit_function() noexcept;
    void _emit_constant(const Function::Constant &c) noexcept;
    void _emit_indention() noexcept;
    void _emit_access_chain(luisa::span<const Expression *const> chain) noexcept;
    void _analyze_sampled_textures(Function function) noexcept;
    [[nodiscard]] bool _is_texture_sampled(Function function, Variable variable) const noexcept;
    [[nodiscard]] bool _is_texture_sampled(Variable variable) const noexcept;
    [[nodiscard]] bool _is_texture_written(Function function, Variable variable) const noexcept;
    void _emit_call_argument(const Expression *argument, bool sampled) noexcept;
    void _emit_swizzle_reference_temporaries(const Expression *expression) noexcept;
    void _emit_swizzle_reference_writebacks(size_t first) noexcept;

public:
    explicit MetalCodegenAST(StringScratch &scratch) noexcept;
    void emit(Function kernel, luisa::string_view native_include) noexcept;
    [[nodiscard]] static size_t type_size_bytes(const Type *type) noexcept;

public:
    [[nodiscard]] auto print_formats() const noexcept { return luisa::span{_print_formats}; }
    [[nodiscard]] auto argument_sampled() const noexcept { return luisa::span{_argument_sampled}; }
};

}// namespace luisa::compute::metal
