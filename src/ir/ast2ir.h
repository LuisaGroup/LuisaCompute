//
// Created by Mike Smith on 2022/10/17.
//

#pragma once

#include <ast/variable.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <ast/function.h>

#include <rust/ir.hpp>

namespace luisa::compute {

namespace detail {
class FunctionBuilder;
}// namespace detail

class LC_IR_API AST2IR {

private:
    class IrBuilderGuard {

    private:
        AST2IR *_self;
        const ir::IrBuilder *_builder;

    public:
        IrBuilderGuard(AST2IR *self, ir::IrBuilder *builder) noexcept;
        ~IrBuilderGuard() noexcept;
    };

private:
    luisa::unordered_map<uint64_t, ir::CArc<ir::Type>> _struct_types;// maps Type::hash() to ir::Type
    luisa::unordered_map<uint64_t, ir::NodeRef> _constants;          // maps Constant::hash() to ir::NodeRef
    luisa::unordered_map<uint32_t, ir::NodeRef> _variables;          // maps Variable::uid to ir::NodeRef
    luisa::vector<ir::IrBuilder *> _builder_stack;
    Function _function;
    ir::CppOwnedCArc<ir::ModulePools> _pools;
    // how to add using is_avalanching = void; for ir::NodeRef in cbindgen?
    // or you'll have to use the inner size_t as key instead of ir::NodeRef itself. see src/core/stl/unordered_dense.h#L371
    luisa::unordered_map<size_t, ir::NodeRef> assign_map;
    [[nodiscard]] ir::NodeRef get_assign_rhs(ir::NodeRef lhs);

private:
    template<typename T>
    [[nodiscard]] auto _boxed_slice(size_t n) noexcept -> ir::CBoxedSlice<T>;

    template<typename Fn>
    auto _with_builder(Fn &&fn) noexcept;

private:
    [[nodiscard]] ir::IrBuilder *_current_builder() noexcept;
    [[nodiscard]] ir::CArc<ir::Type> _convert_type(const Type *type) noexcept;
    [[nodiscard]] ir::NodeRef _convert_argument(Variable v) noexcept;
    [[nodiscard]] ir::NodeRef _convert_builtin_variable(Variable v) noexcept;
    [[nodiscard]] ir::NodeRef _convert_shared_variable(Variable v) noexcept;
    [[nodiscard]] ir::NodeRef _convert_local_variable(Variable v) noexcept;
    [[nodiscard]] ir::NodeRef _convert_constant(const ConstantData &data) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const UnaryExpr *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const BinaryExpr *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const MemberExpr *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const AccessExpr *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const LiteralExpr *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const RefExpr *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const ConstantExpr *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const CallExpr *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const CastExpr *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const CpuCustomOpExpr *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const GpuCustomOpExpr *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert_expr(const Expression *expr) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const BreakStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const ContinueStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const ReturnStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const ScopeStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const IfStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const LoopStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const ExprStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const SwitchStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const SwitchCaseStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const SwitchDefaultStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const AssignStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const ForStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const CommentStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert(const AutoDiffStmt *stmt) noexcept;
    [[nodiscard]] ir::NodeRef _convert_stmt(const Statement *stmt) noexcept;
    [[nodiscard]] ir::Module _convert_body() noexcept;

    // helper functions
    [[nodiscard]] ir::NodeRef _cast(const Type *type_dst, const Type *type_src, ir::NodeRef node_src) noexcept;
    [[nodiscard]] ir::NodeRef _literal(const Type *type, LiteralExpr::Value value) noexcept;

public:
    [[nodiscard]] luisa::shared_ptr<ir::CArc<ir::KernelModule>> convert_kernel(Function function) noexcept;
    [[nodiscard]] ir::CArc<ir::CallableModule> convert_callable(Function function) noexcept;
    [[nodiscard]] static luisa::shared_ptr<ir::CArc<ir::KernelModule>> build_kernel(Function function) noexcept;
    [[nodiscard]] static ir::CArc<ir::CallableModule> build_callable(Function function) noexcept;
};

}// namespace luisa::compute
