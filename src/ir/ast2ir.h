//
// Created by Mike Smith on 2022/10/17.
//

#pragma once

#include <ast/variable.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <ast/function.h>

#include <luisa_compute_ir/bindings.hpp>

namespace luisa::compute {

namespace detail {
class FunctionBuilder;
}

class LC_IR_API AST2IR {

private:
    class IrBuilderGuard {

    private:
        AST2IR *_self;
        const ir::IrBuilder *_builder;

    public:
<<<<<<< HEAD
        IrBuilderGuard(AST2IR *self, ir::IrBuilder *builder) noexcept
            : _self{self}, _builder{builder} {
            self->_builder_stack.emplace_back(builder);
        }
=======
        IrBuilderGuard(AST2IR *self, ir::IrBuilder *builder) noexcept;
>>>>>>> autodiff
        ~IrBuilderGuard() noexcept;
    };

private:
    luisa::unordered_map<uint64_t, ir::Gc<ir::Type>> _struct_types;// maps Type::hash() to ir::Type
    luisa::unordered_map<uint64_t, ir::NodeRef> _constants;        // maps Constant::hash() to ir::NodeRef
    luisa::unordered_map<uint32_t, ir::NodeRef> _variables;        // maps Variable::uid to ir::NodeRef
    luisa::vector<ir::IrBuilder *> _builder_stack;
    Function _function;

private:
    template<typename T>
<<<<<<< HEAD
    [[nodiscard]] auto _boxed_slice(size_t n) noexcept -> ir::CBoxedSlice<T> {
        if (n == 0u) {
            return {.ptr = nullptr,
                    .len = 0u,
                    .destructor = [](T *, size_t) noexcept {}};
        }
        return {.ptr = luisa::allocate_with_allocator<T>(n),
                .len = n,
                .destructor = [](T *ptr, size_t) noexcept { luisa::deallocate_with_allocator(ptr); }};
    }
=======
    [[nodiscard]] auto _boxed_slice(size_t n) noexcept -> ir::CBoxedSlice<T>;
>>>>>>> autodiff

    template<typename Fn>
    auto _with_builder(Fn &&fn) noexcept;

private:
    [[nodiscard]] ir::IrBuilder *_current_builder() noexcept;
    [[nodiscard]] ir::Gc<ir::Type> _convert_type(const Type *type) noexcept;
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
    [[nodiscard]] ir::NodeRef _convert_stmt(const Statement *stmt) noexcept;
    [[nodiscard]] ir::Module _convert_body() noexcept;

    // helper functions
    [[nodiscard]] ir::NodeRef _cast(const Type *type_dst, const Type *type_src, ir::NodeRef node_src) noexcept;
    [[nodiscard]] ir::NodeRef _literal(const Type *type, LiteralExpr::Value value) noexcept;

public:
    [[nodiscard]] luisa::shared_ptr<ir::Gc<ir::KernelModule>> convert_kernel(Function function) noexcept;
    [[nodiscard]] ir::Gc<ir::CallableModule> convert_callable(Function function) noexcept;
};

}// namespace luisa::compute
