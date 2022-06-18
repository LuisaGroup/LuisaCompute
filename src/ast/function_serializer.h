//
// Created by Mike Smith on 2022/6/17.
//

#pragma once

#include <core/json_fwd.h>
#include <ast/variable.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <ast/function.h>

namespace luisa::compute {

namespace detail {
class FunctionBuilder;
}

class FunctionSerializer {

private:
    mutable json *_constants{};
    mutable json *_functions{};

private:
    [[nodiscard]] json dump(const UnaryExpr *expr) const noexcept;
    [[nodiscard]] json dump(const BinaryExpr *expr) const noexcept;
    [[nodiscard]] json dump(const MemberExpr *expr) const noexcept;
    [[nodiscard]] json dump(const AccessExpr *expr) const noexcept;
    [[nodiscard]] json dump(const LiteralExpr *expr) const noexcept;
    [[nodiscard]] json dump(const RefExpr *expr) const noexcept;
    [[nodiscard]] json dump(const ConstantExpr *expr) const noexcept;
    [[nodiscard]] json dump(const CallExpr *expr) const noexcept;
    [[nodiscard]] json dump(const CastExpr *expr) const noexcept;
    [[nodiscard]] json dump(const BreakStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const ContinueStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const ReturnStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const ScopeStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const IfStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const LoopStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const ExprStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const SwitchStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const SwitchCaseStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const SwitchDefaultStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const AssignStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const ForStmt *stmt) const noexcept;
    [[nodiscard]] json dump(const CommentStmt *stmt) const noexcept;
    [[nodiscard]] json dump_stmt(const Statement *stmt) const noexcept;
    [[nodiscard]] json dump_expr(const Expression *expr) const noexcept;
    void dump(const ConstantData &c) const noexcept;
    void dump(Function f) const noexcept;

public:
    FunctionSerializer() noexcept;
    ~FunctionSerializer() noexcept;
    [[nodiscard]] json serialize(Function function) const noexcept;
    [[nodiscard]] json serialize(const luisa::shared_ptr<const detail::FunctionBuilder> &function) const noexcept;
    [[nodiscard]] luisa::shared_ptr<detail::FunctionBuilder> deserialize(const json &json) const noexcept;
};

}// namespace luisa::compute
