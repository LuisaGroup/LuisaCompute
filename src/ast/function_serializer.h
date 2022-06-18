//
// Created by Mike Smith on 2022/6/17.
//

#pragma once

#include <core/json_fwd.h>
#include <core/binary_buffer.h>
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
    mutable const json *_serialized_functions{};
    mutable const json *_serialized_constants{};
    mutable luisa::vector<luisa::vector<const RefExpr *>> _variable_stack;
    mutable luisa::unordered_map<luisa::string, luisa::shared_ptr<const detail::FunctionBuilder>> _parsed_functions;
    mutable luisa::unordered_map<luisa::string, ConstantData> _parsed_constants;

private:
    // dump to json
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
    // parse from json
    [[nodiscard]] const UnaryExpr *parse_unary_expr(const json &j) const noexcept;
    [[nodiscard]] const BinaryExpr *parse_binary_expr(const json &j) const noexcept;
    [[nodiscard]] const MemberExpr *parse_member_expr(const json &j) const noexcept;
    [[nodiscard]] const AccessExpr *parse_access_expr(const json &j) const noexcept;
    [[nodiscard]] const LiteralExpr *parse_literal_expr(const json &j) const noexcept;
    [[nodiscard]] const RefExpr *parse_ref_expr(const json &j) const noexcept;
    [[nodiscard]] const ConstantExpr *parse_constant_expr(const json &j) const noexcept;
    [[nodiscard]] const CallExpr *parse_call_expr(const json &j) const noexcept;
    [[nodiscard]] const CastExpr *parse_cast_expr(const json &j) const noexcept;
    void parse_break_stmt(const json &j) const noexcept;
    void parse_continue_stmt(const json &j) const noexcept;
    void parse_return_stmt(const json &j) const noexcept;
    void parse_if_stmt(const json &j) const noexcept;
    void parse_loop_stmt(const json &j) const noexcept;
    void parse_expr_stmt(const json &j) const noexcept;
    void parse_switch_stmt(const json &j) const noexcept;
    void parse_switch_case_stmt(const json &j) const noexcept;
    void parse_switch_default_stmt(const json &j) const noexcept;
    void parse_assign_stmt(const json &j) const noexcept;
    void parse_for_stmt(const json &j) const noexcept;
    void parse_comment_stmt(const json &j) const noexcept;
    [[nodiscard]] const Expression *parse_expr(const json &j) const noexcept;
    void parse_stmt(const json &j) const noexcept;
    void parse_constant(luisa::string_view key) const noexcept;
    void parse_function(luisa::string_view key) const noexcept;

public:
    FunctionSerializer() noexcept;
    ~FunctionSerializer() noexcept;
    [[nodiscard]] json to_json(Function function) const noexcept;
    [[nodiscard]] json to_json(const luisa::shared_ptr<const detail::FunctionBuilder> &function) const noexcept;
    [[nodiscard]] luisa::shared_ptr<const detail::FunctionBuilder> from_json(const json &json) const noexcept;
    [[nodiscard]] BinaryBuffer to_binary(Function function) const noexcept;
    [[nodiscard]] BinaryBuffer to_binary(const luisa::shared_ptr<const detail::FunctionBuilder> &function) const noexcept;
    void to_binary(BinaryBuffer &buffer, Function function) const noexcept;
    void to_binary(BinaryBuffer &buffer, const luisa::shared_ptr<const detail::FunctionBuilder> &function) const noexcept;
    [[nodiscard]] luisa::shared_ptr<const detail::FunctionBuilder> from_binary(BinaryBufferReader reader) const noexcept;
};

}// namespace luisa::compute
