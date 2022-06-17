//
// Created by Mike Smith on 2022/6/17.
//

#include <nlohmann/json.hpp>

#include <ast/function_builder.h>
#include <ast/function_serializer.h>

namespace luisa::compute {

FunctionSerializer::FunctionSerializer() noexcept = default;
FunctionSerializer::~FunctionSerializer() noexcept = default;

nlohmann::json FunctionSerializer::dump(const UnaryExpr *expr) const noexcept {
    return {{"tag", "unary"},
            {"type", expr->type()->description()},
            {"operand", dump_expr(expr->operand())}};
}

nlohmann::json FunctionSerializer::dump(const BinaryExpr *expr) const noexcept {
    return {{"tag", "binary"},
            {"type", expr->type()->description()},
            {"lhs", dump_expr(expr->lhs())},
            {"rhs", dump_expr(expr->rhs())}};
}

nlohmann::json FunctionSerializer::dump(const MemberExpr *expr) const noexcept {
    return {{"tag", "member"},
            {"type", expr->type()->description()},
            {"self", dump_expr(expr->self())},
            {"code", expr->code()}};
}

nlohmann::json FunctionSerializer::dump(const AccessExpr *expr) const noexcept {
    return {{"tag", "access"},
            {"type", expr->type()->description()},
            {"range", dump_expr(expr->range())},
            {"index", dump_expr(expr->index())}};
}

nlohmann::json FunctionSerializer::dump(const LiteralExpr *expr) const noexcept {
    auto value = luisa::visit(
        [](auto v) noexcept -> nlohmann::json {
            using T = std::remove_cvref_t<decltype(v)>;
            if constexpr (luisa::is_scalar_v<T>) {
                return v;
            } else if constexpr (luisa::is_vector_v<T>) {
                constexpr auto dim = vector_dimension_v<T>;
                std::array<vector_element_t<T>, dim> a{};
                for (auto i = 0u; i < dim; i++) { a[i] = v[i]; }
                return a;
            } else if constexpr (luisa::is_matrix_v<T>) {
                constexpr auto dim = matrix_dimension_v<T>;
                std::array<float, dim * dim> a{};
                for (auto j = 0u; j < dim; j++) {
                    for (auto i = 0u; i < dim; i++) {
                        a[j * dim + i] = v[j][i];
                    }
                }
                return a;
            } else {
                static_assert(always_false_v<T>);
            }
        },
        expr->value());
    return {{"tag", "literal"},
            {"type", expr->type()->description()},
            {"value", value}};
}

nlohmann::json FunctionSerializer::dump(const RefExpr *expr) const noexcept {
    return {{"tag", "ref"},
            {"type", expr->type()->description()},
            {"variable", expr->variable().uid()}};
}

nlohmann::json FunctionSerializer::dump(const ConstantExpr *expr) const noexcept {
    dump(expr->data());
    return {{"tag", "constant"},
            {"type", expr->type()->description()},
            {"key", luisa::format("{:016x}", expr->data().hash())}};
}

nlohmann::json FunctionSerializer::dump(const CallExpr *expr) const noexcept {
    auto call = ::nlohmann::json::object();
    call.emplace("tag", "call");
    call.emplace("type", expr->type() ? expr->type()->description() : "void");
    std::vector<::nlohmann::json> args;
    args.reserve(expr->arguments().size());
    for (const auto &arg : expr->arguments()) { args.emplace_back(dump_expr(arg)); }
    call.emplace("arguments", std::move(args));
    expr->is_builtin() ? call.emplace("op", to_underlying(expr->op())) :
                         call.emplace("custom", luisa::format("{:016x}", expr->custom().hash()));
    return call;
}

nlohmann::json FunctionSerializer::dump(const CastExpr *expr) const noexcept {
    return {{"tag", "cast"},
            {"op", expr->op() == CastOp::STATIC ? "static" : "bitwise"},
            {"type", expr->type()->description()},
            {"operand", dump_expr(expr->expression())}};
}

nlohmann::json FunctionSerializer::dump(const BreakStmt *) const noexcept {
    return {{"tag", "break"}};
}

nlohmann::json FunctionSerializer::dump(const ContinueStmt *) const noexcept {
    return {{"tag", "continue"}};
}

nlohmann::json FunctionSerializer::dump(const ReturnStmt *stmt) const noexcept {
    auto s = ::nlohmann::json::object();
    s.emplace("tag", "return");
    if (auto v = stmt->expression()) { s.emplace("value", dump_expr(v)); }
    return s;
}

nlohmann::json FunctionSerializer::dump(const ScopeStmt *stmt) const noexcept {
    std::vector<::nlohmann::json> stmts;
    stmts.reserve(stmt->statements().size());
    for (auto s : stmt->statements()) { stmts.emplace_back(dump_stmt(s)); }
    return {{"tag", "scope"}, {"statements", std::move(stmts)}};
}

nlohmann::json FunctionSerializer::dump(const IfStmt *stmt) const noexcept {
    return {{"tag", "if"},
            {"condition", dump_expr(stmt->condition())},
            {"true", dump(stmt->true_branch())},
            {"false", dump(stmt->false_branch())}};
}

nlohmann::json FunctionSerializer::dump(const LoopStmt *stmt) const noexcept {
    return {{"tag", "loop"},
            {"body", dump(stmt->body())}};
}

nlohmann::json FunctionSerializer::dump(const ExprStmt *stmt) const noexcept {
    return {{"tag", "expr"},
            {"expression", dump_expr(stmt->expression())}};
}

nlohmann::json FunctionSerializer::dump(const SwitchStmt *stmt) const noexcept {
    return {{"tag", "switch"},
            {"expression", dump_expr(stmt->expression())},
            {"body", dump(stmt->body())}};
}

nlohmann::json FunctionSerializer::dump(const SwitchCaseStmt *stmt) const noexcept {
    return {{"tag", "switch.case"},
            {"value", dump_expr(stmt->expression())},
            {"body", dump(stmt->body())}};
}

nlohmann::json FunctionSerializer::dump(const SwitchDefaultStmt *stmt) const noexcept {
    return {{"tag", "switch.default"},
            {"body", dump(stmt->body())}};
}

nlohmann::json FunctionSerializer::dump(const AssignStmt *stmt) const noexcept {
    return {{"tag", "assign"},
            {"lhs", dump_expr(stmt->lhs())},
            {"rhs", dump_expr(stmt->rhs())}};
}

nlohmann::json FunctionSerializer::dump(const ForStmt *stmt) const noexcept {
    return {{"tag", "for"},
            {"variable", dump_expr(stmt->variable())},
            {"condition", dump_expr(stmt->condition())},
            {"step", dump_expr(stmt->step())},
            {"body", dump(stmt->body())}};
}

nlohmann::json FunctionSerializer::dump(const CommentStmt *stmt) const noexcept {
    return {{"tag", "comment"}, {"string", stmt->comment()}};
}

void FunctionSerializer::dump(const ConstantData &c) const noexcept {
    auto key = luisa::format("{:016x}", c.hash());
    if (_constants->contains(key.c_str())) { return; }
    auto &obj = _constants->emplace(key.c_str(), nlohmann::json::object()).first.value();
    luisa::visit(
        [&](auto view) noexcept {
            using T = std::remove_cvref_t<decltype(view[0])>;
            obj.emplace("type", Type::of<T>()->description());
            auto data = reinterpret_cast<const uint8_t *>(view.data());
            auto size = view.size_bytes();
            std::string s;
            s.reserve(size * 2);
            static constexpr auto hex_digits = "0123456789abcdef";
            for (size_t i = 0; i < size; ++i) {
                s.push_back(hex_digits[(data[i] >> 4u) & 0x0fu]);
                s.push_back(hex_digits[data[i] & 0x0fu]);
            }
            obj.emplace("data", s);
        },
        c.view());
}

void FunctionSerializer::dump(Function f) const noexcept {
    auto key = luisa::format("{:016x}", f.hash());
    if (_functions->contains(key.c_str())) { return; }
    auto &func = _functions->emplace(key.c_str(), nlohmann::json::object()).first.value();
    auto is_kernel = f.tag() == Function::Tag::KERNEL;
    func.emplace("tag", is_kernel ? "kernel" : "callable");
    if (!is_kernel) { func.emplace("return", f.return_type() ? f.return_type()->description() : "void"); }
    func.emplace("block_size", std::array{f.block_size().x, f.block_size().y, f.block_size().z});
    std::vector<::nlohmann::json> variables(f.builder()->_variable_usages.size(), ::nlohmann::json::object());
    for (auto i = 0u; i < f.arguments().size(); i++) {
        using BufferBinding = detail::FunctionBuilder::BufferBinding;
        using TextureBinding = detail::FunctionBuilder::TextureBinding;
        using BindlessArrayBinding = detail::FunctionBuilder::BindlessArrayBinding;
        using AccelBinding = detail::FunctionBuilder::AccelBinding;
        auto v = f.arguments()[i];
        auto binding = f.builder()->argument_bindings()[i];
        auto &&var = variables[v.uid()];
        var.emplace("type", v.type()->description());
        switch (v.tag()) {
            case Variable::Tag::REFERENCE: var.emplace("tag", "reference"); break;
            case Variable::Tag::BUFFER:
                var.emplace("tag", "buffer");
                if (auto b = luisa::get_if<BufferBinding>(&binding)) {
                    var.emplace("binding", nlohmann::json{{"handle", b->handle},
                                                          {"offset", b->offset_bytes},
                                                          {"size", b->size_bytes}});
                }
                break;
            case Variable::Tag::TEXTURE:
                var.emplace("tag", "texture");
                if (auto b = luisa::get_if<TextureBinding>(&binding)) {
                    var.emplace("binding", nlohmann::json{{"handle", b->handle},
                                                          {"level", b->level}});
                }
                break;
            case Variable::Tag::BINDLESS_ARRAY:
                var.emplace("tag", "bindless_array");
                if (auto b = luisa::get_if<BindlessArrayBinding>(&binding)) {
                    var.emplace("binding", nlohmann::json{{"handle", b->handle}});
                }
                break;
            case Variable::Tag::ACCEL:
                var.emplace("tag", "accel");
                if (auto b = luisa::get_if<AccelBinding>(&binding)) {
                    var.emplace("binding", nlohmann::json{{"handle", b->handle}});
                }
                break;
            default: var.emplace("tag", "argument"); break;
        }
    }
    for (auto v : f.builtin_variables()) {
        switch (v.tag()) {
            case Variable::Tag::THREAD_ID:
                variables[v.uid()].emplace("tag", "thread_id");
                break;
            case Variable::Tag::BLOCK_ID:
                variables[v.uid()].emplace("tag", "block_id");
                break;
            case Variable::Tag::DISPATCH_ID:
                variables[v.uid()].emplace("tag", "dispatch_id");
                break;
            case Variable::Tag::DISPATCH_SIZE:
                variables[v.uid()].emplace("tag", "dispatch_size");
                break;
            default: break;
        }
    }
    for (auto v : f.shared_variables()) {
        variables[v.uid()].emplace("tag", "shared");
        variables[v.uid()].emplace("type", v.type()->description());
    }
    for (auto v : f.local_variables()) {
        variables[v.uid()].emplace("tag", "local");
        variables[v.uid()].emplace("type", v.type()->description());
    }
    func.emplace("variables", std::move(variables));
    func.emplace("body", dump(f.body()));
}

nlohmann::json FunctionSerializer::serialize(Function function) const noexcept {
    auto module = nlohmann::json::object();
    _constants = &module.emplace("constants", nlohmann::json::object()).first.value();
    _functions = &module.emplace("functions", nlohmann::json::object()).first.value();
    dump(function);
    _constants = nullptr;
    _functions = nullptr;
    return module;
}

luisa::shared_ptr<detail::FunctionBuilder> FunctionSerializer::deserialize(const nlohmann::json &json) const noexcept {
    return nullptr;
}

nlohmann::json FunctionSerializer::serialize(const shared_ptr<const detail::FunctionBuilder> &function) const noexcept {
    return serialize(function->function());
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-static-cast-downcast"
nlohmann::json FunctionSerializer::dump_stmt(const Statement *stmt) const noexcept {
    switch (stmt->tag()) {
        case Statement::Tag::BREAK: return dump(static_cast<const BreakStmt *>(stmt));
        case Statement::Tag::CONTINUE: return dump(static_cast<const ContinueStmt *>(stmt));
        case Statement::Tag::RETURN: return dump(static_cast<const ReturnStmt *>(stmt));
        case Statement::Tag::SCOPE: return dump(static_cast<const ScopeStmt *>(stmt));
        case Statement::Tag::IF: return dump(static_cast<const IfStmt *>(stmt));
        case Statement::Tag::LOOP: return dump(static_cast<const LoopStmt *>(stmt));
        case Statement::Tag::EXPR: return dump(static_cast<const ExprStmt *>(stmt));
        case Statement::Tag::SWITCH: return dump(static_cast<const SwitchStmt *>(stmt));
        case Statement::Tag::SWITCH_CASE: return dump(static_cast<const SwitchCaseStmt *>(stmt));
        case Statement::Tag::SWITCH_DEFAULT: return dump(static_cast<const SwitchDefaultStmt *>(stmt));
        case Statement::Tag::ASSIGN: return dump(static_cast<const AssignStmt *>(stmt));
        case Statement::Tag::FOR: return dump(static_cast<const ForStmt *>(stmt));
        case Statement::Tag::COMMENT: return dump(static_cast<const CommentStmt *>(stmt));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid statement.");
}

nlohmann::json FunctionSerializer::dump_expr(const Expression *expr) const noexcept {
    switch (expr->tag()) {
        case Expression::Tag::UNARY: return dump(static_cast<const UnaryExpr *>(expr));
        case Expression::Tag::BINARY: return dump(static_cast<const BinaryExpr *>(expr));
        case Expression::Tag::MEMBER: return dump(static_cast<const MemberExpr *>(expr));
        case Expression::Tag::ACCESS: return dump(static_cast<const AccessExpr *>(expr));
        case Expression::Tag::LITERAL: return dump(static_cast<const LiteralExpr *>(expr));
        case Expression::Tag::REF: return dump(static_cast<const RefExpr *>(expr));
        case Expression::Tag::CONSTANT: return dump(static_cast<const ConstantExpr *>(expr));
        case Expression::Tag::CALL: return dump(static_cast<const CallExpr *>(expr));
        case Expression::Tag::CAST: return dump(static_cast<const CastExpr *>(expr));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid expression.");
}
#pragma clang diagnostic pop

}// namespace luisa::compute
