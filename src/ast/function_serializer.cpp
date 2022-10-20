//
// Created by Mike Smith on 2022/6/17.
//

#include <core/json.h>
#include <ast/function_builder.h>
#include <ast/function_serializer.h>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "NullDereference"
#pragma ide diagnostic ignored "misc-no-recursion"
#pragma ide diagnostic ignored "readability-convert-member-functions-to-static"
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-static-cast-downcast"

namespace luisa::compute {

FunctionSerializer::FunctionSerializer() noexcept = default;
FunctionSerializer::~FunctionSerializer() noexcept = default;

[[nodiscard]] inline auto function_builder() noexcept {
    return detail::FunctionBuilder::current();
}

json FunctionSerializer::dump(const UnaryExpr *expr) const noexcept {
    return {{"tag", "unary"},
            {"type", expr->type()->description()},
            {"op", to_underlying(expr->op())},
            {"operand", dump_expr(expr->operand())}};
}

json FunctionSerializer::dump(const BinaryExpr *expr) const noexcept {
    return {{"tag", "binary"},
            {"type", expr->type()->description()},
            {"op", to_underlying(expr->op())},
            {"lhs", dump_expr(expr->lhs())},
            {"rhs", dump_expr(expr->rhs())}};
}

json FunctionSerializer::dump(const MemberExpr *expr) const noexcept {
    json e{{"tag", "member"},
           {"type", expr->type()->description()},
           {"self", dump_expr(expr->self())}};
    if (expr->is_swizzle()) {
        e.emplace("swizzle", std::array{expr->swizzle_size(), expr->swizzle_code()});
    } else {
        e.emplace("index", expr->member_index());
    }
    return e;
}

json FunctionSerializer::dump(const AccessExpr *expr) const noexcept {
    return {{"tag", "access"},
            {"type", expr->type()->description()},
            {"self", dump_expr(expr->range())},
            {"index", dump_expr(expr->index())}};
}

json FunctionSerializer::dump(const LiteralExpr *expr) const noexcept {
    auto value = luisa::visit(
        [](auto v) noexcept -> json {
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

json FunctionSerializer::dump(const RefExpr *expr) const noexcept {
    return {{"tag", "ref"},
            {"type", expr->type()->description()},
            {"variable", expr->variable().uid()}};
}

json FunctionSerializer::dump(const ConstantExpr *expr) const noexcept {
    dump(expr->data());
    return {{"tag", "constant"},
            {"type", expr->type()->description()},
            {"key", luisa::format("{:016x}", expr->data().hash())}};
}

json FunctionSerializer::dump(const CallExpr *expr) const noexcept {
    auto call = json::object();
    call.emplace("tag", "call");
    call.emplace("type", expr->type() ? expr->type()->description() : "void");
    luisa::vector<json> args;
    args.reserve(expr->arguments().size());
    for (const auto &arg : expr->arguments()) { args.emplace_back(dump_expr(arg)); }
    call.emplace("arguments", std::move(args));
    if (expr->is_builtin()) {
        call.emplace("op", to_underlying(expr->op()));
    } else {
        dump(expr->custom());
        call.emplace("custom", luisa::format("{:016x}", expr->custom().hash()));
    }
    return call;
}

json FunctionSerializer::dump(const CastExpr *expr) const noexcept {
    return {{"tag", "cast"},
            {"op", expr->op() == CastOp::STATIC ? "static" : "bitwise"},
            {"type", expr->type()->description()},
            {"operand", dump_expr(expr->expression())}};
}

json FunctionSerializer::dump(const BreakStmt *) const noexcept {
    return {{"tag", "break"}};
}

json FunctionSerializer::dump(const ContinueStmt *) const noexcept {
    return {{"tag", "continue"}};
}

json FunctionSerializer::dump(const ReturnStmt *stmt) const noexcept {
    auto s = json::object();
    s.emplace("tag", "return");
    if (auto v = stmt->expression()) { s.emplace("value", dump_expr(v)); }
    return s;
}

json FunctionSerializer::dump(const ScopeStmt *stmt) const noexcept {
    luisa::vector<json, luisa::allocator<json>> stmts;
    stmts.reserve(stmt->statements().size());
    for (auto s : stmt->statements()) { stmts.emplace_back(dump_stmt(s)); }
    return {{"tag", "scope"}, {"statements", std::move(stmts)}};
}

json FunctionSerializer::dump(const IfStmt *stmt) const noexcept {
    return {{"tag", "if"},
            {"condition", dump_expr(stmt->condition())},
            {"true", dump(stmt->true_branch())},
            {"false", dump(stmt->false_branch())}};
}

json FunctionSerializer::dump(const LoopStmt *stmt) const noexcept {
    return {{"tag", "loop"},
            {"body", dump(stmt->body())}};
}

json FunctionSerializer::dump(const ExprStmt *stmt) const noexcept {
    return {{"tag", "expr"},
            {"expression", dump_expr(stmt->expression())}};
}

json FunctionSerializer::dump(const SwitchStmt *stmt) const noexcept {
    return {{"tag", "switch"},
            {"expression", dump_expr(stmt->expression())},
            {"body", dump(stmt->body())}};
}

json FunctionSerializer::dump(const SwitchCaseStmt *stmt) const noexcept {
    return {{"tag", "switch.case"},
            {"value", dump_expr(stmt->expression())},
            {"body", dump(stmt->body())}};
}

json FunctionSerializer::dump(const SwitchDefaultStmt *stmt) const noexcept {
    return {{"tag", "switch.default"},
            {"body", dump(stmt->body())}};
}

json FunctionSerializer::dump(const AssignStmt *stmt) const noexcept {
    return {{"tag", "assign"},
            {"lhs", dump_expr(stmt->lhs())},
            {"rhs", dump_expr(stmt->rhs())}};
}

json FunctionSerializer::dump(const ForStmt *stmt) const noexcept {
    return {{"tag", "for"},
            {"variable", dump_expr(stmt->variable())},
            {"condition", dump_expr(stmt->condition())},
            {"step", dump_expr(stmt->step())},
            {"body", dump(stmt->body())}};
}

json FunctionSerializer::dump(const CommentStmt *stmt) const noexcept {
    return {{"tag", "comment"}, {"string", stmt->comment()}};
}

void FunctionSerializer::dump(const ConstantData &c) const noexcept {
    auto key = luisa::format("{:016x}", c.hash());
    LUISA_ASSERT(_constants != nullptr, "Constants map is null");
    if (!_constants->contains(key)) {
        luisa::visit(
            [&](auto view) noexcept {
                using T = std::remove_cvref_t<decltype(view[0])>;
                auto data = reinterpret_cast<const uint8_t *>(view.data());
                auto binary = json::binary({data, data + view.size_bytes()});
                _constants->emplace(key, json{{"type", Type::of<T>()->description()},
                                              {"data", binary}});
            },
            c.view());
    }
}

void FunctionSerializer::dump(Function f) const noexcept {
    LUISA_ASSERT(_functions != nullptr, "Functions map is null");
    auto key = luisa::format("{:016x}", f.hash());
    if (_functions->contains(key.c_str())) { return; }
    auto &func = _functions->emplace(key.c_str(), json::object()).first.value();
    auto is_kernel = f.tag() == Function::Tag::KERNEL;
    func.emplace("tag", is_kernel ? "kernel" : "callable");
    if (is_kernel) {
        auto bs = f.block_size();
        func.emplace("block_size", std::array{bs.x, bs.y, bs.z});
    }
    luisa::vector<json, luisa::allocator<json>> variables(f.builder()->_variable_usages.size(), json::object());
    // built-in variables must be serialized first, as
    // they might also be used as arguments in Callables
    for (auto v : f.builtin_variables()) {
        variables[v.uid()].emplace("type", "vector<uint,3>");
        if (is_kernel) {
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
        } else {
            variables[v.uid()].emplace("tag", "argument");
        }
    }
    // arguments
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
                if (is_kernel) {
                    if (auto b = luisa::get_if<BufferBinding>(&binding)) {
                        var.emplace("binding", json{{"handle", luisa::format("{:016x}", b->handle)},
                                                    {"offset", b->offset_bytes},
                                                    {"size", b->size_bytes}});
                    }
                }
                break;
            case Variable::Tag::TEXTURE:
                var.emplace("tag", "texture");
                if (is_kernel) {
                    if (auto b = luisa::get_if<TextureBinding>(&binding)) {
                        var.emplace("binding", json{{"handle", luisa::format("{:016x}", b->handle)},
                                                    {"level", b->level}});
                    }
                }
                break;
            case Variable::Tag::BINDLESS_ARRAY:
                var.emplace("tag", "bindless_array");
                if (is_kernel) {
                    if (auto b = luisa::get_if<BindlessArrayBinding>(&binding)) {
                        var.emplace("binding", json{{"handle", luisa::format("{:016x}", b->handle)}});
                    }
                }
                break;
            case Variable::Tag::ACCEL:
                var.emplace("tag", "accel");
                if (is_kernel) {
                    if (auto b = luisa::get_if<AccelBinding>(&binding)) {
                        var.emplace("binding", json{{"handle", luisa::format("{:016x}", b->handle)}});
                    }
                }
                break;
            default: var.emplace("tag", "argument"); break;
        }
    }
    // shared variables
    for (auto v : f.shared_variables()) {
        variables[v.uid()].emplace("tag", "shared");
        variables[v.uid()].emplace("type", v.type()->description());
    }
    // local variables
    for (auto v : f.local_variables()) {
        variables[v.uid()].emplace("tag", "local");
        variables[v.uid()].emplace("type", v.type()->description());
    }
    func.emplace("variables", std::move(variables));
    // statements
    func.emplace("body", dump(f.body()));
}

json FunctionSerializer::to_json(Function function) const noexcept {
    auto module = json::object();
    module.emplace("entry", luisa::format("{:016x}", function.hash()));
    _constants = &module.emplace("constants", json::object()).first.value();
    _functions = &module.emplace("functions", json::object()).first.value();
    dump(function);
    _constants = nullptr;
    _functions = nullptr;
    return module;
}

luisa::shared_ptr<const detail::FunctionBuilder> FunctionSerializer::from_json(const json &json) const noexcept {
    _serialized_constants = &json.at("constants");
    _serialized_functions = &json.at("functions");
    auto entry = json.at("entry").get<luisa::string>();
    parse_function(entry);
    auto f = std::move(_parsed_functions.at(entry));
    _parsed_functions.clear();
    _parsed_constants.clear();
    _serialized_functions = nullptr;
    _serialized_constants = nullptr;
    LUISA_ASSERT(_variable_stack.empty(), "Variable stack is not empty");
    return f;
}

json FunctionSerializer::to_json(const shared_ptr<const detail::FunctionBuilder> &function) const noexcept {
    return to_json(function->function());
}

json FunctionSerializer::dump_stmt(const Statement *stmt) const noexcept {
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

json FunctionSerializer::dump_expr(const Expression *expr) const noexcept {
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

BinaryBuffer FunctionSerializer::to_binary(Function function) const noexcept {
    BinaryBuffer buffer;
    to_binary(buffer, function);
    return buffer;
}

BinaryBuffer FunctionSerializer::to_binary(const shared_ptr<const detail::FunctionBuilder> &function) const noexcept {
    return to_binary(function->function());
}

void FunctionSerializer::to_binary(BinaryBuffer &buffer, Function function) const noexcept {
    // TODO
}

void FunctionSerializer::to_binary(BinaryBuffer &buffer, const shared_ptr<const detail::FunctionBuilder> &function) const noexcept {
    to_binary(buffer, function->function());
}

luisa::shared_ptr<const detail::FunctionBuilder> FunctionSerializer::from_binary(BinaryBufferReader reader) const noexcept {
    return nullptr;
}

const UnaryExpr *FunctionSerializer::parse_unary_expr(const json &j) const noexcept {
    auto type = Type::from(j.at("type").get<luisa::string>());
    auto op = static_cast<UnaryOp>(j.at("op").get<uint>());
    auto operand = parse_expr(j.at("operand"));
    return function_builder()->unary(type, op, operand);
}

const BinaryExpr *FunctionSerializer::parse_binary_expr(const json &j) const noexcept {
    auto type = Type::from(j.at("type").get<luisa::string>());
    auto op = static_cast<BinaryOp>(j.at("op").get<uint>());
    auto lhs = parse_expr(j.at("lhs"));
    auto rhs = parse_expr(j.at("rhs"));
    return function_builder()->binary(type, op, lhs, rhs);
}

const Expression *FunctionSerializer::parse_member_expr(const json &j) const noexcept {
    auto type = Type::from(j.at("type").get<luisa::string>());
    auto self = parse_expr(j.at("self"));
    if (auto iter = j.find("index"); iter != j.end()) {
        auto index = iter->get<uint>();
        return function_builder()->member(type, self, index);
    }
    auto swizzle = j.at("swizzle").get<std::array<uint, 2>>();
    return function_builder()->swizzle(type, self, swizzle[0], swizzle[1]);
}

const AccessExpr *FunctionSerializer::parse_access_expr(const json &j) const noexcept {
    auto type = Type::from(j.at("type").get<luisa::string>());
    auto self = parse_expr(j.at("self"));
    auto index = parse_expr(j.at("index"));
    return function_builder()->access(type, self, index);
}

const LiteralExpr *FunctionSerializer::parse_literal_expr(const json &j) const noexcept {
    auto type = Type::from(j.at("type").get<luisa::string>());
    auto v = j.at("value");
    switch (type->tag()) {
        case Type::Tag::BOOL: return function_builder()->literal(type, v.get<bool>());
        case Type::Tag::FLOAT: return function_builder()->literal(type, v.get<float>());
        case Type::Tag::INT: return function_builder()->literal(type, v.get<int>());
        case Type::Tag::UINT: return function_builder()->literal(type, v.get<uint>());
        case Type::Tag::VECTOR:
            switch (type->element()->tag()) {
                case Type::Tag::BOOL:
                    if (type->dimension() == 2u) {
                        return function_builder()->literal(
                            type, make_bool2(v.at(0).get<bool>(), v.at(1).get<bool>()));
                    }
                    if (type->dimension() == 3u) {
                        return function_builder()->literal(
                            type, make_bool3(v.at(0).get<bool>(), v.at(1).get<bool>(), v.at(2).get<bool>()));
                    }
                    if (type->dimension() == 4u) {
                        return function_builder()->literal(
                            type, make_bool4(v.at(0).get<bool>(), v.at(1).get<bool>(),
                                             v.at(2).get<bool>(), v.at(3).get<bool>()));
                    }
                    break;
                case Type::Tag::FLOAT:
                    if (type->dimension() == 2u) {
                        return function_builder()->literal(
                            type, make_float2(v.at(0).get<float>(), v.at(1).get<float>()));
                    }
                    if (type->dimension() == 3u) {
                        return function_builder()->literal(
                            type, make_float3(v.at(0).get<float>(), v.at(1).get<float>(), v.at(2).get<float>()));
                    }
                    if (type->dimension() == 4u) {
                        return function_builder()->literal(
                            type, make_float4(v.at(0).get<float>(), v.at(1).get<float>(),
                                              v.at(2).get<float>(), v.at(3).get<float>()));
                    }
                    break;
                case Type::Tag::INT:
                    if (type->dimension() == 2u) {
                        return function_builder()->literal(
                            type, make_int2(v.at(0).get<int>(), v.at(1).get<int>()));
                    }
                    if (type->dimension() == 3u) {
                        return function_builder()->literal(
                            type, make_int3(v.at(0).get<int>(), v.at(1).get<int>(), v.at(2).get<int>()));
                    }
                    if (type->dimension() == 4u) {
                        return function_builder()->literal(
                            type, make_int4(v.at(0).get<int>(), v.at(1).get<int>(),
                                            v.at(2).get<int>(), v.at(3).get<int>()));
                    }
                    break;
                case Type::Tag::UINT:
                    if (type->dimension() == 2u) {
                        return function_builder()->literal(
                            type, make_uint2(v.at(0).get<uint>(), v.at(1).get<uint>()));
                    }
                    if (type->dimension() == 3u) {
                        return function_builder()->literal(
                            type, make_uint3(v.at(0).get<uint>(), v.at(1).get<uint>(), v.at(2).get<uint>()));
                    }
                    if (type->dimension() == 4u) {
                        return function_builder()->literal(
                            type, make_uint4(v.at(0).get<uint>(), v.at(1).get<uint>(),
                                             v.at(2).get<uint>(), v.at(3).get<uint>()));
                    }
                    break;
                default: break;
            }
            break;
        case Type::Tag::MATRIX:
            switch (type->dimension()) {
                case 2u: return function_builder()->literal(
                    type, make_float2x2(v.at(0).get<float>(), v.at(1).get<float>(),
                                        v.at(2).get<float>(), v.at(3).get<float>()));
                case 3u: return function_builder()->literal(
                    type, make_float3x3(v.at(0).get<float>(), v.at(1).get<float>(), v.at(2).get<float>(),
                                        v.at(3).get<float>(), v.at(4).get<float>(), v.at(5).get<float>(),
                                        v.at(6).get<float>(), v.at(7).get<float>(), v.at(8).get<float>()));
                case 4u: return function_builder()->literal(
                    type, make_float4x4(v.at(0).get<float>(), v.at(1).get<float>(), v.at(2).get<float>(), v.at(3).get<float>(),
                                        v.at(4).get<float>(), v.at(5).get<float>(), v.at(6).get<float>(), v.at(7).get<float>(),
                                        v.at(8).get<float>(), v.at(9).get<float>(), v.at(10).get<float>(), v.at(11).get<float>(),
                                        v.at(12).get<float>(), v.at(13).get<float>(), v.at(14).get<float>(), v.at(15).get<float>()));
                default: break;
            }
            break;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid literal expression.");
}

const RefExpr *FunctionSerializer::parse_ref_expr(const json &j) const noexcept {
    return _variable_stack.back().at(j.at("variable").get<uint>());
}

const ConstantExpr *FunctionSerializer::parse_constant_expr(const json &j) const noexcept {
    auto type = Type::from(j.at("type").get<luisa::string>());
    auto key = j.at("key").get<luisa::string>();
    parse_constant(key);
    auto c = _parsed_constants.at(key);
    return function_builder()->constant(type, c);
}

const CallExpr *FunctionSerializer::parse_call_expr(const json &j) const noexcept {
    auto type = Type::from(j.at("type").get<luisa::string>());
    luisa::vector<const Expression *> args;
    for (auto &arg : j.at("arguments")) { args.emplace_back(parse_expr(arg)); }
    if (auto iter = j.find("op"); iter != j.end()) {
        auto op = static_cast<CallOp>(iter->get<uint>());
        return function_builder()->call(type, op, args);
    }
    auto key = j.at("custom").get<luisa::string>();
    parse_function(key);
    auto custom = _parsed_functions.at(key)->function();
    return function_builder()->call(type, custom, args);
}

const CastExpr *FunctionSerializer::parse_cast_expr(const json &j) const noexcept {
    auto type = Type::from(j.at("type").get<luisa::string>());
    auto expr = parse_expr(j.at("operand"));
    auto op = j.at("op").get<luisa::string>() == "static" ? CastOp::STATIC : CastOp::BITWISE;
    return function_builder()->cast(type, op, expr);
}

void FunctionSerializer::parse_break_stmt(const json &j) const noexcept {
    function_builder()->break_();
}

void FunctionSerializer::parse_continue_stmt(const json &j) const noexcept {
    function_builder()->continue_();
}

void FunctionSerializer::parse_return_stmt(const json &j) const noexcept {
    if (auto iter = j.find("value"); iter != j.end()) {
        function_builder()->return_(parse_expr(*iter));
    } else {
        function_builder()->return_();
    }
}

void FunctionSerializer::parse_if_stmt(const json &j) const noexcept {
    auto cond = parse_expr(j.at("condition"));
    auto stmt = function_builder()->if_(cond);
    function_builder()->with(stmt->true_branch(), [&] {
        for (auto &s : j.at("true").at("statements")) { parse_stmt(s); }
    });
    function_builder()->with(stmt->false_branch(), [&] {
        for (auto &s : j.at("false").at("statements")) { parse_stmt(s); }
    });
}

void FunctionSerializer::parse_loop_stmt(const json &j) const noexcept {
    auto stmt = function_builder()->loop_();
    function_builder()->with(stmt->body(), [&] {
        for (auto &s : j.at("body").at("statements")) { parse_stmt(s); }
    });
}

void FunctionSerializer::parse_expr_stmt(const json &j) const noexcept {
    function_builder()->_void_expr(parse_expr(j.at("expression")));
}

void FunctionSerializer::parse_switch_stmt(const json &j) const noexcept {
    auto cond = parse_expr(j.at("expression"));
    auto stmt = function_builder()->switch_(cond);
    function_builder()->with(stmt->body(), [&] {
        for (auto &s : j.at("body").at("statements")) { parse_stmt(s); }
    });
}

void FunctionSerializer::parse_switch_case_stmt(const json &j) const noexcept {
    auto cond = parse_expr(j.at("value"));
    auto stmt = function_builder()->case_(cond);
    function_builder()->with(stmt->body(), [&] {
        for (auto &s : j.at("body").at("statements")) { parse_stmt(s); }
    });
}

void FunctionSerializer::parse_switch_default_stmt(const json &j) const noexcept {
    auto stmt = function_builder()->default_();
    function_builder()->with(stmt->body(), [&] {
        for (auto &s : j.at("body").at("statements")) { parse_stmt(s); }
    });
}

void FunctionSerializer::parse_assign_stmt(const json &j) const noexcept {
    auto lhs = parse_expr(j.at("lhs"));
    auto rhs = parse_expr(j.at("rhs"));
    function_builder()->assign(lhs, rhs);
}

void FunctionSerializer::parse_for_stmt(const json &j) const noexcept {
    auto var = parse_expr(j.at("variable"));
    auto condition = parse_expr(j.at("condition"));
    auto step = parse_expr(j.at("step"));
    auto stmt = function_builder()->for_(var, condition, step);
    function_builder()->with(stmt->body(), [&] {
        for (auto &s : j.at("body").at("statements")) { parse_stmt(s); }
    });
}

void FunctionSerializer::parse_comment_stmt(const json &j) const noexcept {
    function_builder()->comment_(j.at("string").get<luisa::string>());
}

const Expression *FunctionSerializer::parse_expr(const json &j) const noexcept {
    using namespace std::string_view_literals;
    using parser_type = luisa::function<const Expression *(const FunctionSerializer &, const json &)>;
    static const luisa::unordered_map<luisa::string_view, parser_type> tag_to_parser{
        {"unary"sv, [](const FunctionSerializer &s, const json &j) noexcept { return s.parse_unary_expr(j); }},
        {"binary"sv, [](const FunctionSerializer &s, const json &j) noexcept { return s.parse_binary_expr(j); }},
        {"member"sv, [](const FunctionSerializer &s, const json &j) noexcept { return s.parse_member_expr(j); }},
        {"access"sv, [](const FunctionSerializer &s, const json &j) noexcept { return s.parse_access_expr(j); }},
        {"literal"sv, [](const FunctionSerializer &s, const json &j) noexcept { return s.parse_literal_expr(j); }},
        {"ref"sv, [](const FunctionSerializer &s, const json &j) noexcept { return s.parse_ref_expr(j); }},
        {"constant", [](const FunctionSerializer &s, const json &j) noexcept { return s.parse_constant_expr(j); }},
        {"call"sv, [](const FunctionSerializer &s, const json &j) noexcept { return s.parse_call_expr(j); }},
        {"cast"sv, [](const FunctionSerializer &s, const json &j) noexcept { return s.parse_cast_expr(j); }}};
    return tag_to_parser.at(j.at("tag").get<luisa::string>())(*this, j);
}

void FunctionSerializer::parse_stmt(const json &j) const noexcept {
    using namespace std::string_view_literals;
    using parser_type = luisa::function<void(const FunctionSerializer &, const json &)>;
    static const luisa::unordered_map<luisa::string_view, parser_type> tag_to_parser{
        {"break"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_break_stmt(j); }},
        {"continue"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_continue_stmt(j); }},
        {"return"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_return_stmt(j); }},
        {"if"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_if_stmt(j); }},
        {"loop"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_loop_stmt(j); }},
        {"expr"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_expr_stmt(j); }},
        {"switch"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_switch_stmt(j); }},
        {"switch.case"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_switch_case_stmt(j); }},
        {"switch.default"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_switch_default_stmt(j); }},
        {"assign"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_assign_stmt(j); }},
        {"for"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_for_stmt(j); }},
        {"comment"sv, [](const FunctionSerializer &s, const json &j) noexcept { s.parse_comment_stmt(j); }}};
    tag_to_parser.at(j.at("tag").get<luisa::string>())(*this, j);
}

void FunctionSerializer::parse_constant(luisa::string_view key) const noexcept {
    if (_parsed_constants.contains(key)) { return; }
    auto &c = _serialized_constants->at(key);
    auto &binary = c.at("data").get_binary();
    auto type = c.at("type").get<luisa::string>();
    auto make = [&]<typename T>() noexcept {
        auto data = reinterpret_cast<const T *>(binary.data());
        auto size = binary.size() / sizeof(T);
        auto view = luisa::span<const T>{data, size};
        _parsed_constants.emplace(key, ConstantData::create(view));
    };
    if (type == "bool") {
        make.operator()<bool>();
    } else if (type == "float") {
        make.operator()<float>();
    } else if (type == "int") {
        make.operator()<int>();
    } else if (type == "uint") {
        make.operator()<uint>();
    } else if (type == "vector<bool,2>") {
        make.operator()<bool2>();
    } else if (type == "vector<float,2>") {
        make.operator()<float2>();
    } else if (type == "vector<int,2>") {
        make.operator()<int2>();
    } else if (type == "vector<uint,2>") {
        make.operator()<uint2>();
    } else if (type == "vector<bool,3>") {
        make.operator()<bool3>();
    } else if (type == "vector<float,3>") {
        make.operator()<float3>();
    } else if (type == "vector<int,3>") {
        make.operator()<int3>();
    } else if (type == "vector<uint,3>") {
        make.operator()<uint3>();
    } else if (type == "vector<bool,4>") {
        make.operator()<bool4>();
    } else if (type == "vector<float,4>") {
        make.operator()<float4>();
    } else if (type == "vector<int,4>") {
        make.operator()<int4>();
    } else if (type == "vector<uint,4>") {
        make.operator()<uint4>();
    } else if (type == "matrix<2>") {
        make.operator()<float2x2>();
    } else if (type == "matrix<3>") {
        make.operator()<float3x3>();
    } else if (type == "matrix<4>") {
        make.operator()<float4x4>();
    }
}

void FunctionSerializer::parse_function(luisa::string_view key) const noexcept {
    if (_parsed_functions.contains(key)) { return; }
    auto &f = _serialized_functions->at(key);
    auto parse_variables = [&] {
        auto serialized_variables = _serialized_functions->at(key).at("variables");
        luisa::vector<const RefExpr *> variables;
        variables.reserve(serialized_variables.size());
        auto add = [&](const RefExpr *e) noexcept { variables.emplace_back(e); };
        auto handle = [&](const json &j) noexcept {
            auto h = j.at("handle").get<luisa::string>();
            auto end = h.data() + h.size();
            return std::strtoull(h.data(), &end, 16);
        };
        for (auto &v : serialized_variables) {
            auto tag = v.at("tag").get<luisa::string>();
            auto type = Type::from(v.at("type").get<luisa::string>());
            if (tag == "thread_id") {
                add(function_builder()->thread_id());
            } else if (tag == "block_id") {
                add(function_builder()->block_id());
            } else if (tag == "dispatch_id") {
                add(function_builder()->dispatch_id());
            } else if (tag == "dispatch_size") {
                add(function_builder()->dispatch_size());
            } else if (tag == "reference") {
                add(function_builder()->reference(type));
            } else if (tag == "buffer") {
                if (auto iter = v.find("binding"); iter != v.end()) {
                    auto b = *iter;
                    add(function_builder()->buffer_binding(
                        type, handle(b), b.at("offset").get<size_t>(), b.at("size").get<size_t>()));
                } else {
                    add(function_builder()->buffer(type));
                }
            } else if (tag == "texture") {
                if (auto iter = v.find("binding"); iter != v.end()) {
                    auto b = *iter;
                    add(function_builder()->texture_binding(
                        type, handle(b), b.at("level").get<uint>()));
                } else {
                    add(function_builder()->texture(type));
                }
            } else if (tag == "bindless_array") {
                if (auto iter = v.find("binding"); iter != v.end()) {
                    auto b = *iter;
                    add(function_builder()->bindless_array_binding(handle(b)));
                } else {
                    add(function_builder()->bindless_array());
                }
            } else if (tag == "accel") {
                if (auto iter = v.find("binding"); iter != v.end()) {
                    auto b = *iter;
                    add(function_builder()->accel_binding(handle(b)));
                } else {
                    add(function_builder()->accel());
                }
            } else if (tag == "argument") {
                add(function_builder()->argument(type));
            } else if (tag == "shared") {
                add(function_builder()->shared(type));
            } else {
                add(function_builder()->local(type));
            }
        }
        return variables;
    };
    if (auto tag = f.at("tag").get<luisa::string>(); tag == "kernel") {
        auto bs = f.at("block_size").get<std::array<uint, 3>>();
        auto kernel = detail::FunctionBuilder::define_kernel([&] {
            auto block_size = luisa::make_uint3(bs[0], bs[1], bs[2]);
            function_builder()->set_block_size(block_size);
            _variable_stack.emplace_back(parse_variables());
            for (auto &s : f.at("body").at("statements")) { parse_stmt(s); }
            _variable_stack.pop_back();
        });
        _parsed_functions.emplace(key, kernel);
    } else {
        auto callable = detail::FunctionBuilder::define_callable([&] {
            _variable_stack.emplace_back(parse_variables());
            for (auto &s : f.at("body").at("statements")) { parse_stmt(s); }
            _variable_stack.pop_back();
        });
        _parsed_functions.emplace(key, callable);
    }
}

#pragma clang diagnostic pop

}// namespace luisa::compute
