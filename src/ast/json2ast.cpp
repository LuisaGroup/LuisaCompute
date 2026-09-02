#include <luisa/ast/ast2json.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_set>

#include <yyjson.h>

#include <luisa/ast/function_builder.h>
#include <luisa/core/magic_enum.h>
#include <luisa/core/stl/format.h>

namespace luisa::compute {

bool ASTJsonBindingResolver::resolve_buffer(
    const Type *, uint64_t serialized_handle, size_t serialized_offset,
    size_t serialized_size, Function::BufferBinding &binding,
    luisa::string &) const noexcept {
    binding = Function::BufferBinding{
        serialized_handle, serialized_offset, serialized_size};
    return true;
}

bool ASTJsonBindingResolver::resolve_texture(
    uint64_t serialized_handle, uint32_t serialized_level,
    Function::TextureBinding &binding,
    luisa::string &) const noexcept {
    binding = Function::TextureBinding{
        serialized_handle, serialized_level};
    return true;
}

bool ASTJsonBindingResolver::resolve_bindless_array(
    uint64_t serialized_handle, Function::BindlessArrayBinding &binding,
    luisa::string &) const noexcept {
    binding = Function::BindlessArrayBinding{serialized_handle};
    return true;
}

bool ASTJsonBindingResolver::resolve_accel(
    uint64_t serialized_handle, Function::AccelBinding &binding,
    luisa::string &) const noexcept {
    binding = Function::AccelBinding{serialized_handle};
    return true;
}

namespace {

[[nodiscard]] bool is_remote_safe_custom_type(
    const Type *type) noexcept {
    if (type == nullptr || !type->is_custom()) { return false; }
    auto id = type->description();
    return id == ast_json_indirect_dispatch_buffer_type_name ||
           id == ast_json_ray_query_all_type_name ||
           id == ast_json_ray_query_any_type_name;
}

[[nodiscard]] bool is_indirect_dispatch_buffer_type(
    const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           type->description() ==
               ast_json_indirect_dispatch_buffer_type_name;
}

[[nodiscard]] bool is_ray_query_type(
    const Type *type) noexcept {
    if (type == nullptr || !type->is_custom()) { return false; }
    auto id = type->description();
    return id == ast_json_ray_query_all_type_name ||
           id == ast_json_ray_query_any_type_name;
}

[[nodiscard]] bool is_buffer_variable_type(
    const Type *type) noexcept {
    return type != nullptr &&
           (type->is_buffer() || is_indirect_dispatch_buffer_type(type));
}

[[nodiscard]] bool assignment_types_compatible(
    const Type *lhs, const Type *rhs) noexcept {
    return lhs == rhs ||
           (lhs != nullptr && rhs != nullptr &&
            lhs->is_scalar() && rhs->is_scalar());
}

[[nodiscard]] bool is_assignable_expression(
    const Expression *expression) noexcept {
    if (expression == nullptr) { return false; }
    switch (expression->tag()) {
        case Expression::Tag::REF: return true;
        case Expression::Tag::MEMBER:
            return is_assignable_expression(
                static_cast<const MemberExpr *>(expression)->self());
        case Expression::Tag::ACCESS:
            return is_assignable_expression(
                static_cast<const AccessExpr *>(expression)->range());
        default: return false;
    }
}

class ASTJsonPreflight {

private:
    const ASTJsonLimits &_limits;
    luisa::unordered_set<const detail::FunctionBuilder *> _visited_functions;
    luisa::unordered_set<const detail::FunctionBuilder *> _active_functions;
    luisa::unordered_set<const Type *> _visited_types;
    luisa::unordered_set<const void *> _visited_constants;
    size_t _node_count{};
    size_t _constant_bytes{};
    luisa::string _error;

private:
    void _fail(luisa::string message) noexcept {
        if (_error.empty()) { _error = std::move(message); }
    }

    [[nodiscard]] bool _check_depth(size_t depth) noexcept {
        if (depth > _limits.max_depth) {
            _fail("AST JSON nesting exceeds the configured depth limit.");
            return false;
        }
        return true;
    }

    [[nodiscard]] bool _count_node() noexcept {
        if (++_node_count > _limits.max_nodes) {
            _fail("AST JSON node count exceeds the configured limit.");
            return false;
        }
        return true;
    }

    [[nodiscard]] bool _check_string(luisa::string_view value) noexcept {
        if (value.size() > _limits.max_string_bytes) {
            _fail("AST JSON string exceeds the configured limit.");
            return false;
        }
        return true;
    }

    [[nodiscard]] bool _visit_type(const Type *type, size_t depth) noexcept {
        if (type == nullptr || !_error.empty()) { return _error.empty(); }
        if (!_check_depth(depth)) { return false; }
        if (!_visited_types.emplace(type).second) { return true; }
        if (_visited_types.size() > _limits.max_types) {
            _fail("AST JSON type count exceeds the configured limit.");
            return false;
        }
        if ((type->is_structure() || type->is_buffer() || type->is_texture()) &&
            !type->member_attributes().empty()) {
            _fail("Attributed AST types are not portable in AST JSON schema v1.");
            return false;
        }
        switch (type->tag()) {
            case Type::Tag::COOPERATIVE_VECTOR:
            case Type::Tag::COOPERATIVE_VECTOR_REF:
            case Type::Tag::COOPERATIVE_MATRIX_REF:
                _fail("Cooperative AST types are not supported by AST JSON schema v1.");
                return false;
            case Type::Tag::CUSTOM:
                if (!is_remote_safe_custom_type(type)) {
                    _fail("Custom AST types are not supported by remote-safe AST JSON.");
                    return false;
                }
                return _check_string(type->description());
            case Type::Tag::VECTOR:
            case Type::Tag::MATRIX:
            case Type::Tag::ARRAY:
            case Type::Tag::TEXTURE:
                return _visit_type(type->element(), depth + 1u);
            case Type::Tag::BUFFER:
                if (auto element = type->element();
                    element != nullptr && element->is_custom()) {
                    _fail("Custom AST types cannot be nested in buffers.");
                    return false;
                }
                return _visit_type(type->element(), depth + 1u);
            case Type::Tag::STRUCTURE:
                for (auto member : type->members()) {
                    if (!_visit_type(member, depth + 1u)) { return false; }
                }
                return true;
            default: return true;
        }
    }

    [[nodiscard]] bool _visit_constant(
        ConstantData data, size_t depth) noexcept {
        if (!_visited_constants.emplace(data.raw()).second) { return true; }
        auto size = data.type()->size();
        if (size > _limits.max_constant_bytes -
                       std::min(_constant_bytes, _limits.max_constant_bytes)) {
            _fail("AST JSON constants exceed the configured byte limit.");
            return false;
        }
        _constant_bytes += size;
        return _visit_type(data.type(), depth);
    }

    [[nodiscard]] bool _visit_expr(const Expression *expr,
                                   size_t depth) noexcept {
        if (expr == nullptr || !_error.empty()) { return _error.empty(); }
        if (!_check_depth(depth) || !_count_node() ||
            !_visit_type(expr->type(), depth + 1u)) {
            return false;
        }
        switch (expr->tag()) {
            case Expression::Tag::UNARY:
                return _visit_expr(
                    static_cast<const UnaryExpr *>(expr)->operand(), depth + 1u);
            case Expression::Tag::BINARY: {
                auto e = static_cast<const BinaryExpr *>(expr);
                return _visit_expr(e->lhs(), depth + 1u) &&
                       _visit_expr(e->rhs(), depth + 1u);
            }
            case Expression::Tag::MEMBER:
                return _visit_expr(
                    static_cast<const MemberExpr *>(expr)->self(), depth + 1u);
            case Expression::Tag::ACCESS: {
                auto e = static_cast<const AccessExpr *>(expr);
                return _visit_expr(e->range(), depth + 1u) &&
                       _visit_expr(e->index(), depth + 1u);
            }
            case Expression::Tag::LITERAL: return true;
            case Expression::Tag::REF:
                return _visit_type(
                    static_cast<const RefExpr *>(expr)->variable().type(), depth + 1u);
            case Expression::Tag::CONSTANT: {
                auto data = static_cast<const ConstantExpr *>(expr)->data();
                return _visit_constant(data, depth + 1u);
            }
            case Expression::Tag::CALL: {
                auto e = static_cast<const CallExpr *>(expr);
                if (e->is_external()) {
                    _fail("External AST functions are not supported by remote-safe AST JSON.");
                    return false;
                }
                if (e->is_builtin() && is_atomic_operation(e->op())) {
                    if (e->arguments().empty() ||
                        e->arguments().front()->type() == nullptr ||
                        e->arguments().front()->type()->element() == nullptr) {
                        _fail("Atomic AST access chains require a typed buffer or array.");
                        return false;
                    }
                }
                if (e->is_custom() && !_visit_function(e->custom(), depth + 1u, false)) {
                    return false;
                }
                for (auto arg : e->arguments()) {
                    if (!_visit_expr(arg, depth + 1u)) { return false; }
                }
                return true;
            }
            case Expression::Tag::CAST:
                return _visit_expr(
                    static_cast<const CastExpr *>(expr)->expression(), depth + 1u);
            case Expression::Tag::TYPE_ID:
                return _visit_type(
                    static_cast<const TypeIDExpr *>(expr)->data_type(), depth + 1u);
            case Expression::Tag::STRING_ID:
                return _check_string(
                    static_cast<const StringIDExpr *>(expr)->data());
            case Expression::Tag::FUNC_REF:
                return _visit_function(
                    Function{static_cast<const FuncRefExpr *>(expr)->func()},
                    depth + 1u, false);
            case Expression::Tag::CPUCUSTOM:
                _fail("CPU custom AST operations cannot cross a process boundary.");
                return false;
            case Expression::Tag::GPUCUSTOM:
                _fail("GPU custom AST source cannot cross a remote-safe AST boundary.");
                return false;
        }
        _fail("Unknown AST expression tag.");
        return false;
    }

    [[nodiscard]] bool _visit_stmt(const Statement *stmt,
                                   size_t depth) noexcept {
        if (stmt == nullptr) {
            _fail("AST contains a null statement.");
            return false;
        }
        if (!_check_depth(depth) || !_count_node()) { return false; }
        switch (stmt->tag()) {
            case Statement::Tag::BREAK:
            case Statement::Tag::CONTINUE: return true;
            case Statement::Tag::RETURN:
                return _visit_expr(
                    static_cast<const ReturnStmt *>(stmt)->expression(), depth + 1u);
            case Statement::Tag::SCOPE:
                for (auto child : static_cast<const ScopeStmt *>(stmt)->statements()) {
                    if (!_visit_stmt(child, depth + 1u)) { return false; }
                }
                return true;
            case Statement::Tag::IF: {
                auto s = static_cast<const IfStmt *>(stmt);
                return _visit_expr(s->condition(), depth + 1u) &&
                       _visit_stmt(s->true_branch(), depth + 1u) &&
                       _visit_stmt(s->false_branch(), depth + 1u);
            }
            case Statement::Tag::LOOP:
                return _visit_stmt(
                    static_cast<const LoopStmt *>(stmt)->body(), depth + 1u);
            case Statement::Tag::EXPR:
                return _visit_expr(
                    static_cast<const ExprStmt *>(stmt)->expression(), depth + 1u);
            case Statement::Tag::SWITCH: {
                auto s = static_cast<const SwitchStmt *>(stmt);
                return _visit_expr(s->expression(), depth + 1u) &&
                       _visit_stmt(s->body(), depth + 1u);
            }
            case Statement::Tag::SWITCH_CASE: {
                auto s = static_cast<const SwitchCaseStmt *>(stmt);
                return _visit_expr(s->expression(), depth + 1u) &&
                       _visit_stmt(s->body(), depth + 1u);
            }
            case Statement::Tag::SWITCH_DEFAULT:
                return _visit_stmt(
                    static_cast<const SwitchDefaultStmt *>(stmt)->body(), depth + 1u);
            case Statement::Tag::ASSIGN: {
                auto s = static_cast<const AssignStmt *>(stmt);
                if (!is_assignable_expression(s->lhs())) {
                    _fail("AST assignment left-hand side is not assignable.");
                    return false;
                }
                if (!assignment_types_compatible(
                        s->lhs()->type(), s->rhs()->type())) {
                    _fail("AST assignment operand types are incompatible.");
                    return false;
                }
                return _visit_expr(s->lhs(), depth + 1u) &&
                       _visit_expr(s->rhs(), depth + 1u);
            }
            case Statement::Tag::FOR: {
                auto s = static_cast<const ForStmt *>(stmt);
                if (!is_assignable_expression(s->variable()) ||
                    !assignment_types_compatible(
                        s->variable()->type(), s->step()->type()) ||
                    s->condition()->type() == nullptr ||
                    !s->condition()->type()->is_bool()) {
                    _fail("AST for-loop operands are invalid.");
                    return false;
                }
                return _visit_expr(s->variable(), depth + 1u) &&
                       _visit_expr(s->condition(), depth + 1u) &&
                       _visit_expr(s->step(), depth + 1u) &&
                       _visit_stmt(s->body(), depth + 1u);
            }
            case Statement::Tag::COMMENT:
                return _check_string(
                    static_cast<const CommentStmt *>(stmt)->comment());
            case Statement::Tag::RAY_QUERY: {
                auto s = static_cast<const RayQueryStmt *>(stmt);
                if (!is_ray_query_type(s->query()->type())) {
                    _fail("AST ray-query statement has a non-ray-query value.");
                    return false;
                }
                return _visit_expr(s->query(), depth + 1u) &&
                       _visit_stmt(s->on_triangle_candidate(), depth + 1u) &&
                       _visit_stmt(s->on_procedural_candidate(), depth + 1u);
            }
            case Statement::Tag::AUTO_DIFF:
                return _visit_stmt(
                    static_cast<const AutoDiffStmt *>(stmt)->body(), depth + 1u);
            case Statement::Tag::PRINT: {
                auto s = static_cast<const PrintStmt *>(stmt);
                if (!_check_string(s->format())) { return false; }
                for (auto arg : s->arguments()) {
                    if (!_visit_expr(arg, depth + 1u)) { return false; }
                }
                return true;
            }
            case Statement::Tag::SUSPEND:
                _fail("Coroutine suspend statements are not supported by remote-safe AST JSON.");
                return false;
            case Statement::Tag::DEBUG_BREAK:
                _fail("Debug-break callbacks cannot cross a process boundary.");
                return false;
        }
        _fail("Unknown AST statement tag.");
        return false;
    }

    [[nodiscard]] bool _visit_function(Function function, size_t depth,
                                       bool entry) noexcept {
        if (!function) {
            _fail("Cannot serialize an empty AST function.");
            return false;
        }
        auto builder = function.builder();
        if (_active_functions.contains(builder)) {
            _fail("Recursive AST callable graphs are not supported by schema v1.");
            return false;
        }
        if (!_visited_functions.emplace(builder).second) { return true; }
        if (_visited_functions.size() > _limits.max_functions) {
            _fail("AST JSON function count exceeds the configured limit.");
            return false;
        }
        if (!_check_depth(depth)) { return false; }
        if ((entry && function.tag() != Function::Tag::KERNEL) ||
            (!entry && function.tag() != Function::Tag::CALLABLE)) {
            _fail(entry ?
                      "Remote-safe AST JSON entry must be a compute kernel." :
                      "Remote-safe AST JSON dependencies must be callables.");
            return false;
        }
        if (!function.external_callables().empty()) {
            _fail("External AST functions are not supported by remote-safe AST JSON.");
            return false;
        }
        if (!_check_string(function.name())) { return false; }
        _active_functions.emplace(builder);
        for (auto variable : function.arguments()) {
            if (!_visit_type(variable.type(), depth + 1u) ||
                !_check_string(function.get_variable_name(variable.uid()))) {
                _active_functions.erase(builder);
                return false;
            }
        }
        for (auto variable : function.local_variables()) {
            if (!_visit_type(variable.type(), depth + 1u) ||
                !_check_string(function.get_variable_name(variable.uid()))) {
                _active_functions.erase(builder);
                return false;
            }
        }
        for (auto variable : function.builtin_variables()) {
            if (!_visit_type(variable.type(), depth + 1u) ||
                !_check_string(function.get_variable_name(variable.uid()))) {
                _active_functions.erase(builder);
                return false;
            }
        }
        for (auto variable : function.shared_variables()) {
            if (!_visit_type(variable.type(), depth + 1u) ||
                !_check_string(function.get_variable_name(variable.uid()))) {
                _active_functions.erase(builder);
                return false;
            }
        }
        for (auto constant : function.constants()) {
            if (!_visit_constant(constant, depth + 1u)) {
                _active_functions.erase(builder);
                return false;
            }
        }
        auto ok = _visit_stmt(function.body(), depth + 1u);
        _active_functions.erase(builder);
        return ok;
    }

public:
    explicit ASTJsonPreflight(const ASTJsonLimits &limits) noexcept
        : _limits{limits} {}

    [[nodiscard]] bool run(Function function) noexcept {
        return _visit_function(function, 0u, true);
    }

    [[nodiscard]] luisa::string take_error() noexcept {
        return std::move(_error);
    }
};

class ASTJsonDecoder {

private:
    yyjson_val *_root;
    const ASTJsonLimits &_limits;
    const ASTJsonBindingResolver &_binding_resolver;
    luisa::vector<const Type *> _types;
    luisa::vector<ConstantData> _constants;
    luisa::vector<luisa::shared_ptr<const detail::FunctionBuilder>> _functions;
    luisa::string _error;
    size_t _node_count{};
    size_t _constant_bytes{};

private:
    void _fail(luisa::string_view path, luisa::string message) noexcept {
        if (_error.empty()) {
            _error = luisa::format("AST JSON error at {}: {}", path, message);
        }
    }

    [[nodiscard]] bool _check_keys(
        yyjson_val *object,
        std::initializer_list<luisa::string_view> allowed,
        luisa::string_view path) noexcept {
        if (!yyjson_is_obj(object)) {
            _fail(path, "expected an object.");
            return false;
        }
        std::unordered_set<std::string_view> seen;
        yyjson_obj_iter iter = yyjson_obj_iter_with(object);
        while (auto key = yyjson_obj_iter_next(&iter)) {
            auto key_view = std::string_view{
                yyjson_get_str(key), yyjson_get_len(key)};
            if (!seen.emplace(key_view).second) {
                _fail(path, luisa::format("duplicate member '{}'.", key_view));
                return false;
            }
            auto known = std::any_of(
                allowed.begin(), allowed.end(),
                [key_view](auto candidate) noexcept {
                    return candidate == key_view;
                });
            if (!known) {
                _fail(path, luisa::format("unknown member '{}'.", key_view));
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] yyjson_val *_member(
        yyjson_val *object, const char *key,
        luisa::string_view path, bool required = true) noexcept {
        if (!yyjson_is_obj(object)) {
            _fail(path, "expected an object.");
            return nullptr;
        }
        auto value = yyjson_obj_get(object, key);
        if (required && value == nullptr) {
            _fail(path, luisa::format("missing required member '{}'.", key));
        }
        return value;
    }

    [[nodiscard]] bool _string(
        yyjson_val *value, luisa::string_view &result,
        luisa::string_view path) noexcept {
        if (!yyjson_is_str(value)) {
            _fail(path, "expected a string.");
            return false;
        }
        auto size = yyjson_get_len(value);
        if (size > _limits.max_string_bytes) {
            _fail(path, "string exceeds the configured byte limit.");
            return false;
        }
        result = {yyjson_get_str(value), size};
        return true;
    }

    [[nodiscard]] bool _uint64(
        yyjson_val *value, uint64_t &result,
        luisa::string_view path) noexcept {
        if (!yyjson_is_uint(value)) {
            _fail(path, "expected an unsigned integer.");
            return false;
        }
        result = yyjson_get_uint(value);
        return true;
    }

    [[nodiscard]] bool _uint32(
        yyjson_val *value, uint32_t &result,
        luisa::string_view path) noexcept {
        uint64_t wide{};
        if (!_uint64(value, wide, path)) { return false; }
        if (wide > std::numeric_limits<uint32_t>::max()) {
            _fail(path, "integer is outside the uint32 range.");
            return false;
        }
        result = static_cast<uint32_t>(wide);
        return true;
    }

    [[nodiscard]] bool _size(
        yyjson_val *value, size_t &result,
        luisa::string_view path) noexcept {
        uint64_t wide{};
        if (!_uint64(value, wide, path)) { return false; }
        if (wide > std::numeric_limits<size_t>::max()) {
            _fail(path, "integer is outside the host size range.");
            return false;
        }
        result = static_cast<size_t>(wide);
        return true;
    }

    [[nodiscard]] bool _decimal_uint64(
        yyjson_val *value, uint64_t &result,
        luisa::string_view path) noexcept {
        luisa::string_view text;
        if (!_string(value, text, path)) { return false; }
        auto parsed = std::from_chars(
            text.data(), text.data() + text.size(), result);
        if (parsed.ec != std::errc{} ||
            parsed.ptr != text.data() + text.size() || text.empty()) {
            _fail(path, "expected a canonical decimal uint64 string.");
            return false;
        }
        return true;
    }

    template<typename E>
    [[nodiscard]] bool _enum(
        yyjson_val *value, E &result,
        luisa::string_view path) noexcept {
        luisa::string_view text;
        if (!_string(value, text, path)) { return false; }
        auto parsed = magic_enum::enum_cast<E>(
            std::string_view{text.data(), text.size()});
        if (!parsed) {
            _fail(path, luisa::format("unknown enum value '{}'.", text));
            return false;
        }
        result = *parsed;
        return true;
    }

    [[nodiscard]] bool _array(
        yyjson_val *value, size_t limit,
        luisa::string_view path) noexcept {
        if (!yyjson_is_arr(value)) {
            _fail(path, "expected an array.");
            return false;
        }
        if (yyjson_arr_size(value) > limit) {
            _fail(path, "array exceeds the configured element limit.");
            return false;
        }
        return true;
    }

    [[nodiscard]] bool _count_node(size_t depth,
                                   luisa::string_view path) noexcept {
        if (depth > _limits.max_depth) {
            _fail(path, "nesting exceeds the configured depth limit.");
            return false;
        }
        if (++_node_count > _limits.max_nodes) {
            _fail(path, "node count exceeds the configured limit.");
            return false;
        }
        return true;
    }

    [[nodiscard]] bool _type_index(
        yyjson_val *value, size_t upper_bound,
        const Type *&type, luisa::string_view path) noexcept {
        uint64_t index{};
        if (!_uint64(value, index, path)) { return false; }
        if (index >= upper_bound || index >= _types.size()) {
            _fail(path, "type index is out of range or not topologically ordered.");
            return false;
        }
        type = _types[static_cast<size_t>(index)];
        return true;
    }

    [[nodiscard]] static const Type *_scalar_type(Type::Tag tag) noexcept {
        switch (tag) {
            case Type::Tag::BOOL: return Type::from("bool");
            case Type::Tag::INT8: return Type::from("byte");
            case Type::Tag::UINT8: return Type::from("ubyte");
            case Type::Tag::INT16: return Type::from("short");
            case Type::Tag::UINT16: return Type::from("ushort");
            case Type::Tag::INT32: return Type::from("int");
            case Type::Tag::UINT32: return Type::from("uint");
            case Type::Tag::INT64: return Type::from("long");
            case Type::Tag::UINT64: return Type::from("ulong");
            case Type::Tag::FLOAT16: return Type::from("half");
            case Type::Tag::FLOAT32: return Type::from("float");
            case Type::Tag::FLOAT64: return Type::from("double");
            case Type::Tag::FLOAT8_E4M3: return Type::from("float8e4m3");
            case Type::Tag::FLOAT8_E5M2: return Type::from("float8e5m2");
            case Type::Tag::INT4: return Type::from("int4");
            case Type::Tag::FP4_E2M1: return Type::from("fp4e2m1");
            default: return nullptr;
        }
    }

    [[nodiscard]] bool _decode_types(yyjson_val *array) noexcept {
        if (!_array(array, _limits.max_types, "$.types")) { return false; }
        _types.reserve(yyjson_arr_size(array));
        auto count = yyjson_arr_size(array);
        for (size_t i = 0u; i < count; i++) {
            auto item = yyjson_arr_get(array, i);
            auto path = luisa::format("$.types[{}]", i);
            if (yyjson_is_null(item)) {
                _types.emplace_back(nullptr);
                continue;
            }
            if (!yyjson_is_obj(item)) {
                _fail(path, "expected an object or null.");
                return false;
            }
            Type::Tag tag{};
            if (!_enum(_member(item, "tag", path), tag,
                       luisa::format("{}.tag", path))) {
                return false;
            }
            if (auto scalar = _scalar_type(tag)) {
                if (!_check_keys(item, {"tag"}, path)) { return false; }
                _types.emplace_back(scalar);
                continue;
            }
            switch (tag) {
                case Type::Tag::VECTOR:
                case Type::Tag::MATRIX:
                case Type::Tag::ARRAY:
                case Type::Tag::TEXTURE: {
                    if (!_check_keys(item, {"tag", "element", "dimension"}, path)) {
                        return false;
                    }
                    const Type *element{};
                    uint32_t dimension{};
                    if (!_type_index(
                            _member(item, "element", path), i, element,
                            luisa::format("{}.element", path)) ||
                        !_uint32(
                            _member(item, "dimension", path), dimension,
                            luisa::format("{}.dimension", path))) {
                        return false;
                    }
                    if (element == nullptr) {
                        _fail(path, "composite type element must not be null.");
                        return false;
                    }
                    const Type *type{};
                    if (tag == Type::Tag::VECTOR) {
                        if (!element->is_scalar() || dimension < 2u || dimension > 4u) {
                            _fail(path, "invalid vector element or dimension.");
                            return false;
                        }
                        type = Type::vector(element, dimension);
                    } else if (tag == Type::Tag::MATRIX) {
                        if (!element->is_float32() || dimension < 2u || dimension > 4u) {
                            _fail(path, "matrix must be float32 with dimension 2, 3, or 4.");
                            return false;
                        }
                        type = Type::matrix(dimension);
                    } else if (tag == Type::Tag::ARRAY) {
                        if (dimension == 0u || element->is_resource() ||
                            element->is_custom() ||
                            element->size() > std::numeric_limits<uint32_t>::max() / dimension) {
                            _fail(path, "invalid or overflowing array type.");
                            return false;
                        }
                        type = Type::array(element, dimension);
                    } else {
                        if ((dimension != 2u && dimension != 3u) ||
                            !(element->is_int32() || element->is_uint32() ||
                              element->is_float32())) {
                            _fail(path, "invalid texture element or dimension.");
                            return false;
                        }
                        type = Type::texture(element, dimension);
                    }
                    _types.emplace_back(type);
                    break;
                }
                case Type::Tag::STRUCTURE: {
                    if (!_check_keys(item, {"tag", "alignment", "members"}, path)) {
                        return false;
                    }
                    uint32_t alignment{};
                    auto members_value = _member(item, "members", path);
                    if (!_uint32(
                            _member(item, "alignment", path), alignment,
                            luisa::format("{}.alignment", path)) ||
                        !_array(members_value, _limits.max_types,
                                luisa::format("{}.members", path))) {
                        return false;
                    }
                    if (alignment != 1u && alignment != 4u &&
                        alignment != 8u && alignment != 16u) {
                        _fail(path, "structure alignment must be 1, 4, 8, or 16.");
                        return false;
                    }
                    luisa::vector<const Type *> members;
                    members.reserve(yyjson_arr_size(members_value));
                    size_t layout_size{};
                    size_t max_member_alignment{1u};
                    for (size_t j = 0u; j < yyjson_arr_size(members_value); j++) {
                        const Type *member{};
                        if (!_type_index(
                                yyjson_arr_get(members_value, j), i, member,
                                luisa::format("{}.members[{}]", path, j))) {
                            return false;
                        }
                        if (member == nullptr || member->is_resource() ||
                            member->is_custom()) {
                            _fail(path, "structure members must be portable data types.");
                            return false;
                        }
                        auto member_alignment = member->alignment();
                        max_member_alignment = std::max(
                            max_member_alignment, member_alignment);
                        if (layout_size > std::numeric_limits<uint32_t>::max() -
                                              (member_alignment - 1u)) {
                            _fail(path, "structure layout overflows uint32.");
                            return false;
                        }
                        layout_size = (layout_size + member_alignment - 1u) /
                                      member_alignment * member_alignment;
                        if (member->size() >
                            std::numeric_limits<uint32_t>::max() - layout_size) {
                            _fail(path, "structure layout overflows uint32.");
                            return false;
                        }
                        layout_size += member->size();
                        members.emplace_back(member);
                    }
                    if (alignment < max_member_alignment) {
                        _fail(path, "structure alignment is smaller than a member alignment.");
                        return false;
                    }
                    if (layout_size > std::numeric_limits<uint32_t>::max() -
                                          (alignment - 1u)) {
                        _fail(path, "structure final alignment overflows uint32.");
                        return false;
                    }
                    _types.emplace_back(Type::structure(alignment, members));
                    break;
                }
                case Type::Tag::BUFFER: {
                    if (!_check_keys(item, {"tag", "element"}, path)) { return false; }
                    const Type *element{};
                    if (!_type_index(
                            _member(item, "element", path), i, element,
                            luisa::format("{}.element", path))) {
                        return false;
                    }
                    if (element != nullptr &&
                        (element->is_resource() || element->is_custom())) {
                        _fail(path, "buffer element must be a portable data type.");
                        return false;
                    }
                    _types.emplace_back(Type::buffer(element));
                    break;
                }
                case Type::Tag::BINDLESS_ARRAY:
                    if (!_check_keys(item, {"tag"}, path)) { return false; }
                    _types.emplace_back(Type::from("bindless_array"));
                    break;
                case Type::Tag::ACCEL:
                    if (!_check_keys(item, {"tag"}, path)) { return false; }
                    _types.emplace_back(Type::from("accel"));
                    break;
                case Type::Tag::CUSTOM: {
                    if (!_check_keys(item, {"tag", "id"}, path)) {
                        return false;
                    }
                    luisa::string_view id;
                    if (!_string(
                            _member(item, "id", path), id,
                            luisa::format("{}.id", path))) {
                        return false;
                    }
                    if (id != ast_json_indirect_dispatch_buffer_type_name &&
                        id != ast_json_ray_query_all_type_name &&
                        id != ast_json_ray_query_any_type_name) {
                        _fail(path, "custom type is not on the remote-safe allowlist.");
                        return false;
                    }
                    _types.emplace_back(Type::custom(id));
                    break;
                }
                case Type::Tag::COOPERATIVE_VECTOR:
                case Type::Tag::COOPERATIVE_VECTOR_REF:
                case Type::Tag::COOPERATIVE_MATRIX_REF:
                    _fail(path, "type is not supported by AST JSON schema v1.");
                    return false;
                default:
                    _fail(path, "unknown or unsupported type tag.");
                    return false;
            }
        }
        return true;
    }

    [[nodiscard]] static bool _constant_type_supported(
        const Type *type) noexcept {
        if (type == nullptr || type->is_resource() || type->is_custom()) {
            return false;
        }
        switch (type->tag()) {
            case Type::Tag::BOOL:
            case Type::Tag::INT8:
            case Type::Tag::UINT8:
            case Type::Tag::INT16:
            case Type::Tag::UINT16:
            case Type::Tag::INT32:
            case Type::Tag::UINT32:
            case Type::Tag::INT64:
            case Type::Tag::UINT64:
            case Type::Tag::FLOAT16:
            case Type::Tag::FLOAT32:
            case Type::Tag::FLOAT64: return true;
            case Type::Tag::VECTOR:
            case Type::Tag::MATRIX:
            case Type::Tag::ARRAY:
                return _constant_type_supported(type->element());
            case Type::Tag::STRUCTURE:
                return std::all_of(
                    type->members().begin(), type->members().end(),
                    [](auto member) noexcept {
                        return _constant_type_supported(member);
                    });
            default: return false;
        }
    }

    [[nodiscard]] bool _decode_base64(
        yyjson_val *value, size_t expected_size,
        luisa::vector<std::byte> &bytes,
        luisa::string_view path) noexcept {
        luisa::string_view text;
        if (!_string(value, text, path)) { return false; }
        if (text.size() % 4u != 0u) {
            _fail(path, "Base64 length must be divisible by four.");
            return false;
        }
        auto padding = size_t{0u};
        if (!text.empty() && text.back() == '=') { padding++; }
        if (text.size() > 1u && text[text.size() - 2u] == '=') { padding++; }
        auto decoded_size = text.size() / 4u * 3u - padding;
        if (decoded_size != expected_size) {
            _fail(path, luisa::format(
                            "decoded byte length {} does not match expected {}.",
                            decoded_size, expected_size));
            return false;
        }
        auto decode_char = [](char c) noexcept -> int {
            if (c >= 'A' && c <= 'Z') { return c - 'A'; }
            if (c >= 'a' && c <= 'z') { return c - 'a' + 26; }
            if (c >= '0' && c <= '9') { return c - '0' + 52; }
            if (c == '+') { return 62; }
            if (c == '/') { return 63; }
            return -1;
        };
        bytes.clear();
        bytes.reserve(expected_size);
        for (size_t i = 0u; i < text.size(); i += 4u) {
            uint32_t packed{};
            for (size_t j = 0u; j < 4u; j++) {
                auto c = text[i + j];
                if (c == '=') {
                    if (i + 4u != text.size() || j < 2u) {
                        _fail(path, "invalid Base64 padding.");
                        return false;
                    }
                    packed <<= 6u;
                } else {
                    if (padding != 0u && i + j >= text.size() - padding) {
                        _fail(path, "invalid Base64 padding placement.");
                        return false;
                    }
                    auto digit = decode_char(c);
                    if (digit < 0) {
                        _fail(path, "invalid Base64 character.");
                        return false;
                    }
                    packed = packed << 6u | static_cast<uint32_t>(digit);
                }
            }
            for (size_t j = 0u; j < 3u && bytes.size() < expected_size; j++) {
                bytes.emplace_back(static_cast<std::byte>(
                    (packed >> (16u - static_cast<uint32_t>(j) * 8u)) & 0xffu));
            }
        }
        return bytes.size() == expected_size;
    }

    [[nodiscard]] bool _decode_constants(yyjson_val *array) noexcept {
        if (array == nullptr) { return true; }
        if (!_array(array, _limits.max_nodes, "$.constants")) { return false; }
        _constants.reserve(yyjson_arr_size(array));
        for (size_t i = 0u; i < yyjson_arr_size(array); i++) {
            auto item = yyjson_arr_get(array, i);
            auto path = luisa::format("$.constants[{}]", i);
            if (!_check_keys(item, {"type", "raw"}, path)) { return false; }
            const Type *type{};
            if (!_type_index(
                    _member(item, "type", path), _types.size(), type,
                    luisa::format("{}.type", path))) {
                return false;
            }
            if (!_constant_type_supported(type)) {
                _fail(path, "constant uses a type unsupported by schema v1.");
                return false;
            }
            auto size = type->size();
            if (size > _limits.max_constant_bytes -
                           std::min(_constant_bytes, _limits.max_constant_bytes)) {
                _fail(path, "constants exceed the configured byte limit.");
                return false;
            }
            luisa::vector<std::byte> bytes;
            if (!_decode_base64(
                    _member(item, "raw", path), size, bytes,
                    luisa::format("{}.raw", path))) {
                return false;
            }
            _constant_bytes += size;
            _constants.emplace_back(ConstantData::create(type, bytes.data(), bytes.size()));
        }
        return true;
    }

    struct FunctionContext {
        detail::FunctionBuilder *builder{};
        luisa::vector<const RefExpr *> variables;
        const Type *declared_return_type{};
        size_t function_index{};
        Function::Tag tag{};
    };

    [[nodiscard]] bool _curve_bases(
        yyjson_val *value, CurveBasisSet &set,
        luisa::string_view path) noexcept {
        if (value == nullptr) { return true; }
        if (!_array(value, curve_basis_count, path)) { return false; }
        for (size_t i = 0u; i < yyjson_arr_size(value); i++) {
            CurveBasis basis{};
            if (!_enum(yyjson_arr_get(value, i), basis,
                       luisa::format("{}[{}]", path, i))) {
                return false;
            }
            if (set.test(basis)) {
                _fail(path, "duplicate curve basis.");
                return false;
            }
            set.mark(basis);
        }
        return true;
    }

    [[nodiscard]] bool _usage(
        yyjson_val *value, Usage &usage,
        luisa::string_view path) noexcept {
        if (!_enum(value, usage, path)) { return false; }
        switch (usage) {
            case Usage::NONE:
            case Usage::READ:
            case Usage::WRITE:
            case Usage::READ_WRITE: return true;
        }
        _fail(path, "invalid variable usage mask.");
        return false;
    }

    [[nodiscard]] bool _int32(
        yyjson_val *value, int32_t &result,
        luisa::string_view path) noexcept {
        if (yyjson_is_sint(value)) {
            auto v = yyjson_get_sint(value);
            if (v < std::numeric_limits<int32_t>::min() ||
                v > std::numeric_limits<int32_t>::max()) {
                _fail(path, "integer is outside the int32 range.");
                return false;
            }
            result = static_cast<int32_t>(v);
            return true;
        }
        if (yyjson_is_uint(value)) {
            auto v = yyjson_get_uint(value);
            if (v > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                _fail(path, "integer is outside the int32 range.");
                return false;
            }
            result = static_cast<int32_t>(v);
            return true;
        }
        _fail(path, "expected an int32 integer.");
        return false;
    }

    [[nodiscard]] bool _constant_index(
        yyjson_val *value, ConstantData &constant,
        luisa::string_view path) noexcept {
        uint64_t index{};
        if (!_uint64(value, index, path)) { return false; }
        if (index >= _constants.size()) {
            _fail(path, "constant index is out of range.");
            return false;
        }
        constant = _constants[static_cast<size_t>(index)];
        return true;
    }

    template<size_t index = 0u>
    [[nodiscard]] const LiteralExpr *_make_literal(
        FunctionContext &context, const Type *type,
        luisa::span<const std::byte> bytes,
        luisa::string_view path) noexcept {
        using Variant = LiteralExpr::Value::variant_type;
        if constexpr (index < luisa::variant_size_v<Variant>) {
            using T = luisa::variant_alternative_t<index, Variant>;
            if (Type::of<T>() == type) {
                if (bytes.size() != sizeof(T)) {
                    _fail(path, "literal byte size does not match its C++ value type.");
                    return nullptr;
                }
                T value{};
                std::memcpy(&value, bytes.data(), sizeof(T));
                if constexpr (std::is_same_v<T, bool>) {
                    auto raw = std::to_integer<uint8_t>(bytes.front());
                    if (raw > 1u) {
                        _fail(path, "boolean literal must have value zero or one.");
                        return nullptr;
                    }
                }
                return context.builder->literal(
                    type, LiteralExpr::Value{value});
            }
            return _make_literal<index + 1u>(context, type, bytes, path);
        } else {
            _fail(path, luisa::format(
                            "type '{}' is not a supported literal type.",
                            type == nullptr ? "void" : type->description()));
            return nullptr;
        }
    }

    [[nodiscard]] bool _builtin_call_arity(
        CallOp op, luisa::span<const Expression *const> arguments,
        luisa::string_view path) noexcept {
        auto require = [&](size_t count, bool exact = false) noexcept {
            if ((exact && arguments.size() != count) ||
                (!exact && arguments.size() < count)) {
                _fail(path, luisa::format(
                                "operation {} requires {}{} argument(s), got {}.",
                                luisa::to_string(op), exact ? "exactly " : "at least ",
                                count, arguments.size()));
                return false;
            }
            return true;
        };
        if (op == CallOp::PACK) { return require(3u, true); }
        if (op == CallOp::CLUSTER_LAUNCH_CONTROL_TRY_CANCEL ||
            op == CallOp::CLUSTER_LAUNCH_CONTROL_TRY_CANCEL_MULTICAST) {
            return require(2u, true);
        }
        if (op == CallOp::ASYNC_COPY) { return require(7u, true); }
        if (op == CallOp::INDIRECT_SET_DISPATCH_COUNT) {
            return require(2u, true);
        }
        if (op == CallOp::INDIRECT_SET_DISPATCH_KERNEL) {
            return require(5u, true);
        }
        if (op == CallOp::MBARRIER_INIT ||
            op == CallOp::MBARRIER_ARRIVE_EXPECT_TX ||
            op == CallOp::MBARRIER_TRY_WAIT_PARITY) {
            return require(2u, true);
        }
        switch (op) {
            case CallOp::BUFFER_VOLATILE_WRITE:
            case CallOp::BUFFER_WRITE:
            case CallOp::BINDLESS_BUFFER_WRITE:
            case CallOp::UNIFORM_BINDLESS_BUFFER_WRITE:
            case CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_WRITE:
            case CallOp::TYPED_BINDLESS_BUFFER_WRITE:
            case CallOp::BYTE_BUFFER_VOLATILE_WRITE:
            case CallOp::BYTE_BUFFER_WRITE:
            case CallOp::TEXTURE_WRITE:
            case CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
            case CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY:
            case CallOp::RAY_TRACING_SET_INSTANCE_OPACITY:
            case CallOp::RAY_TRACING_SET_INSTANCE_USER_ID:
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT:
            case CallOp::RAY_QUERY_COMMIT_TRIANGLE:
            case CallOp::RAY_QUERY_COMMIT_PROCEDURAL:
            case CallOp::RAY_QUERY_TERMINATE:
            case CallOp::RAY_QUERY_PROCEED:
            case CallOp::GRADIENT_MARKER:
            case CallOp::ACCUMULATE_GRADIENT:
            case CallOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE:
            case CallOp::COOPERATIVE_VECTOR_ACCUMULATE:
            case CallOp::COOPERATIVE_VECTOR_STORE:
            case CallOp::BINDLESS_COOPERATIVE_VECTOR_STORE:
            case CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE:
            case CallOp::COOPERATIVE_VECTOR_WORKGROUP_STORE:
                return require(1u);
            default: break;
        }
        if (is_atomic_operation(op)) {
            if (!require(op == CallOp::ATOMIC_COMPARE_EXCHANGE ? 3u : 2u)) {
                return false;
            }
            auto type = arguments.front()->type();
            if (type == nullptr || (!type->is_buffer() && !type->is_array()) ||
                type->element() == nullptr) {
                _fail(path, "atomic access chain must begin with a typed buffer or array.");
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] const Expression *_decode_expr(
        yyjson_val *object, FunctionContext &context,
        size_t depth, luisa::string_view path,
        bool allow_null = false, bool allow_void_call = false) noexcept {
        if (yyjson_is_null(object)) {
            if (allow_null) { return nullptr; }
            _fail(path, "null expression is not allowed here.");
            return nullptr;
        }
        if (!_count_node(depth, path) || !yyjson_is_obj(object)) {
            if (_error.empty()) { _fail(path, "expected an expression object."); }
            return nullptr;
        }
        Expression::Tag tag{};
        const Type *type{};
        if (!_enum(_member(object, "tag", path), tag,
                   luisa::format("{}.tag", path)) ||
            !_type_index(
                _member(object, "type", path), _types.size(), type,
                luisa::format("{}.type", path))) {
            return nullptr;
        }
        if (type == nullptr && !(tag == Expression::Tag::CALL && allow_void_call)) {
            _fail(path, "only a top-level call expression may have void type.");
            return nullptr;
        }
        switch (tag) {
            case Expression::Tag::UNARY: {
                if (!_check_keys(object, {"tag", "type", "op", "operand"}, path)) {
                    return nullptr;
                }
                UnaryOp op{};
                auto operand = _decode_expr(
                    _member(object, "operand", path), context, depth + 1u,
                    luisa::format("{}.operand", path));
                if (!_error.empty() ||
                    !_enum(_member(object, "op", path), op,
                           luisa::format("{}.op", path))) {
                    return nullptr;
                }
                return context.builder->unary(type, op, operand);
            }
            case Expression::Tag::BINARY: {
                if (!_check_keys(object, {"tag", "type", "op", "lhs", "rhs"}, path)) {
                    return nullptr;
                }
                BinaryOp op{};
                auto lhs = _decode_expr(
                    _member(object, "lhs", path), context, depth + 1u,
                    luisa::format("{}.lhs", path));
                auto rhs = _decode_expr(
                    _member(object, "rhs", path), context, depth + 1u,
                    luisa::format("{}.rhs", path));
                if (!_error.empty() ||
                    !_enum(_member(object, "op", path), op,
                           luisa::format("{}.op", path))) {
                    return nullptr;
                }
                return context.builder->binary(type, op, lhs, rhs);
            }
            case Expression::Tag::MEMBER: {
                auto swizzle = _member(object, "swizzle", path, false);
                auto member = _member(object, "member", path, false);
                if ((swizzle == nullptr) == (member == nullptr)) {
                    _fail(path, "member expression requires exactly one of member or swizzle.");
                    return nullptr;
                }
                if (!_check_keys(
                        object,
                        swizzle == nullptr ?
                            std::initializer_list<luisa::string_view>{"tag", "type", "self", "member"} :
                            std::initializer_list<luisa::string_view>{"tag", "type", "self", "swizzle"},
                        path)) {
                    return nullptr;
                }
                auto self = _decode_expr(
                    _member(object, "self", path), context, depth + 1u,
                    luisa::format("{}.self", path));
                if (!_error.empty()) { return nullptr; }
                if (member != nullptr) {
                    uint32_t index{};
                    if (!_uint32(member, index, luisa::format("{}.member", path))) {
                        return nullptr;
                    }
                    if (self->type() == nullptr || !self->type()->is_structure() ||
                        index >= self->type()->members().size() ||
                        self->type()->members()[index] != type) {
                        _fail(path, "invalid structure member access.");
                        return nullptr;
                    }
                    return context.builder->member(type, self, index);
                }
                luisa::string_view text;
                if (!_string(swizzle, text, luisa::format("{}.swizzle", path)) ||
                    text.empty() || text.size() > 4u ||
                    self->type() == nullptr || !self->type()->is_vector()) {
                    if (_error.empty()) { _fail(path, "invalid vector swizzle."); }
                    return nullptr;
                }
                uint64_t code{};
                for (size_t i = 0u; i < text.size(); i++) {
                    auto component = std::string_view{"xyzw"}.find(text[i]);
                    if (component == std::string_view::npos ||
                        component >= self->type()->dimension()) {
                        _fail(path, "swizzle component is outside the source vector.");
                        return nullptr;
                    }
                    code |= static_cast<uint64_t>(component) << (i * 4u);
                }
                return context.builder->swizzle(type, self, text.size(), code);
            }
            case Expression::Tag::ACCESS: {
                if (!_check_keys(object, {"tag", "type", "range", "index"}, path)) {
                    return nullptr;
                }
                auto range = _decode_expr(
                    _member(object, "range", path), context, depth + 1u,
                    luisa::format("{}.range", path));
                auto index = _decode_expr(
                    _member(object, "index", path), context, depth + 1u,
                    luisa::format("{}.index", path));
                if (!_error.empty()) { return nullptr; }
                return context.builder->access(type, range, index);
            }
            case Expression::Tag::LITERAL: {
                if (!_check_keys(object, {"tag", "type", "value"}, path)) {
                    return nullptr;
                }
                luisa::vector<std::byte> bytes;
                if (!_decode_base64(
                        _member(object, "value", path), type->size(), bytes,
                        luisa::format("{}.value", path))) {
                    return nullptr;
                }
                return _make_literal(context, type, bytes,
                                     luisa::format("{}.value", path));
            }
            case Expression::Tag::REF: {
                if (!_check_keys(object, {"tag", "type", "variable"}, path)) {
                    return nullptr;
                }
                uint64_t index{};
                if (!_uint64(
                        _member(object, "variable", path), index,
                        luisa::format("{}.variable", path))) {
                    return nullptr;
                }
                if (index >= context.variables.size()) {
                    _fail(path, "variable index is out of range.");
                    return nullptr;
                }
                auto ref = context.variables[static_cast<size_t>(index)];
                if (ref->type() != type) {
                    _fail(path, "reference type does not match its variable.");
                    return nullptr;
                }
                return ref;
            }
            case Expression::Tag::CONSTANT: {
                if (!_check_keys(object, {"tag", "type", "data"}, path)) {
                    return nullptr;
                }
                ConstantData data;
                if (!_constant_index(
                        _member(object, "data", path), data,
                        luisa::format("{}.data", path))) {
                    return nullptr;
                }
                if (data.type() != type) {
                    _fail(path, "constant type does not match its table entry.");
                    return nullptr;
                }
                return context.builder->constant(data);
            }
            case Expression::Tag::CALL: {
                auto args_value = _member(object, "arguments", path);
                if (!_array(args_value, _limits.max_nodes,
                            luisa::format("{}.arguments", path))) {
                    return nullptr;
                }
                CallOp op{};
                if (!_enum(_member(object, "op", path), op,
                           luisa::format("{}.op", path))) {
                    return nullptr;
                }
                luisa::vector<const Expression *> arguments;
                arguments.reserve(yyjson_arr_size(args_value));
                for (size_t i = 0u; i < yyjson_arr_size(args_value); i++) {
                    auto argument = _decode_expr(
                        yyjson_arr_get(args_value, i), context, depth + 1u,
                        luisa::format("{}.arguments[{}]", path, i));
                    if (!_error.empty() || argument == nullptr) { return nullptr; }
                    arguments.emplace_back(argument);
                }
                if (op == CallOp::EXTERNAL ||
                    _member(object, "external", path, false) != nullptr) {
                    _fail(path, "external calls are not supported by schema v1.");
                    return nullptr;
                }
                if (op == CallOp::CUSTOM) {
                    if (!_check_keys(
                            object, {"tag", "type", "op", "custom", "arguments"}, path)) {
                        return nullptr;
                    }
                    uint64_t index{};
                    if (!_uint64(
                            _member(object, "custom", path), index,
                            luisa::format("{}.custom", path))) {
                        return nullptr;
                    }
                    if (index >= context.function_index || index >= _functions.size()) {
                        _fail(path, "custom callable index is not topologically ordered.");
                        return nullptr;
                    }
                    Function callable{_functions[static_cast<size_t>(index)].get()};
                    if (callable.return_type() != type ||
                        callable.arguments().size() != arguments.size()) {
                        _fail(path, "custom callable signature does not match the call.");
                        return nullptr;
                    }
                    for (size_t i = 0u; i < arguments.size(); i++) {
                        if (callable.arguments()[i].type() != arguments[i]->type()) {
                            _fail(path, "custom callable argument type mismatch.");
                            return nullptr;
                        }
                    }
                    return context.builder->call(type, callable, arguments);
                }
                if (!_check_keys(
                        object,
                        _member(object, "curve_bases", path, false) == nullptr ?
                            std::initializer_list<luisa::string_view>{"tag", "type", "op", "arguments"} :
                            std::initializer_list<luisa::string_view>{"tag", "type", "op", "arguments", "curve_bases"},
                        path)) {
                    return nullptr;
                }
                if (!is_builtin_operation(op)) {
                    _fail(path, "call operation is neither builtin nor custom.");
                    return nullptr;
                }
                CurveBasisSet curve_bases;
                if (!_curve_bases(
                        _member(object, "curve_bases", path, false), curve_bases,
                        luisa::format("{}.curve_bases", path)) ||
                    !_builtin_call_arity(op, arguments, path)) {
                    return nullptr;
                }
                return context.builder->call(type, op, arguments, curve_bases);
            }
            case Expression::Tag::CAST: {
                if (!_check_keys(object, {"tag", "type", "op", "expression"}, path)) {
                    return nullptr;
                }
                CastOp op{};
                auto expression = _decode_expr(
                    _member(object, "expression", path), context, depth + 1u,
                    luisa::format("{}.expression", path));
                if (!_error.empty() ||
                    !_enum(_member(object, "op", path), op,
                           luisa::format("{}.op", path))) {
                    return nullptr;
                }
                return context.builder->cast(type, op, expression);
            }
            case Expression::Tag::TYPE_ID: {
                if (!_check_keys(object, {"tag", "type", "data_type"}, path)) {
                    return nullptr;
                }
                const Type *data_type{};
                if (!_type_index(
                        _member(object, "data_type", path), _types.size(), data_type,
                        luisa::format("{}.data_type", path)) ||
                    type != Type::of<ulong>()) {
                    if (_error.empty()) { _fail(path, "invalid type-id result type."); }
                    return nullptr;
                }
                return context.builder->type_id(data_type);
            }
            case Expression::Tag::STRING_ID: {
                if (!_check_keys(object, {"tag", "type", "data"}, path)) {
                    return nullptr;
                }
                luisa::string_view text;
                if (!_string(
                        _member(object, "data", path), text,
                        luisa::format("{}.data", path)) ||
                    type != Type::of<ulong>()) {
                    if (_error.empty()) { _fail(path, "invalid string-id result type."); }
                    return nullptr;
                }
                return context.builder->string_id(luisa::string{text});
            }
            case Expression::Tag::FUNC_REF: {
                if (!_check_keys(object, {"tag", "type", "func"}, path)) {
                    return nullptr;
                }
                uint64_t index{};
                if (!_uint64(
                        _member(object, "func", path), index,
                        luisa::format("{}.func", path))) {
                    return nullptr;
                }
                if (index >= context.function_index || index >= _functions.size() ||
                    type != Type::of<uint64_t>()) {
                    _fail(path, "invalid function-reference target or result type.");
                    return nullptr;
                }
                return context.builder->func_ref(
                    Function{_functions[static_cast<size_t>(index)].get()});
            }
            case Expression::Tag::CPUCUSTOM:
            case Expression::Tag::GPUCUSTOM:
                _fail(path, "custom operation is not supported by remote-safe AST JSON.");
                return nullptr;
        }
        _fail(path, "unknown expression tag.");
        return nullptr;
    }

    [[nodiscard]] bool _decode_scope(
        yyjson_val *object, FunctionContext &context,
        ScopeStmt *scope, size_t depth,
        luisa::string_view path) noexcept {
        if (!_count_node(depth, path) ||
            !_check_keys(object, {"tag", "statements"}, path)) {
            return false;
        }
        Statement::Tag tag{};
        if (!_enum(_member(object, "tag", path), tag,
                   luisa::format("{}.tag", path)) ||
            tag != Statement::Tag::SCOPE) {
            if (_error.empty()) { _fail(path, "expected a scope statement."); }
            return false;
        }
        auto statements = _member(object, "statements", path);
        if (!_array(statements, _limits.max_nodes,
                    luisa::format("{}.statements", path))) {
            return false;
        }
        auto decode_children = [&] {
            for (size_t i = 0u; i < yyjson_arr_size(statements); i++) {
                if (!_decode_statement(
                        yyjson_arr_get(statements, i), context, depth + 1u,
                        luisa::format("{}.statements[{}]", path, i))) {
                    return false;
                }
            }
            return true;
        };
        return context.builder->with(scope, decode_children);
    }

    [[nodiscard]] bool _decode_statement(
        yyjson_val *object, FunctionContext &context,
        size_t depth, luisa::string_view path) noexcept {
        if (!_count_node(depth, path) || !yyjson_is_obj(object)) {
            if (_error.empty()) { _fail(path, "expected a statement object."); }
            return false;
        }
        Statement::Tag tag{};
        if (!_enum(_member(object, "tag", path), tag,
                   luisa::format("{}.tag", path))) {
            return false;
        }
        switch (tag) {
            case Statement::Tag::BREAK:
                if (!_check_keys(object, {"tag"}, path)) { return false; }
                context.builder->break_();
                return true;
            case Statement::Tag::CONTINUE:
                if (!_check_keys(object, {"tag"}, path)) { return false; }
                context.builder->continue_();
                return true;
            case Statement::Tag::RETURN: {
                if (!_check_keys(object, {"tag", "expression"}, path)) { return false; }
                auto expression_value = _member(object, "expression", path);
                auto expression = _decode_expr(
                    expression_value, context, depth + 1u,
                    luisa::format("{}.expression", path), true);
                if (!_error.empty()) { return false; }
                if (context.tag == Function::Tag::KERNEL && expression != nullptr) {
                    _fail(path, "compute kernels cannot return a value.");
                    return false;
                }
                if (context.tag == Function::Tag::CALLABLE &&
                    ((expression == nullptr) != (context.declared_return_type == nullptr) ||
                     (expression != nullptr &&
                      expression->type() != context.declared_return_type))) {
                    _fail(path, "return expression does not match the declared return type.");
                    return false;
                }
                context.builder->return_(expression);
                return true;
            }
            case Statement::Tag::SCOPE:
                _fail(path, "standalone nested scopes are not supported by schema v1.");
                return false;
            case Statement::Tag::IF: {
                if (!_check_keys(object, {"tag", "condition", "true_branch", "false_branch"}, path)) {
                    return false;
                }
                auto condition = _decode_expr(
                    _member(object, "condition", path), context, depth + 1u,
                    luisa::format("{}.condition", path));
                if (_error.empty() &&
                    (condition->type() == nullptr || !condition->type()->is_bool())) {
                    _fail(path, "if condition must be boolean.");
                }
                if (!_error.empty()) { return false; }
                auto statement = context.builder->if_(condition);
                return _decode_scope(
                           _member(object, "true_branch", path), context,
                           statement->true_branch(), depth + 1u,
                           luisa::format("{}.true_branch", path)) &&
                       _decode_scope(
                           _member(object, "false_branch", path), context,
                           statement->false_branch(), depth + 1u,
                           luisa::format("{}.false_branch", path));
            }
            case Statement::Tag::LOOP: {
                if (!_check_keys(object, {"tag", "body"}, path)) { return false; }
                auto statement = context.builder->loop_();
                return _decode_scope(
                    _member(object, "body", path), context,
                    statement->body(), depth + 1u,
                    luisa::format("{}.body", path));
            }
            case Statement::Tag::EXPR: {
                if (!_check_keys(object, {"tag", "expression"}, path)) { return false; }
                auto expression = _decode_expr(
                    _member(object, "expression", path), context, depth + 1u,
                    luisa::format("{}.expression", path), false, true);
                if (!_error.empty()) { return false; }
                if (expression != nullptr) {
                    context.builder->expression_statement(expression);
                }
                return true;
            }
            case Statement::Tag::SWITCH: {
                if (!_check_keys(object, {"tag", "expression", "body"}, path)) {
                    return false;
                }
                auto expression = _decode_expr(
                    _member(object, "expression", path), context, depth + 1u,
                    luisa::format("{}.expression", path));
                if (_error.empty() &&
                    !(expression->type()->is_int32() || expression->type()->is_uint32())) {
                    _fail(path, "switch expression must be int32 or uint32.");
                }
                if (!_error.empty()) { return false; }
                auto statement = context.builder->switch_(expression);
                return _decode_scope(
                    _member(object, "body", path), context,
                    statement->body(), depth + 1u,
                    luisa::format("{}.body", path));
            }
            case Statement::Tag::SWITCH_CASE: {
                if (!_check_keys(object, {"tag", "value", "body"}, path)) { return false; }
                int32_t value{};
                if (!_int32(
                        _member(object, "value", path), value,
                        luisa::format("{}.value", path))) {
                    return false;
                }
                auto literal = context.builder->literal(Type::of<int>(), value);
                auto statement = context.builder->case_(literal);
                return _decode_scope(
                    _member(object, "body", path), context,
                    statement->body(), depth + 1u,
                    luisa::format("{}.body", path));
            }
            case Statement::Tag::SWITCH_DEFAULT: {
                if (!_check_keys(object, {"tag", "body"}, path)) { return false; }
                auto statement = context.builder->default_();
                return _decode_scope(
                    _member(object, "body", path), context,
                    statement->body(), depth + 1u,
                    luisa::format("{}.body", path));
            }
            case Statement::Tag::ASSIGN: {
                if (!_check_keys(object, {"tag", "lhs", "rhs"}, path)) { return false; }
                auto lhs = _decode_expr(
                    _member(object, "lhs", path), context, depth + 1u,
                    luisa::format("{}.lhs", path));
                auto rhs = _decode_expr(
                    _member(object, "rhs", path), context, depth + 1u,
                    luisa::format("{}.rhs", path));
                if (_error.empty() && !is_assignable_expression(lhs)) {
                    _fail(path, "assignment left-hand side is not assignable.");
                }
                if (_error.empty() &&
                    !assignment_types_compatible(
                        lhs->type(), rhs->type())) {
                    _fail(path, luisa::format(
                                    "assignment operand types do not match ('{}' vs '{}').",
                                    lhs->type()->description(),
                                    rhs->type()->description()));
                }
                if (!_error.empty()) { return false; }
                context.builder->assign(lhs, rhs);
                return true;
            }
            case Statement::Tag::FOR: {
                if (!_check_keys(object, {"tag", "variable", "condition", "step", "body"}, path)) {
                    return false;
                }
                auto variable = _decode_expr(
                    _member(object, "variable", path), context, depth + 1u,
                    luisa::format("{}.variable", path));
                auto condition = _decode_expr(
                    _member(object, "condition", path), context, depth + 1u,
                    luisa::format("{}.condition", path));
                auto step = _decode_expr(
                    _member(object, "step", path), context, depth + 1u,
                    luisa::format("{}.step", path));
                if (_error.empty() &&
                    (!is_assignable_expression(variable) ||
                     !assignment_types_compatible(
                         variable->type(), step->type()) ||
                     !condition->type()->is_bool())) {
                    _fail(path, "for-loop operands are invalid.");
                }
                if (!_error.empty()) { return false; }
                auto statement = context.builder->for_(variable, condition, step);
                return _decode_scope(
                    _member(object, "body", path), context,
                    statement->body(), depth + 1u,
                    luisa::format("{}.body", path));
            }
            case Statement::Tag::COMMENT: {
                if (!_check_keys(object, {"tag", "comment"}, path)) { return false; }
                luisa::string_view comment;
                if (!_string(
                        _member(object, "comment", path), comment,
                        luisa::format("{}.comment", path))) {
                    return false;
                }
                context.builder->comment_(luisa::string{comment});
                return true;
            }
            case Statement::Tag::RAY_QUERY: {
                if (!_check_keys(
                        object, {"tag", "query", "on_triangle_candidate", "on_procedural_candidate"}, path)) {
                    return false;
                }
                auto query = _decode_expr(
                    _member(object, "query", path), context, depth + 1u,
                    luisa::format("{}.query", path));
                if (_error.empty() &&
                    (query->tag() != Expression::Tag::REF ||
                     !is_ray_query_type(query->type()))) {
                    _fail(path, "ray-query value must be a ray-query reference expression.");
                }
                if (!_error.empty()) { return false; }
                auto statement = context.builder->ray_query_(
                    static_cast<const RefExpr *>(query));
                return _decode_scope(
                           _member(object, "on_triangle_candidate", path), context,
                           statement->on_triangle_candidate(), depth + 1u,
                           luisa::format("{}.on_triangle_candidate", path)) &&
                       _decode_scope(
                           _member(object, "on_procedural_candidate", path), context,
                           statement->on_procedural_candidate(), depth + 1u,
                           luisa::format("{}.on_procedural_candidate", path));
            }
            case Statement::Tag::AUTO_DIFF: {
                if (!_check_keys(object, {"tag", "body"}, path)) { return false; }
                auto statement = context.builder->autodiff_();
                return _decode_scope(
                    _member(object, "body", path), context,
                    statement->body(), depth + 1u,
                    luisa::format("{}.body", path));
            }
            case Statement::Tag::PRINT: {
                if (!_check_keys(object, {"tag", "format", "arguments"}, path)) {
                    return false;
                }
                luisa::string_view format;
                auto args_value = _member(object, "arguments", path);
                if (!_string(
                        _member(object, "format", path), format,
                        luisa::format("{}.format", path)) ||
                    !_array(args_value, _limits.max_nodes,
                            luisa::format("{}.arguments", path))) {
                    return false;
                }
                luisa::vector<const Expression *> arguments;
                arguments.reserve(yyjson_arr_size(args_value));
                for (size_t i = 0u; i < yyjson_arr_size(args_value); i++) {
                    auto argument = _decode_expr(
                        yyjson_arr_get(args_value, i), context, depth + 1u,
                        luisa::format("{}.arguments[{}]", path, i));
                    if (!_error.empty()) { return false; }
                    arguments.emplace_back(argument);
                }
                context.builder->print_(luisa::string{format}, arguments);
                return true;
            }
            case Statement::Tag::SUSPEND:
                _fail(path, "coroutine suspend is not supported by schema v1.");
                return false;
            case Statement::Tag::DEBUG_BREAK:
                _fail(path, "debug-break callbacks are not portable.");
                return false;
        }
        _fail(path, "unknown statement tag.");
        return false;
    }

    [[nodiscard]] const RefExpr *_decode_variable(
        yyjson_val *object, yyjson_val *binding_value,
        FunctionContext &context, bool is_argument,
        size_t index, luisa::string_view path) noexcept {
        if (!_check_keys(
                object,
                _member(object, "name", path, false) == nullptr ?
                    std::initializer_list<luisa::string_view>{"tag", "type", "usage"} :
                    std::initializer_list<luisa::string_view>{"tag", "type", "usage", "name"},
                path)) {
            return nullptr;
        }
        luisa::string_view tag_name;
        const Type *type{};
        Usage usage{};
        if (!_string(
                _member(object, "tag", path), tag_name,
                luisa::format("{}.tag", path)) ||
            !_type_index(
                _member(object, "type", path), _types.size(), type,
                luisa::format("{}.type", path)) ||
            !_usage(
                _member(object, "usage", path), usage,
                luisa::format("{}.usage", path))) {
            return nullptr;
        }
        if (type == nullptr) {
            _fail(path, "variable type must not be void.");
            return nullptr;
        }
        auto parse_variable_tag = [&]() noexcept -> luisa::optional<Variable::Tag> {
            if (tag_name == "ARGUMENT") { return luisa::nullopt; }
            return magic_enum::enum_cast<Variable::Tag>(
                std::string_view{tag_name.data(), tag_name.size()});
        };
        auto tag = parse_variable_tag();
        if (tag_name != "ARGUMENT" && !tag) {
            _fail(path, luisa::format("unknown variable tag '{}'.", tag_name));
            return nullptr;
        }

        const RefExpr *ref{};
        if (binding_value != nullptr) {
            if (!is_argument || context.tag != Function::Tag::KERNEL) {
                _fail(path, "only kernel arguments may carry resource bindings.");
                return nullptr;
            }
            luisa::string_view binding_tag;
            if (!_string(
                    _member(binding_value, "tag", path), binding_tag,
                    luisa::format("{}.binding.tag", path))) {
                return nullptr;
            }
            uint64_t serialized_handle{};
            if (!_decimal_uint64(
                    _member(binding_value, "handle", path), serialized_handle,
                    luisa::format("{}.binding.handle", path))) {
                return nullptr;
            }
            luisa::string resolver_error;
            if (binding_tag == "BUFFER") {
                if (!tag || *tag != Variable::Tag::BUFFER ||
                    !is_buffer_variable_type(type) ||
                    !_check_keys(binding_value, {"tag", "handle", "offset", "size"},
                                 luisa::format("{}.binding", path))) {
                    if (_error.empty()) { _fail(path, "buffer binding does not match its variable."); }
                    return nullptr;
                }
                uint64_t offset_u64{};
                uint64_t size_u64{};
                if (!_decimal_uint64(
                        _member(binding_value, "offset", path), offset_u64,
                        luisa::format("{}.binding.offset", path)) ||
                    !_decimal_uint64(
                        _member(binding_value, "size", path), size_u64,
                        luisa::format("{}.binding.size", path)) ||
                    offset_u64 > std::numeric_limits<size_t>::max() ||
                    size_u64 > std::numeric_limits<size_t>::max()) {
                    if (_error.empty()) { _fail(path, "buffer binding range exceeds host size_t."); }
                    return nullptr;
                }
                Function::BufferBinding binding;
                if (!_binding_resolver.resolve_buffer(
                        type, serialized_handle,
                        static_cast<size_t>(offset_u64),
                        static_cast<size_t>(size_u64), binding, resolver_error)) {
                    _fail(path, resolver_error.empty() ?
                                    "buffer binding resolver rejected the handle." :
                                    std::move(resolver_error));
                    return nullptr;
                }
                ref = context.builder->buffer_binding(
                    type, binding.handle, binding.offset, binding.size);
            } else if (binding_tag == "TEXTURE") {
                if (!tag || *tag != Variable::Tag::TEXTURE || !type->is_texture() ||
                    !_check_keys(binding_value, {"tag", "handle", "level"},
                                 luisa::format("{}.binding", path))) {
                    if (_error.empty()) { _fail(path, "texture binding does not match its variable."); }
                    return nullptr;
                }
                uint64_t level_u64{};
                if (!_decimal_uint64(
                        _member(binding_value, "level", path), level_u64,
                        luisa::format("{}.binding.level", path)) ||
                    level_u64 > std::numeric_limits<uint32_t>::max()) {
                    if (_error.empty()) { _fail(path, "texture level exceeds uint32."); }
                    return nullptr;
                }
                Function::TextureBinding binding;
                if (!_binding_resolver.resolve_texture(
                        serialized_handle, static_cast<uint32_t>(level_u64),
                        binding, resolver_error)) {
                    _fail(path, resolver_error.empty() ?
                                    "texture binding resolver rejected the handle." :
                                    std::move(resolver_error));
                    return nullptr;
                }
                ref = context.builder->texture_binding(
                    type, binding.handle, binding.level);
            } else if (binding_tag == "BINDLESS_ARRAY") {
                if (!tag || *tag != Variable::Tag::BINDLESS_ARRAY ||
                    !type->is_bindless_array() ||
                    !_check_keys(binding_value, {"tag", "handle"},
                                 luisa::format("{}.binding", path))) {
                    if (_error.empty()) { _fail(path, "bindless binding does not match its variable."); }
                    return nullptr;
                }
                Function::BindlessArrayBinding binding;
                if (!_binding_resolver.resolve_bindless_array(
                        serialized_handle, binding, resolver_error)) {
                    _fail(path, resolver_error.empty() ?
                                    "bindless binding resolver rejected the handle." :
                                    std::move(resolver_error));
                    return nullptr;
                }
                ref = context.builder->bindless_array_binding(binding.handle);
            } else if (binding_tag == "ACCEL") {
                if (!tag || *tag != Variable::Tag::ACCEL || !type->is_accel() ||
                    !_check_keys(binding_value, {"tag", "handle"},
                                 luisa::format("{}.binding", path))) {
                    if (_error.empty()) { _fail(path, "accel binding does not match its variable."); }
                    return nullptr;
                }
                Function::AccelBinding binding;
                if (!_binding_resolver.resolve_accel(
                        serialized_handle, binding, resolver_error)) {
                    _fail(path, resolver_error.empty() ?
                                    "accel binding resolver rejected the handle." :
                                    std::move(resolver_error));
                    return nullptr;
                }
                ref = context.builder->accel_binding(binding.handle);
            } else {
                _fail(path, luisa::format("unknown binding tag '{}'.", binding_tag));
                return nullptr;
            }
        } else if (is_argument) {
            if (tag_name == "ARGUMENT") {
                if (type->is_resource() || type->is_custom()) {
                    _fail(path, "ordinary argument cannot have a resource type.");
                    return nullptr;
                }
                ref = context.builder->argument(type);
            } else {
                switch (*tag) {
                    case Variable::Tag::REFERENCE:
                        ref = context.builder->reference(type);
                        break;
                    case Variable::Tag::BUFFER:
                        if (!is_buffer_variable_type(type)) {
                            _fail(path, "buffer variable must have buffer type.");
                            return nullptr;
                        }
                        ref = context.builder->buffer(type);
                        break;
                    case Variable::Tag::TEXTURE:
                        if (!type->is_texture()) {
                            _fail(path, "texture variable must have texture type.");
                            return nullptr;
                        }
                        ref = context.builder->texture(type);
                        break;
                    case Variable::Tag::BINDLESS_ARRAY:
                        if (!type->is_bindless_array()) {
                            _fail(path, "bindless variable has the wrong type.");
                            return nullptr;
                        }
                        ref = context.builder->bindless_array();
                        break;
                    case Variable::Tag::ACCEL:
                        if (!type->is_accel()) {
                            _fail(path, "accel variable has the wrong type.");
                            return nullptr;
                        }
                        ref = context.builder->accel();
                        break;
                    default:
                        _fail(path, "invalid function-argument variable tag.");
                        return nullptr;
                }
            }
        } else {
            if (tag_name == "ARGUMENT" || !tag) {
                _fail(path, "non-argument variable cannot use ARGUMENT tag.");
                return nullptr;
            }
            if (type->is_custom() &&
                (!is_ray_query_type(type) || *tag != Variable::Tag::LOCAL)) {
                _fail(path, "only allowlisted ray-query custom types may be local variables.");
                return nullptr;
            }
            switch (*tag) {
                case Variable::Tag::LOCAL: ref = context.builder->local(type); break;
                case Variable::Tag::SHARED: ref = context.builder->shared(type); break;
                case Variable::Tag::THREAD_ID: ref = context.builder->thread_id(); break;
                case Variable::Tag::BLOCK_ID: ref = context.builder->block_id(); break;
                case Variable::Tag::DISPATCH_ID: ref = context.builder->dispatch_id(); break;
                case Variable::Tag::DISPATCH_SIZE: ref = context.builder->dispatch_size(); break;
                case Variable::Tag::KERNEL_ID: ref = context.builder->kernel_id(); break;
                case Variable::Tag::WARP_LANE_COUNT: ref = context.builder->warp_lane_count(); break;
                case Variable::Tag::WARP_LANE_ID: ref = context.builder->warp_lane_id(); break;
                case Variable::Tag::RASTER_OBJECT_ID:
                case Variable::Tag::RASTER_BARYCENTRICS:
                case Variable::Tag::RASTER_FRONT_FACING:
                case Variable::Tag::RASTER_BASE_INSTANCE:
                    _fail(path, "raster builtin is not valid in a compute AST.");
                    return nullptr;
                default:
                    _fail(path, "invalid non-argument variable tag.");
                    return nullptr;
            }
        }
        if (ref == nullptr || ref->type() != type ||
            ref->variable().uid() != index) {
            _fail(path, "variable order, type, or binding is not canonical.");
            return nullptr;
        }
        context.builder->mark_variable_usage(ref->variable().uid(), usage);
        if (auto name_value = _member(object, "name", path, false)) {
            luisa::string_view name;
            if (!_string(name_value, name, luisa::format("{}.name", path))) {
                return nullptr;
            }
            context.builder->set_variable_name(ref->variable().uid(), name);
        }
        return ref;
    }

    [[nodiscard]] bool _decode_variables(
        yyjson_val *function, FunctionContext &context,
        luisa::string_view path) noexcept {
        auto variables = _member(function, "variables", path);
        auto arguments = _member(function, "arguments", path);
        if (!_array(variables, _limits.max_nodes, luisa::format("{}.variables", path)) ||
            !_array(arguments, _limits.max_nodes, luisa::format("{}.arguments", path))) {
            return false;
        }
        auto argument_count = yyjson_arr_size(arguments);
        if (argument_count > yyjson_arr_size(variables)) {
            _fail(path, "argument count exceeds variable count.");
            return false;
        }
        for (size_t i = 0u; i < argument_count; i++) {
            uint64_t index{};
            if (!_uint64(
                    yyjson_arr_get(arguments, i), index,
                    luisa::format("{}.arguments[{}]", path, i)) ||
                index != i) {
                if (_error.empty()) {
                    _fail(path, "arguments must be a canonical zero-based variable prefix.");
                }
                return false;
            }
        }
        auto bindings = _member(function, "bound_arguments", path, false);
        auto binding_count = size_t{0u};
        if (bindings != nullptr) {
            if (context.tag != Function::Tag::KERNEL ||
                !_array(bindings, argument_count,
                        luisa::format("{}.bound_arguments", path))) {
                if (_error.empty()) { _fail(path, "only kernels may have bound arguments."); }
                return false;
            }
            binding_count = yyjson_arr_size(bindings);
        }
        context.variables.reserve(yyjson_arr_size(variables));
        for (size_t i = 0u; i < yyjson_arr_size(variables); i++) {
            auto binding = i < binding_count ? yyjson_arr_get(bindings, i) : nullptr;
            auto ref = _decode_variable(
                yyjson_arr_get(variables, i), binding, context,
                i < argument_count, i,
                luisa::format("{}.variables[{}]", path, i));
            if (ref == nullptr) { return false; }
            context.variables.emplace_back(ref);
        }
        return true;
    }

    [[nodiscard]] bool _decode_function(
        yyjson_val *object, size_t index, size_t entry) noexcept {
        auto path = luisa::format("$.functions[{}]", index);
        if (!_check_keys(
                object,
                {"tag", "name", "allowed_warp_size", "curve_bases",
                 "variables", "arguments", "bound_arguments", "block_size",
                 "return_type", "body", "constants"},
                path)) {
            return false;
        }
        Function::Tag tag{};
        if (!_enum(_member(object, "tag", path), tag,
                   luisa::format("{}.tag", path))) {
            return false;
        }
        auto expected_tag = index == entry ?
                                Function::Tag::KERNEL :
                                Function::Tag::CALLABLE;
        if (tag != expected_tag) {
            _fail(path, index == entry ?
                            "entry function must be a compute kernel." :
                            "non-entry functions must be callables.");
            return false;
        }
        const Type *declared_return_type{};
        if (tag == Function::Tag::CALLABLE) {
            auto return_value = _member(object, "return_type", path);
            if (!_type_index(
                    return_value, _types.size(), declared_return_type,
                    luisa::format("{}.return_type", path)) ||
                _member(object, "block_size", path, false) != nullptr ||
                _member(object, "bound_arguments", path, false) != nullptr) {
                if (_error.empty()) { _fail(path, "callable has kernel-only metadata."); }
                return false;
            }
        } else if (_member(object, "return_type", path, false) != nullptr) {
            _fail(path, "kernel must not declare a return type.");
            return false;
        }

        auto builder = luisa::make_shared<detail::FunctionBuilder>(tag);
        auto success = true;
        {
            detail::FunctionBuilder::FunctionStackGuard guard{builder.get()};
            if (auto name_value = _member(object, "name", path, false)) {
                luisa::string_view name;
                success = _string(name_value, name, luisa::format("{}.name", path));
                if (success) { builder->set_name(name); }
            }
            if (success) {
                if (auto warp_value = _member(object, "allowed_warp_size", path, false)) {
                    uint32_t warp{};
                    success = _uint32(
                        warp_value, warp,
                        luisa::format("{}.allowed_warp_size", path));
                    if (success && warp != 1u && warp != 2u && warp != 4u &&
                        warp != 8u && warp != 16u && warp != 32u &&
                        warp != 64u && warp != 128u) {
                        _fail(path, "invalid allowed warp size.");
                        success = false;
                    }
                    if (success) { builder->set_allowed_warp_size(static_cast<uint8_t>(warp)); }
                }
            }
            if (success && tag == Function::Tag::KERNEL) {
                auto block = _member(object, "block_size", path);
                success = _array(block, 3u, luisa::format("{}.block_size", path)) &&
                          yyjson_arr_size(block) == 3u;
                uint32_t dimensions[3]{};
                for (size_t i = 0u; success && i < 3u; i++) {
                    success = _uint32(
                        yyjson_arr_get(block, i), dimensions[i],
                        luisa::format("{}.block_size[{}]", path, i));
                }
                auto product = static_cast<uint64_t>(dimensions[0]) *
                               dimensions[1] * dimensions[2];
                if (success && (product == 0u || product > 1024u ||
                                dimensions[2] > 64u)) {
                    _fail(path, "invalid compute block size.");
                    success = false;
                }
                if (success) {
                    builder->set_block_size(
                        uint3{dimensions[0], dimensions[1], dimensions[2]});
                }
            }
            FunctionContext context{
                .builder = builder.get(),
                .declared_return_type = declared_return_type,
                .function_index = index,
                .tag = tag};
            if (success) { success = _decode_variables(object, context, path); }
            if (success) {
                CurveBasisSet bases;
                success = _curve_bases(
                    _member(object, "curve_bases", path), bases,
                    luisa::format("{}.curve_bases", path));
                if (success) { builder->mark_required_curve_basis_set(bases); }
            }
            if (success) {
                auto constants = _member(object, "constants", path);
                success = _array(constants, _constants.size(),
                                 luisa::format("{}.constants", path));
                for (size_t i = 0u; success && i < yyjson_arr_size(constants); i++) {
                    ConstantData ignored;
                    success = _constant_index(
                        yyjson_arr_get(constants, i), ignored,
                        luisa::format("{}.constants[{}]", path, i));
                }
            }
            if (success) {
                success = _decode_scope(
                    _member(object, "body", path), context,
                    builder->body(), 1u,
                    luisa::format("{}.body", path));
            }
        }
        if (!success) { return false; }
        auto function = Function{builder.get()};
        if (tag == Function::Tag::CALLABLE &&
            function.return_type() != declared_return_type) {
            _fail(path, "reconstructed callable return type does not match its declaration.");
            return false;
        }
        _functions.emplace_back(
            luisa::const_pointer_cast<const detail::FunctionBuilder>(builder));
        return true;
    }

    [[nodiscard]] bool _decode_functions(
        yyjson_val *array, size_t entry) noexcept {
        if (!_array(array, _limits.max_functions, "$.functions")) { return false; }
        auto count = yyjson_arr_size(array);
        if (count == 0u || entry >= count || entry + 1u != count) {
            _fail("$.entry", "entry must name the final function in a non-empty table.");
            return false;
        }
        _functions.reserve(count);
        for (size_t i = 0u; i < count; i++) {
            if (!_decode_function(yyjson_arr_get(array, i), i, entry)) {
                return false;
            }
        }
        return true;
    }

public:
    ASTJsonDecoder(yyjson_val *root, const ASTJsonLimits &limits,
                   const ASTJsonBindingResolver &binding_resolver) noexcept
        : _root{root}, _limits{limits},
          _binding_resolver{binding_resolver} {}

    [[nodiscard]] ASTJsonDecodeResult decode() noexcept;
};

ASTJsonDecodeResult ASTJsonDecoder::decode() noexcept {
    if (!_check_keys(
            _root,
            {"schema", "version", "entry", "types", "constants",
             "external_functions", "functions"},
            "$")) {
        return {.error = std::move(_error)};
    }
    luisa::string_view schema;
    uint32_t version{};
    uint64_t entry_u64{};
    if (!_string(_member(_root, "schema", "$"), schema, "$.schema") ||
        schema != "luisa.compute.ast") {
        if (_error.empty()) { _fail("$.schema", "unsupported AST JSON schema."); }
        return {.error = std::move(_error)};
    }
    if (!_uint32(_member(_root, "version", "$"), version, "$.version") ||
        version != ast_json_schema_version) {
        if (_error.empty()) {
            _fail("$.version", luisa::format(
                                   "unsupported schema version {} (expected {}).",
                                   version, ast_json_schema_version));
        }
        return {.error = std::move(_error)};
    }
    if (!_uint64(_member(_root, "entry", "$"), entry_u64, "$.entry") ||
        entry_u64 > std::numeric_limits<size_t>::max()) {
        if (_error.empty()) { _fail("$.entry", "entry index exceeds host size_t."); }
        return {.error = std::move(_error)};
    }
    if (auto external = _member(_root, "external_functions", "$", false)) {
        if (!_array(external, 0u, "$.external_functions") ||
            yyjson_arr_size(external) != 0u) {
            if (_error.empty()) {
                _fail("$.external_functions",
                      "external functions are not supported by schema v1.");
            }
            return {.error = std::move(_error)};
        }
    }
    if (!_decode_types(_member(_root, "types", "$")) ||
        !_decode_constants(_member(_root, "constants", "$", false)) ||
        !_decode_functions(
            _member(_root, "functions", "$"),
            static_cast<size_t>(entry_u64))) {
        return {.error = std::move(_error)};
    }
    return {.function = _functions[static_cast<size_t>(entry_u64)]};
}

struct YYJsonBudget {
    size_t used{};
    size_t limit{};
};

[[nodiscard]] void *yyjson_budget_malloc(void *context, size_t size) noexcept {
    auto budget = static_cast<YYJsonBudget *>(context);
    if (size > budget->limit - std::min(budget->used, budget->limit)) {
        return nullptr;
    }
    auto pointer = std::malloc(size);
    if (pointer != nullptr) { budget->used += size; }
    return pointer;
}

[[nodiscard]] void *yyjson_budget_realloc(
    void *context, void *pointer, size_t old_size, size_t size) noexcept {
    auto budget = static_cast<YYJsonBudget *>(context);
    auto retained = budget->used >= old_size ? budget->used - old_size : 0u;
    if (size > budget->limit - std::min(retained, budget->limit)) {
        return nullptr;
    }
    auto replacement = std::realloc(pointer, size);
    if (replacement != nullptr) { budget->used = retained + size; }
    return replacement;
}

void yyjson_budget_free(void *, void *pointer) noexcept {
    std::free(pointer);
}

}// namespace

ASTJsonEncodeResult try_to_json(Function function,
                                const ASTJsonLimits &limits) noexcept {
    ASTJsonPreflight preflight{limits};
    if (!preflight.run(function)) {
        return {.error = preflight.take_error()};
    }
    auto json = to_json(function);
    if (json.size() > limits.max_document_bytes) {
        return {.error = "AST JSON document exceeds the configured byte limit."};
    }
    yyjson_read_err parse_error{};
    auto document = yyjson_read_opts(
        json.data(), json.size(), YYJSON_READ_NOFLAG, nullptr, &parse_error);
    if (document == nullptr) {
        return {.error = luisa::format(
                    "AST JSON encoder produced invalid JSON at byte {}: {}.",
                    parse_error.pos,
                    parse_error.msg == nullptr ? "unknown parse error" : parse_error.msg)};
    }
    yyjson_doc_free(document);
    return {.json = std::move(json)};
}

ASTJsonDecodeResult from_json(
    luisa::string_view json, const ASTJsonLimits &limits,
    const ASTJsonBindingResolver *binding_resolver) noexcept {
    if (json.empty()) {
        return {.error = "AST JSON document is empty."};
    }
    if (json.size() > limits.max_document_bytes) {
        return {.error = "AST JSON document exceeds the configured byte limit."};
    }
    YYJsonBudget budget{.limit = limits.max_parse_memory_bytes};
    yyjson_alc allocator{
        .malloc = yyjson_budget_malloc,
        .realloc = yyjson_budget_realloc,
        .free = yyjson_budget_free,
        .ctx = &budget};
    yyjson_read_err parse_error{};
    auto document = yyjson_read_opts(
        const_cast<char *>(json.data()), json.size(),
        YYJSON_READ_NOFLAG, &allocator, &parse_error);
    if (document == nullptr) {
        return {.error = luisa::format(
                    "Failed to parse AST JSON at byte {}: {}.",
                    parse_error.pos,
                    parse_error.msg == nullptr ?
                        "memory budget exhausted or unknown parse error" :
                        parse_error.msg)};
    }
    static const ASTJsonBindingResolver identity_resolver;
    ASTJsonDecoder decoder{
        yyjson_doc_get_root(document), limits,
        binding_resolver == nullptr ? identity_resolver : *binding_resolver};
    auto result = decoder.decode();
    yyjson_doc_free(document);
    return result;
}

}// namespace luisa::compute
