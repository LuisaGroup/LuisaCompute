#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <type_traits>

#include <luisa/core/magic_enum.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/hash.h>
#include <luisa/ast/callable_library.h>

namespace luisa::compute {

namespace detail {

[[nodiscard]] inline auto &callable_library_function_builder_deserialize_stack() noexcept {
    static thread_local luisa::vector<FunctionBuilder *> callable_library_stack;
    return callable_library_stack;
}

[[nodiscard]] inline auto callable_library_function_builder_deserialize_stack_top() noexcept {
    LUISA_ASSERT(!callable_library_function_builder_deserialize_stack().empty(), "Empty stack.");
    return callable_library_function_builder_deserialize_stack().back();
}

inline void callable_library_function_builder_deserialize_stack_push(FunctionBuilder *builder) noexcept {
    callable_library_function_builder_deserialize_stack().push_back(builder);
}

inline auto callable_library_function_builder_deserialize_stack_pop() noexcept {
    auto top = callable_library_function_builder_deserialize_stack_top();
    callable_library_function_builder_deserialize_stack().pop_back();
    return top;
}

}// namespace detail

template<typename T>
void CallableLibrary::ser_value(T const &t, luisa::vector<std::byte> &vec) noexcept {
    static_assert(std::is_trivially_destructible_v<T> && !std::is_pointer_v<T>);
    auto last_len = vec.size();
    luisa::enlarge_by(vec, sizeof(T));
    std::memcpy(vec.data() + last_len, &t, sizeof(T));
}
template<typename T>
T CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    static_assert(std::is_trivially_destructible_v<T> && !std::is_pointer_v<T>);
    T t;
    std::memcpy(&t, ptr, sizeof(T));
    ptr += sizeof(T);
    return t;
}
// string: (len: size_t) + (char array)
template<>
void CallableLibrary::ser_value(luisa::string_view const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t.size(), vec);
    auto last_len = vec.size();
    luisa::enlarge_by(vec, t.size());
    std::memcpy(vec.data() + last_len, t.data(), t.size());
}
template<>
void CallableLibrary::ser_value(luisa::string const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(luisa::string_view{t}, vec);
}
template<>
luisa::string CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    luisa::string t;
    auto size = deser_value<size_t>(ptr, pack);
    t.clear();
    t.resize(size);
    std::memcpy(t.data(), ptr, size);
    ptr += size;
    return t;
}
template<>
void CallableLibrary::ser_value(Type const *const &t, luisa::vector<std::byte> &vec) noexcept {
    using namespace std::string_view_literals;
    if (t) {
        ser_value(t->description(), vec);
    } else {
        ser_value("void"sv, vec);
    }
}
template<>
Type const *CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    using namespace std::string_view_literals;
    luisa::string desc = deser_value<luisa::string>(ptr, pack);
    if (desc == "void"sv) return nullptr;
    return Type::from(desc);
}
template<>
luisa::span<const std::byte> CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    auto size = deser_value<size_t>(ptr, pack);
    luisa::span<const std::byte> s{ptr, size};
    ptr += size;
    return s;
}
template<>
void CallableLibrary::ser_value(luisa::span<const std::byte> const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t.size(), vec);
    auto last_len = vec.size();
    luisa::enlarge_by(vec, t.size());
    std::memcpy(vec.data() + last_len, t.data(), t.size());
}
template<>
void CallableLibrary::deser_ptr(Variable *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_type = deser_value<Type const *>(ptr, pack);
    obj->_uid = deser_value<uint32_t>(ptr, pack);
    obj->_tag = deser_value<Variable::Tag>(ptr, pack);
}
template<>
void CallableLibrary::ser_value(Variable const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._type, vec);
    ser_value(t._uid, vec);
    ser_value(t._tag, vec);
}
template<>
Variable CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    Variable v{};
    v._type = deser_value<Type const *>(ptr, pack);
    v._uid = deser_value<uint32_t>(ptr, pack);
    v._tag = deser_value<Variable::Tag>(ptr, pack);
    return v;
}
template<>
void CallableLibrary::deser_ptr(ConstantData *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    Type const *type = deser_value<Type const *>(ptr, pack);
    *obj = ConstantData::create(type, ptr, type->size());
    ptr += type->size();
}
template<>
void CallableLibrary::ser_value(ConstantData const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._type, vec);
    auto last_len = vec.size();
    luisa::enlarge_by(vec, t._type->size());
    std::memcpy(vec.data() + last_len, t._raw, t._type->size());
}
template<>
ConstantData CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    Type const *type = deser_value<Type const *>(ptr, pack);
    auto t = ConstantData::create(type, ptr, type->size());
    ptr += type->size();
    return t;
}
template<>
void CallableLibrary::ser_value(CallOpSet const &t, luisa::vector<std::byte> &vec) noexcept {
    std::array<uint8_t, (call_op_count + 7) / 8> byte_arr{};
    for (size_t i = 0; i < call_op_count; ++i) {
        auto &v = byte_arr[i / 8];
        v |= ((t._bits[i] ? 1 : 0) << (i & 7));
    }
    auto last_len = vec.size();
    luisa::enlarge_by(vec, byte_arr.size());
    std::memcpy(vec.data() + last_len, byte_arr.data(), byte_arr.size());
}
template<>
CallOpSet CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    CallOpSet t;
    auto byte_arr = reinterpret_cast<uint8_t const *>(ptr);
    for (size_t i = 0; i < call_op_count; ++i) {
        auto &v = byte_arr[i / 8];
        t._bits[i] = (v & (1 << (i & 7)));
    }
    ptr += (call_op_count + 7) / 8;
    return t;
}
template<>
void CallableLibrary::ser_value(Expression const &t, luisa::vector<std::byte> &vec) noexcept;
template<>
Expression const *CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept;
template<>
void CallableLibrary::ser_value(UnaryExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._operand, vec);
    ser_value(t._op, vec);
}
template<>
void CallableLibrary::deser_ptr(UnaryExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_operand = deser_value<Expression const *>(ptr, pack);
    obj->_op = deser_value<UnaryOp>(ptr, pack);
}
template<>
void CallableLibrary::ser_value(BinaryExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._lhs, vec);
    ser_value(*t._rhs, vec);
    ser_value(t._op, vec);
}
template<>
void CallableLibrary::deser_ptr(BinaryExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_lhs = deser_value<Expression const *>(ptr, pack);
    obj->_rhs = deser_value<Expression const *>(ptr, pack);
    obj->_op = deser_value<BinaryOp>(ptr, pack);
}
template<>
void CallableLibrary::ser_value(AccessExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._range, vec);
    ser_value(*t._index, vec);
}
template<>
void CallableLibrary::deser_ptr(AccessExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_range = deser_value<Expression const *>(ptr, pack);
    obj->_index = deser_value<Expression const *>(ptr, pack);
}
template<>
void CallableLibrary::ser_value(MemberExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._self, vec);
    ser_value(t._swizzle_size, vec);
    ser_value(t._swizzle_code, vec);
}
template<>
void CallableLibrary::deser_ptr(MemberExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_self = deser_value<Expression const *>(ptr, pack);
    obj->_swizzle_size = deser_value<uint32_t>(ptr, pack);
    obj->_swizzle_code = deser_value<uint32_t>(ptr, pack);
}
template<>
void CallableLibrary::ser_value(LiteralExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._value.index(), vec);
    luisa::visit(
        [&]<typename T>(T const &t) {
            ser_value(sizeof(T), vec);
            ser_value(t, vec);
        },
        t._value);
}

template<>
void CallableLibrary::deser_ptr(LiteralExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    auto index = deser_value<size_t>(ptr, pack);
    auto literal_size = deser_value<size_t>(ptr, pack);
    using V = LiteralExpr::Value::variant_type;
    static constexpr auto n = luisa::variant_size_v<V>;
    auto emplace = [index, ptr, obj, literal_size]<size_t current>(auto &&self, std::integral_constant<size_t, current>) noexcept {
        if constexpr (current < n) {
            if (current == index) {
                using T = luisa::variant_alternative_t<current, V>;
                T value;
                std::memcpy(&value, ptr, literal_size);
                obj->_value.template emplace<T>(value);
            } else {
                self(self, std::integral_constant<size_t, current + 1>{});
            }
        }
    };
    emplace(emplace, std::integral_constant<size_t, 0>{});
    ptr += literal_size;
}
template<>
void CallableLibrary::ser_value(RefExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._variable, vec);
}
template<>
void CallableLibrary::deser_ptr(RefExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_variable = deser_value<Variable>(ptr, pack);
}
template<>
void CallableLibrary::ser_value(ConstantExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._data, vec);
}
template<>
void CallableLibrary::deser_ptr(ConstantExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_data = deser_value<ConstantData>(ptr, pack);
}
template<>
void CallableLibrary::ser_value(CallExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._arguments.size(), vec);
    for (auto &&i : t._arguments) {
        ser_value(*i, vec);
    }
    ser_value(t._op, vec);
    LUISA_ASSERT(!luisa::holds_alternative<CallExpr::ExternalCallee>(t._func),
                 "Callable cannot contain external");
    if (t.is_builtin()) {
        ser_value(t.curve_basis_set().to_u64(), vec);
    } else {
        ser_value(t._func.index(), vec);
    }
    luisa::visit(
        [&]<typename T>(T const &v) {
            if constexpr (std::is_same_v<T, CallExpr::CustomCallee>) {
                ser_value(v->_hash, vec);
            }
        },
        t._func);
}
template<>
void CallableLibrary::deser_ptr(CallExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    auto arg_size = deser_value<size_t>(ptr, pack);
    luisa::enlarge_by(obj->_arguments, arg_size);
    for (auto &&i : obj->_arguments) {
        i = deser_value<Expression const *>(ptr, pack);
    }
    obj->_op = deser_value<CallOp>(ptr, pack);
    if (is_builtin_operation(obj->_op)) {// built-in
        auto curve_basis_set = deser_value<uint64_t>(ptr, pack);
        obj->_curve_basis_set = CurveBasisSet::from_u64(curve_basis_set);
        obj->_func = luisa::monostate{};
    } else {
        obj->_curve_basis_set = {};
        auto index = deser_value<size_t>(ptr, pack);
        if (index == 0) {
            obj->_func = luisa::monostate{};
        } else {
            auto iter = pack.callable_map.find(deser_value<uint64_t>(ptr, pack));
            LUISA_ASSERT(iter != pack.callable_map.end(), "Custom op not found.");
            obj->_func = iter->second.get();
        }
    }
}

template<>
void CallableLibrary::ser_value(CastExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._source, vec);
    ser_value(t._op, vec);
}

template<>
void CallableLibrary::deser_ptr(CastExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_source = deser_value<Expression const *>(ptr, pack);
    obj->_op = deser_value<CastOp>(ptr, pack);
}

template<>
void CallableLibrary::ser_value(TypeIDExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._data_type, vec);
}

template<>
void CallableLibrary::deser_ptr(TypeIDExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_data_type = deser_value<Type const *>(ptr, pack);
}

template<>
void CallableLibrary::ser_value(Expression const &t, luisa::vector<std::byte> &vec) noexcept {
    using namespace std::string_view_literals;
    ser_value(t._type, vec);
    ser_value(t._hash, vec);
    ser_value(t._tag, vec);
    switch (t._tag) {
        case Expression::Tag::UNARY:
            ser_value(*static_cast<UnaryExpr const *>(&t), vec);
            break;
        case Expression::Tag::BINARY:
            ser_value(*static_cast<BinaryExpr const *>(&t), vec);
            break;
        case Expression::Tag::MEMBER:
            ser_value(*static_cast<MemberExpr const *>(&t), vec);
            break;
        case Expression::Tag::ACCESS:
            ser_value(*static_cast<AccessExpr const *>(&t), vec);
            break;
        case Expression::Tag::LITERAL:
            ser_value(*static_cast<LiteralExpr const *>(&t), vec);
            break;
        case Expression::Tag::REF:
            ser_value(*static_cast<RefExpr const *>(&t), vec);
            break;
        case Expression::Tag::CONSTANT:
            ser_value(*static_cast<ConstantExpr const *>(&t), vec);
            break;
        case Expression::Tag::CALL:
            ser_value(*static_cast<CallExpr const *>(&t), vec);
            break;
        case Expression::Tag::CAST:
            ser_value(*static_cast<CastExpr const *>(&t), vec);
            break;
        case Expression::Tag::TYPE_ID:
            ser_value(*static_cast<TypeIDExpr const *>(&t), vec);
            break;
        case Expression::Tag::STRING_ID: [[fallthrough]];
        case Expression::Tag::CPUCUSTOM: [[fallthrough]];
        case Expression::Tag::GPUCUSTOM: [[fallthrough]];
        case Expression::Tag::FUNC_REF: LUISA_NOT_IMPLEMENTED();
    }
}

template<>
Expression const *CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    auto type = deser_value<Type const *>(ptr, pack);
    auto hash = deser_value<uint64_t>(ptr, pack);
    auto tag = deser_value<Expression::Tag>(ptr, pack);
    auto create_expr = [&]<typename T>() {
          // Match the default deleter used by the builder's owning pointers.
          // The node ctors are protected (CallableLibrary is a friend), so
          // allocator-based placement new is used instead of make_unique.
#ifdef LUISA_USE_SYSTEM_STL
          auto expression = luisa::unique_ptr<T>{new T{}};
#else
          auto expression = luisa::unique_ptr<T>{new (luisa::allocate_with_allocator<T>()) T{}};
#endif
        auto expr = expression.get();
        deser_ptr<T *>(expr, ptr, pack);
        expr->_type = type;
        expr->_hash = hash;
        expr->_builder = detail::callable_library_function_builder_deserialize_stack_top();
        expr->_hash_computed = true;
        expr->_tag = tag;
        pack.builder->_all_expressions.emplace_back(std::move(expression));
        return expr;
    };
    switch (tag) {
        case Expression::Tag::UNARY:
            return create_expr.template operator()<UnaryExpr>();
        case Expression::Tag::BINARY:
            return create_expr.template operator()<BinaryExpr>();
        case Expression::Tag::MEMBER:
            return create_expr.template operator()<MemberExpr>();
        case Expression::Tag::ACCESS:
            return create_expr.template operator()<AccessExpr>();
        case Expression::Tag::LITERAL:
            return create_expr.template operator()<LiteralExpr>();
        case Expression::Tag::REF:
            return create_expr.template operator()<RefExpr>();
        case Expression::Tag::CONSTANT:
            return create_expr.template operator()<ConstantExpr>();
        case Expression::Tag::CALL:
            return create_expr.template operator()<CallExpr>();
        case Expression::Tag::CAST:
            return create_expr.template operator()<CastExpr>();
        case Expression::Tag::TYPE_ID:
            return create_expr.template operator()<TypeIDExpr>();
        default:
            return nullptr;
    }
}

template<>
void CallableLibrary::ser_value(Statement const &t, luisa::vector<std::byte> &vec) noexcept;

template<>
Statement *CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept;

template<>
void CallableLibrary::deser_ptr(Statement *obj, std::byte const *&ptr, DeserPackage &pack) noexcept;

template<>
void CallableLibrary::ser_value(ReturnStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    using namespace std::string_view_literals;
    if (t._expr == nullptr) {
        ser_value<uint8_t>(0, vec);
    } else {
        ser_value<uint8_t>(1, vec);
        ser_value(*t._expr, vec);
    }
}

template<>
void CallableLibrary::deser_ptr(ReturnStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    auto contain_ret = deser_value<uint8_t>(ptr, pack);
    if (contain_ret) {
        obj->_expr = deser_value<Expression const *>(ptr, pack);
    }
}

template<>
void CallableLibrary::ser_value(ScopeStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._statements.size(), vec);
    for (auto &&i : t._statements) {
        ser_value(*i, vec);
    }
}

template<>
void CallableLibrary::deser_ptr(ScopeStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    auto size = deser_value<size_t>(ptr, pack);
    luisa::enlarge_by(obj->_statements, size);
    for (auto &&i : obj->_statements) {
        i = deser_value<Statement *>(ptr, pack);
    }
}

template<>
void CallableLibrary::ser_value(IfStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._condition, vec);
    ser_value<Statement>(t._true_branch, vec);
    ser_value<Statement>(t._false_branch, vec);
}

template<>
void CallableLibrary::deser_ptr(IfStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    obj->_condition = deser_value<Expression const *>(ptr, pack);
    deser_ptr<Statement *>(&obj->_true_branch, ptr, pack);
    deser_ptr<Statement *>(&obj->_false_branch, ptr, pack);
}
template<>
void CallableLibrary::ser_value(LoopStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value<Statement>(t._body, vec);
}
template<>
void CallableLibrary::deser_ptr(LoopStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    deser_ptr<Statement *>(&obj->_body, ptr, pack);
}
template<>
void CallableLibrary::ser_value(ExprStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._expr, vec);
}
template<>
void CallableLibrary::deser_ptr(ExprStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_expr = deser_value<Expression const *>(ptr, pack);
}
template<>
void CallableLibrary::ser_value(SwitchStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._expr, vec);
    ser_value<Statement>(t._body, vec);
}
template<>
void CallableLibrary::deser_ptr(SwitchStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    obj->_expr = deser_value<Expression const *>(ptr, pack);
    deser_ptr<Statement *>(&obj->_body, ptr, pack);
}
template<>
void CallableLibrary::ser_value(SwitchCaseStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    if (t.tag() == Statement::Tag::SWITCH_CASE_GROUP) {
        ser_value(t._expressions.size(), vec);
        for (auto expression : t._expressions) { ser_value(*expression, vec); }
    } else {
        ser_value(*t._expr, vec);
    }
    ser_value<Statement>(t._body, vec);
}
template<>
void CallableLibrary::deser_ptr(SwitchCaseStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    if (obj->tag() == Statement::Tag::SWITCH_CASE_GROUP) {
        auto count = deser_value<size_t>(ptr, pack);
        LUISA_ASSERT(count > 1u, "A serialized switch case group needs at least two labels.");
        obj->_expressions.reserve(count);
        for (auto i = 0u; i < count; i++) {
            obj->_expressions.emplace_back(deser_value<Expression const *>(ptr, pack));
        }
        obj->_expr = obj->_expressions.front();
    } else {
        obj->_expr = deser_value<Expression const *>(ptr, pack);
    }
    deser_ptr<Statement *>(&obj->_body, ptr, pack);
}
template<>
void CallableLibrary::ser_value(SwitchDefaultStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value<Statement>(t._body, vec);
}
template<>
void CallableLibrary::deser_ptr(SwitchDefaultStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    deser_ptr<Statement *>(&obj->_body, ptr, pack);
}
template<>
void CallableLibrary::ser_value(ForStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._var, vec);
    ser_value(*t._cond, vec);
    ser_value(*t._step, vec);
    ser_value<Statement>(t._body, vec);
}
template<>
void CallableLibrary::deser_ptr(ForStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    obj->_var = deser_value<Expression const *>(ptr, pack);
    obj->_cond = deser_value<Expression const *>(ptr, pack);
    obj->_step = deser_value<Expression const *>(ptr, pack);
    deser_ptr<Statement *>(&obj->_body, ptr, pack);
}
template<>
void CallableLibrary::ser_value(CommentStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._comment, vec);
}
template<>
void CallableLibrary::deser_ptr(CommentStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_comment = deser_value<luisa::string>(ptr, pack);
}
template<>
void CallableLibrary::ser_value(SuspendStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._token, vec);
    ser_value(t._name, vec);
    ser_value(t._frame_exports.size(), vec);
    for (auto &&frame_export : t._frame_exports) {
        ser_value(frame_export.name, vec);
        ser_value(*frame_export.value, vec);
    }
    ser_value(t._extensions.size(), vec);
    for (auto &&extension : t._extensions) {
        ser_value(luisa::string{extension->schema()}, vec);
        ser_value(extension->version(), vec);
        ser_value(extension->is_annotation(), vec);
        ser_value(extension->fallback(), vec);
        ser_value(extension->bindings().size(), vec);
        for (auto &&binding : extension->bindings()) {
            ser_value(binding.name, vec);
            ser_value(binding.access, vec);
            ser_value(binding.lifetime, vec);
            ser_value(binding.index, vec);
        }
        ser_value(extension->attributes().size(), vec);
        for (auto &&attribute : extension->attributes()) {
            ser_value(attribute.name, vec);
            ser_value(static_cast<uint8_t>(attribute.value.index()), vec);
            luisa::visit(
                [&](auto &&value) noexcept { ser_value(value, vec); },
                attribute.value);
        }
    }
    ser_value(t._extension_binding_values.size(), vec);
    for (auto *value : t._extension_binding_values) {
        ser_value(*value, vec);
    }
}
template<>
void CallableLibrary::deser_ptr(SuspendStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    obj->_token = deser_value<uint32_t>(ptr, pack);
    obj->_name = deser_value<luisa::string>(ptr, pack);
    auto count = deser_value<size_t>(ptr, pack);
    obj->_frame_exports.reserve(count);
    for (size_t i = 0u; i < count; ++i) {
        auto name = deser_value<luisa::string>(ptr, pack);
        auto *value = deser_value<Expression const *>(ptr, pack);
        obj->_frame_exports.emplace_back(CoroFrameExport{
            .value = value, .name = std::move(name)});
        if (value != nullptr) { value->mark(Usage::READ); }
    }
    auto extension_count = deser_value<size_t>(ptr, pack);
    obj->_extensions.reserve(extension_count);
    for (size_t i = 0u; i < extension_count; ++i) {
        auto schema = deser_value<luisa::string>(ptr, pack);
        auto version = deser_value<uint32_t>(ptr, pack);
        auto is_annotation = deser_value<bool>(ptr, pack);
        auto fallback = deser_value<CoroSuspendFallback>(ptr, pack);
        auto binding_count = deser_value<size_t>(ptr, pack);
        luisa::vector<CoroSuspendBinding> bindings;
        bindings.reserve(binding_count);
        for (size_t j = 0u; j < binding_count; ++j) {
            auto name = deser_value<luisa::string>(ptr, pack);
            auto access =
                deser_value<CoroSuspendBindingAccess>(ptr, pack);
            auto lifetime =
                deser_value<CoroSuspendBindingLifetime>(ptr, pack);
            auto index = deser_value<uint32_t>(ptr, pack);
            bindings.emplace_back(CoroSuspendBinding{
                .name = std::move(name),
                .access = access,
                .lifetime = lifetime,
                .index = index});
        }
        auto attribute_count = deser_value<size_t>(ptr, pack);
        luisa::vector<CoroSuspendAttribute> attributes;
        attributes.reserve(attribute_count);
        for (size_t j = 0u; j < attribute_count; ++j) {
            auto name = deser_value<luisa::string>(ptr, pack);
            auto tag = deser_value<uint8_t>(ptr, pack);
            CoroSuspendAttributeValue value;
            switch (tag) {
                case 0u: value = deser_value<bool>(ptr, pack); break;
                case 1u: value = deser_value<int64_t>(ptr, pack); break;
                case 2u: value = deser_value<uint64_t>(ptr, pack); break;
                case 3u: value = deser_value<double>(ptr, pack); break;
                case 4u:
                    value = deser_value<luisa::string>(ptr, pack);
                    break;
                default:
                    LUISA_ERROR_WITH_LOCATION(
                        "Invalid coroutine suspend attribute tag {}.",
                        tag);
            }
            attributes.emplace_back(CoroSuspendAttribute{
                .name = std::move(name), .value = std::move(value)});
        }
        auto extension = is_annotation ?
                             make_coro_suspend_annotation_data(
                                 std::move(schema), version, fallback,
                                 std::move(bindings),
                                 std::move(attributes)) :
                             make_coro_suspend_extension_data(
                                 std::move(schema), version, fallback,
                                 std::move(bindings),
                                 std::move(attributes));
        obj->_extensions.emplace_back(std::move(extension));
    }
    auto binding_value_count = deser_value<size_t>(ptr, pack);
    obj->_extension_binding_values.reserve(binding_value_count);
    for (size_t i = 0u; i < binding_value_count; ++i) {
        obj->_extension_binding_values.emplace_back(
            deser_value<Expression const *>(ptr, pack));
    }
    // Function-level variable usages are serialized separately and were
    // restored before the statement tree. Validate the extension payload here
    // without calling Expression::mark: the ordinary FunctionBuilder stack is
    // intentionally not active during CallableLibrary deserialization, and a
    // temporary FunctionStackGuard would finalize/hash a partially populated
    // builder when it leaves this nested statement routine.
    obj->_validate_extension_bindings();
}
template<>
void CallableLibrary::ser_value(AutoDiffStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value<Statement>(t._body, vec);
}
template<>
void CallableLibrary::deser_ptr(AutoDiffStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    deser_ptr<Statement *>(&obj->_body, ptr, pack);
}
template<>
void CallableLibrary::ser_value(RayQueryStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*static_cast<Expression const *>(t._query), vec);
    ser_value<Statement>(t._on_triangle_candidate, vec);
    ser_value<Statement>(t._on_procedural_candidate, vec);
}
template<>
void CallableLibrary::deser_ptr(RayQueryStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    obj->_query = static_cast<RefExpr const *>(deser_value<Expression const *>(ptr, pack));
    deser_ptr<Statement *>(&obj->_on_triangle_candidate, ptr, pack);
    deser_ptr<Statement *>(&obj->_on_procedural_candidate, ptr, pack);
}
template<>
void CallableLibrary::ser_value(AssignStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._lhs, vec);
    ser_value(*t._rhs, vec);
}
template<>
void CallableLibrary::deser_ptr(AssignStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_lhs = deser_value<Expression const *>(ptr, pack);
    obj->_rhs = deser_value<Expression const *>(ptr, pack);
}

// print stmt
template<>
void CallableLibrary::ser_value(PrintStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._format, vec);
    ser_value(t._args.size(), vec);
    for (auto &&i : t._args) {
        ser_value(*i, vec);
    }
}
template<>
void CallableLibrary::deser_ptr(PrintStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    (void)pack;
    obj->_format = deser_value<luisa::string>(ptr, pack);
    auto size = deser_value<size_t>(ptr, pack);
    obj->_args.reserve(size);
    for (auto i = 0u; i < size; i++) {
        obj->_args.emplace_back(
            deser_value<const Expression *>(ptr, pack));
    }
}

template<>
void CallableLibrary::ser_value(Statement const &t, luisa::vector<std::byte> &vec) noexcept {
    using namespace std::string_view_literals;
    ser_value(t._hash, vec);
    ser_value(t._tag, vec);
    switch (t._tag) {
        case Statement::Tag::RETURN:
            ser_value(*static_cast<ReturnStmt const *>(&t), vec);
            break;
        case Statement::Tag::SCOPE:
            ser_value(*static_cast<ScopeStmt const *>(&t), vec);
            break;
        case Statement::Tag::IF:
            ser_value(*static_cast<IfStmt const *>(&t), vec);
            break;
        case Statement::Tag::LOOP:
            ser_value(*static_cast<LoopStmt const *>(&t), vec);
            break;
        case Statement::Tag::EXPR:
            ser_value(*static_cast<ExprStmt const *>(&t), vec);
            break;
        case Statement::Tag::SWITCH:
            ser_value(*static_cast<SwitchStmt const *>(&t), vec);
            break;
        case Statement::Tag::SWITCH_CASE:
        case Statement::Tag::SWITCH_CASE_GROUP:
            ser_value(*static_cast<SwitchCaseStmt const *>(&t), vec);
            break;
        case Statement::Tag::SWITCH_DEFAULT:
            ser_value(*static_cast<SwitchDefaultStmt const *>(&t), vec);
            break;
        case Statement::Tag::ASSIGN:
            ser_value(*static_cast<AssignStmt const *>(&t), vec);
            break;
        case Statement::Tag::FOR:
            ser_value(*static_cast<ForStmt const *>(&t), vec);
            break;
        case Statement::Tag::COMMENT:
            ser_value(*static_cast<CommentStmt const *>(&t), vec);
            break;
        case Statement::Tag::RAY_QUERY:
            ser_value(*static_cast<RayQueryStmt const *>(&t), vec);
            break;
        case Statement::Tag::AUTO_DIFF:
            ser_value(*static_cast<AutoDiffStmt const *>(&t), vec);
            break;
        case Statement::Tag::SUSPEND:
            ser_value(*static_cast<SuspendStmt const *>(&t), vec);
            break;
        case Statement::Tag::PRINT:
            ser_value(*static_cast<PrintStmt const *>(&t), vec);
            break;
        case Statement::Tag::BREAK:
        case Statement::Tag::CONTINUE:
            break;
        case Statement::Tag::DEBUG_BREAK:
            LUISA_NOT_IMPLEMENTED("Debug break statement serialization is not implemented.");
    }
}
template<>
Statement *CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    auto hash = deser_value<uint64_t>(ptr, pack);
    auto tag = deser_value<Statement::Tag>(ptr, pack);
    auto create_stmt = [&]<typename T, bool construct = true>() {
                  // See the expression site above for the allocator pairing.
#ifdef LUISA_USE_SYSTEM_STL
          auto statement = luisa::unique_ptr<T>{new T{}};
#else
          auto statement = luisa::unique_ptr<T>{new (luisa::allocate_with_allocator<T>()) T{}};
#endif
        auto stmt = statement.get();
        stmt->_hash = hash;
        stmt->_hash_computed = true;
        stmt->_tag = tag;
        pack.builder->_all_statements.emplace_back(std::move(statement));
        if constexpr (construct) {
            deser_ptr<T *>(stmt, ptr, pack);
        }
        return stmt;
    };
    switch (tag) {
        case Statement::Tag::BREAK:
            return create_stmt.template operator()<BreakStmt, false>();
        case Statement::Tag::CONTINUE:
            return create_stmt.template operator()<ContinueStmt, false>();
        case Statement::Tag::RETURN:
            return create_stmt.template operator()<ReturnStmt>();
        case Statement::Tag::SCOPE:
            return create_stmt.template operator()<ScopeStmt>();
        case Statement::Tag::IF:
            return create_stmt.template operator()<IfStmt>();
        case Statement::Tag::LOOP:
            return create_stmt.template operator()<LoopStmt>();
        case Statement::Tag::EXPR:
            return create_stmt.template operator()<ExprStmt>();
        case Statement::Tag::SWITCH:
            return create_stmt.template operator()<SwitchStmt>();
        case Statement::Tag::SWITCH_CASE:
        case Statement::Tag::SWITCH_CASE_GROUP:
            return create_stmt.template operator()<SwitchCaseStmt>();
        case Statement::Tag::SWITCH_DEFAULT:
            return create_stmt.template operator()<SwitchDefaultStmt>();
        case Statement::Tag::ASSIGN:
            return create_stmt.template operator()<AssignStmt>();
        case Statement::Tag::FOR:
            return create_stmt.template operator()<ForStmt>();
        case Statement::Tag::COMMENT:
            return create_stmt.template operator()<CommentStmt>();
        case Statement::Tag::RAY_QUERY:
            return create_stmt.template operator()<RayQueryStmt>();
        case Statement::Tag::AUTO_DIFF:
            return create_stmt.template operator()<AutoDiffStmt>();
        case Statement::Tag::SUSPEND:
            return create_stmt.template operator()<SuspendStmt>();
        case Statement::Tag::PRINT:
            return create_stmt.template operator()<PrintStmt>();
        case Statement::Tag::DEBUG_BREAK:
            LUISA_NOT_IMPLEMENTED("Debug break statement deserialization is not implemented.");
    }
    LUISA_ERROR_WITH_LOCATION("Unreachable code.");
}
template<>
void CallableLibrary::deser_ptr(Statement *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    obj->_hash = deser_value<uint64_t>(ptr, pack);
    obj->_tag = deser_value<Statement::Tag>(ptr, pack);
    obj->_hash_computed = true;

    auto create_stmt = [&]<typename T, bool construct = true>() {
        auto stmt = static_cast<T *>(obj);
        if constexpr (construct) {
            deser_ptr<T *>(stmt, ptr, pack);
        }
    };
    switch (obj->_tag) {
        case Statement::Tag::BREAK:
            create_stmt.template operator()<BreakStmt, false>();
            break;
        case Statement::Tag::CONTINUE:
            create_stmt.template operator()<ContinueStmt, false>();
            break;
        case Statement::Tag::RETURN:
            create_stmt.template operator()<ReturnStmt>();
            break;
        case Statement::Tag::SCOPE:
            create_stmt.template operator()<ScopeStmt>();
            break;
        case Statement::Tag::IF:
            create_stmt.template operator()<IfStmt>();
            break;
        case Statement::Tag::LOOP:
            create_stmt.template operator()<LoopStmt>();
            break;
        case Statement::Tag::EXPR:
            create_stmt.template operator()<ExprStmt>();
            break;
        case Statement::Tag::SWITCH:
            create_stmt.template operator()<SwitchStmt>();
            break;
        case Statement::Tag::SWITCH_CASE:
        case Statement::Tag::SWITCH_CASE_GROUP:
            create_stmt.template operator()<SwitchCaseStmt>();
            break;
        case Statement::Tag::SWITCH_DEFAULT:
            create_stmt.template operator()<SwitchDefaultStmt>();
            break;
        case Statement::Tag::ASSIGN:
            create_stmt.template operator()<AssignStmt>();
            break;
        case Statement::Tag::FOR:
            create_stmt.template operator()<ForStmt>();
            break;
        case Statement::Tag::COMMENT:
            create_stmt.template operator()<CommentStmt>();
            break;
        case Statement::Tag::RAY_QUERY:
            create_stmt.template operator()<RayQueryStmt>();
            break;
        case Statement::Tag::AUTO_DIFF:
            create_stmt.template operator()<AutoDiffStmt>();
            break;
        case Statement::Tag::SUSPEND:
            create_stmt.template operator()<SuspendStmt>();
            break;
        case Statement::Tag::PRINT:
            create_stmt.template operator()<PrintStmt>();
            break;
        case Statement::Tag::DEBUG_BREAK:
            LUISA_NOT_IMPLEMENTED("Debug break statement deserialization is not implemented.");
    }
}

void CallableLibrary::deserialize_func_builder(detail::FunctionBuilder &builder, std::byte const *&ptr, DeserPackage &pack) noexcept {
    detail::callable_library_function_builder_deserialize_stack_push(&builder);
    using namespace detail;
    using namespace std::string_view_literals;
    builder._required_curve_bases = CurveBasisSet::from_u64(deser_value<uint64_t>(ptr, pack));
    builder._return_type = deser_value<Type const *>(ptr, pack);
    luisa::enlarge_by(builder._builtin_variables, deser_value<size_t>(ptr, pack));
    for (auto &&i : builder._builtin_variables) {
        i = deser_value<Variable>(ptr, pack);
    }
    luisa::enlarge_by(builder._captured_constants, deser_value<size_t>(ptr, pack));
    for (auto &&i : builder._captured_constants) {
        i = deser_value<ConstantData>(ptr, pack);
    }
    luisa::enlarge_by(builder._arguments, deser_value<size_t>(ptr, pack));
    for (auto &&i : builder._arguments) {
        i = deser_value<Variable>(ptr, pack);
    }
    // Note: variant is not trivially copyable. DO NOT USE `push_back_uninitialized`!!!
    builder._bound_arguments.reserve(builder._arguments.size());
    for (auto i = 0u; i < builder._arguments.size(); i++) {
        builder._bound_arguments.emplace_back(luisa::monostate{});
    }
    auto used_custom_callables_size = deser_value<size_t>(ptr, pack);
    builder._used_custom_callables.clear();
    builder._used_custom_callables.reserve(used_custom_callables_size);
    for (size_t i = 0; i < used_custom_callables_size; ++i) {
        auto iter = pack.callable_map.find(deser_value<uint64_t>(ptr, pack));
        LUISA_ASSERT(iter != pack.callable_map.end(), "Illegal bin-data.");
        builder._used_custom_callables.emplace(iter->second);
    }
    luisa::enlarge_by(builder._local_variables, deser_value<size_t>(ptr, pack));
    for (auto &&i : builder._local_variables) {
        i = deser_value<Variable>(ptr, pack);
    }
    luisa::enlarge_by(builder._shared_variables, deser_value<size_t>(ptr, pack));
    for (auto &&i : builder._shared_variables) {
        i = deser_value<Variable>(ptr, pack);
    }
    size_t variable_usage_size = deser_value<size_t>(ptr, pack) / sizeof(Usage);
    luisa::enlarge_by(builder._variable_usages, variable_usage_size);
    std::memcpy(builder._variable_usages.data(), ptr, luisa::size_bytes(builder._variable_usages));
    ptr += luisa::size_bytes(builder._variable_usages);
    builder._direct_builtin_callables = deser_value<CallOpSet>(ptr, pack);
    builder._propagated_builtin_callables = deser_value<CallOpSet>(ptr, pack);
    builder._tag = deser_value<Function::Tag>(ptr, pack);
    builder._requires_atomic_float = deser_value<bool>(ptr, pack);
    builder._requires_printing = deser_value<bool>(ptr, pack);
    builder._requires_noinline = deser_value<bool>(ptr, pack);
    builder._block_size = deser_value<uint3>(ptr, pack);
    deser_ptr<Statement *>(&builder._body, ptr, pack);
    auto popped = callable_library_function_builder_deserialize_stack_pop();
    LUISA_ASSERT(popped == &builder, "Illegal function builder stack.");
    builder.sort_bindings();
}
void CallableLibrary::serialize_func_builder(detail::FunctionBuilder const &builder, luisa::vector<std::byte> &vec) noexcept {
    using namespace detail;
    using namespace std::string_view_literals;
    for (auto &&i : builder._bound_arguments) {
        LUISA_ASSERT(luisa::holds_alternative<luisa::monostate>(i),
                     "Serialized function cannot contain bound-argument.");
    }
    LUISA_ASSERT(builder._used_external_functions.empty(), "Serialized function cannot contain external-function.");
    ser_value(builder.required_curve_bases().to_u64(), vec);
    // return type
    if (builder._return_type)
        ser_value(builder._return_type.value(), vec);
    else {
        ser_value("void"sv, vec);
    }
    // builtin variables
    ser_value(builder._builtin_variables.size(), vec);
    for (auto &&i : builder._builtin_variables) {
        ser_value(i, vec);
    }
    // constant
    ser_value(builder._captured_constants.size(), vec);
    for (auto &&i : builder._captured_constants) {
        ser_value(i, vec);
    }
    // arguments
    ser_value(builder._arguments.size(), vec);
    for (auto &&i : builder._arguments) {
        ser_value(i, vec);
    }
    // external function
    ser_value(builder._used_custom_callables.size(), vec);
    for (auto &&i : builder._used_custom_callables) {
        ser_value(i->hash(), vec);
    }
    ser_value(builder._local_variables.size(), vec);
    for (auto &&i : builder._local_variables) {
        ser_value(i, vec);
    }
    // shared vars
    ser_value(builder._shared_variables.size(), vec);
    for (auto &&i : builder._shared_variables) {
        ser_value(i, vec);
    }
    // variable usages
    ser_value(luisa::span<const std::byte>{reinterpret_cast<const std::byte *>(builder._variable_usages.data()), luisa::size_bytes(builder._variable_usages)}, vec);
    // direct builtin callables
    ser_value(builder._direct_builtin_callables, vec);
    // propagated builtin callables
    ser_value(builder._propagated_builtin_callables, vec);
    // tag
    ser_value(builder._tag, vec);
    // requires_atomic_float
    ser_value(builder._requires_atomic_float, vec);
    // requires printing
    ser_value(builder._requires_printing, vec);
    // requires a retained call boundary
    ser_value(builder._requires_noinline, vec);
    ser_value(builder._block_size, vec);
    // body
    ser_value(static_cast<Statement const &>(builder._body), vec);
}
CallableLibrary::CallableLibrary() noexcept = default;
void CallableLibrary::load(luisa::span<const std::byte> binary) noexcept {
    _callables.clear();
    if (binary.empty()) { return; }
    DeserPackage pack;
    auto ptr = binary.data();
    auto callable_size = deser_value<size_t>(ptr, pack);
    auto inline_callable_size = deser_value<size_t>(ptr, pack);
    luisa::unordered_map<size_t, luisa::shared_ptr<const detail::FunctionBuilder>> inline_callables;
    _callables.clear();
    _callables.reserve(callable_size);
    inline_callables.reserve(inline_callable_size);
    pack.callable_map.reserve(callable_size + inline_callable_size);
    for (size_t i = 0; i < callable_size + inline_callable_size; ++i) {
        auto hash = deser_value<uint64_t>(ptr, pack);
        auto func = luisa::make_unique<detail::FunctionBuilder>();
        func->_hash = hash;
        func->_hash_computed = true;
        pack.callable_map.try_emplace(hash, std::move(func));
    }
    for (size_t i = 0; i < callable_size; ++i) {
        auto hash = deser_value<uint64_t>(ptr, pack);
        auto iter = pack.callable_map.find(hash);
        LUISA_ASSERT(iter != pack.callable_map.end(), "Illegal bin-data.");
        auto name = deser_value<luisa::string>(ptr, pack);
        pack.builder = iter->second.get();
        deserialize_func_builder(*iter->second, ptr, pack);
        _callables.try_emplace(std::move(name), iter->second);
    }
    for (size_t i = 0; i < inline_callable_size; ++i) {
        auto hash = deser_value<uint64_t>(ptr, pack);
        auto iter = pack.callable_map.find(hash);
        LUISA_ASSERT(iter != pack.callable_map.end(), "Illegal bin-data.");
        pack.builder = iter->second.get();
        deserialize_func_builder(*iter->second, ptr, pack);
    }
}
luisa::vector<std::byte> CallableLibrary::serialize() const noexcept {
    luisa::unordered_map<size_t, luisa::shared_ptr<const detail::FunctionBuilder>> inline_callables;
    luisa::vector<std::byte> vec;
    for (auto &&i : _callables) {
        for (auto &&j : i.second->_used_custom_callables) {
            inline_callables.try_emplace(j->hash(), j);
        }
    }
    for (auto &&i : _callables) {
        inline_callables.erase(i.second->hash());
    }
    ser_value(_callables.size(), vec);
    ser_value(inline_callables.size(), vec);
    for (auto &&i : _callables) {
        ser_value(i.second->hash(), vec);
    }
    for (auto &&i : inline_callables) {
        ser_value(i.second->hash(), vec);
    }
    // Callables
    for (auto &&i : _callables) {
        // hash
        ser_value(i.second->hash(), vec);
        ser_value(i.first, vec);
        serialize_func_builder(*i.second, vec);
    }
    // Inline callables
    for (auto &&i : inline_callables) {
        ser_value(i.second->hash(), vec);
        serialize_func_builder(*i.second, vec);
    }
    return vec;
}
void CallableLibrary::add_callable(luisa::string_view name, const luisa::shared_ptr<const detail::FunctionBuilder> &callable) noexcept {
#ifdef LUISA_USE_SYSTEM_STL
    _callables.try_emplace(luisa::string{name}, callable);
#else
    _callables.try_emplace(name, callable);
#endif
}
CallableLibrary::~CallableLibrary() noexcept = default;
CallableLibrary::CallableLibrary(CallableLibrary &&) noexcept = default;

Function CallableLibrary::get_function(luisa::string_view name) const noexcept {
    auto iter = _callables.find(name);
    if (iter == _callables.end()) [[unlikely]] {
        LUISA_ERROR("Callable {} not found", name);
    }
    auto &func = iter->second;
    return Function{func.get()};
}
luisa::shared_ptr<const detail::FunctionBuilder> CallableLibrary::get_function_builder(luisa::string_view name) const noexcept {
    auto iter = _callables.find(name);
    if (iter == _callables.end()) [[unlikely]] {
        LUISA_ERROR("Callable {} not found", name);
    }
    auto &func = iter->second;
    return func;
}

namespace callable_graph_codec_detail {

constexpr std::array<std::byte, 8u> kGraphMagic{
    std::byte{'L'}, std::byte{'C'}, std::byte{'A'}, std::byte{'S'},
    std::byte{'T'}, std::byte{'G'}, std::byte{0u}, std::byte{1u}};
constexpr uint32_t kGraphVersion = 1u;
constexpr uint32_t kMaxTableCount = 1u << 24u;
constexpr size_t kMaxAstDepth = 4096u;

class ByteWriter {
public:
    luisa::vector<std::byte> bytes;

    template<typename T>
    void write(T const &v) noexcept {
        static_assert(std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>);
        auto old_size = bytes.size();
        luisa::enlarge_by(bytes, sizeof(T));
        std::memcpy(bytes.data() + old_size, &v, sizeof(T));
    }

    void write_bytes(const void *ptr, size_t size) noexcept {
        if (size == 0u) { return; }
        auto old_size = bytes.size();
        luisa::enlarge_by(bytes, size);
        std::memcpy(bytes.data() + old_size, ptr, size);
    }

    void write_string(luisa::string_view s) noexcept {
        write<uint32_t>(static_cast<uint32_t>(s.size()));
        write_bytes(s.data(), s.size());
    }
};

class ByteReader {
    const std::byte *_ptr{nullptr};
    const std::byte *_end{nullptr};

public:
    explicit ByteReader(luisa::span<const std::byte> data) noexcept
        : _ptr{data.data()}, _end{data.data() + data.size()} {}

    [[nodiscard]] size_t remaining() const noexcept {
        return static_cast<size_t>(_end - _ptr);
    }

    template<typename T>
    [[nodiscard]] bool read(T &out) noexcept {
        static_assert(std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>);
        if (remaining() < sizeof(T)) { return false; }
        std::memcpy(&out, _ptr, sizeof(T));
        _ptr += sizeof(T);
        return true;
    }

    [[nodiscard]] bool read_bytes(void *dst, size_t size) noexcept {
        if (remaining() < size) { return false; }
        if (size != 0u) {
            std::memcpy(dst, _ptr, size);
            _ptr += size;
        }
        return true;
    }

    [[nodiscard]] bool read_span(size_t size, const std::byte *&out_ptr) noexcept {
        if (remaining() < size) { return false; }
        out_ptr = _ptr;
        _ptr += size;
        return true;
    }

    [[nodiscard]] bool read_string(luisa::string &out, uint32_t max_len = (1u << 24u)) noexcept {
        uint32_t len = 0u;
        if (!read(len) || len > max_len || remaining() < len) { return false; }
        out.resize(len);
        if (len != 0u) {
            std::memcpy(out.data(), _ptr, len);
            _ptr += len;
        }
        return true;
    }
};

[[nodiscard]] bool is_valid_constant_type(const Type *type, size_t depth = 0u) noexcept {
    if (type == nullptr || depth > 64u) { return false; }
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
        case Type::Tag::BFLOAT16:
        case Type::Tag::FLOAT32:
        case Type::Tag::FLOAT64:
            return type->size() > 0u;
        case Type::Tag::VECTOR:
        case Type::Tag::MATRIX:
        case Type::Tag::ARRAY:
            return is_valid_constant_type(type->element(), depth + 1u);
        case Type::Tag::STRUCTURE: {
            for (auto *m : type->members()) {
                if (!is_valid_constant_type(m, depth + 1u)) { return false; }
            }
            return true;
        }
        default:
            return false;
    }
}

[[nodiscard]] bool is_valid_variable_tag(Variable::Tag tag) noexcept {
    switch (tag) {
        case Variable::Tag::LOCAL:
        case Variable::Tag::SHARED:
        case Variable::Tag::REFERENCE:
        case Variable::Tag::BUFFER:
        case Variable::Tag::TEXTURE:
        case Variable::Tag::BINDLESS_ARRAY:
        case Variable::Tag::ACCEL:
        case Variable::Tag::THREAD_ID:
        case Variable::Tag::BLOCK_ID:
        case Variable::Tag::DISPATCH_ID:
        case Variable::Tag::DISPATCH_SIZE:
        case Variable::Tag::KERNEL_ID:
        case Variable::Tag::WARP_LANE_COUNT:
        case Variable::Tag::WARP_LANE_ID:
        case Variable::Tag::RASTER_OBJECT_ID:
        case Variable::Tag::RASTER_BARYCENTRICS:
        case Variable::Tag::RASTER_FRONT_FACING:
        case Variable::Tag::RASTER_BASE_INSTANCE:
            return true;
    }
    return false;
}

[[nodiscard]] bool is_valid_allowed_warp_size(uint8_t value) noexcept {
    switch (value) {
        case 1u:
        case 2u:
        case 4u:
        case 8u:
        case 16u:
        case 32u:
        case 64u:
        case 128u:
        case 255u:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] bool is_writable_lvalue(const Expression *expr) noexcept {
    if (expr == nullptr) { return false; }
    switch (expr->tag()) {
        case Expression::Tag::REF: return true;
        case Expression::Tag::ACCESS:
            return is_writable_lvalue(static_cast<const AccessExpr *>(expr)->range());
        case Expression::Tag::MEMBER: {
            auto *member = static_cast<const MemberExpr *>(expr);
            return (!member->is_swizzle() || member->swizzle_size() == 1u) &&
                   is_writable_lvalue(member->self());
        }
        default: return false;
    }
}

struct InternedConstantEntry {
    uint32_t type_id{0u};
    ConstantData data;
};

}// namespace callable_graph_codec_detail

struct CallableLibrary::GraphWriteCtx {
    callable_graph_codec_detail::ByteWriter writer;
    luisa::vector<const detail::FunctionBuilder *> func_order;
    luisa::unordered_map<const detail::FunctionBuilder *, uint32_t> func_to_id;
    luisa::unordered_set<const detail::FunctionBuilder *> visiting;

    luisa::vector<const Type *> types;
    luisa::unordered_map<const Type *, uint32_t> type_to_id;

    luisa::vector<callable_graph_codec_detail::InternedConstantEntry> constants;
    luisa::unordered_map<uint64_t, luisa::vector<uint32_t>> const_buckets;

    // Per-function expression table (reset for each function)
    const detail::FunctionBuilder *current_owner{nullptr};
    luisa::unordered_map<const Expression *, uint32_t> expr_to_id;
    callable_graph_codec_detail::ByteWriter expr_writer;
    uint32_t expr_count{0u};
    size_t depth{0u};

    [[nodiscard]] uint32_t intern_type(const Type *type) noexcept {
        if (type == nullptr) { return 0u; }
        auto [iter, inserted] = type_to_id.try_emplace(type, static_cast<uint32_t>(types.size() + 1u));
        if (inserted) {
            if (type->description().empty()) { return UINT32_MAX; }
            types.emplace_back(type);
        }
        return iter->second;
    }

    [[nodiscard]] uint32_t intern_constant(ConstantData const &c) noexcept {
        if (!c || c.type() == nullptr || c.raw() == nullptr) { return UINT32_MAX; }
        if (!callable_graph_codec_detail::is_valid_constant_type(c.type())) { return UINT32_MAX; }
        auto type_id = intern_type(c.type());
        if (type_id == 0u || type_id == UINT32_MAX) { return UINT32_MAX; }
        auto &bucket = const_buckets[c.hash()];
        auto sz = c.type()->size();
        for (uint32_t idx : bucket) {
            auto &e = constants[idx];
            if (e.type_id == type_id && e.data.hash() == c.hash() &&
                e.data.type()->size() == sz &&
                (sz == 0u || std::memcmp(e.data.raw(), c.raw(), sz) == 0)) {
                return idx;
            }
        }
        auto idx = static_cast<uint32_t>(constants.size());
        constants.emplace_back(callable_graph_codec_detail::InternedConstantEntry{type_id, c});
        bucket.emplace_back(idx);
        return idx;
    }
};

struct CallableLibrary::GraphReadCtx {
    callable_graph_codec_detail::ByteReader reader;
    luisa::vector<const Type *> types;// 1-based indexed (types[0] == nullptr)
    luisa::vector<ConstantData> constants;
    luisa::vector<luisa::shared_ptr<detail::FunctionBuilder>> functions;
    size_t depth{0u};

    explicit GraphReadCtx(luisa::span<const std::byte> data) noexcept : reader{data} {}

    [[nodiscard]] bool resolve_nullable_type(uint32_t type_id, const Type *&out) const noexcept {
        if (type_id >= types.size()) { return false; }
        out = types[type_id];
        return true;
    }
    [[nodiscard]] bool resolve_nonnull_type(uint32_t type_id, const Type *&out) const noexcept {
        if (type_id == 0u || type_id >= types.size() || types[type_id] == nullptr) { return false; }
        out = types[type_id];
        return true;
    }
};

template<typename StmtT>
luisa::unique_ptr<StmtT> CallableLibrary::_fg_alloc_stmt() noexcept {
#ifdef LUISA_USE_SYSTEM_STL
    return luisa::unique_ptr<StmtT>{new StmtT{}};
#else
    return luisa::unique_ptr<StmtT>{new (luisa::allocate_with_allocator<StmtT>()) StmtT{}};
#endif
}

template<typename ExprT>
luisa::unique_ptr<ExprT> CallableLibrary::_fg_alloc_expr() noexcept {
#ifdef LUISA_USE_SYSTEM_STL
    return luisa::unique_ptr<ExprT>{new ExprT{}};
#else
    return luisa::unique_ptr<ExprT>{new (luisa::allocate_with_allocator<ExprT>()) ExprT{}};
#endif
}

bool CallableLibrary::_fg_collect(detail::FunctionBuilder const *fb, GraphWriteCtx &ctx) noexcept {
    if (fb == nullptr || !fb->hash_computed()) { return false; }
    if (ctx.func_to_id.contains(fb)) { return true; }
    if (!ctx.visiting.emplace(fb).second) { return false; }// cyclic graph
    if (ctx.depth >= callable_graph_codec_detail::kMaxAstDepth) { return false; }
    if (!fb->_used_external_functions.empty() ||
        !fb->_internalizer_arguments.empty() ||
        !fb->_captured_external_variables.empty() ||
        fb->_arguments.size() != fb->_bound_arguments.size()) {
        return false;
    }
    for (auto &&b : fb->_bound_arguments) {
        if (!luisa::holds_alternative<luisa::monostate>(b)) { return false; }
    }
    if (fb->_return_type && ctx.intern_type(*fb->_return_type) == UINT32_MAX) { return false; }
    auto check_var = [&](Variable const &v) noexcept {
        return v._type != nullptr && ctx.intern_type(v._type) != UINT32_MAX;
    };
    for (auto const &v : fb->_builtin_variables) {
        if (!check_var(v)) { return false; }
    }
    for (auto const &v : fb->_arguments) {
        if (!check_var(v)) { return false; }
    }
    for (auto const &v : fb->_local_variables) {
        if (!check_var(v)) { return false; }
    }
    for (auto const &v : fb->_shared_variables) {
        if (!check_var(v)) { return false; }
    }
    for (auto const &c : fb->_captured_constants) {
        if (ctx.intern_constant(c) == UINT32_MAX) { return false; }
    }

    luisa::unordered_set<const Expression *> seen_exprs;
    auto inspect_expr = [&](auto &self, Expression const *e, bool allow_null, size_t d) noexcept -> bool {
        if (e == nullptr) { return allow_null; }
        if (d >= callable_graph_codec_detail::kMaxAstDepth) { return false; }
        if (!seen_exprs.emplace(e).second) { return true; }
        if (e->builder() != fb) { return false; }
        if (e->_type != nullptr && ctx.intern_type(e->_type) == UINT32_MAX) { return false; }
        switch (e->tag()) {
            case Expression::Tag::UNARY:
                return self(self, static_cast<UnaryExpr const *>(e)->operand(), false, d + 1u);
            case Expression::Tag::BINARY: {
                auto *b = static_cast<BinaryExpr const *>(e);
                return self(self, b->lhs(), false, d + 1u) &&
                       self(self, b->rhs(), false, d + 1u);
            }
            case Expression::Tag::MEMBER:
                return self(self, static_cast<MemberExpr const *>(e)->self(), false, d + 1u);
            case Expression::Tag::ACCESS: {
                auto *a = static_cast<AccessExpr const *>(e);
                return self(self, a->range(), false, d + 1u) &&
                       self(self, a->index(), false, d + 1u);
            }
            case Expression::Tag::LITERAL: return true;
            case Expression::Tag::REF:
                return check_var(static_cast<RefExpr const *>(e)->variable());
            case Expression::Tag::CONSTANT:
                return ctx.intern_constant(static_cast<ConstantExpr const *>(e)->data()) != UINT32_MAX;
            case Expression::Tag::CALL: {
                auto *c = static_cast<CallExpr const *>(e);
                if (c->is_external() ||
                    luisa::holds_alternative<CallExpr::ExternalCallee>(c->_func)) {
                    return false;
                }
                for (auto *arg : c->arguments()) {
                    if (!self(self, arg, false, d + 1u)) { return false; }
                }
                if (!c->is_builtin()) {
                    if (c->_op != CallOp::CUSTOM) { return false; }
                    if (auto *callee = luisa::get_if<CallExpr::CustomCallee>(&c->_func);
                        callee != nullptr && *callee != nullptr) {
                        ++ctx.depth;
                        bool ok = _fg_collect(*callee, ctx);
                        --ctx.depth;
                        if (!ok) { return false; }
                    }
                }
                return true;
            }
            case Expression::Tag::CAST:
                return self(self, static_cast<CastExpr const *>(e)->expression(), false, d + 1u);
            case Expression::Tag::TYPE_ID: {
                auto *dt = static_cast<TypeIDExpr const *>(e)->data_type();
                return dt != nullptr && ctx.intern_type(dt) != UINT32_MAX;
            }
            case Expression::Tag::STRING_ID:
                return true;
            case Expression::Tag::FUNC_REF:
            case Expression::Tag::CPUCUSTOM:
            case Expression::Tag::GPUCUSTOM:
                return false;
        }
        return false;
    };

    auto inspect_stmt = [&](auto &self, Statement const *s, size_t d) noexcept -> bool {
        if (s == nullptr || d >= callable_graph_codec_detail::kMaxAstDepth) { return false; }
        switch (s->tag()) {
            case Statement::Tag::BREAK:
            case Statement::Tag::CONTINUE:
            case Statement::Tag::COMMENT:
                return true;
            case Statement::Tag::RETURN:
                return inspect_expr(inspect_expr, static_cast<ReturnStmt const *>(s)->_expr, true, d + 1u);
            case Statement::Tag::SCOPE: {
                for (auto *child : static_cast<ScopeStmt const *>(s)->_statements) {
                    if (!self(self, child, d + 1u)) { return false; }
                }
                return true;
            }
            case Statement::Tag::IF: {
                auto *ifs = static_cast<IfStmt const *>(s);
                return inspect_expr(inspect_expr, ifs->_condition, false, d + 1u) &&
                       self(self, &ifs->_true_branch, d + 1u) &&
                       self(self, &ifs->_false_branch, d + 1u);
            }
            case Statement::Tag::LOOP: {
                auto *ls = static_cast<LoopStmt const *>(s);
                return inspect_expr(inspect_expr, ls->_while_condition, true, d + 1u) &&
                       self(self, &ls->_body, d + 1u);
            }
            case Statement::Tag::EXPR:
                return inspect_expr(inspect_expr, static_cast<ExprStmt const *>(s)->_expr, false, d + 1u);
            case Statement::Tag::SWITCH: {
                auto *sw = static_cast<SwitchStmt const *>(s);
                return inspect_expr(inspect_expr, sw->_expr, false, d + 1u) &&
                       self(self, &sw->_body, d + 1u);
            }
            case Statement::Tag::SWITCH_CASE: {
                auto *sc = static_cast<SwitchCaseStmt const *>(s);
                return sc->_expressions.empty() &&
                       inspect_expr(inspect_expr, sc->_expr, false, d + 1u) &&
                       self(self, &sc->_body, d + 1u);
            }
            case Statement::Tag::SWITCH_CASE_GROUP: {
                auto *sc = static_cast<SwitchCaseStmt const *>(s);
                if (sc->_expressions.size() <= 1u) { return false; }
                for (auto *e : sc->_expressions) {
                    if (!inspect_expr(inspect_expr, e, false, d + 1u)) { return false; }
                }
                return self(self, &sc->_body, d + 1u);
            }
            case Statement::Tag::SWITCH_DEFAULT:
                return self(self, &static_cast<SwitchDefaultStmt const *>(s)->_body, d + 1u);
            case Statement::Tag::ASSIGN: {
                auto *as = static_cast<AssignStmt const *>(s);
                return inspect_expr(inspect_expr, as->_lhs, false, d + 1u) &&
                       inspect_expr(inspect_expr, as->_rhs, false, d + 1u);
            }
            case Statement::Tag::FOR: {
                auto *fs = static_cast<ForStmt const *>(s);
                return inspect_expr(inspect_expr, fs->_var, false, d + 1u) &&
                       inspect_expr(inspect_expr, fs->_cond, false, d + 1u) &&
                       inspect_expr(inspect_expr, fs->_step, false, d + 1u) &&
                       self(self, &fs->_body, d + 1u);
            }
            case Statement::Tag::RAY_QUERY: {
                auto *rq = static_cast<RayQueryStmt const *>(s);
                return inspect_expr(inspect_expr, rq->_query, false, d + 1u) &&
                       self(self, &rq->_on_triangle_candidate, d + 1u) &&
                       self(self, &rq->_on_procedural_candidate, d + 1u);
            }
            case Statement::Tag::AUTO_DIFF:
                return self(self, &static_cast<AutoDiffStmt const *>(s)->_body, d + 1u);
            case Statement::Tag::SUSPEND: {
                auto *su = static_cast<SuspendStmt const *>(s);
                for (auto const &fe : su->_frame_exports) {
                    if (!inspect_expr(inspect_expr, fe.value, false, d + 1u)) { return false; }
                }
                for (auto const &ext : su->_extensions) {
                    if (ext == nullptr || !ext->binding_projections().empty()) { return false; }
                }
                for (auto *bv : su->_extension_binding_values) {
                    if (!inspect_expr(inspect_expr, bv, false, d + 1u)) { return false; }
                }
                return true;
            }
            case Statement::Tag::PRINT: {
                for (auto *arg : static_cast<PrintStmt const *>(s)->_args) {
                    if (!inspect_expr(inspect_expr, arg, false, d + 1u)) { return false; }
                }
                return true;
            }
            case Statement::Tag::DEBUG_BREAK:
                return false;
        }
        return false;
    };

    if (!inspect_stmt(inspect_stmt, &fb->_body, 0u)) { return false; }

    // Ensure every declared custom callable dependency is also collected in deterministic order
    luisa::vector<const detail::FunctionBuilder *> remaining;
    for (auto const &callee : fb->_used_custom_callables) {
        if (callee == nullptr) { return false; }
        if (!ctx.func_to_id.contains(callee.get())) {
            remaining.emplace_back(callee.get());
        }
    }
    luisa::sort(remaining.begin(), remaining.end(), [](auto *a, auto *b) noexcept {
        if (a->hash() != b->hash()) { return a->hash() < b->hash(); }
        if (a->_tag != b->_tag) { return to_underlying(a->_tag) < to_underlying(b->_tag); }
        return a->_name < b->_name;
    });
    for (auto *callee : remaining) {
        ++ctx.depth;
        bool ok = _fg_collect(callee, ctx);
        --ctx.depth;
        if (!ok) { return false; }
    }

    ctx.visiting.erase(fb);
    auto id = static_cast<uint32_t>(ctx.func_order.size());
    ctx.func_order.emplace_back(fb);
    ctx.func_to_id.emplace(fb, id);
    return true;
}

bool CallableLibrary::_fg_write_expr(Expression const *expr, detail::FunctionBuilder const *owner,
                                     GraphWriteCtx &ctx, uint32_t &out_id) noexcept {
    if (expr == nullptr) {
        out_id = 0u;
        return true;
    }
    if (auto iter = ctx.expr_to_id.find(expr); iter != ctx.expr_to_id.end()) {
        out_id = iter->second;
        return true;
    }
    if (expr->builder() != owner || ctx.depth >= callable_graph_codec_detail::kMaxAstDepth) {
        return false;
    }
    ++ctx.depth;
    bool ok = true;
    uint32_t child0 = 0u, child1 = 0u;
    luisa::vector<uint32_t> call_args;

    switch (expr->_tag) {
        case Expression::Tag::UNARY:
            ok = _fg_write_expr(static_cast<UnaryExpr const *>(expr)->_operand, owner, ctx, child0);
            break;
        case Expression::Tag::BINARY: {
            auto *b = static_cast<BinaryExpr const *>(expr);
            ok = _fg_write_expr(b->_lhs, owner, ctx, child0) &&
                 _fg_write_expr(b->_rhs, owner, ctx, child1);
            break;
        }
        case Expression::Tag::MEMBER:
            ok = _fg_write_expr(static_cast<MemberExpr const *>(expr)->_self, owner, ctx, child0);
            break;
        case Expression::Tag::ACCESS: {
            auto *a = static_cast<AccessExpr const *>(expr);
            ok = _fg_write_expr(a->_range, owner, ctx, child0) &&
                 _fg_write_expr(a->_index, owner, ctx, child1);
            break;
        }
        case Expression::Tag::CALL: {
            auto *c = static_cast<CallExpr const *>(expr);
            call_args.reserve(c->_arguments.size());
            for (auto *arg : c->_arguments) {
                uint32_t aid = 0u;
                if (!_fg_write_expr(arg, owner, ctx, aid) || aid == 0u) {
                    ok = false;
                    break;
                }
                call_args.emplace_back(aid);
            }
            break;
        }
        case Expression::Tag::CAST:
            ok = _fg_write_expr(static_cast<CastExpr const *>(expr)->_source, owner, ctx, child0);
            break;
        case Expression::Tag::LITERAL:
        case Expression::Tag::REF:
        case Expression::Tag::CONSTANT:
        case Expression::Tag::TYPE_ID:
        case Expression::Tag::STRING_ID:
            break;
        default:
            ok = false;
            break;
    }
    --ctx.depth;
    if (!ok) { return false; }

    auto id = ++ctx.expr_count;
    ctx.expr_to_id.emplace(expr, id);
    out_id = id;

    auto &w = ctx.expr_writer;
    uint32_t type_id = expr->_type == nullptr ? 0u : ctx.intern_type(expr->_type);
    if (type_id == UINT32_MAX) { return false; }
    w.write(type_id);
    w.write<uint64_t>(expr->hash());
    w.write<uint32_t>(to_underlying(expr->_tag));
    w.write<uint8_t>(to_underlying(expr->_usage));

    switch (expr->_tag) {
        case Expression::Tag::UNARY: {
            w.write(child0);
            w.write(static_cast<UnaryExpr const *>(expr)->_op);
            break;
        }
        case Expression::Tag::BINARY: {
            w.write(child0);
            w.write(child1);
            w.write(static_cast<BinaryExpr const *>(expr)->_op);
            break;
        }
        case Expression::Tag::MEMBER: {
            auto *m = static_cast<MemberExpr const *>(expr);
            w.write(child0);
            w.write(m->_swizzle_size);
            w.write(m->_swizzle_code);
            break;
        }
        case Expression::Tag::ACCESS: {
            w.write(child0);
            w.write(child1);
            break;
        }
        case Expression::Tag::LITERAL: {
            auto *lit = static_cast<LiteralExpr const *>(expr);
            w.write<uint32_t>(static_cast<uint32_t>(lit->_value.index()));
            luisa::visit([&]<typename T>(T const &val) noexcept {
                w.write<uint32_t>(static_cast<uint32_t>(sizeof(T)));
                w.write(val);
            }, lit->_value);
            break;
        }
        case Expression::Tag::REF: {
            auto const &v = static_cast<RefExpr const *>(expr)->_variable;
            uint32_t vtid = ctx.intern_type(v._type);
            if (vtid == 0u || vtid == UINT32_MAX) { return false; }
            w.write(vtid);
            w.write(v._uid);
            w.write<uint32_t>(to_underlying(v._tag));
            break;
        }
        case Expression::Tag::CONSTANT: {
            uint32_t cid = ctx.intern_constant(static_cast<ConstantExpr const *>(expr)->_data);
            if (cid == UINT32_MAX) { return false; }
            w.write(cid);
            break;
        }
        case Expression::Tag::CALL: {
            auto *c = static_cast<CallExpr const *>(expr);
            w.write<uint32_t>(static_cast<uint32_t>(call_args.size()));
            for (uint32_t aid : call_args) { w.write(aid); }
            w.write(c->_op);
            if (is_builtin_operation(c->_op)) {
                w.write<uint64_t>(c->_curve_basis_set.to_u64());
            } else {
                if (auto *callee = luisa::get_if<CallExpr::CustomCallee>(&c->_func);
                    callee != nullptr && *callee != nullptr) {
                    auto it = ctx.func_to_id.find(*callee);
                    if (it == ctx.func_to_id.end()) { return false; }
                    w.write<uint8_t>(1u);
                    w.write(it->second);
                } else {
                    w.write<uint8_t>(0u);
                }
            }
            break;
        }
        case Expression::Tag::CAST: {
            w.write(child0);
            w.write(static_cast<CastExpr const *>(expr)->_op);
            break;
        }
        case Expression::Tag::TYPE_ID: {
            uint32_t dtid = ctx.intern_type(static_cast<TypeIDExpr const *>(expr)->_data_type);
            if (dtid == 0u || dtid == UINT32_MAX) { return false; }
            w.write(dtid);
            break;
        }
        case Expression::Tag::STRING_ID: {
            w.write_string(static_cast<StringIDExpr const *>(expr)->_data);
            break;
        }
        default:
            return false;
    }
    return true;
}

bool CallableLibrary::_fg_write_stmt(Statement const &stmt, detail::FunctionBuilder const *owner,
                                     GraphWriteCtx &ctx) noexcept {
    if (ctx.depth >= callable_graph_codec_detail::kMaxAstDepth) { return false; }
    auto &w = ctx.writer;
    w.write<uint64_t>(stmt.hash());
    w.write<uint32_t>(to_underlying(stmt._tag));
    auto write_req_expr = [&](Expression const *e) noexcept {
        uint32_t id = 0u;
        if (e == nullptr || !_fg_write_expr(e, owner, ctx, id) || id == 0u) { return false; }
        w.write(id);
        return true;
    };
    auto write_opt_expr = [&](Expression const *e) noexcept {
        uint32_t id = 0u;
        if (!_fg_write_expr(e, owner, ctx, id)) { return false; }
        w.write(id);
        return true;
    };
    ++ctx.depth;
    bool ok = true;
    switch (stmt._tag) {
        case Statement::Tag::BREAK:
        case Statement::Tag::CONTINUE:
            break;
        case Statement::Tag::RETURN:
            ok = write_opt_expr(static_cast<ReturnStmt const &>(stmt)._expr);
            break;
        case Statement::Tag::SCOPE: {
            auto const &sc = static_cast<ScopeStmt const &>(stmt);
            w.write<uint32_t>(static_cast<uint32_t>(sc._statements.size()));
            for (auto *s : sc._statements) {
                if (s == nullptr || !_fg_write_stmt(*s, owner, ctx)) {
                    ok = false;
                    break;
                }
            }
            break;
        }
        case Statement::Tag::IF: {
            auto const &ifs = static_cast<IfStmt const &>(stmt);
            ok = write_req_expr(ifs._condition) &&
                 _fg_write_stmt(ifs._true_branch, owner, ctx) &&
                 _fg_write_stmt(ifs._false_branch, owner, ctx);
            break;
        }
        case Statement::Tag::LOOP: {
            auto const &ls = static_cast<LoopStmt const &>(stmt);
            ok = write_opt_expr(ls._while_condition);
            if (ok) {
                w.write<uint64_t>(static_cast<uint64_t>(ls._while_condition_statement_count));
                ok = _fg_write_stmt(ls._body, owner, ctx);
            }
            break;
        }
        case Statement::Tag::EXPR:
            ok = write_req_expr(static_cast<ExprStmt const &>(stmt)._expr);
            break;
        case Statement::Tag::SWITCH: {
            auto const &sw = static_cast<SwitchStmt const &>(stmt);
            ok = write_req_expr(sw._expr) && _fg_write_stmt(sw._body, owner, ctx);
            break;
        }
        case Statement::Tag::SWITCH_CASE: {
            auto const &sc = static_cast<SwitchCaseStmt const &>(stmt);
            ok = write_req_expr(sc._expr) && _fg_write_stmt(sc._body, owner, ctx);
            break;
        }
        case Statement::Tag::SWITCH_CASE_GROUP: {
            auto const &sc = static_cast<SwitchCaseStmt const &>(stmt);
            w.write<uint32_t>(static_cast<uint32_t>(sc._expressions.size()));
            for (auto *e : sc._expressions) {
                if (!write_req_expr(e)) {
                    ok = false;
                    break;
                }
            }
            if (ok) { ok = _fg_write_stmt(sc._body, owner, ctx); }
            break;
        }
        case Statement::Tag::SWITCH_DEFAULT:
            ok = _fg_write_stmt(static_cast<SwitchDefaultStmt const &>(stmt)._body, owner, ctx);
            break;
        case Statement::Tag::ASSIGN: {
            auto const &as = static_cast<AssignStmt const &>(stmt);
            ok = write_req_expr(as._lhs) && write_req_expr(as._rhs);
            break;
        }
        case Statement::Tag::FOR: {
            auto const &fs = static_cast<ForStmt const &>(stmt);
            ok = write_req_expr(fs._var) && write_req_expr(fs._cond) &&
                 write_req_expr(fs._step) && _fg_write_stmt(fs._body, owner, ctx);
            break;
        }
        case Statement::Tag::COMMENT:
            w.write_string(static_cast<CommentStmt const &>(stmt)._comment);
            break;
        case Statement::Tag::RAY_QUERY: {
            auto const &rq = static_cast<RayQueryStmt const &>(stmt);
            ok = write_req_expr(rq._query) &&
                 _fg_write_stmt(rq._on_triangle_candidate, owner, ctx) &&
                 _fg_write_stmt(rq._on_procedural_candidate, owner, ctx);
            break;
        }
        case Statement::Tag::AUTO_DIFF:
            ok = _fg_write_stmt(static_cast<AutoDiffStmt const &>(stmt)._body, owner, ctx);
            break;
        case Statement::Tag::SUSPEND: {
            auto const &su = static_cast<SuspendStmt const &>(stmt);
            w.write(su._token);
            w.write_string(su._name);
            w.write<uint32_t>(static_cast<uint32_t>(su._frame_exports.size()));
            for (auto const &fe : su._frame_exports) {
                w.write_string(fe.name);
                if (!write_req_expr(fe.value)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) { break; }
            w.write<uint32_t>(static_cast<uint32_t>(su._extensions.size()));
            for (auto const &ext : su._extensions) {
                if (ext == nullptr || !ext->binding_projections().empty()) {
                    ok = false;
                    break;
                }
                w.write_string(ext->schema());
                w.write<uint32_t>(ext->version());
                w.write<uint8_t>(ext->is_annotation() ? 1u : 0u);
                w.write(ext->fallback());
                w.write<uint32_t>(static_cast<uint32_t>(ext->bindings().size()));
                for (auto const &b : ext->bindings()) {
                    w.write_string(b.name);
                    w.write(b.access);
                    w.write(b.lifetime);
                    w.write(b.index);
                }
                w.write<uint32_t>(static_cast<uint32_t>(ext->attributes().size()));
                for (auto const &attr : ext->attributes()) {
                    w.write_string(attr.name);
                    w.write<uint8_t>(static_cast<uint8_t>(attr.value.index()));
                    luisa::visit([&]<typename V>(V const &val) noexcept {
                        if constexpr (std::is_same_v<V, luisa::string>) {
                            w.write_string(val);
                        } else {
                            w.write(val);
                        }
                    }, attr.value);
                }
            }
            if (!ok) { break; }
            w.write<uint32_t>(static_cast<uint32_t>(su._extension_binding_values.size()));
            for (auto *bv : su._extension_binding_values) {
                if (!write_req_expr(bv)) {
                    ok = false;
                    break;
                }
            }
            break;
        }
        case Statement::Tag::PRINT: {
            auto const &pr = static_cast<PrintStmt const &>(stmt);
            w.write_string(pr._format);
            w.write<uint32_t>(static_cast<uint32_t>(pr._args.size()));
            for (auto *arg : pr._args) {
                if (!write_req_expr(arg)) {
                    ok = false;
                    break;
                }
            }
            break;
        }
        default:
            ok = false;
            break;
    }
    --ctx.depth;
    return ok;
}

bool CallableLibrary::_fg_write_func(detail::FunctionBuilder const &fb, GraphWriteCtx &ctx) noexcept {
    ctx.expr_to_id.clear();
    ctx.expr_writer.bytes.clear();
    ctx.expr_count = 0u;
    ctx.current_owner = &fb;

    // Collect expressions in post-order first, then append metadata + expressions + body statement
    auto stmt_bytes_holder = callable_graph_codec_detail::ByteWriter{};
    std::swap(ctx.writer, stmt_bytes_holder);
    if (!_fg_write_stmt(fb._body, &fb, ctx)) {
        std::swap(ctx.writer, stmt_bytes_holder);
        return false;
    }
    std::swap(stmt_bytes_holder, ctx.writer);

    auto &w = ctx.writer;
    w.write<uint64_t>(fb._hash);
    w.write<uint64_t>(fb._xir_cache_hash);
    w.write<uint32_t>(to_underlying(fb._tag));
    w.write_string(fb._name);
    if (fb._return_type) {
        w.write<uint8_t>(1u);
        uint32_t rtid = fb._return_type.value() == nullptr ? 0u : ctx.intern_type(*fb._return_type);
        if (rtid == UINT32_MAX) { return false; }
        w.write(rtid);
    } else {
        w.write<uint8_t>(0u);
        w.write<uint32_t>(0u);
    }
    w.write<uint64_t>(fb._required_curve_bases.to_u64());
    w.write(fb._block_size);
    w.write<uint8_t>(fb._allowed_warp_size);
    uint8_t flags = static_cast<uint8_t>(fb._requires_atomic_float ? 1u : 0u) |
                    static_cast<uint8_t>(fb._requires_printing ? 2u : 0u) |
                    static_cast<uint8_t>(fb._use_cooperative_operations ? 4u : 0u) |
                    static_cast<uint8_t>(fb._requires_noinline ? 8u : 0u);
    w.write(flags);

    luisa::vector<std::pair<luisa::string_view, const Function::FunctionAttribute *>> sorted_attrs;
    sorted_attrs.reserve(fb._func_attributes.size());
    for (auto const &[k, v] : fb._func_attributes) {
        sorted_attrs.emplace_back(luisa::string_view{k}, &v);
    }
    luisa::sort(sorted_attrs.begin(), sorted_attrs.end(),
                [](auto const &a, auto const &b) noexcept { return a.first < b.first; });
    w.write<uint32_t>(static_cast<uint32_t>(sorted_attrs.size()));
    for (auto const &[k, vptr] : sorted_attrs) {
        w.write_string(k);
        w.write<uint32_t>(static_cast<uint32_t>(vptr->index()));
        luisa::visit([&]<typename T>(T const &val) noexcept {
            w.write<uint32_t>(static_cast<uint32_t>(sizeof(T)));
            w.write(val);
        }, *vptr);
    }

    auto write_vars = [&](luisa::span<const Variable> vars) noexcept {
        w.write<uint32_t>(static_cast<uint32_t>(vars.size()));
        for (auto const &v : vars) {
            uint32_t tid = ctx.intern_type(v._type);
            if (tid == 0u || tid == UINT32_MAX) { return false; }
            w.write(tid);
            w.write(v._uid);
            w.write<uint32_t>(to_underlying(v._tag));
        }
        return true;
    };
    if (!write_vars(fb._builtin_variables) ||
        !write_vars(fb._arguments) ||
        !write_vars(fb._local_variables) ||
        !write_vars(fb._shared_variables)) {
        return false;
    }

    w.write<uint32_t>(static_cast<uint32_t>(fb._captured_constants.size()));
    for (auto const &c : fb._captured_constants) {
        uint32_t cid = ctx.intern_constant(c);
        if (cid == UINT32_MAX) { return false; }
        w.write(cid);
    }

    w.write<uint32_t>(static_cast<uint32_t>(fb._variable_usages.size()));
    for (auto u : fb._variable_usages) {
        w.write<uint8_t>(to_underlying(u));
    }

    w.write<uint32_t>(static_cast<uint32_t>(fb._variables_names.size()));
    for (auto const &vn : fb._variables_names) {
        w.write_string(vn);
    }

    auto write_call_ops = [&](CallOpSet const &ops) noexcept {
        std::array<uint8_t, (call_op_count + 7u) / 8u> bits{};
        for (size_t i = 0u; i < call_op_count; ++i) {
            if (ops._bits[i]) { bits[i / 8u] |= static_cast<uint8_t>(1u << (i & 7u)); }
        }
        w.write_bytes(bits.data(), bits.size());
    };
    write_call_ops(fb._direct_builtin_callables);
    write_call_ops(fb._propagated_builtin_callables);

    luisa::vector<uint32_t> callee_ids;
    callee_ids.reserve(fb._used_custom_callables.size());
    for (auto const &c : fb._used_custom_callables) {
        auto it = ctx.func_to_id.find(c.get());
        if (it == ctx.func_to_id.end()) { return false; }
        callee_ids.emplace_back(it->second);
    }
    luisa::sort(callee_ids.begin(), callee_ids.end());
    w.write<uint32_t>(static_cast<uint32_t>(callee_ids.size()));
    for (uint32_t cid : callee_ids) { w.write(cid); }

    w.write<uint32_t>(ctx.expr_count);
    w.write_bytes(ctx.expr_writer.bytes.data(), ctx.expr_writer.bytes.size());
    w.write_bytes(stmt_bytes_holder.bytes.data(), stmt_bytes_holder.bytes.size());
    return true;
}

luisa::vector<std::byte> CallableLibrary::serialize_function_graph(
    luisa::span<const luisa::shared_ptr<const detail::FunctionBuilder>> roots) noexcept {
    GraphWriteCtx ctx;
    for (auto const &root : roots) {
        if (!_fg_collect(root.get(), ctx)) { return {}; }
    }

    // Reserve header + tables inside func_writer after all functions are written
    // Because all types & constants are interned in _fg_collect, the Type/Constant tables are already complete!
    auto &w = ctx.writer;
    w.write_bytes(callable_graph_codec_detail::kGraphMagic.data(),
                  callable_graph_codec_detail::kGraphMagic.size());
    w.write(callable_graph_codec_detail::kGraphVersion);
    w.write<uint32_t>(static_cast<uint32_t>(sizeof(size_t)));

    // Type table (1..T)
    w.write<uint32_t>(static_cast<uint32_t>(ctx.types.size()));
    for (auto *t : ctx.types) {
        w.write_string(t->description());
    }

    // Constant table
    w.write<uint32_t>(static_cast<uint32_t>(ctx.constants.size()));
    for (auto const &entry : ctx.constants) {
        auto sz = entry.data.type()->size();
        w.write(entry.type_id);
        w.write<uint64_t>(static_cast<uint64_t>(sz));
        w.write_bytes(entry.data.raw(), sz);
    }

    // Graph indices
    w.write<uint32_t>(static_cast<uint32_t>(ctx.func_order.size()));
    w.write<uint32_t>(static_cast<uint32_t>(roots.size()));
    for (auto const &root : roots) {
        w.write<uint32_t>(ctx.func_to_id.at(root.get()));
    }

    for (auto *fb : ctx.func_order) {
        if (!_fg_write_func(*fb, ctx)) { return {}; }
    }
    return std::move(w.bytes);
}

bool CallableLibrary::_fg_read_stmt_payload(Statement &stmt, Statement::Tag tag,
                                            detail::FunctionBuilder &fb, GraphReadCtx &ctx) noexcept {
    if (ctx.depth >= callable_graph_codec_detail::kMaxAstDepth) { return false; }
    auto &r = ctx.reader;

    auto read_expr_id = [&](uint32_t &out_id, bool allow_null) noexcept -> bool {
        if (!r.read(out_id)) { return false; }
        if (!allow_null && out_id == 0u) { return false; }
        if (out_id > fb._all_expressions.size()) { return false; }
        return true;
    };
    auto resolve_expr = [&](uint32_t id) noexcept -> const Expression * {
        return id == 0u ? nullptr : fb._all_expressions[id - 1u].get();
    };

    ++ctx.depth;
    bool ok = true;
    switch (tag) {
        case Statement::Tag::BREAK:
        case Statement::Tag::CONTINUE:
            break;
        case Statement::Tag::RETURN: {
            uint32_t eid = 0u;
            ok = read_expr_id(eid, true);
            if (ok) { static_cast<ReturnStmt &>(stmt)._expr = resolve_expr(eid); }
            break;
        }
        case Statement::Tag::SCOPE: {
            uint32_t count = 0u;
            if (!r.read(count) || count > callable_graph_codec_detail::kMaxTableCount) {
                ok = false;
                break;
            }
            auto &sc = static_cast<ScopeStmt &>(stmt);
            sc._statements.reserve(count);
            for (uint32_t i = 0u; i < count; ++i) {
                auto *child = _fg_read_stmt_ptr(fb, ctx);
                if (child == nullptr) {
                    ok = false;
                    break;
                }
                sc._statements.emplace_back(child);
            }
            break;
        }
        case Statement::Tag::IF: {
            uint32_t cid = 0u;
            auto &ifs = static_cast<IfStmt &>(stmt);
            ok = read_expr_id(cid, false);
            if (ok) { ifs._condition = resolve_expr(cid); }
            ok = ok && _fg_read_stmt_into(ifs._true_branch, fb, ctx) &&
                 _fg_read_stmt_into(ifs._false_branch, fb, ctx);
            if (ok && (ifs._true_branch._tag != Statement::Tag::SCOPE ||
                       ifs._false_branch._tag != Statement::Tag::SCOPE)) {
                ok = false;
            }
            break;
        }
        case Statement::Tag::LOOP: {
            uint32_t cid = 0u;
            uint64_t stmt_count = 0u;
            auto &ls = static_cast<LoopStmt &>(stmt);
            ok = read_expr_id(cid, true) && r.read(stmt_count);
            if (ok) {
                ls._while_condition = resolve_expr(cid);
                ls._while_condition_statement_count = static_cast<size_t>(stmt_count);
                ok = _fg_read_stmt_into(ls._body, fb, ctx) &&
                     ls._body._tag == Statement::Tag::SCOPE;
            }
            break;
        }
        case Statement::Tag::EXPR: {
            uint32_t eid = 0u;
            ok = read_expr_id(eid, false);
            if (ok) { static_cast<ExprStmt &>(stmt)._expr = resolve_expr(eid); }
            break;
        }
        case Statement::Tag::SWITCH: {
            uint32_t eid = 0u;
            auto &sw = static_cast<SwitchStmt &>(stmt);
            ok = read_expr_id(eid, false);
            if (ok) {
                sw._expr = resolve_expr(eid);
                ok = _fg_read_stmt_into(sw._body, fb, ctx) &&
                     sw._body._tag == Statement::Tag::SCOPE;
            }
            break;
        }
        case Statement::Tag::SWITCH_CASE: {
            uint32_t eid = 0u;
            auto &sc = static_cast<SwitchCaseStmt &>(stmt);
            ok = read_expr_id(eid, false);
            if (ok) {
                sc._expr = resolve_expr(eid);
                ok = _fg_read_stmt_into(sc._body, fb, ctx) &&
                     sc._body._tag == Statement::Tag::SCOPE;
            }
            break;
        }
        case Statement::Tag::SWITCH_CASE_GROUP: {
            uint32_t count = 0u;
            if (!r.read(count) || count <= 1u || count > callable_graph_codec_detail::kMaxTableCount) {
                ok = false;
                break;
            }
            auto &sc = static_cast<SwitchCaseStmt &>(stmt);
            sc._expressions.reserve(count);
            for (uint32_t i = 0u; i < count; ++i) {
                uint32_t eid = 0u;
                if (!read_expr_id(eid, false)) {
                    ok = false;
                    break;
                }
                sc._expressions.emplace_back(resolve_expr(eid));
            }
            if (ok) {
                sc._expr = sc._expressions.front();
                ok = _fg_read_stmt_into(sc._body, fb, ctx) &&
                     sc._body._tag == Statement::Tag::SCOPE;
            }
            break;
        }
        case Statement::Tag::SWITCH_DEFAULT: {
            auto &sd = static_cast<SwitchDefaultStmt &>(stmt);
            ok = _fg_read_stmt_into(sd._body, fb, ctx) &&
                 sd._body._tag == Statement::Tag::SCOPE;
            break;
        }
        case Statement::Tag::ASSIGN: {
            uint32_t lhs_id = 0u, rhs_id = 0u;
            auto &as = static_cast<AssignStmt &>(stmt);
            ok = read_expr_id(lhs_id, false) && read_expr_id(rhs_id, false);
            if (ok) {
                as._lhs = resolve_expr(lhs_id);
                as._rhs = resolve_expr(rhs_id);
            }
            break;
        }
        case Statement::Tag::FOR: {
            uint32_t vid = 0u, cid = 0u, sid = 0u;
            auto &fs = static_cast<ForStmt &>(stmt);
            ok = read_expr_id(vid, false) && read_expr_id(cid, false) && read_expr_id(sid, false);
            if (ok) {
                fs._var = resolve_expr(vid);
                fs._cond = resolve_expr(cid);
                fs._step = resolve_expr(sid);
                ok = _fg_read_stmt_into(fs._body, fb, ctx) &&
                     fs._body._tag == Statement::Tag::SCOPE;
            }
            break;
        }
        case Statement::Tag::COMMENT:
            ok = r.read_string(static_cast<CommentStmt &>(stmt)._comment);
            break;
        case Statement::Tag::RAY_QUERY: {
            uint32_t qid = 0u;
            auto &rq = static_cast<RayQueryStmt &>(stmt);
            ok = read_expr_id(qid, false);
            if (ok) {
                auto *qexpr = resolve_expr(qid);
                if (qexpr == nullptr || qexpr->tag() != Expression::Tag::REF) {
                    ok = false;
                } else {
                    rq._query = static_cast<const RefExpr *>(qexpr);
                    ok = _fg_read_stmt_into(rq._on_triangle_candidate, fb, ctx) &&
                         rq._on_triangle_candidate._tag == Statement::Tag::SCOPE &&
                         _fg_read_stmt_into(rq._on_procedural_candidate, fb, ctx) &&
                         rq._on_procedural_candidate._tag == Statement::Tag::SCOPE;
                }
            }
            break;
        }
        case Statement::Tag::AUTO_DIFF: {
            auto &ad = static_cast<AutoDiffStmt &>(stmt);
            ok = _fg_read_stmt_into(ad._body, fb, ctx) &&
                 ad._body._tag == Statement::Tag::SCOPE;
            break;
        }
        case Statement::Tag::SUSPEND: {
            auto &su = static_cast<SuspendStmt &>(stmt);
            uint32_t fe_count = 0u, ext_count = 0u, bv_count = 0u;
            if (!r.read(su._token) || !r.read_string(su._name) || !r.read(fe_count) ||
                fe_count > callable_graph_codec_detail::kMaxTableCount) {
                ok = false;
                break;
            }
            su._frame_exports.reserve(fe_count);
            for (uint32_t i = 0u; i < fe_count; ++i) {
                luisa::string name;
                uint32_t vid = 0u;
                if (!r.read_string(name) || !read_expr_id(vid, false)) {
                    ok = false;
                    break;
                }
                su._frame_exports.emplace_back(CoroFrameExport{
                    .value = resolve_expr(vid),
                    .name = std::move(name)});
            }
            if (!ok || !r.read(ext_count) || ext_count > callable_graph_codec_detail::kMaxTableCount) {
                ok = false;
                break;
            }
            su._extensions.reserve(ext_count);
            for (uint32_t i = 0u; i < ext_count; ++i) {
                luisa::string schema;
                uint32_t version = 0u;
                uint8_t is_annotation = 0u;
                CoroSuspendFallback fallback{};
                uint32_t binding_count = 0u, attr_count = 0u;
                if (!r.read_string(schema) || schema.empty() ||
                    !r.read(version) || version == 0u ||
                    !r.read(is_annotation) ||
                    !r.read(fallback) ||
                    !r.read(binding_count) || binding_count > callable_graph_codec_detail::kMaxTableCount) {
                    ok = false;
                    break;
                }
                luisa::vector<CoroSuspendBinding> bindings;
                bindings.reserve(binding_count);
                for (uint32_t j = 0u; j < binding_count; ++j) {
                    CoroSuspendBinding b;
                    if (!r.read_string(b.name) || b.name.empty() ||
                        !r.read(b.access) || !r.read(b.lifetime) || !r.read(b.index)) {
                        ok = false;
                        break;
                    }
                    for (uint32_t k = 0u; k < j; ++k) {
                        if (bindings[k].name == b.name) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) { break; }
                    bindings.emplace_back(std::move(b));
                }
                if (!ok || !r.read(attr_count) || attr_count > callable_graph_codec_detail::kMaxTableCount) {
                    ok = false;
                    break;
                }
                luisa::vector<CoroSuspendAttribute> attributes;
                attributes.reserve(attr_count);
                for (uint32_t j = 0u; j < attr_count; ++j) {
                    CoroSuspendAttribute a;
                    uint8_t vtag = 0u;
                    if (!r.read_string(a.name) || a.name.empty() || !r.read(vtag)) {
                        ok = false;
                        break;
                    }
                    switch (vtag) {
                        case 0u: {
                            bool v{};
                            if (!r.read(v)) { ok = false; }
                            a.value = v;
                            break;
                        }
                        case 1u: {
                            int64_t v{};
                            if (!r.read(v)) { ok = false; }
                            a.value = v;
                            break;
                        }
                        case 2u: {
                            uint64_t v{};
                            if (!r.read(v)) { ok = false; }
                            a.value = v;
                            break;
                        }
                        case 3u: {
                            double v{};
                            if (!r.read(v)) { ok = false; }
                            a.value = v;
                            break;
                        }
                        case 4u: {
                            luisa::string v;
                            if (!r.read_string(v)) { ok = false; }
                            a.value = std::move(v);
                            break;
                        }
                        default:
                            ok = false;
                            break;
                    }
                    if (!ok) { break; }
                    for (uint32_t k = 0u; k < j; ++k) {
                        if (attributes[k].name == a.name) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) { break; }
                    attributes.emplace_back(std::move(a));
                }
                if (!ok) { break; }
                su._extensions.emplace_back(
                    is_annotation != 0u ?
                        make_coro_suspend_annotation_data(
                            std::move(schema), version, fallback,
                            std::move(bindings), std::move(attributes)) :
                        make_coro_suspend_extension_data(
                            std::move(schema), version, fallback,
                            std::move(bindings), std::move(attributes)));
            }
            if (!ok || !r.read(bv_count) || bv_count > callable_graph_codec_detail::kMaxTableCount) {
                ok = false;
                break;
            }
            su._extension_binding_values.reserve(bv_count);
            for (uint32_t i = 0u; i < bv_count; ++i) {
                uint32_t bvid = 0u;
                if (!read_expr_id(bvid, false)) {
                    ok = false;
                    break;
                }
                su._extension_binding_values.emplace_back(resolve_expr(bvid));
            }
            if (ok) {
                luisa::vector<bool> bound(su._extension_binding_values.size(), false);
                for (auto const &ext : su._extensions) {
                    for (auto const &b : ext->bindings()) {
                        if (b.index >= bound.size() || bound[b.index]) {
                            ok = false;
                            break;
                        }
                        bound[b.index] = true;
                        auto *val = su._extension_binding_values[b.index];
                        if (val == nullptr || val->type() == nullptr) {
                            ok = false;
                            break;
                        }
                        if ((b.access == CoroSuspendBindingAccess::write ||
                             b.access == CoroSuspendBindingAccess::read_write) &&
                            !callable_graph_codec_detail::is_writable_lvalue(val)) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) { break; }
                }
                if (ok) {
                    for (bool flag : bound) {
                        if (!flag) {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            break;
        }
        case Statement::Tag::PRINT: {
            auto &pr = static_cast<PrintStmt &>(stmt);
            uint32_t argc = 0u;
            if (!r.read_string(pr._format) || !r.read(argc) ||
                argc > callable_graph_codec_detail::kMaxTableCount) {
                ok = false;
                break;
            }
            pr._args.reserve(argc);
            for (uint32_t i = 0u; i < argc; ++i) {
                uint32_t aid = 0u;
                if (!read_expr_id(aid, false)) {
                    ok = false;
                    break;
                }
                pr._args.emplace_back(resolve_expr(aid));
            }
            break;
        }
        default:
            ok = false;
            break;
    }
    --ctx.depth;
    return ok;
}

bool CallableLibrary::_fg_read_stmt_into(Statement &stmt, detail::FunctionBuilder &fb, GraphReadCtx &ctx) noexcept {
    uint64_t hash = 0u;
    uint32_t raw_tag = 0u;
    if (!ctx.reader.read(hash) || !ctx.reader.read(raw_tag)) { return false; }
    auto tag = static_cast<Statement::Tag>(raw_tag);
    stmt._hash = hash;
    stmt._tag = tag;
    stmt._hash_computed = true;
    return _fg_read_stmt_payload(stmt, tag, fb, ctx);
}

Statement *CallableLibrary::_fg_read_stmt_ptr(detail::FunctionBuilder &fb, GraphReadCtx &ctx) noexcept {
    if (ctx.depth >= callable_graph_codec_detail::kMaxAstDepth) { return nullptr; }
    uint64_t hash = 0u;
    uint32_t raw_tag = 0u;
    if (!ctx.reader.read(hash) || !ctx.reader.read(raw_tag)) { return nullptr; }
    auto tag = static_cast<Statement::Tag>(raw_tag);

    Statement *raw_ptr = nullptr;
    auto init_node = [&]<typename T>() noexcept {
        auto ptr = CallableLibrary::_fg_alloc_stmt<T>();
        auto *p = ptr.get();
        fb._all_statements.emplace_back(std::move(ptr));
        p->_hash = hash;
        p->_tag = tag;
        p->_hash_computed = true;
        raw_ptr = p;
    };

    switch (tag) {
        case Statement::Tag::BREAK: init_node.template operator()<BreakStmt>(); return raw_ptr;
        case Statement::Tag::CONTINUE: init_node.template operator()<ContinueStmt>(); return raw_ptr;
        case Statement::Tag::RETURN: init_node.template operator()<ReturnStmt>(); break;
        case Statement::Tag::SCOPE: init_node.template operator()<ScopeStmt>(); break;
        case Statement::Tag::IF: init_node.template operator()<IfStmt>(); break;
        case Statement::Tag::LOOP: init_node.template operator()<LoopStmt>(); break;
        case Statement::Tag::EXPR: init_node.template operator()<ExprStmt>(); break;
        case Statement::Tag::SWITCH: init_node.template operator()<SwitchStmt>(); break;
        case Statement::Tag::SWITCH_CASE:
        case Statement::Tag::SWITCH_CASE_GROUP:
            init_node.template operator()<SwitchCaseStmt>();
            break;
        case Statement::Tag::SWITCH_DEFAULT: init_node.template operator()<SwitchDefaultStmt>(); break;
        case Statement::Tag::ASSIGN: init_node.template operator()<AssignStmt>(); break;
        case Statement::Tag::FOR: init_node.template operator()<ForStmt>(); break;
        case Statement::Tag::COMMENT: init_node.template operator()<CommentStmt>(); break;
        case Statement::Tag::RAY_QUERY: init_node.template operator()<RayQueryStmt>(); break;
        case Statement::Tag::AUTO_DIFF: init_node.template operator()<AutoDiffStmt>(); break;
        case Statement::Tag::SUSPEND: init_node.template operator()<SuspendStmt>(); break;
        case Statement::Tag::PRINT: init_node.template operator()<PrintStmt>(); break;
        default: return nullptr;
    }
    return _fg_read_stmt_payload(*raw_ptr, tag, fb, ctx) ? raw_ptr : nullptr;
}

bool CallableLibrary::_fg_read_func(detail::FunctionBuilder &fb, GraphReadCtx &ctx) noexcept {
    auto &r = ctx.reader;
    uint32_t raw_tag = 0u, return_type_id = 0u;
    uint8_t has_return_type = 0u, flags = 0u;
    uint64_t curve_bits = 0u;

    if (!r.read(fb._hash) || !r.read(fb._xir_cache_hash) ||
        !r.read(raw_tag) || !r.read_string(fb._name) ||
        !r.read(has_return_type) || !r.read(return_type_id) ||
        !r.read(curve_bits) || !r.read(fb._block_size) ||
        !r.read(fb._allowed_warp_size) || !r.read(flags)) {
        return false;
    }
    fb._hash_computed = true;
    fb._tag = static_cast<Function::Tag>(raw_tag);
    if (fb._tag != Function::Tag::KERNEL && fb._tag != Function::Tag::CALLABLE &&
        fb._tag != Function::Tag::RASTER_STAGE && fb._tag != Function::Tag::COROUTINE) {
        return false;
    }
    if (!callable_graph_codec_detail::is_valid_allowed_warp_size(fb._allowed_warp_size)) {
        return false;
    }
    if (has_return_type != 0u) {
        const Type *rt = nullptr;
        if (!ctx.resolve_nullable_type(return_type_id, rt)) { return false; }
        fb._return_type = rt;
    } else {
        fb._return_type.reset();
    }
    fb._required_curve_bases = CurveBasisSet::from_u64(curve_bits);
    fb._requires_atomic_float = (flags & 1u) != 0u;
    fb._requires_printing = (flags & 2u) != 0u;
    fb._use_cooperative_operations = (flags & 4u) != 0u;
    fb._requires_noinline = (flags & 8u) != 0u;

    uint32_t attr_count = 0u;
    if (!r.read(attr_count) || attr_count > callable_graph_codec_detail::kMaxTableCount) {
        return false;
    }
    using AttrV = Function::FunctionAttribute;
    for (uint32_t i = 0u; i < attr_count; ++i) {
        luisa::string key;
        uint32_t var_idx = 0u, byte_size = 0u;
        if (!r.read_string(key) || !r.read(var_idx) || !r.read(byte_size)) {
            return false;
        }
        const std::byte *val_ptr = nullptr;
        if (!r.read_span(byte_size, val_ptr)) { return false; }
        if (var_idx >= luisa::variant_size_v<AttrV>) { return false; }
        AttrV val;
        bool matched = false;
        auto emplace_attr = [&]<size_t I>(auto &self, std::integral_constant<size_t, I>) noexcept {
            if constexpr (I < luisa::variant_size_v<AttrV>) {
                if (I == var_idx) {
                    using T = luisa::variant_alternative_t<I, AttrV>;
                    if (byte_size == sizeof(T)) {
                        T tmp;
                        std::memcpy(&tmp, val_ptr, sizeof(T));
                        val.template emplace<T>(tmp);
                        matched = true;
                    }
                } else {
                    self(self, std::integral_constant<size_t, I + 1u>{});
                }
            }
        };
        emplace_attr(emplace_attr, std::integral_constant<size_t, 0u>{});
        if (!matched) { return false; }
        fb._func_attributes.emplace(std::move(key), std::move(val));
    }

    auto read_var = [&](Variable &v) noexcept {
        uint32_t tid = 0u, uid = 0u, vraw_tag = 0u;
        if (!r.read(tid) || !r.read(uid) || !r.read(vraw_tag)) { return false; }
        const Type *vt = nullptr;
        if (!ctx.resolve_nonnull_type(tid, vt)) { return false; }
        auto vtag = static_cast<Variable::Tag>(vraw_tag);
        if (!callable_graph_codec_detail::is_valid_variable_tag(vtag)) { return false; }
        v._type = vt;
        v._uid = uid;
        v._tag = vtag;
        return true;
    };
    auto read_vars = [&](luisa::vector<Variable> &out) noexcept {
        uint32_t count = 0u;
        if (!r.read(count) || count > callable_graph_codec_detail::kMaxTableCount) { return false; }
        out.resize(count);
        for (uint32_t i = 0u; i < count; ++i) {
            if (!read_var(out[i])) { return false; }
        }
        return true;
    };
    if (!read_vars(fb._builtin_variables) ||
        !read_vars(fb._arguments) ||
        !read_vars(fb._local_variables) ||
        !read_vars(fb._shared_variables)) {
        return false;
    }
    // Preserve exact unbound argument slot layout without sort_bindings()!
    fb._bound_arguments.reserve(fb._arguments.size());
    for (size_t i = 0u; i < fb._arguments.size(); ++i) {
        fb._bound_arguments.emplace_back(luisa::monostate{});
    }

    uint32_t const_count = 0u;
    if (!r.read(const_count) || const_count > callable_graph_codec_detail::kMaxTableCount) {
        return false;
    }
    fb._captured_constants.reserve(const_count);
    for (uint32_t i = 0u; i < const_count; ++i) {
        uint32_t cid = 0u;
        if (!r.read(cid) || cid >= ctx.constants.size()) { return false; }
        fb._captured_constants.emplace_back(ctx.constants[cid]);
    }

    uint32_t usage_count = 0u;
    if (!r.read(usage_count) || usage_count > callable_graph_codec_detail::kMaxTableCount) {
        return false;
    }
    fb._variable_usages.resize(usage_count);
    for (uint32_t i = 0u; i < usage_count; ++i) {
        uint8_t u = 0u;
        if (!r.read(u) || u > to_underlying(Usage::READ_WRITE)) { return false; }
        fb._variable_usages[i] = static_cast<Usage>(u);
    }
    auto check_var_uid = [&](Variable const &v) noexcept {
        return v._uid < fb._variable_usages.size();
    };
    for (auto const &v : fb._builtin_variables) { if (!check_var_uid(v)) { return false; } }
    for (auto const &v : fb._arguments) { if (!check_var_uid(v)) { return false; } }
    for (auto const &v : fb._local_variables) { if (!check_var_uid(v)) { return false; } }
    for (auto const &v : fb._shared_variables) { if (!check_var_uid(v)) { return false; } }

    uint32_t vname_count = 0u;
    if (!r.read(vname_count) || vname_count > callable_graph_codec_detail::kMaxTableCount) {
        return false;
    }
    fb._variables_names.resize(vname_count);
    for (uint32_t i = 0u; i < vname_count; ++i) {
        if (!r.read_string(fb._variables_names[i])) { return false; }
    }

    auto read_call_ops = [&](CallOpSet &ops) noexcept {
        const std::byte *bits = nullptr;
        constexpr size_t nbytes = (call_op_count + 7u) / 8u;
        if (!r.read_span(nbytes, bits)) { return false; }
        auto *u8 = reinterpret_cast<const uint8_t *>(bits);
        for (size_t i = 0u; i < call_op_count; ++i) {
            ops._bits[i] = (u8[i / 8u] & (1u << (i & 7u))) != 0u;
        }
        return true;
    };
    if (!read_call_ops(fb._direct_builtin_callables) ||
        !read_call_ops(fb._propagated_builtin_callables)) {
        return false;
    }

    uint32_t callee_count = 0u;
    if (!r.read(callee_count) || callee_count > callable_graph_codec_detail::kMaxTableCount) {
        return false;
    }
    fb._used_custom_callables.clear();
    fb._used_custom_callables.reserve(callee_count);
    for (uint32_t i = 0u; i < callee_count; ++i) {
        uint32_t fid = 0u;
        if (!r.read(fid) || fid >= ctx.functions.size()) { return false; }
        fb._used_custom_callables.emplace(ctx.functions[fid]);
    }

    // Expressions in topological order (1..expr_count stored at index 0..expr_count-1)
    uint32_t expr_count = 0u;
    if (!r.read(expr_count) || expr_count > callable_graph_codec_detail::kMaxTableCount) {
        return false;
    }
    fb._all_expressions.reserve(expr_count);
    for (uint32_t eid = 1u; eid <= expr_count; ++eid) {
        uint32_t tid = 0u, raw_etag = 0u;
        uint64_t ehash = 0u;
        uint8_t raw_usage = 0u;
        if (!r.read(tid) || !r.read(ehash) || !r.read(raw_etag) || !r.read(raw_usage)) {
            return false;
        }
        if (raw_usage > to_underlying(Usage::READ_WRITE)) { return false; }
        const Type *et = nullptr;
        if (!ctx.resolve_nullable_type(tid, et)) { return false; }
        auto etag = static_cast<Expression::Tag>(raw_etag);
        auto resolve_prior = [&](uint32_t cid) noexcept -> const Expression * {
            if (cid == 0u || cid >= eid) { return nullptr; }
            return fb._all_expressions[cid - 1u].get();
        };
        auto finish_expr = [&]<typename E>(luisa::unique_ptr<E> ptr) noexcept {
            ptr->_type = et;
            ptr->_hash = ehash;
            ptr->_builder = &fb;
            ptr->_hash_computed = true;
            ptr->_tag = etag;
            ptr->_usage = static_cast<Usage>(raw_usage);
            fb._all_expressions.emplace_back(std::move(ptr));
        };

        switch (etag) {
            case Expression::Tag::UNARY: {
                uint32_t op_id = 0u;
                UnaryOp op{};
                if (!r.read(op_id) || !r.read(op)) { return false; }
                auto *operand = resolve_prior(op_id);
                if (operand == nullptr) { return false; }
                auto ptr = CallableLibrary::_fg_alloc_expr<UnaryExpr>();
                ptr->_operand = operand;
                ptr->_op = op;
                finish_expr(std::move(ptr));
                break;
            }
            case Expression::Tag::BINARY: {
                uint32_t lid = 0u, rid = 0u;
                BinaryOp op{};
                if (!r.read(lid) || !r.read(rid) || !r.read(op)) { return false; }
                auto *lhs = resolve_prior(lid);
                auto *rhs = resolve_prior(rid);
                if (lhs == nullptr || rhs == nullptr) { return false; }
                auto ptr = CallableLibrary::_fg_alloc_expr<BinaryExpr>();
                ptr->_lhs = lhs;
                ptr->_rhs = rhs;
                ptr->_op = op;
                finish_expr(std::move(ptr));
                break;
            }
            case Expression::Tag::MEMBER: {
                uint32_t sid = 0u, sw_sz = 0u, sw_code = 0u;
                if (!r.read(sid) || !r.read(sw_sz) || !r.read(sw_code)) { return false; }
                auto *self = resolve_prior(sid);
                if (self == nullptr) { return false; }
                auto ptr = CallableLibrary::_fg_alloc_expr<MemberExpr>();
                ptr->_self = self;
                ptr->_swizzle_size = sw_sz;
                ptr->_swizzle_code = sw_code;
                finish_expr(std::move(ptr));
                break;
            }
            case Expression::Tag::ACCESS: {
                uint32_t rng_id = 0u, idx_id = 0u;
                if (!r.read(rng_id) || !r.read(idx_id)) { return false; }
                auto *rng = resolve_prior(rng_id);
                auto *idx = resolve_prior(idx_id);
                if (rng == nullptr || idx == nullptr) { return false; }
                auto ptr = CallableLibrary::_fg_alloc_expr<AccessExpr>();
                ptr->_range = rng;
                ptr->_index = idx;
                finish_expr(std::move(ptr));
                break;
            }
            case Expression::Tag::LITERAL: {
                uint32_t var_idx = 0u, lit_sz = 0u;
                const std::byte *lit_ptr = nullptr;
                if (!r.read(var_idx) || !r.read(lit_sz) || !r.read_span(lit_sz, lit_ptr)) {
                    return false;
                }
                using V = LiteralExpr::Value::variant_type;
                if (var_idx >= luisa::variant_size_v<V>) { return false; }
                auto ptr = CallableLibrary::_fg_alloc_expr<LiteralExpr>();
                bool matched = false;
                auto emplace_lit = [&]<size_t I>(auto &self, std::integral_constant<size_t, I>) noexcept {
                    if constexpr (I < luisa::variant_size_v<V>) {
                        if (I == var_idx) {
                            using T = luisa::variant_alternative_t<I, V>;
                            if (lit_sz == sizeof(T)) {
                                T val;
                                std::memcpy(&val, lit_ptr, sizeof(T));
                                ptr->_value.template emplace<T>(val);
                                matched = true;
                            }
                        } else {
                            self(self, std::integral_constant<size_t, I + 1u>{});
                        }
                    }
                };
                emplace_lit(emplace_lit, std::integral_constant<size_t, 0u>{});
                if (!matched) { return false; }
                finish_expr(std::move(ptr));
                break;
            }
            case Expression::Tag::REF: {
                Variable v;
                if (!read_var(v) || v._uid >= fb._variable_usages.size()) { return false; }
                auto ptr = CallableLibrary::_fg_alloc_expr<RefExpr>();
                ptr->_variable = v;
                finish_expr(std::move(ptr));
                break;
            }
            case Expression::Tag::CONSTANT: {
                uint32_t cid = 0u;
                if (!r.read(cid) || cid >= ctx.constants.size()) { return false; }
                auto ptr = CallableLibrary::_fg_alloc_expr<ConstantExpr>();
                ptr->_data = ctx.constants[cid];
                finish_expr(std::move(ptr));
                break;
            }
            case Expression::Tag::CALL: {
                uint32_t argc = 0u;
                if (!r.read(argc) || argc > callable_graph_codec_detail::kMaxTableCount) {
                    return false;
                }
                CallExpr::ArgumentList args;
                args.reserve(argc);
                for (uint32_t j = 0u; j < argc; ++j) {
                    uint32_t aid = 0u;
                    if (!r.read(aid)) { return false; }
                    auto *aexpr = resolve_prior(aid);
                    if (aexpr == nullptr) { return false; }
                    args.emplace_back(aexpr);
                }
                CallOp op{};
                if (!r.read(op)) { return false; }
                auto ptr = CallableLibrary::_fg_alloc_expr<CallExpr>();
                ptr->_arguments = std::move(args);
                ptr->_op = op;
                if (is_builtin_operation(op)) {
                    uint64_t cbs = 0u;
                    if (!r.read(cbs)) { return false; }
                    ptr->_curve_basis_set = CurveBasisSet::from_u64(cbs);
                    ptr->_func = luisa::monostate{};
                } else {
                    if (op != CallOp::CUSTOM) { return false; }
                    uint8_t has_custom = 0u;
                    if (!r.read(has_custom)) { return false; }
                    if (has_custom == 0u) {
                        ptr->_func = luisa::monostate{};
                    } else if (has_custom == 1u) {
                        uint32_t fid = 0u;
                        if (!r.read(fid) || fid >= ctx.functions.size()) { return false; }
                        ptr->_func = ctx.functions[fid].get();
                    } else {
                        return false;
                    }
                }
                finish_expr(std::move(ptr));
                break;
            }
            case Expression::Tag::CAST: {
                uint32_t sid = 0u;
                CastOp op{};
                if (!r.read(sid) || !r.read(op)) { return false; }
                auto *src = resolve_prior(sid);
                if (src == nullptr) { return false; }
                auto ptr = CallableLibrary::_fg_alloc_expr<CastExpr>();
                ptr->_source = src;
                ptr->_op = op;
                finish_expr(std::move(ptr));
                break;
            }
            case Expression::Tag::TYPE_ID: {
                uint32_t dtid = 0u;
                const Type *dt = nullptr;
                if (!r.read(dtid) || !ctx.resolve_nonnull_type(dtid, dt)) { return false; }
                auto ptr = CallableLibrary::_fg_alloc_expr<TypeIDExpr>();
                ptr->_data_type = dt;
                finish_expr(std::move(ptr));
                break;
            }
            case Expression::Tag::STRING_ID: {
                auto ptr = CallableLibrary::_fg_alloc_expr<StringIDExpr>();
                if (!r.read_string(ptr->_data)) { return false; }
                finish_expr(std::move(ptr));
                break;
            }
            default:
                return false;
        }
    }

    if (!_fg_read_stmt_into(fb._body, fb, ctx) ||
        fb._body._tag != Statement::Tag::SCOPE) {
        return false;
    }
    return true;
}

bool CallableLibrary::deserialize_function_graph(
    luisa::span<const std::byte> binary,
    luisa::vector<luisa::shared_ptr<const detail::FunctionBuilder>> &out_roots,
    luisa::shared_ptr<const detail::FunctionBuilder::XIRCacheProvider> xir_provider,
    luisa::span<const size_t> root_xir_provider_indices) noexcept {

    out_roots.clear();
    GraphReadCtx ctx{binary};
    auto &r = ctx.reader;

    std::array<std::byte, 8u> magic{};
    uint32_t version = 0u;
    uint32_t sizeof_size_t = 0u;
    if (!r.read_bytes(magic.data(), magic.size()) ||
        magic != callable_graph_codec_detail::kGraphMagic ||
        !r.read(version) || version != callable_graph_codec_detail::kGraphVersion ||
        !r.read(sizeof_size_t) || sizeof_size_t != sizeof(size_t)) {
        return false;
    }

    uint32_t type_count = 0u;
    if (!r.read(type_count) || type_count > callable_graph_codec_detail::kMaxTableCount) {
        return false;
    }
    ctx.types.reserve(static_cast<size_t>(type_count) + 1u);
    ctx.types.emplace_back(nullptr);
    for (uint32_t i = 0u; i < type_count; ++i) {
        luisa::string desc;
        if (!r.read_string(desc) || desc.empty() || desc == "void") { return false; }
        auto *t = Type::from(desc);
        if (t == nullptr) { return false; }
        ctx.types.emplace_back(t);
    }

    uint32_t const_count = 0u;
    if (!r.read(const_count) || const_count > callable_graph_codec_detail::kMaxTableCount) {
        return false;
    }
    ctx.constants.reserve(const_count);
    for (uint32_t i = 0u; i < const_count; ++i) {
        uint32_t tid = 0u;
        uint64_t sz = 0u;
        const std::byte *ptr = nullptr;
        const Type *ct = nullptr;
        if (!r.read(tid) || !ctx.resolve_nonnull_type(tid, ct) ||
            !callable_graph_codec_detail::is_valid_constant_type(ct) ||
            !r.read(sz) || sz != ct->size() || !r.read_span(static_cast<size_t>(sz), ptr)) {
            return false;
        }
        ctx.constants.emplace_back(ConstantData::create(ct, ptr, ct->size()));
    }

    uint32_t func_count = 0u, root_count = 0u;
    if (!r.read(func_count) || func_count > callable_graph_codec_detail::kMaxTableCount ||
        !r.read(root_count) || root_count > callable_graph_codec_detail::kMaxTableCount) {
        return false;
    }
    luisa::vector<uint32_t> root_ids(root_count);
    for (uint32_t i = 0u; i < root_count; ++i) {
        if (!r.read(root_ids[i]) || root_ids[i] >= func_count) { return false; }
    }

    ctx.functions.reserve(func_count);
    for (uint32_t i = 0u; i < func_count; ++i) {
        ctx.functions.emplace_back(luisa::make_shared<detail::FunctionBuilder>());
    }
    for (uint32_t i = 0u; i < func_count; ++i) {
        if (!_fg_read_func(*ctx.functions[i], ctx)) { return false; }
    }
    if (r.remaining() != 0u) { return false; }

    if (xir_provider != nullptr) {
        for (size_t i = 0u; i < root_ids.size(); ++i) {
            if (i < root_xir_provider_indices.size() &&
                root_xir_provider_indices[i] != static_cast<size_t>(-1)) {
                auto &fb = ctx.functions[root_ids[i]];
                if (fb->_xir_cache_hash != 0u) {
                    fb->_xir_cache_provider = xir_provider;
                    fb->_xir_cache_provider_index = root_xir_provider_indices[i];
                }
            }
        }
    }

    out_roots.reserve(root_count);
    for (uint32_t rid : root_ids) {
        out_roots.emplace_back(ctx.functions[rid]);
    }
    return true;
}

}// namespace luisa::compute
