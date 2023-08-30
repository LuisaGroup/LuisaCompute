#include <luisa/core/magic_enum.h>
#include <luisa/core/logging.h>
#include <luisa/ast/callable_library.h>

namespace luisa::compute {

template<typename T>
void CallableLibrary::ser_value(T const &t, luisa::vector<std::byte> &vec) noexcept {
    static_assert(std::is_trivially_destructible_v<T> && !std::is_pointer_v<T>);
    auto last_len = vec.size();
    vec.push_back_uninitialized(sizeof(T));
    memcpy(vec.data() + last_len, &t, sizeof(T));
}
template<typename T>
T CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    static_assert(std::is_trivially_destructible_v<T> && !std::is_pointer_v<T>);
    T t;
    memcpy(&t, ptr, sizeof(T));
    ptr += sizeof(T);
    return t;
}
// string: (len: size_t) + (char array)
template<>
void CallableLibrary::ser_value(luisa::string_view const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t.size(), vec);
    auto last_len = vec.size();
    vec.push_back_uninitialized(t.size());
    memcpy(vec.data() + last_len, t.data(), t.size());
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
    memcpy(t.data(), ptr, size);
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
void CallableLibrary::ser_value(luisa::span<const std::byte> const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t.size(), vec);
    auto last_len = vec.size();
    vec.push_back_uninitialized(t.size());
    memcpy(vec.data() + last_len, t.data(), t.size());
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
void CallableLibrary::ser_value(ConstantData const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._type, vec);
    auto last_len = vec.size();
    vec.push_back_uninitialized(t._type->size());
    memcpy(vec.data() + last_len, t._raw, t._type->size());
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
    vec.push_back_uninitialized(byte_arr.size());
    memcpy(vec.data() + last_len, byte_arr.data(), byte_arr.size());
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
    auto index = deser_value<size_t>(ptr, pack);
    auto literal_size = deser_value<size_t>(ptr, pack);
    *reinterpret_cast<size_t *>(&obj->_value) = index;
    memcpy(obj->_value.get_as<std::byte *>(), ptr, literal_size);
    ptr += literal_size;
}
template<>
void CallableLibrary::ser_value(RefExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._variable, vec);
}
template<>
void CallableLibrary::deser_ptr(RefExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    obj->_variable = deser_value<Variable>(ptr, pack);
}
template<>
void CallableLibrary::ser_value(ConstantExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._data, vec);
}
template<>
void CallableLibrary::deser_ptr(ConstantExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
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
    ser_value(t._func.index(), vec);
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
    auto arg_size = deser_value<size_t>(ptr, pack);
    obj->_arguments.push_back_uninitialized(arg_size);
    for (auto &&i : obj->_arguments) {
        i = deser_value<Expression const *>(ptr, pack);
    }
    obj->_op = deser_value<CallOp>(ptr, pack);
    auto index = deser_value<size_t>(ptr, pack);
    if (index == 0) {
        obj->_func = luisa::monostate{};
    } else {
        auto iter = pack.callable_map.find(deser_value<uint64_t>(ptr, pack));
        LUISA_ASSERT(iter != pack.callable_map.end(), "Custom op not found.");
        obj->_func = iter->second.get();
    }
}

template<>
void CallableLibrary::ser_value(CastExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._source, vec);
    ser_value(t._op, vec);
}

template<>
void CallableLibrary::deser_ptr(CastExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    obj->_source = deser_value<Expression const *>(ptr, pack);
    obj->_op = deser_value<CastOp>(ptr, pack);
}

template<>
void CallableLibrary::ser_value(TypeIDExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._data_type, vec);
}

template<>
void CallableLibrary::deser_ptr(TypeIDExpr *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
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
        case Expression::Tag::CPUCUSTOM:
        case Expression::Tag::GPUCUSTOM:
            LUISA_ERROR("Un-supported.");
            break;
    }
}

template<>
Expression const *CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    auto type = deser_value<Type const *>(ptr, pack);
    auto hash = deser_value<uint64_t>(ptr, pack);
    auto tag = deser_value<Expression::Tag>(ptr, pack);
    auto create_expr = [&]<typename T>() {
        auto expr = reinterpret_cast<T *>(luisa::detail::allocator_allocate(sizeof(T), alignof(T)));
        new (expr) T{};
        deser_ptr<T *>(expr, ptr, pack);
        expr->_type = type;
        expr->_hash = hash;
        expr->_hash_computed = true;
        expr->_tag = tag;
        pack.builder->_all_expressions.emplace_back(luisa::unique_ptr<Expression>(expr));
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
    obj->_statements.push_back_uninitialized(size);
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
    ser_value(*t._expr, vec);
    ser_value<Statement>(t._body, vec);
}
template<>
void CallableLibrary::deser_ptr(SwitchCaseStmt *obj, std::byte const *&ptr, DeserPackage &pack) noexcept {
    obj->_expr = deser_value<Expression const *>(ptr, pack);
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
    obj->_comment = deser_value<luisa::string>(ptr, pack);
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
    obj->_lhs = deser_value<Expression const *>(ptr, pack);
    obj->_rhs = deser_value<Expression const *>(ptr, pack);
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
        default:
            break;
    }
}
template<>
Statement *CallableLibrary::deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept {
    auto hash = deser_value<uint64_t>(ptr, pack);
    auto tag = deser_value<Statement::Tag>(ptr, pack);
    auto create_stmt = [&]<typename T, bool construct = true>() {
        auto stmt = reinterpret_cast<T *>(luisa::detail::allocator_allocate(sizeof(T), alignof(T)));
        new (stmt) T{};
        stmt->_hash = hash;
        stmt->_hash_computed = true;
        stmt->_tag = tag;
        auto smt_ptr = luisa::unique_ptr<T>{stmt};
        pack.builder->_all_statements.emplace_back(std::move(smt_ptr));
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
        default:
            return nullptr;
    }
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
    }
}

void CallableLibrary::deserialize_func_builder(detail::FunctionBuilder &builder, std::byte const *&ptr, DeserPackage &pack) noexcept {
    using namespace detail;
    using namespace std::string_view_literals;
    builder._return_type = deser_value<Type const *>(ptr, pack);
    builder._builtin_variables.push_back_uninitialized(deser_value<size_t>(ptr, pack));
    for (auto &&i : builder._builtin_variables) {
        i = deser_value<Variable>(ptr, pack);
    }
    builder._captured_constants.push_back_uninitialized(deser_value<size_t>(ptr, pack));
    for (auto &&i : builder._captured_constants) {
        i = deser_value<ConstantData>(ptr, pack);
    }
    builder._arguments.push_back_uninitialized(deser_value<size_t>(ptr, pack));
    for (auto &&i : builder._arguments) {
        i = deser_value<Variable>(ptr, pack);
    }
    // Note: variant is not trivially copyable. DO NOT USE `push_back_uninitialized`!!!
    builder._bound_arguments.reserve(builder._arguments.size());
    for (auto i = 0u; i < builder._arguments.size(); i++) {
        builder._bound_arguments.emplace_back(luisa::monostate{});
    }
    builder._used_custom_callables.resize(deser_value<size_t>(ptr, pack));
    for (auto &&i : builder._used_custom_callables) {
        auto iter = pack.callable_map.find(deser_value<uint64_t>(ptr, pack));
        LUISA_ASSERT(iter != pack.callable_map.end(), "Illegal bin-data.");
        i = iter->second;
    }
    builder._local_variables.push_back_uninitialized(deser_value<size_t>(ptr, pack));
    for (auto &&i : builder._local_variables) {
        i = deser_value<Variable>(ptr, pack);
    }
    builder._shared_variables.push_back_uninitialized(deser_value<size_t>(ptr, pack));
    for (auto &&i : builder._shared_variables) {
        i = deser_value<Variable>(ptr, pack);
    }
    size_t variable_usage_size = deser_value<size_t>(ptr, pack) / sizeof(Usage);
    builder._variable_usages.push_back_uninitialized(variable_usage_size);
    memcpy(builder._variable_usages.data(), ptr, builder._variable_usages.size_bytes());
    ptr += builder._variable_usages.size_bytes();
    builder._direct_builtin_callables = deser_value<CallOpSet>(ptr, pack);
    builder._propagated_builtin_callables = deser_value<CallOpSet>(ptr, pack);
    builder._tag = deser_value<Function::Tag>(ptr, pack);
    builder._requires_atomic_float = deser_value<bool>(ptr, pack);
    deser_ptr<Statement *>(&builder._body, ptr, pack);
}
void CallableLibrary::serialize_func_builder(detail::FunctionBuilder const &builder, luisa::vector<std::byte> &vec) noexcept {
    using namespace detail;
    using namespace std::string_view_literals;
    LUISA_ASSERT(builder.tag() == Function::Tag::CALLABLE, "Only callable can be serialized.");
    for (auto &&i : builder._bound_arguments) {
        LUISA_ASSERT(luisa::holds_alternative<luisa::monostate>(i),
                     "Callable cannot contain bound-argument.");
    }
    LUISA_ASSERT(builder._used_external_functions.empty(), "Callable cannot contain external-function.");
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
    auto before_size = vec.size();
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
    ser_value(luisa::span<const std::byte>{reinterpret_cast<const std::byte *>(builder._variable_usages.data()), builder._variable_usages.size_bytes()}, vec);
    // direct builtin callables
    ser_value(builder._direct_builtin_callables, vec);
    // propagated builtin callables
    ser_value(builder._propagated_builtin_callables, vec);
    // tag
    ser_value(builder._tag, vec);
    // requires_atomic_float
    ser_value(builder._requires_atomic_float, vec);
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
void CallableLibrary::add_callable(luisa::string_view name, luisa::shared_ptr<const detail::FunctionBuilder> callable) noexcept {
    _callables.try_emplace(name, std::move(callable));
}
CallableLibrary::~CallableLibrary() noexcept = default;
CallableLibrary::CallableLibrary(CallableLibrary &&) noexcept = default;
luisa::vector<luisa::string_view> CallableLibrary::names() const noexcept {
    luisa::vector<luisa::string_view> vec;
    vec.reserve(_callables.size());
    for (auto &&i : _callables) {
        vec.emplace_back(i.first);
    }
    return vec;
}
}// namespace luisa::compute
