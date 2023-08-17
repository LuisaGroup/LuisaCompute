#include <luisa/ast/callable_library.h>
#include <luisa/core/logging.h>
namespace luisa::compute {
namespace detail {
struct CallableHeader {
    size_t name_len;
    size_t bin_len;
};
struct InlineCallableHeader {
    size_t hash;
    size_t bin_len;
};
}// namespace detail
template<typename T>
void CallableLibrary::ser_value(T const &t, luisa::vector<std::byte> &vec) noexcept {
    static_assert(std::is_trivially_destructible_v<T> && !std::is_pointer_v<T>);
    auto last_len = vec.size();
    vec.push_back_uninitialized(sizeof(T));
    memcpy(vec.data() + last_len, &t, sizeof(T));
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
void CallableLibrary::ser_value(Type const *const &t, luisa::vector<std::byte> &vec) noexcept {
    using namespace std::string_view_literals;
    if (t) {
        ser_value(t->description(), vec);
    } else {
        ser_value("void"sv, vec);
    }
}
template<>
void CallableLibrary::ser_value(luisa::string const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(luisa::string_view{t}, vec);
}
template<>
void CallableLibrary::ser_value(luisa::span<const std::byte> const &t, luisa::vector<std::byte> &vec) noexcept {
    auto last_len = vec.size();
    vec.push_back_uninitialized(t.size());
    memcpy(vec.data() + last_len, t.data(), t.size());
}
template<>
void CallableLibrary::ser_value(Variable const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t.type()->description(), vec);
    ser_value(t.uid(), vec);
    ser_value(t.tag(), vec);
}
template<>
void CallableLibrary::ser_value(ConstantData const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._type->description(), vec);
    ser_value(luisa::span<const std::byte>{t._raw, t._type->size()}, vec);
    ser_value(t._hash, vec);
}
template<>
void CallableLibrary::ser_value(CallOpSet const &t, luisa::vector<std::byte> &vec) noexcept {
    std::array<uint8_t, (call_op_count + 7) / 8> byte_arr{};
    for (size_t i = 0; i < call_op_count; ++i) {
        auto &v = byte_arr[i / 8];
        v |= ((t._bits[i] ? 1 : 0) << (i & 7));
    }
    ser_value(luisa::span<const std::byte>{reinterpret_cast<const std::byte *>(byte_arr.data()), byte_arr.size()}, vec);
}
// template<>
// void CallableLibrary::ser_value(UnaryExpr const &t, luisa::vector<std::byte> &vec) noexcept;
// template<>
// void CallableLibrary::ser_value(BinaryExpr const &t, luisa::vector<std::byte> &vec) noexcept;
// template<>
// void CallableLibrary::ser_value(MemberExpr const &t, luisa::vector<std::byte> &vec) noexcept;
// template<>
// void CallableLibrary::ser_value(RefExpr const &t, luisa::vector<std::byte> &vec) noexcept;
// template<>
// void CallableLibrary::ser_value(ConstantExpr const &t, luisa::vector<std::byte> &vec) noexcept;
// template<>
// void CallableLibrary::ser_value(CallExpr const &t, luisa::vector<std::byte> &vec) noexcept;
// template<>
// void CallableLibrary::ser_value(CastExpr const &t, luisa::vector<std::byte> &vec) noexcept;
template<>
void CallableLibrary::ser_value(Expression const &t, luisa::vector<std::byte> &vec) noexcept;
template<>
void CallableLibrary::ser_value(UnaryExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._operand, vec);
    ser_value(t._op, vec);
}
template<>
void CallableLibrary::ser_value(BinaryExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._lhs, vec);
    ser_value(*t._rhs, vec);
    ser_value(t._op, vec);
}
template<>
void CallableLibrary::ser_value(AccessExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._range, vec);
    ser_value(*t._index, vec);
}
template<>
void CallableLibrary::ser_value(MemberExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._self, vec);
    ser_value(t._swizzle_size, vec);
    ser_value(t._swizzle_code, vec);
}
template<>
void CallableLibrary::ser_value(LiteralExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._value.index(), vec);
    luisa::visit(
        [&]<typename T>(T const &t) {
            ser_value(t, vec);
        },
        t._value);
}
template<>
void CallableLibrary::ser_value(RefExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._variable, vec);
}
template<>
void CallableLibrary::ser_value(ConstantExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._data, vec);
}
template<>
void CallableLibrary::ser_value(CallExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._arguments.size(), vec);
    for (auto &&i : t._arguments) {
        ser_value(*i, vec);
    }
    ser_value(t._op, vec);
    LUISA_ASSERT(t._func.index() != 2, "Callable cannot contain external");
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
void CallableLibrary::ser_value(CastExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._source, vec);
    ser_value(t._op, vec);
}
template<>
void CallableLibrary::ser_value(TypeIDExpr const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._data_type, vec);
}
template<>
void CallableLibrary::ser_value(Expression const &t, luisa::vector<std::byte> &vec) noexcept {
    using namespace std::string_view_literals;
    if (&t == nullptr) {
        ser_value("null"sv, vec);
        return;
    }
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
void CallableLibrary::ser_value(Statement const &t, luisa::vector<std::byte> &vec) noexcept;
template<>
void CallableLibrary::ser_value(ReturnStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._expr, vec);
}
template<>
void CallableLibrary::ser_value(ScopeStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._statements.size(), vec);
    for (auto &&i : t._statements) {
        ser_value(*i, vec);
    }
}
template<>
void CallableLibrary::ser_value(IfStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._condition, vec);
    ser_value(t._true_branch, vec);
    ser_value(t._false_branch, vec);
}
template<>
void CallableLibrary::ser_value(LoopStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._body, vec);
}
template<>
void CallableLibrary::ser_value(ExprStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._expr, vec);
}
template<>
void CallableLibrary::ser_value(SwitchStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._expr, vec);
    ser_value(t._body, vec);
}
template<>
void CallableLibrary::ser_value(SwitchCaseStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._expr, vec);
    ser_value(t._body, vec);
}
template<>
void CallableLibrary::ser_value(SwitchDefaultStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._body, vec);
}
template<>
void CallableLibrary::ser_value(ForStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._var, vec);
    ser_value(*t._cond, vec);
    ser_value(*t._step, vec);
    ser_value(t._body, vec);
}
template<>
void CallableLibrary::ser_value(CommentStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._comment, vec);
}
template<>
void CallableLibrary::ser_value(AutoDiffStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t._body, vec);
}
template<>
void CallableLibrary::ser_value(RayQueryStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*static_cast<Expression const *>(t._query), vec);
    ser_value(t._on_triangle_candidate, vec);
    ser_value(t._on_procedural_candidate, vec);
}
template<>
void CallableLibrary::ser_value(AssignStmt const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(*t._lhs, vec);
    ser_value(*t._rhs, vec);
}

template<>
void CallableLibrary::ser_value(Statement const &t, luisa::vector<std::byte> &vec) noexcept {
    using namespace std::string_view_literals;
    if (&t == nullptr) {
        ser_value("null"sv, vec);
        return;
    }
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
    }
}
void CallableLibrary::serialize_func_builder(detail::FunctionBuilder const &builder, luisa::vector<std::byte> &vec) noexcept {
    using namespace detail;
    using namespace std::string_view_literals;
    LUISA_ASSERT(builder.tag() == Function::Tag::CALLABLE, "Only callable can be serialized.");
    for (auto &&i : builder._bound_arguments) {
        if (i.index() != 0) [[unlikely]] {
            LUISA_ERROR("Callable cannot contain bound-argument.");
        }
    }
    LUISA_ASSERT(builder._used_external_functions.empty(), "Callable cannot contain external-function.");
    // return type
    ser_value(builder._return_type, vec);
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
    // local vars
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
    // hash
    ser_value(builder._hash, vec);
    // block_size
    ser_value(builder._block_size, vec);
    // tag
    ser_value(builder._tag, vec);
    // requires_atomic_float
    ser_value(builder._requires_atomic_float, vec);
    // body
    ser_value(static_cast<Statement const &>(builder._body), vec);
}
CallableLibrary::CallableLibrary() noexcept = default;
CallableLibrary CallableLibrary::load(luisa::span<const std::byte> binary) noexcept {
    CallableLibrary lib;
    // TODO
    return lib;
}
luisa::vector<std::byte> CallableLibrary::serialize() const noexcept {
    luisa::unordered_map<size_t, luisa::shared_ptr<const detail::FunctionBuilder>> _inline_callables;
    luisa::vector<std::byte> vec;
    for (auto &&i : _callables) {
        for (auto &&j : i.second->_used_custom_callables) {
            _inline_callables.try_emplace(j->hash(), j);
        }
    }
    for(auto&& i : _callables){
        _inline_callables.erase(i.second->hash());
    }
    // Callables
    ser_value(_callables.size(), vec);
    for (auto &&i : _callables) {
        auto last_len = vec.size();
        vec.push_back_uninitialized(sizeof(detail::CallableHeader));
        detail::CallableHeader header{};
        header.name_len = i.first.length();
        auto name_start = vec.size();
        vec.push_back_uninitialized(header.name_len);
        auto before_ser = vec.size();
        serialize_func_builder(*i.second, vec);
        header.bin_len = vec.size() - before_ser;
        memcpy(vec.data() + last_len, &header, sizeof(detail::CallableHeader));
        memcpy(vec.data() + name_start, i.first.data(), i.first.length());
    }
    // Inline callables
    ser_value(_inline_callables.size(), vec);
    for (auto &&i : _inline_callables) {
        auto last_len = vec.size();
        vec.push_back_uninitialized(sizeof(detail::InlineCallableHeader));
        detail::InlineCallableHeader header{};
        header.hash = i.first;
        auto before_ser = vec.size();
        serialize_func_builder(*i.second, vec);
        header.bin_len = vec.size() - before_ser;
        memcpy(vec.data() + last_len, &header, sizeof(detail::InlineCallableHeader));
    }
    return vec;
}
void CallableLibrary::add_callable(luisa::string_view name, luisa::shared_ptr<const detail::FunctionBuilder> callable) noexcept {
    _callables.try_emplace(name, std::move(callable));
}
CallableLibrary::~CallableLibrary() noexcept {}
CallableLibrary::CallableLibrary(CallableLibrary &&) noexcept = default;
}// namespace luisa::compute