#pragma vengine_package serialize

#include <serialize/config.h>
#include <serialize/serialize.h>
#include <vstl/variant_util.h>

namespace luisa::compute {
using namespace toolhub::db;
template<typename T>
struct BasicType;
template<>
struct BasicType<float> {
    using Type = double;
};
template<>
struct BasicType<int> {
    using Type = int64;
};
template<>
struct BasicType<uint> {
    using Type = int64;
};
template<>
struct BasicType<bool> {
    using Type = bool;
};
template<typename T>
using BasicType_t = typename BasicType<T>::Type;
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(Expression const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("hash", t._hash);
    if (t.type())
        r->Set("type", t.type()->_hash);
    r->Set("tag", static_cast<int64>(t.tag()));
    r->Set("usage", static_cast<int64>(t.usage()));
    return r;
}

vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(UnaryExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("operand", t.operand()->_hash);
    r->Set("op", static_cast<int64>(t.op()));
    return r;
}
void AstSerializer::DeSerialize(UnaryExpr &t, IJsonDict *r, DeserVisitor const &evt) {
    t._op = static_cast<UnaryOp>(r->Get("op").get_or<int64>(0));
    t._operand = evt.GetExpr(r->Get("operand").get_or<int64>(0));
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(BinaryExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("lhs", t.lhs()->_hash);
    r->Set("rhs", t.rhs()->_hash);
    r->Set("op", static_cast<int64>(t.op()));
    return r;
}
void AstSerializer::DeSerialize(BinaryExpr &t, IJsonDict *r, DeserVisitor const &evt) {
    t._lhs = evt.GetExpr(r->Get("lhs").get_or<int64>(0));
    t._rhs = evt.GetExpr(r->Get("rhs").get_or<int64>(0));
    t._op = static_cast<BinaryOp>(r->Get("op").get_or<int64>(0));
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(AccessExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("range", t.range()->_hash);
    r->Set("index", t.index()->_hash);
    return r;
}
void AstSerializer::DeSerialize(AccessExpr &t, IJsonDict *r, DeserVisitor const &evt) {
    t._range = evt.GetExpr(r->Get("range").get_or<int64>(0));
    t._index = evt.GetExpr(r->Get("index").get_or<int64>(0));
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(MemberExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("self", t.self()->_hash);
    r->Set("member", t._member);
    return r;
}
void AstSerializer::DeSerialize(MemberExpr &t, IJsonDict *r, DeserVisitor const &evt) {
    t._self = evt.GetExpr(r->Get("self").get_or<int64>(0));
    t._member = r->Get("member").get_or<int64>(0);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(LiteralExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("value", Serialize(t._value, db));
    return r;
}
void AstSerializer::DeSerialize(LiteralExpr &t, IJsonDict *r, DeserVisitor const &evt) {
    auto value = r->Get("value").try_get<IJsonDict *>();
    if (value) {
        DeSerialize(t._value, *value);
    }
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(Variable const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    if (t.type())
        r->Set("type", t.type()->_hash);
    r->Set("uid", t.uid());
    r->Set("tag", static_cast<int64>(t.tag()));
    return r;
}
void AstSerializer::DeSerialize(Variable &t, IJsonDict *r) {
    auto type = r->Get("type").try_get<int64>();
    if (type)
        t._type = Type::get_type(*type);
    else
        t._type = nullptr;
    t._tag = static_cast<Variable::Tag>(r->Get("tag").get_or<int64>(0));
    t._uid = r->Get("uid").get_or<int64>(0);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(RefExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("variable", Serialize(t._variable, db));
    return r;
}
void AstSerializer::DeSerialize(RefExpr &t, IJsonDict *r, DeserVisitor const &evt) {
    auto dd = r->Get("variable").try_get<IJsonDict *>();
    if (dd) {
        DeSerialize(t._variable, *dd);
    }
}
template<typename T>
struct SerArrayVisitor {
    void SerArray(
        IJsonArray &r,
        luisa::span<T const> a) const {
        for (auto &&i : a) {
            r << static_cast<BasicType_t<T>>(i);
        }
    }
    void SerValue(
        IJsonDict *dict,
        IJsonDatabase *db,
        T const &a) const {
        dict->Set("value", static_cast<BasicType_t<T>>(a));
    }
};
template<typename T, size_t n>
struct SerArrayVisitor<luisa::Vector<T, n>> {
    using Type = luisa::Vector<T, n>;
    void SerArray(
        IJsonArray &r,
        luisa::span<Type const> a) const {
        for (auto &&i : a) {
            auto ptr = reinterpret_cast<T const *>(&i);
            for (auto id : vstd::range(n)) {
                r << static_cast<BasicType_t<T>>(ptr[id]);
            }
        }
    }
    void SerValue(
        IJsonDict *dict,
        IJsonDatabase *db,
        Type const &a) const {
        auto arr = db->CreateArray();
        arr->Reserve(n);
        auto ptr = reinterpret_cast<T const *>(&a);
        for (auto id : vstd::range(n)) {
            (*arr) << static_cast<BasicType_t<T>>(ptr[id]);
        }
        dict->Set("value", std::move(arr));
    }
};
template<size_t n>
struct SerArrayVisitor<luisa::Matrix<n>> {
    using Type = luisa::Matrix<n>;
    void SerArray(
        IJsonArray &r,
        luisa::span<Type const> a) const {
        for (auto &&i : a) {
            auto ptr = reinterpret_cast<float const *>(&i);
            for (auto id : vstd::range(n * n)) {
                r << static_cast<double>(ptr[id]);
            }
        }
    }
    void SerValue(
        IJsonDict *dict,
        IJsonDatabase *db,
        Type const &a) const {
        auto arr = db->CreateArray();
        arr->Reserve(n * n);
        auto ptr = reinterpret_cast<float const *>(&a);
        for (auto id : vstd::range(n * n)) {
            (*arr) << static_cast<double>(ptr[id]);
        }
        dict->Set("value", std::move(arr));
    }
};
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(ConstantData const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    auto &&view = t.view();
    r->Set("view_type", view.index());
    r->Set("hash", t._hash);
    auto arr = db->CreateArray();
    luisa::visit(
        [&]<typename T>(T const &t) {
            SerArrayVisitor<std::remove_cvref_t<typename T::element_type>>().SerArray(*arr, t);
        },
        view);
    r->Set("values", std::move(arr));
    return r;
}

template<typename T>
struct DeserArray {
    template<typename Func>
    void operator()(
        IJsonArray &arr,
        DeserVisitor const &evt,
        Func &&setView) const {
        size_t sz = arr.Length() * sizeof(T);
        T *ptr = (T *)evt.Allocate(sz);

        setView(luisa::span<T const>(ptr, arr.Length()));
        for (auto &&i : arr) {
            *ptr = i.get_or<BasicType_t<T>>(0);
            ptr++;
        }
    }
};
template<typename T, size_t n>
struct DeserArray<luisa::Vector<T, n>> {
    template<typename Func>
    void operator()(
        IJsonArray &arr,
        DeserVisitor const &evt,
        Func &&setView) const {
        size_t sz = arr.Length() * sizeof(T);
        T *ptr = (T *)evt.Allocate(sz);
        setView(luisa::span<luisa::Vector<T, n> const>((luisa::Vector<T, n> *)ptr, arr.Length() / n));
        for (auto &&i : arr) {
            *ptr = i.get_or<BasicType_t<T>>(0);
            ptr++;
        }
    }
};
template<size_t n>
struct DeserArray<luisa::Matrix<n>> {
    template<typename Func>
    void operator()(
        IJsonArray &arr,
        DeserVisitor const &evt,
        Func &&setView) const {
        size_t sz = arr.Length() * sizeof(float);
        float *ptr = (float *)evt.Allocate(sz);
        setView(luisa::span<luisa::Matrix<n> const>((luisa::Matrix<n> *)ptr, arr.Length() / (n * n)));
        for (auto &&i : arr) {
            *ptr = i.get_or<double>(0);
            ptr++;
        }
    }
};
void AstSerializer::DeSerialize(ConstantData &t, IJsonDict *r, DeserVisitor const &evt) {
    t._hash = r->Get("hash").get_or<int64>(0);
    auto arrOpt = r->Get("values").try_get<IJsonArray *>();
    auto type = r->Get("view_type").get_or(std::numeric_limits<int64>::max());
    if (arrOpt) {
        auto &&arr = **arrOpt;
        vstd::VariantVisitor_t<basic_types>()(
            [&]<typename T>() {
                auto setFunc = [&](auto &&v) {
                    t._view = v;
                };
                DeserArray<T>().template operator()<decltype(setFunc)>(
                    arr,
                    evt,
                    std::move(setFunc));
            },
            type);
    }
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(LiteralExpr::Value const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    luisa::visit(
        [&]<typename T>(T const &t) {
            if constexpr (!std::is_same_v<T, LiteralExpr::MetaValue>) {
                SerArrayVisitor<T>().SerValue(r.get(), db, t);
            }
        },
        t);
    r->Set("value_type", t.index());
    return r;
}
template<typename T>
struct DeserLiteral {
    void operator()(
        IJsonDict *r,
        LiteralExpr::Value &t) const {
        t = static_cast<T>(r->Get("value").get_or<BasicType_t<T>>(0));
    }
};
template<typename T, size_t n>
struct DeserLiteral<luisa::Vector<T, n>> {
    void operator()(
        IJsonDict *r,
        LiteralExpr::Value &t) const {
        auto arr = r->Get("value").get_or<IJsonArray *>(nullptr);
        if (!arr || arr->Length() < n) return;
        luisa::Vector<T, n> vec;
        T *vecPtr = reinterpret_cast<T *>(&vec);
        for (auto i : vstd::range(n)) {
            vecPtr[i] = arr->Get(i).get_or<BasicType_t<T>>(0);
        }
        t = vec;
    }
};

template<size_t n>
struct DeserLiteral<luisa::Matrix<n>> {
    void operator()(
        IJsonDict *r,
        LiteralExpr::Value &t) const {
        auto arr = r->Get("value").get_or<IJsonArray *>(nullptr);
        if (!arr || arr->Length() < (n * n)) return;
        luisa::Matrix<n> vec;
        float *vecPtr = reinterpret_cast<float *>(&vec);
        for (auto i : vstd::range(n * n)) {
            vecPtr[i] = arr->Get(i).get_or<double>(0);
        }
        t = vec;
    }
};
void AstSerializer::DeSerialize(LiteralExpr::Value &t, IJsonDict *r) {
    auto type = r->Get("value_type").try_get<int64>();
    if (!type)
        return;
    vstd::VariantVisitor_t<basic_types>()(
        [&]<typename T>() {
            DeserLiteral<T>()(
                r,
                t);
        },
        type);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(ConstantExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("data"sv, Serialize(t._data, db));
    r->Set("expr"sv, Serialize(static_cast<Expression const &>(t), db));
    return r;
}
void AstSerializer::DeSerialize(ConstantExpr &t, IJsonDict *r, DeserVisitor const &evt) {
    auto data = r->Get("data").get_or<IJsonDict *>(nullptr);
    if (!data) return;
    DeSerialize(t._data, data, evt);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(CallExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr"sv, Serialize(static_cast<Expression const &>(t), db));
    if (t._op == CallOp::CUSTOM) {
        r->Set("custom", t._custom.hash());
    } else {
        r->Set("op", (int64)t._op);
    }
    return r;
}
void AstSerializer::DeSerialize(CallExpr &t, IJsonDict *r, DeserVisitor const &evt) {
    auto customHash = r->Get("custom"sv).try_get<int64>();
    if (customHash) {
        t._custom = evt.GetFunction(*customHash);
        t._op = CallOp::CUSTOM;
    }
    // Call OP
    else {
        t._op = (CallOp)r->Get("op").get_or<int64>(0);
    }
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(CastExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr"sv, Serialize(static_cast<Expression const &>(t), db));
    r->Set("src"sv, t._source->_hash);
    r->Set("op"sv, (int64)t._op);
    return r;
}
void AstSerializer::DeSerialize(CastExpr &t, IJsonDict *r, DeserVisitor const &evt) {
    auto src = r->Get("src"sv).try_get<int64>();
    if (!src) return;
    t._source = evt.GetExpr(src);
    t._op = (CastOp)r->Get("op"sv).get_or<int64>(0);
}
template<typename Func>
bool ExecuteFromExprTag(Expression::Tag tag, Func &&func) {
    switch (tag) {
        case Expression::Tag::UNARY:
            func.template operator()<UnaryExpr>();
            break;
        case Expression::Tag::BINARY:
            func.template operator()<BinaryExpr>();
            break;
        case Expression::Tag::MEMBER:
            func.template operator()<MemberExpr>();
            break;
        case Expression::Tag::ACCESS:
            func.template operator()<AccessExpr>();
            break;
        case Expression::Tag::LITERAL:
            func.template operator()<LiteralExpr>();
            break;
        case Expression::Tag::REF:
            func.template operator()<RefExpr>();
            break;
        case Expression::Tag::CONSTANT:
            func.template operator()<ConstantExpr>();
            break;
        case Expression::Tag::CALL:
            func.template operator()<CallExpr>();
            break;
        case Expression::Tag::CAST:
            func.template operator()<CastExpr>();
            break;
        default: return false;
    }
    return true;
}
template<typename Func>
bool ExecuteFromStmtTag(Statement::Tag tag, Func &&func) {
    switch (tag) {
        case Statement::Tag::BREAK:
            func.template operator()<BreakStmt>();
            break;
        case Statement::Tag::CONTINUE:
            func.template operator()<ContinueStmt>();
            break;
        case Statement::Tag::RETURN:
            func.template operator()<ReturnStmt>();
            break;
        case Statement::Tag::SCOPE:
            func.template operator()<ScopeStmt>();
            break;
        case Statement::Tag::IF:
            func.template operator()<IfStmt>();
            break;
        case Statement::Tag::LOOP:
            func.template operator()<LoopStmt>();
            break;
        case Statement::Tag::EXPR:
            func.template operator()<ExprStmt>();
            break;
        case Statement::Tag::SWITCH:
            func.template operator()<SwitchStmt>();
            break;
        case Statement::Tag::SWITCH_CASE:
            func.template operator()<SwitchCaseStmt>();
            break;
        case Statement::Tag::SWITCH_DEFAULT:
            func.template operator()<SwitchDefaultStmt>();
            break;
        case Statement::Tag::ASSIGN:
            func.template operator()<AssignStmt>();
            break;
        case Statement::Tag::FOR:
            func.template operator()<ForStmt>();
            break;
        case Statement::Tag::COMMENT:
            func.template operator()<CommentStmt>();
            break;
        case Statement::Tag::META:
            func.template operator()<MetaStmt>();
            break;
        default:
            return false;
    }
    return true;
}
Expression *AstSerializer::GenExpr(IJsonDict *dict, DeserVisitor &evt) {
    Expression *t;
    auto r = dict->Get("expr").get_or<IJsonDict *>(nullptr);
    if (!r) return nullptr;
    auto tag = r->Get("tag").try_get<int64>();
    if (!tag) return nullptr;
    auto func = [&]<typename T> {
        auto f = reinterpret_cast<T *>(evt.Allocate(sizeof(T)));
        t = f;
        t->_hash = r->Get("hash").get_or<int64>(0);
        t->_hash_computed = true;
        auto type = r->Get("type").try_get<int64>();
        t->_type = type ? Type::get_type(*type) : nullptr;
        t->_usage = static_cast<Usage>(r->Get("usage").get_or<int64>(0));
        t->_tag = static_cast<Expression::Tag>(*tag);
    };
    if (!ExecuteFromExprTag(static_cast<Expression::Tag>(*tag), func)) return nullptr;
    return t;
}

void AstSerializer::DeserExpr(IJsonDict *dict, Expression *expr, DeserVisitor &evt) {
    auto func = [&]<typename T> {
        DeSerialize(*static_cast<T *>(expr), dict, evt);
    };
    ExecuteFromExprTag(expr->_tag, func);
}
vstd::unique_ptr<IJsonDict> AstSerializer::SerExpr(IJsonDatabase *db, Expression const &expr) {
    vstd::unique_ptr<IJsonDict> dd;
    auto func = [&]<typename T> {
        T const &t = static_cast<T const &>(expr);
        dd = Serialize(t, db);
    };
    if (!ExecuteFromExprTag(expr._tag, func)) return nullptr;
    return dd;
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(Statement const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("hash", s._hash);
    r->Set("tag", (int64)s._tag);
    return r;
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(BreakStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    return r;
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(ContinueStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    return r;
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(ReturnStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    r->Set("expr", s._expr->_hash);
    return r;
}
void AstSerializer::DeSerialize(ReturnStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    auto v = r->Get("expr").try_get<int64>();
    if (v)
        s._expr = evt.GetExpr(*v);
}
void AstSerializer::Serialize(ScopeStmt const &s, IJsonDict *r, IJsonDatabase *db) {
    auto arr = db->CreateArray();
    for (auto &&i : s._statements) {
        arr->Add(i->_hash);
    }
    r->Set("scope"sv, std::move(arr));
}

vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(ScopeStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    Serialize(s, r.get(), db);
    return r;
}
void AstSerializer::DeSerialize(ScopeStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    auto arrPtr = r->Get("scope"sv).try_get<IJsonArray *>();
    if (!arrPtr) return;
    auto arr = *arrPtr;
    s._statements.reserve(arr->Length());
    for (auto &&i : *arr) {
        auto v = i.try_get<int64>();
        if (!v) continue;
        s._statements.emplace_back(evt.GetStmt(*v));
    }
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(IfStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    r->Set("true", Serialize(s._true_branch, db));
    r->Set("false", Serialize(s._false_branch, db));
    return r;
}
void AstSerializer::DeSerialize(IfStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    auto ts = r->Get("true").get_or<IJsonDict *>(nullptr);
    auto fs = r->Get("false").get_or<IJsonDict *>(nullptr);
    if (!ts || !fs) return;
    DeSerialize(s._true_branch, ts, evt);
    DeSerialize(s._false_branch, fs, evt);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(LoopStmt const &s, IJsonDatabase *db) {
    return Serialize(s._body, db);
}
void AstSerializer::DeSerialize(LoopStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    DeSerialize(s._body, r, evt);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(ExprStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    r->Set("expr", s._expr->_hash);
    return r;
}
void AstSerializer::DeSerialize(ExprStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    auto exprHash = r->Get("expr").try_get<int64>();
    if (!exprHash) return;
    s._expr = evt.GetExpr(*exprHash);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(SwitchStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    r->Set("expr", s._expr->_hash);
    Serialize(s._body, r.get(), db);
    return r;
}
void AstSerializer::DeSerialize(SwitchStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    auto exprHash = r->Get("expr").try_get<int64>();
    if (!exprHash) return;
    s._expr = evt.GetExpr(*exprHash);
    DeSerialize(s._body, r, evt);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(SwitchCaseStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    r->Set("expr", s._expr->_hash);
    Serialize(s._body, r.get(), db);
    return r;
}
void AstSerializer::DeSerialize(SwitchCaseStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    auto exprHash = r->Get("expr").try_get<int64>();
    if (!exprHash) return;
    s._expr = evt.GetExpr(*exprHash);
    DeSerialize(s._body, r, evt);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(SwitchDefaultStmt const &s, IJsonDatabase *db) {
    return Serialize(s._body, db);
}
void AstSerializer::DeSerialize(SwitchDefaultStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    DeSerialize(s._body, r, evt);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(AssignStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    r->Set("lhs", s._lhs->_hash);
    r->Set("rhs", s._rhs->_hash);
    r->Set("op", (int64)s._op);
    return r;
}
void AstSerializer::DeSerialize(AssignStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    auto lhs = r->Get("lhs").try_get<int64>();
    auto rhs = r->Get("rhs").try_get<int64>();
    auto op = r->Get("op").try_get<int64>();
    if (!lhs || !rhs || !op) return;
    s._lhs = evt.GetExpr(*lhs);
    s._rhs = evt.GetExpr(*rhs);
    s._op = (AssignOp)*op;
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(ForStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    r->Set("var", s._var->_hash);
    r->Set("cond", s._cond->_hash);
    r->Set("step", s._step->_hash);
    Serialize(s._body, r.get(), db);
    return r;
}
void AstSerializer::DeSerialize(ForStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    auto set = [&](auto name, auto &&ref) {
        auto h = r->Get(name).template try_get<int64>();
        if (!h) return;
        ref = evt.GetExpr(*h);
    };
    set("var", s._var);
    set("cond", s._cond);
    set("step", s._step);
    DeSerialize(s._body, r, evt);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(CommentStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    r->Set("comment", s._comment);
    return r;
}
void AstSerializer::DeSerialize(CommentStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    s._comment = r->Get("comment").get_or<std::string_view>(std::string_view(nullptr, 0));
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(MetaStmt const &s, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("stmt", Serialize(static_cast<Statement const &>(s), db));
    r->Set("comment", s._info);
    Serialize(s._scope, r.get(), db);
    auto childArr = db->CreateArray();
    auto varArr = db->CreateArray();
    for (auto &&i : s._children) {
        childArr->Add(i->_hash);
    }
    for (auto &&i : s._variables) {
        varArr->Add(Serialize(i, db));
    }
    r->Set("child", std::move(childArr));
    r->Set("var", std::move(varArr));
    return r;
}
void AstSerializer::DeSerialize(MetaStmt &s, IJsonDict *r, DeserVisitor const &evt) {
    s._info = r->Get("comment").get_or<std::string_view>(std::string_view(nullptr, 0));
    DeSerialize(s._scope, r, evt);
    auto childArr = r->Get("child").get_or<IJsonArray *>(nullptr);
    auto varArr = r->Get("var").get_or<IJsonArray *>(nullptr);
    //TODO
}
Statement *AstSerializer::GenStmt(IJsonDict *dict, DeserVisitor &evt) {
    Statement *t;
    auto r = dict->Get("expr").get_or<IJsonDict *>(nullptr);
    if (!r) return nullptr;
    auto tag = r->Get("tag").try_get<int64>();
    if (!tag) return nullptr;
    auto func = [&]<typename T> {
        auto f = reinterpret_cast<T *>(evt.Allocate(sizeof(T)));
        t = f;
        t->_hash = r->Get("hash").get_or<int64>(0);
        t->_hash_computed = true;
        t->_tag = static_cast<Statement::Tag>(*tag);
    };
    if (!ExecuteFromStmtTag(static_cast<Statement::Tag>(*tag), func)) return nullptr;
    return t;
}
void AstSerializer::DeserStmt(IJsonDict *dict, Statement *t, DeserVisitor &evt) {
    auto func = [&]<typename T> {
        DeSerialize(*static_cast<T *>(t), dict, evt);
    };
    ExecuteFromStmtTag(t->_tag, func);
}

vstd::unique_ptr<IJsonDict> AstSerializer::SerStmt(IJsonDatabase *db, Statement const &s) {
    vstd::unique_ptr<IJsonDict> dict;
    auto func = [&]<typename T> {
        dict = Serialize(static_cast<T const &>(s), db);
    };
    if (!ExecuteFromStmtTag(s._tag, func)) return nullptr;
    return dict;
}

DeserVisitor::DeserVisitor(
    Function kernel,
    IJsonArray *exprArr,
    IJsonArray *stmtArr) {
    {
        auto addCallables = [&](Function f, auto &&addCallables) -> void {
            auto cs = f.custom_callables();
            for (auto &&i : cs) {
                Function c(i.get());
                callables.Emplace(c.hash(), c);
                addCallables(c, addCallables);
            }
        };
        addCallables(kernel, addCallables);
        for (auto &&i : *exprArr) {
            auto dict = i.get_or<IJsonDict *>(nullptr);
            if (!dict) continue;
            auto e = AstSerializer::GenExpr(dict, *this);
            if (e) {
                expr.Emplace(e->hash(), dict, e);
            }
        }
        for (auto &&i : *stmtArr) {
            auto dict = i.get_or<IJsonDict *>(nullptr);
            if (!dict) continue;
            auto e = AstSerializer::GenStmt(dict, *this);
            if (e) {
                stmt.Emplace(e->hash(), dict, e);
            }
        }
    }
    for (auto &&i : expr) {
        AstSerializer::DeserExpr(i.second.first, i.second.second, *this);
    }
    for (auto &&i : stmt) {
        AstSerializer::DeserStmt(i.second.first, i.second.second, *this);
    }
}

Expression const *DeserVisitor::GetExpr(uint64 hs) const {
    auto ite = expr.Find(hs);
    if (!ite) return nullptr;
    return ite.Value().second;
}
Statement const *DeserVisitor::GetStmt(uint64 hs) const {
    auto ite = stmt.Find(hs);
    if (!ite) return nullptr;
    return ite.Value().second;
}
void *DeserVisitor::Allocate(size_t sz) const {
    return vengine_malloc(sz);
}
Function DeserVisitor::GetFunction(uint64 hs) const {
    auto ite = callables.Find(hs);
    if (!ite) return {};
    return ite.Value();
}
DeserVisitor::~DeserVisitor() {
}
void DeserVisitor::GetExpr(vstd::function<void(Expression *)> const &func) {
    for (auto &&i : expr) {
        func(i.second.second);
        i.second.first = nullptr;
    }
}
void DeserVisitor::GetStmt(vstd::function<void(Statement *)> const &func) {
    for (auto &&i : stmt) {
        func(i.second.second);
        i.second.first = nullptr;
    }
}
}// namespace luisa::compute
