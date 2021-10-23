#include <serialize/serialize.h>
namespace luisa::compute {
vstd::unique_ptr<toolhub::db::IJsonDict> AstSerializer::Serialize(Type const &t, toolhub::db::IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("hash", t.hash());
    r->Set("size", t.size());
    r->Set("align", t.alignment());
    r->Set("tag", static_cast<int64>(t.tag()));
    r->Set("index", t._index);
    r->Set("dim", t.dimension());
    return r;
}
void AstSerializer::DeSerialize(Type &t, IJsonDict *dict) {
    auto getOr = [&](auto &&opt) {
        return dict->Get(opt).get_or<int64>(0);
    };
    t._hash = getOr("hash");
    t._size = getOr("size");
    t._index = getOr("index");
    t._alignment = getOr("alignment");
    t._dimension = getOr("dimension");
    t._tag = static_cast<Type::Tag>(getOr("tag"));
}

vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(TypeData const &t, IJsonDatabase *db) {
    auto data = db->CreateDict();
    data->Set("description", vstd::string_view(t.description.data(), t.description.size()));
    auto members = db->CreateArray();
    members->Reserve(t.members.size());
    for (auto &&i : t.members) {
        members->Add(i->hash());
    }
    data->Set("members", members);
    return data;
}
void AstSerializer::DeSerialize(TypeData &d, IJsonDict *dict) {
    auto descOpt = dict->Get("description").try_get<vstd::string_view>();
    if (descOpt) {
        d.description = luisa::string(descOpt->begin(), descOpt->size());
    }
    auto memberArr = dict->Get("members").try_get<IJsonArray *>();
    if (memberArr) {
        d.members.reserve((*memberArr)->Length());
        for (auto &&i : (**memberArr)) {
            auto it = i.try_get<int64>();
            d.members.push_back(it ? Type::get_type(*it) : nullptr);
        }
    }
}

vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(Expression const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("hash", t.hash());
    r->Set("type", t.type()->hash());
    r->Set("tag", static_cast<int64>(t.tag()));
    r->Set("usage", static_cast<int64>(t.usage()));
    return r;
}
void AstSerializer::DeSerialize(Expression &t, IJsonDict *r, vstd::function<Expression const *(SerHash)> const &getObj) {
    t._hash = r->Get("hash").get_or(0ll);
    t._type = Type::get_type(r->Get("type").get_or(0ll));
    t._tag = static_cast<Expression::Tag>(r->Get("tag").get_or(0ll));
    t._usage = static_cast<Usage>(r->Get("usage").get_or(0ll));
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(UnaryExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("operand", t.operand()->hash());
    r->Set("op", static_cast<int64>(t.op()));
    return r;
}
void AstSerializer::DeSerialize(UnaryExpr &t, IJsonDict *r, vstd::function<Expression const *(SerHash)> const &getObj) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, getObj);
    }
    t._op = static_cast<UnaryOp>(r->Get("op").get_or(0ll));
    t._operand = getObj(r->Get("operand").get_or(0ll));
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(BinaryExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("lhs", t.lhs()->hash());
    r->Set("rhs", t.rhs()->hash());
    r->Set("op", static_cast<int64>(t.op()));
    return r;
}
void AstSerializer::DeSerialize(BinaryExpr &t, IJsonDict *r, vstd::function<Expression const *(SerHash)> const &getObj) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, getObj);
    }
    t._lhs = getObj(r->Get("lhs").get_or(0ll));
    t._rhs = getObj(r->Get("rhs").get_or(0ll));
    t._op = static_cast<BinaryOp>(r->Get("op").get_or(0ll));
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(AccessExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("range", t.range()->hash());
    r->Set("index", t.index()->hash());
    return r;
}
void AstSerializer::DeSerialize(AccessExpr &t, IJsonDict *r, vstd::function<Expression const *(SerHash)> const &getObj) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, getObj);
    }
    t._range = getObj(r->Get("range").get_or(0ll));
    t._index = getObj(r->Get("index").get_or(0ll));
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(MemberExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("self", t.self()->hash());
    r->Set("member", t._member);
    return r;
}
void AstSerializer::DeSerialize(MemberExpr &t, IJsonDict *r, vstd::function<Expression const *(SerHash)> const &getObj) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, getObj);
    }
    t._self = getObj(r->Get("self").get_or(0ll));
    t._member = r->Get("member").get_or(0ll);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(LiteralExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    struct Visitor {
        IJsonDatabase *db;
        IJsonDict *r;
        void operator()(bool a) const {
            r->Set("value", a);
        }
        void operator()(int a) const {
            r->Set("value", int64(a));
        }
        void operator()(uint a) const {
            r->Set("value", int64(a));
        }
        void operator()(float a) const {
            r->Set("value", double(a));
        }
        void operator()(bool2 a) const {
            auto arr = db->CreateArray();
            arr->Add(a.x);
            arr->Add(a.y);
            r->Set("value", std::move(arr));
        }
        void operator()(int2 a) const {
            auto arr = db->CreateArray();
            arr->Add(int64(a.x));
            arr->Add(int64(a.y));
            r->Set("value", std::move(arr));
        }
        void operator()(uint2 a) const {
            auto arr = db->CreateArray();
            arr->Add(int64(a.x));
            arr->Add(int64(a.y));
            r->Set("value", std::move(arr));
        }
        void operator()(float2 a) const {
            auto arr = db->CreateArray();
            arr->Add(float(a.x));
            arr->Add(float(a.y));
            r->Set("value", std::move(arr));
        }
        void operator()(bool3 a) const {
            auto arr = db->CreateArray();
            arr->Add(a.x);
            arr->Add(a.y);
            arr->Add(a.z);
            r->Set("value", std::move(arr));
        }
        void operator()(int3 a) const {
            auto arr = db->CreateArray();
            arr->Add(int64(a.x));
            arr->Add(int64(a.y));
            arr->Add(int64(a.z));
            r->Set("value", std::move(arr));
        }
        void operator()(uint3 a) const {
            auto arr = db->CreateArray();
            arr->Add(int64(a.x));
            arr->Add(int64(a.y));
            arr->Add(int64(a.z));
            r->Set("value", std::move(arr));
        }
        void operator()(float3 a) const {
            auto arr = db->CreateArray();
            arr->Add(float(a.x));
            arr->Add(float(a.y));
            arr->Add(float(a.z));
            r->Set("value", std::move(arr));
        }
        void operator()(bool4 a) const {
            auto arr = db->CreateArray();
            arr->Add(a.x);
            arr->Add(a.y);
            arr->Add(a.z);
            arr->Add(a.w);
            r->Set("value", std::move(arr));
        }
        void operator()(int4 a) const {
            auto arr = db->CreateArray();
            arr->Add(int64(a.x));
            arr->Add(int64(a.y));
            arr->Add(int64(a.z));
            arr->Add(int64(a.w));
            r->Set("value", std::move(arr));
        }
        void operator()(uint4 a) const {
            auto arr = db->CreateArray();
            arr->Add(int64(a.x));
            arr->Add(int64(a.y));
            arr->Add(int64(a.z));
            arr->Add(int64(a.w));
            r->Set("value", std::move(arr));
        }
        void operator()(float4 a) const {
            auto arr = db->CreateArray();
            arr->Add(float(a.x));
            arr->Add(float(a.y));
            arr->Add(float(a.z));
            arr->Add(float(a.w));
            r->Set("value", std::move(arr));
        }
        void operator()(float2x2 a) const {
            auto arr = db->CreateArray();
            auto set = [&](auto &&c) {
                arr->Add(float(c.x));
                arr->Add(float(c.y));
            };
            for (auto &&i : a.cols) {
                set(i);
            }
            r->Set("value", std::move(arr));
        }
        void operator()(float3x3 a) const {
            auto arr = db->CreateArray();
            auto set = [&](auto &&c) {
                arr->Add(float(c.x));
                arr->Add(float(c.y));
                arr->Add(float(c.z));
            };
            for (auto &&i : a.cols) {
                set(i);
            }
            r->Set("value", std::move(arr));
        }
        void operator()(float4x4 a) const {
            auto arr = db->CreateArray();
            auto set = [&](auto &&c) {
                arr->Add(float(c.x));
                arr->Add(float(c.y));
                arr->Add(float(c.z));
                arr->Add(float(c.w));
            };
            for (auto &&i : a.cols) {
                set(i);
            }
            r->Set("value", std::move(arr));
        }
    };
    Visitor v{db, r.get()};
    std::visit(v, t._value);
    r->Set("value_type", t._value.index());
    return r;
}
void AstSerializer::DeSerialize(LiteralExpr &t, IJsonDict *r, vstd::function<Expression const *(SerHash)> const &getObj) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, getObj);
    }
    auto type = r->Get("value_type").try_get<int64>();
    if (!type) return;
    size_t ofst = 0;

    auto getFloat = [&](auto &&arr) {
        auto flt = arr->Get(ofst).get_or(0.0);
        ofst++;
        return flt;
    };
    switch (*type) {
        case 0:
            t._value = r->Get("value").get_or(false);
            break;
        case 1:
            t._value = static_cast<float>(r->Get("value").get_or(0.0));
            break;
        case 2:
            t._value = static_cast<int>(r->Get("value").get_or(0ll));
            break;
        case 3:
            t._value = static_cast<uint>(r->Get("value").get_or(0ll));
            break;
        case 4: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = bool2(
                (*arr)->Get(0).get_or(false),
                (*arr)->Get(1).get_or(false));
        } break;
        case 5: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = float2(
                (*arr)->Get(0).get_or(0.0),
                (*arr)->Get(1).get_or(0.0));
        } break;
        case 6: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = int2(
                (*arr)->Get(0).get_or(0ll),
                (*arr)->Get(1).get_or(0ll));
        } break;
        case 7: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = uint2(
                (*arr)->Get(0).get_or(0ll),
                (*arr)->Get(1).get_or(0ll));
        } break;
        case 8: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = bool3(
                (*arr)->Get(0).get_or(false),
                (*arr)->Get(1).get_or(false),
                (*arr)->Get(2).get_or(false));
        } break;
        case 9: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = float3(
                (*arr)->Get(0).get_or(0.0),
                (*arr)->Get(1).get_or(0.0),
                (*arr)->Get(2).get_or(0.0));
        } break;
        case 10: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = int3(
                (*arr)->Get(0).get_or(0ll),
                (*arr)->Get(1).get_or(0ll),
                (*arr)->Get(2).get_or(0ll));
        } break;
        case 11: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = uint3(
                (*arr)->Get(0).get_or(0ll),
                (*arr)->Get(1).get_or(0ll),
                (*arr)->Get(2).get_or(0ll));
        }

        break;
        case 12: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = bool4(
                (*arr)->Get(0).get_or(false),
                (*arr)->Get(1).get_or(false),
                (*arr)->Get(2).get_or(false),
                (*arr)->Get(3).get_or(false));
        } break;
        case 13: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = float4(
                (*arr)->Get(0).get_or(0.0),
                (*arr)->Get(1).get_or(0.0),
                (*arr)->Get(2).get_or(0.0),
                (*arr)->Get(3).get_or(0.0));
        } break;
        case 14: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = int4(
                (*arr)->Get(0).get_or(0ll),
                (*arr)->Get(1).get_or(0ll),
                (*arr)->Get(2).get_or(0ll),
                (*arr)->Get(3).get_or(0ll));
        } break;
        case 15: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t._value = uint4(
                (*arr)->Get(0).get_or(0ll),
                (*arr)->Get(1).get_or(0ll),
                (*arr)->Get(2).get_or(0ll),
                (*arr)->Get(3).get_or(0ll));
        } break;
        case 16: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            float2x2 v;
            for (auto &&i : v.cols) {
                i.x = getFloat(*arr);
                i.y = getFloat(*arr);
            }
            t._value = v;
        } break;
        case 17: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            float3x3 v;
            for (auto &&i : v.cols) {
                i.x = getFloat(*arr);
                i.y = getFloat(*arr);
                i.z = getFloat(*arr);
            }
            t._value = v;
        } break;
        case 18: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            float4x4 v;
            for (auto &&i : v.cols) {
                i.x = getFloat(*arr);
                i.y = getFloat(*arr);
                i.z = getFloat(*arr);
                i.w = getFloat(*arr);
            }
            t._value = v;
        } break;
    }
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(Variable const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("type", t.type()->hash());
    r->Set("uid", t.uid());
    r->Set("tag", static_cast<int64>(t.tag()));
    return r;
}
void AstSerializer::DeSerialize(Variable &t, IJsonDict *r) {
    t._type = Type::get_type(r->Get("type").get_or(0ll));
    t._tag = static_cast<Variable::Tag>(r->Get("tag").get_or(0ll));
    t._uid = r->Get("uid").get_or(0ll);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(RefExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("variable", Serialize(t._variable, db));
    return r;
}
void AstSerializer::DeSerialize(RefExpr &t, IJsonDict *r, vstd::function<Expression const *(SerHash)> const &getObj) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, getObj);
    }
    auto dd = r->Get("variable").try_get<IJsonDict *>();
    if (dd) {
        DeSerialize(t._variable, *dd);
    }
}
}// namespace luisa::compute
