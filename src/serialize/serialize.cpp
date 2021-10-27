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
        return dict->Get(opt).template get_or<int64>(0);
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
    data->Set("description", std::string_view(t.description.data(), t.description.size()));
    auto members = db->CreateArray();
    members->Reserve(t.members.size());
    for (auto &&i : t.members) {
        members->Add(i->hash());
    }
    data->Set("members", members);
    return data;
}
void AstSerializer::DeSerialize(TypeData &d, IJsonDict *dict) {
    auto descOpt = dict->Get("description").try_get<std::string_view>();
    if (descOpt) {
        d.description = luisa::string(descOpt->data(), descOpt->size());
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
void AstSerializer::DeSerialize(Expression &t, IJsonDict *r, SerializeVisitor const &evt) {
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
void AstSerializer::DeSerialize(UnaryExpr &t, IJsonDict *r, SerializeVisitor const &evt) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, evt);
    }
    t._op = static_cast<UnaryOp>(r->Get("op").get_or(0ll));
    t._operand = evt.getExpr(r->Get("operand").get_or(0ll));
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(BinaryExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("lhs", t.lhs()->hash());
    r->Set("rhs", t.rhs()->hash());
    r->Set("op", static_cast<int64>(t.op()));
    return r;
}
void AstSerializer::DeSerialize(BinaryExpr &t, IJsonDict *r, SerializeVisitor const &evt) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, evt);
    }
    t._lhs = evt.getExpr(r->Get("lhs").get_or(0ll));
    t._rhs = evt.getExpr(r->Get("rhs").get_or(0ll));
    t._op = static_cast<BinaryOp>(r->Get("op").get_or(0ll));
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(AccessExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("range", t.range()->hash());
    r->Set("index", t.index()->hash());
    return r;
}
void AstSerializer::DeSerialize(AccessExpr &t, IJsonDict *r, SerializeVisitor const &evt) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, evt);
    }
    t._range = evt.getExpr(r->Get("range").get_or(0ll));
    t._index = evt.getExpr(r->Get("index").get_or(0ll));
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(MemberExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("self", t.self()->hash());
    r->Set("member", t._member);
    return r;
}
void AstSerializer::DeSerialize(MemberExpr &t, IJsonDict *r, SerializeVisitor const &evt) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, evt);
    }
    t._self = evt.getExpr(r->Get("self").get_or(0ll));
    t._member = r->Get("member").get_or(0ll);
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(LiteralExpr const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    r->Set("expr", Serialize(static_cast<Expression const &>(t), db));
    r->Set("value", Serialize(t._value, db));
    return r;
}
void AstSerializer::DeSerialize(LiteralExpr &t, IJsonDict *r, SerializeVisitor const &evt) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, evt);
    }
    auto value = r->Get("value").try_get<IJsonDict *>();
    if (value) {
        DeSerialize(t._value, *value);
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
void AstSerializer::DeSerialize(RefExpr &t, IJsonDict *r, SerializeVisitor const &evt) {
    auto exprD = r->Get("expr").try_get<IJsonDict *>();
    if (exprD) {
        DeSerialize(static_cast<Expression &>(t), *exprD, evt);
    }
    auto dd = r->Get("variable").try_get<IJsonDict *>();
    if (dd) {
        DeSerialize(t._variable, *dd);
    }
}

vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(ConstantData const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    auto &&view = t.view();
    r->Set("view_type", view.index());
    r->Set("hash", t.hash());
    struct SerValueVisitor {
        IJsonDatabase *db;
        IJsonArray &r;
        void operator()(std::span<const bool> a) const {
            for (auto &&i : a) {
                r << i;
            }
        }
        void operator()(std::span<const int> a) const {
            for (auto &&i : a) {
                r << i;
            }
        }
        void operator()(std::span<const uint> a) const {
            for (auto &&i : a) {
                r << i;
            }
        }
        void operator()(std::span<const float> a) const {
            for (auto &&i : a) {
                r << double(i);
            }
        }
        void operator()(std::span<const bool2> a) const {
            for (auto &&i : a) {
                r << i.x << i.y;
            }
        }
        void operator()(std::span<const int2> a) const {
            for (auto &&i : a) {
                r << i.x << i.y;
            }
        }
        void operator()(std::span<const uint2> a) const {
            for (auto &&i : a) {
                r << i.x << i.y;
            }
        }
        void operator()(std::span<const float2> a) const {
            for (auto &&i : a) {
                r << double(i.x) << double(i.y);
            }
        }
        void operator()(std::span<const bool3> a) const {
            for (auto &&i : a) {
                r << i.x << i.y << i.z;
            }
        }
        void operator()(std::span<const int3> a) const {
            for (auto &&i : a) {
                r << i.x << i.y << i.z;
            }
        }
        void operator()(std::span<const uint3> a) const {
            for (auto &&i : a) {
                r << i.x << i.y << i.z;
            }
        }
        void operator()(std::span<const float3> a) const {
            for (auto &&i : a) {
                r << double(i.x) << double(i.y) << double(i.z);
            }
        }
        void operator()(std::span<const bool4> a) const {
            for (auto &&i : a) {
                r << i.x << i.y << i.z << i.w;
            }
        }
        void operator()(std::span<const int4> a) const {
            for (auto &&i : a) {
                r << i.x << i.y << i.z << i.w;
            }
        }
        void operator()(std::span<const uint4> a) const {
            for (auto &&i : a) {
                r << i.x << i.y << i.z << i.w;
            }
        }
        void operator()(std::span<const float4> a) const {
            for (auto &&i : a) {
                r << double(i.x) << double(i.y) << double(i.z) << double(i.w);
            }
        }
        void operator()(std::span<const float2x2> a) const {
            for (auto &&i : a) {
                for (auto &&c : i.cols)
                    r << double(c.x) << double(c.y);
            }
        }
        void operator()(std::span<const float3x3> a) const {
            for (auto &&i : a) {
                for (auto &&c : i.cols)
                    r << double(c.x) << double(c.y) << double(c.z);
            }
        }
        void operator()(std::span<const float4x4> a) const {
            for (auto &&i : a) {
                for (auto &&c : i.cols)
                    r << double(c.x) << double(c.y) << double(c.z) << double(c.w);
            }
        }
    };
    auto arr = db->CreateArray();
    SerValueVisitor vis{db, *arr};
    std::visit(vis, view);
    r->Set("values", std::move(arr));
    return r;
}
void AstSerializer::DeSerialize(ConstantData &t, IJsonDict *r, SerializeVisitor const &evt) {
    t._hash = r->Get("hash").get_or(0ll);
    auto arrOpt = r->Get("values").try_get<IJsonArray *>();
    auto type = r->Get("view_type").get_or(std::numeric_limits<int64>::max());
    if (arrOpt) {
        auto &&arr = **arrOpt;
        switch (type) {
            case 0: {
                size_t sz = arr.Length();
                bool *ptr = (bool *)evt.allocate(sz);
                t._view = std::span<bool const>(ptr, sz);
                for (auto &&i : arr) {
                    *ptr = i.get_or(false);
                    ptr++;
                }
            } break;
            case 1: {
                size_t sz = arr.Length() * sizeof(float);
                float *ptr = (float *)evt.allocate(sz);
                t._view = std::span<float const>(ptr, sz);
                for (auto&& i : arr) {
                    *ptr = i.get_or(0.0);
                    ptr++;
                }
            } break;
            case 2: {
                size_t sz = arr.Length() * sizeof(int);
                int *ptr = (int *)evt.allocate(sz);
                t._view = std::span<int const>(ptr, sz);
                for (auto &&i : arr) {
                    *ptr = i.get_or(0ll);
                    ptr++;
                }
            }break;
            case 3: {
                size_t sz = arr.Length() * sizeof(int);
                uint *ptr = (uint *)evt.allocate(sz);
                t._view = std::span<uint const>(ptr, sz);
                for (auto &&i : arr) {
                    *ptr = i.get_or(0ll);
                    ptr++;
                }
            } break;
            case 4: {
                size_t sz = arr.Length();
                bool *ptr = (bool *)evt.allocate(sz);
                t._view = std::span<bool2 const>((bool2*)ptr, sz / 2);
                for (auto &&i : arr) {
                    *ptr = i.get_or(false);
                    ptr++;
                }
            } break;
            case 5: {
                size_t sz = arr.Length() * sizeof(float);
                float *ptr = (float *)evt.allocate(sz);
                t._view = std::span<float2 const>((float2*)ptr, sz / 2);
                for (auto &&i : arr) {
                    *ptr = i.get_or(0.0);
                    ptr++;
                }
            } break;
            case 6: {
                size_t sz = arr.Length() * sizeof(int);
                int *ptr = (int *)evt.allocate(sz);
                t._view = std::span<int2 const>((int2*)ptr, sz / 2);
                for (auto &&i : arr) {
                    *ptr = i.get_or(0ll);
                    ptr++;
                }
            } break;
            case 7: {
                size_t sz = arr.Length() * sizeof(int);
                uint *ptr = (uint *)evt.allocate(sz);
                t._view = std::span<uint2 const>((uint2*)ptr, sz/2);
                for (auto &&i : arr) {
                    *ptr = i.get_or(0ll);
                    ptr++;
                }
            } break;
            case 8: {
                size_t sz = arr.Length();
                bool *ptr = (bool *)evt.allocate(sz);
                t._view = std::span<bool3 const>((bool3 *)ptr, sz/3);
                for (auto &&i : arr) {
                    *ptr = i.get_or(false);
                    ptr++;
                }
            } break;
            case 9: {
                size_t sz = arr.Length() * sizeof(float);
                float *ptr = (float *)evt.allocate(sz);
                t._view = std::span<float3 const>((float3 *)ptr, sz / 3);
                for (auto &&i : arr) {
                    *ptr = i.get_or(0.0);
                    ptr++;
                }
            } break;
            case 10: {
                size_t sz = arr.Length() * sizeof(int);
                int *ptr = (int *)evt.allocate(sz);
                t._view = std::span<int3 const>((int3 *)ptr, sz / 3);
                for (auto &&i : arr) {
                    *ptr = i.get_or(0ll);
                    ptr++;
                }
            } break;
            case 11: {
                size_t sz = arr.Length() * sizeof(int);
                uint *ptr = (uint *)evt.allocate(sz);
                t._view = std::span<uint3 const>((uint3 *)ptr, sz / 3);
                for (auto &&i : arr) {
                    *ptr = i.get_or(0ll);
                    ptr++;
                }
            } break;
            case 12: {
                size_t sz = arr.Length();
                bool *ptr = (bool *)evt.allocate(sz);
                t._view = std::span<bool4 const>((bool4 *)ptr, sz / 4);
                for (auto &&i : arr) {
                    *ptr = i.get_or(false);
                    ptr++;
                }
            } break;
            case 13: {
                size_t sz = arr.Length() * sizeof(float);
                float *ptr = (float *)evt.allocate(sz);
                t._view = std::span<float4 const>((float4 *)ptr, sz / 4);
                for (auto &&i : arr) {
                    *ptr = i.get_or(0.0);
                    ptr++;
                }
            } break;
            case 14: {
                size_t sz = arr.Length() * sizeof(int);
                int *ptr = (int *)evt.allocate(sz);
                t._view = std::span<int4 const>((int4 *)ptr, sz / 4);
                for (auto &&i : arr) {
                    *ptr = i.get_or(0ll);
                    ptr++;
                }
            } break;
            case 15: {
                size_t sz = arr.Length() * sizeof(int);
                uint *ptr = (uint *)evt.allocate(sz);
                t._view = std::span<uint4 const>((uint4 *)ptr, sz / 4);
                for (auto &&i : arr) {
                    *ptr = i.get_or(0ll);
                    ptr++;
                }
            } break;
            case 16: {
                size_t sz = arr.Length() * sizeof(float);
                float *ptr = (float *)evt.allocate(sz);
                t._view = std::span<float2x2 const>((float2x2 *)ptr, sz / sizeof(float2x2));
                for (auto &&i : arr) {
                    *ptr = i.get_or(0.0);
                    ptr++;
                }
            } break;
            case 17: {
                size_t sz = arr.Length() * sizeof(float);
                float *ptr = (float *)evt.allocate(sz);
                t._view = std::span<float3x3 const>((float3x3 *)ptr, sz / sizeof(float3x3));
                for (auto &&i : arr) {
                    *ptr = i.get_or(0.0);
                    ptr++;
                }
            } break;
            case 18: {
                size_t sz = arr.Length() * sizeof(float);
                float *ptr = (float *)evt.allocate(sz);
                t._view = std::span<float4x4 const>((float4x4 *)ptr, sz / sizeof(float4x4));
                for (auto &&i : arr) {
                    *ptr = i.get_or(0.0);
                    ptr++;
                }
            } break;
        }
    }
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(LiteralExpr::Value const &t, IJsonDatabase *db) {
    auto r = db->CreateDict();
    struct SerValueVisitor {
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
            (*arr) << (a.x);
            (*arr) << (a.y);
            r->Set("value", std::move(arr));
        }
        void operator()(int2 a) const {
            auto arr = db->CreateArray();
            (*arr) << (int64(a.x));
            (*arr) << (int64(a.y));
            r->Set("value", std::move(arr));
        }
        void operator()(uint2 a) const {
            auto arr = db->CreateArray();
            (*arr) << (int64(a.x));
            (*arr) << (int64(a.y));
            r->Set("value", std::move(arr));
        }
        void operator()(float2 a) const {
            auto arr = db->CreateArray();
            (*arr) << (float(a.x));
            (*arr) << (float(a.y));
            r->Set("value", std::move(arr));
        }
        void operator()(bool3 a) const {
            auto arr = db->CreateArray();
            (*arr) << (a.x);
            (*arr) << (a.y);
            (*arr) << (a.z);
            r->Set("value", std::move(arr));
        }
        void operator()(int3 a) const {
            auto arr = db->CreateArray();
            (*arr) << (int64(a.x));
            (*arr) << (int64(a.y));
            (*arr) << (int64(a.z));
            r->Set("value", std::move(arr));
        }
        void operator()(uint3 a) const {
            auto arr = db->CreateArray();
            (*arr) << (int64(a.x));
            (*arr) << (int64(a.y));
            (*arr) << (int64(a.z));
            r->Set("value", std::move(arr));
        }
        void operator()(float3 a) const {
            auto arr = db->CreateArray();
            (*arr) << (float(a.x));
            (*arr) << (float(a.y));
            (*arr) << (float(a.z));
            r->Set("value", std::move(arr));
        }
        void operator()(bool4 a) const {
            auto arr = db->CreateArray();
            (*arr) << (a.x);
            (*arr) << (a.y);
            (*arr) << (a.z);
            (*arr) << (a.w);
            r->Set("value", std::move(arr));
        }
        void operator()(int4 a) const {
            auto arr = db->CreateArray();
            (*arr) << (int64(a.x));
            (*arr) << (int64(a.y));
            (*arr) << (int64(a.z));
            (*arr) << (int64(a.w));
            r->Set("value", std::move(arr));
        }
        void operator()(uint4 a) const {
            auto arr = db->CreateArray();
            (*arr) << (int64(a.x));
            (*arr) << (int64(a.y));
            (*arr) << (int64(a.z));
            (*arr) << (int64(a.w));
            r->Set("value", std::move(arr));
        }
        void operator()(float4 a) const {
            auto arr = db->CreateArray();
            (*arr) << (float(a.x));
            (*arr) << (float(a.y));
            (*arr) << (float(a.z));
            (*arr) << (float(a.w));
            r->Set("value", std::move(arr));
        }
        void operator()(float2x2 a) const {
            auto arr = db->CreateArray();
            auto set = [&](auto &&c) {
                (*arr) << (float(c.x));
                (*arr) << (float(c.y));
            };
            for (auto &&i : a.cols) {
                set(i);
            }
            r->Set("value", std::move(arr));
        }
        void operator()(float3x3 a) const {
            auto arr = db->CreateArray();
            auto set = [&](auto &&c) {
                (*arr) << (float(c.x));
                (*arr) << (float(c.y));
                (*arr) << (float(c.z));
            };
            for (auto &&i : a.cols) {
                set(i);
            }
            r->Set("value", std::move(arr));
        }
        void operator()(float4x4 a) const {
            auto arr = db->CreateArray();
            auto set = [&](auto &&c) {
                (*arr) << (float(c.x));
                (*arr) << (float(c.y));
                (*arr) << (float(c.z));
                (*arr) << (float(c.w));
            };
            for (auto &&i : a.cols) {
                set(i);
            }
            r->Set("value", std::move(arr));
        }
    };

    SerValueVisitor v{db, r.get()};
    std::visit(v, t);
    r->Set("value_type", t.index());
}
void AstSerializer::DeSerialize(LiteralExpr::Value &t, IJsonDict *r) {
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
            t = r->Get("value").get_or(false);
            break;
        case 1:
            t = static_cast<float>(r->Get("value").get_or(0.0));
            break;
        case 2:
            t = static_cast<int>(r->Get("value").get_or(0ll));
            break;
        case 3:
            t = static_cast<uint>(r->Get("value").get_or(0ll));
            break;
        case 4: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = bool2(
                (*arr)->Get(0).get_or(false),
                (*arr)->Get(1).get_or(false));
        } break;
        case 5: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = float2(
                (*arr)->Get(0).get_or(0.0),
                (*arr)->Get(1).get_or(0.0));
        } break;
        case 6: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = int2(
                (*arr)->Get(0).get_or(0ll),
                (*arr)->Get(1).get_or(0ll));
        } break;
        case 7: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = uint2(
                (*arr)->Get(0).get_or(0ll),
                (*arr)->Get(1).get_or(0ll));
        } break;
        case 8: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = bool3(
                (*arr)->Get(0).get_or(false),
                (*arr)->Get(1).get_or(false),
                (*arr)->Get(2).get_or(false));
        } break;
        case 9: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = float3(
                (*arr)->Get(0).get_or(0.0),
                (*arr)->Get(1).get_or(0.0),
                (*arr)->Get(2).get_or(0.0));
        } break;
        case 10: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = int3(
                (*arr)->Get(0).get_or(0ll),
                (*arr)->Get(1).get_or(0ll),
                (*arr)->Get(2).get_or(0ll));
        } break;
        case 11: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = uint3(
                (*arr)->Get(0).get_or(0ll),
                (*arr)->Get(1).get_or(0ll),
                (*arr)->Get(2).get_or(0ll));
        }

        break;
        case 12: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = bool4(
                (*arr)->Get(0).get_or(false),
                (*arr)->Get(1).get_or(false),
                (*arr)->Get(2).get_or(false),
                (*arr)->Get(3).get_or(false));
        } break;
        case 13: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = float4(
                (*arr)->Get(0).get_or(0.0),
                (*arr)->Get(1).get_or(0.0),
                (*arr)->Get(2).get_or(0.0),
                (*arr)->Get(3).get_or(0.0));
        } break;
        case 14: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = int4(
                (*arr)->Get(0).get_or(0ll),
                (*arr)->Get(1).get_or(0ll),
                (*arr)->Get(2).get_or(0ll),
                (*arr)->Get(3).get_or(0ll));
        } break;
        case 15: {
            auto arr = r->Get("value").try_get<IJsonArray *>();
            if (!arr) break;
            t = uint4(
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
            t = v;
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
            t = v;
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
            t = v;
        } break;
    }
}
vstd::unique_ptr<IJsonDict> AstSerializer::Serialize(ConstantExpr const &t, IJsonDatabase *db) {}
void AstSerializer::DeSerialize(ConstantExpr &t, IJsonDict *r) {}
}// namespace luisa::compute
