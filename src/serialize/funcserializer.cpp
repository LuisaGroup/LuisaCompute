#pragma vengine_package serialize
#include <serialize/config.h>
#include <serialize/serialize.h>
#include <vstl/variant_util.h>
#include <serialize/funcserializer.h>
namespace luisa::compute {
using ReadVar = vstd::VariantVisitor_t<ReadJsonVariant>;
vstd::unique_ptr<IJsonDict> FuncSerializer::GetBuilderSerFunc(detail::FunctionBuilder const *b, IJsonDatabase *db) {
    auto exprArr = db->CreateArray();
    auto stmtArr = db->CreateArray();
    auto argArr = db->CreateArray();
    auto constBindArr = db->CreateArray();
    // auto buffBindArr = db->CreateArray();
    auto callablesArr = db->CreateArray();
    auto builtInCallArr = db->CreateArray();
    auto texBindArr = db->CreateArray();
    auto blkSize = db->CreateArray();
    exprArr->Reserve(b->_all_expressions.size());
    stmtArr->Reserve(b->_all_statements.size());
    argArr->Reserve(b->_arguments.size());
    constBindArr->Reserve(b->_captured_constants.size());
    //buffBindArr->Reserve(b->_captured_buffers.size());
    callablesArr->Reserve(b->_used_custom_callables.size());
    for (auto &&i : b->_all_expressions) {
        exprArr->Add(AstSerializer::SerExpr(db, *i));
    }
    for (auto &&i : b->_all_statements) {
        stmtArr->Add(AstSerializer::SerStmt(db, *i));
    }
    for (auto &&i : b->_arguments) {
        argArr->Add(AstSerializer::Serialize(i, db));
    }
    for (auto &&i : b->_captured_constants) {
        auto cDict = db->CreateDict();
        cDict->Set("data", AstSerializer::Serialize(i.data, db));
        cDict->Set("type", (int64)i.type->hash());
        constBindArr->Add(std::move(cDict));
    }
    for (auto &&i : b->_used_custom_callables) {
        callablesArr->Add((int64)i->_hash);
    }
    for (auto &&i : b->_used_builtin_callables) {
        builtInCallArr->Add(static_cast<int64>(i));
    }
    *blkSize << (int64)b->_block_size.x << (int64)b->_block_size.y << (int64)b->_block_size.z;
    /* for (auto &&i : b->_captured_buffers) {
        auto d = db->CreateDict();
        d->Set("variable", AstSerializer::Serialize(i.variable, db));
        d->Set("handle", i.handle);//TODO: handle map
        d->Set("offset_bytes", i.offset_bytes);
        buffBindArr->Add(std::move(d));
    }*/
    auto rootDict = db->CreateDict();
    rootDict->Set("all_expressions", std::move(exprArr));
    rootDict->Set("all_statements", std::move(stmtArr));
    rootDict->Set("body", AstSerializer::Serialize(b->_body, db));
    rootDict->Set("arguments", std::move(argArr));
    rootDict->Set("captured_constants", std::move(constBindArr));
    rootDict->Set("custom_callables", std::move(callablesArr));
    rootDict->Set("builtin_callables", std::move(builtInCallArr));
    rootDict->Set("tag", static_cast<int64>(b->_tag));
    rootDict->Set("block_size", std::move(blkSize));
    rootDict->Set("hash", (int64)b->_hash);
    rootDict->Set("raytracing", b->_raytracing);
    static_assert(call_op_count <= sizeof(int64) * 2 * 8, "CallOpSet size should less than placeholder");

    //rootDict->Set("captured_buffers", std::move(buffBindArr));

    return rootDict;
}

void FuncSerializer::GetBuilderDeserFunc(IJsonDict *dict, detail::FunctionBuilder *fb, FuncMap &map) {

    auto expr = ReadVar::get_or<IJsonArray *>(dict->Get("all_expressions"), nullptr);
    auto stmt = ReadVar::get_or<IJsonArray *>(dict->Get("all_statements"), nullptr);
    auto blkArr = ReadVar::get_or<IJsonArray *>(dict->Get("block_size"), nullptr);
    auto bodyDict = ReadVar::get_or<IJsonDict *>(dict->Get("body"), nullptr);
    auto customCallables = ReadVar::get_or<IJsonArray *>(dict->Get("custom_callables"), nullptr);
    auto builtinCallables = ReadVar::get_or<IJsonArray *>(dict->Get("builtin_callables"), nullptr);
    auto consts = ReadVar::get_or<IJsonArray *>(dict->Get("captured_constants"), nullptr);
    auto args = ReadVar::get_or<IJsonArray *>(dict->Get("arguments"), nullptr);
    if (!expr ||
        !stmt ||
        !blkArr ||
        !bodyDict ||
        !customCallables ||
        !builtinCallables ||
        !consts ||
        !args)
        return;
    Function f(fb);
    DeserVisitor vis(
        f,
        expr,
        stmt);
    fb->_all_expressions.reserve(vis.exprCount());
    fb->_all_statements.reserve(vis.stmtCount());
    vis.GetExpr([&](Expression *e) {
        fb->_all_expressions.push_back(luisa::unique_ptr<Expression>(e));
    });
    vis.GetStmt([&](Statement *e) {
        fb->_all_statements.push_back(luisa::unique_ptr<Statement>(e));
    });

    if (blkArr->Length() >= 3) {
        fb->_block_size = {
            (uint)ReadVar::get_or<int64>((*blkArr)[0], 0),
            (uint)ReadVar::get_or<int64>((*blkArr)[1], 0),
            (uint)ReadVar::get_or<int64>((*blkArr)[2], 0)};
    }

    AstSerializer::DeSerialize(fb->_body, bodyDict, vis);
    fb->_used_custom_callables.reserve(customCallables->Length());
    for (auto &&i : *customCallables) {
        if (auto v = ReadVar::try_get<int64>(i)) {
            auto ite = map.Find(*v);
            fb->_used_custom_callables.emplace_back(ite.Value().second);
        }
    }
    for (auto &&i : *builtinCallables) {
        if (auto op = ReadVar::try_get<int64>(i)) {
            fb->_used_builtin_callables.mark(static_cast<CallOp>(*op));
        }
    }
    fb->_captured_constants.reserve(consts->Length());
    for (auto &&i : *consts) {
        Function::Constant &cb = fb->_captured_constants.emplace_back();
        auto d = ReadVar::get_or<IJsonDict *>(i, nullptr);
        if (!d) continue;
        auto typeHash = ReadVar::try_get<int64>(d->Get("type"));
        if (typeHash)
            cb.type = Type::find(*typeHash);
        auto data = ReadVar::get_or<IJsonDict *>(d->Get("data"), nullptr);
        if (data)
            AstSerializer::DeSerialize(cb.data, data, vis);
    }
    fb->_arguments.reserve(args->Length());
    for (auto &&i : *args) {
        auto dict = ReadVar::get_or<IJsonDict *>(i, nullptr);
        if (!dict) continue;
        auto &&arg = fb->_arguments.emplace_back();
        AstSerializer::DeSerialize(arg, dict);
    }
    fb->_raytracing = ReadVar::get_or<bool>(dict->Get("raytracing"), false);
    //TODO
}
vstd::unique_ptr<IJsonArray> FuncSerializer::SerKernel(Function f, IJsonDatabase *db) {
    auto arr = db->CreateArray();
    vstd::HashMap<uint64, Function> checkMap;
    auto func = [&](auto &&func, Function f) -> void {
        checkMap.Emplace(f.hash(), f);
        for (auto &&i : f.custom_callables()) {
            Function ff(i.get());
            func(func, ff);
        }
    };
    func(func, f);
    arr->Reserve(checkMap.size());
    for (auto &&i : checkMap) {
        arr->Add(GetBuilderSerFunc(f.builder(), db));
    }
    return arr;
}
Function FuncSerializer::DeserKernel(IJsonArray *arr) {
    FuncMap map;
    detail::FunctionBuilder const *bd = nullptr;
    for (auto &&i : *arr) {
        auto dict = ReadVar::get_or<IJsonDict *>(i, nullptr);
        if (!dict) continue;
        auto tag = ReadVar::try_get<int64>(dict->Get("tag"));
        auto hs = ReadVar::try_get<int64>(dict->Get("hash"));
        auto fb = vengine_new<detail::FunctionBuilder>(static_cast<detail::FunctionBuilder::Tag>(*tag));
        map.Emplace(fb->_hash, dict, fb);
        if (fb->_tag == detail::FunctionBuilder::Tag::KERNEL)
            bd = fb;
    }
    for (auto &&i : map) {
        GetBuilderDeserFunc(
            i.second.first,
            i.second.second.get(),
            map);
    }
    return Function(bd);
}
vstd::unique_ptr<IJsonArray> Serializer_Impl::SerKernel(Function func, IJsonDatabase *db) const {
    return FuncSerializer::SerKernel(func, db);
}
Function Serializer_Impl::DeserKernel(IJsonArray *arr) const {
    return FuncSerializer::DeserKernel(arr);
}

vstd::unique_ptr<IJsonArray> Serializer_Impl::SerTypes(IJsonDatabase *db) const {
    auto arr = db->CreateArray();
    struct SerTypeVisitor : public TypeVisitor {
        IJsonDatabase *db;
        IJsonArray *arr;
        SerTypeVisitor(
            IJsonDatabase *db,
            IJsonArray *arr) : arr(arr), db(db) {}
        void visit(Type const *t) noexcept override {
            arr->Add(t->description());
        }
    };
    SerTypeVisitor v{db, arr.get()};
    Type::traverse(v);
    return arr;
}
vstd::vector<Type const *> Serializer_Impl::DeserTypes(IJsonArray *arr) const {
    vstd::vector<Type const *> allTypes;
    allTypes.push_back_func(
        [&](size_t i) -> Type const * {
            auto ss = ReadVar::try_get<std::string_view>((*arr)[i]);
            if (ss)
                return Type::from(*ss);
            else
                return nullptr;
        },
        arr->Length());
    return allTypes;
}
vstd::optional<Serializer_Impl> serializer;
VSTL_EXPORT_C ISerializer const *Serialize_GetFactory() {
    serializer.New();
    return serializer;
}
}// namespace luisa::compute