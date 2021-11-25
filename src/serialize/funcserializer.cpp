#pragma vengine_package serialize
#include <serialize/serialize.h>
#include <serialize/funcserializer.h>
namespace luisa::compute {
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
        cDict->Set("type", i.type->hash());
        constBindArr->Add(std::move(cDict));
    }
    for (auto &&i : b->_used_custom_callables) {
        callablesArr->Add(i->_hash);
    }
    for (auto &&i : b->_used_builtin_callables) {
        builtInCallArr->Add(static_cast<int64>(i));
    }
    *blkSize << b->_block_size.x << b->_block_size.y << b->_block_size.z;
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
    rootDict->Set("hash", b->_hash);
    rootDict->Set("raytracing", b->_raytracing);
    static_assert(call_op_count <= sizeof(int64) * 2 * 8, "CallOpSet size should less than placeholder");

    //rootDict->Set("captured_buffers", std::move(buffBindArr));

    return rootDict;
}

void FuncSerializer::GetBuilderDeserFunc(IJsonDict *dict, detail::FunctionBuilder *fb, FuncMap &map) {

    auto expr = dict->Get("all_expressions").get_or<IJsonArray *>(nullptr);
    auto stmt = dict->Get("all_statements").get_or<IJsonArray *>(nullptr);
    auto blkArr = dict->Get("block_size").get_or<IJsonArray *>(nullptr);
    auto bodyDict = dict->Get("body").get_or<IJsonDict *>(nullptr);
    auto customCallables = dict->Get("custom_callables").get_or<IJsonArray *>(nullptr);
    auto builtinCallables = dict->Get("builtin_callables").get_or<IJsonArray *>(nullptr);
    auto consts = dict->Get("captured_constants").get_or<IJsonArray *>(nullptr);
    auto args = dict->Get("arguments").get_or<IJsonArray *>(nullptr);
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
            (uint)(*blkArr)[0].get_or<int64>(0),
            (uint)(*blkArr)[1].get_or<int64>(0),
            (uint)(*blkArr)[2].get_or<int64>(0)};
    }

    AstSerializer::DeSerialize(fb->_body, bodyDict, vis);
    fb->_used_custom_callables.reserve(customCallables->Length());
    for (auto &&i : *customCallables) {
        if (auto v = i.try_get<int64>()) {
            auto ite = map.Find(*v);
            fb->_used_custom_callables.emplace_back(ite.Value().second);
        }
    }
    for (auto &&i : *builtinCallables) {
        if (auto op = i.try_get<int64>()) {
            fb->_used_builtin_callables.mark(static_cast<CallOp>(*op));
        }
    }
    fb->_captured_constants.reserve(consts->Length());
    for (auto &&i : *consts) {
        Function::ConstantBinding &cb = fb->_captured_constants.emplace_back();
        auto d = i.get_or<IJsonDict *>(nullptr);
        if (!d) continue;
        auto typeHash = d->Get("type").try_get<int64>();
        if (typeHash)
            cb.type = Type::get_type(*typeHash);
        auto data = d->Get("data").get_or<IJsonDict *>(nullptr);
        if (data)
            AstSerializer::DeSerialize(cb.data, data, vis);
    }
    fb->_arguments.reserve(args->Length());
    for (auto &&i : *args) {
        auto dict = i.get_or<IJsonDict *>(nullptr);
        if (!dict) continue;
        auto &&arg = fb->_arguments.emplace_back();
        AstSerializer::DeSerialize(arg, dict);
    }
    fb->_raytracing = dict->Get("raytracing").get_or<bool>(false);
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
        auto dict = i.get_or<IJsonDict *>(nullptr);
        if (!dict) continue;
        auto tag = dict->Get("tag").try_get<int64>();
        auto hs = dict->Get("hash").try_get<int64>();
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
}// namespace luisa::compute