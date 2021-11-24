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
    static_assert(call_op_count <= sizeof(int64) * 2 * 8, "CallOpSet size should less than placeholder");

    //rootDict->Set("captured_buffers", std::move(buffBindArr));

    return rootDict;
}

detail::FunctionBuilder const *FuncSerializer::GetBuilderSerFunc(IJsonDict *dict) {
    auto fb = vengine_new<detail::FunctionBuilder>();
    Function f(fb);
    auto expr = dict->Get("all_expressions").get_or<IJsonArray *>(nullptr);
    auto stmt = dict->Get("all_statements").get_or<IJsonArray *>(nullptr);
    if (!expr || !stmt)
        return;

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

    auto blkArr = dict->Get("block_size").get_or<IJsonArray *>(nullptr);
    if (blkArr && blkArr->Length() >= 3) {
        fb->_block_size = {
            (*blkArr)[0].get_or<int64>(0),
            (*blkArr)[1].get_or<int64>(0),
            (*blkArr)[2].get_or<int64>(0)};
    }
    fb->_hash = dict->Get("hash").get_or<int64>(0);
    fb->_tag = static_cast<detail::FunctionBuilder::Tag>(dict->Get("tag").get_or<int64>(0));
    auto bodyDict = dict->Get("body").get_or<IJsonDict *>(nullptr);
    if (bodyDict) {
        AstSerializer::DeSerialize(fb->_body, bodyDict, vis);
    }
    //TODO
}
vstd::unique_ptr<IJsonDict> FuncSerializer::SerFunc(Function func, IJsonDatabase *db) {
    return GetBuilderSerFunc(func.builder(), db);
}
Function FuncSerializer::DeserFunc(IJsonDict *dict) {
}

}// namespace luisa::compute