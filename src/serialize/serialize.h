#pragma once
#include <ast/type.h>
#include <serialize/IJsonObject.h>
#include <ast/expression.h>
namespace luisa::compute {
using IJsonDict = toolhub::db::IJsonDict;
using IJsonArray = toolhub::db::IJsonArray;
using IJsonDatabase = toolhub::db::IJsonDatabase;
using SerHash = uint64;
class SerializeVisitor {
public:
    virtual Expression const *GetExpr(SerHash) const = 0;
    virtual void *Allocate(size_t) const = 0;
    virtual Function GetFunction(SerHash) const = 0;
};
class AstSerializer {
public:
    static vstd::unique_ptr<IJsonDict> Serialize(Type const &t, IJsonDatabase *db);
    static void DeSerialize(Type &t, IJsonDict *dict);
    static vstd::unique_ptr<IJsonDict> Serialize(TypeData const &t, IJsonDatabase *db);
    static void DeSerialize(TypeData &t, IJsonDict *dict);
    static vstd::unique_ptr<IJsonDict> Serialize(Expression const &t, IJsonDatabase *db);
    static void DeSerialize(Expression &t, IJsonDict *dict, SerializeVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(UnaryExpr const &t, IJsonDatabase *db);
    static void DeSerialize(UnaryExpr &t, IJsonDict *dict, SerializeVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(BinaryExpr const &t, IJsonDatabase *db);
    static void DeSerialize(BinaryExpr &t, IJsonDict *dict, SerializeVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(AccessExpr const &t, IJsonDatabase *db);
    static void DeSerialize(AccessExpr &t, IJsonDict *dict, SerializeVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(MemberExpr const &t, IJsonDatabase *db);
    static void DeSerialize(MemberExpr &t, IJsonDict *dict, SerializeVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(LiteralExpr const &t, IJsonDatabase *db);
    static void DeSerialize(LiteralExpr &t, IJsonDict *dict, SerializeVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(RefExpr const &t, IJsonDatabase *db);
    static void DeSerialize(RefExpr &t, IJsonDict *dict, SerializeVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(Variable const &t, IJsonDatabase *db);
    static void DeSerialize(Variable &t, IJsonDict *dict);
    static vstd::unique_ptr<IJsonDict> Serialize(ConstantData const &t, IJsonDatabase *db);
    static void DeSerialize(ConstantData &t, IJsonDict *dict, SerializeVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(LiteralExpr::Value const &t, IJsonDatabase *db);
    static void DeSerialize(LiteralExpr::Value &t, IJsonDict *dict);
    static vstd::unique_ptr<IJsonDict> Serialize(ConstantExpr const &t, IJsonDatabase *db);
    static void DeSerialize(ConstantExpr &t, IJsonDict *dict, SerializeVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(CallExpr const &t, IJsonDatabase *db);
    static void DeSerialize(CallExpr &t, IJsonDict *dict, SerializeVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(CastExpr const &t, IJsonDatabase *db);
    static void DeSerialize(CastExpr &t, IJsonDict *dict, SerializeVisitor const &evt);
};
}// namespace luisa::compute