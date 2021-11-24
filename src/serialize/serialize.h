#pragma once
#include <ast/type.h>
#include <serde/IJsonObject.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <ast/function.h>
namespace luisa::compute {
using namespace toolhub::db;
class DeserVisitor {
private:
    template<typename T>
    struct Ser {
        IJsonDict *first;
        T *second;
        template<typename A, typename B>
        Ser(A &&a, B &&b)
            : first(a), second(b) {}
        ~Ser() {
            if (first)
                vengine_delete(second);
        }
    };
    vstd::HashMap<uint64, Function> callables;
    vstd::HashMap<uint64, Ser<Expression>> expr;
    vstd::HashMap<uint64, Ser<Statement>> stmt;

public:
    DeserVisitor(
        Function kernel,
        IJsonArray *exprArr,
        IJsonArray *stmtArr);
    Expression const *GetExpr(uint64) const;
    size_t exprCount() const { return expr.size(); }
    size_t stmtCount() const { return stmt.size(); }
    Statement const *GetStmt(uint64) const;
    void *Allocate(size_t) const;
    Function GetFunction(uint64) const;
    ~DeserVisitor();
    void GetExpr(vstd::function<void(Expression *)> const &func);
    void GetStmt(vstd::function<void(Statement *)> const &func);
};

class AstSerializer {
public:
    //Expr
    static vstd::unique_ptr<IJsonDict> Serialize(Type const &t, IJsonDatabase *db);
    static void DeSerialize(Type &t, IJsonDict *dict);
    static vstd::unique_ptr<IJsonDict> Serialize(TypeData const &t, IJsonDatabase *db);
    static void DeSerialize(TypeData &t, IJsonDict *dict);
    static vstd::unique_ptr<IJsonDict> Serialize(Expression const &t, IJsonDatabase *db);
    static vstd::unique_ptr<IJsonDict> Serialize(UnaryExpr const &t, IJsonDatabase *db);
    static void DeSerialize(UnaryExpr &t, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(BinaryExpr const &t, IJsonDatabase *db);
    static void DeSerialize(BinaryExpr &t, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(AccessExpr const &t, IJsonDatabase *db);
    static void DeSerialize(AccessExpr &t, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(MemberExpr const &t, IJsonDatabase *db);
    static void DeSerialize(MemberExpr &t, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(LiteralExpr const &t, IJsonDatabase *db);
    static void DeSerialize(LiteralExpr &t, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(RefExpr const &t, IJsonDatabase *db);
    static void DeSerialize(RefExpr &t, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(Variable const &t, IJsonDatabase *db);
    static void DeSerialize(Variable &t, IJsonDict *dict);
    static vstd::unique_ptr<IJsonDict> Serialize(ConstantData const &t, IJsonDatabase *db);
    static void DeSerialize(ConstantData &t, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(LiteralExpr::Value const &t, IJsonDatabase *db);
    static void DeSerialize(LiteralExpr::Value &t, IJsonDict *dict);
    static vstd::unique_ptr<IJsonDict> Serialize(ConstantExpr const &t, IJsonDatabase *db);
    static void DeSerialize(ConstantExpr &t, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(CallExpr const &t, IJsonDatabase *db);
    static void DeSerialize(CallExpr &t, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(CastExpr const &t, IJsonDatabase *db);
    static void DeSerialize(CastExpr &t, IJsonDict *dict, DeserVisitor const &evt);
    //Stmt
    static vstd::unique_ptr<IJsonDict> Serialize(Statement const &stmt, IJsonDatabase *db);
    static vstd::unique_ptr<IJsonDict> Serialize(BreakStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(BreakStmt &stmt, IJsonDict *dict, DeserVisitor const &evt) {}
    static vstd::unique_ptr<IJsonDict> Serialize(ContinueStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(ContinueStmt &stmt, IJsonDict *dict, DeserVisitor const &evt) {}
    static vstd::unique_ptr<IJsonDict> Serialize(ReturnStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(ReturnStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);
    static void Serialize(ScopeStmt const &stmt, IJsonDict *dict, IJsonDatabase *db);
    static vstd::unique_ptr<IJsonDict> Serialize(ScopeStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(ScopeStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(IfStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(IfStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(LoopStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(LoopStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(ExprStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(ExprStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(SwitchStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(SwitchStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(SwitchCaseStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(SwitchCaseStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(SwitchDefaultStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(SwitchDefaultStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(AssignStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(AssignStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(ForStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(ForStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);
    static vstd::unique_ptr<IJsonDict> Serialize(CommentStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(CommentStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);

    static vstd::unique_ptr<IJsonDict> Serialize(MetaStmt const &stmt, IJsonDatabase *db);
    static void DeSerialize(MetaStmt &stmt, IJsonDict *dict, DeserVisitor const &evt);
    static Expression *GenExpr(IJsonDict *dict, DeserVisitor &evt);
    static void DeserExpr(IJsonDict *dict, Expression *expr, DeserVisitor &evt);
    static vstd::unique_ptr<IJsonDict> SerExpr(IJsonDatabase *db, Expression const &expr);
    static Statement *GenStmt(IJsonDict *dict, DeserVisitor &evt);
    static void DeserStmt(IJsonDict *dict, Statement *stmt, DeserVisitor &evt);
    static vstd::unique_ptr<IJsonDict> SerStmt(IJsonDatabase *db, Statement const &s);
};
}// namespace luisa::compute