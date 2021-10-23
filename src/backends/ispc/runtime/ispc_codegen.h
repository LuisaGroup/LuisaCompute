#pragma once


#include <vstl/Common.h>
#include <ast/function.h>
#include <ast/expression.h>
#include <ast/statement.h>
using namespace luisa;
using namespace luisa::compute;
namespace lc::ispc {
class StringExprVisitor final : public ExprVisitor {

public:
    void visit(const UnaryExpr *expr) override;
    void visit(const BinaryExpr *expr) override;
    void visit(const MemberExpr *expr) override;
    void visit(const AccessExpr *expr) override;
    void visit(const LiteralExpr *expr) override;
    void visit(const RefExpr *expr) override;
    void visit(const CallExpr *expr) override;
    void visit(const CastExpr *expr) override;
    void visit(const ConstantExpr *expr) override;
    StringExprVisitor(std::string &str);
    ~StringExprVisitor();

protected:
    std::string *str;
};
class StringStateVisitor final : public StmtVisitor {
public:
    void visit(const BreakStmt *state) override;
    void visit(const ContinueStmt *state) override;
    void visit(const ReturnStmt *state) override;
    void visit(const ScopeStmt *state) override;
    void visit(const DeclareStmt *state) override;
    void visit(const IfStmt *state) override;
    void visit(const LoopStmt *state) override;
    void visit(const ExprStmt *state) override;
    void visit(const SwitchStmt *state) override;
    void visit(const SwitchCaseStmt *state) override;
    void visit(const SwitchDefaultStmt *state) override;
    void visit(const AssignStmt *state) override;
    void visit(const ForStmt *) override;
    StringStateVisitor(std::string &str, Function func);
    ~StringStateVisitor();

protected:
    std::string *str;
    Function func;
};
}// namespace lc::ispc