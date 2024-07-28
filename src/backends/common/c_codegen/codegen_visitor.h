#pragma once
#include <luisa/ast/expression.h>
#include <luisa/ast/statement.h>
#include "codegen_utils.h"
namespace luisa::compute {
class CodegenVisitor : public ExprVisitor, public StmtVisitor {
public:
    Clanguage_CodegenUtils &utils;
    vstd::StringBuilder &sb;
    void visit(const UnaryExpr *) override;
    void visit(const BinaryExpr *) override;
    void visit(const MemberExpr *) override;
    void visit(const AccessExpr *) override;
    void visit(const LiteralExpr *) override;
    void visit(const RefExpr *) override;
    void visit(const ConstantExpr *) override;
    void visit(const CallExpr *) override;
    void visit(const CastExpr *) override;
    void visit(const TypeIDExpr *) override;
    void visit(const StringIDExpr *) override;
    void visit(const FuncRefExpr *) override;

    void visit(const BreakStmt *) override;
    void visit(const ContinueStmt *) override;
    void visit(const ReturnStmt *) override;
    void visit(const ScopeStmt *) override;
    void visit(const IfStmt *) override;
    void visit(const LoopStmt *) override;
    void visit(const ExprStmt *) override;
    void visit(const SwitchStmt *) override;
    void visit(const SwitchCaseStmt *) override;
    void visit(const SwitchDefaultStmt *) override;
    void visit(const AssignStmt *) override;
    void visit(const ForStmt *) override;
    void visit(const CommentStmt *) override;
    void visit(const RayQueryStmt *) override;
    void visit(const PrintStmt *) override;
    CodegenVisitor(
        vstd::StringBuilder &sb,
        luisa::string_view entry_name,
        Clanguage_CodegenUtils &utils,
        Function func);
    ~CodegenVisitor();
};
};// namespace luisa::compute