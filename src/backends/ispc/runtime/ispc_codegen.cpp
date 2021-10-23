#pragma vengine_package ispc_vsproject

#include "ispc_codegen.h"
namespace lc::ispc {
void StringExprVisitor::visit(const UnaryExpr *expr) {}
void StringExprVisitor::visit(const BinaryExpr *expr) {}
void StringExprVisitor::visit(const MemberExpr *expr) {}
void StringExprVisitor::visit(const AccessExpr *expr) {}
void StringExprVisitor::visit(const LiteralExpr *expr) {}
void StringExprVisitor::visit(const RefExpr *expr) {}
void StringExprVisitor::visit(const CallExpr *expr) {}
void StringExprVisitor::visit(const CastExpr *expr) {}
void StringExprVisitor::visit(const ConstantExpr *expr) {}
StringExprVisitor::StringExprVisitor(std::string &str) {}
StringExprVisitor::~StringExprVisitor() {}

void StringStateVisitor::visit(const BreakStmt *state) {}
void StringStateVisitor::visit(const ContinueStmt *state) {}
void StringStateVisitor::visit(const ReturnStmt *state) {}
void StringStateVisitor::visit(const ScopeStmt *state) {}
void StringStateVisitor::visit(const DeclareStmt *state) {}
void StringStateVisitor::visit(const IfStmt *state) {}
void StringStateVisitor::visit(const LoopStmt *state) {}
void StringStateVisitor::visit(const ExprStmt *state) {}
void StringStateVisitor::visit(const SwitchStmt *state) {}
void StringStateVisitor::visit(const SwitchCaseStmt *state) {}
void StringStateVisitor::visit(const SwitchDefaultStmt *state) {}
void StringStateVisitor::visit(const AssignStmt *state) {}
void StringStateVisitor::visit(const ForStmt *) {}
StringStateVisitor::StringStateVisitor(std::string &str, Function func) {}
StringStateVisitor::~StringStateVisitor() {}
}// namespace lc::ispc