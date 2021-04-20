#pragma once
#include <ast/function.h>
#include <ast/expression.h>
#include <ast/statement.h>

namespace luisa::compute {
struct StmtVisitor;
class Function;
struct ExprVisitor;
class StringStateVisitor;
class Variable;
class Type;
class TypeVisitor;
class CodegenUtility {
public:
	static void GetCodegen(Function func, vengine::string& str);
	static void GetVariableName(Variable const& type, vengine::string& str);
	static void GetTypeName(Type const& type, vengine::string& str);
	static void GetFunctionDecl(Function func, vengine::string& str);
	static void PrintConstant(Function::ConstantBinding const& binding, vengine::string& result);

	static void ClearStructType();
	static void RegistStructType(Type const* type);
	static void IterateStructType(TypeVisitor* visitor);
	static void PrintStructType(vengine::string& str);
	static void GetFunctionName(CallExpr const* expr, vengine::string& result);
	static void PrintUniform(
		Function func,
		vengine::string& result);

	static size_t PrintGlobalVariables(
		std::initializer_list<std::span<const Variable>> values,
		vengine::string& result);
	static void SeparateVariables(
		Function func,
		vengine::vector<Variable const*>& buffers,
		vengine::vector<Variable const*>& textures,
		vengine::vector<Variable const*>& globalSRVValues,
		vengine::vector<Variable const*>& globalUAVValues,
		vengine::vector<Variable const*>& groupSharedValues);
};
class StringExprVisitor final : public ExprVisitor {

public:
	void visit(const UnaryExpr* expr) override;
	void visit(const BinaryExpr* expr) override;
	void visit(const MemberExpr* expr) override;
	void visit(const AccessExpr* expr) override;
	void visit(const LiteralExpr* expr) override;
	void visit(const RefExpr* expr) override;
	void visit(const CallExpr* expr) override;
	void visit(const CastExpr* expr) override;
	void visit(const ConstantExpr* expr) override;
	StringExprVisitor(vengine::string& str);
	~StringExprVisitor();

private:
	vengine::string* str;
	void BeforeVisit();
	void AfterVisit();
};
class StringStateVisitor final : public StmtVisitor {
public:
	void visit(const BreakStmt* state) override;
	void visit(const ContinueStmt* state) override;
	void visit(const ReturnStmt* state) override;
	void visit(const ScopeStmt* state) override;
	void visit(const DeclareStmt* state) override;
	void visit(const IfStmt* state) override;
	void visit(const WhileStmt* state) override;
	void visit(const ExprStmt* state) override;
	void visit(const SwitchStmt* state) override;
	void visit(const SwitchCaseStmt* state) override;
	void visit(const SwitchDefaultStmt* state) override;
	void visit(const AssignStmt* state) override;
	void visit(const ForStmt*) override;
	StringStateVisitor(vengine::string& str);
	~StringStateVisitor();

private:
	vengine::string* str;
};
}// namespace luisa::compute
