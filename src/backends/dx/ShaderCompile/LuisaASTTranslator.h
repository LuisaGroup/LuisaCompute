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
	static void GetCodegen(Function func, vstd::string& str, HashMap<uint, size_t>& varOffsets, size_t& cbufferSize);
	static void GetVariableName(Variable const& type, vstd::string& str);
	static void GetVariableName(Variable::Tag type, uint id, vstd::string& str);
	static void GetVariableName(Type::Tag type, uint id, vstd::string& str);
	static void GetTypeName(Type const& type, vstd::string& str, bool isWritable = false);
	static void GetFunctionDecl(Function func, vstd::string& str);
	static void PrintConstant(Function::ConstantBinding const& binding, vstd::string& result);

	static void ClearStructType();
	static void RegistStructType(Type const* type);
	static void IterateStructType(TypeVisitor* visitor);
	static void PrintStructType(vstd::string& str);
	static void GetFunctionName(CallExpr const* expr, vstd::string& result, Runnable<void()>&& func);
	static void PrintUniform(
		Function func,
		vstd::string& result);

	static size_t PrintGlobalVariables(
		Function func,
		std::initializer_list<std::span<const Variable>> values,
		HashMap<uint, size_t>& varOffsets,
		vstd::string& result);
	static void SeparateVariables(
		Function func,
		vstd::vector<Variable const*>& buffers,
		vstd::vector<Variable const*>& textures,
		vstd::vector<Variable const*>& globalSRVValues,
		vstd::vector<Variable const*>& globalUAVValues,
		vstd::vector<Variable const*>& groupSharedValues);
};
class StringExprVisitor : public ExprVisitor {

public:
	virtual void visit(const UnaryExpr* expr) override;
	virtual void visit(const BinaryExpr* expr) override;
	virtual void visit(const MemberExpr* expr) override;
	virtual void visit(const AccessExpr* expr) override;
	virtual void visit(const LiteralExpr* expr) override;
	virtual void visit(const RefExpr* expr) override;
	virtual void visit(const CallExpr* expr) override;
	virtual void visit(const CastExpr* expr) override;
	virtual void visit(const ConstantExpr* expr) override;
	StringExprVisitor(vstd::string& str);
	~StringExprVisitor();

protected:
	vstd::string* str;
	void BeforeVisit();
	void AfterVisit();
};
class StringStateVisitor : public StmtVisitor {
public:
	virtual void visit(const BreakStmt* state) override;
	virtual void visit(const ContinueStmt* state) override;
	virtual void visit(const ReturnStmt* state) override;
	virtual void visit(const ScopeStmt* state) override;
	virtual void visit(const DeclareStmt* state) override;
	virtual void visit(const IfStmt* state) override;
	virtual void visit(const WhileStmt* state) override;
	virtual void visit(const ExprStmt* state) override;
	virtual void visit(const SwitchStmt* state) override;
	virtual void visit(const SwitchCaseStmt* state) override;
	virtual void visit(const SwitchDefaultStmt* state) override;
	virtual void visit(const AssignStmt* state) override;
	virtual void visit(const ForStmt*) override;
	StringStateVisitor(vstd::string& str, Function func);
	~StringStateVisitor();

protected:
	vstd::string* str;
	Function func;
};
}// namespace luisa::compute
