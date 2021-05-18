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
	static void GetCodegen(Function func, vengine::string& str, HashMap<uint, size_t>& varOffsets, size_t& cbufferSize);
	static void GetVariableName(Variable const& type, vengine::string& str);
	static void GetVariableName(Variable::Tag type, uint id, vengine::string& str);
	static void GetVariableName(Type::Tag type, uint id, vengine::string& str);
	static void GetTypeName(Type const& type, vengine::string& str, bool isWritable = false);
	static void GetFunctionDecl(Function func, vengine::string& str);
	static void PrintConstant(Function::ConstantBinding const& binding, vengine::string& result);

	static void ClearStructType();
	static void RegistStructType(Type const* type);
	static void IterateStructType(TypeVisitor* visitor);
	static void PrintStructType(vengine::string& str);
	static void GetFunctionName(CallExpr const* expr, vengine::string& result, Runnable<void()>&& func);
	static void PrintUniform(
		Function func,
		vengine::string& result);

	static size_t PrintGlobalVariables(
		Function func,
		std::initializer_list<std::span<const Variable>> values,
		HashMap<uint, size_t>& varOffsets,
		vengine::string& result);
	static void SeparateVariables(
		Function func,
		vengine::vector<Variable const*>& buffers,
		vengine::vector<Variable const*>& textures,
		vengine::vector<Variable const*>& globalSRVValues,
		vengine::vector<Variable const*>& globalUAVValues,
		vengine::vector<Variable const*>& groupSharedValues);
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
	StringExprVisitor(vengine::string& str);
	~StringExprVisitor();

protected:
	vengine::string* str;
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
	StringStateVisitor(vengine::string& str, Function func);
	~StringStateVisitor();

protected:
	vengine::string* str;
	Function func;
};
}// namespace luisa::compute
