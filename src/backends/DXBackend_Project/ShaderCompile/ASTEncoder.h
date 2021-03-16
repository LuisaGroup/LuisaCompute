#pragma once
#include <ast/function.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <Common/Common.h>
namespace luisa::compute {
class ASTExprEncoder : public ExprVisitor {
public:
	void visit(const UnaryExpr*) override;
	void visit(const BinaryExpr*) override;
	void visit(const MemberExpr*) override;
	void visit(const AccessExpr*) override;
	void visit(const LiteralExpr*) override;
	void visit(const RefExpr*) override;
	void visit(const ConstantExpr*) override;
	void visit(const CallExpr*) override;
	void visit(const CastExpr*) override;
	vengine::vector<uint8_t>* const data;
	ASTExprEncoder(vengine::vector<uint8_t>* const data) : data(data) {}
	~ASTExprEncoder();

private:
	template <typename T>
	void Push(T const& data) {
		static_assert(!std::is_pointer_v<T>, "Cannot be pointer!");
		Push(reinterpret_cast<uint8_t const*>(&data), sizeof(T));
	}
	void Push(uint8_t const* data, size_t sz);
};
class ASTStmtEncoder : public StmtVisitor {
public:
	void visit(const BreakStmt*) override;
	void visit(const ContinueStmt*) override;
	void visit(const ReturnStmt*) override;
	void visit(const ScopeStmt*) override;
	void visit(const DeclareStmt*) override;
	void visit(const IfStmt*) override;
	void visit(const WhileStmt*) override;
	void visit(const ExprStmt*) override;
	void visit(const SwitchStmt*) override;
	void visit(const SwitchCaseStmt*) override;
	void visit(const SwitchDefaultStmt*) override;
	void visit(const AssignStmt*) override;
	void visit(const ForStmt*) override;
	vengine::vector<uint8_t>* const data;
	ASTStmtEncoder(vengine::vector<uint8_t>* const data) : data(data) {}
	~ASTStmtEncoder();

private:
	template<typename T>
	void Push(T data) {
		static_assert(!std::is_pointer_v<T>, "Cannot be pointer!");
		Push(reinterpret_cast<uint8_t const*>(&data), sizeof(T));
	}
	void Push(uint8_t const* data, size_t sz);
};
}// namespace luisa::compute