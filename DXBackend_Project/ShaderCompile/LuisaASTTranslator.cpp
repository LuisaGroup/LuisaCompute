#include <ast/interface.h>
#include "../Common/Common.h"
#include "../Common/VObject.h"
#include "LuisaASTTranslator.h"
namespace luisa::compute {

void StringExprVisitor::visit(const UnaryExpr* expr) {
	BeforeVisit();
	switch (expr->op()) {
		case UnaryOp::PLUS://+x
			str += '+';
			break;
		case UnaryOp::MINUS://-x
			str += '-';
			break;
		case UnaryOp::NOT://!x
			str += '!';
			break;
		case UnaryOp::BIT_NOT://~x
			str += '~';
			break;
	}
	StringExprVisitor vis;
	expr->accept(vis);
	str += vis.ToString();
	AfterVisit();
}
void StringExprVisitor::visit(const BinaryExpr* expr) {
	auto IsMulFuncCall = [&]() -> bool {
		if (expr->op() == BinaryOp::MUL) {
			if ((expr->lhs()->type()->is_matrix() && (!expr->rhs()->type()->is_scalar()))
				|| (expr->rhs()->type()->is_matrix() && (!expr->lhs()->type()->is_scalar()))) {
				return true;
			}
		}
		return false;
	};
	StringExprVisitor vis;
	if (IsMulFuncCall()) {
		str = "mul(";
		expr->rhs()->accept(vis);//Reverse matrix
		str += vis.ToString();
		str += ',';
		expr->lhs()->accept(vis);
		str += vis.ToString();
		str += ')';

	} else {
		BeforeVisit();
		expr->lhs()->accept(vis);
		str += vis.ToString();
		expr->rhs()->accept(vis);
		str += vis.ToString();
		switch (expr->op()) {
			case BinaryOp::ADD:
				str += '+';
				break;
			case BinaryOp::SUB:
				str += '-';
				break;
			case BinaryOp::MUL:
				str += '*';
				break;
			case BinaryOp::DIV:
				str += '/';
				break;
			case BinaryOp::MOD:
				str += '%';
				break;
			case BinaryOp::BIT_AND:
				str += '&';
				break;
			case BinaryOp::BIT_OR:
				str += '|';
				break;
			case BinaryOp::BIT_XOR:
				str += '^';
				break;
			case BinaryOp::SHL:
				str += "<<";
				break;
			case BinaryOp::SHR:
				str += ">>";
				break;
			case BinaryOp::AND:
				str += "&&";
				break;
			case BinaryOp::OR:
				str += "||";
				break;
			case BinaryOp::LESS:
				str += '<';
				break;
			case BinaryOp::GREATER:
				str += '>';
				break;
			case BinaryOp::LESS_EQUAL:
				str += "<=";
				break;
			case BinaryOp::GREATER_EQUAL:
				str += ">=";
				break;
			case BinaryOp::EQUAL:
				str += "==";
				break;
			case BinaryOp::NOT_EQUAL:
				str += "!=";
				break;
		}
		AfterVisit();
	}
}
void StringExprVisitor::visit(const MemberExpr* expr) {
	expr->self()->accept(*this);
	str += ".v";
	str += vengine::to_string(expr->member_index());
}
void StringExprVisitor::visit(const AccessExpr* expr) {
	expr->range()->accept(*this);
	str += '[';
	StringExprVisitor vis;
	expr->index()->accept(vis);
	str += vis.ToString();
	str += ']';
}
void StringExprVisitor::visit(const RefExpr* expr) {
	Variable v = expr->variable();
	if (v.type()->is_vector() && v.type()->element()->size() < 4) {
		//TODO
	} else {
		str = std::move(GetVariableName(v));
	}
}
void StringExprVisitor::visit(const LiteralExpr* expr) {
	LiteralExpr::Value const& value = expr->value();
	switch (value.index()) {
		case 0: {
			bool const& v = std::get<0>(value);
			str = v ? "true" : "false";
		} break;
		case 1: {
			float const& v = std::get<1>(value);
			str = vengine::to_string(v);
		} break;
		case 2: {
			int8_t const& v = std::get<2>(value);
			str = vengine::to_string(v);
		} break;
		case 3: {
			uint8_t const& v = std::get<3>(value);
			str = vengine::to_string(v);
		} break;
		case 4: {
			int16_t const& v = std::get<4>(value);
			str = vengine::to_string(v);
		} break;
		case 5: {
			uint16_t const& v = std::get<5>(value);
			str = vengine::to_string(v);
		} break;
		case 6: {
			int32_t const& v = std::get<6>(value);
			str = vengine::to_string(v);
		} break;
		case 7: {
			uint32_t const& v = std::get<7>(value);
			str = vengine::to_string(v);
		} break;
		case 8: {
			auto const& v = std::get<8>(value);
			str = "bool2(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += ')';
		} break;
		case 9: {
			auto const& v = std::get<9>(value);
			str = "float2(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += ')';
		} break;
		case 10: {
			auto const& v = std::get<10>(value);
			str = "int2(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += ')';
		} break;
		case 11: {
			auto const& v = std::get<11>(value);
			str = "uint2(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += ')';
		} break;
		case 12: {
			auto const& v = std::get<12>(value);
			str = "int2(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += ')';
		} break;
		case 13: {
			auto const& v = std::get<13>(value);
			str = "uint2(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += ')';
		} break;
		case 14: {
			auto const& v = std::get<14>(value);
			str = "int2(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += ')';
		} break;
		case 15: {
			auto const& v = std::get<15>(value);
			str = "uint2(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += ')';
		} break;
		case 16: {
			auto const& v = std::get<16>(value);
			str = "bool3(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += ')';
		} break;
		case 17: {
			auto const& v = std::get<17>(value);
			str = "float3(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += ')';
		} break;
		case 18: {
			auto const& v = std::get<18>(value);
			str = "int3(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += ')';
		} break;
		case 19: {
			auto const& v = std::get<19>(value);
			str = "uint3(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += ')';
		} break;
		case 20: {
			auto const& v = std::get<20>(value);
			str = "int3(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += ')';
		} break;
		case 21: {
			auto const& v = std::get<21>(value);
			str = "uint3(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += ')';
		} break;
		case 22: {
			auto const& v = std::get<22>(value);
			str = "int3(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += ')';
		} break;
		case 23: {
			auto const& v = std::get<23>(value);
			str = "uint3(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += ')';
		} break;
		case 24: {
			auto const& v = std::get<24>(value);
			str = "bool4(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += vengine::to_string(v.w);
			str += ')';
		} break;
		case 25: {
			auto const& v = std::get<25>(value);
			str = "float4(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += vengine::to_string(v.w);
			str += ')';
		} break;
		case 26: {
			auto const& v = std::get<26>(value);
			str = "int4(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += vengine::to_string(v.w);
			str += ')';
		} break;
		case 27: {
			auto const& v = std::get<27>(value);
			str = "uint4(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += vengine::to_string(v.w);
			str += ')';
		} break;
		case 28: {
			auto const& v = std::get<28>(value);
			str = "int4(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += vengine::to_string(v.w);
			str += ')';
		} break;
		case 29: {
			auto const& v = std::get<29>(value);
			str = "uint4(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += vengine::to_string(v.w);
			str += ')';
		} break;
		case 30: {
			auto const& v = std::get<30>(value);
			str = "int4(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += vengine::to_string(v.w);
			str += ')';
		} break;
		case 31: {
			auto const& v = std::get<31>(value);
			str = "uint4(";
			str += vengine::to_string(v.x);
			str += vengine::to_string(v.y);
			str += vengine::to_string(v.z);
			str += vengine::to_string(v.w);
			str += ')';
		} break;
		case 32: {
			float3x3 const& v = std::get<32>(value);
			str = "float3x3(";
			float3 c0 = v[0];
			float3 c1 = v[1];
			float3 c2 = v[2];
			str += vengine::to_string(c0.x) + ',' + vengine::to_string(c1.x) + ',' + vengine::to_string(c2.x) + ',';
			str += vengine::to_string(c0.y) + ',' + vengine::to_string(c1.y) + ',' + vengine::to_string(c2.y) + ',';
			str += vengine::to_string(c0.z) + ',' + vengine::to_string(c1.z) + ',' + vengine::to_string(c2.z) + ')';
		} break;
		case 33: {
			float4x4 const& v = std::get<33>(value);
			str = "float4x4(";
			float4 c0 = v[0];
			float4 c1 = v[1];
			float4 c2 = v[2];
			float4 c3 = v[3];
			str += vengine::to_string(c0.x) + ',' + vengine::to_string(c1.x) + ',' + vengine::to_string(c2.x) + ',' + vengine::to_string(c3.x) + ',';
			str += vengine::to_string(c0.y) + ',' + vengine::to_string(c1.y) + ',' + vengine::to_string(c2.y) + ',' + vengine::to_string(c3.y) + ',';
			str += vengine::to_string(c0.z) + ',' + vengine::to_string(c1.z) + ',' + vengine::to_string(c2.z) + ',' + vengine::to_string(c3.z) + ')';
		} break;
	}
}
void StringExprVisitor::visit(const CallExpr* expr) {
	str.clear();
	str.push_back_all(expr->name().data(), expr->name().size());
	str += '(';
	auto&& args = expr->arguments();
	StringExprVisitor vis;
	for (auto&& i : args) {
		i->accept(vis);
		str += vis.ToString();
		str += ',';
	}
	str[str.size() - 1] = ')';
}
void StringExprVisitor::visit(const CastExpr* expr) {
	//TODO: After finish type system
}
vengine::string const& StringExprVisitor::ToString() const {
	return str;
}
StringExprVisitor::StringExprVisitor() {}
StringExprVisitor::StringExprVisitor(
	StringExprVisitor&& v)
	: str(std::move(v.str)) {
}

StringExprVisitor::~StringExprVisitor() {
}

void StringExprVisitor::BeforeVisit() {
	str = '(';
}
void StringExprVisitor::AfterVisit() {
	str += ')';
}

void StringStateVisitor::visit(const BreakStmt* state) {
	str = "break;\n";
}

void StringStateVisitor::visit(const ContinueStmt* state) {
	str = "continue;\n";
}

void StringStateVisitor::visit(const ReturnStmt* state) {
	if (state->expression()) {
		str = "return ";
		StringExprVisitor exprVis;
		state->expression()->accept(exprVis);
		str += exprVis.ToString();
		str += ";\n";
	} else {
		str = "return;\n";
	}
}

void StringStateVisitor::visit(const ScopeStmt* state) {
	str = "{\n";
	StringStateVisitor sonVisitor;
	for (auto&& i : state->statements()) {
		i->accept(sonVisitor);
		str += sonVisitor.ToString();
	}
	str += "}\n";
}

void StringStateVisitor::visit(const DeclareStmt* state) {
	auto var = state->variable();
	auto varName = var.type()->description();
	str.clear();
	str.push_back_all(varName.data(), varName.size());
	str += ' ';
	str += GetVariableName(var);
	//TODO: Different Declare in Fxxking HLSL
}

void StringStateVisitor::visit(const IfStmt* state) {
	StringExprVisitor exprVisitor;
	StringStateVisitor stateVisitor;
	str = "if (";
	state->condition()->accept(exprVisitor);
	str += exprVisitor.ToString();
	str += ')';
	state->true_branch()->accept(stateVisitor);
	str += stateVisitor.ToString();
	str += "else";
	state->false_branch()->accept(stateVisitor);
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const WhileStmt* state) {
	StringStateVisitor stateVisitor;
	StringExprVisitor exprVisitor;

	str = "while (";
	state->condition()->accept(exprVisitor);
	str += exprVisitor.ToString();
	str += ")";
	state->body()->accept(stateVisitor);
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const ExprStmt* state) {
	StringExprVisitor exprVisitor;
	state->expression()->accept(exprVisitor);
	str = std::move(exprVisitor.str);
	str += ";\n";
}

void StringStateVisitor::visit(const SwitchStmt* state) {
	str = "switch (";
	StringStateVisitor stateVisitor;
	StringExprVisitor exprVisitor;
	state->expression()->accept(exprVisitor);
	state->body()->accept(stateVisitor);
	str += exprVisitor.ToString();
	str += ")";
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const SwitchCaseStmt* state) {
	str = "case ";
	StringStateVisitor stateVisitor;
	StringExprVisitor exprVisitor;
	state->expression()->accept(exprVisitor);
	state->body()->accept(stateVisitor);
	str += exprVisitor.ToString();
	str += ":\n";
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const SwitchDefaultStmt* state) {
	str = "default:\n";
	StringStateVisitor stateVisitor;
	state->body()->accept(stateVisitor);
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const AssignStmt* state) {
	StringExprVisitor exprVisitor;
	state->lhs()->accept(exprVisitor);
	str = exprVisitor.ToString();
	switch (state->op()) {
		case AssignOp::ASSIGN:
			str += '=';
			break;
		case AssignOp::ADD_ASSIGN:
			str += "+=";
			break;
		case AssignOp::SUB_ASSIGN:
			str += "-=";
			break;
		case AssignOp::MUL_ASSIGN:
			str += "*=";
			break;
		case AssignOp::DIV_ASSIGN:
			str += "/=";
			break;
		case AssignOp::MOD_ASSIGN:
			str += "%=";
			break;
		case AssignOp::BIT_AND_ASSIGN:
			str += "&=";
			break;
		case AssignOp::BIT_OR_ASSIGN:
			str += "|=";
			break;
		case AssignOp::BIT_XOR_ASSIGN:
			str += "^=";
			break;
		case AssignOp::SHL_ASSIGN:
			str += "<<=";
			break;
		case AssignOp::SHR_ASSIGN:
			str += ">>=";
			break;
	}
	state->rhs()->accept(exprVisitor);
	str += exprVisitor.ToString();
	str += ";\n";
}

StringStateVisitor::StringStateVisitor() {
}

StringStateVisitor::~StringStateVisitor() {
}

StringStateVisitor::StringStateVisitor(StringStateVisitor&& st)
	: str(std::move(str)) {
}

vengine::string const& StringStateVisitor::ToString() const {
	return str;
}

vengine::string GetVariableName(Variable const& type) {
	return "v" + type.uid();
}

}// namespace luisa::compute