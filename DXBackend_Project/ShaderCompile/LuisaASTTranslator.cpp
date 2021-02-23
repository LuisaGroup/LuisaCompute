#include <ast/interface.h>
#include "../Common/Common.h"
#include "../Common/VObject.h"
#include "LuisaASTTranslator.h"
namespace luisa::compute {
class StringExprVisitor final : public ExprVisitor {
public:
	void visit(const UnaryExpr* expr) override {
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
	void visit(const BinaryExpr* expr) override {
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
			expr->lhs()->accept(vis);
			str += vis.ToString();
			str += ',';
			expr->rhs()->accept(vis);
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
	void visit(const MemberExpr* expr) override {
		expr->accept(*this);
		str += ".v";
		str += vengine::to_string(expr->member_index());
	}
	void visit(const AccessExpr* expr) override {
		expr->accept(*this);
		str += '[';
		StringExprVisitor vis;
		expr->index()->accept(vis);
		str += vis.ToString();
		str += ']';
	}
	void visit(const ValueExpr* expr) override {
		ValueExpr::Value const& value = expr->value();
		//TODO: After finish type system
		switch (value.index()) {
			case 0: {
				Variable const& v = std::get<0>(value);
			} break;
			case 1: {
				bool const& v = std::get<1>(value);
			} break;
			case 2: {
				float const& v = std::get<2>(value);
			} break;
			case 3: {
				int8_t const& v = std::get<3>(value);
			} break;
			case 4: {
				uint8_t const& v = std::get<4>(value);
			} break;
			case 5: {
				int16_t const& v = std::get<5>(value);
			} break;
			case 6: {
				uint16_t const& v = std::get<6>(value);
			} break;
			case 7: {
				int32_t const& v = std::get<7>(value);
			} break;
			case 8: {
				uint32_t const& v = std::get<8>(value);
			} break;
			case 9: {
				bool2 const& v = std::get<9>(value);
			} break;
			case 10: {
				float2 const& v = std::get<10>(value);
			} break;
			case 11: {
				char2 const& v = std::get<11>(value);
			} break;
			case 12: {
				uchar2 const& v = std::get<12>(value);
			} break;
			case 13: {
				short2 const& v = std::get<13>(value);
			} break;
			case 14: {
				ushort2 const& v = std::get<14>(value);
			} break;
			case 15: {
				int2 const& v = std::get<15>(value);
			} break;
			case 16: {
				uint2 const& v = std::get<16>(value);
			} break;
			case 17: {
				bool3 const& v = std::get<17>(value);
			} break;
			case 18: {
				float3 const& v = std::get<18>(value);
			} break;
			case 19: {
				char3 const& v = std::get<19>(value);
			} break;
			case 20: {
				uchar3 const& v = std::get<20>(value);
			} break;
			case 21: {
				short3 const& v = std::get<21>(value);
			} break;
			case 22: {
				ushort3 const& v = std::get<22>(value);
			} break;
			case 23: {
				int3 const& v = std::get<23>(value);
			} break;
			case 24: {
				uint3 const& v = std::get<24>(value);
			} break;
			case 25: {
				bool4 const& v = std::get<25>(value);
			} break;
			case 26: {
				float4 const& v = std::get<26>(value);
			} break;
			case 27: {
				char4 const& v = std::get<27>(value);
			} break;
			case 28: {
				uchar4 const& v = std::get<28>(value);
			} break;
			case 29: {
				short4 const& v = std::get<29>(value);
			} break;
			case 30: {
				ushort4 const& v = std::get<30>(value);
			} break;
			case 31: {
				int4 const& v = std::get<31>(value);
			} break;
			case 32: {
				uint4 const& v = std::get<32>(value);
			} break;
			case 33: {
				float3x3 const& v = std::get<33>(value);
			} break;
			case 34: {
				float4x4 const& v = std::get<34>(value);
			} break;
		}
	}
	void visit(const CallExpr* expr) override {
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
	void visit(const CastExpr* expr) override {
		//TODO: After finish type system
	}
	vengine::string const& ToString() const {
		return str;
	}
	StringExprVisitor() {}
	StringExprVisitor(
		StringExprVisitor&& v)
		: str(std::move(v.str)) {
	}

private:
	vengine::string str;
	void BeforeVisit() {
		str = '(';
	}
	void AfterVisit() {
		str += ')';
	}
};
}// namespace luisa::compute