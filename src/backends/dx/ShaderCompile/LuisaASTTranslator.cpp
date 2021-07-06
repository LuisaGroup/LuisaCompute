#include <core/clock.h>
#include <ast/interface.h>
#include <ast/constant_data.h>
#include <Common/Common.h>
#include <Common/VObject.h>
#include <Common/linq.h>
#include <ShaderCompile/LuisaASTTranslator.h>

namespace luisa::compute {
static bool _IsVarWritable(Function func, Variable i) {
	return ((uint)func.variable_usage(i.uid()) & (uint)Variable::Usage::WRITE) != 0;
}
void CodegenUtility::GetCodegen(Function func, vengine::string& str, HashMap<uint, size_t>& varOffsets, size_t& cbufferSize) {
	{
		vengine::string function_buffer;
		function_buffer.reserve(65535 * 4);
		str.reserve(65535 * 4);
		str << "#include \"Include.cginc\"\n"_sv;
		CodegenUtility::ClearStructType();

		for (auto cust : func.custom_callables()) {
			auto&& callable = Function::callable(cust);
			StringStateVisitor vis(function_buffer, callable);
			CodegenUtility::GetFunctionDecl(callable, function_buffer);
			callable.body()->accept(vis);
		}

		CodegenUtility::GetFunctionDecl(func, function_buffer);
		{
			StringStateVisitor vis(function_buffer, func);
			func.body()->accept(vis);
		}
		CodegenUtility::PrintStructType(str);

		for (auto& i : func.constants()) {
			CodegenUtility::PrintConstant(i, str);
		}
		for (auto& i : func.shared_variables()) {
			str << "groupshared "_sv;
			CodegenUtility::GetTypeName(*i.type(), str, _IsVarWritable(func, i));
			str << ' ';
			CodegenUtility::GetVariableName(i, str);
			str << '['
				<< vengine::to_string(i.type()->dimension())
				<< "];\n"_sv;
		}
		CodegenUtility::PrintUniform(func, str);
		cbufferSize = CodegenUtility::PrintGlobalVariables(
			func,
			{func.builtin_variables(),
			 func.arguments()},
			varOffsets,
			str);
		str << function_buffer;
	}
	str << "[numthreads(8,8,1)]\nvoid CSMain(uint3 thd_id:SV_GROUPTHREADID,uint3 blk_id:SV_GROUPID,uint3 dsp_id:SV_DISPATCHTHREADID){\n_kernel_"_sv;
	vengine::to_string(func.uid(), str);
	str += "(thd_id,blk_id,dsp_id);\n}\n";
}

template<typename T>
struct GetSpanType;
template<typename T>
struct GetSpanType<std::span<T const>> {
	using Type = T;
};
template<typename T>
struct GetName {
	static vengine::string Get() {
		if constexpr (std::is_same_v<bool, T>) {
			return "bool"_sv;
		} else if constexpr (std::is_same_v<float, T>) {
			return "float"_sv;
		} else if constexpr (std::is_same_v<int, T>) {
			return "int"_sv;
		} else if constexpr (std::is_same_v<uint, T>) {
			return "uint"_sv;
		} else {
			return "unknown"_sv;
		}
	}
};
template<typename EleType, size_t N>
struct GetName<Vector<EleType, N>> {
	static vengine::string Get() {
		return GetName<EleType>::Get() + vengine::to_string(N);
	}
};
template<size_t N>
struct GetName<Matrix<N>> {
	static vengine::string Get() {
		auto num = vengine::to_string(N);
		return GetName<float>::Get() + num + 'x' + num;
	}
};
template<typename T>
struct PrintValue {
	void operator()(T const& v, vengine::string& str) {
		vengine::to_string((int)v, str);
	}
};
template<>
struct PrintValue<float> {
	void operator()(float const& v, vengine::string& str) {
		vengine::to_string(v, str);
	}
};

template<typename EleType, size_t N>
struct PrintValue<Vector<EleType, N>> {
	using T = typename Vector<EleType, N>;
	void operator()(T const& v, vengine::string& varName) {
		varName += GetName<T>::Get();
		varName += '(';
		for (size_t i = 0; i < N; ++i) {
			vengine::to_string(v[i], varName);
			varName += ',';
		}
		varName[varName.size() - 1] = ')';
	}
};

template<size_t N>
struct PrintValue<Matrix<N>> {
	using T = Matrix<N>;
	using EleType = float;
	void operator()(T const& v, vengine::string& varName) {
		varName += GetName<T>::Get();
		varName += "("_sv;
		PrintValue<Vector<EleType, N>> vecPrinter;
		for (size_t i = 0; i < N; ++i) {
			vecPrinter(v[i], varName);
			varName += ',';
		}
		varName[varName.size() - 1] = ')';
	}
};
void StringExprVisitor::visit(const UnaryExpr* expr) {
	BeforeVisit();
	switch (expr->op()) {
		case UnaryOp::PLUS://+x
			(*str) += '+';
			break;
		case UnaryOp::MINUS://-x
			(*str) += '-';
			break;
		case UnaryOp::NOT://!x
			(*str) += '!';
			break;
		case UnaryOp::BIT_NOT://~x
			(*str) += '~';
			break;
	}
	StringExprVisitor vis(*str);
	expr->operand()->accept(vis);
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
	StringExprVisitor vis(*str);
	if (IsMulFuncCall()) {
		(*str) += "mul("_sv;
		expr->rhs()->accept(vis);//Reverse matrix
		(*str) += ',';
		expr->lhs()->accept(vis);
		(*str) += ')';

	} else {
		BeforeVisit();
		expr->lhs()->accept(vis);
		switch (expr->op()) {
			case BinaryOp::ADD:
				(*str) += '+';
				break;
			case BinaryOp::SUB:
				(*str) += '-';
				break;
			case BinaryOp::MUL:
				(*str) += '*';
				break;
			case BinaryOp::DIV:
				(*str) += '/';
				break;
			case BinaryOp::MOD:
				(*str) += '%';
				break;
			case BinaryOp::BIT_AND:
				(*str) += '&';
				break;
			case BinaryOp::BIT_OR:
				(*str) += '|';
				break;
			case BinaryOp::BIT_XOR:
				(*str) += '^';
				break;
			case BinaryOp::SHL:
				(*str) += "<<"_sv;
				break;
			case BinaryOp::SHR:
				(*str) += ">>"_sv;
				break;
			case BinaryOp::AND:
				(*str) += "&&"_sv;
				break;
			case BinaryOp::OR:
				(*str) += "||"_sv;
				break;
			case BinaryOp::LESS:
				(*str) += '<';
				break;
			case BinaryOp::GREATER:
				(*str) += '>';
				break;
			case BinaryOp::LESS_EQUAL:
				(*str) += "<="_sv;
				break;
			case BinaryOp::GREATER_EQUAL:
				(*str) += ">="_sv;
				break;
			case BinaryOp::EQUAL:
				(*str) += "=="_sv;
				break;
			case BinaryOp::NOT_EQUAL:
				(*str) += "!="_sv;
				break;
		}
		expr->rhs()->accept(vis);
		AfterVisit();
	}
}
void StringExprVisitor::visit(const MemberExpr* expr) {
	expr->self()->accept(*this);
	if (expr->is_swizzle()) {
		char const* xyzw = "xyzw";
		(*str) << '.';
		for (auto i : vengine::range(expr->swizzle_size())) {
			(*str) << xyzw[i];
		}
	} else {
		(*str) += ".v"_sv;
		vengine::to_string(expr->member_index(), (*str));
	}
}
void StringExprVisitor::visit(const AccessExpr* expr) {
	expr->range()->accept(*this);
	(*str) += '[';
	StringExprVisitor vis(*str);
	expr->index()->accept(vis);
	(*str) += ']';
}
void StringExprVisitor::visit(const RefExpr* expr) {
	Variable v = expr->variable();
	CodegenUtility::RegistStructType(v.type());
	CodegenUtility::GetVariableName(v, *str);
}
void StringExprVisitor::visit(const LiteralExpr* expr) {
	LiteralExpr::Value const& value = expr->value();
	std::visit([&](auto&& value) -> void {
		using T = std::remove_cvref_t<decltype(value)>;
		PrintValue<T> prt;
		prt(value, (*str));
	},
			   expr->value());
}
void StringExprVisitor::visit(const CallExpr* expr) {
	CodegenUtility::GetFunctionName(expr, *str, [&]() {
		(*str) += '(';
		auto&& args = expr->arguments();
		StringExprVisitor vis(*str);
		for (auto&& i : args) {
			i->accept(vis);
			(*str) += ',';
		}
		(*str)[str->size() - 1] = ')';
	});
}
void StringExprVisitor::visit(const CastExpr* expr) {
	BeforeVisit();
	(*str) += '(';
	CodegenUtility::GetTypeName(*expr->type(), *str);
	(*str) += ')';
	StringExprVisitor vis(*str);
	expr->expression()->accept(vis);
	AfterVisit();
}
void StringExprVisitor::visit(const ConstantExpr* expr) {
	(*str) += "c";
	vengine::to_string(expr->hash(), (*str));
}
StringExprVisitor::StringExprVisitor(vengine::string& str)
	: str(&str) {}

StringExprVisitor::~StringExprVisitor() {
}

void StringExprVisitor::BeforeVisit() {
	(*str) += '(';
}
void StringExprVisitor::AfterVisit() {
	(*str) += ')';
}

void StringStateVisitor::visit(const BreakStmt* state) {
	(*str) += "break;\n"_sv;
}

void StringStateVisitor::visit(const ContinueStmt* state) {
	(*str) += "continue;\n"_sv;
}

void StringStateVisitor::visit(const ReturnStmt* state) {
	if (state->expression()) {
		(*str) += "return "_sv;
		StringExprVisitor exprVis(*str);
		state->expression()->accept(exprVis);
		(*str) += ";\n"_sv;
	} else {
		(*str) += "return;\n"_sv;
	}
}

void StringStateVisitor::visit(const ScopeStmt* state) {
	(*str) += "{\n"_sv;
	StringStateVisitor sonVisitor(*str, func);
	for (auto&& i : state->statements()) {
		i->accept(sonVisitor);
	}
	(*str) += "}\n"_sv;
}

void StringStateVisitor::visit(const DeclareStmt* state) {
	auto var = state->variable();
	CodegenUtility::RegistStructType(var.type());
	CodegenUtility::GetTypeName(*var.type(), *str, _IsVarWritable(func, var));
	(*str) += ' ';
	vengine::string varTempName;
	CodegenUtility::GetVariableName(var, varTempName);
	(*str) += varTempName;
	if (!var.type()->is_structure()) {
		StringExprVisitor vis(*str);
		if (var.type()->is_scalar()) {
			if (!state->initializer().empty()) {
				(*str) += '=';
				for (auto&& i : state->initializer()) {
					i->accept(vis);
				}
			}
		} else if (!state->initializer().empty()) {
			(*str) += '=';
			if (state->initializer().size() == 1 && *state->initializer()[0]->type() == *var.type()) {
				state->initializer()[0]->accept(vis);
			} else {
				CodegenUtility::GetTypeName(*var.type(), (*str), _IsVarWritable(func, var));
				(*str) += '(';
				for (auto&& i : state->initializer()) {
					i->accept(vis);
					(*str) += ',';
				}
				(*str)[str->size() - 1] = ')';
			}
		}
		(*str) += ";\n"_sv;
	} else {

		StringExprVisitor vis(*str);
		size_t count = 0;
		if (state->initializer().size() == 1) {
			auto&& i = *state->initializer().begin();
			(*str) += '=';
			i->accept(vis);
			(*str) += ";\n"_sv;
		} else {
			(*str) += ";\n"_sv;
			for (auto&& i : state->initializer()) {
				(*str) += varTempName;
				(*str) += ".v"_sv;
				vengine::to_string(count, (*str));

				(*str) += '=';
				i->accept(vis);
				(*str) += ";\n"_sv;
				count++;
			}
		}
	}
}

void StringStateVisitor::visit(const IfStmt* state) {
	StringExprVisitor exprVisitor(*str);
	StringStateVisitor stateVisitor(*str, func);
	(*str) += "if ("_sv;
	state->condition()->accept(exprVisitor);
	(*str) += ')';
	state->true_branch()->accept(stateVisitor);
	if (state->false_branch()) {
		(*str) += "else"_sv;
		state->false_branch()->accept(stateVisitor);
	}
}

void StringStateVisitor::visit(const WhileStmt* state) {
	StringStateVisitor stateVisitor(*str, func);
	StringExprVisitor exprVisitor(*str);

	(*str) += "[loop]\nwhile ("_sv;
	state->condition()->accept(exprVisitor);
	(*str) += ')';
	state->body()->accept(stateVisitor);
}

void StringStateVisitor::visit(const ExprStmt* state) {
	StringExprVisitor exprVisitor(*str);
	state->expression()->accept(exprVisitor);
	(*str) += ";\n"_sv;
}

void StringStateVisitor::visit(const SwitchStmt* state) {
	(*str) += "switch ("_sv;
	StringStateVisitor stateVisitor(*str, func);
	StringExprVisitor exprVisitor(*str);
	state->expression()->accept(exprVisitor);
	(*str) += ')';
	state->body()->accept(stateVisitor);
}

void StringStateVisitor::visit(const SwitchCaseStmt* state) {
	(*str) += "case "_sv;
	StringStateVisitor stateVisitor(*str, func);
	StringExprVisitor exprVisitor(*str);
	state->expression()->accept(exprVisitor);
	(*str) += ":\n"_sv;
	state->body()->accept(stateVisitor);
}

void StringStateVisitor::visit(const SwitchDefaultStmt* state) {
	(*str) += "default:\n"_sv;
	StringStateVisitor stateVisitor(*str, func);
	state->body()->accept(stateVisitor);
}

void StringStateVisitor::visit(const AssignStmt* state) {
	StringExprVisitor exprVisitor(*str);
	state->lhs()->accept(exprVisitor);
	switch (state->op()) {
		case AssignOp::ASSIGN:
			(*str) += '=';
			break;
		case AssignOp::ADD_ASSIGN:
			(*str) += "+="_sv;
			break;
		case AssignOp::SUB_ASSIGN:
			(*str) += "-="_sv;
			break;
		case AssignOp::MUL_ASSIGN:
			(*str) += "*="_sv;
			break;
		case AssignOp::DIV_ASSIGN:
			(*str) += "/="_sv;
			break;
		case AssignOp::MOD_ASSIGN:
			(*str) += "%="_sv;
			break;
		case AssignOp::BIT_AND_ASSIGN:
			(*str) += "&="_sv;
			break;
		case AssignOp::BIT_OR_ASSIGN:
			(*str) += "|="_sv;
			break;
		case AssignOp::BIT_XOR_ASSIGN:
			(*str) += "^="_sv;
			break;
		case AssignOp::SHL_ASSIGN:
			(*str) += "<<="_sv;
			break;
		case AssignOp::SHR_ASSIGN:
			(*str) += ">>="_sv;
			break;
	}
	state->rhs()->accept(exprVisitor);
	(*str) += ";\n"_sv;
}

void StringStateVisitor::visit(ForStmt const* expr) {
	StringStateVisitor stmtVis(*str, func);
	StringExprVisitor expVis(*str);
	(*str) += "[loop]\nfor("_sv;
	expr->initialization()->accept(stmtVis);
	expr->condition()->accept(expVis);
	(*str) += ';';
	expr->update()->accept(stmtVis);
	(*str)[str->size() - 2] = ')';
	expr->body()->accept(stmtVis);
}

StringStateVisitor::StringStateVisitor(vengine::string& str, Function func)
	: str(&str), func(func) {
}

StringStateVisitor::~StringStateVisitor() {
}

void CodegenUtility::GetVariableName(Variable::Tag type, uint id, vengine::string& str) {
	switch (type) {
		case Variable::Tag::BLOCK_ID:
			str += "blk_id"_sv;
			break;
		case Variable::Tag::DISPATCH_ID:
			str += "dsp_id"_sv;
			break;
		case Variable::Tag::THREAD_ID:
			str += "thd_id"_sv;
			break;
		case Variable::Tag::LOCAL:
			str += "_v"_sv;
			vengine::to_string(id, str);
			break;
		case Variable::Tag::BUFFER:
			str += "_b"_sv;
			vengine::to_string(id, str);
			break;
		case Variable::Tag::TEXTURE:
			str += "_t"_sv;
			vengine::to_string(id, str);
			break;
		default:
			str += 'v';
			vengine::to_string(id, str);
			break;
	}
}

void CodegenUtility::GetVariableName(Type::Tag type, uint id, vengine::string& str) {
	switch (type) {
		case Type::Tag::BUFFER:
			str += "_b"_sv;
			vengine::to_string(id, str);
			break;
		case Type::Tag::TEXTURE:
			str += "_t"_sv;
			vengine::to_string(id, str);
			break;
		default:
			str += 'v';
			vengine::to_string(id, str);
			break;
	}
}

void CodegenUtility::GetVariableName(Variable const& type, vengine::string& str) {
	GetVariableName(type.tag(), type.uid(), str);
}

void CodegenUtility::GetTypeName(Type const& type, vengine::string& str, bool isWritable) {
	switch (type.tag()) {
		case Type::Tag::ARRAY:
			CodegenUtility::GetTypeName(*type.element(), str, isWritable);
			return;
		case Type::Tag::ATOMIC:
			CodegenUtility::GetTypeName(*type.element(), str, isWritable);
			return;
		case Type::Tag::BOOL:
			str += "bool"_sv;
			return;
		case Type::Tag::FLOAT:
			str += "float"_sv;
			return;
		case Type::Tag::INT:
			str += "int"_sv;
			return;
		case Type::Tag::UINT:
			str += "uint"_sv;
			return;
		case Type::Tag::MATRIX: {
			auto dim = vengine::to_string(type.dimension());
			CodegenUtility::GetTypeName(*type.element(), str, isWritable);
			str += dim;
			str += 'x';
			str += dim;
		}
			return;
		case Type::Tag::VECTOR: {
			CodegenUtility::GetTypeName(*type.element(), str, isWritable);
			vengine::to_string(type.dimension(), str);
		}
			return;
		case Type::Tag::STRUCTURE:
			str += 'T';
			vengine::to_string(type.index(), str);
			return;
		case Type::Tag::BUFFER:
			if (isWritable) {
				str += "RWStructuredBuffer<"_sv;
			} else {
				str += "StructuredBuffer<"_sv;
			}
			GetTypeName(*type.element(), str, isWritable);
			str += '>';
			break;
		case Type::Tag::TEXTURE: {
			if (isWritable) {
				str += "RWTexture"_sv;
			} else {
				str += "Texture"_sv;
			}
			vengine::to_string(type.dimension(), str);
			str += "D<"_sv;
			GetTypeName(*type.element(), str, isWritable);
			if (type.tag() != Type::Tag::VECTOR) {
				str += '4';
			}
			str += '>';
		} break;
	}
}

void CodegenUtility::GetFunctionDecl(Function func, vengine::string& data) {

	if (func.return_type()) {
		CodegenUtility::GetTypeName(*func.return_type(), data);
	} else {
		data += "void"_sv;
	}
	switch (func.tag()) {
		case Function::Tag::CALLABLE: {
			data += " custom_"_sv;
			vengine::to_string(func.uid(), data);
			if (func.arguments().empty()) {
				data += "()"_sv;
			} else {
				data += '(';
				for (auto&& i : func.arguments()) {
					RegistStructType(i.type());
					CodegenUtility::GetTypeName(*i.type(), data, _IsVarWritable(func, i));
					data += ' ';
					CodegenUtility::GetVariableName(i, data);
					data += ',';
				}
				data[data.size() - 1] = ')';
			}
		} break;
		default:
			data += " _kernel_"_sv;
			vengine::to_string(func.uid(), data);
			data += "(uint3 thd_id,uint3 blk_id,uint3 dsp_id)"_sv;
			break;
	}
}

void CodegenUtility::PrintConstant(Function::ConstantBinding const& binding, vengine::string& result) {
	result += "static const "_sv;
	auto valueView = ConstantData::view(binding.hash);
	std::visit([&](auto&& value) -> void {
		using SpanT = std::remove_cvref_t<decltype(value)>;
		using T = GetSpanType<SpanT>::Type;
		PrintValue<T> prt;
		result += GetName<T>::Get();
		result += " c"_sv;
		vengine::to_string(binding.hash, result);

		if (binding.type->is_array()) {
			result += "[]={"_sv;
			for (auto&& i : value) {
				prt(i, result);
				result += ',';
			}
			if (result[result.size() - 1] == ',') {
				result.erase(result.size() - 1);
			}
			result += "};\n"_sv;
		} else {
			result += '=';
			prt(value[0], result);
			result += ";\n"_sv;
		}
	},
			   valueView);
}
static thread_local StackObject<HashMap<Type const*, bool>, true> codegenStructType;
void CodegenUtility::ClearStructType() {
	codegenStructType.New();
	codegenStructType->Clear();
}

void CodegenUtility::GetFunctionName(CallExpr const* expr, vengine::string& result, Runnable<void()>&& func) {
	auto IsType = [](Type const* const type, Type::Tag const tag, uint const vecEle) {
		return vengine::select(
			[&]() {
				return false;
			},
			[&]() { return vengine::select(
						[&]() {
							return type->tag() == tag;
						},
						[&]() {
							return type->element()->tag() == tag && type->dimension() == vecEle;
						},
						[&]() {
							return vecEle > 1;
						}); },
			[&]() {
				return (vecEle > 1) == (type->tag() == Type::Tag::VECTOR);
			});
		/*
		if (isVec != (type->tag() == Type::Tag::VECTOR))
			return false;
		auto typeTag = isVec ? type->element()->tag() : type->tag();
		return typeTag == tag;*/
	};
	switch (expr->op()) {
		case CallOp::CUSTOM:
			result << "custom_"_sv << vengine::to_string(expr->uid());
			break;
		case CallOp::ALL:
			result << "all"_sv;
			break;
		case CallOp::ANY:
			result << "any"_sv;
			break;
		case CallOp::NONE:
			result << "!any"_sv;
			break;
		case CallOp::SELECT: {
			const auto thirdArg = expr->arguments()[2]->type();
			vengine::select(
				[&]() { result << "_select_int"_sv; },
				[&]() { result << "_select_bool"_sv; },
				[&]() { return vengine::select(
							[&](auto&& t) {
								return t->element()->tag() == Type::Tag::BOOL;
							},
							[&](auto&& t) {
								return t->tag() == Type::Tag::BOOL;
							},
							[](auto&& t) {
								return t->is_scalar();
							},
							thirdArg); });
		} break;
		case CallOp::CLAMP:
			result << "clamp"_sv;
			break;
		case CallOp::LERP:
			result << "lerp"_sv;
			break;
		case CallOp::SATURATE:
			result << "saturate"_sv;
			break;
		case CallOp::SIGN:
			result << "sign"_sv;
			break;
		case CallOp::STEP:
			result << "step"_sv;
			break;
		case CallOp::SMOOTHSTEP:
			result << "smoothstep"_sv;
			break;
		case CallOp::ABS:
			result << "abs"_sv;
			break;
		case CallOp::MIN:
			result << "min"_sv;
			break;
		case CallOp::POW:
			result << "pow"_sv;
			break;
		case CallOp::CLZ:
			result << "firstbithigh"_sv;
			break;
		case CallOp::CTZ:
			result << "firstbitlow"_sv;
			break;
		case CallOp::POPCOUNT:
			result << "countbits"_sv;
			break;
		case CallOp::REVERSE:
			result << "reversebits"_sv;
			break;
		case CallOp::ISINF:
			result << "_isinf"_sv;
			break;
		case CallOp::ISNAN:
			result << "_isnan"_sv;
			break;
		case CallOp::ACOS:
			result << "acos"_sv;
			break;
		case CallOp::ACOSH:
			result << "_acosh"_sv;
			break;
		case CallOp::ASIN:
			result << "asin"_sv;
			break;
		case CallOp::ASINH:
			result << "_asinh"_sv;
			break;
		case CallOp::ATAN:
			result << "atan"_sv;
			break;
		case CallOp::ATAN2:
			result << "atan2"_sv;
			break;
		case CallOp::ATANH:
			result << "_atanh"_sv;
			break;
		case CallOp::COS:
			result << "cos"_sv;
			break;
		case CallOp::COSH:
			result << "cosh"_sv;
			break;
		case CallOp::SIN:
			result << "sin"_sv;
			break;
		case CallOp::SINH:
			result << "sinh"_sv;
			break;
		case CallOp::TAN:
			result << "tan"_sv;
			break;
		case CallOp::TANH:
			result << "tanh"_sv;
			break;
		case CallOp::EXP:
			result << "exp"_sv;
			break;
		case CallOp::EXP2:
			result << "exp2"_sv;
			break;
		case CallOp::EXP10:
			result << "_exp10"_sv;
			break;
		case CallOp::LOG:
			result << "log"_sv;
			break;
		case CallOp::LOG2:
			result << "log2"_sv;
			break;
		case CallOp::LOG10:
			result << "log10"_sv;
			break;
		case CallOp::SQRT:
			result << "sqrt"_sv;
			break;
		case CallOp::RSQRT:
			result << "rsqrt"_sv;
			break;
		case CallOp::CEIL:
			result << "ceil"_sv;
			break;
		case CallOp::FLOOR:
			result << "floor"_sv;
			break;
		case CallOp::FRACT:
			result << "fract"_sv;
			break;
		case CallOp::TRUNC:
			result << "trunc"_sv;
			break;
		case CallOp::ROUND:
			result << "round"_sv;
			break;
		case CallOp::DEGREES:
			result << "degrees"_sv;
			break;
		case CallOp::RADIANS:
			result << "radians"_sv;
			break;
		case CallOp::FMA:
			result << "fma"_sv;
			break;
		case CallOp::COPYSIGN:
			result << "_copysign"_sv;
			break;
		case CallOp::CROSS:
			result << "cross"_sv;
			break;
		case CallOp::DOT:
			result << "dot"_sv;
			break;
		case CallOp::DISTANCE:
			result << "distance"_sv;
			break;
		case CallOp::DISTANCE_SQUARED:
			result << "_distance_sqr"_sv;
			break;
		case CallOp::LENGTH:
			result << "length"_sv;
			break;
		case CallOp::LENGTH_SQUARED:
			result << "_length_sqr"_sv;
			break;
		case CallOp::NORMALIZE:
			result << "normalize"_sv;
			break;
		case CallOp::FACEFORWARD:
			result << "faceforward"_sv;
			break;
		case CallOp::DETERMINANT:
			result << "determinant"_sv;
			break;
		case CallOp::TRANSPOSE:
			result << "transpose"_sv;
			break;
		case CallOp::INVERSE:
			result << "_inverse"_sv;
			break;
		case CallOp::GROUP_MEMORY_BARRIER:
			result << "GroupMemoryBarrierWithGroupSync"_sv;
			break;
		case CallOp::DEVICE_MEMORY_BARRIER:
			result << "DeviceMemoryBarrierWithGroupSync"_sv;
			break;
		case CallOp::ALL_MEMORY_BARRIER:
			result << "AllMemoryBarrierWithGroupSync"_sv;
			break;
			///TODO: atomic operation
		case CallOp::ATOMIC_LOAD:
			//result << "_atomic_load"_sv;
			break;
		case CallOp::ATOMIC_STORE:
			//result << "_atomic_store"_sv;
			break;
		case CallOp::ATOMIC_EXCHANGE:
			result << "InterlockedExchange"_sv;
			break;
		case CallOp::ATOMIC_COMPARE_EXCHANGE:
			result << "InterlockedCompareExchange"_sv;
			break;
		case CallOp::ATOMIC_FETCH_ADD:
			result << "InterlockedAdd"_sv;
			break;
		case CallOp::ATOMIC_FETCH_SUB:
			result << "InterlockedAdd"_sv;
			break;
		case CallOp::ATOMIC_FETCH_AND:
			result << "InterlockedAnd"_sv;
			break;
		case CallOp::ATOMIC_FETCH_OR:
			result << "InterlockedOr"_sv;
			break;
		case CallOp::ATOMIC_FETCH_XOR:
			result << "InterlockedXor"_sv;
			break;
		case CallOp::ATOMIC_FETCH_MIN:
			result << "InterlockedMin"_sv;
			break;
		case CallOp::ATOMIC_FETCH_MAX:
			result << "InterlockedMax"_sv;
			break;
		case CallOp::TEXTURE_READ: {
			auto args = expr->arguments();
			StringExprVisitor vis(result);
			args[0]->accept(vis);
			result << '[';
			args[1]->accept(vis);
			result << ']';
		}
			return;
		case CallOp::TEXTURE_WRITE: {
			auto args = expr->arguments();
			StringExprVisitor vis(result);
			args[0]->accept(vis);
			result << '[';
			args[1]->accept(vis);
			if (args[2]->type()->is_vector()) {
				result << "]=to_tex("_sv;
				args[2]->accept(vis);
				result << ')';
			} else {
				result << "]="_sv;
				args[2]->accept(vis);
			}
		}
			return;
		case CallOp::TEXTURE_SAMPLE:
			/*
			auto args = expr->arguments();
			result << '(';
			StringExprVisitor vis(result);
			args[0]->accept(vis);
			result << ".SampleLevel("
				   << ')';
			*/
			return;
		case CallOp::MAKE_BOOL2:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 2))
				result << "make_bool2"_sv;

			break;
		case CallOp::MAKE_BOOL3:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 3))
				result << "make_bool3"_sv;

			break;
		case CallOp::MAKE_BOOL4:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 4))
				result << "make_bool4"_sv;

			break;
		case CallOp::MAKE_UINT2:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 2))
				result << "make_uint2"_sv;

			break;
		case CallOp::MAKE_UINT3:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 3))
				result << "make_uint3"_sv;
			break;
		case CallOp::MAKE_UINT4:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 4))
				result << "make_uint4"_sv;

			break;
		case CallOp::MAKE_INT2:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 2))
				result << "make_int2"_sv;

			break;
		case CallOp::MAKE_INT3:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 3))
				result << "make_int3"_sv;

			break;
		case CallOp::MAKE_INT4:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 4))
				result << "make_int4"_sv;

			break;
		case CallOp::MAKE_FLOAT2:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 2))
				result << "make_float2"_sv;

			break;
		case CallOp::MAKE_FLOAT3:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 3))
				result << "make_float3"_sv;

			break;
		case CallOp::MAKE_FLOAT4:
			if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 4))
				result << "make_float4"_sv;

			break;
		default:
			VEngine_Log("Function Not Implemented"_sv);
			VENGINE_EXIT;
	}
	func();
}
void CodegenUtility::RegistStructType(Type const* type) {
	if (type->is_structure())
		codegenStructType->Insert(type, true);
	else if (type->is_buffer()) {
		RegistStructType(type->element());
	}
}
struct BuiltIn_RunTypeVisitor : public TypeVisitor {
	TypeVisitor* vis;
	void visit(const Type* t) noexcept override {
		if (codegenStructType->Contains(t)) {
			vis->visit(t);
		}
	}
};
struct Print_RunTypeVisitor : public TypeVisitor {
	vengine::string* strPtr;
	void visit(const Type* t) noexcept override {
		if (!codegenStructType->Contains(t)) return;
		auto&& str = *strPtr;
		str += "struct T"_sv;
		vengine::to_string(t->index(), str);
		str += "{\n"_sv;
		size_t count = 0;
		for (auto&& mem : t->members()) {
			CodegenUtility::GetTypeName(*mem, str);
			str << ' ';
			CodegenUtility::GetVariableName(mem->tag(), count, str);
			count++;
			str += ";\n"_sv;
		}
		str += "};\n"_sv;
	}
};
void CodegenUtility::IterateStructType(TypeVisitor* visitor) {
	BuiltIn_RunTypeVisitor vis;
	vis.vis = visitor;
	Type::traverse(vis);
}
void CodegenUtility::PrintStructType(vengine::string& str) {
	Print_RunTypeVisitor vis;
	vis.strPtr = &str;
	Type::traverse(vis);
}
void CodegenUtility::PrintUniform(
	Function func,
	//Buffer Binding
	vengine::string& result) {
	auto argss = {func.builtin_variables(), func.arguments()};
	auto buffers = func.captured_buffers();
	auto texs = func.captured_textures();
	uint tCount = 0;
	uint uCount = 0;
	auto ProcessBuffer = [&](Variable const& var) {
		bool enableRandomWrite = _IsVarWritable(func, var);
		GetTypeName(*var.type(), result, enableRandomWrite);
		result << ' ';
		CodegenUtility::GetVariableName(var, result);
		if (enableRandomWrite) {
			result += ":register(u"_sv;
			vengine::to_string(uCount, result);
			uCount++;
		} else {
			result += ":register(t"_sv;
			vengine::to_string(tCount, result);
			tCount++;
		}
		result += ");\n"_sv;
	};
	for (auto& args : argss) {
		for (auto& var : args) {
			switch (var.tag()) {
				case Variable::Tag::BUFFER:
				case Variable::Tag::TEXTURE:
					ProcessBuffer(var);
					break;
			}
		}
	}
	for (size_t i = 0; i < buffers.size(); ++i) {
		ProcessBuffer(buffers[i].variable);
	}
	//TODO: Texture Binding
}
size_t CodegenUtility::PrintGlobalVariables(
	Function func,
	std::initializer_list<std::span<const Variable>> values,
	HashMap<uint, size_t>& varOffsets,
	vengine::string& result) {
	vengine::vector<Variable const*> scalarArr;
	vengine::vector<Variable const*> vec2Arr;
	vengine::vector<Variable const*> vec3Arr;
	vengine::vector<Variable const*> vec4Arr;
	constexpr size_t ELE_SIZE = 4;
	size_t cbufferSize = 0;
	size_t alignCount = 0;
	auto GetUniform = [&](Type const& type, Variable const& var) -> void {
		switch (type.tag()) {
			///////////// Invalid types
			case Type::Tag::ARRAY:
				VEngine_Log("Uniform Variable Cannot be array!\n"_sv);
				throw 0;
			case Type::Tag::BOOL:
				VEngine_Log("Uniform Variable Cannot be bool!\n"_sv);
				throw 0;
			case Type::Tag::ATOMIC:
				VEngine_Log("Uniform Variable Cannot be atomic!\n"_sv);
				throw 0;
			///////////// Valid Types
			case Type::Tag::MATRIX:
				if (type.dimension() != 4) {
					VEngine_Log("Uniform Matrix Only Allow 4x4 Matrix!\n"_sv);
					throw 0;
				}
				CodegenUtility::GetTypeName(type, result, _IsVarWritable(func, var));
				result << ' ';
				CodegenUtility::GetVariableName(var, result);
				result += ";\n"_sv;
				varOffsets.Insert(var.uid(), cbufferSize);
				cbufferSize += ELE_SIZE * 4 * 4;
				break;
			case Type::Tag::VECTOR:
				switch (type.dimension()) {
					case 2:
						vec2Arr.push_back(&var);
						break;
					case 3:
						vec3Arr.push_back(&var);
						break;
					case 4:
						vec4Arr.push_back(&var);
						break;
				}
				break;
			case Type::Tag::UINT:
			case Type::Tag::INT:
			case Type::Tag::FLOAT:
				scalarArr.push_back(&var);
				break;
		}
	};
	result += "cbuffer Params:register(b0)\n{\n"_sv;

	for (auto& spans : values) {
		for (auto& var : spans) {
			switch (var.tag()) {
				case Variable::Tag::DISPATCH_SIZE:
				case Variable::Tag::UNIFORM:

					auto&& type = *var.type();
					GetUniform(type, var);
					break;
			};
		}
	}
	for (auto& vec4 : vec4Arr) {
		varOffsets.Insert(vec4->uid(), cbufferSize);
		cbufferSize += ELE_SIZE * 4;
		GetTypeName(*vec4->type(), result);
		result += ' ';
		CodegenUtility::GetVariableName(*vec4, result);
		result += ";\n"_sv;
	}
	auto PrintScalar = [&](Variable const* var) -> void {
		GetTypeName(*var->type(), result);
		result += ' ';
		CodegenUtility::GetVariableName(*var, result);

		result += ";\n"_sv;
	};

	for (auto& vec3 : vec3Arr) {
		varOffsets.Insert(vec3->uid(), cbufferSize);
		cbufferSize += ELE_SIZE * 3;
		GetTypeName(*vec3->type(), result);
		result += ' ';
		CodegenUtility::GetVariableName(*vec3, result);
		result += ";\n"_sv;
		if (!scalarArr.empty()) {
			auto v = scalarArr.erase_last();
			varOffsets.Insert(v->uid(), cbufferSize);
			PrintScalar(v);
		} else {
			result += "float __a"_sv;
			vengine::to_string(alignCount, result);
			result += ";\n"_sv;
			alignCount++;
		}
		cbufferSize += ELE_SIZE;
	}

	for (auto& vec2 : vec2Arr) {
		varOffsets.Insert(vec2->uid(), cbufferSize);
		cbufferSize += ELE_SIZE * 2;
		GetTypeName(*vec2->type(), result);
		result += ' ';
		CodegenUtility::GetVariableName(*vec2, result);
		result += ";\n"_sv;
	}

	for (auto& vec : scalarArr) {
		varOffsets.Insert(vec->uid(), cbufferSize);
		cbufferSize += ELE_SIZE;
		PrintScalar(vec);
	}
	result += "}\n"_sv;//End cbuffer
	return cbufferSize;
}
void CodegenUtility::SeparateVariables(
	Function func,
	vengine::vector<Variable const*>& buffers,
	vengine::vector<Variable const*>& textures,
	vengine::vector<Variable const*>& globalSRVValues,
	vengine::vector<Variable const*>& globalUAVValues,
	vengine::vector<Variable const*>& groupSharedValues) {
	buffers.clear();
	textures.clear();
	globalSRVValues.clear();
	globalUAVValues.clear();
	groupSharedValues.clear();
}
}// namespace luisa::compute
