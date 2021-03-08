#include <ast/interface.h>
#include <ast/constant_data.h>
#include <Common/Common.h>
#include <Common/VObject.h>
#include "LuisaASTTranslator.h"

namespace luisa::compute {
#ifdef NDEBUG
DLL_EXPORT void CodegenBody(Function const* func) {
	vengine::vengine_init_malloc();
	std::cout << "Start Working" << std::endl;
	CodegenUtility::ClearStructType();
	StringStateVisitor vis;
	func->body()->accept(vis);
	{
		vengine::string str;
		CodegenUtility::PrintStructType(str);
		std::cout << str << std::endl;
	}
	{
		vengine::string str;
		for (auto& i : func->constants()) {
			CodegenUtility::PrintConstant(i, str);
		}
		std::cout << str << std::endl;
	}
	std::cout << CodegenUtility::GetFunctionDecl(func) << std::endl;
	std::cout << vis.ToString() << std::endl;
	std::cout << "Finished" << std::endl;
}
#endif
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
			return "bool"vstr;
		} else if constexpr (std::is_same_v<float, T>) {
			return "float"vstr;
		} else if constexpr (std::is_same_v<int, T>) {
			return "int"vstr;
		} else if constexpr (std::is_same_v<uint, T>) {
			return "uint"vstr;
		} else {
			return "unknown"vstr;
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
		str += vengine::to_string((int)v);
	}
};
template<>
struct PrintValue<float> {
	void operator()(float const& v, vengine::string& str) {
		str += vengine::to_string(v);
	}
};

template<typename EleType, size_t N>
struct PrintValue<Vector<EleType, N>> {
	using T = typename Vector<EleType, N>;
	void operator()(T const& v, vengine::string& varName) {
		varName += GetName<T>::Get();
		varName += '(';
		for (size_t i = 0; i < N; ++i) {
			varName += vengine::to_string(v[i]);
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
		varName += "("vstr;
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
		str = "mul("vstr;
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
				str += "<<"vstr;
				break;
			case BinaryOp::SHR:
				str += ">>"vstr;
				break;
			case BinaryOp::AND:
				str += "&&"vstr;
				break;
			case BinaryOp::OR:
				str += "||"vstr;
				break;
			case BinaryOp::LESS:
				str += '<';
				break;
			case BinaryOp::GREATER:
				str += '>';
				break;
			case BinaryOp::LESS_EQUAL:
				str += "<="vstr;
				break;
			case BinaryOp::GREATER_EQUAL:
				str += ">="vstr;
				break;
			case BinaryOp::EQUAL:
				str += "=="vstr;
				break;
			case BinaryOp::NOT_EQUAL:
				str += "!="vstr;
				break;
		}
		expr->rhs()->accept(vis);
		str += vis.ToString();
		AfterVisit();
	}
}
void StringExprVisitor::visit(const MemberExpr* expr) {
	expr->self()->accept(*this);
	str += ".v"vstr;
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
	CodegenUtility::RegistStructType(v.type());
	if (v.type()->is_vector() && v.type()->element()->size() < 4) {
		//TODO
	} else {
		str = std::move(CodegenUtility::GetVariableName(v));
	}
}
void StringExprVisitor::visit(const LiteralExpr* expr) {
	LiteralExpr::Value const& value = expr->value();
	str.clear();
	std::visit([&](auto&& value) -> void {
		using T = std::remove_cvref_t<decltype(value)>;
		PrintValue<T> prt;
		prt(value, str);
	},
			   expr->value());
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
	BeforeVisit();
	str += '(';
	str += CodegenUtility::GetTypeName(*expr->type());
	str += ')';
	StringExprVisitor vis;
	expr->expression()->accept(vis);
	str += vis.str;
	AfterVisit();
}
void StringExprVisitor::visit(const ConstantExpr* expr) {
	str = "c" + vengine::to_string(expr->hash());
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
	str = "break;\n"vstr;
}

void StringStateVisitor::visit(const ContinueStmt* state) {
	str = "continue;\n"vstr;
}

void StringStateVisitor::visit(const ReturnStmt* state) {
	if (state->expression()) {
		str = "return "vstr;
		StringExprVisitor exprVis;
		state->expression()->accept(exprVis);
		str += exprVis.ToString();
		str += ";\n"vstr;
	} else {
		str = "return;\n"vstr;
	}
}

void StringStateVisitor::visit(const ScopeStmt* state) {
	str = "{\n"vstr;
	StringStateVisitor sonVisitor;
	for (auto&& i : state->statements()) {
		i->accept(sonVisitor);
		str += sonVisitor.ToString();
	}
	str += "}\n"vstr;
}

void StringStateVisitor::visit(const DeclareStmt* state) {
	auto var = state->variable();
	CodegenUtility::RegistStructType(var.type());
	auto varName = CodegenUtility::GetTypeName(*var.type());
	str.clear();
	str.push_back_all(varName.data(), varName.size());
	str += ' ';
	auto varTempName = CodegenUtility::GetVariableName(var);
	str += varTempName;
	if (!var.type()->is_structure()) {
		StringExprVisitor vis;
		if (var.type()->is_scalar()) {
			if (!state->initializer().empty()) {
				str += '=';
				for (auto&& i : state->initializer()) {
					i->accept(vis);
					str += vis.str;
				}
			}
		} else if (!state->initializer().empty()) {
			str += '=';
			str += CodegenUtility::GetTypeName(*var.type());
			str += '(';
			for (auto&& i : state->initializer()) {
				i->accept(vis);
				str += vis.str;
				str += ',';
			}
			str[str.size() - 1] = ')';
		}
		str += ";\n"vstr;
	} else {

		StringExprVisitor vis;
		size_t count = 0;
		if (state->initializer().size() == 1) {
			auto&& i = *state->initializer().begin();
			i->accept(vis);
			str += '=';
			str += vis.str;
			str += ";\n"vstr;
		} else {
			str += ";\n"vstr;
			for (auto&& i : state->initializer()) {
				i->accept(vis);
				str += varTempName;
				str += ".v"vstr;
				str += vengine::to_string(count);

				str += '=';
				str += vis.str;
				str += ";\n"vstr;
				count++;
			}
		}
	}
	//TODO: Different Declare in Fxxking HLSL
}

void StringStateVisitor::visit(const IfStmt* state) {
	StringExprVisitor exprVisitor;
	StringStateVisitor stateVisitor;
	str = "if ("vstr;
	state->condition()->accept(exprVisitor);
	str += exprVisitor.ToString();
	str += ')';
	state->true_branch()->accept(stateVisitor);
	str += stateVisitor.ToString();
	str += "else"vstr;
	state->false_branch()->accept(stateVisitor);
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const WhileStmt* state) {
	StringStateVisitor stateVisitor;
	StringExprVisitor exprVisitor;

	str = "while ("vstr;
	state->condition()->accept(exprVisitor);
	str += exprVisitor.ToString();
	str += ')';
	state->body()->accept(stateVisitor);
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const ExprStmt* state) {
	StringExprVisitor exprVisitor;
	state->expression()->accept(exprVisitor);
	str = std::move(exprVisitor.str);
	str += ";\n"vstr;
}

void StringStateVisitor::visit(const SwitchStmt* state) {
	str = "switch ("vstr;
	StringStateVisitor stateVisitor;
	StringExprVisitor exprVisitor;
	state->expression()->accept(exprVisitor);
	state->body()->accept(stateVisitor);
	str += exprVisitor.ToString();
	str += ')';
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const SwitchCaseStmt* state) {
	str = "case "vstr;
	StringStateVisitor stateVisitor;
	StringExprVisitor exprVisitor;
	state->expression()->accept(exprVisitor);
	state->body()->accept(stateVisitor);
	str += exprVisitor.ToString();
	str += ":\n"vstr;
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const SwitchDefaultStmt* state) {
	str = "default:\n"vstr;
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
			str += "+="vstr;
			break;
		case AssignOp::SUB_ASSIGN:
			str += "-="vstr;
			break;
		case AssignOp::MUL_ASSIGN:
			str += "*="vstr;
			break;
		case AssignOp::DIV_ASSIGN:
			str += "/="vstr;
			break;
		case AssignOp::MOD_ASSIGN:
			str += "%="vstr;
			break;
		case AssignOp::BIT_AND_ASSIGN:
			str += "&="vstr;
			break;
		case AssignOp::BIT_OR_ASSIGN:
			str += "|="vstr;
			break;
		case AssignOp::BIT_XOR_ASSIGN:
			str += "^="vstr;
			break;
		case AssignOp::SHL_ASSIGN:
			str += "<<="vstr;
			break;
		case AssignOp::SHR_ASSIGN:
			str += ">>="vstr;
			break;
	}
	state->rhs()->accept(exprVisitor);
	str += exprVisitor.ToString();
	str += ";\n"vstr;
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

vengine::string CodegenUtility::GetVariableName(Variable const& type) {
	return "v" + vengine::to_string(type.uid());
}

vengine::string CodegenUtility::GetTypeName(Type const& type) {
	switch (type.tag()) {
		case Type::Tag::ARRAY:
			return CodegenUtility::GetTypeName(*type.element());
		case Type::Tag::ATOMIC:
			return CodegenUtility::GetTypeName(*type.element());
		case Type::Tag::BOOL:
			return "bool"vstr;
		case Type::Tag::FLOAT:
			return "float"vstr;
		case Type::Tag::INT:
			return "int"vstr;
		case Type::Tag::UINT:
			return "uint"vstr;

		case Type::Tag::MATRIX: {
			auto dim = vengine::to_string(type.dimension());
			return CodegenUtility::GetTypeName(*type.element()) + dim + 'x' + dim;
		}
		case Type::Tag::VECTOR: {
			auto dim = vengine::to_string(type.dimension());
			return CodegenUtility::GetTypeName(*type.element()) + dim;
		}
		case Type::Tag::STRUCTURE:
			return "T" + vengine::to_string(type.index());
			break;
	}
}

vengine::string CodegenUtility::GetFunctionDecl(Function const* func) {
	vengine::string data;
	if (func->return_type()) {
		data = CodegenUtility::GetTypeName(*func->return_type());
	} else {
		data = "void"vstr;
	}
	switch (func->tag()) {
		case Function::Tag::CALLABLE:
			data += " custom_"vstr;
			break;
		default:
			data += " kernel_"vstr;
			//TODO: kernel specific declare
			break;
	}
	data += vengine::to_string(func->uid());
	if (func->arguments().empty()) {
		data += "()"vstr;
	} else {
		data += '(';
		for (auto&& i : func->arguments()) {
			RegistStructType(i.type());
			data += CodegenUtility::GetTypeName(*i.type());
			data += " "vstr;
			data += CodegenUtility::GetVariableName(i);
			data += ',';
		}
		data[data.size() - 1] = ')';
	}
	return data;
}

void CodegenUtility::PrintConstant(Function::ConstantBinding const& binding, vengine::string& result) {
	result = "static const "vstr;
	auto valueView = ConstantData::view(binding.hash);
	std::visit([&](auto&& value) -> void {
		using SpanT = std::remove_cvref_t<decltype(value)>;
		using T = GetSpanType<SpanT>::Type;
		PrintValue<T> prt;
		result += GetName<T>::Get();
		result += " c"vstr;
		result += vengine::to_string(binding.hash);

		if (binding.type->is_array()) {
			result += "[]={"vstr;
			for (auto&& i : value) {
				prt(i, result);
				result += ',';
			}
			if (result[result.size() - 1] == ',') {
				result.erase(result.size() - 1);
			}
			result += "};\n"vstr;
		} else {
			result += '=';
			prt(value[0], result);
			result += ";\n"vstr;
		}
	},
			   valueView);
}
static StackObject<HashMap<Type const*, bool>, true> codegenStructType;
void CodegenUtility::ClearStructType() {
	codegenStructType.New();
	codegenStructType->Clear();
}
void CodegenUtility::RegistStructType(Type const* type) {
	if (type->is_structure())
		codegenStructType->Insert(type, true);
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
		str += "struct T"vstr;
		str += vengine::to_string(t->index());
		str += "{\n"vstr;
		size_t count = 0;
		for (auto&& mem : t->members()) {
			str += CodegenUtility::GetTypeName(*mem);
			str += " v"vstr;
			str += vengine::to_string(count);
			count++;
			str += ";\n"vstr;
		}
		str += "};\n"vstr;
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
}// namespace luisa::compute
