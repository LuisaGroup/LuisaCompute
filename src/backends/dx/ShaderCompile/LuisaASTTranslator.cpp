#include <ast/interface.h>
#include <ast/constant_data.h>
#include <Common/Common.h>
#include <Common/VObject.h>
#include "LuisaASTTranslator.h"

namespace luisa::compute {
#ifdef NDEBUG
DLL_EXPORT void CodegenBody(Function func) {
	vengine::vengine_init_malloc(malloc, free);
	//LUISA_INFO("HLSL codegen started.");
	vengine::string function_buffer;
	vengine::string decl_buffer;
	function_buffer.reserve(65535 * 4);
	decl_buffer.reserve(65535 * 4);

	using namespace std::chrono_literals;
	CodegenUtility::ClearStructType();
	auto t0 = std::chrono::high_resolution_clock::now();

	StringStateVisitor vis(function_buffer);

	for (auto cust : func.custom_callables()) {
		auto&& callable = Function::callable(cust);
		CodegenUtility::GetFunctionDecl(callable, function_buffer);
		callable.body()->accept(vis);
	}
	CodegenUtility::GetFunctionDecl(func, function_buffer);
	func.body()->accept(vis);

	CodegenUtility::PrintStructType(decl_buffer);

	for (auto& i : func.constants()) {
		CodegenUtility::PrintConstant(i, decl_buffer);
	}

	CodegenUtility::PrintUniform(func.captured_buffers(), func.captured_textures(), decl_buffer);
	CodegenUtility::PrintGlobalVariables(func.arguments(), decl_buffer);
	auto t1 = std::chrono::high_resolution_clock::now();

	LUISA_INFO("HLSL codegen finished in {} ms.", (t1 - t0) / 1ns * 1e-6);

	/*std::cout << decl_buffer;
	std::cout << function_buffer;
	/*<< CodegenUtility::GetFunctionDecl(func) << "\n"
			  << vis.ToString() << std::endl;*/
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
	expr->accept(vis);
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
	if (expr->self()->type()->is_structure()) {
		(*str) += ".v"_sv;
		vengine::to_string(expr->member_index(), (*str));
	} else {
		switch (expr->member_index()) {
			case 0:
				(*str) += ".x"_sv;
				break;
			case 1:
				(*str) += ".y"_sv;
				break;
			case 2:
				(*str) += ".z"_sv;
				break;
			default:
				(*str) += ".w"_sv;
				break;
		}
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
	if (v.type()->is_vector() && v.type()->element()->size() < 4) {
		//TODO
	} else {
		CodegenUtility::GetVariableName(v, *str);
	}
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
	str->push_back_all(expr->name().data(), expr->name().size());
	(*str) += '(';
	auto&& args = expr->arguments();
	StringExprVisitor vis(*str);
	for (auto&& i : args) {
		i->accept(vis);
		(*str) += ',';
	}
	(*str)[str->size() - 1] = ')';
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
	StringStateVisitor sonVisitor(*str);
	for (auto&& i : state->statements()) {
		i->accept(sonVisitor);
	}
	(*str) += "}\n"_sv;
}

void StringStateVisitor::visit(const DeclareStmt* state) {
	auto var = state->variable();
	CodegenUtility::RegistStructType(var.type());
	CodegenUtility::GetTypeName(*var.type(), *str);
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
			CodegenUtility::GetTypeName(*var.type(), (*str));
			(*str) += '(';
			for (auto&& i : state->initializer()) {
				i->accept(vis);
				(*str) += ',';
			}
			(*str)[str->size() - 1] = ')';
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
	//TODO: Different Declare in Fxxking HLSL
}

void StringStateVisitor::visit(const IfStmt* state) {
	StringExprVisitor exprVisitor(*str);
	StringStateVisitor stateVisitor(*str);
	(*str) += "if ("_sv;
	state->condition()->accept(exprVisitor);
	(*str) += ')';
	state->true_branch()->accept(stateVisitor);
	(*str) += "else"_sv;
	state->false_branch()->accept(stateVisitor);
}

void StringStateVisitor::visit(const WhileStmt* state) {
	StringStateVisitor stateVisitor(*str);
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
	StringStateVisitor stateVisitor(*str);
	StringExprVisitor exprVisitor(*str);
	state->expression()->accept(exprVisitor);
	(*str) += ')';
	state->body()->accept(stateVisitor);
}

void StringStateVisitor::visit(const SwitchCaseStmt* state) {
	(*str) += "case "_sv;
	StringStateVisitor stateVisitor(*str);
	StringExprVisitor exprVisitor(*str);
	state->expression()->accept(exprVisitor);
	(*str) += ":\n"_sv;
	state->body()->accept(stateVisitor);
}

void StringStateVisitor::visit(const SwitchDefaultStmt* state) {
	(*str) += "default:\n"_sv;
	StringStateVisitor stateVisitor(*str);
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
	StringStateVisitor stmtVis(*str);
	StringExprVisitor expVis(*str);
	(*str) += "[loop]\nfor("_sv;
	expr->initialization()->accept(stmtVis);
	expr->condition()->accept(expVis);
	(*str) += ';';
	expr->update()->accept(stmtVis);
	(*str)[str->size() - 2] = ')';
	expr->body()->accept(stmtVis);
}

StringStateVisitor::StringStateVisitor(vengine::string& str)
	: str(&str) {
}

StringStateVisitor::~StringStateVisitor() {
}

void CodegenUtility::GetVariableName(Variable const& type, vengine::string& str) {
	str += 'v';
	vengine::to_string(type.uid(), str);
}

void CodegenUtility::GetTypeName(Type const& type, vengine::string& str) {
	switch (type.tag()) {
		case Type::Tag::ARRAY:
			CodegenUtility::GetTypeName(*type.element(), str);
			return;
		case Type::Tag::ATOMIC:
			CodegenUtility::GetTypeName(*type.element(), str);
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
			CodegenUtility::GetTypeName(*type.element(), str);
			str += dim;
			str += 'x';
			str += dim;
		}
			return;
		case Type::Tag::VECTOR: {
			auto dim = vengine::to_string(type.dimension());
			CodegenUtility::GetTypeName(*type.element(), str);
			str += dim;
		}
			return;
		case Type::Tag::STRUCTURE:
			str += 'T';
			vengine::to_string(type.index(), str);
			return;
	}
}

void CodegenUtility::GetFunctionDecl(Function func, vengine::string& data) {

	if (func.return_type()) {
		CodegenUtility::GetTypeName(*func.return_type(), data);
	} else {
		data += "void"_sv;
	}
	switch (func.tag()) {
		case Function::Tag::CALLABLE:
			data += " custom_"_sv;
			break;
		default:
			data += " kernel_"_sv;
			//TODO: kernel specific declare
			break;
	}
	vengine::to_string(func.uid(), data);
	if (func.arguments().empty()) {
		data += "()"_sv;
	} else {
		data += '(';
		for (auto&& i : func.arguments()) {
			RegistStructType(i.type());
			CodegenUtility::GetTypeName(*i.type(), data);
			data += ' ';
			CodegenUtility::GetVariableName(i, data);
			data += ',';
		}
		data[data.size() - 1] = ')';
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
		str += "struct T"_sv;
		vengine::to_string(t->index(), str);
		str += "{\n"_sv;
		size_t count = 0;
		for (auto&& mem : t->members()) {
			CodegenUtility::GetTypeName(*mem, str);
			str += " v"_sv;
			vengine::to_string(count, str);
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
	//Buffer Binding
	std::span<const Function::BufferBinding> buffers,
	std::span<const Function::TextureBinding> texs,
	vengine::string& result) {
	for (size_t i = 0; i < buffers.size(); ++i) {
		result += "StructuredBuffer<"_sv;
		auto&& var = buffers[i].variable;
		GetTypeName(*var.type(), result);
		result += "> v"_sv;
		vengine::to_string(var.uid(), result);
		result += ":register(t"_sv;
		vengine::to_string(i, result);
		result += ");\n"_sv;
	}
	//TODO: Texture Binding
}
size_t CodegenUtility::PrintGlobalVariables(
	std::span<const Variable> values,
	vengine::string& result) {
	vengine::vector<Variable const*> scalarArr;
	vengine::vector<Variable const*> vec2Arr;
	vengine::vector<Variable const*> vec3Arr;
	vengine::vector<Variable const*> vec4Arr;
	constexpr size_t ELE_SIZE = 4;
	size_t cbufferSize = 0;
	size_t alignCount = 0;
	result += "cbuffer Params:register(b0)\n{\n"_sv;
	for (auto& var : values) {
		auto&& type = *var.type();
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
				result += "float4x4 v"_sv;
				vengine::to_string(var.uid(), result);
				result += ";\n"_sv;
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
	}
	for (auto& vec4 : vec4Arr) {
		cbufferSize += ELE_SIZE * 4;
		GetTypeName(*vec4->type(), result);
		result += "4 v"_sv;
		vengine::to_string(vec4->uid(), result);
		result += ";\n"_sv;
	}
	auto PrintScalar = [&](Variable const* var) -> void {
		GetTypeName(*var->type(), result);
		result += " v"_sv;
		vengine::to_string(var->uid(), result);
		result += ";\n"_sv;
	};

	for (auto& vec3 : vec4Arr) {
		cbufferSize += ELE_SIZE * 4;
		GetTypeName(*vec3->type(), result);
		result += "3 v"_sv;
		vengine::to_string(vec3->uid(), result);
		result += ";\n"_sv;
		if (!scalarArr.empty()) {
			auto v = scalarArr.erase_last();
			PrintScalar(v);
		} else {
			result += "float __a"_sv;
			vengine::to_string(alignCount, result);
			result += ";\n"_sv;
			alignCount++;
		}
	}

	for (auto& vec2 : vec2Arr) {
		cbufferSize += ELE_SIZE * 2;
		GetTypeName(*vec2->type(), result);
		result += "2 v"_sv;
		vengine::to_string(vec2->uid(), result);
		result += ";\n"_sv;
	}

	for (auto& vec : scalarArr) {
		cbufferSize += ELE_SIZE;
		GetTypeName(*vec->type(), result);
		result += " v"_sv;
		vengine::to_string(vec->uid(), result);
		result += ";\n"_sv;
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
