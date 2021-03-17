#include <ast/interface.h>
#include <ast/constant_data.h>
#include <Common/Common.h>
#include <Common/VObject.h>
#include "LuisaASTTranslator.h"

namespace luisa::compute {
#ifdef NDEBUG
DLL_EXPORT void CodegenBody(Function func) {
	vengine::vengine_init_malloc(malloc, free);
	LUISA_INFO("HLSL codegen started.");
	vengine::string string_buffer;
	string_buffer.reserve(4095u);
	auto t0 = std::chrono::high_resolution_clock::now();
	CodegenUtility::ClearStructType();
	StringStateVisitor vis;
	func.body()->accept(vis);
	{
		vengine::string str;
		str.reserve(1023u);
		CodegenUtility::PrintStructType(str);
		for (auto& i : func.constants()) {
			CodegenUtility::PrintConstant(i, str);
		}
		CodegenUtility::PrintUniform(func.captured_buffers(), func.captured_textures(), str);
		CodegenUtility::PrintGlobalVariables(func.arguments(), str);
		string_buffer += str;
		string_buffer += "\n";
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	using namespace std::chrono_literals;
	LUISA_INFO("HLSL codegen finished in {} ms.", (t1 - t0) / 1ns * 1e-6);

	std::cout << string_buffer
			  << CodegenUtility::GetFunctionDecl(func) << "\n"
			  << vis.ToString() << std::endl;
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
		str = "mul("_sv;
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
				str += "<<"_sv;
				break;
			case BinaryOp::SHR:
				str += ">>"_sv;
				break;
			case BinaryOp::AND:
				str += "&&"_sv;
				break;
			case BinaryOp::OR:
				str += "||"_sv;
				break;
			case BinaryOp::LESS:
				str += '<';
				break;
			case BinaryOp::GREATER:
				str += '>';
				break;
			case BinaryOp::LESS_EQUAL:
				str += "<="_sv;
				break;
			case BinaryOp::GREATER_EQUAL:
				str += ">="_sv;
				break;
			case BinaryOp::EQUAL:
				str += "=="_sv;
				break;
			case BinaryOp::NOT_EQUAL:
				str += "!="_sv;
				break;
		}
		expr->rhs()->accept(vis);
		str += vis.ToString();
		AfterVisit();
	}
}
void StringExprVisitor::visit(const MemberExpr* expr) {
	expr->self()->accept(*this);
	str += ".v"_sv;
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
	str = "break;\n"_sv;
}

void StringStateVisitor::visit(const ContinueStmt* state) {
	str = "continue;\n"_sv;
}

void StringStateVisitor::visit(const ReturnStmt* state) {
	if (state->expression()) {
		str = "return "_sv;
		StringExprVisitor exprVis;
		state->expression()->accept(exprVis);
		str += exprVis.ToString();
		str += ";\n"_sv;
	} else {
		str = "return;\n"_sv;
	}
}

void StringStateVisitor::visit(const ScopeStmt* state) {
	str = "{\n"_sv;
	StringStateVisitor sonVisitor;
	for (auto&& i : state->statements()) {
		i->accept(sonVisitor);
		str += sonVisitor.ToString();
	}
	str += "}\n"_sv;
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
		str += ";\n"_sv;
	} else {

		StringExprVisitor vis;
		size_t count = 0;
		if (state->initializer().size() == 1) {
			auto&& i = *state->initializer().begin();
			i->accept(vis);
			str += '=';
			str += vis.str;
			str += ";\n"_sv;
		} else {
			str += ";\n"_sv;
			for (auto&& i : state->initializer()) {
				i->accept(vis);
				str += varTempName;
				str += ".v"_sv;
				str += vengine::to_string(count);

				str += '=';
				str += vis.str;
				str += ";\n"_sv;
				count++;
			}
		}
	}
	//TODO: Different Declare in Fxxking HLSL
}

void StringStateVisitor::visit(const IfStmt* state) {
	StringExprVisitor exprVisitor;
	StringStateVisitor stateVisitor;
	str = "if ("_sv;
	state->condition()->accept(exprVisitor);
	str += exprVisitor.ToString();
	str += ')';
	state->true_branch()->accept(stateVisitor);
	str += stateVisitor.ToString();
	str += "else"_sv;
	state->false_branch()->accept(stateVisitor);
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const WhileStmt* state) {
	StringStateVisitor stateVisitor;
	StringExprVisitor exprVisitor;

	str = "[loop]\nwhile ("_sv;
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
	str += ";\n"_sv;
}

void StringStateVisitor::visit(const SwitchStmt* state) {
	str = "switch ("_sv;
	StringStateVisitor stateVisitor;
	StringExprVisitor exprVisitor;
	state->expression()->accept(exprVisitor);
	state->body()->accept(stateVisitor);
	str += exprVisitor.ToString();
	str += ')';
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const SwitchCaseStmt* state) {
	str = "case "_sv;
	StringStateVisitor stateVisitor;
	StringExprVisitor exprVisitor;
	state->expression()->accept(exprVisitor);
	state->body()->accept(stateVisitor);
	str += exprVisitor.ToString();
	str += ":\n"_sv;
	str += stateVisitor.ToString();
}

void StringStateVisitor::visit(const SwitchDefaultStmt* state) {
	str = "default:\n"_sv;
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
			str += "+="_sv;
			break;
		case AssignOp::SUB_ASSIGN:
			str += "-="_sv;
			break;
		case AssignOp::MUL_ASSIGN:
			str += "*="_sv;
			break;
		case AssignOp::DIV_ASSIGN:
			str += "/="_sv;
			break;
		case AssignOp::MOD_ASSIGN:
			str += "%="_sv;
			break;
		case AssignOp::BIT_AND_ASSIGN:
			str += "&="_sv;
			break;
		case AssignOp::BIT_OR_ASSIGN:
			str += "|="_sv;
			break;
		case AssignOp::BIT_XOR_ASSIGN:
			str += "^="_sv;
			break;
		case AssignOp::SHL_ASSIGN:
			str += "<<="_sv;
			break;
		case AssignOp::SHR_ASSIGN:
			str += ">>="_sv;
			break;
	}
	state->rhs()->accept(exprVisitor);
	str += exprVisitor.ToString();
	str += ";\n"_sv;
}

void StringStateVisitor::visit(ForStmt const* expr) {
	StringStateVisitor stmtVis;
	StringExprVisitor expVis;
	str = "[loop]\nfor("_sv;
	expr->initialization()->accept(stmtVis);
	str += stmtVis.ToString();
	str += ';';
	expr->condition()->accept(expVis);
	str += expVis.ToString();
	str += ';';
	expr->update()->accept(stmtVis);
	str += stmtVis.ToString();
	str += ')';
	expr->body()->accept(stmtVis);
	str += stmtVis.ToString();
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
			return "bool"_sv;
		case Type::Tag::FLOAT:
			return "float"_sv;
		case Type::Tag::INT:
			return "int"_sv;
		case Type::Tag::UINT:
			return "uint"_sv;

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

vengine::string CodegenUtility::GetFunctionDecl(Function func) {
	vengine::string data;
	data.reserve(1023u);
	if (func.return_type()) {
		data = CodegenUtility::GetTypeName(*func.return_type());
	} else {
		data = "void"_sv;
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
	data += vengine::to_string(func.uid());
	if (func.arguments().empty()) {
		data += "()"_sv;
	} else {
		data += '(';
		for (auto&& i : func.arguments()) {
			RegistStructType(i.type());
			data += CodegenUtility::GetTypeName(*i.type());
			data += ' ';
			data += CodegenUtility::GetVariableName(i);
			data += ',';
		}
		data[data.size() - 1] = ')';
	}
	return data;
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
		result += vengine::to_string(binding.hash);

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
		str += vengine::to_string(t->index());
		str += "{\n"_sv;
		size_t count = 0;
		for (auto&& mem : t->members()) {
			str += CodegenUtility::GetTypeName(*mem);
			str += " v"_sv;
			str += vengine::to_string(count);
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
		result += GetTypeName(*var.type());
		result += "> v"_sv;
		result += vengine::to_string(var.uid());
		result += ":register(t"_sv;
		result += vengine::to_string(i);
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
	result += "cbuffer Params:register(b0){"_sv;
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
				result += vengine::to_string(var.uid());
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
		result += GetTypeName(*vec4->type());
		result += "4 v"_sv;
		result += vengine::to_string(vec4->uid());
		result += ";\n"_sv;
	}
	auto PrintScalar = [&](Variable const* var) -> void {
		result += GetTypeName(*var->type());
		result += " v"_sv;
		result += vengine::to_string(var->uid());
		result += ";\n"_sv;
	};

	for (auto& vec3 : vec4Arr) {
		cbufferSize += ELE_SIZE * 4;
		result += GetTypeName(*vec3->type());
		result += "3 v"_sv;
		result += vengine::to_string(vec3->uid());
		result += ";\n"_sv;
		if (!scalarArr.empty()) {
			auto v = scalarArr.erase_last();
			PrintScalar(v);
		} else {
			result += "float __a"_sv;
			result += vengine::to_string(alignCount);
			result += ";\n"_sv;
			alignCount++;
		}
	}

	for (auto& vec2 : vec2Arr) {
		cbufferSize += ELE_SIZE * 2;
		result += GetTypeName(*vec2->type());
		result += "2 v"_sv;
		result += vengine::to_string(vec2->uid());
		result += ";\n"_sv;
	}

	for (auto& vec : scalarArr) {
		cbufferSize += ELE_SIZE;
		result += GetTypeName(*vec->type());
		result += " v"_sv;
		result += vengine::to_string(vec->uid());
		result += ";\n"_sv;
	}
	result += "}\n"_sv;//End cbuffer
	return cbufferSize;
}
void CodegenUtility::SeparateVariables(
	Function const& func,
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
