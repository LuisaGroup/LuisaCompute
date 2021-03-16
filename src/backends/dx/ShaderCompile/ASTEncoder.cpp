#include "ASTEncoder.h"
#include <Common/DLL.h>
#include <Utility/MD5.h>
namespace luisa::compute {
template<typename T>
struct GetBinary {
	static std::array<uint8_t, sizeof(T)> Get(T const& value) {
		std::array<uint8_t, sizeof(T)> arr{};
		if constexpr (std::is_same_v<bool, T>) {
			arr[0] = (uint8_t)value;
		} else if constexpr (std::is_same_v<float, T>) {
			*(float*)arr.data() = value;
			//return "float"_sv;
		} else if constexpr (std::is_same_v<int, T>) {
			*(int*)arr.data() = value;
		} else if constexpr (std::is_same_v<uint, T>) {
			*(uint*)arr.data() = value;
		} else {
			static_assert(always_false_v<T>, "Unknown Type!");
		}
		return arr;
	}
};
template<typename EleType, size_t N>
struct GetBinary<Vector<EleType, N>> {
	using ArrayType = std::array<uint8_t, sizeof(EleType) * N>;
	static ArrayType Get(Vector<EleType, N> const& value) {
		ArrayType sumArray{};
		for (size_t i = 0; i < N; ++i) {
			auto arr = GetBinary<EleType>::Get(value[i]);
			constexpr size_t byteSize = arr.size();
			memcpy(&sumArray[i * byteSize], arr.data(), byteSize);
		}
		return sumArray;
	}
};
template<size_t N>
struct GetBinary<Matrix<N>> {
	using ArrayType = std::array<uint8_t, N * N * sizeof(float)>;
	static ArrayType Get(Matrix<N> const& value) {
		ArrayType sumArray{};
		for (size_t i = 0; i < N; ++i) {
			auto arr = GetBinary<Vector<float, N>>::Get(value[i]);
			constexpr size_t byteSize = arr.size();
			memcpy(&sumArray[i * byteSize], arr.data(), byteSize);
		}
		return sumArray;
	}
};

static void PushToVector(uint8_t const* dd, size_t sz, vengine::vector<uint8_t>* data) {
	size_t offset = data->size();
	data->resize(offset + sz);
	memcpy(data->data() + offset, dd, sz);
}

static void GetTypeBinary(Type const* type, vengine::vector<uint8_t>* result) {
	if (!type) return;
	struct Data {
		uint typeTag;
		uint size;
		uint dim;
	};
	Data data;
	data.typeTag = (uint)type->tag() + 65536 * 256;
	data.size = (uint)type->size() + 65536 * 257;
	data.dim = 1;
	switch (type->tag()) {
		case Type::Tag::ARRAY:
		case Type::Tag::MATRIX:
		case Type::Tag::VECTOR:
			data.dim = (uint)type->dimension();
			GetTypeBinary(type->element(), result);
			break;
		case Type::Tag::STRUCTURE:
			for (auto& subType : type->members()) {
				GetTypeBinary(subType, result);
			}
			break;
	}
	data.dim += 65536 * 258;
	PushToVector((uint8_t const*)&data, sizeof(data), result);
	auto hash = type->hash();
	PushToVector((uint8_t const*)&hash, sizeof(hash), result);
}

static void GetVariableBinary(Variable const& var, vengine::vector<uint8_t>* result) {
	GetTypeBinary(var.type(), result);
	uint data = var.uid() + 65536 * 259;
	uint data2 = (uint)var.tag() + 65536 * 260;
	PushToVector((uint8_t const*)&data, sizeof(data), result);
	PushToVector((uint8_t const*)&data2, sizeof(data2), result);
}

void ASTExprEncoder::visit(const UnaryExpr* expr) {
	expr->operand()->accept(*this);
	Push<uint>((uint)(expr->op()) + 65536 * 1);
}
void ASTExprEncoder::visit(const BinaryExpr* expr) {
	expr->lhs()->accept(*this);
	Push<uint>((uint)(expr->op()) + 65536 * 2);
	expr->rhs()->accept(*this);
}
void ASTExprEncoder::visit(const MemberExpr* expr) {
	expr->self()->accept(*this);
	Push<uint>((uint)expr->member_index() + 65536 * 3);
}
void ASTExprEncoder::visit(const AccessExpr* expr) {
	expr->range()->accept(*this);
	Push<uint>(65536 * 4);
	expr->index()->accept(*this);
}
void ASTExprEncoder::visit(const LiteralExpr* expr) {
	Push(expr->value().index() + 65536 * 5);
	size_t sz = data->size();
	std::visit([&](auto&& value) -> void {
		auto arr = GetBinary<std::remove_cvref_t<decltype(value)>>::Get(value);
		Push(arr.data(), arr.size());
	},
			   expr->value());
	//sz = data->size() - sz;
//	for (size_t i = sz; i < data->size(); ++i) {
//		std::cout << (uint)(*data)[i];
//	}
}
void ASTExprEncoder::visit(const RefExpr* expr) {
	Push<uint>(65536 * 6);
	GetVariableBinary(expr->variable(), data);
}
void ASTExprEncoder::visit(const ConstantExpr* expr) {
	Push<uint>(65536 * 7);
	Push(expr->hash());
}
void ASTExprEncoder::visit(const CallExpr* expr) {
	Push((uint8_t const*)expr->name().data(), expr->name().size());
	for (auto& arg : expr->arguments()) {
		arg->accept(*this);
	}

	Push<uint>(65536 * 8 + expr->uid());
}
void ASTExprEncoder::visit(const CastExpr* expr) {
	Push<uint>(65536 * 9 + (uint)expr->op());
	expr->expression()->accept(*this);
}

void ASTExprEncoder::Push(uint8_t const* dd, size_t sz) {
	PushToVector(dd, sz, data);
}

ASTExprEncoder::~ASTExprEncoder() {
}

void ASTStmtEncoder::visit(const BreakStmt* stmt) {
	Push<uint>(65536 * 10);
}
void ASTStmtEncoder::visit(const ContinueStmt* stmt) {
	Push<uint>(65536 * 11);
}
void ASTStmtEncoder::visit(const ReturnStmt* stmt) {
	Push<uint>(65536 * 12);
	ASTExprEncoder expr(data);
	stmt->expression()->accept(expr);
}
void ASTStmtEncoder::visit(const ScopeStmt* stmt) {
	Push<uint>(65536 * 13);
	for (auto& st : stmt->statements()) {
		st->accept(*this);
	}
}
void ASTStmtEncoder::visit(const DeclareStmt* stmt) {
	Push<uint>(65536 * 14);
	GetVariableBinary(stmt->variable(), data);
	ASTExprEncoder expr(data);
	Push<uint>(65536 * 15);
	for (auto& i : stmt->initializer()) {
		i->accept(expr);
	}
}
void ASTStmtEncoder::visit(const IfStmt* stmt) {
	Push<uint>(65536 * 16);
	ASTExprEncoder expr(data);
	stmt->condition()->accept(expr);
	stmt->true_branch()->accept(*this);
	Push<uint>(65536 * 17);
	stmt->false_branch()->accept(*this);
}
void ASTStmtEncoder::visit(const WhileStmt* stmt) {
	Push<uint>(65536 * 18);
	ASTExprEncoder expr(data);
	stmt->condition()->accept(expr);
	Push<uint>(65536 * 19);
	stmt->body()->accept(*this);
}
void ASTStmtEncoder::visit(const ExprStmt* stmt) {
	Push<uint>(65536 * 20);
	ASTExprEncoder expr(data);
	stmt->expression()->accept(expr);
}
void ASTStmtEncoder::visit(const SwitchStmt* stmt) {
	Push<uint>(65536 * 21);
	ASTExprEncoder expr(data);
	stmt->expression()->accept(expr);
	Push<uint>(65536 * 22);
	stmt->body()->accept(*this);
}
void ASTStmtEncoder::visit(const SwitchCaseStmt* stmt) {
	Push<uint>(65536 * 23);
	ASTExprEncoder expr(data);
	stmt->expression()->accept(expr);
	Push<uint>(65536 * 24);
	stmt->body()->accept(*this);
}
void ASTStmtEncoder::visit(const SwitchDefaultStmt* stmt) {
	Push<uint>(65536 * 25);
	stmt->body()->accept(*this);
}
void ASTStmtEncoder::visit(const AssignStmt* stmt) {
	ASTExprEncoder expr(data);
	stmt->lhs()->accept(expr);
	Push<uint>(65536 * 26);
	stmt->rhs()->accept(expr);
	Push<uint>(65536 * 27 + (uint)stmt->op());
}
void ASTStmtEncoder::visit(const ForStmt* stmt) {
	ASTExprEncoder expr(data);
	Push<uint>(65536 * 28);
	stmt->initialization()->accept(*this);
	Push<uint>(65536 * 29);
	stmt->condition()->accept(expr);
	Push<uint>(65536 * 30);
	stmt->update()->accept(*this);
	Push<uint>(65536 * 31);
	stmt->update()->accept(*this);
}
void ASTStmtEncoder::Push(uint8_t const* dd, size_t sz) {
	PushToVector(dd, sz, data);
}
ASTStmtEncoder::~ASTStmtEncoder() {
}

void SerializeMD5_Result(Function func, vengine::vector<uint8_t>& result) {
	result.push_back(0);
	for (auto& i : func.shared_variables()) {
		GetVariableBinary(i, &result);
	}
	result.push_back(1);
	for (auto& i : func.arguments()) {
		GetVariableBinary(i, &result);
	}
	result.push_back(2);
	for (auto& i : func.custom_callables()) {
		auto customFunc = Function::callable(i);
		SerializeMD5_Result(customFunc, result);
	}
	result.push_back(3);
	GetTypeBinary(func.return_type(), &result);
	result.push_back(4);
	for (auto& i : func.constants()) {
		GetTypeBinary(i.type, &result);
		PushToVector((uint8_t const*)&i.hash, sizeof(i.hash), &result);
	}
	result.push_back(5);
	for (auto& i : func.captured_buffers()) {
		GetVariableBinary(i.variable, &result);
	}
	result.push_back(6);
	for (auto& i : func.arguments()) {
		GetVariableBinary(i, &result);
	}
	result.push_back(7);
	ASTStmtEncoder encoder(&result);
	func.body()->accept(encoder);
	//TODO: texture binding
}

DLL_EXPORT void SerializeMD5(Function func) {
	
	LUISA_INFO("Hello, Husky!");
	
	vengine::vengine_init_malloc(malloc, free);
//	vengine::vengine_init_malloc();
	vengine::vector<uint8_t> result;
	result.reserve(65536);
	
	LUISA_INFO("HLSL codegen started.");
	auto t_husky = std::chrono::high_resolution_clock::now();
	SerializeMD5_Result(func, result);
	auto t_husky_stop = std::chrono::high_resolution_clock::now();
	using namespace std::chrono_literals;
	LUISA_INFO("HLSL codegen finished in {} ms.", (t_husky_stop - t_husky) / 1ns * 1e-6);
	
	std::span<uint8_t> arr(result.data(), result.size());

	MD5 md5(arr);
	auto&& md5Data = md5.GetDigest();
	auto intData = (uint64_t const*)md5Data.data();
	std::cout << std::endl;

	std::cout << intData[0] << intData[1];

	std::cout << std::endl;
}
}// namespace luisa::compute
