#pragma once
#include "../Common/Common.h"
namespace LCShader {
struct IDisposable {
	virtual ~IDisposable() noexcept = default;
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};
////////////////////////////////////////////////////////////////////////////////////////// Enums
enum class GPUResourceType : uint32_t {
	Buffer,			  //StructuredBuffer<float> buffer;
	Texture,		  //Texture2D<float> tex;
	BufferSparseArray,//StructuredBuffer<float> buffer[];
	TextureSparseArray//Texture2D<float> tex[];
};
enum class TextureDimension : uint32_t {
	Tex2D,
	Tex2DArray,
	Tex3D
};
////////////////////////////////////////////////////////////////////////////////////////// Global
struct VarType : public IDisposable {
	virtual ~VarType() noexcept = default;

protected:
	VarType() {}
};
enum class NumOpType : uint32_t {
	Add,
	Sub,
	Mul,
	Div,
	And,
	Or,
	Left,
	Right
};
enum class NumVarType : uint32_t {
	Int,
	UInt,
	Float,
	Bool
};

struct NumVar final : public VarType {

	uint32_t rowLength;
	uint32_t columeLength;
	NumVarType varType;
	bool IsVector() const {
		return (rowLength == 1) || (columeLength == 1);
	}

	bool Check() {
		//Empty NumVar Not Allowed
		if ((rowLength == 0) || (columeLength == 0))
			return false;
		if (IsVector() && (columeLength != 1)) {
			auto a = columeLength;
			columeLength = rowLength;
			rowLength = a;
		}
		return true;
	}
	bool IsDefaultType() const {
		return (rowLength <= 4) && (columeLength <= 4);
	}

	bool IsSingleValue() const {
		return (rowLength == 1) && (columeLength == 1);
	}
	bool operator==(NumVar const& v) const {
		return (rowLength == v.rowLength) && (columeLength == v.columeLength);// No Type Here
	}
	bool operator!=(NumVar const& v) const {
		return !operator==(v);
	}
};
struct VarNode;
struct CustomVarType final : public VarType {
	vengine::string typeName;
	vengine::vector<VarNode const*> definedVars;
};

struct TextureType final : public VarType {
	vengine::string texName;
	NumVar type;
	TextureDimension dimension;
};
////////////////////////////////////////////////////////////////////////////////////////// Logic

struct LogicASTNode : public IDisposable {
	virtual ~LogicASTNode() noexcept = default;

protected:
	LogicASTNode() {}
};

struct ConstantValueNode final : public LogicASTNode {
	enum class ConstVarType {
		Float,
		UInt,
		Int
	};
	struct ConstVar {
		ConstVarType type;
		union {
			float floatValue;
			int32_t intValue;
			uint32_t uintValue;
		};
	};
	vengine::vector<ConstVar> vars;
};
struct VarNode final : public LogicASTNode {
	VarType const* type;
	vengine::string valueName;
	vengine::vector<uint> arrLength;
	bool isConstant;
};
struct VarRefNode final : public LogicASTNode {
	VarNode const* type;
	VarRefNode const* subObj;
	vengine::vector<LogicASTNode const*> arrIndices;
};
struct OperationNode final : public LogicASTNode {
	LogicASTNode const* leftValue;
	LogicASTNode const* rightValue;
	NumOpType opType;
};
struct FunctionDefine final : public IDisposable {
	vengine::string funcName;
	VarType const* ret;
	vengine::vector<std::pair<VarType const*, vengine::string>> args;
	vengine::vector<LogicASTNode const*> logics;
};
struct FunctionCallNode final : public LogicASTNode {
	FunctionDefine const* targetFunc;
	vengine::vector<LogicASTNode const*> args;
};
struct IfNode final : public LogicASTNode {
	LogicASTNode const* conditionNode;
	vengine::vector<LogicASTNode const*> trueNodes;
	vengine::vector<LogicASTNode const*> falseNodes;
};
struct WhileNode final : public LogicASTNode {
	LogicASTNode const* conditionNode;
	vengine::vector<LogicASTNode const*> logicNodes;
};
struct AssignNode final : public LogicASTNode {
	VarRefNode const* leftValue;
	LogicASTNode const* rightValue;
};
}// namespace LCShader