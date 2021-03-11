#include "ShaderVarUtility.h"
#include <Common/Common.h>
#include <Utility/QuickSort.h>
#include <ast/type.h>
#include <ast/expression.h>
namespace LCShader {

void ShaderVarUtility::GetVarName(NumVar const& v, vengine::string& varTypeName) {
	if (v.IsDefaultType()) {
		switch (v.varType) {
			case NumVarType::Float:
				varTypeName = "float";
				break;
			case NumVarType::Int:
				varTypeName = "int";
				break;
			case NumVarType::UInt:
				varTypeName = "uint";
				break;
			case NumVarType::Bool:
				varTypeName = "bool";
				break;
		}
	} else {
		switch (v.varType) {
			case NumVarType::Float:
				varTypeName = "LCFloat";
				break;
			case NumVarType::Int:
				varTypeName = "LCInt";
				break;
			case NumVarType::UInt:
				varTypeName = "LCUInt";
				break;
			case NumVarType::Bool:
				varTypeName = "LCBool";
				break;
		}
	}
	if (!v.IsSingleValue()) {
		if (v.IsVector()) {
			varTypeName += vengine::to_string(v.rowLength);
		} else {
			varTypeName += vengine::to_string(v.columeLength);
			varTypeName += 'x';
			varTypeName += vengine::to_string(v.rowLength);
		}
	}
}
void ShaderVarUtility::GetOperatorName(NumOpType type, vengine::string& result) {
	switch (type) {
		case NumOpType::And:
			result = "AND";
			break;
		case NumOpType::Xor:
			result = "XOR";
			break;
		case NumOpType::Or:
			result = "OR";
			break;
		case NumOpType::Left:
			result = "LEFT";
			break;
		case NumOpType::Right:
			result = "RIGHT";
			break;
		case NumOpType::Add:
			result = "ADD";
			break;
		case NumOpType::Sub:
			result = "SUB";
			break;
		case NumOpType::Div:
			result = "DIV";
			break;
		case NumOpType::Mul:
			result = "MUL";
			break;
		case NumOpType::LogicAnd:
			result = "LOGICAND";
			break;
		case NumOpType::LogicOr:
			result = "LOGICOR";
			break;
	}
}
bool ShaderVarUtility::GetVarFunc(Posture const& pos, vengine::string& result, NumVar& resultVarType) {
	auto&& v0 = pos.v0;
	auto&& v1 = pos.v1;
	auto&& opType = pos.opType;
	bool v0IsVector = v0.IsVector();
	bool v0IsSingle = v0.IsSingleValue();
	bool v1IsVector = v1.IsVector();
	bool v1IsSingle = v1.IsSingleValue();
	//////// Type Check
	switch (opType) {
		case NumOpType::And:
		case NumOpType::Or:
		case NumOpType::Xor:
		case NumOpType::Left:
		case NumOpType::Right:
			if (v0.varType == NumVarType::Float || v1.varType == NumVarType::Float) {
				result = "Can not use float format!";
				return false;
			}
			break;
		case NumOpType::LogicAnd:
		case NumOpType::LogicOr:
			if (v0.varType != NumVarType::Bool || v1.varType != NumVarType::Bool) {
				result = "Can not use non-bool format!";
				return false;
			}
			break;

	}
	if (!(v0IsSingle || v1IsSingle)) {
		switch (opType) {
			case NumOpType::Mul:
				if (v1IsVector) {
					if (v1.rowLength != v0.rowLength) {
						result = "Wrong Format!";
						return false;
					}
				} else if (v0IsVector) {
					if (v0.rowLength != v1.columeLength) {
						result = "Wrong Format!";
						return false;
					}
				} else {
					if (v0.rowLength != v1.columeLength) {
						result = "Wrong Format!";
						return false;
					}
				}
				break;
			case NumOpType::Add:
			case NumOpType::Sub:
			case NumOpType::Div:
			case NumOpType::And:
			case NumOpType::Xor:
			case NumOpType::Or:
			case NumOpType::Left:
			case NumOpType::Right:
			case NumOpType::LogicAnd:
			case NumOpType::LogicOr:
				if (v0 != v1) {
					result = "Wrong Format!";
					return false;
				}
				break;
			
		}
	}
	//////// Result Type
	if (v0.IsSingleValue()) {
		resultVarType = v1;
	} else if (v1.IsSingleValue()) {
		resultVarType = v0;
	} else {
		switch (opType) {
			case NumOpType::Mul:
				if (v1IsVector) {
					resultVarType.varType = v0.varType;
					resultVarType.rowLength = v1.rowLength;
					resultVarType.columeLength = 1;
				} else if (v0IsVector) {
					resultVarType.rowLength = v1.rowLength;
					resultVarType.columeLength = 1;
					resultVarType.varType = v0.varType;
				} else {
					resultVarType.varType = v0.varType;
					resultVarType.rowLength = v1.rowLength;
					resultVarType.columeLength = v0.columeLength;
				}
				break;
			case NumOpType::Add:
			case NumOpType::Sub:
			case NumOpType::Div:
			case NumOpType::And:
			case NumOpType::Or:
			case NumOpType::Xor:
			case NumOpType::Left:
			case NumOpType::Right:
			case NumOpType::LogicAnd:
			case NumOpType::LogicOr:
				resultVarType = v0;
				break;
		}
	}
	if (v0.varType == NumVarType::Float || v1.varType == NumVarType::Float) {
		resultVarType.varType = NumVarType::Float;
	}
	result.clear();

	if (!(v0.IsDefaultType() && v1.IsDefaultType())) {

		auto IndexString = [](uint2 ind, bool isVector, bool isSingle) -> vengine::string {
			if (isSingle) {
				return "";
			}
			if (isVector) {
				uint v = (ind.x == 0) ? ind.y : ind.x;
				vengine::string str = "[";
				str += vengine::to_string(v);
				str += ']';
				return str;
			} else {
				vengine::string str = "[";
				str += vengine::to_string(ind.y);
				str += "][";
				str += vengine::to_string(ind.x);
				str += ']';
				return str;
			}
		};
		//Vector

		GetVarName(resultVarType, result);
		result += " result;\n";
		bool resultIsVector = resultVarType.IsVector();
		bool resultIsSingle = resultVarType.IsSingleValue();

		//TODO: Calculate
		auto IterateAllElements = [&](char const* opt) -> void {
			for (uint x = 0; x < resultVarType.rowLength; ++x)
				for (uint y = 0; y < resultVarType.columeLength; ++y) {
					result += "result.v";
					result += IndexString(uint2(x, y), resultIsVector, resultIsSingle);
					result += '=';
					result += "v0.v";
					result += IndexString(uint2(x, y), v0IsVector, v0IsSingle);
					result += opt;
					result += "v1.v";
					result += IndexString(uint2(x, y), v1IsVector, v1IsSingle);
					result += ";\n";
				}
		};
		switch (opType) {
			case NumOpType::Add:
				IterateAllElements("+");
				break;
			case NumOpType::Sub:
				IterateAllElements("-");
				break;
			case NumOpType::Mul:
				//TODO
				if ((v0IsSingle || v1IsSingle)
					|| (v0IsVector && v1IsVector)) {
					IterateAllElements("*");
				} else {
					for (uint x = 0; x < resultVarType.rowLength; ++x)
						for (uint y = 0; y < resultVarType.columeLength; ++y) {
							result += "result.v";
							result += IndexString(uint2(x, y), resultIsVector, resultIsSingle);
							result += '=';
							for (uint z = 0; z < v0.rowLength; ++z) {
								result += "v0.v";
								result += IndexString(uint2(z, y), v0IsVector, v0IsSingle);
								result += '*';
								result += "v1.v";
								result += IndexString(uint2(x, z), v1IsVector, v1IsSingle);
								if (z < v0.rowLength - 1) {
									result += '+';
								}
							}
							result += ";\n";
						}
				}
				break;
			case NumOpType::Div:
				IterateAllElements("/");
				break;
			case NumOpType::And:
				IterateAllElements("&");
				break;
			case NumOpType::Xor:
				IterateAllElements("^");
				break;
			case NumOpType::Or:
				IterateAllElements("|");
				break;
			case NumOpType::Left:
				IterateAllElements(">>");
				break;
			case NumOpType::Right:
				IterateAllElements("<<");
				break;
		}
		result += "return result;";
	}

	return true;
}
bool ShaderVarUtility::GeneratePosturesString(HashMap<Posture, bool> const& postures, vengine::string& results) {
	vengine::string fomula;
	vengine::string resultVarName;
	vengine::string v0TypeName;
	vengine::string v1TypeName;
	vengine::string operatorName;
	bool v = true;
	postures.IterateAll([&](Posture const& pos, bool v) -> bool {
		if (!v) return true;
		NumVar resultVar;
		if (!GetVarFunc(pos, fomula, resultVar)) {
			v = false;
			results = fomula;
			return false;
		}
		if (fomula.empty()) return true;
		GetVarName(resultVar, resultVarName);
		GetVarName(pos.v0, v0TypeName);
		GetVarName(pos.v1, v1TypeName);
		GetOperatorName(pos.opType, operatorName);

		results += resultVarName;
		results += " __LC_FUNC_";
		results += v0TypeName;
		results += '_';
		results += operatorName;
		results += '_';
		results += v1TypeName;
		results += "__(const ";
		results += v0TypeName;
		results += " v0,const ";
		results += v1TypeName;
		results += " v1){\n";
		results += fomula;
		results += "\n}\n";
		return true;
	});
	return v;
}
bool ShaderVarUtility::GenerateVarDefineStrings(HashMap<NumVar, bool> const& vars, vengine::string& results) {
	vengine::string varName;
	vars.IterateAll([&](NumVar const& var, bool enabled) -> void {
		if (!enabled) return;
		if (var.IsDefaultType()) return;
		results += "struct ";
		GetVarName(var, varName);
		results += varName;
		results += "{\n";
		switch (var.varType) {
			case NumVarType::Float:
				results += "float v";
				break;
			case NumVarType::UInt:
				results += "uint v";
				break;
			case NumVarType::Int:
				results += "int v";
				break;
			case NumVarType::Bool:
				results += "bool v";
				break;
		}
		if (!var.IsSingleValue()) {
			if (var.IsVector()) {
				results += '[';
				results += vengine::to_string(var.rowLength);
				results += ']';
			} else {
				results += '[';
				results += vengine::to_string(var.columeLength);
				results += "][";
				results += vengine::to_string(var.rowLength);
				results += ']';
			}
		}
		results += ";\n}\n";
	});
	return true;
}
bool ShaderVarUtility::GenerateCBuffer(vengine::vector<VarNode const*> const& inputVarNode, vengine::vector<VarNode const*>& outVarNode, vengine::vector<VarNode*>& alignValue, vengine::string& errorMessage) {
	outVarNode.clear();
	outVarNode.reserve(inputVarNode.size() * 1.5);
	vengine::vector<VarNode const*> vec3Arr;
	vengine::vector<VarNode const*> vec2Arr;
	vengine::vector<VarNode const*> singArr;
	for (auto& i : inputVarNode) {
		if (i->isConstant) continue;
		VarType const* vT = i->type;
		if (typeid(*vT) != typeid(NumVar)) {
			errorMessage = "Uniform variable cannot use custom type!";
			return false;
		}
		NumVar const* nv = static_cast<NumVar const*>(vT);
		if (nv->varType == NumVarType::Bool) {
			errorMessage = "Uniform variable can not be bool type!";
			return false;
		}
		if (!nv->IsDefaultType()) {
			errorMessage = "Uniform variable wrong format!";
			return false;
		}

		if (nv->IsVector()) {
			switch (nv->rowLength) {
				case 1:
					singArr.push_back(i);
					break;
				case 2:
					vec2Arr.push_back(i);
					break;
				case 3:
					vec3Arr.push_back(i);
					break;
				default:
					outVarNode.push_back(i);
					break;
			}

		} else {
			if (nv->rowLength != 4 || nv->columeLength != 4) {
				errorMessage = "Uniform variable wrong format!";
				return false;
			}
			outVarNode.push_back(i);
		}
	}
	uint alignIndex = 0;
	for (auto& i : vec3Arr) {
		outVarNode.push_back(i);
		if (singArr.empty()) {
			NumVar* alignVar = new NumVar();
			alignVar->rowLength = 1;
			alignVar->columeLength = 1;
			alignVar->varType = NumVarType::Float;
			VarNode* vn = new VarNode();
			vn->isConstant = false;
			vn->type = alignVar;
			vn->valueName = "__GLOBAL_ALIGN_VALUE_";
			vn->valueName += vengine::to_string(alignIndex);
			vn->valueName += "__";
			alignIndex++;
			alignValue.push_back(vn);
			outVarNode.push_back(vn);
		} else {
			outVarNode.push_back(singArr.erase_last());
		}
	}
	for (auto& i : vec2Arr) {
		outVarNode.push_back(i);
	}
	for (auto& i : singArr) {
		outVarNode.push_back(i);
	}
	return true;
}
}// namespace LCShader