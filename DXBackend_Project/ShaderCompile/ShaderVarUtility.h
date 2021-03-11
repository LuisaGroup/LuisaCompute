#pragma once
#include "ShaderVar.h"
#include <Common/Common.h>
namespace vengine {
template<>
struct hash<LCShader::Posture> {
	size_t operator()(LCShader::Posture const& v) const {
		using namespace LCShader;
		static const hash<NumVar> h;
		static const hash<NumOpType> nH;
		return h(v.v0) ^ h(v.v1) ^ nH(v.opType);
	}
};
}// namespace vengine
namespace LCShader {
class ShaderVarUtility {
	/// Get Variable Operator Function
	/// Default Type Result: v0 * v1;
	/// Custom Type Result: v0[0] * v1[0] + v0[1] * v1[1] + ...
	/// Success: result
	/// Failed: error message


public:
	static void GetVarName(NumVar const& v, vengine::string& result);
	static void GetOperatorName(NumOpType type, vengine::string& result);
	static bool GetVarFunc(Posture const& pos, vengine::string& result, NumVar& resultType);
	static bool GeneratePosturesString(HashMap<Posture, bool> const& postures, vengine::string& results);
	static bool GenerateVarDefineStrings(HashMap<NumVar, bool> const& vars, vengine::string& results);
	static bool GenerateCBuffer(vengine::vector<VarNode const*> const& inputVarNode, vengine::vector<VarNode const*>& outVarNode, vengine::vector<VarNode*>& alignValue, vengine::string& errorMessage);
};
}// namespace LCShader
