#include <RenderComponent/IShader.h>
int32_t IShader::GetPropertyRootSigPos(uint id) const {
	auto ite = mVariablesDict.Find(id);
	if (!ite) return -1;
	return (int32_t)ite.Value();
}
bool IShader::VariableReflection(uint id, void const* targetObj, uint& rootSigPos, ShaderVariable& varResult) const {
	if (targetObj == nullptr) return false;
	auto ite = mVariablesDict.Find(id);
	if (!ite)
		return false;
	rootSigPos = ite.Value();
	varResult = mVariablesVector[rootSigPos];
	return true;
}