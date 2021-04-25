#pragma once
#include <Common/GFXUtil.h>
#include <Common/VObject.h>
#include <Common/Runnable.h>
#include <ShaderCompile/ShaderUniforms.h>
#include <ast/function.h>
#include <runtime/context.h>

namespace SCompile {
class HLSLCompiler {
public:
	static bool ErrorHappened();
	static void PrintErrorMessages();
	static void InitRegisterData();
	static void CompileShader(
		vengine::string const& fileName,
		vengine::vector<ShaderVariable> const& vars,
		vengine::vector<PassDescriptor> const& passDescs,
		vengine::vector<char> const& customData,
		vengine::vector<char>& resultData);
	static void CompileComputeShader(
		vengine::string const& fileName,
		vengine::vector<ShaderVariable> const& vars,
		vengine::string const& passDesc,
		vengine::vector<char> const& customData,
		vengine::vector<char>& resultData);
	static void CompileDXRShader(
		vengine::string const& fileName,
		vengine::vector<ShaderVariable> const& vars,
		CompileDXRHitGroup const& passDescs,
		uint64 raypayloadMaxSize,
		uint64 recursiveCount,
		vengine::vector<char> const& customData,
		vengine::vector<char>& resultData);
	//TODO: Texture Binding, Bindless Texture
	static void GetShaderVariables(
		luisa::compute::Function const& func,
		vengine::vector<ShaderVariable>& result);
	static bool CheckNeedReCompile(
		std::array<uint8_t, 16> const& md5,
		vengine::string const& shaderFileName);

private:
};
}// namespace SCompile
