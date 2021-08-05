#pragma once
#include <Common/GFXUtil.h>
#include <util/VObject.h>
#include <util/Runnable.h>
#include <ShaderCompile/ShaderUniforms.h>
#include <ast/function.h>
#include <runtime/context.h>

namespace SCompile {
class HLSLCompiler {
public:
	static bool ErrorHappened();
	static void PrintErrorMessages();
	static void InitRegisterData();
	static void CompileComputeShader(
		vstd::string const& fileName,
		vstd::vector<ShaderVariable> const& vars,
		vstd::string const& passDesc,
		vstd::vector<char> const& customData,
		vstd::vector<char>& resultData);
	static void CompileDXRShader(
		vstd::string const& fileName,
		vstd::vector<ShaderVariable> const& vars,
		CompileDXRHitGroup const& passDescs,
		uint64 raypayloadMaxSize,
		uint64 recursiveCount,
		vstd::vector<char> const& customData,
		vstd::vector<char>& resultData);
	//TODO: Texture Binding, Bindless Texture
	static void GetShaderVariables(
		luisa::compute::Function const& func,
		vstd::vector<ShaderVariable>& result);
	static bool CheckNeedReCompile(
		std::array<uint8_t, 16> const& md5,
		vstd::string const& shaderFileName);

private:
};
}// namespace SCompile
