#pragma once
#include <Common/GFXUtil.h>
#include <Common/VObject.h>
#include <Common/Runnable.h>
#include "ShaderUniforms.h"
#include <ast/function.h>
namespace SCompile {
class HLSLCompiler {
public:
	static void CompileShader(
		vengine::string const& fileName,
		vengine::vector<ShaderVariable> const& vars,
		vengine::vector<PassDescriptor> const& passDescs,
		vengine::vector<char> const& customData,
		vengine::string const& tempFilePath,
		vengine::vector<char>& resultData);
	static void CompileComputeShader(
		vengine::string const& fileName,
		vengine::vector<ShaderVariable> const& vars,
		vengine::vector<KernelDescriptor> const& passDescs,
		vengine::vector<char> const& customData,
		vengine::string const& tempFilePath,
		vengine::vector<char>& resultData);
	static void CompileDXRShader(
		vengine::string const& fileName,
		vengine::vector<ShaderVariable> const& vars,
		CompileDXRHitGroup const& passDescs,
		uint64 raypayloadMaxSize,
		uint64 recursiveCount,
		vengine::vector<char> const& customData,
		vengine::string const& tempFilePath,
		vengine::vector<char>& resultData);
	//TODO: Texture Binding, Bindless Texture
	static void GetShaderVariables(
		luisa::compute::Function const& func,
		vengine::vector<ShaderVariable>& result);
};
}// namespace SCompile