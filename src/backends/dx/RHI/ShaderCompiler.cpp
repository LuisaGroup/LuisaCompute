#include <RHI/ShaderCompiler.h>
#include <ast/function.h>
#include <runtime/context.h>
#include <ShaderCompile/HLSLCompiler.h>
#include <File/Path.h>
#include <Utility/BinaryReader.h>
#include <ShaderCompile/LuisaASTTranslator.h>
namespace luisa::compute {
static bool ShaderCompiler_NeedCodegen(Function kernel, vengine::string const& path, vengine::string& codegenResult) {
	Path filePath(path);
	CodegenUtility::GetCodegen(kernel, codegenResult);
	if (filePath.Exists()) {
		BinaryReader reader(path);
		if (reader.GetLength() != codegenResult.length())
			return true;
		vengine::string fileData;
		fileData.resize(codegenResult.length());
		reader.Read(fileData.data(), fileData.size());
		return true;
		//fileData != codegenResult;
	}
	return true;
}

void ShaderCompiler::TryCompileCompute(uint32_t uid) {
	using namespace SCompile;
	auto kernel = Function::kernel(uid);
	vengine::string path = "DXCompiledShader/"_sv;
	Path folder(path);
	folder.TryCreateDirectory();
	vengine::string fileStrPath = path;
	vengine::string uidStr = vengine::to_string(uid);
	fileStrPath << uidStr
				<< ".compute"_sv;
	vengine::string codegenResult;
	//Whether need re-compile
	if (ShaderCompiler_NeedCodegen(kernel, fileStrPath, codegenResult)){
		{
			std::ofstream ofs(fileStrPath.data(), std::ios::binary);
			ofs.write(codegenResult.data(), codegenResult.size());
		}
		vengine::string resultStr = path;
		resultStr << uidStr
				  << ".output"_sv;
		vengine::vector<ShaderVariable> vars;
		//TODO: custom data not used
		vengine::vector<char> customData;
		vengine::vector<char> resultData;
		HLSLCompiler::GetShaderVariables(
			kernel,
			vars);
		HLSLCompiler::CompileComputeShader(
			fileStrPath,
			vars,
			"CSMain"_sv,
			customData,
			resultData);
		std::ofstream ofs(resultStr.c_str(), std::ios::binary);
		ofs.write(resultData.data(), resultData.size());
	}
	//TODO: read
}
#ifdef NDEBUG
DLL_EXPORT void CodegenBody(Function func, const Context &ctx) {
	vengine::vengine_init_malloc();
	SCompile::HLSLCompiler::InitRegisterData(ctx);
	luisa::compute::ShaderCompiler::TryCompileCompute(func.uid());
}
#endif
}// namespace luisa::compute