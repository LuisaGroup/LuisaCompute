#include <RHI/ShaderCompiler.h>
#include <ast/function.h>
#include <runtime/context.h>
#include <ShaderCompile/HLSLCompiler.h>
#include <File/Path.h>
#include <Utility/BinaryReader.h>
#include <ShaderCompile/LuisaASTTranslator.h>
#include <Utility/MD5.h>
#include <Common/linq.h>
namespace luisa::compute {
namespace ShaderCompiler_Global {
struct Data {
	HashMap<uint, ShaderCompiler::ConstBufferData> globalVarOffsets;
	std::mutex mtx;
};
static StackObject<Data, true> data;

ShaderCompiler::ConstBufferData* GetCBufferData(uint kernel_uid) {
	using namespace ShaderCompiler_Global;
	data.New();
	ShaderCompiler::ConstBufferData* curData = nullptr;
	{
		std::lock_guard lck(data->mtx);
		auto ite = data->globalVarOffsets.Find(kernel_uid);
		if (!ite) {
			ite = data->globalVarOffsets.Emplace(
				kernel_uid);
		}
		curData = &ite.Value();
	}
	return curData;
}
static bool ShaderCompiler_NeedCodegen(Function kernel, vengine::string const& path, vengine::string const& md5Path, vengine::string& codegenResult, std::array<uint8_t, MD5::MD5_SIZE>& md5Result) {
	data.New();
	ShaderCompiler::ConstBufferData* curData = GetCBufferData(kernel.uid());
	Path filePath(path);
	CodegenUtility::GetCodegen(kernel, codegenResult, curData->offsets, curData->cbufferSize);
	md5Result = MD5::GetMD5FromString(codegenResult);
	using namespace vengine::linq;
	if (!filePath.Exists()) 
		return true;
	{
		BinaryReader md5Reader(md5Path);
		vengine::vector<uint8_t> md5Data(md5Reader.GetLength());
		md5Reader.Read(reinterpret_cast<char*>(md5Data.data()), md5Data.size());
		if (vengine::array_same(md5Data, md5Result)) return false;
	}
	//TODO: Probably other checks
	return true;
}
};// namespace ShaderCompiler_Global

void ShaderCompiler::TryCompileCompute(uint32_t uid) {
	using namespace SCompile;
	auto kernel = Function::kernel(uid);
	vengine::string path = ".cache/"_sv;
	Path folder(path);
	folder.TryCreateDirectory();
	vengine::string fileStrPath = path;
	vengine::string uidStr = vengine::to_string(uid);
	fileStrPath << uidStr
				<< ".compute"_sv;
	vengine::string md5Path = path;
	md5Path << uidStr << ".md5"_sv;
	vengine::string codegenResult;
	std::array<uint8_t, MD5::MD5_SIZE> md5Result;
	//Whether need re-compile
	if (ShaderCompiler_Global::ShaderCompiler_NeedCodegen(kernel, fileStrPath, md5Path, codegenResult, md5Result)) {
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
		{
			std::ofstream ofs(resultStr.c_str(), std::ios::binary);
			ofs.write(resultData.data(), resultData.size());
		}
		{
			std::ofstream ofs(md5Path.c_str(), std::ios::binary);
			ofs.write(reinterpret_cast<char const*>(md5Result.data()), md5Result.size());
		}
	}
	//TODO: read
}
ShaderCompiler::ConstBufferData const& ShaderCompiler::GetCBufferData(uint kernel_uid) {
	return *ShaderCompiler_Global::GetCBufferData(kernel_uid);
}
#ifdef NDEBUG
DLL_EXPORT void CodegenBody(Function func, const Context& ctx) {
	vengine::vengine_init_malloc();
	SCompile::HLSLCompiler::InitRegisterData();
	luisa::compute::ShaderCompiler::TryCompileCompute(func.uid());
}
#endif
}// namespace luisa::compute
