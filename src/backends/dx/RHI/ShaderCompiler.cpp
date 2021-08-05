#include <RHI/ShaderCompiler.h>
#include <ast/function.h>
#include <runtime/context.h>
#include <ShaderCompile/HLSLCompiler.h>
#include <File/Path.h>
#include <Utility/BinaryReader.h>
#include <ShaderCompile/LuisaASTTranslator.h>
#include <Utility/MD5.h>
#include <util/linq.h>
namespace luisa::compute {
namespace ShaderCompiler_Global {
struct Data {
	HashMap<Function, ShaderCompiler::ConstBufferData> globalVarOffsets;
	std::mutex mtx;
};
static StackObject<Data, true> data;

ShaderCompiler::ConstBufferData* GetCBufferData(Function func) {
	using namespace ShaderCompiler_Global;
	data.New();
	ShaderCompiler::ConstBufferData* curData = nullptr;
	{
		std::lock_guard lck(data->mtx);
		auto ite = data->globalVarOffsets.Find(func);
		if (!ite) {
			ite = data->globalVarOffsets.Emplace(
				func);
		}
		curData = &ite.Value();
	}
	return curData;
}
static bool ShaderCompiler_NeedCodegen(Function kernel, vstd::string const& path, vstd::string const& md5Path, vstd::string& codegenResult, std::array<uint8_t, MD5::MD5_SIZE>& md5Result) {
	data.New();
	ShaderCompiler::ConstBufferData* curData = GetCBufferData(kernel);
	Path filePath(path);
	CodegenUtility::GetCodegen(kernel, codegenResult, curData->offsets, curData->cbufferSize);
	md5Result = MD5::GetMD5FromString(codegenResult);
	using namespace vstd::linq;
	if (!filePath.Exists())
		return true;
	{
		BinaryReader md5Reader(md5Path);
		if (!md5Reader) return true;
		vstd::vector<uint8_t> md5Data(md5Reader.GetLength());
		md5Reader.Read(reinterpret_cast<char*>(md5Data.data()), md5Data.size());
		if (vstd::array_same(md5Data, md5Result)) return false;
	}
	//TODO: Probably other checks
	return true;
}
};// namespace ShaderCompiler_Global

void ShaderCompiler::TryCompileCompute(Function func) {
	using namespace SCompile;
	vstd::string path = ".cache/"_sv;
	Path folder(path);
	folder.TryCreateDirectory();
	vstd::string fileStrPath = path;
	vstd::string uidStr = vstd::to_string(func.hash());
	fileStrPath << uidStr
				<< ".compute"_sv;
	vstd::string md5Path = path;
	md5Path << uidStr << ".md5"_sv;
	vstd::string codegenResult;
	std::array<uint8_t, MD5::MD5_SIZE> md5Result;
	//Whether need re-compile
	if (ShaderCompiler_Global::ShaderCompiler_NeedCodegen(func, fileStrPath, md5Path, codegenResult, md5Result)) {
		{
			std::ofstream ofs(fileStrPath.data(), std::ios::binary);
			ofs.write(codegenResult.data(), codegenResult.size());
		}
		vstd::string resultStr = path;
		resultStr << uidStr
				  << ".output"_sv;

		vstd::vector<ShaderVariable> vars;
		//TODO: custom data not used
		vstd::vector<char> customData;
		vstd::vector<char> resultData;
		HLSLCompiler::GetShaderVariables(
			func,
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
ShaderCompiler::ConstBufferData const& ShaderCompiler::GetCBufferData(Function func) {
	return *ShaderCompiler_Global::GetCBufferData(func);
}
#ifdef NDEBUG
DLL_EXPORT void CodegenBody(Function func, const Context& ctx) {
	SCompile::HLSLCompiler::InitRegisterData();
	luisa::compute::ShaderCompiler::TryCompileCompute(func);
}
#endif
}// namespace luisa::compute
