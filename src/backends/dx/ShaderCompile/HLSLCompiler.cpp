#include <ShaderCompile/HLSLCompiler.h>
#include <CJsonObject/CJsonObject.hpp>
#include <Utility/StringUtility.h>
#include <fstream>
#include <Utility/BinaryReader.h>
#include <File/Path.h>
#include <ShaderCompile/ShaderUniforms.h>
#include <Utility/BinaryReader.h>
namespace SCompile {
static constexpr bool g_needCommandOutput = false;
static const HANDLE g_hChildStd_IN_Rd = NULL;
static const HANDLE g_hChildStd_IN_Wr = NULL;
static const HANDLE g_hChildStd_OUT_Rd = NULL;
static const HANDLE g_hChildStd_OUT_Wr = NULL;
struct ProcessorData {
	_PROCESS_INFORMATION piProcInfo;
	bool bSuccess;
};
void CreateChildProcess(vstd::string const& cmd, ProcessorData* data) {
	if constexpr (g_needCommandOutput) {
		std::cout << cmd << std::endl;
		system(cmd.c_str());
		memset(data, 0, sizeof(ProcessorData));
		return;
	}

	PROCESS_INFORMATION piProcInfo;

	static HANDLE g_hInputFile = NULL;

	//PROCESS_INFORMATION piProcInfo;
	STARTUPINFO siStartInfo;
	BOOL bSuccess = FALSE;

	// Set up members of the PROCESS_INFORMATION structure.

	ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));

	// Set up members of the STARTUPINFO structure.
	// This structure specifies the STDIN and STDOUT handles for redirection.

	ZeroMemory(&siStartInfo, sizeof(STARTUPINFO));
	siStartInfo.cb = sizeof(STARTUPINFO);
	siStartInfo.hStdError = g_hChildStd_OUT_Wr;
	siStartInfo.hStdOutput = g_hChildStd_OUT_Wr;
	siStartInfo.hStdInput = g_hChildStd_IN_Rd;
	siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

// Create the child process.
#ifdef UNICODE
	bSuccess = CreateProcess(NULL,
							 vstd::wstring(cmd).data(),// command line
							 NULL,						  // process security attributes
							 NULL,						  // primary thread security attributes
							 TRUE,						  // handles are inherited
							 0,							  // creation flags
							 NULL,						  // use parent's environment
							 NULL,						  // use parent's current directory
							 &siStartInfo,				  // STARTUPINFO pointer
							 &piProcInfo);				  // receives PROCESS_INFORMATION
#else
	bSuccess = CreateProcess(NULL,
							 cmd.data(),  // command line
							 NULL,		  // process security attributes
							 NULL,		  // primary thread security attributes
							 TRUE,		  // handles are inherited
							 0,			  // creation flags
							 NULL,		  // use parent's environment
							 NULL,		  // use parent's current directory
							 &siStartInfo,// STARTUPINFO pointer
							 &piProcInfo);// receives PROCESS_INFORMATION
#endif
	data->bSuccess = bSuccess;
	data->piProcInfo = piProcInfo;
}

void WaitChildProcess(ProcessorData* data)
// Create a child process that uses the previously created pipes for STDIN and STDOUT.
{

	// If an error occurs, exit the application.
	if (data->bSuccess) {
		auto&& piProcInfo = data->piProcInfo;
		// Close handles to the child process and its primary thread.
		// Some applications might keep these handles to monitor the status
		// of the child process, for example.
		WaitForSingleObject(piProcInfo.hProcess, INFINITE);
		WaitForSingleObject(piProcInfo.hThread, INFINITE);
		CloseHandle(piProcInfo.hProcess);
		CloseHandle(piProcInfo.hThread);
		// Close handles to the stdin and stdout pipes no longer needed by the child process.
		// If they are not explicitly closed, there is no way to recognize that the child process has ended.

		CloseHandle(g_hChildStd_OUT_Wr);
		CloseHandle(g_hChildStd_IN_Rd);
	}
}
vstd::string tempPath;

vstd::string fxcStart;
vstd::string dxcStart;
vstd::string shaderTypeCmd;
vstd::string funcName;
vstd::string output;
vstd::string macro_compile;
vstd::string dxcversion;
vstd::string dxcpath;
vstd::string fxcversion;
vstd::string fxcpath;
vstd::string pathFolder;
static spin_mutex outputMtx;
static vstd::vector<vstd::string> errorMessage;

enum class Compiler : bool {
	DXC = false,
	FXC = true
};

Compiler computeCompilerUsage;
Compiler rasterizeCompilerUsage;
Compiler rayTracingCompilerUsage;
std::atomic_bool inited = false;
void HLSLCompiler::InitRegisterData() {
	if (inited.exchange(true)) return;
	vstd::string folderPath = "VEngineCompiler/CompilerToolkit"_sv;

	shaderTypeCmd = " /T "_sv;
	funcName = " /E "_sv;
	output = " /Fo "_sv;
	macro_compile = " /D "_sv;
	using namespace neb;
	std::unique_ptr<CJsonObject> obj(ReadJson(folderPath + "/register.json"_sv));
	if (!obj) {
		VEngine_Log("Register.json not found in HLSLCompiler folder!"_sv);
		VENGINE_EXIT;
	}
	vstd::string value;
	CJsonObject sonObj;
	auto GenerateSettings = [&](vstd::string& settings) -> void {
		settings.clear();
		int sz = sonObj.GetArraySize();
		static vstd::string SplitString = " /"_sv;
		for (int i = 0; i < sz; ++i) {
			if (sonObj.Get(i, value))
				settings += SplitString + value;
		}
	};
	if (obj->Get("DXC"_sv, sonObj)) {
		if (sonObj.Get("SM"_sv, value)) {
			dxcversion = value;
		}
		if (sonObj.Get("Path"_sv, value)) {
			dxcpath = vstd::string("VEngineCompiler/CompilerToolkit/"_sv) + value;
		}
		if (sonObj.Get("Settings"_sv, sonObj) && sonObj.IsArray()) {
			GenerateSettings(dxcStart);
		}
	}
	if (obj->Get("FXC"_sv, sonObj)) {
		if (sonObj.Get("SM"_sv, value)) {
			fxcversion = value;
		}
		if (sonObj.Get("Path"_sv, value)) {
			fxcpath = vstd::string("VEngineCompiler/CompilerToolkit/"_sv) + value;
		}
		if (sonObj.Get("Settings"_sv, sonObj) && sonObj.IsArray()) {
			GenerateSettings(fxcStart);
		}
	}
	if (obj->Get("CompilerUsage"_sv, sonObj)) {
		if (sonObj.Get("Rasterize"_sv, value)) {
			StringUtil::ToLower(value);
			rasterizeCompilerUsage = (Compiler)(value == "fxc"_sv);
		}
		if (sonObj.Get("RayTracing"_sv, value)) {
			StringUtil::ToLower(value);
			rayTracingCompilerUsage = (Compiler)(value == "fxc"_sv);
		}
		if (sonObj.Get("Compute"_sv, value)) {
			StringUtil::ToLower(value);
			computeCompilerUsage = (Compiler)(value == "fxc"_sv);
		}
	}
	//	if (obj->Get("TempFolder"_sv, value)) {
	//		tempPath = value;
	//		Path(tempPath).TryCreateDirectory();
	//	}
	//	if (obj->Get("CompileResultFolder"_sv, pathFolder)) {
	//		if (!pathFolder.empty()) {
	//			char lst = pathFolder.end()[-1];
	//			if (lst != '/' && lst != '\\') {
	//				pathFolder += '/';
	//			}
	//		}
	//	} else
	//		pathFolder.clear();
}
struct CompileFunctionCommand {
	vstd::string name;
	ShaderType type;
};
void GenerateCompilerCommand(
	vstd::string const& fileName,
	vstd::string const& functionName,
	vstd::string const& resultFileName,
	ShaderType shaderType,
	Compiler compiler,
	vstd::string& cmdResult) {
	vstd::string const* compilerPath = nullptr;
	vstd::string const* compileShaderVersion = nullptr;
	vstd::string const* start = nullptr;
	switch (compiler) {
		case Compiler::FXC:
			compilerPath = &fxcpath;
			compileShaderVersion = &fxcversion;
			start = &fxcStart;
			break;
		case Compiler::DXC:
			compilerPath = &dxcpath;
			compileShaderVersion = &dxcversion;
			start = &dxcStart;
			break;
		default:
			std::cout << "Unsupported Compiler!"_sv << std::endl;
			return;
	}
	vstd::string shaderTypeName;
	switch (shaderType) {
		case ShaderType::ComputeShader:
			shaderTypeName = "cs_"_sv;
			break;
		case ShaderType::VertexShader:
			shaderTypeName = "vs_"_sv;
			break;
		case ShaderType::HullShader:
			shaderTypeName = "hs_"_sv;
			break;
		case ShaderType::DomainShader:
			shaderTypeName = "ds_"_sv;
			break;
		case ShaderType::GeometryShader:
			shaderTypeName = "gs_"_sv;
			break;
		case ShaderType::PixelShader:
			shaderTypeName = "ps_"_sv;
			break;
		case ShaderType::RayTracingShader:
			shaderTypeName = "lib_"_sv;
			break;
		default:
			shaderTypeName = " "_sv;
			break;
	}
	shaderTypeName += *compileShaderVersion;
	cmdResult.clear();
	cmdResult.reserve(50);
	cmdResult << *compilerPath << *start << shaderTypeCmd << shaderTypeName;
	if (!functionName.empty()) {
		cmdResult += funcName;
		cmdResult += functionName;
	}
	cmdResult += output;
	cmdResult += resultFileName;
	cmdResult += " "_sv;
	cmdResult += fileName;
}

template<typename T>
void PutIn(vstd::vector<char>& c, const T& data) {
	T* cc = &((T&)data);
	uint64 siz = c.size();
	c.resize(siz + sizeof(T));
	memcpy(c.data() + siz, cc, sizeof(T));
}
void PutIn(vstd::vector<char>& c, void* data, uint64 dataSize) {
	if (dataSize == 0) return;
	uint64 siz = c.size();
	c.resize(siz + dataSize);
	memcpy(c.data() + siz, data, dataSize);
}
template<>
void PutIn<vstd::string>(vstd::vector<char>& c, vstd::string const& data) {
	PutIn<uint>(c, (uint)data.length());
	uint64 siz = c.size();
	c.resize(siz + data.length());
	memcpy(c.data() + siz, data.data(), data.length());
}
template<typename T>
void DragData(std::ifstream& ifs, T& data) {
	ifs.read((char*)&data, sizeof(T));
}
template<>
void DragData<vstd::string>(std::ifstream& ifs, vstd::string& str) {
	uint32_t length = 0;
	DragData<uint32_t>(ifs, length);
	str.clear();
	str.resize(length);
	ifs.read(str.data(), length);
}

void PutInSerializedObjectAndData(
	vstd::vector<char> const& serializeObj,
	vstd::vector<char>& resultData,
	vstd::vector<ShaderVariable> const& vars) {
	PutIn<uint64_t>(resultData, serializeObj.size());
	PutIn(resultData, serializeObj.data(), serializeObj.size());

	PutIn<uint>(resultData, (uint)vars.size());
	for (auto i = vars.begin(); i != vars.end(); ++i) {
		PutIn<uint>(resultData, i->varID);
		PutIn<ShaderVariableType>(resultData, i->type);
		PutIn<uint>(resultData, i->tableSize);
		PutIn<uint>(resultData, i->registerPos);
		PutIn<uint>(resultData, i->space);
	}
}

bool HLSLCompiler::ErrorHappened() {
	return !errorMessage.empty();
}

void HLSLCompiler::PrintErrorMessages() {
	for (auto& msg : errorMessage) {
		std::cout << msg << '\n';
	}
	errorMessage.clear();
}

void HLSLCompiler::CompileComputeShader(
	vstd::string const& fileName,
	vstd::vector<ShaderVariable> const& vars,
	vstd::string const& passDesc,
	vstd::vector<char> const& customData,
	vstd::vector<char>& resultData) {
	resultData.clear();
	resultData.reserve(65536);
	PutInSerializedObjectAndData(
		customData,
		resultData,
		vars);

	auto func = [&](vstd::string const& str, ProcessorData* data) -> bool {
		uint64_t fileSize;
		WaitChildProcess(data);
		//CreateChildProcess(command);
		//TODO
		//system(command.c_str());
		fileSize = 0;
		BinaryReader ifs(str);
		if (!ifs) return false;
		fileSize = ifs.GetLength();
		PutIn<uint64_t>(resultData, fileSize);
		if (fileSize == 0) return false;
		uint64 originSize = resultData.size();
		resultData.resize(fileSize + originSize);
		ifs.Read(resultData.data() + originSize, fileSize);
		return true;
	};
	vstd::string kernelCommand;

	PutIn<uint>(resultData, 1);
	static std::atomic_uint temp_count = 0;
	vstd::string tempFile = tempPath;
	tempFile << vstd::to_string(temp_count++)
			 << ".obj"_sv;
	GenerateCompilerCommand(
		fileName,
		passDesc,
		tempFile,
		ShaderType::ComputeShader,
		computeCompilerUsage,
		kernelCommand);
	ProcessorData data;
	CreateChildProcess(kernelCommand, &data);
	if (!func(tempFile, &data)) {
		std::lock_guard<spin_mutex> lck(outputMtx);
		std::cout << kernelCommand << '\n';
		std::cout << vstd::string("ComputeShader "_sv) + fileName + " Failed!"_sv << std::endl;
		return;
	}

	PutIn<uint>(resultData, 1);
	PutIn(resultData, passDesc);
	PutIn<uint>(resultData, 0);
	remove(tempFile.c_str());
}
void HLSLCompiler::CompileDXRShader(
	vstd::string const& fileName,
	vstd::vector<ShaderVariable> const& vars,
	CompileDXRHitGroup const& passDescs,
	uint64 raypayloadMaxSize,
	uint64 recursiveCount,
	vstd::vector<char> const& customData,
	vstd::vector<char>& resultData) {
	resultData.clear();
	resultData.reserve(65536);
	if (raypayloadMaxSize == 0) {
		std::lock_guard<spin_mutex> lck(outputMtx);
		std::cout << "Raypayload Invalid! \n"_sv;
		std::cout << vstd::string("DXRShader "_sv) + fileName + " Failed!"_sv;

		return;
	}
	PutInSerializedObjectAndData(
		customData,
		resultData,
		vars);
	PutIn<uint64>(resultData, recursiveCount);
	PutIn<uint64>(resultData, raypayloadMaxSize);
	PutIn<vstd::string>(resultData, passDescs.name);
	PutIn<uint>(resultData, passDescs.shaderType);
	for (auto& func : passDescs.functions) {
		PutIn<vstd::string>(resultData, func);
	}

	auto func = [&](vstd::string const& str, ProcessorData* data) -> bool {
		uint64_t fileSize;
		WaitChildProcess(data);
		//CreateChildProcess(command);
		//TODO
		//system(command.c_str());
		fileSize = 0;
		std::ifstream ifs(str.data(), std::ios::binary);
		if (!ifs) return false;
		ifs.seekg(0, std::ios::end);
		fileSize = ifs.tellg();
		ifs.seekg(0, std::ios::beg);
		PutIn<uint64_t>(resultData, fileSize);
		if (fileSize == 0) return false;
		uint64 originSize = resultData.size();
		resultData.resize(fileSize + originSize);
		ifs.read(resultData.data() + originSize, fileSize);
		return true;
	};
	vstd::string kernelCommand;
	GenerateCompilerCommand(
		fileName, vstd::string(), tempPath, ShaderType::RayTracingShader, rayTracingCompilerUsage, kernelCommand);
	ProcessorData data;
	CreateChildProcess(kernelCommand, &data);
	if (!func(tempPath, &data)) {
		std::lock_guard<spin_mutex> lck(outputMtx);
		std::cout << vstd::string("DXRShader "_sv) + fileName + " Failed!"_sv;

	}
	remove(tempPath.c_str());
}

void HLSLCompiler::GetShaderVariables(
	luisa::compute::Function const& func,
	vstd::vector<ShaderVariable>& result) {
	using namespace luisa::compute;
	uint registPos = 0;
	uint rwRegistPos = 0;
	//CBuffer
	result.emplace_back(
		(uint)-1,
		ShaderVariableType::ConstantBuffer,
		1,
		0,
		0);
	auto ProcessBuffer = [&](Variable const& buffer) {
		if (((uint)func.variable_usage(buffer.uid()) & (uint)luisa::compute::Variable::Usage::WRITE) != 0) {
			auto& var = result.emplace_back(
				buffer.uid(),
				ShaderVariableType::RWStructuredBuffer,
				1,
				rwRegistPos,
				0);
			rwRegistPos++;
		} else {
			auto& var = result.emplace_back(
				buffer.uid(),
				ShaderVariableType::StructuredBuffer,
				1,
				registPos,
				0);
			registPos++;
		}
	};
	auto ProcessTexture = [&](Variable const& buffer) {
		if (((uint)func.variable_usage(buffer.uid()) & (uint)luisa::compute::Variable::Usage::WRITE) != 0) {
			auto& var = result.emplace_back(
				buffer.uid(),
				ShaderVariableType::UAVDescriptorHeap,
				1,
				rwRegistPos,
				0);
			rwRegistPos++;
		} else {
			auto& var = result.emplace_back(
				buffer.uid(),
				ShaderVariableType::SRVDescriptorHeap,
				1,
				registPos,
				0);
			registPos++;
		}
	};
	//Buffer
	for (auto& i : func.captured_buffers()) {
		auto& buffer = i.variable;
		ProcessBuffer(buffer);
	}
	for (auto& i : func.captured_textures()) {
		auto& buffer = i.variable;
		ProcessTexture(buffer);
	}
	for (auto& i : func.arguments()) {
		switch (i.tag()) {
			case Variable::Tag::BUFFER:
				ProcessBuffer(i);
				break;
			case Variable::Tag::TEXTURE:
				ProcessTexture(i);
				break;
		}
	}
}
bool HLSLCompiler::CheckNeedReCompile(std::array<uint8_t, 16> const& md5, vstd::string const& shaderFileName) {
	vstd::string md5Path = shaderFileName + ".md5"_sv;
	{
		BinaryReader binReader(md5Path);
		if (binReader && binReader.GetLength() >= md5.size()) {
			std::array<uint64, 2> file;
			binReader.Read((char*)file.data(), md5.size());
			uint64 const* compare = (uint64 const*)md5.data();
			if (compare[0] == file[0] && compare[1] == file[1]) return true;
		}
	}
	std::ofstream ofs(md5Path.c_str(), std::ios::binary);
	ofs.write((char const*)md5.data(), md5.size());
	return false;
}
}// namespace SCompile
