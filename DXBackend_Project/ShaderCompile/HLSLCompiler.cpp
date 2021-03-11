#include "HLSLCompiler.h"
#include <CJsonObject/CJsonObject.hpp>
#include <Windows.h>
#include <Utility/StringUtility.h>
#include <fstream>

#include "ShaderUniforms.h"
namespace SCompile {
static bool g_needCommandOutput = true;
static const HANDLE g_hChildStd_IN_Rd = NULL;
static const HANDLE g_hChildStd_IN_Wr = NULL;
static const HANDLE g_hChildStd_OUT_Rd = NULL;
static const HANDLE g_hChildStd_OUT_Wr = NULL;
struct ProcessorData {
	_PROCESS_INFORMATION piProcInfo;
	bool bSuccess;
};
void CreateChildProcess(const vengine::string& cmd, ProcessorData* data) {
	if (g_needCommandOutput) {
		std::cout << cmd << std::endl;
		system(cmd.c_str());
		memset(data, 0, sizeof(ProcessorData));
		return;
	}

	PROCESS_INFORMATION piProcInfo;

	static HANDLE g_hInputFile = NULL;
	vengine::wstring ws;
	ws.resize(cmd.length());
	for (uint i = 0; i < cmd.length(); ++i) {
		ws[i] = cmd[i];
	}

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

	bSuccess = CreateProcess(NULL,
							 ws.data(),	  // command line
							 NULL,		  // process security attributes
							 NULL,		  // primary thread security attributes
							 TRUE,		  // handles are inherited
							 0,			  // creation flags
							 NULL,		  // use parent's environment
							 NULL,		  // use parent's current directory
							 &siStartInfo,// STARTUPINFO pointer
							 &piProcInfo);// receives PROCESS_INFORMATION
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
vengine::string tempPath;

vengine::string fxcStart;
vengine::string dxcStart;
vengine::string shaderTypeCmd;
vengine::string funcName;
vengine::string output;
vengine::string macro_compile;
vengine::string dxcversion;
vengine::string dxcpath;
vengine::string fxcversion;
vengine::string fxcpath;
vengine::string pathFolder;
static spin_mutex outputMtx;
static vengine::vector<vengine::string> errorMessage;

enum class Compiler : bool {
	DXC = false,
	FXC = true
};

Compiler computeCompilerUsage;
Compiler rasterizeCompilerUsage;
Compiler rayTracingCompilerUsage;
void InitRegisteData() {
	using namespace neb;
	std::unique_ptr<CJsonObject> obj(ReadJson("HLSLCompiler/register.json"_sv));
	if (!obj) {
		std::cout << "Register.txt not found in HLSLCompiler folder!"_sv << std::endl;
		system("pause");
		exit(0);
	}
	vengine::string value;
	CJsonObject sonObj;
	auto GenerateSettings = [&](vengine::string& settings) -> void {
		settings.clear();
		int sz = sonObj.GetArraySize();
		static vengine::string SplitString = " /"_sv;
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
			dxcpath = vengine::string("HLSLCompiler\\"_sv) + value;
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
			fxcpath = vengine::string("HLSLCompiler\\"_sv) + value;
		}
		if (sonObj.Get("Settings"_sv, sonObj) && sonObj.IsArray()) {
			GenerateSettings(fxcStart);
		}
	}
	if (obj->Get("CompilerUsage"_sv, sonObj)) {
		if (sonObj.Get("Rasterize"_sv, value)) {
			StringUtil::ToLower(value);
			rasterizeCompilerUsage = (Compiler)(value == "fxc");
		}
		if (sonObj.Get("RayTracing"_sv, value)) {
			StringUtil::ToLower(value);
			rayTracingCompilerUsage = (Compiler)(value == "fxc");
		}
		if (sonObj.Get("Compute"_sv, value)) {
			StringUtil::ToLower(value);
			computeCompilerUsage = (Compiler)(value == "fxc");
		}
	}
	if (obj->Get("TempFolder"_sv, value)) {
		tempPath = value;
	}
	if (obj->Get("CompileResultFolder"_sv, pathFolder)) {
		if (!pathFolder.empty()) {
			char lst = pathFolder.end()[-1];
			if (lst != '/' && lst != '\\') {
				pathFolder += '/';
			}
		}
	} else
		pathFolder.clear();
}
struct CompileFunctionCommand {
	vengine::string name;
	ShaderType type;
	ObjectPtr<vengine::vector<vengine::string>> macros;
};
void GenerateCompilerCommand(
	const vengine::string& fileName,
	const vengine::string& functionName,
	const vengine::string& resultFileName,
	const ObjectPtr<vengine::vector<vengine::string>>& macros,
	ShaderType shaderType,
	Compiler compiler,
	vengine::string& cmdResult) {
	const vengine::string* compilerPath = nullptr;
	const vengine::string* compileShaderVersion = nullptr;
	const vengine::string* start = nullptr;
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
	vengine::string shaderTypeName;
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
	cmdResult += *compilerPath + *start + shaderTypeCmd + shaderTypeName;
	if (macros && !macros->empty()) {
		for (auto ite = macros->begin(); ite != macros->end(); ++ite) {
			cmdResult += macro_compile;
			cmdResult += *ite + "=1 "_sv;
		}
	}
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
void PutIn(vengine::vector<char>& c, const T& data) {
	T* cc = &((T&)data);
	uint64 siz = c.size();
	c.resize(siz + sizeof(T));
	memcpy(c.data() + siz, cc, sizeof(T));
}
void PutIn(vengine::vector<char>& c, void* data, uint64 dataSize) {
	if (dataSize == 0) return;
	uint64 siz = c.size();
	c.resize(siz + dataSize);
	memcpy(c.data() + siz, data, dataSize);
}
template<>
void PutIn<vengine::string>(vengine::vector<char>& c, const vengine::string& data) {
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
void DragData<vengine::string>(std::ifstream& ifs, vengine::string& str) {
	uint32_t length = 0;
	DragData<uint32_t>(ifs, length);
	str.clear();
	str.resize(length);
	ifs.read(str.data(), length);
}
struct PassFunction {
	vengine::string name;
	ObjectPtr<vengine::vector<vengine::string>> macros;
	PassFunction(const vengine::string& name,
				 const ObjectPtr<vengine::vector<vengine::string>>& macros) : name(name),
																			  macros(macros) {}
	PassFunction() {}
	bool operator==(const PassFunction& p) const noexcept {
		bool cur = macros.operator bool();
		bool pCur = p.macros.operator bool();
		if (name == p.name) {
			if ((!pCur && !cur)) {
				return true;
			} else if (cur && pCur && macros->size() == p.macros->size()) {
				for (uint i = 0; i < macros->size(); ++i)
					if ((*macros)[i] != (*p.macros)[i]) return false;
				return true;
			}
		}
		return false;
	}
	bool operator!=(const PassFunction& p) const noexcept {
		return !operator==(p);
	}
};
struct PassFunctionHash {
	uint64 operator()(const PassFunction& pf) const noexcept {
		vengine::hash<vengine::string> hashStr;
		uint64 h = hashStr(pf.name);
		if (pf.macros) {
			for (uint i = 0; i < pf.macros->size(); ++i) {
				h <<= 4;
				h ^= hashStr((*pf.macros)[i]);
			}
		}
		return h;
	}
};

void PutInSerializedObjectAndData(
	vengine::vector<char> const& serializeObj,
	vengine::vector<char>& resultData,
	vengine::vector<ShaderVariable> vars) {
	PutIn<uint64_t>(resultData, serializeObj.size());
	PutIn(resultData, serializeObj.data(), serializeObj.size());

	PutIn<uint>(resultData, (uint)vars.size());
	for (auto i = vars.begin(); i != vars.end(); ++i) {
		PutIn<vengine::string>(resultData, i->name);
		PutIn<ShaderVariableType>(resultData, i->type);
		PutIn<uint>(resultData, i->tableSize);
		PutIn<uint>(resultData, i->registerPos);
		PutIn<uint>(resultData, i->space);
	}
}

void HLSLCompiler::CompileShader(
	const vengine::string& fileName,
	vengine::vector<ShaderVariable> const& vars,
	vengine::vector<PassDescriptor> const& passDescs,
	vengine::vector<char> const& customData,
	const vengine::string& tempFilePath,
	vengine::vector<char>& resultData) {
	resultData.clear();
	resultData.reserve(65536);
	PutInSerializedObjectAndData(
		customData,
		resultData,
		vars);
	auto func = [&](ProcessorData* pData, vengine::string const& str) -> bool {
		//TODO
		uint64_t fileSize;
		WaitChildProcess(pData);
		//CreateChildProcess(,);
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
	HashMap<PassFunction, std::pair<ShaderType, uint>, PassFunctionHash> passMap(passDescs.size() * 2);
	for (auto i = passDescs.begin(); i != passDescs.end(); ++i) {
		auto findFunc = [&](const vengine::string& namestr, const ObjectPtr<vengine::vector<vengine::string>>& macros, ShaderType type) -> void {
			PassFunction name(namestr, macros);

			if (name.name.empty()) return;
			if (!passMap.Contains(name)) {
				passMap.Insert(name, std::pair<ShaderType, uint>(type, (uint)passMap.Size()));
			}
		};
		findFunc(i->vertex, i->macros, ShaderType::VertexShader);
		findFunc(i->hull, i->macros, ShaderType::HullShader);
		findFunc(i->domain, i->macros, ShaderType::DomainShader);
		findFunc(i->fragment, i->macros, ShaderType::PixelShader);
	}
	vengine::vector<CompileFunctionCommand> functionNames(passMap.Size());
	PutIn<uint>(resultData, (uint)passMap.Size());
	passMap.IterateAll([&](PassFunction const& key, std::pair<ShaderType, uint>& value) -> void {
		CompileFunctionCommand cmd;
		cmd.macros = key.macros;
		cmd.name = key.name;
		cmd.type = value.first;
		functionNames[value.second] = cmd;
	});
	vengine::string commandCache;
	ProcessorData data;
	vengine::vector<vengine::string> strs(functionNames.size());
	for (uint i = 0; i < functionNames.size(); ++i) {
		strs[i] = tempFilePath + vengine::to_string(i);
		GenerateCompilerCommand(
			fileName, functionNames[i].name, strs[i],
			functionNames[i].macros, functionNames[i].type, rasterizeCompilerUsage, commandCache);
		CreateChildProcess(commandCache, &data);
		if (!func(&data, strs[i])) {
			std::lock_guard<spin_mutex> lck(outputMtx);
			errorMessage.emplace_back(std::move(vengine::string("Shader "_sv) + fileName + " Failed!"_sv));
			return;
		}
	}

	PutIn<uint>(resultData, (uint)passDescs.size());
	for (auto i = passDescs.begin(); i != passDescs.end(); ++i) {
		PutIn(resultData, i->name);
		PutIn(resultData, i->rasterizeState);
		PutIn(resultData, i->depthStencilState);
		PutIn(resultData, i->blendState);
		auto PutInFunc = [&](const vengine::string& value, const ObjectPtr<vengine::vector<vengine::string>>& macros) -> void {
			PassFunction psf(value, macros);
			if (value.empty() || !passMap.Contains(psf))
				PutIn<int>(resultData, -1);
			else
				PutIn<int>(resultData, (int)passMap[psf].second);
		};
		PutInFunc(i->vertex, i->macros);
		PutInFunc(i->hull, i->macros);
		PutInFunc(i->domain, i->macros);
		PutInFunc(i->fragment, i->macros);
	}
	for (auto i = strs.begin(); i != strs.end(); ++i)
		remove(i->c_str());
}
void HLSLCompiler::CompileComputeShader(
	const vengine::string& fileName,
	vengine::vector<ShaderVariable> const& vars,
	vengine::vector<PassDescriptor> const& passDescs,
	vengine::vector<char> const& customData,
	const vengine::string& tempFilePath,
	vengine::vector<char>& resultData) {
	resultData.clear();
	resultData.reserve(65536);
	PutInSerializedObjectAndData(
		customData,
		resultData,
		vars);
	using PassMap = HashMap<PassFunction, uint, PassFunctionHash>;
	PassMap passMap(passDescs.size() * 2);

	auto func = [&](vengine::string const& str, ProcessorData* data) -> bool {
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
	vengine::string kernelCommand;

	vengine::vector<std::pair<PassMap::Iterator, vengine::string>> strs;
	strs.reserve(passDescs.size());
	for (auto&& i : passDescs) {
		PassFunction func(
			i.name,
			i.macros);
		auto ite = passMap.Find(func);
		if (ite) continue;
		uint index = strs.size();
		ite = passMap.Insert(func, index);
		strs.push_back({ite, std::move(tempFilePath + vengine::to_string(index))});
	}
	PutIn<uint>(resultData, strs.size());
	for (auto&& i : strs) {
		PassFunction const& pass = i.first.Key();
		GenerateCompilerCommand(
			fileName,
			pass.name,
			i.second,
			pass.macros,
			ShaderType::ComputeShader,
			computeCompilerUsage,
			kernelCommand);
		ProcessorData data;
		CreateChildProcess(kernelCommand, &data);
		if (!func(i.second, &data)) {
			std::lock_guard<spin_mutex> lck(outputMtx);
			errorMessage.emplace_back(std::move(
				"ComputeShader " + fileName + " Failed!"_sv));

			return;
		}
	}
	PutIn<uint>(resultData, (uint)passDescs.size());
	for (auto&& i : passDescs) {
		PassFunction func(
			i.name,
			i.macros);
		auto ite = passMap.Find(func);
		PutIn(resultData, i.name);
		PutIn<uint>(resultData, ite.Value());
	}
	/*
	for (auto i = passDescs.begin(); i != passDescs.end(); ++i) {
		strs[i.GetIndex()] = tempFilePath + vengine::to_string(i.GetIndex());
		GenerateCompilerCommand(
			fileName, i->name, strs[i.GetIndex()], i->macros, ShaderType::ComputeShader, computeCompilerUsage, kernelCommand);
		ProcessorData data;
		CreateChildProcess(kernelCommand, &data);
		if (!func(strs[i.GetIndex()], i->name, &data)) {
			std::lock_guard<spin_mutex> lck(outputMtx);
			errorMessage.push_back(
				"ComputeShader " + fileName + " Failed!"_sv);

			return;
		}
	}*/
	for (auto i = strs.begin(); i != strs.end(); ++i)
		remove(i->second.c_str());
}
void HLSLCompiler::CompileDXRShader(
	const vengine::string& fileName,
	vengine::vector<ShaderVariable> const& vars,
	DXRHitGroup const& passDescs,
	uint64 raypayloadMaxSize,
	uint64 recursiveCount,
	vengine::vector<char> const& customData,
	const vengine::string& tempFilePath,
	vengine::vector<char>& resultData) {
	resultData.clear();
	resultData.reserve(65536);
	if (raypayloadMaxSize == 0) {
		std::lock_guard<spin_mutex> lck(outputMtx);
		std::cout << "Raypayload Invalid! \n"_sv;
		errorMessage.emplace_back(std::move(
			vengine::string("DXRShader "_sv) + fileName + " Failed!"_sv));
		return;
	}
	PutInSerializedObjectAndData(
		customData,
		resultData,
		vars);
	PutIn<uint64>(resultData, recursiveCount);
	PutIn<uint64>(resultData, raypayloadMaxSize);
	PutIn<vengine::string>(resultData, passDescs.name);
	PutIn<uint>(resultData, passDescs.shaderType);
	for (auto& func : passDescs.functionIndex) {
		PutIn<vengine::string>(resultData, func);
	}

	auto func = [&](vengine::string const& str, ProcessorData* data) -> bool {
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
	vengine::string kernelCommand;
	GenerateCompilerCommand(
		fileName, vengine::string(), tempFilePath, nullptr, ShaderType::RayTracingShader, rayTracingCompilerUsage, kernelCommand);
	ProcessorData data;
	CreateChildProcess(kernelCommand, &data);
	if (!func(tempFilePath, &data)) {
		std::lock_guard<spin_mutex> lck(outputMtx);
		errorMessage.emplace_back(std::move(
			vengine::string("DXRShader "_sv) + fileName + " Failed!"_sv));
	}
	remove(tempFilePath.c_str());
}

}// namespace SCompile