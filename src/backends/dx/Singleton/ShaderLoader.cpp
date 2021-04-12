//#endif
#include <Singleton/ShaderLoader.h>
#include <Utility/StringUtility.h>
#include <RenderComponent/PSOContainer.h>
#include <RenderComponent/Shader.h>
#include <RenderComponent/ComputeShader.h>
#include <RenderComponent/RayShader.h>

using namespace neb;
#define StructuredBuffer SCompile::StructuredBuffer_S
#define SHADER_MULTICORE_COMPILE
int32_t ShaderLoader::shaderIDCount = 0;
namespace ShaderGlobalNameSpace {
struct ShaderGlobal {
	HashMap<vengine::string, Shader*> shaderMap;
	HashMap<vengine::string, ComputeShader*> computeShaderMap;
	HashMap<vengine::string, RayShader*> rayShaderMap;
	CJsonObject computeJsons;
	CJsonObject shaderJsons;
	CJsonObject rayShaderJson;
	GFXDevice* device;
	std::mutex mtx;
	ShaderGlobal() : shaderMap(256),
					 computeShaderMap(256),
					 rayShaderMap(256) {}
};
StackObject<ShaderGlobal> globalData;
}// namespace ShaderGlobalNameSpace
Shader* ShaderLoader::LoadShader(const vengine::string& name, GFXDevice* device, const vengine::string& path) {
	using namespace ShaderGlobalNameSpace;
	Shader* sh = new Shader(name, device, path);
	return globalData->shaderMap.Insert(name, sh).Value();
}
RayShader* ShaderLoader::LoadRayShader(const vengine::string& name, GFXDevice* device, const vengine::string& path) {
	using namespace ShaderGlobalNameSpace;
	RayShader* sh = new RayShader(static_cast<ID3D12Device5*>(device), path);
	globalData->rayShaderMap.Insert(name, sh);
	return sh;
}
RayShader const* ShaderLoader::GetRayShader(const vengine::string& name) {
	using namespace ShaderGlobalNameSpace;
	std::lock_guard lck(globalData->mtx);
	auto ite = globalData->rayShaderMap.Find(name);
	if (ite && ite.Value())
		return ite.Value();
	vengine::string path;
	if (globalData->rayShaderJson.Get(name, path)) {
		return LoadRayShader(name, globalData->device, path);
	}
	return nullptr;
}
ComputeShader* ShaderLoader::LoadComputeShader(const vengine::string& name, GFXDevice* device, const vengine::string& path) {
	using namespace ShaderGlobalNameSpace;
	ComputeShader* sh = new ComputeShader(name, path, device);
	return globalData->computeShaderMap.Insert(name, sh).Value();
}
Shader const* ShaderLoader::GetShader(const vengine::string& name) {
	using namespace ShaderGlobalNameSpace;
	std::lock_guard lck(globalData->mtx);
	auto ite = globalData->shaderMap.Find(name);
	if (ite && ite.Value()) return ite.Value();
	vengine::string path;
	if (globalData->shaderJsons.Get(name, path)) {
		return LoadShader(name, globalData->device, path);
	}
	return nullptr;
}
ComputeShader const* ShaderLoader::GetComputeShader(const vengine::string& name) {
	using namespace ShaderGlobalNameSpace;
	std::lock_guard lck(globalData->mtx);
	auto ite = globalData->computeShaderMap.Find(name);
	if (ite && ite.Value()) return ite.Value();
	vengine::string path;
	if (globalData->computeJsons.Get(name, path)) {
		return LoadComputeShader(name, globalData->device, path);
	}
	return nullptr;
}
void ShaderLoader::ReleaseShader(Shader const* shader) {
	using namespace ShaderGlobalNameSpace;
	std::lock_guard lck(globalData->mtx);
	auto ite = globalData->shaderMap.Find(shader->GetName());
	if (!ite || shader != ite.Value()) return;
	PSOContainer::ReleasePSO(shader);
	delete ite.Value();
	globalData->shaderMap.Remove(ite);
}
void ShaderLoader::ReleaseComputeShader(ComputeShader const* shader) {
	using namespace ShaderGlobalNameSpace;
	std::lock_guard lck(globalData->mtx);
	auto ite = globalData->computeShaderMap.Find(shader->GetName());
	if (!ite || shader != ite.Value()) return;
	delete ite.Value();
	globalData->computeShaderMap.Remove(ite);
}
void ShaderLoader::Init(GFXDevice* device) {
	using namespace ShaderGlobalNameSpace;
	globalData.New();
	globalData->device = device;
	CJsonObject* jsonObj = ReadJson("Data/ShaderCompileList.json");
	if (jsonObj) {
		jsonObj->Get("compute", globalData->computeJsons);
		jsonObj->Get("shader", globalData->shaderJsons);
		jsonObj->Get("rayShader", globalData->rayShaderJson);
	}
}
void ShaderLoader::Reload(GFXDevice* device, JobBucket* bucket, HashMap<Shader const*, vengine::vector<JobHandle>>& shaderHandles) {
	using namespace ShaderGlobalNameSpace;
	globalData->shaderMap.IterateAll([&](vengine::string const& name, Shader*& shader) -> void {
		shaderHandles.Insert(shader).Value().push_back(bucket->GetTask({}, [&name, &shader]() -> void {
			vengine::string path;
			shader->~Shader();
			globalData->shaderJsons.Get(name, path);
			new (shader) Shader(name, globalData->device, path);
		}));
	});
	globalData->rayShaderMap.IterateAll(([&](vengine::string const& name, RayShader*& shader) -> void {
		vengine::string path;
		shader->~RayShader();
		globalData->rayShaderJson.Get(name, path);
		new (shader) RayShader(globalData->device, path);
		}));
	globalData->computeShaderMap.IterateAll([&](vengine::string const& name, ComputeShader*& shader) -> void {
		bucket->GetTask({}, [&name, &shader]() -> void {
			vengine::string path;
			shader->~ComputeShader();
			globalData->computeJsons.Get(name, path);
			new (shader) ComputeShader(name, path, globalData->device);
		});
	});
}
void ShaderLoader::Dispose() {
	using namespace ShaderGlobalNameSpace;
	globalData.Delete();
}
#undef StructuredBuffer
