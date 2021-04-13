//#endif
#include <Singleton/ShaderLoader.h>
#include <Utility/StringUtility.h>
#include <RenderComponent/PSOContainer.h>
#include <RenderComponent/Shader.h>
#include <RenderComponent/ComputeShader.h>
#include <RenderComponent/RayShader.h>

using namespace neb;
thread_local ShaderLoaderGlobal* ShaderLoader::current = nullptr;
#define StructuredBuffer SCompile::StructuredBuffer_S
#define SHADER_MULTICORE_COMPILE
int32_t ShaderLoader::shaderIDCount = 0;
class ShaderLoaderGlobal {
public:
	HashMap<vengine::string, Shader*> shaderMap;
	HashMap<vengine::string, ComputeShader*> computeShaderMap;
	HashMap<vengine::string, RayShader*> rayShaderMap;
	CJsonObject computeJsons;
	CJsonObject shaderJsons;
	CJsonObject rayShaderJson;
	GFXDevice* device;
	std::mutex mtx;
	ShaderLoaderGlobal() : shaderMap(256),
						   computeShaderMap(256),
						   rayShaderMap(256) {}
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};
Shader* ShaderLoader::LoadShader(const vengine::string& name, GFXDevice* device, const vengine::string& path) {
	
	Shader* sh = new Shader(name, device, path);
	return current->shaderMap.Insert(name, sh).Value();
}
RayShader* ShaderLoader::LoadRayShader(const vengine::string& name, GFXDevice* device, const vengine::string& path) {
	
	RayShader* sh = new RayShader(static_cast<ID3D12Device5*>(device), path);
	current->rayShaderMap.Insert(name, sh);
	return sh;
}
RayShader const* ShaderLoader::GetRayShader(const vengine::string& name) {
	
	std::lock_guard lck(current->mtx);
	auto ite = current->rayShaderMap.Find(name);
	if (ite && ite.Value())
		return ite.Value();
	vengine::string path;
	if (current->rayShaderJson.Get(name, path)) {
		return LoadRayShader(name, current->device, path);
	}
	return nullptr;
}
ComputeShader* ShaderLoader::LoadComputeShader(const vengine::string& name, GFXDevice* device, const vengine::string& path) {
	
	ComputeShader* sh = new ComputeShader(name, path, device);
	return current->computeShaderMap.Insert(name, sh).Value();
}
Shader const* ShaderLoader::GetShader(const vengine::string& name) {
	
	std::lock_guard lck(current->mtx);
	auto ite = current->shaderMap.Find(name);
	if (ite && ite.Value()) return ite.Value();
	vengine::string path;
	if (current->shaderJsons.Get(name, path)) {
		return LoadShader(name, current->device, path);
	}
	return nullptr;
}
ComputeShader const* ShaderLoader::GetComputeShader(const vengine::string& name) {
	
	std::lock_guard lck(current->mtx);
	auto ite = current->computeShaderMap.Find(name);
	if (ite && ite.Value()) return ite.Value();
	vengine::string path;
	if (current->computeJsons.Get(name, path)) {
		return LoadComputeShader(name, current->device, path);
	}
	return nullptr;
}
void ShaderLoader::ReleaseShader(Shader const* shader) {
	
	std::lock_guard lck(current->mtx);
	auto ite = current->shaderMap.Find(shader->GetName());
	if (!ite || shader != ite.Value()) return;
	PSOContainer::ReleasePSO(shader);
	delete ite.Value();
	current->shaderMap.Remove(ite);
}
void ShaderLoader::ReleaseComputeShader(ComputeShader const* shader) {
	
	std::lock_guard lck(current->mtx);
	auto ite = current->computeShaderMap.Find(shader->GetName());
	if (!ite || shader != ite.Value()) return;
	delete ite.Value();
	current->computeShaderMap.Remove(ite);
}
ShaderLoaderGlobal* ShaderLoader::Init(GFXDevice* device) {
	
	ShaderLoaderGlobal* glb = new ShaderLoaderGlobal();
	glb->device = device;
	CJsonObject* jsonObj = ReadJson("Data/ShaderCompileList.json");
	if (jsonObj) {
		jsonObj->Get("compute", glb->computeJsons);
		jsonObj->Get("shader", glb->shaderJsons);
		jsonObj->Get("rayShader", glb->rayShaderJson);
	}
	return glb;
}
void ShaderLoader::Reload(GFXDevice* device, JobBucket* bucket, HashMap<Shader const*, vengine::vector<JobHandle>>& shaderHandles) {
	
	current->shaderMap.IterateAll([&](vengine::string const& name, Shader*& shader) -> void {
		shaderHandles.Insert(shader).Value().push_back(bucket->GetTask({}, [&name, &shader]() -> void {
			vengine::string path;
			shader->~Shader();
			current->shaderJsons.Get(name, path);
			new (shader) Shader(name, current->device, path);
		}));
	});
	current->rayShaderMap.IterateAll(([&](vengine::string const& name, RayShader*& shader) -> void {
		vengine::string path;
		shader->~RayShader();
		current->rayShaderJson.Get(name, path);
		new (shader) RayShader(current->device, path);
	}));
	current->computeShaderMap.IterateAll([&](vengine::string const& name, ComputeShader*& shader) -> void {
		bucket->GetTask({}, [&name, &shader]() -> void {
			vengine::string path;
			shader->~ComputeShader();
			current->computeJsons.Get(name, path);
			new (shader) ComputeShader(name, path, current->device);
		});
	});
}
void ShaderLoader::Dispose(ShaderLoaderGlobal* glb) {
	
	delete glb;
}
#undef StructuredBuffer
