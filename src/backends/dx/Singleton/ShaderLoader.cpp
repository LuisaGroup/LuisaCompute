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
	GFXDevice* device;
	std::mutex mtx;
	ShaderLoaderGlobal()
		: shaderMap(256),
		  computeShaderMap(256),
		  rayShaderMap(256) {}
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};
Shader* ShaderLoader::LoadShader(const vengine::string& name, GFXDevice* device, const vengine::string& path) {

	Shader* sh = new Shader(name, device, path);
	return current->shaderMap.Insert(name, sh).Value();
}
RayShader* ShaderLoader::LoadRayShader(const vengine::string& name, GFXDevice* device, const vengine::string& path) {

	RayShader* sh = new RayShader(device, path);
	current->rayShaderMap.Insert(name, sh);
	return sh;
}
RayShader const* ShaderLoader::GetRayShader(const vengine::string& name) {

	std::lock_guard lck(current->mtx);
	auto ite = current->rayShaderMap.Find(name);
	if (ite && ite.Value())
		return ite.Value();
	return LoadRayShader(name, current->device, name);
}
ComputeShader* ShaderLoader::LoadComputeShader(const vengine::string& name, GFXDevice* device, const vengine::string& path) {

	ComputeShader* sh = new ComputeShader(name, path, device);
	return current->computeShaderMap.Insert(name, sh).Value();
}
Shader const* ShaderLoader::GetShader(const vengine::string& name) {
	std::lock_guard lck(current->mtx);
	auto ite = current->shaderMap.Find(name);
	if (ite && ite.Value()) return ite.Value();
	return LoadShader(name, current->device, name);
}
ComputeShader const* ShaderLoader::GetComputeShader(const vengine::string& name) {
	std::lock_guard lck(current->mtx);
	auto ite = current->computeShaderMap.Find(name);
	if (ite && ite.Value()) return ite.Value();
	return LoadComputeShader(name, current->device, name);
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
	return glb;
}
void ShaderLoader::Reload(GFXDevice* device, JobBucket* bucket, HashMap<Shader const*, vengine::vector<JobHandle>>& shaderHandles) {

	current->shaderMap.IterateAll([&](vengine::string const& name, Shader*& shader) -> void {
		shaderHandles.Insert(shader).Value().push_back(bucket->GetTask({}, [&name, &shader]() -> void {
			shader->~Shader();
			new (shader) Shader(name, current->device, name);
		}));
	});
	current->rayShaderMap.IterateAll(([&](vengine::string const& name, RayShader*& shader) -> void {
		
		shader->~RayShader();
		new (shader) RayShader(current->device, name);
	}));
	current->computeShaderMap.IterateAll([&](vengine::string const& name, ComputeShader*& shader) -> void {
		bucket->GetTask({}, [&name, &shader]() -> void {
			shader->~ComputeShader();
			new (shader) ComputeShader(name, name, current->device);
		});
	});
}
void ShaderLoader::Dispose(ShaderLoaderGlobal* glb) {

	delete glb;
}
#undef StructuredBuffer
