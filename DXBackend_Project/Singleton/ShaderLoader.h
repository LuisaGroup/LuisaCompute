#pragma once
#include "../Common/GFXUtil.h"
#include "../Common/HashMap.h"
#include "../CJsonObject/CJsonObject.hpp"
#include "../JobSystem/JobInclude.h"
class Shader;
class ComputeShader;
class RayShader;

class ShaderLoader
{
private:
	static int32_t shaderIDCount;
	static Shader* LoadShader(const vengine::string& name, GFXDevice* device, const vengine::string& path);
	static RayShader* LoadRayShader(const vengine::string& name, GFXDevice* device, const vengine::string& path);
	static ComputeShader* LoadComputeShader(const vengine::string& name, GFXDevice* device, const vengine::string& path);
	//static void CompileShaders(GFXDevice* device, JobBucket* bucket, const vengine::string& path);
	ShaderLoader() = delete;

public:

	static Shader const* GetShader(const vengine::string& name);
	static RayShader const* GetRayShader(const vengine::string& name);
	static ComputeShader const* GetComputeShader(const vengine::string& name);
	static void ReleaseShader(Shader const* shader);
	static void ReleaseComputeShader(ComputeShader const* shader);
	static void Init(GFXDevice* device);
	static void Reload(GFXDevice* device, JobBucket* bucket, HashMap<Shader const*, vengine::vector<JobHandle>>& shaderHandles);
	static void Dispose();
	KILL_COPY_CONSTRUCT(ShaderLoader)
};