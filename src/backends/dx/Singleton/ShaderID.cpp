//#endif
#include <Singleton/ShaderID.h>
namespace ShaderIDGlobalNamespace {

struct ShaderIDGlobal {
	const uint INIT_CAPACITY = 100;
	uint currentCount;
	HashMap<vengine::string, uint> allShaderIDs;
	HashMap<uint, uint> lcShaderIndices;
	uint mPerCameraBuffer;
	uint mPerMaterialBuffer;
	uint mPerObjectBuffer;
	uint mainTex;
	uint params;
	spin_mutex mtx;
};
static StackObject<ShaderIDGlobal, true> glb;
}// namespace ShaderIDGlobalNamespace
uint ShaderID::PropertyToID(const vengine::string& str) {
	using namespace ShaderIDGlobalNamespace;
	std::lock_guard lck(glb->mtx);
	auto ite = glb->allShaderIDs.Find(str);
	if (!ite) {
		uint value = glb->currentCount;
		glb->allShaderIDs.Insert(str, glb->currentCount);
		++glb->currentCount;
		return value;
	}
	return ite.Value();
}
uint ShaderID::PropertyToID(uint uid) {
	using namespace ShaderIDGlobalNamespace;
	std::lock_guard lck(glb->mtx);
	auto ite = glb->lcShaderIndices.Find(uid);
	if (!ite) {
		uint value = glb->currentCount;
		glb->lcShaderIndices.Insert(uid, glb->currentCount);
		++glb->currentCount;
		return value;
	}
	return ite.Value();
}
uint ShaderID::GetPerCameraBufferID() { return ShaderIDGlobalNamespace::glb->mPerCameraBuffer; }
uint ShaderID::GetPerMaterialBufferID() { return ShaderIDGlobalNamespace::glb->mPerMaterialBuffer; }
uint ShaderID::GetPerObjectBufferID() { return ShaderIDGlobalNamespace::glb->mPerObjectBuffer; }
uint ShaderID::GetMainTex() { return ShaderIDGlobalNamespace::glb->mainTex; }
uint ShaderID::GetParams() { return ShaderIDGlobalNamespace::glb->params; }

void ShaderID::Init() {
	using namespace ShaderIDGlobalNamespace;
	if (glb) return;
	glb.New();
	glb->mPerCameraBuffer = PropertyToID("Per_Camera_Buffer");
	glb->mPerMaterialBuffer = PropertyToID("Per_Material_Buffer");
	glb->mPerObjectBuffer = PropertyToID("Per_Object_Buffer");
	glb->mainTex = PropertyToID("_MainTex");
	glb->params = PropertyToID("Params");
}
