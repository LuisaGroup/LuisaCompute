//#endif
#include <Singleton/ShaderID.h>
namespace ShaderIDGlobalNamespace {

struct ShaderIDGlobal {
	const uint32_t INIT_CAPACITY = 100;
	uint32_t currentCount;
	HashMap<vengine::string, uint32_t> allShaderIDs;
	uint32_t mPerCameraBuffer;
	uint32_t mPerMaterialBuffer;
	uint32_t mPerObjectBuffer;
	uint32_t mainTex;
	uint32_t params;
	spin_mutex mtx;
};
static StackObject<ShaderIDGlobal, true> glb;
}// namespace ShaderIDGlobalNamespace
uint32_t ShaderID::PropertyToID(const vengine::string& str) {
	using namespace ShaderIDGlobalNamespace;
	std::lock_guard lck(glb->mtx);
	auto ite = glb->allShaderIDs.Find(str);
	if (!ite) {
		uint32_t value = glb->currentCount;
		glb->allShaderIDs.Insert(str, glb->currentCount);
		++glb->currentCount;
		return value;
	} else {
		return ite.Value();
	}
}
uint32_t ShaderID::GetPerCameraBufferID() { return ShaderIDGlobalNamespace::glb->mPerCameraBuffer; }
uint32_t ShaderID::GetPerMaterialBufferID() { return ShaderIDGlobalNamespace::glb->mPerMaterialBuffer; }
uint32_t ShaderID::GetPerObjectBufferID() { return ShaderIDGlobalNamespace::glb->mPerObjectBuffer; }
uint32_t ShaderID::GetMainTex() { return ShaderIDGlobalNamespace::glb->mainTex; }
uint32_t ShaderID::GetParams() { return ShaderIDGlobalNamespace::glb->params; }

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
