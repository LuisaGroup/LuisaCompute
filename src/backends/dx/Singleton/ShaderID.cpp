//#endif
#include <Singleton/ShaderID.h>
namespace ShaderIDGlobalNamespace {

struct ShaderIDGlobal {
	static constexpr uint INIT_CAPACITY = 128;
	uint currentCount{};
	HashMap<uint, uint> lcShaderIndices{INIT_CAPACITY};
	spin_mutex mtx;
	ShaderIDGlobal() noexcept = default;
};
static StackObject<ShaderIDGlobal, true> glb;
}// namespace ShaderIDGlobalNamespace

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
void ShaderID::Init() {
	using namespace ShaderIDGlobalNamespace;
	if (glb) return;
	glb.New();
}
