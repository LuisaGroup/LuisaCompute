//#endif
#include <Singleton/ShaderID.h>
HashMap<vengine::string, uint32_t, vengine::hash<vengine::string>, std::equal_to<vengine::string>, false> ShaderID::allShaderIDs;
uint32_t ShaderID::currentCount = 0;
uint32_t ShaderID::mPerCameraBuffer = 0;
uint32_t ShaderID::mPerMaterialBuffer = 0;
uint32_t ShaderID::mPerObjectBuffer = 0;
uint32_t ShaderID::mainTex = 0;
uint32_t ShaderID::params = 0;
vengine::vector<vengine::string> ShaderID::allShaderNames;
std::mutex ShaderID::mtx;
uint32_t ShaderID::PropertyToID(const vengine::string& str) {
	std::lock_guard lck(mtx);
	auto ite = allShaderIDs.Find(str);
	if (!ite) {
		uint32_t value = currentCount;
		allShaderIDs.Insert(str, currentCount);
		allShaderNames.push_back(str);
		++currentCount;
		return value;
	} else {
		return ite.Value();
	}
}
vengine::string const& ShaderID::IDToProperty(uint32_t id) {
	return allShaderNames[id];
}
void ShaderID::Init() {
	mPerCameraBuffer = PropertyToID("Per_Camera_Buffer");
	mPerMaterialBuffer = PropertyToID("Per_Material_Buffer");
	mPerObjectBuffer = PropertyToID("Per_Object_Buffer");
	mainTex = PropertyToID("_MainTex");
	params = PropertyToID("Params");
}
