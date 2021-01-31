#pragma once
#include <mutex>
#include "../Common/HashMap.h"
#include "../Common/vstring.h"
#include "../Common/vector.h"
class ShaderID
{
	static const uint32_t INIT_CAPACITY = 100;
	static uint32_t currentCount;
	static HashMap<vengine::string, uint32_t, vengine::hash<vengine::string>, std::equal_to<vengine::string>, false> allShaderIDs;
	static vengine::vector<vengine::string> allShaderNames;
	static uint32_t mPerCameraBuffer;
	static uint32_t mPerMaterialBuffer;
	static uint32_t mPerObjectBuffer;
	static uint32_t mainTex;
	static uint32_t params;
	static std::mutex mtx;
public:
	static void Init();
	static uint32_t GetPerCameraBufferID() { return mPerCameraBuffer; }
	static uint32_t GetPerMaterialBufferID() { return mPerMaterialBuffer; }
	static uint32_t GetPerObjectBufferID() { return mPerObjectBuffer; }
	static uint32_t GetMainTex() { return mainTex; }
	static uint32_t PropertyToID(const vengine::string& str);
	static uint32_t GetParams() { return params; }
	static vengine::string const& IDToProperty(uint32_t id);
};

