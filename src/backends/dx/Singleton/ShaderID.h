#pragma once
#include <Common/HashMap.h>
#include <Common/vstring.h>
#include <Common/vector.h>
class VENGINE_DLL_RENDERER ShaderID {
public:
	static void Init();
	static uint32_t GetPerCameraBufferID();
	static uint32_t GetPerMaterialBufferID();
	static uint32_t GetPerObjectBufferID();
	static uint32_t GetMainTex();
	static uint32_t GetParams();

	static uint32_t PropertyToID(const vengine::string& str);
};

