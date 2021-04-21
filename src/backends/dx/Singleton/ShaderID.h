#pragma once
#include <Common/HashMap.h>
#include <Common/vstring.h>
#include <Common/vector.h>
class VENGINE_DLL_RENDERER ShaderID {
public:
	static void Init();
	static uint GetPerCameraBufferID();
	static uint GetPerMaterialBufferID();
	static uint GetPerObjectBufferID();
	static uint GetMainTex();
	static uint GetParams();

	static uint PropertyToID(const vengine::string& str);
	static uint PropertyToID(uint uid);
};

