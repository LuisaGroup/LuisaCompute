#pragma once
#include <Common/HashMap.h>
#include <Common/vstring.h>
#include <Common/vector.h>
class VENGINE_DLL_RENDERER ShaderID {
public:
	static void Init();
	static uint PropertyToID(const vengine::string& str);
	static uint PropertyToID(uint uid);
};

