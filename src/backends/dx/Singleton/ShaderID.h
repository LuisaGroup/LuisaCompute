#pragma once
#include <util/HashMap.h>
#include <util/vstring.h>
#include <util/vector.h>
class VENGINE_DLL_RENDERER ShaderID {
public:
	static void Init();
	static uint PropertyToID(uint uid);
};

