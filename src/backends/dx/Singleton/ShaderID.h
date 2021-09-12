#pragma once
#include <vstl/HashMap.h>
#include <vstl/vstring.h>
#include <vstl/vector.h>
class VENGINE_DLL_RENDERER ShaderID {
public:
	static void Init();
	static uint PropertyToID(uint uid);
};

