#pragma once
#include <core/vstl/HashMap.h>
#include <Common/vstring.h>
#include <core/vstl/vector.h>
class VENGINE_DLL_RENDERER ShaderID {
public:
	static void Init();
	static uint PropertyToID(uint uid);
};

