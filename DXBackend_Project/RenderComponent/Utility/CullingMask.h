#pragma once
#include <stdint.h>
enum class CullingMask : uint32_t
{
	NONE = 0,
	ALL = 4294967295,
	GEOMETRY = 1,
	CASCADE_SHADOWMAP = 2,
	SPOT_SHADOWMAP = 4,
	CUBE_SHADOWMAP = 8
};