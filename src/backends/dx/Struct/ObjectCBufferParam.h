#pragma once
#include <Common/GFXUtil.h>
struct ObjectCBufferParam
{
	float4x4 _LastLocalToWorld;
	float4x4 _LocalToWorld;
	uint4 _ID;
};