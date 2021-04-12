#pragma once
#include <Common/Common.h>
namespace lc_rhi {
struct TransformData {
	float3 position;
	float4 rotation;
	float3 localScale;
};
}// namespace lc_rhi