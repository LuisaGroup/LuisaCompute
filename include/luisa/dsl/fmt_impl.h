#pragma once

#include <luisa/dsl/struct.h>


LUISA_DERIVE_FMT(luisa::compute::float2, float2,x, y);
LUISA_DERIVE_FMT(luisa::compute::float3, float3,x, y, z);
LUISA_DERIVE_FMT(luisa::compute::float4, float4,x, y, z, w);

LUISA_DERIVE_FMT(luisa::compute::int2, int2,x, y);
LUISA_DERIVE_FMT(luisa::compute::int3, int3,x, y, z);
LUISA_DERIVE_FMT(luisa::compute::int4, int4,x, y, z, w);

LUISA_DERIVE_FMT(luisa::compute::uint2, uint2, x, y);
LUISA_DERIVE_FMT(luisa::compute::uint3, uint3, x, y, z);
LUISA_DERIVE_FMT(luisa::compute::uint4, uint4, x, y, z, w);

LUISA_DERIVE_FMT(luisa::compute::bool2, bool2, x, y);
LUISA_DERIVE_FMT(luisa::compute::bool3, bool3, x, y, z);
LUISA_DERIVE_FMT(luisa::compute::bool4, bool4, x, y, z, w);


LUISA_DERIVE_FMT(luisa::compute::float2x2, float2x2,  cols[0], cols[1]);
LUISA_DERIVE_FMT(luisa::compute::float3x3, float3x3,  cols[0], cols[1], cols[2]);
LUISA_DERIVE_FMT(luisa::compute::float4x4, float4x4,  cols[0], cols[1], cols[2], cols[3]);