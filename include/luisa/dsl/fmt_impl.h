#pragma once

#include <luisa/dsl/struct.h>


LUISA_DERIVE_FMT(luisa::compute::float2, x, y);
LUISA_DERIVE_FMT(luisa::compute::float3, x, y, z);
LUISA_DERIVE_FMT(luisa::compute::float4, x, y, z, w);

LUISA_DERIVE_FMT(luisa::compute::int2, x, y);
LUISA_DERIVE_FMT(luisa::compute::int3, x, y, z);
LUISA_DERIVE_FMT(luisa::compute::int4, x, y, z, w);

LUISA_DERIVE_FMT(luisa::compute::uint2, x, y);
LUISA_DERIVE_FMT(luisa::compute::uint3, x, y, z);
LUISA_DERIVE_FMT(luisa::compute::uint4, x, y, z, w);

LUISA_DERIVE_FMT(luisa::compute::bool2, x, y);
LUISA_DERIVE_FMT(luisa::compute::bool3, x, y, z);
LUISA_DERIVE_FMT(luisa::compute::bool4, x, y, z, w);


LUISA_DERIVE_FMT(luisa::compute::float2x2, cols[0], cols[1]);
LUISA_DERIVE_FMT(luisa::compute::float3x3, cols[0], cols[1], cols[2]);
LUISA_DERIVE_FMT(luisa::compute::float4x4, cols[0], cols[1], cols[2], cols[3]);