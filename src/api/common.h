#pragma once
#include <core/platform.h>
#include <stdint.h>
#include <stddef.h>

// Uppercase names prefixed with underscores are reserved for the standard library.
#define LUISA_API_DECL_TYPE(TypeName) \
    typedef struct TypeName##_st {    \
        uint64_t __dummy;             \
    } TypeName##_st;                  \
    typedef TypeName##_st *TypeName

LUISA_API_DECL_TYPE(LCType);
LUISA_API_DECL_TYPE(LCExpression);
LUISA_API_DECL_TYPE(LCConstantData);
LUISA_API_DECL_TYPE(LCStmt);

LUISA_API_DECL_TYPE(LCContext);
LUISA_API_DECL_TYPE(LCDevice);
LUISA_API_DECL_TYPE(LCShader);
LUISA_API_DECL_TYPE(LCBuffer);
LUISA_API_DECL_TYPE(LCTexture);
LUISA_API_DECL_TYPE(LCStream);
LUISA_API_DECL_TYPE(LCEvent);
LUISA_API_DECL_TYPE(LCCommandList);
LUISA_API_DECL_TYPE(LCCommand);
LUISA_API_DECL_TYPE(LCBindlessArray);
LUISA_API_DECL_TYPE(LCMesh);
LUISA_API_DECL_TYPE(LCAccel);

#undef LUISA_API_DECL_TYPE

typedef enum LCAccelUsageHint {
    LC_FAST_TRACE, // build with best quality
    LC_FAST_UPDATE,// optimize for frequent update, usually with compaction
    LC_FAST_BUILD  // optimize for frequent rebuild, maybe without compaction
} LCAccelUsageHint;
typedef enum LCAccelBuildRequest {
    LC_PREFER_UPDATE,
    LC_FORCE_BUILD,
} LCAccelBuildRequest;

typedef enum LCPixelStorage {

    LC_BYTE1,
    LC_BYTE2,
    LC_BYTE4,

    LC_SHORT1,
    LC_SHORT2,
    LC_SHORT4,

    LC_INT1,
    LC_INT2,
    LC_INT4,

    LC_HALF1,
    LC_HALF2,
    LC_HALF4,

    LC_FLOAT1,
    LC_FLOAT2,
    LC_FLOAT4
} LCPixelStorage;

typedef enum LCPixelFormat {

    LC_R8SInt,
    LC_R8UInt,
    LC_R8UNorm,

    LC_RG8SInt,
    LC_RG8UInt,
    LC_RG8UNorm,

    LC_RGBA8SInt,
    LC_RGBA8UInt,
    LC_RGBA8UNorm,

    LC_R16SInt,
    LC_R16UInt,
    LC_R16UNorm,

    LC_RG16SInt,
    LC_RG16UInt,
    LC_RG16UNorm,

    LC_RGBA16SInt,
    LC_RGBA16UInt,
    LC_RGBA16UNorm,

    LC_R32SInt,
    LC_R32UInt,

    LC_RG32SInt,
    LC_RG32UInt,

    LC_RGBA32SInt,
    LC_RGBA32UInt,

    LC_R16F,
    LC_RG16F,
    LC_RGBA16F,

    LC_R32F,
    LC_RG32F,
    LC_RGBA32F
} LCPixelFormat;

typedef struct lc_uint3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} lc_uint3;

typedef enum LCAccelBuildModficationFlags {
    LC_ACCEL_MESH = 1u << 0u,
    LC_ACCEL_TRANSFORM = 1u << 1u,
    LC_ACCEL_VISIBILITY_ON = 1u << 2u,
    LC_ACCEL_VISIBILITY_OFF = 1u << 3u,
    LC_ACCEL_VISIBILITY = LC_ACCEL_VISIBILITY_ON | LC_ACCEL_VISIBILITY_OFF
} LCAccelBuildModficationFlags;

typedef struct LCAccelBuildModification {
    uint32_t index;
    LCAccelBuildModficationFlags flags;
    uint64_t mesh;
    float affine[12];
} LCAccelBuildModification;

typedef enum LCSamplerFilter {
    LC_POINT,
    LC_LINEAR_POINT,
    LC_LINEAR_LINEAR,
    LC_ANISOTROPIC
} LCSamplerFilter;

typedef enum LCSamplerAddress {
    LC_EDGE,
    LC_REPEAT,
    LC_MIRROR,
    LC_ZERO
} LCSamplerAddress;

typedef struct LCSampler {
    LCSamplerFilter filter;
    LCSamplerAddress address;
} LCSampler;

