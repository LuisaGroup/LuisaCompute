#ifndef __SHADER_CONFIG_
#define __SHADER_CONFIG_
//#define ENABLE_VSM

#ifdef ENABLE_VSM
#define ENABLE_MSM
#endif

typedef float CBFloat;
typedef float2 CBFloat2;
typedef float3 CBFloat3;
typedef float4 CBFloat4;

typedef uint CBUInt;
typedef uint2 CBUInt2;
typedef uint3 CBUInt3;
typedef uint4 CBUInt4;

typedef int CBInt;
typedef int2 CBInt2;
typedef int3 CBInt3;
typedef int4 CBInt4;

typedef float4x4 CBFloat4x4;
typedef float3x4 CBFloat3x4;
typedef float4x3 CBFloat4x3;
typedef float3x3 CBFloat3x3;

#endif