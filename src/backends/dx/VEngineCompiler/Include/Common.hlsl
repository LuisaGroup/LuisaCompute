#include "ShaderConfig.cginc"
#ifndef _BaseCommon_
#define _BaseCommon_

#define MaterialFloat float
#define MaterialFloat2 float2
#define MaterialFloat3 float3
#define MaterialFloat4 float4
#define MaterialFloat3x3 float3x3
#define MaterialFloat4x4 float4x4
#define MaterialFloat4x3 float4x3

#define PI 3.1415926
#define Inv_PI 0.3183091
#define Two_PI 6.2831852
#define Inv_Two_PI 0.15915494

#define RANDOM(seed) (sin(cos(seed * 1354.135748 + 13.546184) * 1354.135716 + 32.6842317))
float Square(float x)
{
    return x * x;
}

float2 Square(float2 x)
{
    return x * x;
}

float3 Square(float3 x)
{
    return x * x;
}

float4 Square(float4 x)
{
    return x * x;
}

float pow2(float x)
{
    return x * x;
}

float2 pow2(float2 x)
{
    return x * x;
}

float3 pow2(float3 x)
{
    return x * x;
}

float4 pow2(float4 x)
{
    return x * x;
}

float pow3(float x)
{
    return x * x * x;
}

float2 pow3(float2 x)
{
    return x * x * x;
}

float3 pow3(float3 x)
{
    return x * x * x;
}

float4 pow3(float4 x)
{
    return x * x * x;
}

float pow4(float x)
{
    float xx = x * x;
    return xx * xx;
}

float2 pow4(float2 x)
{
    float2 xx = x * x;
    return xx * xx;
}

float3 pow4(float3 x)
{
    float3 xx = x * x;
    return xx * xx;
}

float4 pow4(float4 x)
{
    float4 xx = x * x;
    return xx * xx;
}

float pow5(float x)
{
    float xx = x * x;
    return xx * xx * x;
}

float2 pow5(float2 x)
{
    float2 xx = x * x;
    return xx * xx * x;
}

float3 pow5(float3 x)
{
    float3 xx = x * x;
    return xx * xx * x;
}

float4 pow5(float4 x)
{
    float4 xx = x * x;
    return xx * xx * x;
}

float pow6(float x)
{
    float xx = x * x;
    return xx * xx * xx;
}

float2 pow6(float2 x)
{
    float2 xx = x * x;
    return xx * xx * xx;
}

float3 pow6(float3 x)
{
    float3 xx = x * x;
    return xx * xx * xx;
}

float4 pow6(float4 x)
{
    float4 xx = x * x;
    return xx * xx * xx;
}

inline float acosFast(float inX)
{
    float x = abs(inX);
    float res = -0.156583f * x + (0.5 * PI);
    res *= sqrt(1 - x);
    return (inX >= 0) ? res : PI - res;
}

inline float asinFast(float x)
{
    return (0.5 * PI) - acosFast(x);
}

inline float ClampedPow(float X, float Y)
{
	return pow(max(abs(X), 0.000001), Y);
}

inline float CharlieL(float x, float r)
{
    r = saturate(r);
    r = 1 - (1 - r) * (1 - r);

    float a = lerp(25.3245, 21.5473, r);
    float b = lerp(3.32435, 3.82987, r);
    float c = lerp(0.16801, 0.19823, r);
    float d = lerp(-1.27393, -1.97760, r);
    float e = lerp(-4.85967, -4.32054, r);

    return a / (1 + b * pow(x, c)) + d * x + e;
}

void ConvertAnisotropyToRoughness(float Roughness, float Anisotropy, out float RoughnessT, out float RoughnessB) {
	Roughness *= Roughness;
    float AnisoAspect = sqrt(1 - 0.9 * Anisotropy);
    RoughnessT = Roughness / AnisoAspect; 
    RoughnessB = Roughness * AnisoAspect; 
}

float3 ComputeGrainNormal(float3 grainDir, float3 V) {
	float3 B = cross(-V, grainDir);
	return cross(B, grainDir);
}

float3 GetAnisotropicModifiedNormal(float3 grainDir, float3 N, float3 V, float Anisotropy) {
	float3 grainNormal = ComputeGrainNormal(grainDir, V);
	return normalize(lerp(N, grainNormal, Anisotropy));
}


#endif
