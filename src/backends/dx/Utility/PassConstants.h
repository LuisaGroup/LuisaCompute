#pragma once
#include <Common/Common.h>
#include <Common/MathHelper.h>
struct PassConstants
{
	float4x4 View = MathHelper::Identity4x4();
	float4x4 InvView = MathHelper::Identity4x4();
	float4x4 Proj = MathHelper::Identity4x4();
	float4x4 InvProj = MathHelper::Identity4x4();
	float4x4 ViewProj = MathHelper::Identity4x4();
	float4x4 InvViewProj = MathHelper::Identity4x4();
	float4x4 nonJitterVP = MathHelper::Identity4x4();
	float4x4 nonJitterInverseVP = MathHelper::Identity4x4();
	float4x4 lastVP = MathHelper::Identity4x4();
	float4x4 lastInverseVP = MathHelper::Identity4x4();
	float4x4 _FlipProj = MathHelper::Identity4x4();
	float4x4 _FlipInvProj = MathHelper::Identity4x4();
	float4x4 _FlipVP = MathHelper::Identity4x4();
	float4x4 _FlipInvVP = MathHelper::Identity4x4();
	float4x4 _FlipNonJitterVP = MathHelper::Identity4x4();
	float4x4 _FlipNonJitterInverseVP = MathHelper::Identity4x4();
	float4x4 _FlipLastVP = MathHelper::Identity4x4();
	float4x4 _FlipInverseLastVP = MathHelper::Identity4x4();
	float4 _ZBufferParams;
	float4 _RandomSeed;
	float3 worldSpaceCameraPos;
	float NearZ = 0.0f;
	float FarZ = 0.0f;
};