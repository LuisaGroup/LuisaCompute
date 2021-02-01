#pragma once
#include "../IPipelineResource.h"
#include "../../Common/GFXUtil.h"
class RenderTexture;
struct CameraTransformData : public IPipelineResource
{
	float2 jitter;
	float2 lastFrameJitter;
	Math::Matrix4 nonJitteredVPMatrix;
	Math::Matrix4 nonJitteredProjMatrix;
	Math::Matrix4 lastNonJitterVP;
	Math::Matrix4 lastProj;

	Math::Matrix4 lastFlipNonJitterVP;
	Math::Matrix4 lastFlipProj;
	Math::Matrix4 nonJitteredFlipVP;
	Math::Matrix4 nonJitteredFlipProj;


	Math::Vector3 lastCameraRight;
	Math::Vector3 lastCameraUp;
	Math::Vector3 lastCameraPosition;
	Math::Vector3 lastCameraForward;
};