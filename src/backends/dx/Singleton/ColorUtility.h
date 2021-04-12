#pragma once
#include <Common/Common.h>
class  ColorUtility
{
public:
	static float StandardIlluminantY(float x);
	static Math::Vector3 CIExyToLMS(float x, float y);
	static Math::Vector3 ComputeColorBalance(float temperature, float tint);
	static Math::Vector3 ColorToLift(Math::Vector4 color);
	static Math::Vector3 ColorToInverseGamma(Math::Vector4 color);
	static Math::Vector3 ColorToGain(Math::Vector4 color);
	static float LogCToLinear(float x);
	static float LinearToLogC(float x);
	static uint ToHex(Math::Vector4 c);
	static Math::Vector4 ToRGBA(uint hex);
};