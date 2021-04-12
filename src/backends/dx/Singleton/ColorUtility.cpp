#include <Singleton/ColorUtility.h>
using namespace Math;
const float logC_cut = 0.011361f;
const float logC_a = 5.555556f;
const float logC_b = 0.047996f;
const float logC_c = 0.244161f;
const float logC_d = 0.386036f;
const float logC_e = 5.301883f;
const float logC_f = 0.092819f;
float ColorUtility::StandardIlluminantY(float x) {
	return 2.87f * x - 3 * x * x - 0.27509507f;
}
// CIE xy chromaticity to CAT02 LMS.
// http://en.wikipedia.org/wiki/LMS_color_space#CAT02
Vector3 ColorUtility::CIExyToLMS(float x, float y) {
	float Y = 1;
	float X = Y * x / y;
	float Z = Y * (1 - x - y) / y;
	float L = 0.7328f * X + 0.4296f * Y - 0.1624f * Z;
	float M = -0.7036f * X + 1.6975f * Y + 0.0061f * Z;
	float S = 0.0030f * X + 0.0136f * Y + 0.9834f * Z;
	return {L, M, S};
}
Vector3 ColorUtility::ComputeColorBalance(float temperature, float tint) {
	// Range ~[-1.67;1.67] works best
	float t1 = temperature / 60;
	float t2 = tint / 60;
	// Get the CIE xy chromaticity of the reference white point.
	// Note: 0.31271 = x value on the D65 white point
	float x = 0.31271f - t1 * (t1 < 0 ? 0.1f : 0.05f);
	float y = StandardIlluminantY(x) + t2 * 0.05f;
	// Calculate the coefficients in the LMS space.
	Vector4 w1v = {0.949237f, 1.03542f, 1.08728f, 1};// D65 white point
	Vector4 w2v = CIExyToLMS(x, y);
	float4& w1 = (float4&)w1v;
	float4& w2 = (float4&)w2v;
	return {w1.x / w2.x, w1.y / w2.y, w1.z / w2.z};
}
// Alpha/w is offset
Vector3 ColorUtility::ColorToLift(Vector4 color) {
	// Shadows
	float3 S = {color.GetX(), color.GetY(), color.GetZ()};
	float lumLift = S.x * 0.2126f + S.y * 0.7152f + S.z * 0.0722f;
	S = {S.x - lumLift, S.y - lumLift, S.z - lumLift};
	float liftOffset = color.GetW();
	return {S.x + liftOffset, S.y + liftOffset, S.z + liftOffset};
}
// Alpha/w is offset
Vector3 ColorUtility::ColorToInverseGamma(Vector4 color) {
	// Midtones
	float3 M = {color.GetX(), color.GetY(), color.GetZ()};
	float lumGamma = M.x * 0.2126f + M.y * 0.7152f + M.z * 0.0722f;
	M = {M.x - lumGamma, M.y - lumGamma, M.z - lumGamma};
	float gammaOffset = (float)color.GetW() + 1;
	return {
		1 / Max(M.x + gammaOffset, 1e-03f),
		1 / Max(M.y + gammaOffset, 1e-03f),
		1 / Max(M.z + gammaOffset, 1e-03f)};
}
// Alpha/w is offset
Vector3 ColorUtility::ColorToGain(Vector4 color) {
	// Highlights
	float3 H = {color.GetX(), color.GetY(), color.GetZ()};
	float lumGain = H.x * 0.2126f + H.y * 0.7152f + H.z * 0.0722f;
	H = {H.x - lumGain, H.y - lumGain, H.z - lumGain};
	float gainOffset = (float)color.GetW() + 1;
	return {H.x + gainOffset, H.y + gainOffset, H.z + gainOffset};
}
// Alexa LogC converters (El 1000)
// See http://www.vocas.nl/webfm_send/964
float ColorUtility::LogCToLinear(float x) {
	return x > logC_e * logC_cut + logC_f
			   ? (pow(10, (x - logC_d) / logC_c) - logC_b) / logC_a
			   : (x - logC_f) / logC_e;
}
float ColorUtility::LinearToLogC(float x) {
	return x > logC_cut
			   ? logC_c * log10(logC_a * x + logC_b) + logC_d
			   : logC_e * x + logC_f;
}
uint ColorUtility::ToHex(Vector4 cc) {
	float4& c = (float4&)cc;
	return ((uint)(c.w * 255) << 24)
		   | ((uint)(c.x * 255) << 16)
		   | ((uint)(c.y * 255) << 8)
		   | ((uint)(c.z * 255));
}
Vector4 ColorUtility::ToRGBA(uint hex) {
	return {
		(float)(((hex >> 16) & 0xff) / 255.0),// r
		(float)(((hex >> 8) & 0xff) / 255.0), // g
		(float)(((hex)&0xff) / 255.0),		  // b
		(float)(((hex >> 24) & 0xff) / 255.0) // a
	};
}
