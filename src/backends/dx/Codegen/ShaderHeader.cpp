#pragma vengine_package vengine_directx
#include <vstl/Common.h>
namespace toolhub::directx {
vstd::string_view GetHLSLHeader() {
    return R"(

#pragma pack_matrix(row_major)

#define INFINITY_f 3.40282347e+37
SamplerState samplers[16] : register(s0, space1);

float4x4 _inverse(float4x4 m) {
	float n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
	float n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
	float n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
	float n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];
	float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
	float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
	float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
	float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;
	float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
	float idet = 1.0f / det;
	float4x4 ret;
	ret[0][0] = t11 * idet;
	ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
	ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
	ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

	ret[1][0] = t12 * idet;
	ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
	ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
	ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

	ret[2][0] = t13 * idet;
	ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
	ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
	ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

	ret[3][0] = t14 * idet;
	ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
	ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
	ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

	return ret;
}

float3x4 _inverse(float3x4 m) {
	float3 c = float3(m[0][0], m[1][0], m[2][0]);
	float3 c2 = float3(m[0][1], m[1][1], m[2][1]);
	float3 c3 = float3(m[0][2], m[1][2], m[2][2]);
	float3 lhs = float3(c2.x, c3.x, c.x);
	float3 flt = float3(c2.y, c3.y, c.y);
	float3 rhs = float3(c2.z, c3.z, c.z);
	float3 flt2 = flt * rhs.yzx - flt.yzx * rhs;
	float3 c4 = lhs.yzx * rhs - lhs * rhs.yzx;
	float3 c5 = lhs * flt.yzx - lhs.yzx * flt;
	float rhs2 = 1.0 / dot(lhs.zxy * flt2, 1);
	return float3x4(float4(flt2, 0), float4(c4, 0), float4(c5, 0)) * rhs2;
}
template<typename T>
T _acosh(T v) { return log(v + sqrt(v * v - 1)); }
template<typename T>
T _asinh(T v) { return log(v + sqrt(v * v + 1)); }
template<typename T>
T _atanh(T v) { return 0.5 * log((1 + v) / (1 - v)); }
template<typename T>
T _exp10(T v) { return pow(10, v); };
template <typename T>
float _length_sqr(T x) { return dot(x, x); }
bool _isnan(float x) {
	return (asuint(x) & 0x7FFFFFFF) > 0x7F800000;
}
bool2 _isnan(float2 x) {
	return (asuint(x) & 0x7FFFFFFF) > 0x7F800000;
}
bool3 _isnan(float3 x) {
	return (asuint(x) & 0x7FFFFFFF) > 0x7F800000;
}
bool4 _isnan(float4 x) {
	return (asuint(x) & 0x7FFFFFFF) > 0x7F800000;
}
bool _isinf(float x) {
	return (asuint(x) & 0x7FFFFFFF) == 0x7F800000;
}
bool2 _isinf(float2 x) {
	return (asuint(x) & 0x7FFFFFFF) == 0x7F800000;
}
bool3 _isinf(float3 x) {
	return (asuint(x) & 0x7FFFFFFF) == 0x7F800000;
}
bool4 _isinf(float4 x) {
	return (asuint(x) & 0x7FFFFFFF) == 0x7F800000;
}
template <typename T>
T selectVec(T a, T b, bool c) {
	if (c) return b;
	else return a;
}
template <typename T>
T selectVec2(T a, T b, bool2 c){
	return T(
	selectVec(a.x, b.x, c.x),
	selectVec(a.y, b.y, c.y));
}
template <typename T>
T selectVec3(T a, T b, bool3 c){
	return T(
	selectVec(a.x, b.x, c.x),
	selectVec(a.y, b.y, c.y),
	selectVec(a.z, b.z, c.z));
}
template <typename T>
T selectVec4(T a, T b, bool4 c){
	return T(	
	selectVec(a.x, b.x, c.x),
	selectVec(a.y, b.y, c.y),
	selectVec(a.z, b.z, c.z),
	selectVec(a.w, b.w, c.w));
}

float copysign(float a, float b) { return asfloat((asuint(a) & 0x7fffffffu) | (asuint(b) & 0x80000000u)); }
float2 copysign(float2 a, float2 b) { return asfloat((asuint(a) & 0x7fffffffu) | (asuint(b) & 0x80000000u)); }
float3 copysign(float3 a, float3 b) { return asfloat((asuint(a) & 0x7fffffffu) | (asuint(b) & 0x80000000u)); }
float4 copysign(float4 a, float4 b) { return asfloat((asuint(a) & 0x7fffffffu) | (asuint(b) & 0x80000000u)); }

float fma(float a, float b, float c) { return a * b + c; }
float2 fma(float2 a, float2 b, float2 c) { return a * b + c; }
float3 fma(float3 a, float3 b, float3 c) { return a * b + c; }
float4 fma(float4 a, float4 b, float4 c) { return a * b + c; }

float2x2 make_float2x2(float m00, float m01,
                       float m10, float m11) {
	return float2x2(
		m00, m01,
        m10, m11);
}

float3x4 make_float3x3(float m00, float m01, float m02,
                       float m10, float m11, float m12,
                       float m20, float m21, float m22) {
	return float3x4(
		m00, m01, m02, 0.f,
        m10, m11, m12, 0.f,
        m20, m21, m22, 0.f);
}

float4x4 make_float4x4(float m00, float m01, float m02, float m03,
                       float m10, float m11, float m12, float m13,
                       float m20, float m21, float m22, float m23,
                       float m30, float m31, float m32, float m33) {
	return float4x4(
		m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23,
        m30, m31, m32, m33);
}

float3x4 make_float3x3(float3 c0, float3 c1, float3 c2) {
	return float3x4(float4(c0, 0.f), float4(c1, 0.f), float4(c2, 0.f));
}

float2x2 make_float2x2(float2 c0, float2 c1) { return float2x2(c0, c1); }
float4x4 make_float4x4(float4 c0, float4 c1, float4 c2, float4 c3) { return float4x4(c0, c1, c2, c3); }

float2x2 my_transpose(float2x2 m) { return transpose(m); }
float3x4 my_transpose(float3x4 m) {
  float4x3 mm = transpose(m);
  return make_float3x3(mm[0], mm[1], mm[2]);
}
float4x4 my_transpose(float4x4 m) { return transpose(m); }

float4x4 Mul(float4x4 a, float4x4 b){ return mul(a, b);}
float3x4 Mul(float3x4 a, float3x4 b){ return mul(a, float4x4(b, 0.f, 0.f, 0.f, 0.f));}
float2x2 Mul(float2x2 a, float2x2 b){ return mul(a, b);}

// Note: do not swap a and b: already swapped in codegen
float4 Mul(float4x4 b, float4 a){ return mul(b, a);}
float3 Mul(float3x4 b, float3 a){ return mul(b, float4(a, 0.f));}
float2 Mul(float2x2 b, float2 a){ return mul(b, a);}

struct WrappedFloat3x3 {
    row_major float3x4 m;
};

#define bfread(bf,idx) (bf[(idx)])
#define bfreadVec3(bf,idx) (bf[(idx)].xyz)
#define bfreadMat3(bf,idx) (bf[idx].m)
#define bfwrite(bf,idx,value) (bf[(idx)]=(value))
#define bfwriteVec3(bf,idx,value) (bf[(idx)]=float4((value), 0))
#define bfwriteMat3(bf,idx,value) (bf[idx].m=value)
struct BdlsStruct{
	uint buffer;
	uint tex2D;
	uint tex3D;
	uint tex2DX: 16;
	uint tex2DY: 16;
	uint tex3DX: 16;
	uint tex3DY: 16;
	uint tex3DZ: 16;
	uint samp2D: 8;
	uint samp3D: 8;
};
#define Smptx(tex, uv) (tex[uv])
#define Writetx(tex, uv, value) (tex[uv] = value)
#define BINDLESS_ARRAY StructuredBuffer<BdlsStruct>
Texture3D<float4> _BindlessTex3D[]:register(t0,space1);
Texture2D<float4> _BindlessTex[]:register(t0,space1);
template <typename T>
T fract(T x){ return x - floor(x);}
float4 SampleTex2D(BINDLESS_ARRAY arr, uint index, float2 uv, float level){
	BdlsStruct s = arr[index];
	SamplerState samp = samplers[s.samp2D];
	return _BindlessTex[s.tex2D].SampleLevel(samp, uv, level);
}
float4 SampleTex2D(BINDLESS_ARRAY arr, uint index, float2 uv){
	return SampleTex2D(arr, index, uv, 0);
}
float4 SampleTex2D(BINDLESS_ARRAY arr, uint index, float2 uv, float2 ddx, float2 ddy){
	BdlsStruct s = arr[index];
	SamplerState samp = samplers[s.samp2D];
	return _BindlessTex[s.tex2D].SampleGrad(samp, uv, ddx, ddy);
}
float4 SampleTex3D(BINDLESS_ARRAY arr, uint index, float3 uv, float level){
	BdlsStruct s = arr[index];
	SamplerState samp = samplers[s.samp3D];
	return _BindlessTex3D[s.tex3D].SampleLevel(samp, uv, level);
}
float4 SampleTex3D(BINDLESS_ARRAY arr, uint index, float3 uv){
	return SampleTex3D(arr, index, uv, 0);
}
float4 SampleTex3D(BINDLESS_ARRAY arr, uint index, float3 uv, float3 ddx, float3 ddy){
	BdlsStruct s = arr[index];
	SamplerState samp = samplers[s.samp3D];
	return _BindlessTex3D[s.tex3D].SampleGrad(samp, uv, ddx, ddy);
}
float4 ReadTex2D(BINDLESS_ARRAY arr,  uint index, uint2 coord, uint level){
	BdlsStruct s = arr[index]; 
	return _BindlessTex[s.tex2D].Load(uint3(coord, level));
}
float4 ReadTex3D(BINDLESS_ARRAY arr,  uint index, uint3 coord, uint level){
	BdlsStruct s = arr[index]; 
	return _BindlessTex3D[s.tex3D].Load(uint4(coord, level));
}
float4 ReadTex2D(BINDLESS_ARRAY arr,  uint index, uint2 coord){
	return ReadTex2D(arr, index, 0);
}
float4 ReadTex3D(BINDLESS_ARRAY arr, uint index,  uint3 coord){
	return ReadTex3D(arr, index, 0);
}
uint2 Tex2DSize(BINDLESS_ARRAY arr, uint index){
	BdlsStruct s = arr[index]; 
	return uint2(s.tex2DX, s.tex2DY);
}
uint3 Tex3DSize(BINDLESS_ARRAY arr, uint index){
	BdlsStruct s = arr[index]; 
	return uint3(s.tex3DX, s.tex3DY, s.tex3DZ);
}
uint2 Tex2DSize(BINDLESS_ARRAY arr, uint index, uint level){
	return max(Tex2DSize(arr, index) >> level, 1u);
}
uint3 Tex3DSize(BINDLESS_ARRAY arr, uint index, uint level){
	return max(Tex3DSize(arr, index) >> level, 1u);
}
#define READ_BUFFER(arr, arrIdx, idx, bf) (bf[arr[arrIdx].buffer][idx])
#define READ_BUFFERVec3(arr, arrIdx, idx, bf) (bf[arr[arrIdx].buffer][idx].xyz)
)"sv;
}
vstd::string_view GetRayTracingHeader() {
    return R"(
#define CLOSEST_HIT_RAY_FLAG (RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES)
#define ANY_HIT_RAY_FLAG (RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER)
RayPayload TraceClosest(RaytracingAccelerationStructure accel, LCRayDesc rayDesc){
	RayDesc ray;
	ray.Origin = float3(rayDesc.v0.v[0], rayDesc.v0.v[1], rayDesc.v0.v[2]);
	ray.Direction = float3(rayDesc.v2.v[0], rayDesc.v2.v[1], rayDesc.v2.v[2]);
	ray.TMin = rayDesc.v1;
	ray.TMax = rayDesc.v3;
	RayQuery<CLOSEST_HIT_RAY_FLAG> q;
	q.TraceRayInline(
	accel,
	CLOSEST_HIT_RAY_FLAG,
	~0,
	ray);
	RayPayload payload;
	q.Proceed();
	if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
	{
		payload.v0 = q.CommittedInstanceIndex();
		payload.v1 = q.CommittedPrimitiveIndex();
		payload.v2 = q.CommittedTriangleBarycentrics();
	}
	else {
		payload.v0 = 4294967295u;
	}
	return payload;
}
bool TraceAny(RaytracingAccelerationStructure accel, LCRayDesc rayDesc){
	RayDesc ray;
	ray.Origin = float3(rayDesc.v0.v[0], rayDesc.v0.v[1], rayDesc.v0.v[2]);
	ray.Direction = float3(rayDesc.v2.v[0], rayDesc.v2.v[1], rayDesc.v2.v[2]);
	ray.TMin = rayDesc.v1;
	ray.TMax = rayDesc.v3;
	RayQuery<ANY_HIT_RAY_FLAG> q;
	q.TraceRayInline(
	accel,
	ANY_HIT_RAY_FLAG,
	~0,
	ray);
	q.Proceed();
	return (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
}

float4x4 InstMatrix(StructuredBuffer<WrappedFloat3x3> instBuffer, uint index){
	float3x4 m = instBuffer[index].m;
	return float4x4(m, float4(0, 0, 0, 1));
}
)"sv;
}
}// namespace toolhub::directx
