
#include <vstl/Common.h>
namespace toolhub::directx {
vstd::string_view GetAccelProcessCompute() {
    return R"(
struct MeshInst{
float4 p0;
float4 p1;
float4 p2;
uint InstanceID : 24;
uint InstanceMask : 8;
uint InstanceContributionToHitGroupIndex : 24;
uint Flags : 8;
uint2 accelStructPtr;
};
struct InputInst{
uint index;
uint flags;
uint2 mesh;
float4 p0;
float4 p1;
float4 p2;
};
RWStructuredBuffer<MeshInst> _InstBuffer : register(u0);
StructuredBuffer<InputInst> _SetBuffer : register(t0);
cbuffer _Global : register(b0){
uint dsp;
uint count;
}
static const uint flag_visibility_on = 1u << 2u;
[numthreads(64,1,1)]
void main(uint id : SV_DISPATCHTHREADID){
if(id >= dsp) return;
InputInst v = _SetBuffer[id];
if(v.index >= count) return;
MeshInst r;
r.p0 = v.p0;
r.p1 = v.p1;
r.p2 = v.p2;
r.InstanceID = v.index;
if((v.flags & flag_visibility_on) != 0)
r.InstanceMask = 255;
else
r.InstanceMask = 0;

r.InstanceContributionToHitGroupIndex = 0;
r.Flags = 0;
r.accelStructPtr = v.mesh;
_InstBuffer[v.index] = r;
}
)"sv;
}
vstd::string_view GetHeader() {
    return R"(
#pragma pack_matrix(row_major)
#define INFINITY_f 3.40282347e+37
SamplerState samplers[16] : register(s0, space1);

float determinant(float2x2 m) {
return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}
float determinant(float3x4 m) {
return m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) - m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) + m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z);
}
float determinant(float4x4 m) {
const float coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
const float coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
const float coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
const float coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
const float coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
const float coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
const float coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
const float coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
const float coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
const float coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
const float coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
const float coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
const float coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
const float coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
const float coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
const float coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
const float coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
const float coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
const float4 fac0 = float4(coef00, coef00, coef02, coef03);
const float4 fac1 = float4(coef04, coef04, coef06, coef07);
const float4 fac2 = float4(coef08, coef08, coef10, coef11);
const float4 fac3 = float4(coef12, coef12, coef14, coef15);
const float4 fac4 = float4(coef16, coef16, coef18, coef19);
const float4 fac5 = float4(coef20, coef20, coef22, coef23);
const float4 Vec0 = float4(m[1].x, m[0].x, m[0].x, m[0].x);
const float4 Vec1 = float4(m[1].y, m[0].y, m[0].y, m[0].y);
const float4 Vec2 = float4(m[1].z, m[0].z, m[0].z, m[0].z);
const float4 Vec3 = float4(m[1].w, m[0].w, m[0].w, m[0].w);
const float4 inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
const float4 inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
const float4 inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
const float4 inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
const float4 sign_a = float4(+1.0f, -1.0f, +1.0f, -1.0f);
const float4 sign_b = float4(-1.0f, +1.0f, -1.0f, +1.0f);
const float4 inv_0 = inv0 * sign_a;
const float4 inv_1 = inv1 * sign_b;
const float4 inv_2 = inv2 * sign_a;
const float4 inv_3 = inv3 * sign_b;
const float4 dot0 = m[0] * float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
return dot0.x + dot0.y + dot0.z + dot0.w;
}

float2x2 inverse(float2x2 m) {
const float one_over_determinant = 1.0f / (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
return float2x2(m[1][1] * one_over_determinant,
-m[0][1] * one_over_determinant,
-m[1][0] * one_over_determinant,
+m[0][0] * one_over_determinant);
}

float3x4 inverse(float3x4 m) {
const float one_over_determinant = 1.0f /
(m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -
m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) +
m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));
return float3x4(
(m[1].y * m[2].z - m[2].y * m[1].z) * one_over_determinant,
(m[2].y * m[0].z - m[0].y * m[2].z) * one_over_determinant,
(m[0].y * m[1].z - m[1].y * m[0].z) * one_over_determinant,
0.f,
(m[2].x * m[1].z - m[1].x * m[2].z) * one_over_determinant,
(m[0].x * m[2].z - m[2].x * m[0].z) * one_over_determinant,
(m[1].x * m[0].z - m[0].x * m[1].z) * one_over_determinant,
0.f,
(m[1].x * m[2].y - m[2].x * m[1].y) * one_over_determinant,
(m[2].x * m[0].y - m[0].x * m[2].y) * one_over_determinant,
(m[0].x * m[1].y - m[1].x * m[0].y) * one_over_determinant,
0.f);
}

float4x4 inverse(float4x4 m) {
const float coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
const float coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
const float coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
const float coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
const float coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
const float coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
const float coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
const float coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
const float coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
const float coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
const float coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
const float coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
const float coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
const float coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
const float coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
const float coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
const float coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
const float coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
const float4 fac0 = float4(coef00, coef00, coef02, coef03);
const float4 fac1 = float4(coef04, coef04, coef06, coef07);
const float4 fac2 = float4(coef08, coef08, coef10, coef11);
const float4 fac3 = float4(coef12, coef12, coef14, coef15);
const float4 fac4 = float4(coef16, coef16, coef18, coef19);
const float4 fac5 = float4(coef20, coef20, coef22, coef23);
const float4 Vec0 = float4(m[1].x, m[0].x, m[0].x, m[0].x);
const float4 Vec1 = float4(m[1].y, m[0].y, m[0].y, m[0].y);
const float4 Vec2 = float4(m[1].z, m[0].z, m[0].z, m[0].z);
const float4 Vec3 = float4(m[1].w, m[0].w, m[0].w, m[0].w);
const float4 inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
const float4 inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
const float4 inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
const float4 inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
const float4 sign_a = float4(+1.0f, -1.0f, +1.0f, -1.0f);
const float4 sign_b = float4(-1.0f, +1.0f, -1.0f, +1.0f);
const float4 inv_0 = inv0 * sign_a;
const float4 inv_1 = inv1 * sign_b;
const float4 inv_2 = inv2 * sign_a;
const float4 inv_3 = inv3 * sign_b;
const float4 dot0 = m[0] * float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
const float dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
const float one_over_determinant = 1.0f / dot1;
return float4x4(inv_0 * one_over_determinant,
inv_1 * one_over_determinant,
inv_2 * one_over_determinant,
inv_3 * one_over_determinant);
}

template<typename T>
T _acosh(T v) { return log(v + sqrt(v * v - 1)); }
template<typename T>
T _asinh(T v) { return log(v + sqrt(v * v + 1)); }
template<typename T>
T _atanh(T v) { return 0.5 * log((1 + v) / (1 - v)); }
template<typename T>
T _exp10(T v) { return pow(10, v); };
template<typename T>
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
template<typename T>
T selectVec(T a, T b, bool c) {
if (c)
return b;
else
return a;
}
template<typename T>
T selectVec2(T a, T b, bool2 c) {
return T(
selectVec(a.x, b.x, c.x),
selectVec(a.y, b.y, c.y));
}
template<typename T>
T selectVec3(T a, T b, bool3 c) {
return T(
selectVec(a.x, b.x, c.x),
selectVec(a.y, b.y, c.y),
selectVec(a.z, b.z, c.z));
}
template<typename T>
T selectVec4(T a, T b, bool4 c) {
return T(
selectVec(a.x, b.x, c.x),
selectVec(a.y, b.y, c.y),
selectVec(a.z, b.z, c.z),
selectVec(a.w, b.w, c.w));
}

template<typename T>
T copysign(T a, T b) { return asfloat((asuint(a) & 0x7fffffffu) | (asuint(b) & 0x80000000u)); }
template<typename T>
T fma(T a, T b, T c) { return a * b + c; }

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

float4x4 Mul(float4x4 a, float4x4 b) { return mul(a, b); }
float3x4 Mul(float3x4 a, float3x4 b) { return mul(a, float4x4(b, 0.f, 0.f, 0.f, 0.f)); }
float2x2 Mul(float2x2 a, float2x2 b) { return mul(a, b); }

// Note: do not swap a and b: already swapped in codegen
float4 Mul(float4x4 b, float4 a) { return mul(a, b); }
float3 Mul(float3x4 b, float3 a) { return mul(a, b).xyz; }
float2 Mul(float2x2 b, float2 a) { return mul(a, b); }

struct WrappedFloat2x2 {
row_major float2x2 m;
};

struct WrappedFloat3x3 {
row_major float3x4 m;
};

struct WrappedFloat4x4 {
row_major float4x4 m;
};

#define bfread(bf, idx) (bf[(idx)])
#define bfreadVec3(bf, idx) (bf[(idx)].xyz)
#define bfreadMat(bf, idx) (bf[idx].m)
#define bfwrite(bf, idx, value) (bf[(idx)] = (value))
#define bfwriteVec3(bf, idx, value) (bf[(idx)] = float4((value), 0))
#define bfwriteMat(bf, idx, value) (bf[idx].m = value)
struct BdlsStruct {
uint buffer;
uint tex2D;
uint tex3D;
uint tex2DX : 16;
uint tex2DY : 16;
uint tex3DX : 16;
uint tex3DY : 16;
uint tex3DZ : 16;
uint samp2D : 8;
uint samp3D : 8;
};
#define Smptx(tex, uv) (tex[uv])
#define Writetx(tex, uv, value) (tex[uv] = value)
#define BINDLESS_ARRAY StructuredBuffer<BdlsStruct>
Texture2D<float4> _BindlessTex[] : register(t0, space1);
Texture3D<float4> _BindlessTex3D[] : register(t0, space2);
template<typename T>
T fract(T x) { return x - floor(x); }
float4 SampleTex2D(BINDLESS_ARRAY arr, uint index, float2 uv, float level) {
BdlsStruct s = arr[index];
SamplerState samp = samplers[s.samp2D];
return _BindlessTex[s.tex2D].SampleLevel(samp, uv, level);
}
float4 SampleTex2D(BINDLESS_ARRAY arr, uint index, float2 uv) {
return SampleTex2D(arr, index, uv, 0);
}
float4 SampleTex2D(BINDLESS_ARRAY arr, uint index, float2 uv, float2 ddx, float2 ddy) {
BdlsStruct s = arr[index];
SamplerState samp = samplers[s.samp2D];
return _BindlessTex[s.tex2D].SampleGrad(samp, uv, ddx, ddy);
}
float4 SampleTex3D(BINDLESS_ARRAY arr, uint index, float3 uv, float level) {
BdlsStruct s = arr[index];
SamplerState samp = samplers[s.samp3D];
return _BindlessTex3D[s.tex3D].SampleLevel(samp, uv, level);
}
float4 SampleTex3D(BINDLESS_ARRAY arr, uint index, float3 uv) {
return SampleTex3D(arr, index, uv, 0);
}
float4 SampleTex3D(BINDLESS_ARRAY arr, uint index, float3 uv, float3 ddx, float3 ddy) {
BdlsStruct s = arr[index];
SamplerState samp = samplers[s.samp3D];
return _BindlessTex3D[s.tex3D].SampleGrad(samp, uv, ddx, ddy);
}
float4 ReadTex2D(BINDLESS_ARRAY arr, uint index, uint2 coord, uint level) {
BdlsStruct s = arr[index];
return _BindlessTex[s.tex2D].Load(uint3(coord, level));
}
float4 ReadTex3D(BINDLESS_ARRAY arr, uint index, uint3 coord, uint level) {
BdlsStruct s = arr[index];
return _BindlessTex3D[s.tex3D].Load(uint4(coord, level));
}
float4 ReadTex2D(BINDLESS_ARRAY arr, uint index, uint2 coord) {
return ReadTex2D(arr, index, coord, 0);
}
float4 ReadTex3D(BINDLESS_ARRAY arr, uint index, uint3 coord) {
return ReadTex3D(arr, index, coord, 0);
}
uint2 Tex2DSize(BINDLESS_ARRAY arr, uint index) {
BdlsStruct s = arr[index];
return uint2(s.tex2DX, s.tex2DY);
}
uint3 Tex3DSize(BINDLESS_ARRAY arr, uint index) {
BdlsStruct s = arr[index];
return uint3(s.tex3DX, s.tex3DY, s.tex3DZ);
}
uint2 Tex2DSize(BINDLESS_ARRAY arr, uint index, uint level) {
return max(Tex2DSize(arr, index) >> level, 1u);
}
uint3 Tex3DSize(BINDLESS_ARRAY arr, uint index, uint level) {
return max(Tex3DSize(arr, index) >> level, 1u);
}
#define READ_BUFFER(arr, arrIdx, idx, bf) (bf[arr[arrIdx].buffer][idx])
#define READ_BUFFERVec3(arr, arrIdx, idx, bf) (bf[arr[arrIdx].buffer][idx].xyz)
struct MeshInst {
float4 p0;
float4 p1;
float4 p2;
uint InstanceID : 24;
uint InstanceMask : 8;
uint InstanceContributionToHitGroupIndex : 24;
uint Flags : 8;
uint2 accelStructPtr;
};
template<typename T>
float4x4 InstMatrix(T instBuffer, uint index) {
MeshInst v = instBuffer[index];
return float4x4(
float4(v.p0.x, v.p1.x, v.p2.x, 0),
float4(v.p0.y, v.p1.y, v.p2.y, 0),
float4(v.p0.z, v.p1.z, v.p2.z, 0),
float4(v.p0.w, v.p1.w, v.p2.w, 1));
}
void SetInstTransform(inout MeshInst i, float4x4 mat) {
i.p0 = float4(mat[0].x, mat[1].x,  mat[2].x,  mat[3].x);
i.p1 = float4(mat[0].y, mat[1].y,  mat[2].y,  mat[3].y);
i.p2 = float4(mat[0].z, mat[1].z,  mat[2].z,  mat[3].z);
}
void SetAccelTransform(RWStructuredBuffer<MeshInst> buffer, uint index, float4x4 mat) {
SetInstTransform(buffer[index], mat);
}
void SetAccelVis(RWStructuredBuffer<MeshInst> buffer, uint index, bool vis) {
buffer[index].InstanceMask = vis ? 255 : 0;
}
template<typename Buffer, typename Idx, typename T>
T _atomic_exchange(Buffer a, Idx idx, T b) {
T r;
InterlockedExchange(a[idx], b, r);
return r;
}
template<typename Buffer, typename Idx, typename T>
T _atomic_compare_exchange(Buffer a, Idx idx, T b, T c) {
T r;
InterlockedCompareExchange(a[idx], b, c, r);
return r;
}
template<typename Buffer, typename Idx, typename T>
T _atomic_add(Buffer a, Idx idx, T b) {
T r;
InterlockedAdd(a[idx], b, r);
return r;
}
template<typename Buffer, typename Idx, typename T>
T _atomic_sub(Buffer a, Idx idx, T b) {
T r;
InterlockedAdd(a[idx], -b, r);
return r;
}
template<typename Buffer, typename Idx, typename T>
T _atomic_and(Buffer a, Idx idx, T b) {
T r;
InterlockedAnd(a[idx], b, r);
return r;
}
template<typename Buffer, typename Idx, typename T>
T _atomic_or(Buffer a, Idx idx, T b) {
T r;
InterlockedOr(a[idx], b, r);
return r;
}
template<typename Buffer, typename Idx, typename T>
T _atomic_xor(Buffer a, Idx idx, T b) {
T r;
InterlockedXor(a[idx], b, r);
return r;
}
template<typename Buffer, typename Idx, typename T>
T _atomic_min(Buffer a, Idx idx, T b) {
T r;
InterlockedMin(a[idx], b, r);
return r;
}
template<typename Buffer, typename Idx, typename T>
T _atomic_max(Buffer a, Idx idx, T b) {
T r;
InterlockedMax(a[idx], b, r);
return r;
}
struct FLOATV3 {
float v[3];
};
struct LCRayDesc {
FLOATV3 v0;
float v1;
FLOATV3 v2;
float v3;
};
struct RayPayload {
uint v0;
uint v1;
float2 v2;
};
)";
}
vstd::string_view GetRayTracingHeader() {
    return R"(
#define CLOSEST_HIT_RAY_FLAG (RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES)
#define ANY_HIT_RAY_FLAG (RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES)
RayPayload TraceClosest(RaytracingAccelerationStructure accel, LCRayDesc rayDesc) {
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
if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
payload.v0 = q.CommittedInstanceIndex();
payload.v1 = q.CommittedPrimitiveIndex();
payload.v2 = q.CommittedTriangleBarycentrics();
} else {
payload.v0 = 4294967295u;
}
return payload;
}
bool TraceAny(RaytracingAccelerationStructure accel, LCRayDesc rayDesc) {
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
return q.CommittedStatus() != COMMITTED_NOTHING;
}
)";
}
}// namespace toolhub::directx
