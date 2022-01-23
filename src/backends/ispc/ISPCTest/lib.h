#pragma once
// TODO: implement layout transition
#define lc_buffer_read(buffer, index) ((buffer)[index])
#define lc_buffer_write(buffer, index, value) (void)((buffer)[index] = (value))

// Vector
typedef float<2> float2;
typedef float<3> float3;
typedef float<4> float4;
typedef int<2> int2;
typedef int<3> int3;
typedef int<4> int4;
typedef uint<2> uint2;
typedef uint<3> uint3;
typedef uint<4> uint4;
typedef bool<2> bool2;
typedef bool<3> bool3;
typedef bool<4> bool4;

// Matrix
struct float2x2 { float m[2][2]; };
struct float3x3 { float m[3][3]; };
struct float4x4 { float m[4][4]; };
//---------------------------------------------------------------------------------------------------------------------
// INTRINSICS
//---------------------------------------------------------------------------------------------------------------------

// absinline int2 _int2(int s){int2 r={s,s};return r;}
inline int2 _int2(int x,int y){int2 r={x,y};return r;}
inline int2 _int2(int2 v){int2 r={v.x,v.y};return r;}
inline int2 _int2(int3 v){int2 r={v.x,v.y};return r;}
inline int2 _int2(int4 v){int2 r={v.x,v.y};return r;}
inline int2 _int2(uint2 v){int2 r={v.x,v.y};return r;}
inline int2 _int2(uint3 v){int2 r={v.x,v.y};return r;}
inline int2 _int2(uint4 v){int2 r={v.x,v.y};return r;}
inline int2 _int2(float2 v){int2 r={v.x,v.y};return r;}
inline int2 _int2(float3 v){int2 r={v.x,v.y};return r;}
inline int2 _int2(float4 v){int2 r={v.x,v.y};return r;}
inline int2 _int2(bool2 v){int2 r={v.x,v.y};return r;}
inline int2 _int2(bool3 v){int2 r={v.x,v.y};return r;}
inline int2 _int2(bool4 v){int2 r={v.x,v.y};return r;}
inline int3 _int3(int s){int3 r={s,s,s};return r;}
inline int3 _int3(int x,int y,int z){int3 r={x,y,z};return r;}
inline int3 _int3(int x,int2 yz){int3 r={x,yz.x,yz.y};return r;}
inline int3 _int3(int2 xy,int z){int3 r={xy.x,xy.y,z};return r;}
inline int3 _int3(int3 v){int3 r={v.x,v.y,v.z};return r;}
inline int3 _int3(int4 v){int3 r={v.x,v.y,v.z};return r;}
inline int3 _int3(uint3 v){int3 r={v.x,v.y,v.z};return r;}
inline int3 _int3(uint4 v){int3 r={v.x,v.y,v.z};return r;}
inline int3 _int3(float3 v){int3 r={v.x,v.y,v.z};return r;}
inline int3 _int3(float4 v){int3 r={v.x,v.y,v.z};return r;}
inline int3 _int3(bool3 v){int3 r={v.x,v.y,v.z};return r;}
inline int3 _int3(bool4 v){int3 r={v.x,v.y,v.z};return r;}
inline int4 _int4(int s){int4 r={s,s,s,s};return r;}
inline int4 _int4(int x,int y,int z,int w){int4 r={x,y,z,w};return r;}
inline int4 _int4(int x,int y,int2 zw){int4 r={x,y,zw.x,zw.y};return r;}
inline int4 _int4(int x,int2 yz,int w){int4 r={x,yz.x,yz.y,w};return r;}
inline int4 _int4(int2 xy,int z,int w){int4 r={xy.x,xy.y,z,w};return r;}
inline int4 _int4(int2 xy,int2 zw){int4 r={xy.x,xy.y,zw.x,zw.y};return r;}
inline int4 _int4(int x,int3 yzw){int4 r={x,yzw.x,yzw.y,yzw.z};return r;}
inline int4 _int4(int3 xyz,int w){int4 r={xyz.x,xyz.y,xyz.z,w};return r;}
inline int4 _int4(int4 v){int4 r={v.x,v.y,v.z,v.w};return r;}
inline int4 _int4(uint4 v){int4 r={v.x,v.y,v.z,v.w};return r;}
inline int4 _int4(float4 v){int4 r={v.x,v.y,v.z,v.w};return r;}
inline int4 _int4(bool4 v){int4 r={v.x,v.y,v.z,v.w};return r;}
inline uint2 _uint2(uint s){uint2 r={s,s};return r;}
inline uint2 _uint2(uint x,uint y){uint2 r={x,y};return r;}
inline uint2 _uint2(int2 v){uint2 r={v.x,v.y};return r;}
inline uint2 _uint2(int3 v){uint2 r={v.x,v.y};return r;}
inline uint2 _uint2(int4 v){uint2 r={v.x,v.y};return r;}
inline uint2 _uint2(uint2 v){uint2 r={v.x,v.y};return r;}
inline uint2 _uint2(uint3 v){uint2 r={v.x,v.y};return r;}
inline uint2 _uint2(uint4 v){uint2 r={v.x,v.y};return r;}
inline uint2 _uint2(float2 v){uint2 r={v.x,v.y};return r;}
inline uint2 _uint2(float3 v){uint2 r={v.x,v.y};return r;}
inline uint2 _uint2(float4 v){uint2 r={v.x,v.y};return r;}
inline uint2 _uint2(bool2 v){uint2 r={v.x,v.y};return r;}
inline uint2 _uint2(bool3 v){uint2 r={v.x,v.y};return r;}
inline uint2 _uint2(bool4 v){uint2 r={v.x,v.y};return r;}
inline uint3 _uint3(uint s){uint3 r={s,s,s};return r;}
inline uint3 _uint3(uint x,uint y,uint z){uint3 r={x,y,z};return r;}
inline uint3 _uint3(uint x,uint2 yz){uint3 r={x,yz.x,yz.y};return r;}
inline uint3 _uint3(uint2 xy,uint z){uint3 r={xy.x,xy.y,z};return r;}
inline uint3 _uint3(int3 v){uint3 r={v.x,v.y,v.z};return r;}
inline uint3 _uint3(int4 v){uint3 r={v.x,v.y,v.z};return r;}
inline uint3 _uint3(uint3 v){uint3 r={v.x,v.y,v.z};return r;}
inline uint3 _uint3(uint4 v){uint3 r={v.x,v.y,v.z};return r;}
inline uint3 _uint3(float3 v){uint3 r={v.x,v.y,v.z};return r;}
inline uint3 _uint3(float4 v){uint3 r={v.x,v.y,v.z};return r;}
inline uint3 _uint3(bool3 v){uint3 r={v.x,v.y,v.z};return r;}
inline uint3 _uint3(bool4 v){uint3 r={v.x,v.y,v.z};return r;}
inline uint4 _uint4(uint s){uint4 r={s,s,s,s};return r;}
inline uint4 _uint4(uint x,uint y,uint z,uint w){uint4 r={x,y,z,w};return r;}
inline uint4 _uint4(uint x,uint y,uint2 zw){uint4 r={x,y,zw.x,zw.y};return r;}
inline uint4 _uint4(uint x,uint2 yz,uint w){uint4 r={x,yz.x,yz.y,w};return r;}
inline uint4 _uint4(uint2 xy,uint z,uint w){uint4 r={xy.x,xy.y,z,w};return r;}
inline uint4 _uint4(uint2 xy,uint2 zw){uint4 r={xy.x,xy.y,zw.x,zw.y};return r;}
inline uint4 _uint4(uint x,uint3 yzw){uint4 r={x,yzw.x,yzw.y,yzw.z};return r;}
inline uint4 _uint4(uint3 xyz,uint w){uint4 r={xyz.x,xyz.y,xyz.z,w};return r;}
inline uint4 _uint4(int4 v){uint4 r={v.x,v.y,v.z,v.w};return r;}
inline uint4 _uint4(uint4 v){uint4 r={v.x,v.y,v.z,v.w};return r;}
inline uint4 _uint4(float4 v){uint4 r={v.x,v.y,v.z,v.w};return r;}
inline uint4 _uint4(bool4 v){uint4 r={v.x,v.y,v.z,v.w};return r;}
inline float2 _float2(float s){float2 r={s,s};return r;}
inline float2 _float2(float x,float y){float2 r={x,y};return r;}
inline float2 _float2(int2 v){float2 r={v.x,v.y};return r;}
inline float2 _float2(int3 v){float2 r={v.x,v.y};return r;}
inline float2 _float2(int4 v){float2 r={v.x,v.y};return r;}
inline float2 _float2(uint2 v){float2 r={v.x,v.y};return r;}
inline float2 _float2(uint3 v){float2 r={v.x,v.y};return r;}
inline float2 _float2(uint4 v){float2 r={v.x,v.y};return r;}
inline float2 _float2(float2 v){float2 r={v.x,v.y};return r;}
inline float2 _float2(float3 v){float2 r={v.x,v.y};return r;}
inline float2 _float2(float4 v){float2 r={v.x,v.y};return r;}
inline float2 _float2(bool2 v){float2 r={v.x,v.y};return r;}
inline float2 _float2(bool3 v){float2 r={v.x,v.y};return r;}
inline float2 _float2(bool4 v){float2 r={v.x,v.y};return r;}
inline float3 _float3(float s){float3 r={s,s,s};return r;}
inline float3 _float3(float x,float y,float z){float3 r={x,y,z};return r;}
inline float3 _float3(float x,float2 yz){float3 r={x,yz.x,yz.y};return r;}
inline float3 _float3(float2 xy,float z){float3 r={xy.x,xy.y,z};return r;}
inline float3 _float3(int3 v){float3 r={v.x,v.y,v.z};return r;}
inline float3 _float3(int4 v){float3 r={v.x,v.y,v.z};return r;}
inline float3 _float3(uint3 v){float3 r={v.x,v.y,v.z};return r;}
inline float3 _float3(uint4 v){float3 r={v.x,v.y,v.z};return r;}
inline float3 _float3(float3 v){float3 r={v.x,v.y,v.z};return r;}
inline float3 _float3(float4 v){float3 r={v.x,v.y,v.z};return r;}
inline float3 _float3(bool3 v){float3 r={v.x,v.y,v.z};return r;}
inline float3 _float3(bool4 v){float3 r={v.x,v.y,v.z};return r;}
inline float4 _float4(float s){float4 r={s,s,s,s};return r;}
inline float4 _float4(float x,float y,float z,float w){float4 r={x,y,z,w};return r;}
inline float4 _float4(float x,float y,float2 zw){float4 r={x,y,zw.x,zw.y};return r;}
inline float4 _float4(float x,float2 yz,float w){float4 r={x,yz.x,yz.y,w};return r;}
inline float4 _float4(float2 xy,float z,float w){float4 r={xy.x,xy.y,z,w};return r;}
inline float4 _float4(float2 xy,float2 zw){float4 r={xy.x,xy.y,zw.x,zw.y};return r;}
inline float4 _float4(float x,float3 yzw){float4 r={x,yzw.x,yzw.y,yzw.z};return r;}
inline float4 _float4(float3 xyz,float w){float4 r={xyz.x,xyz.y,xyz.z,w};return r;}
inline float4 _float4(int4 v){float4 r={v.x,v.y,v.z,v.w};return r;}
inline float4 _float4(uint4 v){float4 r={v.x,v.y,v.z,v.w};return r;}
inline float4 _float4(float4 v){float4 r={v.x,v.y,v.z,v.w};return r;}
inline float4 _float4(bool4 v){float4 r={v.x,v.y,v.z,v.w};return r;}
inline bool2 _bool2(bool s){bool2 r={s,s};return r;}
inline bool2 _bool2(bool x,bool y){bool2 r={x,y};return r;}
inline bool2 _bool2(int2 v){bool2 r={v.x,v.y};return r;}
inline bool2 _bool2(int3 v){bool2 r={v.x,v.y};return r;}
inline bool2 _bool2(int4 v){bool2 r={v.x,v.y};return r;}
inline bool2 _bool2(uint2 v){bool2 r={v.x,v.y};return r;}
inline bool2 _bool2(uint3 v){bool2 r={v.x,v.y};return r;}
inline bool2 _bool2(uint4 v){bool2 r={v.x,v.y};return r;}
inline bool2 _bool2(float2 v){bool2 r={v.x,v.y};return r;}
inline bool2 _bool2(float3 v){bool2 r={v.x,v.y};return r;}
inline bool2 _bool2(float4 v){bool2 r={v.x,v.y};return r;}
inline bool2 _bool2(bool2 v){bool2 r={v.x,v.y};return r;}
inline bool2 _bool2(bool3 v){bool2 r={v.x,v.y};return r;}
inline bool2 _bool2(bool4 v){bool2 r={v.x,v.y};return r;}
inline bool3 _bool3(bool s){bool3 r={s,s,s};return r;}
inline bool3 _bool3(bool x,bool y,bool z){bool3 r={x,y,z};return r;}
inline bool3 _bool3(bool x,bool2 yz){bool3 r={x,yz.x,yz.y};return r;}
inline bool3 _bool3(bool2 xy,bool z){bool3 r={xy.x,xy.y,z};return r;}
inline bool3 _bool3(int3 v){bool3 r={v.x,v.y,v.z};return r;}
inline bool3 _bool3(int4 v){bool3 r={v.x,v.y,v.z};return r;}
inline bool3 _bool3(uint3 v){bool3 r={v.x,v.y,v.z};return r;}
inline bool3 _bool3(uint4 v){bool3 r={v.x,v.y,v.z};return r;}
inline bool3 _bool3(float3 v){bool3 r={v.x,v.y,v.z};return r;}
inline bool3 _bool3(float4 v){bool3 r={v.x,v.y,v.z};return r;}
inline bool3 _bool3(bool3 v){bool3 r={v.x,v.y,v.z};return r;}
inline bool3 _bool3(bool4 v){bool3 r={v.x,v.y,v.z};return r;}
inline bool4 _bool4(bool s){bool4 r={s,s,s,s};return r;}
inline bool4 _bool4(bool x,bool y,bool z,bool w){bool4 r={x,y,z,w};return r;}
inline bool4 _bool4(bool x,bool y,bool2 zw){bool4 r={x,y,zw.x,zw.y};return r;}
inline bool4 _bool4(bool x,bool2 yz,bool w){bool4 r={x,yz.x,yz.y,w};return r;}
inline bool4 _bool4(bool2 xy,bool z,bool w){bool4 r={xy.x,xy.y,z,w};return r;}
inline bool4 _bool4(bool2 xy,bool2 zw){bool4 r={xy.x,xy.y,zw.x,zw.y};return r;}
inline bool4 _bool4(bool x,bool3 yzw){bool4 r={x,yzw.x,yzw.y,yzw.z};return r;}
inline bool4 _bool4(bool3 xyz,bool w){bool4 r={xyz.x,xyz.y,xyz.z,w};return r;}
inline bool4 _bool4(int4 v){bool4 r={v.x,v.y,v.z,v.w};return r;}
inline bool4 _bool4(uint4 v){bool4 r={v.x,v.y,v.z,v.w};return r;}
inline bool4 _bool4(float4 v){bool4 r={v.x,v.y,v.z,v.w};return r;}
inline bool4 _bool4(bool4 v){bool4 r={v.x,v.y,v.z,v.w};return r;}
#define select_scale(a,b,c) ((c)?(b):(a))
inline float2 select(float2 f, float2 t, bool2 v){float2 r={(v.x?t.x:f.x),(v.y?t.y:f.y)};return r;}
inline float3 select(float3 f, float3 t, bool3 v){float3 r={(v.x?t.x:f.x),(v.y?t.y:f.y),(v.z?t.z:f.z)};return r;}
inline float4 select(float4 f, float4 t, bool4 v){float4 r={(v.x?t.x:f.x),(v.y?t.y:f.y),(v.z?t.z:f.z),(v.w?t.w:f.w)};return r;}
inline uint2 select(uint2 f, uint2 t, bool2 v){uint2 r={(v.x?t.x:f.x),(v.y?t.y:f.y)};return r;}
inline uint3 select(uint3 f, uint3 t, bool3 v){uint3 r={(v.x?t.x:f.x),(v.y?t.y:f.y),(v.z?t.z:f.z)};return r;}
inline uint4 select(uint4 f, uint4 t, bool4 v){uint4 r={(v.x?t.x:f.x),(v.y?t.y:f.y),(v.z?t.z:f.z),(v.w?t.w:f.w)};return r;}
inline int2 select(int2 f, int2 t, bool2 v){int2 r={(v.x?t.x:f.x),(v.y?t.y:f.y)};return r;}
inline int3 select(int3 f, int3 t, bool3 v){int3 r={(v.x?t.x:f.x),(v.y?t.y:f.y),(v.z?t.z:f.z)};return r;}
inline int4 select(int4 f, int4 t, bool4 v){int4 r={(v.x?t.x:f.x),(v.y?t.y:f.y),(v.z?t.z:f.z),(v.w?t.w:f.w)};return r;}
inline bool2 select(bool2 f, bool2 t, bool2 v){bool2 r={(v.x?t.x:f.x),(v.y?t.y:f.y)};return r;}
inline bool3 select(bool3 f, bool3 t, bool3 v){bool3 r={(v.x?t.x:f.x),(v.y?t.y:f.y),(v.z?t.z:f.z)};return r;}
inline bool4 select(bool4 f, bool4 t, bool4 v){bool4 r={(v.x?t.x:f.x),(v.y?t.y:f.y),(v.z?t.z:f.z),(v.w?t.w:f.w)};return r;}

inline float2 abs(float2 f) { float2 r = { abs(f.x), abs(f.y) }; return r; }
inline float3 abs(float3 f) { float3 r = { abs(f.x), abs(f.y), abs(f.z) }; return r; }
inline float4 abs(float4 f) { float4 r = { abs(f.x), abs(f.y), abs(f.z), abs(f.w) }; return r; }

// acos
inline float2 acos(float2 f) { float2 r = { acos(f.x), acos(f.y) }; return r; }
inline float3 acos(float3 f) { float3 r = { acos(f.x), acos(f.y), acos(f.z) }; return r; }
inline float4 acos(float4 f) { float4 r = { acos(f.x), acos(f.y), acos(f.z), acos(f.w) }; return r; }

// all
inline bool all(float2 p) { return (p.x != 0 && p.y != 0); }
inline bool all(float3 p) { return (p.x != 0 && p.y != 0 && p.z != 0); }
inline bool all(float4 p) { return (p.x != 0 && p.y != 0 && p.z != 0 && p.w != 0); }

// any
inline bool any(float2 p) { return (p.x != 0 || p.y != 0); }
inline bool any(float3 p) { return (p.x != 0 || p.y != 0 || p.z != 0); }
inline bool any(float4 p) { return (p.x != 0 || p.y != 0 || p.z != 0 || p.w != 0); }

// asin
inline float2 asin(float2 f) { float2 r = { asin(f.x), asin(f.y) }; return r; }
inline float3 asin(float3 f) { float3 r = { asin(f.x), asin(f.y), asin(f.z) }; return r; }
inline float4 asin(float4 f) { float4 r = { asin(f.x), asin(f.y), asin(f.z), asin(f.w) }; return r; }

// atan
inline float2 atan(float2 f) { float2 r = { atan(f.x), atan(f.y) }; return r; }
inline float3 atan(float3 f) { float3 r = { atan(f.x), atan(f.y), atan(f.z) }; return r; }
inline float4 atan(float4 f) { float4 r = { atan(f.x), atan(f.y), atan(f.z), atan(f.w) }; return r; }

// atan2
inline float2 atan2(float2 x, float2 y) { float2 r = { atan2(y.x, x.x), atan2(y.y, x.y) }; return r; }
inline float3 atan2(float3 x, float3 y) { float3 r = { atan2(y.x, x.x), atan2(y.y, x.y), atan2(y.z, x.z) }; return r; }
inline float4 atan2(float4 x, float4 y) { float4 r = { atan2(y.x, x.x), atan2(y.y, x.y), atan2(y.z, x.z), atan2(y.w, x.w) }; return r; }

// ceil
inline float2 ceil(float2 f) { float2 r = { ceil(f.x), ceil(f.y) }; return r; }
inline float3 ceil(float3 f) { float3 r = { ceil(f.x), ceil(f.y), ceil(f.z) }; return r; }
inline float4 ceil(float4 f) { float4 r = { ceil(f.x), ceil(f.y), ceil(f.z), ceil(f.w) }; return r; }

// clamp
inline float2 clamp(float2 f, float2 minVal, float2 maxVal) { float2 r = { clamp(f.x, minVal.x, maxVal.x), clamp(f.y, minVal.y, maxVal.y) }; return r; }
inline float3 clamp(float3 f, float3 minVal, float3 maxVal) { float3 r = { clamp(f.x, minVal.x, maxVal.x), clamp(f.y, minVal.y, maxVal.y), clamp(f.z, minVal.z, maxVal.z) }; return r; }
inline float4 clamp(float4 f, float4 minVal, float4 maxVal) { float4 r = { clamp(f.x, minVal.x, maxVal.x), clamp(f.y, minVal.y, maxVal.y), clamp(f.z, minVal.z, maxVal.z), clamp(f.w, minVal.w, maxVal.w) }; return r; }

// cos
inline float2 cos(float2 f) { float2 r = { cos(f.x), cos(f.y) }; return r; }
inline float3 cos(float3 f) { float3 r = { cos(f.x), cos(f.y), cos(f.z) }; return r; }
inline float4 cos(float4 f) { float4 r = { cos(f.x), cos(f.y), cos(f.z), cos(f.w) }; return r; }

// cosh
inline float cosh(float f) { return (exp(f) + exp(-f)) / 2.0f; }
inline float2 cosh(float2 f) { float2 r = { cosh(f.x), cosh(f.y) }; return r; }
inline float3 cosh(float3 f) { float3 r = { cosh(f.x), cosh(f.y), cosh(f.z) }; return r; }
inline float4 cosh(float4 f) { float4 r = { cosh(f.x), cosh(f.y), cosh(f.z), cosh(f.w) }; return r; }

// cross
inline float3 cross(float3 a, float3 b)
{
float3 r;
r.x = a.y * b.z - a.z * b.y;
r.y = a.z * b.x - a.x * b.z;
r.z = a.x * b.y - a.y * b.x;
return r;
}
// determinant
inline float determinant(float2x2 m)
{
return m.m[0][0] * m.m[1][1] - m.m[1][0] * m.m[0][1];
}
inline float determinant(float3x3 m)
{
return m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[2][1] * m.m[1][2])
- m.m[1][0] * (m.m[0][1] * m.m[2][2] - m.m[2][1] * m.m[0][2])
+ m.m[2][0] * (m.m[0][1] * m.m[1][2] - m.m[1][1] * m.m[0][2]);
}
//float determinant(float4x4 m)
//{
// float2x2 a = float2x2(m);
// float2x2 b = float2x2(m[2].xy, m[3].xy);
// float2x2 c = float2x2(m[0].zw, m[1].zw);
// float2x2 d = float2x2(m[2].zw, m[3].zw);
// float s = determinant(a);
// return s*determinant(d - (1.0 / s)*c*float2x2(a[1][1], -a[0][1], -a[1][0], a[0][0])*b);
//}

// dot
inline float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// exp
inline float2 exp(float2 f) { float2 r = { exp(f.x), exp(f.y) }; return r; }
inline float3 exp(float3 f) { float3 r = { exp(f.x), exp(f.y), exp(f.z) }; return r; }
inline float4 exp(float4 f) { float4 r = { exp(f.x), exp(f.y), exp(f.z), exp(f.w) }; return r; }

// exp2
inline float2 exp2(float2 f) { float2 r = { pow(2, f.x), pow(2, f.y) }; return r; }
inline float3 exp2(float3 f) { float3 r = { pow(2, f.x), pow(2, f.y), pow(2, f.z) }; return r; }
inline float4 exp2(float4 f) { float4 r = { pow(2, f.x), pow(2, f.y), pow(2, f.z), pow(2, f.w) }; return r; }

// floor
inline float2 floor(float2 f) { float2 r = { floor(f.x), floor(f.y) }; return r; }
inline float3 floor(float3 f) { float3 r = { floor(f.x), floor(f.y),floor(f.z) }; return r; }
inline float4 floor(float4 f) { float4 r = { floor(f.x), floor(f.y),floor(f.z), floor(f.w) }; return r; }

// fmod
inline float fmod(float x, float y) { return x - y * floor(x / y); }
inline float2 fmod(float2 f, float m) { float2 r = { fmod(f.x, m), fmod(f.y, m) }; return r; }
inline float2 fmod(float2 f, float2 m) { float2 r = { fmod(f.x, m.x), fmod(f.y, m.y) }; return r; }
inline float3 fmod(float3 f, float m) { float3 r = { fmod(f.x, m), fmod(f.y, m), fmod(f.z, m) }; return r; }
inline float3 fmod(float3 f, float3 m) { float3 r = { fmod(f.x, m.x), fmod(f.y, m.y), fmod(f.z, m.z) }; return r; }
inline float4 fmod(float4 f, float m) { float4 r = { fmod(f.x, m), fmod(f.y, m), fmod(f.z, m), fmod(f.w, m) }; return r; }
inline float4 fmod(float4 f, float4 m) { float4 r = { fmod(f.x, m.x), fmod(f.y, m.y), fmod(f.z, m.z), fmod(f.w, m.w) }; return r; }

// frac
inline float frac(float f) { return f - floor(f); }
inline float2 frac(float2 f) { return f - floor(f); }
inline float3 frac(float3 f) { return f - floor(f); }
inline float4 frac(float4 f) { return f - floor(f); }

// length
inline float length(float2 p) { return sqrt(dot(p, p)); }
inline float length(float3 p) { return sqrt(dot(p, p)); }
inline float length(float4 p) { return sqrt(dot(p, p)); }
inline float length_sqr(float2 p) { return dot(p, p); }
inline float length_sqr(float3 p) { return dot(p, p); }
inline float length_sqr(float4 p) { return dot(p, p); }

// lerp
inline float lerp(float a, float b, float s) { return a + s * (b - a); }
inline float2 lerp(float2 a, float2 b, float2 s) { float2 r = { lerp(a.x, b.x, s.x), lerp(a.y, b.y, s.y) }; return r; }
inline float3 lerp(float3 a, float3 b, float3 s) { float3 r = { lerp(a.x, b.x, s.x), lerp(a.y, b.y, s.y), lerp(a.z, b.z, s.z) }; return r; }
inline float4 lerp(float4 a, float4 b, float4 s) { float4 r = { lerp(a.x, b.x, s.x), lerp(a.y, b.y, s.y), lerp(a.z, b.z, s.z), lerp(a.w, b.w, s.w) }; return r; }

// log
inline float2 log(float2 f) { float2 r = { log(f.x), log(f.y) }; return r; }
inline float3 log(float3 f) { float3 r = { log(f.x), log(f.y), log(f.z) }; return r; }
inline float4 log(float4 f) { float4 r = { log(f.x), log(f.y), log(f.z), log(f.w) }; return r; }

// log10
inline float log10(float f) { return (log(f) / log(10.0f)); }
inline float2 log10(float2 f) { float2 r = { log10(f.x), log10(f.y) }; return r; }
inline float3 log10(float3 f) { float3 r = { log10(f.x), log10(f.y), log10(f.z) }; return r; }
inline float4 log10(float4 f) { float4 r = { log10(f.x), log10(f.y), log10(f.z), log10(f.w) }; return r; }

// log2
inline float log2(float f) { return (log(f) / log(2.0f)); }
inline float2 log2(float2 f) { float2 r = { log2(f.x), log2(f.y) }; return r; }
inline float3 log2(float3 f) { float3 r = { log2(f.x), log2(f.y), log2(f.z) }; return r; }
inline float4 log2(float4 f) { float4 r = { log2(f.x), log2(f.y), log2(f.z), log2(f.w) }; return r; }

// mad
inline float mad(float a, float b, float s) { return a * b + s; }
inline float2 mad(float2 a, float2 b, float2 s) { return a * b + s; }
inline float3 mad(float3 a, float3 b, float3 s) { return a * b + s; }
inline float4 mad(float4 a, float4 b, float4 s) { return a * b + s; }

// max
inline float2 max(float2 a, float2 b) { float2 r = { max(a.x, b.x), max(a.y, b.y) }; return r; }
inline float3 max(float3 a, float3 b) { float3 r = { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) }; return r; }
inline float4 max(float4 a, float4 b) { float4 r = { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w) }; return r; }

// min
inline float2 min(float2 a, float2 b) { float2 r = { min(a.x, b.x), min(a.y, b.y) }; return r; }
inline float3 min(float3 a, float3 b) { float3 r = { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) }; return r; }
inline float4 min(float4 a, float4 b) { float4 r = { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w) }; return r; }

// mul
inline float2x2 mul(float2x2 a, float2x2 b)
{
float2x2 r;

// naive
for (int i = 0; i < 2; i++)
{
for (int j = 0; j < 2; j++)
{
r.m[i][j] = 0.0f;

for (int p = 0; p < 2; p++)
r.m[i][j] += a.m[i][p] * b.m[p][j];
}
}

return r;
}
inline float3x3 mul(float3x3 a, float3x3 b)
{
float3x3 r;

// naive
for (int i = 0; i < 3; i++)
{
for (int j = 0; j < 3; j++)
{
r.m[i][j] = 0.0f;

for (int p = 0; p < 3; p++)
r.m[i][j] += a.m[i][p] * b.m[p][j];
}
}

return r;
}
inline float4x4 mul(float4x4 a, float4x4 b)
{
float4x4 r;

// naive
for (int i = 0; i < 4; i++)
{
for (int j = 0; j < 4; j++)
{
r.m[i][j] = 0.0f;

for (int p = 0; p < 4; p++)
r.m[i][j] += a.m[i][p] * b.m[p][j];
}
}

return r;
}

// normalize
inline float2 normalize(float2 f) { return f / length(f); }
inline float3 normalize(float3 f) { return f / length(f); }
inline float4 normalize(float4 f) { return f / length(f); }

// pow
inline float2 pow(float2 f, float m) { float2 r = { pow(f.x, m), pow(f.y, m) }; return r; }
inline float2 pow(float2 f, float2 m) { float2 r = { pow(f.x, m.x), pow(f.y, m.y) }; return r; }
inline float3 pow(float3 f, float m) { float3 r = { pow(f.x, m), pow(f.y, m), pow(f.z, m) }; return r; }
inline float3 pow(float3 f, float3 m) { float3 r = { pow(f.x, m.x), pow(f.y, m.y), pow(f.z, m.z) }; return r; }
inline float4 pow(float4 f, float m) { float4 r = { pow(f.x, m), pow(f.y, m), pow(f.z, m), pow(f.w, m) }; return r; }
inline float4 pow(float4 f, float4 m) { float4 r = { pow(f.x, m.x), pow(f.y, m.y), pow(f.z, m.z), pow(f.w, m.w) }; return r; }

inline float2 rcp(float2 f) { float2 r = { rcp(f.x), rcp(f.y) }; return r; }
inline float3 rcp(float3 f) { float3 r = { rcp(f.x), rcp(f.y), rcp(f.z) }; return r; }
inline float4 rcp(float4 f) { float4 r = { rcp(f.x), rcp(f.y), rcp(f.z), rcp(f.w) }; return r; }

// reflect
inline float2 reflect(float2 i, float2 n) { return (i - 2.0f * n * dot(n, i)); }
inline float3 reflect(float3 i, float3 n) { return (i - 2.0f * n * dot(n, i)); }
inline float4 reflect(float4 i, float4 n) { return (i - 2.0f * n * dot(n, i)); }

// refract
inline float2 refract(float2 i, float2 n, float rindex)
{
float2 r;

float k = 1.0f - rindex * rindex * (1.0f - dot(n, i) * dot(n, i));
if (k < 0.0f)
r = 0;
else
r = rindex * i - (rindex * dot(n, i) + sqrt(k)) * n;

return r;
}
inline float3 refract(float3 i, float3 n, float rindex)
{
float3 r;

float k = 1.0f - rindex * rindex * (1.0f - dot(n, i) * dot(n, i));
if (k < 0.0f)
r = 0;
else
r = rindex * i - (rindex * dot(n, i) + sqrt(k)) * n;

return r;
}
inline float4 refract(float4 i, float4 n, float rindex)
{
float4 r;

float k = 1.0f - rindex * rindex * (1.0f - dot(n, i) * dot(n, i));
if (k < 0.0f)
r = 0;
else
r = rindex * i - (rindex * dot(n, i) + sqrt(k)) * n;

return r;
}

// round
inline float2 round(float2 f) { float2 r = { round(f.x), round(f.y) }; return r; }
inline float3 round(float3 f) { float3 r = { round(f.x), round(f.y), round(f.z) }; return r; }
inline float4 round(float4 f) { float4 r = { round(f.x), round(f.y), round(f.z), round(f.w) }; return r; }

// rsqrt
inline float2 rsqrt(float2 f) { float2 r = { rsqrt(f.x), rsqrt(f.y) }; return r; }
inline float3 rsqrt(float3 f) { float3 r = { rsqrt(f.x), rsqrt(f.y), rsqrt(f.z) }; return r; }
inline float4 rsqrt(float4 f) { float4 r = { rsqrt(f.x), rsqrt(f.y), rsqrt(f.z), rsqrt(f.w) }; return r; }

// sin
inline float2 sin(float2 f) { float2 r = { sin(f.x), sin(f.y) }; return r; }
inline float3 sin(float3 f) { float3 r = { sin(f.x), sin(f.y), sin(f.z) }; return r; }
inline float4 sin(float4 f) { float4 r = { sin(f.x), sin(f.y), sin(f.z), sin(f.w) }; return r; }

// sinh
inline float sinh(float f) { return (exp(f) - exp(-f)) / 2.0f; }
inline float2 sinh(float2 f) { float2 r = { sinh(f.x), sinh(f.y) }; return r; }
inline float3 sinh(float3 f) { float3 r = { sinh(f.x), sinh(f.y), sinh(f.z) }; return r; }
inline float4 sinh(float4 f) { float4 r = { sinh(f.x), sinh(f.y), sinh(f.z), sinh(f.w) }; return r; }

inline uint clz(uint8 x)
{
static const uint clz_lookup[16] = { 4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
uint8 upper = x >> 4;
uint8 lower = x & 0x0F;
return upper ? clz_lookup[upper] : 4 + clz_lookup[lower];
}

inline uint clz(uint16 x)
{
uint8 upper = (uint8)(x >> 8);
uint8 lower = (uint8)(x & 0xFF);
return upper ? clz(upper) : 16 + clz(lower);
}
inline uint clz(uint x)
{
uint16 upper = (uint16)(x >> 16);
uint16 lower = (uint16)(x & 0xFFFF);
return upper ? clz(upper) : 16 + clz(lower);
}
inline uint2 clz(uint2 x) { uint2 r = { clz(x.x), clz(x.y) }; return r; }
inline uint3 clz(uint3 x) { uint3 r = { clz(x.x), clz(x.y), clz(x.z) }; return r; }
inline uint4 clz(uint4 x) { uint4 r = { clz(x.x), clz(x.y), clz(x.z), clz(x.w) }; return r; }

inline uint ctz(uint8 x)
{
static const uint clz_lookup[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 4 };
uint8 upper = x >> 4;
uint8 lower = x & 0x0F;
return lower ? clz_lookup[upper] : 4 + clz_lookup[lower];
}

inline uint ctz(uint16 x)
{
uint8 upper = (uint8)(x >> 8);
uint8 lower = (uint8)(x & 0xFF);
return lower ? ctz(upper) : 16 + ctz(lower);
}
inline uint ctz(uint x)
{
uint16 upper = (uint16)(x >> 16);
uint16 lower = (uint16)(x & 0xFFFF);
return lower ? ctz(upper) : 16 + ctz(lower);
}
inline uint2 ctz(uint2 x) { uint2 r = { ctz(x.x), ctz(x.y) }; return r; }
inline uint3 ctz(uint3 x) { uint3 r = { ctz(x.x), ctz(x.y), ctz(x.z) }; return r; }
inline uint4 ctz(uint4 x) { uint4 r = { ctz(x.x), ctz(x.y), ctz(x.z), ctz(x.w) }; return r; }
// sqrt
inline float2 sqrt(float2 f) { float2 r = { sqrt(f.x), sqrt(f.y) }; return r; }
inline float3 sqrt(float3 f) { float3 r = { sqrt(f.x), sqrt(f.y), sqrt(f.z) }; return r; }
inline float4 sqrt(float4 f) { float4 r = { sqrt(f.x), sqrt(f.y), sqrt(f.z), sqrt(f.w) }; return r; }

// step
inline float step(float y, float x) { return x >= y ? 1.0 : 0.0; }
inline float2 step(float2 y, float2 x) { float2 r = { step(y.x, x.x), step(y.y, x.y) }; return r; }
inline float3 step(float3 y, float3 x) { float3 r = { step(y.x, x.x), step(y.y, x.y), step(y.z, x.z) }; return r; }
inline float4 step(float4 y, float4 x) { float4 r = { step(y.x, x.x), step(y.y, x.y), step(y.z, x.z), step(x.w, y.w) }; return r; }

// tan
inline float2 tan(float2 f) { float2 r = { tan(f.x), tan(f.y) }; return r; }
inline float3 tan(float3 f) { float3 r = { tan(f.x), tan(f.y), tan(f.z) }; return r; }
inline float4 tan(float4 f) { float4 r = { tan(f.x), tan(f.y), tan(f.z), tan(f.w) }; return r; }

// tanh
inline float tanh(float f) { return sinh(f) / cosh(f); }
inline float2 tanh(float2 f) { float2 r = { tanh(f.x), tanh(f.y) }; return r; }
inline float3 tanh(float3 f) { float3 r = { tanh(f.x), tanh(f.y), tanh(f.z) }; return r; }
inline float4 tanh(float4 f) { float4 r = { tanh(f.x), tanh(f.y), tanh(f.z), tanh(f.w) }; return r; }

inline bool2 isnan(float2 f) { bool2 r = { isnan(f.x), isnan(f.y) }; return r; }
inline bool3 isnan(float3 f) { bool3 r = { isnan(f.x), isnan(f.y), isnan(f.z) }; return r; }
inline bool4 isnan(float4 f) { bool4 r = { isnan(f.x), isnan(f.y), isnan(f.z), isnan(f.w) }; return r; }

inline bool isinf(float f) { return f == floatbits(0x7f800000); }
inline bool2 isinf(float2 f) { bool2 r = { isinf(f.x), isinf(f.y) }; return r; }
inline bool3 isinf(float3 f) { bool3 r = { isinf(f.x), isinf(f.y), isinf(f.z) }; return r; }
inline bool4 isinf(float4 f) { bool4 r = { isinf(f.x), isinf(f.y), isinf(f.z), isinf(f.w) }; return r; }

inline uint popcount(uint n)
{
#define POW2(c)	(1U << (c))
#define MASK(c)	((uint)(-1) / (POW2(POW2(c)) + 1U))
#define COUNT(x, c)	((x) & MASK(c)) + (((x)>>(POW2(c))) & MASK(c))
n = COUNT(n, 0);
n = COUNT(n, 1);
n = COUNT(n, 2);
n = COUNT(n, 3);
n = COUNT(n, 4);
//n = COUNT(n, 5);	// uncomment this line for 64-bit integers
return n;
#undef COUNT
#undef MASK
#undef POW2
}
inline uint2 popcount(uint2 f) { uint2 r = { popcount(f.x), popcount(f.y) }; return r; }
inline uint3 popcount(uint3 f) { uint3 r = { popcount(f.x), popcount(f.y), popcount(f.z) }; return r; }
inline uint4 popcount(uint4 f) { uint4 r = { popcount(f.x), popcount(f.y), popcount(f.z), popcount(f.w) }; return r; }
inline uint reverse(uint n) {
uint rev = 0;
while (n > 0) {
rev <<= 1;
if ((n & 1) == 1) {
rev ^= 1;
}
n >>= 1;
}
return rev;
}
inline uint2 reverse(uint2 n) { uint2 r = { reverse(n.x), reverse(n.y) }; return r; }
inline uint3 reverse(uint3 n) { uint3 r = { reverse(n.x), reverse(n.y), reverse(n.z) }; return r; }
inline uint4 reverse(uint4 n) { uint4 r = { reverse(n.x), reverse(n.y), reverse(n.z), reverse(n.w) }; return r; }
// transpose
inline float2x2 transpose(float2x2 m)
{
float2x2 r;
r.m[0][0] = m.m[0][0];
r.m[0][1] = m.m[1][0];
r.m[1][0] = m.m[0][1];
r.m[1][1] = m.m[1][1];
return r;
}
inline float3x3 transpose(float3x3 m)
{
float3x3 r;
r.m[0][0] = m.m[0][0];
r.m[0][1] = m.m[1][0];
r.m[0][2] = m.m[2][0];
r.m[1][0] = m.m[0][1];
r.m[1][1] = m.m[1][1];
r.m[1][2] = m.m[2][1];
r.m[2][0] = m.m[0][2];
r.m[2][1] = m.m[1][2];
r.m[2][2] = m.m[2][2];
return r;
}
inline float4x4 transpose(float4x4 m)
{
float4x4 r;
r.m[0][0] = m.m[0][0];
r.m[0][1] = m.m[1][0];
r.m[0][2] = m.m[2][0];
r.m[0][3] = m.m[3][0];
r.m[1][0] = m.m[0][1];
r.m[1][1] = m.m[1][1];
r.m[1][2] = m.m[2][1];
r.m[1][3] = m.m[3][1];
r.m[2][0] = m.m[0][2];
r.m[2][1] = m.m[1][2];
r.m[2][2] = m.m[2][2];
r.m[2][3] = m.m[3][2];
r.m[3][0] = m.m[0][3];
r.m[3][1] = m.m[1][3];
r.m[3][2] = m.m[2][3];
r.m[3][3] = m.m[3][3];
return r;
}

// trunc
inline int2 trunc(float2 f) { int2 r = { trunc(f.x), trunc(f.y) }; return r; }
inline int3 trunc(float3 f) { int3 r = { trunc(f.x), trunc(f.y), trunc(f.z) }; return r; }
inline int4 trunc(float4 f) { int4 r = { trunc(f.x), trunc(f.y), trunc(f.z), trunc(f.w) }; return r; }

//-------------------------------------------------------------------------------------------------
// TEXTURES, SAMPLERS, et Al.
//-------------------------------------------------------------------------------------------------
#include "Types.h"
inline int _atomic_exchange(uniform int* v, const int a){return atomic_swap_global(v, a);}
inline uint _atomic_exchange(uniform uint* v, const uint a){return atomic_swap_global(v, a);}
inline int _atomic_add(uniform int* v, const int a){return atomic_add_global(v, a);}
inline uint _atomic_add(uniform uint* v, const uint a){return atomic_add_global(v, a);}
inline int _atomic_sub(uniform int* v, const int a){return atomic_subtract_global(v, a);}
inline uint _atomic_sub(uniform uint* v, const uint a){return atomic_subtract_global(v, a);}
inline int _atomic_min(uniform int* v, const int a){return atomic_min_global(v, a);}
inline uint _atomic_min(uniform uint* v, const uint a){return atomic_min_global(v, a);}
inline int _atomic_max(uniform int* v, const int a){return atomic_max_global(v, a);}
inline uint _atomic_max(uniform uint* v, const uint a){return atomic_max_global(v, a);}
inline int _atomic_and(uniform int *v, const int a) { return atomic_and_global(v, a); }
inline uint _atomic_and(uniform uint *v, const uint a) { return atomic_and_global(v, a); }
inline int _atomic_or(uniform int *v, const int a) { return atomic_or_global(v, a); }
inline uint _atomic_or(uniform uint *v, const uint a) { return atomic_or_global(v, a); }
inline int _atomic_xor(uniform int *v, const int a) { return atomic_xor_global(v, a); }
inline uint _atomic_xor(uniform uint *v, const uint a) { return atomic_xor_global(v, a); }
inline int _atomic_compare_exchange(uniform int *v, const int a, const int b) { return atomic_compare_exchange_global(v, a, b); }
inline uint _atomic_compare_exchange(uniform uint *v, const uint a, const uint b) { return atomic_compare_exchange_global(v, a, b); }

// float2 GetCoord(float2 u, TEXTURE_ADDRESS_MODE addr) {
// switch (addr) {
// case TEXTURE_ADDRESS_WRAP:
// u = fmod(u, 1.0f);
// break;
// case TEXTURE_ADDRESS_CLAMP:
// u = clamp(u, _float2(0.0f), _float2(1.0f));
// break;
// case TEXTURE_ADDRESS_MIRROR:
// {
// int2 mulOne = _int2(trunc(u)) % 2;
// if (mulOne.x)
// u.x = frac(u.x);
// else
// u.x = 1.0f - frac(u.x);
// if (mulOne.y)
// u.y = frac(u.y);
// else
// u.y = 1.0f - frac(u.y);
// }
// break;
// }
// return u;
// }
// float3 GetCoord(float3 u, TEXTURE_ADDRESS_MODE addr) {
// switch (addr) {
// case TEXTURE_ADDRESS_WRAP:
// u = fmod(u, 1.0f);
// break;
// case TEXTURE_ADDRESS_CLAMP:
// u = clamp(u, _float3(0.0f), _float3(1.0f));
// break;
// case TEXTURE_ADDRESS_MIRROR:
// {
// int3 mulOne = _int3(trunc(u)) % 2;
// if (mulOne.x)
// u.x = frac(u.x);
// else
// u.x = 1.0f - frac(u.x);
// if (mulOne.y)
// u.y = frac(u.y);
// else
// u.y = 1.0f - frac(u.y);
// if (mulOne.z)
// u.z = frac(u.z);
// else
// u.z = 1.0f - frac(u.z);
// }
// break;
// }
// return u;
// }

// float SampleFloatArray(void* pData, uint index, TEXTURE_BIT_COUNT bitCount){
// 	uint8* p = (uint8*)pData;
// 	p += bitCount * index;
// 	switch(bitCount){
// 		case BIT_8:
// 		return ((float)*p) / 255f;
// 		case BIT_16:
// 		return half_to_float_fast(*((uint16*)p));
// 		case BIT_32:
// 		return *((float*)p);
// 	}
// 	return 0;
// }
// float4 GetColor(float* pData, uint index, uint numComponents){
// float4 f;
// switch(numComponents){
// case 1:
// f = _float4(pData[index + 0],
// 0,
// 0,
// 0);
// break;
// case 2:
// f = _float4(pData[index + 0],
// pData[index + 1],
// 0,
// 0);
// break;
// case 3:
// f = _float4(pData[index + 0],
// pData[index + 1],
// pData[index + 2],
// 0);
// break;
// case 4:
// f = _float4(pData[index + 0],
// pData[index + 1],
// pData[index + 2],
// pData[index + 3]);
// break;
// default:
// f = _float4(0,0,0,0);
// }
// return f;
// }
// float4 Smp2DPoint(const Texture2D* pTexture, SamplerState const& sampler, float2 uv, const float lodLevelf)
// {
// uint lodLevel = (uint)lodLevelf;
// if(lodLevel >= pTexture->lodLevel) lodLevel = pTexture->lodLevel - 1;
// float* pData = pTexture->pData[lodLevel];
// uv =	GetCoord(uv, sampler.address);
// uint index = pTexture->numComponents * ((uint)(uv.y * (pTexture->height - 1)) * pTexture->width + (uint)(uv.x * (pTexture->width - 1)));
// return GetColor(pData, index, pTexture->numComponents);
// }

// inline float sign(float v){
// if (v > 0) return 1;
// else if(v < 0) return -1;
// return 0;
// }
// inline float2 sign(float2 v){float2 r={sign(v.x),sign(v.y)};return r;}
// inline float3 sign(float3 v){float3 r={sign(v.x),sign(v.y),sign(v.z)};return r;}
// float4 Smp2DBi(const Texture2D* pTexture, SamplerState const& sampler, const float2 uv, const float lodLevel){
// float2 uvSign = sign(uv);
// float4 v0 = Smp2DPoint(pTexture, sampler, uv, lodLevel);
// float4 v1 = Smp2DPoint(pTexture, sampler, uv + _float2(uvSign.x, 0), lodLevel);
// float4 v2 = Smp2DPoint(pTexture, sampler, uv + _float2(0, uvSign.y), lodLevel);
// float4 v3 = Smp2DPoint(pTexture, sampler, uv + uvSign, lodLevel);
// float2 fracUV = frac(uv);
// float4 hori0 = lerp(v0, v1, _float4(fracUV.x));
// float4 hori1 = lerp(v2, v3, _float4(fracUV.x));
// return lerp(hori0, hori1, _float4(fracUV.y));
// }
// float4 Smp2DTri(const Texture2D* pTexture, SamplerState const& sampler, const float2 uv, const float lodLevel){
// float lod = max(0, lodLevel);
// float4 v0 = Smp2DBi(pTexture, sampler, uv, lod);
// float4 v1 = Smp2DBi(pTexture, sampler, uv, lod + 1);
// return lerp(v0, v1, frac(lod));
// }

// float4 Smp3DPoint(const Texture3D* pTexture, SamplerState const& sampler, float3 uv, const float lodLevelf){
// uint lodLevel = (uint)lodLevelf;
// if(lodLevel >= pTexture->lodLevel) lodLevel = pTexture->lodLevel - 1;
// float* pData = pTexture->pData[lodLevel];
// uv =	GetCoord(uv, sampler.address);
// uint index = pTexture->numComponents * 
// ((uint)(uv.z * (pTexture->depth - 1)) * pTexture->height * pTexture->width
//  + (uint)(uv.y * (pTexture->height - 1)) * pTexture->width
//  + (uint)(uv.x * (pTexture->width - 1)));
// return GetColor(pData, index, pTexture->numComponents);
// }
// float4 Smp3DBi(const Texture3D* pTexture, SamplerState const& sampler, const float3 uv, const float lodLevel){
// float3 uvSign = sign(uv);
// const float3 ofst[] = {
// 0,0,0,
// uvSign.x, 0,0,
// 0, uvSign.y,0,
// uvSign.x, uvSign.y,0,
// 0,0,uvSign.z,
// uvSign.x, 0,uvSign.z,
// 0, uvSign.y,uvSign.z,
// uvSign.x, uvSign.y,uvSign.z,
// };
// float4 vs[8];
// for (uint i = 0; i < 8; ++i){
// vs[i] = Smp3DPoint(pTexture, sampler, uv + ofst[i], lodLevel);
// }
// float3 fracUV = frac(uv);
// for(uint i = 0; i < 4; ++i){
// vs[i] = lerp(vs[i], vs[i + 4], fracUV.z);
// }
// for(uint i = 0; i < 2; ++i){
// vs[i] = lerp(vs[i], vs[i + 2], fracUV.y);
// }
// return lerp(vs[0], vs[1], fracUV.x);
// }

// float4 Smp3DTri(const Texture3D* pTexture, SamplerState const& sampler, const float3 uv, const float lodLevel){
// float lod = max(0, lodLevel);
// float4 v0 = Smp3DBi(pTexture, sampler, uv, lod);
// float4 v1 = Smp3DBi(pTexture, sampler, uv, lod + 1);
// return lerp(v0, v1, frac(lod));
// }

#include <embree3/rtcore.isph>

Hit trace_closest(uniform RTCScene scene, Ray ray) {
	uniform RTCIntersectContext intersectCtx;
	rtcInitIntersectContext(&intersectCtx);
	RTCRay r;
	r.org_x = ray.v0[0];
	r.org_y = ray.v0[1];
	r.org_z = ray.v0[2];
	r.tnear = ray.v1;
	r.dir_x = ray.v2[0];
	r.dir_y = ray.v2[1];
	r.dir_z = ray.v2[2];
	r.tfar = ray.v3;
	r.mask = 0xffffu;
	r.flags = 0;
	RTCRayHit rh;
	rh.ray = r;
	rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
	rtcIntersectV(scene, &intersectCtx, &rh);
	Hit hit;
	RTCHit h = rh.hit;
	hit.v0 = h.geomID;
	hit.v1 = h.primID;
	hit.v2[0] = h.u;
	hit.v2[1] = h.v;
	return hit;
}

bool trace_any(uniform RTCScene scene, Ray ray) {
	// TODO: check availability
	uniform RTCIntersectContext intersectCtx;
	rtcInitIntersectContext(&intersectCtx);
	RTCRay r;
	r.org_x = ray.v0[0];
	r.org_y = ray.v0[1];
	r.org_z = ray.v0[2];
	r.tnear = ray.v1;
	r.dir_x = ray.v2[0];
	r.dir_y = ray.v2[1];
	r.dir_z = ray.v2[2];
	r.tfar = ray.v3;
	r.mask = 0xffffu;
	r.flags = 0;
	rtcOccludedV(scene, &intersectCtx, &r);
	if(ray.v3 != r.tfar) return true;
	else return false;
}

#define LC_BINDLESS_BUFFER_READ_TYPE(T) \
inline T lc_bindless_buffer_read_##T(uniform LCBindlessArray array, int index, int i) { \
	T* buffer = (T*)array.v0[index]; \
	return buffer[i]; \
}

LC_BINDLESS_BUFFER_READ_TYPE(uint);

// #define LC_BINDLESS_TEXTURE2D_READ_TYPE(T) \
// inline T lc_bindless_texture2d_read_##T(uniform LCBindlessArray array, uint index, uint2 p) { \
// 	Texture2D* tex = (Texture2D*)array.v1[index]; \
// 	return texture_read(tex, p, 0); \
// }

// LC_BINDLESS_TEXTURE2D_READ_TYPE(float4);

#define LC_BINDLESS_TEXTURE2D_SAMPLE_TYPE(T) \
inline T lc_bindless_texture2d_sample(uniform LCBindlessArray array, uint index, float2 p) { \
	print(">>\n");\
	Texture2D* tex = (Texture2D*)array.v1[index]; \
	return texture_sample_tmp(tex, p, 0); \
}

LC_BINDLESS_TEXTURE2D_SAMPLE_TYPE(float4);

inline uint2 lc_bindless_texture2d_size(uniform LCBindlessArray array, int index, int level) {
	uint x = array.v3[index * 2 + 0];
	uint y = array.v3[index * 2 + 1];
	return _uint2(max(x >> level, 1u), max(y >> level, 1u));
}

inline uint3 lc_bindless_texture3d_size(uniform LCBindlessArray array, int index, int level) {
	uint x = array.v4[index * 3 + 0];
	uint y = array.v4[index * 3 + 1];
	uint z = array.v4[index * 3 + 2];
	return _uint3(max(x >> level, 1u), max(y >> level, 1u), max(z >> level, 1u));
}
