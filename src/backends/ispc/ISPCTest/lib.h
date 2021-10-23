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

// abs
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
inline float2 clamp(float2 f, float minVal, float maxVal) { float2 r = { clamp(f.x, minVal, maxVal), clamp(f.y, minVal, maxVal) }; return r; }
inline float3 clamp(float3 f, float minVal, float maxVal) { float3 r = { clamp(f.x, minVal, maxVal), clamp(f.y, minVal, maxVal), clamp(f.z, minVal, maxVal) }; return r; }
inline float4 clamp(float4 f, float minVal, float maxVal) { float4 r = { clamp(f.x, minVal, maxVal), clamp(f.y, minVal, maxVal), clamp(f.z, minVal, maxVal), clamp(f.w, minVal, maxVal) }; return r; }

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

// degrees
inline float degrees(float f) { return (180 * f) / PI; }
inline float2 degrees(float2 f) { float2 r = { degrees(f.x), degrees(f.y) }; return r; }
inline float3 degrees(float3 f) { float3 r = { degrees(f.x), degrees(f.y), degrees(f.z) }; return r; }
inline float4 degrees(float4 f) { float4 r = { degrees(f.x), degrees(f.y), degrees(f.z), degrees(f.w) }; return r; }

// determinant
float determinant(float2x2 m)
{
	return m.m[0][0] * m.m[1][1] - m.m[1][0] * m.m[0][1];
}
float determinant(float3x3 m)
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

// distance
inline float distance(float a, float b) { return abs(a - b); }
inline float distance(float2 a, float2 b) { return sqrt(dot(a, b)); }
inline float distance(float3 a, float3 b) { return sqrt(dot(a, b)); }
inline float distance(float4 a, float4 b) { return sqrt(dot(a, b)); }

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

// lerp
inline float lerp(float a, float b, float s) { return a + s * (b - a); }
inline float2 lerp(float2 a, float2 b, float s) { float2 r = { lerp(a.x, b.x, s), lerp(a.y, b.y, s) }; return r; }
inline float3 lerp(float3 a, float3 b, float s) { float3 r = { lerp(a.x, b.x, s), lerp(a.y, b.y, s), lerp(a.z, b.z, s) }; return r; }
inline float4 lerp(float4 a, float4 b, float s) { float4 r = { lerp(a.x, b.x, s), lerp(a.y, b.y, s), lerp(a.z, b.z, s), lerp(a.w, b.w, s) }; return r; }

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

// radians
inline float radians(float f) { return (PI * f) / 180.0f; }
inline float2 radians(float2 f) { return (PI * f) / 180.0f; }
inline float3 radians(float3 f) { return (PI * f) / 180.0f; }
inline float4 radians(float4 f) { return (PI * f) / 180.0f; }

// rcp
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

// saturate
inline float saturate(float f) { return clamp(f, 0.0f, 1.0f); }
inline float2 saturate(float2 f) { return clamp(f, 0.0f, 1.0f); }
inline float3 saturate(float3 f) { return clamp(f, 0.0f, 1.0f); }
inline float4 saturate(float4 f) { return clamp(f, 0.0f, 1.0f); }

// sign
inline float sign(float f) { return f < 0 ? -1 : 1; }
inline float2 sign(float2 f) { float2 r = { f.x < 0 ? -1 : 1, f.y < 0 ? -1 : 1 }; return r; }
inline float3 sign(float3 f) { float3 r = { f.x < 0 ? -1 : 1, f.y < 0 ? -1 : 1, f.z < 0 ? -1 : 1 }; return r; }
inline float4 sign(float4 f) { float4 r = { f.x < 0 ? -1 : 1, f.y < 0 ? -1 : 1, f.z < 0 ? -1 : 1, f.w < 0 ? -1 : 1 }; return r; }

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

// smoothstep
inline float smoothstep(float minValue, float maxValue, float x)
{
	float t;
	t = saturate((x - minValue) / (maxValue - minValue));
	return t * t * (3.0f - 2.0f * t);
}
inline float2 smoothstep(float2 a, float2 b, float x) { float2 r = { smoothstep(a.x, b.x, x), smoothstep(a.y, b.y, x) }; return r; }
inline float3 smoothstep(float3 a, float3 b, float x) { float3 r = { smoothstep(a.x, b.x, x), smoothstep(a.y, b.y, x), smoothstep(a.z, b.z, x) }; return r; }
inline float4 smoothstep(float4 a, float4 b, float x) { float4 r = { smoothstep(a.x, b.x, x), smoothstep(a.y, b.y, x), smoothstep(a.z, b.z, x), smoothstep(a.w, b.w, x) }; return r; }

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

uint popcount(uint n)
{
#define POW2(c)      (1U << (c))
#define MASK(c)      ((uint)(-1) / (POW2(POW2(c)) + 1U))
#define COUNT(x, c)  ((x) & MASK(c)) + (((x)>>(POW2(c))) & MASK(c))
	n = COUNT(n, 0);
	n = COUNT(n, 1);
	n = COUNT(n, 2);
	n = COUNT(n, 3);
	n = COUNT(n, 4);
	//	n = COUNT(n, 5);  // uncomment this line for 64-bit integers
	return n;
#undef COUNT
#undef MASK
#undef POW2
}
inline uint2 popcount(uint2 f) { uint2 r = { popcount(f.x), popcount(f.y) }; return r; }
inline uint3 popcount(uint3 f) { uint3 r = { popcount(f.x), popcount(f.y), popcount(f.z) }; return r; }
inline uint4 popcount(uint4 f) { uint4 r = { popcount(f.x), popcount(f.y), popcount(f.z), popcount(f.w) }; return r; }
uint reverse(uint n) {
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
uint2 reverse(uint2 n) { uint2 r = { reverse(n.x), reverse(n.y) }; return r; }
uint3 reverse(uint3 n) { uint3 r = { reverse(n.x), reverse(n.y), reverse(n.z) }; return r; }
uint4 reverse(uint4 n) { uint4 r = { reverse(n.x), reverse(n.y), reverse(n.z), reverse(n.w) }; return r; }
// transpose
float2x2 transpose(float2x2 m)
{
	float2x2 r;
	r.m[0][0] = m.m[0][0];
	r.m[0][1] = m.m[1][0];
	r.m[1][0] = m.m[0][1];
	r.m[1][1] = m.m[1][1];
	return r;
}
float3x3 transpose(float3x3 m)
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
float4x4 transpose(float4x4 m)
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

float GetCoord(float u, TEXTURE_ADDRESS_MODE addr) {
	switch (addr) {
	case TEXTURE_ADDRESS_WRAP:
		u = fmod(u, 1.0f);
		break;
	case TEXTURE_ADDRESS_CLAMP:
		u = clamp(u, 0.0f, 1.0f);
		break;
	case TEXTURE_ADDRESS_MIRROR:
		if ((int)trunc(u) % 2)
			u = frac(u);
		else
			u = 1.0f - frac(u);
		break;
	}
}
float4 SampleTexture2D(const Texture2D* pTexture, const SamplerState& sampler, const float2 uv)
{
	float u = uv.x;
	float v = uv.y;
	u = GetCoord(u, sampler.addressU);
	v = GetCoord(v, sampler.addressV);

	u *= (pTexture->width - 1);
	v *= (pTexture->height - 1);

	int index = pTexture->numComponents * (trunc(v) * pTexture->width + trunc(u));
	float4 f = { pTexture->pData[index + 0],
	pTexture->pData[index + 1],
	pTexture->pData[index + 2],
	pTexture->pData[index + 3] };
	return f;
}

inline float2 make_float2(float x,float y) { float2 f = {x, y}; return f;}
inline float2 make_float2(float2 xy) { return xy;}

inline float3 make_float3(float x,float y,float z){float3 f = {x,y,z}; return f;};
inline float3 make_float3(float2 xy,float z){float3 f = {xy.x,xy.y,z}; return f;};
inline float3 make_float3(float x,float2 yz){float3 f = {x,yz.x,yz.y}; return f;};
inline float3 make_float3(float3 xyz) { return xyz;}

inline float4 make_float4(float x,float y,float z,float w){float4 f = {x,y,z,w}; return f;}
inline float4 make_float4(float2 xy,float z,float w){float4 f = {xy.x, xy.y,z,w}; return f;}
inline float4 make_float4(float3 xyz,float w){float4 f = {xyz.x, xyz.y,xyz.z,w}; return f;}
inline float4 make_float4(float4 xyzw){return xyzw;}
inline float4 make_float4(float x,float2 yz,float w){float4 f = {x,yz.x,yz.y,w}; return f;}
inline float4 make_float4(float x,float3 yzw){float4 f = {x,yzw.x,yzw.y,yzw.z}; return f;}
inline float4 make_float4(float x,float y,float2 zw){float4 f = {x,y,zw.x,zw.y}; return f;}


inline uint2 make_uint2(uint x,uint y) { uint2 f = {x, y}; return f;}
inline uint2 make_uint2(uint2 xy) { return xy;}

inline uint3 make_uint3(uint x,uint y,uint z){uint3 f = {x,y,z}; return f;};
inline uint3 make_uint3(uint2 xy,uint z){uint3 f = {xy.x,xy.y,z}; return f;};
inline uint3 make_uint3(uint x,uint2 yz){uint3 f = {x,yz.x,yz.y}; return f;};
inline uint3 make_uint3(uint3 xyz) { return xyz;}

inline uint4 make_uint4(uint x,uint y,uint z,uint w){uint4 f = {x,y,z,w}; return f;}
inline uint4 make_uint4(uint2 xy,uint z,uint w){uint4 f = {xy.x, xy.y,z,w}; return f;}
inline uint4 make_uint4(uint3 xyz,uint w){uint4 f = {xyz.x, xyz.y,xyz.z,w}; return f;}
inline uint4 make_uint4(uint4 xyzw){return xyzw;}
inline uint4 make_uint4(uint x,uint2 yz,uint w){uint4 f = {x,yz.x,yz.y,w}; return f;}
inline uint4 make_uint4(uint x,uint3 yzw){uint4 f = {x,yzw.x,yzw.y,yzw.z}; return f;}
inline uint4 make_uint4(uint x,uint y,uint2 zw){uint4 f = {x,y,zw.x,zw.y}; return f;}


inline int2 make_int2(int x,int y) { int2 f = {x, y}; return f;}
inline int2 make_int2(int2 xy) { return xy;}

inline int3 make_int3(int x,int y,int z){int3 f = {x,y,z}; return f;};
inline int3 make_int3(int2 xy,int z){int3 f = {xy.x,xy.y,z}; return f;};
inline int3 make_int3(int x,int2 yz){int3 f = {x,yz.x,yz.y}; return f;};
inline int3 make_int3(int3 xyz) { return xyz;}

inline int4 make_int4(int x,int y,int z,int w){int4 f = {x,y,z,w}; return f;}
inline int4 make_int4(int2 xy,int z,int w){int4 f = {xy.x, xy.y,z,w}; return f;}
inline int4 make_int4(int3 xyz,int w){int4 f = {xyz.x, xyz.y,xyz.z,w}; return f;}
inline int4 make_int4(int4 xyzw){return xyzw;}
inline int4 make_int4(int x,int2 yz,int w){int4 f = {x,yz.x,yz.y,w}; return f;}
inline int4 make_int4(int x,int3 yzw){int4 f = {x,yzw.x,yzw.y,yzw.z}; return f;}
inline int4 make_int4(int x,int y,int2 zw){int4 f = {x,y,zw.x,zw.y}; return f;}


inline bool2 make_bool2(bool x,bool y) { bool2 f = {x, y}; return f;}
inline bool2 make_bool2(bool2 xy) { return xy;}

inline bool3 make_bool3(bool x,bool y,bool z){bool3 f = {x,y,z}; return f;};
inline bool3 make_bool3(bool2 xy,bool z){bool3 f = {xy.x,xy.y,z}; return f;};
inline bool3 make_bool3(bool x,bool2 yz){bool3 f = {x,yz.x,yz.y}; return f;};
inline bool3 make_bool3(bool3 xyz) { return xyz;}

inline bool4 make_bool4(bool x,bool y,bool z,bool w){bool4 f = {x,y,z,w}; return f;}
inline bool4 make_bool4(bool2 xy,bool z,bool w){bool4 f = {xy.x, xy.y,z,w}; return f;}
inline bool4 make_bool4(bool3 xyz,bool w){bool4 f = {xyz.x, xyz.y,xyz.z,w}; return f;}
inline bool4 make_bool4(bool4 xyzw){return xyzw;}
inline bool4 make_bool4(bool x,bool2 yz,bool w){bool4 f = {x,yz.x,yz.y,w}; return f;}
inline bool4 make_bool4(bool x,bool3 yzw){bool4 f = {x,yzw.x,yzw.y,yzw.z}; return f;}
inline bool4 make_bool4(bool x,bool y,bool2 zw){bool4 f = {x,y,zw.x,zw.y}; return f;}

inline float2x2 make_float2x2(float2 c0,float2 c1){
float2x2 f;
f.m[0][0]=c0.x;
f.m[1][0]=c0.y;
f.m[0][1]=c1.x;
f.m[1][1]=c1.y;
return f;
}
inline float3x3 make_float3x3(float3 c0,float3 c1,float3 c2){
float3x3 f;
f.m[0][0]=c0.x;
f.m[1][0]=c0.y;
f.m[2][0]=c0.z;
f.m[0][1]=c1.x;
f.m[1][1]=c1.y;
f.m[2][1]=c1.z;
f.m[0][2]=c2.x;
f.m[1][2]=c2.y;
f.m[2][2]=c2.z;
return f;
}
inline float4x4 make_float4x4(float4 c0,float4 c1,float4 c2,float4 c3){
float4x4 f;
f.m[0][0]=c0.x;
f.m[1][0]=c0.y;
f.m[2][0]=c0.z;
f.m[3][0]=c0.w;
f.m[0][1]=c1.x;
f.m[1][1]=c1.y;
f.m[2][1]=c1.z;
f.m[3][1]=c1.w;
f.m[0][2]=c2.x;
f.m[1][2]=c2.y;
f.m[2][2]=c2.z;
f.m[3][2]=c2.w;
f.m[0][3]=c2.x;
f.m[1][3]=c2.y;
f.m[2][3]=c2.z;
f.m[3][3]=c2.w;
return f;
}
