#pragma once
#include <stdint.h>
#include <math.h>

#pragma once
#include "detail/c11.inc"

// Vector
typedef struct {
	_Alignas(8) float x;
	float y;
} float2;
typedef struct {
	_Alignas(16) float x, y, z;
} float3;
typedef struct {
	_Alignas(16) float x;
	float y;
	float z;
	float w;
} float4;
typedef struct {
	_Alignas(16) double x;
	double y;
} double2;
typedef struct {
	_Alignas(16) double x;
	double y, z;
} double3;
typedef struct {
	_Alignas(16) double x;
	double y, z, w;
} double4;
typedef struct {
	_Alignas(16) uint64_t x;
	uint64_t y;
} uint64_t2;
typedef struct {
	_Alignas(16) uint64_t x;
	uint64_t y, z;
} uint64_t3;
typedef struct {
	_Alignas(16) uint64_t x;
	uint64_t y, z, w;
} uint64_t4;
typedef struct {
	_Alignas(16) int64_t x;
	int64_t y;
} int64_t2;
typedef struct {
	_Alignas(16) int64_t x;
	int64_t y, z;
} int64_t3;
typedef struct {
	_Alignas(16) int64_t x;
	int64_t y, z, w;
} int64_t4;
typedef struct {
	_Alignas(8) uint32_t x;
	uint32_t y;
} uint32_t2;
typedef struct {
	_Alignas(16) uint32_t x;
	uint32_t y, z;
} uint32_t3;
typedef struct {
	_Alignas(8) uint32_t x;
	uint32_t y, z, w;
} uint32_t4;
typedef struct {
	_Alignas(16) int32_t x;
	int32_t y;
} int32_t2;
typedef struct {
	_Alignas(16) int32_t x;
	int32_t y, z;
} int32_t3;
typedef struct {
	_Alignas(16) int32_t x;
	int32_t y, z, w;
} int32_t4;
typedef struct {
	_Alignas(4) uint16_t x;
	uint16_t y;
} uint16_t2;
typedef struct {
	_Alignas(8) uint16_t x;
	uint16_t y, z;
} uint16_t3;
typedef struct {
	_Alignas(8) uint16_t x;
	uint16_t y, z, w;
} uint16_t4;
typedef struct {
	_Alignas(4) int16_t x;
	int16_t y;
} int16_t2;
typedef struct {
	_Alignas(8) int16_t x;
	int16_t y, z;
} int16_t3;
typedef struct {
	_Alignas(8) int16_t x;
	int16_t y, z, w;
} int16_t4;
typedef struct {
	_Alignas(2) uint8_t x;
	uint8_t y;
} uint8_t2;
typedef struct {
	_Alignas(4) uint8_t x;
	uint8_t y, z;
} uint8_t3;
typedef struct {
	_Alignas(4) uint8_t x;
	uint8_t y, z, w;
} uint8_t4;
typedef struct {
	_Alignas(2) int8_t x;
	int8_t y;
} int8_t2;
typedef struct {
	_Alignas(4) int8_t x;
	int8_t y, z;
} int8_t3;
typedef struct {
	_Alignas(4) int8_t x;
	int8_t y, z, w;
} int8_t4;
typedef struct {
	_Alignas(2) bool x;
	bool y;
} bool2;
typedef struct {
	_Alignas(4) bool x;
	bool y, z;
} bool3;
typedef struct {
	_Alignas(4) bool x;
	bool y, z, w;
} bool4;
// Matrix
typedef struct {
	_Alignas(8) float2 c0, c1;
} float2x2;
typedef struct {
	_Alignas(16) float3 c0, c1, c2;
} float3x3;
typedef struct {
	_Alignas(16) float4 c0, c1, c2, c3;
} float4x4;

inline float2x2 make_float2x2_0(float2 a, float2 b) {
	return (float2x2){a, b};
}
inline float2x2 make_float2x2_1(float a0, float a1, float a2, float a3) {
	float2x2 f;
	f.c0 = (float2){a0, a1};
	f.c1 = (float2){a2, a3};
	return f;
}
inline float3x3 make_float3x3_0(float3 a, float3 b, float3 c) {
	return (float3x3){a, b, c};
}
inline float3x3 make_float3x3_1(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8) {
	float3x3 f;
	f.c0 = (float3){a0, a1, a2};
	f.c1 = (float3){a3, a4, a5};
	f.c2 = (float3){a6, a7, a8};
	return f;
}
inline float4x4 make_float4x4_0(float4 a, float4 b, float4 c, float4 d) {
	return (float4x4){a, b, c, d};
}
inline float4x4 make_float4x4_1(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8, float a9, float a10, float a11, float a12, float a13, float a14, float a15) {
	float4x4 f;
	f.c0 = (float4){a0, a1, a2, a3};
	f.c1 = (float4){a4, a5, a6, a7};
	f.c2 = (float4){a8, a9, a10, a11};
	f.c3 = (float4){a12, a13, a14, a15};
	return f;
}
inline float2 mul_float2x2_float2(float2x2 m, float2 v) {
	float2 r;
	r.x = v.x * m.c0.x + v.y * m.c1.x;
	r.y = v.x * m.c0.y + v.y * m.c1.y;
	return r;
}
inline float3 mul_float3x3_float3(float3x3 m, float3 v) {
	float3 r;
	r.x = v.x * m.c0.x + v.y * m.c1.x + v.z * m.c2.x;
	r.y = v.x * m.c0.y + v.y * m.c1.y + v.z * m.c2.y;
	r.z = v.x * m.c0.z + v.y * m.c1.z + v.z * m.c2.z;
	return r;
}
inline float4 mul_float4x4_float4(float4x4 m, float4 v) {
	float4 r;
	r.x = v.x * m.c0.x + v.y * m.c1.x + v.z * m.c2.x + v.w * m.c3.x;
	r.y = v.x * m.c0.y + v.y * m.c1.y + v.z * m.c2.y + v.w * m.c3.y;
	r.z = v.x * m.c0.z + v.y * m.c1.z + v.z * m.c2.z + v.w * m.c3.z;
	r.w = v.x * m.c0.w + v.y * m.c1.w + v.z * m.c2.w + v.w * m.c3.w;
	return r;
}
inline float2x2 mul_float2x2_float2x2(float2x2 a, float2x2 b) {
	float2x2 r;
	r.c0 = mul_float2x2_float2(a, b.c0);
	r.c1 = mul_float2x2_float2(a, b.c1);
	return r;
}
inline float3x3 mul_float3x3_float3x3(float3x3 a, float3x3 b) {
	float3x3 r;
	r.c0 = mul_float3x3_float3(a, b.c0);
	r.c1 = mul_float3x3_float3(a, b.c1);
	r.c2 = mul_float3x3_float3(a, b.c2);
	return r;
}
inline float4x4 mul_float4x4_float4x4(float4x4 a, float4x4 b) {
	float4x4 r;
	r.c0 = mul_float4x4_float4(a, b.c0);
	r.c1 = mul_float4x4_float4(a, b.c1);
	r.c2 = mul_float4x4_float4(a, b.c2);
	r.c3 = mul_float4x4_float4(a, b.c3);
	return r;
}
inline float2x2 transpose_float2x2(float2x2 a) {
	float2x2 v;
	v.c0 = (float2){a.c0.x, a.c1.x};
	v.c1 = (float2){a.c0.y, a.c1.y};
	return v;
}
inline float3x3 transpose_float3x3(float3x3 a) {
	float3x3 v;
	v.c0 = (float3){a.c0.x, a.c1.x, a.c2.x};
	v.c1 = (float3){a.c0.y, a.c1.y, a.c2.y};
	v.c2 = (float3){a.c0.z, a.c1.z, a.c2.z};
	return v;
}
inline float4x4 transpose_float4x4(float4x4 a) {
	float4x4 v;
	v.c0 = (float4){a.c0.x, a.c1.x, a.c2.x, a.c3.x};
	v.c1 = (float4){a.c0.y, a.c1.y, a.c2.y, a.c3.y};
	v.c2 = (float4){a.c0.z, a.c1.z, a.c2.z, a.c3.z};
	v.c3 = (float4){a.c0.w, a.c1.w, a.c2.w, a.c3.w};
	return v;
}
#define GET(ELE, value, index) (((ELE*)&(value))[index])
#define ACCESS(ELE, value, index) (((ELE*)(value).v0)[index])
#define DEREF(ELE, value) (*((ELE*)(value).v0))
#define ADDR_OF(value) ((uint64_t) & (value))

inline float determinant_float2x2(float2x2 m) {
	return m.c0.x * m.c1.y - m.c1.x * m.c0.y;
}
inline float determinant_float3x3(float3x3 m) {
	return m.c0.x * (m.c1.y * m.c2.z - m.c2.y * m.c1.z) - m.c1.x * (m.c0.y * m.c2.z - m.c2.y * m.c0.z) + m.c2.x * (m.c0.y * m.c1.z - m.c1.y * m.c0.z);
}
float determinant_float4x4(float4x4 m);
float2x2 inverse_float2x2(float2x2 m);
float3x3 inverse_float3x3(float3x3 m);
float4x4 inverse_float4x4(float4x4 m);
#if defined(__clang__)// Clang
#define LUISA_ASSUME(x) __builtin_assume(x)
#define LUISA_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)// MSVC
#define LUISA_ASSUME(x) __assume(x)
#define LUISA_UNREACHABLE() __assume(false)
#else// GCC
#define LUISA_UNREACHABLE() __builtin_unreachable()
#define LUISA_ASSUME(x) \
	if (!(x)) __builtin_unreachable()
#endif

#define to_string(type, array, size) ((type){(uint64_t)(&(array)), size})

void rtti_call(uint64_t usr_data, uint64_t func_ptr, char const* type_desc, uint64_t type_desc_size, void* ptr);

void memzero(void* ptr, uint64_t size);
void memone(void* ptr, uint64_t size);
int32_t lc_memcmp(uint64_t dst, uint64_t src, uint64_t size);
void lc_memcpy(uint64_t dst, uint64_t src, uint64_t size);
void lc_memmove(uint64_t dst, uint64_t src, uint64_t size);
uint64_t persist_malloc(uint64_t size);
uint64_t temp_malloc(uint64_t size);
void persist_free(uint64_t ptr);
void push_str(char const* ptr, uint64_t size);
void push_bool(bool v);
void push_bool2(bool2 v);
void push_bool3(bool3 v);
void push_bool4(bool4 v);
void push_float(float v);
void push_double(double v);
void push_int32_t(int32_t v);
void push_int8_t(int8_t v);
void push_int16_t(int16_t v);
void push_int64_t(int64_t v);
void push_uint32_t(uint32_t v);
void push_uint8_t(uint8_t v);
void push_uint16_t(uint16_t v);
void push_uint64_t(uint64_t v);

void push_float2(float2 v);
void push_double2(double2 v);
void push_int32_t2(int32_t2 v);
void push_int8_t2(int8_t2 v);
void push_int16_t2(int16_t2 v);
void push_int64_t2(int64_t2 v);
void push_uint32_t2(uint32_t2 v);
void push_uint8_t2(uint8_t2 v);
void push_uint16_t2(uint16_t2 v);
void push_uint64_t2(uint64_t2 v);

void push_float3(float3 v);
void push_double3(double3 v);
void push_int32_t3(int32_t3 v);
void push_int8_t3(int8_t3 v);
void push_int16_t3(int16_t3 v);
void push_int64_t3(int64_t3 v);
void push_uint32_t3(uint32_t3 v);
void push_uint8_t3(uint8_t3 v);
void push_uint16_t3(uint16_t3 v);
void push_uint64_t3(uint64_t3 v);

void push_float4(float4 v);
void push_double4(double4 v);
void push_int32_t4(int32_t4 v);
void push_int8_t4(int8_t4 v);
void push_int16_t4(int16_t4 v);
void push_int64_t4(int64_t4 v);
void push_uint32_t4(uint32_t4 v);
void push_uint8_t4(uint8_t4 v);
void push_uint16_t4(uint16_t4 v);
void push_uint64_t4(uint64_t4 v);

void push_float2x2(uint64_t4 v);
void push_float3x3(uint64_t4 v);
void push_float4x4(uint64_t4 v);
void invoke_print();

void check_access(uint64_t size, uint64_t idx);
