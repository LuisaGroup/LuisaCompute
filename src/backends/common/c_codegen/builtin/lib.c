#include "header.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
inline float4 float4_mul(float4 a, float4 b) {
	return (float4){a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
inline float4 float4_mul_scale(float4 a, float b) {
	return (float4){a.x * b, a.y * b, a.z * b, a.w * b};
}
inline float4 float4_add(float4 a, float4 b) {
	return (float4){a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
inline float4 float4_minus(float4 a, float4 b) {
	return (float4){a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
float determinant_float4x4(float4x4 m) {
	float coef00 = m.c2.z * m.c3.w - m.c3.z * m.c2.w;
	float coef02 = m.c1.z * m.c3.w - m.c3.z * m.c1.w;
	float coef03 = m.c1.z * m.c2.w - m.c2.z * m.c1.w;
	float coef04 = m.c2.y * m.c3.w - m.c3.y * m.c2.w;
	float coef06 = m.c1.y * m.c3.w - m.c3.y * m.c1.w;
	float coef07 = m.c1.y * m.c2.w - m.c2.y * m.c1.w;
	float coef08 = m.c2.y * m.c3.z - m.c3.y * m.c2.z;
	float coef10 = m.c1.y * m.c3.z - m.c3.y * m.c1.z;
	float coef11 = m.c1.y * m.c2.z - m.c2.y * m.c1.z;
	float coef12 = m.c2.x * m.c3.w - m.c3.x * m.c2.w;
	float coef14 = m.c1.x * m.c3.w - m.c3.x * m.c1.w;
	float coef15 = m.c1.x * m.c2.w - m.c2.x * m.c1.w;
	float coef16 = m.c2.x * m.c3.z - m.c3.x * m.c2.z;
	float coef18 = m.c1.x * m.c3.z - m.c3.x * m.c1.z;
	float coef19 = m.c1.x * m.c2.z - m.c2.x * m.c1.z;
	float coef20 = m.c2.x * m.c3.y - m.c3.x * m.c2.y;
	float coef22 = m.c1.x * m.c3.y - m.c3.x * m.c1.y;
	float coef23 = m.c1.x * m.c2.y - m.c2.x * m.c1.y;
	float4 fac0 = (float4){coef00, coef00, coef02, coef03};
	float4 fac1 = (float4){coef04, coef04, coef06, coef07};
	float4 fac2 = (float4){coef08, coef08, coef10, coef11};
	float4 fac3 = (float4){coef12, coef12, coef14, coef15};
	float4 fac4 = (float4){coef16, coef16, coef18, coef19};
	float4 fac5 = (float4){coef20, coef20, coef22, coef23};
	float4 Vec0 = (float4){m.c1.x, m.c0.x, m.c0.x, m.c0.x};
	float4 Vec1 = (float4){m.c1.y, m.c0.y, m.c0.y, m.c0.y};
	float4 Vec2 = (float4){m.c1.z, m.c0.z, m.c0.z, m.c0.z};
	float4 Vec3 = (float4){m.c1.w, m.c0.w, m.c0.w, m.c0.w};
	float4 inv0 = float4_add(float4_minus(float4_mul(Vec1, fac0), float4_mul(Vec2, fac1)), float4_mul(Vec3, fac2));
	float4 inv1 = float4_add(float4_minus(float4_mul(Vec0, fac0), float4_mul(Vec2, fac3)), float4_mul(Vec3, fac4));
	float4 inv2 = float4_add(float4_minus(float4_mul(Vec0, fac1), float4_mul(Vec1, fac3)), float4_mul(Vec3, fac5));
	float4 inv3 = float4_add(float4_minus(float4_mul(Vec0, fac2), float4_mul(Vec1, fac4)), float4_mul(Vec2, fac5));
	float4 sign_a = (float4){+1.0f, -1.0f, +1.0f, -1.0f};
	float4 sign_b = (float4){-1.0f, +1.0f, -1.0f, +1.0f};
	float4 inv_0 = float4_mul(inv0, sign_a);
	float4 inv_1 = float4_mul(inv1, sign_b);
	float4 inv_2 = float4_mul(inv2, sign_a);
	float4 inv_3 = float4_mul(inv3, sign_b);
	float4 dot0 = float4_mul(m.c0, (float4){inv_0.x, inv_1.x, inv_2.x, inv_3.x});
	return dot0.x + dot0.y + dot0.z + dot0.w;
}

typedef struct {
	uint64_t (*persist_malloc)(uint64_t);
	uint64_t (*temp_malloc)(uint64_t);
	void (*persist_free)(uint64_t);
	void (*push_print_str)(char const* ptr, uint64_t len);
	void (*push_print_value)(void* value, uint32_t type);
	void (*print)();
} FuncTable;
static FuncTable func_table;

#ifdef _MSC_VER
#define LUISA_EXPORT_API __declspec(dllexport)
#else
#define LUISA_EXPORT_API __attribute__((visibility("default")))
#endif

LUISA_EXPORT_API void* set_functable_e9f41b3d9cbc4eaea8306f531b1eb997(void* ptr) {
	memcpy(&func_table, ptr, sizeof(FuncTable));
	return &func_table;
}
void memzero(void* ptr, uint64_t size) {
	memset(ptr, 0, size);
}
void memone(void* ptr, uint64_t size) {
	memset(ptr, 1, size);
}
int32_t lc_memcmp(uint64_t dst, uint64_t src, uint64_t size) {
	return memcmp((void const*)dst, (void const*)src, size);
}
void lc_memcpy(uint64_t dst, uint64_t src, uint64_t size) {
	memcpy((void*)dst, (void const*)src, size);
}
void lc_memmove(uint64_t dst, uint64_t src, uint64_t size) {
	memmove((void*)dst, (void const*)src, size);
}
uint64_t persist_malloc(uint64_t size) {
	return (uint64_t)func_table.persist_malloc(size);
}
uint64_t temp_malloc(uint64_t size) {
	return (uint64_t)func_table.temp_malloc(size);
}
void persist_free(uint64_t ptr) {
	func_table.persist_free(ptr);
}

float2x2 inverse_float2x2(float2x2 m) {
	float one_over_determinant = 1.0f / (m.c0.x * m.c1.y - m.c1.x * m.c0.y);
	return (float2x2){(float2){m.c1.y * one_over_determinant,
							   -m.c0.y * one_over_determinant},
					  (float2){-m.c1.x * one_over_determinant,
							   +m.c0.x * one_over_determinant}};
}
float3x3 inverse_float3x3(float3x3 m) {
	float one_over_determinant = 1.0f /
								 (m.c0.x * (m.c1.y * m.c2.z - m.c2.y * m.c1.z) -
								  m.c1.x * (m.c0.y * m.c2.z - m.c2.y * m.c0.z) +
								  m.c2.x * (m.c0.y * m.c1.z - m.c1.y * m.c0.z));
	return (float3x3){
		(float3){(m.c1.y * m.c2.z - m.c2.y * m.c1.z) * one_over_determinant,
				 (m.c2.y * m.c0.z - m.c0.y * m.c2.z) * one_over_determinant,
				 (m.c0.y * m.c1.z - m.c1.y * m.c0.z) * one_over_determinant},
		(float3){(m.c2.x * m.c1.z - m.c1.x * m.c2.z) * one_over_determinant,
				 (m.c0.x * m.c2.z - m.c2.x * m.c0.z) * one_over_determinant,
				 (m.c1.x * m.c0.z - m.c0.x * m.c1.z) * one_over_determinant},
		(float3){(m.c1.x * m.c2.y - m.c2.x * m.c1.y) * one_over_determinant,
				 (m.c2.x * m.c0.y - m.c0.x * m.c2.y) * one_over_determinant,
				 (m.c0.x * m.c1.y - m.c1.x * m.c0.y) * one_over_determinant}};
}
float4x4 inverse_float4x4(float4x4 m) {
	float coef00 = m.c2.z * m.c3.w - m.c3.z * m.c2.w;
	float coef02 = m.c1.z * m.c3.w - m.c3.z * m.c1.w;
	float coef03 = m.c1.z * m.c2.w - m.c2.z * m.c1.w;
	float coef04 = m.c2.y * m.c3.w - m.c3.y * m.c2.w;
	float coef06 = m.c1.y * m.c3.w - m.c3.y * m.c1.w;
	float coef07 = m.c1.y * m.c2.w - m.c2.y * m.c1.w;
	float coef08 = m.c2.y * m.c3.z - m.c3.y * m.c2.z;
	float coef10 = m.c1.y * m.c3.z - m.c3.y * m.c1.z;
	float coef11 = m.c1.y * m.c2.z - m.c2.y * m.c1.z;
	float coef12 = m.c2.x * m.c3.w - m.c3.x * m.c2.w;
	float coef14 = m.c1.x * m.c3.w - m.c3.x * m.c1.w;
	float coef15 = m.c1.x * m.c2.w - m.c2.x * m.c1.w;
	float coef16 = m.c2.x * m.c3.z - m.c3.x * m.c2.z;
	float coef18 = m.c1.x * m.c3.z - m.c3.x * m.c1.z;
	float coef19 = m.c1.x * m.c2.z - m.c2.x * m.c1.z;
	float coef20 = m.c2.x * m.c3.y - m.c3.x * m.c2.y;
	float coef22 = m.c1.x * m.c3.y - m.c3.x * m.c1.y;
	float coef23 = m.c1.x * m.c2.y - m.c2.x * m.c1.y;
	float4 fac0 = (float4){coef00, coef00, coef02, coef03};
	float4 fac1 = (float4){coef04, coef04, coef06, coef07};
	float4 fac2 = (float4){coef08, coef08, coef10, coef11};
	float4 fac3 = (float4){coef12, coef12, coef14, coef15};
	float4 fac4 = (float4){coef16, coef16, coef18, coef19};
	float4 fac5 = (float4){coef20, coef20, coef22, coef23};
	float4 Vec0 = (float4){m.c1.x, m.c0.x, m.c0.x, m.c0.x};
	float4 Vec1 = (float4){m.c1.y, m.c0.y, m.c0.y, m.c0.y};
	float4 Vec2 = (float4){m.c1.z, m.c0.z, m.c0.z, m.c0.z};
	float4 Vec3 = (float4){m.c1.w, m.c0.w, m.c0.w, m.c0.w};
	float4 inv0 = float4_add(float4_minus(float4_mul(Vec1, fac0), float4_mul(Vec2, fac1)), float4_mul(Vec3, fac2));
	float4 inv1 = float4_add(float4_minus(float4_mul(Vec0, fac0), float4_mul(Vec2, fac3)), float4_mul(Vec3, fac4));
	float4 inv2 = float4_add(float4_minus(float4_mul(Vec0, fac1), float4_mul(Vec1, fac3)), float4_mul(Vec3, fac5));
	float4 inv3 = float4_add(float4_minus(float4_mul(Vec0, fac2), float4_mul(Vec1, fac4)), float4_mul(Vec2, fac5));
	float4 sign_a = (float4){+1.0f, -1.0f, +1.0f, -1.0f};
	float4 sign_b = (float4){-1.0f, +1.0f, -1.0f, +1.0f};
	float4 inv_0 = float4_mul(inv0, sign_a);
	float4 inv_1 = float4_mul(inv1, sign_b);
	float4 inv_2 = float4_mul(inv2, sign_a);
	float4 inv_3 = float4_mul(inv3, sign_b);
	float4 dot0 = float4_mul(m.c0, (float4){inv_0.x, inv_1.x, inv_2.x, inv_3.x});
	float dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
	float one_over_determinant = 1.0f / dot1;
	return (float4x4){float4_mul_scale(inv_0, one_over_determinant),
					  float4_mul_scale(inv_1, one_over_determinant),
					  float4_mul_scale(inv_2, one_over_determinant),
					  float4_mul_scale(inv_3, one_over_determinant)};
}
void push_str(char const* ptr, uint64_t size) { func_table.push_print_str(ptr, size); }

void push_bool(bool v) { func_table.push_print_value(&v, 0); }
void push_bool2(bool2 v) { func_table.push_print_value(&v, 1); }
void push_bool3(bool3 v) { func_table.push_print_value(&v, 2); }
void push_bool4(bool4 v) { func_table.push_print_value(&v, 3); }
void push_float(float v) { func_table.push_print_value(&v, 4); }
void push_double(double v) { func_table.push_print_value(&v, 5); }
void push_int32_t(int32_t v) { func_table.push_print_value(&v, 6); }
void push_int8_t(int8_t v) { func_table.push_print_value(&v, 7); }
void push_int16_t(int16_t v) { func_table.push_print_value(&v, 8); }
void push_int64_t(int64_t v) { func_table.push_print_value(&v, 9); }
void push_uint32_t(uint32_t v) { func_table.push_print_value(&v, 10); }
void push_uint8_t(uint8_t v) { func_table.push_print_value(&v, 11); }
void push_uint16_t(uint16_t v) { func_table.push_print_value(&v, 12); }
void push_uint64_t(uint64_t v) { func_table.push_print_value(&v, 13); }

void push_float2(float2 v) { func_table.push_print_value(&v, 14); }
void push_double2(double2 v) { func_table.push_print_value(&v, 15); }
void push_int32_t2(int32_t2 v) { func_table.push_print_value(&v, 16); }
void push_int8_t2(int8_t2 v) { func_table.push_print_value(&v, 17); }
void push_int16_t2(int16_t2 v) { func_table.push_print_value(&v, 18); }
void push_int64_t2(int64_t2 v) { func_table.push_print_value(&v, 19); }
void push_uint32_t2(uint32_t2 v) { func_table.push_print_value(&v, 20); }
void push_uint8_t2(uint8_t2 v) { func_table.push_print_value(&v, 21); }
void push_uint16_t2(uint16_t2 v) { func_table.push_print_value(&v, 22); }
void push_uint64_t2(uint64_t2 v) { func_table.push_print_value(&v, 23); }

void push_float3(float3 v) { func_table.push_print_value(&v, 24); }
void push_double3(double3 v) { func_table.push_print_value(&v, 25); }
void push_int32_t3(int32_t3 v) { func_table.push_print_value(&v, 26); }
void push_int8_t3(int8_t3 v) { func_table.push_print_value(&v, 27); }
void push_int16_t3(int16_t3 v) { func_table.push_print_value(&v, 28); }
void push_int64_t3(int64_t3 v) { func_table.push_print_value(&v, 29); }
void push_uint32_t3(uint32_t3 v) { func_table.push_print_value(&v, 30); }
void push_uint8_t3(uint8_t3 v) { func_table.push_print_value(&v, 31); }
void push_uint16_t3(uint16_t3 v) { func_table.push_print_value(&v, 32); }
void push_uint64_t3(uint64_t3 v) { func_table.push_print_value(&v, 33); }

void push_float4(float4 v) { func_table.push_print_value(&v, 34); }
void push_double4(double4 v) { func_table.push_print_value(&v, 35); }
void push_int32_t4(int32_t4 v) { func_table.push_print_value(&v, 36); }
void push_int8_t4(int8_t4 v) { func_table.push_print_value(&v, 37); }
void push_int16_t4(int16_t4 v) { func_table.push_print_value(&v, 38); }
void push_int64_t4(int64_t4 v) { func_table.push_print_value(&v, 39); }
void push_uint32_t4(uint32_t4 v) { func_table.push_print_value(&v, 40); }
void push_uint8_t4(uint8_t4 v) { func_table.push_print_value(&v, 41); }
void push_uint16_t4(uint16_t4 v) { func_table.push_print_value(&v, 42); }
void push_uint64_t4(uint64_t4 v) { func_table.push_print_value(&v, 43); }

void push_float2x2(uint64_t4 v) { func_table.push_print_value(&v, 44); }
void push_float3x3(uint64_t4 v) { func_table.push_print_value(&v, 45); }
void push_float4x4(uint64_t4 v) { func_table.push_print_value(&v, 46); }

void invoke_print() {
	func_table.print();
}
typedef struct {
	char const* ptr;
	uint64_t len;
} CharBuffer;
void rtti_call(uint64_t usr_data, uint64_t func_ptr, char const* type_desc, uint64_t type_desc_size, void* ptr) {
	CharBuffer c;
	c.ptr = type_desc;
	c.len = type_desc_size;
	((void (*)(uint64_t, CharBuffer, uint64_t))(func_ptr))(usr_data, c, (uint64_t)ptr);
}
void check_access(uint64_t size, uint64_t idx) {
	if (size <= idx) [[unlikely]] {
		printf("Index %llu out of range [0, %llu).\n", idx, size);
		exit(1);
	}
}