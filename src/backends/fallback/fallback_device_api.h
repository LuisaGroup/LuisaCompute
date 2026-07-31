#pragma once

namespace luisa::compute {
struct MotionInstanceTransform;
}// namespace luisa::compute

#ifndef LUISA_COMPUTE_FALLBACK_DEVICE_LIB
#include <cstdint>
#include <cstddef>
namespace luisa::compute::fallback::api {
#else
using int8_t = signed char;
using int16_t = short;
using int32_t = int;
using int64_t = long long;
using uint8_t = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
using size_t = unsigned long long;
using MotionInstanceTransform = luisa::compute::MotionInstanceTransform;
#endif

extern "C" {

using uint = unsigned int;

struct alignas(8) float2 {
    float x, y;
};

struct alignas(16) float3 {
    float x, y, z;
};

struct alignas(16) float4 {
    float x, y, z, w;
};

struct alignas(16) uint4 {
    uint x, y, z, w;
};

struct alignas(16) int4 {
    int x, y, z, w;
};

struct alignas(8) uint2 {
    uint x, y;
};

struct alignas(16) uint3 {
    uint x, y, z;
};

struct float2x2 {
    float2 cols[2];
};

struct float3x3 {
    float3 cols[3];
};

struct float4x4 {
    float4 cols[4];
};

struct alignas(16) Ray {
    float origin[3];
    float t_min;
    float direction[3];
    float t_max;
};

struct alignas(8) SurfaceHit {
    uint inst;
    uint prim;
    float2 bary;
    float committed_ray_t;
};
static_assert(sizeof(SurfaceHit) == 24u, "SurfaceHit size mismatch");
static_assert(alignof(SurfaceHit) == 8u, "SurfaceHit align mismatch");

struct alignas(8) AABBHit {
    uint inst;
    uint prim;
};
static_assert(sizeof(AABBHit) == 8u, "AABBHit size mismatch");
static_assert(alignof(AABBHit) == 8u, "AABBHit align mismatch");

enum struct HitType : uint {
    Miss = 0,
    Surface = 1,   // triangle or curve
    Procedural = 2,// bounding box (for procedural primitives)
};

struct alignas(8) CommittedHit {
    uint inst;
    uint prim;
    float2 bary;
    HitType hit_type;// HitType
    float committed_ray_t;
};
static_assert(sizeof(CommittedHit) == 24u, "CommittedHit size mismatch");
static_assert(alignof(CommittedHit) == 8u, "CommittedHit align mismatch");

struct alignas(16u) TextureView {
    void *_data;
    uint _width : 16u;
    uint _height : 16u;
    uint _depth : 16u;
    uint _storage : 8u;
    uint _dimension : 4u;
    uint _pixel_stride_shift : 4u;
};

struct alignas(16u) PackedTextureView {
    void *data;
    uint64_t extra;
};

static_assert(sizeof(TextureView) == 16 && sizeof(PackedTextureView) == 16);

struct alignas(16) Texture;

struct alignas(16) BindlessSlot {
    const void *buffer;
    uint64_t _compressed_buffer_size_sampler_2d_sampler_3d;
    const Texture *tex2d;
    const Texture *tex3d;
};

struct alignas(16) BindlessArrayView {
    const BindlessSlot *slots;
    size_t size;
};

/* implementations */

[[nodiscard]] int4 luisa_fallback_texture2d_read_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y) noexcept;
[[nodiscard]] uint4 luisa_fallback_texture2d_read_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y) noexcept;
[[nodiscard]] float4 luisa_fallback_texture2d_read_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y) noexcept;

void luisa_fallback_texture2d_write_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y, float4 value) noexcept;
void luisa_fallback_texture2d_write_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint4 value) noexcept;
void luisa_fallback_texture2d_write_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y, int4 value) noexcept;

[[nodiscard]] int4 luisa_fallback_texture3d_read_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z) noexcept;
[[nodiscard]] uint4 luisa_fallback_texture3d_read_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z) noexcept;
[[nodiscard]] float4 luisa_fallback_texture3d_read_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z) noexcept;

void luisa_fallback_texture3d_write_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z, float4 value) noexcept;
void luisa_fallback_texture3d_write_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z, uint4 value) noexcept;
void luisa_fallback_texture3d_write_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z, int4 value) noexcept;

[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample(const Texture *handle, uint sampler, float u, float v) noexcept;
[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample_level(const Texture *handle, uint sampler, float u, float v, float level) noexcept;
[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample_grad(const Texture *handle, uint sampler, float u, float v, float dudx, float dudy, float dvdx, float dvdy) noexcept;
[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample_grad_level(const Texture *handle, uint sampler, float u, float v, float dudx, float dudy, float dvdx, float dvdy, float level) noexcept;

[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample(const Texture *handle, uint sampler, float u, float v, float w) noexcept;
[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample_level(const Texture *handle, uint sampler, float u, float v, float w, float level) noexcept;
[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample_grad(const Texture *handle, uint sampler, float u, float v, float w, float dudx, float dvdx, float dwdx, float dudy, float dvdy, float dwdy) noexcept;
[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample_grad_level(const Texture *handle, uint sampler, float u, float v, float w, float dudx, float dvdx, float dwdx, float dudy, float dvdy, float dwdy, float level) noexcept;

[[nodiscard]] float4 luisa_fallback_bindless_texture2d_read(const Texture *handle, uint x, uint y) noexcept;
[[nodiscard]] float4 luisa_fallback_bindless_texture2d_read_level(const Texture *handle, uint x, uint y, uint level) noexcept;

[[nodiscard]] float4 luisa_fallback_bindless_texture3d_read(const Texture *handle, uint x, uint y, uint z) noexcept;
[[nodiscard]] float4 luisa_fallback_bindless_texture3d_read_level(const Texture *handle, uint x, uint y, uint z, uint level) noexcept;

struct alignas(16) EmbreeRay;
struct alignas(16) EmbreeHit;
struct alignas(16) EmbreeRayHit;

struct alignas(16) AccelInstance {
    float affine[12];
    uint8_t mask;
    uint8_t opaque : 1;
    uint8_t dirty : 1;
    uint8_t is_curve : 1;
    uint8_t is_motion : 1;
    uint8_t is_srt : 1;
    uint8_t is_matrix : 1;
    uint8_t motion_steps;
    uint user_id;
    MotionInstanceTransform *motion;
};

static_assert(sizeof(AccelInstance) == 64u);

struct alignas(16) AccelView {
    void *embree_scene;
    // The instance vector is populated and may be reallocated by an
    // asynchronous accel-build command. Shader arguments are encoded when
    // they are submitted, before that command necessarily runs, so retain a
    // stable pointer to the backend-owned data pointer rather than snapshotting
    // the current (possibly null or soon stale) vector address.
    AccelInstance **instances;
};

void luisa_fallback_accel_trace_closest(void *handle, EmbreeRayHit *ray_hit) noexcept;
void luisa_fallback_accel_trace_any(void *handle, EmbreeRay *ray) noexcept;

struct alignas(16) LC_RayQueryObject;

[[nodiscard]] size_t luisa_fallback_ray_query_object_size() noexcept;
[[nodiscard]] size_t luisa_fallback_ray_query_object_alignment() noexcept;

struct alignas(16) RayQueryCandidate {
    uint inst;
    uint prim;
    float2 bary;
    float t;
    float pad;
    int committed;
    int terminated;
};

static constexpr uint64_t luisa_fallback_embree_accel_user_data_flags_opaque = 1u << 0u;
static constexpr uint64_t luisa_fallback_embree_accel_user_data_flags_curve = 1u << 1u;

using RayQueryOnSurfaceFunc = void(LC_RayQueryObject *, const void *capture) noexcept;
using RayQueryOnProceduralFunc = void(LC_RayQueryObject *, const void *capture) noexcept;

struct RayQueryIntersectFunctionArguments;
struct RayQueryOccludedFunctionArguments;

void luisa_fallback_ray_query_procedural_intersect_function(const RayQueryIntersectFunctionArguments *args) noexcept;
void luisa_fallback_ray_query_procedural_occluded_function(const RayQueryOccludedFunctionArguments *args) noexcept;

void luisa_fallback_ray_query_pipeline_all(LC_RayQueryObject *query_object, const void *capture, RayQueryOnSurfaceFunc *on_surface, RayQueryOnProceduralFunc *on_procedural) noexcept;
void luisa_fallback_ray_query_pipeline_any(LC_RayQueryObject *query_object, const void *capture, RayQueryOnSurfaceFunc *on_surface, RayQueryOnProceduralFunc *on_procedural) noexcept;
}

#ifndef LUISA_COMPUTE_FALLBACK_DEVICE_LIB
}// namespace luisa::compute::fallback
#endif
