#include "fallback_texture.h"
#include "fallback_texture_bc.h"
#include "fallback_accel.h"

namespace luisa::compute::fallback::api {

void luisa_bc7_read(const FallbackTextureView *tex, uint x, uint y, float4 &out) noexcept {
    auto block_pos = make_uint2(x / 4, y / 4);
    auto block_per_row = tex->size2d().x / 4;
    const bc::BC7Block *bc_block = reinterpret_cast<const bc::BC7Block *>(tex->data()) +
                                   (block_pos.x + block_pos.y * block_per_row);
    bc_block->Decode(x % 4, y % 4, reinterpret_cast<float *>(&out));
}

void luisa_bc6h_read(const FallbackTextureView *tex, int x, int y, float4 &out) noexcept {
    auto block_pos = make_uint2(x / 4, y / 4);
    auto block_per_row = tex->size2d().x / 4;
    const bc::BC6HBlock *bc_block = reinterpret_cast<const bc::BC6HBlock *>(tex->data()) + (block_pos.x + block_pos.y * block_per_row);
    bc_block->Decode(false, x % 4, y % 4, reinterpret_cast<float *>(&out));
}

[[nodiscard]] int4 luisa_fallback_texture2d_read_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 2D int texture");
    auto v = tex->read2d<int>(make_uint2(x, y));
    return {v.x, v.y, v.z, v.w};
}

[[nodiscard]] uint4 luisa_fallback_texture2d_read_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 2D uint texture");
    auto v = tex->read2d<uint>(make_uint2(x, y));
    return {v.x, v.y, v.z, v.w};
}

[[nodiscard]] float4 luisa_fallback_texture2d_read_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    switch (tex->storage()) {
        case PixelStorage::BC7: {
            float4 out{};
            luisa_bc7_read(tex, x, y, out);
            return out;
        }
        case PixelStorage::BC6: {
            float4 out{};
            luisa_bc6h_read(tex, x, y, out);
            return out;
        }
        default: {
            auto v = tex->read2d<float>(make_uint2(x, y));
            return {v.x, v.y, v.z, v.w};
        }
    }
}

void luisa_fallback_texture2d_write_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y, float4 value) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 2D texture write");
    auto v = luisa::make_float4(value.x, value.y, value.z, value.w);
    tex->write2d<float>(make_uint2(x, y), v);
}

void luisa_fallback_texture2d_write_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint4 value) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 2D texture write");
    auto v = luisa::make_uint4(value.x, value.y, value.z, value.w);
    tex->write2d<uint>(make_uint2(x, y), v);
}

void luisa_fallback_texture2d_write_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y, int4 value) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 2D texture write");
    auto v = luisa::make_int4(value.x, value.y, value.z, value.w);
    tex->write2d<int>(make_uint2(x, y), v);
}

[[nodiscard]] int4 luisa_fallback_texture3d_read_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = tex->read3d<int>(make_uint3(x, y, z));
    return {v.x, v.y, v.z, v.w};
}

[[nodiscard]] uint4 luisa_fallback_texture3d_read_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = tex->read3d<uint>(make_uint3(x, y, z));
    return {v.x, v.y, v.z, v.w};
}

[[nodiscard]] float4 luisa_fallback_texture3d_read_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = tex->read3d<float>(make_uint3(x, y, z));
    return {v.x, v.y, v.z, v.w};
}

void luisa_fallback_texture3d_write_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z, float4 value) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = luisa::make_float4(value.x, value.y, value.z, value.w);
    tex->write3d<float>(make_uint3(x, y, z), v);
}

void luisa_fallback_texture3d_write_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z, uint4 value) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = luisa::make_uint4(value.x, value.y, value.z, value.w);
    tex->write3d<uint>(make_uint3(x, y, z), v);
}

void luisa_fallback_texture3d_write_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z, int4 value) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = luisa::make_int4(value.x, value.y, value.z, value.w);
    tex->write3d<int>(make_uint3(x, y, z), v);
}

template<typename T>
[[nodiscard]] inline auto texture_coord_point(Sampler::Address address, T uv, T s) noexcept {
    switch (address) {
        case Sampler::Address::EDGE: return luisa::clamp(uv, 0.0f, one_minus_epsilon) * s;
        case Sampler::Address::REPEAT: return luisa::fract(uv) * s;
        case Sampler::Address::MIRROR: {
            uv = luisa::fmod(luisa::abs(uv), T{2.f});
            uv = select(2.f - uv, uv, uv < T{1.f});
            return luisa::min(uv, one_minus_epsilon) * s;
        }
        case Sampler::Address::ZERO: return luisa::select(uv * s, T{65536.f}, uv < 0.f || uv >= 1.f);
    }
    return T{65536.f};
}

[[nodiscard]] inline auto texture_sample_point(FallbackTextureView view, Sampler::Address address, luisa::float2 uv) noexcept {
    auto size = make_float2(view.size2d());
    auto p = texture_coord_point(address, uv, size);
    auto c = make_uint2(p);
    auto handle = reinterpret_cast<const PackedTextureView *>(&view);
    return bit_cast<luisa::float4>(luisa_fallback_texture2d_read_float(handle->data, handle->extra, c.x, c.y));
}

[[nodiscard]] inline auto texture_coord_linear(Sampler::Address address, luisa::float2 uv, luisa::float2 s) noexcept {
    auto inv_s = 1.f / s;
    auto c_min = texture_coord_point(address, uv - .5f * inv_s, s);
    auto c_max = texture_coord_point(address, uv + .5f * inv_s, s);
    return std::make_pair<luisa::float2, luisa::float2>(luisa::min(c_min, c_max), luisa::max(c_min, c_max));
}

[[nodiscard]] inline auto texture_sample_linear(FallbackTextureView view, Sampler::Address address, luisa::float2 uv) noexcept {
    auto size = make_float2(view.size2d());
    auto [st_min, st_max] = texture_coord_linear(address, uv, size);
    auto t = luisa::fract(st_max);
    auto c0 = make_uint2(st_min);
    auto c1 = make_uint2(st_max);
    auto handle = reinterpret_cast<const PackedTextureView *>(&view);
    auto v00 = bit_cast<luisa::float4>(luisa_fallback_texture2d_read_float(handle->data, handle->extra, c0.x, c0.y));
    auto v01 = bit_cast<luisa::float4>(luisa_fallback_texture2d_read_float(handle->data, handle->extra, c1.x, c0.y));
    auto v10 = bit_cast<luisa::float4>(luisa_fallback_texture2d_read_float(handle->data, handle->extra, c0.x, c1.y));
    auto v11 = bit_cast<luisa::float4>(luisa_fallback_texture2d_read_float(handle->data, handle->extra, c1.x, c1.y));
    return luisa::lerp(luisa::lerp(v00, v01, t.x), luisa::lerp(v10, v11, t.x), t.y);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample(const Texture *handle, uint sampler, float u, float v) noexcept {
    auto tex = reinterpret_cast<const FallbackTexture *>(handle);
    auto s = Sampler::decode(sampler);
    auto view = tex->view(0);
    auto result = s.filter() == Sampler::Filter::POINT ?
                      texture_sample_point(view, s.address(), make_float2(u, v)) :
                      texture_sample_linear(view, s.address(), make_float2(u, v));
    return {result.x, result.y, result.z, result.w};
}

[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample_level(const Texture *handle, uint sampler, float u, float v, float level) noexcept {
    auto tex = reinterpret_cast<const FallbackTexture *>(handle);
    auto s = Sampler::decode(sampler);
    auto filter = s.filter();
    if (level <= 0.f || tex->mip_levels() == 0u || filter == Sampler::Filter::POINT) {
        return luisa_fallback_bindless_texture2d_sample(handle, sampler, u, v);
    }
    auto level0 = std::min(static_cast<uint32_t>(level), tex->mip_levels() - 1u);
    auto view0 = tex->view(level0);
    auto v0 = texture_sample_linear(view0, s.address(), make_float2(u, v));
    if (level0 == tex->mip_levels() - 1u || filter == Sampler::Filter::LINEAR_POINT) {
        return {v0.x, v0.y, v0.z, v0.w};
    }
    auto view1 = tex->view(level0 + 1);
    auto v1 = texture_sample_linear(view1, s.address(), make_float2(u, v));
    auto result = luisa::lerp(v0, v1, luisa::fract(level));
    return {result.x, result.y, result.z, result.w};
}

// TODO: implement
[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample_grad(const Texture *handle, uint sampler, float u, float v, float dudx, float dudy, float dvdx, float dvdy) noexcept {
    return luisa_fallback_bindless_texture2d_sample(handle, sampler, u, v);
}

// TODO: implement
[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample_grad_level(const Texture *handle, uint sampler, float u, float v, float dudx, float dudy, float dvdx, float dvdy, float level) noexcept {
    return luisa_fallback_bindless_texture2d_sample_level(handle, sampler, u, v, level);
}

[[nodiscard]] inline auto texture_sample_point(FallbackTextureView view, Sampler::Address address, luisa::float3 uv) noexcept {
    auto size = make_float3(view.size3d());
    auto c = make_uint3(texture_coord_point(address, uv, size));
    auto handle = reinterpret_cast<const PackedTextureView *>(&view);
    return bit_cast<luisa::float4>(luisa_fallback_texture3d_read_float(handle->data, handle->extra, c.x, c.y, c.z));
}

[[nodiscard]] inline auto texture_coord_linear(Sampler::Address address, luisa::float3 uv, luisa::float3 size) noexcept {
    auto s = make_float3(size);
    auto inv_s = 1.f / s;
    auto c_min = texture_coord_point(address, uv - .5f * inv_s, s);
    auto c_max = texture_coord_point(address, uv + .5f * inv_s, s);
    return std::make_pair(luisa::min(c_min, c_max), luisa::max(c_min, c_max));
}

[[nodiscard]] inline auto texture_sample_linear(FallbackTextureView view, Sampler::Address address, luisa::float3 uvw) noexcept {
    auto size = make_float3(view.size3d());
    auto [st_min, st_max] = texture_coord_linear(address, uvw, size);
    auto t = luisa::fract(st_max);
    auto c0 = make_uint3(st_min);
    auto c1 = make_uint3(st_max);
    auto handle = reinterpret_cast<const PackedTextureView *>(&view);
    auto v000 = bit_cast<luisa::float4>(luisa_fallback_texture3d_read_float(handle->data, handle->extra, c0.x, c0.y, c0.z));
    auto v001 = bit_cast<luisa::float4>(luisa_fallback_texture3d_read_float(handle->data, handle->extra, c1.x, c0.y, c0.z));
    auto v010 = bit_cast<luisa::float4>(luisa_fallback_texture3d_read_float(handle->data, handle->extra, c0.x, c1.y, c0.z));
    auto v011 = bit_cast<luisa::float4>(luisa_fallback_texture3d_read_float(handle->data, handle->extra, c1.x, c1.y, c0.z));
    auto v100 = bit_cast<luisa::float4>(luisa_fallback_texture3d_read_float(handle->data, handle->extra, c0.x, c0.y, c1.z));
    auto v101 = bit_cast<luisa::float4>(luisa_fallback_texture3d_read_float(handle->data, handle->extra, c1.x, c0.y, c1.z));
    auto v110 = bit_cast<luisa::float4>(luisa_fallback_texture3d_read_float(handle->data, handle->extra, c0.x, c1.y, c1.z));
    auto v111 = bit_cast<luisa::float4>(luisa_fallback_texture3d_read_float(handle->data, handle->extra, c1.x, c1.y, c1.z));
    return luisa::lerp(luisa::lerp(luisa::lerp(v000, v001, t.x),
                                   luisa::lerp(v010, v011, t.x), t.y),
                       luisa::lerp(luisa::lerp(v100, v101, t.x),
                                   luisa::lerp(v110, v111, t.x), t.y),
                       t.z);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample(const Texture *handle, uint sampler, float u, float v, float w) noexcept {
    auto tex = reinterpret_cast<const FallbackTexture *>(handle);
    auto s = Sampler::decode(sampler);
    auto view = tex->view(0);
    auto result = s.filter() == Sampler::Filter::POINT ?
                      texture_sample_point(view, s.address(), make_float3(u, v, w)) :
                      texture_sample_linear(view, s.address(), make_float3(u, v, w));
    return {result.x, result.y, result.z, result.w};
}

[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample_level(const Texture *handle, uint sampler, float u, float v, float w, float level) noexcept {
    auto tex = reinterpret_cast<const FallbackTexture *>(handle);
    auto s = Sampler::decode(sampler);
    auto filter = s.filter();
    if (level <= 0.f || tex->mip_levels() == 0u || filter == Sampler::Filter::POINT) {
        return luisa_fallback_bindless_texture3d_sample(handle, sampler, u, v, w);
    }
    auto level0 = std::min(static_cast<uint32_t>(level), tex->mip_levels() - 1u);
    auto view0 = tex->view(level0);
    auto v0 = texture_sample_linear(view0, s.address(), make_float3(u, v, w));
    if (level0 == tex->mip_levels() - 1u || filter == Sampler::Filter::LINEAR_POINT) {
        return {v0.x, v0.y, v0.z, v0.w};
    }
    auto view1 = tex->view(level0 + 1);
    auto v1 = texture_sample_linear(view1, s.address(), make_float3(u, v, w));
    auto result = luisa::lerp(v0, v1, luisa::fract(level));
    return {result.x, result.y, result.z, result.w};
}

// TODO: implement
[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample_grad(const Texture *handle, uint sampler, float u, float v, float w, float dudx, float dvdx, float dwdx, float dudy, float dvdy, float dwdy) noexcept {
    return luisa_fallback_bindless_texture3d_sample(handle, sampler, u, v, w);
}

// TODO: implement
[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample_grad_level(const Texture *handle, uint sampler, float u, float v, float w, float dudx, float dvdx, float dwdx, float dudy, float dvdy, float dwdy, float level) noexcept {
    return luisa_fallback_bindless_texture3d_sample_level(handle, sampler, u, v, w, level);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture2d_read_level(const Texture *handle, uint x, uint y, uint level) noexcept {
    auto texture = reinterpret_cast<const FallbackTexture *>(handle);
    auto levels = texture->mip_levels();
    LUISA_ASSUME(levels > 0);
    if (level >= levels) { return {}; }
    auto view = texture->view(level);
    auto packed = reinterpret_cast<const PackedTextureView *>(&view);
    return luisa_fallback_texture2d_read_float(packed->data, packed->extra, x, y);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture2d_read(const Texture *handle, uint x, uint y) noexcept {
    return luisa_fallback_bindless_texture2d_read_level(handle, x, y, 0u);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture3d_read_level(const Texture *handle, uint x, uint y, uint z, uint level) noexcept {
    auto texture = reinterpret_cast<const FallbackTexture *>(handle);
    auto levels = texture->mip_levels();
    LUISA_ASSUME(levels > 0);
    if (level >= levels) { return {}; }
    auto view = texture->view(level);
    auto packed = reinterpret_cast<const PackedTextureView *>(&view);
    return luisa_fallback_texture3d_read_float(packed->data, packed->extra, x, y, z);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture3d_read(const Texture *handle, uint x, uint y, uint z) noexcept {
    return luisa_fallback_bindless_texture3d_read_level(handle, x, y, z, 0u);
}

struct alignas(16) RayQueryObject {
    AccelView accel;
    RayQueryCandidate candidate;
    RTCRayHit ray_hit;
    Ray world_ray;
};

struct alignas(16) RayQueryContext {
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    RTCIntersectContext rtc_ctx;
#else
    RTCRayQueryContext rtc_ctx;
#endif
    RayQueryObject *q;
};

struct RayQueryContextEx {
    RayQueryContext base;
    const void *capture;
    float current_t;
    float original_tnear;
    RayQueryOnSurfaceFunc *on_surface;
    RayQueryOnProceduralFunc *on_procedural;
};

// possibly a curve hit, we should modify hit.v to -1
static void ray_trace_fix_hit_for_curves(RTCScene scene, RTCRayHit *rh) noexcept {
    if (auto inst = rh->hit.instID[0]; inst != RTC_INVALID_GEOMETRY_ID && rh->hit.v == 0.f) {
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
        auto flags = reinterpret_cast<uint64_t>(rtcGetGeometryUserData(rtcGetGeometry(scene, inst)));
#else
        auto flags = reinterpret_cast<uint64_t>(rtcGetGeometryUserDataFromScene(scene, inst));
#endif
        if (flags & luisa_fallback_embree_accel_user_data_flags_curve) {
            rh->hit.v = -1.f;
        }
    }
}

void luisa_fallback_accel_trace_closest(void *handle, EmbreeRayHit *ray_hit) noexcept {

    auto scene = static_cast<RTCScene>(handle);
    auto rh = reinterpret_cast<RTCRayHit *>(ray_hit);

    RayQueryContext ctx;
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.rtc_ctx);
#else
    rtcInitRayQueryContext(&ctx.rtc_ctx);
#endif
    ctx.q = nullptr;

#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcIntersect1(scene, &ctx.rtc_ctx, rh);
#else
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.context = &ctx.rtc_ctx;
    rtcIntersect1(scene, rh, &args);
#endif
    ray_trace_fix_hit_for_curves(scene, rh);
}

void luisa_fallback_accel_trace_any(void *handle, EmbreeRay *ray) noexcept {

    auto scene = static_cast<RTCScene>(handle);
    auto r = reinterpret_cast<RTCRay *>(ray);

    RayQueryContext ctx;
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.rtc_ctx);
#else
    rtcInitRayQueryContext(&ctx.rtc_ctx);
#endif
    ctx.q = nullptr;

#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcOccluded1(scene, &ctx.rtc_ctx, r);
#else
    RTCOccludedArguments args;
    rtcInitOccludedArguments(&args);
    args.context = &ctx.rtc_ctx;
    rtcOccluded1(scene, r, &args);
#endif
}

size_t luisa_fallback_ray_query_object_size() noexcept {
    return sizeof(RayQueryObject);
}

size_t luisa_fallback_ray_query_object_alignment() noexcept {
    return alignof(RayQueryObject);
}

static void ray_query_update_current_t(RayQueryContextEx *ctx, float new_t) noexcept {
    ctx->current_t = std::min(ctx->current_t, new_t);
    ctx->base.q->world_ray.t_max = ctx->current_t;
}

// Luisa ray queries use a closed lower bound, as do the hardware backends.
// Embree's triangle intersector may reject an exactly reconstructed tnear
// hit before its filter callback runs. Widen only the internal traversal
// interval by one representable value, then reject candidates below the
// original bound in the filter. The smallest-normal fallback keeps the
// widening effective when FTZ/DAZ would erase a subnormal nextafter result.
[[nodiscard]] static float ray_query_embree_tnear(float tnear) noexcept {
    if (!std::isfinite(tnear)) { return tnear; }
    auto widened = std::nextafter(
        tnear, -std::numeric_limits<float>::infinity());
    if (std::abs(widened) <
        std::numeric_limits<float>::min()) {
        widened = -std::numeric_limits<float>::min();
    }
    return widened;
}

[[nodiscard]] static bool ray_query_candidate_in_range(
    const RayQueryContextEx *ctx, float t) noexcept {
    return t >= ctx->original_tnear;
}

static void ray_query_decode_surface_candidate(RayQueryCandidate *out, const RTCRay *ray, const RTCHit *hit) noexcept {
    out->inst = hit->instID[0];
    out->prim = hit->primID;
    out->bary = {hit->u, hit->v};
    out->t = ray->tfar;
    out->committed = false;
    out->terminated = false;
}

static void ray_query_decode_procedural_candidate(RayQueryCandidate *out, const RayQueryContext *ctx, const RTCRay *ray, uint prim) noexcept {
    out->inst = ctx->rtc_ctx.instID[0];
    out->prim = prim;
    out->committed = false;
    out->terminated = false;
}

[[nodiscard]] static const AccelInstance &ray_query_surface_instance(
    const RayQueryContextEx *ctx, const RTCHit *hit) noexcept {
    auto inst = hit->instID[0];
    LUISA_DEBUG_ASSERT(
        inst != RTC_INVALID_GEOMETRY_ID,
        "Ray query surface hit is missing its top-level instance ID.");
    return ctx->base.q->accel.instances[inst];
}

void luisa_fallback_ray_query_terminate_ray(RTCRay *ray) noexcept {
    ray->tfar = -std::numeric_limits<float>::infinity();
}

void luisa_fallback_ray_query_procedural_intersect_function(const RayQueryIntersectFunctionArguments *args_in) noexcept {
    auto args = reinterpret_cast<const RTCIntersectFunctionNArguments *>(args_in);
    LUISA_DEBUG_ASSERT(args->N == 1u, "Only single ray is support.");
    LUISA_DEBUG_ASSERT(args->valid[0] == -1, "Only valid ray is support.");
    args->valid[0] = 0;
    auto ctx = reinterpret_cast<RayQueryContextEx *>(args->context);
    if (auto q = ctx->base.q) {
        if (auto on_procedural = ctx->on_procedural) {
            auto candidate = &q->candidate;
            auto ray_hit = reinterpret_cast<RTCRayHit *>(args->rayhit);
            ray_query_decode_procedural_candidate(candidate, &ctx->base, &ray_hit->ray, args->primID);
            const auto embree_tnear = ray_hit->ray.tnear;
            ray_hit->ray.tnear = ctx->original_tnear;
            on_procedural(reinterpret_cast<LC_RayQueryObject *>(q), ctx->capture);
            ray_hit->ray.tnear = embree_tnear;
            if (candidate->committed &&
                ray_query_candidate_in_range(ctx, candidate->t) &&
                candidate->t <= ray_hit->ray.tfar) {
                args->valid[0] = -1;
                ray_hit->ray.tfar = candidate->t;
                ray_query_update_current_t(ctx, candidate->t);
                ray_hit->hit = {
                    .Ng_x = 0.f,
                    .Ng_y = 0.f,
                    .Ng_z = 0.f,
                    .u = -1.f,
                    .v = -1.f,
                    .primID = args->primID,
                    .geomID = args->geomID,
                    .instID = {candidate->inst},
                };
            }
            if (candidate->terminated) {
                luisa_fallback_ray_query_terminate_ray(&ray_hit->ray);
            }
        }
    }
}

void luisa_fallback_ray_query_procedural_occluded_function(const RayQueryOccludedFunctionArguments *args_in) noexcept {
    auto args = reinterpret_cast<const RTCOccludedFunctionNArguments *>(args_in);
    LUISA_DEBUG_ASSERT(args->N == 1u, "Only single ray is support.");
    LUISA_DEBUG_ASSERT(args->valid[0] == -1, "Only valid ray is support.");
    args->valid[0] = 0;
    auto ctx = reinterpret_cast<RayQueryContextEx *>(args->context);
    if (auto q = ctx->base.q) {
        if (auto on_procedural = ctx->on_procedural) {
            auto candidate = &q->candidate;
            auto ray = reinterpret_cast<RTCRay *>(args->ray);
            ray_query_decode_procedural_candidate(candidate, &ctx->base, ray, args->primID);
            const auto embree_tnear = ray->tnear;
            ray->tnear = ctx->original_tnear;
            on_procedural(reinterpret_cast<LC_RayQueryObject *>(q), ctx->capture);
            ray->tnear = embree_tnear;
            if (candidate->committed &&
                ray_query_candidate_in_range(ctx, candidate->t) &&
                candidate->t <= ray->tfar) {
                args->valid[0] = -1;
                ray->tfar = candidate->t;
                q->ray_hit.hit = {
                    .Ng_z = candidate->t,
                    .u = -1.f,
                    .v = -1.f,
                    .primID = candidate->prim,
                    .instID = {candidate->inst},
                };
            }
            if (candidate->terminated) {
                luisa_fallback_ray_query_terminate_ray(ray);
            }
        }
    }
}

static void luisa_fallback_ray_query_surface_intersect_filter_function(const RTCFilterFunctionNArguments *args) noexcept {
    LUISA_DEBUG_ASSERT(args->N == 1u, "Only single ray is support.");
    LUISA_DEBUG_ASSERT(args->valid[0] == -1, "Only valid ray is support.");
    auto ctx = reinterpret_cast<RayQueryContextEx *>(args->context);
    auto ray = reinterpret_cast<RTCRay *>(args->ray);
    auto hit = reinterpret_cast<RTCHit *>(args->hit);
    auto &&instance = ray_query_surface_instance(ctx, hit);
    if (instance.is_curve) {// curve
        hit->v = -1.f;
    }
    if (!ray_query_candidate_in_range(ctx, ray->tfar)) {
        args->valid[0] = 0;
        return;
    }
    if (instance.opaque) {// opaque, always commit
        ray_query_update_current_t(ctx, ray->tfar);
    } else if (auto on_surface = ctx->on_surface) {
        auto q = ctx->base.q;
        auto candidate = &q->candidate;
        ray_query_decode_surface_candidate(candidate, ray, hit);
        const auto embree_tnear = ray->tnear;
        ray->tnear = ctx->original_tnear;
        on_surface(reinterpret_cast<LC_RayQueryObject *>(q), ctx->capture);
        ray->tnear = embree_tnear;
        if (candidate->committed) {
            ray_query_update_current_t(ctx, ray->tfar);
        } else {
            args->valid[0] = 0;
        }
        if (candidate->terminated) {
            luisa_fallback_ray_query_terminate_ray(ray);
        }
    } else {
        args->valid[0] = 0;
    }
}

static void luisa_fallback_ray_query_surface_occluded_filter_function(const RTCFilterFunctionNArguments *args) noexcept {
    LUISA_DEBUG_ASSERT(args->N == 1u, "Only single ray is support.");
    LUISA_DEBUG_ASSERT(args->valid[0] == -1, "Only valid ray is support.");
    static constexpr auto record_hit_data = [](RayQueryObject *q, const RTCRay *ray, const RTCHit *hit) noexcept {
        q->ray_hit.hit.Ng_x = 0.f;
        q->ray_hit.hit.Ng_y = 0.f;
        q->ray_hit.hit.Ng_z = ray->tfar;
        q->ray_hit.hit.u = hit->u;
        q->ray_hit.hit.v = hit->v;
        q->ray_hit.hit.primID = hit->primID;
        q->ray_hit.hit.geomID = hit->geomID;
        q->ray_hit.hit.instID[0] = hit->instID[0];
    };
    auto ctx = reinterpret_cast<RayQueryContextEx *>(args->context);
    auto ray = reinterpret_cast<RTCRay *>(args->ray);
    auto hit = reinterpret_cast<RTCHit *>(args->hit);
    auto &&instance = ray_query_surface_instance(ctx, hit);
    if (instance.is_curve) {// curve
        hit->v = -1.f;
    }
    if (!ray_query_candidate_in_range(ctx, ray->tfar)) {
        args->valid[0] = 0;
        return;
    }
    auto q = ctx->base.q;
    if (instance.opaque) {// opaque
        record_hit_data(q, ray, hit);
    } else if (auto on_surface = ctx->on_surface) {
        auto candidate = &q->candidate;
        ray_query_decode_surface_candidate(candidate, ray, hit);
        const auto embree_tnear = ray->tnear;
        ray->tnear = ctx->original_tnear;
        on_surface(reinterpret_cast<LC_RayQueryObject *>(q), ctx->capture);
        ray->tnear = embree_tnear;
        if (candidate->committed) {
            record_hit_data(q, ray, hit);
        } else {
            args->valid[0] = 0;
        }
        if (candidate->terminated) {
            luisa_fallback_ray_query_terminate_ray(ray);
        }
    } else {
        args->valid[0] = 0;
    }
}

void luisa_fallback_ray_query_pipeline_all(LC_RayQueryObject *query_object, const void *capture, RayQueryOnSurfaceFunc *on_surface, RayQueryOnProceduralFunc *on_procedural) noexcept {

    auto q = reinterpret_cast<RayQueryObject *>(query_object);
    auto scene = static_cast<RTCScene>(q->accel.embree_scene);

    RayQueryContextEx ctx;
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.base.rtc_ctx);
#else
    rtcInitRayQueryContext(&ctx.base.rtc_ctx);
#endif
    ctx.base.q = q;
    ctx.capture = capture;
    ctx.current_t = q->world_ray.t_max;
    ctx.original_tnear = q->ray_hit.ray.tnear;
    ctx.on_surface = on_surface;
    ctx.on_procedural = on_procedural;
    q->ray_hit.ray.tnear =
        ray_query_embree_tnear(ctx.original_tnear);

#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    ctx.base.rtc_ctx.filter = luisa_fallback_ray_query_surface_intersect_filter_function;
    rtcIntersect1(scene, &ctx.base.rtc_ctx, &q->ray_hit);
#else
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.context = &ctx.base.rtc_ctx;
    args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
    args.filter = luisa_fallback_ray_query_surface_intersect_filter_function;
    rtcIntersect1(scene, &q->ray_hit, &args);
#endif
    q->ray_hit.ray.tnear = ctx.original_tnear;
    q->ray_hit.hit.Ng_z = ctx.current_t;
    ray_trace_fix_hit_for_curves(scene, &q->ray_hit);
}

void luisa_fallback_ray_query_pipeline_any(LC_RayQueryObject *query_object, const void *capture, RayQueryOnSurfaceFunc *on_surface, RayQueryOnProceduralFunc *on_procedural) noexcept {

    auto q = reinterpret_cast<RayQueryObject *>(query_object);
    auto scene = static_cast<RTCScene>(q->accel.embree_scene);

    RayQueryContextEx ctx;
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.base.rtc_ctx);
#else
    rtcInitRayQueryContext(&ctx.base.rtc_ctx);
#endif
    ctx.base.q = q;
    ctx.capture = capture;
    ctx.current_t = q->world_ray.t_max;
    ctx.original_tnear = q->ray_hit.ray.tnear;
    ctx.on_surface = on_surface;
    ctx.on_procedural = on_procedural;
    q->ray_hit.ray.tnear =
        ray_query_embree_tnear(ctx.original_tnear);

#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    ctx.base.rtc_ctx.filter = luisa_fallback_ray_query_surface_occluded_filter_function;
    rtcOccluded1(scene, &ctx.base.rtc_ctx, &q->ray_hit.ray);
#else
    RTCOccludedArguments args;
    rtcInitOccludedArguments(&args);
    args.context = &ctx.base.rtc_ctx;
    args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
    args.filter = luisa_fallback_ray_query_surface_occluded_filter_function;
    rtcOccluded1(scene, &q->ray_hit.ray, &args);
#endif
    q->ray_hit.ray.tnear = ctx.original_tnear;
}

}// namespace luisa::compute::fallback::api
