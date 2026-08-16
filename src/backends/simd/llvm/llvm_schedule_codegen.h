#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace llvm {
class Function;
class Module;
}// namespace llvm

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::simd::schedule {
class Function;
}// namespace luisa::compute::simd::schedule

namespace luisa::compute::simd {

inline constexpr size_t simd_max_cooperative_frame_bytes =
    4u * 1024u * 1024u;
inline constexpr size_t simd_max_shared_memory_bytes =
    1u * 1024u * 1024u;
inline constexpr uint32_t simd_cooperative_packet_running =
    UINT32_MAX - 2u;
inline constexpr uint32_t simd_cooperative_packet_complete =
    UINT32_MAX - 1u;
inline constexpr uint32_t simd_cooperative_packet_inactive =
    UINT32_MAX;

// Host-side buffer descriptor passed through the packet argument buffer. The
// descriptor is intentionally backend-local: a runtime Buffer handle is
// resolved to this plain view immediately before a dispatch.
struct alignas(16) SIMDHostBufferView {
    void *data{nullptr};
    size_t size_bytes{0u};
};

// Bindless resources are resolved by the runtime into a dense host table.
// Keep the slot representation plain and backend-local so JIT code and packet
// callbacks never depend on runtime C++ object layouts.
struct alignas(8) SIMDHostBindlessTextureSlot {
    void *texture{nullptr};
    // Non-null only for BYTE1 mip zero. This lets the JIT lower the common
    // uniform-slot 2D sampling path to target-native fixed-vector IR without
    // depending on SIMDTexture/FallbackTexture C++ object layouts. Other
    // formats and mip paths continue through the packet callback.
    const std::byte *byte1_mip0{nullptr};
    uint64_t metadata{0u};
};

static constexpr auto simd_bindless_linear_point_mirror_sampler_code =
    (1u << 2u) | 2u;

static constexpr auto simd_bindless_texture_extent_bits = 20u;
static constexpr auto simd_bindless_texture_extent_mask =
    (uint64_t{1u} << simd_bindless_texture_extent_bits) - 1u;

[[nodiscard]] constexpr uint64_t simd_bindless_texture_metadata(
    uint32_t sampler_code, uint32_t width,
    uint32_t height, uint32_t depth) noexcept {
    return (sampler_code & 0x0fu) |
           ((static_cast<uint64_t>(width) &
             simd_bindless_texture_extent_mask)
            << 4u) |
           ((static_cast<uint64_t>(height) &
             simd_bindless_texture_extent_mask)
            << 24u) |
           ((static_cast<uint64_t>(depth) &
             simd_bindless_texture_extent_mask)
            << 44u);
}

[[nodiscard]] constexpr uint32_t simd_bindless_texture_sampler(
    const SIMDHostBindlessTextureSlot &slot) noexcept {
    return static_cast<uint32_t>(slot.metadata & 0x0fu);
}

static_assert(sizeof(SIMDHostBindlessTextureSlot) == 24u);

struct alignas(16) SIMDHostBindlessSlot {
    SIMDHostBufferView buffer{};
    SIMDHostBindlessTextureSlot texture2d{};
    SIMDHostBindlessTextureSlot texture3d{};
};

static_assert(sizeof(SIMDHostBindlessSlot) == 64u);

// Bindless texture callbacks consume one SoA packet. The runtime groups lanes
// that resolve to the same texture/sampler before sampling, while slot_indices
// remain free to diverge. A null sampler_codes pointer selects the sampler
// stored in each slot. For a non-gradient sample, levels is either null (mip
// zero) or an explicit LOD vector. Gradient-derived LOD is computed in JIT
// fixed-vector IR from the immutable slot extent and passed through levels.
// Results contain four (sample/read) or three (size) consecutive component
// vectors.
using SIMDHostBindlessTextureSample = void(
    const SIMDHostBindlessSlot *slots, size_t slot_count,
    uint32_t dimension, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *slot_indices,
    const uint32_t *sampler_codes,
    const float *u, const float *v, const float *w,
    const float *levels, float *values);
using SIMDHostBindlessTextureRead = void(
    const SIMDHostBindlessSlot *slots, size_t slot_count,
    uint32_t dimension, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *slot_indices,
    const uint32_t *x, const uint32_t *y, const uint32_t *z,
    const uint32_t *levels, float *values);
using SIMDHostBindlessTextureSize = void(
    const SIMDHostBindlessSlot *slots, size_t slot_count,
    uint32_t dimension, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *slot_indices,
    const uint32_t *levels, uint32_t *values);

struct alignas(16) SIMDHostBindlessArrayView {
    const SIMDHostBindlessSlot *slots{nullptr};
    size_t size{0u};
    SIMDHostBindlessTextureSample *sample_texture{nullptr};
    SIMDHostBindlessTextureRead *read_texture{nullptr};
    SIMDHostBindlessTextureSize *size_texture{nullptr};
};

// Acceleration-structure callbacks consume an in-place, component-major packet
// with Embree's public RTCRay/RTCHit field order. Direct trace does not expose
// Embree's application-defined ray ID, so the JIT stores Embree-compatible
// -1/0 validity lanes in that packet field. The runtime derives valid from the
// packet instead of round-tripping a packed mask. Each packet field is one
// vector of lane_count 32-bit words. The stable field indices below are shared
// by the target-independent JIT and the runtime; the runtime statically proves
// them against the configured Embree headers. W2 is still padded to Embree's
// W4 ABI, while W1 alone may use the scalar API. W4/W8/W16 pass the scratch and
// its embedded valid array directly to Embree, which writes results in place.
inline constexpr auto simd_host_accel_ray_tfar_field = 8u;
inline constexpr auto simd_host_accel_ray_id_field = 10u;
inline constexpr auto simd_host_accel_hit_u_field = 15u;
inline constexpr auto simd_host_accel_hit_v_field = 16u;
inline constexpr auto simd_host_accel_hit_prim_field = 17u;
inline constexpr auto simd_host_accel_hit_geom_field = 18u;
inline constexpr auto simd_host_accel_hit_inst_field = 19u;
using SIMDHostAccelTraceClosest = void(
    void *accel, uint32_t lane_count, void *ray_hit_packet);
using SIMDHostAccelTraceAny = void(
    void *accel, uint32_t lane_count, void *ray_packet);

enum class SIMDHostRayQueryCandidateKind : uint32_t {
    none = 0u,
    surface = 1u,
    procedural = 2u,
};

struct alignas(8) SIMDHostRayQuerySurfaceHit {
    uint32_t inst{~0u};
    uint32_t prim{~0u};
    float bary[2]{};
    float t{0.0f};
    uint32_t reserved{0u};
};

struct alignas(8) SIMDHostRayQueryCommittedHit {
    uint32_t inst{~0u};
    uint32_t prim{~0u};
    float bary[2]{};
    uint32_t kind{0u};
    float t{0.0f};
};

struct alignas(8) SIMDHostRayQueryProceduralHit {
    uint32_t inst{~0u};
    uint32_t prim{~0u};
};

inline constexpr auto simd_host_ray_query_candidate_batch_capacity = 32u;
inline constexpr auto simd_host_ray_query_terminated_status_shift = 0u;
inline constexpr auto simd_host_ray_query_surface_status_shift = 16u;
inline constexpr auto simd_host_ray_query_procedural_status_shift = 32u;
inline constexpr auto simd_host_ray_query_valid_status_shift = 48u;

struct SIMDHostRayQueryState;
using SIMDHostAccelRayQueryProceed = void(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states);
using SIMDHostAccelRayQueryProceedStatus = uint64_t(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states);

// One fixed-size state belongs to each logical lane and simultaneously live
// query object. Noninterfering construction sites may share storage after a
// fail-closed Schedule-IR liveness proof. Embree traversal is still issued
// once per active packet; these AoS records only let the ordinary SIMD CFG
// retain candidate/commit state across divergent handler iterations without a
// second callback-side PC machine.
struct alignas(16) SIMDHostRayQueryState {
    void *accel{nullptr};
    SIMDHostAccelRayQueryProceed *proceed{nullptr};
    float world_ray[8]{};
    float time{0.0f};
    uint32_t visibility_mask{0xffu};
    uint32_t terminate_on_first{0u};
    uint32_t cursor_valid{0u};
    uint32_t cursor_inst{~0u};
    uint32_t cursor_prim{~0u};
    float cursor_t{0.0f};
    uint32_t candidate_kind{0u};
    uint32_t candidate_committed{0u};
    uint32_t terminated{0u};
    uint32_t procedural_cursor_valid{0u};
    uint32_t reserved{0u};
    uint32_t procedural_cursor_inst{~0u};
    uint32_t procedural_cursor_prim{~0u};
    SIMDHostRayQuerySurfaceHit candidate{};
    SIMDHostRayQueryCommittedHit committed{};
    uint32_t candidate_batch_count{0u};
    uint32_t candidate_batch_index{0u};
    uint32_t candidate_batch_has_more{0u};
    uint32_t candidate_batch_initialized{0u};
    SIMDHostRayQuerySurfaceHit
        candidate_batch[simd_host_ray_query_candidate_batch_capacity]{};
    uint32_t procedural_batch_count{0u};
    uint32_t procedural_batch_index{0u};
    uint32_t procedural_batch_has_more{0u};
    uint32_t procedural_batch_initialized{0u};
    SIMDHostRayQueryProceduralHit
        procedural_batch[simd_host_ray_query_candidate_batch_capacity]{};
};

// The proceed callback has already classified every active lane before it
// returns. Carry that classification back across the host/JIT boundary so a
// proven query owner can retain the three hot predicates in one scalar mask
// instead of gathering candidate_kind/terminated from the large AoS record.
// Each field occupies the physical lane bits [0, 16). The fields intentionally
// remain independent: explicit termination does not clear candidate_kind in
// the public state. Inactive-lane bits are always clear;
// [48, 64) is reserved for JIT-side per-lane initialization validity.
[[nodiscard]] inline uint64_t simd_host_ray_query_pack_status(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    auto remaining = active_mask_bits & lane_mask;
    auto status = uint64_t{0u};
    while (remaining != 0u) {
        auto lane = static_cast<uint32_t>(std::countr_zero(remaining));
        auto bit = uint64_t{1u} << lane;
        remaining &= remaining - 1u;
        auto *state = states[lane];
        if (state->terminated != 0u) {
            status |= bit << simd_host_ray_query_terminated_status_shift;
        }
        if (state->candidate_kind == static_cast<uint32_t>(
                                         SIMDHostRayQueryCandidateKind::surface)) {
            status |= bit << simd_host_ray_query_surface_status_shift;
        } else if (state->candidate_kind == static_cast<uint32_t>(
                                                SIMDHostRayQueryCandidateKind::procedural)) {
            status |= bit << simd_host_ray_query_procedural_status_shift;
        }
    }
    return status;
}

// Procedural W16 queries commonly return a fully active cohort. In that
// case a sequential pass is cheaper than repeatedly finding and clearing the
// next set bit. Sparse cohorts retain the baseline packer so inactive entries
// may remain null and are never accessed.
[[nodiscard]] inline uint64_t simd_host_ray_query_pack_procedural_wide_status(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    auto active = active_mask_bits & lane_mask;
    if (active != lane_mask) {
        return simd_host_ray_query_pack_status(
            lane_count, active, states);
    }
    auto terminated = uint64_t{0u};
    auto surface = uint64_t{0u};
    auto procedural = uint64_t{0u};
    auto bit = uint64_t{1u};
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++, bit <<= 1u) {
        auto *state = states[lane];
        auto kind = state->candidate_kind;
        terminated |= state->terminated != 0u ? bit : 0u;
        surface |= kind == static_cast<uint32_t>(
                               SIMDHostRayQueryCandidateKind::surface) ?
                       bit :
                       0u;
        procedural |= kind == static_cast<uint32_t>(
                                  SIMDHostRayQueryCandidateKind::procedural) ?
                          bit :
                          0u;
    }
    return terminated |
           (surface << simd_host_ray_query_surface_status_shift) |
           (procedural << simd_host_ray_query_procedural_status_shift);
}

// Status-aware JIT kernels call this side entry. The status entry and the
// construction-selected plain provider form an internal ABI pair: the entry
// dispatches through that provider, which must reject any active state carrying
// a different plain callback, then scans the active lanes once to return the
// packed hot predicates. Proven JIT cohorts can therefore avoid repeating the
// same plain-callback gather without weakening the provider-side check.
[[nodiscard]] uint64_t simd_host_ray_query_proceed_status(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept;

// A W16 procedural accel selects this entry. Dense cohorts pack
// post-proceed status with one sequential state-pointer pass; sparse cohorts
// retain the bit-scan baseline so inactive/null lanes are never touched.
[[nodiscard]] uint64_t simd_host_ray_query_proceed_wide_procedural_status(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept;

// Provider-native W16 procedural entry. Unlike the generic paired wrapper,
// this publishes each lane's status while advancing an already cached batch
// or installing a newly scanned Embree batch, so it does not reread the large
// per-lane state in a separate post-proceed pass.
[[nodiscard]] uint64_t
simd_host_ray_query_proceed_wide_procedural_fused_status(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept;

[[nodiscard]] constexpr bool
simd_host_ray_query_use_procedural_wide_status(
    uint32_t lane_count,
    bool has_procedural_instances,
    bool enable_procedural_dense_status) noexcept {
    return lane_count == 16u &&
           has_procedural_instances &&
           enable_procedural_dense_status;
}

// Device-side instance metadata reads use this stable plain-data table rather
// than depending on SIMDAccel's C++ object or vector layout. The table object
// itself remains at a fixed address while a build may replace its data pointer.
enum class SIMDHostAccelMotionMode : uint32_t {
    matrix = 0u,
    srt = 1u,
};
inline constexpr size_t simd_host_accel_motion_frame_size = 64u;

enum class SIMDHostAccelGeometryKind : uint8_t {
    triangle = 0u,
    curve = 1u,
    procedural = 2u,
};

struct alignas(16) SIMDHostAccelInstance {
    float affine[12]{};
    uint32_t user_id{0u};
    uint8_t mask{0xffu};
    uint8_t opaque{1u};
    uint8_t dirty{0u};
    uint8_t geometry_kind{
        static_cast<uint8_t>(SIMDHostAccelGeometryKind::triangle)};
    uint64_t reserved_motion_alignment{0u};
    // Null for a static instance. Motion frames retain the public
    // MotionInstanceTransform 64-byte layout; motion_mode uses
    // AccelMotionMode's MATRIX=0/SRT=1 wire values.
    void *motion_frames{nullptr};
    uint32_t motion_keyframe_count{0u};
    uint32_t motion_mode{0u};
};

struct SIMDHostAccelCommittedInstance {
    uint8_t geometry_kind{
        static_cast<uint8_t>(SIMDHostAccelGeometryKind::triangle)};
    uint8_t opaque{1u};
};
static_assert(sizeof(SIMDHostAccelCommittedInstance) == 2u);

struct alignas(16) SIMDHostAccelInstanceTable {
    SIMDHostAccelInstance *data{nullptr};
    size_t size{0u};
    SIMDHostAccelRayQueryProceedStatus *ray_query_proceed_status{nullptr};
    SIMDHostAccelRayQueryProceedStatus *ray_query_proceed_wide_status{nullptr};
    // Geometry classification belongs to the last committed Embree scene,
    // not necessarily to the desired public table above. A buffer-only
    // primitive replacement or resize must not reinterpret stale BVH hits
    // using the new primitive kind.
    const SIMDHostAccelCommittedInstance *committed_instances{nullptr};
    size_t committed_size{0u};
};

struct alignas(16) SIMDHostAccelView {
    void *accel{nullptr};
    SIMDHostAccelTraceClosest *trace_closest{nullptr};
    SIMDHostAccelTraceAny *trace_any{nullptr};
    const SIMDHostAccelInstanceTable *instances{nullptr};
    SIMDHostAccelRayQueryProceed *ray_query_proceed{nullptr};
    // W8/W16 provider with runtime dense/sparse cohort selection. Narrow
    // specializations select ray_query_proceed instead.
    SIMDHostAccelRayQueryProceed *ray_query_proceed_wide{nullptr};
};
static_assert(sizeof(SIMDHostAccelTraceClosest *) == sizeof(void *));
static_assert(sizeof(SIMDHostAccelTraceAny *) == sizeof(void *));
static_assert(sizeof(SIMDHostAccelRayQueryProceed *) == sizeof(void *));
static_assert(sizeof(SIMDHostAccelRayQueryProceedStatus *) == sizeof(void *));
static_assert(sizeof(SIMDHostRayQuerySurfaceHit) == 24u);
static_assert(sizeof(SIMDHostRayQueryCommittedHit) == 24u);
static_assert(sizeof(SIMDHostRayQueryProceduralHit) == 8u);
static_assert(sizeof(SIMDHostRayQueryState) == 1216u);
static_assert(offsetof(SIMDHostRayQueryState, world_ray) == 16u);
static_assert(offsetof(SIMDHostRayQueryState, candidate_kind) == 76u);
static_assert(offsetof(SIMDHostRayQueryState, candidate_committed) == 80u);
static_assert(offsetof(SIMDHostRayQueryState, terminated) == 84u);
static_assert(offsetof(SIMDHostRayQueryState, procedural_cursor_valid) == 88u);
static_assert(offsetof(SIMDHostRayQueryState, candidate) == 104u);
static_assert(offsetof(SIMDHostRayQueryState, committed) == 128u);
static_assert(offsetof(SIMDHostRayQueryCommittedHit, inst) == 0u);
static_assert(offsetof(SIMDHostRayQueryCommittedHit, prim) == 4u);
static_assert(offsetof(SIMDHostRayQueryCommittedHit, bary) == 8u);
static_assert(offsetof(SIMDHostRayQueryCommittedHit, kind) == 16u);
static_assert(offsetof(SIMDHostRayQueryCommittedHit, t) == 20u);
static_assert(offsetof(SIMDHostRayQueryState, candidate_batch_count) == 152u);
static_assert(offsetof(SIMDHostRayQueryState, candidate_batch) == 168u);
static_assert(offsetof(SIMDHostRayQueryState, procedural_batch_count) == 936u);
static_assert(offsetof(SIMDHostRayQueryState, procedural_batch) == 952u);
static_assert(sizeof(SIMDHostAccelInstance) == 80u);
static_assert(offsetof(SIMDHostAccelInstance, affine) == 0u);
static_assert(offsetof(SIMDHostAccelInstance, user_id) == 48u);
static_assert(offsetof(SIMDHostAccelInstance, mask) == 52u);
static_assert(offsetof(SIMDHostAccelInstance, opaque) == 53u);
static_assert(offsetof(SIMDHostAccelInstance, dirty) == 54u);
static_assert(offsetof(SIMDHostAccelInstance, geometry_kind) == 55u);
static_assert(offsetof(SIMDHostAccelInstance, motion_frames) == 64u);
static_assert(offsetof(SIMDHostAccelInstance, motion_keyframe_count) == 72u);
static_assert(offsetof(SIMDHostAccelInstance, motion_mode) == 76u);
static_assert(sizeof(SIMDHostAccelInstanceTable) == 48u);
static_assert(offsetof(SIMDHostAccelInstanceTable, data) == 0u);
static_assert(offsetof(SIMDHostAccelInstanceTable, size) == sizeof(void *));
static_assert(offsetof(
                  SIMDHostAccelInstanceTable,
                  ray_query_proceed_status) == 2u * sizeof(void *));
static_assert(offsetof(
                  SIMDHostAccelInstanceTable,
                  ray_query_proceed_wide_status) == 3u * sizeof(void *));
static_assert(offsetof(
                  SIMDHostAccelInstanceTable,
                  committed_instances) == 4u * sizeof(void *));
static_assert(offsetof(
                  SIMDHostAccelInstanceTable,
                  committed_size) == 5u * sizeof(void *));
static_assert(offsetof(SIMDHostAccelView, accel) == 0u);
static_assert(offsetof(SIMDHostAccelView, trace_closest) == sizeof(void *));
static_assert(offsetof(SIMDHostAccelView, trace_any) == 2u * sizeof(void *));
static_assert(offsetof(SIMDHostAccelView, instances) == 3u * sizeof(void *));
static_assert(offsetof(SIMDHostAccelView, ray_query_proceed) ==
              4u * sizeof(void *));
static_assert(offsetof(SIMDHostAccelView, ray_query_proceed_wide) ==
              5u * sizeof(void *));
static_assert(sizeof(SIMDHostAccelView) == 6u * sizeof(void *));

// Texture callbacks operate once per SIMD packet. Coordinates are SoA vectors
// of lane_count elements and values contain four consecutive component
// vectors. inactive_mask_bits uses its low lane_count bits. This keeps the JIT
// ABI target-independent while allowing the runtime to batch coherent texels
// and iterate only set bits for a sparse cohort.
using SIMDHostTextureRead = void(
    void *texture, uint32_t level, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *x,
    const uint32_t *y, const uint32_t *z, void *values);
using SIMDHostTextureWrite = void(
    void *texture, uint32_t level, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *x,
    const uint32_t *y, const uint32_t *z, const void *values);
using SIMDHostTextureSize = uint32_t(
    void *texture, uint32_t level, uint32_t axis);
using SIMDHostTextureSample = void(
    void *texture, uint32_t base_level, uint32_t dimension,
    uint32_t lane_count, uint64_t active_mask_bits,
    const uint32_t *sampler_codes, const float *u,
    const float *v, const float *w, const float *levels,
    float *values);

struct alignas(16) SIMDHostTextureView {
    void *texture{nullptr};
    SIMDHostTextureRead *read_float{nullptr};
    SIMDHostTextureRead *read_uint{nullptr};
    SIMDHostTextureWrite *write_float{nullptr};
    SIMDHostTextureWrite *write_uint{nullptr};
    SIMDHostTextureSize *size{nullptr};
    uint32_t level{0u};
    uint32_t dimension{0u};
    // Appended after the established read/write/size fields. On the supported
    // 64-bit hosts this consumes the descriptor's existing tail padding, so
    // the argument-slot size and every earlier field offset stay unchanged.
    SIMDHostTextureSample *sample_float{nullptr};
};
static_assert(offsetof(SIMDHostTextureView, texture) == 0u);
static_assert(offsetof(SIMDHostTextureView, read_float) == sizeof(void *));
static_assert(offsetof(SIMDHostTextureView, size) == 5u * sizeof(void *));
static_assert(offsetof(SIMDHostTextureView, level) == 6u * sizeof(void *));
static_assert(offsetof(SIMDHostTextureView, dimension) ==
              6u * sizeof(void *) + sizeof(uint32_t));
static_assert(offsetof(SIMDHostTextureView, sample_float) ==
              7u * sizeof(void *));
static_assert(sizeof(SIMDHostTextureView) == 8u * sizeof(void *));

// Per-packet launch state. thread_index is the first linear thread within the
// current block; lane i executes thread_index + i. The generated entry derives
// thread_id and dispatch_id in fixed LLVM vectors and masks threads outside a
// non-divisible dispatch extent.
struct SIMDPacketLaunchConfig {
    uint32_t block_id[3]{};
    uint32_t dispatch_size[3]{};
    uint32_t block_size[3]{1u, 1u, 1u};
    uint32_t thread_index{0u};
    uint32_t kernel_id{0u};
    // Runtime-only block-range batching advances block_id in flattened grid
    // order. Ordinary packet and block-local packet-batch entries ignore this
    // field, so their physical ABI remains unchanged.
    uint32_t grid_size[3]{1u, 1u, 1u};
    // Debug side effects are invoked indirectly through the launch record so
    // the portable JIT module has no backend-private unresolved symbols. The
    // context is immutable for the duration of one synchronous dispatch.
    void *debug_context{nullptr};
    void (*print_callback)(
        void *context, uint64_t format_id,
        const void *arguments) noexcept {nullptr};
    void (*assert_fail_callback)(const char *message) noexcept {nullptr};
    // Cooperative-block kernels append all mutable synchronization state to
    // the established packet record. The runtime callbacks keep coroutine
    // allocation and shared storage out of the portable JIT symbol table.
    // Ordinary kernels leave these fields null and retain the previous ABI
    // prefix unchanged.
    void *shared_memory{nullptr};
    uint32_t *barrier_ids{nullptr};
    // One scalar epoch per packet and per tracked natural loop. The packet
    // coroutine publishes exact enclosing-loop instances before suspension;
    // the block wrapper compares them together with the static barrier ID.
    uint64_t *barrier_loop_epochs{nullptr};
    void *(*cooperative_block_begin)(
        size_t shared_memory_size) noexcept {nullptr};
    void *(*cooperative_frame_alloc)(size_t size) noexcept {nullptr};
    void (*cooperative_frame_free)(void *memory) noexcept {nullptr};
};

struct SIMDLLVMPrintFormat {
    std::string format{};
    std::vector<const Type *> argument_types{};
};

// Packet ABI:
//   void entry(ptr argument_buffer, ptr return_lanes,
//              ptr launch_config, i32 active_lane_count)
//
// Arguments are packed in declaration order at 16-byte boundaries. Value
// arguments use their Luisa ABI layout and buffer resources use
// SIMDHostBufferView. Scalar returns are written contiguously per lane; runtime
// kernels are normally void and pass a null return_lanes pointer.
struct LLVMScheduleCodegenResult {
    ::llvm::Function *entry{nullptr};
    // Optional runtime-only wrapper with the same physical signature. Its
    // fourth argument is a packet count rather than an active-lane count; it
    // advances launch_config.thread_index by specialization_width and invokes
    // entry once per packet. The packet body becomes internal in this mode:
    // statically small W16 batches construct only an unrolled direct-call
    // shell, while a target-profitable W8 batch may inline one body into the
    // dynamic loop so LLVM can reuse the frame and hoist invariants. LLVM's
    // normal inliner may still inline a sufficiently small internal direct-CFG
    // body. Standalone/direct compilation leaves this null and retains entry
    // as its exported ABI.
    ::llvm::Function *packet_batch_entry{nullptr};
    // Optional runtime-only wrapper around packet_batch_entry. The fourth
    // physical argument is a count of consecutive flattened blocks. The
    // wrapper advances launch_config.block_id in x-major grid order and
    // resets thread_index before issuing each complete block.
    ::llvm::Function *block_batch_entry{nullptr};
    size_t argument_buffer_size{0u};
    std::vector<SIMDLLVMPrintFormat> print_formats{};
    size_t schedule_block_count{0u};
    size_t convergence_point_count{0u};
    bool scalar_frame_metadata{false};
    size_t state_slot_count{0u};
    size_t coalesced_state_slot_count{0u};
    size_t general_colored_state_slot_count{0u};
    size_t spilled_instruction_count{0u};
    size_t cold_state_slot_count{0u};
    size_t stack_pinned_state_slot_count{0u};
    size_t ray_query_count{0u};
    size_t ray_query_scratch_slot_count{0u};
    size_t ray_query_scratch_bytes{0u};
    size_t ray_query_status_slot_count{0u};
    size_t ray_query_state_handle_slot_count{0u};
    size_t uniform_buffer_broadcast_count{0u};
    size_t contiguous_buffer_read_count{0u};
    size_t contiguous_buffer_write_count{0u};
    size_t transposed_buffer_read_count{0u};
    size_t transposed_buffer_write_count{0u};
    size_t paired_leaf_gather_count{0u};
    size_t predicated_memory_diamond_count{0u};
    size_t predicated_memory_instruction_count{0u};
    size_t local_predicated_diamond_count{0u};
    size_t local_predicated_two_sided_diamond_count{0u};
    size_t local_predicated_assignment_diamond_count{0u};
    size_t local_predicated_block_count{0u};
    size_t local_predicated_instruction_count{0u};
    size_t nested_predicated_region_count{0u};
    size_t nested_predicated_block_count{0u};
    size_t nested_predicated_instruction_count{0u};
    size_t chained_predicated_region_count{0u};
    size_t chained_predicated_transition_count{0u};
    size_t chained_predicated_block_count{0u};
    size_t chained_predicated_nested_tail_count{0u};
    size_t chained_predicated_terminal_block_count{0u};
    size_t chained_predicated_terminal_instruction_count{0u};
    size_t predicated_loop_count{0u};
    size_t predicated_loop_block_count{0u};
    size_t predicated_loop_instruction_count{0u};
    size_t predicated_loop_batch_iteration_count{0u};
    size_t structured_early_exit_loop_count{0u};
    size_t structured_early_exit_loop_block_count{0u};
    size_t structured_early_exit_loop_instruction_count{0u};
    size_t structured_early_exit_loop_absorbed_block_count{0u};
    size_t cohort_uniform_loop_branch_count{0u};
    size_t coherent_mask_reuse_count{0u};
    size_t all_on_region_version_count{0u};
    size_t all_on_region_block_count{0u};
    size_t all_on_region_instruction_count{0u};
    size_t convergence_token_guard_count{0u};
    size_t return_frame_guard_count{0u};
    size_t direct_divergent_child_count{0u};
    size_t unit_dimension_mask_elision_count{0u};
    size_t linear_1d_thread_id_count{0u};
    size_t linear_1d_packet_tail_narrowing_count{0u};
    size_t linear_1d_block_coalescing_count{0u};
    size_t shared_memory_size{0u};
    size_t block_barrier_count{0u};
    size_t block_barrier_loop_epoch_count{0u};
    std::vector<std::vector<uint32_t>> block_barrier_loop_epochs{};
    bool cooperative_block{false};
    bool direct_control_flow{false};
    std::string error{};

    [[nodiscard]] bool succeeded() const noexcept {
        return entry != nullptr && error.empty();
    }
};

// Lowers a supported reducible Schedule IR function to target-independent
// LLVM fixed vectors. No target ISA or hardware SIMD intrinsic is selected
// here; the LLVM target machine owns legalization, instruction selection,
// register allocation, and scheduling.
[[nodiscard]] LLVMScheduleCodegenResult lower_schedule_to_llvm(
    ::llvm::Module &module, const schedule::Function &function,
    uint32_t specialization_width, std::string_view entry_name = {},
    bool enable_fast_math = false,
    // A zero dimension selects the generic launch-config path. Nonzero static
    // dimensions must be powers of two and are lowered with shifts and masks.
    std::array<uint32_t, 3u> static_block_size = {},
    bool enable_uniform_buffer_broadcast = true,
    bool enable_lane_affine_buffer = true,
    // Target-unaware callers keep the portable 32-bit leaf path. The runtime
    // compiler enables pairing only after querying the host TargetMachine.
    bool enable_paired_leaf_gather = false,
    // The device-owned worker pool is fixed before shader compilation. It is
    // a profitability input only; generated packet semantics do not depend on
    // this count. Standalone callers conservatively model one worker.
    uint32_t dispatch_worker_count = 1u,
    // Runtime compilation supplies a host-TTI profitability decision. Direct
    // lowering callers default to enabled so target-independent semantic tests
    // can exercise the transformation without pretending to query a host.
    bool enable_native_predicated_loop = true,
    // Runtime kernels may amortize their host/JIT boundary across all packets
    // in one block. Eligible linear 1D kernels may additionally narrow their
    // final packet and coalesce a proven block-agnostic range. Standalone
    // lowering keeps the single-packet ABI only.
    bool enable_packet_batch_entry = false,
    // This is a target-profitability decision supplied by the host JIT. The
    // portable default retains direct packet calls in the batch wrapper; a
    // separate bounded source-shape policy decides whether W16 may use it.
    bool enable_inlined_packet_batch = false,
    // Runtime workers may amortize their JIT boundary across a consecutive
    // block range. This requires a statically known packet count and keeps the
    // block-local packet wrapper internal. A guarded linear-1D refinement may
    // collapse a proven block-agnostic range into one packet loop.
    bool enable_block_batch_entry = false);

}// namespace luisa::compute::simd
