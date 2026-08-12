#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace llvm {
class Function;
class Module;
}// namespace llvm

namespace luisa::compute::simd::schedule {
class Function;
}// namespace luisa::compute::simd::schedule

namespace luisa::compute::simd {

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
struct alignas(16) SIMDHostBindlessTextureSlot {
    void *texture{nullptr};
    uint32_t sampler_code{0u};
    uint32_t reserved{0u};
};

struct alignas(16) SIMDHostBindlessSlot {
    SIMDHostBufferView buffer{};
    SIMDHostBindlessTextureSlot texture2d{};
    SIMDHostBindlessTextureSlot texture3d{};
};

// Bindless texture callbacks consume one SoA packet. The runtime groups lanes
// that resolve to the same texture/sampler before sampling, while slot_indices
// remain free to diverge. A null sampler_codes pointer selects the sampler
// stored in each slot; a null levels pointer selects mip zero. Results contain
// four (sample/read) or three (size) consecutive component vectors.
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

// Acceleration-structure callbacks consume component-major ray packets:
// origin.xyz, t_min, direction.xyz, and t_max are eight consecutive float
// vectors. Motion queries additionally provide one time vector; static queries
// pass a null time pointer and use time zero. Closest-hit ids contain instance
// and primitive vectors; hit values contain barycentric u/v and committed t
// vectors. W2 is padded to Embree's four-wide ABI by the runtime, while W1
// alone may use the scalar API.
using SIMDHostAccelTraceClosest = void(
    void *accel, uint32_t lane_count, uint64_t active_mask_bits,
    const float *ray_components, const uint32_t *visibility_masks,
    const float *times, uint32_t *hit_ids, float *hit_values);
using SIMDHostAccelTraceAny = void(
    void *accel, uint32_t lane_count, uint64_t active_mask_bits,
    const float *ray_components, const uint32_t *visibility_masks,
    const float *times, uint32_t *occluded);

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

inline constexpr auto simd_host_ray_query_candidate_batch_capacity = 32u;

struct SIMDHostRayQueryState;
using SIMDHostAccelRayQueryProceed = void(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states);

// One fixed-size state belongs to each logical lane and query construction
// site. Embree traversal is still issued once per active packet; these AoS
// records only let the ordinary SIMD CFG retain candidate/commit state across
// divergent handler iterations without a second callback-side PC machine.
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
    uint32_t reserved[2]{};
    SIMDHostRayQuerySurfaceHit candidate{};
    SIMDHostRayQueryCommittedHit committed{};
    uint32_t candidate_batch_count{0u};
    uint32_t candidate_batch_index{0u};
    uint32_t candidate_batch_has_more{0u};
    uint32_t candidate_batch_initialized{0u};
    SIMDHostRayQuerySurfaceHit
        candidate_batch[simd_host_ray_query_candidate_batch_capacity]{};
};

// Device-side instance metadata reads use this stable plain-data table rather
// than depending on SIMDAccel's C++ object or vector layout. The table object
// itself remains at a fixed address while a build may replace its data pointer.
enum class SIMDHostAccelMotionMode : uint32_t {
    matrix = 0u,
    srt = 1u,
};
inline constexpr size_t simd_host_accel_motion_frame_size = 64u;

struct alignas(16) SIMDHostAccelInstance {
    float affine[12]{};
    uint32_t user_id{0u};
    uint8_t mask{0xffu};
    uint8_t opaque{1u};
    uint8_t dirty{0u};
    uint8_t curve{0u};
    uint64_t reserved_motion_alignment{0u};
    // Null for a static instance. Motion frames retain the public
    // MotionInstanceTransform 64-byte layout; motion_mode uses
    // AccelMotionMode's MATRIX=0/SRT=1 wire values.
    void *motion_frames{nullptr};
    uint32_t motion_keyframe_count{0u};
    uint32_t motion_mode{0u};
};

struct alignas(16) SIMDHostAccelInstanceTable {
    SIMDHostAccelInstance *data{nullptr};
    size_t size{0u};
};

struct alignas(16) SIMDHostAccelView {
    void *accel{nullptr};
    SIMDHostAccelTraceClosest *trace_closest{nullptr};
    SIMDHostAccelTraceAny *trace_any{nullptr};
    const SIMDHostAccelInstanceTable *instances{nullptr};
    SIMDHostAccelRayQueryProceed *ray_query_proceed{nullptr};
    void *reserved{nullptr};
};
static_assert(sizeof(SIMDHostAccelTraceClosest *) == sizeof(void *));
static_assert(sizeof(SIMDHostAccelTraceAny *) == sizeof(void *));
static_assert(sizeof(SIMDHostAccelRayQueryProceed *) == sizeof(void *));
static_assert(sizeof(SIMDHostRayQuerySurfaceHit) == 24u);
static_assert(sizeof(SIMDHostRayQueryCommittedHit) == 24u);
static_assert(sizeof(SIMDHostRayQueryState) == 928u);
static_assert(offsetof(SIMDHostRayQueryState, world_ray) == 16u);
static_assert(offsetof(SIMDHostRayQueryState, candidate) == 96u);
static_assert(offsetof(SIMDHostRayQueryState, committed) == 120u);
static_assert(offsetof(SIMDHostRayQueryState, candidate_batch_count) == 144u);
static_assert(offsetof(SIMDHostRayQueryState, candidate_batch) == 160u);
static_assert(sizeof(SIMDHostAccelInstance) == 80u);
static_assert(offsetof(SIMDHostAccelInstance, affine) == 0u);
static_assert(offsetof(SIMDHostAccelInstance, user_id) == 48u);
static_assert(offsetof(SIMDHostAccelInstance, mask) == 52u);
static_assert(offsetof(SIMDHostAccelInstance, curve) == 55u);
static_assert(offsetof(SIMDHostAccelInstance, motion_frames) == 64u);
static_assert(offsetof(SIMDHostAccelInstance, motion_keyframe_count) == 72u);
static_assert(offsetof(SIMDHostAccelInstance, motion_mode) == 76u);
static_assert(sizeof(SIMDHostAccelInstanceTable) == 16u);
static_assert(offsetof(SIMDHostAccelView, accel) == 0u);
static_assert(offsetof(SIMDHostAccelView, trace_closest) == sizeof(void *));
static_assert(offsetof(SIMDHostAccelView, trace_any) == 2u * sizeof(void *));
static_assert(offsetof(SIMDHostAccelView, instances) == 3u * sizeof(void *));
static_assert(offsetof(SIMDHostAccelView, ray_query_proceed) ==
              4u * sizeof(void *));
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

struct alignas(16) SIMDHostTextureView {
    void *texture{nullptr};
    SIMDHostTextureRead *read_float{nullptr};
    SIMDHostTextureRead *read_uint{nullptr};
    SIMDHostTextureWrite *write_float{nullptr};
    SIMDHostTextureWrite *write_uint{nullptr};
    SIMDHostTextureSize *size{nullptr};
    uint32_t level{0u};
    uint32_t dimension{0u};
};

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
    size_t argument_buffer_size{0u};
    size_t uniform_buffer_broadcast_count{0u};
    size_t contiguous_buffer_read_count{0u};
    size_t contiguous_buffer_write_count{0u};
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
    bool enable_lane_affine_buffer = true);

}// namespace luisa::compute::simd
