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
