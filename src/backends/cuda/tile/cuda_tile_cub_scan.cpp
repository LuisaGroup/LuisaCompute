#include "cuda_tile_cub_scan.h"
#include <luisa/core/stl/format.h>
#include <limits>

namespace luisa::compute::cuda::native_tile {
namespace {
namespace cub_scan_detail {

[[nodiscard]] luisa::string_view scalar(tile::ScalarType type) noexcept {
    using tile::ScalarType;
    switch (type) {
        case ScalarType::BOOL: return "bool";
        case ScalarType::INT32: return "int";
        case ScalarType::UINT32: return "unsigned";
        case ScalarType::INT64: return "long long";
        case ScalarType::UINT64: return "unsigned long long";
        case ScalarType::FLOAT16: return "__half";
        case ScalarType::BFLOAT16: return "__nv_bfloat16";
        case ScalarType::FLOAT32: return "float";
        default: return {};
    }
}

// BlockScan consumes blocked local arrays: thread t, item j is column
// base + 8*t + j. Global uint4 transport changes representation only. The
// only reused shared state is Scan::TempStorage; every nonfirst iteration
// synchronizes before its next Scan, including all threads in the CTA.
constexpr luisa::string_view source_prefix = R"cub(#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_scan.cuh>

namespace luisa_tile_cub_detail {
struct Fp16Storage {
    __device__ static float load(unsigned short bits) { return __half2float(__ushort_as_half(bits)); }
    __device__ static unsigned short store(float value) { return __half_as_ushort(__float2half_rn(value)); }
};
struct Bf16Storage {
    __device__ static float load(unsigned short bits) { return __bfloat162float(__ushort_as_bfloat16(bits)); }
    __device__ static unsigned short store(float value) { return __bfloat16_as_ushort(__float2bfloat16_rn(value)); }
};

template<typename Codec, int Threads, int Width>
struct RowScan {
    static constexpr int Items = 8;
    static constexpr int Chunk = Threads * Items;
    static_assert(Threads == 128 || Threads == 256 || Threads == 512 || Threads == 1024);
    static_assert(Width > 0 && Width % Chunk == 0);
    static_assert(sizeof(unsigned short) == 2 && sizeof(uint4) == 16 && alignof(uint4) == 16);
    using Scan = cub::BlockScan<float, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
    struct Storage { typename Scan::TempStorage scan; };

    __device__ __forceinline__ static void run(const unsigned short *input,
                                               unsigned short *output, Storage &storage) {
        const auto row_start = static_cast<unsigned long long>(blockIdx.x) * Width;
        float carry = 0.0f;
#pragma unroll 1
        for (int base = 0; base < Width; base += Chunk) {
            unsigned short bits[Items];
            float values[Items];
            const auto thread_offset = row_start + base + static_cast<unsigned long long>(threadIdx.x) * Items;
            const uint4 *vectors_in = reinterpret_cast<const uint4 *>(input + thread_offset);
            uint4 packed = vectors_in[0];
            bits[0] = static_cast<unsigned short>(packed.x);
            bits[1] = static_cast<unsigned short>(packed.x >> 16);
            bits[2] = static_cast<unsigned short>(packed.y);
            bits[3] = static_cast<unsigned short>(packed.y >> 16);
            bits[4] = static_cast<unsigned short>(packed.z);
            bits[5] = static_cast<unsigned short>(packed.z >> 16);
            bits[6] = static_cast<unsigned short>(packed.w);
            bits[7] = static_cast<unsigned short>(packed.w >> 16);
            if (base != 0) { __syncthreads(); }
#pragma unroll
            for (int item = 0; item < Items; ++item) { values[item] = Codec::load(bits[item]); }
            float aggregate;
            Scan{storage.scan}.InclusiveSum(values, values, aggregate);
#pragma unroll
            for (int item = 0; item < Items; ++item) {
                float cumulative = base == 0 ? values[item] : __fadd_rn(carry, values[item]);
                bits[item] = Codec::store(__fadd_rn(0.0f, cumulative));
            }
            carry = base == 0 ? aggregate : __fadd_rn(carry, aggregate);
            uint4 *vectors_out = reinterpret_cast<uint4 *>(output + thread_offset);
            uint4 result;
            result.x = static_cast<unsigned>(bits[0]) | (static_cast<unsigned>(bits[1]) << 16);
            result.y = static_cast<unsigned>(bits[2]) | (static_cast<unsigned>(bits[3]) << 16);
            result.z = static_cast<unsigned>(bits[4]) | (static_cast<unsigned>(bits[5]) << 16);
            result.w = static_cast<unsigned>(bits[6]) | (static_cast<unsigned>(bits[7]) << 16);
            vectors_out[0] = result;
        }
    }
};
}// namespace luisa_tile_cub_detail

)cub";

}
}// namespace ::cub_scan_detail

CubScanArtifact generate_cub_scan(const tile::Function &function, const Artifact &original,
                                  uint32_t threads) noexcept {
    CubScanArtifact result;
    auto fail = [&](luisa::string_view why) noexcept {
        result.error.assign(why.data(), why.size());
        result.source.clear();
        return std::move(result);
    };
    if (!original.ok() || original.scan_chunk_extent != 0u || original.independent_axis_extent != 0u ||
        original.streaming_scan_chunk_extent != 0u || original.partition_rows != 0u ||
        original.aligned16_buffer_mask != 0u || original.aligned16_partition_loads != 0u ||
        !original.aligned16_entry.empty() || !original.streaming_scan_entry.empty() || !original.partition_entry.empty()) {
        return fail("CUB prefix requires a successful untransformed original Tile artifact");
    }
    if (threads != 128u && threads != 256u && threads != 512u && threads != 1024u) {
        return fail("CUB prefix threads must be 128, 256, 512 or 1024");
    }
    auto proof = tile::analyze_closed_prefix(function);
    if (!proof.ok()) { return fail(proof.error); }
    auto rows = proof.logical_independent_extent;
    auto width = proof.logical_contribution_extent;
    auto chunk = threads * 8u;
    constexpr auto launch_limit = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    if ((proof.storage != tile::ScalarType::FLOAT16 && proof.storage != tile::ScalarType::BFLOAT16) ||
        proof.original.independent_extent_per_program != 1u || proof.original.programs != rows ||
        proof.original.tail_valid_extent != 0u || rows > launch_limit || width > launch_limit ||
        width != proof.collective.contribution_extent || width % chunk != 0u) {
        return fail("CUB prefix requires complete narrow-storage rows with width divisible by 8*threads");
    }
    auto root = function.body().block(0u);
    if (root->argument_count() != original.arguments.size() || original.arguments.size() > 31u ||
        original.grid != std::array<uint32_t, 3u>{static_cast<uint32_t>(rows), 1u, 1u} ||
        original.block != std::array<uint32_t, 3u>{1u, 1u, 1u}) {
        return fail("CUB prefix original direct-buffer ABI or launch geometry disagrees with IR");
    }
    auto input_slot = proof.disjoint.input.argument_index;
    auto output_slot = proof.disjoint.output.argument_index;
    if (input_slot >= original.arguments.size() || output_slot >= original.arguments.size() || input_slot == output_slot ||
        proof.disjoint.input.byte_offset != 0u || proof.disjoint.output.byte_offset != 0u ||
        proof.disjoint.input.byte_count == 0u || proof.disjoint.input.byte_count != proof.disjoint.output.byte_count) {
        return fail("CUB prefix root intervals do not match the direct-buffer ABI");
    }
    for (auto slot = size_t{0u}; slot < original.arguments.size(); slot++) {
        auto &&argument = original.arguments[slot];
        auto &&type = root->argument(slot)->type();
        if (!type.is_view() || type.scalar_type() != argument.element || cub_scan_detail::scalar(argument.element).empty() ||
            argument.read != (slot == input_slot) || argument.written != (slot == output_slot)) {
            return fail("CUB prefix resource type/effects disagree with the closed root view chain");
        }
        if ((slot == input_slot || slot == output_slot) && argument.minimum_size_bytes != proof.disjoint.input.byte_count) {
            return fail("CUB prefix resource size disagrees with its complete root interval");
        }
    }
    result.threads = threads;
    result.chunk_extent = chunk;
    result.grid = {static_cast<uint32_t>(rows), 1u, 1u};
    result.block = {threads, 1u, 1u};
    result.guard = StreamingScanGuard{input_slot, output_slot, proof.disjoint.input.byte_count, proof.disjoint.output.byte_count};
    result.alignment_mask = (uint32_t{1u} << input_slot) | (uint32_t{1u} << output_slot);
    result.source.assign(cub_scan_detail::source_prefix.data(), cub_scan_detail::source_prefix.size());
    result.source += luisa::format("extern \"C\" __global__ __launch_bounds__({})\nvoid {}(", threads, result.entry);
    for (auto slot = size_t{0u}; slot < original.arguments.size(); slot++) {
        if (slot != 0u) { result.source += ", "; }
        result.source += luisa::format("{} *buffer{}", cub_scan_detail::scalar(original.arguments[slot].element), slot);
    }
    auto codec = proof.storage == tile::ScalarType::FLOAT16 ? "Fp16Storage" : "Bf16Storage";
    result.source += luisa::format(
        ") {{\n    using Row = luisa_tile_cub_detail::RowScan<luisa_tile_cub_detail::{}, {}, {}>;\n"
        "    __shared__ Row::Storage storage;\n"
        "    Row::run(reinterpret_cast<const unsigned short *>(buffer{}), reinterpret_cast<unsigned short *>(buffer{}), storage);\n}}\n",
        codec, threads, width, input_slot, output_slot);
    return result;
}

}// namespace luisa::compute::cuda::native_tile
