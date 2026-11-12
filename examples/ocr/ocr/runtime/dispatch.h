#pragma once
// OCR engine -- launch geometry helpers (Phase 2). Host-only.
//
// Every kernel factory in ocr/ops has a matching plan_* helper here. All
// plans use TOTAL-THREAD dispatch semantics (LuisaCompute dispatch(n) takes
// a total thread count, not a block count) and every kernel guards its tail
// (`if (id < count)`); nothing is ever clamped or dropped. When a 1D launch
// would exceed the portable 65535-group envelope (DX-like backends), the
// plan flattens the launch into a 2D grid: dispatch.x stays a multiple of
// the block size and dispatch.y carries the remaining block rows; kernels
// recover the linear id with `dispatch_y() * dispatch_x_total + dispatch_x()`
// (see ocr_linear_id() in ops/ops_common.h).
//
// All plans are checked: dimensions that do not fit the uint32 kernel
// envelope return a Status instead of truncating. Zero-sized work produces
// an empty plan (dispatch nothing), never a degenerate 0-thread launch.
#include <cstdint>
#include <utility>

#include "ocr/status.h"

namespace luisa::ocr {

// Portable group-count cap per grid dimension (D3D12
// D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION). CUDA/Vulkan allow more,
// but the portable baseline plans for the smallest envelope.
inline constexpr uint32_t kMaxGroupsPerDim = 65535u;
// Default block width of the 1D elementwise/row kernels.
inline constexpr uint32_t kDefaultBlock1D = 256u;

// Checked uint64 -> uint32 conversion for kernel arguments.
[[nodiscard]] Result<uint32_t> checked_u32(uint64_t value, luisa::string_view what);

// A planned launch. `total` is the TOTAL thread count per dimension (what
// compute::Shader::dispatch expects); `groups` is the derived group count.
// empty() == true means "dispatch nothing" (zero work), which callers must
// honor by skipping the dispatch entirely.
struct LaunchPlan1D {
    uint32_t total_x{0};
    uint32_t total_y{1};
    uint32_t block{0};
    uint32_t groups_x{0};
    uint32_t groups_y{1};
    uint64_t work_items{0};// logical element/row count the kernel must cover

    [[nodiscard]] bool empty() const noexcept { return work_items == 0u; }
    [[nodiscard]] bool is_2d() const noexcept { return total_y != 1u; }
};

// Plan `count` work items over 1D blocks of `block` threads, flattening into
// a 2D grid when the group count would exceed kMaxGroupsPerDim. The kernel
// must use the flattened linear id when plan.is_2d(). `count` may be 0
// (empty plan) or exceed uint32 as long as the PADDED thread envelope fits;
// the kernel receives the count as uint32, so counts > UINT32_MAX are
// rejected (the caller must chunk -- see plan_row_chunks).
[[nodiscard]] Result<LaunchPlan1D> plan_threads(uint64_t count, uint32_t block = kDefaultBlock1D);

// 2D plan: one thread per (row, col) of a [rows, cols] output, blocks of
// (block_x, block_y). Flattens the y dimension when needed. Kernels use
// dispatch_x()/dispatch_y() directly when !flattened_y, otherwise recover
// row = linear / cols.
struct LaunchPlan2D {
    uint32_t total_x{0};
    uint32_t total_y{0};
    uint32_t block_x{0};
    uint32_t block_y{0};
    uint32_t groups_x{0};
    uint32_t groups_y{0};
    uint64_t rows{0};
    uint64_t cols{0};
    uint32_t padded_cols{0};// in-kernel column stride for flattened recovery
    bool flattened_y{false};// true: y was folded into x; recover row/col in-kernel

    [[nodiscard]] bool empty() const noexcept { return rows == 0u || cols == 0u; }
};

[[nodiscard]] Result<LaunchPlan2D> plan_threads_2d(uint64_t rows, uint64_t cols,
                                                   uint32_t block_x = 16u, uint32_t block_y = 16u);

// Row/query chunking: split `rows` rows into chunks of at most `max_rows`
// (never dropping rows). Used to bound command lists and to keep per-launch
// row counts inside the uint32 envelope. Returns the chunk count and the
// row range of chunk i via chunk_range().
struct RowChunkPlan {
    uint64_t rows{0};
    uint32_t max_rows{0};
    uint64_t chunk_count{0};

    [[nodiscard]] bool empty() const noexcept { return rows == 0u; }
    // [begin, end) of chunk `index`; end is clamped to rows for the tail.
    [[nodiscard]] Result<std::pair<uint64_t, uint64_t>> chunk_range(uint64_t index) const;
};

[[nodiscard]] Result<RowChunkPlan> plan_row_chunks(uint64_t rows, uint32_t max_rows);

// Shared-memory guard: the portable HLSL/DX floor is 32 KiB per group.
inline constexpr uint64_t kPortableSharedMemoryBytes = 32ull * 1024ull;
[[nodiscard]] Status check_shared_memory_budget(uint64_t bytes);

}// namespace luisa::ocr