// OCR engine -- launch geometry helpers (Phase 2). Host-only.
#include "ocr/runtime/dispatch.h"

#include "ocr/quantization.h"

namespace luisa::ocr {

[[nodiscard]] Result<uint32_t> checked_u32(uint64_t value, luisa::string_view what) {
    if (value > UINT32_MAX) {
        return Result<uint32_t>::failure(StatusCode::invalid_argument,
                                         luisa::string{what} + " exceeds the uint32 kernel envelope",
                                         luisa::string{what});
    }
    return Result<uint32_t>::success(static_cast<uint32_t>(value));
}

namespace {

// Overflow-safe ceil(a/b) in 64-bit (a + b - 1 can wrap near UINT64_MAX).
[[nodiscard]] constexpr uint64_t ceil_div(uint64_t a, uint64_t b) noexcept {
    return a / b + (a % b != 0u ? 1u : 0u);
}

[[nodiscard]] Result<uint32_t> ceil_div_u32(uint64_t a, uint64_t b, luisa::string_view what) {
    return checked_u32(ceil_div(a, b), what);
}

// Overflow-safe ceil(sqrt(v)): the naive `r*r < v` loop wraps for v near
// UINT64_MAX. Doubling + binary search compare via division, never wrapping.
[[nodiscard]] uint64_t ceil_sqrt(uint64_t v) noexcept {
    if (v <= 1u) { return v; }
    uint64_t hi = 1u;
    while (hi < v / hi) { hi <<= 1u; }// hi >= sqrt(v); hi <= 2^32, no overflow
    uint64_t lo = hi >> 1u;           // lo < sqrt(v)
    while (lo + 1u < hi) {
        const uint64_t mid = lo + (hi - lo) / 2u;
        // mid >= sqrt(v)  <=>  mid*mid >= v, evaluated without overflow:
        // mid > v/mid implies mid*mid > v; equality needs the exact product
        // (mid <= 2^32 here, so mid*mid fits uint64).
        const uint64_t q = v / mid;
        if (mid > q || (mid == q && mid * mid >= v)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return hi;
}

// Prove the flattened linear-id envelope: ocr_linear_id() is uint32, so the
// TOTAL thread count total_x * total_y must fit uint32 or the plan is refused.
[[nodiscard]] Status check_linear_envelope(uint64_t total_x, uint64_t total_y) {
    uint64_t total = 0;
    if (!checked_mul(total_x, total_y, total) || total > UINT32_MAX) {
        return Status::error(StatusCode::invalid_argument,
                             "total thread count exceeds the uint32 linear-id envelope; chunk the work",
                             "total_threads");
    }
    return Status::success();
}

}// namespace

[[nodiscard]] Result<LaunchPlan1D> plan_threads(uint64_t count, uint32_t block) {
    using R = Result<LaunchPlan1D>;
    if (block == 0u || block > 1024u || (block % 32u) != 0u) {
        return R::failure(StatusCode::invalid_argument,
                          "block size must be a multiple of 32 in [32, 1024]", "block");
    }
    LaunchPlan1D plan;
    plan.block = block;
    plan.work_items = count;
    if (count == 0u) { return R::success(plan); }// empty: dispatch nothing
    if (count > UINT32_MAX) {
        return R::failure(StatusCode::invalid_argument,
                          "work item count exceeds uint32; chunk the launch (plan_row_chunks)", "count");
    }
    const uint64_t groups = ceil_div(count, block);
    const uint64_t padded = groups * block;// <= UINT32_MAX + 1023, fits uint64
    if (groups <= kMaxGroupsPerDim) {
        auto tx = checked_u32(padded, "dispatch.x");
        if (!tx.ok()) { return R::failure(tx.status); }
        plan.total_x = *tx.value;
        plan.total_y = 1u;
        plan.groups_x = static_cast<uint32_t>(groups);
        plan.groups_y = 1u;
        return R::success(plan);
    }
    // Flatten into a 2D grid: x carries `cols` block-columns, y carries the
    // remaining block rows. The uint32 linear-id envelope requires
    //   (cols*block) * ceil(groups/cols) <= UINT32_MAX.
    // Using ceil(groups/cols) <= groups/cols + 1, the sufficient bound
    //   groups*block + cols*block <= UINT32_MAX
    // gives an exact, non-heuristic column budget:
    //   cols <= UINT32_MAX/block - groups.
    // Start balanced at ceil(sqrt(groups)) and clamp to that budget; if even
    // the minimum (cols = ceil(sqrt)) cannot fit, the launch is refused.
    const uint64_t max_total_blocks = UINT32_MAX / block;
    if (groups > max_total_blocks) {
        return R::failure(StatusCode::invalid_argument,
                          "launch exceeds the uint32 linear-id envelope; chunk the work", "count");
    }
    const uint64_t col_budget = max_total_blocks - groups;
    uint64_t cols = ceil_sqrt(groups);
    if (cols > col_budget) { cols = col_budget; }
    if (cols == 0u) {
        return R::failure(StatusCode::invalid_argument,
                          "launch exceeds the uint32 linear-id envelope; chunk the work", "count");
    }
    const uint64_t rows = ceil_div(groups, cols);
    if (cols > kMaxGroupsPerDim || rows > kMaxGroupsPerDim) {
        return R::failure(StatusCode::invalid_argument,
                          "launch exceeds the portable 2D group envelope; chunk the work", "count");
    }
    uint64_t total_x = 0;
    if (!checked_mul(cols, block, total_x)) {
        return R::failure(StatusCode::invalid_argument, "dispatch.x overflows uint64", "dispatch.x");
    }
    auto env = check_linear_envelope(total_x, rows);
    if (!env.ok()) { return R::failure(env); }
    auto tx = checked_u32(total_x, "dispatch.x");
    if (!tx.ok()) { return R::failure(tx.status); }
    auto ty = checked_u32(rows, "dispatch.y");
    if (!ty.ok()) { return R::failure(ty.status); }
    plan.total_x = *tx.value;
    plan.total_y = *ty.value;
    plan.groups_x = static_cast<uint32_t>(cols);
    plan.groups_y = static_cast<uint32_t>(rows);
    return R::success(plan);
}

[[nodiscard]] Result<LaunchPlan2D> plan_threads_2d(uint64_t rows, uint64_t cols,
                                                   uint32_t block_x, uint32_t block_y) {
    using R = Result<LaunchPlan2D>;
    if (block_x == 0u || block_y == 0u) {
        return R::failure(StatusCode::invalid_argument, "block dimensions must be nonzero", "block");
    }
    const uint64_t threads = static_cast<uint64_t>(block_x) * block_y;
    if (threads > 1024u || (threads % 32u) != 0u) {
        return R::failure(StatusCode::invalid_argument,
                          "block thread count must be a multiple of 32 in [32, 1024]", "block");
    }
    LaunchPlan2D plan;
    plan.block_x = block_x;
    plan.block_y = block_y;
    plan.rows = rows;
    plan.cols = cols;
    if (rows == 0u || cols == 0u) { return R::success(plan); }// empty
    auto gx = ceil_div_u32(cols, block_x, "grid.x groups");
    if (!gx.ok()) { return R::failure(gx.status); }
    auto gy = ceil_div_u32(rows, block_y, "grid.y groups");
    if (!gy.ok()) { return R::failure(gy.status); }
    if (*gx.value > kMaxGroupsPerDim) {
        return R::failure(StatusCode::invalid_argument,
                          "column count exceeds the portable group envelope; chunk columns", "cols");
    }
    uint64_t padded_cols = 0, padded_rows = 0;
    if (!checked_mul(static_cast<uint64_t>(*gx.value), block_x, padded_cols) ||
        !checked_mul(static_cast<uint64_t>(*gy.value), block_y, padded_rows)) {
        return R::failure(StatusCode::invalid_argument, "padded 2D extent overflows uint64", "extent");
    }
    if (*gy.value <= kMaxGroupsPerDim) {
        auto env = check_linear_envelope(padded_cols, padded_rows);
        if (!env.ok()) { return R::failure(env); }
        auto tx = checked_u32(padded_cols, "dispatch.x");
        if (!tx.ok()) { return R::failure(tx.status); }
        auto ty = checked_u32(padded_rows, "dispatch.y");
        if (!ty.ok()) { return R::failure(ty.status); }
        plan.total_x = *tx.value;
        plan.total_y = *ty.value;
        plan.groups_x = *gx.value;
        plan.groups_y = *gy.value;
        plan.padded_cols = *tx.value;
        plan.flattened_y = false;
        return R::success(plan);
    }
    // Flatten the row axis into x: the linear id spans padded_cols columns per
    // row. fold block-columns move into x, the remaining row groups into y.
    // BOTH resulting group counts must fit the cap, and the total thread count
    // must fit the uint32 linear-id envelope.
    const uint64_t row_groups = *gy.value;
    const uint64_t fold = ceil_sqrt(row_groups);
    const uint64_t y_groups = ceil_div(row_groups, fold);
    uint64_t x_groups = 0, total_x = 0, total_y = 0;
    if (!checked_mul(static_cast<uint64_t>(*gx.value), fold, x_groups) ||
        !checked_mul(padded_cols, fold, total_x) ||
        !checked_mul(y_groups, block_y, total_y)) {
        return R::failure(StatusCode::invalid_argument, "flattened 2D extent overflows uint64", "extent");
    }
    if (x_groups > kMaxGroupsPerDim || y_groups > kMaxGroupsPerDim) {
        return R::failure(StatusCode::invalid_argument,
                          "flattened 2D launch exceeds the portable group envelope; chunk rows/cols", "extent");
    }
    auto env = check_linear_envelope(total_x, total_y);
    if (!env.ok()) { return R::failure(env); }
    if (total_x % block_x != 0u) {
        return R::failure(StatusCode::internal_error, "flattened dispatch.x is not block-aligned", "dispatch.x");
    }
    auto tx = checked_u32(total_x, "dispatch.x");
    if (!tx.ok()) { return R::failure(tx.status); }
    auto ty = checked_u32(total_y, "dispatch.y");
    if (!ty.ok()) { return R::failure(ty.status); }
    plan.total_x = *tx.value;
    plan.total_y = *ty.value;
    plan.groups_x = static_cast<uint32_t>(x_groups);
    plan.groups_y = static_cast<uint32_t>(y_groups);
    plan.padded_cols = static_cast<uint32_t>(padded_cols);
    plan.flattened_y = true;
    return R::success(plan);
}

[[nodiscard]] Result<RowChunkPlan> plan_row_chunks(uint64_t rows, uint32_t max_rows) {
    using R = Result<RowChunkPlan>;
    if (max_rows == 0u) {
        return R::failure(StatusCode::invalid_argument, "max_rows must be nonzero", "max_rows");
    }
    RowChunkPlan plan;
    plan.rows = rows;
    plan.max_rows = max_rows;
    plan.chunk_count = ceil_div(rows, max_rows);
    return R::success(plan);
}

[[nodiscard]] Result<std::pair<uint64_t, uint64_t>> RowChunkPlan::chunk_range(uint64_t index) const {
    using R = Result<std::pair<uint64_t, uint64_t>>;
    if (index >= chunk_count) {
        return R::failure(StatusCode::invalid_argument, "chunk index out of range", "index");
    }
    const uint64_t begin = index * max_rows;
    uint64_t end = 0;
    if (!checked_add(begin, max_rows, end) || end > rows) { end = rows; }
    return R::success(std::make_pair(begin, end));
}

[[nodiscard]] Status check_shared_memory_budget(uint64_t bytes) {
    if (bytes > kPortableSharedMemoryBytes) {
        return Status::error(StatusCode::invalid_argument,
                             "shared memory exceeds the portable 32 KiB floor", "shared");
    }
    return Status::success();
}

}// namespace luisa::ocr