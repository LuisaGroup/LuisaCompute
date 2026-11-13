// OCR engine -- layout/region scheduling implementation (Phase 6).
// Pure host planning + bounded detector cache. See the header for the contract.
#include "ocr/runtime/scheduler.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace luisa::ocr {

[[nodiscard]] luisa::string_view layout_block_type_name(LayoutBlockType type) noexcept {
    switch (type) {
        case LayoutBlockType::text: return "text";
        case LayoutBlockType::title: return "title";
        case LayoutBlockType::table: return "table";
        case LayoutBlockType::formula: return "formula";
        case LayoutBlockType::figure: return "figure";
        case LayoutBlockType::header: return "header";
        case LayoutBlockType::footer: return "footer";
        case LayoutBlockType::list: return "list";
        case LayoutBlockType::other: return "other";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// SchedulerLimits
// ---------------------------------------------------------------------------

[[nodiscard]] Status SchedulerLimits::validate() const {
    if (max_patches_per_vision_batch == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "max_patches_per_vision_batch must be nonzero", "max_patches_per_vision_batch");
    }
    if (max_regions_per_batch == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "max_regions_per_batch must be nonzero", "max_regions_per_batch");
    }
    if (max_decode_batch == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_decode_batch must be nonzero", "max_decode_batch");
    }
    if (max_output_tokens_per_region == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "max_output_tokens_per_region must be nonzero", "max_output_tokens_per_region");
    }
    if (max_output_tokens_per_page == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "max_output_tokens_per_page must be nonzero", "max_output_tokens_per_page");
    }
    if (max_wall_time_ms < 0.0 || !std::isfinite(max_wall_time_ms)) {
        return Status::error(StatusCode::invalid_argument,
                             "max_wall_time_ms must be finite and >= 0", "max_wall_time_ms");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// RegionScheduler
// ---------------------------------------------------------------------------

[[nodiscard]] Result<luisa::vector<RegionBatch>> RegionScheduler::plan_vision_batches(
    luisa::span<const RegionTask> tasks) const {
    using R = Result<luisa::vector<RegionBatch>>;
    if (auto s = _limits.validate(); !s.ok()) { return R::failure(s); }
    if (tasks.empty()) { return R::success({}); }

    // Stable sort by approx_patches DESCENDING, tie-break on region_id ASC so
    // the plan is a total order and repeat runs are identical.
    luisa::vector<uint32_t> order(tasks.size());
    for (uint32_t i = 0; i < tasks.size(); i++) { order[i] = i; }
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (tasks[a].approx_patches != tasks[b].approx_patches) {
            return tasks[a].approx_patches > tasks[b].approx_patches;
        }
        return tasks[a].region_id < tasks[b].region_id;
    });

    // A single oversized crop is refused, never silently split: the caller must
    // shrink the crop or raise the cap deliberately (plan §7 memory rule).
    for (auto idx : order) {
        if (tasks[idx].approx_patches > _limits.max_patches_per_vision_batch) {
            return R::failure(StatusCode::out_of_memory,
                              "a region's patch count exceeds max_patches_per_vision_batch; "
                              "shrink the crop or raise the cap",
                              "max_patches_per_vision_batch");
        }
    }

    // First-fit-decreasing packing under BOTH the patch and region caps.
    luisa::vector<RegionBatch> batches;
    for (auto idx : order) {
        const uint64_t patches = tasks[idx].approx_patches;
        bool placed = false;
        for (auto &batch : batches) {
            if (batch.region_indices.size() >= _limits.max_regions_per_batch) { continue; }
            if (batch.total_patches + patches > _limits.max_patches_per_vision_batch) { continue; }
            batch.region_indices.push_back(idx);
            batch.total_patches += patches;
            placed = true;
            break;
        }
        if (!placed) {
            RegionBatch batch;
            batch.region_indices.push_back(idx);
            batch.total_patches = patches;
            batches.push_back(std::move(batch));
        }
    }
    return R::success(std::move(batches));
}

[[nodiscard]] Result<luisa::vector<RegionBatch>> RegionScheduler::plan_decode_batches(
    luisa::span<const uint32_t> region_indices) const {
    using R = Result<luisa::vector<RegionBatch>>;
    if (auto s = _limits.validate(); !s.ok()) { return R::failure(s); }
    luisa::vector<RegionBatch> batches;
    for (size_t i = 0; i < region_indices.size(); i += _limits.max_decode_batch) {
        RegionBatch batch;
        const size_t end = std::min(region_indices.size(), i + _limits.max_decode_batch);
        for (size_t j = i; j < end; j++) { batch.region_indices.push_back(region_indices[j]); }
        batches.push_back(std::move(batch));
    }
    return R::success(std::move(batches));
}

[[nodiscard]] Result<luisa::vector<uint32_t>> RegionScheduler::restore_order(
    luisa::span<const RegionTask> tasks) {
    using R = Result<luisa::vector<uint32_t>>;
    luisa::vector<uint32_t> order(tasks.size());
    for (uint32_t i = 0; i < tasks.size(); i++) { order[i] = i; }
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (tasks[a].page_index != tasks[b].page_index) { return tasks[a].page_index < tasks[b].page_index; }
        if (tasks[a].reading_order != tasks[b].reading_order) { return tasks[a].reading_order < tasks[b].reading_order; }
        return tasks[a].region_id < tasks[b].region_id;
    });
    return R::success(std::move(order));
}

[[nodiscard]] luisa::vector<uint32_t> layout_sort_pages_by_index(
    luisa::span<const LayoutPageResult> pages) {
    luisa::vector<uint32_t> order(pages.size());
    for (uint32_t i = 0; i < pages.size(); i++) { order[i] = i; }
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return pages[a].page_index < pages[b].page_index;
    });
    return order;
}

[[nodiscard]] Status RegionScheduler::preflight_page_output_budget(uint32_t region_count) const {
    if (region_count == 0u) { return Status::success(); }
    // The page budget must hold at least one output token per region; otherwise
    // even a minimal decode cannot complete. This is a conservative preflight,
    // not a claim about actual generation length.
    if (static_cast<uint64_t>(region_count) > _limits.max_output_tokens_per_page) {
        return Status::error(StatusCode::out_of_memory,
                             "region count exceeds the per-page output-token budget",
                             "max_output_tokens_per_page");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// LayoutCache
// ---------------------------------------------------------------------------

uint32_t LayoutCache::store(LayoutCacheKey key, CachedLayout layout) {
    for (auto &entry : _entries) {
        if (entry.key == key) {
            entry.layout = std::move(layout);
            return static_cast<uint32_t>(_entries.size());
        }
    }
    if (_entries.size() >= _max_entries) {
        _entries.erase(_entries.begin());// FIFO eviction
    }
    _entries.push_back(Entry{key, std::move(layout)});
    return static_cast<uint32_t>(_entries.size());
}

[[nodiscard]] const CachedLayout *LayoutCache::find(LayoutCacheKey key) const noexcept {
    for (const auto &entry : _entries) {
        if (entry.key == key) { return &entry.layout; }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Hash helpers
// ---------------------------------------------------------------------------

[[nodiscard]] uint64_t layout_hash_bytes(luisa::span<const uint8_t> bytes) noexcept {
    uint64_t hash = 14695981039346656037ull;// FNV offset basis
    for (auto b : bytes) {
        hash ^= static_cast<uint64_t>(b);
        hash *= 1099511628211ull;// FNV prime
    }
    return hash;
}

[[nodiscard]] uint64_t layout_hash_geometry(uint32_t source_width, uint32_t source_height,
                                            uint32_t det_width, uint32_t det_height,
                                            uint32_t page_index) noexcept {
    // Pack the five 32-bit fields little-endian into a byte span and hash.
    uint8_t bytes[20];
    const uint32_t fields[5] = {source_width, source_height, det_width, det_height, page_index};
    for (uint32_t i = 0; i < 5u; i++) {
        bytes[i * 4u + 0u] = static_cast<uint8_t>(fields[i] & 0xFFu);
        bytes[i * 4u + 1u] = static_cast<uint8_t>((fields[i] >> 8u) & 0xFFu);
        bytes[i * 4u + 2u] = static_cast<uint8_t>((fields[i] >> 16u) & 0xFFu);
        bytes[i * 4u + 3u] = static_cast<uint8_t>((fields[i] >> 24u) & 0xFFu);
    }
    return layout_hash_bytes(luisa::span<const uint8_t>{bytes, 20});
}

}// namespace luisa::ocr