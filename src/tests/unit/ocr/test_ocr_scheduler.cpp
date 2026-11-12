// Test for ocr/runtime/scheduler (Phase 6, host-only).
// This test covers:
// - token-budgeted vision/prefill batch packing (patch + region caps);
// - descending patch-count sort with stable tie-breaking;
// - oversized-crop refusal (never silently split);
// - decode batch packing under max_decode_batch;
// - restore to original page/reading order;
// - per-page output-token budget preflight;
// - detector-output cache (content/geometry key, FIFO eviction, retry hit);
// - deterministic FNV hashes; empty inputs; invalid limits.
#include "ut/ut.hpp"

#include <luisa/core/stl/vector.h>

#include "ocr/runtime/scheduler.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] RegionTask task(uint32_t id, uint32_t patches, uint32_t page = 0, uint32_t order = 0) {
    RegionTask t;
    t.region_id = id;
    t.page_index = page;
    t.reading_order = order;
    t.approx_patches = patches;
    t.approx_tokens = patches;
    t.source_box = output::DbRect{0, 0, 10, 10};
    return t;
}

[[nodiscard]] SchedulerLimits limits(uint32_t max_patches, uint32_t max_regions, uint32_t max_decode) {
    SchedulerLimits l;
    l.max_patches_per_vision_batch = max_patches;
    l.max_regions_per_batch = max_regions;
    l.max_decode_batch = max_decode;
    return l;
}

}// namespace

void test_vision_batch_packing() {
    RegionScheduler sched{limits(100, 3, 1)};
    // Patch counts: 60, 50, 40, 30, 20. Sorted desc: 60,50,40,30,20.
    // FFD under cap 100 / 3 regions: [60,40], [50,30,20].
    luisa::vector<RegionTask> tasks{task(0, 60), task(1, 50), task(2, 40), task(3, 30), task(4, 20)};
    auto r = sched.plan_vision_batches(tasks);
    expect(r.ok());
    expect(r.value->size() == 2u);
    // Every batch respects both caps.
    for (const auto &b : *r.value) {
        expect(b.region_indices.size() <= 3u);
        expect(b.total_patches <= 100u);
    }
    // All regions covered exactly once.
    luisa::vector<uint32_t> seen;
    for (const auto &b : *r.value) {
        for (auto idx : b.region_indices) { seen.push_back(idx); }
    }
    expect(seen.size() == 5u);
    // Descending patch order within the plan: first batch holds the largest.
    expect((*r.value)[0].total_patches >= (*r.value)[1].total_patches);
}

void test_vision_batch_stable_tie() {
    RegionScheduler sched{limits(1000, 8, 1)};
    // Equal patch counts -> stable region_id ascending order.
    luisa::vector<RegionTask> tasks{task(5, 10), task(2, 10), task(9, 10), task(1, 10)};
    auto r = sched.plan_vision_batches(tasks);
    expect(r.ok());
    expect(r.value->size() == 1u);
    const auto &idx = (*r.value)[0].region_indices;
    expect(idx.size() == 4u);
    // region_id ascending: tasks[3]=id1, tasks[1]=id2, tasks[0]=id5, tasks[2]=id9.
    expect(idx[0] == 3u && idx[1] == 1u && idx[2] == 0u && idx[3] == 2u);
}

void test_vision_batch_oversized_refused() {
    RegionScheduler sched{limits(50, 4, 1)};
    luisa::vector<RegionTask> tasks{task(0, 60)};// exceeds per-batch patch cap
    auto r = sched.plan_vision_batches(tasks);
    expect(!r.ok());
    expect(r.status.code == StatusCode::out_of_memory);
}

void test_vision_batch_empty() {
    RegionScheduler sched{limits(100, 4, 1)};
    auto r = sched.plan_vision_batches({});
    expect(r.ok());
    expect(r.value->empty());
}

void test_decode_batches() {
    RegionScheduler sched{limits(100, 4, 2)};
    luisa::vector<uint32_t> ids{0, 1, 2, 3, 4};
    auto r = sched.plan_decode_batches(ids);
    expect(r.ok());
    expect(r.value->size() == 3u);// [0,1],[2,3],[4]
    expect((*r.value)[0].region_indices.size() == 2u);
    expect((*r.value)[2].region_indices.size() == 1u);
    // Order preserved.
    expect((*r.value)[0].region_indices[0] == 0u);
    expect((*r.value)[1].region_indices[0] == 2u);
}

void test_restore_order() {
    // Tasks out of reading order; restore by (page, reading_order, region_id).
    luisa::vector<RegionTask> tasks{
        task(0, 10, 0, 2),
        task(1, 10, 0, 0),
        task(2, 10, 1, 0),
        task(3, 10, 0, 1),
    };
    auto r = RegionScheduler::restore_order(tasks);
    expect(r.ok());
    expect(r.value->size() == 4u);
    // page0 order0 (idx1), page0 order1 (idx3), page0 order2 (idx0), page1 (idx2).
    expect((*r.value)[0] == 1u);
    expect((*r.value)[1] == 3u);
    expect((*r.value)[2] == 0u);
    expect((*r.value)[3] == 2u);
}

void test_page_output_budget() {
    SchedulerLimits l = limits(100, 4, 1);
    l.max_output_tokens_per_page = 8;
    RegionScheduler sched{l};
    expect(sched.preflight_page_output_budget(8u).ok());
    expect(!sched.preflight_page_output_budget(9u).ok());
    expect(sched.preflight_page_output_budget(0u).ok());
}

void test_cache_hit_and_eviction() {
    LayoutCache cache{2};
    LayoutCacheKey k1{1, 10};
    LayoutCacheKey k2{2, 20};
    LayoutCacheKey k3{3, 30};
    CachedLayout l1;
    l1.regions.push_back(task(0, 5));
    expect(cache.store(k1, l1) == 1u);
    expect(cache.find(k1) != nullptr);
    expect(cache.find(k2) == nullptr);
    // Geometry mismatch -> miss even with same content hash.
    LayoutCacheKey k1g{1, 99};
    expect(cache.find(k1g) == nullptr);
    expect(cache.store(k2, l1) == 2u);
    expect(cache.store(k3, l1) == 2u);// evicts k1 (FIFO)
    expect(cache.find(k1) == nullptr);
    expect(cache.find(k2) != nullptr);
    expect(cache.find(k3) != nullptr);
    // Overwrite existing key does not grow.
    expect(cache.store(k2, l1) == 2u);
    cache.clear();
    expect(cache.size() == 0u);
}

void test_hashes_deterministic() {
    luisa::vector<uint8_t> bytes{1, 2, 3, 4};
    const auto h1 = layout_hash_bytes(bytes);
    const auto h2 = layout_hash_bytes(bytes);
    expect(h1 == h2);
    luisa::vector<uint8_t> other{1, 2, 3, 5};
    expect(layout_hash_bytes(other) != h1);
    const auto g1 = layout_hash_geometry(64, 64, 16, 16, 0);
    const auto g2 = layout_hash_geometry(64, 64, 16, 16, 0);
    expect(g1 == g2);
    expect(layout_hash_geometry(64, 64, 16, 16, 1) != g1);
}

void test_invalid_limits() {
    SchedulerLimits l = limits(0, 4, 1);
    expect(!l.validate().ok());
    RegionScheduler sched{l};
    luisa::vector<RegionTask> tasks{task(0, 10)};
    expect(!sched.plan_vision_batches(tasks).ok());
}

static auto test_ocr_scheduler_registration = [] {
    "scheduler_vision_batch_packing"_test = [] { test_vision_batch_packing(); };
    "scheduler_vision_batch_stable_tie"_test = [] { test_vision_batch_stable_tie(); };
    "scheduler_vision_batch_oversized_refused"_test = [] { test_vision_batch_oversized_refused(); };
    "scheduler_vision_batch_empty"_test = [] { test_vision_batch_empty(); };
    "scheduler_decode_batches"_test = [] { test_decode_batches(); };
    "scheduler_restore_order"_test = [] { test_restore_order(); };
    "scheduler_page_output_budget"_test = [] { test_page_output_budget(); };
    "scheduler_cache_hit_and_eviction"_test = [] { test_cache_hit_and_eviction(); };
    "scheduler_hashes_deterministic"_test = [] { test_hashes_deterministic(); };
    "scheduler_invalid_limits"_test = [] { test_invalid_limits(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}