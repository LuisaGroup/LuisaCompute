// Test for ocr/runtime/dispatch.h + tensor.h (Phase 2, host-only).
// Covers: TOTAL-thread plans, zero/ragged sizes, >65535-group flattening
// (1D and 2D), uint32 envelope overflow refusal, row chunking incl. tails,
// shared-memory budget guard; tensor checked math, strides, view ranges,
// overlap detection, storage/compute dtype separation.
#include "ut/ut.hpp"

#include "ocr/runtime/dispatch.h"
#include "ocr/runtime/tensor.h"
#include "ocr/runtime/workspace.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_plan_threads_basic() {
    auto zero = plan_threads(0);
    expect(zero.ok());
    expect(zero.value->empty());
    auto one = plan_threads(1);
    expect(one.ok());
    expect(one.value->total_x == 256u);// padded to block
    expect(one.value->groups_x == 1u);
    expect(!one.value->is_2d());
    auto ragged = plan_threads(1027);
    expect(ragged.ok());
    expect(ragged.value->total_x == 1280u);// ceil(1027/256)*256
    expect(ragged.value->work_items == 1027u);
    // Every requested item is covered: total threads >= count.
    expect(ragged.value->total_x >= 1027u);
    // Invalid block sizes refused.
    expect(!plan_threads(10, 0).ok());
    expect(!plan_threads(10, 100).ok());// not multiple of 32
    expect(!plan_threads(10, 2048).ok());
    // uint32 envelope: count > UINT32_MAX refused (must chunk).
    expect(!plan_threads(static_cast<uint64_t>(UINT32_MAX) + 1u).ok());
}

void test_plan_threads_flatten() {
    // 256 * 70000 items -> 70000 groups > 65535: must flatten to 2D.
    const uint64_t count = 256ull * 70000ull;
    auto plan = plan_threads(count);
    expect(plan.ok());
    expect(plan.value->is_2d());
    expect(plan.value->groups_x <= kMaxGroupsPerDim);
    expect(plan.value->groups_y <= kMaxGroupsPerDim);
    // Full coverage: total threads >= count, x stays block-multiple.
    const uint64_t total = static_cast<uint64_t>(plan.value->total_x) * plan.value->total_y;
    expect(total >= count);
    expect(plan.value->total_x % 256u == 0u);
    // Exactly at the cap: no flattening needed.
    auto at_cap = plan_threads(256ull * 65535ull);
    expect(at_cap.ok());
    expect(!at_cap.value->is_2d());
    expect(at_cap.value->groups_x == 65535u);
}

void test_plan_2d() {
    auto zero = plan_threads_2d(0, 10);
    expect(zero.ok());
    expect(zero.value->empty());
    auto small = plan_threads_2d(37, 41);
    expect(small.ok());
    expect(!small.value->flattened_y);
    expect(small.value->total_x == 48u);// ceil(41/16)*16
    expect(small.value->total_y == 48u);// ceil(37/16)*16
    // Rows needing > 65535 row-groups: 16 * 70000 rows.
    auto big = plan_threads_2d(16ull * 70000ull, 32);
    expect(big.ok());
    expect(big.value->flattened_y);
    expect(big.value->groups_x <= kMaxGroupsPerDim);
    expect(big.value->groups_y <= kMaxGroupsPerDim);
    const uint64_t total = static_cast<uint64_t>(big.value->total_x) * big.value->total_y;
    expect(total >= 16ull * 70000ull * 32ull);
    // Column count beyond the envelope is refused, never clamped.
    expect(!plan_threads_2d(4, 16ull * 70000ull).ok());
}

void test_row_chunks() {
    auto plan = plan_row_chunks(1027, 256);
    expect(plan.ok());
    expect(plan.value->chunk_count == 5u);
    uint64_t covered = 0;
    for (uint64_t i = 0; i < plan.value->chunk_count; i++) {
        auto r = plan.value->chunk_range(i);
        expect(r.ok());
        expect(r.value->first == i * 256u);
        covered += r.value->second - r.value->first;
    }
    expect(covered == 1027u);// no dropped rows
    auto tail = plan.value->chunk_range(4);
    expect(tail.value->second == 1027u);
    expect(!plan.value->chunk_range(5).ok());
    expect(!plan_row_chunks(10, 0).ok());
    auto empty = plan_row_chunks(0, 256);
    expect(empty.ok());
    expect(empty.value->empty());
    // Overflow-safe tail: rows near UINT64_MAX clamp instead of failing.
    auto huge = plan_row_chunks(UINT64_MAX - 3u, 1024u);
    expect(huge.ok());
    auto last = huge.value->chunk_range(huge.value->chunk_count - 1u);
    expect(last.ok());
    expect(last.value->second == UINT64_MAX - 3u);
}

void test_checked_u32_shared() {
    expect(checked_u32(UINT32_MAX, "x").ok());
    expect(!checked_u32(static_cast<uint64_t>(UINT32_MAX) + 1u, "x").ok());
    expect(check_shared_memory_budget(32u * 1024u).ok());
    expect(!check_shared_memory_budget(32u * 1024u + 1u).ok());
}

void test_tensor_desc() {
    auto m = make_matrix_desc(37, 41);
    expect(m.ok());
    expect(m.value->element_count().ok());
    expect(*m.value->element_count().value == 37ull * 41ull);
    expect(m.value->is_contiguous());
    const uint64_t idx23[] = {2, 3};
    auto off = m.value->linear_offset(idx23);
    expect(off.ok());
    expect(*off.value == 2ull * 41ull + 3ull);
    const uint64_t idx370[] = {37, 0};
    expect(!m.value->linear_offset(idx370).ok());// OOB
    const uint64_t idx1[] = {1};
    expect(!m.value->linear_offset(idx1).ok());// wrong rank
    // Overflow shape refused.
    TensorDesc big;
    big.shape = {UINT64_MAX / 2u, 4};
    expect(!big.element_count().ok());
    // Non-contiguous strides validate; extent checked.
    TensorDesc s;
    s.shape = {4, 4};
    s.strides = {8, 1};// padded rows
    expect(s.normalize().ok());
    expect(s.validate().ok());
    expect(!s.is_contiguous());
    const uint64_t idx33[] = {3, 3};
    auto so = s.linear_offset(idx33);
    expect(so.ok());
    expect(*so.value == 27u);
    // Rank cap.
    TensorDesc r;
    r.shape.assign(9, 1);
    expect(!r.normalize().ok());
    // Empty tensor.
    TensorDesc e;
    e.shape = {0, 5};
    expect(e.normalize().ok());
    expect(e.validate().ok());
    expect(*e.element_count().value == 0u);
}

void test_buffer_view_desc() {
    auto m = make_matrix_desc(4, 4);
    BufferViewDesc v;
    v.tensor = *m.value;
    v.byte_offset = 256;
    expect(v.validate_within(256 + 64).ok());
    expect(!v.validate_within(256 + 63).ok());
    // Misaligned offset refused.
    v.byte_offset = 254;
    expect(!v.validate_within(4096).ok());
    v.byte_offset = 256;
    BufferViewDesc w = v;
    w.byte_offset = 256 + 32;
    auto ov = v.overlaps(w);
    expect(ov.ok());
    expect(*ov.value);
    w.byte_offset = 256 + 64;
    expect(!*v.overlaps(w).value);// adjacent, not overlapping
    // Empty views never overlap.
    BufferViewDesc empty;
    empty.tensor.shape = {0, 4};
    expect(empty.tensor.normalize().ok());
    expect(!*v.overlaps(empty).value);
    // Storage vs compute dtype sizes are distinct.
    expect(tensor_dtype_size(TensorDType::f32) == 4u);
    expect(tensor_dtype_size(TensorDType::f16) == 2u);
    expect(!is_f32_compute_dtype(TensorDType::f16));
    auto bytes = tensor_byte_size(TensorDType::f32, UINT64_MAX / 4u + 1u);
    expect(!bytes.ok());
}

// Adversarial envelope tests: uint64 overflow safety + uint32 linear-id proof.
void test_dispatch_adversarial() {
    // ceil-div overflow: counts near UINT64_MAX must be REJECTED (count >
    // UINT32_MAX), never wrap into a small plan.
    expect(!plan_threads(UINT64_MAX).ok());
    expect(!plan_threads(UINT64_MAX - 1u).ok());
    expect(!plan_threads(static_cast<uint64_t>(UINT32_MAX) + 1u).ok());
    // Largest valid count: UINT32_MAX. Padded total = ceil(UINT32_MAX/256)*256
    // = 2^32 -> exceeds uint32 total_x -> must reject (envelope), not wrap.
    expect(!plan_threads(UINT32_MAX).ok());
    // 0xFFFFFF00 also pads to 2^32 (16777216 groups * 256) -> reject.
    expect(!plan_threads(0xFFFFFF00ull).ok());
    // 0xFFFFFE00 needs 16777215 groups; the uint32 envelope forces cols<=257,
    // which pushes rows past the 65535 group cap -> correctly rejected.
    expect(!plan_threads(0xFFFFFE00ull).ok());
    // Largest accepted count (verified by scan): 0xFFFEFF00 -> 16776959
    // groups, cols=256, rows=65535 (at the cap), total=65536*65535<=UINT32_MAX.
    auto ok_max = plan_threads(0xFFFEFF00ull);
    expect(ok_max.ok());
    expect(ok_max.value->is_2d());
    const uint64_t total = static_cast<uint64_t>(ok_max.value->total_x) * ok_max.value->total_y;
    expect(total >= 0xFFFEFF00ull);
    expect(total <= UINT32_MAX);// linear-id envelope proven
    expect(ok_max.value->groups_x <= kMaxGroupsPerDim);
    expect(ok_max.value->groups_y <= kMaxGroupsPerDim);
    // One block past it (0xFFFF0000) is rejected: the sqrt split's padding
    // pushes the envelope past uint32.
    expect(!plan_threads(0xFFFF0000ull).ok());
    // 2D: rows/cols near UINT64_MAX must reject without wrapping.
    expect(!plan_threads_2d(UINT64_MAX, 4).ok());
    expect(!plan_threads_2d(4, UINT64_MAX).ok());
    expect(!plan_threads_2d(UINT64_MAX / 2, UINT64_MAX / 2).ok());
    // Large gx*fold: cols just under the x-group cap, rows forcing flattening.
    // gx = 65535 (cols = 65535*16), rows = 16*70000 -> fold must keep
    // gx*fold <= 65535: with gx already at the cap, fold must be 1 -> the
    // y-groups stay > cap -> REJECT (never silently exceed the group cap).
    expect(!plan_threads_2d(16ull * 70000ull, 16ull * 65535ull).ok());
    // A flattenable case with gx>1: verify groups_x = gx*fold <= cap and the
    // linear envelope holds.
    auto flat = plan_threads_2d(16ull * 70000ull, 48);// gx=3
    expect(flat.ok());
    expect(flat.value->flattened_y);
    expect(flat.value->groups_x <= kMaxGroupsPerDim);
    expect(flat.value->groups_y <= kMaxGroupsPerDim);
    const uint64_t ftotal = static_cast<uint64_t>(flat.value->total_x) * flat.value->total_y;
    expect(ftotal <= UINT32_MAX);
    expect(ftotal >= 16ull * 70000ull * 48ull);
    expect(flat.value->padded_cols == 48u);
    // Row chunking near UINT64_MAX: no overflow, full coverage of the tail.
    auto huge = plan_row_chunks(UINT64_MAX, 1024u);
    expect(huge.ok());
    expect(huge.value->chunk_count == UINT64_MAX / 1024u + 1u);
    auto last = huge.value->chunk_range(huge.value->chunk_count - 1u);
    expect(last.ok());
    expect(last.value->second == UINT64_MAX);
    auto first = huge.value->chunk_range(0);
    expect(first.ok());
    expect(first.value->first == 0u && first.value->second == 1024u);
}

// S4: strided views reserve their full footprint.
void test_strided_views() {
    // 4x4 logical with row stride 8 (padded rows): extent = 3*8+3+1 = 28.
    TensorDesc t;
    t.shape = {4, 4};
    t.strides = {8, 1};
    expect(t.normalize().ok());
    auto ext = t.strided_extent();
    expect(ext.ok());
    expect(*ext.value == 28u);
    expect(*t.element_count().value == 16u);// logical count differs
    BufferViewDesc v;
    v.tensor = t;
    v.byte_offset = 0;
    auto bytes = v.byte_size();
    expect(bytes.ok());
    expect(*bytes.value == 28u * 4u);// footprint, not 16*4
    expect(v.validate_within(112).ok());
    expect(!v.validate_within(111).ok());// element_count-based check would pass
    // Overlap uses the strided footprint: a view starting at element 20
    // (byte 80) overlaps the padded tail of v even though 20 >= count 16.
    BufferViewDesc w;
    w.tensor = make_matrix_desc(1, 4).value.value();
    w.byte_offset = 80;
    auto ov = v.overlaps(w);
    expect(ov.ok());
    expect(*ov.value);
    // Contiguous view: extent == count.
    auto c = make_matrix_desc(4, 4);
    expect(*c.value->strided_extent().value == 16u);
    // Empty strided view: extent 0.
    TensorDesc e;
    e.shape = {0, 4};
    e.strides = {8, 1};
    expect(*e.strided_extent().value == 0u);
    // Overflowing strides rejected: (3-1)*(UINT64_MAX/2+1) overflows uint64.
    TensorDesc big;
    big.shape = {3, 2};
    big.strides = {UINT64_MAX / 2 + 1u, 1};
    expect(!big.strided_extent().ok());
    // Addition overflow: two large stride terms sum past UINT64_MAX.
    TensorDesc big2;
    big2.shape = {2, 2};
    big2.strides = {UINT64_MAX - 4u, 8};
    expect(!big2.strided_extent().ok());
}

// S7: allocation alias tracking (host-side struct semantics).
void test_workspace_alias_tracking() {
    WorkspaceAllocation a{0, 128, 0, 0};
    WorkspaceAllocation b{0, 128, 0, 1};// same offset, newer epoch
    expect(!a.aliases(b));
    WorkspaceAllocation c{64, 128, 0, 0};// same epoch, intersecting
    expect(a.aliases(c));
    WorkspaceAllocation d{128, 128, 0, 0};// adjacent
    expect(!a.aliases(d));
    WorkspaceAllocation e{0, 128, 1, 0};// different generation
    expect(!a.aliases(e));
}

}// namespace

static auto test_ocr_runtime_registration = [] {
    "dispatch_plan_threads_basic"_test = [] { test_plan_threads_basic(); };
    "dispatch_plan_threads_flatten"_test = [] { test_plan_threads_flatten(); };
    "dispatch_plan_2d"_test = [] { test_plan_2d(); };
    "dispatch_row_chunks"_test = [] { test_row_chunks(); };
    "dispatch_checked_u32_shared"_test = [] { test_checked_u32_shared(); };
    "tensor_desc"_test = [] { test_tensor_desc(); };
    "tensor_buffer_view"_test = [] { test_buffer_view_desc(); };
    "dispatch_adversarial_envelopes"_test = [] { test_dispatch_adversarial(); };
    "tensor_strided_views"_test = [] { test_strided_views(); };
    "workspace_alias_tracking"_test = [] { test_workspace_alias_tracking(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}