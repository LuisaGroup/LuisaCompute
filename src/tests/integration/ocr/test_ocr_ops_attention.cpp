// Test for ocr/ops/ops_attention (Phase 2).
// Covers: reference vs online-softmax A/B; all five mask kinds incl. a
// DeepSeek-style CUSTOM_BLOCK_MASK (visual bidirectional, queries see all
// visual + prior queries); GQA + MHA; ragged q/k lengths; softcap; relative
// bias; contiguous-KV decode with per-batch lengths; fully-masked rows.
// Oracle: independent double host reference. Tolerance 2e-4 + 2e-4*|ref|.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_attention.h"
#include "ocr/runtime/dispatch_adapter.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 2e-4f;
constexpr float kRtol = 2e-4f;

[[nodiscard]] RefAttentionMask to_ref_mask(const AttentionMaskDescriptor &m) {
    RefAttentionMask r;
    r.kind = static_cast<RefMaskKind>(static_cast<uint32_t>(m.kind));
    r.window = m.window;
    r.prefix_len = m.prefix_len;
    r.block_size = m.block_size;
    r.block_mask = m.block_mask;
    r.rel_bias = m.rel_bias;
    return r;
}

struct PrefillCase {
    uint32_t q_len, k_len, q_heads, kv_heads, head_dim;
    float softcap;
    AttentionMaskDescriptor mask;
};

void run_prefill(Device &device, const PrefillCase &c, const char *what) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    const float scale = 1.f / std::sqrt(static_cast<float>(c.head_dim));
    luisa::vector<float> q(static_cast<size_t>(c.q_len) * c.q_heads * c.head_dim);
    luisa::vector<float> k(static_cast<size_t>(c.k_len) * c.kv_heads * c.head_dim);
    luisa::vector<float> v(k.size());
    lcg.fill(q, -1.f, 1.f);
    lcg.fill(k, -1.f, 1.f);
    lcg.fill(v, -1.f, 1.f);
    expect(c.mask.validate(c.q_len, c.k_len, c.q_heads).ok()) << what;
    luisa::vector<float> ref;
    expect(ref_attention(q, k, v, c.q_len, c.k_len, c.q_heads, c.kv_heads, c.head_dim,
                         scale, c.softcap, to_ref_mask(c.mask), ref).ok());
    auto bq = ocr::test::upload(device, stream, luisa::span<const float>{q});
    auto bk = ocr::test::upload(device, stream, luisa::span<const float>{k});
    auto bv = ocr::test::upload(device, stream, luisa::span<const float>{v});
    auto dummy_u = device.create_buffer<uint32_t>(1);
    auto dummy_f = device.create_buffer<float>(1);
    auto mask_words = pack_block_mask_words(c.mask.block_mask);
    auto bmask_storage = c.mask.block_mask.empty()
                             ? Buffer<uint32_t>{}
                             : ocr::test::upload(device, stream, luisa::span<const uint32_t>{mask_words});
    auto bbias_storage = c.mask.rel_bias.empty()
                             ? Buffer<float>{}
                             : ocr::test::upload(device, stream, luisa::span<const float>{c.mask.rel_bias});
    auto bmask = c.mask.block_mask.empty() ? dummy_u.view() : bmask_storage.view();
    auto bbias = c.mask.rel_bias.empty() ? dummy_f.view() : bbias_storage.view();
    auto plan = plan_attention_prefill_dispatch(c.q_len, c.q_heads);
    expect(plan.ok());
    // A/B: reference and online paths compiled and checked SEPARATELY.
    for (int path = 0; path < 2; path++) {
        auto shader = device.compile(path == 0 ? make_attention_prefill_reference_kernel()
                                               : make_attention_prefill_online_kernel());
        expect(shader.compile_ok());
        auto bo = device.create_buffer<float>(q.size());
        OCR_DISPATCH_1D(stream, shader(bq, bk, bv, bo, bmask, bbias, c.q_len, c.k_len, c.q_heads, c.kv_heads,
                         c.head_dim, scale, c.softcap, static_cast<uint32_t>(c.mask.kind),
                         c.mask.window, c.mask.prefix_len, c.mask.block_size,
                         c.mask.rel_bias.empty() ? 0u : 1u), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, q.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, path == 0 ? "attention reference" : "attention online");
    }
}

void test_prefill_masks(Device &device) {
    // Ragged GQA: q_len != k_len exercises the end-aligned causal diagonal.
    {
        PrefillCase c{7, 7, 8, 2, 16, 0.f, {}};
        c.mask.kind = AttentionMaskKind::bidirectional;
        run_prefill(device, c, "bidirectional gqa");
    }
    {
        PrefillCase c{9, 9, 4, 4, 8, 0.f, {}};
        c.mask.kind = AttentionMaskKind::causal;
        run_prefill(device, c, "causal mha");
    }
    {
        PrefillCase c{5, 13, 6, 3, 12, 0.f, {}};// short query tail over long KV
        c.mask.kind = AttentionMaskKind::causal;
        run_prefill(device, c, "causal prefill tail");
    }
    {
        PrefillCase c{11, 11, 4, 2, 16, 0.f, {}};
        c.mask.kind = AttentionMaskKind::windowed;
        c.mask.window = 4;
        run_prefill(device, c, "windowed");
    }
    {
        PrefillCase c{10, 10, 4, 2, 8, 0.f, {}};
        c.mask.kind = AttentionMaskKind::prefix_causal;
        c.mask.prefix_len = 4;
        run_prefill(device, c, "prefix causal");
    }
    {
        // DeepSeek-style: 8 visual tokens (bidirectional among themselves) +
        // 8 query tokens (see ALL visual + prior queries). Block size 4:
        // blocks 0,1 = visual; blocks 2,3 = queries. 4x4 block mask.
        PrefillCase c{16, 16, 4, 2, 8, 0.f, {}};
        c.mask.kind = AttentionMaskKind::custom_block_mask;
        c.mask.block_size = 4;
        c.mask.block_mask.assign(16, 0);
        for (uint32_t qb = 0; qb < 4; qb++) {
            for (uint32_t kb = 0; kb < 4; kb++) {
                const bool visual_q = qb < 2, visual_k = kb < 2;
                uint8_t allow = 0;
                if (visual_q) { allow = visual_k ? 1 : 0; }// visual sees visual only
                else { allow = (visual_k || kb <= qb) ? 1 : 0; }// query: all visual + prior queries
                c.mask.block_mask[static_cast<size_t>(qb) * 4 + kb] = allow;
            }
        }
        run_prefill(device, c, "deepseek custom block");
    }
    {
        // Softcap + relative bias.
        PrefillCase c{6, 9, 4, 2, 16, 30.f, {}};
        c.mask.kind = AttentionMaskKind::causal;
        ocr::test::Lcg lcg;
        c.mask.rel_bias.resize(static_cast<size_t>(c.q_heads) * c.q_len * c.k_len);
        lcg.fill(c.mask.rel_bias, -0.5f, 0.5f);
        run_prefill(device, c, "softcap + rel bias");
    }
    {
        // Fully-masked row: window 1 at position 0 of a causal tail with
        // k_len > q_len leaves every key visible, so force an all-zero mask.
        PrefillCase c{4, 4, 2, 2, 8, 0.f, {}};
        c.mask.kind = AttentionMaskKind::custom_block_mask;
        c.mask.block_size = 2;
        c.mask.block_mask.assign(4, 0);
        run_prefill(device, c, "fully masked");
    }
}

void test_decode(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t batch = 3u, capacity = 17u, q_heads = 6u, kv_heads = 2u, head_dim = 12u;
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    luisa::vector<float> q(static_cast<size_t>(batch) * q_heads * head_dim);
    luisa::vector<float> k(static_cast<size_t>(batch) * capacity * kv_heads * head_dim);
    luisa::vector<float> v(k.size());
    lcg.fill(q, -1.f, 1.f);
    lcg.fill(k, -1.f, 1.f);
    lcg.fill(v, -1.f, 1.f);
    luisa::vector<uint32_t> kv_len = {5u, 17u, 0u};// ragged + empty
    luisa::vector<int32_t> positions = {4, 16, 0};
    auto dummy_u = device.create_buffer<uint32_t>(1);
    for (auto kind : {AttentionMaskKind::bidirectional, AttentionMaskKind::causal,
                      AttentionMaskKind::windowed, AttentionMaskKind::custom_block_mask}) {
        AttentionMaskDescriptor mask;
        mask.kind = kind;
        mask.window = 6;
        if (kind == AttentionMaskKind::custom_block_mask) {
            // S3: block grid over ceil(17/4)=5 key blocks; query blocks 0..4
            // (positions 4/4=1, 16/4=4, 0/4=0). Mask: qb sees kb <= qb.
            mask.block_size = 4;
            const uint32_t k_blocks = (capacity + 3u) / 4u;
            mask.block_mask.assign(static_cast<size_t>(5) * k_blocks, 0);
            for (uint32_t qb = 0; qb < 5; qb++) {
                for (uint32_t kb = 0; kb < k_blocks; kb++) {
                    mask.block_mask[static_cast<size_t>(qb) * k_blocks + kb] = kb <= qb ? 1 : 0;
                }
            }
        }
        // S2/S3 host validation before launch.
        expect(validate_attention_decode_inputs(kv_len, capacity, mask).ok());
        luisa::vector<float> ref;
        expect(ref_attention_decode(q, k, v, kv_len, positions, batch, capacity, q_heads, kv_heads,
                                    head_dim, scale, 0.f, to_ref_mask(mask), ref).ok());
        auto shader = device.compile(make_attention_decode_kernel());
        expect(shader.compile_ok());
        auto bq = ocr::test::upload(device, stream, luisa::span<const float>{q});
        auto bk = ocr::test::upload(device, stream, luisa::span<const float>{k});
        auto bv = ocr::test::upload(device, stream, luisa::span<const float>{v});
        auto bl = ocr::test::upload(device, stream, luisa::span<const uint32_t>{kv_len});
        auto bp = ocr::test::upload(device, stream, luisa::span<const int32_t>{positions});
        auto mask_words = pack_block_mask_words(mask.block_mask);
        auto bmask_storage = mask.block_mask.empty() ? Buffer<uint32_t>{}
                             : ocr::test::upload(device, stream, luisa::span<const uint32_t>{mask_words});
        auto bmask = mask.block_mask.empty() ? dummy_u.view() : bmask_storage.view();
        auto bo = device.create_buffer<float>(q.size());
        auto plan = plan_attention_decode_dispatch(batch, q_heads);
        OCR_DISPATCH_1D(stream, shader(bq, bk, bv, bo, bl, bp, bmask, batch, capacity, q_heads, kv_heads, head_dim,
                         scale, 0.f, static_cast<uint32_t>(kind), mask.window, 0u, mask.block_size,
                         static_cast<uint32_t>(mask.block_mask.size())), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, q.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, "attention decode");
    }
}

// S2: kv_len > kv_capacity must be rejected host-side AND clamped in-kernel
// (defined behavior == same result as kv_len == capacity; never OOB).
void test_decode_invalid_kv_len(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t batch = 1u, capacity = 9u, q_heads = 2u, kv_heads = 2u, head_dim = 8u;
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    luisa::vector<float> q(static_cast<size_t>(batch) * q_heads * head_dim);
    luisa::vector<float> k(static_cast<size_t>(batch) * capacity * kv_heads * head_dim);
    luisa::vector<float> v(k.size());
    lcg.fill(q, -1.f, 1.f);
    lcg.fill(k, -1.f, 1.f);
    lcg.fill(v, -1.f, 1.f);
    const luisa::vector<uint32_t> bad_len = {capacity + 100u};
    const luisa::vector<uint32_t> clamp_len = {capacity};
    const luisa::vector<int32_t> positions = {static_cast<int32_t>(capacity) - 1};
    AttentionMaskDescriptor mask;// bidirectional
    // Host validation REJECTS the invalid length.
    expect(!validate_attention_decode_inputs(bad_len, capacity, mask).ok());
    expect(validate_attention_decode_inputs(clamp_len, capacity, mask).ok());
    // Oracle at the clamped length.
    luisa::vector<float> ref;
    expect(ref_attention_decode(q, k, v, clamp_len, positions, batch, capacity, q_heads, kv_heads,
                                head_dim, scale, 0.f, to_ref_mask(mask), ref).ok());
    auto shader = device.compile(make_attention_decode_kernel());
    auto bq = ocr::test::upload(device, stream, luisa::span<const float>{q});
    auto bk = ocr::test::upload(device, stream, luisa::span<const float>{k});
    auto bv = ocr::test::upload(device, stream, luisa::span<const float>{v});
    auto bl = ocr::test::upload(device, stream, luisa::span<const uint32_t>{bad_len});
    auto bp = ocr::test::upload(device, stream, luisa::span<const int32_t>{positions});
    auto dummy_u = device.create_buffer<uint32_t>(1);
    auto bo = device.create_buffer<float>(q.size());
    auto plan = plan_attention_decode_dispatch(batch, q_heads);
    // Deliberately skip host validation: the kernel must clamp, not read OOB.
    OCR_DISPATCH_1D(stream, shader(bq, bk, bv, bo, bl, bp, dummy_u, batch, capacity, q_heads, kv_heads, head_dim,
                     scale, 0.f, static_cast<uint32_t>(mask.kind), 0u, 0u, 0u, 0u), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, q.size());
    stream << synchronize();
    ocr::test::expect_close(got, ref, kAtol, kRtol, "decode invalid kv_len clamped");
}

// S3: CUSTOM_BLOCK_MASK with block_size==0 must mask EVERYTHING (zero out),
// never behave bidirectional.
void test_decode_block_mask_fail_safe(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t batch = 1u, capacity = 6u, q_heads = 2u, kv_heads = 1u, head_dim = 4u;
    luisa::vector<float> q(static_cast<size_t>(batch) * q_heads * head_dim, 1.f);
    luisa::vector<float> k(static_cast<size_t>(batch) * capacity * kv_heads * head_dim, 1.f);
    luisa::vector<float> v(k.size(), 3.f);
    const luisa::vector<uint32_t> kv_len = {capacity};
    const luisa::vector<int32_t> positions = {5};
    auto shader = device.compile(make_attention_decode_kernel());
    auto bq = ocr::test::upload(device, stream, luisa::span<const float>{q});
    auto bk = ocr::test::upload(device, stream, luisa::span<const float>{k});
    auto bv = ocr::test::upload(device, stream, luisa::span<const float>{v});
    auto bl = ocr::test::upload(device, stream, luisa::span<const uint32_t>{kv_len});
    auto bp = ocr::test::upload(device, stream, luisa::span<const int32_t>{positions});
    auto dummy_u = device.create_buffer<uint32_t>(1);
    auto bo = device.create_buffer<float>(q.size());
    auto plan = plan_attention_decode_dispatch(batch, q_heads);
    OCR_DISPATCH_1D(stream, shader(bq, bk, bv, bo, bl, bp, dummy_u, batch, capacity, q_heads, kv_heads, head_dim,
                     1.f, 0.f, static_cast<uint32_t>(AttentionMaskKind::custom_block_mask), 0u, 0u, 0u, 0u), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, q.size());
    stream << synchronize();
    bool all_zero = true;
    for (auto v_ : got) { all_zero = all_zero && v_ == 0.f; }
    expect(all_zero) << "block_size==0 custom mask must produce zero, not bidirectional";
    // Host validator rejects it too.
    AttentionMaskDescriptor m;
    m.kind = AttentionMaskKind::custom_block_mask;
    m.block_size = 0;
    expect(!validate_attention_decode_inputs(kv_len, capacity, m).ok());
}

void test_mask_validation() {
    AttentionMaskDescriptor m;
    m.kind = AttentionMaskKind::windowed;
    expect(!m.validate(4, 4, 2).ok());// window 0 invalid
    m.kind = AttentionMaskKind::custom_block_mask;
    m.block_size = 0;
    expect(!m.validate(4, 4, 2).ok());
    m.block_size = 2;
    m.block_mask.assign(3, 1);// wrong size (needs 4)
    expect(!m.validate(4, 4, 2).ok());
    m.block_mask.assign(4, 1);
    expect(m.validate(4, 4, 2).ok());
}

// Exact OOB regression: position=1000 with a 2-byte mask. qb=250, idx>=500
// is far beyond mask_bytes=2 -> every key disallowed WITHOUT reading -> zero
// output, matching the oracle. Must not read OOB on either backend.
void test_decode_mask_oob_position(Device &device) {
    auto stream = device.create_stream();
    constexpr uint32_t batch = 1u, capacity = 8u, q_heads = 1u, kv_heads = 1u, head_dim = 4u;
    luisa::vector<float> q(static_cast<size_t>(batch) * q_heads * head_dim, 1.f);
    luisa::vector<float> k(static_cast<size_t>(batch) * capacity * kv_heads * head_dim, 1.f);
    luisa::vector<float> v(k.size(), 3.f);
    const luisa::vector<uint32_t> kv_len = {8u};
    const luisa::vector<int32_t> positions = {1000};
    AttentionMaskDescriptor mask;
    mask.kind = AttentionMaskKind::custom_block_mask;
    mask.block_size = 4;
    mask.block_mask = {1, 1};// only 2 bytes: covers qb=0 only
    // Position-aware host validator REJECTS this (mask can't cover qb=250).
    expect(!validate_attention_decode_inputs(kv_len, positions, capacity, mask).ok());
    // Oracle: idx = 250*2 + kb >= 500 > 2 -> all masked -> zero.
    luisa::vector<float> ref;
    expect(ref_attention_decode(q, k, v, kv_len, positions, batch, capacity, q_heads, kv_heads,
                                head_dim, 1.f, 0.f, to_ref_mask(mask), ref).ok());
    bool ref_zero = true;
    for (auto x : ref) { ref_zero = ref_zero && x == 0.f; }
    expect(ref_zero) << "oracle must be zero for out-of-range mask position";
    auto shader = device.compile(make_attention_decode_kernel());
    auto bq = ocr::test::upload(device, stream, luisa::span<const float>{q});
    auto bk = ocr::test::upload(device, stream, luisa::span<const float>{k});
    auto bv = ocr::test::upload(device, stream, luisa::span<const float>{v});
    auto bl = ocr::test::upload(device, stream, luisa::span<const uint32_t>{kv_len});
    auto bp = ocr::test::upload(device, stream, luisa::span<const int32_t>{positions});
    auto words = pack_block_mask_words(mask.block_mask);
    auto bmask = ocr::test::upload(device, stream, luisa::span<const uint32_t>{words});
    auto bo = device.create_buffer<float>(q.size());
    auto plan = plan_attention_decode_dispatch(batch, q_heads);
    // Deliberately skip host validation: the kernel bound must prevent OOB.
    OCR_DISPATCH_1D(stream, shader(bq, bk, bv, bo, bl, bp, bmask, batch, capacity, q_heads, kv_heads, head_dim,
                     1.f, 0.f, static_cast<uint32_t>(mask.kind), 0u, 0u, mask.block_size,
                     static_cast<uint32_t>(mask.block_mask.size())), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, q.size());
    stream << synchronize();
    ocr::test::expect_close(got, ref, 0.f, 0.f, "decode mask OOB position -> zero, no OOB");
    // A covering mask (qb up to 250) is accepted by the position-aware validator.
    AttentionMaskDescriptor big = mask;
    big.block_mask.assign(static_cast<size_t>(251) * 2, 1);
    expect(validate_attention_decode_inputs(kv_len, positions, capacity, big).ok());
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "attention_prefill_all_masks"_test = [&] { test_prefill_masks(device); };
    "attention_decode_contiguous_kv"_test = [&] { test_decode(device); };
    "attention_mask_validation"_test = [] { test_mask_validation(); };
    "attention_decode_invalid_kv_len"_test = [&] { test_decode_invalid_kv_len(device); };
    "attention_decode_block_mask_fail_safe"_test = [&] { test_decode_block_mask_fail_safe(device); };
    "attention_decode_mask_oob_position"_test = [&] { test_decode_mask_oob_position(device); };
}