// Test for ocr/runtime/hybrid_gdn_executor -- hybrid cache (Phase 9).
// This test covers:
// - capacity math from ACTUAL geometry (KV + recurrent + conv + metadata);
// - per-layer/head/request recurrent state + Conv1D ring state isolation;
// - device position/step/state metadata; periodic full-attention contiguous KV;
// - completion / eviction / reset / reuse semantics with NO cross-request leak;
// - paged KV remains unsupported (fails closed).
// Synthetic data only; no model/download.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"
#include "ocr_decode_test_utils.h"
#include "ocr_gdn_test_utils.h"

#include <cmath>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/runtime/hybrid_gdn_executor.h"
#include "ocr/runtime/memory_budget.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

HybridCacheConfig make_config() {
    HybridCacheConfig c;
    c.num_full_layers = 2;
    c.num_gdn_layers = 3;
    c.batch = 1;
    c.num_kv_heads = 2;
    c.head_dim = 8;
    c.max_seq_len = 32;
    c.gdn_heads = 2;
    c.gdn_head_dim = 4;
    c.conv_kernel_size = 4;
    c.dtype_bytes = 4;
    return c;
}

void test_capacity_math() {
    auto c = make_config();
    expect(c.validate().ok());
    // KV = full_layers * batch * tokens * kv_heads * head_dim * 2 * bytes.
    const uint64_t kv = 2ull * 1 * 32 * 2 * 8 * 2 * 4;
    // GDN = gdn_layers * batch * heads * d * d * bytes.
    const uint64_t rec = 3ull * 1 * 2 * 4 * 4 * 4;
    // Conv = gdn_layers * batch * (3*qk_dim) * (K-1) * bytes; qk_dim=2*4=8.
    const uint64_t conv = 3ull * 1 * (3 * 8) * 3 * 4;
    const uint64_t meta = 4ull * 1 * 4;
    expect(eq(*c.kv_bytes().value, kv)) << "KV bytes from actual dims";
    expect(eq(*c.recurrent_bytes().value, rec)) << "recurrent bytes from actual dims";
    expect(eq(*c.conv_bytes().value, conv)) << "conv bytes from actual dims";
    expect(eq(*c.total_bytes().value, kv + rec + conv + meta)) << "total bytes";
    // Overflow / bad geometry fails closed.
    auto bad = c;
    bad.conv_kernel_size = 1;
    expect(!bad.validate().ok());
    auto bad2 = c;
    bad2.dtype_bytes = 2;
    expect(!bad2.validate().ok());
}

void test_budget_preflight(Device &device) {
    auto stream = device.create_stream();
    auto c = make_config();
    // A budget smaller than the requirement refuses BEFORE allocation.
    MemoryBudget tight(MemoryLimits{*c.total_bytes().value - 1u, 0});
    HybridGdnCache cache;
    auto s = cache.create(device, stream, c, &tight);
    expect(!s.ok()) << "tight budget refuses hybrid cache";
    expect(s.code == StatusCode::out_of_memory);
    // A sufficient budget succeeds and records the resident bytes.
    MemoryBudget ok(MemoryLimits{*c.total_bytes().value + 4096u, 0});
    HybridGdnCache cache2;
    auto s2 = cache2.create(device, stream, c, &ok);
    expect(s2.ok()) << "sufficient budget allocates";
    expect(cache2.valid());
    expect(eq(cache2.device_bytes(), *c.total_bytes().value)) << "resident bytes == planned";
}

void test_state_isolation_and_metadata(Device &device) {
    auto stream = device.create_stream();
    auto c = make_config();
    HybridGdnCache cache;
    LUISA_ASSERT(cache.create(device, stream, c).ok(), "cache create");
    // Fresh state is zero (recurrent + conv ring + metadata).
    luisa::vector<float> rec, ring;
    LUISA_ASSERT(cache.download_recurrent(stream, 0, 0, rec).ok(), "dl rec");
    LUISA_ASSERT(cache.download_conv_ring(stream, 0, 0, ring).ok(), "dl ring");
    stream << synchronize();
    bool zero = true;
    for (auto v : rec) { if (v != 0.0f) { zero = false; } }
    for (auto v : ring) { if (v != 0.0f) { zero = false; } }
    expect(zero) << "fresh recurrent + conv state is zero";
    luisa::vector<uint32_t> kv_len, step, state;
    luisa::vector<int32_t> pos;
    LUISA_ASSERT(cache.state().download(stream, kv_len, pos, step, state).ok(), "dl meta");
    stream << synchronize();
    expect(eq(kv_len[0], 0u));
    expect(eq(state[0], static_cast<uint32_t>(DecodeSeqState::empty)));
    // Per-layer recurrent views are distinct (write one, others stay zero).
    auto rec0 = cache.recurrent(0, 0);
    auto rec1 = cache.recurrent(1, 0);
    expect(static_cast<bool>(rec0));
    expect(rec0.size() == rec1.size());
    // Write a sentinel into layer 0's recurrent state via a host upload.
    luisa::vector<float> sentinel(rec0.size(), 1.5f);
    stream << rec0.copy_from(luisa::span<const float>{sentinel}) << synchronize();
    luisa::vector<float> rec0_dl, rec1_dl;
    LUISA_ASSERT(cache.download_recurrent(stream, 0, 0, rec0_dl).ok(), "dl0");
    LUISA_ASSERT(cache.download_recurrent(stream, 1, 0, rec1_dl).ok(), "dl1");
    stream << synchronize();
    bool l0_set = true, l1_zero = true;
    for (auto v : rec0_dl) { if (v != 1.5f) { l0_set = false; } }
    for (auto v : rec1_dl) { if (v != 0.0f) { l1_zero = false; } }
    expect(l0_set) << "layer 0 recurrent written";
    expect(l1_zero) << "layer 1 recurrent isolated (no leak)";
}

void test_completion_reset_reuse(Device &device) {
    auto stream = device.create_stream();
    auto c = make_config();
    HybridGdnCache cache;
    LUISA_ASSERT(cache.create(device, stream, c).ok(), "cache create");
    // Bind a sequence, dirty the state, then complete + release + reuse.
    LUISA_ASSERT(cache.bind_sequence(0, 5).ok(), "bind");
    expect(eq(cache.length(0), 5u));
    // Occupied slot refuses a second bind.
    expect(!cache.bind_sequence(0, 3).ok()) << "occupied slot refuses rebind";
    auto rec0 = cache.recurrent(0, 0);
    luisa::vector<float> dirty(rec0.size(), 2.0f);
    stream << rec0.copy_from(luisa::span<const float>{dirty}) << synchronize();
    // Completion records finished metadata + releases the host slot.
    {
        auto list = CommandList::create();
        LUISA_ASSERT(cache.record_finished(list, 0).ok(), "finished");
        stream << list.commit() << synchronize();
    }
    expect(eq(cache.length(0), 0u)) << "released slot length 0";
    luisa::vector<uint32_t> kv_len, step, state;
    luisa::vector<int32_t> pos;
    LUISA_ASSERT(cache.state().download(stream, kv_len, pos, step, state).ok(), "dl meta");
    stream << synchronize();
    expect(eq(state[0], static_cast<uint32_t>(DecodeSeqState::finished))) << "finished metadata";
    expect(eq(kv_len[0], 0u)) << "finished kv_len 0 (never attended on reuse)";
    // Reuse: a fresh bind succeeds. record_finished() is METADATA-ONLY, so the
    // stale recurrent bytes ARE still present in the released slot here (the
    // delta-rule kernels do NOT overwrite the whole state: they accumulate onto
    // it). Safety comes from two explicit mechanisms, both tested:
    //   * reset() clears recurrent + conv + metadata for ALL sequences (below);
    //   * every from-scratch EXECUTOR prefill zeroes THIS request's recurrent +
    //     conv state after bind (see gdn_hybrid_reused_cache_no_leak, which uses
    //     NONZERO GDN weights and NO reset() between the two generate calls).
    LUISA_ASSERT(cache.bind_sequence(0, 4).ok(), "rebind after release");
    {
        auto list = CommandList::create();
        LUISA_ASSERT(cache.reset(list).ok(), "reset");
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> rec_after;
    LUISA_ASSERT(cache.download_recurrent(stream, 0, 0, rec_after).ok(), "dl after reset");
    stream << synchronize();
    bool cleared = true;
    for (auto v : rec_after) { if (v != 0.0f) { cleared = false; } }
    expect(cleared) << "reset clears recurrent state (no cross-request leak)";
    expect(eq(cache.length(0), 0u)) << "reset clears host length";
}

void test_kv_append_and_capacity(Device &device) {
    auto stream = device.create_stream();
    auto c = make_config();
    HybridGdnCache cache;
    LUISA_ASSERT(cache.create(device, stream, c).ok(), "cache create");
    // Append within capacity succeeds; overrun fails closed.
    LUISA_ASSERT(cache.bind_sequence(0, 0).ok(), "bind");
    expect(cache.append_tokens(0, 32).ok()) << "append to capacity";
    expect(!cache.append_tokens(0, 1).ok()) << "append beyond capacity refuses";
    // Full-attention KV views are per-full-layer and sized to the capacity.
    auto k0 = cache.k_layer(0);
    auto k1 = cache.k_layer(1);
    const size_t want = static_cast<size_t>(c.max_seq_len) * c.num_kv_heads * c.head_dim;
    expect(eq(k0.size(), want));
    expect(eq(k1.size(), want));
}

void test_paged_unsupported(Device &device) {
    // Paged KV remains optional/unsupported (plan §7.2 item 2).
    KvCacheConfig kc;
    kc.num_layers = 1;
    kc.num_kv_heads = 1;
    kc.head_dim = 4;
    kc.max_seq_len = 8;
    PagedKvCache paged;
    auto s = paged.create(device, kc, 16);
    expect(!s.ok());
    expect(s.code == StatusCode::unsupported);
}

// ---------------------------------------------------------------------------
// Full hybrid decoder: layered CPU parity (prefill / decode / chunked / tokens).
// ---------------------------------------------------------------------------
constexpr float kExecAtol = 4e-3f;
constexpr float kExecRtol = 4e-3f;

struct HybridFixture {
    ocr::test::P7TempDir tmp;
    HybridGdnExecutorConfig cfg;
    ocr::test::HybridSyntheticWeights weights;
    WeightStore store;
    luisa::unique_ptr<HybridGdnExecutor> exec;
    [[nodiscard]] bool init(Device &device, uint32_t seed = 0x9E37u, bool self_loop = false) {
        cfg = ocr::test::hybrid_synthetic_config();
        if (auto s = cfg.validate(); !s.ok()) { LUISA_WARNING("cfg validate: {}", s.message); return false; }
        weights = self_loop ? ocr::test::make_self_loop_weights(cfg, seed)
                            : ocr::test::generate_hybrid_weights(cfg, seed);
        const auto dpath = tmp.path / "hybrid_decoder.lcocr";
        HybridGdnTensorNames names;
        if (auto s = ocr::test::write_hybrid_decoder_package(dpath, cfg, names, weights); !s.ok()) {
            LUISA_WARNING("write hybrid package: {}", s.message);
            return false;
        }
        {
            auto reader = PackageReader::open(dpath);
            if (!reader.ok()) { LUISA_WARNING("open package: {}", reader.status.message); return false; }
            const auto expected = hybrid_gdn_expected_tensors(cfg, names);
            if (auto s = hybrid_gdn_validate_tensors((*reader.value)->tensors(), expected); !s.ok()) {
                LUISA_WARNING("hybrid tensor gate: {}", s.message);
                return false;
            }
        }
        auto stream = device.create_stream();
        if (auto s = ocr::test::upload_hybrid_weights(device, stream, dpath, store); !s.ok()) {
            LUISA_WARNING("upload hybrid: {}", s.message);
            return false;
        }
        exec = luisa::make_unique<HybridGdnExecutor>();
        if (auto s = exec->create(device, stream, cfg, names, store); !s.ok()) {
            LUISA_WARNING("hybrid create: {}", s.message);
            return false;
        }
        return true;
    }
};

// Prefill parity: every row's logits + GDN recurrent/conv state + full KV vs
// the independent token-by-token CPU oracle.
void test_hybrid_prefill_parity(Device &device) {
    HybridFixture f;
    LUISA_ASSERT(f.init(device), "hybrid fixture init");
    auto stream = device.create_stream();
    const uint32_t prompt_len = 7, vocab = f.cfg.decoder.vocab_size;
    luisa::vector<uint32_t> prompt = {1, 5, 3, 9, 2, 7, 4};
    DecodeRequest request;
    request.prompt_tokens = prompt;
    // Oracle: process the prompt token by token.
    ocr::test::HybridCpuOracle oracle(f.cfg, f.weights);
    luisa::vector<float> logits_ref;
    for (auto t : prompt) {
        auto row = oracle.step(t);
        logits_ref.insert(logits_ref.end(), row.begin(), row.end());
    }
    // Device prefill (all logits).
    HybridGdnCache cache;
    LUISA_ASSERT(cache.create(device, stream, ocr::test::hybrid_cache_config(f.cfg, 64)).ok(), "cache");
    auto logits_buf = device.create_buffer<float>(static_cast<size_t>(prompt_len) * vocab);
    {
        auto list = CommandList::create();
        LUISA_ASSERT(f.exec->record_prefill_all_logits(list, request, cache, logits_buf.view()).ok(), "prefill");
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> logits_dev(static_cast<size_t>(prompt_len) * vocab);
    stream << logits_buf.copy_to(luisa::span<float>{logits_dev}) << synchronize();
    ocr::test::expect_close(logits_dev, logits_ref, kExecAtol, kExecRtol, "hybrid prefill all-row logits");
    // GDN recurrent + conv state after prefill (per GDN layer).
    for (uint32_t gl = 0; gl < f.cfg.num_gdn_layers(); gl++) {
        luisa::vector<float> rec_dev, ring_dev;
        LUISA_ASSERT(cache.download_recurrent(stream, gl, 0, rec_dev).ok(), "dl rec");
        LUISA_ASSERT(cache.download_conv_ring(stream, gl, 0, ring_dev).ok(), "dl ring");
        stream << synchronize();
        ocr::test::expect_close(rec_dev, oracle.recurrent(gl), kExecAtol, kExecRtol, "hybrid prefill recurrent state");
        ocr::test::expect_close(ring_dev, oracle.ring(gl), kExecAtol, kExecRtol, "hybrid prefill conv ring state");
    }
    // Full-attention KV after prefill (per full layer, committed prefix).
    for (uint32_t fl = 0; fl < f.cfg.num_full_layers(); fl++) {
        luisa::vector<float> k_dev, v_dev;
        LUISA_ASSERT(cache.download_kv(stream, fl, 0, k_dev, v_dev).ok(), "dl kv");
        stream << synchronize();
        const size_t n = static_cast<size_t>(prompt_len) * f.cfg.decoder.kv_dim();
        luisa::vector<float> k_ref(oracle.kv_k(fl).begin(), oracle.kv_k(fl).begin() + n);
        luisa::vector<float> v_ref(oracle.kv_v(fl).begin(), oracle.kv_v(fl).begin() + n);
        ocr::test::expect_close(k_dev, k_ref, kExecAtol, kExecRtol, "hybrid prefill KV K");
        ocr::test::expect_close(v_dev, v_ref, kExecAtol, kExecRtol, "hybrid prefill KV V");
    }
}

// Full-sequence prefill + N decode steps == token-by-token oracle (logits +
// carried GDN state + KV growth). Also verifies recurrent == chunked prefill.
void test_hybrid_decode_and_chunked(Device &device) {
    HybridFixture f;
    LUISA_ASSERT(f.init(device), "hybrid fixture init");
    auto stream = device.create_stream();
    const uint32_t vocab = f.cfg.decoder.vocab_size;
    luisa::vector<uint32_t> prompt = {2, 6, 1, 8, 3};
    const uint32_t n_decode = 5;
    // Oracle: prompt + greedy decode.
    ocr::test::HybridCpuOracle oracle(f.cfg, f.weights);
    luisa::vector<float> last_logits;
    for (auto t : prompt) { last_logits = oracle.step(t); }
    luisa::vector<uint32_t> tokens_ref;
    luisa::vector<float> decode_logits_ref;
    for (uint32_t s = 0; s < n_decode; s++) {
        luisa::vector<int32_t> am;
        LUISA_ASSERT(ref_argmax(last_logits, 1, vocab, am).ok(), "oracle argmax");
        const uint32_t tok = static_cast<uint32_t>(am[0]);
        tokens_ref.push_back(tok);
        last_logits = oracle.step(tok);
        decode_logits_ref.insert(decode_logits_ref.end(), last_logits.begin(), last_logits.end());
    }
    // Device: recurrent prefill (last logits) + decode steps.
    auto run_decode = [&](uint32_t chunk_size, luisa::vector<uint32_t> &tokens_out,
                          luisa::vector<float> &decode_logits_out, HybridGdnCache &cache) {
        DecodeRequest request;
        request.prompt_tokens = prompt;
        luisa::vector<float> logits(vocab);
        {
            auto list = CommandList::create();
            LUISA_ASSERT(f.exec->record_prefill_chunked(list, request, cache, chunk_size, false, logits, {}).ok(), "prefill");
            stream << list.commit() << synchronize();
        }
        for (uint32_t s = 0; s < n_decode; s++) {
            luisa::vector<int32_t> am;
            LUISA_ASSERT(ref_argmax(logits, 1, vocab, am).ok(), "argmax");
            const uint32_t tok = static_cast<uint32_t>(am[0]);
            tokens_out.push_back(tok);
            auto list = CommandList::create();
            LUISA_ASSERT(f.exec->record_decode_step(list, tok, cache, logits).ok(), "decode");
            stream << list.commit() << synchronize();
            decode_logits_out.insert(decode_logits_out.end(), logits.begin(), logits.end());
        }
    };
    // Recurrent path.
    HybridGdnCache cache_rec;
    LUISA_ASSERT(cache_rec.create(device, stream, ocr::test::hybrid_cache_config(f.cfg, 64)).ok(), "cache rec");
    luisa::vector<uint32_t> tokens_rec;
    luisa::vector<float> logits_rec;
    run_decode(0u, tokens_rec, logits_rec, cache_rec);
    expect(tokens_rec == tokens_ref) << "hybrid recurrent decode tokens == oracle";
    ocr::test::expect_close(logits_rec, decode_logits_ref, kExecAtol, kExecRtol, "hybrid recurrent decode logits");
    // Carried GDN state after decode == oracle.
    for (uint32_t gl = 0; gl < f.cfg.num_gdn_layers(); gl++) {
        luisa::vector<float> rec_dev;
        LUISA_ASSERT(cache_rec.download_recurrent(stream, gl, 0, rec_dev).ok(), "dl rec");
        stream << synchronize();
        ocr::test::expect_close(rec_dev, oracle.recurrent(gl), kExecAtol, kExecRtol, "hybrid decode recurrent state");
    }
    // Chunked path (nondivisible chunk 2) must produce the SAME tokens + logits.
    HybridGdnCache cache_chunk;
    LUISA_ASSERT(cache_chunk.create(device, stream, ocr::test::hybrid_cache_config(f.cfg, 64)).ok(), "cache chunk");
    luisa::vector<uint32_t> tokens_chunk;
    luisa::vector<float> logits_chunk;
    run_decode(2u, tokens_chunk, logits_chunk, cache_chunk);
    expect(tokens_chunk == tokens_ref) << "hybrid chunked decode tokens == oracle";
    expect(tokens_chunk == tokens_rec) << "hybrid chunked == recurrent tokens";
    ocr::test::expect_close(logits_chunk, logits_rec, kExecAtol, kExecRtol, "hybrid chunked == recurrent logits");
}

// Repeat-run determinism: two identical requests produce EXACTLY equal tokens.
void test_hybrid_repeat_determinism(Device &device) {
    HybridFixture f;
    LUISA_ASSERT(f.init(device), "hybrid fixture init");
    auto stream = device.create_stream();
    const uint32_t vocab = f.cfg.decoder.vocab_size;
    luisa::vector<uint32_t> prompt = {4, 2, 9, 1, 6, 3, 8};
    const auto run_once = [&](luisa::vector<uint32_t> &tokens) {
        HybridGdnCache cache;
        LUISA_ASSERT(cache.create(device, stream, ocr::test::hybrid_cache_config(f.cfg, 64)).ok(), "cache");
        DecodeRequest request;
        request.prompt_tokens = prompt;
        luisa::vector<float> logits(vocab);
        {
            auto list = CommandList::create();
            LUISA_ASSERT(f.exec->record_prefill(list, request, cache, logits).ok(), "prefill");
            stream << list.commit() << synchronize();
        }
        for (uint32_t s = 0; s < 6; s++) {
            luisa::vector<int32_t> am;
            LUISA_ASSERT(ref_argmax(logits, 1, vocab, am).ok(), "argmax");
            const uint32_t tok = static_cast<uint32_t>(am[0]);
            tokens.push_back(tok);
            auto list = CommandList::create();
            LUISA_ASSERT(f.exec->record_decode_step(list, tok, cache, logits).ok(), "decode");
            stream << list.commit() << synchronize();
        }
    };
    luisa::vector<uint32_t> run1, run2;
    run_once(run1);
    run_once(run2);
    expect(run1 == run2) << "hybrid repeat-run exact tokens";
    expect(!run1.empty());
}

// ---------------------------------------------------------------------------
// Controller wiring: early EOS / truncation / stop strings / seeded sampling,
// using the controlled self-loop weights (greedy decode repeats the last
// prompt token exactly). The compact Unigram tokenizer (id 4 = "rld") provides
// deterministic text for stop-string handling.
// ---------------------------------------------------------------------------
void test_hybrid_controller_eos_stop_sampling(Device &device) {
    HybridFixture f;
    LUISA_ASSERT(f.init(device, 0x51EEu, true), "self-loop fixture init");
    auto stream = device.create_stream();
    HybridGdnCache cache;
    LUISA_ASSERT(cache.create(device, stream, ocr::test::hybrid_cache_config(f.cfg, 64)).ok(), "cache");
    HybridDecodeController controller;
    DecodeRequest request;
    request.prompt_tokens = {1u, 2u, 4u};// last token 4 -> self-loop on 4

    // 1. Early EOS: eos == the self-loop token -> exactly one generated token.
    {
        HybridDecodeOptions opts;
        opts.sampling.greedy = true;
        opts.max_new_tokens = 8;
        opts.eos_ids = {4u};
        LUISA_ASSERT(controller.create(*f.exec, nullptr, opts).ok(), "controller eos");
        auto r = controller.generate(stream, request, cache);
        expect(r.ok()) << "controller early-EOS generate";
        if (r.ok()) {
            expect(r.value->hit_eos);
            expect(eq(r.value->tokens.size(), 1u));
            expect(eq(r.value->tokens[0], 4u));
            expect(eq(r.value->steps, 1u));
        }
        // Completion released the slot: a new request can bind immediately.
        expect(eq(cache.length(0), 0u)) << "controller releases cache on completion";
    }
    // 2. Truncation: no reachable EOS -> max_new_tokens budget + diagnostic.
    {
        HybridDecodeOptions opts;
        opts.sampling.greedy = true;
        opts.max_new_tokens = 5;
        opts.eos_ids = {23u};// never produced by the self-loop on 4
        LUISA_ASSERT(controller.create(*f.exec, nullptr, opts).ok(), "controller trunc");
        auto r = controller.generate(stream, request, cache);
        expect(r.ok());
        if (r.ok()) {
            expect(r.value->truncated);
            expect(!r.value->hit_eos);
            expect(eq(r.value->tokens.size(), 5u));
            bool all4 = true;
            for (auto t : r.value->tokens) { if (t != 4u) { all4 = false; } }
            expect(all4) << "self-loop greedy tokens";
            expect(!r.value->diagnostics.empty()) << "truncation diagnostic";
        }
    }
    // 3. Stop string via the real incremental detokenizer (id 4 = "rld"). The
    //    stop "rld" is border-free (no proper prefix equals a suffix), so it
    //    matches cleanly on the first generated self-loop token; the stop text
    //    is excluded, leaving empty committed text (stop at the first token).
    {
        auto tok = ocr::test::load_compact_tokenizer();
        LUISA_ASSERT(tok != nullptr, "compact tokenizer");
        HybridDecodeOptions opts;
        opts.sampling.greedy = true;
        opts.max_new_tokens = 12;
        opts.eos_ids = {23u};
        opts.stop_strings = {"rld"};
        LUISA_ASSERT(controller.create(*f.exec, tok.get(), opts).ok(), "controller stop");
        auto r = controller.generate(stream, request, cache);
        expect(r.ok());
        if (r.ok()) {
            expect(r.value->hit_stop_string) << "stop string matched";
            expect(!r.value->hit_eos);
            // Stop text never leaks: the stop is the first token, so nothing is
            // emitted before it.
            expect(r.value->text.empty()) << "stop text excluded from output";
        }
    }
    // 4. Seeded sampling determinism: same seed + logits -> same tokens.
    {
        const auto sample_run = [&](luisa::vector<uint32_t> &tokens) {
            HybridDecodeOptions opts;
            opts.sampling.greedy = false;
            opts.sampling.temperature = 0.9f;
            opts.sampling.top_k = 4;
            opts.sampling.seed = 42u;
            opts.max_new_tokens = 6;
            opts.eos_ids = {23u};
            LUISA_ASSERT(controller.create(*f.exec, nullptr, opts).ok(), "controller sampling");
            auto r = controller.generate(stream, request, cache);
            expect(r.ok()) << "sampling generate";
            if (r.ok()) { tokens = r.value->tokens; }
        };
        luisa::vector<uint32_t> s1, s2;
        sample_run(s1);
        sample_run(s2);
        expect(s1 == s2) << "seeded sampling repeat-run exact tokens";
        expect(eq(s1.size(), 6u));
    }
    // 5. Invalid options fail closed before any device work.
    {
        HybridDecodeOptions bad;
        bad.max_new_tokens = 0;
        expect(!controller.create(*f.exec, nullptr, bad).ok());
        HybridDecodeOptions bad_eos;
        bad_eos.max_new_tokens = 4;
        bad_eos.eos_ids = {999u};// outside vocab
        expect(!controller.create(*f.exec, nullptr, bad_eos).ok());
    }
}

// ---------------------------------------------------------------------------
// Executor-level chunk-size independence: for chunk sizes 1 / nondivisible /
// whole-sequence, EVERY chunk boundary's logits + GDN recurrent state + Conv
// ring + full KV must equal the straight-recurrent oracle at that boundary.
// ---------------------------------------------------------------------------
void test_hybrid_chunk_boundaries(Device &device) {
    HybridFixture f;
    LUISA_ASSERT(f.init(device), "hybrid fixture init");
    auto stream = device.create_stream();
    const uint32_t vocab = f.cfg.decoder.vocab_size;
    const uint32_t total = 13;
    luisa::vector<uint32_t> prompt = {2, 6, 1, 8, 3, 5, 9, 4, 7, 2, 6, 1, 8};
    const uint32_t hidden = f.cfg.decoder.hidden_size;
    for (uint32_t chunk : {1u, 5u, 13u}) {
        // Boundaries: chunk, 2*chunk, ... and the final (possibly partial) one.
        luisa::vector<uint32_t> boundaries;
        for (uint32_t b = chunk; b < total; b += chunk) { boundaries.push_back(b); }
        boundaries.push_back(total);
        for (uint32_t b : boundaries) {
            // Oracle prefix [0, b).
            ocr::test::HybridCpuOracle oracle(f.cfg, f.weights);
            luisa::vector<float> logits_ref;
            for (uint32_t t = 0; t < b; t++) {
                auto row = oracle.step(prompt[t]);
                logits_ref.insert(logits_ref.end(), row.begin(), row.end());
            }
            // Device chunked prefill of the prefix.
            HybridGdnCache cache;
            LUISA_ASSERT(cache.create(device, stream, ocr::test::hybrid_cache_config(f.cfg, 64)).ok(), "cache");
            DecodeRequest request;
            request.prompt_tokens.assign(prompt.begin(), prompt.begin() + b);
            auto logits_buf = device.create_buffer<float>(static_cast<size_t>(b) * vocab);
            {
                auto list = CommandList::create();
                LUISA_ASSERT(f.exec->record_prefill_chunked(list, request, cache, chunk, true, {}, logits_buf.view()).ok(),
                             "chunked prefix prefill");
                stream << list.commit() << synchronize();
            }
            luisa::vector<float> logits_dev(static_cast<size_t>(b) * vocab);
            stream << logits_buf.copy_to(luisa::span<float>{logits_dev}) << synchronize();
            ocr::test::expect_close(logits_dev, logits_ref, kExecAtol, kExecRtol, "hybrid chunk-boundary logits");
            for (uint32_t gl = 0; gl < f.cfg.num_gdn_layers(); gl++) {
                luisa::vector<float> rec_dev, ring_dev;
                LUISA_ASSERT(cache.download_recurrent(stream, gl, 0, rec_dev).ok(), "dl rec");
                LUISA_ASSERT(cache.download_conv_ring(stream, gl, 0, ring_dev).ok(), "dl ring");
                stream << synchronize();
                ocr::test::expect_close(rec_dev, oracle.recurrent(gl), kExecAtol, kExecRtol,
                                        "hybrid chunk-boundary recurrent state");
                ocr::test::expect_close(ring_dev, oracle.ring(gl), kExecAtol, kExecRtol,
                                        "hybrid chunk-boundary conv ring state");
            }
            for (uint32_t fl = 0; fl < f.cfg.num_full_layers(); fl++) {
                luisa::vector<float> k_dev, v_dev;
                LUISA_ASSERT(cache.download_kv(stream, fl, 0, k_dev, v_dev).ok(), "dl kv");
                stream << synchronize();
                const size_t n = static_cast<size_t>(b) * f.cfg.decoder.kv_dim();
                luisa::vector<float> k_ref(oracle.kv_k(fl).begin(), oracle.kv_k(fl).begin() + n);
                luisa::vector<float> v_ref(oracle.kv_v(fl).begin(), oracle.kv_v(fl).begin() + n);
                ocr::test::expect_close(k_dev, k_ref, kExecAtol, kExecRtol, "hybrid chunk-boundary KV K");
                ocr::test::expect_close(v_dev, v_ref, kExecAtol, kExecRtol, "hybrid chunk-boundary KV V");
            }
            (void)hidden;
        }
    }
    // chunk_size larger than the workspace row capacity fails closed.
    {
        HybridGdnCache cache;
        LUISA_ASSERT(cache.create(device, stream, ocr::test::hybrid_cache_config(f.cfg, 64)).ok(), "cache");
        DecodeRequest request;
        request.prompt_tokens = prompt;
        luisa::vector<float> logits(vocab);
        auto list = CommandList::create();
        auto s = f.exec->record_prefill_chunked(list, request, cache, f.cfg.prefill_chunk_rows + 1u, false, logits, {});
        expect(!s.ok()) << "oversized chunk refused";
    }
}

// ---------------------------------------------------------------------------
// S1 REGRESSION (Phase 9 review): a REUSED cache must not leak GDN state across
// requests. Uses NONZERO GDN weights (the ordinary random synthetic weights --
// NOT the self-loop zero-weight fixture, which would hide the leak because a
// zeroed GDN layer contributes nothing regardless of its input state).
//
// Two full generate() calls run on ONE cache with NO reset() in between (the
// controller's completion path is metadata-only). The second run's prefill
// logits, decode logits, tokens, and final GDN recurrent + Conv ring state must
// equal a run on a FRESH cache. Before the fix the delta rule accumulated onto
// the previous request's state, so this test failed.
// ---------------------------------------------------------------------------
void test_hybrid_reused_cache_no_leak(Device &device) {
    HybridFixture f;
    LUISA_ASSERT(f.init(device), "hybrid fixture init (NONZERO GDN weights)");
    auto stream = device.create_stream();
    const uint32_t vocab = f.cfg.decoder.vocab_size;
    const uint32_t hidden = f.cfg.decoder.hidden_size;
    const uint32_t n_decode = 4;

    // Sanity: the fixture really has nonzero GDN weights, so a stale state would
    // change the output (otherwise this regression could not detect the leak).
    bool gdn_nonzero = false;
    for (const auto &L : f.weights.gdn) {
        for (auto v : L.q_w) { if (v != 0.0f) { gdn_nonzero = true; } }
        for (auto v : L.conv_q_w) { if (v != 0.0f) { gdn_nonzero = true; } }
        for (auto v : L.norm_w) { if (v != 0.0f) { gdn_nonzero = true; } }
    }
    expect(gdn_nonzero) << "regression uses NONZERO GDN weights";

    struct RunOut {
        luisa::vector<float> prefill_logits;
        luisa::vector<float> decode_logits;
        luisa::vector<uint32_t> tokens;
        luisa::vector<luisa::vector<float>> recurrent;
        luisa::vector<luisa::vector<float>> ring;
    };
    // One full request (prefill all-row logits + greedy decode steps) on `cache`.
    const auto run_request = [&](HybridGdnCache &cache, luisa::span<const uint32_t> prompt, RunOut &out) {
        DecodeRequest request;
        request.prompt_tokens.assign(prompt.begin(), prompt.end());
        const uint32_t plen = static_cast<uint32_t>(prompt.size());
        auto logits_buf = device.create_buffer<float>(static_cast<size_t>(plen) * vocab);
        {
            auto list = CommandList::create();
            LUISA_ASSERT(f.exec->record_prefill_all_logits(list, request, cache, logits_buf.view()).ok(), "prefill");
            stream << list.commit() << synchronize();
        }
        out.prefill_logits.resize(static_cast<size_t>(plen) * vocab);
        stream << logits_buf.copy_to(luisa::span<float>{out.prefill_logits}) << synchronize();
        // Last prompt row's logits drive the first sampled token.
        luisa::vector<float> logits(out.prefill_logits.end() - vocab, out.prefill_logits.end());
        for (uint32_t s = 0; s < n_decode; s++) {
            luisa::vector<int32_t> am;
            LUISA_ASSERT(ref_argmax(logits, 1, vocab, am).ok(), "argmax");
            const uint32_t tok = static_cast<uint32_t>(am[0]);
            out.tokens.push_back(tok);
            auto list = CommandList::create();
            LUISA_ASSERT(f.exec->record_decode_step(list, tok, cache, logits).ok(), "decode");
            stream << list.commit() << synchronize();
            out.decode_logits.insert(out.decode_logits.end(), logits.begin(), logits.end());
        }
        for (uint32_t gl = 0; gl < f.cfg.num_gdn_layers(); gl++) {
            luisa::vector<float> rec, ring;
            LUISA_ASSERT(cache.download_recurrent(stream, gl, 0, rec).ok(), "dl rec");
            LUISA_ASSERT(cache.download_conv_ring(stream, gl, 0, ring).ok(), "dl ring");
            stream << synchronize();
            out.recurrent.push_back(std::move(rec));
            out.ring.push_back(std::move(ring));
        }
        // Completion is METADATA-ONLY (no reset()): exactly what the controller
        // does. The next prefill must zero the GDN state itself.
        auto list = CommandList::create();
        LUISA_ASSERT(cache.record_finished(list, 0).ok(), "finished");
        stream << list.commit() << synchronize();
        cache.release(0);
        (void)hidden;
    };

    const luisa::vector<uint32_t> prompt_a = {1u, 5u, 3u, 9u, 2u, 7u, 4u};
    const luisa::vector<uint32_t> prompt_b = {6u, 2u, 8u, 1u, 3u};

    // Reference: each request on its OWN fresh cache.
    RunOut ref_a, ref_b;
    {
        HybridGdnCache fresh;
        LUISA_ASSERT(fresh.create(device, stream, ocr::test::hybrid_cache_config(f.cfg, 64)).ok(), "fresh A");
        run_request(fresh, prompt_a, ref_a);
    }
    {
        HybridGdnCache fresh;
        LUISA_ASSERT(fresh.create(device, stream, ocr::test::hybrid_cache_config(f.cfg, 64)).ok(), "fresh B");
        run_request(fresh, prompt_b, ref_b);
    }
    // Reused: request A then request B on ONE cache, no reset() between them.
    RunOut reuse_a, reuse_b;
    {
        HybridGdnCache shared;
        LUISA_ASSERT(shared.create(device, stream, ocr::test::hybrid_cache_config(f.cfg, 64)).ok(), "shared");
        run_request(shared, prompt_a, reuse_a);
        run_request(shared, prompt_b, reuse_b);
    }
    // The FIRST request on the shared cache matches its fresh reference.
    ocr::test::expect_close(reuse_a.prefill_logits, ref_a.prefill_logits, 0.0f, 0.0f, "reused-cache run A prefill logits (exact)");
    expect(reuse_a.tokens == ref_a.tokens) << "reused-cache run A tokens";
    // The SECOND request must be BIT-IDENTICAL to a fresh-cache run: this is the
    // leak regression (stale recurrent/conv state would change every logit).
    ocr::test::expect_close(reuse_b.prefill_logits, ref_b.prefill_logits, 0.0f, 0.0f,
                            "reused-cache run B prefill logits == fresh (exact)");
    ocr::test::expect_close(reuse_b.decode_logits, ref_b.decode_logits, 0.0f, 0.0f,
                            "reused-cache run B decode logits == fresh (exact)");
    expect(reuse_b.tokens == ref_b.tokens) << "reused-cache run B tokens == fresh";
    for (uint32_t gl = 0; gl < f.cfg.num_gdn_layers(); gl++) {
        ocr::test::expect_close(reuse_b.recurrent[gl], ref_b.recurrent[gl], 0.0f, 0.0f,
                                "reused-cache run B recurrent state == fresh (exact)");
        ocr::test::expect_close(reuse_b.ring[gl], ref_b.ring[gl], 0.0f, 0.0f,
                                "reused-cache run B conv ring == fresh (exact)");
    }
    // Also prove the two requests actually produce DIFFERENT state (so equality
    // above is not trivially satisfied by a degenerate all-zero state).
    bool state_differs = false;
    for (uint32_t gl = 0; gl < f.cfg.num_gdn_layers() && !state_differs; gl++) {
        for (size_t i = 0; i < ref_a.recurrent[gl].size(); i++) {
            if (ref_a.recurrent[gl][i] != ref_b.recurrent[gl][i]) { state_differs = true; break; }
        }
    }
    expect(state_differs) << "requests A/B reach different GDN states (non-degenerate check)";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "gdn_cache_capacity_math"_test = [&] { test_capacity_math(); };
    "gdn_cache_budget_preflight"_test = [&] { test_budget_preflight(device); };
    "gdn_cache_state_isolation_and_metadata"_test = [&] { test_state_isolation_and_metadata(device); };
    "gdn_cache_completion_reset_reuse"_test = [&] { test_completion_reset_reuse(device); };
    "gdn_cache_kv_append_and_capacity"_test = [&] { test_kv_append_and_capacity(device); };
    "gdn_cache_paged_unsupported"_test = [&] { test_paged_unsupported(device); };
    "gdn_hybrid_prefill_parity"_test = [&] { test_hybrid_prefill_parity(device); };
    "gdn_hybrid_decode_and_chunked"_test = [&] { test_hybrid_decode_and_chunked(device); };
    "gdn_hybrid_repeat_determinism"_test = [&] { test_hybrid_repeat_determinism(device); };
    "gdn_hybrid_controller_eos_stop_sampling"_test = [&] { test_hybrid_controller_eos_stop_sampling(device); };
    "gdn_hybrid_chunk_boundaries"_test = [&] { test_hybrid_chunk_boundaries(device); };
    "gdn_hybrid_reused_cache_no_leak"_test = [&] { test_hybrid_reused_cache_no_leak(device); };
}
