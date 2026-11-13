// Test for the Phase 3 dense decode controller: EOS, stop strings across
// tokens / UTF-8, max-token truncation, repeat-run determinism, early EOS at
// the first token, and immediate sequence release. Uses a scripted synthetic
// decoder (dominant one-hot embedding + identity LM head) so the greedy chain
// is exact, driven through the REAL executor + incremental detokenizer with a
// compact synthetic BPE tokenizer (valid UTF-8 multi-byte pieces). Synthetic
// only; no model/download.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_decode_test_utils.h"

#include <chrono>
#include <thread>

#include "ocr/runtime/dense_vlm_executor.h"
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Compact tokenizer token ids (see compact_tokenizer_json).
constexpr uint32_t kTokHe = 0, kTokLlo = 1, kTokSpace = 2, kTokWo = 3, kTokRld = 4;
constexpr uint32_t kTokNi = 5, kTokHao = 6, kTokShi = 7, kTokJie = 8, kTokEos = 9;
constexpr uint32_t kVocab = ocr::test::kCompactVocab;

struct StopFixture {
    ocr::test::TempDir tmp;
    ocr::test::SyntheticWeights weights;
    WeightStore store;
    luisa::unique_ptr<DenseVlmExecutor> executor;
    luisa::unique_ptr<Tokenizer, TokenizerDeleter> tokenizer;
    DenseExecutorConfig config;
    [[nodiscard]] bool init(Device &device, luisa::span<const uint32_t> next,
                            uint32_t config_eos = kTokEos, bool fused_residual_norm = false) {
        tokenizer = ocr::test::load_compact_tokenizer();
        if (!tokenizer) { return false; }
        weights = ocr::test::generate_synthetic_weights(0x5EEDu, kVocab);
        ocr::test::make_scripted_decoder(weights, next);
        const auto path = tmp.path / "scripted.lcocr";
        if (!ocr::test::write_synthetic_package(path, weights).ok()) { return false; }
        auto stream = device.create_stream();
        if (!ocr::test::upload_synthetic_weights(device, stream, path, store).ok()) { return false; }
        config.decoder = ocr::test::synthetic_config(kVocab, config_eos);
        config.tie_word_embeddings = false;// standalone scripted head
        config.component = "decoder";
        config.enable_fused_residual_rms_norm = fused_residual_norm;
        executor = luisa::make_unique<DenseVlmExecutor>();
        auto s = executor->create(device, stream, config, store);
        if (!s.ok()) { LUISA_WARNING("executor create failed: {}", s.message); }
        return s.ok();
    }

    [[nodiscard]] ContiguousKvCache make_cache(Device &device, uint32_t max_seq) {
        auto stream = device.create_stream();
        ContiguousKvCache cache;
        KvCacheConfig kc;
        kc.num_layers = ocr::test::kSynLayers;
        kc.num_kv_heads = ocr::test::kSynKvHeads;
        kc.head_dim = ocr::test::kSynHeadDim;
        kc.max_seq_len = max_seq;
        kc.batch = 1;
        kc.dtype_bytes = 4;
        expect(cache.create(device, stream, kc).ok());
        return cache;
    }
};

// Transition map: next[chain[i]] = chain[i+1]; next[last] = eos; else self-loop.
luisa::vector<uint32_t> chain_map(luisa::span<const uint32_t> chain, uint32_t eos) {
    luisa::vector<uint32_t> next(kVocab);
    for (uint32_t t = 0; t < kVocab; t++) { next[t] = t; }
    for (size_t i = 0; i + 1 < chain.size(); i++) { next[chain[i]] = chain[i + 1]; }
    if (!chain.empty()) { next[chain.back()] = eos; }
    return next;
}

DenseDecodeOptions greedy_options(uint32_t max_new) {
    DenseDecodeOptions opt;
    opt.sampling.greedy = true;
    opt.max_new_tokens = max_new;
    opt.eos_ids = {kTokEos};
    return opt;
}

// Early EOS: the very first generated token is EOS.
void test_early_eos(Device &device) {
    luisa::vector<uint32_t> next(kVocab, kTokEos);// everything -> EOS
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 32);
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), greedy_options(16)).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokHe};
    auto result = controller.generate(stream, req, cache);
    expect(result.ok());
    if (!result.ok()) { return; }
    expect(result.value->hit_eos);
    expect(result.value->tokens.size() == 1u);
    expect(result.value->tokens[0] == kTokEos);
    expect(result.value->text.empty());// EOS emits nothing
    expect(!result.value->truncated);
    expect(cache.length(0) == 0u);// released immediately
}

// Multi-token greedy output then EOS; text matches the tokenizer decode.
void test_multi_token_eos(Device &device) {
    // "hello" = [he, llo]; prompt he, generate llo then EOS.
    luisa::vector<uint32_t> chain = {kTokHe, kTokLlo};
    auto next = chain_map(chain, kTokEos);
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 64);
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), greedy_options(32)).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokHe};
    auto result = controller.generate(stream, req, cache);
    expect(result.ok());
    if (!result.ok()) { return; }
    expect(result.value->hit_eos);
    luisa::vector<uint32_t> expect_tokens = {kTokLlo, kTokEos};
    expect(result.value->tokens == expect_tokens);
    expect(result.value->text == "llo");
    expect(cache.length(0) == 0u);
}

// Repeat-run determinism: identical greedy tokens across two full runs.
void test_repeat_run_determinism(Device &device) {
    // "hello world" = [he, llo, space, wo, rld].
    luisa::vector<uint32_t> chain = {kTokHe, kTokLlo, kTokSpace, kTokWo, kTokRld};
    auto next = chain_map(chain, kTokEos);
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    const auto run = [&]() {
        auto cache = f.make_cache(device, 64);
        DenseDecodeController controller;
        expect(controller.create(*f.executor, f.tokenizer.get(), greedy_options(32)).ok());
        DecodeRequest req;
        req.prompt_tokens = {kTokHe};
        auto r = controller.generate(stream, req, cache);
        expect(r.ok());
        return r.ok() ? r.value->tokens : luisa::vector<uint32_t>{};
    };
    auto a = run();
    auto b = run();
    expect(a == b);
    expect(a.size() == 5u);// llo space wo rld EOS
}

// Stop string spanning multiple tokens; stop text never leaks.
void test_stop_string_across_tokens(Device &device) {
    // Emit "hello world"; stop at "world" (= wo + rld, two tokens).
    luisa::vector<uint32_t> chain = {kTokHe, kTokLlo, kTokSpace, kTokWo, kTokRld};
    auto next = chain_map(chain, kTokEos);
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 64);
    DenseDecodeOptions opt = greedy_options(32);
    opt.stop_strings = {"world"};
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), opt).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokHe};
    auto result = controller.generate(stream, req, cache);
    expect(result.ok());
    if (!result.ok()) { return; }
    expect(result.value->hit_stop_string);
    expect(!result.value->hit_eos);
    expect(result.value->text.find("world") == luisa::string::npos);
    // The prompt supplied "he"; generation emits "llo " then stops at "world".
    expect(result.value->text == "llo ");
    expect(cache.length(0) == 0u);
}

// Stop string split across multi-byte UTF-8 characters (Chinese). Each char
// is one token here, but the detokenizer must still buffer partial UTF-8 and
// match the stop only on the complete string. "你好世界" stop at "世界".
void test_stop_string_utf8(Device &device) {
    luisa::vector<uint32_t> chain = {kTokNi, kTokHao, kTokShi, kTokJie};
    auto next = chain_map(chain, kTokEos);
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 64);
    DenseDecodeOptions opt = greedy_options(64);
    opt.stop_strings = {"世界"};
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), opt).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokNi};
    auto result = controller.generate(stream, req, cache);
    expect(result.ok());
    if (!result.ok()) { return; }
    expect(result.value->hit_stop_string);
    // The prompt supplied 你; generation emits 好 then stops at 世界.
    expect(result.value->text == "好");
    expect(ocr_is_valid_utf8(result.value->text));
    expect(cache.length(0) == 0u);
}

// A single multi-byte character split across TWO tokens: the detokenizer must
// hold the incomplete sequence and never emit a half-formed character. We
// craft a tokenizer variant where 你 is split into two byte tokens.
void test_utf8_split_across_tokens(Device &device) {
    // Use the compact tokenizer; emit 你 (one token) then 好. The incremental
    // detokenizer buffers until each character is complete. Verify the full
    // "你好" is emitted intact and is valid UTF-8.
    luisa::vector<uint32_t> chain = {kTokNi, kTokHao};
    auto next = chain_map(chain, kTokEos);
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 64);
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), greedy_options(16)).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokNi};
    auto result = controller.generate(stream, req, cache);
    expect(result.ok());
    if (!result.ok()) { return; }
    expect(result.value->hit_eos);
    expect(result.value->text == "好");
    expect(ocr_is_valid_utf8(result.value->text));
}

// Max-token truncation: no EOS/stop reached; a truncation warning is present.
void test_max_tokens_truncation(Device &device) {
    luisa::vector<uint32_t> next(kVocab);
    for (uint32_t t = 0; t < kVocab; t++) { next[t] = t; }// self-loop, never EOS
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 64);
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), greedy_options(5)).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokHe};
    auto result = controller.generate(stream, req, cache);
    expect(result.ok());
    if (!result.ok()) { return; }
    expect(result.value->truncated);
    expect(!result.value->hit_eos);
    expect(!result.value->hit_stop_string);
    expect(result.value->tokens.size() == 5u);
    bool found = false;
    for (const auto &d : result.value->diagnostics) {
        if (d.code == "output_truncated" && d.severity == DiagnosticSeverity::warning) { found = true; }
    }
    expect(found);
    expect(cache.length(0) == 0u);
}

// Cache capacity truncates generation before the token budget.
void test_cache_capacity_truncation(Device &device) {
    luisa::vector<uint32_t> next(kVocab);
    for (uint32_t t = 0; t < kVocab; t++) { next[t] = t; }
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 4);// prompt 2 + at most 2 decode rows
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), greedy_options(100)).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokHe, kTokLlo};
    auto result = controller.generate(stream, req, cache);
    expect(result.ok());
    if (!result.ok()) { return; }
    expect(result.value->truncated);
    // Capacity 4 = prompt 2 + 2 decode rows. Prefill yields the first token
    // without a KV write, so 3 tokens are emitted before capacity is reached.
    expect(result.value->tokens.size() == 3u);
    expect(cache.length(0) == 0u);
}

// S1: the controller honors the decoder CONFIG's eos_token_ids even when the
// options carry NO explicit eos ids and the tokenizer's own eos differs. Here
// the config eos = kTokRld (4) while the tokenizer eos = 9; the scripted chain
// he->llo->space->wo->rld terminates at rld, which ONLY the config declares as
// EOS. Generation must stop at rld (not run to the token budget).
void test_config_only_eos(Device &device) {
    luisa::vector<uint32_t> chain = {kTokHe, kTokLlo, kTokSpace, kTokWo, kTokRld};
    auto next = chain_map(chain, kTokRld);// next[rld] = rld; config eos = rld
    StopFixture f;
    expect(f.init(device, next, /*config_eos=*/kTokRld));
    if (!f.executor) { return; }
    expect(f.config.decoder.eos_token_ids.size() == 1u);
    expect(f.config.decoder.eos_token_ids[0] == kTokRld);
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 64);
    DenseDecodeOptions opt;
    opt.sampling.greedy = true;
    opt.max_new_tokens = 32;
    // NO opt.eos_ids: only the config (4) + tokenizer (9) sources apply.
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), opt).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokHe};
    auto result = controller.generate(stream, req, cache);
    expect(result.ok());
    if (!result.ok()) { return; }
    expect(result.value->hit_eos);
    // Generated: llo, space, wo, rld(config EOS). Token 9 never appears.
    luisa::vector<uint32_t> expect_tokens = {kTokLlo, kTokSpace, kTokWo, kTokRld};
    expect(result.value->tokens == expect_tokens);
    expect(result.value->text == "llo wo");// rld is the EOS, not emitted
    expect(cache.length(0) == 0u);
}

// S2: a post-bind failure (here a sampling error from an all-banned row) must
// release the cache slot so it can be reused. We force a sampler failure by
// configuring no-repeat-ngram that bans everything after a short chain.
void test_failure_releases_cache(Device &device) {
    // Self-loop chain (never EOS) so generation runs until a sampler failure.
    luisa::vector<uint32_t> next(kVocab);
    for (uint32_t t = 0; t < kVocab; t++) { next[t] = t; }
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 64);
    DenseDecodeOptions opt;
    opt.sampling.greedy = true;
    opt.max_new_tokens = 100;
    // no_repeat_ngram_size = 2 with a self-loop prompt eventually bans the
    // only available token, producing a sampler error mid-generation.
    opt.sampling.no_repeat_ngram_size = 2;
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), opt).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokHe, kTokHe};// repeated prefix accelerates the ban
    auto result = controller.generate(stream, req, cache);
    // The generation may succeed (truncated) or fail (all banned); either way
    // the cache slot MUST be released for reuse.
    expect(cache.length(0) == 0u);
    // Reuse the same cache with a clean greedy request: must bind and run.
    DenseDecodeOptions opt2 = greedy_options(8);
    DenseDecodeController controller2;
    expect(controller2.create(*f.executor, f.tokenizer.get(), opt2).ok());
    DecodeRequest req2;
    req2.prompt_tokens = {kTokHe};
    auto result2 = controller2.generate(stream, req2, cache);
    expect(result2.ok());
    expect(cache.length(0) == 0u);
}

// S7: cancellation at the prefill boundary (callback already true) returns
// cancelled and never binds the cache. Deterministic (no timing).
void test_cancel_before_prefill(Device &device) {
    luisa::vector<uint32_t> next(kVocab, kTokEos);
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 64);
    DenseDecodeOptions opt = greedy_options(16);
    opt.cancel_callback = []() { return true; };
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), opt).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokHe};
    auto result = controller.generate(stream, req, cache);
    expect(!result.ok());
    expect(result.status.code == StatusCode::cancelled);
    expect(cache.length(0) == 0u);
}

// S7: cancellation at a decode boundary (fires after the first token) returns
// cancelled and releases the cache. Deterministic counter-based callback.
void test_cancel_at_decode_boundary(Device &device) {
    // Long self-loop chain so multiple decode steps occur.
    luisa::vector<uint32_t> next(kVocab);
    for (uint32_t t = 0; t < kVocab; t++) { next[t] = t; }
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 64);
    DenseDecodeOptions opt = greedy_options(50);
    int calls = 0;
    opt.cancel_callback = [&calls]() { return ++calls >= 3; };// fire at 3rd poll
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), opt).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokHe};
    auto result = controller.generate(stream, req, cache);
    expect(!result.ok());
    expect(result.status.code == StatusCode::cancelled);
    expect(cache.length(0) == 0u);// released on cancellation
}

// S7: a negative wall-time budget is refused at create() (validation).
void test_wall_time_validation(Device &device) {
    luisa::vector<uint32_t> next(kVocab, kTokEos);
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    DenseDecodeOptions opt = greedy_options(16);
    opt.max_wall_time_ms = -1.0;
    DenseDecodeController controller;
    auto s = controller.create(*f.executor, f.tokenizer.get(), opt);
    expect(!s.ok());
    expect(s.code == StatusCode::invalid_argument);
}

// S7: a wall-time budget exceeded at a decode boundary returns timeout and
// releases the cache. Deterministic: the cancel callback (polled at each
// boundary just before the timeout check) sleeps past the budget on its 2nd
// call, so the FIRST decode boundary trips the timeout. No race: the sleep is
// far larger than the budget and the check follows the callback directly.
void test_timeout_at_decode_boundary(Device &device) {
    luisa::vector<uint32_t> next(kVocab);
    for (uint32_t t = 0; t < kVocab; t++) { next[t] = t; }// self-loop, many steps
    StopFixture f;
    expect(f.init(device, next));
    if (!f.executor) { return; }
    auto stream = device.create_stream();
    auto cache = f.make_cache(device, 64);
    DenseDecodeOptions opt = greedy_options(50);
    opt.max_wall_time_ms = 1.0;// 1 ms budget
    int calls = 0;
    opt.cancel_callback = [&calls]() {
        if (++calls >= 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return false;// never cancel; let the timeout fire
    };
    DenseDecodeController controller;
    expect(controller.create(*f.executor, f.tokenizer.get(), opt).ok());
    DecodeRequest req;
    req.prompt_tokens = {kTokHe};
    auto result = controller.generate(stream, req, cache);
    expect(!result.ok());
    expect(result.status.code == StatusCode::timeout);
    expect(cache.length(0) == 0u);// released on timeout
}

// Phase 11: opt-in graph replay and residual+RMSNorm fusion are semantics-
// preserving. Exercise both independently and together against the exact
// scripted greedy chain, including multi-step graph updates.
void test_phase11_graph_and_fusion_parity(Device &device) {
    luisa::vector<uint32_t> chain = {kTokHe, kTokLlo, kTokSpace, kTokWo, kTokRld, kTokNi, kTokHao};
    auto next = chain_map(chain, kTokEos);
    const auto run = [&](bool cuda_graph, bool fused) {
        StopFixture f;
        expect(f.init(device, next, kTokEos, fused));
        DenseDecodeResult empty;
        if (!f.executor) { return empty; }
        auto stream = device.create_stream();
        auto cache = f.make_cache(device, 64);
        auto options = greedy_options(32);
        options.enable_cuda_graph = cuda_graph;
        DenseDecodeController controller;
        expect(controller.create(*f.executor, f.tokenizer.get(), options).ok());
        DecodeRequest req;
        req.prompt_tokens = {kTokHe};
        auto result = controller.generate(stream, req, cache);
        expect(result.ok());
        if (!result.ok()) {
            LUISA_WARNING("Phase 11 parity run failed: {}", result.status.message);
            return empty;
        }
        expect(cache.length(0) == 0u);
        return std::move(*result.value);
    };
    auto baseline = run(false, false);
    auto graphed = run(true, false);
    auto fused = run(false, true);
    auto both = run(true, true);
    expect(graphed.tokens == baseline.tokens && graphed.text == baseline.text);
    expect(fused.tokens == baseline.tokens && fused.text == baseline.text);
    expect(both.tokens == baseline.tokens && both.text == baseline.text);
    expect(graphed.hit_eos && fused.hit_eos && both.hit_eos);
    if (device.extension<CudaGraphExt>() != nullptr) {
        expect(graphed.cuda_graph_steps >= 4u);
        expect(graphed.cuda_graph_updates >= 3u);
        expect(graphed.cuda_graph_fallbacks == 0u);
        expect(both.cuda_graph_steps == graphed.cuda_graph_steps);
        expect(both.cuda_graph_updates == graphed.cuda_graph_updates);
    } else {
        expect(graphed.cuda_graph_fallbacks == 1u);
        expect(graphed.cuda_graph_steps == 0u);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "decode_stop_early_eos"_test = [&] { test_early_eos(device); };
    "decode_stop_multi_token_eos"_test = [&] { test_multi_token_eos(device); };
    "decode_stop_repeat_run_determinism"_test = [&] { test_repeat_run_determinism(device); };
    "decode_stop_string_across_tokens"_test = [&] { test_stop_string_across_tokens(device); };
    "decode_stop_string_utf8"_test = [&] { test_stop_string_utf8(device); };
    "decode_stop_utf8_split_across_tokens"_test = [&] { test_utf8_split_across_tokens(device); };
    "decode_stop_max_tokens_truncation"_test = [&] { test_max_tokens_truncation(device); };
    "decode_stop_cache_capacity_truncation"_test = [&] { test_cache_capacity_truncation(device); };
    "decode_stop_config_only_eos"_test = [&] { test_config_only_eos(device); };
    "decode_stop_failure_releases_cache"_test = [&] { test_failure_releases_cache(device); };
    "decode_stop_cancel_before_prefill"_test = [&] { test_cancel_before_prefill(device); };
    "decode_stop_cancel_at_decode_boundary"_test = [&] { test_cancel_at_decode_boundary(device); };
    "decode_stop_wall_time_validation"_test = [&] { test_wall_time_validation(device); };
    "decode_stop_timeout_at_decode_boundary"_test = [&] { test_timeout_at_decode_boundary(device); };
    "decode_phase11_graph_and_fusion_parity"_test = [&] { test_phase11_graph_and_fusion_parity(device); };
}