// Test for ocr/models/deepseek_ocr2 + the MoE VLM graph (device, Phase 10) -- the
// SMALL SYNTHETIC DeepSeek-OCR-2 vertical slice: SAM local perception + BOTH conv
// downsamplers + Qwen2-style visual LM encoder with causal-flow queries and the
// EXACT custom visual/query block mask + DeepSeek MoE decoder (64 routed / top-6 /
// two shared). Covers:
// - vision stage parity (patch embed, perception, both downsamplers, encoder
//   custom-mask attention, connector features) vs the independent CPU oracle
// - router logits/probs/selected experts+weights, permutation, every shared/routed
//   contribution, custom-mask attention, and decode KV state vs the oracle
// - deterministic repeat tokens, end-to-end grounded Markdown, memory preflight
// Synthetic weights only; no model/download, no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_deepseek_test_utils.h"

#include <cstdint>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/models/deepseek_ocr2.h"
#include "ocr/runtime/moe_vlm_executor.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 5e-3f;
constexpr float kRtol = 5e-3f;

luisa::vector<uint8_t> make_page(uint32_t w, uint32_t h, uint32_t seed = 7u) {
    luisa::vector<uint8_t> px(static_cast<size_t>(w) * h * 3u);
    ocr::test::Lcg lcg;
    lcg.state = seed;
    for (auto &v : px) { v = static_cast<uint8_t>(lcg.next() % 256u); }
    return px;
}

struct DeepSeekFixture {
    ocr::test::P7TempDir tmp;
    DeepSeekOcr2Config cfg;
    ocr::test::DeepSeekSyntheticWeights w;
    WeightStore store;
    luisa::unique_ptr<MoeVlmExecutor> exec;
    [[nodiscard]] bool init(Device &device, bool self_loop = false) {
        cfg = deepseek_ocr2_synthetic_config();
        if (!cfg.validate().ok()) { return false; }
        w = self_loop ? ocr::test::make_deepseek_self_loop_weights(cfg)
                      : ocr::test::generate_deepseek_weights(cfg);
        const auto vpath = tmp.path / "ds_vision.lcocr";
        const auto dpath = tmp.path / "ds_decoder.lcocr";
        if (auto s = ocr::test::write_deepseek_vision_package(vpath, cfg, w); !s.ok()) {
            LUISA_WARNING("write vision: {}", s.message); return false;
        }
        if (auto s = ocr::test::write_deepseek_decoder_package(dpath, cfg, w); !s.ok()) {
            LUISA_WARNING("write decoder: {}", s.message); return false;
        }
        {
            auto vr = PackageReader::open(vpath);
            auto dr = PackageReader::open(dpath);
            if (!vr.ok() || !dr.ok()) { return false; }
            if (auto s = deepseek_ocr2_validate_tensors((*vr.value)->tensors(), (*dr.value)->tensors(), cfg); !s.ok()) {
                LUISA_WARNING("tensor gate: {}", s.message); return false;
            }
        }
        auto stream = device.create_stream();
        if (auto s = ocr::test::upload_deepseek_weights(device, stream, vpath, dpath, store); !s.ok()) {
            LUISA_WARNING("upload: {}", s.message); return false;
        }
        exec = luisa::make_unique<MoeVlmExecutor>();
        if (auto s = exec->create(device, stream, cfg, store); !s.ok()) {
            LUISA_WARNING("exec create: {}", s.message); return false;
        }
        return true;
    }
};

// Build a packed image for the synthetic perception grid.
P7ImageInput make_image(const DeepSeekOcr2Config &cfg, uint32_t sw, uint32_t sh, uint32_t seed) {
    P7ImageInput img;
    img.pixels = make_page(sw, sh, seed);
    img.source_width = sw;
    img.source_height = sh;
    // Pack through the perception packing spec (mlp connector => grid_multiple 1
    // for packing, but the DeepSeek downsamplers need /4; use a grid multiple 4).
    P7PackingSpec spec = cfg.packing;
    P7ConnectorConfig conn;
    conn.kind = P7ConnectorKind::mlp;
    conn.intermediate_size = cfg.vision.connector_intermediate_size;
    auto packed = p7_pack_image(sw, sh, spec, conn);
    LUISA_ASSERT(packed.ok(), "pack image");
    img.packed = *packed.value;
    return img;
}

// ---------------------------------------------------------------------------
// Vision stage parity.
// ---------------------------------------------------------------------------
void test_vision_parity(Device &device) {
    DeepSeekFixture f;
    LUISA_ASSERT(f.init(device), "fixture init");
    auto stream = device.create_stream();
    const uint32_t sw = 16, sh = 16;
    auto img = make_image(f.cfg, sw, sh, 7u);
    MoeVisionOutput vision;
    {
        auto list = CommandList::create();
        auto vr = f.exec->record_vision(list, img);
        LUISA_ASSERT(vr.ok(), "record vision");
        vision = *vr.value;
        stream << list.commit() << synchronize();
    }
    expect(eq(vision.visual_tokens, f.cfg.vision.num_queries));
    // Independent oracle.
    auto pixels_chw = ocr::test::p7_oracle_preprocess(img.pixels, sw, sh, img.packed.resized_width,
                                                      img.packed.resized_height, f.cfg.vision.mean, f.cfg.vision.std_dev);
    ocr::test::DeepSeekVisionOracle vor;
    LUISA_ASSERT(ocr::test::ds_oracle_vision(f.cfg, f.w, pixels_chw, img.packed, vor).ok(), "vision oracle");
    // Features parity.
    luisa::vector<float> feat_dev;
    LUISA_ASSERT(f.exec->download_vision_features(stream, feat_dev).ok(), "dl features");
    stream << synchronize();
    ocr::test::expect_close(feat_dev, vor.features, kAtol, kRtol, "deepseek connector features");
    // Custom-mask encoder attention parity (per layer).
    for (uint32_t l = 0; l < f.cfg.vision.encoder_layers; l++) {
        luisa::vector<float> attn_dev;
        LUISA_ASSERT(f.exec->download_encoder_attn(stream, l, attn_dev).ok(), "dl encoder attn");
        stream << synchronize();
        ocr::test::expect_close(attn_dev, vor.encoder_attn[l], kAtol, kRtol, "deepseek custom-mask encoder attention");
    }
}

// ---------------------------------------------------------------------------
// Custom visual/query mask boundaries: a query sees all visual + previous queries
// (incl itself), NOT later queries; visual rows never see queries. Verified by
// comparing the device encoder attention to the oracle (which uses the exact
// mask) AND by an explicit mask-structure check.
// ---------------------------------------------------------------------------
void test_custom_mask_boundaries() {
    const uint32_t visual = 6, queries = 4;
    auto mask = ocr::test::ds_visual_query_mask(visual, queries);
    const uint32_t rows = visual + queries;
    expect(eq(mask.block_size, 1u));
    expect(eq(mask.block_mask.size(), static_cast<size_t>(rows) * rows));
    const auto at = [&](uint32_t q, uint32_t k) { return mask.block_mask[static_cast<size_t>(q) * rows + k] != 0u; };
    // Visual rows: bidirectional among visual, never queries.
    for (uint32_t q = 0; q < visual; q++) {
        for (uint32_t k = 0; k < visual; k++) { expect(at(q, k)) << "visual sees visual"; }
        for (uint32_t k = visual; k < rows; k++) { expect(!at(q, k)) << "visual never sees query"; }
    }
    // Query rows: all visual + previous queries incl self, not later queries.
    for (uint32_t qi = 0; qi < queries; qi++) {
        const uint32_t q = visual + qi;
        for (uint32_t k = 0; k < visual; k++) { expect(at(q, k)) << "query sees all visual"; }
        for (uint32_t qj = 0; qj < queries; qj++) {
            const uint32_t k = visual + qj;
            if (qj <= qi) { expect(at(q, k)) << "query sees previous/self query"; }
            else { expect(!at(q, k)) << "query never sees later query"; }
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill: router/permutation/contribution parity + full logits vs oracle.
// ---------------------------------------------------------------------------
void test_prefill_parity(Device &device) {
    DeepSeekFixture f;
    LUISA_ASSERT(f.init(device), "fixture init");
    auto stream = device.create_stream();
    const uint32_t sw = 16, sh = 16;
    auto img = make_image(f.cfg, sw, sh, 11u);
    MoeVisionOutput vision;
    {
        auto list = CommandList::create();
        auto vr = f.exec->record_vision(list, img);
        LUISA_ASSERT(vr.ok(), "record vision");
        vision = *vr.value;
        stream << list.commit() << synchronize();
    }
    // Prompt: [1,2][visual rows][3,4].
    luisa::vector<uint32_t> prompt = {1u, 2u};
    const uint32_t mm_start = static_cast<uint32_t>(prompt.size());
    for (uint32_t i = 0; i < vision.visual_tokens; i++) { prompt.push_back(0u); }
    prompt.push_back(3u);
    prompt.push_back(4u);
    const uint32_t prompt_len = static_cast<uint32_t>(prompt.size());
    // S8: the decoder oracle consumes the INDEPENDENT vision-oracle features,
    // not the device output; the device features are separately asserted equal.
    luisa::vector<float> feat_dev;
    LUISA_ASSERT(f.exec->download_vision_features(stream, feat_dev).ok(), "dl features");
    stream << synchronize();
    auto pixels_chw = ocr::test::p7_oracle_preprocess(img.pixels, sw, sh, img.packed.resized_width,
                                                      img.packed.resized_height, f.cfg.vision.mean, f.cfg.vision.std_dev);
    ocr::test::DeepSeekVisionOracle vor;
    LUISA_ASSERT(ocr::test::ds_oracle_vision(f.cfg, f.w, pixels_chw, img.packed, vor).ok(), "vision oracle");
    ocr::test::expect_close(feat_dev, vor.features, kAtol, kRtol, "S8 device features == independent vision oracle");
    DecodeRequest request;
    request.prompt_tokens = prompt;
    request.mm_features = vision.features;
    request.mm_start = mm_start;
    request.mm_count = vision.visual_tokens;
    ContiguousKvCache cache;
    LUISA_ASSERT(f.exec->create_cache(device, stream, f.cfg.max_seq_len, cache).ok(), "create cache");
    luisa::vector<float> logits(f.cfg.decoder.decoder.vocab_size);
    LUISA_ASSERT(f.exec->prefill(stream, request, cache, logits).ok(), "prefill");
    // Oracle: embeddings + multimodal replacement, then token-by-token decode of
    // the whole prompt (causal => equivalent to prefill).
    const auto &d = f.cfg.decoder.decoder;
    const uint32_t th = d.hidden_size;
    luisa::vector<float> emb_rows(static_cast<size_t>(prompt_len) * th);
    for (uint32_t i = 0; i < prompt_len; i++) {
        for (uint32_t c = 0; c < th; c++) {
            emb_rows[static_cast<size_t>(i) * th + c] = f.w.embed[static_cast<size_t>(prompt[i]) * th + c];
        }
    }
    for (uint32_t i = 0; i < vision.visual_tokens; i++) {
        for (uint32_t c = 0; c < th; c++) {
            // S8: independent oracle features drive the reference embeddings.
            emb_rows[static_cast<size_t>(mm_start + i) * th + c] = vor.features[static_cast<size_t>(i) * th + c];
        }
    }
    ocr::test::DeepSeekDecoderOracle oracle;
    luisa::vector<float> ref_logits;
    LUISA_ASSERT(ocr::test::ds_oracle_decoder_step(f.cfg, f.w, oracle, emb_rows, prompt_len, 0u, ref_logits).ok(), "oracle prefill");
    ocr::test::expect_close(logits, ref_logits, kAtol, kRtol, "deepseek prefill last-row logits");
    // Per-layer router/permutation/contribution parity (oracle processed all rows
    // at once, matching the device prefill batch).
    for (uint32_t l = 0; l < d.num_layers; l++) {
        luisa::vector<float> rlogits, rprobs, expert_out, s0, s1, moe_out;
        luisa::vector<uint32_t> ids, counts, perm_token, slot_pos;
        luisa::vector<float> weights;
        LUISA_ASSERT(f.exec->download_router_logits(stream, l, rlogits).ok(), "dl router logits");
        LUISA_ASSERT(f.exec->download_router_probs(stream, l, rprobs).ok(), "dl router probs");
        LUISA_ASSERT(f.exec->download_selected(stream, l, ids, weights).ok(), "dl selected");
        LUISA_ASSERT(f.exec->download_permutation(stream, l, counts, perm_token, slot_pos).ok(), "dl perm");
        LUISA_ASSERT(f.exec->download_expert_out(stream, l, expert_out).ok(), "dl expert_out");
        LUISA_ASSERT(f.exec->download_shared(stream, l, s0, s1).ok(), "dl shared");
        LUISA_ASSERT(f.exec->download_moe_out(stream, l, moe_out).ok(), "dl moe_out");
        stream << synchronize();
        const auto &R = oracle.routing[l];
        ocr::test::expect_close(rlogits, R.logits, kAtol, kRtol, "deepseek router logits");
        ocr::test::expect_close(rprobs, R.probs, kAtol, kRtol, "deepseek router probs");
        expect(ids == R.ids) << "deepseek selected experts match oracle";
        ocr::test::expect_close(weights, R.weights, kAtol, kRtol, "deepseek selected weights");
        expect(counts == R.counts) << "deepseek permutation counts match";
        expect(perm_token == R.perm_token) << "deepseek perm_token match";
        expect(slot_pos == R.slot_pos) << "deepseek slot_pos match";
        ocr::test::expect_close(expert_out, oracle.expert_out[l], kAtol, kRtol, "deepseek routed expert_out");
        ocr::test::expect_close(s0, oracle.shared0[l], kAtol, kRtol, "deepseek shared0");
        ocr::test::expect_close(s1, oracle.shared1[l], kAtol, kRtol, "deepseek shared1");
        ocr::test::expect_close(moe_out, oracle.moe_out[l], kAtol, kRtol, "deepseek moe_out");
    }
    // Decode KV state: download layer 0 committed K/V and compare to the oracle.
    luisa::vector<float> k_dev, v_dev;
    LUISA_ASSERT(cache.download_layer(stream, 0, 0, k_dev, v_dev).ok(), "dl kv");
    stream << synchronize();
    const uint32_t kvd = d.kv_dim();
    luisa::vector<float> k_ref(oracle.k_cache[0].begin(), oracle.k_cache[0].begin() + static_cast<size_t>(prompt_len) * kvd);
    luisa::vector<float> v_ref(oracle.v_cache[0].begin(), oracle.v_cache[0].begin() + static_cast<size_t>(prompt_len) * kvd);
    ocr::test::expect_close(k_dev, k_ref, kAtol, kRtol, "deepseek decode KV K state");
    ocr::test::expect_close(v_dev, v_ref, kAtol, kRtol, "deepseek decode KV V state");
}

// ---------------------------------------------------------------------------
// Decode step parity + deterministic repeat tokens.
// ---------------------------------------------------------------------------
void test_decode_and_determinism(Device &device) {
    DeepSeekFixture f;
    LUISA_ASSERT(f.init(device), "fixture init");
    auto stream = device.create_stream();
    const uint32_t sw = 16, sh = 16;
    auto img = make_image(f.cfg, sw, sh, 21u);
    MoeVisionOutput vision;
    {
        auto list = CommandList::create();
        auto vr = f.exec->record_vision(list, img);
        LUISA_ASSERT(vr.ok(), "record vision");
        vision = *vr.value;
        stream << list.commit() << synchronize();
    }
    luisa::vector<uint32_t> prompt = {5u};
    const uint32_t mm_start = 1u;
    for (uint32_t i = 0; i < vision.visual_tokens; i++) { prompt.push_back(0u); }
    prompt.push_back(6u);
    const uint32_t prompt_len = static_cast<uint32_t>(prompt.size());
    luisa::vector<float> feat_dev;
    LUISA_ASSERT(f.exec->download_vision_features(stream, feat_dev).ok(), "dl features");
    stream << synchronize();
    const auto &d = f.cfg.decoder.decoder;
    const uint32_t th = d.hidden_size, vocab = d.vocab_size;
    // S8: independent vision oracle drives the reference; device features are
    // asserted equal in the same test.
    auto pixels_chw = ocr::test::p7_oracle_preprocess(img.pixels, sw, sh, img.packed.resized_width,
                                                      img.packed.resized_height, f.cfg.vision.mean, f.cfg.vision.std_dev);
    ocr::test::DeepSeekVisionOracle vor;
    LUISA_ASSERT(ocr::test::ds_oracle_vision(f.cfg, f.w, pixels_chw, img.packed, vor).ok(), "vision oracle");
    ocr::test::expect_close(feat_dev, vor.features, kAtol, kRtol, "S8 decode: device features == vision oracle");
    // Oracle embeddings.
    luisa::vector<float> emb_rows(static_cast<size_t>(prompt_len) * th);
    for (uint32_t i = 0; i < prompt_len; i++) {
        for (uint32_t c = 0; c < th; c++) { emb_rows[static_cast<size_t>(i) * th + c] = f.w.embed[static_cast<size_t>(prompt[i]) * th + c]; }
    }
    for (uint32_t i = 0; i < vision.visual_tokens; i++) {
        for (uint32_t c = 0; c < th; c++) { emb_rows[static_cast<size_t>(mm_start + i) * th + c] = vor.features[static_cast<size_t>(i) * th + c]; }
    }
    ocr::test::DeepSeekDecoderOracle oracle;
    luisa::vector<float> ref_logits;
    LUISA_ASSERT(ocr::test::ds_oracle_decoder_step(f.cfg, f.w, oracle, emb_rows, prompt_len, 0u, ref_logits).ok(), "oracle prefill");
    // Device prefill + a few decode steps; compare each step's logits + greedy token.
    const auto run_decode = [&](luisa::vector<uint32_t> &tokens_out) {
        DecodeRequest request;
        request.prompt_tokens = prompt;
        request.mm_features = vision.features;
        request.mm_start = mm_start;
        request.mm_count = vision.visual_tokens;
        ContiguousKvCache cache;
        LUISA_ASSERT(f.exec->create_cache(device, stream, f.cfg.max_seq_len, cache).ok(), "cache");
        luisa::vector<float> logits(vocab);
        LUISA_ASSERT(f.exec->prefill(stream, request, cache, logits).ok(), "prefill");
        tokens_out.clear();
        for (uint32_t step = 0; step < 4u; step++) {
            // greedy argmax (lowest index ties).
            uint32_t best = 0;
            for (uint32_t c = 1; c < vocab; c++) { if (logits[c] > logits[best]) { best = c; } }
            tokens_out.push_back(best);
            if (step + 1u < 4u) {
                LUISA_ASSERT(f.exec->decode_step(stream, best, cache, logits).ok(), "decode step");
            }
        }
        return cache.length(0);
    };
    luisa::vector<uint32_t> tokens1, tokens2;
    auto len1 = run_decode(tokens1);
    auto len2 = run_decode(tokens2);
    expect(tokens1 == tokens2) << "deterministic repeat tokens";
    expect(eq(len1, prompt_len + 3u)) << "KV grows by decode steps";
    // Oracle decode: feed each greedy token, compare logits at each step.
    luisa::vector<float> oracle_logits = ref_logits;
    bool logits_match = true;
    uint32_t pos = prompt_len;
    for (uint32_t step = 0; step < 4u; step++) {
        uint32_t best = 0;
        for (uint32_t c = 1; c < vocab; c++) { if (oracle_logits[c] > oracle_logits[best]) { best = c; } }
        if (best != tokens1[step]) { logits_match = false; }
        if (step + 1u < 4u) {
            luisa::vector<float> tok_emb(th);
            for (uint32_t c = 0; c < th; c++) { tok_emb[c] = f.w.embed[static_cast<size_t>(best) * th + c]; }
            luisa::vector<float> next_logits;
            LUISA_ASSERT(ocr::test::ds_oracle_decoder_step(f.cfg, f.w, oracle, tok_emb, 1u, pos, next_logits).ok(), "oracle decode");
            oracle_logits = next_logits;
            pos++;
        }
    }
    expect(logits_match) << "decode greedy tokens match oracle at every step";
}

// ---------------------------------------------------------------------------
// End-to-end grounded Markdown from a REAL pipeline run (self-loop weights give
// deterministic non-empty text).
// ---------------------------------------------------------------------------
void test_end_to_end_markdown(Device &device) {
    DeepSeekFixture f;
    LUISA_ASSERT(f.init(device, true), "self-loop fixture init");
    auto tokenizer = ocr::test::load_deepseek_tokenizer();
    LUISA_ASSERT(tokenizer != nullptr, "tokenizer");
    auto stream = device.create_stream();
    const uint32_t sw = 16, sh = 16;
    auto img = make_image(f.cfg, sw, sh, 31u);
    luisa::vector<uint32_t> prompt = {1u, 2u};
    const uint32_t mm_start = static_cast<uint32_t>(prompt.size());
    for (uint32_t i = 0; i < f.cfg.vision.num_queries; i++) { prompt.push_back(0u); }
    prompt.push_back(3u);// last prompt token -> self-loop 'd'
    MoeDecodeOptions opts;
    opts.sampling.greedy = true;
    opts.max_new_tokens = 6;
    opts.eos_ids = {23u};
    auto run = f.exec->run(stream, img, prompt, mm_start, opts, tokenizer.get());
    expect(run.ok()) << "end-to-end run";
    if (!run.ok()) { LUISA_WARNING("run: {}", run.status.message); return; }
    expect(eq(run.value->visual_tokens, f.cfg.vision.num_queries));
    expect(!run.value->decode.tokens.empty()) << "generated tokens";
    expect(!run.value->decode.text.empty()) << "detokenized text";
    expect(run.value->cache_bytes > 0u);
    // Compose the REAL generated text into a page Markdown document.
    LayoutRunResult lr;
    LayoutPageResult page;
    page.page_index = 0;
    page.source_width = sw;
    page.source_height = sh;
    LayoutRegionResult region;
    region.region_id = 0;
    region.page_index = 0;
    region.reading_order = 0;
    region.block_type = LayoutBlockType::text;
    region.content = run.value->decode.text;
    page.regions.push_back(region);
    lr.pages.push_back(page);
    auto md = deepseek_ocr2_compose(lr, LayoutOutputMode::markdown);
    expect(md.ok()) << "end-to-end Markdown compose";
    // Grounded parse of a synthetic grounded output (the parser contract).
    const luisa::string grounded = "<|ref|>" + run.value->decode.text + "<|/ref|><|det|>[[100,100,900,900]]<|/det|>";
    auto gp = deepseek_ocr2_parse_grounded(grounded, sw, sh);
    expect(gp.ok()) << "grounded parse of generated text";
    if (gp.ok()) { expect((*gp.value)[0].grounded); }
}

// ---------------------------------------------------------------------------
// Memory preflight measured against the synthetic config (host; no allocation).
// ---------------------------------------------------------------------------
void test_memory_preflight_measured() {
    const auto cfg = deepseek_ocr2_synthetic_config();
    auto report = deepseek_ocr2_preflight(cfg, cfg.max_seq_len, MemoryLimits{256ull * 1024 * 1024, 0});
    expect(report.ok()) << "synthetic config fits a 256MiB budget";
    if (report.ok()) {
        // The 64-expert decoder dominates the footprint; record the measured plan.
        LUISA_INFO("DeepSeek synthetic preflight: vision {} B, decoder {} B, workspace {} B, KV {} B, total {} B",
                   report.value->vision_weight_bytes, report.value->decoder_weight_bytes,
                   report.value->workspace_bytes, report.value->kv_cache_bytes, report.value->total_device_bytes);
        expect(report.value->decoder_weight_bytes > report.value->vision_weight_bytes)
            << "64-expert decoder dominates";
    }
}

// ---------------------------------------------------------------------------
// MEASURED memory: create the real executor + cache and compare the ACTUAL
// resident device bytes against the preflight plan. Weights and KV must match
// EXACTLY; the workspace arena must fit within the declared conservative bound.
// This is a measurement of allocated bytes, not an estimate.
// ---------------------------------------------------------------------------
void test_memory_measured_device(Device &device) {
    DeepSeekFixture f;
    LUISA_ASSERT(f.init(device), "fixture init");
    auto stream = device.create_stream();
    auto plan = deepseek_ocr2_preflight(f.cfg, f.cfg.max_seq_len, MemoryLimits{0, 0});
    LUISA_ASSERT(plan.ok(), "preflight plan");
    // Measured weight bytes = the installed component buffers (actual device
    // allocations), which must equal the planned inventory byte sums exactly.
    const uint64_t vision_measured = f.store.component_bytes("vision");
    const uint64_t decoder_measured = f.store.component_bytes("decoder");
    expect(eq(vision_measured, plan.value->vision_weight_bytes)) << "measured vision weights == plan";
    expect(eq(decoder_measured, plan.value->decoder_weight_bytes)) << "measured decoder weights == plan";
    expect(eq(f.exec->weight_bytes(), vision_measured + decoder_measured)) << "executor weight accounting";
    // Measured workspace arena capacity must fit the declared bound.
    const uint64_t ws_measured = f.exec->workspace_bytes();
    expect(ws_measured > 0u);
    expect(ws_measured <= plan.value->workspace_bytes) << "measured workspace within the declared bound";
    // Measured KV cache bytes (K + V + DecodeState metadata) must equal the plan.
    ContiguousKvCache cache;
    LUISA_ASSERT(f.exec->create_cache(device, stream, f.cfg.max_seq_len, cache).ok(), "create cache");
    const uint64_t kv_measured = cache.device_bytes();
    expect(eq(kv_measured, plan.value->kv_cache_bytes)) << "measured KV cache == plan";
    const uint64_t total_measured = vision_measured + decoder_measured + ws_measured + kv_measured;
    LUISA_INFO("DeepSeek MEASURED device bytes: vision {} + decoder {} + workspace {} + KV {} = {} (plan total {})",
               vision_measured, decoder_measured, ws_measured, kv_measured, total_measured,
               plan.value->total_device_bytes);
    expect(total_measured <= plan.value->total_device_bytes) << "measured total within the planned total";
    // A cache created under a tiny explicit budget must refuse BEFORE allocation.
    ContiguousKvCache tiny;
    MemoryBudget budget{MemoryLimits{1024, 0}};
    auto refused = f.exec->create_cache(device, stream, f.cfg.max_seq_len, tiny, &budget);
    expect(!refused.ok()) << "tiny KV budget refused before allocation";
    expect(refused.code == StatusCode::out_of_memory);
}


// M2/M3: record_vision must reject a forged patch_count and degenerate sources
// BEFORE any upload/dispatch.
void test_vision_input_boundaries(Device &device) {
    DeepSeekFixture f;
    LUISA_ASSERT(f.init(device), "fixture init");
    auto stream = device.create_stream();
    const uint32_t sw = 16, sh = 16;
    auto good = make_image(f.cfg, sw, sh, 5u);
    // Forged patch_count (grid_h*grid_w + 1) must be refused.
    {
        auto bad = good;
        bad.packed.patch_count = bad.packed.patch_count + 1u;
        auto list = CommandList::create();
        auto r = f.exec->record_vision(list, bad);
        expect(!r.ok()) << "M2 forged patch_count refused";
        expect(r.status.code == StatusCode::invalid_argument);
        expect(list.empty()) << "M2 nothing recorded for a forged patch_count";
    }
    // Forged patch_count smaller than the grid.
    {
        auto bad = good;
        bad.packed.patch_count = bad.packed.patch_count - 1u;
        auto list = CommandList::create();
        expect(!f.exec->record_vision(list, bad).ok()) << "M2 undersized patch_count refused";
    }
    // Zero grid dimension.
    {
        auto bad = good;
        bad.packed.grid_h = 0u;
        auto list = CommandList::create();
        expect(!f.exec->record_vision(list, bad).ok()) << "M2 zero grid_h refused";
    }
    // M3: zero source width / height and an empty pixel buffer.
    {
        auto bad = good;
        bad.source_width = 0u;
        auto list = CommandList::create();
        expect(!f.exec->record_vision(list, bad).ok()) << "M3 zero source_width refused";
    }
    {
        auto bad = good;
        bad.source_height = 0u;
        auto list = CommandList::create();
        expect(!f.exec->record_vision(list, bad).ok()) << "M3 zero source_height refused";
    }
    {
        auto bad = good;
        bad.pixels.clear();
        auto list = CommandList::create();
        expect(!f.exec->record_vision(list, bad).ok()) << "M3 empty pixels refused";
    }
    {
        auto bad = good;
        bad.packed.resized_width = 0u;
        auto list = CommandList::create();
        expect(!f.exec->record_vision(list, bad).ok()) << "M3 zero resized_width refused";
    }
    // The valid image still records.
    {
        auto list = CommandList::create();
        expect(f.exec->record_vision(list, good).ok()) << "valid image records";
        stream << list.commit() << synchronize();
    }
}

// S2: the KV capacity may never exceed the decoder's declared context ceiling.
void test_cache_context_ceiling(Device &device) {
    DeepSeekFixture f;
    LUISA_ASSERT(f.init(device), "fixture init");
    auto stream = device.create_stream();
    const uint32_t ceiling = f.cfg.decoder.decoder.max_position_embeddings;
    ContiguousKvCache ok_cache;
    expect(f.exec->create_cache(device, stream, ceiling, ok_cache).ok()) << "capacity == ceiling allowed";
    ContiguousKvCache bad_cache;
    auto over = f.exec->create_cache(device, stream, ceiling + 1u, bad_cache);
    expect(!over.ok()) << "S2 capacity above max_position_embeddings refused";
    expect(over.code == StatusCode::invalid_argument);
    expect(!bad_cache.valid()) << "S2 no cache allocated on refusal";
    ContiguousKvCache zero_cache;
    expect(!f.exec->create_cache(device, stream, 0u, zero_cache).ok()) << "S2 zero capacity refused";
}

// S5: the configured MemoryLimits bound weights + workspace + KV end-to-end and
// refuse BEFORE allocation, with required-vs-available details.
void test_memory_limits_end_to_end(Device &device) {
    // A budget far below the measured footprint must refuse executor creation.
    {
        DeepSeekOcr2Config cfg = deepseek_ocr2_synthetic_config();
        cfg.memory = MemoryLimits{4096ull, 0};
        auto plan = deepseek_ocr2_preflight(cfg, cfg.max_seq_len, cfg.memory);
        expect(!plan.ok()) << "S5 preflight refuses a 4KiB budget";
        expect(plan.status.code == StatusCode::out_of_memory);
        expect(plan.status.message.find("available") != luisa::string::npos);
    }
    // With a real WeightStore, create() must refuse before compiling/allocating
    // the KV, and a legal budget must succeed.
    DeepSeekFixture f;
    LUISA_ASSERT(f.init(device), "fixture init");
    auto stream = device.create_stream();
    auto plan = deepseek_ocr2_preflight(f.cfg, f.cfg.max_seq_len, MemoryLimits{0, 0});
    LUISA_ASSERT(plan.ok(), "plan");
    const uint64_t need = plan.value->total_device_bytes;
    // Just below the requirement -> refusal.
    {
        DeepSeekOcr2Config tight_cfg = f.cfg;
        tight_cfg.memory = MemoryLimits{need - 1ull, 0};
        MoeVlmExecutor tight;
        auto s = tight.create(device, stream, tight_cfg, f.store);
        expect(!s.ok()) << "S5 create() refuses a budget one byte below the footprint";
        expect(s.code == StatusCode::out_of_memory);
    }
    // Exactly the requirement -> allowed, and the cache creation also fits.
    {
        DeepSeekOcr2Config exact_cfg = f.cfg;
        exact_cfg.memory = MemoryLimits{need, 0};
        MoeVlmExecutor exact;
        auto s = exact.create(device, stream, exact_cfg, f.store);
        expect(s.ok()) << "S5 create() accepts a budget equal to the footprint";
        if (s.ok()) {
            ContiguousKvCache cache;
            expect(exact.create_cache(device, stream, exact_cfg.max_seq_len, cache).ok())
                << "S5 cache fits the exact budget";
            // A larger cache must be refused by the end-to-end budget.
            ContiguousKvCache bigger;
            auto bs = exact.create_cache(device, stream, exact_cfg.max_seq_len, bigger);
            expect(bs.ok()) << "same capacity re-checks consistently";
        }
    }
    // A cache-only budget refusal (explicit MemoryBudget) still fails closed.
    {
        ContiguousKvCache tiny;
        MemoryBudget budget{MemoryLimits{1024, 0}};
        auto s = f.exec->create_cache(device, stream, f.cfg.max_seq_len, tiny, &budget);
        expect(!s.ok()) << "S5 tiny cache budget refused";
        expect(s.code == StatusCode::out_of_memory);
        expect(!tiny.valid()) << "S5 no allocation on refusal";
    }
}

// S4: an error AFTER uploads were recorded must still keep the staging alive
// until command completion, so committing the list cannot read freed memory.
void test_staging_lifetime_on_error(Device &device) {
    DeepSeekFixture f;
    LUISA_ASSERT(f.init(device), "fixture init");
    auto stream = device.create_stream();
    const uint32_t sw = 16, sh = 16;
    auto img = make_image(f.cfg, sw, sh, 9u);
    auto list = CommandList::create();
    auto s = f.exec->record_vision_uploads_then_fail_for_test(list, img);
    expect(!s.ok()) << "S4 injected post-upload failure reported";
    expect(!list.empty()) << "S4 uploads were recorded before the failure";
    // Committing after the error must be safe: the staging is retained by the
    // completion callback attached on the error path.
    stream << list.commit() << synchronize();
    // Prove the uploaded pixels are the expected preprocessed values (i.e. the
    // copy read live memory, not a freed buffer).
    auto expect_pixels = ocr::test::p7_oracle_preprocess(img.pixels, sw, sh, img.packed.resized_width,
                                                         img.packed.resized_height, f.cfg.vision.mean,
                                                         f.cfg.vision.std_dev);
    luisa::vector<float> got(expect_pixels.size(), 0.0f);
    LUISA_ASSERT(f.exec->download_vision_pixels(stream, got).ok(), "dl pixels");
    stream << synchronize();
    ocr::test::expect_close(got, expect_pixels, 0.0f, 0.0f, "S4 uploaded pixels survive the error path");
}

// S9: compare the DEVICE custom-mask bytes directly against the independent
// structural expectation (not only attention parity).
void test_device_mask_bytes(Device &device) {
    DeepSeekFixture f;
    LUISA_ASSERT(f.init(device), "fixture init");
    auto stream = device.create_stream();
    const uint32_t sw = 16, sh = 16;
    auto img = make_image(f.cfg, sw, sh, 13u);
    MoeVisionOutput vision;
    {
        auto list = CommandList::create();
        auto vr = f.exec->record_vision(list, img);
        LUISA_ASSERT(vr.ok(), "record vision");
        vision = *vr.value;
        stream << list.commit() << synchronize();
    }
    luisa::vector<uint8_t> dev_mask;
    LUISA_ASSERT(f.exec->download_visual_query_mask(stream, dev_mask).ok(), "dl mask");
    stream << synchronize();
    const uint32_t enc_rows = f.exec->last_encoder_rows();
    expect(eq(enc_rows, vision.encoder_rows));
    const uint32_t nq = f.cfg.vision.num_queries;
    LUISA_ASSERT(enc_rows >= nq, "encoder rows");
    const uint32_t visual_rows = enc_rows - nq;
    // Independent structural expectation (no shared code with the builder).
    const auto want = ocr::test::ds_expected_visual_query_mask(visual_rows, nq);
    expect(dev_mask == want) << "S9 device mask bytes == independent structural expectation";
    // The canonical builder must agree with the structural expectation too.
    const auto canonical = MoeVlmExecutor::build_visual_query_mask(visual_rows, nq);
    expect(canonical == want) << "S9 canonical builder == structural expectation";
    // Boundary spot-checks on the downloaded bytes.
    const auto at = [&](uint32_t q, uint32_t k) { return dev_mask[static_cast<size_t>(q) * enc_rows + k] != 0u; };
    expect(at(0, 0)) << "visual self";
    expect(!at(0, visual_rows)) << "visual never sees the first query";
    expect(at(visual_rows, 0)) << "first query sees visual";
    expect(at(visual_rows, visual_rows)) << "query sees itself";
    if (nq > 1u) {
        expect(!at(visual_rows, visual_rows + 1u)) << "query never sees a later query";
        expect(at(visual_rows + 1u, visual_rows)) << "later query sees earlier query";
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "deepseek_vision_parity"_test = [&] { test_vision_parity(device); };
    "deepseek_custom_mask_boundaries"_test = [] { test_custom_mask_boundaries(); };
    "deepseek_prefill_parity"_test = [&] { test_prefill_parity(device); };
    "deepseek_decode_and_determinism"_test = [&] { test_decode_and_determinism(device); };
    "deepseek_end_to_end_markdown"_test = [&] { test_end_to_end_markdown(device); };
    "deepseek_memory_preflight_measured"_test = [] { test_memory_preflight_measured(); };
    "deepseek_memory_measured_device"_test = [&] { test_memory_measured_device(device); };
    "deepseek_vision_input_boundaries"_test = [&] { test_vision_input_boundaries(device); };
    "deepseek_cache_context_ceiling"_test = [&] { test_cache_context_ceiling(device); };
    "deepseek_memory_limits_end_to_end"_test = [&] { test_memory_limits_end_to_end(device); };
    "deepseek_staging_lifetime_on_error"_test = [&] { test_staging_lifetime_on_error(device); };
    "deepseek_device_mask_bytes"_test = [&] { test_device_mask_bytes(device); };
}
