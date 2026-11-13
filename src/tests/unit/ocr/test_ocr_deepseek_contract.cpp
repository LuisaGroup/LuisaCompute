// Test for ocr/models/deepseek_ocr2 host contract (Phase 10, no device).
// This test covers:
// - the EXACT 64 routed / top-6 / two-shared expert contract (full + synthetic)
// - config validation, expert-contract refusal on scaled counts, config parsing
// - tensor inventory, signature gate, real-artifact fail-closed gate
// - memory preflight (fits a measured synthetic config; refuses before allocation)
// - grounded Markdown parsing (source coordinates) + invalid output handling
// - compose modes (plain/markdown; spotting/extract fail closed)
#include "ut/ut.hpp"

#include <cmath>
#include <cstdint>

#include "ocr/models/deepseek_ocr2.h"
#include "ocr/runtime/layout_result.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_expert_contract() {
    const auto declared = deepseek_ocr2_declared_decoder_config();
    expect(declared.validate().ok()) << "declared config valid";
    expect(eq(declared.num_experts, 64u));
    expect(eq(declared.top_k, 6u));
    expect(eq(declared.num_shared_experts, 2u));
    expect(deepseek_ocr2_validate_expert_contract(declared).ok()) << "declared 64/6/2 contract";
    // Scaled hidden dims are allowed; scaled expert count / top-k are REFUSED.
    auto bad_count = declared;
    bad_count.num_experts = 32u;
    expect(!deepseek_ocr2_validate_expert_contract(bad_count).ok()) << "scaled expert count refused";
    auto bad_topk = declared;
    bad_topk.top_k = 3u;
    expect(!deepseek_ocr2_validate_expert_contract(bad_topk).ok()) << "scaled top-k refused";
    auto bad_shared = declared;
    bad_shared.num_shared_experts = 1u;
    expect(!deepseek_ocr2_validate_expert_contract(bad_shared).ok()) << "scaled shared count refused";
    // The synthetic slice keeps the EXACT contract.
    const auto synth = deepseek_ocr2_synthetic_config();
    expect(synth.validate().ok()) << "synthetic config valid";
    expect(eq(synth.decoder.num_experts, 64u));
    expect(eq(synth.decoder.top_k, 6u));
    expect(eq(synth.decoder.num_shared_experts, 2u));
    expect(deepseek_ocr2_validate_expert_contract(synth.decoder).ok());
    // Scaled hidden dims (synthetic hidden 16 != declared 2048) still pass.
    expect(synth.decoder.decoder.hidden_size != declared.decoder.hidden_size);
}

void test_config_parse() {
    const luisa::string json = R"({
      "model_type": "deepseek_ocr_2",
      "decoder_config": {
        "vocab_size": 24, "hidden_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
        "intermediate_size": 32, "max_position_embeddings": 128,
        "moe_intermediate_size": 16, "shared_intermediate_size": 16
      }
    })";
    auto parsed = parse_deepseek_ocr2_config(json);
    expect(parsed.ok()) << "valid config parses";
    if (parsed.ok()) {
        expect(eq(parsed.value->decoder.decoder.num_layers, 2u));
        expect(eq(parsed.value->decoder.num_experts, 64u));
        expect(eq(parsed.value->decoder.top_k, 6u));
    }
    // A scaled expert count in the config fails closed.
    const luisa::string bad = R"({"decoder_config":{"vocab_size":24,"hidden_size":16,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":32,"max_position_embeddings":128,"num_experts":8}})";
    auto b = parse_deepseek_ocr2_config(bad);
    expect(!b.ok()) << "scaled expert count in config refused";
    // Missing decoder_config fails closed.
    expect(!parse_deepseek_ocr2_config(R"({"model_type":"deepseek_ocr_2"})").ok());
}

void test_signature_and_real_gate() {
    const auto cfg = deepseek_ocr2_synthetic_config();
    auto real = deepseek_ocr2_validate_real_artifact("some-rev", "some-fp");
    expect(!real.ok());
    expect(real.code == StatusCode::unvalidated);
    DeepSeekOcr2Signature empty;
    expect(!deepseek_ocr2_validate_signature(cfg, empty).ok());
    DeepSeekOcr2Signature good;
    good.checkpoint_revision = "synthetic-rev";
    good.config_fingerprint = "synthetic-fp";
    for (const auto &e : deepseek_ocr2_vision_expected_tensors(cfg)) { good.tensor_names.push_back(e.name); }
    for (const auto &e : deepseek_ocr2_decoder_expected_tensors(cfg)) { good.tensor_names.push_back(e.name); }
    expect(deepseek_ocr2_validate_signature(cfg, good).ok()) << "exact signature accepted";
    auto bad = good;
    bad.tensor_names.push_back("model.layers.0.mlp.experts.999.gate_proj.weight");
    expect(!deepseek_ocr2_validate_signature(cfg, bad).ok()) << "unexpected tensor refused";
    auto missing = good;
    missing.tensor_names.pop_back();
    expect(!deepseek_ocr2_validate_signature(cfg, missing).ok()) << "missing tensor refused";
    // The inventory carries exactly 64 routed experts per layer + 2 shared.
    const auto dec = deepseek_ocr2_decoder_expected_tensors(cfg);
    uint32_t expert_tensors = 0, shared_tensors = 0, router_tensors = 0;
    for (const auto &e : dec) {
        if (e.name.find(".mlp.experts.") != luisa::string::npos) { expert_tensors++; }
        if (e.name.find(".mlp.shared_experts.") != luisa::string::npos) { shared_tensors++; }
        if (e.name.find(".mlp.gate.weight") != luisa::string::npos) { router_tensors++; }
    }
    const uint32_t nl = cfg.decoder.decoder.num_layers;
    expect(eq(expert_tensors, nl * 64u * 3u)) << "64 routed experts x 3 projections per layer";
    expect(eq(shared_tensors, nl * 2u * 3u)) << "2 shared experts x 3 projections per layer";
    expect(eq(router_tensors, nl)) << "one router per layer";
}

void test_memory_preflight() {
    const auto cfg = deepseek_ocr2_synthetic_config();
    auto ok = deepseek_ocr2_preflight(cfg, 128, MemoryLimits{64ull * 1024 * 1024, 0});
    expect(ok.ok()) << "sufficient budget fits";
    if (ok.ok()) {
        expect(ok.value->fits);
        expect(ok.value->decoder_weight_bytes > 0u);
        expect(ok.value->vision_weight_bytes > 0u);
        expect(ok.value->kv_cache_bytes > 0u);
        expect(ok.value->total_device_bytes > 0u);
    }
    // A tiny budget refuses BEFORE allocation with required-vs-available details.
    auto tight = deepseek_ocr2_preflight(cfg, 128, MemoryLimits{1024, 0});
    expect(!tight.ok());
    expect(tight.status.code == StatusCode::out_of_memory);
    expect(tight.status.message.find("requires") != luisa::string::npos) << "required-vs-available message";
    expect(tight.status.message.find("available") != luisa::string::npos);
}

void test_grounded_markdown() {
    // Valid grounded block: coordinates in [0,1000) scaled to page pixels.
    const luisa::string text = "<|ref|>hello world<|/ref|><|det|>[[100,200,500,600]]<|/det|>";
    auto parsed = deepseek_ocr2_parse_grounded(text, 1000, 2000);
    expect(parsed.ok()) << "valid grounded markdown parses";
    if (parsed.ok()) {
        expect(eq(parsed.value->size(), 1u));
        const auto &b = (*parsed.value)[0];
        expect(b.grounded);
        expect(b.text == "hello world");
        expect(eq(b.boxes.size(), 1u));
        // x: 100/1000*1000=100, 500/1000*1000=500; y: 200/1000*2000=400, 600/1000*2000=1200.
        expect(std::abs(b.boxes[0].x0 - 100.0f) < 1e-3f);
        expect(std::abs(b.boxes[0].y0 - 400.0f) < 1e-3f);
        expect(std::abs(b.boxes[0].x1 - 500.0f) < 1e-3f);
        expect(std::abs(b.boxes[0].y1 - 1200.0f) < 1e-3f);
    }
    // Multiple boxes + plain text around grounding.
    const luisa::string multi = "intro <|ref|>a<|/ref|><|det|>[[0,0,100,100],[200,200,300,300]]<|/det|> outro";
    auto m = deepseek_ocr2_parse_grounded(multi, 100, 100);
    expect(m.ok());
    if (m.ok()) {
        expect(eq(m.value->size(), 3u)) << "plain + grounded + plain";
        expect(eq((*m.value)[1].boxes.size(), 2u));
    }
    // Plain markdown (no grounding) is accepted.
    auto plain = deepseek_ocr2_parse_grounded("just text", 100, 100);
    expect(plain.ok());
    if (plain.ok()) { expect(!(*plain.value)[0].grounded); }
    // Invalid: unclosed ref.
    expect(!deepseek_ocr2_parse_grounded("<|ref|>oops", 100, 100).ok());
    // Invalid: ref without det.
    expect(!deepseek_ocr2_parse_grounded("<|ref|>x<|/ref|> no det", 100, 100).ok());
    // Invalid: out-of-range coordinate.
    expect(!deepseek_ocr2_parse_grounded("<|ref|>x<|/ref|><|det|>[[0,0,1001,100]]<|/det|>", 100, 100).ok());
    // Invalid: degenerate box (x1<=x0).
    expect(!deepseek_ocr2_parse_grounded("<|ref|>x<|/ref|><|det|>[[100,0,100,100]]<|/det|>", 100, 100).ok());
    // Invalid: unclosed det.
    expect(!deepseek_ocr2_parse_grounded("<|ref|>x<|/ref|><|det|>[[0,0,100,100]]", 100, 100).ok());
    // Invalid: empty box list.
    expect(!deepseek_ocr2_parse_grounded("<|ref|>x<|/ref|><|det|>[[]]<|/det|>", 100, 100).ok());
}

void test_compose_modes() {
    LayoutRunResult run;
    LayoutPageResult page;
    page.page_index = 0;
    page.source_width = 16;
    page.source_height = 16;
    LayoutRegionResult region;
    region.region_id = 0;
    region.page_index = 0;
    region.reading_order = 0;
    region.block_type = LayoutBlockType::text;
    region.content = "hello world";
    page.regions.push_back(region);
    run.pages.push_back(page);
    auto plain = deepseek_ocr2_compose(run, LayoutOutputMode::plain);
    expect(plain.ok()) << "plain compose";
    auto md = deepseek_ocr2_compose(run, LayoutOutputMode::markdown);
    expect(md.ok()) << "markdown compose";
    auto spot = deepseek_ocr2_compose(run, LayoutOutputMode::spotting);
    expect(!spot.ok());
    expect(spot.status.code == StatusCode::unsupported);
    auto ext = deepseek_ocr2_compose(run, LayoutOutputMode::extract);
    expect(!ext.ok());
}


// M1: a zero head count must fail closed BEFORE deriving the default head_dim
// (previously divided by zero). A non-dividing hidden_size is refused too.
void test_parse_zero_heads() {
    const luisa::string zero_heads = R"({"decoder_config":{"vocab_size":24,"hidden_size":16,"num_hidden_layers":2,"num_attention_heads":0,"num_key_value_heads":2,"intermediate_size":32,"max_position_embeddings":128}})";
    auto z = parse_deepseek_ocr2_config(zero_heads);
    expect(!z.ok()) << "num_attention_heads=0 fails closed (no divide by zero)";
    expect(z.status.code == StatusCode::invalid_argument);
    const luisa::string indivisible = R"({"decoder_config":{"vocab_size":24,"hidden_size":17,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":32,"max_position_embeddings":128}})";
    auto i = parse_deepseek_ocr2_config(indivisible);
    expect(!i.ok()) << "hidden_size not divisible by heads fails closed";
    // An explicit head_dim still needs a nonzero head count.
    const luisa::string explicit_hd = R"({"decoder_config":{"vocab_size":24,"hidden_size":16,"num_hidden_layers":2,"num_attention_heads":0,"num_key_value_heads":2,"head_dim":4,"intermediate_size":32,"max_position_embeddings":128}})";
    expect(!parse_deepseek_ocr2_config(explicit_hd).ok()) << "explicit head_dim does not bypass the zero-head guard";
}

// M4: a downsampler must never ENLARGE the grid (stride 1 + pad 2 would), and the
// refusal happens in config validation, i.e. BEFORE any workspace sizing.
void test_downsampler_cannot_enlarge() {
    auto cfg = deepseek_ocr2_synthetic_config();
    // stride 1, kernel 3, pad 2 => out = in + 2 (enlarges). Must be refused.
    cfg.vision.down1 = DeepSeekConvDownsampler{16u, 3u, 1u, 2u};
    expect(!cfg.vision.down1.validate().ok()) << "stride1/pad2 downsampler refused";
    expect(!cfg.validate().ok()) << "config validation refuses the enlarging downsampler";
    expect(!deepseek_ocr2_workspace_plan(cfg).ok() || !cfg.validate().ok())
        << "no workspace is sized from an invalid downsampler";
    // Same for the second downsampler.
    auto cfg2 = deepseek_ocr2_synthetic_config();
    cfg2.vision.down2 = DeepSeekConvDownsampler{16u, 5u, 1u, 3u};
    expect(!cfg2.validate().ok()) << "down2 enlarging refused";
    // A stride-1 kernel-3 pad-1 downsampler preserves the grid and is allowed.
    auto ok_cfg = deepseek_ocr2_synthetic_config();
    ok_cfg.vision.down1 = DeepSeekConvDownsampler{16u, 3u, 1u, 1u};
    expect(ok_cfg.vision.down1.validate().ok()) << "grid-preserving stride-1 conv allowed";
    // stride 2 with excessive padding enlarges at the smallest legal input.
    auto bad2 = deepseek_ocr2_synthetic_config();
    bad2.vision.down1 = DeepSeekConvDownsampler{16u, 3u, 2u, 4u};
    expect(!bad2.vision.down1.validate().ok()) << "stride2 with excessive pad refused";
    // The declared synthetic downsamplers are valid.
    const auto good = deepseek_ocr2_synthetic_config();
    expect(good.vision.down1.validate().ok());
    expect(good.vision.down2.validate().ok());
}

// S1: tied word embeddings are refused (the MoE decoder ships a standalone head),
// and the preflight no longer invents tied-head bytes.
void test_tied_embeddings_refused() {
    auto cfg = deepseek_ocr2_synthetic_config();
    cfg.decoder.tie_word_embeddings = true;
    auto v = cfg.decoder.validate();
    expect(!v.ok()) << "tied word embeddings refused";
    expect(v.code == StatusCode::unsupported);
    expect(!cfg.validate().ok());
    expect(!deepseek_ocr2_preflight(cfg, 128, MemoryLimits{0, 0}).ok()) << "preflight refuses a tied config";
    // The valid (untied) preflight decoder bytes equal the inventory sum exactly:
    // there is no extra tied-head transpose buffer.
    const auto good = deepseek_ocr2_synthetic_config();
    auto rep = deepseek_ocr2_preflight(good, 128, MemoryLimits{0, 0});
    LUISA_ASSERT(rep.ok(), "untied preflight");
    uint64_t sum = 0;
    for (const auto &e : deepseek_ocr2_decoder_expected_tensors(good)) {
        uint64_t elems = 1;
        for (auto d : e.shape) { elems *= d; }
        sum += elems * 4ull;
    }
    expect(eq(rep.value->decoder_weight_bytes, sum)) << "decoder bytes == inventory (no tied-head bytes)";
}

// M8: compose MUST invoke the grounded parser: markers are stripped into
// canonical content and coordinates populate the block source geometry.
void test_compose_grounding() {
    const auto make_run = [](const luisa::string &content) {
        LayoutRunResult run;
        LayoutPageResult page;
        page.page_index = 0;
        page.source_width = 1000;
        page.source_height = 2000;
        LayoutRegionResult r;
        r.region_id = 0;
        r.page_index = 0;
        r.reading_order = 0;
        r.block_type = LayoutBlockType::text;
        r.content = content;
        page.regions.push_back(r);
        run.pages.push_back(page);
        return run;
    };
    // Valid grounding: canonical text has NO markers, and the box is populated.
    auto ok = deepseek_ocr2_compose(make_run("<|ref|>hello world<|/ref|><|det|>[[100,200,500,600]]<|/det|>"),
                                   LayoutOutputMode::markdown);
    expect(ok.ok()) << "valid grounded compose";
    if (ok.ok()) {
        expect(ok.value->text.find("<|ref|>") == luisa::string::npos) << "ref marker stripped";
        expect(ok.value->text.find("<|det|>") == luisa::string::npos) << "det marker stripped";
        expect(ok.value->text.find("hello world") != luisa::string::npos) << "canonical text kept";
        expect(eq(ok.value->blocks.size(), 1u));
        if (!ok.value->blocks.empty()) {
            const auto &b = ok.value->blocks[0];
            expect(!b.source_box.empty()) << "grounded block carries a non-empty source box";
            expect(std::abs(b.source_box.x0 - 100.0f) < 1e-3f);
            expect(std::abs(b.source_box.y0 - 400.0f) < 1e-3f);
            expect(std::abs(b.source_box.x1 - 500.0f) < 1e-3f);
            expect(std::abs(b.source_box.y1 - 1200.0f) < 1e-3f);
            expect(eq(b.polygon.size(), 4u)) << "box corners preserved";
        }
    }
    // Plain (ungrounded) content still composes.
    expect(deepseek_ocr2_compose(make_run("plain text"), LayoutOutputMode::markdown).ok());
    // Invalid grounding must FAIL CLOSED through compose (previously leaked).
    expect(!deepseek_ocr2_compose(make_run("<|ref|>x<|/ref|><|det|>[[0,0,1001,100]]<|/det|>"),
                                  LayoutOutputMode::markdown).ok()) << "out-of-range coords refused by compose";
    expect(!deepseek_ocr2_compose(make_run("<|ref|>x<|/ref|>"), LayoutOutputMode::markdown).ok())
        << "ref without det refused by compose";
    expect(!deepseek_ocr2_compose(make_run("<|ref|>x"), LayoutOutputMode::markdown).ok())
        << "unclosed ref refused by compose";
    expect(!deepseek_ocr2_compose(make_run("<|ref|>x<|/ref|><|det|>[[100,0,100,100]]<|/det|>"),
                                  LayoutOutputMode::markdown).ok()) << "degenerate box refused by compose";
    // Zero page dimensions cannot scale coordinates.
    {
        auto run = make_run("<|ref|>x<|/ref|><|det|>[[0,0,10,10]]<|/det|>");
        run.pages[0].source_width = 0;
        expect(!deepseek_ocr2_compose(run, LayoutOutputMode::markdown).ok()) << "zero page width refused";
    }
}

// S7: the coordinate range is the CLOSED [0,1000] (docs agree) and trailing
// commas are rejected.
void test_grounded_bounds_and_trailing_comma() {
    // 1000 is legal and maps to the full page edge.
    auto edge = deepseek_ocr2_parse_grounded("<|ref|>x<|/ref|><|det|>[[0,0,1000,1000]]<|/det|>", 100, 200);
    expect(edge.ok()) << "closed upper bound 1000 accepted";
    if (edge.ok()) {
        const auto &b = (*edge.value)[0].boxes[0];
        expect(std::abs(b.x1 - 100.0f) < 1e-3f);
        expect(std::abs(b.y1 - 200.0f) < 1e-3f);
    }
    expect(!deepseek_ocr2_parse_grounded("<|ref|>x<|/ref|><|det|>[[0,0,1001,1000]]<|/det|>", 100, 200).ok())
        << "1001 out of range";
    // Trailing comma before the closing bracket.
    expect(!deepseek_ocr2_parse_grounded("<|ref|>x<|/ref|><|det|>[[0,0,10,10],]<|/det|>", 100, 100).ok())
        << "trailing comma refused";
    expect(!deepseek_ocr2_parse_grounded("<|ref|>x<|/ref|><|det|>[[0,0,10,10], ]<|/det|>", 100, 100).ok())
        << "trailing comma with whitespace refused";
    // Two boxes without a trailing comma remain valid.
    expect(deepseek_ocr2_parse_grounded("<|ref|>x<|/ref|><|det|>[[0,0,10,10],[20,20,30,30]]<|/det|>", 100, 100).ok());
}

}// namespace

static auto test_deepseek_contract_registration = [] {
    "deepseek_expert_contract"_test = [] { test_expert_contract(); };
    "deepseek_config_parse"_test = [] { test_config_parse(); };
    "deepseek_signature_and_real_gate"_test = [] { test_signature_and_real_gate(); };
    "deepseek_memory_preflight"_test = [] { test_memory_preflight(); };
    "deepseek_grounded_markdown"_test = [] { test_grounded_markdown(); };
    "deepseek_compose_modes"_test = [] { test_compose_modes(); };
    "deepseek_parse_zero_heads"_test = [] { test_parse_zero_heads(); };
    "deepseek_downsampler_cannot_enlarge"_test = [] { test_downsampler_cannot_enlarge(); };
    "deepseek_tied_embeddings_refused"_test = [] { test_tied_embeddings_refused(); };
    "deepseek_compose_grounding"_test = [] { test_compose_grounding(); };
    "deepseek_grounded_bounds_and_trailing_comma"_test = [] { test_grounded_bounds_and_trailing_comma(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
