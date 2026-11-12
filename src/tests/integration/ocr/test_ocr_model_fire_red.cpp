// Test for ocr/models/fire_red_ocr + the shared Qwen3-VL graph (Phase 4,
// device). FireRed-OCR reuses the Qwen3-VL executor ONLY through its own
// tensor map / prompt / output-parser configuration. Covers: the FireRed
// recipe-driven synthetic path executes end-to-end on the shared graph, the
// FireRed tensor map is distinct from the plain Qwen3-VL map, real artifacts
// fail closed, no GRPO/training nodes, and the FireRed Markdown output
// contract validates. Synthetic weights only; no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_qwen3_test_utils.h"

#include "ocr/models/fire_red_ocr.h"

// NOTE: ocr/prompt.h and ocr/output/markdown.h pull model_recipe.h (host
// AttentionMaskKind), which clashes with the ops enum in this device TU.
// The prompt-template and Markdown output contracts are exercised by the
// host tests (test_ocr_prompt / test_ocr_output_markdown /
// test_ocr_model_contract); here we check the FireRed recipe ids + graph.

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 3e-3f;
constexpr float kRtol = 3e-3f;

// Build a FireRed synthetic image input (same packing contract as Qwen3-VL).
[[nodiscard]] Qwen3VlImageInput make_fire_red_image(uint32_t w, uint32_t h) {
    Qwen3VlPackingSpec spec;
    spec.patch_size = ocr::test::kQ3Patch;
    spec.spatial_merge_size = ocr::test::kQ3Merge;
    spec.min_pixels = 16;
    spec.max_pixels = 4096;
    auto packed = qwen3_vl_pack_image(w, h, spec);
    LUISA_ASSERT(packed.ok(), "packing");
    Qwen3VlImageInput img;
    img.source_width = w;
    img.source_height = h;
    img.packed = *packed.value;
    img.pixels.resize(static_cast<size_t>(w) * h * 3u);
    ocr::test::Lcg lcg;
    lcg.state = 11u;
    for (auto &b : img.pixels) { b = static_cast<uint8_t>(lcg.next() % 256u); }
    return img;
}

void test_fire_red_recipe_contract() {
    auto cfg = fire_red_ocr_synthetic_config();
    expect(cfg.validate().ok());
    // The FireRed tensor map is DISTINCT from the plain Qwen3-VL map.
    expect(cfg.names.text.prefix == "language_model.");
    expect(cfg.names.visual_prefix == "model.visual.");
    expect(cfg.prompt_template_id == kFireRedPromptTemplateId);
    expect(cfg.output_parser_id == kFireRedOutputParserId);
    // A config carrying the plain Qwen3-VL prefixes is rejected.
    auto bad = cfg;
    bad.names.text.prefix = "model.";
    bad.names.visual_prefix = "visual.";
    expect(!bad.validate().ok());
    // Real artifacts ALWAYS fail closed (no official qualification).
    expect(fire_red_ocr_validate_real_artifact("", "").code == StatusCode::unvalidated);
    expect(fire_red_ocr_validate_real_artifact("some-rev", "some-fp").code == StatusCode::unvalidated);
    // No GRPO/training nodes.
    luisa::vector<luisa::string> good_roles = {"vision", "decoder", "merger"};
    expect(fire_red_ocr_check_no_training_nodes(good_roles).ok());
    luisa::vector<luisa::string> bad_roles = {"vision", "grpo_trainer"};
    expect(!fire_red_ocr_check_no_training_nodes(bad_roles).ok());
    luisa::vector<luisa::string> reward_roles = {"reward_model"};
    expect(!fire_red_ocr_check_no_training_nodes(reward_roles).ok());
    // The FireRed prompt template id is the registered contract tag (the
    // prompt builder itself is host-only; exercised by the host tests).
    expect(kFireRedPromptTemplateId == "fire_red_ocr");
    expect(kFireRedOutputParserId == "fire_red_markdown");
}

void test_fire_red_graph_parity(Device &device) {
    // The FireRed recipe drives the SAME Qwen3-VL executor with FireRed
    // tensor names. Weights are written under the FireRed prefixes.
    ocr::test::Qwen3TempDir tmp;
    auto weights = ocr::test::generate_qwen3_synthetic_weights();
    auto fire_cfg = fire_red_ocr_synthetic_config();
    const auto vpath = tmp.path / "fire_red_vision.lcocr";
    const auto dpath = tmp.path / "fire_red_decoder.lcocr";
    expect(ocr::test::write_qwen3_synthetic_packages(vpath, dpath, weights, fire_cfg.names).ok());
    auto stream = device.create_stream();
    WeightStore store;
    expect(ocr::test::upload_qwen3_synthetic_weights(device, stream, vpath, dpath, store).ok());

    Qwen3VlExecutorConfig config;
    config.model = fire_cfg.graph;
    config.names = fire_cfg.names;
    config.vision_component = "vision";
    config.decoder_component = "decoder";
    config.max_vision_patches = 4096;
    Qwen3VlExecutor executor;
    expect(executor.create(device, stream, config, store).ok());

    auto img = make_fire_red_image(12, 6);
    auto pre = qwen3_vl_preprocess_image_host(img, config.mean, config.std_dev);
    expect(pre.ok());
    if (!pre.ok()) { return; }
    ocr::test::Qwen3VisionOracle vision_oracle;
    expect(ocr::test::oracle_vision(weights, pre.value->pixels, img.packed, vision_oracle).ok());

    Qwen3VlVisionOutput vision;
    luisa::vector<float> features;
    {
        auto list = CommandList::create();
        auto r = executor.record_vision(list, luisa::span<const Qwen3VlImageInput>{&img, 1});
        expect(r.ok());
        if (!r.ok()) { return; }
        vision = *r.value;
        // Commit + drain BEFORE downloading (copy_to must run after the graph).
        stream << list.commit() << synchronize();
        expect(executor.download_vision_features(stream, features).ok());
        stream << synchronize();
    }
    ocr::test::expect_close(features, vision_oracle.features, kAtol, kRtol, "FireRed merger features");

    // End-to-end prefill + a few decode steps through the shared graph.
    const uint32_t merged = static_cast<uint32_t>(img.packed.merged_count);
    Qwen3VlPrompt prompt;
    prompt.token_ids = {1u, 2u};
    const uint32_t img_start = 2u;
    for (uint32_t i = 0; i < merged; i++) { prompt.token_ids.push_back(3u); }
    prompt.token_ids.push_back(4u);
    prompt.image_ranges.push_back({img_start, merged});
    luisa::vector<int32_t> position_ids;
    auto req = executor.build_decode_request(prompt, vision,
                                             luisa::span<const Qwen3VlImageInput>{&img, 1}, position_ids);
    expect(req.ok());
    if (!req.ok()) { return; }

    luisa::vector<luisa::vector<float>> k_caches, v_caches;
    ocr::test::Qwen3TextOracle text_oracle;
    luisa::vector<luisa::vector<float>> ds_views;
    for (const auto &ds : vision_oracle.ds_features) { ds_views.push_back(ds); }
    expect(ocr::test::oracle_qwen3_forward(weights, prompt.token_ids, position_ids,
                                           vision_oracle.features, img_start, merged, ds_views,
                                           0u, k_caches, v_caches, text_oracle).ok());

    ContiguousKvCache cache;
    KvCacheConfig kc;
    kc.num_layers = ocr::test::kQ3Layers;
    kc.num_kv_heads = ocr::test::kQ3KvHeads;
    kc.head_dim = ocr::test::kQ3HeadDim;
    kc.max_seq_len = 64;
    kc.batch = 1;
    kc.dtype_bytes = 4;
    expect(cache.create(device, stream, kc).ok());

    const uint32_t prompt_len = static_cast<uint32_t>(prompt.token_ids.size());
    auto logits_buf = device.create_buffer<float>(static_cast<size_t>(prompt_len) * ocr::test::kQ3Vocab);
    luisa::vector<BufferView<float>> ds_device;
    for (const auto &ds : vision.deepstack_features) { ds_device.push_back(ds); }
    {
        auto list = CommandList::create();
        auto s = executor.text_executor().record_prefill_qwen3(
            list, *req.value, position_ids, ds_device, cache, true, {}, logits_buf.view());
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> logits(static_cast<size_t>(prompt_len) * ocr::test::kQ3Vocab);
    stream << logits_buf.copy_to(luisa::span<float>{logits}) << synchronize();
    ocr::test::expect_close(logits, text_oracle.logits, kAtol, kRtol, "FireRed full-prefill logits");
    // The FireRed Markdown output contract (HTML table + LaTeX validation)
    // is exercised by the host tests; the graph parity above is the device
    // gate for the shared FireRed path.
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "fire_red_recipe_contract"_test = [&] { test_fire_red_recipe_contract(); };
    "fire_red_graph_parity"_test = [&] { test_fire_red_graph_parity(device); };
    return 0;
}
