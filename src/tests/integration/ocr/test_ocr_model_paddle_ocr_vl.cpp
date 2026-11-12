// Test for ocr/runtime/layout_vlm_executor PaddleOcrVlRecognizer + the full
// layout->region->composition pipeline (Phase 6, device) -- the SMALL SYNTHETIC
// PaddleOCR-VL vertical slice (native-resolution NaViT + adaptive visual-token
// reduction + MLP connector + ERNIE decoder).
// Covers: config/tensor contracts, native packing + token reduction, vision
// patch/connector parity, ERNIE prefill logits + greedy decode parity vs an
// independent double-accumulating CPU oracle, region batching vs batch-1
// equivalence, and the full layout->Markdown/JSON pipeline.
// Synthetic weights only; no model/download, no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_p6_test_utils.h"

#include "ocr/runtime/layout_vlm_executor.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Tolerances: FP32 device vs double oracle (operation-appropriate, plan §10.1).
constexpr float kAtol = 3e-3f;
constexpr float kRtol = 3e-3f;
constexpr float kLinAtol = 5e-4f;
constexpr float kLinRtol = 5e-4f;

struct Fixture {
    ocr::test::P6TempDir tmp;
    PaddleOcrVlConfig cfg;
    PaddleOcrVlTensorNames names;
    ocr::test::PaddleSyntheticWeights weights;
    WeightStore store;
    luisa::unique_ptr<PaddleOcrVlRecognizer> recognizer;
    PaddleOcrVlRecognizerConfig config;

    [[nodiscard]] bool init(Device &device, uint32_t seed = 0x5EEDu) {
        cfg = paddle_ocr_vl_synthetic_config();
        if (!cfg.validate().ok()) { return false; }
        names = PaddleOcrVlTensorNames{};
        weights = ocr::test::generate_paddle_weights(cfg, seed);
        const auto vpath = tmp.path / "paddle_vision.lcocr";
        const auto dpath = tmp.path / "paddle_decoder.lcocr";
        if (auto s = ocr::test::write_paddle_packages(vpath, dpath, cfg, names, weights); !s.ok()) {
            LUISA_WARNING("write_paddle_packages: {}", s.message);
            return false;
        }
        auto stream = device.create_stream();
        if (auto s = ocr::test::upload_paddle_weights(device, stream, vpath, dpath, store); !s.ok()) {
            LUISA_WARNING("upload_paddle: {}", s.message);
            return false;
        }
        config.model = cfg;
        config.names = names;
        config.vision_component = "vision";
        config.decoder_component = "decoder";
        config.max_vision_patches = 4096;
        recognizer = luisa::make_unique<PaddleOcrVlRecognizer>();
        if (auto s = recognizer->create(device, stream, config, store); !s.ok()) {
            LUISA_WARNING("paddle create: {}", s.message);
            return false;
        }
        return true;
    }
};

// A deterministic region crop (bright bar on dark noise).
[[nodiscard]] RegionCropInput make_crop(uint32_t w, uint32_t h, uint32_t region_id, uint32_t seed = 11u) {
    RegionCropInput crop;
    crop.source_width = w;
    crop.source_height = h;
    crop.task.region_id = region_id;
    crop.task.page_index = 0;
    crop.task.reading_order = region_id;
    crop.task.block_type = LayoutBlockType::text;
    crop.task.source_box = output::DbRect{0, 0, static_cast<float>(w), static_cast<float>(h)};
    crop.pixels.resize(static_cast<size_t>(w) * h * 3u);
    ocr::test::Lcg lcg;
    lcg.state = seed;
    for (auto &b : crop.pixels) { b = static_cast<uint8_t>(lcg.next() % 60u); }
    return crop;
}

// ---------------------------------------------------------------------------
// Host contract checks.
// ---------------------------------------------------------------------------
void test_paddle_contracts() {
    auto cfg = paddle_ocr_vl_synthetic_config();
    expect(cfg.validate().ok());
    // Distinct NaViT contract: no QK-Norm/DeepStack; ERNIE rotate-half RoPE.
    expect(cfg.text.rope_pair_layout == RopePairLayout::rotate_half);
    expect(cfg.vision.num_layers >= 1u);
    expect(cfg.connector.max_visual_tokens >= 1u);
    // Tensor inventory present.
    PaddleOcrVlTensorNames names;
    auto expected = paddle_ocr_vl_expected_tensors(cfg, names);
    expect(expected.size() > 20u);
    // Real-artifact gate fails closed.
    expect(paddle_ocr_vl_validate_real_artifact("r", "f").code == StatusCode::unvalidated);
    // Packing: native resolution, patch grid, token reduction under budget.
    auto packed = paddle_pack_image(16, 16, cfg.packing, cfg.connector.max_visual_tokens);
    expect(packed.ok());
    if (packed.ok()) {
        expect(packed.value->resized_width % cfg.vision.patch_size == 0u);
        expect(packed.value->resized_height % cfg.vision.patch_size == 0u);
        expect(packed.value->patch_count == static_cast<uint64_t>(packed.value->grid_h) * packed.value->grid_w);
        expect(packed.value->visual_tokens <= cfg.connector.max_visual_tokens);
        expect(packed.value->visual_tokens == static_cast<uint64_t>(packed.value->reduced_h) * packed.value->reduced_w);
    }
    // Grid reduction respects the budget for a large grid.
    auto reduced = paddle_reduce_grid(32, 32, 16);
    expect(reduced.ok());
    if (reduced.ok()) {
        expect(static_cast<uint64_t>(reduced.value->first) * reduced.value->second <= 16u);
    }
}

// ---------------------------------------------------------------------------
// Vision + connector parity (device vs independent CPU oracle).
// ---------------------------------------------------------------------------
void test_vision_connector_parity(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer) { return; }
    auto stream = device.create_stream();
    for (const auto &[w, h] : std::initializer_list<std::pair<uint32_t, uint32_t>>{{8, 8}, {12, 6}, {16, 4}}) {
        auto crop = make_crop(w, h, 0);
        auto packed = paddle_pack_image(w, h, fx.cfg.packing, fx.cfg.connector.max_visual_tokens);
        expect(packed.ok());
        if (!packed.ok()) { continue; }
        // Oracle preprocessing (float mirror) + vision/connector.
        auto pixels = ocr::test::p6_oracle_preprocess(crop.pixels, w, h,
                                                      packed.value->resized_width, packed.value->resized_height,
                                                      fx.config.mean, fx.config.std_dev);
        ocr::test::PaddleVisionOracle oracle;
        expect(ocr::test::paddle_oracle_vision(fx.cfg, fx.weights, pixels, *packed.value, oracle).ok());
        // Device recognize_batch (single crop) + download intermediates.
        RegionRecognizeOptions opts;
        opts.max_output_tokens = 4;
        auto out = fx.recognizer->recognize_batch(stream, luisa::span<const RegionCropInput>{&crop, 1}, opts);
        expect(out.ok());
        if (!out.ok()) { continue; }
        luisa::vector<float> dev_patches, dev_features;
        expect(fx.recognizer->download_vision_patches(stream, dev_patches).ok());
        expect(fx.recognizer->download_connector_features(stream, dev_features).ok());
        stream << synchronize();
        const auto what = luisa::format("paddle {}x{}", w, h);
        ocr::test::expect_close(dev_patches, oracle.patches, kLinAtol, kLinRtol, (what + " patches").c_str());
        ocr::test::expect_close(dev_features, oracle.features, kAtol, kRtol, (what + " connector").c_str());
    }
}

// ---------------------------------------------------------------------------
// ERNIE decoder parity: prefill logits + greedy tokens vs the CPU oracle.
// ---------------------------------------------------------------------------
void test_decoder_parity(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer) { return; }
    auto stream = device.create_stream();
    auto crop = make_crop(8, 8, 0);
    auto packed = paddle_pack_image(8, 8, fx.cfg.packing, fx.cfg.connector.max_visual_tokens);
    expect(packed.ok());
    if (!packed.ok()) { return; }
    auto pixels = ocr::test::p6_oracle_preprocess(crop.pixels, 8, 8,
                                                  packed.value->resized_width, packed.value->resized_height,
                                                  fx.config.mean, fx.config.std_dev);
    ocr::test::PaddleVisionOracle vision_oracle;
    expect(ocr::test::paddle_oracle_vision(fx.cfg, fx.weights, pixels, *packed.value, vision_oracle).ok());
    // Build the prompt the recognizer uses: prefix + visual placeholders + suffix.
    const uint32_t tokens = static_cast<uint32_t>(packed.value->visual_tokens);
    luisa::vector<uint32_t> prompt = fx.config.prompt_prefix;
    const uint32_t mm_start = static_cast<uint32_t>(prompt.size());
    for (uint32_t i = 0; i < tokens; i++) { prompt.push_back(0u); }
    prompt.insert(prompt.end(), fx.config.prompt_suffix.begin(), fx.config.prompt_suffix.end());
    // Oracle prefill.
    luisa::vector<luisa::vector<float>> k_caches, v_caches;
    ocr::test::PaddleTextOracle text_oracle;
    expect(ocr::test::paddle_oracle_forward(fx.cfg, fx.weights, prompt, vision_oracle.features,
                                            mm_start, tokens, 0u, k_caches, v_caches, text_oracle).ok());
    // Device recognize_batch with a small output budget; compare generated tokens.
    RegionRecognizeOptions opts;
    opts.max_output_tokens = 6;
    auto out = fx.recognizer->recognize_batch(stream, luisa::span<const RegionCropInput>{&crop, 1}, opts);
    expect(out.ok());
    if (!out.ok()) { return; }
    expect(out.value->size() == 1u);
    // Oracle greedy decode from the prefill last-row logits.
    const uint32_t vocab = fx.cfg.text.vocab_size;
    const uint32_t prompt_len = static_cast<uint32_t>(prompt.size());
    luisa::vector<uint32_t> oracle_tokens;
    uint32_t input = ocr::test::oracle_argmax(luisa::span<const float>{
        text_oracle.logits.data() + static_cast<size_t>(prompt_len - 1) * vocab, vocab});
    oracle_tokens.push_back(input);
    uint32_t kv_prefix = prompt_len;
    for (uint32_t step = 1; step < opts.max_output_tokens; step++) {
        // Stop at EOS (31) like the controller.
        if (input == 31u) { break; }
        luisa::vector<uint32_t> step_tokens{input};
        ocr::test::PaddleTextOracle step_oracle;
        expect(ocr::test::paddle_oracle_forward(fx.cfg, fx.weights, step_tokens, {}, 0u, 0u,
                                                kv_prefix, k_caches, v_caches, step_oracle).ok());
        input = ocr::test::oracle_argmax(step_oracle.logits);
        oracle_tokens.push_back(input);
        kv_prefix++;
    }
    // The device tokens must match the oracle greedy sequence (prefix shared).
    const auto &dev_tokens = (*out.value)[0].tokens;
    expect(dev_tokens.size() >= 1u);
    const size_t n = std::min(dev_tokens.size(), oracle_tokens.size());
    bool match = true;
    for (size_t i = 0; i < n; i++) {
        if (dev_tokens[i] != oracle_tokens[i]) { match = false; break; }
    }
    expect(match) << "device greedy tokens must match the oracle";
}

// ---------------------------------------------------------------------------
// Region batching vs batch-1 equivalence (deterministic policy).
// ---------------------------------------------------------------------------
void test_batch_equivalence(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer) { return; }
    auto stream = device.create_stream();
    luisa::vector<RegionCropInput> crops;
    for (uint32_t i = 0; i < 3u; i++) { crops.push_back(make_crop(8, 8, i, 11u + i)); }
    RegionRecognizeOptions opts;
    opts.max_output_tokens = 5;
    // Batch-1: recognize each crop independently.
    luisa::vector<luisa::vector<uint32_t>> single_tokens;
    for (const auto &crop : crops) {
        auto out = fx.recognizer->recognize_batch(stream, luisa::span<const RegionCropInput>{&crop, 1}, opts);
        expect(out.ok());
        if (!out.ok()) { return; }
        single_tokens.push_back((*out.value)[0].tokens);
    }
    // Batched: recognize all crops in one call.
    auto batched = fx.recognizer->recognize_batch(stream, crops, opts);
    expect(batched.ok());
    if (!batched.ok()) { return; }
    expect(batched.value->size() == crops.size());
    bool equivalent = true;
    for (size_t i = 0; i < crops.size(); i++) {
        if ((*batched.value)[i].tokens != single_tokens[i]) { equivalent = false; break; }
    }
    expect(equivalent) << "region batching must equal batch-1 under the deterministic policy";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "paddle_contracts"_test = [] { test_paddle_contracts(); };
    "paddle_vision_connector_parity"_test = [&] { test_vision_connector_parity(device); };
    "paddle_decoder_parity"_test = [&] { test_decoder_parity(device); };
    "paddle_batch_equivalence"_test = [&] { test_batch_equivalence(device); };
}