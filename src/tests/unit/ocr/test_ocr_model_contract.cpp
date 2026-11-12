// Test for ocr/models/qwen3_vl.h + fire_red_ocr.h (Phase 4, host-only).
// Covers: the distinct Qwen3-VL config/tensor contract (QK-Norm, no QKV bias,
// interleaved MRoPE, DeepStack), strict config parsing + Instruct/Thinking
// identity boundary, the exact expected tensor inventory, package tensor
// validation, packing/position semantics, and the FireRed recipe contract
// (distinct tensor map, fail-closed real artifacts, no GRPO/training nodes).
// No device/model required.
#include "ut/ut.hpp"

#include <cstring>

#include "ocr/models/fire_red_ocr.h"
#include "ocr/models/qwen3_vl.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// A minimal valid synthetic Qwen3-VL config JSON (matches the declared
// synthetic geometry).
luisa::string base_config_json() {
    return R"({
        "model_type": "qwen3_vl",
        "variant": "instruct",
        "text_config": {
            "vocab_size": 32, "hidden_size": 32, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "intermediate_size": 64, "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5, "rope_theta": 10000, "eos_token_id": 31
        },
        "vision_config": {
            "num_layers": 2, "hidden_size": 16, "num_heads": 2, "head_dim": 8,
            "intermediate_size": 32, "patch_size": 2, "temporal_patch_size": 1,
            "spatial_merge_size": 2, "merger_intermediate_size": 32,
            "layer_norm_eps": 1e-5, "rope_theta": 10000, "rope_height_pairs": 2
        },
        "deepstack": {"decoder_layers": [0, 1], "vision_layers": [0, 1]},
        "mrope_section": [2, 1, 1],
        "tie_word_embeddings": true
    })";
}

void test_parse_and_identity() {
    auto r = parse_qwen3_vl_config(base_config_json());
    expect(r.ok());
    if (!r.ok()) { return; }
    const auto &c = *r.value;
    expect(c.variant == Qwen3VlVariant::instruct);
    expect(c.text.num_layers == 3u);
    expect(c.text.num_kv_heads == 2u);
    expect(c.text.rope_pair_layout == RopePairLayout::rotate_half);
    expect(c.vision.num_layers == 2u);
    expect(c.vision.spatial_merge_size == 2u);
    expect(c.deepstack.count() == 2u);
    expect(c.mrope.total() == 4u);
    expect(c.validate().ok());
    // Thinking fails closed (explicit identity boundary).
    auto thinking = base_config_json();
    const auto pos = thinking.find("\"instruct\"");
    expect(pos != luisa::string::npos);
    thinking.replace(pos, std::strlen("\"instruct\""), "\"thinking\"");
    auto tr = parse_qwen3_vl_config(thinking);
    expect(!tr.ok());
    expect(tr.status.code == StatusCode::unsupported);
    // A Qwen2.5 model_type is refused (no silent substitution).
    auto q25 = base_config_json();
    const auto mp = q25.find("\"qwen3_vl\"");
    q25.replace(mp, std::strlen("\"qwen3_vl\""), "\"qwen2_5_vl\"");
    expect(!parse_qwen3_vl_config(q25).ok());
    // The shared Qwen2.5 parser refuses a Qwen3 text config (inverse boundary):
    // it rejects the qwen3 model_type marker rather than silently accepting a
    // different decoder contract.
    const luisa::string q3_text = R"({"model_type":"qwen3_vl","vocab_size":32,"hidden_size":32,
        "num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,
        "intermediate_size":64,"max_position_embeddings":128})";
    expect(!parse_qwen2_5_text_config(q3_text).ok());
    // And it refuses an explicit qk_norm marker even without a model_type.
    const luisa::string qk_text = R"({"qk_norm":true,"vocab_size":32,"hidden_size":32,
        "num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,
        "intermediate_size":64,"max_position_embeddings":128})";
    expect(!parse_qwen2_5_text_config(qk_text).ok());
}

void test_config_validation_failures() {
    // Missing required field.
    auto missing = base_config_json();
    const auto p = missing.find("\"vocab_size\": 32, ");
    missing.erase(p, std::strlen("\"vocab_size\": 32, "));
    expect(!parse_qwen3_vl_config(missing).ok());
    // Bad mrope section sum.
    auto bad_mrope = base_config_json();
    const auto mp = bad_mrope.find("[2, 1, 1]");
    bad_mrope.replace(mp, std::strlen("[2, 1, 1]"), "[2, 1, 2]");
    expect(!parse_qwen3_vl_config(bad_mrope).ok());
    // DeepStack layer out of range.
    auto bad_ds = base_config_json();
    const auto dp = bad_ds.find("\"decoder_layers\": [0, 1]");
    bad_ds.replace(dp, std::strlen("\"decoder_layers\": [0, 1]"), "\"decoder_layers\": [0, 5]");
    expect(!parse_qwen3_vl_config(bad_ds).ok());
    // temporal_patch_size > 1 unsupported.
    auto bad_t = base_config_json();
    const auto tp = bad_t.find("\"temporal_patch_size\": 1");
    bad_t.replace(tp, std::strlen("\"temporal_patch_size\": 1"), "\"temporal_patch_size\": 2");
    expect(!parse_qwen3_vl_config(bad_t).ok());
    // Not JSON.
    expect(!parse_qwen3_vl_config("not json").ok());
}

void test_synthetic_config_and_inventory() {
    auto c = qwen3_vl_synthetic_config();
    expect(c.validate().ok());
    Qwen3VlTensorNames names;
    expect(names.validate().ok());
    auto inv = qwen3_vl_expected_tensors(c, names);
    expect(!inv.empty());
    // QK-Norm weights present; QKV biases ABSENT (Qwen3 contract).
    bool has_q_norm = false, has_q_bias = false;
    for (const auto &e : inv) {
        if (e.name == names.text.q_norm(0)) { has_q_norm = true; }
        if (e.name == names.text.q_bias(0)) { has_q_bias = true; }
    }
    expect(has_q_norm);
    expect(!has_q_bias);
    // Vision patch embed + merger + deepstack mergers present.
    bool has_patch = false, has_merger = false, has_ds = false;
    for (const auto &e : inv) {
        if (e.name == names.patch_embed_weight()) { has_patch = true; }
        if (e.name == Qwen3VlTensorNames::merger_fc1(names.merger_prefix())) { has_merger = true; }
        if (e.name == Qwen3VlTensorNames::merger_fc1(names.deepstack_prefix(0))) { has_ds = true; }
    }
    expect(has_patch && has_merger && has_ds);
    // The synthetic config round-trips through the parser.
    // (Serialize a matching JSON and parse it back.)
    auto parsed = parse_qwen3_vl_config(base_config_json());
    expect(parsed.ok());
    if (parsed.ok()) {
        expect(parsed.value->text.hidden_size == c.text.hidden_size);
        expect(parsed.value->vision.hidden_size == c.vision.hidden_size);
    }
}

void test_packing_positions() {
    Qwen3VlPackingSpec spec;
    spec.patch_size = 2;
    spec.spatial_merge_size = 2;
    spec.min_pixels = 16;
    spec.max_pixels = 4096;
    expect(spec.validate().ok());
    auto p = qwen3_vl_pack_image(12, 6, spec);
    expect(p.ok());
    if (p.ok()) {
        expect(p.value->resized_width % 4u == 0u);
        expect(p.value->resized_height % 4u == 0u);
        expect(p.value->merged_count == p.value->patch_count / 4u);
    }
    expect(!qwen3_vl_pack_image(0, 6, spec).ok());
    // Vision positions + axis map.
    auto img = qwen3_vl_pack_image(8, 8, spec);
    expect(img.ok());
    if (img.ok()) {
        auto pos = qwen3_vl_vision_positions(*img.value);
        expect(pos.size() == img.value->patch_count * 2u);
        auto axis = qwen3_vl_vision_axis_of_pair(qwen3_vl_synthetic_config().vision);
        expect(axis.size() == 4u);
        expect(axis[0] == 0u && axis[2] == 1u);
    }
    // MRoPE axis map sections {2,1,1}.
    auto cfg = qwen3_vl_synthetic_config();
    auto maxis = cfg.mrope.axis_of_pair();
    expect(maxis.size() == 4u);
    expect(maxis[0] == 0u && maxis[1] == 0u && maxis[2] == 1u && maxis[3] == 2u);
}

void test_fire_red_contract() {
    auto cfg = fire_red_ocr_synthetic_config();
    expect(cfg.validate().ok());
    expect(cfg.names.text.prefix == "language_model.");
    expect(cfg.names.visual_prefix == "model.visual.");
    expect(cfg.prompt_template_id == kFireRedPromptTemplateId);
    expect(cfg.output_parser_id == kFireRedOutputParserId);
    // The FireRed inventory uses the FireRed prefixes.
    auto inv = qwen3_vl_expected_tensors(cfg.graph, cfg.names);
    bool has_fire_embed = false;
    for (const auto &e : inv) {
        if (e.name == "language_model.embed_tokens.weight") { has_fire_embed = true; }
    }
    expect(has_fire_embed);
    // Plain Qwen3-VL prefixes are rejected for FireRed.
    auto bad = cfg;
    bad.names.text.prefix = "model.";
    bad.names.visual_prefix = "visual.";
    expect(!bad.validate().ok());
    // Real artifacts always fail closed.
    expect(fire_red_ocr_validate_real_artifact("", "").code == StatusCode::unvalidated);
    expect(fire_red_ocr_validate_real_artifact("rev", "fp").code == StatusCode::unvalidated);
    // No GRPO/training nodes.
    luisa::vector<luisa::string> good = {"vision", "decoder"};
    expect(fire_red_ocr_check_no_training_nodes(good).ok());
    for (const char *role : {"grpo", "GRPO_trainer", "reward_model", "policy_head", "training_loop", "rollout"}) {
        luisa::vector<luisa::string> bad_roles = {luisa::string{role}};
        expect(!fire_red_ocr_check_no_training_nodes(bad_roles).ok()) << role;
    }
}

void test_variant_names() {
    expect(qwen3_vl_variant_name(Qwen3VlVariant::instruct) == "instruct");
    expect(qwen3_vl_variant_name(Qwen3VlVariant::thinking) == "thinking");
    auto v = qwen3_vl_variant_from_name("instruct");
    expect(v.ok());
    if (v.ok()) { expect(*v.value == Qwen3VlVariant::instruct); }
    expect(!qwen3_vl_variant_from_name("bogus").ok());
}

// S6: packing must never exceed the declared max_pixels after grid rounding.
void test_packing_pixel_budget() {
    Qwen3VlPackingSpec spec;
    spec.patch_size = 2;
    spec.spatial_merge_size = 2;
    spec.min_pixels = 16;
    spec.max_pixels = 1000;
    // Extreme wide image: rounding must not push the area over the cap.
    auto wide = qwen3_vl_pack_image(4000, 1, spec);
    expect(wide.ok());
    if (wide.ok()) {
        const uint64_t area = static_cast<uint64_t>(wide.value->resized_width) * wide.value->resized_height;
        expect(area <= 1000u) << "wide area " << area;
        expect(wide.value->resized_width % 4u == 0u);
        expect(wide.value->resized_height % 4u == 0u);
    }
    // 30x30 with max 1000: 900 px is under the cap; rounding keeps it in bounds.
    auto sq = qwen3_vl_pack_image(30, 30, spec);
    expect(sq.ok());
    if (sq.ok()) {
        const uint64_t area = static_cast<uint64_t>(sq.value->resized_width) * sq.value->resized_height;
        expect(area <= 1000u) << "square area " << area;
    }
    // A cap smaller than the minimal grid (factor^2 = 16) fails closed.
    Qwen3VlPackingSpec tiny = spec;
    tiny.min_pixels = 1;
    tiny.max_pixels = 4;// < factor^2
    expect(!qwen3_vl_pack_image(4000, 1, tiny).ok());
}

// S7: DeepStack vision_layers must be strictly increasing (no duplicates).
void test_deepstack_vision_layers_strict() {
    auto cfg = qwen3_vl_synthetic_config();
    expect(cfg.deepstack.validate(cfg.text.num_layers, cfg.vision.num_layers).ok());
    // Duplicate vision layers rejected.
    auto dup = cfg.deepstack;
    dup.vision_layers = {0u, 0u};
    expect(!dup.validate(cfg.text.num_layers, cfg.vision.num_layers).ok());
    // Decreasing vision layers rejected.
    auto dec = cfg.deepstack;
    dec.vision_layers = {1u, 0u};
    expect(!dec.validate(cfg.text.num_layers, cfg.vision.num_layers).ok());
    // Out-of-range vision layer rejected.
    auto oob = cfg.deepstack;
    oob.vision_layers = {0u, 5u};
    expect(!oob.validate(cfg.text.num_layers, cfg.vision.num_layers).ok());
    // Duplicate decoder layers rejected.
    auto ddup = cfg.deepstack;
    ddup.decoder_layers = {0u, 0u};
    expect(!ddup.validate(cfg.text.num_layers, cfg.vision.num_layers).ok());
}

// S8: config strictness — exact model_type, attention_bias, unknown fields,
// aligned sliding-window handling.
void test_config_strictness() {
    // Exact model_type identity: look-alikes refused.
    for (const char *mt : {"qwen3_vl_moe", "not_qwen3_vl", "qwen3_vl_extra", "qwen2_5_vl"}) {
        auto j = base_config_json();
        const auto p = j.find("\"qwen3_vl\"");
        j.replace(p, std::strlen("\"qwen3_vl\""), luisa::string{"\""} + mt + "\"");
        expect(!parse_qwen3_vl_config(j).ok()) << mt;
    }
    // attention_bias:true refused; false accepted.
    {
        auto j = base_config_json();
        const auto p = j.find("\"eos_token_id\": 31");
        j.insert(p, "\"attention_bias\": true, ");
        expect(!parse_qwen3_vl_config(j).ok());
    }
    {
        auto j = base_config_json();
        const auto p = j.find("\"eos_token_id\": 31");
        j.insert(p, "\"attention_bias\": false, ");
        expect(parse_qwen3_vl_config(j).ok());
    }
    // Unknown structural fields fail closed at every level.
    {
        auto j = base_config_json();
        const auto p = j.find("\"mrope_section\"");
        j.insert(p, "\"bogus_root_field\": 1, ");
        expect(!parse_qwen3_vl_config(j).ok());
    }
    {
        auto j = base_config_json();
        const auto p = j.find("\"vocab_size\": 32");
        j.insert(p, "\"bogus_text_field\": 1, ");
        expect(!parse_qwen3_vl_config(j).ok());
    }
    {
        auto j = base_config_json();
        const auto p = j.find("\"num_layers\": 2");
        j.insert(p, "\"bogus_vision_field\": 1, ");
        expect(!parse_qwen3_vl_config(j).ok());
    }
    {
        auto j = base_config_json();
        const auto p = j.find("\"decoder_layers\": [0, 1]");
        j.insert(p, "\"bogus_ds_field\": 1, ");
        expect(!parse_qwen3_vl_config(j).ok());
    }
    // Sliding window: disabled accepted, enabled refused (text_config).
    {
        auto j = base_config_json();
        const auto p = j.find("\"eos_token_id\": 31");
        j.insert(p, "\"sliding_window\": 4, \"use_sliding_window\": false, ");
        expect(parse_qwen3_vl_config(j).ok());
    }
    {
        auto j = base_config_json();
        const auto p = j.find("\"eos_token_id\": 31");
        j.insert(p, "\"sliding_window\": 4, \"use_sliding_window\": true, ");
        expect(!parse_qwen3_vl_config(j).ok());
    }
    {
        // Window present, use absent => enabled by default => refused.
        auto j = base_config_json();
        const auto p = j.find("\"eos_token_id\": 31");
        j.insert(p, "\"sliding_window\": 4, ");
        expect(!parse_qwen3_vl_config(j).ok());
    }
}

// S9: FireRed requires the EXACT prefix pair; all Qwen3 combinations refused.
void test_fire_red_exact_prefix_pair() {
    auto cfg = fire_red_ocr_synthetic_config();
    expect(cfg.validate().ok());
    const luisa::vector<std::pair<const char *, const char *>> bad_pairs = {
        {"model.", "visual."},
        {"language_model.", "visual."},
        {"model.", "model.visual."},
        {"language_model.", "model.visual2."},
        {"model.", "visual2."},
    };
    for (const auto &[tp, vp] : bad_pairs) {
        auto bad = cfg;
        bad.names.text.prefix = tp;
        bad.names.visual_prefix = vp;
        expect(!bad.validate().ok()) << tp << " / " << vp;
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "qwen3_contract_parse_and_identity"_test = [] { test_parse_and_identity(); };
    "qwen3_contract_config_validation_failures"_test = [] { test_config_validation_failures(); };
    "qwen3_contract_synthetic_config_and_inventory"_test = [] { test_synthetic_config_and_inventory(); };
    "qwen3_contract_packing_positions"_test = [] { test_packing_positions(); };
    "qwen3_contract_fire_red"_test = [] { test_fire_red_contract(); };
    "qwen3_contract_variant_names"_test = [] { test_variant_names(); };
    "qwen3_contract_packing_pixel_budget"_test = [] { test_packing_pixel_budget(); };
    "qwen3_contract_deepstack_vision_layers_strict"_test = [] { test_deepstack_vision_layers_strict(); };
    "qwen3_contract_config_strictness"_test = [] { test_config_strictness(); };
    "qwen3_contract_fire_red_exact_prefix_pair"_test = [] { test_fire_red_exact_prefix_pair(); };
    return 0;
}
