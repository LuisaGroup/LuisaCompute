// Test for ocr/models/qwen2_5_vl.h (Phase 3, host-only).
// Covers: strict Qwen2.5 text-config parsing (integer-valued rope_theta/
// rms_norm_eps via yyjson_get_num), rotate-half RoPE selection, Qwen3/MoE/GDN
// fail-closed, disabled-vs-enabled sliding_window, the exact expected tensor
// inventory (standalone lm_head [hidden,vocab] packed kn), and per-weight
// shape/layout/element-count validation. No device/model required.
#include "ut/ut.hpp"

#include <cstring>

#include "ocr/models/qwen2_5_vl.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// A minimal valid Qwen2.5-style text config (rotate-half is selected).
luisa::string base_config() {
    return R"({
        "model_type": "qwen2_5_vl",
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0
    })";
}

void test_parse_basic_and_rotate_half() {
    auto r = parse_qwen2_5_text_config(base_config());
    expect(r.ok());
    if (!r.ok()) { return; }
    const auto &c = *r.value;
    expect(c.vocab_size == 32u);
    expect(c.hidden_size == 16u);
    expect(c.num_layers == 2u);
    expect(c.num_attention_heads == 4u);
    expect(c.num_kv_heads == 2u);
    expect(c.head_dim == 4u);// hidden/heads default
    expect(c.intermediate_size == 32u);
    expect(c.max_position_embeddings == 64u);
    // Qwen2.5-style configs select rotate-half, NOT the adjacent convention.
    expect(c.rope_pair_layout == RopePairLayout::rotate_half);
    expect(rope_pair_layout_name(c.rope_pair_layout) == "rotate_half");
    expect(rope_pair_layout_name(RopePairLayout::adjacent) == "adjacent");
}

// S6: integer-valued rope_theta / rms_norm_eps must parse (yyjson_get_num,
// not yyjson_get_real which returns 0 for an integer node).
void test_parse_integer_floats() {
    auto json = base_config();
    // Replace the real literals with integer literals.
    const auto replace = [&json](const char *from, const char *to) {
        auto pos = json.find(from);
        expect(pos != luisa::string::npos);
        if (pos != luisa::string::npos) { json.replace(pos, std::strlen(from), to); }
    };
    replace("\"rope_theta\": 1000000.0", "\"rope_theta\": 1000000");
    replace("\"rms_norm_eps\": 1e-6", "\"rms_norm_eps\": 1");
    auto r = parse_qwen2_5_text_config(json);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->rope_theta == 1000000.0f);
    expect(r.value->rms_norm_eps == 1.0f);
}

void test_parse_text_config_wrapper() {
    auto json = R"({"text_config": )" + base_config() + "}";
    auto r = parse_qwen2_5_text_config(json);
    expect(r.ok());
    if (r.ok()) { expect(r.value->rope_pair_layout == RopePairLayout::rotate_half); }
}

void test_parse_qwen3_rejected() {
    auto json = base_config();
    auto pos = json.find("\"qwen2_5_vl\"");
    json.replace(pos, std::strlen("\"qwen2_5_vl\""), "\"qwen3_vl\"");
    auto r = parse_qwen2_5_text_config(json);
    expect(!r.ok());
    expect(r.status.code == StatusCode::unsupported);
}

void test_parse_moe_gdn_rejected() {
    for (const char *marker : {"\"num_experts\": 8,", "\"qk_norm\": true,",
                               "\"linear_num_key_heads\": 4,"}) {
        auto json = base_config();
        json.insert(1, marker);// insert right after the opening brace
        auto r = parse_qwen2_5_text_config(json);
        expect(!r.ok());
        expect(r.status.code == StatusCode::unsupported);
    }
}

// Sliding window: disabled (use_sliding_window=false or null window) is the
// ordinary dense path and is accepted; enabled is unsupported.
void test_parse_sliding_window() {
    // use_sliding_window=false with a window set: accepted (disabled).
    {
        auto json = base_config();
        json.insert(1, "\"use_sliding_window\": false, \"sliding_window\": 4096,");
        auto r = parse_qwen2_5_text_config(json);
        expect(r.ok());
    }
    // sliding_window=null: accepted.
    {
        auto json = base_config();
        json.insert(1, "\"sliding_window\": null,");
        auto r = parse_qwen2_5_text_config(json);
        expect(r.ok());
    }
    // use_sliding_window=false alone: accepted.
    {
        auto json = base_config();
        json.insert(1, "\"use_sliding_window\": false,");
        auto r = parse_qwen2_5_text_config(json);
        expect(r.ok());
    }
    // Enabled window (use_sliding_window=true): unsupported.
    {
        auto json = base_config();
        json.insert(1, "\"use_sliding_window\": true, \"sliding_window\": 4096,");
        auto r = parse_qwen2_5_text_config(json);
        expect(!r.ok());
        expect(r.status.code == StatusCode::unsupported);
    }
    // A positive window with no explicit disable (HF defaults to enabled):
    // unsupported.
    {
        auto json = base_config();
        json.insert(1, "\"sliding_window\": 4096,");
        auto r = parse_qwen2_5_text_config(json);
        expect(!r.ok());
        expect(r.status.code == StatusCode::unsupported);
    }
}

void test_parse_eos_scalar_and_list() {
    {
        auto json = base_config();
        json.insert(1, "\"eos_token_id\": 31,");
        auto r = parse_qwen2_5_text_config(json);
        expect(r.ok());
        if (r.ok()) {
            expect(r.value->eos_token_ids.size() == 1u);
            expect(r.value->eos_token_ids[0] == 31u);
        }
    }
    {
        auto json = base_config();
        json.insert(1, "\"eos_token_id\": [31, 30],");
        auto r = parse_qwen2_5_text_config(json);
        expect(r.ok());
        if (r.ok()) { expect(r.value->eos_token_ids.size() == 2u); }
    }
}

// M1: the expected inventory declares the standalone lm_head as [hidden,vocab]
// packed kn (NOT [vocab,hidden]); tied makes it optional.
void test_expected_tensor_inventory() {
    DenseDecoderConfig c;
    c.vocab_size = 32;
    c.hidden_size = 16;
    c.num_layers = 2;
    c.num_attention_heads = 4;
    c.num_kv_heads = 2;
    c.head_dim = 4;
    c.intermediate_size = 32;
    c.max_position_embeddings = 64;
    DenseDecoderTensorNames names;
    auto inv = dense_decoder_expected_tensors(c, names, /*tie=*/false);
    const ExpectedTensor *head = nullptr;
    const ExpectedTensor *q0 = nullptr;
    for (const auto &e : inv) {
        if (e.name == names.lm_head()) { head = &e; }
        if (e.name == names.q_proj(0)) { q0 = &e; }
    }
    expect(head != nullptr);
    expect(q0 != nullptr);
    if (head == nullptr || q0 == nullptr) { return; }
    // lm_head [hidden, vocab] = [16, 32], layout kn, required when not tied.
    expect(head->shape.size() == 2u);
    expect(head->shape[0] == 16u);
    expect(head->shape[1] == 32u);
    expect(head->layout == TensorLayout::kn);
    expect(head->required);
    // q_proj [hidden, q_dim] = [16, 16], layout kn.
    expect(q0->shape[0] == 16u);
    expect(q0->shape[1] == 16u);
    expect(q0->layout == TensorLayout::kn);
    // Tied: lm_head optional.
    auto tied_inv = dense_decoder_expected_tensors(c, names, /*tie=*/true);
    for (const auto &e : tied_inv) {
        if (e.name == names.lm_head()) { expect(!e.required); }
    }
}

// M1: per-weight descriptor validation rejects wrong shape/layout/count.
void test_weight_desc_validation() {
    DenseDecoderConfig c;
    c.vocab_size = 32;
    c.hidden_size = 16;
    c.num_layers = 1;
    c.num_attention_heads = 4;
    c.num_kv_heads = 2;
    c.head_dim = 4;
    c.intermediate_size = 32;
    c.max_position_embeddings = 64;
    DenseDecoderTensorNames names;
    auto inv = dense_decoder_expected_tensors(c, names, false);
    const ExpectedTensor *q0 = nullptr;
    for (const auto &e : inv) {
        if (e.name == names.q_proj(0)) { q0 = &e; }
    }
    expect(q0 != nullptr);
    if (q0 == nullptr) { return; }
    // Correct descriptor: [16,16] kn, 256 elements.
    TensorDesc good;
    good.dtype = TensorDType::f32;
    good.layout = TensorLayoutTag::kn;
    good.shape = {16, 16};
    expect(good.normalize().ok());
    expect(validate_decoder_weight_desc(good, 256, *q0).ok());
    // Undersized element count.
    expect(!validate_decoder_weight_desc(good, 240, *q0).ok());
    // Wrong layout (row_major instead of kn).
    TensorDesc bad_layout = good;
    bad_layout.layout = TensorLayoutTag::row_major;
    expect(!validate_decoder_weight_desc(bad_layout, 256, *q0).ok());
    // Wrong shape (transposed [16,16] is same; use [16,15]).
    TensorDesc bad_shape = good;
    bad_shape.shape = {16, 15};
    expect(bad_shape.normalize().ok());
    expect(!validate_decoder_weight_desc(bad_shape, 240, *q0).ok());
}

}// namespace

static auto test_ocr_decoder_config_registration = [] {
    "decoder_config_parse_basic_rotate_half"_test = [] { test_parse_basic_and_rotate_half(); };
    "decoder_config_parse_integer_floats"_test = [] { test_parse_integer_floats(); };
    "decoder_config_parse_text_config_wrapper"_test = [] { test_parse_text_config_wrapper(); };
    "decoder_config_parse_qwen3_rejected"_test = [] { test_parse_qwen3_rejected(); };
    "decoder_config_parse_moe_gdn_rejected"_test = [] { test_parse_moe_gdn_rejected(); };
    "decoder_config_parse_sliding_window"_test = [] { test_parse_sliding_window(); };
    "decoder_config_parse_eos"_test = [] { test_parse_eos_scalar_and_list(); };
    "decoder_config_expected_inventory"_test = [] { test_expected_tensor_inventory(); };
    "decoder_config_weight_desc_validation"_test = [] { test_weight_desc_validation(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}