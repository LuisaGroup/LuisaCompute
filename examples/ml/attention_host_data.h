// Multi-Head Latent Attention (MLA) Example -- host input data.

#pragma once

#include <luisa/core/stl.h>

namespace mla {

// Host-side input tensors and MLA weights (deterministically initialized).
struct AttentionHostData {
    // MHA inputs (kept for the baseline path).
    luisa::vector<float> q;
    luisa::vector<float> k;
    luisa::vector<float> v;

    // MLA inputs and weights.
    luisa::vector<float> h;    // hidden states h_t
    luisa::vector<float> wq;   // query projection
    luisa::vector<float> wdkv; // low-rank joint KV compression
    luisa::vector<float> wuk;  // content key up-projection
    luisa::vector<float> wuv;  // value up-projection
    luisa::vector<float> wkr;  // decoupled RoPE key projection
};

// Allocate and fill all host tensors with the deterministic sin/cos patterns.
[[nodiscard]] AttentionHostData make_host_data();

}// namespace mla
