// Multi-Head Latent Attention (MLA) Example -- host input data.

#include "attention_config.h"
#include "attention_host_data.h"

#include <cmath>

namespace mla {

AttentionHostData make_host_data() {
    AttentionHostData data;

    // MHA inputs (kept for the baseline path).
    data.q.resize(qkv_size);
    data.k.resize(qkv_size);
    data.v.resize(qkv_size);
    for (uint32_t i = 0u; i < qkv_size; ++i) {
        auto fi = static_cast<float>(i);
        data.q[i] = std::sin(fi * 0.13f + 0.5f) * 0.5f + 0.5f;
        data.k[i] = std::cos(fi * 0.17f + 1.0f) * 0.5f + 0.5f;
        data.v[i] = std::sin(fi * 0.11f + 2.0f) * 0.3f + 0.3f;
    }

    // MLA inputs and weights.
    data.h.resize(hidden_size);
    data.wq.resize(wq_size);
    data.wdkv.resize(wdkv_size);
    data.wuk.resize(wuk_size);
    data.wuv.resize(wuv_size);
    data.wkr.resize(wkr_size);

    for (uint32_t i = 0u; i < hidden_size; ++i) {
        data.h[i] = std::sin(static_cast<float>(i) * 0.05f + 0.3f) * 0.4f + 0.4f;
    }
    for (uint32_t i = 0u; i < data.wq.size(); ++i) {
        data.wq[i] = std::sin(static_cast<float>(i) * 0.07f) * 0.01f;
    }
    for (uint32_t i = 0u; i < data.wdkv.size(); ++i) {
        data.wdkv[i] = std::sin(static_cast<float>(i) * 0.09f + 0.1f) * 0.02f;
    }
    for (uint32_t i = 0u; i < data.wuk.size(); ++i) {
        data.wuk[i] = std::sin(static_cast<float>(i) * 0.11f + 0.2f) * 0.02f;
    }
    for (uint32_t i = 0u; i < data.wuv.size(); ++i) {
        data.wuv[i] = std::sin(static_cast<float>(i) * 0.13f + 0.3f) * 0.02f;
    }
    for (uint32_t i = 0u; i < data.wkr.size(); ++i) {
        data.wkr[i] = std::sin(static_cast<float>(i) * 0.15f + 0.4f) * 0.02f;
    }

    return data;
}

}// namespace mla
