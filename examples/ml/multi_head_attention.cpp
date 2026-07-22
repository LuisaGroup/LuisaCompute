// Multi-Head Latent Attention (MLA) Example
//
// This example implements two attention paths selected by `use_mla`:
//
//   use_mla = false  -- classic Multi-Head Attention (MHA) reference.
//                       Three GPU passes: QK^T + scale, softmax, AV.
//
//   use_mla = true   -- Multi-Head Latent Attention reference.
//                       Demonstrates the DeepSeek-V2/V3 techniques:
//                         * Low-rank joint compression of K/V via W_DKV.
//                         * Decoupled RoPE: content queries + positional queries,
//                           separate W_KR decoupled keys.
//                         * Matrix absorption for the content score:
//                           q^T (W_UK c_t^KV) = (W_UK^T q)^T c_t^KV,
//                           so the full per-head content key tensor is never
//                           materialised during inference.
//                         * Reduced KV cache: cache only c_t^KV (latent_dim) and
//                           k_t^R (num_heads * rope_dim) per token.
//
// Both paths are verified against a matching CPU reference with tolerance 1e-4f.
//
// The example is split into focused modules:
//   attention_config.h          -- compile-time dimensions, sizes, index helpers, RoPE utils
//   attention_kernels.h/.cpp    -- DSL kernel factories (MHA / MLA / cooperative-vector)
//   attention_host_data.h/.cpp  -- deterministic host input initialization
//   attention_cpu_reference.*   -- multi-threaded CPU reference implementation
//   attention_runner.h/.cpp     -- device buffers, upload, compile/dispatch, download
//   multi_head_attention.cpp    -- entry point: argument parsing and verification

#include <cmath>
#include <cstdlib>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "attention_config.h"
#include "attention_cpu_reference.h"
#include "attention_host_data.h"
#include "attention_runner.h"

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [use_mla=1] [cooperative_vector=0]", argv[0]);
        return 1;
    }

    LUISA_INFO("Multi-Head Attention Example");
    LUISA_INFO("Backend: {}", argv[1]);

    // -- Attention path switch ---------------------------------------------
    bool use_mla = true;
    if (argc >= 3) {
        use_mla = (std::atoi(argv[2]) != 0);
    }

    bool cooperative_vector = false;
    if (argc >= 4) {
        cooperative_vector = (std::atoi(argv[3]) != 0);
    }

    LUISA_INFO("Path: {}", use_mla ? "MLA" : "MHA");
    LUISA_INFO("Cooperative vector: {}", cooperative_vector ? "enabled" : "disabled");

    // -- Host data initialization ------------------------------------------
    auto host = mla::make_host_data();

    // -- Context & Device --------------------------------------------------
    Context ctx{argv[0]};
    Device device = ctx.create_device(argv[1]);
    Stream stream = device.create_stream();

    // -- Device buffers & upload -------------------------------------------
    auto buffers = mla::create_device_buffers(device);
    mla::upload_host_data(stream, buffers, host);

    // -- Compile and dispatch the selected attention path ------------------
    mla::run_attention(device, stream, buffers, use_mla, cooperative_vector);

    // -- Download results --------------------------------------------------
    luisa::vector<float> O_gpu;
    mla::download_output(stream, buffers, O_gpu);

    // -- CPU Reference -----------------------------------------------------
    LUISA_INFO("Running CPU reference ...");
    Clock cpu_clock;
    auto O_cpu = mla::run_cpu_reference(host, use_mla);
    double cpu_ms = cpu_clock.toc();
    LUISA_INFO("  CPU reference completed in {:.2f} ms", cpu_ms);

    // -- Performance / memory summary --------------------------------------
    LUISA_INFO("-- Summary ----------------------------------------------");
    LUISA_INFO("  Mode: {}", use_mla ? "MLA" : "MHA");
    LUISA_INFO("  Matrix config: batch={}, heads={}, seq_len={}, head_dim={}",
               mla::batch, mla::num_heads, mla::seq_len, mla::head_dim);
    if (use_mla) {
        LUISA_INFO("  MLA config: hidden_dim={}, latent_dim={}, rope_dim={}, content_dim={}",
                   mla::hidden_dim, mla::latent_dim, mla::rope_dim, mla::content_dim);
        size_t mha_kv_bytes = (2ull * mla::num_heads * mla::head_dim) * sizeof(float);
        size_t mla_kv_bytes = (static_cast<size_t>(mla::latent_dim) + mla::num_heads * mla::rope_dim) * sizeof(float);
        float ratio = static_cast<float>(mla_kv_bytes) / static_cast<float>(mha_kv_bytes);
        LUISA_INFO("  KV-cache per token: MHA={} B, MLA={} B, ratio={:.2f}%",
                   mha_kv_bytes, mla_kv_bytes, ratio * 100.0f);
    }

    // -- Verification ------------------------------------------------------
    constexpr float tolerance = 1e-4f;
    float max_diff = 0.0f;
    uint max_idx = 0u;
    bool all_finite = true;
    for (uint i = 0u; i < mla::qkv_size; ++i) {
        if (!std::isfinite(O_gpu[i]) || !std::isfinite(O_cpu[i])) {
            all_finite = false;
            max_idx = i;
            break;
        }
        float diff = std::abs(O_gpu[i] - O_cpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    bool passed = all_finite && max_diff <= tolerance;

    LUISA_INFO("Verification: {}", passed ? "PASSED" : "FAILED");
    if (all_finite) {
        LUISA_INFO("  Max absolute error: {} at index {} (GPU={}, CPU={})",
                   max_diff, max_idx, O_gpu[max_idx], O_cpu[max_idx]);
    } else {
        LUISA_INFO("  Non-finite output at index {} (GPU={}, CPU={})",
                   max_idx, O_gpu[max_idx], O_cpu[max_idx]);
    }

    if (!passed) {
        uint printed = 0u;
        for (uint i = 0u; i < mla::qkv_size && printed < 5u; ++i) {
            float diff = std::abs(O_gpu[i] - O_cpu[i]);
            if (!std::isfinite(O_gpu[i]) || !std::isfinite(O_cpu[i]) || diff > tolerance) {
                LUISA_INFO("  Mismatch[{}]: GPU={}, CPU={}, diff={}", i, O_gpu[i], O_cpu[i], diff);
                ++printed;
            }
        }
    }

    return passed ? 0 : 1;
}
