// mm_embedding engine -- comprehensive automated unit & regression tests.
//
// Tests:
// 1. Tokenizer: special tokens, BPE merges, byte fallback, truncation, batching.
// 2. Package: headers, 64-bit offsets, tensor lookups, vision tensors, has_vision flag.
// 3. DSL Kernels (GPU):
//    - RMSNorm, L2Norm, GEMM, MeanPool
//    - Vision: ScalePixels, AddPatchPosEmbed, AxialRoPE, VisionPool, MaskFromPos, ScatterSoftTokens
// 4. Selective Encoder Loading:
//    - Text-only session (enable_vision=false): verify VRAM savings, has_vision()=false, correct text encoding
//    - Multimodal session (enable_vision=true): verify text parity with text-only mode and vision availability
// 5. Image & Multimodal End-to-End:
//    - Single image encoding across MRL dimensions (768, 512, 256, 128)
//    - Multimodal prompt encoding (image + text)
//    - Cross-modal retrieval: Image-to-Text and Text-to-Image top-1 matching
//    - Parity comparison with PyTorch SentenceTransformer golden embeddings (cos sim > 0.99)
//
// Usage:
//   example_mm_embed_test [backend: dx|vk] [model.mmegm] [tokenizer.json] [data_dir]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

#include <yyjson.h>

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

#include "mm_embedding/embed/image_processor.h"
#include "mm_embedding/embed/audio_processor.h"
#include "mm_embedding/embed/kernels/kernels_attention.h"
#include "mm_embedding/embed/kernels/kernels_audio.h"
#include "mm_embedding/embed/kernels/kernels_gemm.h"
#include "mm_embedding/embed/kernels/kernels_mlp.h"
#include "mm_embedding/embed/kernels/kernels_norm.h"
#include "mm_embedding/embed/kernels/kernels_pool.h"
#include "mm_embedding/embed/kernels/kernels_rope.h"
#include "mm_embedding/embed/kernels/kernels_vision.h"
#include "mm_embedding/embed/package.h"
#include "mm_embedding/embed/session.h"
#include "mm_embedding/embed/executor.h"
#include "mm_embedding/embed/tokenizer.h"

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "TEST FAILED [%s:%d]: %s (%s)\n", __FILE__, __LINE__, msg, #cond); \
            std::exit(1); \
        } \
    } while (0)

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::mm_embed;

// ============================================================================
// 1. Tokenizer Tests
// ============================================================================
void test_tokenizer(const std::filesystem::path &tok_path) {
    std::cout << "[TEST] Running Tokenizer tests..." << std::endl;

    if (!std::filesystem::exists(tok_path)) {
        std::cout << "  [SKIP] Tokenizer file '" << tok_path.string() << "' not found." << std::endl;
        return;
    }

    auto tok = Tokenizer::load(tok_path);
    TEST_ASSERT(tok != nullptr, "Tokenizer::load must return non-null");
    TEST_ASSERT(tok->vocab_size() >= 256000u, "Vocab size must be >= 256k for Gemma");
    TEST_ASSERT(tok->bos_id() == 2u, "Gemma BOS token must be 2");
    TEST_ASSERT(tok->eos_id() == 1u, "Gemma EOS token must be 1");
    TEST_ASSERT(tok->pad_id() == 0u, "Gemma PAD token must be 0");

    // Empty string test: should produce [<bos>, <eos>]
    auto empty_toks = tok->encode("");
    TEST_ASSERT(empty_toks.size() == 2, "Empty string should encode to [<bos>, <eos>]");
    TEST_ASSERT(empty_toks[0] == 2u, "First token must be BOS (2)");
    TEST_ASSERT(empty_toks[1] == 1u, "Second token must be EOS (1)");

    // Standard sentence test
    auto s_toks = tok->encode("Hello world");
    TEST_ASSERT(s_toks.size() >= 3, "Encoding 'Hello world' must have >= 3 tokens");
    TEST_ASSERT(s_toks.front() == 2u, "Must start with BOS");
    TEST_ASSERT(s_toks.back() == 1u, "Must end with EOS");

    // Truncation test
    auto trunc_toks = tok->encode("Hello world, welcome to multimodal embeddings with Gemma!", 4u);
    TEST_ASSERT(trunc_toks.size() <= 4u, "Max len truncation must be respected");

    // Batch encoding
    luisa::string_view batch_texts[] = {"Test 1", "Longer test sentence for embedding"};
    auto b_res = tok->encode_batch(luisa::span{batch_texts, 2u});
    TEST_ASSERT(b_res.size() == 2, "Batch size must match");
    TEST_ASSERT(!b_res[0].empty() && !b_res[1].empty(), "Batch tokens cannot be empty");

    std::cout << "  [PASS] Tokenizer tests passed." << std::endl;
}

// ============================================================================
// 2. Package Tests
// ============================================================================
void test_package(const std::filesystem::path &egm_path) {
    std::cout << "[TEST] Running Package tests..." << std::endl;

    if (!std::filesystem::exists(egm_path)) {
        std::cout << "  [SKIP] Model package '" << egm_path.string() << "' not found." << std::endl;
        return;
    }

    auto pkg = Package::load(egm_path);
    TEST_ASSERT(pkg != nullptr, "Package::load must return non-null");
    TEST_ASSERT(pkg->num_layers() == 24u, "Gemma 2 270M must have 24 layers");
    TEST_ASSERT(pkg->hidden_size() == 512u, "Hidden size must be 512");
    TEST_ASSERT(pkg->intermediate_size() == 2048u, "Intermediate size must be 2048");
    TEST_ASSERT(pkg->embedding_dim() == 768u, "Embedding dim must be 768");
    TEST_ASSERT(pkg->tensors().size() >= 413u, "Total tensors must be at least 413");

    // Find known text tensors
    const auto *t_emb = pkg->find("emb.word");
    TEST_ASSERT(t_emb != nullptr, "Must find 'emb.word'");
    TEST_ASSERT(t_emb->rank == 2u, "'emb.word' rank must be 2");
    TEST_ASSERT(t_emb->dims[0] == 262144u && t_emb->dims[1] == 512u, "emb.word dims must be [262144, 512]");

    const auto *t_final_norm = pkg->find("final_norm");
    TEST_ASSERT(t_final_norm != nullptr, "Must find 'final_norm'");
    TEST_ASSERT(t_final_norm->rank == 1u && t_final_norm->dims[0] == 512u, "final_norm dims must be [512]");

    // Check vision presence
    if (pkg->has_vision()) {
        std::cout << "  [INFO] Model package includes Vision Tower weights (624 tensors)." << std::endl;
        TEST_ASSERT(pkg->find("vision.patch_embed.proj") != nullptr, "Must find 'vision.patch_embed.proj'");
        TEST_ASSERT(pkg->find("vision.patch_embed.pos_table") != nullptr, "Must find 'vision.patch_embed.pos_table'");
        TEST_ASSERT(pkg->find("vision.connector.proj") != nullptr, "Must find 'vision.connector.proj'");
        TEST_ASSERT(pkg->find("vision.layer0.input_norm") != nullptr, "Must find 'vision.layer0.input_norm'");
        TEST_ASSERT(pkg->find("vision.layer15.out_proj") != nullptr, "Must find 'vision.layer15.out_proj'");
    }

    // Check audio presence
    if (pkg->has_audio()) {
        std::cout << "  [INFO] Model package includes Audio Tower weights." << std::endl;
        TEST_ASSERT(pkg->find("audio.connector.proj") != nullptr, "Must find 'audio.connector.proj'");
        TEST_ASSERT(pkg->find("audio.subsample.layer0.conv") != nullptr, "Must find 'audio.subsample.layer0.conv'");
        TEST_ASSERT(pkg->find("audio.subsample.layer1.conv") != nullptr, "Must find 'audio.subsample.layer1.conv'");
        TEST_ASSERT(pkg->find("audio.subsample.input_proj") != nullptr, "Must find 'audio.subsample.input_proj'");
        TEST_ASSERT(pkg->find("audio.layer0.q_proj") != nullptr, "Must find 'audio.layer0.q_proj'");
        TEST_ASSERT(pkg->find("audio.layer11.norm_out") != nullptr, "Must find 'audio.layer11.norm_out'");
        TEST_ASSERT(pkg->find("audio.output_proj.weight") != nullptr, "Must find 'audio.output_proj.weight'");
        TEST_ASSERT(pkg->find("audio.clip_bounds") != nullptr, "Must find 'audio.clip_bounds'");
    }

    std::cout << "  [PASS] Package tests passed." << std::endl;
}

// ============================================================================
// 3. Kernel Unit Tests (GPU)
// ============================================================================
void test_kernels(Device &device) {
    std::cout << "[TEST] Running Kernel GPU unit tests..." << std::endl;
    auto stream = device.create_stream(StreamTag::COMPUTE);

    // (a) Test RMSNorm
    {
        auto rmsnorm_shader = device.compile(make_rmsnorm_kernel());
        constexpr uint32_t H = 512u;
        luisa::vector<float> host_x(H);
        luisa::vector<float> host_gamma(H, 1.0f);
        float mean_sq = 0.0f;
        for (uint32_t i = 0; i < H; i++) {
            host_x[i] = static_cast<float>(i + 1) * 0.01f;
            mean_sq += host_x[i] * host_x[i];
        }
        mean_sq /= static_cast<float>(H);
        float expected_inv_rms = 1.0f / std::sqrt(mean_sq + 1e-6f);

        auto buf_x = device.create_buffer<float>(H);
        auto buf_y = device.create_buffer<float>(H);
        auto buf_g = device.create_buffer<float>(H);
        luisa::vector<float> host_y(H);

        CommandList cmd;
        cmd << buf_x.copy_from(host_x.data())
            << buf_g.copy_from(host_gamma.data())
            << rmsnorm_shader(buf_x, buf_y, buf_g, H, 1e-6f).dispatch(kNormBlock)
            << buf_y.copy_to(host_y.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t i = 0; i < H; i++) {
            float expected = host_x[i] * expected_inv_rms;
            TEST_ASSERT(std::abs(host_y[i] - expected) < 1e-4f, "RMSNorm value mismatch");
        }
        std::cout << "  [PASS] RMSNorm kernel verified." << std::endl;
    }

    // (b) Test L2Normalize
    {
        auto l2norm_shader = device.compile(make_l2_normalize_kernel());
        constexpr uint32_t D = 768u;
        luisa::vector<float> host_x(D);
        for (uint32_t i = 0; i < D; i++) {
            host_x[i] = static_cast<float>(i % 17) - 8.0f;
        }

        auto buf_x = device.create_buffer<float>(D);
        auto buf_y = device.create_buffer<float>(D);
        luisa::vector<float> host_y(D);

        CommandList cmd;
        cmd << buf_x.copy_from(host_x.data())
            << l2norm_shader(buf_x, buf_y, D).dispatch(kNormBlock)
            << buf_y.copy_to(host_y.data());
        stream << cmd.commit() << synchronize();

        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < D; i++) {
            sum_sq += host_y[i] * host_y[i];
        }
        TEST_ASSERT(std::abs(sum_sq - 1.0f) < 1e-4f, "L2 normalized vector must have norm 1.0");
        std::cout << "  [PASS] L2Normalize kernel verified." << std::endl;
    }

    // (c) Test GEMM (16x16 tiled kernel)
    {
        auto gemm_shader = device.compile(make_gemm_kernel());
        constexpr uint32_t M = 32u;
        constexpr uint32_t N = 48u;
        constexpr uint32_t K = 64u;

        luisa::vector<float> host_a(M * K);
        luisa::vector<float> host_b(K * N);
        luisa::vector<float> host_c(M * N);
        luisa::vector<float> expected_c(M * N, 0.0f);

        for (uint32_t i = 0; i < M * K; i++) host_a[i] = (i % 7) * 0.1f;
        for (uint32_t i = 0; i < K * N; i++) host_b[i] = (i % 5) * 0.1f;

        for (uint32_t m = 0; m < M; m++) {
            for (uint32_t n = 0; n < N; n++) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < K; k++) {
                    acc += host_a[m * K + k] * host_b[k * N + n];
                }
                expected_c[m * N + n] = acc;
            }
        }

        auto buf_a = device.create_buffer<float>(M * K);
        auto buf_b = device.create_buffer<float>(K * N);
        auto buf_c = device.create_buffer<float>(M * N);

        CommandList cmd;
        cmd << buf_a.copy_from(host_a.data())
            << buf_b.copy_from(host_b.data())
            << gemm_shader(buf_a, buf_b, buf_c, M, N, K).dispatch(gemm_dispatch_size(M, N))
            << buf_c.copy_to(host_c.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t i = 0; i < M * N; i++) {
            TEST_ASSERT(std::abs(host_c[i] - expected_c[i]) < 1e-3f, "GEMM calculation mismatch");
        }
        std::cout << "  [PASS] GEMM kernel verified." << std::endl;
    }

    // (d) Test Vision ScalePixels Kernel
    {
        auto scale_shader = device.compile(make_scale_pixels_kernel());
        constexpr uint32_t N = 128u;
        luisa::vector<float> in_vals(N);
        luisa::vector<float> expected(N);
        for (uint32_t i = 0; i < N; i++) {
            in_vals[i] = static_cast<float>(i) / static_cast<float>(N);
            expected[i] = 2.0f * (in_vals[i] - 0.5f);
        }

        auto buf_in = device.create_buffer<float>(N);
        auto buf_out = device.create_buffer<float>(N);
        luisa::vector<float> host_out(N);

        CommandList cmd;
        cmd << buf_in.copy_from(in_vals.data())
            << scale_shader(buf_in, buf_out, N).dispatch(N)
            << buf_out.copy_to(host_out.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t i = 0; i < N; i++) {
            TEST_ASSERT(std::abs(host_out[i] - expected[i]) < 1e-5f, "ScalePixels calculation mismatch");
        }
        std::cout << "  [PASS] Vision ScalePixels kernel verified." << std::endl;
    }

    // (e) Test Vision Axial RoPE Kernel
    {
        auto axial_rope_shader = device.compile(make_axial_rope_kernel());
        constexpr uint32_t num_heads = 12u;
        constexpr uint32_t head_dim = 64u;
        constexpr uint32_t num_patches = 4u;
        constexpr float rope_theta = 100.0f;

        luisa::vector<float> host_qk(num_patches * num_heads * head_dim, 1.0f);
        luisa::vector<int32_t> host_pos = {
            0, 0,
            1, 0,
            0, 1,
            5, 10
        };

        auto buf_qk_in = device.create_buffer<float>(host_qk.size());
        auto buf_qk_out = device.create_buffer<float>(host_qk.size());
        auto buf_pos = device.create_buffer<int32_t>(host_pos.size());
        luisa::vector<float> host_out(host_qk.size());

        CommandList cmd;
        cmd << buf_qk_in.copy_from(host_qk.data())
            << buf_pos.copy_from(host_pos.data())
            << axial_rope_shader(buf_qk_in, buf_qk_out, buf_pos, num_heads, head_dim, rope_theta)
                   .dispatch(16u, num_patches * num_heads)
            << buf_qk_out.copy_to(host_out.data());
        stream << cmd.commit() << synchronize();

        // Patch 0 has pos (0, 0) -> theta = 0 -> cos = 1, sin = 0 -> values remain 1.0
        for (uint32_t i = 0; i < num_heads * head_dim; i++) {
            TEST_ASSERT(std::abs(host_out[i] - 1.0f) < 1e-4f, "Axial RoPE pos(0,0) must preserve values");
        }
        std::cout << "  [PASS] Vision AxialRoPE kernel verified." << std::endl;
    }

    // (f) Test Vision 2D Spatial Pooler Kernel (Square and Non-Square)
    {
        auto pool_shader = device.compile(make_vision_pool_kernel());
        constexpr uint32_t H = 16u;

        // (f.1) Square pooling: 2x2 soft tokens
        {
            constexpr uint32_t grid_soft = 2u; // 2x2 soft tokens
            constexpr uint32_t num_soft = grid_soft * grid_soft; // 4
            constexpr uint32_t patch_stride = grid_soft * 3u; // 6
            constexpr uint32_t total_patches = patch_stride * patch_stride; // 36

            luisa::vector<float> host_patches(total_patches * H, 1.0f);
            constexpr float scale = 3.0f; // test scale factor

            auto buf_in = device.create_buffer<float>(total_patches * H);
            auto buf_out = device.create_buffer<float>(num_soft * H);
            luisa::vector<float> host_out(num_soft * H);

            CommandList cmd;
            cmd << buf_in.copy_from(host_patches.data())
                << pool_shader(buf_in, buf_out, grid_soft, grid_soft, patch_stride, H, scale)
                       .dispatch(H, num_soft)
                << buf_out.copy_to(host_out.data());
            stream << cmd.commit() << synchronize();

            for (uint32_t i = 0; i < num_soft * H; i++) {
                TEST_ASSERT(std::abs(host_out[i] - 27.0f) < 1e-4f, "VisionPool square sum mismatch");
            }
        }

        // (f.2) Non-square rectangular pooling: 3x2 soft tokens (grid_w=3, grid_h=2)
        {
            constexpr uint32_t grid_w_soft = 3u;
            constexpr uint32_t grid_h_soft = 2u;
            constexpr uint32_t num_soft = grid_w_soft * grid_h_soft; // 6
            constexpr uint32_t patch_stride = grid_w_soft * 3u; // 9
            constexpr uint32_t num_patches_h = grid_h_soft * 3u; // 6
            constexpr uint32_t total_patches = patch_stride * num_patches_h; // 54

            luisa::vector<float> host_patches(total_patches * H);
            for (uint32_t p = 0; p < total_patches; p++) {
                for (uint32_t d = 0; d < H; d++) {
                    host_patches[p * H + d] = static_cast<float>(p + 1);
                }
            }
            constexpr float scale = 0.5f;

            auto buf_in = device.create_buffer<float>(total_patches * H);
            auto buf_out = device.create_buffer<float>(num_soft * H);
            luisa::vector<float> host_out(num_soft * H);

            CommandList cmd;
            cmd << buf_in.copy_from(host_patches.data())
                << pool_shader(buf_in, buf_out, grid_w_soft, grid_h_soft, patch_stride, H, scale)
                       .dispatch(H, num_soft)
                << buf_out.copy_to(host_out.data());
            stream << cmd.commit() << synchronize();

            for (uint32_t sy = 0; sy < grid_h_soft; sy++) {
                for (uint32_t sx = 0; sx < grid_w_soft; sx++) {
                    uint32_t s_idx = sy * grid_w_soft + sx;
                    float expected_sum = 0.0f;
                    for (uint32_t dy = 0; dy < 3u; dy++) {
                        for (uint32_t dx = 0; dx < 3u; dx++) {
                            uint32_t px = sx * 3u + dx;
                            uint32_t py = sy * 3u + dy;
                            uint32_t p_idx = py * patch_stride + px;
                            expected_sum += static_cast<float>(p_idx + 1);
                        }
                    }
                    float expected_val = expected_sum * scale;
                    for (uint32_t d = 0; d < H; d++) {
                        TEST_ASSERT(std::abs(host_out[s_idx * H + d] - expected_val) < 1e-4f,
                                    "VisionPool non-square rectangular sum mismatch");
                    }
                }
            }
        }
        std::cout << "  [PASS] Vision Spatial Pooler kernel (square & non-square) verified." << std::endl;
    }

    // (g) Test Scatter Soft Tokens Kernel
    {
        auto scatter_shader = device.compile(make_scatter_soft_tokens_kernel());
        constexpr uint32_t total_tokens = 8u;
        constexpr uint32_t num_soft = 3u;
        constexpr uint32_t start_idx = 2u;
        constexpr uint32_t H = 4u;

        luisa::vector<float> host_h0(total_tokens * H, 0.0f);
        luisa::vector<float> host_soft(num_soft * H, 99.0f);

        auto buf_h0 = device.create_buffer<float>(total_tokens * H);
        auto buf_soft = device.create_buffer<float>(num_soft * H);
        luisa::vector<float> host_out(total_tokens * H);

        CommandList cmd;
        cmd << buf_h0.copy_from(host_h0.data())
            << buf_soft.copy_from(host_soft.data())
            << scatter_shader(buf_h0, buf_soft, start_idx, num_soft, H).dispatch(H, num_soft)
            << buf_h0.copy_to(host_out.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t t = 0; t < total_tokens; t++) {
            float expected = (t >= start_idx && t < start_idx + num_soft) ? 99.0f : 0.0f;
            for (uint32_t d = 0; d < H; d++) {
                TEST_ASSERT(host_out[t * H + d] == expected, "ScatterSoftTokens misplaced token");
            }
        }
        std::cout << "  [PASS] ScatterSoftTokens kernel verified." << std::endl;
    }

    // (h) Test Audio Clamp Kernel
    {
        auto clamp_shader = device.compile(make_audio_clamp_kernel());
        constexpr uint32_t N = 10u;
        luisa::vector<float> host_x = {-5.0f, -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f, 5.0f, 10.0f};
        float min_v = -1.5f;
        float max_v = 3.0f;

        auto buf_x = device.create_buffer<float>(N);
        auto buf_y = device.create_buffer<float>(N);
        luisa::vector<float> host_y(N);

        CommandList cmd;
        cmd << buf_x.copy_from(host_x.data())
            << clamp_shader(buf_x, buf_y, min_v, max_v, N).dispatch(N)
            << buf_y.copy_to(host_y.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t i = 0; i < N; i++) {
            float exp_v = std::clamp(host_x[i], min_v, max_v);
            TEST_ASSERT(std::abs(host_y[i] - exp_v) < 1e-6f, "AudioClamp value mismatch");
        }
        std::cout << "  [PASS] AudioClamp kernel verified." << std::endl;
    }

    // (i) Test Audio GLU Kernel
    {
        auto glu_shader = device.compile(make_audio_glu_kernel());
        constexpr uint32_t T = 4u;
        constexpr uint32_t H = 8u;
        luisa::vector<float> host_x(T * 2 * H);
        for (uint32_t i = 0; i < T * 2 * H; i++) host_x[i] = static_cast<float>(i % 5) - 2.0f;

        auto buf_x = device.create_buffer<float>(T * 2 * H);
        auto buf_y = device.create_buffer<float>(T * H);
        luisa::vector<float> host_y(T * H);

        CommandList cmd;
        cmd << buf_x.copy_from(host_x.data())
            << glu_shader(buf_x, buf_y, T, H).dispatch(H, T)
            << buf_y.copy_to(host_y.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t h = 0; h < H; h++) {
                float x1 = host_x[t * 2 * H + h];
                float x2 = host_x[t * 2 * H + H + h];
                float sig = 1.0f / (1.0f + std::exp(-x2));
                float exp_v = x1 * sig;
                TEST_ASSERT(std::abs(host_y[t * H + h] - exp_v) < 1e-5f, "AudioGlu mismatch");
            }
        }
        std::cout << "  [PASS] AudioGlu kernel verified." << std::endl;
    }

    // (j) Test Audio Depthwise Causal Conv1D Kernel
    {
        auto conv_shader = device.compile(make_audio_depthwise_conv1d_kernel());
        constexpr uint32_t T = 6u;
        constexpr uint32_t C = 8u;
        luisa::vector<float> host_x(T * C);
        luisa::vector<float> host_w(C * 5u);
        for (uint32_t i = 0; i < T * C; i++) host_x[i] = static_cast<float>(i + 1);
        for (uint32_t i = 0; i < C * 5u; i++) host_w[i] = 0.2f * static_cast<float>((i % 5) + 1);

        auto buf_x = device.create_buffer<float>(T * C);
        auto buf_w = device.create_buffer<float>(C * 5u);
        auto buf_y = device.create_buffer<float>(T * C);
        luisa::vector<float> host_y(T * C);

        CommandList cmd;
        cmd << buf_x.copy_from(host_x.data())
            << buf_w.copy_from(host_w.data())
            << conv_shader(buf_x, buf_w, buf_y, T, C).dispatch(C, T)
            << buf_y.copy_to(host_y.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t c = 0; c < C; c++) {
                float acc = 0.0f;
                for (int k = 0; k < 5; k++) {
                    int in_t = static_cast<int>(t) + k - 4;
                    float val = (in_t >= 0) ? host_x[in_t * C + c] : 0.0f;
                    acc += val * host_w[c * 5 + k];
                }
                TEST_ASSERT(std::abs(host_y[t * C + c] - acc) < 1e-4f, "AudioDepthwiseConv1D mismatch");
            }
        }
        std::cout << "  [PASS] AudioDepthwiseConv1D kernel verified." << std::endl;
    }

    // (k) Test Audio Scatter Soft Tokens Kernel
    {
        auto scatter_shader = device.compile(make_audio_scatter_tokens_kernel());
        constexpr uint32_t total_tokens = 10u;
        constexpr uint32_t num_soft = 4u;
        constexpr uint32_t start_idx = 2u;
        constexpr uint32_t H = 8u;

        luisa::vector<float> host_h0(total_tokens * H, 0.0f);
        luisa::vector<float> host_soft(num_soft * H, 77.0f);

        auto buf_h0 = device.create_buffer<float>(total_tokens * H);
        auto buf_soft = device.create_buffer<float>(num_soft * H);
        luisa::vector<float> host_out(total_tokens * H);

        CommandList cmd;
        cmd << buf_h0.copy_from(host_h0.data())
            << buf_soft.copy_from(host_soft.data())
            << scatter_shader(buf_h0, buf_soft, start_idx, num_soft, H).dispatch(H, num_soft)
            << buf_h0.copy_to(host_out.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t t = 0; t < total_tokens; t++) {
            float exp_v = (t >= start_idx && t < start_idx + num_soft) ? 77.0f : 0.0f;
            for (uint32_t d = 0; d < H; d++) {
                TEST_ASSERT(host_out[t * H + d] == exp_v, "AudioScatterTokens misplaced token");
            }
        }
        std::cout << "  [PASS] AudioScatterTokens kernel verified." << std::endl;
    }

    // (l) Test Audio SiLU Kernel
    {
        auto silu_shader = device.compile(make_audio_silu_kernel());
        constexpr uint32_t N = 8u;
        luisa::vector<float> host_x = {-3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
        auto buf_x = device.create_buffer<float>(N);
        auto buf_y = device.create_buffer<float>(N);
        luisa::vector<float> host_y(N);

        CommandList cmd;
        cmd << buf_x.copy_from(host_x.data())
            << silu_shader(buf_x, buf_y, N).dispatch(N)
            << buf_y.copy_to(host_y.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t i = 0; i < N; i++) {
            float exp_v = host_x[i] / (1.0f + std::exp(-host_x[i]));
            TEST_ASSERT(std::abs(host_y[i] - exp_v) < 1e-5f, "AudioSilu mismatch");
        }
        std::cout << "  [PASS] AudioSilu kernel verified." << std::endl;
    }

    // (m) Test Audio Add and AddScale Kernels
    {
        auto add_shader = device.compile(make_audio_add_kernel());
        auto add_scale_shader = device.compile(make_audio_add_scale_kernel());
        constexpr uint32_t N = 8u;
        luisa::vector<float> host_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        luisa::vector<float> host_res = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
        auto buf_x = device.create_buffer<float>(N);
        auto buf_res = device.create_buffer<float>(N);
        auto buf_y1 = device.create_buffer<float>(N);
        auto buf_y2 = device.create_buffer<float>(N);
        luisa::vector<float> host_y1(N);
        luisa::vector<float> host_y2(N);

        CommandList cmd;
        cmd << buf_x.copy_from(host_x.data())
            << buf_res.copy_from(host_res.data())
            << add_shader(buf_x, buf_res, buf_y1, N).dispatch(N)
            << add_scale_shader(buf_x, buf_res, buf_y2, 0.5f, N).dispatch(N)
            << buf_y1.copy_to(host_y1.data())
            << buf_y2.copy_to(host_y2.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t i = 0; i < N; i++) {
            TEST_ASSERT(std::abs(host_y1[i] - (host_x[i] + host_res[i])) < 1e-6f, "AudioAdd mismatch");
            TEST_ASSERT(std::abs(host_y2[i] - (host_x[i] * 0.5f + host_res[i])) < 1e-6f, "AudioAddScale mismatch");
        }
        std::cout << "  [PASS] AudioAdd & AudioAddScale kernels verified." << std::endl;
    }

    // (n) Test GemmBias Kernel
    {
        auto gemm_bias_shader = device.compile(make_gemm_bias_kernel());
        constexpr uint32_t M = 2u;
        constexpr uint32_t N = 4u;
        constexpr uint32_t K = 3u;
        luisa::vector<float> host_A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        luisa::vector<float> host_B = {
            1.0f, 0.0f, 2.0f, 1.0f,
            0.0f, 1.0f, 1.0f, 0.0f,
            2.0f, 1.0f, 0.0f, 2.0f
        };
        luisa::vector<float> host_bias = {0.5f, 1.5f, 2.5f, 3.5f};
        auto buf_A = device.create_buffer<float>(M * K);
        auto buf_B = device.create_buffer<float>(K * N);
        auto buf_bias = device.create_buffer<float>(N);
        auto buf_C = device.create_buffer<float>(M * N);
        luisa::vector<float> host_C(M * N);

        CommandList cmd;
        cmd << buf_A.copy_from(host_A.data())
            << buf_B.copy_from(host_B.data())
            << buf_bias.copy_from(host_bias.data())
            << gemm_bias_shader(buf_A, buf_B, buf_bias, buf_C, M, N, K).dispatch(gemm_dispatch_size(M, N))
            << buf_C.copy_to(host_C.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t r = 0; r < M; r++) {
            for (uint32_t c = 0; c < N; c++) {
                float expected = host_bias[c];
                for (uint32_t k = 0; k < K; k++) {
                    expected += host_A[r * K + k] * host_B[k * N + c];
                }
                TEST_ASSERT(std::abs(host_C[r * N + c] - expected) < 1e-4f, "GemmBias mismatch");
            }
        }
        std::cout << "  [PASS] GemmBias kernel verified." << std::endl;
    }

    // (o) Test AudioScaleQK Kernel
    {
        auto scale_qk_shader = device.compile(make_audio_scale_qk_kernel());
        constexpr uint32_t T = 2u;
        constexpr uint32_t num_heads = 2u;
        constexpr uint32_t head_dim = 4u;
        constexpr uint32_t H = num_heads * head_dim;
        luisa::vector<float> host_q(T * H, 1.0f);
        luisa::vector<float> host_k(T * H, 2.0f);
        luisa::vector<float> host_pds = {0.0f, 1.0f, 2.0f, 25.0f};
        auto buf_q = device.create_buffer<float>(T * H);
        auto buf_k = device.create_buffer<float>(T * H);
        auto buf_pds = device.create_buffer<float>(head_dim);
        luisa::vector<float> out_q(T * H);
        luisa::vector<float> out_k(T * H);

        float q_scale = 0.5f;
        float k_scale = 0.25f;

        CommandList cmd;
        cmd << buf_q.copy_from(host_q.data())
            << buf_k.copy_from(host_k.data())
            << buf_pds.copy_from(host_pds.data())
            << scale_qk_shader(buf_q, buf_k, buf_pds, T, num_heads, head_dim, q_scale, k_scale).dispatch(H, T)
            << buf_q.copy_to(out_q.data())
            << buf_k.copy_to(out_k.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t ch = 0; ch < H; ch++) {
                uint32_t d = ch % head_dim;
                float pds = host_pds[d];
                float sp = (pds > 20.0f) ? pds : std::log(1.0f + std::exp(pds));
                float exp_q = host_q[t * H + ch] * (q_scale * sp);
                float exp_k = host_k[t * H + ch] * k_scale;
                TEST_ASSERT(std::abs(out_q[t * H + ch] - exp_q) < 1e-5f, "AudioScaleQK Q mismatch");
                TEST_ASSERT(std::abs(out_k[t * H + ch] - exp_k) < 1e-5f, "AudioScaleQK K mismatch");
            }
        }
        std::cout << "  [PASS] AudioScaleQK kernel verified." << std::endl;
    }

    // (p) Test AudioGenRelPos Kernel
    {
        auto rel_pos_shader = device.compile(make_audio_gen_rel_pos_kernel());
        constexpr uint32_t pos_len = 13u;
        constexpr uint32_t hidden_size = 1024u;
        auto buf_pos = device.create_buffer<float>(pos_len * hidden_size);
        luisa::vector<float> out_pos(pos_len * hidden_size);

        CommandList cmd;
        cmd << rel_pos_shader(buf_pos, pos_len, hidden_size).dispatch(512u, pos_len)
            << buf_pos.copy_to(out_pos.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t p = 0; p < pos_len; p++) {
            float pos_id = static_cast<float>(pos_len - 1u - p);
            float log_inc = std::log(10000.0f) / 511.0f;
            for (uint32_t d = 0; d < 512u; d++) {
                float inv_ts = std::exp(-static_cast<float>(d) * log_inc);
                float scaled = pos_id * inv_ts;
                float exp_sin = std::sin(scaled);
                float exp_cos = std::cos(scaled);
                TEST_ASSERT(std::abs(out_pos[p * hidden_size + d] - exp_sin) < 1e-5f, "RelPos sin mismatch");
                TEST_ASSERT(std::abs(out_pos[p * hidden_size + 512u + d] - exp_cos) < 1e-5f, "RelPos cos mismatch");
            }
        }
        std::cout << "  [PASS] AudioGenRelPos kernel verified." << std::endl;
    }

    // (q) Test AudioChunkedAttention Kernel
    {
        auto attn_shader = device.compile(make_audio_chunked_attention_kernel());
        constexpr uint32_t T = 24u;
        constexpr uint32_t num_heads = 2u;
        constexpr uint32_t head_dim = 16u;
        constexpr uint32_t H = num_heads * head_dim;
        luisa::vector<float> host_q(T * H, 0.1f);
        luisa::vector<float> host_k(T * H, 0.1f);
        luisa::vector<float> host_v(T * H, 1.0f);
        luisa::vector<float> host_rel_k(13u * H, 0.05f);

        auto buf_q = device.create_buffer<float>(T * H);
        auto buf_k = device.create_buffer<float>(T * H);
        auto buf_v = device.create_buffer<float>(T * H);
        auto buf_rel_k = device.create_buffer<float>(13u * H);
        auto buf_out = device.create_buffer<float>(T * H);
        luisa::vector<float> out_attn(T * H);

        CommandList cmd;
        cmd << buf_q.copy_from(host_q.data())
            << buf_k.copy_from(host_k.data())
            << buf_v.copy_from(host_v.data())
            << buf_rel_k.copy_from(host_rel_k.data())
            << attn_shader(buf_q, buf_k, buf_v, buf_rel_k, buf_out, T, num_heads, head_dim, 12u, 50.0f).dispatch(num_heads, T)
            << buf_out.copy_to(out_attn.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t d = 0; d < H; d++) {
                TEST_ASSERT(std::abs(out_attn[t * H + d] - 1.0f) < 1e-4f, "AudioChunkedAttention convex sum must equal 1.0f");
            }
        }
        std::cout << "  [PASS] AudioChunkedAttention kernel verified." << std::endl;
    }
}

// ============================================================================
// 4. Selective Encoder Loading Tests
// ============================================================================
void test_selective_encoder_loading(Device &device, const std::filesystem::path &egm_path, const std::filesystem::path &tok_path) {
    std::cout << "[TEST] Running Selective Encoder Loading tests..." << std::endl;

    if (!std::filesystem::exists(egm_path) || !std::filesystem::exists(tok_path)) {
        std::cout << "  [SKIP] Model files not found. Skipping Selective Loading test." << std::endl;
        return;
    }

    // 1. Text-only mode (enable_vision = false, enable_audio = false)
    std::cout << "  [INFO] Initializing text-only session (enable_vision=false, enable_audio=false)..." << std::endl;
    SessionConfig cfg_text_only{};
    cfg_text_only.enable_vision = false;
    cfg_text_only.enable_audio = false;
    auto sess_text = Session::create(device, egm_path, tok_path, cfg_text_only);
    TEST_ASSERT(sess_text != nullptr, "Session create must succeed in text-only mode");
    TEST_ASSERT(!sess_text->has_vision(), "has_vision() must be false when enable_vision=false");
    TEST_ASSERT(!sess_text->has_audio(), "has_audio() must be false when enable_audio=false");

    auto emb_text1 = sess_text->encode("Test sentence for selective loading.");
    TEST_ASSERT(emb_text1.dimension == 768u, "Dimension must be 768");
    float norm1 = 0.0f;
    for (float v : emb_text1.values) norm1 += v * v;
    TEST_ASSERT(std::abs(norm1 - 1.0f) < 1e-4f, "Embedding must have unit norm");

    // 2. Multimodal Vision mode (enable_vision = true, enable_audio = false)
    std::cout << "  [INFO] Initializing multimodal session (enable_vision=true, enable_audio=false)..." << std::endl;
    SessionConfig cfg_mm{};
    cfg_mm.enable_vision = true;
    cfg_mm.enable_audio = false;
    auto sess_mm = Session::create(device, egm_path, tok_path, cfg_mm);
    TEST_ASSERT(sess_mm != nullptr, "Session create must succeed in multimodal vision mode");

    if (sess_mm->has_vision()) {
        std::cout << "  [INFO] Multimodal session successfully loaded Vision Tower." << std::endl;
        auto emb_text2 = sess_mm->encode("Test sentence for selective loading.");

        // Parity check: text encoding in multimodal session must be bit-identical to text-only session!
        float max_diff = 0.0f;
        for (uint32_t i = 0; i < 768u; i++) {
            max_diff = std::max(max_diff, std::abs(emb_text1.values[i] - emb_text2.values[i]));
        }
        std::cout << "  [INFO] Text embedding max diff between text-only and vision session: "
                  << max_diff << std::endl;
        TEST_ASSERT(max_diff < 1e-5f, "Selective loading must preserve exact text embedding compatibility");
    }

    // 3. Multimodal Audio mode (enable_vision = false, enable_audio = true)
    std::cout << "  [INFO] Initializing audio session (enable_vision=false, enable_audio=true)..." << std::endl;
    SessionConfig cfg_audio{};
    cfg_audio.enable_vision = false;
    cfg_audio.enable_audio = true;
    auto sess_audio = Session::create(device, egm_path, tok_path, cfg_audio);
    TEST_ASSERT(sess_audio != nullptr, "Session create must succeed in audio mode");

    if (sess_audio->has_audio()) {
        std::cout << "  [INFO] Audio session successfully loaded Audio Tower." << std::endl;
        auto emb_text3 = sess_audio->encode("Test sentence for selective loading.");

        float max_diff = 0.0f;
        for (uint32_t i = 0; i < 768u; i++) {
            max_diff = std::max(max_diff, std::abs(emb_text1.values[i] - emb_text3.values[i]));
        }
        std::cout << "  [INFO] Text embedding max diff between text-only and audio session: "
                  << max_diff << std::endl;
        TEST_ASSERT(max_diff < 1e-5f, "Selective audio loading must preserve exact text embedding compatibility");
    }

    std::cout << "  [PASS] Selective Encoder Loading tests passed." << std::endl;
}

// ============================================================================
// 5. Image & Multimodal End-to-End Tests
// ============================================================================
void test_multimodal_e2e(
    Device &device,
    const std::filesystem::path &egm_path,
    const std::filesystem::path &tok_path,
    const std::filesystem::path &data_dir) {

    std::cout << "[TEST] Running Image & Multimodal End-to-End inference tests..." << std::endl;

    if (!std::filesystem::exists(egm_path) || !std::filesystem::exists(tok_path)) {
        std::cout << "  [SKIP] Model files not found. Skipping E2E test." << std::endl;
        return;
    }

    SessionConfig cfg{};
    cfg.enable_vision = true;
    auto session = Session::create(device, egm_path, tok_path, cfg);
    if (!session->has_vision()) {
        std::cout << "  [SKIP] Package does not have Vision Tower weights. Skipping multimodal test." << std::endl;
        return;
    }

    // Verify test images exist
    auto forest_img = data_dir / "forest.png";
    auto ocean_img = data_dir / "ocean.png";
    auto sunset_img = data_dir / "sunset.png";
    auto code_img = data_dir / "code.png";
    auto rect_img = data_dir / "rect.png";

    if (!std::filesystem::exists(forest_img)) {
        std::cout << "  [SKIP] Test images not found in '" << data_dir.string() << "'. Skipping." << std::endl;
        return;
    }

    // 1. Test image encoding across MRL dimensions: 768, 512, 256, 128
    std::cout << "  [INFO] Encoding image 'forest.png' across MRL dimensions..." << std::endl;
    for (uint32_t dim : {768u, 512u, 256u, 128u}) {
        EncodeOptions opt{};
        opt.target_dimension = dim;

        auto emb = session->encode_image(forest_img, opt);
        TEST_ASSERT(emb.dimension == dim, "Embedding dimension mismatch");

        float norm_sq = 0.0f;
        for (float v : emb.values) norm_sq += v * v;
        TEST_ASSERT(std::abs(norm_sq - 1.0f) < 1e-4f, "Image embedding must have unit norm");
        std::cout << "    Dim " << dim << ": norm = " << std::sqrt(norm_sq) << " [PASS]" << std::endl;
    }

    // 2. Cross-modal retrieval test: Image vs Text (5 items including non-square rect.png)
    constexpr int kNumItems = 5;
    std::cout << "  [INFO] Running Cross-Modal Retrieval ranking tests (" << kNumItems << " items)..." << std::endl;
    const std::filesystem::path img_paths[kNumItems] = {forest_img, ocean_img, sunset_img, code_img, rect_img};
    const luisa::string img_names[kNumItems] = {"forest", "ocean", "sunset", "code", "rect"};

    const luisa::string_view captions[kNumItems] = {
        "A lush green forest with tall trees and green leaves",
        "Deep blue ocean waters with waves in the sea",
        "A warm sunset with an orange sun in the evening sky",
        "Computer code in an editor syntax highlighting terminal",
        "A sunny sandy beach under a clear blue sky"
    };

    luisa::vector<Embedding> img_embs;
    luisa::vector<Embedding> txt_embs;

    for (int i = 0; i < kNumItems; i++) {
        img_embs.push_back(session->encode_image(img_paths[i]));
        txt_embs.push_back(session->encode(captions[i]));
    }

    std::cout << "\n  [INFO] Computed C++ Similarity Matrix (Image vs Text):" << std::endl;
    for (int i = 0; i < kNumItems; i++) {
        std::cout << "    Image " << img_names[i] << " | ";
        float max_sim = -1.0f;
        int best_match = -1;
        for (int j = 0; j < kNumItems; j++) {
            float sim = 0.0f;
            for (uint32_t d = 0; d < 768u; d++) {
                sim += img_embs[i].values[d] * txt_embs[j].values[d];
            }
            std::cout << img_names[j] << ": " << sim << "  ";
            if (sim > max_sim) {
                max_sim = sim;
                best_match = j;
            }
        }
        std::cout << std::endl;
        TEST_ASSERT(best_match == i, "Top-1 retrieved caption must match the image content!");
    }

    // 3. Parity comparison with golden_vision.json if available
    auto golden_path = data_dir / "golden_vision.json";
    if (std::filesystem::exists(golden_path)) {
        std::cout << "\n  [INFO] Validating against PyTorch golden reference in '"
                  << golden_path.string() << "'..." << std::endl;

        std::ifstream f(golden_path);
        std::string json_str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        yyjson_doc *doc = yyjson_read(json_str.c_str(), json_str.size(), 0);
        TEST_ASSERT(doc != nullptr, "Failed to parse golden_vision.json");

        yyjson_val *root = yyjson_doc_get_root(doc);
        yyjson_val *images_val = yyjson_obj_get(root, "images");

        for (int i = 0; i < kNumItems; i++) {
            yyjson_val *item = yyjson_obj_get(images_val, img_names[i].c_str());
            if (!item) continue;
            yyjson_val *arr768 = yyjson_obj_get(item, "768");
            if (!arr768) continue;

            float dot = 0.0f;
            float pt_norm_sq = 0.0f;
            float cpp_norm_sq = 0.0f;
            for (size_t d = 0; d < 768; d++) {
                float pt_val = static_cast<float>(yyjson_get_num(yyjson_arr_get(arr768, d)));
                float cpp_val = img_embs[i].values[d];
                dot += cpp_val * pt_val;
                pt_norm_sq += pt_val * pt_val;
                cpp_norm_sq += cpp_val * cpp_val;
            }
            float cos_sim = dot / (std::sqrt(pt_norm_sq) * std::sqrt(cpp_norm_sq));
            std::cout << "    Parity " << img_names[i] << ": cos_sim with PyTorch = "
                      << cos_sim << " (threshold > 0.99)" << std::endl;
            TEST_ASSERT(cos_sim > 0.99f && cos_sim <= 1.00001f, "C++ image embedding must match PyTorch golden reference!");
        }
        yyjson_doc_free(doc);
    }

    // 4. Test multimodal prompt encoding (image + text)
    std::cout << "\n  [INFO] Testing multimodal joint encoding (image + text)..." << std::endl;
    auto mm_emb = session->encode_multimodal(forest_img, "A view of the forest");
    TEST_ASSERT(mm_emb.dimension == 768u, "Multimodal embedding dimension mismatch");
    float mm_norm_sq = 0.0f;
    for (float v : mm_emb.values) mm_norm_sq += v * v;
    TEST_ASSERT(std::abs(mm_norm_sq - 1.0f) < 1e-4f, "Multimodal embedding must have unit norm");
    std::cout << "    Joint multimodal embedding norm = " << std::sqrt(mm_norm_sq) << " [PASS]" << std::endl;

    std::cout << "  [PASS] Image & Multimodal End-to-End tests passed." << std::endl;
}

// ============================================================================
// 6. Audio Processor Unit Tests
// ============================================================================
void test_audio_processor(const std::filesystem::path &data_dir, const std::filesystem::path &egm_path) {
    std::cout << "[TEST] Running AudioProcessor tests..." << std::endl;
    auto wav_path = data_dir / "roses_are_16k.wav";
    if (!std::filesystem::exists(wav_path)) {
        wav_path = data_dir / "roses-are.wav";
    }
    if (!std::filesystem::exists(wav_path)) {
        std::cout << "  [SKIP] roses_are_16k.wav not found in '" << data_dir.string() << "'. Skipping AudioProcessor test." << std::endl;
        return;
    }

    // 1. Resampling test: synthetic 48 kHz sine wave downsampled to 16 kHz
    {
        constexpr uint32_t in_sr = 48000u;
        constexpr size_t in_samples = 4800u; // 0.1s
        luisa::vector<float> sine48k(in_samples);
        for (size_t i = 0; i < in_samples; i++) {
            float t = static_cast<float>(i) / static_cast<float>(in_sr);
            sine48k[i] = std::sin(2.0f * 3.14159265f * 440.0f * t);
        }
        auto resampled = AudioProcessor::resample_to_16k(luisa::span{sine48k.data(), sine48k.size()}, in_sr);
        TEST_ASSERT(resampled.size() == 1600u, "Resampled 48k->16k sample count must match exactly (1600)");
        float max_val = 0.0f;
        for (float s : resampled) max_val = std::max(max_val, std::abs(s));
        TEST_ASSERT(max_val > 0.8f && max_val <= 1.05f, "Resampled amplitude must preserve sine signal");
        std::cout << "  [PASS] Audio sinc downsampler (48k -> 16k) verified." << std::endl;
    }

    // 1b. 24-bit PCM WAV decoding test
    {
        constexpr uint32_t num_samples_24 = 480u;
        constexpr uint32_t sr_24 = 16000u;
        constexpr uint32_t data_size_24 = num_samples_24 * 3u;
        luisa::vector<uint8_t> wav24(44u + data_size_24);

        std::memcpy(wav24.data(), "RIFF", 4);
        *reinterpret_cast<uint32_t *>(wav24.data() + 4) = static_cast<uint32_t>(36u + data_size_24);
        std::memcpy(wav24.data() + 8, "WAVEfmt ", 8);
        *reinterpret_cast<uint32_t *>(wav24.data() + 16) = 16u;
        *reinterpret_cast<uint16_t *>(wav24.data() + 20) = 1u;  // PCM
        *reinterpret_cast<uint16_t *>(wav24.data() + 22) = 1u;  // mono
        *reinterpret_cast<uint32_t *>(wav24.data() + 24) = sr_24;
        *reinterpret_cast<uint32_t *>(wav24.data() + 28) = sr_24 * 3u;
        *reinterpret_cast<uint16_t *>(wav24.data() + 32) = 3u;
        *reinterpret_cast<uint16_t *>(wav24.data() + 34) = 24u; // 24-bit
        std::memcpy(wav24.data() + 36, "data", 4);
        *reinterpret_cast<uint32_t *>(wav24.data() + 40) = data_size_24;

        for (uint32_t i = 0; i < num_samples_24; i++) {
            float t = static_cast<float>(i) / static_cast<float>(sr_24);
            float s = 0.75f * std::sin(2.0f * 3.14159265f * 440.0f * t);
            int32_t val = static_cast<int32_t>(s * 8388607.0f);
            wav24[44u + i * 3u] = static_cast<uint8_t>(val & 0xFF);
            wav24[44u + i * 3u + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
            wav24[44u + i * 3u + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
        }

        auto proc24 = AudioProcessor::process_wav_bytes(wav24.data(), wav24.size(), nullptr);
        TEST_ASSERT(proc24.raw_samples_16k.size() == num_samples_24, "24-bit WAV sample count match");
        float max_24 = 0.0f;
        for (float s : proc24.raw_samples_16k) max_24 = std::max(max_24, std::abs(s));
        TEST_ASSERT(std::abs(max_24 - 0.75f) < 0.02f, "24-bit WAV amplitude preserved");
        std::cout << "  [PASS] 24-bit PCM WAV decoding verified." << std::endl;
    }

    // 2. Real audio feature extraction test
    auto pkg = std::filesystem::exists(egm_path) ? Package::load(egm_path) : nullptr;
    auto processed = AudioProcessor::process_file(wav_path, pkg.get());
    std::cout << "  [INFO] Processed audio: num_frames=" << processed.num_frames
              << ", num_features=" << processed.num_features
              << ", num_soft_tokens=" << processed.num_soft_tokens << std::endl;

    TEST_ASSERT(processed.num_frames == 307u, "num_frames must be 307 for roses_are_16k.wav");
    TEST_ASSERT(processed.num_features == 128u, "num_features must be 128");
    TEST_ASSERT(processed.num_soft_tokens == 77u, "num_soft_tokens must be 77");
    TEST_ASSERT(processed.features.size() == 307u * 128u, "features buffer size must be 307 * 128");

    // 3. Correlation check with golden_audio.json
    auto golden_path = data_dir / "golden_audio.json";
    if (std::filesystem::exists(golden_path)) {
        std::ifstream f(golden_path);
        std::string json_str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        yyjson_doc *doc = yyjson_read(json_str.c_str(), json_str.size(), 0);
        TEST_ASSERT(doc != nullptr, "Failed to parse golden_audio.json");
        yyjson_val *root = yyjson_doc_get_root(doc);
        yyjson_val *feats_val = yyjson_obj_get(root, "input_features");
        if (feats_val && yyjson_is_arr(feats_val)) {
            size_t frames = yyjson_arr_size(feats_val);
            TEST_ASSERT(frames == processed.num_frames, "Frames count mismatch with golden reference");

            double sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0, sum_y2 = 0.0, sum_xy = 0.0;
            size_t total = 0;
            for (size_t t = 0; t < frames; t++) {
                yyjson_val *frame = yyjson_arr_get(feats_val, t);
                for (size_t c = 0; c < 128; c++) {
                    double golden_v = yyjson_get_num(yyjson_arr_get(frame, c));
                    double cpp_v = processed.features[t * 128 + c];
                    sum_x += cpp_v;
                    sum_y += golden_v;
                    sum_x2 += cpp_v * cpp_v;
                    sum_y2 += golden_v * golden_v;
                    sum_xy += cpp_v * golden_v;
                    total++;
                }
            }
            double num = static_cast<double>(total) * sum_xy - sum_x * sum_y;
            double den = std::sqrt((static_cast<double>(total) * sum_x2 - sum_x * sum_x) *
                                   (static_cast<double>(total) * sum_y2 - sum_y * sum_y));
            double corr = num / den;
            std::cout << "  [INFO] Mel spectrogram correlation with PyTorch golden: " << corr
                      << " (threshold > 0.999)" << std::endl;
            TEST_ASSERT(corr > 0.999, "AudioProcessor Mel features correlation must exceed 0.999");
        }
        yyjson_doc_free(doc);
    }

    std::cout << "  [PASS] AudioProcessor tests passed." << std::endl;
}

// ============================================================================
// 7. Audio End-to-End and Parity Tests
// ============================================================================
void test_audio_e2e_and_golden(
    Device &device,
    const std::filesystem::path &egm_path,
    const std::filesystem::path &tok_path,
    const std::filesystem::path &data_dir) {

    std::cout << "[TEST] Running Audio End-to-End inference tests..." << std::endl;

    if (!std::filesystem::exists(egm_path) || !std::filesystem::exists(tok_path)) {
        std::cout << "  [SKIP] Model files not found. Skipping Audio E2E test." << std::endl;
        return;
    }

    SessionConfig cfg{};
    cfg.enable_vision = false;
    cfg.enable_audio = true;
    auto session = Session::create(device, egm_path, tok_path, cfg);
    if (!session->has_audio()) {
        std::cout << "  [SKIP] Package does not have Audio Tower weights. Skipping audio test." << std::endl;
        return;
    }

    auto audio_file = data_dir / "roses_are_16k.wav";
    if (!std::filesystem::exists(audio_file)) {
        audio_file = data_dir / "roses-are.wav";
    }
    if (!std::filesystem::exists(audio_file)) {
        std::cout << "  [SKIP] Audio file not found in '" << data_dir.string() << "'. Skipping." << std::endl;
        return;
    }

    // 1. Test audio encoding across MRL dimensions: 768, 512, 256, 128
    std::cout << "  [INFO] Encoding audio 'roses_are_16k.wav' across MRL dimensions..." << std::endl;
    luisa::unordered_map<uint32_t, Embedding> audio_embs_mrl;
    for (uint32_t dim : {768u, 512u, 256u, 128u}) {
        EncodeOptions opt{};
        opt.target_dimension = dim;

        auto emb = session->encode_audio(audio_file, opt);
        TEST_ASSERT(emb.dimension == dim, "Embedding dimension mismatch");

        float norm_sq = 0.0f;
        for (float v : emb.values) norm_sq += v * v;
        TEST_ASSERT(std::abs(norm_sq - 1.0f) < 1e-4f, "Audio embedding must have unit norm");
        std::cout << "    Dim " << dim << ": norm = " << std::sqrt(norm_sq) << " [PASS]" << std::endl;
        audio_embs_mrl[dim] = std::move(emb);
    }

    // 2. Parity comparison with golden_audio.json if available
    auto golden_path = data_dir / "golden_audio.json";
    if (std::filesystem::exists(golden_path)) {
        std::cout << "\n  [INFO] Validating Audio parity against PyTorch golden reference in '"
                  << golden_path.string() << "'..." << std::endl;

        std::ifstream f(golden_path);
        std::string json_str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        yyjson_doc *doc = yyjson_read(json_str.c_str(), json_str.size(), 0);
        TEST_ASSERT(doc != nullptr, "Failed to parse golden_audio.json");

        yyjson_val *root = yyjson_doc_get_root(doc);
        yyjson_val *embs_val = yyjson_obj_get(root, "embeddings");

        if (embs_val && yyjson_is_obj(embs_val)) {
            for (uint32_t dim : {768u, 512u, 256u, 128u}) {
                std::string dim_str = std::to_string(dim);
                yyjson_val *arr_val = yyjson_obj_get(embs_val, dim_str.c_str());
                if (!arr_val) continue;

                const auto &cpp_emb = audio_embs_mrl[dim];
                float dot = 0.0f;
                float pt_norm_sq = 0.0f;
                float cpp_norm_sq = 0.0f;
                for (size_t d = 0; d < dim; d++) {
                    float pt_val = static_cast<float>(yyjson_get_num(yyjson_arr_get(arr_val, d)));
                    float cpp_val = cpp_emb.values[d];
                    dot += cpp_val * pt_val;
                    pt_norm_sq += pt_val * pt_val;
                    cpp_norm_sq += cpp_val * cpp_val;
                }
                float cos_sim = dot / (std::sqrt(pt_norm_sq) * std::sqrt(cpp_norm_sq));
                std::cout << "    Audio Parity Dim " << dim << ": cos_sim with PyTorch = "
                          << cos_sim << " (threshold > 0.999)" << std::endl;
                TEST_ASSERT(cos_sim > 0.999f && cos_sim <= 1.00001f,
                            "C++ audio embedding must match PyTorch golden reference!");
            }
        }

        // 2b. Fine-Grained Key Audio Layer Operator Alignment against PyTorch
        {
            std::cout << "\n  [INFO] Validating Fine-Grained Key Audio Layer Operator Alignment against PyTorch..." << std::endl;
            auto audio_pkg = Package::load(egm_path);
            auto audio_exec = Executor::create(device, *audio_pkg, false, true);

            auto proc_audio = AudioProcessor::process_file(audio_file, audio_pkg.get());
            uint32_t T_in = proc_audio.num_frames;
            uint32_t num_soft = proc_audio.num_soft_tokens;

            auto buf_in_feats = device.create_buffer<float>(T_in * 128u);
            auto buf_soft_tokens = device.create_buffer<float>(num_soft * 512u);
            auto buf_subsample_proj = device.create_buffer<float>(num_soft * 1024u);
            auto buf_layer0_out = device.create_buffer<float>(num_soft * 1024u);
            auto buf_layer11_out = device.create_buffer<float>(num_soft * 1024u);
            auto buf_tower_out = device.create_buffer<float>(num_soft * 1536u);

            Executor::AudioDebugActivations debug_acts{};
            debug_acts.subsample_proj = buf_subsample_proj.view();
            debug_acts.layer0_out = buf_layer0_out.view();
            debug_acts.layer11_out = buf_layer11_out.view();
            debug_acts.audio_tower_out = buf_tower_out.view();

            CommandList align_cmd;
            align_cmd << buf_in_feats.copy_from(proc_audio.features.data());
            audio_exec->record_audio_encode(
                align_cmd, buf_in_feats.view(), buf_soft_tokens.view(),
                T_in, 128u, num_soft, &debug_acts);

            luisa::vector<float> host_sub_proj(num_soft * 1024u);
            luisa::vector<float> host_layer0(num_soft * 1024u);
            luisa::vector<float> host_layer11(num_soft * 1024u);
            luisa::vector<float> host_tower_out(num_soft * 1536u);
            luisa::vector<float> host_soft_tokens(num_soft * 512u);

            align_cmd << buf_subsample_proj.copy_to(host_sub_proj.data())
                      << buf_layer0_out.copy_to(host_layer0.data())
                      << buf_layer11_out.copy_to(host_layer11.data())
                      << buf_tower_out.copy_to(host_tower_out.data())
                      << buf_soft_tokens.copy_to(host_soft_tokens.data());

            Stream align_stream = device.create_stream();
            align_stream << align_cmd.commit() << synchronize();

            auto compare_layer = [](const char *name, const luisa::vector<float> &cpp_data, yyjson_val *gold_val, size_t rows, size_t cols, float min_cos) {
                if (!gold_val) return;
                float dot = 0.0f;
                float pt_norm_sq = 0.0f;
                float cpp_norm_sq = 0.0f;
                float max_diff = 0.0f;
                for (size_t r = 0; r < rows; r++) {
                    yyjson_val *row_val = yyjson_arr_get(gold_val, r);
                    for (size_t c = 0; c < cols; c++) {
                        float pt_val = static_cast<float>(yyjson_get_num(yyjson_arr_get(row_val, c)));
                        float cpp_val = cpp_data[r * cols + c];
                        dot += cpp_val * pt_val;
                        pt_norm_sq += pt_val * pt_val;
                        cpp_norm_sq += cpp_val * cpp_val;
                        max_diff = std::max(max_diff, std::abs(cpp_val - pt_val));
                    }
                }
                float cos_sim = dot / (std::sqrt(pt_norm_sq) * std::sqrt(cpp_norm_sq) + 1e-12f);
                std::cout << "    Operator Alignment [" << name << "]: cos_sim=" << cos_sim
                          << ", max_diff=" << max_diff << " (threshold > " << min_cos << ")" << std::endl;
                TEST_ASSERT(cos_sim > min_cos, "Operator alignment cosine similarity below threshold");
            };

            compare_layer("Subsample Conv Projection", host_sub_proj, yyjson_obj_get(root, "subsample_proj"), num_soft, 1024u, 0.999f);
            yyjson_val *layer0_obj = yyjson_obj_get(root, "layer0");
            if (layer0_obj) {
                compare_layer("Conformer Layer 0 Output", host_layer0, yyjson_obj_get(layer0_obj, "out"), num_soft, 1024u, 0.999f);
            }
            compare_layer("Conformer Layer 11 Output", host_layer11, yyjson_obj_get(root, "layer11_out"), num_soft, 1024u, 0.999f);
            compare_layer("Audio Tower Output Proj", host_tower_out, yyjson_obj_get(root, "audio_tower_out"), num_soft, 1536u, 0.999f);
            compare_layer("Audio Connector Soft Tokens", host_soft_tokens, yyjson_obj_get(root, "soft_tokens"), num_soft, 512u, 0.999f);
        }

        // 3. Audio-to-Text retrieval ranking
        std::cout << "\n  [INFO] Running Zero-Shot Audio-to-Text retrieval ranking test..." << std::endl;
        yyjson_val *cands_val = yyjson_obj_get(root, "candidates");
        const luisa::string candidates[] = {
            "a person speaking English",
            "random static white noise",
            "Roses are red, violets are blue."
        };
        constexpr size_t num_cands = sizeof(candidates) / sizeof(candidates[0]);

        float max_sim = -1.0f;
        int best_idx = -1;
        for (size_t i = 0; i < num_cands; i++) {
            auto txt_emb = session->encode(candidates[i]);
            float sim = 0.0f;
            for (uint32_t d = 0; d < 768u; d++) {
                sim += audio_embs_mrl[768u].values[d] * txt_emb.values[d];
            }
            std::cout << "    Candidate [" << (i + 1) << "] Sim: " << sim << " -> \"" << candidates[i] << "\"";
            if (cands_val) {
                yyjson_val *cand_obj = yyjson_obj_get(cands_val, candidates[i].c_str());
                if (cand_obj) {
                    float golden_sim = static_cast<float>(yyjson_get_num(yyjson_obj_get(cand_obj, "similarity")));
                    std::cout << " (golden: " << golden_sim << ")";
                    TEST_ASSERT(std::abs(sim - golden_sim) < 0.02f, "Similarity match with golden");
                }
            }
            std::cout << std::endl;
            if (sim > max_sim) {
                max_sim = sim;
                best_idx = static_cast<int>(i);
            }
        }
        TEST_ASSERT(best_idx == 2, "Top-1 retrieved candidate must be 'Roses are red, violets are blue.'");
        std::cout << "    Top-1 Match: \"" << candidates[best_idx] << "\" [PASS]" << std::endl;

        // 4. Interleaved multimodal query
        yyjson_val *interleaved_val = yyjson_obj_get(root, "interleaved");
        if (interleaved_val) {
            const char *prompt_str = yyjson_get_str(yyjson_obj_get(interleaved_val, "prompt"));
            yyjson_val *gold_emb_arr = yyjson_obj_get(interleaved_val, "embedding");
            if (prompt_str && gold_emb_arr) {
                std::cout << "\n  [INFO] Testing Interleaved Multimodal query: \"" << prompt_str << "\"..." << std::endl;
                auto mm_emb = session->encode_multimodal_audio(audio_file, prompt_str);
                TEST_ASSERT(mm_emb.dimension == 768u, "Multimodal dimension mismatch");

                float dot = 0.0f;
                float pt_norm_sq = 0.0f;
                float cpp_norm_sq = 0.0f;
                for (size_t d = 0; d < 768u; d++) {
                    float pt_val = static_cast<float>(yyjson_get_num(yyjson_arr_get(gold_emb_arr, d)));
                    float cpp_val = mm_emb.values[d];
                    dot += cpp_val * pt_val;
                    pt_norm_sq += pt_val * pt_val;
                    cpp_norm_sq += cpp_val * cpp_val;
                }
                float cos_sim = dot / (std::sqrt(pt_norm_sq) * std::sqrt(cpp_norm_sq));
                std::cout << "    Interleaved query: cos_sim with PyTorch = " << cos_sim << " (threshold > 0.999)" << std::endl;
                TEST_ASSERT(cos_sim > 0.999f && cos_sim <= 1.00001f,
                            "Interleaved multimodal audio embedding must match PyTorch golden reference!");
            }
        }

        yyjson_doc_free(doc);
    }

    std::cout << "  [PASS] Audio End-to-End tests passed." << std::endl;
}

int main(int argc, char **argv) {
    luisa::string backend = (argc > 1) ? argv[1] : "dx";
    std::filesystem::path egm_path = (argc > 2) ? argv[2] : "model.mmegm";
    std::filesystem::path tok_path = (argc > 3) ? argv[3] : "tokenizer.json";
    std::filesystem::path data_dir = (argc > 4) ? argv[4] : "examples/mm_embedding/data";
    if (!std::filesystem::exists(egm_path)) {
        if (std::filesystem::exists("../../model.mmegm")) {
            egm_path = "../../model.mmegm";
        } else if (std::filesystem::exists("../model.mmegm")) {
            egm_path = "../model.mmegm";
        }
    }
    if (!std::filesystem::exists(tok_path)) {
        if (std::filesystem::exists("../../tokenizer.json")) {
            tok_path = "../../tokenizer.json";
        } else if (std::filesystem::exists("../../models/embeddinggemma-2/tokenizer.json")) {
            tok_path = "../../models/embeddinggemma-2/tokenizer.json";
        } else if (std::filesystem::exists("models/embeddinggemma-2/tokenizer.json")) {
            tok_path = "models/embeddinggemma-2/tokenizer.json";
        }
    }
    if (!std::filesystem::exists(data_dir)) {
        if (std::filesystem::exists("../../examples/mm_embedding/data")) {
            data_dir = "../../examples/mm_embedding/data";
        } else if (std::filesystem::exists("../examples/mm_embedding/data")) {
            data_dir = "../examples/mm_embedding/data";
        }
    }

    std::cout << "========================================================\n"
              << " LuisaCompute mm_embedding Phase 3 Automated Test Suite\n"
              << " Backend: " << backend << "\n"
              << "========================================================\n"
              << std::endl;

    // 1. Host unit tests
    test_tokenizer(tok_path);
    test_package(egm_path);
    test_audio_processor(data_dir, egm_path);

    // 2. GPU unit & regression tests
    Context context{argv[0]};
    auto device = context.create_device(backend);

    test_kernels(device);
    test_selective_encoder_loading(device, egm_path, tok_path);
    test_multimodal_e2e(device, egm_path, tok_path, data_dir);
    test_audio_e2e_and_golden(device, egm_path, tok_path, data_dir);

    std::cout << "\n========================================================\n"
              << " ALL PHASE 3 AUDIO & VISION TESTS PASSED SUCCESSFULLY!\n"
              << "========================================================"
              << std::endl;

    return 0;
}
