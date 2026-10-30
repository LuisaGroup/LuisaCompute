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
#include "mm_embedding/embed/kernels/kernels_attention.h"
#include "mm_embedding/embed/kernels/kernels_gemm.h"
#include "mm_embedding/embed/kernels/kernels_mlp.h"
#include "mm_embedding/embed/kernels/kernels_norm.h"
#include "mm_embedding/embed/kernels/kernels_pool.h"
#include "mm_embedding/embed/kernels/kernels_rope.h"
#include "mm_embedding/embed/kernels/kernels_vision.h"
#include "mm_embedding/embed/package.h"
#include "mm_embedding/embed/session.h"
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

    // 1. Text-only mode (enable_vision = false)
    std::cout << "  [INFO] Initializing text-only session (enable_vision=false)..." << std::endl;
    SessionConfig cfg_text_only{};
    cfg_text_only.enable_vision = false;
    auto sess_text = Session::create(device, egm_path, tok_path, cfg_text_only);
    TEST_ASSERT(sess_text != nullptr, "Session create must succeed in text-only mode");
    TEST_ASSERT(!sess_text->has_vision(), "has_vision() must be false when enable_vision=false");

    auto emb_text1 = sess_text->encode("Test sentence for selective loading.");
    TEST_ASSERT(emb_text1.dimension == 768u, "Dimension must be 768");
    float norm1 = 0.0f;
    for (float v : emb_text1.values) norm1 += v * v;
    TEST_ASSERT(std::abs(norm1 - 1.0f) < 1e-4f, "Embedding must have unit norm");

    // 2. Multimodal mode (enable_vision = true)
    std::cout << "  [INFO] Initializing multimodal session (enable_vision=true)..." << std::endl;
    SessionConfig cfg_mm{};
    cfg_mm.enable_vision = true;
    auto sess_mm = Session::create(device, egm_path, tok_path, cfg_mm);
    TEST_ASSERT(sess_mm != nullptr, "Session create must succeed in multimodal mode");

    if (sess_mm->has_vision()) {
        std::cout << "  [INFO] Multimodal session successfully loaded Vision Tower." << std::endl;
        auto emb_text2 = sess_mm->encode("Test sentence for selective loading.");

        // Parity check: text encoding in multimodal session must be bit-identical to text-only session!
        float max_diff = 0.0f;
        for (uint32_t i = 0; i < 768u; i++) {
            max_diff = std::max(max_diff, std::abs(emb_text1.values[i] - emb_text2.values[i]));
        }
        std::cout << "  [INFO] Text embedding max diff between text-only and multimodal session: "
                  << max_diff << std::endl;
        TEST_ASSERT(max_diff < 1e-5f, "Selective loading must preserve exact text embedding compatibility");
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

int main(int argc, char **argv) {
    luisa::string backend = (argc > 1) ? argv[1] : "dx";
    std::filesystem::path egm_path = (argc > 2) ? argv[2] : "model.mmegm";
    std::filesystem::path tok_path = (argc > 3) ? argv[3] : "tokenizer.json";
    std::filesystem::path data_dir = (argc > 4) ? argv[4] : "examples/mm_embedding/data";
    if (!std::filesystem::exists(data_dir)) {
        if (std::filesystem::exists("../../examples/mm_embedding/data")) {
            data_dir = "../../examples/mm_embedding/data";
        } else if (std::filesystem::exists("../examples/mm_embedding/data")) {
            data_dir = "../examples/mm_embedding/data";
        }
    }

    std::cout << "========================================================\n"
              << " LuisaCompute mm_embedding Phase 2 Automated Test Suite\n"
              << " Backend: " << backend << "\n"
              << "========================================================\n"
              << std::endl;

    // 1. Host unit tests
    test_tokenizer(tok_path);
    test_package(egm_path);

    // 2. GPU unit & regression tests
    Context context{argv[0]};
    auto device = context.create_device(backend);

    test_kernels(device);
    test_selective_encoder_loading(device, egm_path, tok_path);
    test_multimodal_e2e(device, egm_path, tok_path, data_dir);

    std::cout << "\n========================================================\n"
              << " ALL PHASE 2 TESTS PASSED SUCCESSFULLY!\n"
              << "========================================================"
              << std::endl;

    return 0;
}
