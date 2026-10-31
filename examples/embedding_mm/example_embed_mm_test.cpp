// embedding_mm engine -- comprehensive automated unit & regression tests.
//
// Tests:
// 1. Tokenizer: special tokens, BPE merges, byte fallback, truncation, batching.
// 2. Package: headers, 64-bit offsets, tensor lookups, vision & audio tensors.
// 3. DSL Kernels (GPU):
//    - RMSNorm, L2Norm, GEMM, GemmBias
//    - Vision: ScalePixels, AxialRoPE, VisionPool (square & non-square), ScatterSoftTokens
//    - Audio: AudioClamp, AudioGlu, AudioDepthwiseConv1D, AudioScatterTokens, AudioSilu,
//             AudioAdd, AudioAddScale, AudioScaleQK, AudioGenRelPos, AudioChunkedAttention
// 4. Selective Encoder Loading:
//    - Text-only (~270M), Text+Vision (~440M), Text+Audio (~570M), Full (~740M)
// 5. FeatureLoader (.mmfeat binary bundles):
//    - Image (default 280 & high-res 1120), Audio, Video (default & 16-frame timestamped), Multimodal
// 6. Official Tutorial E2E & Layer-by-Layer 0-Drift Parity (vs PyTorch golden_official.json):
//    - Text & MRL (768, 512, 256, 128) SearchQuery vs Document
//    - Vision Tower intermediate activations (patch_embed, vit_layer0, pooled_out, soft_tokens)
//    - Working with Images (default 280 & custom 1120) + similarity ranking
//    - Audio Conformer intermediate activations (subsample_proj, layer0, layer11, tower_out, soft_tokens)
//    - Working with Audio + similarity ranking
//    - Working with Video (default 1fps & custom 16-frame timestamped) + similarity ranking
//    - Interleaved Multimodal Inputs (query_multimodal, query_interleaved, query_media_sequence)
//
// Usage:
//   example_embed_mm_test [backend: dx|cuda|vk] [model.mmegm] [tokenizer.json] [data_dir]

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

#include "embedding_mm/embed/executor.h"
#include "embedding_mm/embed/feature_loader.h"
#include "embedding_mm/embed/kernels/kernels_attention.h"
#include "embedding_mm/embed/kernels/kernels_audio.h"
#include "embedding_mm/embed/kernels/kernels_gemm.h"
#include "embedding_mm/embed/kernels/kernels_mlp.h"
#include "embedding_mm/embed/kernels/kernels_norm.h"
#include "embedding_mm/embed/kernels/kernels_pool.h"
#include "embedding_mm/embed/kernels/kernels_rope.h"
#include "embedding_mm/embed/kernels/kernels_vision.h"
#include "embedding_mm/embed/package.h"
#include "embedding_mm/embed/session.h"
#include "embedding_mm/embed/tokenizer.h"

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "TEST FAILED [%s:%d]: %s (%s)\n", __FILE__, __LINE__, msg, #cond); \
            std::exit(1); \
        } \
    } while (0)

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::embed_mm;

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

    auto empty_toks = tok->encode("");
    TEST_ASSERT(empty_toks.size() == 2, "Empty string should encode to [<bos>, <eos>]");
    TEST_ASSERT(empty_toks[0] == 2u, "First token must be BOS (2)");
    TEST_ASSERT(empty_toks[1] == 1u, "Second token must be EOS (1)");

    auto s_toks = tok->encode("Hello world");
    TEST_ASSERT(s_toks.size() >= 3, "Encoding 'Hello world' must have >= 3 tokens");
    TEST_ASSERT(s_toks.front() == 2u, "Must start with BOS");
    TEST_ASSERT(s_toks.back() == 1u, "Must end with EOS");

    auto trunc_toks = tok->encode("Hello world, welcome to multimodal embeddings with Gemma!", 4u);
    TEST_ASSERT(trunc_toks.size() <= 4u, "Max len truncation must be respected");

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

    const auto *t_emb = pkg->find("emb.word");
    TEST_ASSERT(t_emb != nullptr, "Must find 'emb.word'");
    TEST_ASSERT(t_emb->rank == 2u, "'emb.word' rank must be 2");
    TEST_ASSERT(t_emb->dims[0] == 262144u && t_emb->dims[1] == 512u, "emb.word dims must be [262144, 512]");

    const auto *t_final_norm = pkg->find("final_norm");
    TEST_ASSERT(t_final_norm != nullptr, "Must find 'final_norm'");
    TEST_ASSERT(t_final_norm->rank == 1u && t_final_norm->dims[0] == 512u, "final_norm dims must be [512]");

    if (pkg->has_vision()) {
        std::cout << "  [INFO] Model package includes Vision Tower weights." << std::endl;
        TEST_ASSERT(pkg->find("vision.patch_embed.proj") != nullptr, "Must find 'vision.patch_embed.proj'");
        TEST_ASSERT(pkg->find("vision.patch_embed.pos_table") != nullptr, "Must find 'vision.patch_embed.pos_table'");
        TEST_ASSERT(pkg->find("vision.connector.proj") != nullptr, "Must find 'vision.connector.proj'");
        TEST_ASSERT(pkg->find("vision.layer0.input_norm") != nullptr, "Must find 'vision.layer0.input_norm'");
        TEST_ASSERT(pkg->find("vision.layer15.out_proj") != nullptr, "Must find 'vision.layer15.out_proj'");
    }

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
        luisa::vector<int32_t> host_pos = {0, 0, 1, 0, 0, 1, 5, 10};

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

        for (uint32_t i = 0; i < num_heads * head_dim; i++) {
            TEST_ASSERT(std::abs(host_out[i] - 1.0f) < 1e-4f, "Axial RoPE pos(0,0) must preserve values");
        }
        std::cout << "  [PASS] Vision AxialRoPE kernel verified." << std::endl;
    }

    // (f) Test Vision 2D Spatial Pooler Kernel (Square and Non-Square)
    {
        auto pool_shader = device.compile(make_vision_pool_kernel());
        constexpr uint32_t H = 16u;

        {
            constexpr uint32_t grid_soft = 2u;
            constexpr uint32_t num_soft = grid_soft * grid_soft;
            constexpr uint32_t patch_stride = grid_soft * 3u;
            constexpr uint32_t total_patches = patch_stride * patch_stride;

            luisa::vector<float> host_patches(total_patches * H, 1.0f);
            constexpr float scale = 3.0f;

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
        std::cout << "  [PASS] Vision Spatial Pooler kernel verified." << std::endl;
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

    // (h) Test Audio Clamp, GLU, DepthwiseConv1D, SiLU, Add, GemmBias, ScaleQK, RelPos, ChunkedAttn
    {
        auto clamp_shader = device.compile(make_audio_clamp_kernel());
        constexpr uint32_t N = 10u;
        luisa::vector<float> host_x = {-5.0f, -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f, 5.0f, 10.0f};
        auto buf_x = device.create_buffer<float>(N);
        auto buf_y = device.create_buffer<float>(N);
        luisa::vector<float> host_y(N);

        CommandList cmd;
        cmd << buf_x.copy_from(host_x.data())
            << clamp_shader(buf_x, buf_y, -1.5f, 3.0f, N).dispatch(N)
            << buf_y.copy_to(host_y.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t i = 0; i < N; i++) {
            float exp_v = std::clamp(host_x[i], -1.5f, 3.0f);
            TEST_ASSERT(std::abs(host_y[i] - exp_v) < 1e-6f, "AudioClamp value mismatch");
        }
        std::cout << "  [PASS] Audio Conformer atomic kernels verified." << std::endl;
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

    SessionConfig cfg_text_only{false, false};
    auto sess_text = Session::create(device, egm_path, tok_path, cfg_text_only);
    TEST_ASSERT(sess_text != nullptr, "Session create must succeed in text-only mode");
    TEST_ASSERT(!sess_text->has_vision(), "has_vision() must be false when enable_vision=false");
    TEST_ASSERT(!sess_text->has_audio(), "has_audio() must be false when enable_audio=false");

    auto emb_text1 = sess_text->encode("Test sentence for selective loading.");
    TEST_ASSERT(emb_text1.dimension == 768u, "Dimension must be 768");

    SessionConfig cfg_mm{true, false};
    auto sess_mm = Session::create(device, egm_path, tok_path, cfg_mm);
    if (sess_mm->has_vision()) {
        auto emb_text2 = sess_mm->encode("Test sentence for selective loading.");
        float max_diff = 0.0f;
        for (uint32_t i = 0; i < 768u; i++) {
            max_diff = std::max(max_diff, std::abs(emb_text1.values[i] - emb_text2.values[i]));
        }
        TEST_ASSERT(max_diff < 1e-5f, "Selective loading must preserve exact text embedding compatibility");
    }

    SessionConfig cfg_audio{false, true};
    auto sess_audio = Session::create(device, egm_path, tok_path, cfg_audio);
    if (sess_audio->has_audio()) {
        auto emb_text3 = sess_audio->encode("Test sentence for selective loading.");
        float max_diff = 0.0f;
        for (uint32_t i = 0; i < 768u; i++) {
            max_diff = std::max(max_diff, std::abs(emb_text1.values[i] - emb_text3.values[i]));
        }
        TEST_ASSERT(max_diff < 1e-5f, "Selective audio loading must preserve exact text embedding compatibility");
    }

    std::cout << "  [PASS] Selective Encoder Loading tests passed." << std::endl;
}

// ============================================================================
// 5. FeatureLoader (.mmfeat) Unit Tests
// ============================================================================
void test_feature_loader(const std::filesystem::path &data_dir) {
    std::cout << "[TEST] Running FeatureLoader (.mmfeat) tests..." << std::endl;

    auto img_280 = data_dir / "GoldenGate.png.mmfeat";
    auto img_1120 = data_dir / "GoldenGate_1120.png.mmfeat";
    auto aud_feat = data_dir / "roses-are.wav.mmfeat";
    auto vid_def = data_dir / "cat_dozing.mp4.mmfeat";
    auto vid_16f = data_dir / "cat_dozing_16frames.mp4.mmfeat";
    auto inter_mm = data_dir / "interleaved_multimodal.mmfeat";

    if (!std::filesystem::exists(img_280)) {
        std::cout << "  [SKIP] .mmfeat files not found in '" << data_dir.string() << "'." << std::endl;
        return;
    }

    TEST_ASSERT(FeatureLoader::peek_modality(img_280) == ModalityKind::Image, "GoldenGate.png.mmfeat must be Image");
    auto im280 = FeatureLoader::load_image(data_dir / "GoldenGate.png");// sidecar resolution test
    TEST_ASSERT(im280.max_patches == 2520u && im280.num_soft_tokens == 256u, "GoldenGate 280 budget dims");

    if (std::filesystem::exists(img_1120)) {
        auto im1120 = FeatureLoader::load_image(img_1120);
        TEST_ASSERT(im1120.max_patches == 10080u && im1120.num_soft_tokens == 1089u, "GoldenGate 1120 budget dims");
    }

    if (std::filesystem::exists(aud_feat)) {
        TEST_ASSERT(FeatureLoader::peek_modality(aud_feat) == ModalityKind::Audio, "roses-are.wav.mmfeat must be Audio");
        auto aud = FeatureLoader::load_audio(data_dir / "roses-are.wav");
        TEST_ASSERT(aud.num_frames == 307u && aud.num_features == 128u && aud.num_soft_tokens == 77u, "roses-are.wav dims");
    }

    if (std::filesystem::exists(vid_def) && std::filesystem::exists(vid_16f)) {
        TEST_ASSERT(FeatureLoader::peek_modality(vid_def) == ModalityKind::Video, "cat_dozing.mp4.mmfeat must be Video");
        auto v29 = FeatureLoader::load_video(vid_def);
        auto v16 = FeatureLoader::load_video(vid_16f);
        TEST_ASSERT(v29.num_frames == 29u, "Default 1fps video must have 29 frames");
        TEST_ASSERT(v16.num_frames == 16u && !v16.input_ids.empty(), "Custom 16-frame video must have 16 frames + timestamped input_ids");
    }

    if (std::filesystem::exists(inter_mm)) {
        TEST_ASSERT(FeatureLoader::peek_modality(inter_mm) == ModalityKind::Multimodal, "interleaved_multimodal.mmfeat must be Multimodal");
        auto mm = FeatureLoader::load_multimodal(inter_mm);
        TEST_ASSERT(mm.images.size() == 2u && mm.audios.size() == 1u && mm.videos.empty(), "interleaved_multimodal counts");
    }

    std::cout << "  [PASS] FeatureLoader (.mmfeat) tests passed." << std::endl;
}

// ============================================================================
// 6. Official Tutorial E2E & Layer-by-Layer 0-Drift Parity Suite
// ============================================================================
static float check_mrl_parity(const Embedding &cpp_emb, yyjson_val *mrl_obj, uint32_t dim) {
    char key[16];
    std::snprintf(key, sizeof(key), "%u", dim);
    yyjson_val *arr = yyjson_obj_get(mrl_obj, key);
    TEST_ASSERT(arr != nullptr && yyjson_is_arr(arr), "Missing golden MRL array");
    float dot = 0.0f, c_norm = 0.0f, p_norm = 0.0f;
    for (uint32_t d = 0; d < dim; d++) {
        float pt = static_cast<float>(yyjson_get_num(yyjson_arr_get(arr, d)));
        float cp = cpp_emb.values[d];
        dot += cp * pt;
        c_norm += cp * cp;
        p_norm += pt * pt;
    }
    return dot / (std::sqrt(c_norm) * std::sqrt(p_norm) + 1e-12f);
}

void test_official_golden_suite(
    Device &device,
    const std::filesystem::path &egm_path,
    const std::filesystem::path &tok_path,
    const std::filesystem::path &data_dir) {

    std::cout << "[TEST] Running Official Tutorial E2E & Layer-by-Layer 0-Drift Parity Suite..." << std::endl;

    auto golden_path = data_dir / "golden_official.json";
    if (!std::filesystem::exists(egm_path) || !std::filesystem::exists(tok_path) || !std::filesystem::exists(golden_path)) {
        std::cout << "  [SKIP] Model or golden_official.json not found. Skipping E2E parity suite." << std::endl;
        return;
    }

    std::ifstream f(golden_path);
    std::string json_str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    yyjson_doc *doc = yyjson_read(json_str.c_str(), json_str.size(), 0);
    TEST_ASSERT(doc != nullptr, "Failed to parse golden_official.json");
    yyjson_val *root = yyjson_doc_get_root(doc);

    SessionConfig full_cfg{true, true};
    auto session = Session::create(device, egm_path, tok_path, full_cfg);

    // ------------------------------------------------------------------------
    // 6.1 Text & MRL Truncation (768, 512, 256, 128)
    // ------------------------------------------------------------------------
    std::cout << "\n  [6.1] Validating Official Text & MRL (768, 512, 256, 128) Parity..." << std::endl;
    yyjson_val *txt_sec = yyjson_obj_get(root, "text");
    const char *q_str = yyjson_get_str(yyjson_obj_get(txt_sec, "query"));
    const char *d_str = yyjson_get_str(yyjson_obj_get(txt_sec, "document"));
    yyjson_val *q_gold = yyjson_obj_get(txt_sec, "query_embeddings");
    yyjson_val *d_gold = yyjson_obj_get(txt_sec, "document_embeddings");

    for (uint32_t dim : {768u, 512u, 256u, 128u}) {
        EncodeOptions opt{512u, dim};
        auto q_emb = session->encode_query(q_str, opt);
        auto d_emb = session->encode_document(d_str, "none", opt);
        float q_cos = check_mrl_parity(q_emb, q_gold, dim);
        float d_cos = check_mrl_parity(d_emb, d_gold, dim);
        std::printf("    Text MRL Dim %u: Query cos_sim=%.6f, Doc cos_sim=%.6f [PASS]\n", dim, q_cos, d_cos);
        TEST_ASSERT(q_cos > 0.9999f && d_cos > 0.9999f, "Text MRL parity must exceed 0.9999");
    }

    // ------------------------------------------------------------------------
    // 6.2 Vision Tower Layer-by-Layer 0-Drift + Default (280) & High-Res (1120)
    // ------------------------------------------------------------------------
    std::cout << "\n  [6.2] Validating Vision Tower Layer-by-Layer 0-Drift & Image Retrieval (280 & 1120)..." << std::endl;
    yyjson_val *vis_sec = yyjson_obj_get(root, "vision");
    yyjson_val *vis_acts = yyjson_obj_get(vis_sec, "layer_activations");

    {
        auto pkg = Package::load(egm_path);
        auto exec = Executor::create(device, *pkg, true, false);
        auto im = FeatureLoader::load_image(data_dir / "GoldenGate.png.mmfeat");

        uint32_t max_p = im.max_patches;
        uint32_t num_soft = im.num_soft_tokens;
        auto buf_px = device.create_buffer<float>(max_p * 768u);
        auto buf_pos = device.create_buffer<int32_t>(max_p * 2u);
        auto buf_soft = device.create_buffer<float>(num_soft * 512u);
        auto buf_patch = device.create_buffer<float>(max_p * 768u);
        auto buf_l0 = device.create_buffer<float>(max_p * 768u);
        auto buf_pool = device.create_buffer<float>(num_soft * 768u);

        Executor::VisionDebugActivations dbg{buf_patch.view(), buf_l0.view(), buf_pool.view()};
        CommandList cmd;
        cmd << buf_px.copy_from(im.pixel_values.data())
            << buf_pos.copy_from(im.position_ids.data());
        exec->record_vision_encode(cmd, buf_px.view(), buf_pos.view(), buf_soft.view(),
                                   1u, im.num_patches_w, im.num_patches_h, max_p, num_soft, &dbg);

        luisa::vector<float> h_patch(max_p * 768u), h_l0(max_p * 768u), h_pool(num_soft * 768u), h_soft(num_soft * 512u);
        cmd << buf_patch.copy_to(h_patch.data())
            << buf_l0.copy_to(h_l0.data())
            << buf_pool.copy_to(h_pool.data())
            << buf_soft.copy_to(h_soft.data());
        auto s = device.create_stream();
        s << cmd.commit() << synchronize();

        auto check_slice_2d = [](const char *name, const luisa::vector<float> &cpp, size_t stride, size_t rows, size_t cols, yyjson_val *gold) {
            float dot = 0.0f, cn = 0.0f, pn = 0.0f;
            size_t idx = 0;
            for (size_t r = 0; r < rows; r++) {
                for (size_t c = 0; c < cols; c++) {
                    float pt = static_cast<float>(yyjson_get_num(yyjson_arr_get(gold, idx++)));
                    float cp = cpp[r * stride + c];
                    dot += cp * pt;
                    cn += cp * cp;
                    pn += pt * pt;
                }
            }
            float cs = dot / (std::sqrt(cn) * std::sqrt(pn) + 1e-12f);
            std::printf("    Vision Layer [%-26s] cos_sim = %.6f [PASS]\n", name, cs);
            TEST_ASSERT(cs > 0.9999f, "Vision layer 0-drift alignment failed");
        };

        check_slice_2d("Patch Embed + 2D Pos", h_patch, 768, 64, 16, yyjson_obj_get(vis_acts, "patch_embed_slice"));
        check_slice_2d("ViT Layer 0 Output", h_l0, 768, 64, 16, yyjson_obj_get(vis_acts, "vit_layer0_slice"));
        check_slice_2d("Spatial 3x3 Pooler", h_pool, 768, 32, 16, yyjson_obj_get(vis_acts, "pooled_slice"));
        check_slice_2d("Vision Connector SoftToks", h_soft, 512, 32, 16, yyjson_obj_get(vis_acts, "soft_tokens_slice"));
    }

    for (uint32_t dim : {768u, 512u, 256u, 128u}) {
        EncodeOptions opt{512u, dim};
        auto g280 = session->encode_image(data_dir / "GoldenGate.png.mmfeat", opt);
        auto k280 = session->encode_image(data_dir / "kitchen_painting.jpg.mmfeat", opt);
        float cs_g280 = check_mrl_parity(g280, yyjson_obj_get(vis_sec, "GoldenGate_280"), dim);
        float cs_k280 = check_mrl_parity(k280, yyjson_obj_get(vis_sec, "kitchen_painting_280"), dim);
        TEST_ASSERT(cs_g280 > 0.9999f && cs_k280 > 0.9999f, "Image 280 MRL parity failed");
    }
    {
        auto g1120 = session->encode_image(data_dir / "GoldenGate_1120.png.mmfeat");
        auto k1120 = session->encode_image(data_dir / "kitchen_painting_1120.jpg.mmfeat");
        float cs_g1120 = check_mrl_parity(g1120, yyjson_obj_get(vis_sec, "GoldenGate_1120"), 768u);
        float cs_k1120 = check_mrl_parity(k1120, yyjson_obj_get(vis_sec, "kitchen_painting_1120"), 768u);
        std::printf("    Image High-Res (1120 budget, 1089 soft tokens): GoldenGate cos_sim=%.6f, Kitchen cos_sim=%.6f [PASS]\n",
                    cs_g1120, cs_k1120);
        TEST_ASSERT(cs_g1120 > 0.9999f && cs_k1120 > 0.9999f, "Image 1120 MRL parity failed");
    }

    // ------------------------------------------------------------------------
    // 6.3 Audio Conformer Layer-by-Layer 0-Drift & Audio Retrieval
    // ------------------------------------------------------------------------
    std::cout << "\n  [6.3] Validating Audio Conformer Layer-by-Layer 0-Drift & Audio Retrieval..." << std::endl;
    yyjson_val *aud_sec = yyjson_obj_get(root, "audio");
    yyjson_val *aud_acts = yyjson_obj_get(aud_sec, "layer_activations");
    {
        auto pkg = Package::load(egm_path);
        auto exec = Executor::create(device, *pkg, false, true);
        auto aud = FeatureLoader::load_audio(data_dir / "roses-are.wav.mmfeat");
        uint32_t T_in = aud.num_frames;
        uint32_t num_soft = aud.num_soft_tokens;

        auto b_in = device.create_buffer<float>(T_in * 128u);
        auto b_soft = device.create_buffer<float>(num_soft * 512u);
        auto b_sub = device.create_buffer<float>(num_soft * 1024u);
        auto b_l0 = device.create_buffer<float>(num_soft * 1024u);
        auto b_l11 = device.create_buffer<float>(num_soft * 1024u);
        auto b_tow = device.create_buffer<float>(num_soft * 1536u);

        Executor::AudioDebugActivations dbg{b_sub.view(), b_l0.view(), b_l11.view(), b_tow.view()};
        CommandList cmd;
        cmd << b_in.copy_from(aud.features.data());
        exec->record_audio_encode(cmd, b_in.view(), b_soft.view(), T_in, 128u, num_soft, &dbg);

        luisa::vector<float> h_sub(num_soft * 1024u), h_l0(num_soft * 1024u), h_l11(num_soft * 1024u), h_tow(num_soft * 1536u), h_soft(num_soft * 512u);
        cmd << b_sub.copy_to(h_sub.data())
            << b_l0.copy_to(h_l0.data())
            << b_l11.copy_to(h_l11.data())
            << b_tow.copy_to(h_tow.data())
            << b_soft.copy_to(h_soft.data());
        auto s = device.create_stream();
        s << cmd.commit() << synchronize();

        auto check_mat = [](const char *name, const luisa::vector<float> &cpp, size_t rows, size_t cols, yyjson_val *gold) {
            float dot = 0.0f, cn = 0.0f, pn = 0.0f;
            for (size_t r = 0; r < rows; r++) {
                yyjson_val *row_v = yyjson_arr_get(gold, r);
                for (size_t c = 0; c < cols; c++) {
                    float pt = static_cast<float>(yyjson_get_num(yyjson_arr_get(row_v, c)));
                    float cp = cpp[r * cols + c];
                    dot += cp * pt;
                    cn += cp * cp;
                    pn += pt * pt;
                }
            }
            float cs = dot / (std::sqrt(cn) * std::sqrt(pn) + 1e-12f);
            std::printf("    Audio Layer  [%-26s] cos_sim = %.6f [PASS]\n", name, cs);
            TEST_ASSERT(cs > 0.9999f, "Audio layer 0-drift alignment failed");
        };

        check_mat("Subsample Conv Projection", h_sub, num_soft, 1024, yyjson_obj_get(aud_acts, "subsample_proj"));
        check_mat("Conformer Layer 0 Output", h_l0, num_soft, 1024, yyjson_obj_get(aud_acts, "layer0_out"));
        check_mat("Conformer Layer 11 Output", h_l11, num_soft, 1024, yyjson_obj_get(aud_acts, "layer11_out"));
        check_mat("Audio Tower Output Proj", h_tow, num_soft, 1536, yyjson_obj_get(aud_acts, "audio_tower_out"));
        check_mat("Audio Connector SoftToks", h_soft, num_soft, 512, yyjson_obj_get(aud_acts, "soft_tokens"));
    }

    for (uint32_t dim : {768u, 512u, 256u, 128u}) {
        EncodeOptions opt{512u, dim};
        auto a_emb = session->encode_audio(data_dir / "roses-are.wav.mmfeat", opt);
        float cs = check_mrl_parity(a_emb, yyjson_obj_get(aud_sec, "roses_are"), dim);
        TEST_ASSERT(cs > 0.9999f, "Audio MRL parity failed");
    }

    // ------------------------------------------------------------------------
    // 6.4 Video Spatio-Temporal E2E (Default 1fps & Custom 16-Frame Timestamped)
    // ------------------------------------------------------------------------
    std::cout << "\n  [6.4] Validating Video Spatio-Temporal Parity (Default 1fps & 16-Frame Timestamped)..." << std::endl;
    yyjson_val *vid_sec = yyjson_obj_get(root, "video");
    for (uint32_t dim : {768u, 512u, 256u, 128u}) {
        EncodeOptions opt{512u, dim};
        auto v_def = session->encode_video(data_dir / "cat_dozing.mp4.mmfeat", opt);
        auto v_16f = session->encode_video(data_dir / "cat_dozing_16frames.mp4.mmfeat", opt);
        float cs_def = check_mrl_parity(v_def, yyjson_obj_get(vid_sec, "cat_dozing_default"), dim);
        float cs_16f = check_mrl_parity(v_16f, yyjson_obj_get(vid_sec, "cat_dozing_16frames"), dim);
        std::printf("    Video MRL Dim %u: Default(29f) cos_sim=%.6f, Timestamped(16f) cos_sim=%.6f [PASS]\n",
                    dim, cs_def, cs_16f);
        TEST_ASSERT(cs_def > 0.9999f && cs_16f > 0.9999f, "Video MRL parity failed");
    }

    // ------------------------------------------------------------------------
    // 6.5 Interleaved Multimodal Inputs (3 Official Queries)
    // ------------------------------------------------------------------------
    std::cout << "\n  [6.5] Validating Interleaved Multimodal Inputs (3 Official Queries)..." << std::endl;
    yyjson_val *inter_sec = yyjson_obj_get(root, "interleaved");
    auto mm1 = session->encode_multimodal(data_dir / "interleaved_multimodal.mmfeat");
    auto mm2 = session->encode_multimodal(data_dir / "interleaved_prompt.mmfeat");
    auto mm3 = session->encode_multimodal(data_dir / "interleaved_media_seq.mmfeat");

    float cs_mm1 = check_mrl_parity(mm1, yyjson_obj_get(inter_sec, "query_multimodal"), 768u);
    float cs_mm2 = check_mrl_parity(mm2, yyjson_obj_get(inter_sec, "query_interleaved"), 768u);
    float cs_mm3 = check_mrl_parity(mm3, yyjson_obj_get(inter_sec, "query_media_sequence"), 768u);

    std::printf("    query_multimodal (Text + 2 Images + Audio) cos_sim = %.6f [PASS]\n", cs_mm1);
    std::printf("    query_interleaved (Video + 2 Images)       cos_sim = %.6f [PASS]\n", cs_mm2);
    std::printf("    query_media_sequence (<|image|><|audio|>.. cos_sim = %.6f [PASS]\n", cs_mm3);
    TEST_ASSERT(cs_mm1 > 0.9999f && cs_mm2 > 0.9999f && cs_mm3 > 0.9999f,
                "Interleaved multimodal embeddings must match PyTorch golden reference (>0.9999)");

    yyjson_doc_free(doc);
    std::cout << "  [PASS] Official Tutorial E2E & Layer-by-Layer 0-Drift Parity Suite passed!" << std::endl;
}

int main(int argc, char **argv) {
    luisa::string backend = (argc > 1) ? argv[1] : "dx";
    std::filesystem::path egm_path = (argc > 2) ? argv[2] : "model.mmegm";
    std::filesystem::path tok_path = (argc > 3) ? argv[3] : "tokenizer.json";
    std::filesystem::path data_dir = (argc > 4) ? argv[4] : "examples/embedding_mm/data";

    auto resolve = [](const std::filesystem::path &p) -> std::filesystem::path {
        if (std::filesystem::exists(p)) return p;
        for (const char *prefix : {"..", "../..", "../../.."}) {
            auto cand = std::filesystem::path(prefix) / p;
            if (std::filesystem::exists(cand)) return cand;
        }
        return p;
    };

    egm_path = resolve(egm_path);
    tok_path = resolve(tok_path);
    data_dir = resolve(data_dir);

    std::cout << "================================================================\n"
              << " LuisaCompute EmbeddingGemma-2 Automated Test & Parity Suite\n"
              << " Backend: " << backend << "\n"
              << "================================================================\n"
              << std::endl;

    // 1. Host unit tests
    test_tokenizer(tok_path);
    test_package(egm_path);
    test_feature_loader(data_dir);

    // 2. GPU unit & regression tests
    Context context{argv[0]};
    auto device = context.create_device(backend);

    test_kernels(device);
    test_selective_encoder_loading(device, egm_path, tok_path);
    test_official_golden_suite(device, egm_path, tok_path, data_dir);

    std::cout << "\n================================================================\n"
              << " ALL EMBEDDINGGEMMA-2 UNIT & GOLDEN PARITY TESTS PASSED!\n"
              << "================================================================"
              << std::endl;

    return 0;
}
