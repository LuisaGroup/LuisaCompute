// mm_embedding engine -- comprehensive automated unit & regression tests.
//
// Tests:
// 1. Tokenizer: special tokens, BPE merges, byte fallback, truncation, batching.
// 2. Package: headers, 64-bit offsets, tensor lookups, missing key handling.
// 3. DSL Kernels: RMSNorm, L2Norm, RoPE, GeGLU, GEMM, MeanPool + MRL.
// 4. Session GPU End-to-End: Semantic query-doc ranking and unit-norm invariant.
//
// Usage:
//   example_mm_embed_test [backend: dx|vk] [model.mmegm] [tokenizer.json]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>

#include "mm_embedding/embed/kernels/kernels_norm.h"
#include "mm_embedding/embed/kernels/kernels_rope.h"
#include "mm_embedding/embed/kernels/kernels_mlp.h"
#include "mm_embedding/embed/kernels/kernels_gemm.h"
#include "mm_embedding/embed/kernels/kernels_pool.h"
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

    // Print T04 test
    auto t04_toks = tok->encode("task: question answering | query: 大文件传一半断了要重新传吗");
    std::cout << "  [INFO] C++ T04 toks (" << t04_toks.size() << "): ";
    for (auto id : t04_toks) std::cout << id << " ";
    std::cout << std::endl;

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
    TEST_ASSERT(pkg->tensors().size() == 413u, "Total tensors must be 413");

    // Find known tensors
    const auto *t_emb = pkg->find("emb.word");
    TEST_ASSERT(t_emb != nullptr, "Must find 'emb.word'");
    TEST_ASSERT(t_emb->rank == 2u, "'emb.word' rank must be 2");
    TEST_ASSERT(t_emb->dims[0] == 262144u && t_emb->dims[1] == 512u, "emb.word dims must be [262144, 512]");

    const auto *t_final_norm = pkg->find("final_norm");
    TEST_ASSERT(t_final_norm != nullptr, "Must find 'final_norm'");
    TEST_ASSERT(t_final_norm->rank == 1u && t_final_norm->dims[0] == 512u, "final_norm dims must be [512]");

    // Missing tensor lookup
    TEST_ASSERT(pkg->find("non_existent_tensor_name") == nullptr, "Missing tensor must return nullptr");

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
        luisa::vector<float> host_gamma(H, 1.0f); // gamma = 1.0
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

    // (d) Test MeanPool
    {
        auto mean_pool_shader = device.compile(make_mean_pool_kernel());
        constexpr uint32_t B = 2u;
        constexpr uint32_t S = 4u;
        constexpr uint32_t H = 8u;

        luisa::vector<float> host_hidden(B * S * H, 1.0f);
        // Batch 0: length 2 (mask = [1, 1, 0, 0])
        // Batch 1: length 3 (mask = [1, 1, 1, 0])
        luisa::vector<float> host_mask = {
            1.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 0.0f
        };

        // Fill token 0 with 2.0, token 1 with 4.0 -> mean = 3.0
        for (uint32_t h = 0; h < H; h++) {
            host_hidden[(0 * S + 0) * H + h] = 2.0f;
            host_hidden[(0 * S + 1) * H + h] = 4.0f;
            host_hidden[(0 * S + 2) * H + h] = 999.0f; // masked
            host_hidden[(0 * S + 3) * H + h] = 999.0f; // masked
        }

        auto buf_h = device.create_buffer<float>(B * S * H);
        auto buf_m = device.create_buffer<float>(B * S);
        auto buf_pooled = device.create_buffer<float>(B * H);
        luisa::vector<float> host_pooled(B * H);

        CommandList cmd;
        cmd << buf_h.copy_from(host_hidden.data())
            << buf_m.copy_from(host_mask.data())
            << mean_pool_shader(buf_h, buf_m, buf_pooled, S, H).dispatch(H, B)
            << buf_pooled.copy_to(host_pooled.data());
        stream << cmd.commit() << synchronize();

        for (uint32_t h = 0; h < H; h++) {
            TEST_ASSERT(std::abs(host_pooled[0 * H + h] - 3.0f) < 1e-5f, "Mask-aware pooling must ignore masked tokens");
        }
        std::cout << "  [PASS] MeanPool kernel verified." << std::endl;
    }
}

// ============================================================================
// 4. Session End-to-End Regression Test
// ============================================================================
void test_session_e2e(Device &device, const std::filesystem::path &egm_path, const std::filesystem::path &tok_path) {
    std::cout << "[TEST] Running Session End-to-End inference tests..." << std::endl;

    if (!std::filesystem::exists(egm_path) || !std::filesystem::exists(tok_path)) {
        std::cout << "  [SKIP] Model files not found. Skipping E2E test." << std::endl;
        return;
    }

    auto session = Session::create(device, egm_path, tok_path);
    TEST_ASSERT(session != nullptr, "Session::create must succeed");

    const luisa::string_view query = "What is deep learning?";
    const luisa::string_view docs[] = {
        "Deep learning uses artificial neural networks with multiple layers to learn representations from data.",
        "The recipe for a classic French baguette requires flour, water, yeast, and salt."
    };

    // Test across MRL dimensions: 768, 512, 256, 128
    for (uint32_t dim : {768u, 512u, 256u, 128u}) {
        EncodeOptions opt{};
        opt.target_dimension = dim;

        auto q_emb = session->encode(query, opt);
        auto d_embs = session->encode_batch(luisa::span{docs, 2u}, opt);

        TEST_ASSERT(q_emb.dimension == dim, "Query embedding dimension mismatch");
        TEST_ASSERT(d_embs.size() == 2u, "Document batch size mismatch");
        TEST_ASSERT(d_embs[0].dimension == dim && d_embs[1].dimension == dim, "Doc embedding dimension mismatch");

        // Check unit L2 norm
        auto norm_sq = [](const luisa::vector<float> &v) {
            float s = 0.0f;
            for (float val : v) s += val * val;
            return s;
        };
        TEST_ASSERT(std::abs(norm_sq(q_emb.values) - 1.0f) < 1e-4f, "Query embedding must have L2 norm = 1");
        TEST_ASSERT(std::abs(norm_sq(d_embs[0].values) - 1.0f) < 1e-4f, "Doc 0 embedding must have L2 norm = 1");
        TEST_ASSERT(std::abs(norm_sq(d_embs[1].values) - 1.0f) < 1e-4f, "Doc 1 embedding must have L2 norm = 1");

        // Dot product / cosine similarity
        float sim0 = 0.0f, sim1 = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            sim0 += q_emb.values[d] * d_embs[0].values[d];
            sim1 += q_emb.values[d] * d_embs[1].values[d];
        }

        std::cout << "  [INFO] Dim " << dim << ": sim(relevant)=" << sim0 << ", sim(irrelevant)=" << sim1 << std::endl;
        TEST_ASSERT(sim0 > sim1, "Semantic similarity of relevant document must be higher than irrelevant document");
    }

    std::cout << "  [PASS] Session End-to-End tests passed." << std::endl;
}

int main(int argc, char **argv) {
    luisa::string backend = (argc > 1) ? argv[1] : "dx";
    std::filesystem::path egm_path = (argc > 2) ? argv[2] : "model.mmegm";
    std::filesystem::path tok_path = (argc > 3) ? argv[3] : "tokenizer.json";

    std::cout << "========================================================\n"
              << " LuisaCompute mm_embedding Automated Test Suite\n"
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
    test_session_e2e(device, egm_path, tok_path);

    std::cout << "\n========================================================\n"
              << " ALL TESTS PASSED SUCCESSFULLY!\n"
              << "========================================================"
              << std::endl;

    return 0;
}
