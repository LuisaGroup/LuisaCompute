// =============================================================================
// cnn_inference.cpp — New Tile DSL inference for TinyCNN
// =============================================================================
// Port of backup_old_tile/examples/tensor/cnn_inference.cpp. Loads the
// weights/input/reference probabilities exported by the PyTorch script
// (cnn_train.py -> cnn_input.bin), builds the im2col and bias-folded weight
// matrices on the host, then runs the five Tile DSL kernels (cnn_kernels.cpp)
// on a Luisa device and verifies that the probabilities match both the
// PyTorch reference and an independent host CPU reference (tol = 1e-3, the
// old driver's tolerance).
//
// Invoked through example_tensor's main() with the `--cnn` flag:
//   example_tensor <backend> --cnn [cnn_input.bin] [--bench]
//
// Host flow mirrors the old driver exactly: conv2/fc1/fc2 im2col matrices
// that depend on a previous layer's *device* output are rebuilt on the host
// from the downloaded device result between kernel dispatches.
// =============================================================================

#include "cnn_inference.h"
#include "cnn_kernels.h"
#include "tensor_kernels.h"

#include <luisa/core/binary_file_stream.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace tensor_example::cnn {

namespace {

// ---- exported model (cnn_train.py layout) -----------------------------------
struct ExportedModel {
    int B = 0;
    int num_classes = 0;
    luisa::vector<float> input;    // B*1*8*8
    luisa::vector<float> w1, b1;   // conv1
    luisa::vector<float> w2, b2;   // conv2
    luisa::vector<float> wfc1, bfc1;
    luisa::vector<float> wfc2, bfc2;
    luisa::vector<float> ref_probs;// B*num_classes
};

// Reads the little-endian binary produced by cnn_train.py.
bool read_export(const luisa::string &path, ExportedModel &m) {
    luisa::BinaryFileStream f(path);
    if (!f) { return false; }
    auto read_f32 = [&](luisa::vector<float> &v, size_t n) {
        v.resize(n);
        if (n == 0) { return; }
        f.read(luisa::span<std::byte>{reinterpret_cast<std::byte *>(v.data()), n * sizeof(float)});
    };
    auto read_i32 = [&]() {
        int v = 0;
        f.read(luisa::span<std::byte>{reinterpret_cast<std::byte *>(&v), sizeof(v)});
        return v;
    };
    char magic[9] = {};
    f.read(luisa::span<std::byte>{reinterpret_cast<std::byte *>(magic), 8});
    if (std::strncmp(magic, "LUISACNN", 8) != 0) { return false; }
    m.B = read_i32();
    m.num_classes = read_i32();
    read_f32(m.input, static_cast<size_t>(m.B) * 64);
    read_f32(m.w1, 36);
    read_f32(m.b1, 4);
    read_f32(m.w2, 288);
    read_f32(m.b2, 8);
    read_f32(m.wfc1, 32 * 128);
    read_f32(m.bfc1, 32);
    read_f32(m.wfc2, 4 * 32);
    read_f32(m.bfc2, 4);
    read_f32(m.ref_probs, static_cast<size_t>(m.B) * m.num_classes);
    return true;
}

// ---------------------------------------------------------------------------
// Host reference: the exact forward pass (conv via im2col dot products, relu,
// fc, row-wise softmax), used to sanity-check the device result independently
// of PyTorch. Host arrays use [b][co][h][w] layout.
// ---------------------------------------------------------------------------
luisa::vector<float> host_reference(const ExportedModel &m) {
    const int B = m.B;
    luisa::vector<float> y1(B * 4 * 6 * 6);
    luisa::vector<float> y2(B * 8 * 4 * 4);
    luisa::vector<float> f1(B * 32);
    luisa::vector<float> logits(B * 4);
    luisa::vector<float> probs(B * 4);

    for (int b = 0; b < B; ++b) {
        for (int co = 0; co < 4; ++co) {
            for (int oh = 0; oh < 6; ++oh) {
                for (int ow = 0; ow < 6; ++ow) {
                    float s = m.b1[co];
                    for (int kh = 0; kh < 3; ++kh) {
                        for (int kw = 0; kw < 3; ++kw) {
                            s += m.input[b * 64 + (oh + kh) * 8 + (ow + kw)] *
                                 m.w1[co * 9 + kh * 3 + kw];
                        }
                    }
                    y1[((b * 4 + co) * 6 + oh) * 6 + ow] = s > 0.0f ? s : 0.0f;
                }
            }
        }
    }
    for (int b = 0; b < B; ++b) {
        for (int co = 0; co < 8; ++co) {
            for (int oh = 0; oh < 4; ++oh) {
                for (int ow = 0; ow < 4; ++ow) {
                    float s = m.b2[co];
                    for (int ci = 0; ci < 4; ++ci) {
                        for (int kh = 0; kh < 3; ++kh) {
                            for (int kw = 0; kw < 3; ++kw) {
                                s += y1[((b * 4 + ci) * 6 + (oh + kh)) * 6 + (ow + kw)] *
                                     m.w2[((co * 4 + ci) * 9 + kh * 3) + kw];
                            }
                        }
                    }
                    y2[((b * 8 + co) * 4 + oh) * 4 + ow] = s > 0.0f ? s : 0.0f;
                }
            }
        }
    }
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < 32; ++h) {
            float s = m.bfc1[h];
            for (int i = 0; i < 128; ++i) { s += y2[b * 128 + i] * m.wfc1[h * 128 + i]; }
            f1[b * 32 + h] = s > 0.0f ? s : 0.0f;
        }
    }
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < 4; ++c) {
            float s = m.bfc2[c];
            for (int h = 0; h < 32; ++h) { s += f1[b * 32 + h] * m.wfc2[c * 32 + h]; }
            logits[b * 4 + c] = s;
        }
    }
    for (int b = 0; b < B; ++b) {
        float mx = logits[b * 4];
        for (int c = 1; c < 4; ++c) { mx = mx > logits[b * 4 + c] ? mx : logits[b * 4 + c]; }
        float sum = 0.0f;
        for (int c = 0; c < 4; ++c) { sum += std::exp(logits[b * 4 + c] - mx); }
        for (int c = 0; c < 4; ++c) { probs[b * 4 + c] = std::exp(logits[b * 4 + c] - mx) / sum; }
    }
    return probs;
}

// ---------------------------------------------------------------------------
// Host building of the weight matrices / base im2col columns. The per-layer
// im2col data rows that depend on a previous layer's *device* output (col2,
// colfc1, colfc2) are filled by the caller after each kernel runs.
// ---------------------------------------------------------------------------
struct HostMats {
    luisa::vector<float> w1m;   // [4, 10]
    luisa::vector<float> col1;  // [10, B*36]
    luisa::vector<float> w2m;   // [8, 37]
    luisa::vector<float> wfc1t; // [129, 32]
    luisa::vector<float> wfc2t; // [33, 4]
    luisa::vector<float> colfc1;// [B, 129]  (bias row pre-set)
    luisa::vector<float> colfc2;// [B, 33]   (bias row pre-set)
};

HostMats build_host_mats(const ExportedModel &m) {
    const int B = m.B;
    constexpr int KK1 = 10, P1 = 36;
    constexpr int KK2 = 37;
    constexpr int KFC1 = 129, KFC2 = 33;
    HostMats h;

    // conv1 weight [4,10] = w1[co,kh*3+kw] + bias
    h.w1m.assign(4 * KK1, 0.0f);
    for (int co = 0; co < 4; ++co) {
        for (int k = 0; k < 9; ++k) { h.w1m[co * KK1 + k] = m.w1[co * 9 + k]; }
        h.w1m[co * KK1 + 9] = m.b1[co];
    }
    // conv1 im2col [10, B*36]
    h.col1.assign(KK1 * B * P1, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int oh = 0; oh < 6; ++oh) {
            for (int ow = 0; ow < 6; ++ow) {
                const int p = b * P1 + oh * 6 + ow;
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        h.col1[(kh * 3 + kw) * B * P1 + p] =
                            m.input[b * 64 + (oh + kh) * 8 + (ow + kw)];
                    }
                }
                h.col1[9 * B * P1 + p] = 1.0f;// bias row
            }
        }
    }
    // conv2 weight [8,37]
    h.w2m.assign(8 * KK2, 0.0f);
    for (int co = 0; co < 8; ++co) {
        for (int ci = 0; ci < 4; ++ci) {
            for (int k = 0; k < 9; ++k) { h.w2m[co * KK2 + ci * 9 + k] = m.w2[((co * 4 + ci) * 9) + k]; }
        }
        h.w2m[co * KK2 + 36] = m.b2[co];
    }
    // fc1 weight [129,32]
    h.wfc1t.assign(KFC1 * 32, 0.0f);
    for (int i = 0; i < 128; ++i) {
        for (int h2 = 0; h2 < 32; ++h2) { h.wfc1t[i * 32 + h2] = m.wfc1[h2 * 128 + i]; }
    }
    for (int h2 = 0; h2 < 32; ++h2) { h.wfc1t[128 * 32 + h2] = m.bfc1[h2]; }
    // fc2 weight [33,4]
    h.wfc2t.assign(KFC2 * 4, 0.0f);
    for (int h2 = 0; h2 < 32; ++h2) {
        for (int c = 0; c < 4; ++c) { h.wfc2t[h2 * 4 + c] = m.wfc2[c * 32 + h2]; }
    }
    for (int c = 0; c < 4; ++c) { h.wfc2t[32 * 4 + c] = m.bfc2[c]; }
    // fc1 / fc2 im2col base: bias row pre-set to 1.0, data rows filled later
    h.colfc1.assign(B * KFC1, 0.0f);
    h.colfc2.assign(B * KFC2, 0.0f);
    for (int b = 0; b < B; ++b) {
        h.colfc1[b * KFC1 + 128] = 1.0f;
        h.colfc2[b * KFC2 + 32] = 1.0f;
    }
    return h;
}

}// namespace

// =============================================================================
// run_cnn_inference — driver (invoked by example_tensor's main with --cnn)
// =============================================================================
int run_cnn_inference(int argc, char *argv[]) {
    using namespace luisa;
    using namespace luisa::compute;

    // Collect the backend name and the optional .bin path from the positional
    // arguments (skipping flags such as --cnn / --bench), so the same
    // executable serves both the kernel suite and the CNN inference modes.
    auto has_flag = [&](luisa::string_view f) {
        for (auto i = 1; i < argc; ++i) {
            if (argv != nullptr && argv[i] != nullptr && luisa::string_view{argv[i]} == f) { return true; }
        }
        return false;
    };
    luisa::string_view backend{};
    luisa::string_view bin_path{"cnn_input.bin"};
    int positional = 0;
    for (auto i = 1; i < argc; ++i) {
        if (argv != nullptr && argv[i] != nullptr && !luisa::string_view{argv[i]}.starts_with("--")) {
            if (positional == 0) { backend = argv[i]; } else if (positional == 1) { bin_path = argv[i]; }
            ++positional;
        }
    }
    auto do_bench = has_flag("--bench");

    if (backend.empty()) {
        LUISA_INFO("Usage: {} <backend> --cnn [cnn_input.bin] [--bench]", argv[0]);
        return 1;
    }

      ExportedModel model;
      // Try the path as given first, then repo-layout fallbacks so the demo
      // works from the build tree or the source tree.
      luisa::string resolved_path{bin_path};
      auto try_read = [&]() noexcept {
          return read_export(resolved_path, model);
      };
      if (!try_read()) {
          resolved_path = luisa::format("examples/tensor/{}", bin_path);
          if (!try_read()) {
              resolved_path = luisa::format("../../examples/tensor/{}", bin_path);
              if (!try_read()) {
                  record("cnn_read_export", false,
                         luisa::format("failed to read '{}' (run examples/tensor/cnn_train.py first)", bin_path));
                  return 1;
              }
          }
      }
    const int B = model.B;
    if (B != cnn::CB || model.num_classes != cnn::CNC) {
        record("cnn_dims", false,
               luisa::format("exported model dims (B={}, classes={}) do not match the tile kernels "
                             "(B={}, classes={})",
                             B, model.num_classes, static_cast<int>(cnn::CB), static_cast<int>(cnn::CNC)));
        return 1;
    }
    record("cnn_read_export", true,
           luisa::format("loaded '{}': B={}, classes={}", resolved_path, B, model.num_classes));

    auto mats = build_host_mats(model);

    // ---- device -------------------------------------------------------------
    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();

    constexpr uint32_t P1 = 36, P2 = 16;
    constexpr uint32_t KK1 = 10, KK2 = 37, KFC1 = 129, KFC2 = 33;
    constexpr uint32_t NC = 4, F1 = 32, C2 = 8;

    auto make_buf = [&](auto &&host_vec, size_t n) {
        auto buf = device.create_buffer<float>(static_cast<uint32_t>(n));
        stream << buf.copy_from(luisa::span{host_vec.data(), host_vec.size()}) << synchronize();
        return buf;
    };

    auto bufW1 = make_buf(mats.w1m, 4u * KK1);
    auto bufCol1 = make_buf(mats.col1, KK1 * static_cast<uint32_t>(B) * P1);
    auto bufY1 = device.create_buffer<float>(4u * static_cast<uint32_t>(B) * P1);
    auto bufW2 = make_buf(mats.w2m, C2 * KK2);
    luisa::vector<float> zero_col2(KK2 * static_cast<uint32_t>(B) * P2, 0.0f);
    auto bufCol2 = make_buf(zero_col2, zero_col2.size());
    auto bufY2 = device.create_buffer<float>(C2 * static_cast<uint32_t>(B) * P2);
    auto bufColFc1 = make_buf(mats.colfc1, static_cast<uint32_t>(B) * KFC1);
    auto bufWfc1T = make_buf(mats.wfc1t, KFC1 * F1);
    auto bufF1 = device.create_buffer<float>(static_cast<uint32_t>(B) * F1);
    auto bufColFc2 = make_buf(mats.colfc2, static_cast<uint32_t>(B) * KFC2);
    auto bufWfc2T = make_buf(mats.wfc2t, KFC2 * NC);
    auto bufLogits = device.create_buffer<float>(static_cast<uint32_t>(B) * NC);
    auto bufProbs = device.create_buffer<float>(static_cast<uint32_t>(B) * NC);

    // ---- compile the five tile kernels --------------------------------------
    luisa::Clock c_compile;
    auto sh_conv1 = compile_tile(device, cnn::make_conv1_relu_kernel(), "cnn_conv1_relu");
    auto sh_conv2 = compile_tile(device, cnn::make_conv2_relu_kernel(), "cnn_conv2_relu");
    auto sh_fc1 = compile_tile(device, cnn::make_fc1_relu_kernel(), "cnn_fc1_relu");
    auto sh_fc2 = compile_tile(device, cnn::make_fc2_kernel(), "cnn_fc2");
    auto sh_softmax = compile_tile(device, cnn::make_cnn_softmax_kernel(), "cnn_softmax");
    double compile_ms = c_compile.toc();
    if (!sh_conv1 || !sh_conv2 || !sh_fc1 || !sh_fc2 || !sh_softmax) {
        record("cnn_compile", false, "one or more kernels failed to compile (see log above)");
        return 1;
    }
    record("cnn_compile", true, luisa::format("5 kernels in {:.3f} ms", compile_ms));

    // ---- run the forward pass -----------------------------------------------
    luisa::Clock clock;
    // conv1
    stream << sh_conv1(bufW1, bufCol1, bufY1).dispatch() << synchronize();
    luisa::vector<float> y1(4u * static_cast<uint32_t>(B) * P1);// [C1][B*36]
    stream << bufY1.copy_to(luisa::span{y1}) << synchronize();

    // conv2 im2col from the device y1 ([co][b*36+h*6+w] layout)
    luisa::vector<float> col2(KK2 * static_cast<uint32_t>(B) * P2, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int oh = 0; oh < 4; ++oh) {
            for (int ow = 0; ow < 4; ++ow) {
                const int p = b * static_cast<int>(P2) + oh * 4 + ow;
                for (int ci = 0; ci < 4; ++ci) {
                    for (int kh = 0; kh < 3; ++kh) {
                        for (int kw = 0; kw < 3; ++kw) {
                            col2[(ci * 9 + kh * 3 + kw) * B * P2 + p] =
                                y1[ci * B * 36 + b * 36 + (oh + kh) * 6 + (ow + kw)];
                        }
                    }
                }
                col2[36 * B * P2 + p] = 1.0f;
            }
        }
    }
    stream << bufCol2.copy_from(luisa::span{col2}) << synchronize();
    stream << sh_conv2(bufW2, bufCol2, bufY2).dispatch() << synchronize();
    luisa::vector<float> y2(C2 * static_cast<uint32_t>(B) * P2);// [C2][B*16]
    stream << bufY2.copy_to(luisa::span{y2}) << synchronize();

    // fc1 im2col from the device y2 ([co][b*16+h*4+w] layout)
    luisa::vector<float> colfc1(static_cast<uint32_t>(B) * KFC1, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int co = 0; co < 8; ++co) {
            for (int h = 0; h < 4; ++h) {
                for (int w = 0; w < 4; ++w) {
                    colfc1[b * KFC1 + co * 16 + h * 4 + w] =
                        y2[co * B * P2 + b * 16 + h * 4 + w];
                }
            }
        }
        colfc1[b * KFC1 + 128] = 1.0f;
    }
    stream << bufColFc1.copy_from(luisa::span{colfc1}) << synchronize();
    stream << sh_fc1(bufColFc1, bufWfc1T, bufF1).dispatch() << synchronize();
    luisa::vector<float> f1(static_cast<uint32_t>(B) * F1);// [B][32]
    stream << bufF1.copy_to(luisa::span{f1}) << synchronize();

    // fc2 im2col from the device f1 ([b][h] layout)
    luisa::vector<float> colfc2(static_cast<uint32_t>(B) * KFC2, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < 32; ++h) { colfc2[b * KFC2 + h] = f1[b * 32 + h]; }
        colfc2[b * KFC2 + 32] = 1.0f;
    }
    stream << bufColFc2.copy_from(luisa::span{colfc2}) << synchronize();
    stream << sh_fc2(bufColFc2, bufWfc2T, bufLogits).dispatch() << synchronize();

    // softmax
    stream << sh_softmax(bufLogits, bufProbs).dispatch() << synchronize();
    double gpu_ms = clock.toc();

    luisa::vector<float> probs(static_cast<uint32_t>(B) * NC);
    stream << bufProbs.copy_to(luisa::span{probs}) << synchronize();

    // ---- verify against PyTorch reference and host reference ----------------
    auto host_probs = host_reference(model);

    float err_pt = 0.0f, err_host = 0.0f;
    for (int i = 0; i < B * NC; ++i) {
        err_pt = std::max(err_pt, std::fabs(probs[i] - model.ref_probs[i]));
        err_host = std::max(err_host, std::fabs(probs[i] - host_probs[i]));
    }
    LUISA_INFO("TinyCNN tile inference on '{}' completed in {:.2f} ms.", backend, gpu_ms);
    for (int b = 0; b < B; ++b) {
        LUISA_INFO("  sample {}: device=[{:.6f} {:.6f} {:.6f} {:.6f}] "
                   "pytorch=[{:.6f} {:.6f} {:.6f} {:.6f}]",
                   b, probs[b * 4 + 0], probs[b * 4 + 1], probs[b * 4 + 2], probs[b * 4 + 3],
                   model.ref_probs[b * 4 + 0], model.ref_probs[b * 4 + 1],
                   model.ref_probs[b * 4 + 2], model.ref_probs[b * 4 + 3]);
    }
    LUISA_INFO("  max |device - pytorch| = {:.3e}", err_pt);
    LUISA_INFO("  max |device - host  | = {:.3e}", err_host);

    // Old driver's tolerance: 1e-3 on both references.
    check("cnn_probs_vs_pytorch", err_pt, 1e-3);
    check("cnn_probs_vs_host", err_host, 1e-3);

    // ---- comprehensive benchmark -------------------------------------------
    if (do_bench) {
        constexpr int R = 4000;// dispatch iterations per kernel
        LUISA_INFO("=== benchmark ({} backend) ===", backend);
        LUISA_INFO("  backend compile      : {:.3f} ms (5 kernels)", compile_ms);
        auto bench = [&](luisa::string_view name, auto &&dispatch_fn) {
            dispatch_fn();
            stream << synchronize();
            luisa::Clock c;
            for (int i = 0; i < R; ++i) { dispatch_fn(); }
            stream << synchronize();
            double us = c.toc() * 1000.0 / R;
            LUISA_INFO("  dispatch {:>8s} : {:8.3f} us/run", name, us);
            return us;
        };
        double d_conv1 = bench("conv1", [&] { stream << sh_conv1(bufW1, bufCol1, bufY1).dispatch(); });
        double d_conv2 = bench("conv2", [&] { stream << sh_conv2(bufW2, bufCol2, bufY2).dispatch(); });
        double d_fc1 = bench("fc1", [&] { stream << sh_fc1(bufColFc1, bufWfc1T, bufF1).dispatch(); });
        double d_fc2 = bench("fc2", [&] { stream << sh_fc2(bufColFc2, bufWfc2T, bufLogits).dispatch(); });
        double d_sm = bench("softmax", [&] { stream << sh_softmax(bufLogits, bufProbs).dispatch(); });
        LUISA_INFO("  dispatch total       : {:8.3f} us/run", d_conv1 + d_conv2 + d_fc1 + d_fc2 + d_sm);
    }
    return failure_count() == 0 ? 0 : 1;
}

}// namespace tensor_example::cnn
