// M3 end-to-end gate for the bge-m3 embedding engine (embedding_model_plan.md
// §6 M3): Session-level encode of a pinned corpus subset through the full
// executor graph (embed_lookup -> 24 x {GEMM/attention/GEMM/LN} -> slice_cls
// -> l2_normalize), compared against golden_dense.bin (worst cosine >= the
// package gate on ALL sentences), PLUS the layer-trace oracle against
// golden_layers.bin (per-layer max-abs-diff gate for corpus sentences 0 and 3
// -- the D4 bisect oracle proving per-layer, not just final, correctness).
//
// Package-format gates (probed from the BGM3 header, (version, precision) PAIR
// validated against the loader's accepted table -- never "non-v2 means fp32"):
//   v1 fp32            : cosine >= 0.9999, per-layer <= 1e-3  (unchanged)
//   v2 bf16 | fp16     : cosine >= 0.999,  per-layer <= 0.5   (unchanged)
// v3 int8 (symmetric): cosine >= 0.98, per-layer <= 2.0 DIAGNOSTIC
// regression gate (NOT a model-quality gate: per-tensor int8 trace drift can
// approach O(1)); for v3 the DENSE cosine gate is the authoritative model-quality
// gate and the 2.0 layer bound only catches gross upload/layout corruption
// (measured v3 trace worst: 1.173 / 0.7301 on the two traced sentences).
//
// Attention: naive trio only (all corpus sentences fit S <= 512 after the
// 512-token truncation); the flash path is never touched here (B2/M4).
//
// Run (device test; backend = argv[1], package = argv[2] optional):
//   xmake run -o build/embed_b1 test_embed_executor dx
//   xmake run -o build/embed_b1 test_embed_executor vk
//   xmake run -o build/embed_b1 test_embed_executor fallback
//   (optionally: ... dx D:/compute/bge-m3/bge-m3/model.bgm)
#include "ut/ut.hpp"
#include "test_device.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>

#include <luisa/core/clock.h>
#include <luisa/core/fiber.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/stream.h>

#include "embedding/embed/session.h"
#include "embedding/embed/package.h"
#include "embedding/embed/tokenizer.h"
#include "embedding/embed/kernels/kernels_embed.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::embed;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

luisa::vector<char> read_file(const std::filesystem::path &p) {
    if (std::FILE *f = std::fopen(p.string().c_str(), "rb"); f != nullptr) {
        luisa::vector<char> data;
        char buf[65536];
        size_t got = 0u;
        while ((got = std::fread(buf, 1u, sizeof(buf), f)) > 0u) {
            data.insert(data.end(), buf, buf + static_cast<ptrdiff_t>(got));
        }
        std::fclose(f);
        return data;
    }
    LUISA_ERROR("cannot open test data file \"{}\"", p.string());
}

luisa::string unescape_corpus_line(luisa::string_view line) {
    luisa::string out;
    out.reserve(line.size());
    size_t i = 0u;
    while (i < line.size()) {
        if (line[i] == '\\' && i + 1u < line.size()) {
            const auto c = line[i + 1u];
            if (c == 'n') {
                out.push_back('\n');
                i += 2u;
                continue;
            }
            if (c == 'r') {
                out.push_back('\r');
                i += 2u;
                continue;
            }
            if (c == '\\') {
                out.push_back('\\');
                i += 2u;
                continue;
            }
        }
        out.push_back(line[i]);
        i++;
    }
    return out;
}

luisa::vector<luisa::string> load_corpus(const std::filesystem::path &dir) {
    auto raw = read_file(dir / "corpus.txt");
    luisa::vector<luisa::string> lines;
    size_t start = 0u;
    auto emit_line = [&](size_t end) {
        luisa::string_view line{raw.data() + start, end - start};
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1u);// CRLF tolerance
        }
        lines.push_back(unescape_corpus_line(line));
        start = end + 1u;
    };
    for (size_t i = 0u; i < raw.size(); i++) {
        if (raw[i] == '\n') {
            emit_line(i);
        }
    }
    if (start < raw.size()) {
        emit_line(raw.size());
    }
    return lines;
}

luisa::vector<float> read_f32_file(const std::filesystem::path &p) {
    auto raw = read_file(p);
    luisa::vector<float> out(raw.size() / 4u);
    std::memcpy(out.data(), raw.data(), out.size() * 4u);
    return out;
}

// cosine of two fp32 vectors (goldens are unit-norm; full formula anyway)
float cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) {
        return -1.0f;
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

constexpr size_t kDenseDim = 1024u;
constexpr uint32_t kTraceSentences[] = {0u, 3u};
constexpr size_t kTraceSentenceRows[] = {15u, 17u};// golden_layers.bin rows (meta)
constexpr size_t kTraceSentenceBase[] = {0u, 1536000u};

// O-G / quant gates. fp32 v1 and half v2 numbers are the historical ones and
// must not move; int8 v3 gets its own, honest tolerances.
constexpr float kFp32GoldenGate = 0.9999f;
constexpr float kHalfGoldenGate = 0.999f;
constexpr float kInt8GoldenGate = 0.98f;
constexpr float kFp32LayerGate = 1e-3f;
constexpr float kHalfLayerGate = 0.5f;
// v3 per-layer gate: DIAGNOSTIC/regression only (gross layout/upload
// corruption), not a model-quality gate -- see the trace section.
constexpr float kInt8LayerDiagnosticGate = 2.0f;

// BGM3 package header probe. Layout (examples/embedding/embed/package.cpp is the
// spec): magic 'B','G','M','3' | u32 version | u32 precision | u32 tensor_count.
struct PackageHeader {
    uint32_t version = 0u;
    uint32_t precision = 0u;
    bool magic_ok = false;
};

PackageHeader probe_package_header(const std::filesystem::path &bgm) {
    PackageHeader h;
    if (auto *f = std::fopen(bgm.string().c_str(), "rb"); f != nullptr) {
        uint8_t raw[16] = {};
        const auto got = std::fread(raw, 1u, 16u, f);
        std::fclose(f);
        if (got == 16u && raw[0] == 'B' && raw[1] == 'G' && raw[2] == 'M' && raw[3] == '3') {
            h.magic_ok = true;
            h.version = static_cast<uint32_t>(raw[4]) | (static_cast<uint32_t>(raw[5]) << 8u) |
                        (static_cast<uint32_t>(raw[6]) << 16u) | (static_cast<uint32_t>(raw[7]) << 24u);
            h.precision = static_cast<uint32_t>(raw[8]) | (static_cast<uint32_t>(raw[9]) << 8u) |
                          (static_cast<uint32_t>(raw[10]) << 16u) | (static_cast<uint32_t>(raw[11]) << 24u);
        }
    }
    return h;
}

enum class PackageKind { fp32, half, int8 };

// Validate the (version, precision) PAIR against the loader's accepted table
// (v1 + 0 = fp32, v2 + 1|2 = bf16|fp16, v3 + 3 = per-tensor symmetric int8).
// An unimplemented pair is rejected here instead of being silently mapped onto a
// known gate family -- in particular "not v2" must never mean either fp32 or
// int8.
PackageKind classify_package(const std::filesystem::path &bgm) {
    const auto h = probe_package_header(bgm);
    if (!h.magic_ok) {
        // No readable BGM3 header: Package::load raises the authoritative error
        // right below (Session::create), so keep the fp32 gate here rather than
        // guessing a low-precision one.
        LUISA_WARNING("package \"{}\" has no readable BGM3 header; keeping the fp32 gates until the loader reports the real error",
                      bgm.string());
        return PackageKind::fp32;
    }
    if (h.version == 1u && h.precision == 0u) {
        return PackageKind::fp32;
    }
    if (h.version == 2u && (h.precision == 1u || h.precision == 2u)) {
        return PackageKind::half;
    }
    if (h.version == 3u && h.precision == 3u) {
        return PackageKind::int8;
    }
    LUISA_ERROR("unsupported BGM3 version {} precision {} in \"{}\": v1 is fp32 only (precision 0), "
                "v2 is bf16|fp16 only (precision 1|2), v3 is int8 only (precision 3)",
                h.version, h.precision, bgm.string());
}

}// namespace

int main(int argc, char *argv[]) {
    luisa::fiber::scheduler fiber_scheduler;
    auto [context, device] = luisa::test::create_device(argc, argv);
    luisa::test::log_test_backend(argv[1], device);
    const std::filesystem::path data_dir{LUISA_EMBED_TEST_DATA_DIR};
    const std::filesystem::path bgm = (argc > 2 ? std::filesystem::path{argv[2]} : std::filesystem::path{"D:/compute/bge-m3/bge-m3/model.bgm"});
    // O-G / quant precision gating (same policy as test_embed_e2e): the golden
    // cosine gate follows the probed + pair-validated package format.
    const auto pkg_kind = classify_package(bgm);
    const bool pkg_half = pkg_kind == PackageKind::half;
    const bool pkg_int8 = pkg_kind == PackageKind::int8;
    const float golden_gate = pkg_int8 ? kInt8GoldenGate : (pkg_half ? kHalfGoldenGate : kFp32GoldenGate);
    LUISA_INFO("test_embed_executor: package \"{}\" -> {} package, dense/worst cosine gate >= {:.4f}",
               bgm.string(),
               pkg_int8 ? "int8 (v3, per-tensor symmetric)" : (pkg_half ? "half-precision (v2)" : "fp32 (v1)"),
               golden_gate);
    const auto corpus = load_corpus(data_dir);
    const auto gold_dense = read_f32_file(data_dir / "golden_dense.bin");
    const auto gold_layers = read_f32_file(data_dir / "golden_layers.bin");
    expect(gold_dense.size() == 184u * kDenseDim) << "golden_dense.bin shape";
    expect(corpus.size() >= 184u) << "corpus size";

    // ---- Session (M3 uses the single-worker sync path; the naive attention
    // path covers every sentence: max corpus length 512 after truncation).
    luisa::compute::Clock create_clock;
    auto session = Session::create(device, bgm, data_dir / "tokenizer.json", 1u);
    LUISA_INFO("Session created in {:.1f} ms (package load + weight upload + kernel compile)",
               create_clock.toc());

    const bool trace_only = std::getenv("BGEM3_TRACE_ONLY") != nullptr;// dev iteration helper
    // ---- dense parity: corpus sentences 0..15 against golden_dense.bin ----
    float worst_cosine = 1.0f;
    uint32_t worst_idx = 0u;
    luisa::compute::Clock enc_clock;
    for (uint32_t i = 0; !trace_only && i < 16u; i++) {
        EncodeOptions opt;
        opt.max_seq_len = 512u;
        auto emb = session->encode(corpus[i], opt);
        expect(emb.dense.size() == kDenseDim) << "dense dim for sentence " << i;
        const auto c = cosine(emb.dense.data(), gold_dense.data() + static_cast<size_t>(i) * kDenseDim, kDenseDim);
        worst_cosine = std::min(worst_cosine, c);
        worst_idx = (c == worst_cosine ? i : worst_idx);
        LUISA_INFO("sentence {:3d}: cosine = {:.6f}", i, c);
        expect(c >= golden_gate) << "cosine >= package gate for sentence " << i;
    }
    LUISA_INFO("M3 dense gate: 16 sentences, worst cosine = {:.6f} (sentence {}), gate >= {:.4f} ({})",
               worst_cosine, worst_idx, golden_gate,
               pkg_int8 ? "int8 v3 gate" : (pkg_half ? "half v2 gate" : "fp32 v1 gate"));
    expect(worst_cosine >= golden_gate) << "worst cosine gate";

    // ---- full-corpus timing (report evidence; every sentence gated too when
    // the run is not on the slow fallback reference).
    if (!trace_only) {
        const bool full_gate = std::strcmp(argv[1], "fallback") != 0;
        luisa::compute::Clock corpus_clock;
        float worst_all = 1.0f;
        for (size_t i = 0; i < corpus.size(); i++) {
            auto emb = session->encode(corpus[i]);
            if (full_gate) {
                const auto c = cosine(emb.dense.data(), gold_dense.data() + i * kDenseDim, kDenseDim);
                worst_all = std::min(worst_all, c);
            }
        }
        const auto total_ms = corpus_clock.toc();
        LUISA_INFO("full corpus ({} sentences) encode: {:.1f} ms total, {:.1f} ms/sentence, worst cosine {}",
                   corpus.size(), total_ms, total_ms / static_cast<double>(corpus.size()),
                   full_gate ? luisa::format("{:.6f}", worst_all) : luisa::string{"(not gated on fallback)"});
        if (full_gate) {
            expect(worst_all >= golden_gate) << "full-corpus worst cosine";
        }
    }

    // ---- diagnostic (BGEM3_DIAG=1): host-side reference embedding block
    // computed directly from the package tables, compared against the traced
    // slot 0 and against golden_layers.bin layer 0. Separates
    // ids/tokenization errors, weight-upload errors, trace-plumbing errors and
    // golden-index errors.
    const bool diag = std::getenv("BGEM3_DIAG") != nullptr && pkg_kind == PackageKind::fp32;
    if (std::getenv("BGEM3_DIAG") != nullptr && pkg_kind != PackageKind::fp32) {
        LUISA_INFO("BGEM3_DIAG skipped: the host-side embedding reference reader below is fp32-blob-only (v2 half packages expose Tensor::half_data, v3 int8 packages expose Tensor::int8_data + int8_scale); run it on a v1 fp32 package.");
    }
    luisa::vector<float> diag_ref;
    luisa::vector<uint32_t> diag_ids;
    if (diag) {
        auto pkg = Package::load(bgm);
        auto tok = Tokenizer::load(data_dir / "tokenizer.json");
        diag_ids = tok->encode(corpus[0], 512u);
        const auto N = static_cast<uint32_t>(diag_ids.size());
        const uint32_t Hh = 1024u;
        const auto *word = pkg->find("emb.word")->data;
        const auto *pos = pkg->find("emb.pos")->data;
        const auto *g = pkg->find("emb.ln.gamma")->data;
        const auto *bt = pkg->find("emb.ln.beta")->data;
        diag_ref.assign(static_cast<size_t>(N) * Hh, 0.0f);
        for (uint32_t s2 = 0; s2 < N; s2++) {
            // same math as kernel #1: word+pos, two-pass mean/var LN
            luisa::vector<float> row(Hh);
            double mean = 0.0;
            for (uint32_t d = 0; d < Hh; d++) {
                row[d] = word[static_cast<size_t>(diag_ids[s2]) * Hh + d] +
                         pos[static_cast<size_t>(s2 + 2u) * Hh + d];
                mean += row[d];
            }
            mean /= static_cast<double>(Hh);
            double var = 0.0;
            for (uint32_t d = 0; d < Hh; d++) {
                const double dv = static_cast<double>(row[d]) - mean;
                var += dv * dv;
            }
            var /= static_cast<double>(Hh);
            const double rstd = 1.0 / std::sqrt(var + 1e-5);
            for (uint32_t d = 0; d < Hh; d++) {
                diag_ref[static_cast<size_t>(s2) * Hh + d] =
                    static_cast<float>((static_cast<double>(row[d]) - mean) * rstd * g[d] + bt[d]);
            }
        }
        // ids match golden_tokens? print first few + length
        LUISA_INFO("DIAG tokens for sentence 0: n={}, ids[0..5]: {} {} {} {} {}", N,
                   diag_ids[0], diag_ids[1], diag_ids[2], diag_ids[3], diag_ids[4]);
        // DIAG v3: run embed_lookup directly on the test's own stream with
        // own uploads (bypasses WeightStore/Scheduler/Executor plumbing).
        {
            auto st = device.create_stream();
            const auto word_elems = static_cast<size_t>(pkg->find("emb.word")->element_count());
            const auto pos_elems = static_cast<size_t>(pkg->find("emb.pos")->element_count());
            auto wbuf = device.create_buffer<float>(word_elems);
            auto pbuf = device.create_buffer<float>(pos_elems);
            auto gbuf = device.create_buffer<float>(1024);
            auto bbuf = device.create_buffer<float>(1024);
            auto ibuf = device.create_buffer<uint32_t>(N);
            auto obuf = device.create_buffer<float>(static_cast<size_t>(N) * 1024);
            st << wbuf.copy_from(luisa::span{pkg->find("emb.word")->data, word_elems})
               << pbuf.copy_from(luisa::span{pkg->find("emb.pos")->data, pos_elems})
               << gbuf.copy_from(luisa::span{pkg->find("emb.ln.gamma")->data, 1024ul})
               << bbuf.copy_from(luisa::span{pkg->find("emb.ln.beta")->data, 1024ul})
               << ibuf.copy_from(luisa::span{diag_ids.data(), static_cast<size_t>(N)})
               << synchronize();
            auto shader = device.compile(make_embed_lookup_kernel());
            luisa::vector<float> got(static_cast<size_t>(N) * 1024);
            st << shader(wbuf.view(), pbuf.view(), ibuf.view(0, N), gbuf.view(), bbuf.view(),
                         obuf.view(), N, 1024u, 1e-5f, 2u)
                      .dispatch(N * 256u)
               << obuf.copy_to(luisa::span{got})
               << synchronize();
            double d = 0.0;
            for (size_t i = 0; i < got.size(); i++) {
                d = std::max(d, static_cast<double>(std::abs(got[i] - diag_ref[i])));
            }
            LUISA_INFO("DIAG direct embed_lookup (test stream) vs host-reference: max_abs_diff = {:.3e}", d);
        }
        double d_ref_gold = 0.0;
        for (size_t i = 0; i < diag_ref.size(); i++) {
            d_ref_gold = std::max(d_ref_gold, static_cast<double>(std::abs(diag_ref[i] - gold_layers[i])));
        }
        LUISA_INFO("DIAG host-reference embedding vs golden_layers[0]: max_abs_diff = {:.3e}", d_ref_gold);
    }

    // ---- layer trace against golden_layers.bin (D4 oracle, sentences 0/3).
    // Callback semantics (executor.h): invoked on the caller thread AFTER the
    // worker event fence completes, with a stable BufferView of the per-layer
    // trace slot; the consumer downloads through its own stream.
    // ONE download stream for the whole trace section, alive for its full
    // scope: the validation layer keys per-resource usage by stream handle, and
    // destroying+recreating streams between sentences lets a new unnamed stream
    // inherit the recycled handle, making still-live trace slots look "in use".
    auto dl_stream = device.create_stream();
    for (uint32_t si = 0; si < 2u; si++) {
        const auto sent = kTraceSentences[si];
        luisa::vector<luisa::vector<float>> host;
        uint32_t traced_rows = 0u, traced_cols = 0u;
        session->set_layer_trace([&](uint32_t layer, BufferView<float> activations, uint32_t rows, uint32_t cols) noexcept {
            (void)layer;
            luisa::vector<float> dst(static_cast<size_t>(rows) * cols);
            dl_stream << activations.copy_to(luisa::span{dst}) << synchronize();
            traced_rows = rows;
            traced_cols = cols;
            host.emplace_back(std::move(dst));
        });
        auto emb = session->encode(corpus[sent]);
        session->set_layer_trace({});// unset: zero overhead afterwards
        (void)emb;

        expect(host.size() == 25u) << "25 traced layer states for sentence " << sent;
        expect(traced_cols == kDenseDim) << "trace cols";
        expect(traced_rows == static_cast<uint32_t>(kTraceSentenceRows[si]))
            << "trace rows match golden_layers.bin rows for sentence " << sent;

        const size_t stride = kTraceSentenceRows[si] * kDenseDim;
        if (diag && si == 0u) {
            double d = 0.0;
            for (size_t i = 0; i < diag_ref.size() && i < host[0].size(); i++) {
                d = std::max(d, static_cast<double>(std::abs(diag_ref[i] - host[0][i])));
            }
            LUISA_INFO("DIAG host-reference vs traced slot0: max_abs_diff = {:.3e} (ref n={}, slot n={})",
                       d, diag_ref.size(), host[0].size());
        }
        float worst_layer_diff = 0.0f;
        // O-G: the 1e-3 per-layer trace gate compares against the FROZEN fp32
        // golden_layers.bin; a half package's weights are RNE-quantized, so the
        // quantization drift (bf16: observed worst 1.0e-1) legitimately exceeds
        // it. The authoritative half-package gate remains the end-to-end cosine
        // (>= 0.999 above). For fp32 the 1e-3 gate is unchanged; for half
        // packages it widens to 0.5 -- still catches upload/layout corruption
        // (wrong-tensor mismatches are O(1)-O(10) diffs), not a softening of the
        // fp32 suite.
        // QUANT (v3 int8): per-tensor symmetric int8 is far more lossy than the
        // half formats, so the traced drift against the fp32 golden trace is
        // can approach O(1) (measured 1.173 and 0.7301 on the two traced sentences).
        // The dense cosine gate above is authoritative for v3 model quality;
        // kInt8LayerDiagnosticGate (2.0) only catches gross layout/upload
        // corruption -- e.g. a wrong tensor scale, a wrong blob stride, or a
        // missing dequantization (O(1e2)+ diffs). The DENSE golden cosine gate
        // (>= 0.98 above, full corpus in test_embed_e2e) is the AUTHORITATIVE
        // model-quality gate for int8; this per-layer number is not a quality
        // claim. Every layer is still printed and compared on every package.
        const float layer_gate = pkg_int8 ? kInt8LayerDiagnosticGate
                                          : (pkg_half ? kHalfLayerGate : kFp32LayerGate);
        const bool layer_gate_is_diagnostic = pkg_int8;
        for (uint32_t layer = 0; layer < 25u; layer++) {
            const float *golden = gold_layers.data() + kTraceSentenceBase[si] / 4u + static_cast<size_t>(layer) * stride;
            const auto &got = host[layer];
            float max_diff = 0.0f;
            for (size_t i = 0; i < got.size(); i++) {
                max_diff = std::max(max_diff, std::abs(got[i] - golden[i]));
            }
            worst_layer_diff = std::max(worst_layer_diff, max_diff);
            LUISA_INFO("sentence {} layer {:2d}: max_abs_diff = {:.3e}", sent, layer, max_diff);
            expect(max_diff <= layer_gate)
                << (layer_gate_is_diagnostic ? "layer (int8 DIAGNOSTIC gate) " : "layer ")
                << layer << " diff <= gate (sentence " << sent << ")";
        }
        if (layer_gate_is_diagnostic) {
            LUISA_INFO("sentence {} worst per-layer max-abs-diff = {:.3e} (int8 DIAGNOSTIC/regression gate <= {:.1f}; "
                       "the dense cosine gate >= {:.4f} is the AUTHORITATIVE model-quality gate -- per-tensor symmetric "
                       "int8 trace drift approaching O(1) is expected and is not a model-quality statement)",
                       sent, worst_layer_diff, layer_gate, golden_gate);
        } else {
            LUISA_INFO("sentence {} worst per-layer max-abs-diff = {:.3e} (gate <= {:.1e})",
                       sent, worst_layer_diff, layer_gate);
        }
    }

    return 0;
}
