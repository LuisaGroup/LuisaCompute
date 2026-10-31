// Optimization-phase benchmark for the bge-m3 embedding engine
// (embedding_model_opt_plan.md O0.1/O0.2 -- the ruler every optimization is
// measured with; built by the parent, not delegated).
//
// What is measured
// ----------------
// Session mode (default): end-to-end encode latency through the public Session
//   API -- the number the plan's acceptance table cares about. min/median/mean
//   over --iters samples after --warmup warm-up encodes (lc_optimize §10.8:
//   sub-0.2 ms kernels have +-50 % noise -- distrust single runs; the engine's
//   per-encode time is milliseconds, so min-of-N with warm-up is the
//   methodology). Cold start (Session::create: package load + weight upload +
//   kernel compile) is timed once and reported separately. Tokenizer-only time
//   is measured on the same text for the host-side share.
//
// Executor mode (--executor): direct Executor micro-bench on a dedicated
//   stream: host CommandList record cost vs commit+synchronize wall time.
//   This is the submission-overhead ruler for plan items H1/H3 (CommandList
//   batching, pre-recorded templates).
//
// Roofline mode (--roofline): device DRAM bandwidth (streaming float4 copy)
//   and peak fp32 FLOPS (FMA loop) anchors for the O0.2 table; arithmetic
//   intensity of each engine shape is computed from these, not assumed.
//
// Per-dispatch GPU stage breakdown (per-layer-type / attention / heads) is
// NOT attempted here: host timers cannot see inside a committed CommandList.
// That is what O0.3 (RenderDoc capture) is for -- see
// scripts/embedding/renderdoc_capture.md.
//
// Usage
// -----
//   benchmark_embed <backend> [--model DIR] [--seq N] [--batch B]
//                   [--iters N] [--warmup N] [--csv FILE] [--heads]
//                   [--executor] [--roofline] [--text "sentence"] [--note S]
//
// Defaults: seq=512 batch=1 iters=20 warmup=3, dense-only output.
// Model dir: --model > BGEM3_MODEL_DIR env > error. Expects model.bgm and
// tokenizer.json inside.
//
// Examples:
//   xmake run benchmark_embed dx --seq 512 --batch 1
//   xmake run benchmark_embed vk --seq 512 --batch 32 --iters 10 --heads
//   xmake run benchmark_embed dx --roofline
//   xmake run benchmark_embed dx --executor --seq 8192 --batch 1
#include "ut/ut.hpp"// boost.ut cfg used by test_device.h
#include "test_device.h"

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>

#include <luisa/core/clock.h>
#include <luisa/core/fiber.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "embed/executor.h"
#include "embed/package.h"
#include "embed/session.h"
#include "embed/tokenizer.h"

using namespace luisa;
using namespace luisa::compute;
using luisa::compute::Buffer;
using luisa::compute::Device;
using luisa::compute::Stream;

namespace {

struct Options {
    std::filesystem::path model_dir{};
    uint32_t seq{512u};
    uint32_t batch{1u};
    uint32_t iters{20u};
    uint32_t warmup{3u};
    std::filesystem::path csv{};
    bool heads{false};
    bool executor_mode{false};
    bool roofline{false};
    // EncodeOptions::max_seq_len pass-through. Default 512 = EncodeOptions
    // default, so --seq 512 measures the plan's S=512 row. Wave-1 lesson: a
    // larger default silently re-shapes batch jobs (S=561 pads to bucket 2048
    // > kMaxJobRows -> multi-chunk), changing both cost AND the code path.
    uint32_t maxseq{512u};
    luisa::string text{};
    luisa::string note{};
};

[[nodiscard]] uint32_t parse_uint32(const char *text, const char *name, uint32_t minimum) {
    auto input = luisa::string_view{text};
    uint32_t value{};
    auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    LUISA_ASSERT(result.ec == std::errc{} && result.ptr == input.data() + input.size() && value >= minimum,
                 "{} must be an integer >= {}, got \"{}\".", name, minimum, text);
    return value;
}

[[nodiscard]] Options parse_args(int argc, char *argv[]) {
    Options opt;
    if (const auto *env = std::getenv("BGEM3_MODEL_DIR"); env != nullptr && env[0] != '\0') {
        opt.model_dir = env;
    }
    for (int i = 2; i < argc; i++) {
        const luisa::string_view arg{argv[i]};
        const auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                LUISA_ERROR_WITH_LOCATION("flag {} requires a value", arg);
            }
            return argv[++i];
        };
        if (arg == "--model") {
            opt.model_dir = need("model");
        } else if (arg == "--seq") {
            opt.seq = parse_uint32(need("seq"), "seq", 1u);
        } else if (arg == "--batch") {
            opt.batch = parse_uint32(need("batch"), "batch", 1u);
        } else if (arg == "--iters") {
            opt.iters = parse_uint32(need("iters"), "iters", 1u);
        } else if (arg == "--warmup") {
            opt.warmup = parse_uint32(need("warmup"), "warmup", 0u);
        } else if (arg == "--csv") {
            opt.csv = need("csv");
        } else if (arg == "--maxseq") {
            opt.maxseq = parse_uint32(need("maxseq"), "maxseq", 2u);
        } else if (arg == "--heads") {
            opt.heads = true;
        } else if (arg == "--executor") {
            opt.executor_mode = true;
        } else if (arg == "--roofline") {
            opt.roofline = true;
        } else if (arg == "--text") {
            opt.text = need("text");
        } else if (arg == "--note") {
            opt.note = need("note");
        } else {
            LUISA_ERROR_WITH_LOCATION("unknown argument: {}", arg);
        }
    }
    if (opt.model_dir.empty()) {
        LUISA_ERROR_WITH_LOCATION(
            "no model dir: pass --model DIR or set BGEM3_MODEL_DIR (expects model.bgm + tokenizer.json inside)");
    }
    return opt;
}

struct Statistics {
    double min{};
    double median{};
    double mean{};
};

[[nodiscard]] Statistics summarize(luisa::vector<double> samples) {
    LUISA_ASSERT(!samples.empty(), "No samples to summarize.");
    luisa::sort(samples.begin(), samples.end());
    const auto sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    return {samples.front(), samples[samples.size() / 2u],
            sum / static_cast<double>(samples.size())};
}

// Repeat a base paragraph until the tokenizer yields >= target tokens. The
// engine pads batches to the longest member, so an over-long text lands in a
// predictable bucket; the ACTUAL token count is what gets reported and gated.
[[nodiscard]] luisa::string make_text(const luisa::embed::Tokenizer &tok,
                                      uint32_t target_tokens) {
    static constexpr luisa::string_view kBase =
        "The quick brown fox jumps over the lazy dog while seventeen curious "
        "sparrows count the autumn leaves falling silently across the old "
        "courtyard near the riverbank, and somewhere downstream a patient "
        "fisherman mends his nets as the morning fog lifts slowly.\n";
    luisa::string text;
    for (uint32_t attempt = 0; attempt < 4096u; attempt++) {
        text.append(kBase.data(), kBase.size());
        if (tok.encode(text, 8192u).size() >= target_tokens) {
            break;
        }
    }
    return text;
}

void append_csv(const Options &opt, luisa::string_view backend, luisa::string_view mode,
                uint32_t actual_seq, double cold_ms, double tok_min,
                const Statistics &enc, double record_min, double wall_min) {
    if (opt.csv.empty()) {
        return;
    }
    const bool need_header = !std::filesystem::exists(opt.csv) ||
                             std::filesystem::file_size(opt.csv) == 0u;
    std::ofstream out{opt.csv, std::ios::app};
    if (!out) {
        LUISA_WARNING("benchmark_embed: cannot open csv \"{}\" -- row dropped.", opt.csv.string());
        return;
    }
    if (need_header) {
        out << "backend,mode,seq,batch,heads,iters,note,cold_start_ms,tok_min_ms,"
               "enc_min_ms,enc_median_ms,enc_mean_ms,record_min_ms,wall_min_ms\n";
    }
    out << backend << ',' << mode << ',' << actual_seq << ',' << opt.batch << ','
        << (opt.heads ? 1 : 0) << ',' << opt.iters << ',' << opt.note << ','
        << cold_ms << ',' << tok_min << ',' << enc.min << ',' << enc.median << ','
        << enc.mean << ',' << record_min << ',' << wall_min << "\n";
}

//--- roofline anchors (O0.2) -------------------------------------------------

void run_roofline(Device &device, luisa::string_view backend) {
    Stream stream = device.create_stream();

    // DRAM bandwidth: one float4 read + one float4 write per thread. Grid is
    // exactly at the DX 65535-block cap (65535 x 256 threads x 16 B = 268 MB
    // traffic per direction).
    const uint32_t n4 = 65535u * 256u;
    auto in = device.create_buffer<float4>(n4);
    auto out = device.create_buffer<float4>(n4);
    Kernel1D fill4 = [](BufferFloat4 dst, Float4 value) noexcept {
        set_block_size(256u);
        dst.write(dispatch_id().x, value);
    };
    auto fill_shader = device.compile(fill4, compute::ShaderOption{.name = "roofline_fill"});
    Kernel1D stream_kernel = [](BufferFloat4 src, BufferFloat4 dst) noexcept {
        set_block_size(256u);
        const auto i = dispatch_id().x;
        dst.write(i, src.read(i) * 1.0000001f);
    };
    auto stream_shader = device.compile(stream_kernel, compute::ShaderOption{.name = "roofline_stream"});
    stream << fill_shader(in.view(), make_float4(1.0f)).dispatch(n4)
           << fill_shader(out.view(), make_float4(0.0f)).dispatch(n4) << synchronize();
    luisa::vector<double> bw;
    for (uint32_t i = 0u; i < 10u; i++) {
        Clock clock;
        stream << stream_shader(in.view(), out.view()).dispatch(n4) << synchronize();
        bw.emplace_back(clock.toc());
    }
    const auto bw_min = summarize(bw).min;
    const double bw_gbps = static_cast<double>(n4) * 16.0 * 2.0 / (bw_min * 1.0e6);

    // Peak fp32 FLOPS: 4 independent float4 accumulator chains (ILP 4 -- a
    // single dependent chain is latency-bound and would under-report by >10x),
    // native C++ loop with compile-time trip count 1024 (unrolled at emission
    // time per the DSL rule). Occupancy fills the device; the number is a
    // practical-peak anchor, not a datasheet quote.
    Kernel1D flops_kernel = [](BufferFloat sink) noexcept {
        set_block_size(256u);
        // NOTE: must be Var<float>, not `auto x = 1.0f` -- a plain host float
        // makes the whole loop constant-fold at emission time and the kernel
        // body would be empty (the anchor would measure an empty dispatch).
        Var<float> x0 = 1.0000001f;
        Var<float> x1 = 0.9999999f;
        Var<float> x2 = 1.0000002f;
        Var<float> x3 = 0.9999998f;
        for (auto j = 0u; j < 1024u; j++) {
            x0 = x0 * 1.0000001f + 0.0000001f;
            x1 = x1 * 1.0000001f - 0.0000001f;
            x2 = x2 * 1.0000001f + 0.0000002f;
            x3 = x3 * 1.0000001f - 0.0000002f;
        }
        // Never-true guard keeps the chains alive without polluting the cache.
        $if (dispatch_id().x == 0xFFFFFFFFu) {
            sink.write(0u, x0 + x1 + x2 + x3);
        };
    };
    auto sink_buf = device.create_buffer<float>(1u);
    auto flops_shader = device.compile(flops_kernel, compute::ShaderOption{.name = "roofline_flops"});
    const uint32_t flop_blocks = 76u * 8u;// 76 SMs on the dev GPU, 8 blocks each
    luisa::vector<double> fl;
    for (uint32_t i = 0u; i < 5u; i++) {
        Clock clock;
        stream << flops_shader(sink_buf.view()).dispatch(flop_blocks * 256u) << synchronize();
        fl.emplace_back(clock.toc());
    }
    const auto fl_min = summarize(fl).min;
    const double tflops = static_cast<double>(flop_blocks) * 256.0 *
                          1024.0 /*unrolled fma*/ * 4.0 /*chains*/ * 2.0 / (fl_min * 1.0e6) / 1.0e3;

    LUISA_INFO("=== roofline anchors (backend {}) ===", backend);
    LUISA_INFO("DRAM bandwidth : {:>10.1f} GB/s   (streaming float4 r+w, 537 MB moved, min-of-10)", bw_gbps);
    LUISA_INFO("fp32 FLOPS     : {:>10.2f} TFLOPS (4 ILP chains, 1024 unrolled FMA, min-of-5)", tflops);
    LUISA_INFO("note: anchors are practical-peak measurements on this device, not datasheet numbers");
}

//--- Session-mode e2e bench ---------------------------------------------------

void run_session_bench(Device &device, const Options &opt, luisa::string_view backend) {
    const auto bgm = opt.model_dir / "model.bgm";
    const auto tokjson = opt.model_dir / "tokenizer.json";
    if (!std::filesystem::exists(bgm) || !std::filesystem::exists(tokjson)) {
        LUISA_ERROR_WITH_LOCATION("model dir \"{}\" must contain model.bgm and tokenizer.json",
                                  opt.model_dir.string());
    }

    Clock cold_clock;
    auto session = luisa::embed::Session::create(device, bgm, tokjson, 1u);
    const double cold_ms = cold_clock.toc();

    auto tokenizer = luisa::embed::Tokenizer::load(tokjson);
    const auto text = opt.text.empty() ? make_text(*tokenizer, opt.seq) : opt.text;
    const auto ids0 = tokenizer->encode(text, 8192u);
    const uint32_t actual_seq = static_cast<uint32_t>(ids0.size());
    LUISA_INFO("session bench: text tokenizes to {} tokens (target {}), batch {}, heads {}",
               actual_seq, opt.seq, opt.batch, opt.heads);

    luisa::embed::EncodeOptions eopt;
    eopt.want_sparse = opt.heads;
    eopt.want_colbert = opt.heads;
    eopt.max_seq_len = opt.maxseq;// pass-through so --seq >512 reaches the flash path

    // Tokenizer-only ruler (same work the session does per text).
    luisa::vector<double> tok_samples;
    for (uint32_t i = 0u; i < opt.iters; i++) {
        Clock clock;
        auto ids = tokenizer->encode(text, 8192u);
        (void)ids;
        tok_samples.emplace_back(clock.toc());
    }
    const auto tok = summarize(tok_samples);

    // Warm-up: allocator steady state + lazy kernel compiles (flash path) must
    // not leak into the samples.
    for (uint32_t i = 0u; i < opt.warmup; i++) {
        if (opt.batch == 1u) {
            auto e = session->encode(text, eopt);
            (void)e;
        } else {
            luisa::vector<luisa::string_view> texts(opt.batch, luisa::string_view{text});
            auto es = session->encode_batch(texts, eopt);
            (void)es;
        }
    }

    luisa::vector<double> enc;
    for (uint32_t i = 0u; i < opt.iters; i++) {
        Clock clock;
        if (opt.batch == 1u) {
            auto e = session->encode(text, eopt);
            (void)e;
        } else {
            luisa::vector<luisa::string_view> texts(opt.batch, luisa::string_view{text});
            auto es = session->encode_batch(texts, eopt);
            (void)es;
        }
        enc.emplace_back(clock.toc());
    }
    const auto enc_stats = summarize(enc);

    LUISA_INFO("=== session e2e (backend {}, seq {}, batch {}, heads {}) ===",
               backend, actual_seq, opt.batch, opt.heads);
    LUISA_INFO("cold start (load+upload+compile): {:>10.1f} ms", cold_ms);
    LUISA_INFO("tokenize only       min/median/mean: {:>8.3f} / {:>8.3f} / {:>8.3f} ms", tok.min, tok.median, tok.mean);
    LUISA_INFO("encode e2e          min/median/mean: {:>8.3f} / {:>8.3f} / {:>8.3f} ms", enc_stats.min, enc_stats.median, enc_stats.mean);
    if (opt.batch > 1u) {
        LUISA_INFO("per-embedding (batch wall / {}):    {:>8.3f} ms", opt.batch, enc_stats.min / static_cast<double>(opt.batch));
    }
    append_csv(opt, backend, "session", actual_seq, cold_ms, tok.min, enc_stats, 0.0, 0.0);
}

//--- Executor-mode submission ruler -------------------------------------------

void run_executor_bench(Device &device, const Options &opt, luisa::string_view backend) {
    const auto bgm = opt.model_dir / "model.bgm";
    const auto tokjson = opt.model_dir / "tokenizer.json";
    auto package = luisa::embed::Package::load(bgm);
    if (package == nullptr) {
        LUISA_ERROR_WITH_LOCATION("failed to load package \"{}\"", bgm.string());
    }
    auto tokenizer = luisa::embed::Tokenizer::load(tokjson);
    const auto text = opt.text.empty() ? make_text(*tokenizer, opt.seq) : opt.text;
    const auto ids0 = tokenizer->encode(text, 8192u);
    const uint32_t S = static_cast<uint32_t>(ids0.size());
    const uint32_t B = opt.batch;
    const uint32_t H = package->hidden_size();
    const uint32_t rows = B * S;
    LUISA_INFO("executor bench: B={} S={} rows={} H={} (naive trio for S<=1024, flash above)",
               B, S, rows, H);

    Stream stream = device.create_stream();
    luisa::embed::WeightStore weights{device, *package};
        CommandList upload;
        weights.record_upload(upload);
        auto conversion_staging = weights.take_conversion_sources();
        stream << std::move(upload).commit() << synchronize();
        conversion_staging.sources.clear();

    Clock compile_clock;
    luisa::embed::Executor executor{device, weights, stream};
    LUISA_INFO("executor kernel compile: {:.1f} ms", compile_clock.toc());

    std::vector<uint32_t> ids(rows, 1u);
    std::vector<float> mask(rows, 1.0f);
    std::vector<float> dense(static_cast<size_t>(B) * H, 0.0f);
    std::vector<float> sparse(opt.heads ? rows : 0u, 0.0f);
    std::vector<float> colbert(opt.heads ? static_cast<size_t>(rows) * H : 0u, 0.0f);
    for (uint32_t b = 0u; b < B; b++) {
        std::copy(ids0.begin(), ids0.end(), ids.begin() + static_cast<size_t>(b) * S);
    }

    luisa::embed::EncodeJob job;
    job.batch = B;
    job.seq = S;
    job.ids = luisa::span<const uint32_t>{ids.data(), ids.size()};
    job.mask = luisa::span<const float>{mask.data(), mask.size()};
    job.want_sparse = opt.heads;
    job.want_colbert = opt.heads;
    job.dense_out = luisa::span<float>{dense.data(), dense.size()};
    job.sparse_out = luisa::span<float>{sparse.data(), sparse.size()};
    job.colbert_out = luisa::span<float>{colbert.data(), colbert.size()};

    // Warm-up (pool growth + lazy flash compile).
    for (uint32_t i = 0u; i < opt.warmup + 1u; i++) {
        CommandList cl;
        executor.record_encode(cl, job);
        stream << std::move(cl).commit() << synchronize();
    }

    luisa::vector<double> record_samples, wall_samples;
    for (uint32_t i = 0u; i < opt.iters; i++) {
        Clock wall_clock;
        CommandList cl;
        Clock record_clock;
        executor.record_encode(cl, job);
        record_samples.emplace_back(record_clock.toc());
        stream << std::move(cl).commit() << synchronize();
        wall_samples.emplace_back(wall_clock.toc());
    }
    const auto rec = summarize(record_samples);
    const auto wall = summarize(wall_samples);
    const double gpu_min = wall.min - rec.min;// queue-idle: wall = record + gpu + sync overhead

    LUISA_INFO("=== executor submission (backend {}, B {} x S {}) ===", backend, B, S);
    LUISA_INFO("host record min/median: {:>8.3f} / {:>8.3f} ms  (CommandList build cost)", rec.min, rec.median);
    LUISA_INFO("record+commit+sync   : {:>8.3f} / {:>8.3f} / {:>8.3f} ms (min/median/mean)", wall.min, wall.median, wall.mean);
    LUISA_INFO("gpu+sync estimate    : {:>8.3f} ms  (wall min - record min)", gpu_min);
    append_csv(opt, backend, "executor", S, 0.0, 0.0, wall, rec.min, wall.min);
}

}// namespace

int main(int argc, char *argv[]) {
    luisa::fiber::scheduler fiber_scheduler;
    if (argc < 2) {
        LUISA_ERROR_WITH_LOCATION(
            "Usage: {} <backend> [--model DIR] [--seq N] [--batch B] [--iters N] "
            "[--warmup N] [--csv FILE] [--heads] [--executor] [--roofline] [--text S] [--note S]",
            argv[0]);
        return 1;
    }
    const auto opt = parse_args(argc, argv);
    log_level_info();
    auto [context, device] = test::create_device(argc, argv);
    const luisa::string_view backend{argv[1]};

    if (opt.roofline) {
        run_roofline(device, backend);
        return 0;
    }
    if (opt.executor_mode) {
        run_executor_bench(device, opt, backend);
        return 0;
    }
    run_session_bench(device, opt, backend);
    return 0;
}
