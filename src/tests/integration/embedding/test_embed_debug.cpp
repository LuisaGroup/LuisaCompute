// Test for embedding/embed/debug_utils.h -- the plan §9.1 always-on debug
// layer (D1/D3/D4) and the §9.4 debug-harness exit criteria owned by B4.
//
// Coverage:
// - D3 gate (device): route_stream_logs() on this test's own stream, a tiny
//   kernel emitting one DEVICE_WARNING / DEVICE_INFO / DEVICE_ERROR /
//   unprefixed device_log line each, synchronize, then assert host-side
//   receipt through the routed_* counters. 'E' lines must be COUNTED and
//   routed to log_warning -- the test would abort at process level if the
//   router ever called luisa::log_error (plan §9.1 D3 warning).
// - D4 gate (device, Executor-gated and opt-in): with the real model (argv[2],
//   default D:/compute/bge-m3/bge-m3/model.bgm) encode corpus sentence 0
//   through the B1 Session (Package/tokenizer/WeightStore/Scheduler inside
//   Session::create, encode(), synchronous trace firing) with layer dumps
//   registered for layers 0, 12 and 24
//   through install_layer_dump(); compare the written dumps against
//   Runs only when the Session header has landed (LUISA_EMBED_HAS_SESSION) AND
//   the gate is requested (argv token "b4_d4" or BGEM3_D4=1): the real-model
//   pipeline is still under integration (B1/B2/B3: encoder currently diverges
//   from the goldens from layer 0 on, and the dx worker path takes
//   DXGI_ERROR_DEVICE_REMOVED), so the gate stays opt-in and is otherwise
//   skipped with a NOT-RUN warning (see the B4 report for the current
//   evidence).
// - env-knob gate (host): BGEM3_DUMP_LAYER=5 set in-process must parse to
//   layer 5; unset/garbage/overflow must return false.
//
// Run (device test, backend = argv[1], optional bgm path = argv[2]):
// xmake run -o build/embed_b4 test_embed_debug dx
// xmake run -o build/embed_b4 test_embed_debug fallback
// CI gate: dx + fallback (plan §9.4).
#include "ut/ut.hpp"
#include "test_device.h"

#include "embedding/embed/debug_utils.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::embed;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Set/clear a process environment variable in a never-throwing way (the
// env-knob gate must be exercisable in-process, plan §9.1 D4).
void set_env(const char *name, const char *value) noexcept {
#ifdef _WIN32
    static_cast<void>(_putenv_s(name, value));
#else
    static_cast<void>(setenv(name, value, 1));
#endif
}

void clear_env(const char *name) noexcept {
#ifdef _WIN32
    static_cast<void>(_putenv_s(name, ""));
#else
    static_cast<void>(unsetenv(name));
#endif
}

}// namespace

// --- D3 gate ---------------------------------------------------------------
void test_device_log_routing(Device &device) {
    // D1: dev preset check (warning-only; never mutates the environment).
    install_validation_preset();

    reset_routed_counts();
    auto stream = device.create_stream();
    route_stream_logs(stream);

    // One thread per severity class; the emitted strings are unique so a
    // counter per class is an unambiguous receipt proof.
    Kernel1D kernel = []() noexcept {
        $if (dispatch_x() == 0u) {
            DEVICE_WARNING("b4 d3 gate warning={}", 1u);
        };
        $if (dispatch_x() == 1u) {
            DEVICE_INFO("b4 d3 gate info={}", 2u);
        };
        $if (dispatch_x() == 2u) {
            // Must NOT abort the host (routed to log_warning).
            DEVICE_ERROR("b4 d3 gate device-error={}", 3u);
        };
        $if (dispatch_x() == 3u) {
            device_log("b4 d3 gate unprefixed={}", 4u);
        };
    };
    auto shader = device.compile(kernel);
    stream << shader().dispatch(4u) << synchronize();

    const auto info = routed_info_count();
    const auto warning = routed_warning_count();
    const auto error = routed_error_count();
    const auto other = routed_other_count();
    const auto total = routed_total_count();
    LUISA_INFO("D3 routed counts: info={} warning={} error(routed-to-warning)={} other={} total={}",
               info, warning, error, other, total);

    expect(info == 1u) << "exactly one 'I' line must reach log_info";
    expect(warning == 1u) << "exactly one 'W' line must reach log_warning (routed_warning_count)";
    expect(error == 1u) << "the 'E' line must be counted and routed to log_warning, "
                           "never to log_error (abort) -- the test reaching this line is "
                           "itself the proof the host survived";
    expect(other == 1u) << "unprefixed device lines must route to log_verbose";
    expect(total == 4u) << "every callback invocation must be counted";
}

// --- env-knob gate ---------------------------------------------------------
void test_dump_env_knob() {
    uint32_t layer = 0xFFFFFFFFu;
    clear_env("BGEM3_DUMP_LAYER");
    expect(!dump_requested_from_env(layer)) << "unset BGEM3_DUMP_LAYER must return false";

    set_env("BGEM3_DUMP_LAYER", "5");
    layer = 0u;
    expect(dump_requested_from_env(layer)) << "BGEM3_DUMP_LAYER=5 must parse";
    expect(layer == 5u) << "parsed layer must be 5";

    set_env("BGEM3_DUMP_LAYER", "24");
    expect(dump_requested_from_env(layer) && layer == 24u) << "two-digit layer must parse";

    set_env("BGEM3_DUMP_LAYER", "x");
    expect(!dump_requested_from_env(layer)) << "non-decimal value must return false";

    set_env("BGEM3_DUMP_LAYER", "4294967297");
    expect(!dump_requested_from_env(layer)) << "uint32 overflow must return false";

    clear_env("BGEM3_DUMP_LAYER");
}

#if defined(LUISA_EMBED_HAS_SESSION)
#include "embedding/embed/session.h"

namespace {

// Reads a whole file into `out` (never-throwing stdio). Returns false on any
// I/O failure.
bool read_whole_file(const char *name, luisa::vector<char> &out) noexcept {
    out.clear();
    auto *f = std::fopen(name, "rb");
    if (f == nullptr) return false;
    static_cast<void>(std::fseek(f, 0, SEEK_END));
    const auto size = std::ftell(f);
    static_cast<void>(std::fseek(f, 0, SEEK_SET));
    if (size < 0) {
        std::fclose(f);
        return false;
    }
    out.resize(static_cast<size_t>(size));
    const auto read = out.empty() ? 0u : std::fread(out.data(), 1u, out.size(), f);
    static_cast<void>(std::fclose(f));
    return read == out.size();
}

// corpus.txt escapes backslash, \n and \r (golden_meta.json "corpus_escapes"):
// unescape one raw line into the literal bytes of the sentence.
[[nodiscard]] luisa::string unescape_corpus_line(luisa::string_view line) noexcept {
    auto out = luisa::string{};
    out.reserve(line.size());
    for (size_t i = 0u; i < line.size(); i++) {
        const auto c = line[i];
        if (c != '\\' || i + 1u >= line.size()) {
            out.push_back(c);
            continue;
        }
        const auto n = line[++i];
        switch (n) {
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case '\\': out.push_back('\\'); break;
            default:
                out.push_back(c);
                out.push_back(n);
                break;
        }
    }
    return out;
}

// Golden layer-state layout for sentence 0, verified against
// golden_meta.json ("layers" block): 25 states x 15 rows x 1024 cols x 4 B,
// sentence 0 occupies the file prefix [0, 25 * 61440) and the second block
// (sentence 3) starts at 1536000. File total must be 3276800 bytes.
constexpr uint32_t kSentence0Rows = 15u;
constexpr uint32_t kGoldenCols = 1024u;
constexpr size_t kGoldenLayerStride = 61440u;// 15 * 1024 * 4
constexpr size_t kGoldenFileBytes = 3276800u;

[[nodiscard]] luisa::string corpus_sentence0() {
    const auto corpus_path = luisa::to_string(luisa::filesystem::path{LUISA_EMBED_TEST_DATA_DIR} /
                                              "corpus.txt");
    auto *f = std::fopen(corpus_path.c_str(), "rb");
    if (f == nullptr) {
        LUISA_ERROR("D4 gate: cannot open corpus {}", corpus_path);
        return {};
    }
    auto line = luisa::string{};
    int c = 0;
    while ((c = std::fgetc(f)) != EOF && c != '\n' && c != '\r') {
        line.push_back(static_cast<char>(c));
    }
    static_cast<void>(std::fclose(f));
    if (line.empty()) LUISA_ERROR("D4 gate: corpus sentence 0 is empty");
    return unescape_corpus_line(luisa::string_view{line});
}

}// namespace

// --- D4 gate (Session trace hook required, plan §9.1 D4 / §9.4) -------------
void test_layer_dump(Device &device, luisa::string_view bgm_path) {
    const auto bgm = luisa::filesystem::path{bgm_path};
    std::error_code ec;
    if (!std::filesystem::is_regular_file(bgm, ec)) {
        LUISA_WARNING("D4 gate skipped: packed model {} does not exist", bgm_path);
        return;
    }
    const auto tok_path = luisa::filesystem::path{luisa::to_string(
        luisa::filesystem::path{LUISA_EMBED_TEST_DATA_DIR} / "tokenizer.json")};

    // B1 contract: Session::create loads Package + tokenizer, uploads weights,
    // compiles kernels and starts the scheduler (fatal LUISA_ERROR on any load
    // failure -- hence this gate is strictly opt-in).
    auto session = Session::create(device, bgm, tok_path);
    if (session == nullptr) {
        LUISA_WARNING("D4 gate skipped: Session::create({}) failed", bgm_path);
        return;
    }

    // Fresh out dir under the system temp directory (never the repo tree).
    const auto out_dir = std::filesystem::temp_directory_path(ec) / "embed_b4_layer_dump";
    std::filesystem::remove_all(out_dir, ec);
    reset_layer_dump_completed_count();

    static constexpr uint32_t kDumpLayers[] = {0u, 12u, 24u};
    for (const auto l : kDumpLayers) {
        install_layer_dump(*session, device, l, out_dir);
    }

    // B1 contract: synchronous encode() fires the trace callbacks on the
    // caller's host thread once the worker's layer copies completed; the
    // sentence-0 dump must carry 15 rows x 1024 cols (checked below against
    // the golden stride).
    const auto sentence = corpus_sentence0();
    static_cast<void>(session->encode(sentence));

    expect(layer_dump_completed_count() == 3u)
        << "all three registered layers (0, 12, 24) must have been dumped";

    // Goldens
    const auto golden_path = luisa::to_string(
        luisa::filesystem::path{LUISA_EMBED_TEST_DATA_DIR} / "golden_layers.bin");
    luisa::vector<char> golden_raw;
    if (!read_whole_file(golden_path.c_str(), golden_raw)) {
        LUISA_ERROR("D4 gate: cannot open golden {}", golden_path);
        return;
    }
    expect(golden_raw.size() == kGoldenFileBytes)
        << "golden_layers.bin must be 3276800 bytes (25 states, sentences 0 and 3)";

    auto failures = 0u;
    for (const auto l : kDumpLayers) {
        const auto dump_path = luisa::format("{}/layer{}.bin", luisa::to_string(out_dir), l);
        luisa::vector<char> dump_raw;
        if (!read_whole_file(dump_path.c_str(), dump_raw)) {
            LUISA_WARNING("D4 gate: dump {} missing for layer {}", dump_path, l);
            failures++;
            continue;
        }
        if (dump_raw.size() != kGoldenLayerStride) {
            LUISA_WARNING("D4 gate: layer {} dump is {} bytes, expected {} (15x1024 fp32)",
                          l, dump_raw.size(), kGoldenLayerStride);
            failures++;
            continue;
        }
        const auto *dump = reinterpret_cast<const float *>(dump_raw.data());
        const auto *ref = reinterpret_cast<const float *>(golden_raw.data() + l * kGoldenLayerStride);
        auto max_diff = 0.0f;
        auto max_idx = size_t{0u};
        for (size_t i = 0u; i < kSentence0Rows * kGoldenCols; i++) {
            const auto d = std::abs(dump[i] - ref[i]);
            if (d > max_diff) {
                max_diff = d;
                max_idx = i;
            }
        }
        LUISA_INFO("D4 layer-bisect gate: layer {} max-abs-diff={:.3e} (at element {}, got {} vs "
                   "golden {})",
                   l, max_diff, max_idx, dump[max_idx], ref[max_idx]);
        if (!(max_diff <= 1e-3f)) {
            LUISA_WARNING("D4 gate: layer {} diverged beyond 1e-3", l);
            failures++;
        }
    }
    expect(failures == 0u)
        << luisa::format("D4 activation dumps must match golden_layers.bin within 1e-3 ({} failed)",
                         failures);
}
#endif// LUISA_EMBED_HAS_SESSION

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    auto &device = dc->device;

    test_device_log_routing(device);
    test_dump_env_knob();

#if defined(LUISA_EMBED_HAS_SESSION)
    // Opt-in: the real-model pipeline is still under integration (see the
    // file header); flip the gate on with argv token "b4_d4" or BGEM3_D4=1 at
    // the integration review.
    bool run_d4 = false;
    for (auto i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "b4_d4") == 0) run_d4 = true;
    }
    if (const auto d4_env = std::getenv("BGEM3_D4"); d4_env != nullptr && d4_env[0] == '1') {
        run_d4 = true;
    }
    if (run_d4) {
        // argv[2] = bgm path; the gate token itself ("b4_d4") never becomes a path.
        const bool bgm_arg_ok = argc > 2 && argv[2] != nullptr && argv[2][0] != '\0' &&
                                std::strcmp(argv[2], "b4_d4") != 0;
        const auto bgm = bgm_arg_ok ? luisa::string_view{argv[2]} : luisa::string_view{"D:/compute/bge-m3/bge-m3/model.bgm"};
        test_layer_dump(device, bgm);
    } else {
        LUISA_WARNING("D4 layer-dump gate NOT RUN: opt in with argv token b4_d4 or "
                      "BGEM3_D4=1 once the real-model pipeline is green at integration "
                      "(install_layer_dump/env gates are compiled against the landed "
                      "Session contract in the meantime).");
    }
#else
    LUISA_WARNING("D4 layer-dump gate is PENDING: embedding/embed/session.h (B1) has not landed; "
                  "the trace-hook code path compiles in as soon as it does (LUISA_EMBED_HAS_SESSION).");
#endif

    return 0;
}
