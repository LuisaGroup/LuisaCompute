// Test for ocr/ops/ops_decode sampling building blocks (Phase 3).
// Covers: greedy argmax parity (device kernel vs host sampler vs oracle),
// temperature softmax device parity, deterministic sampling (same seed =>
// identical tokens; different seed => different), top-k/top-p validation,
// repetition penalty + no-repeat-ngram processors, invalid parameters.
// The sampler is host-side by design (host decode scheduling); the device
// softmax/argmax kernels are checked against the same logits. Synthetic only.
#include "ut/ut.hpp"
#include "test_device.h"

#include <limits>
#include "ocr_decode_test_utils.h"

#include "ocr/ops/ops_decode.h"
#include "ocr/runtime/dispatch_adapter.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 2e-4f;
constexpr float kRtol = 2e-4f;

// Deterministic logits row generator.
luisa::vector<float> make_logits(uint32_t vocab, uint32_t seed) {
    ocr::test::Lcg lcg;
    lcg.state = seed;
    luisa::vector<float> v(vocab);
    lcg.fill(v, -3.0f, 3.0f);
    return v;
}

void test_greedy_argmax_parity(Device &device) {
    auto stream = device.create_stream();
    constexpr uint32_t vocab = 64, rows = 5;
    auto logits = make_logits(vocab * rows, 0x11u);
    auto buf = ocr::test::upload(device, stream, luisa::span<const float>{logits});
    auto out = device.create_buffer<int32_t>(rows);
    auto plan = plan_argmax_dispatch(rows);
    expect(plan.ok());
    auto shader = device.compile(make_argmax_kernel());
    expect(shader.compile_ok());
    OCR_DISPATCH_1D(stream, shader(buf, out, rows, vocab), *plan.value);
    luisa::vector<int32_t> got;
    ocr::test::download(stream, out, got, rows);
    stream << synchronize();
    // Host sampler (greedy) + oracle agree with the device kernel per row.
    DecodeSamplingConfig cfg;// greedy default
    for (uint32_t r = 0; r < rows; r++) {
        luisa::span<const float> row{logits.data() + static_cast<size_t>(r) * vocab, vocab};
        DeterministicSampler sampler{cfg, vocab, {}};
        auto picked = sampler.sample(row);
        expect(picked.ok());
        expect(static_cast<int32_t>(*picked.value) == got[r]);
        expect(ocr::test::oracle_argmax(row) == static_cast<uint32_t>(got[r]));
    }
}

void test_softmax_device_parity(Device &device) {
    auto stream = device.create_stream();
    constexpr uint32_t vocab = 48, rows = 3;
    const float temperature = 0.7f;
    auto logits = make_logits(vocab * rows, 0x22u);
    auto buf = ocr::test::upload(device, stream, luisa::span<const float>{logits});
    auto out = device.create_buffer<float>(logits.size());
    auto plan = plan_softmax_dispatch(rows);
    expect(plan.ok());
    auto shader = device.compile(make_softmax_kernel());
    expect(shader.compile_ok());
    OCR_DISPATCH_1D(stream, shader(buf, out, rows, vocab, temperature), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, out, got, logits.size());
    stream << synchronize();
    luisa::vector<float> ref;
    expect(ref_softmax(logits, rows, vocab, temperature, ref).ok());
    ocr::test::expect_close(got, ref, kAtol, kRtol, "temperature softmax device parity");
    // Each row sums to 1.
    for (uint32_t r = 0; r < rows; r++) {
        double sum = 0.0;
        for (uint32_t c = 0; c < vocab; c++) { sum += got[static_cast<size_t>(r) * vocab + c]; }
        expect(std::abs(sum - 1.0) < 1e-3);
    }
}

void test_sampling_determinism() {
    constexpr uint32_t vocab = 32;
    DecodeSamplingConfig cfg;
    cfg.greedy = false;
    cfg.temperature = 0.9f;
    cfg.top_k = 8;
    cfg.top_p = 0.95f;
    cfg.seed = 12345;
    expect(cfg.validate(vocab).ok());
    // Two samplers with the same seed + same logits stream produce identical
    // token sequences (the RNG advances once per sampled token).
    const auto run = [&]() {
        const luisa::vector<uint32_t> prompt_a = {1u, 2u, 3u};
        DeterministicSampler sampler{cfg, vocab, prompt_a};
        luisa::vector<uint32_t> tokens;
        for (uint32_t step = 0; step < 24; step++) {
            auto logits = make_logits(vocab, 0x33u + step);
            auto t = sampler.sample(logits);
            expect(t.ok());
            tokens.push_back(*t.value);
        }
        return tokens;
    };
    auto a = run();
    auto b = run();
    expect(a == b);
    // A different seed changes the sequence (with overwhelming probability).
    cfg.seed = 99999;
    auto c = run();
    expect(c != a);
}

void test_greedy_determinism() {
    constexpr uint32_t vocab = 32;
    DecodeSamplingConfig cfg;// greedy
    const auto run = [&]() {
        DeterministicSampler sampler{cfg, vocab, {}};
        luisa::vector<uint32_t> tokens;
        for (uint32_t step = 0; step < 16; step++) {
            auto logits = make_logits(vocab, 0x44u + step);
            tokens.push_back(*sampler.sample(logits).value);
        }
        return tokens;
    };
    expect(run() == run());
}

void test_top_k_restricts() {
    constexpr uint32_t vocab = 16;
    DecodeSamplingConfig cfg;
    cfg.greedy = false;
    cfg.temperature = 1.0f;
    cfg.top_k = 1;// only the single best token survives
    cfg.seed = 7;
    DeterministicSampler sampler{cfg, vocab, {}};
    auto logits = make_logits(vocab, 0x55u);
    const uint32_t best = ocr::test::oracle_argmax(logits);
    for (int i = 0; i < 8; i++) {
        auto t = sampler.sample(logits);
        expect(t.ok());
        expect(*t.value == best);
    }
}

void test_repetition_penalty() {
    constexpr uint32_t vocab = 8;
    // Logits where token 0 is best; a strong repetition penalty on the seen
    // token 0 must shift the greedy choice to token 1.
    luisa::vector<float> logits = {5.0f, 4.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f};
    DecodeSamplingConfig cfg;// greedy
    cfg.repetition_penalty = 2.0f;
    const luisa::vector<uint32_t> seen0 = {0u};
    DeterministicSampler penalized{cfg, vocab, seen0};// token 0 already seen
    auto t = penalized.sample(logits);
    expect(t.ok());
    expect(*t.value == 1u);// 5/2=2.5 < 4
    // Without the penalty token 0 wins.
    DecodeSamplingConfig plain;
    DeterministicSampler unpenalized{plain, vocab, seen0};
    expect(*unpenalized.sample(logits).value == 0u);
}

void test_no_repeat_ngram() {
    constexpr uint32_t vocab = 8;
    // History [1,2,3,1,2]: the trailing prefix [1,2] previously occurred at
    // index 0 followed by token 3, so token 3 is banned (no-repeat-3gram).
    DecodeSamplingConfig cfg;
    cfg.no_repeat_ngram_size = 3;
    const luisa::vector<uint32_t> hist = {1u, 2u, 3u, 1u, 2u};
    DeterministicSampler sampler{cfg, vocab, hist};
    // Make token 3 the overwhelming favorite; it must still be banned.
    luisa::vector<float> logits = {0.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto t = sampler.sample(logits);
    expect(t.ok());
    expect(*t.value != 3u);
    // With the ban lifted (ngram disabled) token 3 wins.
    DecodeSamplingConfig plain;
    DeterministicSampler free_sampler{plain, vocab, hist};
    expect(*free_sampler.sample(logits).value == 3u);
}

void test_invalid_parameters() {
    constexpr uint32_t vocab = 16;
    DecodeSamplingConfig cfg;
    cfg.greedy = false;
    cfg.temperature = 0.0f;
    expect(!cfg.validate(vocab).ok());// temperature must be > 0
    cfg.temperature = -1.0f;
    expect(!cfg.validate(vocab).ok());
    cfg.temperature = 1.0f;
    cfg.top_k = vocab + 1;
    expect(!cfg.validate(vocab).ok());// top_k > vocab
    cfg.top_k = 0;
    cfg.top_p = 0.0f;
    expect(!cfg.validate(vocab).ok());// top_p must be > 0
    cfg.top_p = 1.5f;
    expect(!cfg.validate(vocab).ok());// top_p must be <= 1
    cfg.top_p = 1.0f;
    cfg.repetition_penalty = 0.0f;
    expect(!cfg.validate(vocab).ok());
    cfg.repetition_penalty = 1.0f;
    expect(cfg.validate(vocab).ok());
    // Greedy ignores temperature/top-k/top-p but still checks repetition.
    DecodeSamplingConfig greedy;
    greedy.temperature = 0.0f;// ignored for greedy
    greedy.top_p = 0.0f;      // ignored for greedy
    expect(greedy.validate(vocab).ok());
    // Zero vocab refused.
    expect(!cfg.validate(0).ok());
    // Sampler rejects a wrong-size logits row.
    DeterministicSampler sampler{cfg, vocab, {}};
    luisa::vector<float> wrong(vocab + 1, 0.0f);
    expect(!sampler.sample(wrong).ok());
}

void test_splitmix_determinism() {
    uint64_t a = 42, b = 42;
    for (int i = 0; i < 100; i++) {
        expect(decode_splitmix64(a) == decode_splitmix64(b));
    }
    uint64_t c = 43;
    expect(decode_splitmix64(a) != decode_splitmix64(c));
}

// S4: non-finite logits and all-banned rows must return a clear error, never
// a silent token 0 or a banned token. -inf is an allowed masked entry; NaN
// and +inf are invalid. The RNG must not advance on a failed sample.
void test_nonfinite_and_all_banned() {
    constexpr uint32_t vocab = 8;
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // All-NaN row: greedy and sampling both error.
    {
        luisa::vector<float> logits(vocab, nan);
        DecodeSamplingConfig greedy;
        DeterministicSampler g{greedy, vocab, {}};
        expect(!g.sample(logits).ok());
        DecodeSamplingConfig samp;
        samp.greedy = false;
        samp.seed = 42;
        DeterministicSampler s{samp, vocab, {}};
        expect(!s.sample(logits).ok());
    }
    // All -inf row (fully masked): error, not token 0.
    {
        luisa::vector<float> logits(vocab, -inf);
        DecodeSamplingConfig greedy;
        DeterministicSampler g{greedy, vocab, {}};
        expect(!g.sample(logits).ok());
        DecodeSamplingConfig samp;
        samp.greedy = false;
        samp.seed = 42;
        DeterministicSampler s{samp, vocab, {}};
        expect(!s.sample(logits).ok());
    }
    // All +inf row: invalid logits -> error.
    {
        luisa::vector<float> logits(vocab, inf);
        DecodeSamplingConfig greedy;
        DeterministicSampler g{greedy, vocab, {}};
        expect(!g.sample(logits).ok());
    }
    // Mixed NaN/-inf with one finite entry: the finite entry is selected
    // (greedy), proving -inf/NaN are skipped, not silently chosen.
    {
        luisa::vector<float> logits(vocab, -inf);
        logits[3] = nan;
        logits[5] = 1.5f;// only finite selectable
        DecodeSamplingConfig greedy;
        DeterministicSampler g{greedy, vocab, {}};
        auto t = g.sample(logits);
        expect(t.ok());
        expect(*t.value == 5u);
        DecodeSamplingConfig samp;
        samp.greedy = false;
        samp.seed = 7;
        DeterministicSampler s{samp, vocab, {}};
        auto ts = s.sample(logits);
        expect(ts.ok());
        expect(*ts.value == 5u);
    }
    // All-banned via no-repeat-ngram: history forces every token banned.
    // With vocab 2 and no_repeat_ngram_size 2, history [0,1,0,1] bans the
    // completion of the trailing prefix [1] -> bans 0; and [0] prefix bans 1.
    {
        constexpr uint32_t v2 = 2;
        DecodeSamplingConfig cfg;
        cfg.no_repeat_ngram_size = 2;
        // Trailing prefix [0]; earlier windows [0,1] bans 1 and [1,1] bans 0
        // (only windows strictly before the trailing prefix count), so BOTH
        // tokens are banned -> no finite selectable logit.
        const luisa::vector<uint32_t> hist = {0u, 1u, 1u, 0u, 0u};
        luisa::vector<float> logits = {10.0f, 9.0f};
        DeterministicSampler g{cfg, v2, hist};
        auto t = g.sample(logits);
        expect(!t.ok());// both tokens banned -> error, not a banned token
        DecodeSamplingConfig samp;
        samp.greedy = false;
        samp.seed = 3;
        samp.no_repeat_ngram_size = 2;
        DeterministicSampler s{samp, v2, hist};
        expect(!s.sample(logits).ok());
    }
    // RNG does not advance on failure: a failed sample followed by a valid
    // one draws the SAME first value as a fresh sampler (seed unchanged).
    {
        DecodeSamplingConfig cfg;
        cfg.greedy = false;
        cfg.seed = 99;
        luisa::vector<float> bad(vocab, -inf);
        luisa::vector<float> good = make_logits(vocab, 0x77u);
        DeterministicSampler a{cfg, vocab, {}};
        expect(!a.sample(bad).ok());// failure: RNG must not advance
        auto ta = a.sample(good);
        expect(ta.ok());
        DeterministicSampler b{cfg, vocab, {}};
        auto tb = b.sample(good);// fresh: first draw
        expect(tb.ok());
        expect(*ta.value == *tb.value);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "decode_sampling_greedy_argmax_parity"_test = [&] { test_greedy_argmax_parity(device); };
    "decode_sampling_softmax_device_parity"_test = [&] { test_softmax_device_parity(device); };
    "decode_sampling_determinism"_test = [&] { test_sampling_determinism(); };
    "decode_sampling_greedy_determinism"_test = [&] { test_greedy_determinism(); };
    "decode_sampling_top_k_restricts"_test = [&] { test_top_k_restricts(); };
    "decode_sampling_repetition_penalty"_test = [&] { test_repetition_penalty(); };
    "decode_sampling_no_repeat_ngram"_test = [&] { test_no_repeat_ngram(); };
    "decode_sampling_invalid_parameters"_test = [&] { test_invalid_parameters(); };
    "decode_sampling_splitmix_determinism"_test = [&] { test_splitmix_determinism(); };
    "decode_sampling_nonfinite_and_all_banned"_test = [&] { test_nonfinite_and_all_banned(); };
}