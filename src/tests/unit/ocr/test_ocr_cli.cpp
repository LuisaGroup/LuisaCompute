// Test for examples/ocr/ocr/cli.h/.cpp -- the reusable Phase 0 CLI parser and
// structured status report. Host-only: no device, no backend, no model.
//
// This test covers:
// - --help short-circuit (no device/model needed);
// - bad/unknown arguments and missing values -> invalid_argument;
// - a valid Phase 0 invocation prepares a session config + request that pass
//   validation and then fail closed with `unvalidated` (no fabricated output);
// - temperature > 0 switches greedy -> non-greedy; 0 stays greedy;
// - pixel bounds (separate overview/crop + compatibility ceiling);
// - precision flags (automatic default + explicit override);
// - UTF-8 path/prompt handling and invalid-UTF-8 rejection;
// - status report JSON serialization and I/O error propagation.
#include "ut/ut.hpp"

#include <cstdio>
#include <string>

#include <luisa/core/stl/filesystem.h>

#include "ocr/api.h"
#include "ocr/cli.h"
#include "ocr/session.h"
#include "ocr/status.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Parse a fixed argv (argv[0] is a dummy program name). The literals live for
// the whole test, so the pointers remain valid.
[[nodiscard]] Result<CliArgs> parse(std::initializer_list<const char *> args) {
    luisa::vector<const char *> argv;
    argv.emplace_back("example_ocr");
    for (auto a : args) { argv.emplace_back(a); }
    return parse_cli_args(static_cast<int>(argv.size()), argv.data());
}

void test_help() {
    const auto parsed = parse({"--help"});
    expect(parsed.ok());
    expect(parsed.value->help);
    const auto short_help = parse({"-h"});
    expect(short_help.ok());
    expect(short_help.value->help);
    // --help wins even alongside other arguments.
    const auto mixed = parse({"--backend", "vk", "--help"});
    expect(mixed.ok());
    expect(mixed.value->help);
    expect(!cli_usage_text().empty());
}

void test_bad_arguments() {
    expect(parse({"--no-such-flag"}).status.code == StatusCode::invalid_argument);
    expect(parse({"--backend"}).status.code == StatusCode::invalid_argument);// missing value
    expect(parse({"--mode", "bogus"}).status.code == StatusCode::invalid_argument);
    expect(parse({"--model", "not_a_model"}).status.code == StatusCode::invalid_argument);
    expect(parse({"--model", "pp_ocrv6"}).status.code == StatusCode::invalid_argument);// bare family
    expect(parse({"--max-decode-batch", "abc"}).status.code == StatusCode::invalid_argument);
    expect(parse({"--max-decode-batch", "-1"}).status.code == StatusCode::invalid_argument);
    expect(parse({"--temperature", "nan"}).status.code == StatusCode::invalid_argument);
    expect(parse({"--temperature", "inf"}).status.code == StatusCode::invalid_argument);
    expect(parse({"--weight-precision", "fp8"}).status.code == StatusCode::invalid_argument);
    expect(parse({"--cache-policy", "bogus"}).status.code == StatusCode::invalid_argument);
    expect(parse({"--offload", "bogus"}).status.code == StatusCode::invalid_argument);
    expect(parse({"--graph", "bogus"}).status.code == StatusCode::invalid_argument);
}

void test_valid_phase0_preparation() {
    const auto parsed = parse({"--backend", "vk", "--package", "model.lcocr",
                               "--input", "page0.png", "--mode", "plain"});
    expect(parsed.ok());
    expect(!parsed.value->help);
    const CliArgs &args = *parsed.value;
    expect(args.session.backend == "vk");
    expect(args.pages.size() == 1u);
    // The prepared config + request pass host validation...
    expect(validate_session_config(args.session).ok());
    expect(Session::preflight_request(args.pages, args.options).ok());
    // ...and then fail closed (no validated model, no fabricated output).
    const auto session = Session::create(args.session);
    expect(!session.ok());
    expect(session.status.code == StatusCode::unvalidated);
}

void test_temperature_greedy_switch() {
    // Default: greedy.
    {
        const auto parsed = parse({"--backend", "vk", "--package", "m", "--input", "p"});
        expect(parsed.ok());
        expect(parsed.value->options.sampling.greedy);
    }
    // Explicit positive temperature -> non-greedy.
    {
        const auto parsed = parse({"--backend", "vk", "--package", "m", "--input", "p",
                                   "--temperature", "0.7"});
        expect(parsed.ok());
        expect(!parsed.value->options.sampling.greedy);
        expect(parsed.value->options.sampling.temperature > 0.0f);
    }
    // Zero temperature stays greedy.
    {
        const auto parsed = parse({"--backend", "vk", "--package", "m", "--input", "p",
                                   "--temperature", "0"});
        expect(parsed.ok());
        expect(parsed.value->options.sampling.greedy);
    }
}

void test_pixel_bounds() {
    const auto parsed = parse({"--backend", "vk", "--package", "m", "--input", "p",
                               "--max-overview-pixels", "1000",
                               "--max-crop-pixels", "2000",
                               "--max-pixels", "1500"});
    expect(parsed.ok());
    const auto &options = parsed.value->options;
    expect(options.max_overview_pixels == 1000u);
    expect(options.max_crop_pixels == 2000u);
    expect(options.max_pixels == 1500u);
    const auto bounds = effective_pixel_bounds(options);
    expect(bounds.overview == 1000u);// min(1000, 1500)
    expect(bounds.crop == 1500u);// min(2000, 1500)
}

void test_precision_flags() {
    // Default policy is the automatic sentinel.
    {
        const auto parsed = parse({"--backend", "vk", "--package", "m", "--input", "p"});
        expect(parsed.ok());
        expect(parsed.value->options.precision.weights == Precision::automatic);
        expect(parsed.value->session.precision.weights == Precision::automatic);
    }
    // Explicit override reaches both session and request policies.
    {
        const auto parsed = parse({"--backend", "vk", "--package", "m", "--input", "p",
                                   "--weight-precision", "bf16",
                                   "--activation-precision", "fp16",
                                   "--cache-precision", "fp32"});
        expect(parsed.ok());
        expect(parsed.value->options.precision.weights == Precision::bf16);
        expect(parsed.value->options.precision.activations == Precision::fp16);
        expect(parsed.value->options.precision.cache == Precision::fp32);
        expect(parsed.value->session.precision.weights == Precision::bf16);
    }
    // Explicit automatic is accepted.
    {
        const auto parsed = parse({"--backend", "vk", "--package", "m", "--input", "p",
                                   "--weight-precision", "automatic"});
        expect(parsed.ok());
        expect(parsed.value->options.precision.weights == Precision::automatic);
    }
}

void test_utf8_and_paths() {
    // A UTF-8 prompt with non-ASCII bytes passes through unchanged.
    {
        const auto parsed = parse({"--backend", "vk", "--package", "m", "--input", "p",
                                   "--mode", "extract", "--prompt", "提取发票总额"});
        expect(parsed.ok());
        expect(parsed.value->options.prompt == "提取发票总额");
    }
    // Invalid UTF-8 in a path is rejected (not silently treated as ACP).
    {
        const char bad_path[] = {'p', 'a', 'g', 'e', static_cast<char>(0xff), '.', 'p', 'n', 'g', '\0'};
        const auto parsed = parse({"--backend", "vk", "--package", "m", "--input", bad_path});
        expect(parsed.status.code == StatusCode::invalid_argument);
    }
    // extract without schema/prompt fails preflight.
    {
        const auto parsed = parse({"--backend", "vk", "--package", "m", "--input", "p",
                                   "--mode", "extract"});
        expect(parsed.ok());
        expect(Session::preflight_request(parsed.value->pages, parsed.value->options).code == StatusCode::invalid_argument);
    }
}

void test_status_report() {
    const auto status = Status::error(StatusCode::unvalidated, "no model \"validated\"\nyet", "model");
    const auto json = status_report_json(status);
    expect(json.find("\"ok\":false") != luisa::string::npos);
    expect(json.find("\"code\":\"unvalidated\"") != luisa::string::npos);
    expect(json.find("\\\"validated\\\"") != luisa::string::npos);// escaped quotes
    expect(json.find("\\n") != luisa::string::npos);// escaped newline
    expect(json_escape("a\"b\\c") == "a\\\"b\\\\c");
    // Writing to a file succeeds and produces the same JSON.
    {
        const auto path = luisa::filesystem::path{"ocr_cli_test_report.json"};
        const auto write = write_status_report(status, path);
        expect(write.ok());
        std::FILE *file = std::fopen(luisa::to_string(path).c_str(), "rb");
        expect(file != nullptr);
        if (file != nullptr) {
            char buffer[512];
            const auto n = std::fread(buffer, 1, sizeof(buffer) - 1, file);
            buffer[n] = '\0';
            std::fclose(file);
            expect(luisa::string{buffer} == json);
        }
        std::remove(luisa::to_string(path).c_str());
    }
    // Writing to an impossible path reports io_error (not swallowed).
    {
        const auto bad_path = luisa::filesystem::path{"nonexistent_dir_zz/ocr_report.json"};
        const auto write = write_status_report(status, bad_path);
        expect(!write.ok());
        expect(write.code == StatusCode::io_error);
    }
}

}// namespace

static auto test_ocr_cli_registration = [] {
    "help"_test = [] { test_help(); };
    "bad_arguments"_test = [] { test_bad_arguments(); };
    "valid_phase0_preparation"_test = [] { test_valid_phase0_preparation(); };
    "temperature_greedy_switch"_test = [] { test_temperature_greedy_switch(); };
    "pixel_bounds"_test = [] { test_pixel_bounds(); };
    "precision_flags"_test = [] { test_precision_flags(); };
    "utf8_and_paths"_test = [] { test_utf8_and_paths(); };
    "status_report"_test = [] { test_status_report(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
