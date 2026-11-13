// example_ocr -- command-line front end for the OCR engine.
//
// The CLI parses and validates the full surface, then fails closed: no
// model is validated, so the tool never claims inference succeeded and never
// fabricates output. --help and --list-models (the offline model support
// report, Phase 12) work without a device or model. Later phases wire the
// same arguments into Session::create()/recognize() unchanged; the success
// path already serializes via ocr/output/json_output.h (Phase 1).
//
// Encoding: Windows narrow argv is the ANSI code page, NOT UTF-8. On Windows
// we use wmain and convert every argument with WideCharToMultiByte(CP_UTF8)
// so BOTH paths and prompt/schema/language/guidance reach the parser as UTF-8.
// POSIX main passes UTF-8 bytes through unchanged. The parser (ocr/cli.h)
// then decodes paths with path_from_utf8. No ACP bytes are treated as UTF-8.
//
// Exit codes: 0 = help / --list-models; 1 = bad arguments; 2 = engine refusal
// (unvalidated/unsupported) reported as structured JSON; 3 = report I/O error.
#include <cstdio>
#include <cwchar>
#include <utility>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/api.h"
#include "ocr/cli.h"
#include "ocr/output/json_output.h"
#include "ocr/session.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace luisa::ocr;

namespace {

#ifdef _WIN32
// Convert one wide argument to UTF-8. Returns false on conversion failure
// (e.g. an unpaired surrogate); never throws and never silently substitutes
// ACP bytes. An empty argument converts to an empty string.
[[nodiscard]] bool wide_arg_to_utf8(const wchar_t *wide, luisa::string &out) noexcept {
    out.clear();
    if (wide == nullptr) { return true; }
    const auto length = static_cast<int>(::wcslen(wide));
    if (length == 0) { return true; }
    const int needed = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide, length,
                                             nullptr, 0, nullptr, nullptr);
    if (needed <= 0) { return false; }
    out.resize(static_cast<size_t>(needed));
    const int written = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide, length,
                                              out.data(), needed, nullptr, nullptr);
    if (written != needed) {
        out.clear();
        return false;
    }
    return true;
}
#endif

// Shared body: consumes UTF-8 argument bytes. argv[0] is the program name.
[[nodiscard]] int run_cli(int argc, const char *const *argv) {
    const auto parsed = parse_cli_args(argc, argv);
    if (!parsed.ok()) {
        const auto report = write_status_report(parsed.status, {});
        return report.ok() ? 1 : 3;
    }
    const CliArgs &args = *parsed.value;
    if (args.help) {
        const auto usage = cli_usage_text();
        std::fputs(usage.c_str(), stdout);
        return 0;
    }
    if (args.list_models) {
        // Offline qualification report (Phase 12): printed BEFORE any session
        // validation/device work; needs no backend, package, or input.
        const auto report = model_support_report_json();
        if (std::fputs(report.c_str(), stdout) < 0) { return 3; }
        return 0;
    }
    // Validate the full surface before touching any device/model.
    const auto config_status = validate_session_config(args.session);
    if (!config_status.ok()) {
        const auto report = write_status_report(config_status, args.output);
        return report.ok() ? 1 : 3;
    }
    const auto request_status = Session::preflight_request(args.pages, args.options);
    if (!request_status.ok()) {
        const auto report = write_status_report(request_status, args.output);
        return report.ok() ? 1 : 3;
    }
    // Report the resolved configuration, then the expected refusal.
    const auto bounds = effective_pixel_bounds(args.options);
    std::fprintf(stderr,
                 "example_ocr: backend=%s model=%s mode=%s precision=%s/%s/%s "
                 "overview_pixels=%llu crop_pixels=%llu tokens_region=%u tokens_page=%u\n",
                 args.session.backend.c_str(),
                 args.session.model.empty() ? "(from manifest)" : args.session.model.c_str(),
                 luisa::string{recognize_mode_name(args.options.mode)}.c_str(),
                 luisa::string{precision_name(args.options.precision.weights)}.c_str(),
                 luisa::string{precision_name(args.options.precision.activations)}.c_str(),
                 luisa::string{precision_name(args.options.precision.cache)}.c_str(),
                 static_cast<unsigned long long>(bounds.overview),
                 static_cast<unsigned long long>(bounds.crop),
                 args.options.max_output_tokens_per_region,
                 args.options.max_output_tokens_per_page);
    const auto session = Session::create(args.session);
    if (!session.ok()) {
        const auto report = write_status_report(session.status, args.output);
        return report.ok() ? 2 : 3;
    }
    // Unreachable today (create always refuses). Kept so later phases
    // only replace the refusal above.
    const auto result = (*session.value)->recognize(args.pages, args.options);
    if (!result.ok()) {
        const auto report = write_status_report(result.status, args.output);
        return report.ok() ? 2 : 3;
    }
    // Serialize the FULL RecognizeResult schema (ocr/output/json_output.h) and
    // write it to args.output/stdout. This path is reached only when a later
    // phase produces a real result; Phase 0/1 never fabricates OCR output.
    const auto write = output::write_result(*result.value, args.output);
    if (!write.ok()) {
        const auto report = write_status_report(write, args.output);
        return report.ok() ? 3 : 3;
    }
    return 0;
}

}// namespace

#ifdef _WIN32
int wmain(int argc, wchar_t *argv[]) {
    luisa::vector<luisa::string> utf8_storage;
    utf8_storage.reserve(static_cast<size_t>(argc));
    luisa::vector<const char *> utf8_argv;
    utf8_argv.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; i++) {
        luisa::string converted;
        if (!wide_arg_to_utf8(argv[i], converted)) {
            const auto status = Status::error(StatusCode::invalid_argument,
                                              "command-line argument is not representable as UTF-8", "argv");
            const auto report = write_status_report(status, {});
            return report.ok() ? 1 : 3;
        }
        utf8_storage.emplace_back(std::move(converted));
        utf8_argv.emplace_back(utf8_storage.back().c_str());
    }
    return run_cli(argc, utf8_argv.data());
}
#else
int main(int argc, char *argv[]) {
    return run_cli(argc, argv);
}
#endif
