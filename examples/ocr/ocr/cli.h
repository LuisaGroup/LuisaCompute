#pragma once
// OCR engine -- reusable command-line surface (Phase 0 contract;
// Phase 12 adds the offline --list-models support report).
//
// The argument parser and the structured status report live here so they can
// be exercised by host tests without a device or model. The parser consumes
// UTF-8 argument bytes; example_ocr.cpp is responsible for producing UTF-8
// argv (Windows wmain + WideCharToMultiByte, POSIX main passes UTF-8 through).
//
// Host-only header: no DSL/runtime device includes.
#include <cstdint>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/api.h"
#include "ocr/session.h"
#include "ocr/status.h"

namespace luisa::ocr {

// Fully parsed command line. `help` and `list_models` short-circuit
// everything else (no backend/package/input required).
struct CliArgs {
    SessionConfig session;
    RecognizeOptions options;
    luisa::vector<PageInput> pages;
    luisa::filesystem::path output;
    bool help{false};
    // Phase 12: --list-models prints the registry-driven model support report
    // (model_support_report_json) and exits 0 without touching any device or
    // model artifact.
    bool list_models{false};
};

// Parse UTF-8 argument bytes (argv[0] is the program name and is skipped).
// Returns invalid_argument on any malformed/unknown argument; `help` is set
// when --help/-h appears and `list_models` when --list-models appears. Path
// arguments are decoded with path_from_utf8 and rejected if not valid UTF-8.
// No ACP bytes are ever treated as UTF-8 here.
[[nodiscard]] Result<CliArgs> parse_cli_args(int argc, const char *const *argv);

// Usage text (also used by --help). Written to stdout by the caller.
[[nodiscard]] luisa::string cli_usage_text();

// Minimal JSON string escaping (UTF-8 passthrough; control chars, quotes and
// backslashes escaped). Exposed for tests.
[[nodiscard]] luisa::string json_escape(luisa::string_view text);

// Serialize a failure Status as the structured refusal JSON the CLI emits.
[[nodiscard]] luisa::string status_report_json(const Status &status);

// Phase 12 offline qualification surface: registry-driven model support
// report as valid JSON. Emits target_count (17 selectable targets), one
// entry per target with target/model/family/support_state/validated, and the
// explicit global real_model_validated=false / session_status="unvalidated".
// Synthetic host/device parity does NOT make any entry validated; the report
// is derived from the same default planned table Session will consult once
// real loading is enabled, and needs no device, package, or network.
[[nodiscard]] luisa::string model_support_report_json();

// Write a status report to `output` (or stderr when empty). Returns ok on
// success; an I/O failure is reported as io_error rather than swallowed.
[[nodiscard]] Status write_status_report(const Status &status,
                                         const luisa::filesystem::path &output);

}// namespace luisa::ocr
