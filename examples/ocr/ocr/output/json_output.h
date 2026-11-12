#pragma once
// OCR engine -- full RecognizeResult JSON serialization (Phase 1). Host-only.
//
// Serializes the COMPLETE public schema (text, lines/blocks with coordinates,
// page/order/content/confidence presence, all model metadata/policies/router
// info, diagnostics and counters) plus a strict status/error form. Uses yyjson
// with correct ownership: every dynamic string/key is copied into the
// document; the writer buffer is freed with the matching allocator. Non-finite
// floats are rejected (never emitted as invalid JSON).
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>

#include "ocr/output/types.h"
#include "ocr/status.h"

namespace luisa::ocr::output {

struct JsonOutputOptions {
    bool pretty{true};
};

// Serialize a successful result. Returns io_error/invalid_argument on a
// non-finite float (rather than emitting invalid JSON).
[[nodiscard]] Result<luisa::string> serialize_result(const RecognizeResult &result,
                                                     const JsonOutputOptions &options = {});

// Serialize a failure status (the strict error form).
[[nodiscard]] Result<luisa::string> serialize_status(const Status &status,
                                                     const JsonOutputOptions &options = {});

// Write a result to a file (or stdout when the path is empty). Useful helper
// for the CLI and tools; uses serialize_result internally.
[[nodiscard]] Status write_result(const RecognizeResult &result,
                                  const luisa::filesystem::path &path,
                                  const JsonOutputOptions &options = {});

// Write a status report to a file (or stderr when the path is empty).
[[nodiscard]] Status write_status(const Status &status,
                                  const luisa::filesystem::path &path,
                                  const JsonOutputOptions &options = {});

}// namespace luisa::ocr::output