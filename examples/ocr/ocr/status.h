#pragma once
// OCR engine (examples/ocr) -- recoverable status/result contract (Phase 0).
//
// Ordinary failures (bad arguments, unsupported capabilities, unknown or
// unvalidated packages, resource-limit refusals, cancellation) are reported
// through Status/Result values, never through exceptions or LUISA_ERROR.
// LUISA_ASSERT/LUISA_ERROR remain reserved for broken internal invariants.
//
// This header is host-only: it must not include DSL/runtime device headers.
#include <cstdint>
#include <utility>

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::ocr {

// Coarse, stable error categories. `none` means success; every other code is
// recoverable and carries a human-readable message plus (where useful) the
// offending argument/field name.
enum class StatusCode : uint32_t {
    ok = 0,
    invalid_argument,          // malformed or contradictory caller input
    unsupported,               // valid request, but no validated path implements it
    unvalidated,               // artifact/recipe identity is not pinned & verified
    not_found,                 // required package/manifest/component is absent
    io_error,                  // reading/parsing an artifact failed
    out_of_memory,             // memory-limit preflight or allocation refusal
    cancelled,                 // caller cancellation callback fired
    timeout,                   // wall-time budget exceeded at a safe boundary
    internal_error,            // engine bug detected and contained, not a crash
};

// Stable lowercase names ("ok", "invalid_argument", ...). Unknown values
// render as "unknown" instead of asserting, so diagnostics stay recoverable.
[[nodiscard]] luisa::string_view status_code_name(StatusCode code) noexcept;

struct Status {
    StatusCode code{StatusCode::ok};
    luisa::string message;
    // Optional machine-readable detail: the argument, field, tensor, or
    // component the failure refers to ("" when not applicable).
    luisa::string detail;

    [[nodiscard]] bool ok() const noexcept { return code == StatusCode::ok; }

    [[nodiscard]] static Status success() { return {}; }
    [[nodiscard]] static Status error(StatusCode code, luisa::string message,
                                      luisa::string detail = {}) {
        return Status{code, std::move(message), std::move(detail)};
    }
};

// Severity of a non-fatal diagnostic attached to a successful result. Warnings
// such as truncation, output repair, offload, or fallback must be visible to
// callers; none of them silently changes model semantics.
enum class DiagnosticSeverity : uint32_t {
    info = 0,
    warning,
    error,
};

[[nodiscard]] luisa::string_view diagnostic_severity_name(DiagnosticSeverity severity) noexcept;

struct Diagnostic {
    DiagnosticSeverity severity{DiagnosticSeverity::info};
    luisa::string code;// stable machine-readable tag, e.g. "output_truncated"
    luisa::string message;
};

// Value-or-status result. On failure `value` is empty and `status` explains
// why; on success `status.ok()` holds. Copyable/movable like its members.
template<typename T>
struct Result {
    Status status;
    luisa::optional<T> value;

    [[nodiscard]] bool ok() const noexcept { return status.ok() && value.has_value(); }

    [[nodiscard]] static Result success(T v) {
        return Result{Status::success(), luisa::optional<T>{std::move(v)}};
    }
    [[nodiscard]] static Result failure(Status s) {
        return Result{std::move(s), luisa::nullopt};
    }
    [[nodiscard]] static Result failure(StatusCode code, luisa::string message,
                                        luisa::string detail = {}) {
        return failure(Status::error(code, std::move(message), std::move(detail)));
    }
};

}// namespace luisa::ocr
