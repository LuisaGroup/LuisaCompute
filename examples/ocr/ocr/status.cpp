// OCR engine -- status code/severity name tables (Phase 0).
#include "ocr/status.h"

namespace luisa::ocr {

[[nodiscard]] luisa::string_view status_code_name(StatusCode code) noexcept {
    switch (code) {
        case StatusCode::ok: return "ok";
        case StatusCode::invalid_argument: return "invalid_argument";
        case StatusCode::unsupported: return "unsupported";
        case StatusCode::unvalidated: return "unvalidated";
        case StatusCode::not_found: return "not_found";
        case StatusCode::io_error: return "io_error";
        case StatusCode::out_of_memory: return "out_of_memory";
        case StatusCode::cancelled: return "cancelled";
        case StatusCode::timeout: return "timeout";
        case StatusCode::internal_error: return "internal_error";
    }
    return "unknown";
}

[[nodiscard]] luisa::string_view diagnostic_severity_name(DiagnosticSeverity severity) noexcept {
    switch (severity) {
        case DiagnosticSeverity::info: return "info";
        case DiagnosticSeverity::warning: return "warning";
        case DiagnosticSeverity::error: return "error";
    }
    return "unknown";
}

}// namespace luisa::ocr
