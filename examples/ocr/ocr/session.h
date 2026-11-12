#pragma once
// OCR engine -- Session orchestration contract (frozen in Phase 0;
// unchanged in Phase 1).
//
// One Session conceptually owns ONE model (and its declared dependent
// components). Phase 0 froze the creation/recognition surface and its
// argument validation, then fails closed: no recipe is validated, so
// create() always refuses after validating its arguments and recognize()
// never produces output. There is no mock OCR path.
//
// Host-only header: no DSL/runtime device includes. Later phases add device
// binding through additional create() overloads without changing these types.
#include <cstdint>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>

#include "ocr/api.h"
#include "ocr/model_registry.h"
#include "ocr/status.h"

namespace luisa::ocr {

struct SessionConfig {
    // Backend plugin name ("cuda", "vk", "dx", "fallback", ...). Validated as
    // a non-empty identifier here; actual device creation happens in later
    // phases and reports recoverable failures.
    luisa::string backend;
    // Packed .lcocr package and/or its manifest. At least one must be set;
    // The session does not read either file yet: Phase 1 adds host readers
    // (ocr/package.*) but no recipe is validated, so none is wired in.
    luisa::filesystem::path package;
    luisa::filesystem::path manifest;
    // Optional explicit target selector ("pp_ocrv6:medium", "glm_ocr", ...).
    // When empty, later phases derive it from the manifest -- never from an
    // outer model_type alone.
    luisa::string model;
    // Registry policy; the unsafe experimental override remains unvalidated.
    RegistryPolicy registry_policy;
    // Session-level default precision policy (per-request overrides allowed).
    PrecisionPolicy precision;
};

class Session {
public:
    // Validates the configuration, then fails closed with
    // StatusCode::unvalidated because no model is validated yet.
    // Argument errors are reported as invalid_argument BEFORE the
    // unvalidated refusal, so callers can distinguish bad input from the
    // (expected) support gap.
    [[nodiscard]] static Result<luisa::unique_ptr<Session>> create(const SessionConfig &config);

    ~Session() noexcept;
    Session(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(const Session &) = delete;
    Session &operator=(Session &&) = delete;

    // Preflights the request (validate_request), honors cancellation, then
    // fails closed: no validated model exists, so no inference is attempted
    // and no output is fabricated.
    [[nodiscard]] Result<RecognizeResult> recognize(
        luisa::span<const PageInput> pages, const RecognizeOptions &options);

    // Host-side request preflight shared by recognize(). Exposed so callers
    // and tests can validate pages/options without a device or model:
    // - at least one page, every page path non-empty;
    // - structural option validation (validate_recognize_options);
    // - extract without schema/prompt is rejected here as a generic rule
    //   (recipes may relax it only through their declared capabilities).
    [[nodiscard]] static Status validate_request(
        luisa::span<const PageInput> pages, const RecognizeOptions &options);

    // Full host-side preflight used by recognize(): validate_request() FIRST,
    // then the cancellation callback. Ordering is contractual -- an invalid
    // request reports invalid_argument even when cancellation would also fire;
    // a valid-but-cancelled request reports cancelled. No device/model needed.
    [[nodiscard]] static Status preflight_request(
        luisa::span<const PageInput> pages, const RecognizeOptions &options);

    [[nodiscard]] const ModelMetadata &metadata() const noexcept { return _metadata; }

private:
    Session() noexcept;
    ModelMetadata _metadata;
};

// Structural validation of a session configuration (no device, no files).
[[nodiscard]] Status validate_session_config(const SessionConfig &config);

}// namespace luisa::ocr
