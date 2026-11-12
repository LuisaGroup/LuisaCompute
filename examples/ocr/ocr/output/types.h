#pragma once
// OCR engine -- public output types (Phase 1). Host-only.
//
// The result schema is the Phase 0 RecognizeResult contract (ocr/api.h); this
// header re-exports it under output/ and adds the serialization surface. The
// JSON serializer (json_output.h) writes the FULL schema, never a subset.
#include "ocr/api.h"
#include "ocr/status.h"

namespace luisa::ocr::output {

// Re-export the frozen public result types so output consumers include one
// header. These are the SAME types the session produces (no parallel schema).
using luisa::ocr::BlockType;
using luisa::ocr::Diagnostic;
using luisa::ocr::DiagnosticSeverity;
using luisa::ocr::ModelMetadata;
using luisa::ocr::OcrBlock;
using luisa::ocr::OcrLine;
using luisa::ocr::PerformanceCounters;
using luisa::ocr::RecognizeResult;
using luisa::ocr::RouterMetadata;
using luisa::ocr::SourceBox;
using luisa::ocr::SourcePoint;
using luisa::ocr::SourceRegion;

}// namespace luisa::ocr::output