#pragma once
// OCR engine -- strict Markdown block composition/validation (Phase 4).
// Host-only.
//
// Parses a generative VLM's Markdown output into ordered blocks (text, title,
// table, formula, figure, list, ...) with STRICT structural validation:
//   * tables must be valid HTML or OTSL (output/table.h) -- a malformed table
//     is an ERROR, never silently downgraded to text;
//   * formulas must pass LaTeX delimiter/content sanity (check_latex);
//   * fenced code blocks must be closed;
//   * heading levels must be 1..6;
//   * unknown/unsupported constructs fail closed (no plain-text fallback).
//
// Source regions / reading order: blocks are produced in document order; the
// caller attaches SourceRegion/reading_order when composing a RecognizeResult
// (the parser preserves the order it was given). No confidence is produced.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/output/types.h"
#include "ocr/status.h"

namespace luisa::ocr::output {

// One parsed Markdown block. `content` is the canonical block text (for a
// table: the original HTML/OTSL source; for a formula: the LaTeX body WITHOUT
// the outer delimiters). `raw` is the exact source span.
struct MarkdownBlock {
    BlockType type{BlockType::text};
    luisa::string content;
    luisa::string raw;
    // Heading level (1..6) for title blocks; 0 otherwise.
    uint32_t heading_level{0};
    // Source span in the input text (byte offsets, [begin, end)).
    uint64_t begin{0};
    uint64_t end{0};
};

struct MarkdownParseOptions {
    // When true, a table block must parse as HTML or OTSL (strict). When
    // false, table-looking blocks are still detected but not validated
    // (NOT recommended; the default is strict).
    bool validate_tables{true};
    // When true, formula blocks must pass LaTeX sanity (strict).
    bool validate_formulas{true};
};

struct MarkdownParseResult {
    luisa::vector<MarkdownBlock> blocks;
    // Non-fatal diagnostics (e.g. an empty document). Errors are returned as
    // a failing Status, never as a diagnostic.
    luisa::vector<Diagnostic> diagnostics;
};

// Parse + validate a Markdown document. Fails closed (invalid_argument) on
// any structural violation; never returns a partial result on error.
[[nodiscard]] Result<MarkdownParseResult> parse_markdown(luisa::string_view text,
                                                         const MarkdownParseOptions &options = {});

// Compose ordered blocks back into canonical Markdown (block content joined
// by blank lines; titles rendered as '#' * level + content; tables/formulas
// re-wrapped in their delimiters). Used to produce RecognizeResult::text.
[[nodiscard]] Result<luisa::string> compose_markdown(luisa::span<const MarkdownBlock> blocks);

// LaTeX delimiter/content sanity check. Accepts $...$, $$...$$, \(...\),
// \[...\] as the OUTER delimiters of `body_with_delims`; the inner content
// must be non-empty, must not contain an unescaped outer delimiter, must have
// balanced braces, and must not contain a nested math fence. Returns
// invalid_argument with a specific message on violation.
[[nodiscard]] Status check_latex(luisa::string_view body_with_delims);

// Strip the outer LaTeX delimiters, returning the inner body. Fails closed
// when the delimiters are not a recognized pair.
[[nodiscard]] Result<luisa::string> strip_latex_delimiters(luisa::string_view body_with_delims);

}// namespace luisa::ocr::output
