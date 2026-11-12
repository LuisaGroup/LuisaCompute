#pragma once
// OCR engine -- schema-directed JSON extraction (Phase 4). Host-only.
//
// Parses a generative VLM's JSON output against a DELIBERATELY SUPPORTED
// JSON-schema subset. Unknown schema features FAIL CLOSED (unsupported),
// never silently ignored; a non-conforming document is an ERROR, never a
// plain-text fallback. JSON repair is OFF by default; when explicitly enabled
// it produces diagnostics for every repair it applies.
//
// Supported schema subset (documented, frozen for Phase 4):
//   type: "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
//   object: properties, required, additionalProperties (bool or schema)
//   array: items (single schema), minItems, maxItems
//   string: minLength, maxLength, enum
//   number/integer: minimum, maximum, enum
//   boolean/null: (no keywords beyond type)
//   any schema: enum, const, title, description (annotations ignored)
//   $ref: NOT supported (fail closed)
//   allOf/anyOf/oneOf/not, pattern, patternProperties, format, dependent*,
//   if/then/else, unevaluated*, prefixItems, contains, multipleOf,
//   uniqueItems, propertyNames, minProperties/maxProperties: NOT supported.
//
// The parser is strict JSON (RFC 8259): no trailing commas, no comments, no
// NaN/Infinity, no single quotes, no unquoted keys. Duplicate object keys are
// rejected. Numbers must be finite.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/status.h"

namespace luisa::ocr::output {

// A parsed JSON value (owning tree). Object key order is preserved.
struct JsonValue {
    enum class Kind : uint32_t {
        null = 0,
        boolean,
        number,
        string,
        array,
        object,
    };
    Kind kind{Kind::null};
    bool bool_value{false};
    double number_value{0.0};
    luisa::string string_value;
    luisa::vector<JsonValue> array_items;
    // Object members in source order.
    luisa::vector<luisa::string> object_keys;
    luisa::vector<JsonValue> object_values;

    [[nodiscard]] const JsonValue *find(luisa::string_view key) const noexcept;
};

// Strict JSON parse (no repair). Fails closed on any syntax violation.
[[nodiscard]] Result<JsonValue> parse_json_strict(luisa::string_view text);

// JSON repair options. DEFAULT OFF. When enabled, ONLY these repairs are
// attempted, each reported as a diagnostic:
//   * strip a single leading/trailing Markdown code fence (```json ... ```);
//   * strip leading/trailing whitespace;
//   * remove trailing commas before } or ].
// Nothing else is repaired (no quote fixing, no key quoting, no NaN).
struct JsonRepairOptions {
    bool enabled{false};
};

struct JsonParseResult {
    JsonValue value;
    luisa::vector<Diagnostic> diagnostics;// repairs applied (empty when none)
    bool repaired{false};
};

// Parse JSON with the configured repair policy. When repair is disabled this
// is exactly parse_json_strict. When enabled, repairs are applied and EACH
// one is reported; if the result still does not parse, the error is returned
// (never a silent fallback).
[[nodiscard]] Result<JsonParseResult> parse_json(luisa::string_view text,
                                                 const JsonRepairOptions &repair = {});

// Validate a parsed schema document against the supported subset. Returns
// unsupported for any unknown/unsupported keyword, invalid_argument for a
// malformed schema. This is a SEPARATE step from document validation so a
// caller can preflight a schema once.
[[nodiscard]] Status validate_extraction_schema(const JsonValue &schema);

// Validate a parsed JSON document against a (already validated) schema.
// Fails closed (invalid_argument) on any violation with a path-qualified
// message. `path` is the root path label (e.g. "$").
[[nodiscard]] Status validate_json_against_schema(const JsonValue &value, const JsonValue &schema,
                                                  luisa::string_view path = "$");

// One-shot: parse the schema text, validate the subset, parse the document
// (with the repair policy), and validate the document. Fails closed at the
// first violation; never returns a partial/plain-text result.
struct ExtractionResult {
    JsonValue value;
    luisa::vector<Diagnostic> diagnostics;
    bool repaired{false};
};
[[nodiscard]] Result<ExtractionResult> extract_json(luisa::string_view document,
                                                    luisa::string_view schema,
                                                    const JsonRepairOptions &repair = {});

// Serialize a JsonValue back to canonical strict JSON (used for the canonical
// RecognizeResult::text in extract mode). Object key order is preserved.
[[nodiscard]] Result<luisa::string> serialize_json(const JsonValue &value);

}// namespace luisa::ocr::output
