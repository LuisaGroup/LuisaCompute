// OCR engine -- schema-directed JSON extraction (Phase 4). Host-only.
// Strict JSON parse + deliberately supported schema subset; repair OFF by
// default with diagnostics. See extract.h.
#include "ocr/output/extract.h"

#include <cmath>
#include <cstring>
#include <utility>

#include <luisa/core/stl/format.h>

namespace luisa::ocr::output {

[[nodiscard]] const JsonValue *JsonValue::find(luisa::string_view key) const noexcept {
    if (kind != Kind::object) { return nullptr; }
    for (size_t i = 0; i < object_keys.size(); i++) {
        if (object_keys[i] == key) { return &object_values[i]; }
    }
    return nullptr;
}

namespace {

// ---------------------------------------------------------------------------
// Strict JSON parser (recursive descent, depth-bounded).
// ---------------------------------------------------------------------------

class JsonParser {
public:
    explicit JsonParser(luisa::string_view text) noexcept : _text(text) {}

    [[nodiscard]] Result<JsonValue> parse() {
        using R = Result<JsonValue>;
        skip_ws();
        auto v = parse_value(0);
        if (!v.ok()) { return R::failure(v.status); }
        skip_ws();
        if (_pos != _text.size()) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("JSON has trailing content at byte {}", _pos), "json");
        }
        return R::success(std::move(*v.value));
    }

private:
    static constexpr uint32_t kMaxDepth = 64;

    [[nodiscard]] bool eof() const noexcept { return _pos >= _text.size(); }
    [[nodiscard]] char peek() const noexcept { return _text[_pos]; }

    void skip_ws() noexcept {
        while (!eof()) {
            const char c = peek();
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { _pos++; }
            else { break; }
        }
    }

    [[nodiscard]] Status fail(luisa::string msg) const {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("{} at byte {}", msg, _pos), "json");
    }

    [[nodiscard]] Result<JsonValue> parse_value(uint32_t depth) {
        using R = Result<JsonValue>;
        if (depth > kMaxDepth) {
            return R::failure(fail("JSON nesting is too deep"));
        }
        if (eof()) { return R::failure(fail("JSON value is missing")); }
        const char c = peek();
        switch (c) {
            case '{': return parse_object(depth);
            case '[': return parse_array(depth);
            case '"': {
                auto s = parse_string();
                if (!s.ok()) { return R::failure(s.status); }
                JsonValue v;
                v.kind = JsonValue::Kind::string;
                v.string_value = std::move(*s.value);
                return R::success(std::move(v));
            }
            case 't': return parse_literal("true", JsonValue::Kind::boolean, true);
            case 'f': return parse_literal("false", JsonValue::Kind::boolean, false);
            case 'n': return parse_literal("null", JsonValue::Kind::null, false);
            default:
                if (c == '-' || (c >= '0' && c <= '9')) { return parse_number(); }
                return R::failure(fail("JSON value has an unexpected character"));
        }
    }

    [[nodiscard]] Result<JsonValue> parse_literal(luisa::string_view lit, JsonValue::Kind kind, bool b) {
        using R = Result<JsonValue>;
        if (_text.compare(_pos, lit.size(), lit) != 0) {
            return R::failure(fail("JSON literal is malformed"));
        }
        _pos += lit.size();
        JsonValue v;
        v.kind = kind;
        v.bool_value = b;
        return R::success(std::move(v));
    }

    [[nodiscard]] Result<JsonValue> parse_number() {
        using R = Result<JsonValue>;
        const size_t start = _pos;
        if (!eof() && peek() == '-') { _pos++; }
        // Integer part: 0 or [1-9][0-9]*.
        if (eof()) { return R::failure(fail("JSON number is truncated")); }
        if (peek() == '0') {
            _pos++;
        } else if (peek() >= '1' && peek() <= '9') {
            while (!eof() && peek() >= '0' && peek() <= '9') { _pos++; }
        } else {
            return R::failure(fail("JSON number has an invalid integer part"));
        }
        // Fraction.
        if (!eof() && peek() == '.') {
            _pos++;
            const size_t frac_start = _pos;
            while (!eof() && peek() >= '0' && peek() <= '9') { _pos++; }
            if (_pos == frac_start) { return R::failure(fail("JSON number has an empty fraction")); }
        }
        // Exponent.
        if (!eof() && (peek() == 'e' || peek() == 'E')) {
            _pos++;
            if (!eof() && (peek() == '+' || peek() == '-')) { _pos++; }
            const size_t exp_start = _pos;
            while (!eof() && peek() >= '0' && peek() <= '9') { _pos++; }
            if (_pos == exp_start) { return R::failure(fail("JSON number has an empty exponent")); }
        }
        const auto num_str = _text.substr(start, _pos - start);
        // Parse to double (checked finite).
        char *end = nullptr;
        const double value = std::strtod(luisa::string{num_str}.c_str(), &end);
        if (!std::isfinite(value)) {
            return R::failure(fail("JSON number is not finite"));
        }
        JsonValue v;
        v.kind = JsonValue::Kind::number;
        v.number_value = value;
        return R::success(std::move(v));
    }

    // Parse a JSON string into UTF-8 (validating escapes / surrogate pairs).
    [[nodiscard]] Result<luisa::string> parse_string() {
        using R = Result<luisa::string>;
        if (eof() || peek() != '"') { return R::failure(fail("JSON string is not quoted")); }
        _pos++;
        luisa::string out;
        while (!eof()) {
            const char c = peek();
            if (c == '"') {
                _pos++;
                return R::success(std::move(out));
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                return R::failure(fail("JSON string contains an unescaped control character"));
            }
            if (c == '\\') {
                _pos++;
                if (eof()) { return R::failure(fail("JSON string escape is truncated")); }
                const char e = peek();
                _pos++;
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        auto cp = parse_hex4();
                        if (!cp.ok()) { return R::failure(cp.status); }
                        uint32_t code = *cp.value;
                        if (code >= 0xD800 && code <= 0xDBFF) {
                            // High surrogate: require a low surrogate.
                            if (_pos + 1 >= _text.size() || _text[_pos] != '\\' || _text[_pos + 1] != 'u') {
                                return R::failure(fail("JSON string has an unpaired high surrogate"));
                            }
                            _pos += 2;
                            auto lo = parse_hex4();
                            if (!lo.ok()) { return R::failure(lo.status); }
                            if (*lo.value < 0xDC00 || *lo.value > 0xDFFF) {
                                return R::failure(fail("JSON string has an invalid low surrogate"));
                            }
                            code = 0x10000 + ((code - 0xD800) << 10) + (*lo.value - 0xDC00);
                        } else if (code >= 0xDC00 && code <= 0xDFFF) {
                            return R::failure(fail("JSON string has an unpaired low surrogate"));
                        }
                        append_utf8(out, code);
                        break;
                    }
                    default:
                        return R::failure(fail("JSON string has an invalid escape"));
                }
                continue;
            }
            out += c;
            _pos++;
        }
        return R::failure(fail("JSON string is not closed"));
    }

    [[nodiscard]] Result<uint32_t> parse_hex4() {
        using R = Result<uint32_t>;
        if (_pos + 4 > _text.size()) { return R::failure(fail("JSON \\u escape is truncated")); }
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) {
            const char c = _text[_pos++];
            uint32_t d = 0;
            if (c >= '0' && c <= '9') { d = c - '0'; }
            else if (c >= 'a' && c <= 'f') { d = c - 'a' + 10; }
            else if (c >= 'A' && c <= 'F') { d = c - 'A' + 10; }
            else { return R::failure(fail("JSON \\u escape has a non-hex digit")); }
            v = v * 16 + d;
        }
        return R::success(v);
    }

    static void append_utf8(luisa::string &out, uint32_t code) {
        if (code < 0x80) {
            out += static_cast<char>(code);
        } else if (code < 0x800) {
            out += static_cast<char>(0xC0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else if (code < 0x10000) {
            out += static_cast<char>(0xE0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (code >> 18));
            out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        }
    }

    [[nodiscard]] Result<JsonValue> parse_array(uint32_t depth) {
        using R = Result<JsonValue>;
        _pos++;// '['
        JsonValue v;
        v.kind = JsonValue::Kind::array;
        skip_ws();
        if (!eof() && peek() == ']') {
            _pos++;
            return R::success(std::move(v));
        }
        while (true) {
            skip_ws();
            auto item = parse_value(depth + 1);
            if (!item.ok()) { return R::failure(item.status); }
            v.array_items.push_back(std::move(*item.value));
            skip_ws();
            if (eof()) { return R::failure(fail("JSON array is not closed")); }
            if (peek() == ']') {
                _pos++;
                return R::success(std::move(v));
            }
            if (peek() != ',') { return R::failure(fail("JSON array expects ',' or ']'")); }
            _pos++;
            skip_ws();
            if (!eof() && peek() == ']') {
                return R::failure(fail("JSON array has a trailing comma"));
            }
        }
    }

    [[nodiscard]] Result<JsonValue> parse_object(uint32_t depth) {
        using R = Result<JsonValue>;
        _pos++;// '{'
        JsonValue v;
        v.kind = JsonValue::Kind::object;
        skip_ws();
        if (!eof() && peek() == '}') {
            _pos++;
            return R::success(std::move(v));
        }
        while (true) {
            skip_ws();
            if (eof() || peek() != '"') { return R::failure(fail("JSON object key is not a quoted string")); }
            auto key = parse_string();
            if (!key.ok()) { return R::failure(key.status); }
            // Duplicate keys are rejected.
            for (const auto &k : v.object_keys) {
                if (k == *key.value) {
                    return R::failure(fail(luisa::format("JSON object has a duplicate key '{}'", *key.value)));
                }
            }
            skip_ws();
            if (eof() || peek() != ':') { return R::failure(fail("JSON object expects ':' after a key")); }
            _pos++;
            skip_ws();
            auto val = parse_value(depth + 1);
            if (!val.ok()) { return R::failure(val.status); }
            v.object_keys.push_back(std::move(*key.value));
            v.object_values.push_back(std::move(*val.value));
            skip_ws();
            if (eof()) { return R::failure(fail("JSON object is not closed")); }
            if (peek() == '}') {
                _pos++;
                return R::success(std::move(v));
            }
            if (peek() != ',') { return R::failure(fail("JSON object expects ',' or '}'")); }
            _pos++;
            skip_ws();
            if (!eof() && peek() == '}') {
                return R::failure(fail("JSON object has a trailing comma"));
            }
        }
    }

    luisa::string_view _text;
    size_t _pos{0};
};

// ---------------------------------------------------------------------------
// Schema subset
// ---------------------------------------------------------------------------

[[nodiscard]] bool is_schema_type(luisa::string_view t) noexcept {
    return t == "object" || t == "array" || t == "string" || t == "number" ||
           t == "integer" || t == "boolean" || t == "null";
}

// Keywords supported per type (plus the universal annotations/enum/const).
[[nodiscard]] bool keyword_supported(luisa::string_view type, luisa::string_view key) noexcept {
    // Universal.
    if (key == "type" || key == "enum" || key == "const" || key == "title" || key == "description") {
        return true;
    }
    if (type == "object") {
        return key == "properties" || key == "required" || key == "additionalProperties";
    }
    if (type == "array") {
        return key == "items" || key == "minItems" || key == "maxItems";
    }
    if (type == "string") {
        return key == "minLength" || key == "maxLength";
    }
    if (type == "number" || type == "integer") {
        return key == "minimum" || key == "maximum";
    }
    return false;
}

[[nodiscard]] Status validate_schema_node(const JsonValue &schema, luisa::string_view path, uint32_t depth) {
    if (depth > 64) {
        return Status::error(StatusCode::invalid_argument, "schema nesting is too deep", luisa::string{path});
    }
    if (schema.kind != JsonValue::Kind::object) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("schema at {} must be an object", path), luisa::string{path});
    }
    // Determine the declared type (default: any supported type is allowed
    // only when no type-specific keyword is present; we require an explicit
    // type for clarity, but allow its absence for pure enum/const schemas).
    luisa::string type;
    if (const auto *t = schema.find("type"); t != nullptr) {
        if (t->kind != JsonValue::Kind::string) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("schema at {} 'type' must be a string", path), luisa::string{path});
        }
        type = t->string_value;
        if (!is_schema_type(type)) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("schema at {} has unsupported type '{}'", path, type), luisa::string{path});
        }
    }
    for (size_t i = 0; i < schema.object_keys.size(); i++) {
        const auto &key = schema.object_keys[i];
        const auto &val = schema.object_values[i];
        if (!keyword_supported(type, key)) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("schema at {} uses unsupported keyword '{}'", path, key),
                                 luisa::string{path});
        }
        const auto child_path = luisa::format("{}.{}", path, key);
        if (key == "properties") {
            if (val.kind != JsonValue::Kind::object) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::format("schema at {} 'properties' must be an object", path), luisa::string{path});
            }
            for (size_t k = 0; k < val.object_keys.size(); k++) {
                auto s = validate_schema_node(val.object_values[k],
                                              luisa::format("{}.{}", child_path, val.object_keys[k]), depth + 1);
                if (!s.ok()) { return s; }
            }
        } else if (key == "required") {
            if (val.kind != JsonValue::Kind::array) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::format("schema at {} 'required' must be an array", path), luisa::string{path});
            }
            for (const auto &item : val.array_items) {
                if (item.kind != JsonValue::Kind::string) {
                    return Status::error(StatusCode::invalid_argument,
                                         luisa::format("schema at {} 'required' entries must be strings", path),
                                         luisa::string{path});
                }
            }
        } else if (key == "additionalProperties") {
            if (val.kind == JsonValue::Kind::boolean) {
                // ok
            } else if (val.kind == JsonValue::Kind::object) {
                auto s = validate_schema_node(val, child_path, depth + 1);
                if (!s.ok()) { return s; }
            } else {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::format("schema at {} 'additionalProperties' must be a boolean or schema", path),
                                     luisa::string{path});
            }
        } else if (key == "items") {
            if (val.kind != JsonValue::Kind::object) {
                return Status::error(StatusCode::unsupported,
                                     luisa::format("schema at {} 'items' must be a single schema (tuple items unsupported)", path),
                                     luisa::string{path});
            }
            auto s = validate_schema_node(val, child_path, depth + 1);
            if (!s.ok()) { return s; }
        } else if (key == "minItems" || key == "maxItems" || key == "minLength" || key == "maxLength") {
            if (val.kind != JsonValue::Kind::number || val.number_value < 0.0 ||
                val.number_value != std::floor(val.number_value)) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::format("schema at {} '{}' must be a nonnegative integer", path, key),
                                     luisa::string{path});
            }
        } else if (key == "minimum" || key == "maximum") {
            if (val.kind != JsonValue::Kind::number) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::format("schema at {} '{}' must be a number", path, key), luisa::string{path});
            }
        } else if (key == "enum") {
            if (val.kind != JsonValue::Kind::array || val.array_items.empty()) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::format("schema at {} 'enum' must be a non-empty array", path), luisa::string{path});
            }
        }
        // const/title/description: annotations, no structural check.
    }
    // Cross-keyword consistency.
    if (const auto *mn = schema.find("minItems"); mn != nullptr) {
        if (const auto *mx = schema.find("maxItems"); mx != nullptr) {
            if (mn->number_value > mx->number_value) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::format("schema at {} minItems > maxItems", path), luisa::string{path});
            }
        }
    }
    if (const auto *mn = schema.find("minLength"); mn != nullptr) {
        if (const auto *mx = schema.find("maxLength"); mx != nullptr) {
            if (mn->number_value > mx->number_value) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::format("schema at {} minLength > maxLength", path), luisa::string{path});
            }
        }
    }
    if (const auto *mn = schema.find("minimum"); mn != nullptr) {
        if (const auto *mx = schema.find("maximum"); mx != nullptr) {
            if (mn->number_value > mx->number_value) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::format("schema at {} minimum > maximum", path), luisa::string{path});
            }
        }
    }
    return Status::success();
}

// UTF-8 codepoint count (for string minLength/maxLength).
[[nodiscard]] uint64_t utf8_length(luisa::string_view s) noexcept {
    uint64_t n = 0;
    for (size_t i = 0; i < s.size(); i++) {
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) { n++; }
    }
    return n;
}

[[nodiscard]] bool json_equal(const JsonValue &a, const JsonValue &b) noexcept {
    if (a.kind != b.kind) { return false; }
    switch (a.kind) {
        case JsonValue::Kind::null: return true;
        case JsonValue::Kind::boolean: return a.bool_value == b.bool_value;
        case JsonValue::Kind::number: return a.number_value == b.number_value;
        case JsonValue::Kind::string: return a.string_value == b.string_value;
        case JsonValue::Kind::array: {
            if (a.array_items.size() != b.array_items.size()) { return false; }
            for (size_t i = 0; i < a.array_items.size(); i++) {
                if (!json_equal(a.array_items[i], b.array_items[i])) { return false; }
            }
            return true;
        }
        case JsonValue::Kind::object: {
            if (a.object_keys.size() != b.object_keys.size()) { return false; }
            for (size_t i = 0; i < a.object_keys.size(); i++) {
                const auto *bv = b.find(a.object_keys[i]);
                if (bv == nullptr || !json_equal(a.object_values[i], *bv)) { return false; }
            }
            return true;
        }
    }
    return false;
}

[[nodiscard]] Status validate_node(const JsonValue &value, const JsonValue &schema,
                                   luisa::string_view path, uint32_t depth) {
    if (depth > 64) {
        return Status::error(StatusCode::invalid_argument, "JSON nesting is too deep", luisa::string{path});
    }
    const auto fail = [&](luisa::string msg) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("{}: {}", path, msg), luisa::string{path});
    };
    // const / enum.
    if (const auto *c = schema.find("const"); c != nullptr) {
        if (!json_equal(value, *c)) { return fail("value does not match const"); }
    }
    if (const auto *e = schema.find("enum"); e != nullptr) {
        bool found = false;
        for (const auto &item : e->array_items) {
            if (json_equal(value, item)) { found = true; break; }
        }
        if (!found) { return fail("value is not in enum"); }
    }
    const auto *type_val = schema.find("type");
    if (type_val == nullptr) { return Status::success(); }// enum/const-only schema
    const auto &type = type_val->string_value;

    if (type == "object") {
        if (value.kind != JsonValue::Kind::object) { return fail("expected an object"); }
        // required.
        if (const auto *req = schema.find("required"); req != nullptr) {
            for (const auto &r : req->array_items) {
                if (value.find(r.string_value) == nullptr) {
                    return fail(luisa::format("missing required property '{}'", r.string_value));
                }
            }
        }
        const auto *props = schema.find("properties");
        const auto *additional = schema.find("additionalProperties");
        for (size_t i = 0; i < value.object_keys.size(); i++) {
            const auto &key = value.object_keys[i];
            const auto &val = value.object_values[i];
            const JsonValue *prop_schema = nullptr;
            if (props != nullptr) { prop_schema = props->find(key); }
            const auto child_path = luisa::format("{}.{}", path, key);
            if (prop_schema != nullptr) {
                auto s = validate_node(val, *prop_schema, child_path, depth + 1);
                if (!s.ok()) { return s; }
            } else if (additional != nullptr) {
                if (additional->kind == JsonValue::Kind::boolean) {
                    if (!additional->bool_value) {
                        return fail(luisa::format("additional property '{}' is not allowed", key));
                    }
                } else {
                    auto s = validate_node(val, *additional, child_path, depth + 1);
                    if (!s.ok()) { return s; }
                }
            }
            // No additionalProperties keyword => additional allowed (default).
        }
        return Status::success();
    }
    if (type == "array") {
        if (value.kind != JsonValue::Kind::array) { return fail("expected an array"); }
        if (const auto *mn = schema.find("minItems"); mn != nullptr) {
            if (static_cast<double>(value.array_items.size()) < mn->number_value) {
                return fail("array has fewer than minItems elements");
            }
        }
        if (const auto *mx = schema.find("maxItems"); mx != nullptr) {
            if (static_cast<double>(value.array_items.size()) > mx->number_value) {
                return fail("array has more than maxItems elements");
            }
        }
        if (const auto *items = schema.find("items"); items != nullptr) {
            for (size_t i = 0; i < value.array_items.size(); i++) {
                auto s = validate_node(value.array_items[i], *items,
                                       luisa::format("{}[{}]", path, i), depth + 1);
                if (!s.ok()) { return s; }
            }
        }
        return Status::success();
    }
    if (type == "string") {
        if (value.kind != JsonValue::Kind::string) { return fail("expected a string"); }
        const auto len = utf8_length(value.string_value);
        if (const auto *mn = schema.find("minLength"); mn != nullptr) {
            if (static_cast<double>(len) < mn->number_value) { return fail("string is shorter than minLength"); }
        }
        if (const auto *mx = schema.find("maxLength"); mx != nullptr) {
            if (static_cast<double>(len) > mx->number_value) { return fail("string is longer than maxLength"); }
        }
        return Status::success();
    }
    if (type == "number" || type == "integer") {
        if (value.kind != JsonValue::Kind::number) { return fail("expected a number"); }
        if (type == "integer" && value.number_value != std::floor(value.number_value)) {
            return fail("expected an integer");
        }
        if (const auto *mn = schema.find("minimum"); mn != nullptr) {
            if (value.number_value < mn->number_value) { return fail("number is below minimum"); }
        }
        if (const auto *mx = schema.find("maximum"); mx != nullptr) {
            if (value.number_value > mx->number_value) { return fail("number is above maximum"); }
        }
        return Status::success();
    }
    if (type == "boolean") {
        if (value.kind != JsonValue::Kind::boolean) { return fail("expected a boolean"); }
        return Status::success();
    }
    if (type == "null") {
        if (value.kind != JsonValue::Kind::null) { return fail("expected null"); }
        return Status::success();
    }
    return fail("unsupported schema type");
}

}// namespace

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

[[nodiscard]] Result<JsonValue> parse_json_strict(luisa::string_view text) {
    JsonParser parser{text};
    return parser.parse();
}

[[nodiscard]] Result<JsonParseResult> parse_json(luisa::string_view text, const JsonRepairOptions &repair) {
    using R = Result<JsonParseResult>;
    if (!repair.enabled) {
        auto v = parse_json_strict(text);
        if (!v.ok()) { return R::failure(v.status); }
        JsonParseResult out;
        out.value = std::move(*v.value);
        return R::success(std::move(out));
    }
    // Repair: only the documented, diagnostic-producing transforms.
    JsonParseResult out;
    out.repaired = false;
    luisa::string working{text};
    const auto note = [&](const char *code, luisa::string msg) {
        out.diagnostics.push_back(Diagnostic{DiagnosticSeverity::warning, code, std::move(msg)});
        out.repaired = true;
    };
    // 1. Strip a single leading/trailing Markdown code fence.
    {
        // Leading whitespace.
        size_t b = 0;
        while (b < working.size() && (working[b] == ' ' || working[b] == '\t' || working[b] == '\r' || working[b] == '\n')) { b++; }
        size_t e = working.size();
        while (e > b && (working[e - 1] == ' ' || working[e - 1] == '\t' || working[e - 1] == '\r' || working[e - 1] == '\n')) { e--; }
        if (b != 0 || e != working.size()) {
            working = working.substr(b, e - b);
            note("json_repair_trim", "stripped leading/trailing whitespace");
        }
        // ```json ... ``` or ``` ... ```
        if (working.compare(0, 3, "```") == 0) {
            const auto first_nl = working.find('\n');
            const auto last_fence = working.rfind("```");
            if (first_nl != luisa::string::npos && last_fence != luisa::string::npos && last_fence > first_nl) {
                const auto lang = working.substr(3, first_nl - 3);
                // Only strip when the opening fence is bare or "json".
                auto lang_trim = lang;
                while (!lang_trim.empty() && (lang_trim.back() == ' ' || lang_trim.back() == '\r')) { lang_trim.pop_back(); }
                if (lang_trim.empty() || lang_trim == "json") {
                    working = working.substr(first_nl + 1, last_fence - first_nl - 1);
                    note("json_repair_unfence", "stripped a Markdown code fence");
                }
            }
        }
    }
    // 2. Remove trailing commas before } or ] (outside strings).
    {
        luisa::string cleaned;
        cleaned.reserve(working.size());
        bool in_string = false;
        bool escaped = false;
        bool removed = false;
        for (size_t i = 0; i < working.size(); i++) {
            const char c = working[i];
            if (in_string) {
                cleaned += c;
                if (escaped) { escaped = false; }
                else if (c == '\\') { escaped = true; }
                else if (c == '"') { in_string = false; }
                continue;
            }
            if (c == '"') {
                in_string = true;
                cleaned += c;
                continue;
            }
            if (c == ',') {
                // Look ahead past whitespace for } or ].
                size_t j = i + 1;
                while (j < working.size() && (working[j] == ' ' || working[j] == '\t' || working[j] == '\r' || working[j] == '\n')) { j++; }
                if (j < working.size() && (working[j] == '}' || working[j] == ']')) {
                    removed = true;
                    continue;// drop the comma
                }
            }
            cleaned += c;
        }
        if (removed) {
            working = std::move(cleaned);
            note("json_repair_trailing_comma", "removed trailing comma(s) before } or ]");
        }
    }
    auto v = parse_json_strict(working);
    if (!v.ok()) { return R::failure(v.status); }
    out.value = std::move(*v.value);
    return R::success(std::move(out));
}

[[nodiscard]] Status validate_extraction_schema(const JsonValue &schema) {
    return validate_schema_node(schema, "$", 0);
}

[[nodiscard]] Status validate_json_against_schema(const JsonValue &value, const JsonValue &schema,
                                                  luisa::string_view path) {
    return validate_node(value, schema, path, 0);
}

[[nodiscard]] Result<ExtractionResult> extract_json(luisa::string_view document, luisa::string_view schema,
                                                    const JsonRepairOptions &repair) {
    using R = Result<ExtractionResult>;
    auto schema_val = parse_json_strict(schema);
    if (!schema_val.ok()) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::format("extraction schema is not valid JSON: {}", schema_val.status.message), "schema");
    }
    if (auto s = validate_extraction_schema(*schema_val.value); !s.ok()) { return R::failure(s); }
    auto doc = parse_json(document, repair);
    if (!doc.ok()) { return R::failure(doc.status); }
    if (auto s = validate_json_against_schema(doc.value->value, *schema_val.value); !s.ok()) {
        return R::failure(s);
    }
    ExtractionResult out;
    out.value = std::move(doc.value->value);
    out.diagnostics = std::move(doc.value->diagnostics);
    out.repaired = doc.value->repaired;
    return R::success(std::move(out));
}

[[nodiscard]] Result<luisa::string> serialize_json(const JsonValue &value) {
    using R = Result<luisa::string>;
    // Iterative-safe recursive serializer (depth-bounded by the parser).
    luisa::string out;
    // A small recursive lambda via a helper struct.
    struct Serializer {
        luisa::string &out;
        [[nodiscard]] Status emit(const JsonValue &v, uint32_t depth) {
            if (depth > 64) {
                return Status::error(StatusCode::invalid_argument, "JSON nesting is too deep", "json");
            }
            switch (v.kind) {
                case JsonValue::Kind::null: out += "null"; break;
                case JsonValue::Kind::boolean: out += v.bool_value ? "true" : "false"; break;
                case JsonValue::Kind::number: {
                    if (!std::isfinite(v.number_value)) {
                        return Status::error(StatusCode::invalid_argument, "JSON number is not finite", "json");
                    }
                    // Integer-valued numbers emit without a fraction.
                    if (v.number_value == std::floor(v.number_value) &&
                        std::abs(v.number_value) < 9.0e15) {
                        out += luisa::format("{}", static_cast<int64_t>(v.number_value));
                    } else {
                        char buf[64];
                        std::snprintf(buf, sizeof(buf), "%.17g", v.number_value);
                        out += buf;
                    }
                    break;
                }
                case JsonValue::Kind::string: emit_string(v.string_value); break;
                case JsonValue::Kind::array: {
                    out += '[';
                    for (size_t i = 0; i < v.array_items.size(); i++) {
                        if (i != 0) { out += ','; }
                        if (auto s = emit(v.array_items[i], depth + 1); !s.ok()) { return s; }
                    }
                    out += ']';
                    break;
                }
                case JsonValue::Kind::object: {
                    out += '{';
                    for (size_t i = 0; i < v.object_keys.size(); i++) {
                        if (i != 0) { out += ','; }
                        emit_string(v.object_keys[i]);
                        out += ':';
                        if (auto s = emit(v.object_values[i], depth + 1); !s.ok()) { return s; }
                    }
                    out += '}';
                    break;
                }
            }
            return Status::success();
        }
        void emit_string(luisa::string_view s) {
            out += '"';
            for (char c : s) {
                switch (c) {
                    case '"': out += "\\\""; break;
                    case '\\': out += "\\\\"; break;
                    case '\b': out += "\\b"; break;
                    case '\f': out += "\\f"; break;
                    case '\n': out += "\\n"; break;
                    case '\r': out += "\\r"; break;
                    case '\t': out += "\\t"; break;
                    default:
                        if (static_cast<unsigned char>(c) < 0x20) {
                            char buf[8];
                            std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                            out += buf;
                        } else {
                            out += c;
                        }
                }
            }
            out += '"';
        }
    };
    Serializer ser{out};
    if (auto s = ser.emit(value, 0); !s.ok()) { return R::failure(s); }
    return R::success(std::move(out));
}

}// namespace luisa::ocr::output
