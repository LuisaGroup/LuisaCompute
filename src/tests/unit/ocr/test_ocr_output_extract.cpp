// Test for ocr/output/extract.h/.cpp (Phase 4, host-only).
// Covers: strict JSON parsing (types, Unicode, duplicate keys, trailing
// commas, NaN/Infinity, control chars), the deliberately supported schema
// subset (object/array/string/number/integer/boolean/null, properties,
// required, additionalProperties, items, min/max, enum, const), unknown
// schema features fail-closed, document validation, repair OFF by default
// and diagnostic-producing repair when enabled, canonical serialization.
// No device/model required.
#include "ut/ut.hpp"

#include "ocr/output/extract.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace luisa::ocr::output;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_strict_parse() {
    auto r = parse_json_strict(R"({"a":1,"b":[true,false,null],"c":"x","d":2.5})");
    expect(r.ok());
    if (r.ok()) {
        expect(r.value->kind == JsonValue::Kind::object);
        expect(r.value->object_keys.size() == 4u);
        expect(r.value->find("a")->number_value == 1.0);
        expect(r.value->find("b")->array_items.size() == 3u);
        expect(r.value->find("c")->string_value == "x");
        expect(r.value->find("d")->number_value == 2.5);
    }
    // Unicode + escapes.
    auto u = parse_json_strict(R"({"s":"你好 \u00e9 \"q\" \\ \/ \n"})");
    expect(u.ok());
    if (u.ok()) {
        const auto &s = u.value->find("s")->string_value;
        expect(s.find("你好") != luisa::string::npos);
        expect(s.find("\xC3\xA9") != luisa::string::npos);// é
        expect(s.find("\"q\"") != luisa::string::npos);
    }
    // Surrogate pair (emoji U+1F600).
    auto sp = parse_json_strict(R"("\ud83d\ude00")");
    expect(sp.ok());
    if (sp.ok()) { expect(sp.value->string_value == "\xF0\x9F\x98\x80"); }
}

void test_strict_parse_invalid() {
    expect(!parse_json_strict("").ok());
    expect(!parse_json_strict("{").ok());
    expect(!parse_json_strict("[1,]").ok());// trailing comma
    expect(!parse_json_strict("{\"a\":1,}").ok());// trailing comma
    expect(!parse_json_strict("{\"a\":1,\"a\":2}").ok());// duplicate key
    expect(!parse_json_strict("{a:1}").ok());// unquoted key
    expect(!parse_json_strict("{'a':1}").ok());// single quotes
    expect(!parse_json_strict("NaN").ok());
    expect(!parse_json_strict("Infinity").ok());
    expect(!parse_json_strict("01").ok());// leading zero
    expect(!parse_json_strict("1.").ok());
    expect(!parse_json_strict("\"unterminated").ok());
    expect(!parse_json_strict("\"bad\\x\"").ok());
    expect(!parse_json_strict("\"\\ud800\"").ok());// unpaired surrogate
    expect(!parse_json_strict("[1] extra").ok());// trailing content
    expect(!parse_json_strict("\"a\nb\"").ok());// unescaped control char
}

void test_schema_subset() {
    const luisa::string schema = R"({
        "type":"object",
        "properties":{
            "name":{"type":"string","minLength":1,"maxLength":10},
            "age":{"type":"integer","minimum":0,"maximum":150},
            "tags":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":3},
            "active":{"type":"boolean"},
            "meta":{"type":"null"}
        },
        "required":["name","age"],
        "additionalProperties":false
    })";
    auto sv = parse_json_strict(schema);
    expect(sv.ok());
    if (!sv.ok()) { return; }
    expect(validate_extraction_schema(*sv.value).ok());
    // Valid document.
    auto doc = parse_json_strict(R"({"name":"Ada","age":36,"tags":["x"],"active":true,"meta":null})");
    expect(doc.ok());
    if (doc.ok()) { expect(validate_json_against_schema(*doc.value, *sv.value).ok()); }
    // Missing required.
    auto miss = parse_json_strict(R"({"name":"Ada"})");
    expect(miss.ok());
    if (miss.ok()) { expect(!validate_json_against_schema(*miss.value, *sv.value).ok()); }
    // Additional property rejected.
    auto extra = parse_json_strict(R"({"name":"Ada","age":36,"other":1})");
    expect(extra.ok());
    if (extra.ok()) { expect(!validate_json_against_schema(*extra.value, *sv.value).ok()); }
    // Wrong type.
    auto wrong = parse_json_strict(R"({"name":"Ada","age":"old"})");
    expect(wrong.ok());
    if (wrong.ok()) { expect(!validate_json_against_schema(*wrong.value, *sv.value).ok()); }
    // Integer vs number.
    auto frac = parse_json_strict(R"({"name":"Ada","age":3.5})");
    expect(frac.ok());
    if (frac.ok()) { expect(!validate_json_against_schema(*frac.value, *sv.value).ok()); }
    // Array bounds.
    auto empty_tags = parse_json_strict(R"({"name":"Ada","age":36,"tags":[]})");
    expect(empty_tags.ok());
    if (empty_tags.ok()) { expect(!validate_json_against_schema(*empty_tags.value, *sv.value).ok()); }
}

void test_schema_additional_properties_schema() {
    const luisa::string schema = R"({
        "type":"object",
        "properties":{"a":{"type":"string"}},
        "additionalProperties":{"type":"number"}
    })";
    auto sv = parse_json_strict(schema);
    expect(sv.ok());
    if (!sv.ok()) { return; }
    expect(validate_extraction_schema(*sv.value).ok());
    auto ok_doc = parse_json_strict(R"({"a":"x","b":2})");
    expect(ok_doc.ok());
    if (ok_doc.ok()) { expect(validate_json_against_schema(*ok_doc.value, *sv.value).ok()); }
    auto bad_doc = parse_json_strict(R"({"a":"x","b":"not-number"})");
    expect(bad_doc.ok());
    if (bad_doc.ok()) { expect(!validate_json_against_schema(*bad_doc.value, *sv.value).ok()); }
}

void test_schema_enum_const() {
    const luisa::string schema = R"({"type":"string","enum":["red","green","blue"]})";
    auto sv = parse_json_strict(schema);
    expect(sv.ok());
    if (!sv.ok()) { return; }
    expect(validate_extraction_schema(*sv.value).ok());
    auto ok_doc = parse_json_strict(R"("green")");
    expect(ok_doc.ok());
    if (ok_doc.ok()) { expect(validate_json_against_schema(*ok_doc.value, *sv.value).ok()); }
    auto bad_doc = parse_json_strict(R"("purple")");
    expect(bad_doc.ok());
    if (bad_doc.ok()) { expect(!validate_json_against_schema(*bad_doc.value, *sv.value).ok()); }
    // const.
    auto cs = parse_json_strict(R"({"const":42})");
    expect(cs.ok());
    if (cs.ok()) {
        expect(validate_extraction_schema(*cs.value).ok());
        auto cd = parse_json_strict("42");
        expect(cd.ok());
        if (cd.ok()) { expect(validate_json_against_schema(*cd.value, *cs.value).ok()); }
    }
}

void test_schema_unknown_features_fail_closed() {
    // Each unsupported keyword must be rejected (never silently ignored).
    for (const char *kw : {"pattern", "format", "allOf", "anyOf", "oneOf", "not",
                           "patternProperties", "uniqueItems", "multipleOf",
                           "minProperties", "maxProperties", "propertyNames",
                           "prefixItems", "contains", "if", "then", "else",
                           "dependentRequired", "unevaluatedProperties", "$ref"}) {
        const auto schema = luisa::string{R"({"type":"string",")"} + kw + R"(":1})";
        auto sv = parse_json_strict(schema);
        expect(sv.ok()) << kw;
        if (sv.ok()) {
            auto s = validate_extraction_schema(*sv.value);
            expect(!s.ok()) << "keyword must fail closed: " << kw;
            expect(s.code == StatusCode::unsupported) << kw;
        }
    }
    // Unknown type.
    auto ut = parse_json_strict(R"({"type":"date"})");
    expect(ut.ok());
    if (ut.ok()) { expect(validate_extraction_schema(*ut.value).code == StatusCode::unsupported); }
    // Malformed schema (required not array).
    auto bad = parse_json_strict(R"({"type":"object","required":"name"})");
    expect(bad.ok());
    if (bad.ok()) { expect(!validate_extraction_schema(*bad.value).ok()); }
}

void test_extract_one_shot() {
    const luisa::string schema = R"({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]})";
    auto ok = extract_json(R"({"x":5})", schema);
    expect(ok.ok());
    if (ok.ok()) {
        expect(ok.value->value.find("x")->number_value == 5.0);
        expect(!ok.value->repaired);
    }
    // Non-conforming document fails (no plain-text fallback).
    expect(!extract_json(R"({"x":"five"})", schema).ok());
    expect(!extract_json("not json at all", schema).ok());
    // Invalid schema fails.
    expect(!extract_json(R"({"x":5})", R"({"type":"object","pattern":"x"})").ok());
}

void test_repair_default_off() {
    const luisa::string schema = R"({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]})";
    // A fenced document with a trailing comma: repair OFF must fail.
    const luisa::string doc = "```json\n{\"x\":5,}\n```";
    expect(!extract_json(doc, schema).ok());
    // Repair ON: succeeds with diagnostics.
    JsonRepairOptions repair;
    repair.enabled = true;
    auto r = extract_json(doc, schema, repair);
    expect(r.ok());
    if (r.ok()) {
        expect(r.value->repaired);
        expect(r.value->diagnostics.size() >= 1u);
        expect(r.value->value.find("x")->number_value == 5.0);
    }
    // Repair ON but still invalid: fails (never a silent fallback).
    expect(!extract_json("```json\n{\"x\":}\n```", schema, repair).ok());
}

void test_serialize() {
    auto v = parse_json_strict(R"({"b":1,"a":[1,2.5,"x",true,null]})");
    expect(v.ok());
    if (!v.ok()) { return; }
    auto s = serialize_json(*v.value);
    expect(s.ok());
    if (s.ok()) {
        // Key order preserved.
        expect(s.value->find("\"b\"") < s.value->find("\"a\""));
        // Round-trip.
        auto rt = parse_json_strict(*s.value);
        expect(rt.ok());
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "extract_strict_parse"_test = [] { test_strict_parse(); };
    "extract_strict_parse_invalid"_test = [] { test_strict_parse_invalid(); };
    "extract_schema_subset"_test = [] { test_schema_subset(); };
    "extract_schema_additional_properties_schema"_test = [] { test_schema_additional_properties_schema(); };
    "extract_schema_enum_const"_test = [] { test_schema_enum_const(); };
    "extract_schema_unknown_features_fail_closed"_test = [] { test_schema_unknown_features_fail_closed(); };
    "extract_one_shot"_test = [] { test_extract_one_shot(); };
    "extract_repair_default_off"_test = [] { test_repair_default_off(); };
    "extract_serialize"_test = [] { test_serialize(); };
    return 0;
}
