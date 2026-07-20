#pragma once

#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <istream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <luisa/luisa-compute.h>

namespace luisa::test {

struct PbrtCurveData {
    luisa::vector<float4> control_points;
    luisa::vector<uint32_t> segments;
    float3 aabb_min;
    float3 aabb_max;
};

struct PbrtCurveParseResult {
    PbrtCurveData data;
    std::string error;

    [[nodiscard]] explicit operator bool() const noexcept { return error.empty(); }
};

namespace detail {

class PbrtCurveParser {
private:
    std::istream &_stream;
    PbrtCurveParseResult _result;
    size_t _offset{0u};
    bool _failed{false};

private:
    void _fail(std::string_view message) noexcept {
        if (!_failed) {
            _result.error = std::string{message} + " at byte " + std::to_string(_offset) + ".";
            _failed = true;
        }
    }

    [[nodiscard]] bool _eof() noexcept {
        return _stream.peek() == std::char_traits<char>::eof();
    }

    [[nodiscard]] std::optional<char> _peek() noexcept {
        auto c = _stream.peek();
        if (c == std::char_traits<char>::eof()) {
            _fail("Unexpected EOF");
            return std::nullopt;
        }
        return static_cast<char>(c);
    }

    [[nodiscard]] std::optional<char> _pop() noexcept {
        auto c = _stream.get();
        if (c == std::char_traits<char>::eof()) {
            _fail("Unexpected EOF");
            return std::nullopt;
        }
        _offset++;
        return static_cast<char>(c);
    }

    [[nodiscard]] bool _match(char expected) noexcept {
        auto c = _pop();
        if (!c) { return false; }
        if (*c != expected) {
            _fail("Unexpected character");
            return false;
        }
        return true;
    }

    void _skip_whitespaces() noexcept {
        while (!_failed && !_eof()) {
            auto c = _peek();
            if (!c || !std::isspace(static_cast<unsigned char>(*c))) { break; }
            static_cast<void>(_pop());
        }
    }

    [[nodiscard]] std::optional<std::string> _read_string() noexcept {
        _skip_whitespaces();
        if (_failed || !_match('"')) { return std::nullopt; }
        std::string value;
        while (!_failed) {
            if (_eof()) {
                _fail("Unexpected EOF while reading string");
                return std::nullopt;
            }
            auto c = _pop();
            if (!c) { return std::nullopt; }
            if (*c == '"') { return value; }
            value.push_back(*c);
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::string> _read_token() noexcept {
        _skip_whitespaces();
        std::string value;
        while (!_failed && !_eof()) {
            auto c = _peek();
            if (!c || std::isspace(static_cast<unsigned char>(*c))) { break; }
            auto consumed = _pop();
            if (!consumed) { return std::nullopt; }
            value.push_back(*consumed);
        }
        if (value.empty()) {
            _fail("Expected token");
            return std::nullopt;
        }
        return value;
    }

    [[nodiscard]] std::optional<float> _read_float() noexcept {
        _skip_whitespaces();
        std::string token;
        auto is_float_character = [](char c) noexcept {
            auto u = static_cast<unsigned char>(c);
            return std::isdigit(u) || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E';
        };
        while (!_failed && !_eof()) {
            auto c = _peek();
            if (!c || !is_float_character(*c)) { break; }
            auto consumed = _pop();
            if (!consumed) { return std::nullopt; }
            token.push_back(*consumed);
        }
        if (token.empty()) {
            _fail("Expected floating-point value");
            return std::nullopt;
        }
        errno = 0;
        char *parsed_end = nullptr;
        auto value = std::strtof(token.c_str(), &parsed_end);
        if (errno == ERANGE || parsed_end != token.data() + token.size() || !std::isfinite(value)) {
            _fail("Invalid floating-point value");
            return std::nullopt;
        }
        return value;
    }

    [[nodiscard]] bool _skip_property_value() noexcept {
        while (!_failed) {
            if (_eof()) {
                _fail("Unexpected EOF while reading property");
                return false;
            }
            auto c = _peek();
            if (!c) { return false; }
            if (*c == ']') { return true; }
            static_cast<void>(_pop());
        }
        return false;
    }

    [[nodiscard]] bool _parse_curve() noexcept {
        auto token = _read_token();
        if (!token) { return false; }
        if (*token != "Shape") {
            _fail("Expected Shape token");
            return false;
        }
        auto shape = _read_string();
        if (!shape) { return false; }
        if (*shape != "curve") {
            _fail("Expected curve shape");
            return false;
        }

        luisa::vector<float3> vertices;
        std::optional<float> width;
        std::optional<float> width0;
        std::optional<float> width1;

        while (!_failed) {
            _skip_whitespaces();
            if (_eof()) { break; }
            auto c = _peek();
            if (!c) { return false; }
            if (*c != '"') { break; }

            auto property = _read_string();
            if (!property) { return false; }
            _skip_whitespaces();
            if (!_match('[')) { return false; }
            if (*property == "point3 P") {
                while (!_failed) {
                    _skip_whitespaces();
                    if (_eof()) {
                        _fail("Unexpected EOF while reading control points");
                        return false;
                    }
                    auto next = _peek();
                    if (!next) { return false; }
                    if (*next == ']') { break; }
                    auto x = _read_float();
                    auto y = _read_float();
                    auto z = _read_float();
                    if (!x || !y || !z) { return false; }
                    vertices.emplace_back(make_float3(*x, *y, *z));
                }
            } else if (*property == "float width") {
                width = _read_float();
                if (!width) { return false; }
            } else if (*property == "float width0") {
                width0 = _read_float();
                if (!width0) { return false; }
            } else if (*property == "float width1") {
                width1 = _read_float();
                if (!width1) { return false; }
            } else if (!_skip_property_value()) {
                return false;
            }
            _skip_whitespaces();
            if (!_match(']')) { return false; }
        }

        if (vertices.size() < 4u) {
            _fail("A cubic curve requires at least four control points");
            return false;
        }
        auto start_width = width0.value_or(width.value_or(width1.value_or(0.0f)));
        auto end_width = width1.value_or(width.value_or(width0.value_or(0.0f)));
        if (!(start_width > 0.0f) || !(end_width > 0.0f)) {
            _fail("Curve widths must be positive");
            return false;
        }

        auto offset = static_cast<uint32_t>(_result.data.control_points.size());
        auto denominator = static_cast<float>(vertices.size() - 1u);
        for (auto i = 0u; i < vertices.size(); i++) {
            auto p = vertices[i];
            auto t = static_cast<float>(i) / denominator;
            auto curve_width = std::lerp(start_width, end_width, t);
            _result.data.control_points.emplace_back(make_float4(p, curve_width));
            _result.data.aabb_min = min(_result.data.aabb_min, p);
            _result.data.aabb_max = max(_result.data.aabb_max, p);
        }
        for (auto i = 0u; i + 3u < vertices.size(); i++) {
            _result.data.segments.emplace_back(offset + i);
        }
        return true;
    }

public:
    explicit PbrtCurveParser(std::istream &stream) noexcept
        : _stream{stream} {
        static constexpr auto infinity = std::numeric_limits<float>::infinity();
        _result.data.aabb_min = make_float3(infinity);
        _result.data.aabb_max = make_float3(-infinity);
    }

    [[nodiscard]] PbrtCurveParseResult parse() noexcept {
        while (!_failed) {
            _skip_whitespaces();
            if (_eof()) { break; }
            if (!_parse_curve()) { break; }
        }
        if (!_failed && _result.data.segments.empty()) {
            _fail("No curve segments found");
        }
        return std::move(_result);
    }
};

}// namespace detail

[[nodiscard]] inline PbrtCurveParseResult parse_pbrt_curve_stream(std::istream &stream) noexcept {
    return detail::PbrtCurveParser{stream}.parse();
}

[[nodiscard]] inline PbrtCurveParseResult parse_pbrt_curve_file(
    const std::filesystem::path &path) noexcept {
    std::ifstream file{path};
    if (!file.is_open()) {
        PbrtCurveParseResult result;
        result.error = "Failed to open curve file: " + path.string();
        return result;
    }
    return parse_pbrt_curve_stream(file);
}

}// namespace luisa::test
