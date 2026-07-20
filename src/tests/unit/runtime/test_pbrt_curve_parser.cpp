#include "ut/ut.hpp"
#include "integration/runtime/pbrt_curve_parser.h"

#include <cmath>
#include <sstream>

using namespace luisa;
using namespace luisa::test;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] PbrtCurveParseResult parse(luisa::string_view text) noexcept {
    std::istringstream stream{std::string{text}};
    return parse_pbrt_curve_stream(stream);
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "pbrt_curve_parser_reads_segments_widths_and_unknown_properties"_test = [] {
        auto result = parse(
            R"(Shape "curve"
               "string type" ["cylinder"]
               "point3 P" [0 0 0 1 2 3 2 4 6 3 6 9 4 8 12]
               "float width0" [5e-1]
               "float width1" [1e-1])");
        expect(static_cast<bool>(result)) << result.error;
        if (!result) { return; }
        expect(result.data.control_points.size() == 5u);
        expect(result.data.segments.size() == 2u);
        expect(result.data.segments[0u] == 0u);
        expect(result.data.segments[1u] == 1u);
        expect(std::abs(result.data.control_points.front().w - 0.5f) < 1e-6f);
        expect(std::abs(result.data.control_points.back().w - 0.1f) < 1e-6f);
        expect(all(result.data.aabb_min == make_float3(0.0f)));
        expect(all(result.data.aabb_max == make_float3(4.0f, 8.0f, 12.0f)));
    };

    "pbrt_curve_parser_applies_constant_width_to_both_ends"_test = [] {
        auto result = parse(
            R"(Shape "curve" "point3 P" [0 0 0 1 0 0 2 0 0 3 0 0]
               "float width" [0.25])");
        expect(static_cast<bool>(result)) << result.error;
        if (!result) { return; }
        expect(result.data.control_points.size() == 4u);
        expect(result.data.segments.size() == 1u);
        for (auto point : result.data.control_points) {
            expect(std::abs(point.w - 0.25f) < 1e-6f);
        }
    };

    "pbrt_curve_parser_rejects_malformed_inputs"_test = [] {
        constexpr std::array malformed{
            R"(Shape "curve)",
            R"(Shape "curve" "point3 P" [0 0 nope 1 0 0 2 0 0 3 0 0] "float width" [1])",
            R"(Shape "curve" "point3 P" [0 0 0 1 0 0 2 0 0 1e999 0 0] "float width" [1])",
            R"(Shape "curve" "point3 P" [0 0 0 1 0 0 2 0 0 1e2e3 0 0] "float width" [1])",
            R"(Shape "curve" "point3 P" [0 0 0 1 0 0 2 0 0] "float width" [1])",
            R"(Shape "curve" "point3 P" [0 0 0 1 0 0 2 0 0 3 0 0])",
            R"(Shape "curve" "point3 P" [0 0 0 1 0 0 2 0 0 3 0 0] "float width" [0])",
            R"(Shape "curve" "point3 P" [0 0 0 1 0 0 2 0 0 3 0 0] "string type" ["cylinder")"};
        for (auto text : malformed) {
            auto result = parse(text);
            expect(!static_cast<bool>(result));
            expect(!result.error.empty());
        }
    };

    return 0;
}
