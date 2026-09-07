// Test for the shared example command-line parser.
// This test covers strict values, uint32 boundaries, extension arguments,
// and raw-reference read/write modes without creating a device.

#include "ut/ut.hpp"

#include "reference_compare.h"

#include <cstdint>
#include <initializer_list>
#include <vector>

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] ref::ExampleOptions parse_options(
    std::initializer_list<const char *> arguments) {
    std::vector<char *> argv;
    argv.reserve(arguments.size());
    for (auto *argument : arguments) {
        argv.emplace_back(const_cast<char *>(argument));
    }
    return ref::ExampleOptions::parse(
        static_cast<int>(argv.size()), argv.data());
}

void expect_invalid(std::initializer_list<const char *> arguments) {
    auto options = parse_options(arguments);
    expect(!options.valid());
    expect(!options.error_message.empty());
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "example_options_accept_valid_shared_options"_test = [] {
        auto options = parse_options({"example", "vk", "--spp", "17", "--iterations", "23",
                                      "--max-spp-per-dispatch", "5", "-c", "reference.png"});
        expect(options.valid());
        expect(options.error_message.empty());
        expect(options.offline);
        expect(eq(options.spp, uint32_t{17u}));
        expect(eq(options.iterations, uint32_t{23u}));
        expect(options.max_spp_per_dispatch.has_value());
        expect(eq(*options.max_spp_per_dispatch, uint32_t{5u}));
        expect(options.compare_path.has_value());
        expect(options.compare_path->generic_string() == "reference.png");

        auto max_value = parse_options({
            "example", "vk", "--spp", "4294967295"});
        expect(max_value.valid());
        expect(eq(max_value.spp, uint32_t{0xffffffffu}));
    };

    "example_options_reject_missing_values"_test = [] {
        expect_invalid({"example", "vk", "--compare"});
        expect_invalid({"example", "vk", "-c"});
        expect_invalid({"example", "vk", "--compare", "--offline"});
        expect_invalid({"example", "vk", "--spp"});
        expect_invalid({"example", "vk", "--iterations"});
        expect_invalid({"example", "vk", "--max-spp-per-dispatch"});
        expect_invalid({"example", "vk", "--out_ref"});
        expect_invalid({"example", "vk", "--out_ref", "write"});
        expect_invalid({"example", "vk", "--out_ref", "read"});
    };

    "example_options_reject_junk_unsigned_value"_test = [] {
        expect_invalid({"example", "vk", "--spp", "12samples"});
        expect_invalid({"example", "vk", "--iterations", "12passes"});
        expect_invalid({"example", "vk", "--max-spp-per-dispatch", "12samples"});
    };

    "example_options_reject_negative_unsigned_value"_test = [] {
        expect_invalid({"example", "vk", "--spp", "-1"});
        expect_invalid({"example", "vk", "--iterations", "-1"});
        expect_invalid({"example", "vk", "--max-spp-per-dispatch", "-1"});
    };

    "example_options_reject_uint32_overflow"_test = [] {
        expect_invalid({"example", "vk", "--spp", "4294967296"});
        expect_invalid({"example", "vk", "--iterations", "4294967296"});
        expect_invalid({"example", "vk", "--max-spp-per-dispatch", "4294967296"});
    };

    "example_options_reject_zero_iterations"_test = [] {
        expect_invalid({"example", "vk", "--iterations", "0"});
        expect_invalid({"example", "vk", "--max-spp-per-dispatch", "0"});
    };

    "example_options_coexist_with_extension_options"_test = [] {
        auto options = parse_options({
            "example", "vk", "--scheduler", "wavefront",
            "--resolution", "512", "--sample-dispatch",
            "--spp", "64", "--offline"});
        expect(options.valid());
        expect(options.offline);
        expect(eq(options.spp, uint32_t{64u}));
        expect(!options.compare_path.has_value());
    };

    "example_options_parse_out_ref_modes"_test = [] {
        auto write = parse_options({
            "example", "vk", "--out_ref", "write", "output.bin"});
        expect(write.valid());
        expect(write.offline);
        expect(write.out_ref_write);
        expect(write.out_ref_path.has_value());
        expect(write.out_ref_path->generic_string() == "output.bin");

        auto read = parse_options({
            "example", "vk", "--out_ref", "read", "input.bin"});
        expect(read.valid());
        expect(read.offline);
        expect(!read.out_ref_write);
        expect(read.out_ref_path.has_value());
        expect(read.out_ref_path->generic_string() == "input.bin");

        expect_invalid({
            "example", "vk", "--out_ref", "append", "output.bin"});
    };
}
