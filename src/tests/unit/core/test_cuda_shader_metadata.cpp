#include "ut/ut.hpp"
#include "cuda_shader_metadata.h"

#include <array>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::cuda;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// A sidecar written before the payload field existed, independent of the serializer.
constexpr string_view kLegacyMetadata =
    "CHECKSUM 0000000000000123 KIND RAY_TRACING DEBUG FALSE "
    "TRACE_CLOSEST FALSE TRACE_ANY FALSE RAY_QUERY TRUE PRINTING FALSE "
    "MOTION_BLUR FALSE MAX_REGISTER_COUNT 0 BLOCK_SIZE 64 1 1 "
    "ARGUMENT_TYPES 2 buffer<uint> accel ARGUMENT_USAGES 2 WRITE READ "
    "FORMAT_TYPES 0 CURVE_BASES 0 ";

[[nodiscard]] CUDAShaderMetadata make_metadata(uint32_t payload_count) {
    return CUDAShaderMetadata{
        .checksum = 0x123u,
        .curve_bases = {},
        .kind = CUDAShaderMetadata::Kind::RAY_TRACING,
        .enable_debug = false,
        .requires_trace_closest = false,
        .requires_trace_any = false,
        .requires_ray_query = true,
        .ray_query_payload_count = payload_count,
        .requires_printing = false,
        .requires_motion_blur = false,
        .max_register_count = 0u,
        .block_size = make_uint3(64u, 1u, 1u),
        .argument_types = {"buffer<uint>", "accel"},
        .argument_usages = {Usage::WRITE, Usage::READ},
        .format_types = {}};
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "cuda_shader_metadata_payload_roundtrip"_test = [] {
        for (auto payload_count : {2u, 32u}) {
            auto original = make_metadata(payload_count);
            auto serialized = serialize_cuda_shader_metadata(original);
            auto field = payload_count == 2u ? "RAY_QUERY_PAYLOAD_COUNT 2 " : "RAY_QUERY_PAYLOAD_COUNT 32 ";
            expect(serialized.find(field) != string::npos);
            auto parsed = deserialize_cuda_shader_metadata(serialized);
            expect(parsed.has_value()) << "payload count:" << payload_count;
            if (parsed) {
                expect(eq(parsed->ray_query_payload_count, payload_count));
                expect(*parsed == original);
            }
        }
    };

    "cuda_shader_metadata_legacy_payload_defaults_to_two"_test = [] {
        auto parsed = deserialize_cuda_shader_metadata(kLegacyMetadata);
        expect(parsed.has_value());
        if (parsed) {
            expect(eq(parsed->ray_query_payload_count, 2u));
            expect(*parsed == make_metadata(2u));
        }
    };

    "cuda_shader_metadata_rejects_invalid_payload_counts"_test = [] {
        constexpr std::array<string_view, 16u> invalid_values{
            "0", "1", "3", "31", "33", "-1", "+2", "2junk",
            "2.0", "0x2", "NaN", "4294967298", "4294967328",
            "18446744073709551615", "18446744073709551616", ""};
        for (auto value : invalid_values) {
            auto serialized = string{kLegacyMetadata};
            serialized.append("RAY_QUERY_PAYLOAD_COUNT ").append(value);
            expect(!deserialize_cuda_shader_metadata(serialized).has_value())
                << "invalid payload token:" << value;
        }
    };

    "cuda_shader_metadata_rejects_duplicate_payload_fields"_test = [] {
        for (auto payload_count : {2u, 32u}) {
            auto serialized = serialize_cuda_shader_metadata(make_metadata(payload_count));
            for (auto suffix : {"RAY_QUERY_PAYLOAD_COUNT 2 ", "RAY_QUERY_PAYLOAD_COUNT 32 "}) {
                auto duplicate = serialized;
                duplicate.append(suffix);
                expect(!deserialize_cuda_shader_metadata(duplicate).has_value())
                    << "original payload count:" << payload_count << "duplicate:" << suffix;
            }
        }
    };

    "cuda_shader_metadata_equality_includes_payload_count"_test = [] {
        auto ast = make_metadata(2u);
        auto llvm = ast;
        expect(ast == llvm);
        llvm.ray_query_payload_count = 32u;
        expect(ast != llvm);
        expect(llvm != ast);
        expect(llvm == make_metadata(32u));
    };
}
