#include "ut/ut.hpp"
#include "cuda_shader_metadata.h"
#include "llvm_codegen/cuda_codegen_llvm_optix_ir.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <string>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::cuda;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// A synthetic container for routing and envelope bounds, not a complete
// OptiX module. The backend remains responsible for validating its IR body.
constexpr std::array<std::byte, 32u> kOptixIrEnvelope{
    std::byte{0xed}, std::byte{0x43}, std::byte{0x4e}, std::byte{0x7f},
    std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
    std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
    std::byte{24}, std::byte{0}, std::byte{2}, std::byte{0},
    std::byte{24}, std::byte{0}, std::byte{0}, std::byte{0},
    std::byte{28}, std::byte{0}, std::byte{0}, std::byte{0},
    std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
    std::byte{0x42}, std::byte{0x43}, std::byte{0xc0}, std::byte{0xde}};

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

    "cuda_optix_ir_encoder_fast_math_ftz_option"_test = [] {
        // A raw bitcode signature suffices for this CPU-only envelope test.
        constexpr std::array bitcode{
            std::byte{0x42}, std::byte{0x43}, std::byte{0xc0}, std::byte{0xde}};
        auto precise = luisa_compute_cuda_llvm_encode_optix_ir(luisa::span{bitcode}, 89u, false);
        auto default_options = luisa_compute_cuda_llvm_encode_optix_ir(luisa::span{bitcode}, 89u);
        auto fast = luisa_compute_cuda_llvm_encode_optix_ir(luisa::span{bitcode}, 89u, true);
        // The precise header remains the original format, byte for byte.
        constexpr std::array<uint8_t, 44u> precise_header{
            0xed, 0x43, 0x4e, 0x7f, 1, 0x43, 2, 0x73, 3, 2, 7, 0,
            24, 0, 2, 0, 44, 0, 0, 0, 48, 0, 0, 0,
            1, 0, 0x7a, 3, 2, 0, 0, 0, 3, 0, 0, 0,
            99, 0, 0, 0, 0, 0, 0, 0};
        constexpr std::array<uint8_t, 48u> fast_header{
            0xed, 0x43, 0x4e, 0x7f, 1, 0x43, 2, 0x73, 3, 2, 7, 0,
            24, 0, 2, 0, 48, 0, 0, 0, 52, 0, 0, 0,
            1, 0, 0x7a, 3, 2, 0, 0, 0, 3, 0, 0, 0,
            13, 0, 1, 0, 99, 0, 0, 0, 0, 0, 0, 0};
        expect(precise == default_options);
        expect(eq(precise.size(), size_t{48u} + bitcode.size()));
        expect(eq(fast.size(), size_t{52u} + bitcode.size()));
        if (precise.size() == 48u + bitcode.size() && fast.size() == 52u + bitcode.size()) {
            for (auto i = size_t{0u}; i < precise_header.size(); i++) {
                expect(eq(static_cast<uint8_t>(precise[i]), precise_header[i])) << "precise byte:" << i;
            }
            for (auto i = size_t{0u}; i < fast_header.size(); i++) {
                expect(eq(static_cast<uint8_t>(fast[i]), fast_header[i])) << "fast byte:" << i;
            }
            // FTZ changes only the header; seed and encoded bitcode are identical.
            expect(precise.substr(44u) == fast.substr(48u));
            for (const auto *encoded : {&precise, &fast}) {
                auto bytes = luisa::span{
                    reinterpret_cast<const std::byte *>(encoded->data()), encoded->size()};
                expect(cuda_shader_code_matches_format(bytes, CUDAShaderMetadata::CodeFormat::OPTIX_IR));
            }
        }
    };

    "cuda_shader_metadata_payload_roundtrip"_test = [] {
        for (auto payload_count : {2u, 3u, 5u, 31u, 32u}) {
            auto original = make_metadata(payload_count);
            auto serialized = serialize_cuda_shader_metadata(original);
            auto field = "RAY_QUERY_PAYLOAD_COUNT " + std::to_string(payload_count) + " ";
            expect(serialized.find(string_view{field}) != string::npos);
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
        constexpr std::array<string_view, 14u> invalid_values{
            "0", "1", "33", "-1", "+2", "2junk",
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
        for (auto payload_count : {2u, 3u, 5u, 31u, 32u}) {
            auto serialized = serialize_cuda_shader_metadata(make_metadata(payload_count));
            for (auto suffix : {"RAY_QUERY_PAYLOAD_COUNT 2 ", "RAY_QUERY_PAYLOAD_COUNT 3 ",
                                "RAY_QUERY_PAYLOAD_COUNT 5 ", "RAY_QUERY_PAYLOAD_COUNT 32 "}) {
                auto duplicate = serialized;
                duplicate.append(suffix);
                expect(!deserialize_cuda_shader_metadata(duplicate).has_value())
                    << "original payload count:" << payload_count << "duplicate:" << suffix;
            }
        }
    };

    "cuda_shader_metadata_equality_includes_payload_count"_test = [] {
        for (auto lhs : {2u, 3u, 5u, 31u, 32u}) {
            for (auto rhs : {2u, 3u, 5u, 31u, 32u}) {
                expect((make_metadata(lhs) == make_metadata(rhs)) == (lhs == rhs))
                    << "payload counts:" << lhs << rhs;
            }
        }
    };

    "cuda_shader_metadata_code_format_defaults_to_ptx"_test = [] {
        auto metadata = CUDAShaderMetadata{};
        expect(metadata.code_format == CUDAShaderMetadata::CodeFormat::PTX);
    };

    "cuda_shader_metadata_code_format_roundtrip"_test = [] {
        for (auto format : {CUDAShaderMetadata::CodeFormat::PTX, CUDAShaderMetadata::CodeFormat::OPTIX_IR}) {
            for (auto payload_count : {2u, 3u, 5u, 31u, 32u}) {
                auto original = make_metadata(payload_count);
                original.code_format = format;
                auto serialized = serialize_cuda_shader_metadata(original);
                auto field = format == CUDAShaderMetadata::CodeFormat::PTX ? "CODE_FORMAT PTX " : "CODE_FORMAT OPTIX_IR ";
                expect(serialized.find(field) != string::npos);
                auto parsed = deserialize_cuda_shader_metadata(serialized);
                expect(parsed.has_value()) << field << "payload count:" << payload_count;
                if (parsed) {
                    expect(parsed->code_format == format);
                    expect(eq(parsed->ray_query_payload_count, payload_count));
                    expect(*parsed == original);
                }
            }
        }
    };

    "cuda_shader_metadata_legacy_code_format_defaults_to_ptx"_test = [] {
        for (auto kind : {"RAY_TRACING", "COMPUTE", "TILE"}) {
            auto serialized = string{kLegacyMetadata};
            serialized.replace(serialized.find("RAY_TRACING"), string_view{"RAY_TRACING"}.size(), kind);
            if (string_view{kind} != "RAY_TRACING") {
                serialized.replace(serialized.find("RAY_QUERY TRUE"), string_view{"RAY_QUERY TRUE"}.size(), "RAY_QUERY FALSE");
            }
            auto parsed = deserialize_cuda_shader_metadata(serialized);
            expect(parsed.has_value()) << "legacy kind:" << kind;
            if (parsed) {
                expect(parsed->code_format == CUDAShaderMetadata::CodeFormat::PTX);
                expect(eq(parsed->ray_query_payload_count, 2u));
            }
        }
    };

    "cuda_shader_metadata_rejects_unknown_code_format"_test = [] {
        constexpr std::array<string_view, 8u> invalid_values{
            "", "ptx", "optix_ir", "OPTIX", "LLVM_IR", "0", "PTXjunk", "OPTIX_IRjunk"};
        for (auto value : invalid_values) {
            auto serialized = string{kLegacyMetadata};
            serialized.append("CODE_FORMAT ").append(value);
            expect(!deserialize_cuda_shader_metadata(serialized).has_value())
                << "invalid code format:" << value;
        }
    };

    "cuda_shader_metadata_rejects_duplicate_code_format"_test = [] {
        for (auto first : {"PTX", "OPTIX_IR"}) {
            for (auto second : {"PTX", "OPTIX_IR"}) {
                auto serialized = string{kLegacyMetadata};
                serialized.append("CODE_FORMAT ").append(first).append(" CODE_FORMAT ").append(second).append(" ");
                expect(!deserialize_cuda_shader_metadata(serialized).has_value())
                    << "duplicate code formats:" << first << second;
            }
        }
    };

    "cuda_shader_metadata_optix_ir_requires_ray_tracing"_test = [] {
        for (auto kind : {"COMPUTE", "TILE"}) {
            auto base = string{kLegacyMetadata};
            base.replace(base.find("RAY_TRACING"), string_view{"RAY_TRACING"}.size(), kind);
            base.replace(base.find("RAY_QUERY TRUE"), string_view{"RAY_QUERY TRUE"}.size(), "RAY_QUERY FALSE");
            // Cross-field validation must not depend on sidecar token order.
            for (auto format_first : {false, true}) {
                auto ir = format_first ? string{"CODE_FORMAT OPTIX_IR "}.append(base) : string{base}.append("CODE_FORMAT OPTIX_IR ");
                expect(!deserialize_cuda_shader_metadata(ir).has_value())
                    << "OptiX IR kind:" << kind << "format before kind:" << format_first;
                auto ptx = format_first ? string{"CODE_FORMAT PTX "}.append(base) : string{base}.append("CODE_FORMAT PTX ");
                expect(deserialize_cuda_shader_metadata(ptx).has_value())
                    << "PTX kind:" << kind << "format before kind:" << format_first;
            }
        }
        auto before_kind = string{"CODE_FORMAT OPTIX_IR "}.append(kLegacyMetadata);
        auto parsed = deserialize_cuda_shader_metadata(before_kind);
        expect(parsed.has_value());
        if (parsed) {
            expect(parsed->code_format == CUDAShaderMetadata::CodeFormat::OPTIX_IR);
        }
    };

    "cuda_shader_metadata_equality_includes_code_format"_test = [] {
        auto ptx = make_metadata(32u);
        auto ir = ptx;
        expect(ptx == ir);
        ir.code_format = CUDAShaderMetadata::CodeFormat::OPTIX_IR;
        expect(ptx != ir);
        expect(ir != ptx);
        ptx.code_format = CUDAShaderMetadata::CodeFormat::OPTIX_IR;
        expect(ptx == ir);
    };


    "cuda_shader_code_format_rejects_empty_and_unknown"_test = [] {
        auto empty = luisa::span<const std::byte>{};
        expect(!cuda_shader_code_matches_format(empty, CUDAShaderMetadata::CodeFormat::PTX));
        expect(!cuda_shader_code_matches_format(empty, CUDAShaderMetadata::CodeFormat::OPTIX_IR));
        expect(!cuda_shader_code_matches_format(luisa::span{kOptixIrEnvelope}, static_cast<CUDAShaderMetadata::CodeFormat>(255u)));
    };

    "cuda_shader_code_format_distinguishes_text_and_ir_envelope"_test = [] {
        constexpr std::array<std::byte, 4u> ptx{std::byte{'.'}, std::byte{'v'}, std::byte{'e'}, std::byte{'r'}};
        expect(cuda_shader_code_matches_format(luisa::span{ptx}, CUDAShaderMetadata::CodeFormat::PTX));
        expect(!cuda_shader_code_matches_format(luisa::span{ptx}, CUDAShaderMetadata::CodeFormat::OPTIX_IR));
        expect(cuda_shader_code_matches_format(luisa::span{kOptixIrEnvelope}, CUDAShaderMetadata::CodeFormat::OPTIX_IR));
        expect(!cuda_shader_code_matches_format(luisa::span{kOptixIrEnvelope}, CUDAShaderMetadata::CodeFormat::PTX));
    };

    "cuda_shader_code_format_rejects_truncated_ir"_test = [] {
        auto code = luisa::span{kOptixIrEnvelope};
        for (auto size = size_t{0u}; size <= 28u; size++) {
            expect(!cuda_shader_code_matches_format(code.first(size), CUDAShaderMetadata::CodeFormat::OPTIX_IR))
                << "truncated envelope length:" << size;
        }
        // Even an incomplete IR container must not be routed to the PTX path
        // once its complete four-byte binary signature has been identified.
        expect(!cuda_shader_code_matches_format(code.first(4u), CUDAShaderMetadata::CodeFormat::PTX));
    };

    "cuda_shader_code_format_rejects_invalid_ir_envelope_bounds"_test = [] {
        constexpr std::array<std::pair<size_t, uint8_t>, 8u> mutations{{
            {0u, 0u}, {12u, 23u}, {14u, 1u}, {16u, 23u},
            {16u, 29u}, {20u, 23u}, {20u, 32u}, {23u, 255u}}};
        for (auto [offset, value] : mutations) {
            auto code = kOptixIrEnvelope;
            code[offset] = static_cast<std::byte>(value);
            expect(!cuda_shader_code_matches_format(luisa::span{code}, CUDAShaderMetadata::CodeFormat::OPTIX_IR))
                << "invalid envelope byte:" << offset << "value:" << static_cast<unsigned>(value);
        }
    };

}
