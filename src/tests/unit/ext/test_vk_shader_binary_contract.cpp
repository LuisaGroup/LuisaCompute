// Test for the Vulkan persisted shader-artifact contract and production codec.
// This test covers complete compute/raster round trips, integrity failures,
// Vulkan SPIR-V validation, entry-point stages, and feature reconciliation.

#include "ut/ut.hpp"

#include "shader_artifact_codec.h"
#include "shader_binary_contract.h"
#include "shader_interface_plan.h"

#include <luisa/core/binary_io.h>
#include <luisa/core/stl/filesystem.h>

#if defined(LUISA_XIR_TO_SPIRV) || defined(LUISA_AST_LLVM_TO_SPIRV)
#include <spirv-tools/libspirv.hpp>
#endif

#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;
using namespace std::string_view_literals;

namespace {

[[nodiscard]] vstd::MD5 test_md5(vstd::string_view value) {
    return vstd::MD5{value};
}

class MemoryBinaryStream final : public BinaryStream {
private:
    luisa::vector<std::byte> _data;
    size_t _position{};

public:
    explicit MemoryBinaryStream(luisa::span<const std::byte> data)
        : _data{data.begin(), data.end()} {}

    [[nodiscard]] size_t length() const noexcept override {
        return _data.size();
    }

    [[nodiscard]] size_t pos() const noexcept override {
        return _position;
    }

    void read(luisa::span<std::byte> dst) noexcept override {
        if (_position > _data.size() ||
            dst.size() > _data.size() - _position) {
            std::memset(dst.data(), 0, dst.size());
            _position = _data.size();
            return;
        }
        std::memcpy(dst.data(), _data.data() + _position, dst.size());
        _position += dst.size();
    }
};

class MemoryBinaryIO final : public BinaryIO {
private:
    struct Entry {
        luisa::string name;
        luisa::vector<std::byte> bytes;
    };

    mutable Entry _bytecode;
    mutable Entry _cache;
    mutable Entry _internal;

    [[nodiscard]] static luisa::unique_ptr<BinaryStream> _read(
        const Entry &entry, luisa::string_view name) noexcept {
        if (entry.name != name || entry.bytes.empty()) { return nullptr; }
        return luisa::make_unique<MemoryBinaryStream>(
            luisa::span<const std::byte>{entry.bytes});
    }

    [[nodiscard]] static luisa::filesystem::path _write(
        Entry &entry, luisa::string_view name,
        luisa::span<const std::byte> data) noexcept {
        entry.name = name;
        entry.bytes.assign(data.begin(), data.end());
        return {};
    }

public:
    void clear_shader_cache() const noexcept override {
        _cache = {};
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream> read_shader_bytecode(
        luisa::string_view name) const noexcept override {
        return _read(_bytecode, name);
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream> read_shader_cache(
        luisa::string_view name) const noexcept override {
        return _read(_cache, name);
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream> read_internal_shader(
        luisa::string_view name) const noexcept override {
        return _read(_internal, name);
    }

    luisa::filesystem::path write_shader_bytecode(
        luisa::string_view name,
        luisa::span<const std::byte> data) const noexcept override {
        return _write(_bytecode, name, data);
    }

    luisa::filesystem::path write_shader_cache(
        luisa::string_view name,
        luisa::span<const std::byte> data) const noexcept override {
        return _write(_cache, name, data);
    }

    luisa::filesystem::path write_internal_shader(
        luisa::string_view name,
        luisa::span<const std::byte> data) const noexcept override {
        return _write(_internal, name, data);
    }
};

[[nodiscard]] std::vector<uint32_t> assemble_shader_module(
    lc::vk::detail::ShaderArtifactSpirvStage stage,
    std::string_view extra_capabilities = {}) {
#if defined(LUISA_XIR_TO_SPIRV) || defined(LUISA_AST_LLVM_TO_SPIRV)
    auto execution_model = std::string_view{};
    auto execution_modes = std::string{};
    switch (stage) {
        case lc::vk::detail::ShaderArtifactSpirvStage::COMPUTE:
            execution_model = "GLCompute";
            execution_modes = "OpExecutionMode %main LocalSize 1 1 1\n";
            break;
        case lc::vk::detail::ShaderArtifactSpirvStage::VERTEX:
            execution_model = "Vertex";
            break;
        case lc::vk::detail::ShaderArtifactSpirvStage::FRAGMENT:
            execution_model = "Fragment";
            execution_modes = "OpExecutionMode %main OriginUpperLeft\n";
            break;
    }
    auto assembly = std::string{"OpCapability Shader\n"};
    assembly.append(extra_capabilities);
    assembly.append("OpMemoryModel Logical GLSL450\nOpEntryPoint ");
    assembly.append(execution_model);
    assembly.append(" %main \"main\"\n");
    assembly.append(execution_modes);
    assembly.append(
        "%void = OpTypeVoid\n"
        "%fn = OpTypeFunction %void\n"
        "%main = OpFunction %void None %fn\n"
        "%entry = OpLabel\n"
        "OpReturn\n"
        "OpFunctionEnd\n");
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    std::vector<uint32_t> words;
    LUISA_ASSERT(tools.Assemble(assembly, &words),
                 "Failed to assemble the Vulkan shader-artifact fixture.");
    return words;
#else
    static_cast<void>(stage);
    static_cast<void>(extra_capabilities);
    return {};
#endif
}

[[nodiscard]] vstd::MD5 byte_md5(
    luisa::span<const std::byte> bytes) {
    return vstd::MD5{vstd::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()}};
}

void replace_compute_header(
    luisa::vector<std::byte> &bytes,
    const lc::vk::detail::ShaderSerHeader &header) {
    LUISA_ASSERT(bytes.size() >= sizeof(header));
    std::memcpy(bytes.data(), &header, sizeof(header));
}

[[nodiscard]] lc::vk::detail::ShaderSerHeader compute_header(
    luisa::span<const std::byte> bytes) {
    lc::vk::detail::ShaderSerHeader header{};
    LUISA_ASSERT(bytes.size() >= sizeof(header));
    std::memcpy(&header, bytes.data(), sizeof(header));
    return header;
}

[[nodiscard]] constexpr lc::hlsl::Property sampler_property() noexcept {
    return {
        lc::hlsl::ShaderVariableType::SamplerHeap, 1u, 0u,
        lc::vk::detail::descriptor_interface_sampler_count};
}

[[nodiscard]] std::array<lc::hlsl::Property, 5u>
compute_round_trip_properties() noexcept {
    return {
        lc::hlsl::Property{lc::hlsl::ShaderVariableType::StructuredBuffer, 0u, 0u, 1u},
        lc::hlsl::Property{lc::hlsl::ShaderVariableType::ConstantBuffer, 0u, 1u, 1u},
        lc::hlsl::Property{lc::hlsl::ShaderVariableType::RWStructuredBuffer, 0u, 2u, 1u},
        lc::hlsl::Property{lc::hlsl::ShaderVariableType::RWStructuredBuffer, 0u, 3u, 1u},
        sampler_property()};
}

[[nodiscard]] std::array<lc::hlsl::Property, 4u>
compute_printer_properties() noexcept {
    return {
        lc::hlsl::Property{lc::hlsl::ShaderVariableType::StructuredBuffer, 0u, 0u, 1u},
        lc::hlsl::Property{lc::hlsl::ShaderVariableType::RWStructuredBuffer, 0u, 1u, 1u},
        lc::hlsl::Property{lc::hlsl::ShaderVariableType::RWStructuredBuffer, 0u, 2u, 1u},
        sampler_property()};
}

[[nodiscard]] lc::vk::SavedArgument uniform_int_argument() noexcept {
    auto argument = lc::vk::SavedArgument{};
    argument.tag = luisa::compute::Type::Tag::INT32;
    argument.var_usage = luisa::compute::Usage::READ;
    argument.struct_size = sizeof(int32_t);
    return argument;
}

[[nodiscard]] lc::vk::detail::ShaderSerHeader
make_compute_semantic_header() {
    using namespace lc::vk::detail;
    ShaderSerHeader header{
        .header_ver = kShaderSerVersion,
        .pipeline_ver = kXIRPipelineVersion,
        .md5 = test_md5("compute-shader"),
        .type_md5 = test_md5("compute-type"),
        .property_md5 = test_md5("compute-properties"),
        .argument_md5 = test_md5("compute-arguments"),
        .spv_md5 = test_md5("compute-spirv"),
        .property_size = 7u,
        .spv_byte_size = 44u,
        .block_size = {8u, 4u, 2u},
        .kernel_arg_count = 3u,
        .printer_count = 2u,
        .printer_size_bytes = 17u,
        .printer_md5 = test_md5("compute-printers"),
        .validation_count = 5u,
        .required_subgroup_size = 32u,
        .constant_ubo_size = 29u,
        .constant_ubo_md5 = test_md5("compute-constants"),
        .use_bindless_buffer = 1u,
        .use_bindless_tex2d = 0u,
        .use_bindless_tex3d = 1u,
        .codegen_dialect = static_cast<uint8_t>(
            ShaderCodegenDialect::XIR_SPIRV),
        .required_spirv_features =
            lc::spirv::target_feature::sampler_anisotropy |
            lc::spirv::target_feature::shader_int64};
    header.semantic_header_md5 = shader_semantic_header_md5(header);
    return header;
}

[[nodiscard]] lc::vk::detail::RasterSerHeader
make_raster_semantic_header() {
    using namespace lc::vk::detail;
    RasterSerHeader header{
        .header_ver = kShaderSerVersion,
        .pipeline_ver = kXIRPipelineVersion,
        .md5 = test_md5("raster-shader"),
        .type_md5 = test_md5("raster-type"),
        .property_md5 = test_md5("raster-properties"),
        .argument_md5 = test_md5("raster-arguments"),
        .vert_spv_md5 = test_md5("raster-vertex"),
        .pixel_spv_md5 = test_md5("raster-pixel"),
        .property_size = 5u,
        .vert_spv_byte_size = 60u,
        .pixel_spv_byte_size = 68u,
        .kernel_arg_count = 4u,
        .printer_count = 1u,
        .printer_size_bytes = 13u,
        .printer_md5 = test_md5("raster-printers"),
        .validation_count = 9u,
        .use_bindless_buffer = 0u,
        .use_bindless_tex2d = 1u,
        .use_bindless_tex3d = 1u,
        .codegen_dialect = static_cast<uint8_t>(
            ShaderCodegenDialect::HLSL_SPIRV),
        .required_spirv_features =
            lc::spirv::target_feature::sampler_anisotropy |
            lc::spirv::target_feature::shader_int64};
    header.semantic_header_md5 = raster_semantic_header_md5(header);
    return header;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_native_bindless_metadata_is_an_exact_optional_argument_role"_test = [] {
        using namespace lc::vk;
        using namespace lc::vk::detail;
        using lc::hlsl::Property;
        using lc::hlsl::ShaderVariableType;
        constexpr auto unbounded =
            std::numeric_limits<uint32_t>::max();
        auto bindless_argument = SavedArgument{};
        bindless_argument.tag =
            luisa::compute::Type::Tag::BINDLESS_ARRAY;
        bindless_argument.var_usage = luisa::compute::Usage::READ_WRITE;
        bindless_argument.set_native_bindless_roles(
            lc::spirv::kernel_argument_role::none);
        constexpr std::array typed_properties{
            Property{ShaderVariableType::StructuredBuffer, 0u, 0u, 1u},
            Property{ShaderVariableType::SPIRVIndirectDispatch, 0u, 1u, 1u},
            Property{ShaderVariableType::SamplerHeap, 1u, 0u,
                     descriptor_interface_sampler_count},
            Property{ShaderVariableType::SRVBufferHeap, 2u, 0u,
                     unbounded}};
        constexpr std::array mixed_properties{
            Property{ShaderVariableType::StructuredBuffer, 0u, 0u, 1u},
            Property{ShaderVariableType::SPIRVBindlessBufferMetadata,
                     0u, 1u, 1u},
            Property{ShaderVariableType::SPIRVIndirectDispatch, 0u, 2u, 1u},
            Property{ShaderVariableType::SamplerHeap, 1u, 0u,
                     descriptor_interface_sampler_count},
            Property{ShaderVariableType::SRVBufferHeap, 2u, 0u,
                     unbounded}};
        auto request = [&](auto &&properties) noexcept {
            return ShaderInterfaceRequest{
                .properties = properties,
                .arguments = luisa::span{&bindless_argument, 1u},
                .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
                .dialect = ShaderCodegenDialect::XIR_SPIRV,
                .use_buffer_bindless = true};
        };

        auto typed = plan_shader_interface(
            request(luisa::span{typed_properties}));
        expect(static_cast<bool>(typed))
            << shader_interface_error_name(typed.error);
        expect(eq(typed.resource_binding_count, 1u));
        expect(eq(typed.local_binding_count, 2u));

        auto mixed = plan_shader_interface(
            request(luisa::span{mixed_properties}));
        expect(static_cast<bool>(mixed))
            << shader_interface_error_name(mixed.error);
        expect(eq(mixed.resource_binding_count, 2u));
        expect(eq(mixed.local_binding_count, 3u));
    };

    "vk_shader_binary_sizes_are_overflow_checked"_test = [] {
        uint64_t value = 0u;
        expect(lc::vk::detail::checked_binary_product(7u, 11u, value));
        expect(eq(value, 77u));
        expect(!lc::vk::detail::checked_binary_product(
            std::numeric_limits<uint64_t>::max(), 2u, value));

        expect(lc::vk::detail::checked_binary_total(
            5u, {7u, 11u, 13u}, value));
        expect(eq(value, 36u));
        expect(!lc::vk::detail::checked_binary_total(
            std::numeric_limits<uint64_t>::max() - 1u,
            {1u, 1u}, value));
    };

    "vk_shader_binary_spirv_framing_is_exact"_test = [] {
        constexpr std::array<uint32_t, 5u> header{
            0x07230203u, 0x00010500u, 0u, 1u, 0u};
        expect(lc::vk::detail::valid_spirv_byte_size(sizeof(header)));
        expect(!lc::vk::detail::valid_spirv_byte_size(sizeof(header) - 1u));
        expect(lc::vk::detail::valid_spirv_header(header));

        auto invalid_magic = header;
        invalid_magic[0] = 0u;
        expect(!lc::vk::detail::valid_spirv_header(invalid_magic));
        auto invalid_bound = header;
        invalid_bound[3] = 0u;
        expect(!lc::vk::detail::valid_spirv_header(invalid_bound));
        auto invalid_schema = header;
        invalid_schema[4] = 1u;
        expect(!lc::vk::detail::valid_spirv_header(invalid_schema));

        expect(!lc::vk::detail::valid_spirv_byte_size(
            lc::vk::detail::max_spirv_module_byte_size + sizeof(uint32_t)))
            << "a word-aligned but implausibly large module must be rejected before allocation";
    };

    "vk_pipeline_cache_artifact_rejects_truncation_and_corruption"_test = [] {
        using namespace lc::vk::detail;
        std::array<std::byte, 32u> payload{};
        for (auto i = 0u; i < payload.size(); ++i) {
            payload[i] = static_cast<std::byte>(i);
        }
        PipelineCacheArtifactHeader header{
            .payload_size = payload.size(),
            .payload_md5 = pipeline_cache_payload_md5(payload)};
        constexpr auto header_size =
            sizeof(PipelineCacheArtifactHeader);
        expect(valid_pipeline_cache_artifact_framing(
            header, header_size + payload.size()));
        expect(valid_pipeline_cache_artifact_payload(header, payload));
        expect(!valid_pipeline_cache_artifact_framing(
            header, header_size + payload.size() - 1u));
        expect(!valid_pipeline_cache_artifact_framing(
            header, header_size + payload.size() + 1u));

        auto corrupted = payload;
        corrupted.back() ^= std::byte{0x80u};
        expect(!valid_pipeline_cache_artifact_payload(
            header, corrupted));

        auto bad_magic = header;
        bad_magic.magic ^= 1u;
        expect(!valid_pipeline_cache_artifact_framing(
            bad_magic, header_size + payload.size()));
        auto bad_version = header;
        bad_version.version++;
        expect(!valid_pipeline_cache_artifact_framing(
            bad_version, header_size + payload.size()));
        auto bad_reserved = header;
        bad_reserved.reserved = 1u;
        expect(!valid_pipeline_cache_artifact_framing(
            bad_reserved, header_size + payload.size()));
        auto oversized = header;
        oversized.payload_size =
            max_pipeline_cache_byte_size + 1u;
        expect(!valid_pipeline_cache_artifact_framing(
            oversized, header_size + oversized.payload_size));
    };

    "vk_shader_binary_table_limits_fail_closed_before_allocation"_test = [] {
        using namespace lc::vk::detail;
        expect(valid_shader_table_sizes(
            max_shader_property_count,
            max_shader_argument_count,
            max_shader_printer_count,
            max_shader_printer_payload_size));
        expect(!valid_shader_table_sizes(
            max_shader_property_count + 1u, 0u, 0u, 0u));
        expect(!valid_shader_table_sizes(
            0u, max_shader_argument_count + 1u, 0u, 0u));
        expect(!valid_shader_table_sizes(
            0u, 0u, max_shader_printer_count + 1u, 0u));
        expect(!valid_shader_table_sizes(
            0u, 0u, 0u, max_shader_printer_payload_size + 1u));
    };

    "vk_shader_binary_flags_are_canonical_bytes"_test = [] {
        using lc::vk::detail::valid_binary_flag;
        expect(valid_binary_flag(0u));
        expect(valid_binary_flag(1u));
        expect(!valid_binary_flag(2u));
        expect(!valid_binary_flag(std::numeric_limits<uint8_t>::max()));
    };

    "vk_shader_constant_payload_uses_one_fresh_and_cache_contract"_test = [] {
        using namespace lc::vk::detail;
        constexpr uint64_t device_limit = 16u * 1024u;
        expect(valid_shader_constant_payload_size(
            device_limit, device_limit));
        expect(!valid_shader_constant_payload_size(
            device_limit + 1u, device_limit));

        constexpr auto unlimited_device =
            std::numeric_limits<uint64_t>::max();
        expect(valid_shader_constant_payload_size(
            max_shader_constant_payload_size, unlimited_device));
        expect(!valid_shader_constant_payload_size(
            max_shader_constant_payload_size + 1u,
            unlimited_device));
    };

    "vk_shader_binary_spirv_feature_requirements_fail_closed"_test = [] {
        using namespace lc::spirv;
        constexpr auto required =
            target_feature::sampler_anisotropy |
            target_feature::shader_int64;

        constexpr auto exact = check_spirv_target_feature_requirements(
            required, required);
        expect(static_cast<bool>(exact));
        expect(eq(exact.unknown_required_bits, 0u));
        expect(eq(exact.missing_required_bits, 0u));

        constexpr auto superset = check_spirv_target_feature_requirements(
            required,
            required | target_feature::ray_query);
        expect(static_cast<bool>(superset));

        constexpr auto missing = check_spirv_target_feature_requirements(
            required, target_feature::sampler_anisotropy);
        expect(!static_cast<bool>(missing));
        expect(eq(missing.unknown_required_bits, 0u));
        expect(eq(missing.missing_required_bits,
                  target_feature::shader_int64));
        constexpr auto missing_list = list_spirv_target_features(
            missing.missing_required_bits);
        expect(eq(missing_list.count, 1u));
        expect(missing_list.features[0].name == "shaderInt64"sv);

        constexpr SpirvTargetFeatureMask unknown_bit = 1ull << 63u;
        constexpr auto unknown = check_spirv_target_feature_requirements(
            required | unknown_bit, target_feature::known_mask);
        expect(!static_cast<bool>(unknown));
        expect(eq(unknown.unknown_required_bits, unknown_bit));
        expect(eq(unknown.missing_required_bits, 0u));
    };

    "vk_compute_semantic_header_is_canonical_and_covers_every_field"_test = [] {
        using namespace lc::vk::detail;
        auto header = make_compute_semantic_header();
        expect(valid_shader_semantic_header(header));
        expect(header.semantic_header_md5.to_string(false) ==
               "3fbd5542f264277e7e8267ca3df494fa"sv)
            << "the compute semantic header must retain its canonical "
               "little-endian field order";
        expect(eq(shader_semantic_header_byte_size, 200u));

        auto rejects = [&](std::string_view field, auto mutate) {
            auto tampered = header;
            mutate(tampered);
            expect(!valid_shader_semantic_header(tampered))
                << "compute semantic digest omitted field " << field;
        };
        rejects("header_ver", [](auto &h) { h.header_ver++; });
        rejects("pipeline_ver", [](auto &h) { h.pipeline_ver++; });
        rejects("md5", [](auto &h) { h.md5 = test_md5("tampered"); });
        rejects("type_md5", [](auto &h) { h.type_md5 = test_md5("tampered"); });
        rejects("property_md5", [](auto &h) { h.property_md5 = test_md5("tampered"); });
        rejects("argument_md5", [](auto &h) { h.argument_md5 = test_md5("tampered"); });
        rejects("spv_md5", [](auto &h) { h.spv_md5 = test_md5("tampered"); });
        rejects("property_size", [](auto &h) { h.property_size++; });
        rejects("spv_byte_size", [](auto &h) { h.spv_byte_size++; });
        rejects("block_size[0]", [](auto &h) { h.block_size[0]++; });
        rejects("block_size[1]", [](auto &h) { h.block_size[1]++; });
        rejects("block_size[2]", [](auto &h) { h.block_size[2]++; });
        rejects("kernel_arg_count", [](auto &h) { h.kernel_arg_count++; });
        rejects("printer_count", [](auto &h) { h.printer_count++; });
        rejects("printer_size_bytes", [](auto &h) { h.printer_size_bytes++; });
        rejects("printer_md5", [](auto &h) { h.printer_md5 = test_md5("tampered"); });
        rejects("validation_count", [](auto &h) { h.validation_count++; });
        rejects("required_subgroup_size", [](auto &h) { h.required_subgroup_size++; });
        rejects("constant_ubo_size", [](auto &h) { h.constant_ubo_size++; });
        rejects("constant_ubo_md5", [](auto &h) { h.constant_ubo_md5 = test_md5("tampered"); });
        rejects("use_bindless_buffer", [](auto &h) { h.use_bindless_buffer ^= 1u; });
        rejects("use_bindless_tex2d", [](auto &h) { h.use_bindless_tex2d ^= 1u; });
        rejects("use_bindless_tex3d", [](auto &h) { h.use_bindless_tex3d ^= 1u; });
        rejects("codegen_dialect", [](auto &h) {
            h.codegen_dialect = static_cast<uint8_t>(
                ShaderCodegenDialect::LLVM_SPIRV);
        });
        rejects("required_spirv_features", [](auto &h) {
            h.required_spirv_features ^=
                lc::spirv::target_feature::ray_query;
        });
        rejects("semantic_header_md5", [](auto &h) {
            h.semantic_header_md5 = vstd::MD5{};
        });
    };

    "vk_raster_semantic_header_is_canonical_and_covers_every_field"_test = [] {
        using namespace lc::vk::detail;
        auto header = make_raster_semantic_header();
        expect(valid_raster_semantic_header(header));
        expect(header.semantic_header_md5.to_string(false) ==
               "c8de5e3ce0d230f117d768ae487cd6da"sv)
            << "the raster semantic header must retain its canonical "
               "little-endian field order";
        expect(eq(raster_semantic_header_byte_size, 184u));

        auto rejects = [&](std::string_view field, auto mutate) {
            auto tampered = header;
            mutate(tampered);
            expect(!valid_raster_semantic_header(tampered))
                << "raster semantic digest omitted field " << field;
        };
        rejects("header_ver", [](auto &h) { h.header_ver++; });
        rejects("pipeline_ver", [](auto &h) { h.pipeline_ver++; });
        rejects("md5", [](auto &h) { h.md5 = test_md5("tampered"); });
        rejects("type_md5", [](auto &h) { h.type_md5 = test_md5("tampered"); });
        rejects("property_md5", [](auto &h) { h.property_md5 = test_md5("tampered"); });
        rejects("argument_md5", [](auto &h) { h.argument_md5 = test_md5("tampered"); });
        rejects("vert_spv_md5", [](auto &h) { h.vert_spv_md5 = test_md5("tampered"); });
        rejects("pixel_spv_md5", [](auto &h) { h.pixel_spv_md5 = test_md5("tampered"); });
        rejects("property_size", [](auto &h) { h.property_size++; });
        rejects("vert_spv_byte_size", [](auto &h) { h.vert_spv_byte_size++; });
        rejects("pixel_spv_byte_size", [](auto &h) { h.pixel_spv_byte_size++; });
        rejects("kernel_arg_count", [](auto &h) { h.kernel_arg_count++; });
        rejects("printer_count", [](auto &h) { h.printer_count++; });
        rejects("printer_size_bytes", [](auto &h) { h.printer_size_bytes++; });
        rejects("printer_md5", [](auto &h) { h.printer_md5 = test_md5("tampered"); });
        rejects("validation_count", [](auto &h) { h.validation_count++; });
        rejects("use_bindless_buffer", [](auto &h) { h.use_bindless_buffer ^= 1u; });
        rejects("use_bindless_tex2d", [](auto &h) { h.use_bindless_tex2d ^= 1u; });
        rejects("use_bindless_tex3d", [](auto &h) { h.use_bindless_tex3d ^= 1u; });
        rejects("codegen_dialect", [](auto &h) {
            h.codegen_dialect = static_cast<uint8_t>(
                ShaderCodegenDialect::LLVM_SPIRV);
        });
        rejects("required_spirv_features", [](auto &h) {
            h.required_spirv_features ^=
                lc::spirv::target_feature::ray_query;
        });
        rejects("semantic_header_md5", [](auto &h) {
            h.semantic_header_md5 = vstd::MD5{};
        });
    };

    "vk_shader_binary_has_at_most_one_native_indirect_descriptor"_test = [] {
        using lc::hlsl::Property;
        using lc::hlsl::ShaderVariableType;
        constexpr Property ordinary{
            ShaderVariableType::StructuredBuffer, 0u, 0u, 1u};
        constexpr Property indirect_1{
            ShaderVariableType::SPIRVIndirectDispatch, 0u, 1u, 1u};
        constexpr Property indirect_2{
            ShaderVariableType::SPIRVIndirectDispatch, 0u, 2u, 1u};

        constexpr std::array no_indirect{ordinary};
        constexpr std::array one_indirect{ordinary, indirect_1};
        constexpr std::array duplicate_indirect{
            ordinary, indirect_1, indirect_2};
        expect(lc::vk::detail::valid_indirect_dispatch_property_contract(
            no_indirect, false));
        expect(lc::vk::detail::valid_indirect_dispatch_property_contract(
            one_indirect, true));
        expect(!lc::vk::detail::valid_indirect_dispatch_property_contract(
            one_indirect, false))
            << "raster and non-native property tables must reject the hidden descriptor";
        expect(!lc::vk::detail::valid_indirect_dispatch_property_contract(
            duplicate_indirect, true))
            << "runtime binding has exactly one hidden metadata slot";
    };

    "vk_shader_binary_printer_records_are_bounded_and_canonical"_test = [] {
        constexpr std::array valid{
            'x', '=', '\0', 'u', 'i', 'n', 't', '\0',
            'y', '\0', 'f', 'l', 'o', 'a', 't', '\0'};
        std::vector<std::pair<std::string_view, std::string_view>> records;
        auto accepted = lc::vk::detail::for_each_printer_record(
            luisa::span{valid}, 2u,
            [&](auto name, auto type) { records.emplace_back(name, type); });
        expect(accepted);
        expect(eq(records.size(), 2u));
        expect(records[0].first == "x="sv);
        expect(records[0].second == "uint"sv);
        expect(records[1].first == "y"sv);
        expect(records[1].second == "float"sv);

        constexpr std::array missing_type_terminator{
            'x', '\0', 'u', 'i', 'n', 't'};
        expect(!lc::vk::detail::for_each_printer_record(
            luisa::span{missing_type_terminator}, 1u,
            [](auto, auto) noexcept {}));
        expect(!lc::vk::detail::for_each_printer_record(
            luisa::span{valid}, 1u,
            [](auto, auto) noexcept {}))
            << "trailing bytes must not be silently ignored";
        expect(!lc::vk::detail::for_each_printer_record(
            luisa::span{valid}, 9u,
            [](auto, auto) noexcept {}))
            << "record count must fit at least two terminators per record";
    };

#if defined(LUISA_XIR_TO_SPIRV) || defined(LUISA_AST_LLVM_TO_SPIRV)
    "vk_compute_shader_artifact_codec_round_trips_through_binary_io"_test = [] {
        using namespace lc::vk::detail;
        auto spirv = assemble_shader_module(
            ShaderArtifactSpirvStage::COMPUTE);
        auto properties = compute_round_trip_properties();
        std::array arguments{uniform_int_argument()};
        auto printer_argument_pack = luisa::compute::Type::structure(
            {luisa::compute::Type::from("uint")});
        std::array printers{
            std::pair{luisa::string{"value={}"},
                      printer_argument_pack}};
        constexpr std::array constant_data{
            std::byte{0x11u}, std::byte{0x22u},
            std::byte{0x33u}, std::byte{0x44u}};
        auto shader_md5 = test_md5("codec-compute-shader");
        auto type_md5 = test_md5("codec-compute-types");
        auto bytes = encode_compute_shader_artifact(
            {.properties = properties,
             .arguments = arguments,
             .shader_md5 = shader_md5,
             .type_md5 = type_md5,
             .block_size = {8u, 4u, 1u},
             .spirv = spirv,
             .printers = printers,
             .constant_ubo_data = constant_data,
             .required_subgroup_size = 32u,
             .codegen_dialect = ShaderCodegenDialect::HLSL_SPIRV,
             .required_spirv_features =
                 lc::spirv::target_feature::sampler_anisotropy});

        MemoryBinaryIO io;
        static_cast<void>(io.write_shader_bytecode(
            "compute.vk", bytes));
        auto stream = io.read_shader_bytecode("compute.vk");
        expect(stream != nullptr);
        auto decoded = decode_compute_shader_artifact(
            *stream, shader_md5, type_md5,
            ShaderCodegenDialect::HLSL_SPIRV);
        expect(static_cast<bool>(decoded))
            << shader_artifact_codec_error_name(decoded.error);
        expect(eq(decoded.artifact.properties.size(), properties.size()));
        for (size_t i = 0u; i < properties.size(); ++i) {
            expect(decoded.artifact.properties[i].type == properties[i].type);
            expect(eq(decoded.artifact.properties[i].space_index,
                      properties[i].space_index));
            expect(eq(decoded.artifact.properties[i].register_index,
                      properties[i].register_index));
            expect(eq(decoded.artifact.properties[i].array_size,
                      properties[i].array_size));
        }
        expect(eq(decoded.artifact.arguments.size(), arguments.size()));
        expect(decoded.artifact.arguments[0].tag == arguments[0].tag);
        expect(decoded.artifact.arguments[0].var_usage ==
               arguments[0].var_usage);
        expect(eq(decoded.artifact.arguments[0].struct_size,
                  arguments[0].struct_size));
        expect(eq(decoded.artifact.arguments[0].buffer_metadata_index(),
                  arguments[0].buffer_metadata_index()));
        expect(decoded.artifact.spirv ==
               luisa::vector<uint32_t>{spirv.begin(), spirv.end()});
        expect(eq(decoded.artifact.printers.size(), 1u));
        expect(decoded.artifact.printers[0].first == "value={}"sv);
        expect(decoded.artifact.printers[0].second ==
               printer_argument_pack);
        expect(decoded.artifact.constant_ubo_data ==
               luisa::vector<std::byte>{
                   constant_data.begin(), constant_data.end()});
        expect(eq(decoded.artifact.header.block_size[0], 8u));
        expect(eq(decoded.artifact.header.block_size[1], 4u));
        expect(eq(decoded.artifact.header.required_subgroup_size, 32u));
        expect(eq(
            decoded.artifact.header.required_spirv_features,
            lc::spirv::target_feature::sampler_anisotropy));
    };

    "vk_native_accel_role_word_round_trips_through_artifact_codec"_test = [] {
        using namespace lc::vk;
        using namespace lc::vk::detail;
        auto spirv = assemble_shader_module(
            ShaderArtifactSpirvStage::COMPUTE);
        std::array properties{
            sampler_property(),
            lc::hlsl::Property{
                lc::hlsl::ShaderVariableType::SPIRVAccelInstance,
                0u, 0u, 1u},
            lc::hlsl::Property{
                lc::hlsl::ShaderVariableType::SPIRVIndirectDispatch,
                0u, 1u, 1u}};
        std::array<SavedArgument, 1u> arguments{};
        arguments[0u].tag = luisa::compute::Type::Tag::ACCEL;
        arguments[0u].var_usage = luisa::compute::Usage::READ;
        arguments[0u].set_native_accel_roles(
            SavedArgument::native_accel_role_instance);
        auto bytes = encode_compute_shader_artifact(
            {.properties = properties,
             .arguments = arguments,
             .shader_md5 = test_md5("codec-native-accel-role"),
             .type_md5 = test_md5("codec-native-accel-role-types"),
             .block_size = {1u, 1u, 1u},
             .spirv = spirv,
             .codegen_dialect = ShaderCodegenDialect::XIR_SPIRV});
        MemoryBinaryStream stream{bytes};
        auto decoded = decode_compute_shader_artifact(stream);
        expect(static_cast<bool>(decoded))
            << shader_artifact_codec_error_name(decoded.error);
        expect(eq(decoded.artifact.arguments.size(), size_t{1u}));
        expect(eq(
            decoded.artifact.arguments[0u].resource_aux,
            SavedArgument::native_accel_role_instance));
        expect(decoded.artifact.arguments[0u]
                   .native_accel_uses_instance_buffer());
        expect(!decoded.artifact.arguments[0u]
                    .native_accel_uses_traversal());
    };

    "vk_raster_shader_artifact_codec_round_trips_through_binary_io"_test = [] {
        using namespace lc::vk::detail;
        auto vertex = assemble_shader_module(
            ShaderArtifactSpirvStage::VERTEX);
        auto fragment = assemble_shader_module(
            ShaderArtifactSpirvStage::FRAGMENT);
        std::array properties{sampler_property()};
        auto shader_md5 = test_md5("codec-raster-shader");
        auto type_md5 = test_md5("codec-raster-types");
        auto bytes = encode_raster_shader_artifact(
            {.properties = properties,
             .shader_md5 = shader_md5,
             .type_md5 = type_md5,
             .vertex_spirv = vertex,
             .pixel_spirv = fragment,
             .codegen_dialect = ShaderCodegenDialect::HLSL_SPIRV});

        MemoryBinaryIO io;
        static_cast<void>(io.write_shader_cache("raster.vk", bytes));
        auto stream = io.read_shader_cache("raster.vk");
        expect(stream != nullptr);
        auto decoded = decode_raster_shader_artifact(
            *stream, shader_md5, type_md5,
            ShaderCodegenDialect::HLSL_SPIRV);
        expect(static_cast<bool>(decoded))
            << shader_artifact_codec_error_name(decoded.error);
        expect(eq(decoded.artifact.properties.size(), 1u));
        expect(decoded.artifact.vertex_spirv ==
               luisa::vector<uint32_t>{vertex.begin(), vertex.end()});
        expect(decoded.artifact.pixel_spirv ==
               luisa::vector<uint32_t>{fragment.begin(), fragment.end()});
        expect(decoded.artifact.arguments.empty());
        expect(decoded.artifact.printers.empty());

        MemoryBinaryStream wrong_dialect_stream{bytes};
        auto wrong_dialect = decode_raster_shader_artifact(
            wrong_dialect_stream, shader_md5, type_md5,
            ShaderCodegenDialect::XIR_SPIRV);
        expect(wrong_dialect.error ==
               ShaderArtifactCodecError::CODEGEN_DIALECT_MISMATCH);

        MemoryBinaryStream truncated{
            luisa::span<const std::byte>{bytes.data(), bytes.size() - 1u}};
        auto truncated_result = decode_raster_shader_artifact(
            truncated, shader_md5, type_md5);
        expect(truncated_result.error ==
               ShaderArtifactCodecError::INVALID_SECTION_SIZES);

        auto tampered = bytes;
        tampered.back() ^= std::byte{0x40u};
        MemoryBinaryStream tampered_stream{tampered};
        auto tampered_result = decode_raster_shader_artifact(
            tampered_stream, shader_md5, type_md5);
        expect(tampered_result.error ==
               ShaderArtifactCodecError::SECTION_DIGEST_MISMATCH);
    };

    "vk_shader_artifact_codec_rejects_truncation_and_tampering"_test = [] {
        using namespace lc::vk::detail;
        auto spirv = assemble_shader_module(
            ShaderArtifactSpirvStage::COMPUTE);
        std::array properties{sampler_property()};
        auto shader_md5 = test_md5("codec-integrity-shader");
        auto type_md5 = test_md5("codec-integrity-types");
        auto bytes = encode_compute_shader_artifact(
            {.properties = properties,
             .shader_md5 = shader_md5,
             .type_md5 = type_md5,
             .block_size = {1u, 1u, 1u},
             .spirv = spirv,
             .codegen_dialect = ShaderCodegenDialect::HLSL_SPIRV});

        MemoryBinaryStream truncated{
            luisa::span<const std::byte>{bytes.data(), bytes.size() - 1u}};
        auto truncated_result = decode_compute_shader_artifact(
            truncated, shader_md5, type_md5);
        expect(truncated_result.error ==
               ShaderArtifactCodecError::INVALID_SECTION_SIZES);

        auto tampered = bytes;
        tampered.back() ^= std::byte{0x80u};
        MemoryBinaryStream tampered_stream{tampered};
        auto tampered_result = decode_compute_shader_artifact(
            tampered_stream, shader_md5, type_md5);
        expect(tampered_result.error ==
               ShaderArtifactCodecError::SECTION_DIGEST_MISMATCH);

        auto wrong_type_stream = MemoryBinaryStream{bytes};
        auto identity_result = decode_compute_shader_artifact(
            wrong_type_stream, shader_md5, test_md5("wrong-types"));
        expect(identity_result.error ==
               ShaderArtifactCodecError::IDENTITY_MISMATCH);

        MemoryBinaryStream wrong_dialect_stream{bytes};
        auto dialect_result = decode_compute_shader_artifact(
            wrong_dialect_stream, shader_md5, type_md5,
            ShaderCodegenDialect::XIR_SPIRV);
        expect(dialect_result.error ==
               ShaderArtifactCodecError::CODEGEN_DIALECT_MISMATCH);
    };

    "vk_shader_artifact_codec_validates_spirv_after_recomputed_hashes"_test = [] {
        using namespace lc::vk::detail;
        auto spirv = assemble_shader_module(
            ShaderArtifactSpirvStage::COMPUTE);
        std::array properties{
            sampler_property(),
            lc::hlsl::Property{
                lc::hlsl::ShaderVariableType::SPIRVIndirectDispatch,
                0u, 0u, 1u}};
        auto bytes = encode_compute_shader_artifact(
            {.properties = properties,
             .shader_md5 = test_md5("codec-malformed-spirv"),
             .type_md5 = test_md5("codec-malformed-types"),
             .block_size = {1u, 1u, 1u},
             .spirv = spirv,
             .codegen_dialect = ShaderCodegenDialect::XIR_SPIRV});
        auto header = compute_header(bytes);
        auto spirv_offset = sizeof(ShaderSerHeader) +
                            header.property_size * sizeof(lc::hlsl::Property) +
                            header.kernel_arg_count * sizeof(lc::vk::SavedArgument);
        // Keep instruction framing and the compute "main" entry point intact,
        // but make every declared result id exceed the module id bound. This
        // passes the codec's structural preflight and must reach SPIRV-Tools.
        auto invalid_id_bound = 1u;
        std::memcpy(
            bytes.data() + spirv_offset + 3u * sizeof(uint32_t),
            &invalid_id_bound, sizeof(invalid_id_bound));
        auto spirv_bytes = luisa::span<const std::byte>{
            bytes.data() + spirv_offset,
            static_cast<size_t>(header.spv_byte_size)};
        header.spv_md5 = byte_md5(spirv_bytes);
        header.semantic_header_md5 = shader_semantic_header_md5(header);
        replace_compute_header(bytes, header);

        MemoryBinaryStream stream{bytes};
        auto decoded = decode_compute_shader_artifact(stream);
        expect(decoded.error == ShaderArtifactCodecError::INVALID_SPIRV)
            << "a self-consistent section hash must not bypass Vulkan SPIR-V validation";
    };

    "vk_shader_artifact_codec_rejects_self_consistent_invalid_printer_records"_test = [] {
        using namespace lc::vk::detail;
        auto spirv = assemble_shader_module(
            ShaderArtifactSpirvStage::COMPUTE);
        auto properties = compute_printer_properties();
        std::array arguments{uniform_int_argument()};
        auto printer_argument_pack = luisa::compute::Type::structure(
            {luisa::compute::Type::from("uint")});
        std::array printers{
            std::pair{luisa::string{"value={}"},
                      printer_argument_pack}};
        auto bytes = encode_compute_shader_artifact(
            {.properties = properties,
             .arguments = arguments,
             .shader_md5 = test_md5("codec-invalid-printer"),
             .type_md5 = test_md5("codec-invalid-printer-types"),
             .block_size = {1u, 1u, 1u},
             .spirv = spirv,
             .printers = printers,
             .codegen_dialect = ShaderCodegenDialect::HLSL_SPIRV});
        auto header = compute_header(bytes);
        auto printer_offset = sizeof(ShaderSerHeader) +
                              header.property_size * sizeof(lc::hlsl::Property) +
                              header.kernel_arg_count * sizeof(lc::vk::SavedArgument) +
                              header.spv_byte_size;
        constexpr auto format = "value={}"sv;
        auto decode_after_rehash = [printer_offset](
                                       luisa::vector<std::byte> candidate) {
            auto candidate_header = compute_header(candidate);
            auto printer_bytes = luisa::span<const std::byte>{
                candidate.data() + printer_offset,
                static_cast<size_t>(candidate_header.printer_size_bytes)};
            candidate_header.printer_md5 = byte_md5(printer_bytes);
            candidate_header.semantic_header_md5 =
                shader_semantic_header_md5(candidate_header);
            replace_compute_header(candidate, candidate_header);
            MemoryBinaryStream stream{candidate};
            return decode_compute_shader_artifact(stream).error;
        };

        auto invalid_type = bytes;
        auto type_offset = printer_offset + format.size() + 1u;
        invalid_type[type_offset] = std::byte{'!'};
        expect(decode_after_rehash(std::move(invalid_type)) ==
               ShaderArtifactCodecError::INVALID_PRINTER_PAYLOAD)
            << "a self-consistent artifact must reject an invalid printer type "
               "without entering Type::from's fatal parser";

        auto invalid_format = bytes;
        invalid_format[printer_offset + format.size() - 1u] =
            std::byte{'x'};
        expect(decode_after_rehash(std::move(invalid_format)) ==
               ShaderArtifactCodecError::INVALID_PRINTER_PAYLOAD)
            << "a self-consistent artifact must reject a format string that "
               "would violate ShaderPrintFormatter's brace contract";
    };

    "vk_shader_artifact_codec_rejects_a_valid_module_for_the_wrong_stage"_test = [] {
        using namespace lc::vk::detail;
        auto vertex = assemble_shader_module(
            ShaderArtifactSpirvStage::VERTEX);
        std::array modules{
            luisa::span<const uint32_t>{vertex}};
        constexpr std::array expected_stages{
            ShaderArtifactSpirvStage::COMPUTE};
        auto validation = validate_spirv_artifact_modules(
            ShaderCodegenDialect::HLSL_SPIRV, 0u,
            modules, expected_stages);
        expect(validation.error ==
               ShaderArtifactCodecError::INVALID_SPIRV)
            << "Vulkan validation alone does not prove that an artifact's "
               "entry point matches its pipeline stage";
        expect(eq(validation.failed_module_index, 0u));
    };

    "vk_native_shader_artifact_codec_reconciles_exact_and_union_masks"_test = [] {
        using namespace lc::vk::detail;
        auto int64_spirv = assemble_shader_module(
            ShaderArtifactSpirvStage::COMPUTE,
            "OpCapability Int64\n");
        auto float64_spirv = assemble_shader_module(
            ShaderArtifactSpirvStage::COMPUTE,
            "OpCapability Float64\n");
        constexpr auto emission_owned =
            lc::spirv::target_feature::sampler_anisotropy;
        constexpr auto expected_union =
            emission_owned |
            lc::spirv::target_feature::shader_int64 |
            lc::spirv::target_feature::shader_float64;
        std::array modules{
            luisa::span<const uint32_t>{int64_spirv},
            luisa::span<const uint32_t>{float64_spirv}};
        constexpr std::array stages{
            ShaderArtifactSpirvStage::COMPUTE,
            ShaderArtifactSpirvStage::COMPUTE};
        auto exact = validate_spirv_artifact_modules(
            ShaderCodegenDialect::XIR_SPIRV, expected_union,
            modules, stages);
        expect(static_cast<bool>(exact));
        expect(eq(exact.reconciled_features, expected_union));

        auto omitted = validate_spirv_artifact_modules(
            ShaderCodegenDialect::XIR_SPIRV, emission_owned,
            modules, stages);
        expect(omitted.error ==
               ShaderArtifactCodecError::NATIVE_FEATURE_MASK_MISMATCH);
        expect(eq(omitted.reconciled_features, expected_union));
    };

    "vk_native_shader_artifact_codec_rejects_mask_tampering_but_llvm_does_not_reconcile"_test = [] {
        using namespace lc::vk::detail;
        auto spirv = assemble_shader_module(
            ShaderArtifactSpirvStage::COMPUTE,
            "OpCapability Int64\n");
        std::array native_properties{
            lc::hlsl::Property{
                lc::hlsl::ShaderVariableType::SPIRVIndirectDispatch,
                0u, 0u, 1u},
            sampler_property()};
        constexpr auto native_requirements =
            lc::spirv::target_feature::shader_int64 |
            lc::spirv::target_feature::sampler_anisotropy;
        auto native_bytes = encode_compute_shader_artifact(
            {.properties = native_properties,
             .shader_md5 = test_md5("codec-native-mask"),
             .type_md5 = test_md5("codec-native-types"),
             .block_size = {1u, 1u, 1u},
             .spirv = spirv,
             .codegen_dialect = ShaderCodegenDialect::XIR_SPIRV,
             .required_spirv_features = native_requirements});
        auto native_header = compute_header(native_bytes);
        native_header.required_spirv_features =
            lc::spirv::target_feature::sampler_anisotropy;
        native_header.semantic_header_md5 =
            shader_semantic_header_md5(native_header);
        replace_compute_header(native_bytes, native_header);
        MemoryBinaryStream native_stream{native_bytes};
        auto native_decoded = decode_compute_shader_artifact(native_stream);
        expect(native_decoded.error ==
               ShaderArtifactCodecError::NATIVE_FEATURE_MASK_MISMATCH);

        std::array llvm_properties{sampler_property()};
        auto llvm_bytes = encode_compute_shader_artifact(
            {.properties = llvm_properties,
             .shader_md5 = test_md5("codec-llvm-mask"),
             .type_md5 = test_md5("codec-llvm-types"),
             .block_size = {1u, 1u, 1u},
             .spirv = spirv,
             .codegen_dialect = ShaderCodegenDialect::LLVM_SPIRV,
             .required_spirv_features =
                 lc::spirv::target_feature::sampler_anisotropy});
        MemoryBinaryStream llvm_stream{llvm_bytes};
        auto llvm_decoded = decode_compute_shader_artifact(llvm_stream);
        expect(static_cast<bool>(llvm_decoded))
            << "LLVM artifacts must validate SPIR-V but retain their own persisted feature contract";
        expect(eq(
            llvm_decoded.artifact.header.required_spirv_features,
            lc::spirv::target_feature::sampler_anisotropy));
    };
#endif
}
