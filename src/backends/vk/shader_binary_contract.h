#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/vstl/md5.h>

#include "../common/hlsl/shader_property.h"
#include "../common/spirv/spirv_codegen/target_feature_mask.h"

namespace lc::vk::detail {

// Shader bytecode is still laid out as a fixed-width native header followed
// by individually hashed sections. The integrity digest below never hashes a
// raw struct: padding is deliberately excluded by appending every semantic
// field to a canonical little-endian byte sequence.
inline constexpr uint32_t kShaderSerVersion = 10u;
inline constexpr uint32_t kXIRPipelineVersion = 4u;

struct ShaderSerHeader {
    uint64_t header_ver;
    uint32_t pipeline_ver;
    vstd::MD5 md5;
    vstd::MD5 type_md5;
    vstd::MD5 property_md5;
    vstd::MD5 argument_md5;
    vstd::MD5 spv_md5;
    uint64_t property_size;
    uint64_t spv_byte_size;
    uint32_t block_size[3];
    uint32_t kernel_arg_count;
    uint32_t printer_count;
    uint32_t printer_size_bytes;
    vstd::MD5 printer_md5;
    uint32_t validation_count;
    uint32_t required_subgroup_size;
    uint64_t constant_ubo_size;
    vstd::MD5 constant_ubo_md5;
    uint8_t use_bindless_buffer;
    uint8_t use_bindless_tex2d;
    uint8_t use_bindless_tex3d;
    uint8_t codegen_dialect;
    spirv::SpirvTargetFeatureMask required_spirv_features;
    vstd::MD5 semantic_header_md5;
};

struct RasterSerHeader {
    uint64_t header_ver;
    uint32_t pipeline_ver;
    vstd::MD5 md5;
    vstd::MD5 type_md5;
    vstd::MD5 property_md5;
    vstd::MD5 argument_md5;
    vstd::MD5 vert_spv_md5;
    vstd::MD5 pixel_spv_md5;
    uint64_t property_size;
    uint64_t vert_spv_byte_size;
    uint64_t pixel_spv_byte_size;
    uint32_t kernel_arg_count;
    uint32_t printer_count;
    uint32_t printer_size_bytes;
    vstd::MD5 printer_md5;
    uint32_t validation_count;
    uint8_t use_bindless_buffer;
    uint8_t use_bindless_tex2d;
    uint8_t use_bindless_tex3d;
    uint8_t codegen_dialect;
    spirv::SpirvTargetFeatureMask required_spirv_features;
    vstd::MD5 semantic_header_md5;
};

static_assert(std::is_trivially_copyable_v<ShaderSerHeader>);
static_assert(std::is_trivially_copyable_v<RasterSerHeader>);
static_assert(sizeof(vstd::MD5::MD5Data) == 16u);

namespace shader_binary_detail {

template<size_t Capacity>
class CanonicalHeaderBytes {
private:
    std::array<uint8_t, Capacity> _bytes{};
    size_t _size{};

    void _append(uint8_t value) noexcept {
        LUISA_ASSERT(
            _size < _bytes.size(),
            "Vulkan shader semantic-header encoding exceeded its fixed "
            "capacity of {} bytes.",
            _bytes.size());
        _bytes[_size++] = value;
    }

public:
    void append_u8(uint8_t value) noexcept { _append(value); }

    void append_u32(uint32_t value) noexcept {
        for (auto shift = 0u; shift < 32u; shift += 8u) {
            _append(static_cast<uint8_t>(value >> shift));
        }
    }

    void append_u64(uint64_t value) noexcept {
        for (auto shift = 0u; shift < 64u; shift += 8u) {
            _append(static_cast<uint8_t>(value >> shift));
        }
    }

    void append_md5(const vstd::MD5 &value) noexcept {
        auto data = value.to_binary();
        append_u64(data.data0);
        append_u64(data.data1);
    }

    [[nodiscard]] vstd::MD5 digest() const {
        LUISA_ASSERT(
            _size == _bytes.size(),
            "Vulkan shader semantic-header encoding produced {} bytes, "
            "but its contract requires exactly {} bytes.",
            _size, _bytes.size());
        return vstd::MD5{vstd::span<const uint8_t>{
            _bytes.data(), _size}};
    }
};

}// namespace shader_binary_detail

inline constexpr size_t shader_semantic_header_byte_size = 200u;
inline constexpr size_t raster_semantic_header_byte_size = 184u;

[[nodiscard]] inline vstd::MD5 shader_semantic_header_md5(
    const ShaderSerHeader &header) {
    // Domain separation keeps the two header kinds distinct even if a future
    // version happens to give them an identical field sequence.
    shader_binary_detail::CanonicalHeaderBytes<
        shader_semantic_header_byte_size>
        bytes;
    bytes.append_u64(0x314844524448534Cull);// compute-header domain tag
    bytes.append_u64(header.header_ver);
    bytes.append_u32(header.pipeline_ver);
    bytes.append_md5(header.md5);
    bytes.append_md5(header.type_md5);
    bytes.append_md5(header.property_md5);
    bytes.append_md5(header.argument_md5);
    bytes.append_md5(header.spv_md5);
    bytes.append_u64(header.property_size);
    bytes.append_u64(header.spv_byte_size);
    for (auto size : header.block_size) { bytes.append_u32(size); }
    bytes.append_u32(header.kernel_arg_count);
    bytes.append_u32(header.printer_count);
    bytes.append_u32(header.printer_size_bytes);
    bytes.append_md5(header.printer_md5);
    bytes.append_u32(header.validation_count);
    bytes.append_u32(header.required_subgroup_size);
    bytes.append_u64(header.constant_ubo_size);
    bytes.append_md5(header.constant_ubo_md5);
    bytes.append_u8(header.use_bindless_buffer);
    bytes.append_u8(header.use_bindless_tex2d);
    bytes.append_u8(header.use_bindless_tex3d);
    bytes.append_u8(header.codegen_dialect);
    bytes.append_u64(header.required_spirv_features);
    return bytes.digest();
}

[[nodiscard]] inline vstd::MD5 raster_semantic_header_md5(
    const RasterSerHeader &header) {
    shader_binary_detail::CanonicalHeaderBytes<
        raster_semantic_header_byte_size>
        bytes;
    bytes.append_u64(0x314844525453524Cull);// raster-header domain tag
    bytes.append_u64(header.header_ver);
    bytes.append_u32(header.pipeline_ver);
    bytes.append_md5(header.md5);
    bytes.append_md5(header.type_md5);
    bytes.append_md5(header.property_md5);
    bytes.append_md5(header.argument_md5);
    bytes.append_md5(header.vert_spv_md5);
    bytes.append_md5(header.pixel_spv_md5);
    bytes.append_u64(header.property_size);
    bytes.append_u64(header.vert_spv_byte_size);
    bytes.append_u64(header.pixel_spv_byte_size);
    bytes.append_u32(header.kernel_arg_count);
    bytes.append_u32(header.printer_count);
    bytes.append_u32(header.printer_size_bytes);
    bytes.append_md5(header.printer_md5);
    bytes.append_u32(header.validation_count);
    bytes.append_u8(header.use_bindless_buffer);
    bytes.append_u8(header.use_bindless_tex2d);
    bytes.append_u8(header.use_bindless_tex3d);
    bytes.append_u8(header.codegen_dialect);
    bytes.append_u64(header.required_spirv_features);
    return bytes.digest();
}

[[nodiscard]] inline bool valid_shader_semantic_header(
    const ShaderSerHeader &header) {
    return header.semantic_header_md5 ==
           shader_semantic_header_md5(header);
}

[[nodiscard]] inline bool valid_raster_semantic_header(
    const RasterSerHeader &header) {
    return header.semantic_header_md5 ==
           raster_semantic_header_md5(header);
}

// Bytecode/cache inputs are not trusted merely because their section sizes
// add up. These limits are intentionally far above what the front end can
// generate, while preventing corrupt headers from driving unbounded host
// allocations before semantic validation.
// Up to 256 local bindings plus the sampler, three global heaps, and the
// optional legacy ConstantValue push-constant marker.
inline constexpr uint64_t max_shader_property_count = 261u;
inline constexpr uint64_t max_shader_argument_count = 64u;
inline constexpr uint64_t max_shader_printer_count = 1u << 20u;
inline constexpr uint64_t max_shader_printer_payload_size = 64u << 20u;
inline constexpr uint64_t max_spirv_module_byte_size = 256u << 20u;
inline constexpr uint64_t max_pipeline_cache_byte_size = 256u << 20u;
inline constexpr uint64_t max_shader_constant_payload_size = 64u << 20u;
inline constexpr uint64_t max_shader_validation_count = 1u << 20u;

inline constexpr uint64_t pipeline_cache_artifact_magic =
    0x314f53504b56434cull;// "LCVKPSO1" in little-endian bytes
inline constexpr uint32_t pipeline_cache_artifact_version = 1u;

struct PipelineCacheArtifactHeader {
    uint64_t magic{pipeline_cache_artifact_magic};
    uint32_t version{pipeline_cache_artifact_version};
    uint32_t reserved{};
    uint64_t payload_size{};
    vstd::MD5 payload_md5{};
};

[[nodiscard]] constexpr bool valid_pipeline_cache_artifact_framing(
    const PipelineCacheArtifactHeader &header,
    uint64_t total_size) noexcept {
    return header.magic == pipeline_cache_artifact_magic &&
           header.version == pipeline_cache_artifact_version &&
           header.reserved == 0u &&
           header.payload_size <= max_pipeline_cache_byte_size &&
           total_size >= sizeof(PipelineCacheArtifactHeader) &&
           total_size - sizeof(PipelineCacheArtifactHeader) ==
               header.payload_size;
}

[[nodiscard]] inline vstd::MD5 pipeline_cache_payload_md5(
    luisa::span<const std::byte> payload) {
    return vstd::MD5{vstd::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(payload.data()),
        payload.size_bytes()}};
}

[[nodiscard]] inline bool valid_pipeline_cache_artifact_payload(
    const PipelineCacheArtifactHeader &header,
    luisa::span<const std::byte> payload) {
    return payload.size_bytes() == header.payload_size &&
           pipeline_cache_payload_md5(payload) == header.payload_md5;
}

// Persisted flags use fixed-width integers. Reading arbitrary bytes directly
// into a C++ bool can create a noncanonical representation before validation.
[[nodiscard]] constexpr bool valid_binary_flag(uint8_t value) noexcept {
    return value <= 1u;
}

// Freshly generated and deserialized shaders must apply the same device-facing
// UBO payload limit. The independent serialized-payload cap also protects host
// allocation when a device advertises an unexpectedly large range.
[[nodiscard]] constexpr bool valid_shader_constant_payload_size(
    uint64_t payload_size,
    uint64_t max_uniform_buffer_range) noexcept {
    return payload_size <= max_shader_constant_payload_size &&
           payload_size <= max_uniform_buffer_range;
}

[[nodiscard]] inline bool checked_binary_product(
    uint64_t count, uint64_t stride, uint64_t &result) noexcept {
    if (stride != 0u && count > std::numeric_limits<uint64_t>::max() / stride) {
        return false;
    }
    result = count * stride;
    return true;
}

[[nodiscard]] inline bool checked_binary_total(
    uint64_t header_size, std::initializer_list<uint64_t> section_sizes,
    uint64_t &result) noexcept {
    result = header_size;
    for (auto size : section_sizes) {
        if (size > std::numeric_limits<uint64_t>::max() - result) {
            return false;
        }
        result += size;
    }
    return true;
}

[[nodiscard]] inline bool valid_spirv_byte_size(uint64_t byte_size) noexcept {
    constexpr auto header_size = 5u * sizeof(uint32_t);
    return byte_size >= header_size &&
           byte_size <= max_spirv_module_byte_size &&
           byte_size % sizeof(uint32_t) == 0u &&
           byte_size / sizeof(uint32_t) <=
               std::numeric_limits<size_t>::max();
}

[[nodiscard]] inline bool valid_shader_table_sizes(
    uint64_t property_count, uint64_t argument_count,
    uint64_t printer_count, uint64_t printer_payload_size) noexcept {
    return property_count <= max_shader_property_count &&
           argument_count <= max_shader_argument_count &&
           printer_count <= max_shader_printer_count &&
           printer_payload_size <= max_shader_printer_payload_size;
}

[[nodiscard]] inline bool valid_indirect_dispatch_property_contract(
    luisa::span<const hlsl::Property> properties,
    bool allow_indirect_dispatch) noexcept {
    auto count = uint32_t{0u};
    for (auto property : properties) {
        if (property.type !=
            hlsl::ShaderVariableType::SPIRVIndirectDispatch) {
            continue;
        }
        if (!allow_indirect_dispatch || ++count > 1u) { return false; }
    }
    return true;
}

[[nodiscard]] inline bool valid_spirv_header(
    luisa::span<const uint32_t> words) noexcept {
    // SPIR-V header: magic, version, generator, id bound, reserved schema.
    constexpr uint32_t spirv_magic = 0x07230203u;
    constexpr uint32_t spirv_1_0 = 0x00010000u;
    constexpr uint32_t spirv_1_5 = 0x00010500u;
    return words.size() >= 5u && words[0] == spirv_magic &&
           words[1] >= spirv_1_0 && words[1] <= spirv_1_5 &&
           words[3] != 0u && words[4] == 0u;
}

template<typename F>
[[nodiscard]] bool for_each_printer_record(
    luisa::span<const char> bytes, uint64_t record_count,
    F &&visitor) noexcept(noexcept(std::declval<F &>()(luisa::string_view{}, luisa::string_view{}))) {
    // Every record contains two zero-terminated strings. This lower bound also
    // prevents an attacker-controlled count from causing a long empty loop.
    if (record_count > bytes.size() / 2u) { return false; }
    size_t offset = 0u;
    for (uint64_t i = 0u; i < record_count; ++i) {
        auto remaining = bytes.size() - offset;
        auto *name_end = static_cast<const char *>(
            std::memchr(bytes.data() + offset, '\0', remaining));
        if (name_end == nullptr) { return false; }
        auto name_size = static_cast<size_t>(name_end - (bytes.data() + offset));
        luisa::string_view name{bytes.data() + offset, name_size};
        offset += name_size + 1u;

        remaining = bytes.size() - offset;
        auto *type_end = static_cast<const char *>(
            std::memchr(bytes.data() + offset, '\0', remaining));
        if (type_end == nullptr) { return false; }
        auto type_size = static_cast<size_t>(type_end - (bytes.data() + offset));
        luisa::string_view type{bytes.data() + offset, type_size};
        offset += type_size + 1u;
        visitor(name, type);
    }
    // Trailing bytes would make the record count ambiguous and are rejected.
    return offset == bytes.size();
}

}// namespace lc::vk::detail
