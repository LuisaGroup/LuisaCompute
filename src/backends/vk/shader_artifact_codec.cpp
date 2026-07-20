#include "shader_artifact_codec.h"

#include <cstring>
#include <initializer_list>
#include <limits>
#include <type_traits>

#include <luisa/core/logging.h>

#if defined(LUISA_XIR_TO_SPIRV) || defined(LUISA_AST_LLVM_TO_SPIRV)
#include "../common/spirv/spirv_codegen/optimizer.h"
#endif

namespace lc::vk::detail {
namespace {

constexpr size_t max_shader_printer_type_description_size = 16u * 1024u;
constexpr size_t max_shader_printer_type_depth = 64u;
constexpr size_t max_shader_printer_type_nodes = 4096u;

[[nodiscard]] bool shader_printer_scalar_type(
    const luisa::compute::Type *type) noexcept {
    using Tag = luisa::compute::Type::Tag;
    if (type == nullptr) { return false; }
    switch (type->tag()) {
        case Tag::BOOL:
        case Tag::INT8:
        case Tag::UINT8:
        case Tag::INT16:
        case Tag::UINT16:
        case Tag::INT32:
        case Tag::UINT32:
        case Tag::INT64:
        case Tag::UINT64:
        case Tag::FLOAT16:
        case Tag::FLOAT32:
        case Tag::FLOAT64: return true;
        default: return false;
    }
}

[[nodiscard]] bool shader_printer_value_type(
    const luisa::compute::Type *type, size_t depth = 0u) noexcept {
    if (type == nullptr || depth > max_shader_printer_type_depth) {
        return false;
    }
    if (shader_printer_scalar_type(type) || type->is_matrix()) {
        return true;
    }
    if (type->is_vector()) {
        return shader_printer_scalar_type(type->element());
    }
    if (type->is_array()) {
        return shader_printer_value_type(type->element(), depth + 1u);
    }
    if (type->is_structure()) {
        for (auto member : type->members()) {
            if (!shader_printer_value_type(member, depth + 1u)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

[[nodiscard]] bool shader_printer_placeholder_count(
    luisa::string_view format, size_t &count) noexcept {
    count = 0u;
    for (size_t i = 0u; i < format.size(); ++i) {
        auto c = format[i];
        if (c != '{' && c != '}') { continue; }
        if (++i == format.size()) { return false; }
        auto next = format[i];
        if (c == '{') {
            if (next == '}') {
                count++;
            } else if (next != '{') {
                return false;
            }
        } else if (next != '}') {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool valid_shader_printer_record(
    luisa::string_view format,
    const luisa::compute::Type *argument_pack) noexcept {
    if (argument_pack == nullptr || !argument_pack->is_structure() ||
        !shader_printer_value_type(argument_pack) ||
        argument_pack->description().size() >
            max_shader_printer_type_description_size ||
        format.find('\0') != luisa::string_view::npos) {
        return false;
    }
    size_t placeholder_count;
    return shader_printer_placeholder_count(format, placeholder_count) &&
           placeholder_count <= argument_pack->members().size();
}

// Type::from() reports malformed descriptions with a fatal diagnostic. Shader
// artifacts are input data, so never pass artifact text directly to it:
// validate and reconstruct the narrower ShaderPrintFormatter type dialect.
class ShaderPrinterTypeParser {
private:
    luisa::string_view _text;
    size_t _offset{};
    size_t _node_count{};

private:
    [[nodiscard]] bool _punctuation(char c) noexcept {
        if (_offset < _text.size() && _text[_offset] == c) {
            _offset++;
            return true;
        }
        return false;
    }

    [[nodiscard]] luisa::string_view _identifier() noexcept {
        auto begin = _offset;
        while (_offset < _text.size()) {
            auto c = _text[_offset];
            auto identifier_character =
                (c >= 'a' && c <= 'z') ||
                (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_';
            if (!identifier_character) { break; }
            _offset++;
        }
        return _text.substr(begin, _offset - begin);
    }

    [[nodiscard]] bool _number(size_t &value) noexcept {
        auto begin = _offset;
        constexpr auto limit =
            static_cast<size_t>(std::numeric_limits<uint32_t>::max());
        value = 0u;
        while (_offset < _text.size() &&
               _text[_offset] >= '0' && _text[_offset] <= '9') {
            auto digit = static_cast<size_t>(_text[_offset] - '0');
            if (value > (limit - digit) / 10u) { return false; }
            value = value * 10u + digit;
            _offset++;
        }
        return _offset != begin;
    }

    [[nodiscard]] bool _attribute(
        luisa::compute::Attribute &attribute) noexcept {
        if (!_punctuation('[')) { return true; }
        auto key = _identifier();
        if (key.empty()) { return false; }
        auto value = luisa::string_view{};
        if (_punctuation('(')) {
            value = _identifier();
            if (value.empty() || !_punctuation(')')) { return false; }
        }
        if (!_punctuation(']')) { return false; }
        attribute = luisa::compute::Attribute{
            luisa::string{key}, luisa::string{value}};
        return true;
    }

    [[nodiscard]] static bool _valid_array_layout(
        const luisa::compute::Type *element, size_t dimension) noexcept {
        constexpr auto limit =
            static_cast<size_t>(std::numeric_limits<uint32_t>::max());
        return dimension == 0u || element->size() <= limit / dimension;
    }

    [[nodiscard]] static bool _append_struct_member_layout(
        size_t &offset, const luisa::compute::Type *member) noexcept {
        constexpr auto limit =
            static_cast<size_t>(std::numeric_limits<uint32_t>::max());
        auto alignment = member->alignment();
        if (alignment == 0u || alignment > limit) { return false; }
        auto remainder = offset % alignment;
        auto padding = remainder == 0u ? 0u : alignment - remainder;
        if (padding > limit - offset) { return false; }
        offset += padding;
        if (member->size() > limit - offset) { return false; }
        offset += member->size();
        return true;
    }

    [[nodiscard]] const luisa::compute::Type *_type(size_t depth) noexcept {
        using luisa::compute::Type;
        if (depth > max_shader_printer_type_depth ||
            _node_count++ >= max_shader_printer_type_nodes) {
            return nullptr;
        }
        auto name = _identifier();
        if (name.empty()) { return nullptr; }
        if (name == "bool" || name == "byte" || name == "ubyte" ||
            name == "short" || name == "ushort" || name == "int" ||
            name == "uint" || name == "long" || name == "ulong" ||
            name == "half" || name == "float" || name == "double") {
            return Type::from(name);
        }
        if (name == "vector") {
            size_t dimension;
            if (!_punctuation('<')) { return nullptr; }
            auto element = _type(depth + 1u);
            if (!shader_printer_scalar_type(element) ||
                !_punctuation(',') || !_number(dimension) ||
                dimension < 2u || dimension > 4u ||
                !_punctuation('>')) {
                return nullptr;
            }
            return Type::vector(element, dimension);
        }
        if (name == "matrix") {
            size_t dimension;
            if (!_punctuation('<') || !_number(dimension) ||
                dimension < 2u || dimension > 4u ||
                !_punctuation('>')) {
                return nullptr;
            }
            return Type::matrix(dimension);
        }
        if (name == "array") {
            size_t dimension;
            if (!_punctuation('<')) { return nullptr; }
            auto element = _type(depth + 1u);
            if (element == nullptr || !_punctuation(',') ||
                !_number(dimension) || !_punctuation('>') ||
                !_valid_array_layout(element, dimension)) {
                return nullptr;
            }
            return Type::array(element, dimension);
        }
        if (name == "struct") {
            size_t alignment;
            if (!_punctuation('<') || !_number(alignment) ||
                (alignment != 1u && alignment != 4u &&
                 alignment != 8u && alignment != 16u)) {
                return nullptr;
            }
            luisa::vector<const Type *> members;
            luisa::vector<luisa::compute::Attribute> attributes;
            auto has_attribute = false;
            auto layout_size = size_t{};
            while (_punctuation(',')) {
                auto attribute = luisa::compute::Attribute{};
                if (!_attribute(attribute)) { return nullptr; }
                has_attribute |= static_cast<bool>(attribute);
                auto member = _type(depth + 1u);
                if (member == nullptr || member->alignment() > alignment ||
                    !_append_struct_member_layout(layout_size, member)) {
                    return nullptr;
                }
                members.emplace_back(member);
                attributes.emplace_back(std::move(attribute));
            }
            auto final_remainder = layout_size % alignment;
            auto final_padding = final_remainder == 0u ?
                                     0u :
                                     alignment - final_remainder;
            constexpr auto limit =
                static_cast<size_t>(std::numeric_limits<uint32_t>::max());
            if (!_punctuation('>') || final_padding > limit - layout_size) {
                return nullptr;
            }
            if (has_attribute) {
                return Type::structure(alignment, members, attributes);
            }
            return Type::structure(alignment, members);
        }
        return nullptr;
    }

public:
    explicit ShaderPrinterTypeParser(luisa::string_view text) noexcept
        : _text{text} {}

    [[nodiscard]] const luisa::compute::Type *parse() noexcept {
        if (_text.empty() ||
            _text.size() > max_shader_printer_type_description_size) {
            return nullptr;
        }
        auto type = _type(0u);
        return type != nullptr && _offset == _text.size() &&
                       type->is_structure() &&
                       type->description() == _text ?
                   type :
                   nullptr;
    }
};

[[nodiscard]] vstd::MD5 binary_md5(
    const void *data, size_t size) noexcept {
    return vstd::MD5{vstd::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(data), size}};
}

template<typename T>
[[nodiscard]] vstd::MD5 binary_md5(
    luisa::span<const T> values) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    return binary_md5(values.data(), values.size_bytes());
}

[[nodiscard]] vstd::MD5 shader_property_md5(
    luisa::span<const hlsl::Property> properties) noexcept {
    struct Record {
        uint32_t type;
        uint32_t space;
        uint32_t binding;
        uint32_t count;
    };
    luisa::vector<Record> records;
    records.reserve(properties.size());
    for (auto property : properties) {
        records.emplace_back(Record{
            static_cast<uint32_t>(property.type), property.space_index,
            property.register_index, property.array_size});
    }
    return binary_md5(luisa::span<const Record>{records});
}

[[nodiscard]] vstd::MD5 saved_argument_md5(
    luisa::span<const SavedArgument> arguments) noexcept {
    struct Record {
        uint32_t tag;
        uint32_t usage;
        uint32_t size;
        uint32_t resource_aux;
    };
    luisa::vector<Record> records;
    records.reserve(arguments.size());
    for (auto argument : arguments) {
        records.emplace_back(Record{
            static_cast<uint32_t>(argument.tag),
            static_cast<uint32_t>(argument.var_usage),
            argument.struct_size, argument.resource_aux});
    }
    return binary_md5(luisa::span<const Record>{records});
}

[[nodiscard]] size_t serialized_printer_payload_size(
    luisa::span<const std::pair<luisa::string, luisa::compute::Type const *>> printers) {
    LUISA_ASSERT(
        printers.size() <= max_shader_printer_count,
        "Too many Vulkan shader printer records to serialize: {}.",
        printers.size());
    uint64_t size = 0u;
    for (auto &&[format, type] : printers) {
        LUISA_ASSERT(
            valid_shader_printer_record(format, type),
            "Vulkan shader printer '{}' has an invalid format or argument-pack type.",
            format);
        uint64_t record_size;
        LUISA_ASSERT(
            checked_binary_total(
                2u, {format.size(), type->description().size()}, record_size) &&
                record_size <= max_shader_printer_payload_size - size,
            "Vulkan shader printer payload exceeds the serialized bytecode limit.");
        size += record_size;
    }
    return static_cast<size_t>(size);
}

[[nodiscard]] luisa::vector<std::byte> serialize_printer_payload(
    luisa::span<const std::pair<luisa::string, luisa::compute::Type const *>> printers) {
    auto payload_size = serialized_printer_payload_size(printers);
    luisa::vector<std::byte> payload(payload_size);
    if (payload.empty()) { return payload; }
    auto *dst = payload.data();
    for (auto &&[format, type] : printers) {
        std::memcpy(dst, format.data(), format.size());
        dst += format.size();
        *dst++ = std::byte{0u};
        auto description = type->description();
        std::memcpy(dst, description.data(), description.size());
        dst += description.size();
        *dst++ = std::byte{0u};
    }
    LUISA_ASSERT(dst == payload.data() + payload.size(),
                 "Vulkan shader printer serializer size contract drifted.");
    return payload;
}

[[nodiscard]] bool decode_printers(
    luisa::span<const char> bytes, uint64_t printer_count,
    luisa::vector<std::pair<luisa::string, luisa::compute::Type const *>> &printers) {
    printers.reserve(static_cast<size_t>(printer_count));
    auto valid = true;
    auto records_valid = for_each_printer_record(
        bytes, printer_count,
        [&](luisa::string_view name, luisa::string_view type_description) {
            if (!valid) { return; }
            size_t placeholder_count;
            if (!shader_printer_placeholder_count(
                    name, placeholder_count)) {
                valid = false;
                return;
            }
            ShaderPrinterTypeParser parser{type_description};
            auto type = parser.parse();
            if (type == nullptr ||
                placeholder_count > type->members().size()) {
                valid = false;
                return;
            }
            printers.emplace_back(luisa::string{name}, type);
        });
    return records_valid && valid;
}

[[nodiscard]] size_t checked_serialized_size(
    uint64_t header_size, std::initializer_list<uint64_t> sections) {
    uint64_t size;
    LUISA_ASSERT(
        checked_binary_total(header_size, sections, size) &&
            size <= std::numeric_limits<size_t>::max(),
        "Vulkan shader binary exceeds the addressable host size.");
    return static_cast<size_t>(size);
}

template<typename T>
void append_bytes(std::byte *&dst, luisa::span<const T> values) noexcept {
    if (!values.empty()) {
        std::memcpy(dst, values.data(), values.size_bytes());
        dst += values.size_bytes();
    }
}

template<typename T>
void append_value(std::byte *&dst, const T &value) noexcept {
    std::memcpy(dst, &value, sizeof(T));
    dst += sizeof(T);
}

[[nodiscard]] bool spirv_has_entry_point(
    luisa::span<const uint32_t> words,
    ShaderArtifactSpirvStage expected_stage) noexcept {
    // SPIR-V fixed enumerants: OpEntryPoint = 15; execution models are
    // Vertex = 0, Fragment = 4, and GLCompute = 5.
    constexpr uint32_t op_entry_point = 15u;
    auto expected_model = uint32_t{};
    switch (expected_stage) {
        case ShaderArtifactSpirvStage::COMPUTE: expected_model = 5u; break;
        case ShaderArtifactSpirvStage::VERTEX: expected_model = 0u; break;
        case ShaderArtifactSpirvStage::FRAGMENT: expected_model = 4u; break;
    }
    for (size_t offset = 5u; offset < words.size();) {
        auto instruction = words[offset];
        auto word_count = static_cast<size_t>(instruction >> 16u);
        auto opcode = instruction & 0xffffu;
        if (word_count == 0u || word_count > words.size() - offset) {
            return false;
        }
        if (opcode == op_entry_point && word_count >= 4u &&
            words[offset + 1u] == expected_model) {
            // The runtime creates every pipeline with pName = "main". Check
            // the complete NUL-terminated entry-point name inside this
            // instruction instead of accepting another entry point of the
            // same execution model.
            auto name_words = luisa::span{
                words.data() + offset + 3u, word_count - 3u};
            auto name_bytes = luisa::span{
                reinterpret_cast<const char *>(name_words.data()),
                name_words.size_bytes()};
            auto *terminator = static_cast<const char *>(
                std::memchr(name_bytes.data(), '\0', name_bytes.size()));
            if (terminator != nullptr &&
                static_cast<size_t>(terminator - name_bytes.data()) == 4u &&
                std::memcmp(name_bytes.data(), "main", 4u) == 0) {
                return true;
            }
        }
        offset += word_count;
    }
    return false;
}

}// namespace

const char *shader_artifact_codec_error_name(
    ShaderArtifactCodecError error) noexcept {
    switch (error) {
        case ShaderArtifactCodecError::NONE: return "none";
        case ShaderArtifactCodecError::TRUNCATED_HEADER: return "truncated header";
        case ShaderArtifactCodecError::INVALID_HEADER: return "invalid semantic header";
        case ShaderArtifactCodecError::IDENTITY_MISMATCH: return "shader identity mismatch";
        case ShaderArtifactCodecError::CODEGEN_DIALECT_MISMATCH: return "shader codegen dialect mismatch";
        case ShaderArtifactCodecError::INVALID_SECTION_SIZES: return "invalid section sizes";
        case ShaderArtifactCodecError::SECTION_DIGEST_MISMATCH: return "section digest mismatch";
        case ShaderArtifactCodecError::INVALID_SHADER_INTERFACE: return "invalid shader interface";
        case ShaderArtifactCodecError::INVALID_PRINTER_PAYLOAD: return "invalid printer payload";
        case ShaderArtifactCodecError::INVALID_SPIRV: return "invalid SPIR-V module";
        case ShaderArtifactCodecError::NATIVE_FEATURE_MASK_UNAVAILABLE: return "native feature-mask validation unavailable";
        case ShaderArtifactCodecError::NATIVE_FEATURE_MASK_MISMATCH: return "native feature-mask mismatch";
    }
    return "unknown shader-artifact codec error";
}

SpirvArtifactModuleValidationResult validate_spirv_artifact_modules(
    ShaderCodegenDialect dialect,
    spirv::SpirvTargetFeatureMask persisted_features,
    luisa::span<const luisa::span<const uint32_t>> modules,
    luisa::span<const ShaderArtifactSpirvStage> stages) {
    auto result = SpirvArtifactModuleValidationResult{
        .reconciled_features = persisted_features};
    if ((persisted_features & ~spirv::target_feature::known_mask) != 0u) {
        result.error = ShaderArtifactCodecError::INVALID_HEADER;
        result.diagnostics = luisa::format(
            "Persisted SPIR-V feature mask contains unknown bits 0x{:016x}.",
            persisted_features & ~spirv::target_feature::known_mask);
        return result;
    }
    if (modules.size() != stages.size() || modules.empty()) {
        result.error = ShaderArtifactCodecError::INVALID_SPIRV;
        result.diagnostics = "SPIR-V module/stage list is empty or mismatched.";
        return result;
    }
    for (size_t i = 0u; i < modules.size(); ++i) {
        result.failed_module_index = i;
        auto module = modules[i];
        if (!valid_spirv_header(module) ||
            !spirv_has_entry_point(module, stages[i])) {
            result.error = ShaderArtifactCodecError::INVALID_SPIRV;
            result.diagnostics = "SPIR-V header, instruction framing, or expected main entry point is invalid.";
            return result;
        }
#if defined(LUISA_XIR_TO_SPIRV) || defined(LUISA_AST_LLVM_TO_SPIRV)
        auto validation = spirv::validate_spirv(
            module.data(), module.size());
        if (!validation.valid) {
            result.error = ShaderArtifactCodecError::INVALID_SPIRV;
            result.diagnostics = std::move(validation.diagnostics);
            return result;
        }
        result.has_warning |= validation.has_warning;
        if (!validation.diagnostics.empty()) {
            result.diagnostics.append(validation.diagnostics);
        }
#endif
    }
    result.failed_module_index = 0u;
    if (dialect != ShaderCodegenDialect::XIR_SPIRV) {
        return result;
    }
#if defined(LUISA_XIR_TO_SPIRV) || defined(LUISA_AST_LLVM_TO_SPIRV)
    result.reconciled_features = spirv::reconcile_spirv_target_features(
        modules, persisted_features);
    if (result.reconciled_features != persisted_features) {
        result.error = ShaderArtifactCodecError::NATIVE_FEATURE_MASK_MISMATCH;
        result.diagnostics = luisa::format(
            "Persisted native SPIR-V feature mask 0x{:016x} does not match "
            "the validated module requirements 0x{:016x} (missing "
            "0x{:016x}, stale 0x{:016x}).",
            persisted_features, result.reconciled_features,
            result.reconciled_features & ~persisted_features,
            persisted_features & ~result.reconciled_features);
    }
#else
    result.reconciled_features = 0u;
    result.error = ShaderArtifactCodecError::NATIVE_FEATURE_MASK_UNAVAILABLE;
    result.diagnostics =
        "This build cannot validate native XIR-SPIR-V capability requirements.";
#endif
    return result;
}

luisa::vector<std::byte> encode_compute_shader_artifact(
    const ComputeShaderArtifactEncodeInfo &info) {
    auto interface_plan = plan_shader_interface(
        {.properties = info.properties,
         .arguments = info.arguments,
         .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
         .dialect = info.codegen_dialect,
         .printer_count = static_cast<uint32_t>(info.printers.size()),
         .validation_count = info.validation_count,
         .use_buffer_bindless = info.use_buffer_bindless,
         .use_tex2d_bindless = info.use_tex2d_bindless,
         .use_tex3d_bindless = info.use_tex3d_bindless,
         .has_constant_ubo_payload = !info.constant_ubo_data.empty()});
    LUISA_ASSERT(
        interface_plan,
        "Vulkan compute shader serialization received an invalid runtime interface: {}.",
        shader_interface_error_name(interface_plan.error));
    LUISA_ASSERT(
        (info.required_spirv_features & ~spirv::target_feature::known_mask) == 0u,
        "Vulkan compute shader serialization received unknown SPIR-V target-feature bits 0x{:016x}.",
        info.required_spirv_features & ~spirv::target_feature::known_mask);
    LUISA_ASSERT(
        valid_shader_table_sizes(
            info.properties.size(), info.arguments.size(),
            info.printers.size(), 0u) &&
            valid_spirv_byte_size(info.spirv.size_bytes()) &&
            valid_shader_constant_payload_size(
                info.constant_ubo_data.size_bytes(),
                std::numeric_limits<uint64_t>::max()) &&
            info.validation_count <= max_shader_validation_count &&
            info.block_size[0] != 0u && info.block_size[1] != 0u &&
            info.block_size[2] != 0u &&
            (info.required_subgroup_size == 0u ||
             (info.required_subgroup_size <= std::numeric_limits<uint8_t>::max() &&
              (info.required_subgroup_size &
               (info.required_subgroup_size - 1u)) == 0u)),
        "Vulkan compute shader exceeds the serialized bytecode limits.");
    std::array modules{info.spirv};
    constexpr std::array stages{ShaderArtifactSpirvStage::COMPUTE};
    auto spirv_validation = validate_spirv_artifact_modules(
        info.codegen_dialect, info.required_spirv_features,
        modules, stages);
    LUISA_ASSERT(
        spirv_validation,
        "Vulkan compute shader serialization rejected its SPIR-V artifact ({}): {}",
        shader_artifact_codec_error_name(spirv_validation.error),
        spirv_validation.diagnostics);

    auto printer_payload = serialize_printer_payload(info.printers);
    LUISA_ASSERT(
        valid_shader_table_sizes(
            info.properties.size(), info.arguments.size(),
            info.printers.size(), printer_payload.size()),
        "Vulkan compute shader printer payload exceeds the serialized bytecode limits.");
    auto header = ShaderSerHeader{
        .header_ver = kShaderSerVersion,
        .pipeline_ver = kXIRPipelineVersion,
        .md5 = info.shader_md5,
        .type_md5 = info.type_md5,
        .property_size = info.properties.size(),
        .spv_byte_size = info.spirv.size_bytes(),
        .block_size = {info.block_size[0], info.block_size[1], info.block_size[2]},
        .kernel_arg_count = static_cast<uint32_t>(info.arguments.size()),
        .printer_count = static_cast<uint32_t>(info.printers.size()),
        .printer_size_bytes = static_cast<uint32_t>(printer_payload.size()),
        .validation_count = info.validation_count,
        .required_subgroup_size = info.required_subgroup_size,
        .constant_ubo_size = info.constant_ubo_data.size_bytes(),
        .use_bindless_buffer = static_cast<uint8_t>(info.use_buffer_bindless),
        .use_bindless_tex2d = static_cast<uint8_t>(info.use_tex2d_bindless),
        .use_bindless_tex3d = static_cast<uint8_t>(info.use_tex3d_bindless),
        .codegen_dialect = static_cast<uint8_t>(info.codegen_dialect),
        .required_spirv_features = info.required_spirv_features};
    header.property_md5 = shader_property_md5(info.properties);
    header.argument_md5 = saved_argument_md5(info.arguments);
    header.spv_md5 = binary_md5(info.spirv);
    header.printer_md5 = binary_md5(luisa::span<const std::byte>{printer_payload});
    header.constant_ubo_md5 = binary_md5(info.constant_ubo_data);
    header.semantic_header_md5 = shader_semantic_header_md5(header);

    uint64_t property_bytes;
    uint64_t argument_bytes;
    LUISA_ASSERT(
        checked_binary_product(
            header.property_size, sizeof(hlsl::Property), property_bytes) &&
            checked_binary_product(
                header.kernel_arg_count, sizeof(SavedArgument), argument_bytes),
        "Vulkan compute shader table size overflow during serialization.");
    auto final_size = checked_serialized_size(
        sizeof(ShaderSerHeader),
        {property_bytes, argument_bytes, header.spv_byte_size,
         header.printer_size_bytes, header.constant_ubo_size});
    luisa::vector<std::byte> artifact(final_size);
    auto *dst = artifact.data();
    append_value(dst, header);
    append_bytes(dst, info.properties);
    append_bytes(dst, info.arguments);
    append_bytes(dst, info.spirv);
    append_bytes(dst, luisa::span<const std::byte>{printer_payload});
    append_bytes(dst, info.constant_ubo_data);
    LUISA_ASSERT(dst == artifact.data() + artifact.size(),
                 "Vulkan compute shader serializer size contract drifted.");
    return artifact;
}

luisa::vector<std::byte> encode_raster_shader_artifact(
    const RasterShaderArtifactEncodeInfo &info) {
    auto interface_plan = plan_shader_interface(
        {.properties = info.properties,
         .arguments = info.arguments,
         .stage_mask = DescriptorInterfaceStageMask::RASTER,
         .dialect = info.codegen_dialect,
         .printer_count = static_cast<uint32_t>(info.printers.size()),
         .validation_count = info.validation_count,
         .use_buffer_bindless = info.use_buffer_bindless,
         .use_tex2d_bindless = info.use_tex2d_bindless,
         .use_tex3d_bindless = info.use_tex3d_bindless});
    LUISA_ASSERT(
        interface_plan,
        "Vulkan raster shader serialization received an invalid runtime interface: {}.",
        shader_interface_error_name(interface_plan.error));
    LUISA_ASSERT(
        (info.required_spirv_features & ~spirv::target_feature::known_mask) == 0u,
        "Vulkan raster shader serialization received unknown SPIR-V target-feature bits 0x{:016x}.",
        info.required_spirv_features & ~spirv::target_feature::known_mask);
    LUISA_ASSERT(
        valid_shader_table_sizes(
            info.properties.size(), info.arguments.size(),
            info.printers.size(), 0u) &&
            valid_spirv_byte_size(info.vertex_spirv.size_bytes()) &&
            valid_spirv_byte_size(info.pixel_spirv.size_bytes()) &&
            info.validation_count <= max_shader_validation_count,
        "Vulkan raster shader exceeds the serialized bytecode limits.");
    std::array modules{info.vertex_spirv, info.pixel_spirv};
    constexpr std::array stages{
        ShaderArtifactSpirvStage::VERTEX,
        ShaderArtifactSpirvStage::FRAGMENT};
    auto spirv_validation = validate_spirv_artifact_modules(
        info.codegen_dialect, info.required_spirv_features,
        modules, stages);
    LUISA_ASSERT(
        spirv_validation,
        "Vulkan raster shader serialization rejected its SPIR-V artifact ({}): {}",
        shader_artifact_codec_error_name(spirv_validation.error),
        spirv_validation.diagnostics);

    auto printer_payload = serialize_printer_payload(info.printers);
    LUISA_ASSERT(
        valid_shader_table_sizes(
            info.properties.size(), info.arguments.size(),
            info.printers.size(), printer_payload.size()),
        "Vulkan raster shader printer payload exceeds the serialized bytecode limits.");
    auto header = RasterSerHeader{
        .header_ver = kShaderSerVersion,
        .pipeline_ver = kXIRPipelineVersion,
        .md5 = info.shader_md5,
        .type_md5 = info.type_md5,
        .property_size = info.properties.size(),
        .vert_spv_byte_size = info.vertex_spirv.size_bytes(),
        .pixel_spv_byte_size = info.pixel_spirv.size_bytes(),
        .kernel_arg_count = static_cast<uint32_t>(info.arguments.size()),
        .printer_count = static_cast<uint32_t>(info.printers.size()),
        .printer_size_bytes = static_cast<uint32_t>(printer_payload.size()),
        .validation_count = info.validation_count,
        .use_bindless_buffer = static_cast<uint8_t>(info.use_buffer_bindless),
        .use_bindless_tex2d = static_cast<uint8_t>(info.use_tex2d_bindless),
        .use_bindless_tex3d = static_cast<uint8_t>(info.use_tex3d_bindless),
        .codegen_dialect = static_cast<uint8_t>(info.codegen_dialect),
        .required_spirv_features = info.required_spirv_features};
    header.property_md5 = shader_property_md5(info.properties);
    header.argument_md5 = saved_argument_md5(info.arguments);
    header.vert_spv_md5 = binary_md5(info.vertex_spirv);
    header.pixel_spv_md5 = binary_md5(info.pixel_spirv);
    header.printer_md5 = binary_md5(luisa::span<const std::byte>{printer_payload});
    header.semantic_header_md5 = raster_semantic_header_md5(header);

    uint64_t property_bytes;
    uint64_t argument_bytes;
    LUISA_ASSERT(
        checked_binary_product(
            header.property_size, sizeof(hlsl::Property), property_bytes) &&
            checked_binary_product(
                header.kernel_arg_count, sizeof(SavedArgument), argument_bytes),
        "Vulkan raster shader table size overflow during serialization.");
    auto final_size = checked_serialized_size(
        sizeof(RasterSerHeader),
        {property_bytes, argument_bytes, header.vert_spv_byte_size,
         header.pixel_spv_byte_size, header.printer_size_bytes});
    luisa::vector<std::byte> artifact(final_size);
    auto *dst = artifact.data();
    append_value(dst, header);
    append_bytes(dst, info.properties);
    append_bytes(dst, info.arguments);
    append_bytes(dst, info.vertex_spirv);
    append_bytes(dst, info.pixel_spirv);
    append_bytes(dst, luisa::span<const std::byte>{printer_payload});
    LUISA_ASSERT(dst == artifact.data() + artifact.size(),
                 "Vulkan raster shader serializer size contract drifted.");
    return artifact;
}

ComputeShaderArtifactDecodeResult decode_compute_shader_artifact(
    luisa::BinaryStream &stream,
    luisa::optional<vstd::MD5> expected_shader_md5,
    luisa::optional<vstd::MD5> expected_type_md5,
    luisa::optional<ShaderCodegenDialect> expected_codegen_dialect) {
    auto result = ComputeShaderArtifactDecodeResult{};
    auto fail = [&](ShaderArtifactCodecError error) -> ComputeShaderArtifactDecodeResult {
        result.error = error;
        return std::move(result);
    };
    auto stream_length = stream.length();
    if (stream_length < sizeof(ShaderSerHeader)) {
        return fail(ShaderArtifactCodecError::TRUNCATED_HEADER);
    }
    auto &artifact = result.artifact;
    auto &header = artifact.header;
    stream.read({reinterpret_cast<std::byte *>(&header), sizeof(header)});
    if (header.header_ver != kShaderSerVersion ||
        !valid_shader_semantic_header(header) ||
        header.pipeline_ver != kXIRPipelineVersion ||
        !valid_binary_flag(header.use_bindless_buffer) ||
        !valid_binary_flag(header.use_bindless_tex2d) ||
        !valid_binary_flag(header.use_bindless_tex3d) ||
        !valid_shader_codegen_dialect(header.codegen_dialect) ||
        header.validation_count > max_shader_validation_count ||
        (header.required_spirv_features & ~spirv::target_feature::known_mask) != 0u ||
        header.block_size[0] == 0u || header.block_size[1] == 0u ||
        header.block_size[2] == 0u ||
        (header.required_subgroup_size != 0u &&
         (header.required_subgroup_size > std::numeric_limits<uint8_t>::max() ||
          (header.required_subgroup_size &
           (header.required_subgroup_size - 1u)) != 0u))) {
        return fail(ShaderArtifactCodecError::INVALID_HEADER);
    }
    if ((expected_shader_md5 && *expected_shader_md5 != header.md5) ||
        (expected_type_md5 && *expected_type_md5 != header.type_md5)) {
        return fail(ShaderArtifactCodecError::IDENTITY_MISMATCH);
    }
    if (expected_codegen_dialect &&
        static_cast<ShaderCodegenDialect>(header.codegen_dialect) !=
            *expected_codegen_dialect) {
        result.diagnostics = luisa::format(
            "Artifact codegen dialect {} does not match the required "
            "dialect {}.",
            static_cast<uint32_t>(header.codegen_dialect),
            static_cast<uint32_t>(*expected_codegen_dialect));
        return fail(ShaderArtifactCodecError::CODEGEN_DIALECT_MISMATCH);
    }
    uint64_t property_bytes;
    uint64_t argument_bytes;
    uint64_t expected_size;
    if (!valid_shader_table_sizes(
            header.property_size, header.kernel_arg_count,
            header.printer_count, header.printer_size_bytes) ||
        !valid_spirv_byte_size(header.spv_byte_size) ||
        !valid_shader_constant_payload_size(
            header.constant_ubo_size,
            std::numeric_limits<uint64_t>::max()) ||
        !checked_binary_product(
            header.property_size, sizeof(hlsl::Property), property_bytes) ||
        !checked_binary_product(
            header.kernel_arg_count, sizeof(SavedArgument), argument_bytes) ||
        !checked_binary_total(
            sizeof(header),
            {property_bytes, argument_bytes, header.spv_byte_size,
             header.printer_size_bytes, header.constant_ubo_size},
            expected_size) ||
        expected_size != stream_length) {
        return fail(ShaderArtifactCodecError::INVALID_SECTION_SIZES);
    }

    artifact.properties.resize(static_cast<size_t>(header.property_size));
    artifact.arguments.resize(static_cast<size_t>(header.kernel_arg_count));
    artifact.spirv.resize(
        static_cast<size_t>(header.spv_byte_size / sizeof(uint32_t)));
    luisa::vector<char> printer_data(
        static_cast<size_t>(header.printer_size_bytes));
    artifact.constant_ubo_data.resize(
        static_cast<size_t>(header.constant_ubo_size));
    if (!artifact.properties.empty()) {
        stream.read({reinterpret_cast<std::byte *>(artifact.properties.data()),
                     luisa::size_bytes(artifact.properties)});
    }
    if (!artifact.arguments.empty()) {
        stream.read({reinterpret_cast<std::byte *>(artifact.arguments.data()),
                     luisa::size_bytes(artifact.arguments)});
    }
    stream.read({reinterpret_cast<std::byte *>(artifact.spirv.data()),
                 luisa::size_bytes(artifact.spirv)});
    if (!printer_data.empty()) {
        stream.read({reinterpret_cast<std::byte *>(printer_data.data()),
                     printer_data.size()});
    }
    if (!artifact.constant_ubo_data.empty()) {
        stream.read(artifact.constant_ubo_data);
    }
    if (stream.pos() != stream_length ||
        shader_property_md5(artifact.properties) != header.property_md5 ||
        saved_argument_md5(artifact.arguments) != header.argument_md5 ||
        binary_md5(luisa::span<const uint32_t>{artifact.spirv}) != header.spv_md5 ||
        binary_md5(luisa::span<const char>{printer_data}) != header.printer_md5 ||
        binary_md5(luisa::span<const std::byte>{artifact.constant_ubo_data}) != header.constant_ubo_md5) {
        return fail(ShaderArtifactCodecError::SECTION_DIGEST_MISMATCH);
    }
    auto interface_plan = plan_shader_interface(
        {.properties = artifact.properties,
         .arguments = artifact.arguments,
         .stage_mask = DescriptorInterfaceStageMask::COMPUTE,
         .dialect = static_cast<ShaderCodegenDialect>(header.codegen_dialect),
         .printer_count = header.printer_count,
         .validation_count = header.validation_count,
         .use_buffer_bindless = header.use_bindless_buffer != 0u,
         .use_tex2d_bindless = header.use_bindless_tex2d != 0u,
         .use_tex3d_bindless = header.use_bindless_tex3d != 0u,
         .has_constant_ubo_payload = !artifact.constant_ubo_data.empty()});
    if (!interface_plan) {
        return fail(ShaderArtifactCodecError::INVALID_SHADER_INTERFACE);
    }
    if (!decode_printers(
            printer_data, header.printer_count, artifact.printers)) {
        return fail(ShaderArtifactCodecError::INVALID_PRINTER_PAYLOAD);
    }
    std::array modules{luisa::span<const uint32_t>{artifact.spirv}};
    constexpr std::array stages{ShaderArtifactSpirvStage::COMPUTE};
    auto spirv_validation = validate_spirv_artifact_modules(
        static_cast<ShaderCodegenDialect>(header.codegen_dialect),
        header.required_spirv_features, modules, stages);
    if (!spirv_validation) {
        result.failed_spirv_module_index =
            spirv_validation.failed_module_index;
        result.diagnostics = std::move(spirv_validation.diagnostics);
        return fail(spirv_validation.error);
    }
    result.has_spirv_warning = spirv_validation.has_warning;
    result.diagnostics = std::move(spirv_validation.diagnostics);
    return result;
}

RasterShaderArtifactDecodeResult decode_raster_shader_artifact(
    luisa::BinaryStream &stream,
    luisa::optional<vstd::MD5> expected_shader_md5,
    luisa::optional<vstd::MD5> expected_type_md5,
    luisa::optional<ShaderCodegenDialect> expected_codegen_dialect) {
    auto result = RasterShaderArtifactDecodeResult{};
    auto fail = [&](ShaderArtifactCodecError error) -> RasterShaderArtifactDecodeResult {
        result.error = error;
        return std::move(result);
    };
    auto stream_length = stream.length();
    if (stream_length < sizeof(RasterSerHeader)) {
        return fail(ShaderArtifactCodecError::TRUNCATED_HEADER);
    }
    auto &artifact = result.artifact;
    auto &header = artifact.header;
    stream.read({reinterpret_cast<std::byte *>(&header), sizeof(header)});
    if (header.header_ver != kShaderSerVersion ||
        !valid_raster_semantic_header(header) ||
        header.pipeline_ver != kXIRPipelineVersion ||
        !valid_binary_flag(header.use_bindless_buffer) ||
        !valid_binary_flag(header.use_bindless_tex2d) ||
        !valid_binary_flag(header.use_bindless_tex3d) ||
        !valid_shader_codegen_dialect(header.codegen_dialect) ||
        header.validation_count > max_shader_validation_count ||
        (header.required_spirv_features & ~spirv::target_feature::known_mask) != 0u) {
        return fail(ShaderArtifactCodecError::INVALID_HEADER);
    }
    if ((expected_shader_md5 && *expected_shader_md5 != header.md5) ||
        (expected_type_md5 && *expected_type_md5 != header.type_md5)) {
        return fail(ShaderArtifactCodecError::IDENTITY_MISMATCH);
    }
    if (expected_codegen_dialect &&
        static_cast<uint32_t>(*expected_codegen_dialect) !=
            header.codegen_dialect) {
        return fail(ShaderArtifactCodecError::CODEGEN_DIALECT_MISMATCH);
    }
    uint64_t property_bytes;
    uint64_t argument_bytes;
    uint64_t expected_size;
    if (!valid_shader_table_sizes(
            header.property_size, header.kernel_arg_count,
            header.printer_count, header.printer_size_bytes) ||
        !valid_spirv_byte_size(header.vert_spv_byte_size) ||
        !valid_spirv_byte_size(header.pixel_spv_byte_size) ||
        !checked_binary_product(
            header.property_size, sizeof(hlsl::Property), property_bytes) ||
        !checked_binary_product(
            header.kernel_arg_count, sizeof(SavedArgument), argument_bytes) ||
        !checked_binary_total(
            sizeof(header),
            {property_bytes, argument_bytes, header.vert_spv_byte_size,
             header.pixel_spv_byte_size, header.printer_size_bytes},
            expected_size) ||
        expected_size != stream_length) {
        return fail(ShaderArtifactCodecError::INVALID_SECTION_SIZES);
    }

    artifact.properties.resize(static_cast<size_t>(header.property_size));
    artifact.arguments.resize(static_cast<size_t>(header.kernel_arg_count));
    artifact.vertex_spirv.resize(
        static_cast<size_t>(header.vert_spv_byte_size / sizeof(uint32_t)));
    artifact.pixel_spirv.resize(
        static_cast<size_t>(header.pixel_spv_byte_size / sizeof(uint32_t)));
    luisa::vector<char> printer_data(
        static_cast<size_t>(header.printer_size_bytes));
    if (!artifact.properties.empty()) {
        stream.read({reinterpret_cast<std::byte *>(artifact.properties.data()),
                     luisa::size_bytes(artifact.properties)});
    }
    if (!artifact.arguments.empty()) {
        stream.read({reinterpret_cast<std::byte *>(artifact.arguments.data()),
                     luisa::size_bytes(artifact.arguments)});
    }
    stream.read({reinterpret_cast<std::byte *>(artifact.vertex_spirv.data()),
                 luisa::size_bytes(artifact.vertex_spirv)});
    stream.read({reinterpret_cast<std::byte *>(artifact.pixel_spirv.data()),
                 luisa::size_bytes(artifact.pixel_spirv)});
    if (!printer_data.empty()) {
        stream.read({reinterpret_cast<std::byte *>(printer_data.data()),
                     printer_data.size()});
    }
    if (stream.pos() != stream_length ||
        shader_property_md5(artifact.properties) != header.property_md5 ||
        saved_argument_md5(artifact.arguments) != header.argument_md5 ||
        binary_md5(luisa::span<const uint32_t>{artifact.vertex_spirv}) != header.vert_spv_md5 ||
        binary_md5(luisa::span<const uint32_t>{artifact.pixel_spirv}) != header.pixel_spv_md5 ||
        binary_md5(luisa::span<const char>{printer_data}) != header.printer_md5) {
        return fail(ShaderArtifactCodecError::SECTION_DIGEST_MISMATCH);
    }
    auto interface_plan = plan_shader_interface(
        {.properties = artifact.properties,
         .arguments = artifact.arguments,
         .stage_mask = DescriptorInterfaceStageMask::RASTER,
         .dialect = static_cast<ShaderCodegenDialect>(header.codegen_dialect),
         .printer_count = header.printer_count,
         .validation_count = header.validation_count,
         .use_buffer_bindless = header.use_bindless_buffer != 0u,
         .use_tex2d_bindless = header.use_bindless_tex2d != 0u,
         .use_tex3d_bindless = header.use_bindless_tex3d != 0u});
    if (!interface_plan) {
        return fail(ShaderArtifactCodecError::INVALID_SHADER_INTERFACE);
    }
    if (!decode_printers(
            printer_data, header.printer_count, artifact.printers)) {
        return fail(ShaderArtifactCodecError::INVALID_PRINTER_PAYLOAD);
    }
    std::array modules{
        luisa::span<const uint32_t>{artifact.vertex_spirv},
        luisa::span<const uint32_t>{artifact.pixel_spirv}};
    constexpr std::array stages{
        ShaderArtifactSpirvStage::VERTEX,
        ShaderArtifactSpirvStage::FRAGMENT};
    auto spirv_validation = validate_spirv_artifact_modules(
        static_cast<ShaderCodegenDialect>(header.codegen_dialect),
        header.required_spirv_features, modules, stages);
    if (!spirv_validation) {
        result.failed_spirv_module_index =
            spirv_validation.failed_module_index;
        result.diagnostics = std::move(spirv_validation.diagnostics);
        return fail(spirv_validation.error);
    }
    result.has_spirv_warning = spirv_validation.has_warning;
    result.diagnostics = std::move(spirv_validation.diagnostics);
    return result;
}

}// namespace lc::vk::detail
