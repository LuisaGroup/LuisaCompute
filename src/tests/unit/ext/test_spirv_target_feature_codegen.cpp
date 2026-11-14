// Test for exact Vulkan XIR-to-SPIR-V target-feature accounting.
// This test covers:
// - baseline and subgroup feature requirements
// - storage-image, ray-query, and sampler-selector requirements
// - uniform and divergent unbounded-descriptor indexing contracts

#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <spirv-tools/libspirv.hpp>

#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>

#include "spirv_codegen/entry.h"
#include "spirv_codegen/optimizer.h"
#include "spirv_codegen/texture_sampling.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace xir = luisa::compute::xir;

namespace {

struct CompiledSpirv {
    std::string text;
    std::vector<uint32_t> words;
    lc::spirv::SpirvTargetFeatureMask required_features{};
    std::vector<lc::hlsl::ShaderVariableType> property_types;
    vstd::vector<Usage> argument_usages;
    vstd::vector<lc::spirv::SpirvKernelArgumentRoleMask> argument_roles;
};

constexpr auto all_target_features =
    lc::spirv::SpirvTargetFeatures::from_enabled_mask(
        lc::spirv::target_feature::known_mask);

void set_environment_variable(const char *name,
                              const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

class ScopedEnvironmentVariable {
private:
    const char *_name;
    luisa::optional<std::string> _previous;

public:
    ScopedEnvironmentVariable(const char *name,
                              const char *value) noexcept
        : _name{name} {
        if (auto previous = std::getenv(name)) {
            _previous.emplace(previous);
        }
        set_environment_variable(name, value);
    }
    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            _name, _previous ? _previous->c_str() : nullptr);
    }
    ScopedEnvironmentVariable(
        const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(
        const ScopedEnvironmentVariable &) = delete;
};

template<typename Kernel>
[[nodiscard]] CompiledSpirv compile_spirv_fixture(
    Kernel &&kernel,
    lc::spirv::SpirvTargetFeatures target_features =
        all_target_features) {
    auto result = lc::spirv::SpirvCodegenEntry::compile_spirv(
        kernel.function()->function(),
        ShaderOption{.enable_cache = false}, target_features);
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    std::string text;
    expect(tools.Validate(result.spv_bin.data(), result.spv_bin.size()))
        << "target-feature SPIR-V fixture must validate";
    expect(tools.Disassemble(
        result.spv_bin.data(), result.spv_bin.size(), &text))
        << "failed to disassemble target-feature SPIR-V fixture";
    std::vector<lc::hlsl::ShaderVariableType> property_types;
    property_types.reserve(result.properties.size());
    for (auto property : result.properties) {
        property_types.emplace_back(property.type);
    }
    vstd::vector<Usage> argument_usages;
    argument_usages.reserve(result.argument_usages.size());
    for (auto &&[argument, usage] : result.argument_usages) {
        static_cast<void>(argument);
        argument_usages.emplace_back(usage);
    }
    return {
        .text = std::move(text),
        .words = std::move(result.spv_bin),
        .required_features = result.required_target_features,
        .property_types = std::move(property_types),
        .argument_usages = std::move(argument_usages),
        .argument_roles = std::move(result.argument_roles)};
}

[[nodiscard]] bool contains(luisa::string_view text,
                            luisa::string_view needle) noexcept {
    return text.find(needle) != luisa::string_view::npos;
}

[[nodiscard]] size_t count_property(
    const CompiledSpirv &compiled,
    lc::hlsl::ShaderVariableType type) noexcept {
    return static_cast<size_t>(std::ranges::count(
        compiled.property_types, type));
}

// Exact instruction-shape facts for the DOT2ADD lowering. The fused path must
// emit one OpFDot2MixAcc32VALVE (float result, half2 vectors, float accumulator);
// the unfused path must emit the DXC-equivalent sequence: one half2 OpFMul,
// one OpFConvert to float2, two lane extracts, and two ordered float OpFAdd.
struct Dot2AddSpirvFacts {
    size_t valve_count{0u};
    size_t valve_capability_count{0u};
    size_t valve_extension_count{0u};
    size_t half2_mul_count{0u};
    size_t half2_to_float2_convert_count{0u};
    size_t float_add_count{0u};
    size_t no_contraction_count{0u};
    bool valve_shape_valid{false};
    bool unfused_sequence_valid{false};
    bool parse_succeeded{true};
};

[[nodiscard]] Dot2AddSpirvFacts inspect_dot2add_path(
    luisa::span<const uint32_t> words) noexcept {
    Dot2AddSpirvFacts facts;
    if (words.size() < 5u || words[3u] == 0u) {
        facts.parse_succeeded = false;
        return facts;
    }
    const auto id_bound = words[3u];
    struct FloatType {
        uint32_t width{0u};
        bool valid{false};
    };
    luisa::vector<FloatType> float_types(id_bound);
    luisa::vector<uint32_t> vector_components(id_bound, 0u);
    luisa::vector<uint32_t> vector_element(id_bound, 0u);
    luisa::vector<uint32_t> value_types(id_bound, 0u);
    luisa::vector<spv::Op> value_opcodes(id_bound, spv::Op::OpNop);
    luisa::vector<uint32_t> convert_sources(id_bound, 0u);
    luisa::vector<uint32_t> add_lhs(id_bound, 0u);
    luisa::vector<uint32_t> add_rhs(id_bound, 0u);
    luisa::vector<uint32_t> extract_sources(id_bound, 0u);
    luisa::vector<uint32_t> extract_indices(id_bound, ~0u);
    luisa::vector<bool> no_contraction(id_bound, false);
    auto is_half2 = [&](uint32_t type) noexcept {
        return type < id_bound && vector_components[type] == 2u &&
               vector_element[type] < id_bound &&
               float_types[vector_element[type]].valid &&
               float_types[vector_element[type]].width == 16u;
    };
    auto is_float2 = [&](uint32_t type) noexcept {
        return type < id_bound && vector_components[type] == 2u &&
               vector_element[type] < id_bound &&
               float_types[vector_element[type]].valid &&
               float_types[vector_element[type]].width == 32u;
    };
    auto is_float = [&](uint32_t type) noexcept {
        return type < id_bound && float_types[type].valid &&
               float_types[type].width == 32u;
    };
    for (auto offset = size_t{5u}; offset < words.size();) {
        const auto word_count = words[offset] >> 16u;
        const auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (word_count == 0u || word_count > words.size() - offset) {
            facts.parse_succeeded = false;
            return facts;
        }
        auto result_in_range = [&](uint32_t id) noexcept {
            return id < id_bound;
        };
        switch (opcode) {
            case spv::Op::OpTypeFloat:
                if (word_count >= 3u && result_in_range(words[offset + 1u])) {
                    float_types[words[offset + 1u]] = {words[offset + 2u], true};
                }
                break;
            case spv::Op::OpTypeVector:
                if (word_count == 4u && result_in_range(words[offset + 1u])) {
                    vector_element[words[offset + 1u]] = words[offset + 2u];
                    vector_components[words[offset + 1u]] = words[offset + 3u];
                }
                break;
            case spv::Op::OpCapability:
                if (word_count == 2u &&
                    words[offset + 1u] == static_cast<uint32_t>(
                                              spv::Capability::DotProductFloat16AccFloat32VALVE)) {
                    facts.valve_capability_count++;
                }
                break;
            case spv::Op::OpExtension: {
                // The extension name starts at word 1 as a literal string.
                luisa::string_view name{
                    reinterpret_cast<const char *>(&words[offset + 1u]),
                    (word_count - 1u) * sizeof(uint32_t)};
                if (name.find("SPV_VALVE_mixed_float_dot_product") !=
                    luisa::string_view::npos) {
                    facts.valve_extension_count++;
                }
                break;
            }
            case spv::Op::OpDecorate:
                if (word_count == 3u &&
                    words[offset + 2u] == static_cast<uint32_t>(
                                              spv::Decoration::NoContraction) &&
                    result_in_range(words[offset + 1u])) {
                    no_contraction[words[offset + 1u]] = true;
                    facts.no_contraction_count++;
                }
                break;
            case spv::Op::OpFMul:
                if (word_count == 5u && result_in_range(words[offset + 2u])) {
                    const auto result = words[offset + 2u];
                    value_types[result] = words[offset + 1u];
                    value_opcodes[result] = opcode;
                    if (is_half2(words[offset + 1u])) {
                        facts.half2_mul_count++;
                    }
                }
                break;
            case spv::Op::OpFConvert:
                if (word_count == 4u && result_in_range(words[offset + 2u])) {
                    const auto result = words[offset + 2u];
                    value_types[result] = words[offset + 1u];
                    value_opcodes[result] = opcode;
                    convert_sources[result] = words[offset + 3u];
                    if (is_float2(words[offset + 1u]) &&
                        is_half2(value_types[words[offset + 3u]])) {
                        facts.half2_to_float2_convert_count++;
                    }
                }
                break;
            case spv::Op::OpCompositeExtract:
                if (word_count >= 5u && result_in_range(words[offset + 2u])) {
                    const auto result = words[offset + 2u];
                    value_types[result] = words[offset + 1u];
                    value_opcodes[result] = opcode;
                    extract_sources[result] = words[offset + 3u];
                    extract_indices[result] = words[offset + 4u];
                }
                break;
            case spv::Op::OpFAdd:
                if (word_count == 5u && result_in_range(words[offset + 2u])) {
                    const auto result = words[offset + 2u];
                    value_types[result] = words[offset + 1u];
                    value_opcodes[result] = opcode;
                    add_lhs[result] = words[offset + 3u];
                    add_rhs[result] = words[offset + 4u];
                    if (is_float(words[offset + 1u])) {
                        facts.float_add_count++;
                    }
                }
                break;
            case spv::Op::OpFDot2MixAcc32VALVE:
                if (word_count == 6u && result_in_range(words[offset + 2u])) {
                    facts.valve_count++;
                    const auto result_type = words[offset + 1u];
                    const auto a = words[offset + 3u];
                    const auto b = words[offset + 4u];
                    const auto acc = words[offset + 5u];
                    facts.valve_shape_valid =
                        is_float(result_type) &&
                        is_half2(value_types[a]) &&
                        is_half2(value_types[b]) &&
                        is_float(value_types[acc]);
                }
                break;
            case spv::Op::OpLoad:
            case spv::Op::OpConstant:
            case spv::Op::OpConstantComposite:
            case spv::Op::OpCompositeConstruct:
            case spv::Op::OpFunctionParameter:
            case spv::Op::OpUConvert:
            case spv::Op::OpSConvert:
            case spv::Op::OpBitcast:
                if (word_count >= 3u && result_in_range(words[offset + 2u])) {
                    value_types[words[offset + 2u]] = words[offset + 1u];
                    value_opcodes[words[offset + 2u]] = opcode;
                }
                break;
            default:
                break;
        }
        offset += word_count;
    }
    // Unfused sequence: half2 mul -> float2 convert -> lane extracts ->
    // ordered float adds (x + y, then + acc), with NoContraction on the
    // multiply and both adds when fast math is off.
    for (auto convert = uint32_t{1u}; convert < id_bound; ++convert) {
        const auto source = convert_sources[convert];
        if (value_opcodes[convert] != spv::Op::OpFConvert ||
            !is_float2(value_types[convert]) ||
            source == 0u || source >= id_bound ||
            value_opcodes[source] != spv::Op::OpFMul ||
            !is_half2(value_types[source])) {
            continue;
        }
        // Find the two lane extracts of this convert.
        auto lane_x = uint32_t{0u};
        auto lane_y = uint32_t{0u};
        for (auto candidate = uint32_t{1u}; candidate < id_bound; ++candidate) {
            if (value_opcodes[candidate] != spv::Op::OpCompositeExtract ||
                extract_sources[candidate] != convert) {
                continue;
            }
            if (extract_indices[candidate] == 0u) { lane_x = candidate; }
            if (extract_indices[candidate] == 1u) { lane_y = candidate; }
        }
        if (lane_x == 0u || lane_y == 0u) { continue; }
        // Find the ordered adds: (x + y) then (+ acc).
        for (auto first = uint32_t{1u}; first < id_bound; ++first) {
            if (value_opcodes[first] != spv::Op::OpFAdd ||
                !is_float(value_types[first])) {
                continue;
            }
            auto matches_xy =
                (add_lhs[first] == lane_x && add_rhs[first] == lane_y) ||
                (add_lhs[first] == lane_y && add_rhs[first] == lane_x);
            if (!matches_xy) { continue; }
            for (auto second = uint32_t{1u}; second < id_bound; ++second) {
                if (value_opcodes[second] != spv::Op::OpFAdd ||
                    !is_float(value_types[second])) {
                    continue;
                }
                auto acc_operand = uint32_t{0u};
                if (add_lhs[second] == first) {
                    acc_operand = add_rhs[second];
                } else if (add_rhs[second] == first) {
                    acc_operand = add_lhs[second];
                }
                if (acc_operand == 0u || acc_operand >= id_bound ||
                    !is_float(value_types[acc_operand]) ||
                    value_opcodes[acc_operand] == spv::Op::OpFAdd ||
                    value_opcodes[acc_operand] == spv::Op::OpCompositeExtract) {
                    continue;
                }
                facts.unfused_sequence_valid =
                    no_contraction[source] &&
                    no_contraction[first] &&
                    no_contraction[second];
            }
        }
    }
    return facts;
}

[[nodiscard]] std::unordered_set<int16_t> signed_16bit_constants(
    const std::vector<uint32_t> &words) {
    std::unordered_set<int16_t> result;
    if (words.size() < 5u) { return result; }
    std::vector<bool> signed_16bit_types(words[3u], false);
    for (auto offset = size_t{5u}; offset < words.size();) {
        const auto size = words[offset] >> 16u;
        const auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (size == 0u || size > words.size() - offset) { return {}; }
        if (op == spv::Op::OpTypeInt && size == 4u &&
            words[offset + 1u] < signed_16bit_types.size()) {
            signed_16bit_types[words[offset + 1u]] =
                words[offset + 2u] == 16u && words[offset + 3u] == 1u;
        } else if (op == spv::Op::OpConstant && size == 4u &&
                   words[offset + 1u] < signed_16bit_types.size() &&
                   signed_16bit_types[words[offset + 1u]]) {
            result.emplace(luisa::bit_cast<int16_t>(
                static_cast<uint16_t>(words[offset + 3u])));
        }
        offset += size;
    }
    return result;
}

struct SamplerIntegerType {
    uint32_t width{0u};
    bool is_signed{false};
    bool valid{false};
};

struct SamplerClampFacts {
    size_t uint32_upper_clamp_count{0u};
    size_t unexpected_upper_clamp_count{0u};
    size_t safe_sampler_index_count{0u};
    size_t canonical_sampler_heap_count{0u};
    size_t sampler_heap_access_count{0u};
    size_t bounded_sampler_heap_access_count{0u};
    size_t sampler_heap_load_count{0u};
    size_t bounded_sampler_heap_load_count{0u};
    size_t sampled_image_from_heap_count{0u};
    size_t bounded_sampled_image_count{0u};
    size_t nonuniform_sampler_index_count{0u};
    size_t nonuniform_sampler_pointer_count{0u};
    size_t nonuniform_sampler_load_count{0u};
    size_t nonuniform_sampled_image_count{0u};
    size_t bindless_image_heap_count{0u};
    size_t bindless_image_heap_access_count{0u};
    size_t bindless_image_heap_load_count{0u};
    size_t bindless_image_fetch_count{0u};
    size_t nonuniform_bindless_image_index_count{0u};
    size_t nonuniform_bindless_image_pointer_count{0u};
    size_t nonuniform_bindless_image_load_count{0u};
    size_t nonuniform_bindless_image_fetch_count{0u};
    size_t nonuniform_integer_constant_count{0u};
    bool parse_succeeded{true};
};

[[nodiscard]] SamplerClampFacts inspect_configured_sampler_path(
    luisa::span<const uint32_t> words) noexcept {
    struct ParsedInstruction {
        spv::Op opcode{spv::Op::OpNop};
        size_t offset{0u};
        uint16_t word_count{0u};
    };
    auto facts = SamplerClampFacts{};
    if (words.size() < 5u) {
        facts.parse_succeeded = false;
        return facts;
    }
    auto id_bound = words[3u];
    if (id_bound == 0u) {
        facts.parse_succeeded = false;
        return facts;
    }
    luisa::vector<SamplerIntegerType> integer_types(id_bound);
    luisa::vector<uint32_t> value_types(id_bound, 0u);
    luisa::vector<size_t> defining_offsets(id_bound, ~size_t{0u});
    luisa::vector<luisa::optional<uint64_t>> integer_constants(id_bound);
    luisa::vector<bool> sampler_types(id_bound, false);
    luisa::vector<bool> sampled_image_types(id_bound, false);
    luisa::vector<uint32_t> array_element_types(id_bound, 0u);
    luisa::vector<uint32_t> array_length_ids(id_bound, 0u);
    luisa::vector<uint32_t> runtime_array_element_types(id_bound, 0u);
    luisa::vector<spv::StorageClass> pointer_storage_classes(
        id_bound, spv::StorageClass::Max);
    luisa::vector<uint32_t> pointer_pointee_types(id_bound, 0u);
    luisa::vector<spv::StorageClass> variable_storage_classes(
        id_bound, spv::StorageClass::Max);
    luisa::vector<luisa::optional<uint32_t>> descriptor_sets(id_bound);
    luisa::vector<luisa::optional<uint32_t>> descriptor_bindings(id_bound);
    luisa::vector<bool> nonuniform_ids(id_bound, false);
    luisa::vector<ParsedInstruction> instructions;
    auto record_typed_result =
        [&](size_t offset, spv::Op opcode,
            uint16_t word_count) noexcept {
            if (word_count < 3u) {
                facts.parse_succeeded = false;
                return;
            }
            auto type_id = words[offset + 1u];
            auto result_id = words[offset + 2u];
            if (type_id >= id_bound || result_id >= id_bound) {
                facts.parse_succeeded = false;
                return;
            }
            value_types[result_id] = type_id;
            defining_offsets[result_id] = offset;
        };
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto first_word = words[offset];
        auto word_count = static_cast<uint16_t>(first_word >> 16u);
        auto opcode = static_cast<spv::Op>(first_word & 0xffffu);
        if (word_count == 0u || offset + word_count > words.size()) {
            facts.parse_succeeded = false;
            return facts;
        }
        instructions.emplace_back(ParsedInstruction{
            .opcode = opcode,
            .offset = offset,
            .word_count = word_count});
        switch (opcode) {
            case spv::Op::OpTypeInt: {
                if (word_count != 4u) {
                    facts.parse_succeeded = false;
                    return facts;
                }
                auto result_id = words[offset + 1u];
                if (result_id >= id_bound) {
                    facts.parse_succeeded = false;
                    return facts;
                }
                integer_types[result_id] = {
                    .width = words[offset + 2u],
                    .is_signed = words[offset + 3u] != 0u,
                    .valid = true};
                break;
            }
            case spv::Op::OpTypeSampler: {
                if (word_count != 2u || words[offset + 1u] >= id_bound) {
                    facts.parse_succeeded = false;
                    return facts;
                }
                sampler_types[words[offset + 1u]] = true;
                break;
            }
            case spv::Op::OpTypeImage: {
                if (word_count < 9u || words[offset + 1u] >= id_bound) {
                    facts.parse_succeeded = false;
                    return facts;
                }
                // Operand 6 is the SPIR-V Sampled qualifier: 1 denotes an
                // image consumed by sampling/fetch instructions, while 2 is
                // a storage image. Bindless texture heaps contain the former.
                sampled_image_types[words[offset + 1u]] =
                    words[offset + 7u] == 1u;
                break;
            }
            case spv::Op::OpTypeArray: {
                if (word_count != 4u || words[offset + 1u] >= id_bound ||
                    words[offset + 2u] >= id_bound ||
                    words[offset + 3u] >= id_bound) {
                    facts.parse_succeeded = false;
                    return facts;
                }
                array_element_types[words[offset + 1u]] =
                    words[offset + 2u];
                array_length_ids[words[offset + 1u]] =
                    words[offset + 3u];
                break;
            }
            case spv::Op::OpTypeRuntimeArray: {
                if (word_count != 3u || words[offset + 1u] >= id_bound ||
                    words[offset + 2u] >= id_bound) {
                    facts.parse_succeeded = false;
                    return facts;
                }
                runtime_array_element_types[words[offset + 1u]] =
                    words[offset + 2u];
                break;
            }
            case spv::Op::OpTypePointer: {
                if (word_count != 4u || words[offset + 1u] >= id_bound ||
                    words[offset + 3u] >= id_bound) {
                    facts.parse_succeeded = false;
                    return facts;
                }
                pointer_storage_classes[words[offset + 1u]] =
                    static_cast<spv::StorageClass>(words[offset + 2u]);
                pointer_pointee_types[words[offset + 1u]] =
                    words[offset + 3u];
                break;
            }
            case spv::Op::OpDecorate: {
                if (word_count < 3u || words[offset + 1u] >= id_bound) {
                    facts.parse_succeeded = false;
                    return facts;
                }
                auto target = words[offset + 1u];
                auto decoration =
                    static_cast<spv::Decoration>(words[offset + 2u]);
                if (decoration == spv::Decoration::NonUniformEXT) {
                    if (word_count != 3u) {
                        facts.parse_succeeded = false;
                        return facts;
                    }
                    nonuniform_ids[target] = true;
                } else if (decoration == spv::Decoration::DescriptorSet ||
                           decoration == spv::Decoration::Binding) {
                    if (word_count != 4u) {
                        facts.parse_succeeded = false;
                        return facts;
                    }
                    auto &literal =
                        decoration == spv::Decoration::DescriptorSet ?
                            descriptor_sets[target] :
                            descriptor_bindings[target];
                    literal = words[offset + 3u];
                }
                break;
            }
            case spv::Op::OpConstant:
            case spv::Op::OpFunctionParameter:
            case spv::Op::OpLoad:
            case spv::Op::OpSelect:
            case spv::Op::OpSConvert:
            case spv::Op::OpUConvert:
            case spv::Op::OpBitcast:
            case spv::Op::OpBitwiseAnd:
            case spv::Op::OpIMul:
            case spv::Op::OpIAdd:
            case spv::Op::OpSGreaterThan:
            case spv::Op::OpUGreaterThan:
            case spv::Op::OpAccessChain:
            case spv::Op::OpInBoundsAccessChain:
            case spv::Op::OpSampledImage:
                record_typed_result(offset, opcode, word_count);
                break;
            case spv::Op::OpVariable:
                record_typed_result(offset, opcode, word_count);
                if (word_count < 4u || words[offset + 2u] >= id_bound) {
                    facts.parse_succeeded = false;
                    return facts;
                }
                variable_storage_classes[words[offset + 2u]] =
                    static_cast<spv::StorageClass>(words[offset + 3u]);
                break;
            default: break;
        }
        if (opcode == spv::Op::OpConstant && word_count >= 4u) {
            auto type_id = words[offset + 1u];
            auto result_id = words[offset + 2u];
            if (type_id < id_bound && result_id < id_bound &&
                integer_types[type_id].valid) {
                auto value = static_cast<uint64_t>(words[offset + 3u]);
                if (integer_types[type_id].width == 64u &&
                    word_count == 5u) {
                    value |= static_cast<uint64_t>(
                                 words[offset + 4u])
                             << 32u;
                }
                integer_constants[result_id] = value;
            }
        }
        offset += word_count;
    }
    if (!facts.parse_succeeded) { return facts; }

    for (auto id = uint32_t{1u}; id < id_bound; ++id) {
        if (nonuniform_ids[id] && integer_constants[id].has_value()) {
            facts.nonuniform_integer_constant_count++;
        }
    }

    auto instruction_of =
        [&](uint32_t result_id) noexcept
        -> const ParsedInstruction * {
        if (result_id >= id_bound ||
            defining_offsets[result_id] == ~size_t{0u}) {
            return nullptr;
        }
        for (auto &&instruction : instructions) {
            if (instruction.offset ==
                defining_offsets[result_id]) {
                return &instruction;
            }
        }
        return nullptr;
    };
    auto type_of = [&](uint32_t value_id) noexcept {
        if (value_id >= id_bound ||
            value_types[value_id] >= id_bound) {
            return SamplerIntegerType{};
        }
        return integer_types[value_types[value_id]];
    };
    auto constant_is = [&](uint32_t value_id,
                           uint64_t expected) noexcept {
        return value_id < id_bound &&
               integer_constants[value_id].has_value() &&
               *integer_constants[value_id] == expected;
    };

    std::unordered_set<uint32_t> bounded_selectors;
    for (auto &&instruction : instructions) {
        if (instruction.opcode != spv::Op::OpSelect ||
            instruction.word_count != 6u) {
            continue;
        }
        auto result_id = words[instruction.offset + 2u];
        auto condition_id = words[instruction.offset + 3u];
        auto maximum_id = words[instruction.offset + 4u];
        auto condition = instruction_of(condition_id);
        if (condition == nullptr || condition->word_count != 5u ||
            (condition->opcode != spv::Op::OpSGreaterThan &&
             condition->opcode != spv::Op::OpUGreaterThan) ||
            !constant_is(maximum_id,
                         lc::spirv::spirv_configured_sampler_selector_max)) {
            continue;
        }
        auto selector_type = type_of(result_id);
        if (!selector_type.valid) { continue; }
        auto compared_value = words[condition->offset + 3u];
        auto compared_maximum = words[condition->offset + 4u];
        if (compared_maximum != maximum_id ||
            words[instruction.offset + 5u] != compared_value ||
            type_of(maximum_id).width != selector_type.width ||
            type_of(maximum_id).is_signed != selector_type.is_signed ||
            type_of(compared_value).width != selector_type.width ||
            type_of(compared_value).is_signed !=
                selector_type.is_signed) {
            continue;
        }
        if (condition->opcode == spv::Op::OpUGreaterThan &&
            selector_type.width == 32u && !selector_type.is_signed) {
            facts.uint32_upper_clamp_count++;
            bounded_selectors.emplace(result_id);
        } else {
            facts.unexpected_upper_clamp_count++;
        }
    }

    // A constant selector in the public enum domain is bounded already. This
    // matters for mixed constant/dynamic sampler expressions, where only one
    // side needs a generated clamp.
    for (auto id = uint32_t{1u}; id < id_bound; ++id) {
        if (integer_constants[id] &&
            *integer_constants[id] <
                lc::spirv::spirv_configured_sampler_selector_count) {
            bounded_selectors.emplace(id);
        }
    }

    std::unordered_set<uint32_t> canonical_u32;
    for (auto selector : bounded_selectors) {
        auto type = type_of(selector);
        if (type.valid && type.width == 32u && !type.is_signed) {
            canonical_u32.emplace(selector);
        }
    }

    std::unordered_set<uint32_t> address_bases;
    for (auto &&instruction : instructions) {
        if (instruction.opcode != spv::Op::OpIMul ||
            instruction.word_count != 5u) {
            continue;
        }
        auto left = words[instruction.offset + 3u];
        auto right = words[instruction.offset + 4u];
        if ((canonical_u32.contains(left) &&
             constant_is(
                 right,
                 lc::spirv::spirv_configured_sampler_selector_count)) ||
            (canonical_u32.contains(right) &&
             constant_is(
                 left,
                 lc::spirv::spirv_configured_sampler_selector_count))) {
            address_bases.emplace(words[instruction.offset + 2u]);
        }
    }
    std::unordered_set<uint32_t> safe_sampler_indices;
    for (auto &&instruction : instructions) {
        if (instruction.opcode != spv::Op::OpIAdd ||
            instruction.word_count != 5u) {
            continue;
        }
        auto left = words[instruction.offset + 3u];
        auto right = words[instruction.offset + 4u];
        if ((address_bases.contains(left) &&
             canonical_u32.contains(right)) ||
            (address_bases.contains(right) &&
             canonical_u32.contains(left))) {
            safe_sampler_indices.emplace(
                words[instruction.offset + 2u]);
        }
    }
    facts.safe_sampler_index_count = safe_sampler_indices.size();

    // Identify the exact fixed sampler descriptor heap by its SPIR-V type and
    // ABI binding, rather than by debug names or instruction order.
    std::unordered_set<uint32_t> sampler_array_types;
    for (auto id = uint32_t{1u}; id < id_bound; ++id) {
        if (array_element_types[id] < id_bound &&
            sampler_types[array_element_types[id]] &&
            constant_is(
                array_length_ids[id],
                lc::spirv::spirv_configured_sampler_heap_size)) {
            sampler_array_types.emplace(id);
        }
    }
    std::unordered_set<uint32_t> sampler_heap_pointer_types;
    for (auto id = uint32_t{1u}; id < id_bound; ++id) {
        if (pointer_storage_classes[id] ==
                spv::StorageClass::UniformConstant &&
            sampler_array_types.contains(pointer_pointee_types[id])) {
            sampler_heap_pointer_types.emplace(id);
        }
    }
    std::unordered_set<uint32_t> sampler_heaps;
    for (auto id = uint32_t{1u}; id < id_bound; ++id) {
        if (variable_storage_classes[id] ==
                spv::StorageClass::UniformConstant &&
            sampler_heap_pointer_types.contains(value_types[id]) &&
            descriptor_sets[id] == 1u && descriptor_bindings[id] == 0u) {
            sampler_heaps.emplace(id);
        }
    }
    facts.canonical_sampler_heap_count = sampler_heaps.size();

    std::unordered_set<uint32_t> sampler_pointers;
    std::unordered_set<uint32_t> bounded_sampler_pointers;
    for (auto &&instruction : instructions) {
        if ((instruction.opcode != spv::Op::OpAccessChain &&
             instruction.opcode != spv::Op::OpInBoundsAccessChain) ||
            instruction.word_count < 5u ||
            !sampler_heaps.contains(words[instruction.offset + 3u])) {
            continue;
        }
        facts.sampler_heap_access_count++;
        auto result_id = words[instruction.offset + 2u];
        sampler_pointers.emplace(result_id);
        if (instruction.word_count == 5u &&
            safe_sampler_indices.contains(
                words[instruction.offset + 4u])) {
            facts.bounded_sampler_heap_access_count++;
            bounded_sampler_pointers.emplace(result_id);
            if (nonuniform_ids[words[instruction.offset + 4u]]) {
                facts.nonuniform_sampler_index_count++;
            }
            if (nonuniform_ids[result_id]) {
                facts.nonuniform_sampler_pointer_count++;
            }
        }
    }

    std::unordered_set<uint32_t> sampler_loads;
    std::unordered_set<uint32_t> bounded_sampler_loads;
    for (auto &&instruction : instructions) {
        if (instruction.opcode != spv::Op::OpLoad ||
            instruction.word_count < 4u ||
            !sampler_pointers.contains(
                words[instruction.offset + 3u])) {
            continue;
        }
        facts.sampler_heap_load_count++;
        auto result_id = words[instruction.offset + 2u];
        sampler_loads.emplace(result_id);
        if (bounded_sampler_pointers.contains(
                words[instruction.offset + 3u])) {
            facts.bounded_sampler_heap_load_count++;
            bounded_sampler_loads.emplace(result_id);
            if (nonuniform_ids[result_id]) {
                facts.nonuniform_sampler_load_count++;
            }
        }
    }

    for (auto &&instruction : instructions) {
        if (instruction.opcode != spv::Op::OpSampledImage ||
            instruction.word_count != 5u ||
            !sampler_loads.contains(words[instruction.offset + 4u])) {
            continue;
        }
        facts.sampled_image_from_heap_count++;
        if (bounded_sampler_loads.contains(
                words[instruction.offset + 4u])) {
            facts.bounded_sampled_image_count++;
            if (nonuniform_ids[words[instruction.offset + 2u]]) {
                facts.nonuniform_sampled_image_count++;
            }
        }
    }

    // Follow bindless sampled-image descriptors by type, not by generated
    // names or declaration order: runtime-array<Image Sampled=1> in
    // UniformConstant -> indexed pointer -> image load -> OpImageFetch.
    std::unordered_set<uint32_t> bindless_image_array_types;
    for (auto id = uint32_t{1u}; id < id_bound; ++id) {
        auto element = runtime_array_element_types[id];
        if (element < id_bound && sampled_image_types[element]) {
            bindless_image_array_types.emplace(id);
        }
    }
    std::unordered_set<uint32_t> bindless_image_heap_pointer_types;
    for (auto id = uint32_t{1u}; id < id_bound; ++id) {
        if (pointer_storage_classes[id] ==
                spv::StorageClass::UniformConstant &&
            bindless_image_array_types.contains(
                pointer_pointee_types[id])) {
            bindless_image_heap_pointer_types.emplace(id);
        }
    }
    std::unordered_set<uint32_t> bindless_image_heaps;
    for (auto id = uint32_t{1u}; id < id_bound; ++id) {
        if (variable_storage_classes[id] ==
                spv::StorageClass::UniformConstant &&
            bindless_image_heap_pointer_types.contains(value_types[id])) {
            bindless_image_heaps.emplace(id);
        }
    }
    facts.bindless_image_heap_count = bindless_image_heaps.size();

    std::unordered_set<uint32_t> bindless_image_pointers;
    for (auto &&instruction : instructions) {
        if ((instruction.opcode != spv::Op::OpAccessChain &&
             instruction.opcode != spv::Op::OpInBoundsAccessChain) ||
            instruction.word_count != 5u ||
            !bindless_image_heaps.contains(
                words[instruction.offset + 3u])) {
            continue;
        }
        facts.bindless_image_heap_access_count++;
        auto result_id = words[instruction.offset + 2u];
        auto index_id = words[instruction.offset + 4u];
        bindless_image_pointers.emplace(result_id);
        if (nonuniform_ids[index_id]) {
            facts.nonuniform_bindless_image_index_count++;
        }
        if (nonuniform_ids[result_id]) {
            facts.nonuniform_bindless_image_pointer_count++;
        }
    }

    std::unordered_set<uint32_t> bindless_image_loads;
    for (auto &&instruction : instructions) {
        if (instruction.opcode != spv::Op::OpLoad ||
            instruction.word_count < 4u ||
            !bindless_image_pointers.contains(
                words[instruction.offset + 3u])) {
            continue;
        }
        facts.bindless_image_heap_load_count++;
        auto result_id = words[instruction.offset + 2u];
        bindless_image_loads.emplace(result_id);
        if (nonuniform_ids[result_id]) {
            facts.nonuniform_bindless_image_load_count++;
        }
    }

    for (auto &&instruction : instructions) {
        if (instruction.opcode != spv::Op::OpImageFetch ||
            instruction.word_count < 5u ||
            !bindless_image_loads.contains(
                words[instruction.offset + 3u])) {
            continue;
        }
        facts.bindless_image_fetch_count++;
        if (nonuniform_ids[words[instruction.offset + 3u]]) {
            facts.nonuniform_bindless_image_fetch_count++;
        }
    }
    return facts;
}

void expect_bounded_configured_sampler_path(
    const SamplerClampFacts &facts, size_t access_count,
    bool nonuniform) {
    expect(facts.parse_succeeded)
        << "failed to recognize the configured sampler dataflow";
    expect(eq(facts.canonical_sampler_heap_count, 1u));
    expect(eq(facts.sampler_heap_access_count, access_count));
    expect(eq(facts.bounded_sampler_heap_access_count,
              facts.sampler_heap_access_count));
    expect(eq(facts.sampler_heap_load_count, access_count));
    expect(eq(facts.bounded_sampler_heap_load_count,
              facts.sampler_heap_load_count));
    expect(eq(facts.sampled_image_from_heap_count, access_count));
    expect(eq(facts.bounded_sampled_image_count,
              facts.sampled_image_from_heap_count));
    auto expected_nonuniform = nonuniform ? access_count : 0u;
    expect(eq(facts.nonuniform_sampler_index_count,
              expected_nonuniform));
    expect(eq(facts.nonuniform_sampler_pointer_count,
              expected_nonuniform));
    expect(eq(facts.nonuniform_sampler_load_count,
              expected_nonuniform));
    expect(eq(facts.nonuniform_sampled_image_count,
              expected_nonuniform));
}

void expect_bindless_image_fetch_path(
    const SamplerClampFacts &facts, bool nonuniform) {
    expect(facts.parse_succeeded)
        << "failed to recognize bindless sampled-image dataflow";
    expect(eq(facts.bindless_image_heap_count, 1u));
    expect(eq(facts.bindless_image_heap_access_count, 1u));
    expect(eq(facts.bindless_image_heap_load_count, 1u));
    expect(eq(facts.bindless_image_fetch_count, 1u));
    auto expected_nonuniform = nonuniform ? 1u : 0u;
    expect(eq(facts.nonuniform_bindless_image_index_count,
              expected_nonuniform));
    expect(eq(facts.nonuniform_bindless_image_pointer_count,
              expected_nonuniform));
    expect(eq(facts.nonuniform_bindless_image_load_count,
              expected_nonuniform));
    expect(eq(facts.nonuniform_bindless_image_fetch_count,
              expected_nonuniform));
    expect(eq(facts.nonuniform_integer_constant_count, 0u))
        << "NonUniformEXT must not contaminate interned constant indices";
}

enum class SamplerSelectorSource : uint8_t {
    CONSTANT,
    UNIFORM_ARGUMENT,
    NON_UNIFORM_DISPATCH_ID,
};

[[nodiscard]] Kernel1D<void(Image<float>, Buffer<float4>, uint)>
make_direct_sampler_kernel(
    SamplerSelectorSource source, uint32_t constant_filter = 0u) {
    Kernel1D kernel = [=](ImageFloat image, BufferFloat4 output,
                          UInt uniform_filter) noexcept {
        auto builder = luisa::compute::detail::FunctionBuilder::current();
        auto literal = [&](auto value) noexcept {
            return builder->literal(Type::of<decltype(value)>(), value);
        };
        auto non_uniform_filter = dispatch_id().x & 3u;
        const Expression *filter = nullptr;
        switch (source) {
            case SamplerSelectorSource::CONSTANT:
                filter = literal(constant_filter);
                break;
            case SamplerSelectorSource::UNIFORM_ARGUMENT:
                filter = uniform_filter.expression();
                break;
            case SamplerSelectorSource::NON_UNIFORM_DISPATCH_ID:
                filter = non_uniform_filter.expression();
                break;
        }
        auto sample = builder->call(
            Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE,
            {image.expression(), literal(make_float2(0.5f)), filter,
             literal(0u)});
        output.write(0u, def<float4>(sample));
    };
    return kernel;
}

// Exact instruction-shape facts for the SPV_EXT_split_barrier lowering: one
// OpControlBarrierArriveEXT (6142) with Workgroup/Workgroup/WorkgroupMemory|
// Release (2, 2, 260) and one OpControlBarrierWaitEXT (6143) with
// Workgroup/Workgroup/WorkgroupMemory|Acquire (2, 2, 258), one SplitBarrierEXT
// (6141) capability, and one SPV_EXT_split_barrier extension declaration.
// The classic OpControlBarrier must keep its (2, 2, 264) operands.
struct SplitBarrierSpirvFacts {
    size_t arrive_count{0u};
    size_t wait_count{0u};
    size_t classic_barrier_count{0u};
    size_t capability_count{0u};
    size_t extension_count{0u};
    bool arrive_shape_valid{false};
    bool wait_shape_valid{false};
    bool classic_shape_valid{false};
    bool parse_succeeded{true};
};

[[nodiscard]] SplitBarrierSpirvFacts inspect_split_barrier_path(
    luisa::span<const uint32_t> words) noexcept {
    SplitBarrierSpirvFacts facts;
    if (words.size() < 5u || words[3u] == 0u) {
        facts.parse_succeeded = false;
        return facts;
    }
    const auto id_bound = words[3u];
    luisa::vector<uint32_t> uint_constants(id_bound, ~0u);
    luisa::vector<bool> uint32_types(id_bound, false);
    auto constant_value = [&](uint32_t id) noexcept {
        return id < id_bound ? uint_constants[id] : ~0u;
    };
    auto barrier_shape = [&](size_t offset, size_t word_count,
                             uint32_t semantics) noexcept {
        return word_count == 4u &&
               constant_value(words[offset + 1u]) == 2u &&
               constant_value(words[offset + 2u]) == 2u &&
               constant_value(words[offset + 3u]) == semantics;
    };
    for (auto offset = size_t{5u}; offset < words.size();) {
        const auto word_count = words[offset] >> 16u;
        const auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (word_count == 0u || word_count > words.size() - offset) {
            facts.parse_succeeded = false;
            return facts;
        }
        switch (opcode) {
            case spv::Op::OpTypeInt:
                if (word_count == 4u && words[offset + 1u] < id_bound &&
                    words[offset + 2u] == 32u && words[offset + 3u] == 0u) {
                    uint32_types[words[offset + 1u]] = true;
                }
                break;
            case spv::Op::OpConstant:
                if (word_count == 4u && words[offset + 1u] < id_bound &&
                    uint32_types[words[offset + 1u]] &&
                    words[offset + 2u] < id_bound) {
                    uint_constants[words[offset + 2u]] = words[offset + 3u];
                }
                break;
            case spv::Op::OpCapability:
                if (word_count == 2u &&
                    words[offset + 1u] == static_cast<uint32_t>(
                                              spv::Capability::SplitBarrierEXT)) {
                    facts.capability_count++;
                }
                break;
            case spv::Op::OpExtension: {
                // The extension name starts at word 1 as a literal string.
                luisa::string_view name{
                    reinterpret_cast<const char *>(&words[offset + 1u]),
                    (word_count - 1u) * sizeof(uint32_t)};
                if (name.find("SPV_EXT_split_barrier") !=
                    luisa::string_view::npos) {
                    facts.extension_count++;
                }
                break;
            }
            case spv::Op::OpControlBarrierArriveEXT:
                facts.arrive_count++;
                facts.arrive_shape_valid = barrier_shape(offset, word_count, 260u);
                break;
            case spv::Op::OpControlBarrierWaitEXT:
                facts.wait_count++;
                facts.wait_shape_valid = barrier_shape(offset, word_count, 258u);
                break;
            case spv::Op::OpControlBarrier:
                facts.classic_barrier_count++;
                facts.classic_shape_valid = barrier_shape(offset, word_count, 264u);
                break;
            default: break;
        }
        offset += word_count;
    }
    return facts;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_target_features_baseline_is_zero"_test = [] {
        Kernel1D kernel = [](BufferUInt output) noexcept {
            output.write(0u, 42u);
        };
        auto compiled = compile_spirv_fixture(kernel);
        expect(eq(compiled.required_features, 0u));
        expect(contains(compiled.text, "OpEntryPoint GLCompute"));
        expect(!contains(compiled.text, "OpCapability GroupNonUniform"));
        expect(!contains(compiled.text, "OpCapability RayQueryKHR"));
    };

    "spirv_target_features_bfloat16_lowers_math_through_float32"_test = [] {
        Kernel1D kernel = [](BufferBFloat16 input, BufferBFloat16 output) noexcept {
            BFloat16 a = input.read(0u);
            BFloat16 two = 2.0f;
            output.write(0u, a * two + a);
            output.write(1u, sin(a));
        };
        auto compiled = compile_spirv_fixture(kernel);
        expect(eq(compiled.required_features,
                  lc::spirv::target_feature::shader_bfloat16));
        expect(contains(compiled.text, "OpExtension \"SPV_KHR_bfloat16\""));
        expect(contains(compiled.text, "OpCapability BFloat16TypeKHR"));
        expect(contains(compiled.text, "OpTypeFloat 16 BFloat16KHR"));
        expect(contains(compiled.text, "OpFConvert"));
        expect(contains(compiled.text, "OpFMul"));
        expect(contains(compiled.text, "OpFAdd"));
        expect(contains(compiled.text, " Sin"));
        expect(!contains(compiled.text, "OpCapability Float16"));
        expect(!contains(compiled.text, "OpCapability StorageBuffer16BitAccess"));
    };

    "spirv_target_features_device_clock_is_exact"_test = [] {
        Kernel1D kernel = [](BufferULong output) noexcept {
            output.write(0u, device_clock());
        };
        auto compiled = compile_spirv_fixture(kernel);
        constexpr auto expected =
            lc::spirv::target_feature::shader_device_clock |
            lc::spirv::target_feature::shader_int64;
        expect(eq(compiled.required_features, expected));
        expect(contains(compiled.text,
                        "OpExtension \"SPV_KHR_shader_clock\""));
        expect(contains(compiled.text,
                        "OpCapability ShaderClockKHR"));
        expect(contains(compiled.text, "OpReadClockKHR"));
    };

    "spirv_float_classification_uses_core_boolean_instructions"_test = [] {
        Kernel1D kernel = [](BufferUInt output, Float value) noexcept {
            output.write(0u, ite(luisa::compute::isnan(value), 1u, 0u));
            output.write(1u, ite(luisa::compute::isinf(value), 1u, 0u));
        };
        auto compiled = compile_spirv_fixture(kernel);
        expect(eq(compiled.required_features, 0u));
        expect(contains(compiled.text, "OpIsNan"));
        expect(contains(compiled.text, "OpIsInf"));
        expect(!contains(compiled.text, "Log2"))
            << "isnan must produce a bool with OpIsNan, not a floating-point logarithm";
    };

    "spirv_wide_integer_boolean_cast_constants_are_well_formed"_test = [] {
        Kernel1D kernel = [](BufferULong input,
                             BufferULong output) noexcept {
            auto truth = cast<bool>(input.read(0u));
            output.write(0u, cast<luisa::ulong>(truth));
        };
        auto compiled = compile_spirv_fixture(kernel);
        expect(eq(compiled.required_features,
                  lc::spirv::target_feature::shader_int64));
        expect(contains(compiled.text, "OpINotEqual"));
        expect(contains(compiled.text, "OpSelect"));
    };

    "spirv_narrow_constant_ubo_optimization_respects_storage_features"_test = [] {
        constexpr std::array narrow_values{
            int16_t{-7}, int16_t{11}, int16_t{23}, int16_t{-31}};
        Kernel1D kernel = [&](BufferInt output) noexcept {
            luisa::compute::Constant<int16_t> values{narrow_values};
            auto index = dispatch_x() & 3u;
            output.write(0u, cast<int32_t>(values[index]));
        };
        constexpr auto enabled =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::shader_int16);
        auto compiled = compile_spirv_fixture(kernel, enabled);
        expect(eq(compiled.required_features,
                  lc::spirv::target_feature::shader_int16));
        // A dynamic read of a small constant array may become scalar selects.
        // Its signed literal payload and storage-feature contract matter, not
        // whether the optimizer retains an OpConstantComposite instruction.
        const auto constants = signed_16bit_constants(compiled.words);
        for (auto value : narrow_values) { expect(constants.contains(value)); }
        expect(!contains(
            compiled.text,
            "OpCapability UniformAndStorageBuffer16BitAccess"));
        expect(!contains(compiled.text, "_ConstantUBO"));

        constexpr auto ubo_enabled =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::uniform_storage_buffer_16bit_access);
        auto lowered = compile_spirv_fixture(kernel, ubo_enabled);
        constexpr auto lowered_requirements =
            lc::spirv::target_feature::uniform_storage_buffer_16bit_access;
        expect(eq(lowered.required_features, lowered_requirements));
        expect(contains(
            lowered.text,
            "OpCapability UniformAndStorageBuffer16BitAccess"));
        expect(!contains(lowered.text, "OpCapability Int16"));
        expect(contains(lowered.text, "_ConstantUBO"));
    };

    "spirv_narrow_storage_and_arithmetic_features_are_independent"_test = [] {
        constexpr auto storage_8 =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::storage_buffer_8bit_access);
        Kernel1D load_i8 = [](BufferVar<int8_t> input,
                              BufferInt output) noexcept {
            output.write(0u, cast<int32_t>(input.read(0u)));
        };
        auto stored_i8 = compile_spirv_fixture(load_i8, storage_8);
        expect(eq(stored_i8.required_features,
                  lc::spirv::target_feature::storage_buffer_8bit_access));
        expect(contains(stored_i8.text,
                        "OpCapability StorageBuffer8BitAccess"));
        expect(!contains(stored_i8.text, "OpCapability Int8"));
        expect(contains(stored_i8.text, "OpSConvert"));

        constexpr auto storage_16 =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::storage_buffer_16bit_access);
        Kernel1D load_i16 = [](BufferShort input,
                               BufferInt output) noexcept {
            output.write(0u, cast<int32_t>(input.read(0u)));
        };
        auto stored_i16 = compile_spirv_fixture(load_i16, storage_16);
        expect(eq(stored_i16.required_features,
                  lc::spirv::target_feature::storage_buffer_16bit_access));
        expect(contains(stored_i16.text,
                        "OpCapability StorageBuffer16BitAccess"));
        expect(!contains(stored_i16.text, "OpCapability Int16"));
        expect(contains(stored_i16.text, "OpSConvert"));

        Kernel1D load_f16 = [](BufferHalf input,
                               BufferFloat output) noexcept {
            output.write(0u, cast<float>(input.read(0u)));
        };
        auto stored_f16 = compile_spirv_fixture(load_f16, storage_16);
        expect(eq(stored_f16.required_features,
                  lc::spirv::target_feature::storage_buffer_16bit_access));
        expect(contains(stored_f16.text,
                        "OpCapability StorageBuffer16BitAccess"));
        expect(!contains(stored_f16.text, "OpCapability Float16"));
        expect(contains(stored_f16.text, "OpFConvert"));

        constexpr auto storage_and_constant =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::storage_buffer_16bit_access |
                lc::spirv::target_feature::shader_int16);
        Kernel1D store_i16_constant = [](BufferShort output) noexcept {
            output.write(0u, def<int16_t>(int16_t{7}));
        };
        auto stored_constant = compile_spirv_fixture(
            store_i16_constant, storage_and_constant);
        constexpr auto expected_storage_and_constant =
            lc::spirv::target_feature::storage_buffer_16bit_access |
            lc::spirv::target_feature::shader_int16;
        expect(eq(stored_constant.required_features,
                  expected_storage_and_constant));
        expect(contains(stored_constant.text,
                        "OpCapability StorageBuffer16BitAccess"));
        expect(contains(stored_constant.text, "OpCapability Int16"));

        constexpr auto arithmetic_features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::shader_int8 |
                lc::spirv::target_feature::shader_float16);
        Kernel1D arithmetic = [](BufferInt int_output,
                                 BufferFloat float_output,
                                 Int integer,
                                 Float floating) noexcept {
            auto narrow_integer = cast<int8_t>(integer);
            auto narrow_float = cast<half>(floating);
            int_output.write(0u, cast<int32_t>(narrow_integer));
            float_output.write(
                0u, cast<float>(narrow_float * narrow_float));
        };
        auto arithmetic_spirv =
            compile_spirv_fixture(arithmetic, arithmetic_features);
        constexpr auto expected_arithmetic_features =
            lc::spirv::target_feature::shader_int8 |
            lc::spirv::target_feature::shader_float16;
        expect(eq(arithmetic_spirv.required_features,
                  expected_arithmetic_features));
        expect(contains(arithmetic_spirv.text, "OpCapability Int8"));
        expect(contains(arithmetic_spirv.text, "OpCapability Float16"));
        expect(!contains(arithmetic_spirv.text,
                         "OpCapability StorageBuffer8BitAccess"));
        expect(!contains(arithmetic_spirv.text,
                         "OpCapability StorageBuffer16BitAccess"));

        constexpr auto storage_arithmetic_features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::storage_buffer_16bit_access |
                lc::spirv::target_feature::shader_float16);
        Kernel1D storage_arithmetic = [](
                                          BufferHalf float_input,
                                          BufferHalf float_output) noexcept {
            float_output.write(
                0u, float_input.read(0u) * float_input.read(1u));
        };
        auto storage_arithmetic_spirv = compile_spirv_fixture(
            storage_arithmetic, storage_arithmetic_features);
        constexpr auto expected_storage_arithmetic_features =
            lc::spirv::target_feature::storage_buffer_16bit_access |
            lc::spirv::target_feature::shader_float16;
        expect(eq(storage_arithmetic_spirv.required_features,
                  expected_storage_arithmetic_features));
        expect(contains(storage_arithmetic_spirv.text,
                        "OpCapability StorageBuffer16BitAccess"));
        expect(contains(storage_arithmetic_spirv.text,
                        "OpCapability Float16"));
        expect(contains(storage_arithmetic_spirv.text, "OpFMul"));
    };

    "spirv_copysign_reports_lowering_integer_features_exactly"_test = [] {
        constexpr auto half_features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::storage_buffer_16bit_access |
                lc::spirv::target_feature::shader_float16 |
                lc::spirv::target_feature::shader_int16);
        Kernel1D half_copysign = [](BufferHalf magnitude,
                                    BufferHalf sign,
                                    BufferHalf output) noexcept {
            output.write(0u, copysign(
                                 magnitude.read(0u), sign.read(0u)));
        };
        auto half_spirv = compile_spirv_fixture(
            half_copysign, half_features);
        constexpr auto expected_half_features =
            lc::spirv::target_feature::storage_buffer_16bit_access |
            lc::spirv::target_feature::shader_float16 |
            lc::spirv::target_feature::shader_int16;
        expect(eq(half_spirv.required_features,
                  expected_half_features));
        expect(contains(half_spirv.text,
                        "OpCapability StorageBuffer16BitAccess"));
        expect(contains(half_spirv.text, "OpCapability Float16"));
        expect(contains(half_spirv.text, "OpCapability Int16"));
        expect(contains(half_spirv.text, "OpBitwiseAnd"));
        expect(contains(half_spirv.text, "OpBitwiseOr"));

        constexpr auto double_features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::shader_float64 |
                lc::spirv::target_feature::shader_int64);
        Kernel1D double_copysign = [](
                                       BufferVar<double> magnitude,
                                       BufferVar<double> sign,
                                       BufferVar<double> output) noexcept {
            output.write(0u, copysign(
                                 magnitude.read(0u), sign.read(0u)));
        };
        auto double_spirv = compile_spirv_fixture(
            double_copysign, double_features);
        constexpr auto expected_double_features =
            lc::spirv::target_feature::shader_float64 |
            lc::spirv::target_feature::shader_int64;
        expect(eq(double_spirv.required_features,
                  expected_double_features));
        expect(contains(double_spirv.text, "OpCapability Float64"));
        expect(contains(double_spirv.text, "OpCapability Int64"));
        expect(contains(double_spirv.text, "OpBitwiseAnd"));
        expect(contains(double_spirv.text, "OpBitwiseOr"));
    };

    "spirv_target_features_subgroup_arithmetic_is_exact"_test = [] {
        Kernel1D kernel = [](BufferUInt output) noexcept {
            output.write(0u, warp_active_sum(dispatch_id().x));
        };
        auto compiled = compile_spirv_fixture(kernel);
        constexpr auto expected =
            lc::spirv::target_feature::subgroup_basic |
            lc::spirv::target_feature::subgroup_arithmetic;
        expect(eq(compiled.required_features, expected));
        expect(contains(
            compiled.text, "OpCapability GroupNonUniformArithmetic"));
        expect(contains(compiled.text, "OpGroupNonUniformIAdd"));
        expect(!contains(compiled.text,
                         "OpCapability GroupNonUniformVote"));
        expect(!contains(compiled.text,
                         "OpCapability GroupNonUniformBallot"));
        expect(!contains(compiled.text,
                         "OpCapability GroupNonUniformShuffle"));
    };

    "spirv_target_features_storage_image_write_is_exact"_test = [] {
        Kernel1D kernel = [](ImageFloat image) noexcept {
            image.write(make_uint2(0u), make_float4(1.0f));
        };
        auto compiled = compile_spirv_fixture(kernel);
        expect(eq(
            compiled.required_features,
            lc::spirv::target_feature::storage_image_write_without_format));
        expect(contains(
            compiled.text,
            "OpCapability StorageImageWriteWithoutFormat"));
        expect(contains(compiled.text, "OpImageWrite"));
        expect(!contains(
            compiled.text,
            "OpCapability StorageImageReadWithoutFormat"));

        // The public AST path has no storage-image-read fixture: read-only
        // image arguments use sampled-image descriptors, while read/write
        // kernel arguments receive distinct sampled and storage descriptors.
        // Keep the read bit covered by the pure mask tests until an operation
        // can semantically read through the writable descriptor.
    };

    "spirv_target_features_ray_query_is_exact"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto ray = make_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f));
            output.write(0u, ite(accel.intersect_any(ray, {}), 1u, 0u));
        };
        auto compiled = compile_spirv_fixture(kernel);
        expect(eq(compiled.required_features,
                          lc::spirv::target_feature::ray_query |
                          lc::spirv::target_feature::ray_tracing_opacity_micromap));
        // The KHR spelling of VK_KHR_opacity_micromap requires every ray-query
        // module to declare the OpacityMicromapIdKHR execution mode, so the
        // emitted module must carry both the capability and the mode (and the
        // extension that defines them).
        expect(contains(compiled.text,
                     "OpCapability RayTracingOpacityMicromapExecutionModeKHR"));
        expect(contains(compiled.text, "OpacityMicromapIdKHR"));
        expect(contains(compiled.text,
                     "OpExtension \"SPV_KHR_opacity_micromap\""));
        expect(contains(compiled.text, "OpRayQueryInitializeKHR"));
        expect(eq(count_property(
                      compiled, lc::hlsl::ShaderVariableType::SPIRVAccel),
                  1u));
        expect(eq(count_property(
                      compiled,
                      lc::hlsl::ShaderVariableType::SPIRVAccelInstance),
                  0u));
        expect(!compiled.argument_roles.empty());
        expect(eq(
            compiled.argument_roles.front(),
            lc::spirv::kernel_argument_role::accel_traversal));
        expect(!lc::spirv::spirv_target_feature_is_capability_owned(
            lc::spirv::target_feature::ray_query));
        expect(eq(
            lc::spirv::reconcile_spirv_target_features(
                compiled.words.data(), compiled.words.size(), 0u),
            lc::spirv::target_feature::ray_query))
            << "a present RayQueryKHR capability must recover an omitted "
               "runtime-owned artifact requirement";
    };

    "spirv_instance_only_accel_roles_do_not_require_ray_query"_test = [] {
        constexpr lc::spirv::SpirvTargetFeatures no_optional_features{};
        Kernel1D dead_traversal = [](AccelVar accel,
                                     BufferUInt output) noexcept {
            if_(false, [&] {
                auto ray = make_ray(
                    make_float3(0.0f, 0.0f, 1.0f),
                    make_float3(0.0f, 0.0f, -1.0f));
                output.write(
                    0u, ite(accel.intersect_any(ray, {}), 1u, 0u));
            });
            output.write(0u, 7u);
        };
        expect(dead_traversal.function()->function().requires_raytracing())
            << "the AST oracle must retain the dead traversal builtin";
        auto dead = compile_spirv_fixture(
            dead_traversal, no_optional_features);
        expect(eq(dead.required_features, 0u));
        expect(!contains(dead.text, "OpCapability RayQueryKHR"));
        expect(eq(count_property(
                      dead, lc::hlsl::ShaderVariableType::SPIRVAccel),
                  0u));
        expect(!dead.argument_roles.empty());
        expect(eq(
            dead.argument_roles.front(),
            lc::spirv::kernel_argument_role::none));
        expect(!dead.argument_usages.empty());
        expect(dead.argument_usages.front() == Usage::NONE)
            << "optimized-away accel traversal must not retain a synthetic "
               "read usage beside its exact zero-role mask";

        Kernel1D unused_accel = [](AccelVar, BufferUInt output) noexcept {
            output.write(0u, 7u);
        };
        auto unused = compile_spirv_fixture(
            unused_accel, no_optional_features);
        expect(eq(unused.required_features, 0u));
        expect(eq(count_property(
                      unused, lc::hlsl::ShaderVariableType::SPIRVAccel),
                  0u));
        expect(eq(count_property(
                      unused,
                      lc::hlsl::ShaderVariableType::SPIRVAccelInstance),
                  0u));
        expect(!unused.argument_roles.empty());
        expect(eq(
            unused.argument_roles.front(),
            lc::spirv::kernel_argument_role::none));
        expect(!unused.argument_usages.empty());
        expect(unused.argument_usages.front() == Usage::NONE)
            << "unused accel must preserve the optimized XIR usage exactly";

        Kernel1D read_instance = [](AccelVar accel,
                                    BufferUInt output) noexcept {
            output.write(0u, accel.instance_user_id(0u));
        };
        auto read = compile_spirv_fixture(
            read_instance, no_optional_features);
        expect(eq(read.required_features, 0u));
        expect(!contains(read.text, "OpCapability RayQueryKHR"));
        expect(!contains(read.text, "OpTypeAccelerationStructureKHR"));
        expect(eq(count_property(
                      read, lc::hlsl::ShaderVariableType::SPIRVAccel),
                  0u));
        expect(eq(count_property(
                      read,
                      lc::hlsl::ShaderVariableType::SPIRVAccelInstance),
                  1u));
        expect(!read.argument_roles.empty());
        expect(eq(
            read.argument_roles.front(),
            lc::spirv::kernel_argument_role::accel_instance));
        expect(!read.argument_usages.empty());
        expect(read.argument_usages.front() == Usage::READ);

        Kernel1D write_instance = [](AccelVar accel) noexcept {
            accel.set_instance_visibility(0u, 0xa5u);
        };
        auto write = compile_spirv_fixture(
            write_instance, no_optional_features);
        expect(eq(write.required_features, 0u));
        expect(!contains(write.text, "OpCapability RayQueryKHR"));
        expect(!contains(write.text, "OpTypeAccelerationStructureKHR"));
        expect(eq(count_property(
                      write, lc::hlsl::ShaderVariableType::SPIRVAccel),
                  0u));
        expect(eq(count_property(
                      write,
                      lc::hlsl::ShaderVariableType::SPIRVAccelInstanceRW),
                  1u));
        expect(!write.argument_roles.empty());
        expect(eq(
            write.argument_roles.front(),
            lc::spirv::kernel_argument_role::accel_instance));
        expect(!write.argument_usages.empty());
        expect(write.argument_usages.front() == Usage::WRITE);
    };

    "spirv_target_features_sampler_selectors_are_exact"_test = [] {
        ScopedEnvironmentVariable optimization_level{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto non_anisotropic = compile_spirv_fixture(
            make_direct_sampler_kernel(
                SamplerSelectorSource::CONSTANT, 2u));
        expect(eq(non_anisotropic.required_features, 0u));
        expect(contains(non_anisotropic.text, "OpSampledImage"));
        expect(contains(non_anisotropic.text,
                        "OpImageSampleExplicitLod"));
        expect(contains(non_anisotropic.text, "Lod"));

        auto anisotropic = compile_spirv_fixture(
            make_direct_sampler_kernel(
                SamplerSelectorSource::CONSTANT, 3u));
        expect(eq(anisotropic.required_features,
                  lc::spirv::target_feature::sampler_anisotropy));
        expect(!contains(
            anisotropic.text,
            "OpCapability SampledImageArrayDynamicIndexing"));

        auto uniform_dynamic = compile_spirv_fixture(
            make_direct_sampler_kernel(
                SamplerSelectorSource::UNIFORM_ARGUMENT));
        constexpr auto uniform_expected =
            lc::spirv::target_feature::sampler_anisotropy |
            lc::spirv::target_feature::sampled_image_array_dynamic_indexing;
        expect(eq(uniform_dynamic.required_features, uniform_expected));
        expect(contains(
            uniform_dynamic.text,
            "OpCapability SampledImageArrayDynamicIndexing"));
        expect(!contains(
            uniform_dynamic.text,
            "OpCapability SampledImageArrayNonUniformIndexing"));
        auto uniform_facts = inspect_configured_sampler_path(
            luisa::span<const uint32_t>{uniform_dynamic.words});
        expect(eq(uniform_facts.uint32_upper_clamp_count, 1u));
        expect(eq(uniform_facts.unexpected_upper_clamp_count, 0u));
        expect_bounded_configured_sampler_path(
            uniform_facts, 1u, false);

        auto non_uniform_dynamic = compile_spirv_fixture(
            make_direct_sampler_kernel(
                SamplerSelectorSource::NON_UNIFORM_DISPATCH_ID));
        constexpr auto non_uniform_expected =
            lc::spirv::target_feature::sampler_anisotropy |
            lc::spirv::target_feature::sampled_image_array_non_uniform_indexing;
        expect(eq(non_uniform_dynamic.required_features,
                  non_uniform_expected));
        expect(contains(
            non_uniform_dynamic.text,
            "OpCapability SampledImageArrayNonUniformIndexing"));
        expect(!contains(
            non_uniform_dynamic.text,
            "OpCapability SampledImageArrayDynamicIndexing"));
        auto nonuniform_facts = inspect_configured_sampler_path(
            luisa::span<const uint32_t>{non_uniform_dynamic.words});
        expect(eq(nonuniform_facts.uint32_upper_clamp_count, 1u));
        expect(eq(nonuniform_facts.unexpected_upper_clamp_count, 0u));
        expect_bounded_configured_sampler_path(
            nonuniform_facts, 1u, true);
    };

    "spirv_gradient_sampling_emits_min_lod_operand_and_feature"_test = [] {
        ScopedEnvironmentVariable optimization_level{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        Kernel1D<void(Image<float>, Buffer<float4>)> kernel = [](
                                                                  ImageFloat image,
                                                                  BufferFloat4 output) noexcept {
            auto builder =
                luisa::compute::detail::FunctionBuilder::current();
            auto literal = [&](auto value) noexcept {
                return builder->literal(
                    Type::of<decltype(value)>(), value);
            };
            auto sample = builder->call(
                Type::of<float4>(),
                CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL,
                {image.expression(), literal(make_float2(0.5f)),
                 literal(make_float2(0.25f, 0.0f)),
                 literal(make_float2(0.0f, 0.25f)),
                 literal(1.0f), literal(0u), literal(0u)});
            output.write(0u, def<float4>(sample));
        };
        constexpr auto enabled =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::shader_resource_min_lod);
        auto compiled = compile_spirv_fixture(kernel, enabled);
        expect(eq(compiled.required_features,
                  lc::spirv::target_feature::shader_resource_min_lod));
        expect(contains(compiled.text, "OpCapability MinLod"));
        expect(contains(compiled.text, "OpImageSampleExplicitLod"));
        expect(contains(compiled.text, "Grad|MinLod"));
    };

    "spirv_dynamic_uint32_sampler_selectors_are_bounded_before_heap_access"_test = [] {
        using SelectorKernel = Kernel1D<void(
            Image<float>, Buffer<float4>, uint32_t, uint32_t)>;
        SelectorKernel ast_kernel = [](ImageFloat, BufferFloat4, UInt,
                                       UInt) noexcept {};
        auto ast_function = ast_kernel.function()->function();

        xir::Module module;
        auto *kernel = module.create_kernel();
        kernel->set_block_size(ast_function.block_size());
        auto *texture = kernel->create_resource_argument(
            Type::texture(Type::of<float>(), 2u));
        auto *output = kernel->create_resource_argument(
            Type::buffer(Type::of<float4>()));
        auto *filter = kernel->create_value_argument(
            Type::of<uint32_t>());
        auto *address = kernel->create_value_argument(
            Type::of<uint32_t>());
        auto *uv = module.create_constant_zero(Type::of<float2>());
        xir::XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *sample = builder.call(
            Type::of<float4>(),
            xir::ResourceQueryOp::TEXTURE2D_SAMPLE,
            {texture, uv, filter, address});
        auto *index = module.create_constant_zero(
            Type::of<uint32_t>());
        builder.call(
            xir::ResourceWriteOp::BUFFER_WRITE,
            {output, index, sample});
        builder.return_void();

        auto generic = xir::xir_verify_module(&module);
        expect(generic.succeeded())
            << "uint32 sampler selector fixture must be valid generic XIR";
        if (!generic.succeeded()) { return; }

        ScopedEnvironmentVariable optimization_level{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto compiled =
            lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
                ast_function, &module,
                ShaderOption{.enable_cache = false},
                all_target_features);
        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        std::string diagnostics;
        tools.SetMessageConsumer(
            [&diagnostics](spv_message_level_t, const char *,
                           const spv_position_t &,
                           const char *message) {
                if (!diagnostics.empty()) {
                    diagnostics.push_back('\n');
                }
                diagnostics.append(message);
            });
        expect(tools.Validate(compiled.spv_bin.data(),
                              compiled.spv_bin.size()))
            << "bounded uint32 sampler selector SPIR-V failed Vulkan validation: "
            << diagnostics;
        constexpr auto expected_features =
            lc::spirv::target_feature::sampler_anisotropy |
            lc::spirv::target_feature::sampled_image_array_dynamic_indexing;
        expect(eq(compiled.required_target_features,
                  expected_features));

        auto facts = inspect_configured_sampler_path(
            luisa::span<const uint32_t>{
                compiled.spv_bin.data(),
                compiled.spv_bin.size()});
        expect(facts.parse_succeeded)
            << "failed to recognize the exact uint32 sampler clamp dataflow";
        expect(eq(facts.uint32_upper_clamp_count, 2u));
        expect(eq(facts.unexpected_upper_clamp_count, 0u));
        expect(eq(facts.safe_sampler_index_count, 1u));
        expect_bounded_configured_sampler_path(
            facts, 1u, false);
    };

    "spirv_float32_fma_uses_core_instruction_only_with_shader_fma"_test = [] {
        Kernel1D kernel = [](BufferFloat x, BufferFloat y, BufferFloat z,
                             BufferFloat out) noexcept {
            auto i = dispatch_x();
            out.write(i, fma(x.read(i), y.read(i), z.read(i)));
        };
        lc::spirv::SpirvTargetFeatures features{};
        features.shader_fma_float32 = true;
        auto fused = compile_spirv_fixture(kernel, features);
        expect(contains(fused.text, "OpFmaKHR"));
        expect(contains(fused.text, "OpCapability FMAKHR"));
        expect(contains(fused.text, "OpExtension \"SPV_KHR_fma\""));
        expect(eq(fused.required_features,
                  lc::spirv::target_feature::shader_fma_float32));

        auto extended = compile_spirv_fixture(
            kernel, lc::spirv::SpirvTargetFeatures{});
        expect(!contains(extended.text, "OpFmaKHR"));
        expect(!contains(extended.text, "SPV_KHR_fma"));
        expect(contains(extended.text, " Fma"));
        expect(eq(extended.required_features, 0u));
    };

    "spirv_dot2add_fused_valve_path_is_exact"_test = [] {
        // Isolate emission: the capability-owned VALVE requirement must be
        // recorded even though the trim-capabilities allowlist deliberately
        // excludes capability 6912.
        ScopedEnvironmentVariable optimization_level{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        Kernel1D ast_kernel = [](BufferHalf2, BufferHalf2, BufferFloat,
                                 BufferFloat) noexcept {};
        auto ast_function = ast_kernel.function()->function();
        xir::Module module;
        auto *kernel = module.create_kernel();
        kernel->set_block_size(ast_function.block_size());
        auto *a_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<half2>()));
        auto *b_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<half2>()));
        auto *acc_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<float>()));
        auto *out_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<float>()));
        auto *index = module.create_constant_zero(Type::of<uint32_t>());
        xir::XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *a = builder.call(Type::of<half2>(),
                               xir::ResourceReadOp::BUFFER_READ,
                               {a_buffer, index});
        auto *b = builder.call(Type::of<half2>(),
                               xir::ResourceReadOp::BUFFER_READ,
                               {b_buffer, index});
        auto *acc = builder.call(Type::of<float>(),
                                 xir::ResourceReadOp::BUFFER_READ,
                                 {acc_buffer, index});
        auto *result = builder.call(Type::of<float>(),
                                    xir::ArithmeticOp::DOT2ADD, {a, b, acc});
        builder.call(xir::ResourceWriteOp::BUFFER_WRITE,
                     {out_buffer, index, result});
        builder.return_void();
        expect(xir::xir_verify_module(&module).succeeded());

        auto compile = [&](bool valve_enabled) {
            lc::spirv::SpirvTargetFeatures features{};
            features.shader_float16 = true;
            features.storage_buffer_16bit_access = true;
            features.shader_mixed_float_dot_product_float16_acc_float32 =
                valve_enabled;
            return lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
                ast_function, &module,
                ShaderOption{.enable_cache = false, .enable_fast_math = false},
                features);
        };
        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        auto validate_and_disassemble = [&](const std::vector<uint32_t> &words,
                                            std::string &text) {
            std::string diagnostics;
            tools.SetMessageConsumer(
                [&diagnostics](spv_message_level_t, const char *,
                               const spv_position_t &, const char *message) {
                    if (!diagnostics.empty()) { diagnostics.push_back('\n'); }
                    diagnostics.append(message);
                });
            expect(tools.Validate(words.data(), words.size()))
                << "dot2add SPIR-V failed Vulkan 1.2 validation: " << diagnostics;
            expect(tools.Disassemble(words.data(), words.size(), &text));
        };

        // On-path: one fused OpFDot2MixAcc32VALVE, the VALVE capability and
        // extension, and the exact required feature bit. The half2 operands
        // additionally require the Float16 capability (shader_float16).
        auto fused = compile(true);
        std::string fused_text;
        validate_and_disassemble(fused.spv_bin, fused_text);
        constexpr auto fused_expected =
            lc::spirv::target_feature::shader_mixed_float_dot_product_float16_acc_float32 |
            lc::spirv::target_feature::shader_float16 |
            lc::spirv::target_feature::storage_buffer_16bit_access;
        expect(eq(fused.required_target_features, fused_expected));
        expect(contains(fused_text, "OpFDot2MixAcc32VALVE"));
        expect(contains(fused_text,
                        "OpExtension \"SPV_VALVE_mixed_float_dot_product\""));
        expect(contains(fused_text,
                        "OpCapability DotProductFloat16AccFloat32VALVE"));
        auto fused_facts = inspect_dot2add_path(
            luisa::span<const uint32_t>{fused.spv_bin.data(), fused.spv_bin.size()});
        expect(fused_facts.parse_succeeded);
        expect(eq(fused_facts.valve_count, 1u));
        expect(fused_facts.valve_shape_valid)
            << "OpFDot2MixAcc32VALVE must take half2 a/b, float acc, and return float";
        expect(eq(fused_facts.valve_capability_count, 1u));
        expect(eq(fused_facts.valve_extension_count, 1u));
        expect(eq(fused_facts.half2_mul_count, 0u));
        expect(eq(fused_facts.float_add_count, 0u));

        // Off-path: the DXC-equivalent unfused sequence, no VALVE opcode,
        // capability, extension, or required feature. NoContraction must keep
        // the multiply and both ordered adds from re-fusing.
        auto unfused = compile(false);
        std::string unfused_text;
        validate_and_disassemble(unfused.spv_bin, unfused_text);
        expect(eq(unfused.required_target_features,
                  lc::spirv::target_feature::shader_float16 |
                      lc::spirv::target_feature::storage_buffer_16bit_access));
        expect(!contains(unfused_text, "OpFDot2MixAcc32VALVE"));
        expect(!contains(unfused_text, "SPV_VALVE_mixed_float_dot_product"));
        expect(!contains(unfused_text, "DotProductFloat16AccFloat32VALVE"));
        auto unfused_facts = inspect_dot2add_path(
            luisa::span<const uint32_t>{unfused.spv_bin.data(), unfused.spv_bin.size()});
        expect(unfused_facts.parse_succeeded);
        expect(eq(unfused_facts.valve_count, 0u));
        expect(eq(unfused_facts.valve_capability_count, 0u));
        expect(eq(unfused_facts.valve_extension_count, 0u));
        expect(eq(unfused_facts.half2_mul_count, 1u));
        expect(eq(unfused_facts.half2_to_float2_convert_count, 1u));
        expect(eq(unfused_facts.float_add_count, 2u));
        expect(unfused_facts.unfused_sequence_valid)
            << "the unfused path must be half2 OpFMul -> OpFConvert float2 -> "
               "lane extracts -> ordered float adds, all NoContraction\n"
            << unfused_text;
        expect(eq(unfused_facts.no_contraction_count, 3u));
    };

    "spirv_dot2add_live_requirement_survives_optimization"_test = [] {
        // With the optimizer enabled, the capability-owned VALVE bit must be
        // reconciled from the final binary and retained for a live dot2add.
        ScopedEnvironmentVariable optimization_level{
            "LUISA_SPIRV_OPT_LEVEL", "1"};
        ScopedEnvironmentVariable pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        Kernel1D ast_kernel = [](BufferHalf2, BufferHalf2, BufferFloat,
                                 BufferFloat) noexcept {};
        auto ast_function = ast_kernel.function()->function();
        auto build_module = [&](xir::Module &module) {
            auto *kernel = module.create_kernel();
            kernel->set_block_size(ast_function.block_size());
            auto *a_buffer = kernel->create_resource_argument(
                Type::buffer(Type::of<half2>()));
            auto *b_buffer = kernel->create_resource_argument(
                Type::buffer(Type::of<half2>()));
            auto *acc_buffer = kernel->create_resource_argument(
                Type::buffer(Type::of<float>()));
            auto *out_buffer = kernel->create_resource_argument(
                Type::buffer(Type::of<float>()));
            auto *index = module.create_constant_zero(Type::of<uint32_t>());
            xir::XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            auto *a = builder.call(Type::of<half2>(),
                                   xir::ResourceReadOp::BUFFER_READ,
                                   {a_buffer, index});
            auto *b = builder.call(Type::of<half2>(),
                                   xir::ResourceReadOp::BUFFER_READ,
                                   {b_buffer, index});
            auto *acc = builder.call(Type::of<float>(),
                                     xir::ResourceReadOp::BUFFER_READ,
                                     {acc_buffer, index});
            auto *result = builder.call(Type::of<float>(),
                                        xir::ArithmeticOp::DOT2ADD, {a, b, acc});
            builder.call(xir::ResourceWriteOp::BUFFER_WRITE,
                         {out_buffer, index, result});
            builder.return_void();
        };
        lc::spirv::SpirvTargetFeatures features{};
        features.shader_float16 = true;
        features.storage_buffer_16bit_access = true;
        features.shader_mixed_float_dot_product_float16_acc_float32 = true;
        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        {
            xir::Module module;
            build_module(module);
            expect(xir::xir_verify_module(&module).succeeded());
            auto compiled = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
                ast_function, &module,
                ShaderOption{.enable_cache = false, .enable_fast_math = false},
                features);
            expect(tools.Validate(compiled.spv_bin.data(), compiled.spv_bin.size()));
            constexpr auto expected =
                lc::spirv::target_feature::shader_mixed_float_dot_product_float16_acc_float32 |
                lc::spirv::target_feature::shader_float16 |
                lc::spirv::target_feature::storage_buffer_16bit_access;
            expect(eq(compiled.required_target_features, expected));
            std::string text;
            expect(tools.Disassemble(compiled.spv_bin.data(),
                                     compiled.spv_bin.size(), &text));
            expect(contains(text, "OpFDot2MixAcc32VALVE"));
            expect(contains(text, "OpCapability DotProductFloat16AccFloat32VALVE"));
        }
    };

    "spirv_split_barrier_exact_opcodes_scopes_and_feature"_test = [] {
        // Isolate emission: the capability-owned SplitBarrierEXT requirement
        // must be recorded even though the trim-capabilities allowlist
        // deliberately excludes capability 6141.
        ScopedEnvironmentVariable optimization_level{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        // The classic sync_block() must keep its exact (2, 2, 264)
        // OpControlBarrier alongside the split arrive/wait pair.
        Kernel1D ast_kernel = [](BufferUInt) noexcept {
            sync_block();
            split_barrier_arrive();
            split_barrier_wait();
        };
        auto compiled = compile_spirv_fixture(ast_kernel);
        expect(eq(compiled.required_features,
                  lc::spirv::target_feature::shader_split_barrier));
        expect(contains(compiled.text, "OpControlBarrierArriveEXT"));
        expect(contains(compiled.text, "OpControlBarrierWaitEXT"));
        expect(contains(compiled.text, "OpCapability SplitBarrierEXT"));
        expect(contains(compiled.text,
                        "OpExtension \"SPV_EXT_split_barrier\""));
        auto facts = inspect_split_barrier_path(
            luisa::span<const uint32_t>{compiled.words});
        expect(facts.parse_succeeded);
        expect(eq(facts.arrive_count, 1u));
        expect(eq(facts.wait_count, 1u));
        expect(facts.arrive_shape_valid)
            << "arrive must be OpControlBarrierArriveEXT Workgroup(2), "
               "Workgroup(2), WorkgroupMemory|Release(260)\n"
            << compiled.text;
        expect(facts.wait_shape_valid)
            << "wait must be OpControlBarrierWaitEXT Workgroup(2), "
               "Workgroup(2), WorkgroupMemory|Acquire(258)\n"
            << compiled.text;
        expect(eq(facts.classic_barrier_count, 1u));
        expect(facts.classic_shape_valid)
            << "sync_block must keep OpControlBarrier Workgroup(2), "
               "Workgroup(2), WorkgroupMemory|AcquireRelease(264)\n"
            << compiled.text;
        expect(eq(facts.capability_count, 1u));
        expect(eq(facts.extension_count, 1u));
        expect(lc::spirv::spirv_target_feature_is_capability_owned(
            lc::spirv::target_feature::shader_split_barrier));
        // Reconciliation recovers the requirement from the capability alone.
        auto reconciled = lc::spirv::reconcile_spirv_target_features(
            compiled.words.data(), compiled.words.size(), 0u);
        expect((reconciled & lc::spirv::target_feature::shader_split_barrier) != 0u);
    };

    "spirv_split_barrier_in_reached_callable_is_exact"_test = [] {
        ScopedEnvironmentVariable optimization_level{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        // Callables may use the split barrier when reached from a compute
        // kernel; the callable-reachable call graph is the only accepted path.
        Kernel1D ast_kernel = [](BufferUInt) noexcept {};
        auto ast_function = ast_kernel.function()->function();
        xir::Module module;
        auto *callable = module.create_callable(nullptr);
        xir::XIRBuilder builder;
        builder.set_insertion_point(callable->create_body_block());
        builder.split_barrier_wait();
        builder.return_void();
        auto *kernel = module.create_kernel();
        kernel->set_block_size(ast_function.block_size());
        auto *out_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto *index = module.create_constant_zero(Type::of<uint32_t>());
        auto *value = module.create_constant_one(Type::of<uint32_t>());
        builder.set_insertion_point(kernel->create_body_block());
        builder.split_barrier_arrive();
        builder.call(nullptr, callable, {});
        builder.call(xir::ResourceWriteOp::BUFFER_WRITE,
                     {out_buffer, index, value});
        builder.return_void();
        expect(xir::xir_verify_module(&module).succeeded());
        lc::spirv::SpirvTargetFeatures features{};
        features.shader_split_barrier = true;
        auto compiled = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            ast_function, &module,
            ShaderOption{.enable_cache = false}, features);
        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        std::string diagnostics;
        tools.SetMessageConsumer(
            [&diagnostics](spv_message_level_t, const char *,
                           const spv_position_t &, const char *message) {
                if (!diagnostics.empty()) { diagnostics.push_back('\n'); }
                diagnostics.append(message);
            });
        expect(tools.Validate(compiled.spv_bin.data(), compiled.spv_bin.size()))
            << "split-barrier SPIR-V failed Vulkan 1.2 validation: " << diagnostics;
        expect(eq(compiled.required_target_features,
                  lc::spirv::target_feature::shader_split_barrier));
        auto facts = inspect_split_barrier_path(
            luisa::span<const uint32_t>{compiled.spv_bin});
        expect(facts.parse_succeeded);
        expect(eq(facts.arrive_count, 1u));
        expect(eq(facts.wait_count, 1u));
        expect(facts.arrive_shape_valid);
        expect(facts.wait_shape_valid);
        expect(eq(facts.classic_barrier_count, 0u));
        expect(eq(facts.capability_count, 1u));
        expect(eq(facts.extension_count, 1u));
    };

    "spirv_split_barrier_live_requirement_survives_optimization"_test = [] {
        // With the optimizer enabled, the capability-owned SplitBarrierEXT bit
        // must be reconciled from the final binary and retained for a live
        // arrive/wait pair. The vendored trim pass does not support capability
        // 6141, so the conservative allowlist keeps every declaration intact.
        ScopedEnvironmentVariable optimization_level{
            "LUISA_SPIRV_OPT_LEVEL", "1"};
        ScopedEnvironmentVariable pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        Kernel1D ast_kernel = [](BufferUInt) noexcept {
            sync_block();
            split_barrier_arrive();
            split_barrier_wait();
        };
        auto compiled = compile_spirv_fixture(ast_kernel);
        expect(eq(compiled.required_features,
                  lc::spirv::target_feature::shader_split_barrier));
        auto facts = inspect_split_barrier_path(
            luisa::span<const uint32_t>{compiled.words});
        expect(facts.parse_succeeded);
        expect(eq(facts.arrive_count, 1u));
        expect(eq(facts.wait_count, 1u));
        expect(facts.arrive_shape_valid);
        expect(facts.wait_shape_valid);
        expect(eq(facts.classic_barrier_count, 1u));
        expect(facts.classic_shape_valid);
        expect(eq(facts.capability_count, 1u));
        expect(eq(facts.extension_count, 1u));
        // The vendored trim-capabilities pass does not support capability
        // 6141: the conservative allowlist must refuse to register it for a
        // split-barrier module, so no declaration can be trimmed away.
        auto report = lc::spirv::optimize_spirv(
            compiled.words,
            lc::spirv::spirv_optimizer_options_from_environment());
        expect(report.attempted);
        expect(!report.capability_trim_registered)
            << "trim-capabilities must stay unregistered for SplitBarrierEXT";
    };

    "spirv_split_barrier_absent_without_the_ops"_test = [] {
        // A kernel that only uses the classic barrier must remain byte-exact
        // in its feature contract: no split-barrier capability, extension,
        // opcodes, or required feature even with the feature enabled.
        ScopedEnvironmentVariable optimization_level{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        Kernel1D ast_kernel = [](BufferUInt) noexcept {
            sync_block();
        };
        auto compiled = compile_spirv_fixture(ast_kernel);
        expect(eq(compiled.required_features, 0u));
        expect(!contains(compiled.text, "OpControlBarrierArriveEXT"));
        expect(!contains(compiled.text, "OpControlBarrierWaitEXT"));
        expect(!contains(compiled.text, "SplitBarrierEXT"));
        expect(!contains(compiled.text, "SPV_EXT_split_barrier"));
        auto facts = inspect_split_barrier_path(
            luisa::span<const uint32_t>{compiled.words});
        expect(facts.parse_succeeded);
        expect(eq(facts.classic_barrier_count, 1u));
        expect(facts.classic_shape_valid);
    };

    "spirv_target_features_uniform_bindless_texture_indexing_is_exact"_test = [] {
        constexpr auto expected =
            lc::spirv::target_feature::descriptor_indexing |
            lc::spirv::target_feature::runtime_descriptor_array |
            lc::spirv::target_feature::descriptor_binding_partially_bound |
            lc::spirv::target_feature::descriptor_binding_sampled_image_update_after_bind |
            lc::spirv::target_feature::sampled_image_array_dynamic_indexing;
        constexpr auto features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(expected);
        Kernel1D constant_index = [](BindlessVar bindless,
                                     BufferFloat4 output) noexcept {
            output.write(
                0u, bindless.tex2d(0u, false, true)
                        .read(make_uint2(0u)));
        };
        Kernel1D argument_index = [](BindlessVar bindless,
                                     BufferFloat4 output,
                                     UInt slot) noexcept {
            output.write(
                0u, bindless.tex2d(slot, false, true)
                        .read(make_uint2(0u)));
        };
        Kernel1D typed_argument_index = [](BindlessVar bindless,
                                           BufferFloat4 output,
                                           UInt slot) noexcept {
            output.write(
                0u, bindless.tex2d(slot, true, true)
                        .read(make_uint2(0u)));
        };
        for (auto compiled : {
                 compile_spirv_fixture(constant_index, features),
                 compile_spirv_fixture(argument_index, features),
                 compile_spirv_fixture(typed_argument_index, features)}) {
            expect(eq(compiled.required_features, expected));
            expect(contains(compiled.text,
                            "OpCapability RuntimeDescriptorArray"));
            expect(contains(
                compiled.text,
                "OpCapability SampledImageArrayDynamicIndexing"));
            expect(!contains(
                compiled.text,
                "OpCapability SampledImageArrayNonUniformIndexing"));
            expect(!contains(compiled.text,
                             "OpCapability ShaderNonUniform"));
            expect(!contains(compiled.text, " NonUniform"));
            expect(!lc::spirv::spirv_target_feature_is_capability_owned(
                lc::spirv::target_feature::runtime_descriptor_array));
            auto reconciled =
                lc::spirv::reconcile_spirv_target_features(
                    compiled.words.data(), compiled.words.size(), 0u);
            expect((reconciled &
                    lc::spirv::target_feature::runtime_descriptor_array) !=
                   0u)
                << "a present RuntimeDescriptorArray capability must recover "
                   "an omitted runtime-owned artifact requirement";
            auto facts = inspect_configured_sampler_path(
                luisa::span<const uint32_t>{compiled.words});
            expect_bindless_image_fetch_path(facts, false);
        }
    };

    "spirv_target_features_divergent_bindless_texture_indexing_is_exact"_test = [] {
        constexpr auto expected =
            lc::spirv::target_feature::descriptor_indexing |
            lc::spirv::target_feature::runtime_descriptor_array |
            lc::spirv::target_feature::descriptor_binding_partially_bound |
            lc::spirv::target_feature::descriptor_binding_sampled_image_update_after_bind |
            lc::spirv::target_feature::sampled_image_array_non_uniform_indexing;
        constexpr auto features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(expected);
        Kernel1D kernel = [](BindlessVar bindless,
                             BufferFloat4 output) noexcept {
            output.write(
                0u, bindless.tex2d(dispatch_id().x)
                        .read(make_uint2(0u)));
        };
        Kernel1D typed_kernel = [](BindlessVar bindless,
                                   BufferFloat4 output) noexcept {
            output.write(
                0u, bindless.tex2d(dispatch_id().x, true, false)
                        .read(make_uint2(0u)));
        };
        for (auto compiled : {
                 compile_spirv_fixture(kernel, features),
                 compile_spirv_fixture(typed_kernel, features)}) {
            expect(eq(compiled.required_features, expected));
            expect(contains(compiled.text,
                            "OpCapability RuntimeDescriptorArray"));
            expect(contains(
                compiled.text,
                "OpCapability SampledImageArrayNonUniformIndexing"));
            expect(contains(compiled.text,
                            "OpCapability ShaderNonUniform"));
            expect(!contains(
                compiled.text,
                "OpCapability SampledImageArrayDynamicIndexing"));
            expect(contains(compiled.text, " NonUniform"));
            auto facts = inspect_configured_sampler_path(
                luisa::span<const uint32_t>{compiled.words});
            expect_bindless_image_fetch_path(facts, true);
        }
    };

    "spirv_target_features_uniform_bindless_buffer_indexing_is_exact"_test = [] {
        constexpr auto expected =
            lc::spirv::target_feature::descriptor_indexing |
            lc::spirv::target_feature::runtime_descriptor_array |
            lc::spirv::target_feature::descriptor_binding_partially_bound |
            lc::spirv::target_feature::descriptor_binding_storage_buffer_update_after_bind |
            lc::spirv::target_feature::storage_buffer_array_dynamic_indexing;
        constexpr auto features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(expected);
        Kernel1D constant_index = [](BindlessVar bindless,
                                     BufferUInt output) noexcept {
            output.write(
                0u, bindless.buffer<uint32_t>(0u, false, true).read(0u));
        };
        Kernel1D argument_index = [](BindlessVar bindless,
                                     BufferUInt output,
                                     UInt slot) noexcept {
            output.write(
                0u, bindless.buffer<uint32_t>(slot, false, true).read(0u));
        };
        Kernel1D typed_argument_index = [](BindlessVar bindless,
                                           BufferUInt output,
                                           UInt slot) noexcept {
            output.write(
                0u, bindless.buffer<uint32_t>(slot, true, true).read(0u));
        };
        for (auto compiled : {
                 compile_spirv_fixture(constant_index, features),
                 compile_spirv_fixture(argument_index, features),
                 compile_spirv_fixture(typed_argument_index, features)}) {
            expect(eq(compiled.required_features, expected));
            expect(contains(compiled.text,
                            "OpCapability RuntimeDescriptorArray"));
            expect(contains(
                compiled.text,
                "OpCapability StorageBufferArrayDynamicIndexing"));
            expect(!contains(
                compiled.text,
                "OpCapability StorageBufferArrayNonUniformIndexing"));
            expect(!contains(compiled.text,
                             "OpCapability ShaderNonUniform"));
            expect(!contains(compiled.text, " NonUniform"));
        }
    };

    "spirv_target_features_divergent_bindless_buffer_indexing_is_exact"_test = [] {
        constexpr auto expected =
            lc::spirv::target_feature::descriptor_indexing |
            lc::spirv::target_feature::runtime_descriptor_array |
            lc::spirv::target_feature::descriptor_binding_partially_bound |
            lc::spirv::target_feature::descriptor_binding_storage_buffer_update_after_bind |
            lc::spirv::target_feature::storage_buffer_array_non_uniform_indexing;
        constexpr auto features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(expected);
        Kernel1D kernel = [](BindlessVar bindless,
                             BufferUInt output) noexcept {
            output.write(
                0u, bindless.buffer<uint32_t>(dispatch_id().x).read(0u));
        };
        Kernel1D typed_kernel = [](BindlessVar bindless,
                                   BufferUInt output) noexcept {
            output.write(
                0u, bindless.buffer<uint32_t>(
                                dispatch_id().x, true, false)
                        .read(0u));
        };
        for (auto compiled : {
                 compile_spirv_fixture(kernel, features),
                 compile_spirv_fixture(typed_kernel, features)}) {
            expect(eq(compiled.required_features, expected));
            expect(contains(compiled.text,
                            "OpCapability RuntimeDescriptorArray"));
            expect(contains(
                compiled.text,
                "OpCapability StorageBufferArrayNonUniformIndexing"));
            expect(contains(compiled.text,
                            "OpCapability ShaderNonUniform"));
            expect(!contains(
                compiled.text,
                "OpCapability StorageBufferArrayDynamicIndexing"));
            expect(contains(compiled.text, " NonUniform"));
        }
    };
}
