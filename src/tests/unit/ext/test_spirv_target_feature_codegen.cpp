// Test for exact Vulkan XIR-to-SPIR-V target-feature accounting.
// This test covers:
// - baseline and subgroup feature requirements
// - storage-image, ray-query, and sampler-selector requirements
// - uniform and divergent unbounded-descriptor indexing contracts

#include "ut/ut.hpp"

#include <algorithm>
#include <array>
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
    std::optional<std::string> _previous;

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

[[nodiscard]] bool contains(std::string_view text,
                            std::string_view needle) noexcept {
    return text.find(needle) != std::string_view::npos;
}

[[nodiscard]] size_t count_property(
    const CompiledSpirv &compiled,
    lc::hlsl::ShaderVariableType type) noexcept {
    return static_cast<size_t>(std::ranges::count(
        compiled.property_types, type));
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
    luisa::vector<std::optional<uint64_t>> integer_constants(id_bound);
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
    luisa::vector<std::optional<uint32_t>> descriptor_sets(id_bound);
    luisa::vector<std::optional<uint32_t>> descriptor_bindings(id_bound);
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
        expect(contains(compiled.text, "OpConstantComposite"));
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
                  lc::spirv::target_feature::ray_query));
        expect(contains(compiled.text, "OpCapability RayQueryKHR"));
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
        for (auto compiled : {
                 compile_spirv_fixture(constant_index, features),
                 compile_spirv_fixture(argument_index, features)}) {
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
        auto compiled = compile_spirv_fixture(kernel, features);
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
        for (auto compiled : {
                 compile_spirv_fixture(constant_index, features),
                 compile_spirv_fixture(argument_index, features)}) {
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
        auto compiled = compile_spirv_fixture(kernel, features);
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
    };
}
