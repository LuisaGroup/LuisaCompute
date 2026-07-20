// Test for the Vulkan XIR-to-SPIR-V atomic storage and feature contract.
// This test covers:
// - feature-exact planner choices for buffer and shared float32 atomics
// - explicit float16/float64 and shared compare-exchange boundaries
// - integer-word fallback versus native SPIR-V atomic instructions
// - typed nested aggregate storage for signed and unsigned int64 atomics

#include "ut/ut.hpp"

#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <spirv-tools/libspirv.hpp>

#include <luisa/dsl/sugar.h>

#include "spirv_codegen/entry.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

struct SpirvSignedAtomicLeaf {
    std::array<slong, 2u> values;
};

struct SpirvUnsignedAtomicLeaf {
    std::array<ulong2, 2u> values;
};

struct SpirvAggregateAtomicLeaves {
    uint prefix;
    float native_float;
    SpirvSignedAtomicLeaf signed_leaf;
    SpirvUnsignedAtomicLeaf unsigned_leaf;
};

LUISA_STRUCT(SpirvSignedAtomicLeaf, values) {};
LUISA_STRUCT(SpirvUnsignedAtomicLeaf, values) {};
LUISA_STRUCT(SpirvAggregateAtomicLeaves, prefix, native_float, signed_leaf, unsigned_leaf) {};

namespace {

constexpr auto shared_atomic_fixture_block_size = 32u;

[[nodiscard]] std::string disassemble(
    const std::vector<uint32_t> &words) {
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    std::string text;
    expect(tools.Validate(words.data(), words.size()))
        << "float-atomic SPIR-V fixture must validate";
    expect(tools.Disassemble(words.data(), words.size(), &text))
        << "failed to disassemble float-atomic SPIR-V fixture";
    return text;
}

struct CompiledSpirv {
    std::vector<uint32_t> words;
    std::string text;
    lc::spirv::SpirvTargetFeatureMask required_features{};
};

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

struct FloatTypeFacts {
    uint32_t width;
};

struct PointerTypeFacts {
    spv::StorageClass storage;
    uint32_t pointee;
};

struct FloatAtomicInstruction {
    spv::Op op;
    uint32_t result_type;
    uint32_t pointer;
    uint32_t value;
};

struct FloatAtomicDataflowFacts {
    bool native_fetch_sub_uses_typed_storage_buffer{false};
    bool native_fetch_sub_uses_fnegated_source{false};
    bool shared_exchange_is_typed_float{false};
};

[[nodiscard]] FloatAtomicDataflowFacts inspect_float_atomic_dataflow(
    const std::vector<uint32_t> &words) noexcept {
    std::unordered_map<uint32_t, FloatTypeFacts> float_types;
    std::unordered_map<uint32_t, PointerTypeFacts> pointer_types;
    std::unordered_map<uint32_t, uint32_t> value_types;
    std::unordered_map<uint32_t, uint32_t> fnegate_operands;
    std::unordered_map<uint32_t, uint32_t> float_constants;
    std::vector<FloatAtomicInstruction> atomics;
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            break;
        }
        auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (op == spv::Op::OpTypeFloat && word_count == 3u) {
            float_types.emplace(
                words[offset + 1u],
                FloatTypeFacts{.width = words[offset + 2u]});
        } else if (op == spv::Op::OpTypePointer && word_count == 4u) {
            pointer_types.emplace(
                words[offset + 1u],
                PointerTypeFacts{
                    .storage = static_cast<spv::StorageClass>(
                        words[offset + 2u]),
                    .pointee = words[offset + 3u]});
        } else if (op == spv::Op::OpConstant && word_count == 4u) {
            float_constants.emplace(
                words[offset + 2u], words[offset + 3u]);
        } else if (op == spv::Op::OpFNegate && word_count == 4u) {
            fnegate_operands.emplace(
                words[offset + 2u], words[offset + 3u]);
        } else if ((op == spv::Op::OpAtomicFAddEXT ||
                    op == spv::Op::OpAtomicExchange) &&
                   word_count == 7u) {
            atomics.emplace_back(FloatAtomicInstruction{
                .op = op,
                .result_type = words[offset + 1u],
                .pointer = words[offset + 3u],
                .value = words[offset + 6u]});
        }
        auto has_result = false;
        auto has_result_type = false;
        spv::HasResultAndType(
            op, &has_result, &has_result_type);
        if (has_result && has_result_type && word_count >= 3u) {
            value_types.emplace(
                words[offset + 2u], words[offset + 1u]);
        }
        offset += word_count;
    }

    auto is_f32 = [&](uint32_t type) noexcept {
        auto iter = float_types.find(type);
        return iter != float_types.end() &&
               iter->second.width == 32u;
    };
    auto value_is_f32 = [&](uint32_t value) noexcept {
        auto iter = value_types.find(value);
        return iter != value_types.end() && is_f32(iter->second);
    };
    auto pointer_is_f32 = [&](uint32_t value,
                              spv::StorageClass storage) noexcept {
        auto value_type = value_types.find(value);
        if (value_type == value_types.end()) { return false; }
        auto pointer_type = pointer_types.find(value_type->second);
        return pointer_type != pointer_types.end() &&
               pointer_type->second.storage == storage &&
               is_f32(pointer_type->second.pointee);
    };

    FloatAtomicDataflowFacts facts;
    constexpr auto one_point_two_five_bits = 0x3fa00000u;
    for (auto &&atomic : atomics) {
        if (!is_f32(atomic.result_type)) { continue; }
        if (atomic.op == spv::Op::OpAtomicFAddEXT &&
            pointer_is_f32(atomic.pointer,
                           spv::StorageClass::StorageBuffer)) {
            facts.native_fetch_sub_uses_typed_storage_buffer = true;
            auto negate = fnegate_operands.find(atomic.value);
            if (negate != fnegate_operands.end() &&
                value_is_f32(atomic.value) &&
                value_is_f32(negate->second)) {
                auto source = float_constants.find(negate->second);
                facts.native_fetch_sub_uses_fnegated_source =
                    source != float_constants.end() &&
                    source->second == one_point_two_five_bits;
            }
        } else if (atomic.op == spv::Op::OpAtomicExchange &&
                   pointer_is_f32(atomic.pointer,
                                  spv::StorageClass::Workgroup) &&
                   value_is_f32(atomic.value)) {
            facts.shared_exchange_is_typed_float = true;
        }
    }
    return facts;
}

template<typename Kernel>
[[nodiscard]] CompiledSpirv compile_spirv_fixture(
    Kernel &&kernel, lc::spirv::SpirvTargetFeatures features) {
    ScopedEnvironmentVariable disable_spirv_optimization{
        "LUISA_SPIRV_OPT_LEVEL", "0"};
    ScopedEnvironmentVariable clear_spirv_pass_override{
        "LUISA_SPIRV_OPT_PASSES", nullptr};
    ShaderOption option{.enable_cache = false};
    auto result = lc::spirv::SpirvCodegenEntry::compile_spirv(
        kernel.function()->function(), option, features);
    auto text = disassemble(result.spv_bin);
    return {
        .words = std::move(result.spv_bin),
        .text = std::move(text),
        .required_features = result.required_target_features};
}

template<typename Kernel>
[[nodiscard]] std::string compile_spirv_text(
    Kernel &&kernel, lc::spirv::SpirvTargetFeatures features) {
    return compile_spirv_fixture(
               std::forward<Kernel>(kernel), features)
        .text;
}

[[nodiscard]] bool contains(std::string_view text,
                            std::string_view needle) noexcept {
    return text.find(needle) != std::string_view::npos;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_float_atomic_planner_is_feature_exact"_test = [] {
        using lc::spirv::SpirvFloatAtomicImplementation;
        using lc::spirv::SpirvFloatAtomicStorage;
        using lc::spirv::SpirvTargetFeatures;
        using lc::spirv::plan_spirv_float_atomic;

        constexpr SpirvTargetFeatures none{};
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::EXCHANGE, 32u,
                   SpirvFloatAtomicStorage::BUFFER, none) ==
               SpirvFloatAtomicImplementation::WORD_EXCHANGE);
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::COMPARE_EXCHANGE, 32u,
                   SpirvFloatAtomicStorage::BUFFER, none) ==
               SpirvFloatAtomicImplementation::WORD_COMPARE_EXCHANGE);
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::FETCH_ADD, 32u,
                   SpirvFloatAtomicStorage::BUFFER, none) ==
               SpirvFloatAtomicImplementation::WORD_CAS);

        constexpr SpirvTargetFeatures buffer_exchange_only{
            .shader_buffer_float32_atomics = true};
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::EXCHANGE, 32u,
                   SpirvFloatAtomicStorage::BUFFER,
                   buffer_exchange_only) ==
               SpirvFloatAtomicImplementation::NATIVE_EXCHANGE);
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::FETCH_ADD, 32u,
                   SpirvFloatAtomicStorage::BUFFER,
                   buffer_exchange_only) ==
               SpirvFloatAtomicImplementation::WORD_CAS)
            << "float exchange support must not authorize atomic add";
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::FETCH_MIN, 32u,
                   SpirvFloatAtomicStorage::BUFFER,
                   buffer_exchange_only) ==
               SpirvFloatAtomicImplementation::WORD_CAS)
            << "float exchange support must not authorize atomic min/max";

        constexpr SpirvTargetFeatures buffer_min_max_only{
            .shader_buffer_float32_atomic_min_max = true};
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::FETCH_MIN, 32u,
                   SpirvFloatAtomicStorage::BUFFER,
                   buffer_min_max_only) ==
               SpirvFloatAtomicImplementation::NATIVE_MIN_MAX);
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::FETCH_ADD, 32u,
                   SpirvFloatAtomicStorage::BUFFER,
                   buffer_min_max_only) ==
               SpirvFloatAtomicImplementation::WORD_CAS)
            << "float min/max support must not authorize atomic add";

        constexpr SpirvTargetFeatures native_buffer{
            .shader_buffer_float32_atomics = true,
            .shader_buffer_float32_atomic_add = true,
            .shader_buffer_float32_atomic_min_max = true};
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::EXCHANGE, 32u,
                   SpirvFloatAtomicStorage::BUFFER, native_buffer) ==
               SpirvFloatAtomicImplementation::NATIVE_EXCHANGE);
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::FETCH_ADD, 32u,
                   SpirvFloatAtomicStorage::BUFFER, native_buffer) ==
               SpirvFloatAtomicImplementation::NATIVE_ADD);
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::FETCH_MIN, 32u,
                   SpirvFloatAtomicStorage::BUFFER, native_buffer) ==
               SpirvFloatAtomicImplementation::NATIVE_MIN_MAX);
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::COMPARE_EXCHANGE, 32u,
                   SpirvFloatAtomicStorage::BUFFER, native_buffer) ==
               SpirvFloatAtomicImplementation::WORD_COMPARE_EXCHANGE)
            << "SPIR-V float compare-exchange must stay integer word-backed";

        constexpr SpirvTargetFeatures shared_add_only{
            .shader_shared_float32_atomic_add = true};
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::FETCH_ADD, 32u,
                   SpirvFloatAtomicStorage::SHARED, shared_add_only) ==
               SpirvFloatAtomicImplementation::NATIVE_ADD);
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::EXCHANGE, 32u,
                   SpirvFloatAtomicStorage::SHARED, shared_add_only) ==
               SpirvFloatAtomicImplementation::UNSUPPORTED_FEATURE)
            << "atomic-add support must not authorize float exchange";
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::FETCH_MIN, 32u,
                   SpirvFloatAtomicStorage::SHARED, shared_add_only) ==
               SpirvFloatAtomicImplementation::UNSUPPORTED_FEATURE)
            << "atomic-add support must not authorize float min/max";
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::COMPARE_EXCHANGE, 32u,
                   SpirvFloatAtomicStorage::SHARED, shared_add_only) ==
               SpirvFloatAtomicImplementation::UNSUPPORTED_REPRESENTATION);
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::FETCH_ADD, 16u,
                   SpirvFloatAtomicStorage::BUFFER, native_buffer) ==
               SpirvFloatAtomicImplementation::UNSUPPORTED_WIDTH);
        expect(plan_spirv_float_atomic(
                   xir::AtomicOp::FETCH_ADD, 64u,
                   SpirvFloatAtomicStorage::BUFFER, native_buffer) ==
               SpirvFloatAtomicImplementation::UNSUPPORTED_WIDTH);
    };

    "spirv_atomic_buffer_storage_planner_rejects_incompatible_requirements"_test = [] {
        using lc::spirv::SpirvAtomicBufferStoragePlan;
        using lc::spirv::plan_spirv_atomic_buffer_storage;

        expect(plan_spirv_atomic_buffer_storage({}) ==
               SpirvAtomicBufferStoragePlan::TYPED)
            << "unconstrained scalar and aggregate atomics should use typed storage";
        expect(plan_spirv_atomic_buffer_storage(
                   {.has_float32_word_fallback = true}) ==
               SpirvAtomicBufferStoragePlan::WORD);
        expect(plan_spirv_atomic_buffer_storage(
                   {.contains_bool = true}) ==
               SpirvAtomicBufferStoragePlan::WORD);
        expect(plan_spirv_atomic_buffer_storage(
                   {.has_int64_atomic = true}) ==
               SpirvAtomicBufferStoragePlan::TYPED);
        expect(plan_spirv_atomic_buffer_storage(
                   {.contains_bool = true,
                    .has_int64_atomic = true}) ==
               SpirvAtomicBufferStoragePlan::CONFLICT)
            << "logical-bool word storage cannot provide a typed int64 pointer";
        expect(plan_spirv_atomic_buffer_storage(
                   {.has_float32_word_fallback = true,
                    .has_int64_atomic = true}) ==
               SpirvAtomicBufferStoragePlan::CONFLICT)
            << "float fallback word storage cannot provide a typed int64 pointer";
    };

    "spirv_float_atomic_buffer_fallback_has_no_float_capability"_test = [] {
        Kernel1D kernel = [](BufferFloat values, BufferFloat old_values) noexcept {
            old_values.write(0u, values.atomic(0u).fetch_add(1.25f));
        };
        auto compiled = compile_spirv_fixture(kernel, {});
        expect(contains(compiled.text, "OpAtomicCompareExchange"))
            << "buffer float add fallback must use an integer CAS loop";
        expect(!contains(compiled.text, "OpAtomicFAddEXT"));
        expect(!contains(compiled.text, "AtomicFloat32AddEXT"));
        expect(!contains(compiled.text, "SPV_EXT_shader_atomic_float_add"));
        expect(eq(compiled.required_features, 0u))
            << "integer-word float atomic fallback must not consume a native float-atomic feature";
    };

    "spirv_float_atomic_native_buffer_add_requires_feature"_test = [] {
        Kernel1D kernel = [](BufferFloat values, BufferFloat old_values) noexcept {
            old_values.write(0u, values.atomic(0u).fetch_sub(1.25f));
        };
        auto compiled = compile_spirv_fixture(
            kernel,
            {.shader_buffer_float32_atomic_add = true});
        expect(contains(compiled.text, "OpAtomicFAddEXT"));
        expect(contains(compiled.text, "AtomicFloat32AddEXT"));
        expect(contains(compiled.text, "SPV_EXT_shader_atomic_float_add"));
        expect(eq(
            compiled.required_features,
            lc::spirv::target_feature::shader_buffer_float32_atomic_add));
        auto facts = inspect_float_atomic_dataflow(compiled.words);
        expect(facts.native_fetch_sub_uses_typed_storage_buffer)
            << "native fetch-sub must emit a float32 OpAtomicFAddEXT through "
               "a float32 StorageBuffer pointer";
        expect(facts.native_fetch_sub_uses_fnegated_source)
            << "native fetch-sub must feed OpAtomicFAddEXT from OpFNegate of "
               "the exact 1.25f source constant";
    };

    "spirv_float_atomic_compare_exchange_forces_word_abi"_test = [] {
        Kernel1D kernel = [](BufferFloat values, BufferFloat old_values) noexcept {
            old_values.write(
                0u, values.atomic(0u).compare_exchange(0.0f, 2.0f));
            old_values.write(1u, values.atomic(1u).fetch_add(1.0f));
        };
        auto compiled = compile_spirv_fixture(
            kernel,
            {.shader_buffer_float32_atomics = true,
             .shader_buffer_float32_atomic_add = true,
             .shader_buffer_float32_atomic_min_max = true});
        expect(contains(compiled.text, "OpAtomicCompareExchange"));
        expect(!contains(compiled.text, "OpAtomicFAddEXT"))
            << "one word-only operation must select one consistent buffer ABI";
        expect(!contains(compiled.text, "AtomicFloat32AddEXT"));
        expect(eq(compiled.required_features, 0u))
            << "word-backed compare-exchange must suppress native requirements for the entire buffer ABI";
    };

    "spirv_float_atomic_shared_native_paths_are_structural"_test = [] {
        Kernel1D exchange_kernel = [](BufferFloat old_values) noexcept {
            set_block_size(shared_atomic_fixture_block_size, 1u, 1u);
            Shared<float> value{1u};
            auto lane = thread_id().x;
            $if (lane == 0u) { value.write(0u, 0.0f); };
            sync_block();
            $if (lane == 0u) {
                old_values.write(0u, value.atomic(0u).exchange(1.0f));
            };
        };
        auto exchange = compile_spirv_fixture(
            exchange_kernel,
            {.shader_shared_float32_atomics = true});
        expect(contains(exchange.text, "OpAtomicExchange"));
        expect(!contains(exchange.text, "AtomicFloat32AddEXT"));
        expect(!contains(exchange.text, "AtomicFloat32MinMaxEXT"));
        expect(eq(
            exchange.required_features,
            lc::spirv::target_feature::shader_shared_float32_atomics));
        auto exchange_facts = inspect_float_atomic_dataflow(exchange.words);
        expect(exchange_facts.shared_exchange_is_typed_float)
            << "shared float exchange must use a float32 Workgroup pointer, "
               "float32 value, and float32 OpAtomicExchange result";

        Kernel1D add_kernel = [](BufferFloat old_values) noexcept {
            set_block_size(shared_atomic_fixture_block_size, 1u, 1u);
            Shared<float> value{1u};
            auto lane = thread_id().x;
            $if (lane == 0u) { value.write(0u, 0.0f); };
            sync_block();
            $if (lane == 0u) {
                old_values.write(0u, value.atomic(0u).fetch_add(1.0f));
            };
        };
        auto add = compile_spirv_fixture(
            add_kernel,
            {.shader_shared_float32_atomic_add = true});
        expect(contains(add.text, "OpAtomicFAddEXT"));
        expect(contains(add.text, "AtomicFloat32AddEXT"));
        expect(eq(
            add.required_features,
            lc::spirv::target_feature::shader_shared_float32_atomic_add));

        Kernel1D min_kernel = [](BufferFloat old_values) noexcept {
            set_block_size(shared_atomic_fixture_block_size, 1u, 1u);
            Shared<float> value{1u};
            auto lane = thread_id().x;
            $if (lane == 0u) { value.write(0u, 2.0f); };
            sync_block();
            $if (lane == 0u) {
                old_values.write(0u, value.atomic(0u).fetch_min(1.0f));
            };
        };
        auto min = compile_spirv_fixture(
            min_kernel,
            {.shader_shared_float32_atomic_min_max = true});
        expect(contains(min.text, "OpAtomicFMinEXT"));
        expect(contains(min.text, "AtomicFloat32MinMaxEXT"));
        expect(contains(min.text, "SPV_EXT_shader_atomic_float_min_max"));
        expect(eq(
            min.required_features,
            lc::spirv::target_feature::shader_shared_float32_atomic_min_max));
    };

    "spirv_int64_atomic_records_storage_specific_requirement"_test = [] {
        Kernel1D buffer_kernel = [](
                                     BufferSLong values,
                                     BufferSLong old_values) noexcept {
            old_values.write(0u, values.atomic(0u).fetch_add(1ll));
        };
        auto buffer = compile_spirv_fixture(
            buffer_kernel,
            {.shader_int64 = true,
             .shader_buffer_int64_atomics = true});
        expect(contains(buffer.text, "Int64Atomics"));
        expect(contains(buffer.text, "OpAtomicIAdd"));
        expect(eq(
            buffer.required_features,
            lc::spirv::target_feature::shader_int64 |
                lc::spirv::target_feature::shader_buffer_int64_atomics));

        Kernel1D shared_kernel = [](BufferSLong old_values) noexcept {
            set_block_size(shared_atomic_fixture_block_size, 1u, 1u);
            Shared<slong> value{1u};
            auto lane = thread_id().x;
            $if (lane == 0u) { value.write(0u, 0ll); };
            sync_block();
            $if (lane == 0u) {
                old_values.write(0u, value.atomic(0u).fetch_add(1ll));
            };
        };
        auto shared = compile_spirv_fixture(
            shared_kernel,
            {.shader_int64 = true,
             .shader_shared_int64_atomics = true});
        expect(contains(shared.text, "Int64Atomics"));
        expect(contains(shared.text, "OpAtomicIAdd"));
        expect(eq(
            shared.required_features,
            lc::spirv::target_feature::shader_int64 |
                lc::spirv::target_feature::shader_shared_int64_atomics));
    };

    "spirv_nested_aggregate_int64_atomics_use_typed_signed_pointers"_test = [] {
        Kernel1D kernel = [](BufferVar<SpirvAggregateAtomicLeaves> values) noexcept {
            auto lane = dispatch_id().x & 1u;
            auto atomic = values.atomic(0u);
            atomic.signed_leaf.values[lane].fetch_min(-7ll);
            atomic.unsigned_leaf.values[1u][lane].fetch_max(9ull);
        };
        auto compiled = compile_spirv_fixture(
            kernel,
            {.shader_int64 = true,
             .shader_buffer_int64_atomics = true});
        expect(contains(compiled.text, "Int64Atomics"));
        expect(contains(compiled.text, "OpTypeInt 64 1"));
        expect(contains(compiled.text, "OpTypeInt 64 0"));
        expect(contains(compiled.text, "OpAtomicSMin"))
            << "signed nested int64 leaf must use the signed atomic opcode";
        expect(contains(compiled.text, "OpAtomicUMax"))
            << "unsigned nested int64 leaf must use the unsigned atomic opcode";
        expect(eq(
            compiled.required_features,
            lc::spirv::target_feature::shader_int64 |
                lc::spirv::target_feature::shader_buffer_int64_atomics));
    };

    "spirv_native_float_and_int64_aggregate_share_typed_storage"_test = [] {
        Kernel1D kernel = [](BufferVar<SpirvAggregateAtomicLeaves> values) noexcept {
            auto atomic = values.atomic(0u);
            atomic.signed_leaf.values[0u].fetch_add(1ll);
            atomic.native_float.fetch_add(1.0f);
        };
        auto compiled = compile_spirv_fixture(
            kernel,
            {.shader_int64 = true,
             .shader_buffer_float32_atomic_add = true,
             .shader_buffer_int64_atomics = true});
        expect(contains(compiled.text, "OpTypeInt 64 1"));
        expect(contains(compiled.text, "OpAtomicIAdd"));
        expect(contains(compiled.text, "OpAtomicFAddEXT"));
        expect(eq(
            compiled.required_features,
            lc::spirv::target_feature::shader_int64 |
                lc::spirv::target_feature::shader_buffer_float32_atomic_add |
                lc::spirv::target_feature::shader_buffer_int64_atomics));
    };
}
