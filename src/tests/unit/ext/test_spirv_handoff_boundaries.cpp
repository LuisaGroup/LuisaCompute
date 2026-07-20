// Cross-layer tests for native XIR-to-SPIR-V handoff boundaries that must be
// rejected before the emitter mutates a SPIR-V module.

#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/GLSL.std.450.h>

#include <luisa/ast/type.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

#include "spirv_codegen/atomic_buffer_plan.h"
#include "spirv_codegen/atomic_target_contract.h"
#include "spirv_codegen/call_graph_validation.h"
#include "spirv_codegen/dialect.h"
#include "spirv_codegen/entry.h"
#include "spirv_codegen/kernel_argument_layout.h"
#include "spirv_codegen/texture_sampling.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool has_instruction_diagnostic(
    const lc::spirv::SpirvXIRDialectValidationResult &validation,
    const Instruction *instruction,
    std::string_view needle) noexcept {
    for (auto &&diagnostic : validation.diagnostics) {
        if (diagnostic.instruction == instruction &&
            diagnostic.message.find(needle) != luisa::string::npos) {
            return true;
        }
    }
    return false;
}

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

[[nodiscard]] std::string decode_spirv_literal_string(
    const std::vector<uint32_t> &words, size_t begin,
    size_t end) {
    std::string text;
    for (auto index = begin; index < end; ++index) {
        auto word = words[index];
        for (auto byte = 0u; byte < 4u; ++byte) {
            auto c = static_cast<char>((word >> (byte * 8u)) & 0xffu);
            if (c == '\0') { return text; }
            text.push_back(c);
        }
    }
    return text;
}

[[nodiscard]] bool has_exact_f64_glsl_sqrt(
    const std::vector<uint32_t> &words) {
    std::unordered_set<uint32_t> f64_types;
    std::unordered_set<uint32_t> glsl450_imports;
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            return false;
        }
        auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (op == spv::Op::OpTypeFloat && word_count == 3u &&
            words[offset + 2u] == 64u) {
            f64_types.emplace(words[offset + 1u]);
        } else if (op == spv::Op::OpExtInstImport &&
                   word_count >= 3u &&
                   decode_spirv_literal_string(
                       words, offset + 2u,
                       offset + word_count) == "GLSL.std.450") {
            glsl450_imports.emplace(words[offset + 1u]);
        }
        offset += word_count;
    }
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            return false;
        }
        auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (op == spv::Op::OpExtInst && word_count >= 6u &&
            f64_types.contains(words[offset + 1u]) &&
            glsl450_imports.contains(words[offset + 3u]) &&
            words[offset + 4u] ==
                static_cast<uint32_t>(GLSLstd450Sqrt)) {
            return true;
        }
        offset += word_count;
    }
    return false;
}

[[nodiscard]] lc::spirv::SpirvAtomicBufferModulePlan
plan_atomic_buffers(
    const Module &module,
    const lc::spirv::SpirvTargetFeatures *features) noexcept {
    auto graph = lc::spirv::validate_spirv_reachable_call_graph(
        &module);
    expect(graph.succeeded());
    return lc::spirv::plan_spirv_atomic_buffers(
        luisa::span<const xir::Function *const>{
            graph.functions_post_order.data(),
            graph.functions_post_order.size()},
        {.target_features = features});
}

[[nodiscard]] lc::spirv::SpirvAtomicTargetContractResult
validate_atomic_target(
    const Module &module,
    const lc::spirv::SpirvTargetFeatures &features) noexcept {
    auto graph = lc::spirv::validate_spirv_reachable_call_graph(
        &module);
    expect(graph.succeeded());
    return lc::spirv::validate_spirv_atomic_target_contract(
        luisa::span<const xir::Function *const>{
            graph.functions_post_order.data(),
            graph.functions_post_order.size()},
        features);
}

[[nodiscard]] const Type *make_bool_int64_atomic_element() noexcept {
    return Type::structure(
        {Type::of<bool>(), Type::of<int64_t>()});
}

[[nodiscard]] const Type *make_float_int64_atomic_element() noexcept {
    return Type::structure(
        {Type::of<float>(), Type::of<int64_t>()});
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_texture_sample_operation_shapes_are_exhaustive"_test = [] {
        enum class ExpectedLod : uint8_t {
            IMPLICIT,
            EXPLICIT,
            GRADIENT,
            GRADIENT_WITH_MIN_LOD,
        };
        struct ExpectedShape {
            ResourceQueryOp op;
            bool direct;
            bool is_2d;
            ExpectedLod lod;
            bool configured_sampler;
        };
        constexpr std::array cases{
            ExpectedShape{ResourceQueryOp::TEXTURE2D_SAMPLE, true, true,
                          ExpectedLod::IMPLICIT, true},
            ExpectedShape{ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL, true, true,
                          ExpectedLod::EXPLICIT, true},
            ExpectedShape{ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD, true, true,
                          ExpectedLod::GRADIENT, true},
            ExpectedShape{ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL, true,
                          true, ExpectedLod::GRADIENT_WITH_MIN_LOD, true},
            ExpectedShape{ResourceQueryOp::TEXTURE3D_SAMPLE, true, false,
                          ExpectedLod::IMPLICIT, true},
            ExpectedShape{ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL, true, false,
                          ExpectedLod::EXPLICIT, true},
            ExpectedShape{ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD, true, false,
                          ExpectedLod::GRADIENT, true},
            ExpectedShape{ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL, true,
                          false, ExpectedLod::GRADIENT_WITH_MIN_LOD, true},

            ExpectedShape{ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE, false,
                          true, ExpectedLod::IMPLICIT, false},
            ExpectedShape{ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL,
                          false, true, ExpectedLod::EXPLICIT, false},
            ExpectedShape{ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD,
                          false, true, ExpectedLod::GRADIENT, false},
            ExpectedShape{
                ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL, false,
                true, ExpectedLod::GRADIENT_WITH_MIN_LOD, false},
            ExpectedShape{ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE, false,
                          false, ExpectedLod::IMPLICIT, false},
            ExpectedShape{ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL,
                          false, false, ExpectedLod::EXPLICIT, false},
            ExpectedShape{ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD,
                          false, false, ExpectedLod::GRADIENT, false},
            ExpectedShape{
                ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL, false,
                false, ExpectedLod::GRADIENT_WITH_MIN_LOD, false},

            ExpectedShape{ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
                          false, true, ExpectedLod::IMPLICIT, true},
            ExpectedShape{
                ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,
                false, true, ExpectedLod::EXPLICIT, true},
            ExpectedShape{
                ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,
                false, true, ExpectedLod::GRADIENT, true},
            ExpectedShape{
                ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,
                false, true, ExpectedLod::GRADIENT_WITH_MIN_LOD, true},
            ExpectedShape{ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
                          false, false, ExpectedLod::IMPLICIT, true},
            ExpectedShape{
                ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,
                false, false, ExpectedLod::EXPLICIT, true},
            ExpectedShape{
                ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,
                false, false, ExpectedLod::GRADIENT, true},
            ExpectedShape{
                ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,
                false, false, ExpectedLod::GRADIENT_WITH_MIN_LOD, true},
        };

        std::unordered_set<uint32_t> seen;
        for (auto expected : cases) {
            auto key = static_cast<uint32_t>(expected.op);
            expect(seen.emplace(key).second)
                << "sampling operation appears more than once in the shape table";
            auto actual =
                lc::spirv::spirv_texture_sample_op_info(expected.op);
            expect(actual.valid);
            expect(eq(actual.direct, expected.direct));
            expect(eq(actual.is_2d, expected.is_2d));
            expect(eq(actual.explicit_lod,
                      expected.lod == ExpectedLod::EXPLICIT));
            expect(eq(
                actual.gradients,
                expected.lod == ExpectedLod::GRADIENT ||
                    expected.lod == ExpectedLod::GRADIENT_WITH_MIN_LOD));
            expect(eq(actual.lod_clamp,
                      expected.lod ==
                          ExpectedLod::GRADIENT_WITH_MIN_LOD));
            expect(eq(actual.sampler_operands,
                      expected.configured_sampler));
        }
        expect(eq(seen.size(), cases.size()));
        expect(!lc::spirv::spirv_texture_sample_op_info(
                    ResourceQueryOp::TEXTURE2D_SIZE)
                    .valid);
    };

    "spirv_float64_transcendental_table_fails_at_dialect_handoff"_test = [] {
        constexpr std::array operations{
            ArithmeticOp::ACOS,
            ArithmeticOp::ACOSH,
            ArithmeticOp::ASIN,
            ArithmeticOp::ASINH,
            ArithmeticOp::ATAN,
            ArithmeticOp::ATAN2,
            ArithmeticOp::ATANH,
            ArithmeticOp::COS,
            ArithmeticOp::COSH,
            ArithmeticOp::SIN,
            ArithmeticOp::SINH,
            ArithmeticOp::TAN,
            ArithmeticOp::TANH,
            ArithmeticOp::EXP,
            ArithmeticOp::EXP2,
            ArithmeticOp::EXP10,
            ArithmeticOp::LOG,
            ArithmeticOp::LOG2,
            ArithmeticOp::LOG10,
            ArithmeticOp::POW};
        for (auto op : operations) {
            Module module;
            auto *kernel = module.create_kernel();
            auto *value =
                kernel->create_value_argument(Type::of<double>());
            XIRBuilder builder;
            builder.set_insertion_point(
                kernel->create_body_block());
            auto *instruction =
                op == ArithmeticOp::ATAN2 ||
                        op == ArithmeticOp::POW ?
                    builder.call(Type::of<double>(), op,
                                 {value, value}) :
                    builder.call(Type::of<double>(), op, {value});
            builder.return_void();

            auto generic = xir_verify_module(&module);
            expect(generic.succeeded())
                << "float64 transcendental fixture must be valid generic XIR";
            auto dialect =
                lc::spirv::validate_spirv_xir_codegen_dialect(
                    &module);
            expect(!dialect.succeeded());
            expect(has_instruction_diagnostic(
                dialect, instruction, "GLSL.std.450"));
            expect(has_instruction_diagnostic(
                dialect, instruction, "float64"));
        }
    };

    "spirv_float64_sqrt_remains_supported_at_dialect_handoff"_test = [] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        Module module;
        auto *kernel = module.create_kernel();
        auto *value = kernel->create_value_argument(Type::of<double>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *sqrt = builder.call(
            Type::of<double>(), ArithmeticOp::SQRT, {value});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        expect(lc::spirv::validate_spirv_xir_codegen_dialect(&module)
                   .succeeded());
        Kernel1D ast_kernel = [](Var<double>) noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        auto compiled = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            ast_kernel.function()->function(), &module,
            ShaderOption{.enable_cache = false},
            {.shader_float64 = true});
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
            << "float64 sqrt exact-XIR output failed Vulkan 1.2 validation: "
            << diagnostics;
        expect(has_exact_f64_glsl_sqrt(compiled.spv_bin))
            << "the accepted XIR instruction must emit a float64 "
               "GLSL.std.450 Sqrt, not merely pass dialect validation";
        expect(eq(
            compiled.required_target_features,
            lc::spirv::target_feature::shader_float64));
        expect(sqrt != nullptr);
    };

    "spirv_sampler_constant_range_is_a_backend_only_boundary"_test = [] {
        for (auto invalid_filter : {true, false}) {
            Module module;
            auto *kernel = module.create_kernel();
            auto *texture = kernel->create_resource_argument(
                Type::texture(Type::of<float>(), 2u));
            auto *uv = module.create_constant_zero(
                Type::of<float2>());
            auto *valid = module.create_constant_zero(
                Type::of<uint32_t>());
            const uint32_t invalid_selector = 4u;
            auto *invalid = module.create_constant(
                Type::of<uint32_t>(), &invalid_selector);
            XIRBuilder builder;
            builder.set_insertion_point(
                kernel->create_body_block());
            auto *sample = builder.call(
                Type::of<float4>(),
                ResourceQueryOp::TEXTURE2D_SAMPLE,
                {texture, uv,
                 invalid_filter ? invalid : valid,
                 invalid_filter ? valid : invalid});
            builder.return_void();

            expect(xir_verify_module(&module).succeeded())
                << "sampler enum range is a backend/runtime contract, not a generic XIR shape rule";
            auto dialect =
                lc::spirv::validate_spirv_xir_codegen_dialect(
                    &module);
            expect(!dialect.succeeded());
            expect(has_instruction_diagnostic(
                dialect, sample,
                invalid_filter ? "filter selector 4" :
                                 "address selector 4"));
            expect(has_instruction_diagnostic(
                dialect, sample, "outside [0, 4)"));
        }
    };

    "spirv_sampler_selector_type_is_exactly_uint32"_test = [] {
        expect(lc::spirv::spirv_sampler_selector_type_supported(
            Type::of<uint32_t>()));
        const std::array invalid_types{
            Type::of<int32_t>(),
            Type::of<uint16_t>(),
            Type::of<uint64_t>()};
        for (auto *invalid_type : invalid_types) {
            expect(!lc::spirv::spirv_sampler_selector_type_supported(
                invalid_type));
            for (auto invalid_filter : {true, false}) {
                Module module;
                auto *kernel = module.create_kernel();
                auto *texture = kernel->create_resource_argument(
                    Type::texture(Type::of<float>(), 2u));
                auto *uv = module.create_constant_zero(
                    Type::of<float2>());
                auto *invalid = kernel->create_value_argument(
                    invalid_type);
                auto *valid = module.create_constant_zero(
                    Type::of<uint32_t>());
                XIRBuilder builder;
                builder.set_insertion_point(
                    kernel->create_body_block());
                luisa::compute::xir::Value *filter =
                    invalid_filter ?
                        static_cast<luisa::compute::xir::Value *>(invalid) :
                        static_cast<luisa::compute::xir::Value *>(valid);
                luisa::compute::xir::Value *address =
                    invalid_filter ?
                        static_cast<luisa::compute::xir::Value *>(valid) :
                        static_cast<luisa::compute::xir::Value *>(invalid);
                auto *sample = builder.call(
                    Type::of<float4>(),
                    ResourceQueryOp::TEXTURE2D_SAMPLE,
                    {texture, uv, filter, address});
                builder.return_void();

                expect(!xir_verify_module(&module).succeeded())
                    << "generic XIR must reject configured-sampler selectors that are not uint32";
                auto dialect =
                    lc::spirv::validate_spirv_xir_codegen_dialect(
                        &module);
                expect(!dialect.succeeded());
                expect(has_instruction_diagnostic(
                    dialect, sample,
                    invalid_filter ?
                        "filter selector must be uint32" :
                        "address selector must be uint32"));
            }
        }
    };

    "spirv_sampler_anisotropy_is_checked_at_target_preflight"_test = [] {
        using lc::spirv::SpirvSamplerFilterPlan;
        using lc::spirv::plan_spirv_sampler_filter;
        expect(plan_spirv_sampler_filter(true, 0u, false) ==
               SpirvSamplerFilterPlan::SUPPORTED);
        expect(plan_spirv_sampler_filter(true, 3u, false) ==
               SpirvSamplerFilterPlan::REQUIRES_ANISOTROPY);
        expect(plan_spirv_sampler_filter(false, 0u, false) ==
               SpirvSamplerFilterPlan::REQUIRES_ANISOTROPY);
        expect(plan_spirv_sampler_filter(true, 4u, true) ==
               SpirvSamplerFilterPlan::INVALID_SELECTOR);
        expect(plan_spirv_sampler_filter(true, 3u, true) ==
               SpirvSamplerFilterPlan::SUPPORTED);

        Module module;
        auto *kernel = module.create_kernel();
        auto *texture = kernel->create_resource_argument(
            Type::texture(Type::of<float>(), 2u));
        auto *uv = module.create_constant_zero(Type::of<float2>());
        const uint32_t anisotropic_selector = 3u;
        auto *anisotropic = module.create_constant(
            Type::of<uint32_t>(), &anisotropic_selector);
        auto *address =
            module.create_constant_zero(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *sample = builder.call(
            Type::of<float4>(),
            ResourceQueryOp::TEXTURE2D_SAMPLE,
            {texture, uv, anisotropic, address});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        expect(lc::spirv::validate_spirv_xir_codegen_dialect(
                   &module)
                   .succeeded())
            << "selector 3 is semantically valid independent of target features";
        auto graph =
            lc::spirv::validate_spirv_reachable_call_graph(
                &module);
        expect(graph.succeeded());
        auto functions = luisa::span<const xir::Function *const>{
            graph.functions_post_order.data(),
            graph.functions_post_order.size()};
        auto unavailable =
            lc::spirv::validate_spirv_sampler_target_contract(
                functions, false);
        expect(!unavailable.succeeded());
        expect(eq(unavailable.diagnostics.size(), 1u));
        if (!unavailable.diagnostics.empty()) {
            expect(unavailable.diagnostics.front().instruction ==
                   sample);
            expect(unavailable.diagnostics.front().message.find(
                       "samplerAnisotropy") != luisa::string::npos);
        }
        expect(lc::spirv::validate_spirv_sampler_target_contract(
                   functions, true)
                   .succeeded());
    };

    "spirv_kernel_argument_layout_rejects_direct_buffer_metadata_capacity_overflow"_test = [] {
        constexpr auto uint32_limit =
            std::numeric_limits<uint32_t>::max();
        auto *huge = Type::array(
            Type::of<uint8_t>(), uint32_limit - 7u);
        expect(eq(huge->size(),
                  static_cast<size_t>(uint32_limit - 7u)));

        Module module;
        auto *kernel = module.create_kernel();
        kernel->create_value_argument(huge);
        kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        builder.return_void();

        expect(xir_verify_module(&module).succeeded())
            << "the array and direct buffer are valid generic XIR; only the native argument-block ABI is too small for the metadata record";
        auto layout =
            lc::spirv::plan_spirv_kernel_argument_layout(
                kernel);
        expect(!layout.succeeded());
        expect(layout.status ==
                   lc::ArgumentBlockLayoutStatus::LIMIT_EXCEEDED ||
               layout.status ==
                   lc::ArgumentBlockLayoutStatus::ADDITION_OVERFLOW);
        expect(layout.diagnostic.find(
                   "direct-buffer metadata") != luisa::string::npos);
        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(
                &module);
        expect(!dialect.succeeded());
        expect(std::ranges::any_of(
            dialect.diagnostics, [](auto &&diagnostic) noexcept {
                return diagnostic.message.find(
                           "direct-buffer metadata") !=
                       luisa::string::npos;
            }));
    };

    "spirv_kernel_argument_layout_matches_runtime_metadata_abi"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *first = kernel->create_value_argument(
            Type::of<uint8_t>());
        auto *second = kernel->create_value_argument(
            Type::of<double>());
        kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        builder.return_void();

        auto layout =
            lc::spirv::plan_spirv_kernel_argument_layout(
                kernel);
        expect(layout.succeeded());
        expect(eq(layout.value_arguments.size(), 2u));
        if (layout.value_arguments.size() == 2u) {
            expect(layout.value_arguments[0u].argument == first);
            expect(eq(layout.value_arguments[0u].byte_offset, 0u));
            expect(layout.value_arguments[1u].argument == second);
            expect(eq(layout.value_arguments[1u].byte_offset, 8u));
        }
        expect(eq(layout.buffer_metadata_count, 1u));
        expect(eq(layout.buffer_metadata_offset, 16u));
        expect(eq(layout.final_size, 32u));
    };

    "spirv_kernel_argument_layout_without_buffers_only_word_aligns"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        kernel->create_value_argument(Type::of<uint8_t>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        builder.return_void();

        auto layout =
            lc::spirv::plan_spirv_kernel_argument_layout(kernel);
        expect(layout.succeeded());
        expect(eq(layout.buffer_metadata_count, 0u));
        expect(eq(layout.buffer_metadata_offset, 1u));
        expect(eq(layout.final_size, 4u));
    };

    "spirv_bool_and_int64_atomic_buffer_conflict_fails_in_dialect"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *element = make_bool_int64_atomic_element();
        auto *buffer = kernel->create_resource_argument(
            Type::buffer(element));
        auto *zero =
            module.create_constant_zero(Type::of<uint32_t>());
        auto *member =
            module.create_constant_one(Type::of<uint32_t>());
        auto *one =
            module.create_constant_one(Type::of<int64_t>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        std::array<Value *, 2u> indices{zero, member};
        auto *atomic = builder.atomic_fetch_add(
            Type::of<int64_t>(), buffer,
            luisa::span<Value *const>{indices.data(), indices.size()}, one);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded())
            << "logical bool layout and int64 atomics are independently valid generic XIR features";
        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(
                &module);
        expect(!dialect.succeeded());
        expect(has_instruction_diagnostic(
            dialect, atomic, "logical-bool member"));
        expect(has_instruction_diagnostic(
            dialect, atomic, "one SPIR-V Logical pointer type"));
    };

    "spirv_float_fallback_and_int64_atomic_conflict_is_target_dependent"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *buffer = kernel->create_resource_argument(
            Type::buffer(make_float_int64_atomic_element()));
        auto *zero =
            module.create_constant_zero(Type::of<uint32_t>());
        auto *member_one =
            module.create_constant_one(Type::of<uint32_t>());
        auto *float_one =
            module.create_constant_one(Type::of<float>());
        auto *int64_one =
            module.create_constant_one(Type::of<int64_t>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        std::array<Value *, 2u> float_indices{zero, zero};
        std::array<Value *, 2u> int64_indices{zero, member_one};
        auto *float_atomic = builder.atomic_fetch_add(
            Type::of<float>(), buffer,
            luisa::span<Value *const>{float_indices.data(),
                                      float_indices.size()},
            float_one);
        auto *int64_atomic = builder.atomic_fetch_add(
            Type::of<int64_t>(), buffer,
            luisa::span<Value *const>{int64_indices.data(),
                                      int64_indices.size()},
            int64_one);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        expect(lc::spirv::validate_spirv_xir_codegen_dialect(
                   &module)
                   .succeeded())
            << "native float32 atomic availability is a target, not dialect, property";

        constexpr lc::spirv::SpirvTargetFeatures no_float_atomics{};
        auto fallback = plan_atomic_buffers(
            module, &no_float_atomics);
        expect(!fallback.succeeded());
        expect(eq(fallback.diagnostics.size(), 1u));
        if (!fallback.diagnostics.empty()) {
            expect(fallback.diagnostics.front().instruction ==
                   float_atomic)
                << "the diagnostic should identify the float operation that forces word storage";
            expect(fallback.diagnostics.front().instruction !=
                   int64_atomic);
            expect(fallback.diagnostics.front().message.find(
                       "uint32 word fallback") !=
                   luisa::string::npos);
        }

        constexpr lc::spirv::SpirvTargetFeatures native_add{
            .shader_buffer_float32_atomic_add = true};
        auto native = plan_atomic_buffers(module, &native_add);
        expect(native.succeeded());
        expect(eq(native.assignments.size(), 1u));
        if (!native.assignments.empty()) {
            expect(native.assignments.front().storage ==
                   lc::spirv::SpirvAtomicBufferStoragePlan::TYPED);
        }
    };

    "spirv_shared_float_atomic_feature_is_checked_before_emission"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *zero =
            module.create_constant_zero(Type::of<uint32_t>());
        auto *one = module.create_constant_one(Type::of<float>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *shared = builder.alloca_shared(
            Type::array(Type::of<float>(), 1u));
        std::array<Value *, 1u> indices{zero};
        auto *atomic = builder.atomic_fetch_add(
            Type::of<float>(), shared,
            luisa::span<Value *const>{indices.data(), indices.size()}, one);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        expect(lc::spirv::validate_spirv_xir_codegen_dialect(
                   &module)
                   .succeeded());
        auto unavailable = validate_atomic_target(
            module, lc::spirv::SpirvTargetFeatures{});
        expect(!unavailable.succeeded());
        expect(eq(unavailable.diagnostics.size(), 1u));
        if (!unavailable.diagnostics.empty()) {
            expect(unavailable.diagnostics.front().instruction ==
                   atomic);
            expect(unavailable.diagnostics.front().message.find(
                       "shaderSharedFloat32AtomicAdd") !=
                   luisa::string::npos);
        }
        constexpr lc::spirv::SpirvTargetFeatures supported{
            .shader_shared_float32_atomic_add = true};
        expect(validate_atomic_target(module, supported).succeeded());
    };

    "spirv_int64_atomic_storage_features_are_checked_before_emission"_test = [] {
        for (auto shared_storage : {false, true}) {
            Module module;
            auto *kernel = module.create_kernel();
            auto *zero =
                module.create_constant_zero(Type::of<uint32_t>());
            auto *one =
                module.create_constant_one(Type::of<int64_t>());
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            Value *base = nullptr;
            if (shared_storage) {
                base = builder.alloca_shared(
                    Type::array(Type::of<int64_t>(), 1u));
            } else {
                base = kernel->create_resource_argument(
                    Type::buffer(Type::of<int64_t>()));
            }
            std::array<Value *, 1u> indices{zero};
            auto *atomic = builder.atomic_exchange(
                Type::of<int64_t>(), base,
                luisa::span<Value *const>{indices.data(), indices.size()}, one);
            builder.return_void();

            expect(xir_verify_module(&module).succeeded());
            expect(lc::spirv::validate_spirv_xir_codegen_dialect(
                       &module)
                       .succeeded());
            auto unavailable = validate_atomic_target(
                module, lc::spirv::SpirvTargetFeatures{});
            expect(!unavailable.succeeded());
            expect(eq(unavailable.diagnostics.size(), 1u));
            if (!unavailable.diagnostics.empty()) {
                expect(unavailable.diagnostics.front().instruction ==
                       atomic);
                expect(unavailable.diagnostics.front().message.find(
                           shared_storage ?
                               "shaderSharedInt64Atomics" :
                               "shaderBufferInt64Atomics") !=
                       luisa::string::npos);
            }
            auto supported = lc::spirv::SpirvTargetFeatures{};
            supported.shader_shared_int64_atomics = shared_storage;
            supported.shader_buffer_int64_atomics = !shared_storage;
            expect(validate_atomic_target(module, supported)
                       .succeeded());
        }
    };

    return 0;
}
