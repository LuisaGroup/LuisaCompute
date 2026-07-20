// Test for the Vulkan XIR-to-SPIR-V ray primitive-culling contract.
// This test covers:
// - SkipAABBsKHR flags emitted by direct surface-only tracing
// - the corresponding RayTraversalPrimitiveCullingKHR capability
// - capability absence for generic ray-query traversal without culling flags

#include "ut/ut.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <spirv-tools/libspirv.hpp>

#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/verifier.h>

#include "spirv_codegen/entry.h"
#include "spirv_codegen/utils.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct RayQuerySpirvFacts {
    size_t primitive_culling_capability_count{0u};
    size_t proceed_count{0u};
    size_t loop_merge_count{0u};
    std::vector<uint32_t> initialize_flags;
};

[[nodiscard]] RayQuerySpirvFacts inspect_ray_query_spirv(
    const std::vector<uint32_t> &words) {
    RayQuerySpirvFacts facts;
    if (words.size() < 5u) { return facts; }

    struct Constant {
        uint32_t id;
        uint32_t value;
    };
    std::vector<Constant> constants;
    std::vector<uint32_t> flag_ids;
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto instruction = words[offset];
        auto word_count = static_cast<size_t>(instruction >> 16u);
        auto opcode = static_cast<spv::Op>(instruction & 0xffffu);
        if (word_count == 0u || offset + word_count > words.size()) {
            break;
        }
        if (opcode == spv::Op::OpCapability && word_count >= 2u &&
            words[offset + 1u] ==
                static_cast<uint32_t>(
                    spv::Capability::RayTraversalPrimitiveCullingKHR)) {
            facts.primitive_culling_capability_count++;
        } else if (opcode == spv::Op::OpRayQueryProceedKHR) {
            facts.proceed_count++;
        } else if (opcode == spv::Op::OpLoopMerge) {
            facts.loop_merge_count++;
        } else if (opcode == spv::Op::OpConstant && word_count == 4u) {
            constants.emplace_back(Constant{
                .id = words[offset + 2u],
                .value = words[offset + 3u]});
        } else if (opcode == spv::Op::OpRayQueryInitializeKHR &&
                   word_count >= 9u) {
            // OpRayQueryInitializeKHR operands are query, acceleration
            // structure, ray flags, cull mask, origin, t-min, direction,
            // and t-max. Ray flags are therefore the third operand.
            flag_ids.emplace_back(words[offset + 3u]);
        }
        offset += word_count;
    }
    for (auto flag_id : flag_ids) {
        auto iter = std::find_if(
            constants.cbegin(), constants.cend(),
            [flag_id](auto constant) noexcept {
                return constant.id == flag_id;
            });
        if (iter != constants.cend()) {
            facts.initialize_flags.emplace_back(iter->value);
        }
    }
    std::sort(facts.initialize_flags.begin(),
              facts.initialize_flags.end());
    return facts;
}

template<typename Kernel>
[[nodiscard]] std::vector<uint32_t> compile_spirv(
    Kernel &&kernel) {
    auto result = lc::spirv::SpirvCodegenEntry::compile_spirv(
        kernel.function()->function(),
        ShaderOption{.enable_cache = false},
        {.ray_query = true});
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    expect(tools.Validate(result.spv_bin.data(), result.spv_bin.size()))
        << "ray-query SPIR-V fixture must validate";
    return std::move(result.spv_bin);
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_direct_surface_trace_declares_primitive_culling"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto ray = make_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f));
            auto closest = accel.intersect(ray, {});
            auto any = accel.intersect_any(ray, {});
            output.write(0u, closest->inst);
            output.write(1u, ite(any, 1u, 0u));
        };
        auto facts = inspect_ray_query_spirv(compile_spirv(kernel));
        expect(facts.primitive_culling_capability_count == 1u)
            << "SkipAABBsKHR requires exactly one primitive-culling capability";
        constexpr auto closest_flags =
            spv::RayFlagsMask::OpaqueKHR |
            spv::RayFlagsMask::SkipAABBsKHR;
        constexpr auto any_flags =
            closest_flags |
            spv::RayFlagsMask::TerminateOnFirstHitKHR |
            spv::RayFlagsMask::SkipClosestHitShaderKHR;
        const std::vector<uint32_t> expected{
            static_cast<uint32_t>(closest_flags),
            static_cast<uint32_t>(any_flags)};
        expect(facts.initialize_flags == expected)
            << "direct closest/any tracing must preserve the surface-only ray flags";
        expect(facts.proceed_count == 2u)
            << "direct closest and any tracing each require one static proceed instruction";
        expect(facts.loop_merge_count == 1u)
            << "direct closest tracing must loop until ray-query traversal completes, "
               "while terminate-on-first-hit any tracing remains single-step";
    };

    "spirv_generic_ray_query_omits_primitive_culling"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto ray = make_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f));
            auto hit = accel.traverse(ray, {})
                           .on_surface_candidate(
                               [](SurfaceCandidate &candidate) noexcept {
                                   candidate.commit();
                               })
                           .on_procedural_candidate(
                               [](ProceduralCandidate &) noexcept {})
                           .trace();
            output.write(0u, hit->hit_type);
        };
        auto facts = inspect_ray_query_spirv(compile_spirv(kernel));
        expect(facts.primitive_culling_capability_count == 0u)
            << "generic traversal has no primitive-culling ray flag";
        const std::vector<uint32_t> expected{
            static_cast<uint32_t>(spv::RayFlagsMask::MaskNone)};
        expect(facts.initialize_flags == expected);
    };

    "spirv_true_orphan_ray_query_payload_is_not_emitted"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto ray = make_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f));
            output.write(
                0u, ite(accel.intersect_any(ray, {}), 1u, 0u));
        };
        auto ast_function = kernel.function()->function();
        auto module =
            luisa::compute::spirv::luisa_spirv_backend_translate_ast_to_xir(
                ast_function, ShaderOption{.enable_cache = false});
        xir::KernelFunction *xir_kernel = nullptr;
        for (auto *function : module->function_list()) {
            if (function->derived_function_tag() ==
                xir::DerivedFunctionTag::KERNEL) {
                xir_kernel = static_cast<xir::KernelFunction *>(function);
                break;
            }
        }
        expect(xir_kernel != nullptr);
        if (xir_kernel == nullptr) { return; }

        // This is a generically valid but backend-unsupported ray-query
        // materialization if executable. As a true orphan it has no physical
        // SPIR-V block, so backend dialect/lifetime analysis must not make
        // direct codegen depend on optional DCE. Generic verification remains
        // whole-module and therefore still has to accept the fixture itself.
        auto *orphan = xir_kernel->create_basic_block();
        xir::XIRBuilder builder;
        builder.set_insertion_point(orphan);
        auto *query_type = Type::custom("LC_RayQueryAny");
        auto *query = builder.alloca_local(query_type);
        builder.store(query, module->create_undefined(query_type));
        builder.return_void();
        expect(xir::xir_verify_module(module.get()).succeeded());

        auto result = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            ast_function, module.get(),
            ShaderOption{.enable_cache = false},
            {.ray_query = true});
        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        expect(tools.Validate(
            result.spv_bin.data(), result.spv_bin.size()));
    };
}
