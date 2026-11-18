// Host-only tests for required XIR call semantics at the Metal4 AIR boundary.
// Ordinary external declarations remain supported; native_include cannot
// replace a required typed intrinsic with an arbitrary same-name definition.

#include <array>

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/runtime/rtx/ray.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/metadata/contiguous_copy.h>
#include <luisa/xir/metadata/strided_mma.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

#include "llvm_codegen/metal_codegen_llvm.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void check_external_contract(bool copy, bool required, bool native_include, bool called) {
    xir::Module module;
    auto kernel = module.create_kernel();
    kernel->set_block_size(make_uint3(32u, 1u, 1u));
    auto external = module.create_external_function(nullptr);
    external->set_name("same_external_contract");
    xir::XIRBuilder builder;
    builder.set_insertion_point(kernel->create_body_block());
    auto array_type = Type::array(Type::of<float>(), 8u);
    luisa::vector<xir::Value *> arguments;
    if (copy) {
        auto buffer_type = Type::buffer(Type::of<float>());
        (void)external->create_resource_argument(buffer_type);
        (void)external->create_value_argument(Type::of<uint64_t>());
        (void)external->create_reference_argument(array_type);
        arguments.emplace_back(kernel->create_resource_argument(buffer_type));
        arguments.emplace_back(module.create_constant_zero(Type::of<uint64_t>()));
        arguments.emplace_back(builder.alloca_local(array_type));
        if (required) {
            external->create_metadata<xir::ContiguousCopyMD>()->descriptor = {5u, 4u};
        }
    } else {
        for (auto i = 0u; i < 4u; i++) {
            (void)external->create_reference_argument(array_type);
            arguments.emplace_back(builder.alloca_local(array_type));
        }
        if (required) {
            external->create_metadata<xir::StridedMmaMD>()->descriptor = {
                .output_extents = {3u},
                .lhs_output_strides = {0u},
                .rhs_output_strides = {1u},
                .contraction_extent = 1u,
                .lhs_contraction_stride = 1u,
                .rhs_contraction_stride = 3u,
                .vector_width = 4u};
        }
    }
    if (called) { builder.call(nullptr, external, arguments); }
    builder.return_void();
    auto verified = xir::xir_verify_module(&module);
    expect(verified.succeeded());
    if (!verified.succeeded()) { return; }

    metal::MetalCodegenLLVMConfig config;
    if (native_include) {
        // This is only a preflight input: no LLVM parsing, code generation,
        // linking or device execution occurs in this test. The corresponding
        // ordinary declaration uses exactly the same name and formal types.
        config.native_include = copy ?
                                    "define void @same_external_contract({ ptr addrspace(1), i64 } %source, i64 %offset, ptr %destination) { ret void }" :
                                    "define void @same_external_contract(ptr %lhs, ptr %rhs, ptr %seed, ptr %output) { ret void }";
    }
    luisa::string reason;
    auto supported = metal::luisa_compute_metal_codegen_llvm_supported(module, config, &reason);
    expect(supported == !required) << "copy=" << copy << ", required=" << required
                                   << ", native_include=" << native_include << ", called=" << called << ": " << reason;
    if (required) {
        expect(reason.find(copy ? "contiguous_copy" : "strided_mma") != luisa::string::npos) << reason;
        expect(reason.find("required") != luisa::string::npos) << reason;
    } else {
        expect(reason.empty()) << reason;
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "metal4_preflight_preserves_ordinary_external_boundary"_test = [] {
        for (auto copy : {false, true}) {
            for (auto native_include : {false, true}) {
                for (auto called : {false, true}) {
                    check_external_contract(copy, false, native_include, called);
                }
            }
        }
    };
    "metal4_preflight_rejects_required_native_call_semantics"_test = [] {
        for (auto copy : {false, true}) {
            for (auto native_include : {false, true}) {
                for (auto called : {false, true}) {
                    check_external_contract(copy, true, native_include, called);
                }
            }
        }
    };
    "metal4_preflight_accepts_procedural_and_mixed_ray_query_pipelines"_test = [] {
        xir::Module module;
        auto *kernel = module.create_kernel();
        kernel->set_block_size(make_uint3(32u, 1u, 1u));
        auto *accel_arg = kernel->create_resource_argument(Type::of<Accel>());
        auto *ray_arg = kernel->create_value_argument(Type::of<Ray>());
        auto *mask_arg = kernel->create_value_argument(Type::of<uint>());

        auto *surface_fn = module.create_callable(nullptr);
        auto *surf_q = surface_fn->create_reference_argument(Type::of<RayQueryAll>());
        auto *surf_capture = surface_fn->create_reference_argument(Type::of<float>());
        xir::XIRBuilder sb;
        sb.set_insertion_point(surface_fn->create_body_block());
        auto *tri_hit = sb.call(
            Type::of<TriangleHit>(),
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT,
            std::array<xir::Value *, 1u>{surf_q});
        auto *surf_obj_ray = sb.call(
            Type::of<Ray>(),
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY,
            std::array<xir::Value *, 1u>{surf_q});
        (void)tri_hit;
        (void)surf_obj_ray;
        sb.store(surf_capture, module.create_constant_one(Type::of<float>()));
        sb.call(
            xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE,
            std::array<xir::Value *, 1u>{surf_q});
        sb.return_void();

        auto *procedural_fn = module.create_callable(nullptr);
        auto *proc_q = procedural_fn->create_reference_argument(Type::of<RayQueryAll>());
        auto *proc_capture = procedural_fn->create_reference_argument(Type::of<float>());
        xir::XIRBuilder pb;
        pb.set_insertion_point(procedural_fn->create_body_block());
        (void)pb.call(
            Type::of<ProceduralHit>(),
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT,
            std::array<xir::Value *, 1u>{proc_q});
        (void)pb.call(
            Type::of<Ray>(),
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY,
            std::array<xir::Value *, 1u>{proc_q});
        (void)pb.call(
            Type::of<Ray>(),
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY,
            std::array<xir::Value *, 1u>{proc_q});
        auto *dist = pb.load(Type::of<float>(), proc_capture);
        pb.call(
            xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL,
            std::array<xir::Value *, 2u>{proc_q, dist});
        pb.return_void();

        xir::XIRBuilder kb;
        kb.set_insertion_point(kernel->create_body_block());
        auto *query_obj = kb.alloca_local(Type::of<RayQueryAll>());
        auto *capture_var = kb.alloca_local(Type::of<float>());
        kb.store(capture_var, module.create_constant_one(Type::of<float>()));
        auto *query_init = kb.call(
            Type::of<RayQueryAll>(),
            xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL,
            std::array<xir::Value *, 3u>{accel_arg, ray_arg, mask_arg});
        kb.store(query_obj, query_init);
        kb.ray_query_pipeline(
            query_obj, surface_fn, procedural_fn,
            std::array<xir::Value *, 1u>{capture_var});
        (void)kb.call(
            Type::of<CommittedHit>(),
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT,
            std::array<xir::Value *, 1u>{query_obj});
        kb.return_void();

        expect(xir::xir_verify_module(&module).succeeded());
        luisa::string reason;
        metal::MetalCodegenLLVMConfig config;
        expect(metal::luisa_compute_metal_codegen_llvm_supported(module, config, &reason))
            << reason;
        auto codegen = metal::luisa_compute_metal_codegen_llvm(module, config);
        expect(codegen.module != nullptr);
        expect(codegen.intersection_functions.size() == 2u);
        if (codegen.intersection_functions.size() == 2u) {
            expect(codegen.intersection_functions[0] == "luisa_ray_query_surface_0");
            expect(codegen.intersection_functions[1] == "luisa_ray_query_procedural_0");
        }
    };
    "metal4_preflight_rejects_cross_role_handler_commits"_test = [] {
        xir::Module module;
        auto *kernel = module.create_kernel();
        kernel->set_block_size(make_uint3(32u, 1u, 1u));
        auto *accel_arg = kernel->create_resource_argument(Type::of<Accel>());
        auto *ray_arg = kernel->create_value_argument(Type::of<Ray>());
        auto *mask_arg = kernel->create_value_argument(Type::of<uint>());

        auto *surface_fn = module.create_callable(nullptr);
        (void)surface_fn->create_reference_argument(Type::of<RayQueryAll>());
        xir::XIRBuilder sb;
        sb.set_insertion_point(surface_fn->create_body_block());
        sb.return_void();

        auto *procedural_fn = module.create_callable(nullptr);
        auto *proc_q = procedural_fn->create_reference_argument(Type::of<RayQueryAll>());
        xir::XIRBuilder pb;
        pb.set_insertion_point(procedural_fn->create_body_block());
        // Invalid: triangle commit inside procedural handler.
        pb.call(
            xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE,
            std::array<xir::Value *, 1u>{proc_q});
        pb.return_void();

        xir::XIRBuilder kb;
        kb.set_insertion_point(kernel->create_body_block());
        auto *query_obj = kb.alloca_local(Type::of<RayQueryAll>());
        auto *query_init = kb.call(
            Type::of<RayQueryAll>(),
            xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL,
            std::array<xir::Value *, 3u>{accel_arg, ray_arg, mask_arg});
        kb.store(query_obj, query_init);
        kb.ray_query_pipeline(query_obj, surface_fn, procedural_fn, {});
        kb.return_void();

        luisa::string reason;
        expect(!metal::luisa_compute_metal_codegen_llvm_supported(module, &reason));
        expect(reason.find("procedural intersection handler") != luisa::string::npos)
            << reason;
    };
    return 0;
}
