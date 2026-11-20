#include "ut/ut.hpp"
#include "xir_autodiff.h"
#include <atomic>
#include <thread>
#include <vector>
#include <luisa/luisa-compute.h>
#include <luisa/ast/function_builder.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/no_inline.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/lower_ray_query_to_pipeline.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;
using FB = luisa::compute::detail::FunctionBuilder;

namespace {

shared_ptr<const FB> make_cached_callable(bool cache, string_view comment = "snapshot",
                                          string_view leaf_comment = {}, bool unrelated = false) {
    xir::Module module;
    xir::XIRBuilder builder;
    if (unrelated) {
        auto *unused = module.create_callable(nullptr);
        builder.set_insertion_point(unused->create_body_block());
        builder.return_void();
    }
    auto *leaf = module.create_callable(Type::of<int>());
    leaf->set_name("leaf");
    if (!leaf_comment.empty()) { leaf->add_comment(leaf_comment); }
    static_cast<void>(leaf->create_metadata<xir::NoInlineMD>());
    auto *x = leaf->create_value_argument(Type::of<int>());
    builder.set_insertion_point(leaf->create_body_block());
    auto *sum = builder.call(Type::of<int>(), xir::ArithmeticOp::BINARY_ADD,
                             {x, module.create_constant_one(Type::of<int>())});
    builder.return_(sum);
    auto *root = module.create_callable(Type::of<int>());
    root->set_name("root");
    root->add_comment(comment);
    auto *arg = root->create_value_argument(Type::of<int>());
    builder.set_insertion_point(root->create_body_block());
    auto *call = builder.call(Type::of<int>(), leaf, {arg});
    builder.return_(call);
    return xir::xir_to_ast_translate(*root, {.cache_transformed_functions = cache});
}

}// namespace

int main(int argc, char *argv[]) {
    "only_opted_in_xir_to_ast_attaches_a_cache"_test = [] {
        Callable leaf = [](UInt value) { return value + 1u; };
        Kernel1D kernel = [&](BufferUInt output) { output.write(dispatch_x(), leaf(dispatch_x())); };
        auto ast = kernel.function()->function();
        auto lower = [](luisa::compute::Function f, size_t bodies, size_t imports) {
            xir::AST2XIRTranslationStatistics stats;
            auto ir = xir::ast_to_xir_translate(f, {.statistics = &stats});
            expect(stats.ast_function_translations == bodies);
            expect(stats.cached_function_imports == imports);
            expect(xir::xir_verify_module(ir.get()).succeeded());
            return ir;
        };

        auto module = lower(ast, 2u, 0u);
        const xir::KernelFunction *root = nullptr;
        for (auto *function : module->function_list()) {
            if (function->isa<xir::KernelFunction>()) {
                root = static_cast<const xir::KernelFunction *>(function);
            }
        }
        expect(root != nullptr) << fatal;
        auto plain = xir::xir_to_ast_translate(*root, {});
        auto retained = xir::xir_to_ast_translate(*root, {.cache_transformed_functions = true});
        lower(plain->function(), 2u, 0u);
        lower(retained->function(), 0u, 1u);
        lower(ast, 2u, 0u);
    };
    "cache_survives_source_and_imports_without_ast_lowering"_test = [] {
        auto ast = make_cached_callable(true);
        xir::AST2XIRTranslationStatistics statistics;
        auto *ctx = xir::ast_to_xir_translate_begin({.statistics = &statistics});
        auto *first = xir::ast_to_xir_translate_add_function(ctx, ast->function());
        auto *second = xir::ast_to_xir_translate_add_function(ctx, ast->function());
        expect(first == second);
        auto module = xir::ast_to_xir_translate_finalize(ctx);
        expect(statistics.cached_function_imports == 1u);
        expect(statistics.ast_function_translations == 0u);
        expect(statistics.function_reuses == 1u);
        expect(module->function_list().count_size() == 2u);
        expect(xir::xir_verify_module(module.get()).succeeded());
        auto original = xir::xir_to_flat_text_translate(module.get(), true);
        first->set_name("consumer_mutation");
        auto fresh = xir::ast_to_xir_translate(ast->function(), {});
        expect(xir::xir_to_flat_text_translate(fresh.get(), true) == original);
        expect(first != fresh->function_list().front());

        xir::AST2XIRTranslationStatistics oracle;
        auto lowered = xir::ast_to_xir_translate(ast->function(), {
            .use_transformed_function_cache = false, .statistics = &oracle});
        expect(oracle.cached_function_imports == 0u);
        expect(oracle.ast_function_translations == 2u);
        expect(xir::xir_verify_module(lowered.get()).succeeded());
        xir::AST2XIRTranslationStatistics legacy;
        auto legacy_module = xir::ast_to_xir_translate(ast->function(), {
            .preserve_inline_ray_query_loops = false, .statistics = &legacy});
        expect(legacy.cached_function_imports == 0u);
        expect(legacy.ast_function_translations == 2u);
        expect(xir::xir_verify_module(legacy_module.get()).succeeded());
        // Duplication can internalize/reorder arguments, so it deliberately
        // reconstructs AST without blindly propagating an old XIR snapshot.
        auto duplicate = ast->duplicate();
        xir::AST2XIRTranslationStatistics duplicated_stats;
        auto duplicated = xir::ast_to_xir_translate(duplicate->function(), {.statistics = &duplicated_stats});
        expect(duplicated_stats.cached_function_imports == 0u);
        expect(duplicated_stats.ast_function_translations == 2u);
        expect(xir::xir_verify_module(duplicated.get()).succeeded());
    };
    "cache_identity_and_parallel_consumers"_test = [] {
        auto ast = make_cached_callable(true);
        auto same = make_cached_callable(true, "snapshot", {}, true);
        auto different_leaf = make_cached_callable(true, "snapshot", "callee metadata");
        auto different = make_cached_callable(true, "different preserved metadata");
        expect(ast->hash() == same->hash());
        expect(ast->hash() != different->hash());
        expect(ast->hash() != different_leaf->hash());
        auto plain = make_cached_callable(false);
        expect(plain->hash() == make_cached_callable(false)->hash());
        expect(ast->hash() != plain->hash());
        std::atomic<unsigned> successes{0u};
        std::vector<std::thread> workers;
        for (auto i = 0u; i < 4u; i++) {
            workers.emplace_back([&] {
                xir::AST2XIRTranslationStatistics stats;
                auto module = xir::ast_to_xir_translate(ast->function(), {.statistics = &stats});
                if (stats.cached_function_imports == 1u && stats.ast_function_translations == 0u &&
                    xir::xir_verify_module(module.get()).succeeded()) { successes.fetch_add(1u); }
                module->function_list().front()->set_name("thread_local_mutation");
            });
        }
        for (auto &worker : workers) { worker.join(); }
        expect(successes.load() == 4u);
    };
    "fingerprint_includes_instruction_state_erased_by_xir_to_ast"_test = [] {
        xir::Module module;
        auto *function = module.create_callable(nullptr);
        xir::XIRBuilder builder;
        builder.set_insertion_point(function->create_body_block());
        auto *slot = builder.alloca_local(Type::of<uint>());
        builder.return_void();
        auto hash = [&](bool retain) {
            return xir::xir_to_ast_translate(*function, {.cache_transformed_functions = retain})->hash();
        };
        auto ast_hash = hash(false);
        auto retained_hash = hash(true);
        slot->set_coro_return_selector(7u);
        expect(hash(false) == ast_hash);
        expect(hash(true) != retained_hash);
    };
    "bound_argument_sorting_uses_ast_fallback"_test = [] {
        xir::Module module;
        auto *kernel = module.create_kernel();
        kernel->set_block_size({64u, 1u, 1u});
        static_cast<void>(kernel->create_value_argument(Type::of<int>()));
        static_cast<void>(kernel->create_resource_argument(Type::of<Buffer<int>>()));
        static_cast<void>(kernel->create_resource_argument(Type::of<Buffer<int>>()));
        xir::XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        builder.return_void();
        std::array<luisa::compute::Function::Binding, 3u> bindings{
            monostate{}, luisa::compute::Function::BufferBinding{123u, 0u, 64u},
            luisa::compute::Function::BufferBinding{123u, 0u, 64u}};
        auto ast = xir::xir_to_ast_translate(*kernel, {
            .bound_arguments = bindings, .cache_transformed_functions = true});
        auto oracle = xir::xir_to_ast_translate(*kernel, {.bound_arguments = bindings});
        expect(ast->hash() == oracle->hash());
        expect(ast->function().arguments().front().type()->is_buffer());
        expect(ast->function().arguments().size() == 2u);
        xir::AST2XIRTranslationStatistics stats;
        auto lowered = xir::ast_to_xir_translate(ast->function(), {.statistics = &stats});
        expect(stats.cached_function_imports == 0u);
        expect(stats.ast_function_translations == 1u);
        auto *function = lowered->function_list().front();
        expect(function->arguments().front()->type()->is_buffer());
        expect(xir::xir_verify_module(lowered.get()).succeeded());
    };
    "autodiff_ast_handoff_retains_transformed_xir"_test = [&] {
        Kernel1D kernel = [](BufferFloat output, Float input) {
            auto x = input;
            $autodiff {
                requires_grad(x);
                auto y = x * x;
                backward(y);
                output.write(0u, grad(x));
            };
        };
        auto ast_only = backend_detail::lower_autodiff_to_ast(kernel.function()->function());
        xir::AST2XIRTranslationStatistics ast_only_stats;
        auto ast_only_ir = xir::ast_to_xir_translate(ast_only->function(), {.statistics = &ast_only_stats});
        expect(ast_only_stats.cached_function_imports == 0u);
        expect(ast_only_stats.ast_function_translations == 1u);
        expect(xir::xir_verify_module(ast_only_ir.get()).succeeded());

        auto transformed = backend_detail::lower_autodiff_to_ast(kernel.function()->function(), true);
        expect(!transformed->function().requires_autodiff());
        xir::AST2XIRTranslationStatistics stats;
        auto module = xir::ast_to_xir_translate(transformed->function(), {.statistics = &stats});
        expect(stats.cached_function_imports == 1u);
        expect(stats.ast_function_translations == 0u);
        expect(xir::xir_verify_module(module.get()).succeeded());
        xir::AST2XIRTranslationStatistics oracle;
        auto lowered = xir::ast_to_xir_translate(transformed->function(), {
            .use_transformed_function_cache = false, .statistics = &oracle});
        expect(oracle.ast_function_translations == 1u);
        expect(xir::xir_verify_module(lowered.get()).succeeded());
        if (argc > 1) {
            Context context{argv[0]};
            auto device = context.create_device(argv[1]);
            auto stream = device.create_stream();
            auto output = device.create_buffer<float>(1u);
            Kernel1D<Buffer<float>, float> cached_kernel{transformed};
            Kernel1D<Buffer<float>, float> oracle_kernel{transformed->duplicate()};
            auto shader = device.compile(cached_kernel, {.enable_cache = false});
            auto oracle_shader = device.compile(oracle_kernel, {.enable_cache = false});
            for (auto input : {-3.f, -1.f, 0.f, 1.5f, 4.f}) {
                float cached_value = 0.f, oracle_value = 0.f;
                stream << shader(output, input).dispatch(1u)
                       << output.copy_to(&cached_value) << synchronize();
                stream << oracle_shader(output, input).dispatch(1u)
                       << output.copy_to(&oracle_value) << synchronize();
                expect(cached_value == 2.f * input);
                expect(cached_value == oracle_value);
            }
        }
    };
    "cached_struct_and_array_aggregate_and_diagnostic_inline_metadata"_test = [&] {
        // Exercises two real Coro XIR cache consumer contracts:
        // 1) Struct/array ArithmeticOp::AGGREGATE (including Ray-like and
        //    vector-bearing padded structs) preserved across cached XIR handoff.
        // 2) Diagnostic block/return/call metadata + NoInlineMD on a multi-block
        //    cached callable legalized by backends such as SIMD.
        auto *f32_ty = Type::of<float>();
        auto *u8_ty = Type::of<uint8_t>();
        auto *u32_ty = Type::of<uint>();
        auto *f2_ty = Type::of<float2>();
        auto *f3_ty = Type::of<float3>();
        auto *f4_ty = Type::of<float4>();
        auto *arr_f3_ty = Type::array(f32_ty, 3u);
        auto *arr_vec3_ty = Type::array(f3_ty, 2u);
        auto *ray_ty = Type::structure(16u, {arr_f3_ty, f32_ty, arr_f3_ty, f32_ty});
        auto *padded_ty = Type::structure(16u, {u8_ty, f3_ty, f2_ty, arr_vec3_ty});

        xir::Module xir_mod;
        xir::XIRBuilder b;

        auto *helper = xir_mod.create_callable(f4_ty);
        helper->set_name("cached_aggregate_helper");
        static_cast<void>(helper->create_metadata<xir::NoInlineMD>());
        auto *h_idx = helper->create_value_argument(u32_ty);
        auto *h_entry = helper->create_body_block();
        h_entry->set_name("helper_entry");
        h_entry->add_comment("diagnostic block metadata");
        auto *h_then = helper->create_basic_block();
        auto *h_else = helper->create_basic_block();
        auto *h_merge = helper->create_basic_block();
        h_merge->set_name("helper_merge");

        b.set_insertion_point(h_entry);
        auto *h_slot = b.alloca_local(f32_ty);
        auto make_const = [&]<typename T>(T val) noexcept {
            return xir_mod.create_constant(Type::of<T>(), &val);
        };
        auto *c0u = make_const(0u);
        auto *c1u = make_const(1u);
        auto *c2u = make_const(2u);
        auto *c3u = make_const(3u);
        auto *idx_f = b.static_cast_(f32_ty, h_idx);
        auto *bit = b.call(u32_ty, xir::ArithmeticOp::BINARY_BIT_AND, {h_idx, c1u});
        auto *is_odd = b.call(Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL, {bit, c0u});
        auto *if_inst = b.if_(is_odd);
        if_inst->set_true_target(h_then);
        if_inst->set_false_target(h_else);
        if_inst->set_merge_block(h_merge);

        b.set_insertion_point(h_then);
        b.store(h_slot, b.call(f32_ty, xir::ArithmeticOp::BINARY_ADD,
                               {idx_f, make_const(10.f)}));
        b.br(h_merge);

        b.set_insertion_point(h_else);
        b.store(h_slot, b.call(f32_ty, xir::ArithmeticOp::BINARY_ADD,
                               {idx_f, make_const(20.f)}));
        b.br(h_merge);

        b.set_insertion_point(h_merge);
        auto *v = b.load(f32_ty, h_slot);
        auto *v_plus1 = b.call(f32_ty, xir::ArithmeticOp::BINARY_ADD, {v, make_const(1.f)});
        auto *v_plus2 = b.call(f32_ty, xir::ArithmeticOp::BINARY_ADD, {v, make_const(2.f)});
        auto *orig = b.call(arr_f3_ty, xir::ArithmeticOp::AGGREGATE, {v, v_plus1, v_plus2});
        auto *dir = b.call(arr_f3_ty, xir::ArithmeticOp::AGGREGATE, {v_plus2, v, v_plus1});
        auto *ray = b.call(ray_ty, xir::ArithmeticOp::AGGREGATE,
                           {orig, make_const(0.25f), dir, v_plus2});
        auto *v2 = b.call(f2_ty, xir::ArithmeticOp::AGGREGATE, {v, v_plus1});
        auto *v3 = b.call(f3_ty, xir::ArithmeticOp::AGGREGATE, {v, v_plus1, v_plus2});
        auto *v3_arr = b.call(arr_vec3_ty, xir::ArithmeticOp::AGGREGATE, {v3, v3});
        auto *padded = b.call(padded_ty, xir::ArithmeticOp::AGGREGATE,
                              {make_const(uint8_t{7u}), v3, v2, v3_arr});
        auto *ray_orig = b.call(arr_f3_ty, xir::ArithmeticOp::EXTRACT, {ray, c0u});
        auto *r0 = b.call(f32_ty, xir::ArithmeticOp::EXTRACT, {ray_orig, c1u});
        auto *r1 = b.call(f32_ty, xir::ArithmeticOp::EXTRACT, {ray, c3u});
        auto *p_v2 = b.call(f2_ty, xir::ArithmeticOp::EXTRACT, {padded, c2u});
        auto *r2 = b.call(f32_ty, xir::ArithmeticOp::EXTRACT, {p_v2, c0u});
        auto *p_arr = b.call(arr_vec3_ty, xir::ArithmeticOp::EXTRACT, {padded, c3u});
        auto *p_arr1 = b.call(f3_ty, xir::ArithmeticOp::EXTRACT, {p_arr, c1u});
        auto *r3 = b.call(f32_ty, xir::ArithmeticOp::EXTRACT, {p_arr1, c2u});
        auto *out_f4 = b.call(f4_ty, xir::ArithmeticOp::AGGREGATE, {r0, r1, r2, r3});
        auto *ret = b.return_(out_f4);
        ret->add_comment("diagnostic return metadata");

        auto *kernel = xir_mod.create_kernel();
        kernel->set_block_size({64u, 1u, 1u});
        auto *out_buf = kernel->create_resource_argument(Type::of<Buffer<float4>>());
        auto *k_entry = kernel->create_body_block();
        k_entry->set_name("kernel_entry");
        k_entry->add_comment("caller block diagnostic metadata");
        b.set_insertion_point(k_entry);
        auto *did = xir_mod.create_dispatch_id();
        auto *tid = b.call(u32_ty, xir::ArithmeticOp::EXTRACT, {did, c0u});
        auto *res = b.call(f4_ty, helper, {tid});
        res->add_comment("call site diagnostic metadata");
        b.call(xir::ResourceWriteOp::BUFFER_WRITE, {out_buf, tid, res});
        b.return_void();

        expect(xir::xir_verify_module(&xir_mod).succeeded()) << fatal;
        auto retained = xir::xir_to_ast_translate(*kernel, {.cache_transformed_functions = true});
        xir::AST2XIRTranslationStatistics stats;
        auto imported = xir::ast_to_xir_translate(retained->function(), {.statistics = &stats});
        expect(stats.cached_function_imports == 1u);
        expect(stats.ast_function_translations == 0u);
        expect(xir::xir_verify_module(imported.get()).succeeded());

        if (argc > 1) {
            Context context{argv[0]};
            auto device = context.create_device(argv[1]);
            auto stream = device.create_stream();
            constexpr uint count = 16u;
            auto output = device.create_buffer<float4>(count);
            Kernel1D<Buffer<float4>> cached_kernel{retained};
            Kernel1D<Buffer<float4>> oracle_kernel{retained->duplicate()};
            auto cached_shader = device.compile(cached_kernel, {.enable_cache = false});
            auto oracle_shader = device.compile(oracle_kernel, {.enable_cache = false});
            std::vector<float4> cached_host(count), oracle_host(count);
            stream << cached_shader(output).dispatch(count)
                   << output.copy_to(cached_host.data()) << synchronize();
            stream << oracle_shader(output).dispatch(count)
                   << output.copy_to(oracle_host.data()) << synchronize();
            for (auto i = 0u; i < count; i++) {
                auto base = static_cast<float>(i) + ((i & 1u) != 0u ? 10.f : 20.f);
                expect(cached_host[i].x == base + 1.f);
                expect(cached_host[i].y == base + 2.f);
                expect(cached_host[i].z == base);
                expect(cached_host[i].w == base + 2.f);
                expect(cached_host[i].x == oracle_host[i].x);
                expect(cached_host[i].y == oracle_host[i].y);
                expect(cached_host[i].z == oracle_host[i].z);
                expect(cached_host[i].w == oracle_host[i].w);
            }
        }
    };
    "xir2ast_cache_declines_ray_query_pipeline"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferFloat out) {
            auto hit_t = def(100.f);
            auto hit = accel.traverse(make_ray(make_float3(0.f), make_float3(0.f, 0.f, 1.f), 0.f, 100.f), {})
                           .on_surface_candidate([&](SurfaceCandidate &cand) {
                               auto h = cand.hit();
                               $if(h.committed_ray_t < hit_t) {
                                   hit_t = h.committed_ray_t;
                                   cand.commit();
                               };
                           })
                           .trace();
            out.write(dispatch_x(), ite(hit->miss(), hit_t, hit.committed_ray_t));
        };
        auto ast = kernel.function()->function();

        auto find_kernel = [](xir::Module *m) -> const xir::KernelFunction * {
            for (auto *f : m->function_list()) {
                if (f->isa<xir::KernelFunction>()) {
                    return static_cast<const xir::KernelFunction *>(f);
                }
            }
            return nullptr;
        };
        auto count_tags = [](xir::Module *m, xir::DerivedInstructionTag tag) -> size_t {
            size_t count = 0u;
            for (auto *f : m->function_list()) {
                if (auto *def = f->definition()) {
                    def->traverse_instructions([&](xir::Instruction *inst) noexcept {
                        if (inst->derived_instruction_tag() == tag) { count++; }
                    });
                }
            }
            return count;
        };

        // Pre-split RayQueryPipelineInst (produced by lower_ray_query_to_pipeline,
        // e.g. inside coroutine lowering) must NOT be cached across xir_to_ast -> ast_to_xir
        // so consumers with default preserve_inline_ray_query_loops=true see canonical RayQueryLoopInst.
        auto pipeline_mod = xir::ast_to_xir_translate(ast, {});
        expect(xir::xir_verify_module(pipeline_mod.get()).succeeded()) << fatal;
        expect(xir::lower_ray_query_to_pipeline_pass_run_on_module(pipeline_mod.get()).succeeded()) << fatal;
        expect(xir::xir_verify_module(pipeline_mod.get()).succeeded()) << fatal;
        expect(count_tags(pipeline_mod.get(), xir::DerivedInstructionTag::RAY_QUERY_PIPELINE) > 0u);
        auto *pipeline_root = find_kernel(pipeline_mod.get());
        expect(pipeline_root != nullptr) << fatal;
        auto retained_pipeline = xir::xir_to_ast_translate(*pipeline_root, {.cache_transformed_functions = true});

        xir::AST2XIRTranslationStatistics pipeline_stats;
        auto re_lowered = xir::ast_to_xir_translate(retained_pipeline->function(), {.statistics = &pipeline_stats});
        expect(xir::xir_verify_module(re_lowered.get()).succeeded());
        // The root kernel containing RayQueryPipelineInst must re-lower from AST into a
        // canonical RayQueryLoopInst, while its pure callback callees remain cacheable.
        expect(pipeline_stats.ast_function_translations == 1u);
        expect(count_tags(re_lowered.get(), xir::DerivedInstructionTag::RAY_QUERY_PIPELINE) == 0u);
        expect(count_tags(re_lowered.get(), xir::DerivedInstructionTag::RAY_QUERY_LOOP) > 0u);
    };
}
