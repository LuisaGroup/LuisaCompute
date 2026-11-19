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

shared_ptr<const FB> make_cached_callable(bool cache, string_view comment = "snapshot") {
    xir::Module module;
    auto *leaf = module.create_callable(Type::of<int>());
    leaf->set_name("leaf");
    static_cast<void>(leaf->create_metadata<xir::NoInlineMD>());
    auto *x = leaf->create_value_argument(Type::of<int>());
    xir::XIRBuilder builder;
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
        auto same = make_cached_callable(true);
        auto different = make_cached_callable(true, "different preserved metadata");
        expect(ast->hash() == same->hash());
        expect(ast->hash() != different->hash());
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
        auto transformed = backend_detail::lower_autodiff_to_ast(kernel.function()->function());
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
}
