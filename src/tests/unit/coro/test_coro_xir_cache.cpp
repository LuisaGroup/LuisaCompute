#include "ut/ut.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] const xir::KernelFunction *find_kernel(const xir::Module &module) noexcept {
    for (auto *function : module.function_list()) {
        if (function->isa<xir::KernelFunction>()) {
            return static_cast<const xir::KernelFunction *>(function);
        }
    }
    return nullptr;
}

}// namespace

int main(int argc, char *argv[]) {
    "coro_pipeline_imports_continuations_into_fresh_scheduler_xir"_test = [&] {
        // Construction runs the actual coroutine compiler, including splitting,
        // materializing and caching its transformed continuations.
        Coroutine<void(Buffer<uint>)> coroutine{[](BufferUInt output) {
            UInt value = dispatch_x() ^ 0x3d6fu;
            $suspend("cache_first");
            value += 19u;
            output.write(dispatch_x(), value);
            $suspend("cache_second");
            output.write(dispatch_x(), value * 3u);
        }};
        expect(coroutine.subroutine_count() == 3_u);

        Callable ordinary = [](UInt x) { return x ^ 0x5123u; };
        // Assemble the same entry/resume call patterns used by the scheduler,
        // without creating a Device or compiling a backend shader.
        Kernel1D scheduler = [&](BufferUInt output) {
            auto frame = coroutine.instantiate(dispatch_id(), dispatch_size());
            coroutine.entry()(frame, output);
            coroutine.subroutine("cache_first")(frame, output);
            coroutine.subroutine("cache_second")(frame, output);
            output.write(dispatch_size().x + dispatch_x(), ordinary(dispatch_x()));
        };

        xir::AST2XIRTranslationStatistics cached_stats;
        auto cached = xir::ast_to_xir_translate(scheduler.function()->function(), {.statistics = &cached_stats});
        expect(cached_stats.ast_function_translations == 5_u)
            << "the scheduler root, ordinary callable, and three wrappers should lower from AST";
        expect(cached_stats.cached_function_imports == 3_u);
        expect(xir::xir_verify_module(cached.get()).succeeded());

        xir::AST2XIRTranslationStatistics uncached_stats;
        auto uncached = xir::ast_to_xir_translate(
            scheduler.function()->function(), {.use_transformed_function_cache = false, .statistics = &uncached_stats});
        expect(uncached_stats.cached_function_imports == 0_u);
        expect(uncached_stats.ast_function_translations == cached_stats.ast_function_translations + cached_stats.cached_function_imports);
        expect(xir::xir_verify_module(uncached.get()).succeeded());

        auto *cached_kernel = find_kernel(*cached);
        expect(cached_kernel != nullptr);
        for (auto *definition : cached->function_list()) {
            if (definition == cached_kernel) { continue; }
            definition->set_name("mutation_only_in_first_module");
        }
        expect(xir::xir_verify_module(cached.get()).succeeded());

        xir::AST2XIRTranslationStatistics second_stats;
        auto second = xir::ast_to_xir_translate(scheduler.function()->function(), {.statistics = &second_stats});
        expect(second_stats.ast_function_translations == cached_stats.ast_function_translations);
        expect(second_stats.cached_function_imports == 3_u);
        expect(xir::xir_verify_module(second.get()).succeeded());

        auto *second_kernel = find_kernel(*second);
        expect(second_kernel != nullptr);
        expect(cached_kernel != second_kernel);
        for (auto *definition : second->function_list()) {
            expect(definition->name() != "mutation_only_in_first_module");
        }

        // Optional device oracle: duplicate reconstructs an uncached AST DAG,
        // so the two shaders exercise cache import versus ordinary lowering.
        // Disable shader disk caching to ensure both compiler paths execute.
        if (argc > 1) {
            Kernel1D<Buffer<uint>> oracle{scheduler.function()->duplicate()};
            xir::AST2XIRTranslationStatistics duplicate_stats;
            auto duplicate_ir = xir::ast_to_xir_translate(
                oracle.function()->function(), {.statistics = &duplicate_stats});
            expect(duplicate_stats.cached_function_imports == 0_u);
            expect(duplicate_stats.ast_function_translations == 8_u);
            Context context{argv[0]};
            auto device = context.create_device(argv[1]);
            auto stream = device.create_stream();
            constexpr uint count = 32u;
            auto output = device.create_buffer<uint>(2u * count);
            std::array<uint, 2u * count> cached_values{}, oracle_values{};
            auto shader = device.compile(scheduler, {.enable_cache = false});
            auto oracle_shader = device.compile(oracle, {.enable_cache = false});
            stream << shader(output).dispatch(count)
                   << output.copy_to(cached_values.data()) << synchronize();
            stream << oracle_shader(output).dispatch(count)
                   << output.copy_to(oracle_values.data()) << synchronize();
            for (uint i = 0u; i < count; i++) {
                expect(cached_values[i] == ((i ^ 0x3d6fu) + 19u) * 3u);
                expect(cached_values[count + i] == (i ^ 0x5123u));
            }
            expect(cached_values == oracle_values);
        }

        if (std::getenv("LUISA_XIR_CACHE_BENCHMARK") != nullptr) {
            constexpr size_t iterations = 30u;
            for (bool use_cache : {true, false}) {
                xir::AST2XIRTranslationStatistics totals;
                const auto begin = std::chrono::steady_clock::now();
                for (size_t i = 0u; i < iterations; ++i) {
                    xir::AST2XIRTranslationStatistics stats;
                    auto module = xir::ast_to_xir_translate(
                        scheduler.function()->function(),
                        {.use_transformed_function_cache = use_cache, .statistics = &stats});
                    totals.ast_function_translations += stats.ast_function_translations;
                    totals.cached_function_imports += stats.cached_function_imports;
                    totals.function_reuses += stats.function_reuses;
                }
                const auto elapsed = std::chrono::duration<double, std::milli>{
                    std::chrono::steady_clock::now() - begin}.count();
                std::printf("AST2XIR cache=%s independent_modules=%zu total_ms=%.3f "
                            "ast_translations=%zu cache_imports=%zu function_reuses=%zu\n",
                            use_cache ? "on" : "off", iterations, elapsed,
                            totals.ast_function_translations, totals.cached_function_imports,
                            totals.function_reuses);
            }
        }
    };
}
