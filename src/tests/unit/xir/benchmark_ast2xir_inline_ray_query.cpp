#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/dsl/sugar.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/reconstruct_ray_query_loop.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;

namespace {

struct ModuleShape {
    size_t blocks{0u};
    size_t instructions{0u};
};

struct TranslationResult {
    luisa::unique_ptr<Module> module;
    ModuleShape before_normalization;
    ModuleShape after_normalization;
};

struct TranslationWork {
    bool normalize{false};
    bool verify{false};
};

[[nodiscard]] ModuleShape module_shape(const Module *module) noexcept {
    ModuleShape shape;
    for (auto function : module->function_list()) {
        if (auto definition = function->definition()) {
            definition->traverse_basic_blocks(
                [&](const BasicBlock *block) noexcept {
                    shape.blocks++;
                    for ([[maybe_unused]] auto instruction :
                         block->instructions()) {
                        shape.instructions++;
                    }
                });
        }
    }
    return shape;
}

[[nodiscard]] TranslationResult translate_and_normalize(
    const luisa::compute::Function &function, bool preserve,
    TranslationWork work) {
    auto module = ast_to_xir_translate(
        function,
        {.preserve_inline_ray_query_loops = preserve});
    if (module == nullptr ||
        (work.verify &&
         !xir_verify_module(module.get()).succeeded())) {
        std::cerr << "AST-to-XIR translation failed.\n";
        std::exit(2);
    }
    auto before = module_shape(module.get());
    if (work.normalize) {
        auto info = reconstruct_ray_query_loop_pass_run_on_module(
            module.get());
        if (!info.succeeded() ||
            (work.verify &&
             !xir_verify_module(module.get()).succeeded())) {
            std::cerr << "Ray-query normalization failed.\n";
            std::exit(2);
        }
    }
    auto after = module_shape(module.get());
    return {.module = std::move(module),
            .before_normalization = before,
            .after_normalization = after};
}

[[nodiscard]] double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    auto middle = values.size() / 2u;
    return values.size() % 2u == 0u ?
               (values[middle - 1u] + values[middle]) * 0.5 :
               values[middle];
}

struct BatchResult {
    double microseconds_per_translation{0.0};
    uint64_t checksum{0u};
};

[[nodiscard]] BatchResult run_batch(
    const luisa::compute::Function &function, bool preserve,
    TranslationWork work, size_t iterations) {
    auto begin = std::chrono::steady_clock::now();
    uint64_t checksum = 0u;
    for (auto i = 0u; i < iterations; ++i) {
        auto result = translate_and_normalize(
            function, preserve, work);
        checksum += result.before_normalization.blocks * 17u +
                    result.before_normalization.instructions * 31u +
                    result.after_normalization.blocks * 43u +
                    result.after_normalization.instructions * 59u;
    }
    auto elapsed = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - begin);
    return {.microseconds_per_translation =
                elapsed.count() / static_cast<double>(iterations),
            .checksum = checksum};
}

struct ComparisonResult {
    double direct_median_us{0.0};
    double legacy_median_us{0.0};
    uint64_t checksum{0u};
};

[[nodiscard]] ComparisonResult compare(
    const luisa::compute::Function &function,
    TranslationWork work, size_t iterations,
    size_t rounds) {
    static_cast<void>(run_batch(function, true, work, 3u));
    static_cast<void>(run_batch(function, false, work, 3u));
    std::vector<double> direct_samples;
    std::vector<double> legacy_samples;
    direct_samples.reserve(rounds);
    legacy_samples.reserve(rounds);
    uint64_t checksum = 0u;
    for (auto round = 0u; round < rounds; ++round) {
        if ((round & 1u) == 0u) {
            auto direct = run_batch(
                function, true, work, iterations);
            auto legacy = run_batch(
                function, false, work, iterations);
            direct_samples.emplace_back(
                direct.microseconds_per_translation);
            legacy_samples.emplace_back(
                legacy.microseconds_per_translation);
            checksum += direct.checksum + legacy.checksum;
        } else {
            auto legacy = run_batch(
                function, false, work, iterations);
            auto direct = run_batch(
                function, true, work, iterations);
            direct_samples.emplace_back(
                direct.microseconds_per_translation);
            legacy_samples.emplace_back(
                legacy.microseconds_per_translation);
            checksum += direct.checksum + legacy.checksum;
        }
    }
    return {.direct_median_us = median(std::move(direct_samples)),
            .legacy_median_us = median(std::move(legacy_samples)),
            .checksum = checksum};
}

void print_comparison(
    std::string_view name,
    const ComparisonResult &result) {
    std::cout << name << "_direct_median_us="
              << result.direct_median_us << '\n';
    std::cout << name << "_legacy_median_us="
              << result.legacy_median_us << '\n';
    std::cout << name << "_speedup="
              << result.legacy_median_us /
                     result.direct_median_us
              << "x\n";
}

}// namespace

int main(int argc, char **argv) {
    auto iterations = size_t{100u};
    auto rounds = size_t{9u};
    for (auto i = 1; i < argc; ++i) {
        auto argument = std::string_view{argv[i]};
        constexpr auto iterations_prefix =
            std::string_view{"--iterations="};
        constexpr auto rounds_prefix = std::string_view{"--rounds="};
        if (argument.starts_with(iterations_prefix)) {
            iterations = std::stoul(std::string{
                argument.substr(iterations_prefix.size())});
        } else if (argument.starts_with(rounds_prefix)) {
            rounds = std::stoul(
                std::string{argument.substr(rounds_prefix.size())});
        }
    }
    if (iterations == 0u || rounds == 0u) {
        std::cerr << "iterations and rounds must be non-zero.\n";
        return 2;
    }

    Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
        auto index = dispatch_x();
        auto ray = make_ray(
            make_float3(cast<float>(index), 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 1.0f), 0.0f, 100.0f);
        UInt result = 0u;
        for (auto query_index = 0u; query_index < 4u; ++query_index) {
            auto query = accel.query(ray, {});
            $while (query.proceed()) {
                $if (query.is_surface_candidate()) {
                    auto candidate = query.surface_candidate();
                    auto hit = candidate.hit();
                    result += hit->prim + query_index;
                    $if ((hit->prim & 1u) == 0u) {
                        candidate.commit();
                    };
                }
                $else {
                    auto candidate = query.procedural_candidate();
                    result += candidate.hit()->prim + query_index + 1u;
                    candidate.commit(1.0f);
                };
            };
            result += query.committed_hit()->prim;
        }
        output.write(index, result);
    };
    auto &&function = kernel.function()->function();

    Kernel1D ordinary_kernel = [](
                                   BufferUInt input,
                                   BufferUInt output) noexcept {
        auto index = dispatch_x();
        auto value = input.read(index);
        UInt result = 0u;
        for (auto loop_index = 0u; loop_index < 4u;
             ++loop_index) {
            UInt iteration = 0u;
            $while (iteration < ((value >> loop_index) & 7u)) {
                $if (((value + iteration) & 1u) == 0u) {
                    result += value ^ iteration;
                }
                $else {
                    result += value + iteration;
                };
                iteration += 1u;
            };
        }
        output.write(index, result);
    };
    auto &&ordinary_function =
        ordinary_kernel.function()->function();

    constexpr TranslationWork checked_normalization{
        .normalize = true, .verify = true};
    auto direct = translate_and_normalize(
        function, true, checked_normalization);
    auto legacy = translate_and_normalize(
        function, false, checked_normalization);
    if (xir_to_text_translate(direct.module.get(), false) !=
        xir_to_text_translate(legacy.module.get(), false)) {
        std::cerr << "Normalized direct and legacy XIR differ.\n";
        return 2;
    }

    auto translation = compare(
        function, {}, iterations, rounds);
    auto normalization = compare(
        function, {.normalize = true}, iterations, rounds);
    auto verified_normalization = compare(
        function, checked_normalization, iterations, rounds);
    auto ordinary_direct = translate_and_normalize(
        ordinary_function, true, {});
    auto ordinary_legacy = translate_and_normalize(
        ordinary_function, false, {});
    if (xir_to_text_translate(ordinary_direct.module.get(), false) !=
        xir_to_text_translate(ordinary_legacy.module.get(), false)) {
        std::cerr << "Ordinary-loop XIR differs with preservation enabled.\n";
        return 2;
    }
    auto ordinary_translation = compare(
        ordinary_function, {}, iterations, rounds);
    std::cout << "queries=4 iterations=" << iterations
              << " rounds=" << rounds << '\n';
    std::cout << "direct_shape="
              << direct.before_normalization.blocks << " blocks, "
              << direct.before_normalization.instructions
              << " instructions\n";
    std::cout << "legacy_shape="
              << legacy.before_normalization.blocks << " blocks, "
              << legacy.before_normalization.instructions
              << " instructions\n";
    std::cout << "normalized_shape="
              << direct.after_normalization.blocks << " blocks, "
              << direct.after_normalization.instructions
              << " instructions\n";
    print_comparison("translation", translation);
    print_comparison("normalization", normalization);
    print_comparison(
        "verified_normalization", verified_normalization);
    std::cout << "ordinary_shape="
              << ordinary_direct.before_normalization.blocks
              << " blocks, "
              << ordinary_direct.before_normalization.instructions
              << " instructions\n";
    print_comparison(
        "ordinary_translation", ordinary_translation);
    std::cout << "checksum="
              << translation.checksum + normalization.checksum +
                     verified_normalization.checksum +
                     ordinary_translation.checksum
              << '\n';
    return 0;
}
