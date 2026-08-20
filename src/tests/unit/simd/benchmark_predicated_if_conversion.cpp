#include "llvm_jit.h"
#include "llvm_schedule_codegen.h"
#include "predicated_if_conversion.h"
#include "xir_to_schedule.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

using namespace luisa::compute;
using namespace luisa::compute::simd;

namespace {

enum struct Shape : uint8_t {
    select_only,
    factorable_alu,
};

enum struct Variant : uint8_t {
    scheduled,
    predicated,
};

struct Scenario {
    std::string_view name;
    uint32_t threshold;
    uint32_t active_lanes;
};

struct EntryMetadata {
    std::string name;
    Shape shape{};
    Variant variant{};
    uint32_t width{0u};
    size_t schedule_blocks{0u};
    size_t convergence_points{0u};
    size_t converted_diamonds{0u};
    size_t factored_selects{0u};
    size_t argument_buffer_size{0u};
};

struct LLVMModuleBundle {
    std::unique_ptr<::llvm::LLVMContext> context;
    std::unique_ptr<::llvm::Module> module;
    std::vector<EntryMetadata> entries;
    std::string error;
};

struct AssemblyStats {
    size_t instructions{0u};
    size_t stack_references{0u};
    size_t calls{0u};
    std::string_view widest_register{"scalar"};
};

struct Measurement {
    double scheduled_ns{0.0};
    double predicated_ns{0.0};
};

[[nodiscard]] constexpr std::string_view shape_name(
    Shape shape) noexcept {
    switch (shape) {
        case Shape::select_only: return "select_only";
        case Shape::factorable_alu: return "factorable_alu";
    }
    return {};
}

[[nodiscard]] constexpr std::string_view variant_name(
    Variant variant) noexcept {
    switch (variant) {
        case Variant::scheduled: return "scheduled";
        case Variant::predicated: return "predicated";
    }
    return {};
}

[[nodiscard]] std::string entry_name(
    Shape shape, Variant variant, uint32_t width) {
    return "benchmark_" + std::string{shape_name(shape)} + "_" +
           std::string{variant_name(variant)} + "_w" +
           std::to_string(width);
}

[[nodiscard]] std::string diagnostics_text(
    const schedule::XIRToScheduleResult &result) {
    std::string text;
    for (auto &&diagnostic : result.diagnostics) {
        text += schedule::to_string(diagnostic.code);
        text += ": ";
        text += diagnostic.message;
        text += '\n';
    }
    return text;
}

[[nodiscard]] std::optional<schedule::Function> make_schedule(
    Shape shape, Variant variant, uint32_t width,
    EntryMetadata &metadata, std::string &error) {
    xir::Module module;
    auto *function = module.create_kernel();
    function->set_name(entry_name(shape, variant, width));
    auto *threshold =
        function->create_value_argument(Type::of<uint32_t>());
    auto *entry = function->create_body_block();
    auto *true_block = function->create_basic_block();
    auto *false_block = function->create_basic_block();
    auto *merge = function->create_basic_block();
    entry->set_name("entry");
    merge->set_name("merge");

    auto *lane = module.create_warp_lane_id();
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t three_value = 3u;
    uint32_t five_value = 5u;
    uint32_t false_value = 17u;
    uint32_t true_value = 31u;
    auto *three = module.create_constant(
        Type::of<uint32_t>(), &three_value);
    auto *five = module.create_constant(
        Type::of<uint32_t>(), &five_value);
    auto *false_constant = module.create_constant(
        Type::of<uint32_t>(), &false_value);
    auto *true_constant = module.create_constant(
        Type::of<uint32_t>(), &true_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {lane, threshold});
    builder.cond_br(condition, true_block, false_block);

    builder.set_insertion_point(true_block);
    xir::Value *true_result = true_constant;
    if (shape == Shape::factorable_alu) {
        auto *product = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_MUL,
            {lane, three});
        true_result = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
            {product, one});
    }
    builder.br(merge);

    builder.set_insertion_point(false_block);
    xir::Value *false_result = false_constant;
    if (shape == Shape::factorable_alu) {
        auto *product = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_MUL,
            {lane, five});
        false_result = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
            {product, one});
    }
    builder.br(merge);

    builder.set_insertion_point(merge);
    auto *result = builder.phi(
        Type::of<uint32_t>(),
        {{true_result, true_block},
         {false_result, false_block}});
    result->set_name("result");
    builder.return_void();

    if (!xir::xir_verify_module(&module).succeeded()) {
        error = "input XIR verification failed";
        return std::nullopt;
    }
    auto predication = schedule::PredicatedIfConversionInfo{};
    if (variant == Variant::predicated) {
        predication =
            schedule::predicate_small_varying_diamonds(function);
        if (!predication.if_conversion.changed()) {
            error = "predication policy rejected " +
                    entry_name(shape, variant, width) +
                    " benchmark diamond";
            return std::nullopt;
        }
    }
    if (!xir::xir_verify_module(&module).succeeded()) {
        error = "rewritten XIR verification failed";
        return std::nullopt;
    }
    auto lowered = schedule::lower_xir_to_schedule(
        function, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        error = diagnostics_text(lowered);
        return std::nullopt;
    }
    auto result_id = std::optional<schedule::ValueId>{};
    for (auto &&value : lowered.function->values()) {
        if (value.name == "result") { result_id = value.id; }
    }
    for (auto &block : lowered.function->blocks()) {
        if (block.name == "merge") {
            block.terminator = schedule::ReturnTerminator{result_id};
        }
    }
    if (!result_id || !schedule::verify(*lowered.function).succeeded()) {
        error = "benchmark Schedule IR result rewrite failed";
        return std::nullopt;
    }
    metadata = {
        .name = entry_name(shape, variant, width),
        .shape = shape,
        .variant = variant,
        .width = width,
        .schedule_blocks = lowered.function->blocks().size(),
        .convergence_points =
            lowered.function->convergence_points().size(),
        .converted_diamonds = predication.if_conversion
                                  .converted_diamond_count,
        .factored_selects = predication.select_factoring
                                .factored_select_count,
    };
    return std::move(*lowered.function);
}

[[nodiscard]] LLVMModuleBundle make_llvm_module() {
    LLVMModuleBundle bundle;
    bundle.context = std::make_unique<::llvm::LLVMContext>();
    bundle.module = std::make_unique<::llvm::Module>(
        "predicated-if-conversion-benchmark", *bundle.context);
    constexpr std::array widths{2u, 4u, 8u, 16u};
    constexpr std::array shapes{
        Shape::select_only, Shape::factorable_alu};
    constexpr std::array variants{
        Variant::scheduled, Variant::predicated};
    for (auto width : widths) {
        for (auto shape : shapes) {
            for (auto variant : variants) {
                EntryMetadata metadata;
                auto schedule_function = make_schedule(
                    shape, variant, width, metadata, bundle.error);
                if (!schedule_function) { return bundle; }
                auto codegen = lower_schedule_to_llvm(
                    *bundle.module, *schedule_function, width,
                    metadata.name, false, {width, 1u, 1u});
                if (!codegen.succeeded()) {
                    bundle.error = codegen.error;
                    return bundle;
                }
                metadata.argument_buffer_size =
                    codegen.argument_buffer_size;
                bundle.entries.emplace_back(std::move(metadata));
            }
        }
    }
    if (::llvm::verifyModule(*bundle.module, &::llvm::errs())) {
        bundle.error = "LLVM verification failed";
    }
    return bundle;
}

[[nodiscard]] std::string_view function_assembly(
    std::string_view assembly, std::string_view function) {
    auto label = std::string{"\n"} + std::string{function} + ":";
    auto begin = assembly.find(label);
    if (begin == std::string_view::npos) { return {}; }
    begin += label.size();
    auto end = assembly.find("\n.Lfunc_end", begin);
    if (end == std::string_view::npos) {
        end = assembly.find("\n\t.size", begin);
    }
    if (end == std::string_view::npos) { end = assembly.size(); }
    return assembly.substr(begin, end - begin);
}

[[nodiscard]] AssemblyStats assembly_stats(
    std::string_view assembly, std::string_view function) {
    auto body = function_assembly(assembly, function);
    AssemblyStats stats;
    if (body.find("%zmm") != std::string_view::npos) {
        stats.widest_register = "zmm";
    } else if (body.find("%ymm") != std::string_view::npos) {
        stats.widest_register = "ymm";
    } else if (body.find("%xmm") != std::string_view::npos) {
        stats.widest_register = "xmm";
    }
    for (auto begin = size_t{0u}; begin < body.size();) {
        auto end = body.find('\n', begin);
        if (end == std::string_view::npos) { end = body.size(); }
        auto line = body.substr(begin, end - begin);
        auto first = line.find_first_not_of(" \t");
        if (first != std::string_view::npos &&
            line[first] != '.' && line[first] != '#' &&
            line.back() != ':') {
            stats.instructions++;
            auto instruction = line.substr(first);
            stats.calls += instruction.starts_with("call");
            stats.stack_references +=
                instruction.find("%rsp") != std::string_view::npos ||
                instruction.find("%rbp") != std::string_view::npos;
        }
        begin = end + 1u;
    }
    return stats;
}

[[nodiscard]] const EntryMetadata *find_metadata(
    const std::vector<EntryMetadata> &entries,
    Shape shape, Variant variant, uint32_t width) noexcept {
    auto iter = std::ranges::find_if(
        entries, [&](auto &&entry) noexcept {
            return entry.shape == shape && entry.variant == variant &&
                   entry.width == width;
        });
    return iter == entries.end() ? nullptr : &*iter;
}

[[nodiscard]] std::array<Scenario, 5u> scenarios(
    uint32_t width) noexcept {
    return {{{"all_false", 0u, width},
             {"all_true", width, width},
             {"sparse", 1u, width},
             {"balanced", std::max(1u, width / 2u), width},
             {"tail", std::max(1u, (width - 1u) / 2u),
              width - 1u}}};
}

struct alignas(16) ArgumentBuffer {
    uint32_t threshold{0u};
    std::array<std::byte, 12u> padding{};
};

using Entry = void(
    const void *, uint32_t *, const SIMDPacketLaunchConfig *, uint32_t);

volatile uint32_t benchmark_sink = 0u;

[[nodiscard]] uint32_t expected_value(
    Shape shape, uint32_t lane, uint32_t threshold) noexcept {
    auto take_true = lane < threshold;
    if (shape == Shape::select_only) {
        return take_true ? 31u : 17u;
    }
    return lane * (take_true ? 3u : 5u) + 1u;
}

[[nodiscard]] bool check_entry(
    Entry *entry, Shape shape, uint32_t width,
    const Scenario &scenario) {
    ArgumentBuffer arguments{.threshold = scenario.threshold};
    SIMDPacketLaunchConfig config{};
    config.dispatch_size[0u] = scenario.active_lanes;
    config.dispatch_size[1u] = 1u;
    config.dispatch_size[2u] = 1u;
    config.block_size[0u] = width;
    config.block_size[1u] = 1u;
    config.block_size[2u] = 1u;
    std::array<uint32_t, 16u> output{};
    output.fill(0xdeadbeefu);
    entry(
        &arguments, output.data(), &config,
        scenario.active_lanes);
    for (auto lane = uint32_t{0u}; lane < width; lane++) {
        auto expected = lane < scenario.active_lanes ?
                            expected_value(
                                shape, lane,
                                scenario.threshold) :
                            0xdeadbeefu;
        if (output[lane] != expected) {
            std::cerr << "incorrect " << shape_name(shape)
                      << " W" << width << ' ' << scenario.name
                      << " lane " << lane << ": " << output[lane]
                      << " != " << expected << '\n';
            return false;
        }
    }
    return true;
}

[[nodiscard]] double measure(
    Entry *entry, uint64_t repetitions,
    const ArgumentBuffer &arguments,
    const SIMDPacketLaunchConfig &config,
    uint32_t active_lanes,
    std::array<uint32_t, 16u> &output) {
    auto begin = std::chrono::steady_clock::now();
    for (auto i = uint64_t{0u}; i < repetitions; i++) {
        entry(&arguments, output.data(), &config, active_lanes);
    }
    auto end = std::chrono::steady_clock::now();
    benchmark_sink = benchmark_sink ^
                     output[repetitions % active_lanes];
    return std::chrono::duration<double, std::nano>{end - begin}
               .count() /
           static_cast<double>(repetitions);
}

[[nodiscard]] double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2u];
}

[[nodiscard]] Measurement benchmark_pair(
    Entry *scheduled, Entry *predicated,
    uint32_t width, const Scenario &scenario,
    uint32_t trials, double target_sample_ns) {
    ArgumentBuffer arguments{.threshold = scenario.threshold};
    SIMDPacketLaunchConfig config{};
    config.dispatch_size[0u] = scenario.active_lanes;
    config.dispatch_size[1u] = 1u;
    config.dispatch_size[2u] = 1u;
    config.block_size[0u] = width;
    config.block_size[1u] = 1u;
    config.block_size[2u] = 1u;
    std::array<uint32_t, 16u> output{};
    auto repetitions = uint64_t{1024u};
    while (measure(
               scheduled, repetitions, arguments, config,
               scenario.active_lanes, output) *
               static_cast<double>(repetitions) <
           target_sample_ns) {
        repetitions *= 2u;
    }
    std::vector<double> scheduled_samples;
    std::vector<double> predicated_samples;
    scheduled_samples.reserve(trials);
    predicated_samples.reserve(trials);
    for (auto trial = uint32_t{0u}; trial < trials; trial++) {
        if ((trial & 1u) == 0u) {
            scheduled_samples.emplace_back(measure(
                scheduled, repetitions, arguments, config,
                scenario.active_lanes, output));
            predicated_samples.emplace_back(measure(
                predicated, repetitions, arguments, config,
                scenario.active_lanes, output));
        } else {
            predicated_samples.emplace_back(measure(
                predicated, repetitions, arguments, config,
                scenario.active_lanes, output));
            scheduled_samples.emplace_back(measure(
                scheduled, repetitions, arguments, config,
                scenario.active_lanes, output));
        }
    }
    return {
        .scheduled_ns = median(std::move(scheduled_samples)),
        .predicated_ns = median(std::move(predicated_samples)),
    };
}

template<typename T>
[[nodiscard]] bool parse_integer(
    std::string_view text, T &value) noexcept {
    auto *begin = text.data();
    auto *end = begin + text.size();
    auto result = std::from_chars(begin, end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

}// namespace

int main(int argc, char *argv[]) {
    auto profile = argc >= 2 &&
                   std::string_view{argv[1]} == "--profile";
    auto mode = argc == 2 ? std::string_view{argv[1]} :
                            std::string_view{};
    auto quick = mode == "--quick";
    auto dump_assembly = mode == "--assembly";
    auto profile_variant = Variant::scheduled;
    auto profile_width = uint32_t{0u};
    auto profile_repetitions = uint64_t{0u};
    auto profile_valid = false;
    if (profile && (argc == 4 || argc == 5)) {
        auto variant = std::string_view{argv[2]};
        profile_valid =
            (variant == "scheduled" || variant == "predicated") &&
            parse_integer(std::string_view{argv[3]}, profile_width) &&
            (profile_width == 2u || profile_width == 4u ||
             profile_width == 8u || profile_width == 16u);
        profile_variant = variant == "predicated" ?
                              Variant::predicated :
                              Variant::scheduled;
        if (argc == 5) {
            profile_valid = profile_valid &&
                            parse_integer(
                                std::string_view{argv[4]},
                                profile_repetitions) &&
                            profile_repetitions != 0u;
        } else {
            profile_repetitions = 100000000u;
        }
    }
    if ((!profile && argc > 2) ||
        (!profile && !mode.empty() && !quick && !dump_assembly) ||
        (profile && !profile_valid)) {
        std::cerr << "usage: " << argv[0]
                  << " [--quick|--assembly|--profile "
                     "scheduled|predicated 2|4|8|16 [repetitions]]\n";
        return 1;
    }
    auto trials = quick ? 5u : 9u;
    auto target_sample_ns = quick ? 5.0e6 : 2.0e7;

    auto assembly_module = make_llvm_module();
    if (!assembly_module.error.empty()) {
        std::cerr << assembly_module.error << '\n';
        return 1;
    }
    LLVMJIT assembly_target;
    if (!assembly_target.succeeded()) {
        std::cerr << assembly_target.error() << '\n';
        return 1;
    }
    auto metadata = std::move(assembly_module.entries);
    auto assembly = assembly_target.emit_assembly(
        std::move(assembly_module.module),
        std::move(assembly_module.context));
    if (assembly.empty()) {
        std::cerr << assembly_target.error() << '\n';
        return 1;
    }
    if (dump_assembly) {
        std::cout << assembly;
        return 0;
    }

    auto executable = make_llvm_module();
    if (!executable.error.empty()) {
        std::cerr << executable.error << '\n';
        return 1;
    }
    LLVMJIT jit;
    if (!jit.succeeded() ||
        !jit.add_module(
            std::move(executable.module),
            std::move(executable.context))) {
        std::cerr << jit.error() << '\n';
        return 1;
    }

    if (profile) {
        auto *entry_metadata = find_metadata(
            metadata, Shape::factorable_alu,
            profile_variant, profile_width);
        auto *entry = entry_metadata == nullptr ?
                          nullptr :
                          reinterpret_cast<Entry *>(
                              jit.lookup(entry_metadata->name));
        auto scenario = scenarios(profile_width)[3u];
        if (entry == nullptr ||
            !check_entry(
                entry, Shape::factorable_alu,
                profile_width, scenario)) {
            std::cerr << "profile entry lookup/check failed\n";
            return 1;
        }
        ArgumentBuffer arguments{.threshold = scenario.threshold};
        SIMDPacketLaunchConfig config{};
        config.dispatch_size[0u] = scenario.active_lanes;
        config.dispatch_size[1u] = 1u;
        config.dispatch_size[2u] = 1u;
        config.block_size[0u] = profile_width;
        config.block_size[1u] = 1u;
        config.block_size[2u] = 1u;
        std::array<uint32_t, 16u> output{};
        // Allows `perf stat --delay 500` to exclude LLVM/JIT startup while
        // still observing the complete steady-state loop.
        std::this_thread::sleep_for(std::chrono::seconds{1});
        auto ns = measure(
            entry, profile_repetitions, arguments, config,
            scenario.active_lanes, output);
        std::cout << variant_name(profile_variant) << " W"
                  << profile_width << " factorable_alu balanced: "
                  << std::fixed << std::setprecision(3) << ns
                  << " ns/call over " << profile_repetitions
                  << " calls\n";
        return 0;
    }

    constexpr std::array widths{2u, 4u, 8u, 16u};
    constexpr std::array shapes{
        Shape::select_only, Shape::factorable_alu};
    std::cout
        << "shape,width,scenario,active_lanes,scheduled_ns,"
           "predicated_ns,speedup,delta_ns,scheduled_instructions,"
           "predicated_instructions,scheduled_stack_refs,"
           "predicated_stack_refs,scheduled_calls,predicated_calls,"
           "scheduled_register,"
           "predicated_register\n";
    auto stable = true;
    for (auto width : widths) {
        for (auto shape : shapes) {
            auto *scheduled_metadata = find_metadata(
                metadata, shape, Variant::scheduled, width);
            auto *predicated_metadata = find_metadata(
                metadata, shape, Variant::predicated, width);
            if (scheduled_metadata == nullptr ||
                predicated_metadata == nullptr ||
                scheduled_metadata->argument_buffer_size != 16u ||
                predicated_metadata->argument_buffer_size != 16u ||
                scheduled_metadata->schedule_blocks != 4u ||
                predicated_metadata->schedule_blocks != 2u ||
                scheduled_metadata->convergence_points != 1u ||
                predicated_metadata->convergence_points != 0u ||
                predicated_metadata->converted_diamonds != 1u ||
                predicated_metadata->factored_selects !=
                    (shape == Shape::factorable_alu ? 2u : 0u)) {
                std::cerr << "unexpected lowering metadata for "
                          << shape_name(shape) << " W" << width
                          << '\n';
                return 1;
            }
            auto *scheduled = reinterpret_cast<Entry *>(
                jit.lookup(scheduled_metadata->name));
            auto *predicated = reinterpret_cast<Entry *>(
                jit.lookup(predicated_metadata->name));
            if (scheduled == nullptr || predicated == nullptr) {
                std::cerr << jit.error() << '\n';
                return 1;
            }
            auto scheduled_assembly = assembly_stats(
                assembly, scheduled_metadata->name);
            auto predicated_assembly = assembly_stats(
                assembly, predicated_metadata->name);
            if (scheduled_assembly.instructions <=
                    predicated_assembly.instructions ||
                scheduled_assembly.calls != 0u ||
                predicated_assembly.calls != 0u) {
                std::cerr << "unexpected native shape for "
                          << shape_name(shape) << " W" << width
                          << '\n';
                return 1;
            }
            std::cerr
                << shape_name(shape) << " W" << width << ": "
                << scheduled_metadata->schedule_blocks << " -> "
                << predicated_metadata->schedule_blocks
                << " Schedule blocks, "
                << scheduled_metadata->convergence_points << " -> "
                << predicated_metadata->convergence_points
                << " convergence points, "
                << scheduled_assembly.instructions << " -> "
                << predicated_assembly.instructions
                << " native instructions\n";
            for (auto &&scenario : scenarios(width)) {
                if (!check_entry(
                        scheduled, shape, width, scenario) ||
                    !check_entry(
                        predicated, shape, width, scenario)) {
                    return 1;
                }
                auto measurement = benchmark_pair(
                    scheduled, predicated, width, scenario,
                    trials, target_sample_ns);
                auto speedup = measurement.scheduled_ns /
                               measurement.predicated_ns;
                std::cout
                    << shape_name(shape) << ',' << width << ','
                    << scenario.name << ',' << scenario.active_lanes
                    << ',' << std::fixed << std::setprecision(3)
                    << measurement.scheduled_ns << ','
                    << measurement.predicated_ns << ',' << speedup
                    << ','
                    << measurement.scheduled_ns -
                           measurement.predicated_ns
                    << ',' << scheduled_assembly.instructions << ','
                    << predicated_assembly.instructions << ','
                    << scheduled_assembly.stack_references << ','
                    << predicated_assembly.stack_references << ','
                    << scheduled_assembly.calls << ','
                    << predicated_assembly.calls << ','
                    << scheduled_assembly.widest_register << ','
                    << predicated_assembly.widest_register << '\n';
                stable = stable && std::isfinite(speedup) &&
                         speedup >= 1.05;
            }
        }
    }
    static_cast<void>(benchmark_sink);
    return stable ? 0 : 1;
}
