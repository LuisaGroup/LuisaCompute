#include "llvm_jit.h"
#include "llvm_schedule_codegen.h"
#include "loop_unswitch.h"
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
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

using namespace luisa::compute;
using namespace luisa::compute::simd;

namespace {

static constexpr auto loop_iterations = 32u;

enum struct Variant : uint8_t {
    scheduled,
    unswitched,
};

struct Scenario {
    std::string_view name;
    uint32_t threshold;
    uint32_t active_lanes;
};

struct EntryMetadata {
    std::string name;
    Variant variant{};
    uint32_t width{0u};
    size_t schedule_blocks{0u};
    size_t convergence_points{0u};
    size_t unswitched_loops{0u};
    size_t cloned_blocks{0u};
    size_t cloned_instructions{0u};
    size_t merged_live_outs{0u};
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
    double unswitched_ns{0.0};
};

[[nodiscard]] constexpr std::string_view variant_name(
    Variant variant) noexcept {
    switch (variant) {
        case Variant::scheduled: return "scheduled";
        case Variant::unswitched: return "unswitched";
    }
    return {};
}

[[nodiscard]] std::string entry_name(
    Variant variant, uint32_t width) {
    return "benchmark_loop_" + std::string{variant_name(variant)} +
           "_w" + std::to_string(width);
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

[[nodiscard]] xir::Value *binary(
    xir::XIRBuilder &builder, xir::ArithmeticOp op,
    xir::Value *lhs, xir::Value *rhs) noexcept {
    return builder.call(Type::of<uint32_t>(), op, {lhs, rhs});
}

[[nodiscard]] std::optional<schedule::Function> make_schedule(
    Variant variant, uint32_t width,
    EntryMetadata &metadata, std::string &error) {
    xir::Module module;
    auto *function = module.create_kernel();
    function->set_name(entry_name(variant, width));
    auto *threshold =
        function->create_value_argument(Type::of<uint32_t>());
    auto *preheader = function->create_body_block();
    auto *header = function->create_basic_block();
    auto *body = function->create_basic_block();
    auto *true_block = function->create_basic_block();
    auto *false_block = function->create_basic_block();
    auto *latch = function->create_basic_block();
    auto *exit = function->create_basic_block();
    preheader->set_name("preheader");
    header->set_name("header");
    body->set_name("body");
    true_block->set_name("true_arm");
    false_block->set_name("false_arm");
    latch->set_name("latch");
    exit->set_name("exit");

    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    auto make_constant = [&](uint32_t value) noexcept {
        return module.create_constant(Type::of<uint32_t>(), &value);
    };
    auto *three = make_constant(3u);
    auto *five = make_constant(5u);
    auto *nine = make_constant(9u);
    auto *seventeen = make_constant(17u);
    auto *iterations = make_constant(loop_iterations);

    xir::XIRBuilder builder;
    builder.set_insertion_point(preheader);
    auto *selector = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {lane, threshold});
    auto *initial = binary(
        builder, xir::ArithmeticOp::BINARY_ADD, lane, one);
    builder.br(header);

    builder.set_insertion_point(header);
    auto *index = builder.phi(Type::of<uint32_t>());
    auto *accumulator = builder.phi(Type::of<uint32_t>());
    auto *continue_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {index, iterations});
    builder.cond_br(continue_condition, body, exit);

    builder.set_insertion_point(body);
    builder.cond_br(selector, true_block, false_block);

    builder.set_insertion_point(true_block);
    auto *true_product = binary(
        builder, xir::ArithmeticOp::BINARY_MUL,
        accumulator, three);
    auto *true_sum = binary(
        builder, xir::ArithmeticOp::BINARY_ADD,
        true_product, lane);
    auto *true_key = binary(
        builder, xir::ArithmeticOp::BINARY_ADD, index, nine);
    auto *true_result = binary(
        builder, xir::ArithmeticOp::BINARY_BIT_XOR,
        true_sum, true_key);
    builder.br(latch);

    builder.set_insertion_point(false_block);
    auto *false_product = binary(
        builder, xir::ArithmeticOp::BINARY_MUL,
        accumulator, five);
    auto *false_sum = binary(
        builder, xir::ArithmeticOp::BINARY_ADD,
        false_product, lane);
    auto *false_key = binary(
        builder, xir::ArithmeticOp::BINARY_ADD, index, seventeen);
    auto *false_result = binary(
        builder, xir::ArithmeticOp::BINARY_BIT_XOR,
        false_sum, false_key);
    builder.br(latch);

    builder.set_insertion_point(latch);
    auto *next_accumulator = builder.phi(
        Type::of<uint32_t>(),
        {{true_result, true_block},
         {false_result, false_block}});
    auto *next_index = binary(
        builder, xir::ArithmeticOp::BINARY_ADD, index, one);
    builder.br(header);
    index->add_incoming(zero, preheader);
    index->add_incoming(next_index, latch);
    accumulator->add_incoming(initial, preheader);
    accumulator->add_incoming(next_accumulator, latch);

    builder.set_insertion_point(exit);
    auto *result = binary(
        builder, xir::ArithmeticOp::BINARY_ADD,
        accumulator, zero);
    result->set_name("result");
    builder.return_void();

    if (!xir::xir_verify_module(&module).succeeded()) {
        error = "input XIR verification failed";
        return std::nullopt;
    }
    auto unswitch = schedule::SIMDLoopUnswitchInfo{};
    if (variant == Variant::unswitched) {
        unswitch =
            schedule::unswitch_invariant_varying_loop_condition(
                function);
        if (unswitch.unswitch.unswitched_loop_count != 1u) {
            error = "SIMD policy rejected loop-unswitch benchmark";
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
        if (block.name == "exit") {
            block.terminator = schedule::ReturnTerminator{result_id};
        }
    }
    if (!result_id || !schedule::verify(*lowered.function).succeeded()) {
        error = "benchmark Schedule IR result rewrite failed";
        return std::nullopt;
    }
    metadata = {
        .name = entry_name(variant, width),
        .variant = variant,
        .width = width,
        .schedule_blocks = lowered.function->blocks().size(),
        .convergence_points =
            lowered.function->convergence_points().size(),
        .unswitched_loops =
            unswitch.unswitch.unswitched_loop_count,
        .cloned_blocks = unswitch.unswitch.cloned_block_count,
        .cloned_instructions =
            unswitch.unswitch.cloned_instruction_count,
        .merged_live_outs =
            unswitch.unswitch.merged_live_out_count,
    };
    return std::move(*lowered.function);
}

[[nodiscard]] LLVMModuleBundle make_llvm_module() {
    LLVMModuleBundle bundle;
    bundle.context = std::make_unique<::llvm::LLVMContext>();
    bundle.module = std::make_unique<::llvm::Module>(
        "loop-unswitch-benchmark", *bundle.context);
    constexpr std::array widths{2u, 4u, 8u, 16u};
    constexpr std::array variants{
        Variant::scheduled, Variant::unswitched};
    for (auto width : widths) {
        for (auto variant : variants) {
            EntryMetadata metadata;
            auto schedule_function = make_schedule(
                variant, width, metadata, bundle.error);
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
    Variant variant, uint32_t width) noexcept {
    auto iter = std::ranges::find_if(
        entries, [&](auto &&entry) noexcept {
            return entry.variant == variant && entry.width == width;
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
    uint32_t lane, uint32_t threshold) noexcept {
    auto value = lane + 1u;
    for (auto index = 0u; index < loop_iterations; index++) {
        if (lane < threshold) {
            value = (value * 3u + lane) ^ (index + 9u);
        } else {
            value = (value * 5u + lane) ^ (index + 17u);
        }
    }
    return value;
}

[[nodiscard]] bool check_entry(
    Entry *entry, uint32_t width, const Scenario &scenario) {
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
    entry(&arguments, output.data(), &config, scenario.active_lanes);
    for (auto lane = uint32_t{0u}; lane < width; lane++) {
        auto expected = lane < scenario.active_lanes ?
                            expected_value(lane, scenario.threshold) :
                            0xdeadbeefu;
        if (output[lane] != expected) {
            std::cerr << "incorrect W" << width << ' '
                      << scenario.name << " lane " << lane << ": "
                      << output[lane] << " != " << expected << '\n';
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
    Entry *scheduled, Entry *unswitched,
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
    std::vector<double> unswitched_samples;
    scheduled_samples.reserve(trials);
    unswitched_samples.reserve(trials);
    for (auto trial = uint32_t{0u}; trial < trials; trial++) {
        if ((trial & 1u) == 0u) {
            scheduled_samples.emplace_back(measure(
                scheduled, repetitions, arguments, config,
                scenario.active_lanes, output));
            unswitched_samples.emplace_back(measure(
                unswitched, repetitions, arguments, config,
                scenario.active_lanes, output));
        } else {
            unswitched_samples.emplace_back(measure(
                unswitched, repetitions, arguments, config,
                scenario.active_lanes, output));
            scheduled_samples.emplace_back(measure(
                scheduled, repetitions, arguments, config,
                scenario.active_lanes, output));
        }
    }
    return {
        .scheduled_ns = median(std::move(scheduled_samples)),
        .unswitched_ns = median(std::move(unswitched_samples)),
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
            (variant == "scheduled" || variant == "unswitched") &&
            parse_integer(std::string_view{argv[3]}, profile_width) &&
            (profile_width == 2u || profile_width == 4u ||
             profile_width == 8u || profile_width == 16u);
        profile_variant = variant == "unswitched" ?
                              Variant::unswitched :
                              Variant::scheduled;
        if (argc == 5) {
            profile_valid = profile_valid &&
                            parse_integer(
                                std::string_view{argv[4]},
                                profile_repetitions) &&
                            profile_repetitions != 0u;
        } else {
            profile_repetitions = 10000000u;
        }
    }
    if ((!profile && argc > 2) ||
        (!profile && !mode.empty() && !quick && !dump_assembly) ||
        (profile && !profile_valid)) {
        std::cerr << "usage: " << argv[0]
                  << " [--quick|--assembly|--profile "
                     "scheduled|unswitched 2|4|8|16 [repetitions]]\n";
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
            metadata, profile_variant, profile_width);
        auto *entry = entry_metadata == nullptr ?
                          nullptr :
                          reinterpret_cast<Entry *>(
                              jit.lookup(entry_metadata->name));
        auto scenario = scenarios(profile_width)[3u];
        if (entry == nullptr ||
            !check_entry(entry, profile_width, scenario)) {
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
        std::this_thread::sleep_for(std::chrono::seconds{1});
        auto ns = measure(
            entry, profile_repetitions, arguments, config,
            scenario.active_lanes, output);
        std::cout << variant_name(profile_variant) << " W"
                  << profile_width << " balanced: " << std::fixed
                  << std::setprecision(3) << ns << " ns/call over "
                  << profile_repetitions << " calls\n";
        return 0;
    }

    constexpr std::array widths{2u, 4u, 8u, 16u};
    std::cout
        << "width,scenario,active_lanes,scheduled_ns,unswitched_ns,"
           "speedup,delta_ns,scheduled_instructions,"
           "unswitched_instructions,scheduled_stack_refs,"
           "unswitched_stack_refs,scheduled_calls,unswitched_calls,"
           "scheduled_register,unswitched_register\n";
    auto stable = true;
    for (auto width : widths) {
        auto *scheduled_metadata = find_metadata(
            metadata, Variant::scheduled, width);
        auto *unswitched_metadata = find_metadata(
            metadata, Variant::unswitched, width);
        if (scheduled_metadata == nullptr ||
            unswitched_metadata == nullptr ||
            scheduled_metadata->argument_buffer_size != 16u ||
            unswitched_metadata->argument_buffer_size != 16u ||
            scheduled_metadata->unswitched_loops != 0u ||
            unswitched_metadata->unswitched_loops != 1u ||
            unswitched_metadata->cloned_blocks == 0u ||
            unswitched_metadata->cloned_instructions == 0u ||
            unswitched_metadata->merged_live_outs == 0u) {
            std::cerr << "unexpected lowering metadata for W"
                      << width << ": scheduled_blocks="
                      << (scheduled_metadata == nullptr ?
                              0u :
                              scheduled_metadata->schedule_blocks)
                      << ", unswitched_blocks="
                      << (unswitched_metadata == nullptr ?
                              0u :
                              unswitched_metadata->schedule_blocks)
                      << ", scheduled_convergence="
                      << (scheduled_metadata == nullptr ?
                              0u :
                              scheduled_metadata->convergence_points)
                      << ", unswitched_convergence="
                      << (unswitched_metadata == nullptr ?
                              0u :
                              unswitched_metadata->convergence_points)
                      << ", loops="
                      << (unswitched_metadata == nullptr ?
                              0u :
                              unswitched_metadata->unswitched_loops)
                      << ", cloned_blocks="
                      << (unswitched_metadata == nullptr ?
                              0u :
                              unswitched_metadata->cloned_blocks)
                      << ", cloned_instructions="
                      << (unswitched_metadata == nullptr ?
                              0u :
                              unswitched_metadata->cloned_instructions)
                      << ", live_outs="
                      << (unswitched_metadata == nullptr ?
                              0u :
                              unswitched_metadata->merged_live_outs)
                      << '\n';
            return 1;
        }
        auto *scheduled = reinterpret_cast<Entry *>(
            jit.lookup(scheduled_metadata->name));
        auto *unswitched = reinterpret_cast<Entry *>(
            jit.lookup(unswitched_metadata->name));
        if (scheduled == nullptr || unswitched == nullptr) {
            std::cerr << jit.error() << '\n';
            return 1;
        }
        auto scheduled_assembly = assembly_stats(
            assembly, scheduled_metadata->name);
        auto unswitched_assembly = assembly_stats(
            assembly, unswitched_metadata->name);
        if (scheduled_assembly.calls != 0u ||
            unswitched_assembly.calls != 0u) {
            std::cerr << "unexpected native call in W" << width
                      << " loop benchmark\n";
            return 1;
        }
        std::cerr << "W" << width << ": "
                  << scheduled_metadata->schedule_blocks << " -> "
                  << unswitched_metadata->schedule_blocks
                  << " Schedule blocks, "
                  << scheduled_metadata->convergence_points << " -> "
                  << unswitched_metadata->convergence_points
                  << " convergence points, "
                  << scheduled_assembly.instructions << " -> "
                  << unswitched_assembly.instructions
                  << " native instructions\n";
        for (auto &&scenario : scenarios(width)) {
            if (!check_entry(scheduled, width, scenario) ||
                !check_entry(unswitched, width, scenario)) {
                return 1;
            }
            auto measurement = benchmark_pair(
                scheduled, unswitched, width, scenario,
                trials, target_sample_ns);
            auto speedup = measurement.scheduled_ns /
                           measurement.unswitched_ns;
            std::cout
                << width << ',' << scenario.name << ','
                << scenario.active_lanes << ',' << std::fixed
                << std::setprecision(3)
                << measurement.scheduled_ns << ','
                << measurement.unswitched_ns << ',' << speedup << ','
                << measurement.scheduled_ns -
                       measurement.unswitched_ns
                << ',' << scheduled_assembly.instructions << ','
                << unswitched_assembly.instructions << ','
                << scheduled_assembly.stack_references << ','
                << unswitched_assembly.stack_references << ','
                << scheduled_assembly.calls << ','
                << unswitched_assembly.calls << ','
                << scheduled_assembly.widest_register << ','
                << unswitched_assembly.widest_register << '\n';
            if (scenario.name == "balanced") {
                stable = stable && std::isfinite(speedup) &&
                         speedup >= 1.05;
            }
        }
    }
    static_cast<void>(benchmark_sink);
    return stable ? 0 : 1;
}
