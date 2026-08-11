#include "llvm_warp_collectives.h"
#include "reference_warp_collectives.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

using namespace luisa::compute::simd;

namespace {

[[nodiscard]] bool check(bool condition, const char *expression,
                         const char *file, int line) noexcept {
    if (!condition) {
        std::cerr << file << ':' << line << ": check failed: "
                  << expression << '\n';
    }
    return condition;
}

#define CHECK(EXPR)                                                           \
    do {                                                                      \
        if (!check(static_cast<bool>(EXPR), #EXPR, __FILE__, __LINE__)) {     \
            return false;                                                     \
        }                                                                     \
    } while (false)

template<size_t Width>
struct OutputLayout {
    static constexpr auto sum = 0u;
    static constexpr auto product = 1u;
    static constexpr auto minimum = 2u;
    static constexpr auto maximum = 3u;
    static constexpr auto bit_and = 4u;
    static constexpr auto bit_or = 5u;
    static constexpr auto bit_xor = 6u;
    static constexpr auto count = 7u;
    static constexpr auto all = 8u;
    static constexpr auto any = 9u;
    static constexpr auto all_equal = 10u;
    static constexpr auto first = 11u;
    static constexpr auto ballot = 12u;
    static constexpr auto prefix_sum = ballot + 4u;
    static constexpr auto prefix_product = prefix_sum + Width;
    static constexpr auto read = prefix_product + Width;
    static constexpr auto invalid = read + Width;
    static constexpr auto read_first = invalid + Width;
    static constexpr auto count_words = read_first + Width;
};

[[nodiscard]] std::string take_error(::llvm::Error error) {
    return ::llvm::toString(std::move(error));
}

template<size_t Width>
[[nodiscard]] bool build_function(::llvm::Module &module,
                                  std::string_view name) {
    auto &context = module.getContext();
    ::llvm::IRBuilder<> builder{context};
    auto *i32 = builder.getInt32Ty();
    auto *pointer = builder.getPtrTy();
    auto *function_type = ::llvm::FunctionType::get(
        builder.getVoidTy(),
        {pointer, pointer, i32, i32, pointer}, false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::ExternalLinkage,
        ::llvm::StringRef{name.data(), name.size()}, module);
    auto argument = function->arg_begin();
    auto *value_pointer = &*argument++;
    auto *source_pointer = &*argument++;
    auto *participant_bits = &*argument++;
    auto *predicate_bits = &*argument++;
    auto *output_pointer = &*argument;

    auto *entry = ::llvm::BasicBlock::Create(context, "entry", function);
    builder.SetInsertPoint(entry);
    auto *lane_type = ::llvm::FixedVectorType::get(i32, Width);
    auto *mask_type = ::llvm::FixedVectorType::get(
        builder.getInt1Ty(), Width);
    auto *values = builder.CreateLoad(lane_type, value_pointer, "values");
    values->setAlignment(::llvm::Align{alignof(uint32_t)});
    auto *sources = builder.CreateLoad(lane_type, source_pointer, "sources");
    sources->setAlignment(::llvm::Align{alignof(uint32_t)});
    ::llvm::Value *participants = ::llvm::PoisonValue::get(mask_type);
    ::llvm::Value *predicate = ::llvm::PoisonValue::get(mask_type);
    for (auto lane = uint32_t{0u}; lane < Width; lane++) {
        auto lane_index = builder.getInt32(lane);
        auto *participant = builder.CreateTrunc(
            builder.CreateLShr(participant_bits, lane),
            builder.getInt1Ty());
        auto *predicate_lane = builder.CreateTrunc(
            builder.CreateLShr(predicate_bits, lane),
            builder.getInt1Ty());
        participants = builder.CreateInsertElement(
            participants, participant, lane_index);
        predicate = builder.CreateInsertElement(
            predicate, predicate_lane, lane_index);
    }

    LLVMWarpCollectives collectives{Width};
    auto *sum = collectives.active_sum(builder, values, participants);
    auto *product = collectives.active_product(
        builder, values, participants);
    auto *minimum = collectives.active_min(
        builder, values, participants, false);
    auto *maximum = collectives.active_max(
        builder, values, participants, false);
    auto *bit_and = collectives.active_bit_and(
        builder, values, participants);
    auto *bit_or = collectives.active_bit_or(
        builder, values, participants);
    auto *bit_xor = collectives.active_bit_xor(
        builder, values, participants);
    auto *count = collectives.active_count_bits(
        builder, predicate, participants);
    auto *all = collectives.active_all(builder, predicate, participants);
    auto *any = collectives.active_any(builder, predicate, participants);
    auto *all_equal = collectives.active_all_equal(
        builder, values, participants);
    auto *first = collectives.first_active_lane(builder, participants);
    auto *ballot = collectives.active_bit_mask(
        builder, predicate, participants);
    auto *prefix_sum = collectives.prefix_sum(
        builder, values, participants);
    auto *prefix_product = collectives.prefix_product(
        builder, values, participants);
    auto read = collectives.read_lane(
        builder, values, sources, participants);
    auto read_first = collectives.read_first_active_lane(
        builder, values, participants);
    CHECK(collectives.succeeded());
    CHECK(sum != nullptr && product != nullptr && minimum != nullptr);
    CHECK(maximum != nullptr && bit_and != nullptr && bit_or != nullptr);
    CHECK(bit_xor != nullptr && count != nullptr && all != nullptr);
    CHECK(any != nullptr && all_equal != nullptr && first != nullptr);
    CHECK(ballot != nullptr && prefix_sum != nullptr &&
          prefix_product != nullptr);
    CHECK(read.values != nullptr && read.invalid_lanes != nullptr);
    CHECK(read_first.values != nullptr &&
          read_first.invalid_lanes != nullptr);

    auto store = [&](::llvm::Value *value, uint32_t index) {
        if (value->getType()->isIntegerTy(1u)) {
            value = builder.CreateZExt(value, i32);
        }
        builder.CreateStore(
            value, builder.CreateGEP(
                       i32, output_pointer, builder.getInt32(index)));
    };
    using Out = OutputLayout<Width>;
    store(sum, Out::sum);
    store(product, Out::product);
    store(minimum, Out::minimum);
    store(maximum, Out::maximum);
    store(bit_and, Out::bit_and);
    store(bit_or, Out::bit_or);
    store(bit_xor, Out::bit_xor);
    store(count, Out::count);
    store(all, Out::all);
    store(any, Out::any);
    store(all_equal, Out::all_equal);
    store(first, Out::first);
    for (auto word = uint32_t{0u}; word < 4u; word++) {
        store(builder.CreateExtractElement(ballot, word),
              Out::ballot + word);
    }
    for (auto lane = uint32_t{0u}; lane < Width; lane++) {
        store(builder.CreateExtractElement(prefix_sum, lane),
              Out::prefix_sum + lane);
        store(builder.CreateExtractElement(prefix_product, lane),
              Out::prefix_product + lane);
        store(builder.CreateExtractElement(read.values, lane),
              Out::read + lane);
        store(builder.CreateExtractElement(read.invalid_lanes, lane),
              Out::invalid + lane);
        store(builder.CreateExtractElement(read_first.values, lane),
              Out::read_first + lane);
    }
    builder.CreateRetVoid();
    return true;
}

template<size_t Width>
[[nodiscard]] bool verify_results(const std::array<uint32_t, Width> &values,
                                  const std::array<uint32_t, Width> &sources,
                                  uint32_t participant_bits,
                                  uint32_t predicate_bits,
                                  const std::vector<uint32_t> &output) {
    using Ref = reference::WarpCollectives<Width>;
    using Mask = typename Ref::Mask;
    Mask participants{};
    typename Ref::template Lanes<bool> predicate{};
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        if ((participant_bits & (1u << lane)) != 0u) {
            participants.set(lane);
        }
        predicate[lane] = (predicate_bits & (1u << lane)) != 0u;
    }
    auto expected_sum = Ref::active_sum(participants, values);
    auto expected_product = Ref::active_product(participants, values);
    auto expected_prefix_sum = Ref::prefix_sum(participants, values);
    auto expected_prefix_product = Ref::prefix_product(participants, values);
    auto expected_read = Ref::read_lane(participants, values, sources);
    auto expected_first = Ref::read_first_active_lane(participants, values);
    auto expected_ballot = Ref::active_bit_mask(participants, predicate);
    using Out = OutputLayout<Width>;
    CHECK(output.size() == Out::count_words);
    CHECK(output[Out::sum] == *expected_sum);
    CHECK(output[Out::product] == *expected_product);
    CHECK(output[Out::minimum] == *Ref::active_min(participants, values));
    CHECK(output[Out::maximum] == *Ref::active_max(participants, values));
    CHECK(output[Out::bit_and] ==
          *Ref::active_bit_and(participants, values));
    CHECK(output[Out::bit_or] ==
          *Ref::active_bit_or(participants, values));
    CHECK(output[Out::bit_xor] ==
          *Ref::active_bit_xor(participants, values));
    CHECK(output[Out::count] ==
          Ref::active_count_bits(participants, predicate));
    CHECK(output[Out::all] == Ref::active_all(participants, predicate));
    CHECK(output[Out::any] == Ref::active_any(participants, predicate));
    CHECK(output[Out::all_equal] ==
          Ref::active_all_equal(participants, values));
    CHECK(output[Out::first] == *Ref::first_active_lane(participants));
    for (auto word = size_t{0u}; word < 4u; word++) {
        CHECK(output[Out::ballot + word] == expected_ballot[word]);
    }
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        CHECK(output[Out::prefix_sum + lane] == expected_prefix_sum[lane]);
        CHECK(output[Out::prefix_product + lane] ==
              expected_prefix_product[lane]);
        CHECK(output[Out::read + lane] == expected_read.values[lane]);
        CHECK(output[Out::invalid + lane] ==
              expected_read.invalid_lanes.test(lane));
        auto expected = participants.test(lane) ? *expected_first : 0u;
        CHECK(output[Out::read_first + lane] == expected);
    }
    return true;
}

template<size_t Width>
[[nodiscard]] bool test_jit(uint32_t participant_bits,
                            uint32_t predicate_bits,
                            uint32_t invalid_destination) {
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-warp-collectives", *context);
    auto name = std::string{"simd_collectives_w"} +
                std::to_string(Width);
    CHECK(build_function<Width>(*module, name));
    CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));

    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    module->print(stream, nullptr);
    stream.flush();
    auto lane_spelling = std::string{"<"} + std::to_string(Width) +
                         " x i32>";
    CHECK(ir.find(lane_spelling) != std::string::npos);
    CHECK(ir.find("llvm.vector.reduce.add.v" +
                  std::to_string(Width) + "i32") != std::string::npos);
    CHECK(ir.find("shufflevector") != std::string::npos);
    CHECK(ir.find("llvm.x86.") == std::string::npos);
    CHECK(ir.find("llvm.aarch64.") == std::string::npos);
    CHECK(ir.find("llvm.arm.neon.") == std::string::npos);

    ::llvm::orc::LLJITBuilder jit_builder;
    auto host = ::llvm::orc::JITTargetMachineBuilder::detectHost();
    CHECK(static_cast<bool>(host));
    host->setCodeGenOptLevel(::llvm::CodeGenOptLevel::Aggressive);
    jit_builder.setJITTargetMachineBuilder(std::move(*host));
    auto expected_jit = jit_builder.create();
    CHECK(static_cast<bool>(expected_jit));
    auto jit = std::move(*expected_jit);
    module->setDataLayout(jit->getDataLayout());
    auto thread_safe_module = ::llvm::orc::ThreadSafeModule(
        std::move(module), std::move(context));
    if (auto error = jit->addIRModule(std::move(thread_safe_module))) {
        std::cerr << "LLJIT::addIRModule failed: "
                  << take_error(std::move(error)) << '\n';
        return false;
    }
    auto symbol = jit->lookup(name);
    if (!symbol) {
        std::cerr << "LLJIT::lookup failed: "
                  << take_error(symbol.takeError()) << '\n';
        return false;
    }
    using Function = void(const uint32_t *, const uint32_t *,
                          uint32_t, uint32_t, uint32_t *);
    auto function = symbol->template toPtr<Function>();
    CHECK(function != nullptr);

    std::array<uint32_t, Width> values{};
    std::array<uint32_t, Width> sources{};
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        values[lane] = static_cast<uint32_t>(lane + 2u);
        sources[lane] = 1u;
    }
    sources[invalid_destination] = 2u;
    std::vector<uint32_t> output(
        OutputLayout<Width>::count_words, 0xdeadbeefu);
    function(values.data(), sources.data(), participant_bits,
             predicate_bits, output.data());
    return verify_results(values, sources, participant_bits,
                          predicate_bits, output);
}

}// namespace

int main() {
    if (::llvm::InitializeNativeTarget() ||
        ::llvm::InitializeNativeTargetAsmPrinter()) {
        std::cerr << "failed to initialize LLVM native target\n";
        return 1;
    }
    struct Test {
        std::string_view name;
        bool (*run)();
    };
    constexpr Test tests[]{
        {"LLVM vector warp4", [] { return test_jit<4u>(0x0bu, 0x09u, 3u); }},
        {"LLVM vector warp8", [] { return test_jit<8u>(0x43u, 0x41u, 6u); }},
    };
    auto failures = 0u;
    for (auto test : tests) {
        if (test.run()) {
            std::cout << "[pass] " << test.name << '\n';
        } else {
            std::cerr << "[fail] " << test.name << '\n';
            ++failures;
        }
    }
    return failures == 0u ? 0 : 1;
}
