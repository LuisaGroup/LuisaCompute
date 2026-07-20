// Test for physical SPIR-V instruction-layout boundaries.
// This test covers:
// - exact and one-past 16-bit instruction-word-count limits
// - the extra literal word required by 64-bit OpSwitch selectors
// - the value/block pairs required by OpPhi
// - shared variadic limits for types, calls, composites, and access paths
// - dialect rejection before oversized switches/composites reach emission

#include "ut/ut.hpp"

#include <cerrno>
#include <sstream>
#include <vector>

#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>

#include <SPIRV/disassemble.h>
#include <SPIRV/spirv.hpp11>

#include "spirv_codegen/dialect.h"
#include "spirv_codegen/instruction_layout.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool has_diagnostic(
    const lc::spirv::SpirvXIRDialectValidationResult &result,
    luisa::string_view needle) noexcept {
    for (auto &&diagnostic : result.diagnostics) {
        if (diagnostic.message.find(needle) != luisa::string_view::npos) {
            return true;
        }
    }
    return false;
}

#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
template<typename F>
[[nodiscard]] bool exits_with_status_one(F &&f) noexcept {
    auto pid = fork();
    if (pid < 0) { return false; }
    if (pid == 0) {
        f();
        _exit(0);
    }
    auto status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) { return false; }
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 1;
}
#endif

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
    "spirv_disassembler_rejects_truncated_integer_type_at_its_own_boundary"_test = [] {
        constexpr auto encode_instruction = [](spv::Op opcode, unsigned int word_count) noexcept {
            return (word_count << 16u) | static_cast<unsigned int>(opcode);
        };
        const std::vector<unsigned int> words{
            spv::MagicNumber,
            0x00010000u,
            0u,
            2u,
            0u,
            encode_instruction(spv::Op::OpTypeInt, 2u),
            1u,
            encode_instruction(spv::Op::OpNop, 1u)};

        expect(exits_with_status_one([&] {
            std::ostringstream out;
            spv::Disassemble(out, words);
        }));
    };
#endif

    "spirv_switch_layout_checks_exact_instruction_word_limit"_test = [] {
        auto i32_exact = lc::spirv::plan_spirv_switch_instruction(
            Type::of<int32_t>(), 32766u);
        expect(i32_exact.succeeded());
        expect(eq(i32_exact.max_case_count, size_t{32766u}));
        expect(eq(i32_exact.instruction_word_count, uint32_t{65535u}));
        auto i32_over = lc::spirv::plan_spirv_switch_instruction(
            Type::of<int32_t>(), 32767u);
        expect(!i32_over.succeeded());
        expect(i32_over.diagnostic.find("at most 32766") !=
               luisa::string_view::npos);

        auto u64_exact = lc::spirv::plan_spirv_switch_instruction(
            Type::of<uint64_t>(), 21844u);
        expect(u64_exact.succeeded());
        expect(eq(u64_exact.max_case_count, size_t{21844u}));
        expect(eq(u64_exact.instruction_word_count, uint32_t{65535u}));
        auto u64_over = lc::spirv::plan_spirv_switch_instruction(
            Type::of<uint64_t>(), 21845u);
        expect(!u64_over.succeeded());
        expect(u64_over.diagnostic.find("at most 21844") !=
               luisa::string_view::npos);
    };

    "spirv_phi_layout_checks_exact_instruction_word_limit"_test = [] {
        auto exact = lc::spirv::plan_spirv_phi_instruction(32766u);
        expect(exact.succeeded());
        expect(eq(exact.max_incoming_count, size_t{32766u}));
        expect(eq(exact.instruction_word_count, uint32_t{65535u}));
        auto over = lc::spirv::plan_spirv_phi_instruction(32767u);
        expect(!over.succeeded());
        expect(over.diagnostic.find("at most 32766") !=
               luisa::string_view::npos);
    };

    "spirv_variadic_layout_checks_exact_instruction_word_limit"_test = [] {
        auto struct_exact = lc::spirv::plan_spirv_variadic_instruction(
            "OpTypeStruct", 2u, 65533u);
        expect(struct_exact.succeeded());
        expect(eq(struct_exact.max_item_count, size_t{65533u}));
        expect(eq(struct_exact.instruction_word_count, uint32_t{65535u}));
        auto struct_over = lc::spirv::plan_spirv_variadic_instruction(
            "OpTypeStruct", 2u, 65534u);
        expect(!struct_over.succeeded());
        expect(struct_over.diagnostic.find("at most 65533") !=
               luisa::string_view::npos);

        auto composite_exact =
            lc::spirv::plan_spirv_variadic_instruction(
                "OpCompositeConstruct", 3u, 65532u);
        expect(composite_exact.succeeded());
        expect(eq(composite_exact.max_item_count, size_t{65532u}));
        expect(eq(composite_exact.instruction_word_count,
                  uint32_t{65535u}));
        auto composite_over =
            lc::spirv::plan_spirv_variadic_instruction(
                "OpCompositeConstruct", 3u, 65533u);
        expect(!composite_over.succeeded());
        expect(composite_over.diagnostic.find("at most 65532") !=
               luisa::string_view::npos);

        auto function_call_exact =
            lc::spirv::plan_spirv_variadic_instruction(
                "OpFunctionCall", 4u, 65531u);
        expect(function_call_exact.succeeded());
        expect(eq(function_call_exact.instruction_word_count,
                  uint32_t{65535u}));
        auto function_call_over =
            lc::spirv::plan_spirv_variadic_instruction(
                "OpFunctionCall", 4u, 65532u);
        expect(!function_call_over.succeeded());
        expect(function_call_over.diagnostic.find("at most 65531") !=
               luisa::string_view::npos);

        auto composite_insert_exact =
            lc::spirv::plan_spirv_variadic_instruction(
                "OpCompositeInsert", 5u, 65530u);
        expect(composite_insert_exact.succeeded());
        expect(eq(composite_insert_exact.instruction_word_count,
                  uint32_t{65535u}));
        auto composite_insert_over =
            lc::spirv::plan_spirv_variadic_instruction(
                "OpCompositeInsert", 5u, 65531u);
        expect(!composite_insert_over.succeeded());
        expect(composite_insert_over.diagnostic.find("at most 65530") !=
               luisa::string_view::npos);
    };

    "spirv_dialect_rejects_oversized_aggregate_before_emission"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *element = module.create_constant_zero(Type::of<uint8_t>());
        constexpr auto first_rejected_constituent_count = size_t{65533u};
        luisa::vector<Value *> constituents(
            first_rejected_constituent_count, element);
        auto *array_type = Type::array(
            Type::of<uint8_t>(), first_rejected_constituent_count);
        luisa::vector<uint8_t> constant_data(
            first_rejected_constituent_count, uint8_t{0u});
        auto *large_constant = module.create_constant(
            array_type, constant_data.data());
        auto *zero = module.create_constant_zero(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.call(
            array_type, ArithmeticOp::AGGREGATE,
            luisa::span<Value *const>{constituents});
        builder.call(
            Type::of<uint8_t>(), ArithmeticOp::EXTRACT,
            {large_constant, zero});
        builder.return_void();

        auto result =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!result.succeeded());
        expect(has_diagnostic(result, "OpCompositeConstruct"));
        expect(has_diagnostic(result, "OpConstantComposite"));
        expect(has_diagnostic(result, "16-bit instruction word count"));
        expect(has_diagnostic(result, "at most 65532"));
    };

    "spirv_dialect_rejects_oversized_switch_before_emission"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *selector = kernel->create_value_argument(Type::of<uint64_t>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *switch_inst = builder.switch_(selector);
        auto *target = switch_inst->create_default_block();
        auto *merge = switch_inst->create_merge_block();
        constexpr auto first_rejected_case_count = size_t{21845u};
        for (size_t i = 0u; i < first_rejected_case_count; ++i) {
            switch_inst->add_case(
                static_cast<SwitchInst::case_value_type>(i), target);
        }
        builder.set_insertion_point(target);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();

        auto result =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!result.succeeded());
        expect(has_diagnostic(result, "16-bit instruction word count"));
        expect(has_diagnostic(result, "at most 21844"));
    };

    return 0;
}
