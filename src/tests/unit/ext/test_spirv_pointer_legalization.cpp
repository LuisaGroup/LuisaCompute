// Test for SPIR-V callable pointer-argument legalization.
// This test covers:
// - switch-scoped fallback for a derived reference argument
// - native OpSwitch preservation when no pointer fallback is needed
// - fixed-point specialization and local/shared storage-class boundaries
// - indirect-dispatch, acceleration-structure, and texture ABI boundaries
// - unused-resource preservation
// - deterministic failure when the inline retry cannot remove a recursive call

#include "ut/ut.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include <spirv-tools/libspirv.hpp>

#include <luisa/core/stl/memory.h>
#include <luisa/dsl/dispatch_indirect.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/fix_self_referential.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/verifier.h>

#include "spirv_codegen/entry.h"
#include "spirv_codegen/argument_usage.h"
#include "spirv_codegen/dialect.h"
#include "spirv_codegen/pointer_legalization.h"
#include "spirv_codegen/utils.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void set_environment_variable(const char *name,
                              const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

class ScopedEnvironmentVariable {
private:
    const char *_name;
    std::optional<std::string> _previous;

public:
    ScopedEnvironmentVariable(const char *name,
                              const char *value) noexcept
        : _name{name} {
        if (auto previous = std::getenv(name)) {
            _previous.emplace(previous);
        }
        set_environment_variable(name, value);
    }
    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            _name, _previous ? _previous->c_str() : nullptr);
    }
    ScopedEnvironmentVariable(
        const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(
        const ScopedEnvironmentVariable &) = delete;
};

[[nodiscard]] size_t count_opcode(
    luisa::span<const uint32_t> words, spv::Op expected) noexcept {
    auto count = size_t{0u};
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = words[offset] >> 16u;
        if (word_count == 0u ||
            word_count > words.size() - offset) {
            break;
        }
        auto opcode = static_cast<spv::Op>(
            words[offset] & 0xffffu);
        count += opcode == expected ? 1u : 0u;
        offset += word_count;
    }
    return count;
}

[[nodiscard]] bool validates(
    luisa::span<const uint32_t> words) noexcept {
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    return tools.Validate(words.data(), words.size());
}

[[nodiscard]] bool has_diagnostic(
    const lc::spirv::SpirvXIRDialectValidationResult &result,
    luisa::string_view fragment) noexcept {
    for (auto &&diagnostic : result.diagnostics) {
        if (diagnostic.message.find(fragment) != luisa::string::npos) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] auto make_pointer_switch_kernel() noexcept {
    Callable update = [](UInt &value, UInt selector) noexcept {
        $switch (selector) {
            $case (0u) { value = 11u; };
            $case (1u) { value += 7u; };
            $default { value = 29u; };
        };
    };
    Kernel1D kernel = [&update](BufferUInt output,
                                UInt selector) noexcept {
        Var<std::array<uint32_t, 2u>> values;
        values[0u] = 3u;
        values[1u] = 5u;
        update(values[1u], selector);
        output.write(0u, values[1u]);
    };
    return kernel;
}

[[nodiscard]] auto make_native_switch_kernel() noexcept {
    Kernel1D kernel = [](BufferUInt output, UInt selector) noexcept {
        UInt value = 31u;
        $switch (selector) {
            $case (0u) { value = 3u; };
            $case (7u) { value = 17u; };
            $default { value = 43u; };
        };
        output.write(0u, value);
    };
    return kernel;
}

enum class PointerSwitchOrphan : uint8_t {
    NONE,
    UNTERMINATED,
    MALFORMED_SWITCH,
};

struct PointerSwitchCallableFixture {
    CallableFunction *callable;
    BasicBlock *entry;
    SwitchInst *reachable_switch;
    BasicBlock *orphan;
    SwitchInst *orphan_switch;
};

[[nodiscard]] PointerSwitchCallableFixture
make_pointer_switch_callable(
    Module &module, PointerSwitchOrphan orphan_kind) noexcept {
    auto *uint_type = Type::of<uint32_t>();
    auto *callable = module.create_callable(Type::of<void>());
    auto *value = callable->create_reference_argument(uint_type);
    auto *selector = callable->create_value_argument(uint_type);
    auto *entry = callable->create_body_block();
    auto *one = module.create_constant_one(uint_type);
    XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *reachable_switch = builder.switch_(selector);
    auto *case_block =
        reachable_switch->create_case_block(0u);
    auto *default_block =
        reachable_switch->create_default_block();
    auto *merge_block =
        reachable_switch->create_merge_block();
    builder.set_insertion_point(case_block);
    builder.br(merge_block);
    builder.set_insertion_point(default_block);
    builder.br(merge_block);
    builder.set_insertion_point(merge_block);
    builder.store(value, one);
    builder.return_void();

    BasicBlock *orphan = nullptr;
    SwitchInst *orphan_switch = nullptr;
    if (orphan_kind != PointerSwitchOrphan::NONE) {
        orphan = callable->create_basic_block();
        if (orphan_kind == PointerSwitchOrphan::MALFORMED_SWITCH) {
            builder.set_insertion_point(orphan);
            orphan_switch = builder.switch_(selector);
        }
    }
    return {
        .callable = callable,
        .entry = entry,
        .reachable_switch = reachable_switch,
        .orphan = orphan,
        .orphan_switch = orphan_switch};
}

[[nodiscard]] lc::spirv::SpirvResult compile_exact_xir(
    luisa::compute::Function kernel, const Module *module,
    lc::spirv::SpirvTargetFeatures target_features = {}) {
    ScopedEnvironmentVariable disable_spirv_optimization{
        "LUISA_SPIRV_OPT_LEVEL", "0"};
    ScopedEnvironmentVariable clear_spirv_pass_override{
        "LUISA_SPIRV_OPT_PASSES", nullptr};
    return lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
        kernel, module, ShaderOption{.enable_cache = false},
        target_features);
}

void restore_structured_codegen_boundary(Module *module) {
    static_cast<void>(reg2mem_pass_run_on_module(module));
    auto restructured = restructure_cfg_pass_run_on_module(module);
    expect(restructured.succeeded());
    expect(restructured.restructured_if_count +
               restructured.restructured_switch_count >
           0u)
        << "pointer fallback raw CFG must be recovered as structured control flow";
    auto post_restructure =
        luisa::compute::spirv::create_spirv_codegen_post_restructure_pipeline();
    static_cast<void>(post_restructure.run(module));
    auto fixed = fix_self_referential_pass_run_on_module(module);
    expect(fixed.succeeded());
    auto dialect = lc::spirv::validate_spirv_xir_codegen_dialect(module);
    expect(dialect.succeeded());
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_argument_usage_propagates_sparse_deep_call_chain"_test = [] {
        Module module;
        constexpr auto chain_length = 128u;
        auto *buffer_type =
            Type::buffer(Type::of<uint32_t>());
        luisa::vector<CallableFunction *> callables;
        luisa::vector<ResourceArgument *> arguments;
        luisa::vector<BasicBlock *> bodies;
        callables.reserve(chain_length);
        arguments.reserve(chain_length);
        bodies.reserve(chain_length);
        for (auto i = 0u; i < chain_length; ++i) {
            auto *callable = module.create_callable(nullptr);
            callables.emplace_back(callable);
            arguments.emplace_back(
                callable->create_resource_argument(buffer_type));
            bodies.emplace_back(callable->create_body_block());
        }
        XIRBuilder builder;
        for (auto i = 0u; i + 1u < chain_length; ++i) {
            builder.set_insertion_point(bodies[i]);
            builder.call(nullptr, callables[i + 1u],
                         {arguments[i]});
            builder.return_void();
        }
        auto *zero =
            module.create_constant_zero(Type::of<uint32_t>());
        builder.set_insertion_point(bodies.back());
        builder.call(Type::of<uint32_t>(),
                     ResourceReadOp::BUFFER_READ,
                     {arguments.back(), zero});
        builder.return_void();

        auto *kernel = module.create_kernel();
        auto *kernel_buffer =
            kernel->create_resource_argument(buffer_type);
        auto *kernel_body = kernel->create_body_block();
        builder.set_insertion_point(kernel_body);
        builder.call(nullptr, callables.front(),
                     {kernel_buffer});
        builder.return_void();
        expect(xir_verify_module(&module).succeeded());

        lc::spirv::SpirvFunctionArgumentAnalysisStatistics statistics;
        lc::spirv::SpirvFunctionCallSiteList call_sites;
        auto analysis =
            lc::spirv::analyze_spirv_function_argument_usage(
                &module, &statistics, {}, &call_sites);
        expect(lc::spirv::spirv_function_argument_usage_of(
                   analysis, kernel, kernel_buffer) == Usage::READ);
        expect(eq(call_sites.size(), chain_length));
        auto origins = lc::spirv::
            analyze_spirv_unique_resource_origins_from_call_sites(
                analysis, luisa::span{call_sites});
        expect(eq(origins.size(), chain_length));
        for (auto *argument : arguments) {
            expect(origins.at(argument) == kernel_buffer);
        }
        expect(eq(statistics.structural_closure_count,
                  chain_length + 1u));
        expect(eq(statistics.instruction_scan_count,
                  2u * (chain_length + 1u)));
        expect(eq(statistics.call_dependency_count,
                  chain_length));
        expect(eq(statistics.worklist_pop_count,
                  chain_length + 1u));
        expect(eq(statistics.dependency_visit_count,
                  chain_length));
    };

    "spirv_pointer_switch_fallback_compiles_and_validates"_test = [] {
        auto kernel = make_pointer_switch_kernel();
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        CallInst *pointer_call = nullptr;
        for (auto *function : module->function_list()) {
            if (auto *definition = function->definition()) {
                definition->traverse_instructions(
                    [&](Instruction *instruction) noexcept {
                        if (instruction->isa<CallInst>()) {
                            expect(pointer_call == nullptr)
                                << "the fixture must contain exactly one call";
                            pointer_call =
                                static_cast<CallInst *>(instruction);
                        }
                    });
            }
        }
        expect(pointer_call != nullptr);
        pointer_call->add_comment(
            "mandatory pointer specialization call site");
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());

        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 1u));
        expect(eq(legalized.blocking_function_count, 0u));
        expect(eq(legalized.destructured_blocking_function_count, 0u));
        expect(eq(legalized.destructured_switch_count, 0u));
        expect(eq(legalized.inline_info.inlined_call_count, 1u));
        expect(eq(
            legalized.inline_info
                .consumed_call_site_diagnostic_metadata_count,
            1u));
        expect(eq(legalized.inline_info.skipped_metadata_call_count, 0u));
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(eq(legalized.argument_usage_analysis_count, 2u));
        expect(xir_verify_module(module.get()).succeeded());
        auto intermediate_dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(module.get());
        expect(!intermediate_dialect.succeeded())
            << "switch fallback intentionally produces raw CFG before the "
               "mandatory restructure boundary";
        expect(has_diagnostic(
            intermediate_dialect, "raw indexed branches"));
        expect(has_diagnostic(intermediate_dialect, "SwitchInst"));
        restore_structured_codegen_boundary(module.get());
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get());
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(eq(count_opcode(luisa::span{compiled.spv_bin},
                               spv::Op::OpFunctionCall),
                  0u))
            << "the ABI-sensitive callable must be specialized away";
        expect(eq(count_opcode(luisa::span{compiled.spv_bin},
                               spv::Op::OpSwitch),
                  1u))
            << "pointer specialization must preserve native switch selection";
    };

    "spirv_pointer_legalization_rejects_semantic_call_metadata"_test = [] {
        auto kernel = make_pointer_switch_kernel();
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        CallInst *pointer_call = nullptr;
        for (auto *function : module->function_list()) {
            if (auto *definition = function->definition()) {
                definition->traverse_instructions(
                    [&](Instruction *instruction) noexcept {
                        if (instruction->isa<CallInst>()) {
                            expect(pointer_call == nullptr)
                                << "the fixture must contain exactly one call";
                            pointer_call =
                                static_cast<CallInst *>(instruction);
                        }
                    });
            }
        }
        expect(pointer_call != nullptr);
        pointer_call->metadata_list().push_front(
            luisa::make_managed<Reg2MemSpillMD>(
                Reg2MemSpillKind::CROSS_BLOCK));
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());

        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(!legalized.succeeded());
        expect(eq(legalized.inline_info.inlined_call_count, 0u));
        expect(eq(
            legalized.inline_info
                .consumed_call_site_diagnostic_metadata_count,
            0u));
        expect(eq(legalized.inline_info.skipped_metadata_call_count, 1u));
        expect(legalized.diagnostic.find("metadata=1") !=
               luisa::string::npos);
        expect(pointer_call->is_linked());
        expect(xir_verify_module(module.get()).succeeded());
    };

    "spirv_pointer_legalization_preserves_unblocked_native_switch"_test = [] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto kernel = make_native_switch_kernel();
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u));
        restore_structured_codegen_boundary(module.get());
        auto compiled = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            kernel.function()->function(), module.get(),
            ShaderOption{.enable_cache = false});
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpSwitch) > 0u)
            << "a switch outside pointer specialization must remain OpSwitch";
    };

    "spirv_pointer_legalization_reaches_transitive_fixed_point"_test = [] {
        Callable update = [](UInt &value) noexcept {
            value += 1u;
        };
        Callable relay = [&update](UInt &value) noexcept {
            update(value);
        };
        Kernel1D kernel = [&relay](BufferUInt output) noexcept {
            Var<std::array<uint32_t, 2u>> values;
            values[0u] = 3u;
            values[1u] = 5u;
            relay(values[1u]);
            output.write(0u, values[1u]);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto uint_reference_formal_count = size_t{0u};
        for (auto *function : module->function_list()) {
            for (auto *argument : function->arguments()) {
                if (argument->is_reference()) {
                    expect(argument->type() == Type::of<uint32_t>());
                    uint_reference_formal_count++;
                }
            }
        }
        expect(eq(uint_reference_formal_count, 2u))
            << "the relay and update callables must each retain their UInt& "
               "formal before pointer specialization";
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 2u))
            << "the relay inline must expose and then specialize the inner call";
        expect(eq(legalized.inline_info.inlined_call_count, 2u));
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get());
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(eq(count_opcode(luisa::span{compiled.spv_bin},
                               spv::Op::OpFunctionCall),
                  0u));
    };

    "spirv_pointer_legalization_specializes_shared_alloca_reference"_test = [] {
        Module module;
        auto *uint_type = Type::of<uint32_t>();
        auto *callable = module.create_callable(Type::of<void>());
        auto *reference =
            callable->create_reference_argument(uint_type);
        auto *callable_body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(callable_body);
        builder.store(reference, module.create_constant_one(uint_type));
        builder.return_void();

        auto *xir_kernel = module.create_kernel();
        auto *kernel_body = xir_kernel->create_body_block();
        builder.set_insertion_point(kernel_body);
        auto *shared = builder.alloca_shared(uint_type);
        auto *call = builder.call(nullptr, callable, {shared});
        auto call_lock = call->lock_into<Instruction>();
        builder.return_void();

        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(&module);
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 1u));
        expect(eq(legalized.inline_info.inlined_call_count, 1u));
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(!call_lock->is_linked());
        expect(xir_verify_module(&module).succeeded());

        Kernel1D kernel = []() noexcept {};
        xir_kernel->set_block_size(
            kernel.function()->function().block_size());
        auto compiled = compile_exact_xir(
            kernel.function()->function(), &module);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(eq(count_opcode(luisa::span{compiled.spv_bin},
                               spv::Op::OpFunctionCall),
                  0u));
    };

    "spirv_pointer_legalization_preserves_local_alloca_reference"_test = [] {
        Module module;
        auto *uint_type = Type::of<uint32_t>();
        auto *callable = module.create_callable(Type::of<void>());
        auto *reference =
            callable->create_reference_argument(uint_type);
        auto *callable_body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(callable_body);
        builder.store(reference, module.create_constant_one(uint_type));
        builder.return_void();

        auto *xir_kernel = module.create_kernel();
        auto *kernel_body = xir_kernel->create_body_block();
        builder.set_insertion_point(kernel_body);
        auto *local = builder.alloca_local(uint_type);
        auto *call = builder.call(nullptr, callable, {local});
        builder.return_void();

        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(&module);
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u));
        expect(call->is_linked());
        expect(xir_verify_module(&module).succeeded());

        Kernel1D kernel = []() noexcept {};
        xir_kernel->set_block_size(
            kernel.function()->function().block_size());
        auto compiled = compile_exact_xir(
            kernel.function()->function(), &module);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpFunctionCall) > 0u);
    };

    "spirv_pointer_legalization_specializes_indirect_dispatch_callable"_test = [] {
        Callable author = [](Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_dispatch_count(1u);
            commands.set_kernel(
                0u, make_uint3(1u), make_uint3(1u), 0u);
        };
        Kernel1D kernel = [&author](
                              Var<IndirectDispatchBuffer> commands) noexcept {
            author(commands);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 1u));
        expect(eq(legalized.inline_info.inlined_call_count, 1u));
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get());
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(eq(count_opcode(luisa::span{compiled.spv_bin},
                               spv::Op::OpFunctionCall),
                  0u));
    };

    "spirv_pointer_legalization_preserves_unused_buffer_callable"_test = [] {
        Callable ignore = [](BufferUInt) noexcept {};
        Kernel1D kernel = [&ignore](BufferUInt buffer) noexcept {
            ignore(buffer);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u));
        expect(eq(legalized.inline_info.inlined_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get());
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpFunctionCall) > 0u)
            << "an unused resource formal must not force specialization";
    };

    "spirv_pointer_legalization_outlines_unique_readonly_buffer_origin"_test = [] {
        Callable read = [](BufferUInt input, UInt index) noexcept {
            return input.read(index);
        };
        Kernel1D kernel = [&read](BufferUInt input,
                                  BufferUInt output) noexcept {
            output.write(
                0u, read(input, 0u) + read(input, 1u));
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u))
            << "a callable read-only resource with one kernel origin must "
               "use that module-level descriptor without call-site inlining";
        expect(eq(legalized.inline_info.inlined_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(
                module.get());
        expect(dialect.succeeded())
            << (dialect.diagnostics.empty() ?
                    "unknown dialect failure" :
                    dialect.diagnostics.front().message);
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get());
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(
                   luisa::span{compiled.spv_bin},
                   spv::Op::OpFunctionCall) > 0u)
            << "the read-only callable must remain outlined";
    };

    "spirv_pointer_legalization_outlines_unique_read_write_buffer_origin"_test = [] {
        Callable update = [](BufferUInt values, UInt index) noexcept {
            values.write(index, values.read(index) + 1u);
        };
        Kernel1D kernel = [&update](BufferUInt values) noexcept {
            update(values, 0u);
            update(values, 1u);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());

        lc::spirv::SpirvFunctionCallSiteList call_sites;
        auto analysis =
            lc::spirv::analyze_spirv_function_argument_usage(
                module.get(), nullptr, {}, &call_sites);
        auto origins = lc::spirv::
            analyze_spirv_unique_resource_origins_from_call_sites(
                analysis, luisa::span{call_sites});
        const ResourceArgument *callable_buffer = nullptr;
        const ResourceArgument *kernel_buffer = nullptr;
        for (auto *function : module->function_list()) {
            for (auto *argument : function->arguments()) {
                if (!argument->is_resource() ||
                    !argument->type()->is_buffer()) {
                    continue;
                }
                auto *resource =
                    static_cast<ResourceArgument *>(argument);
                if (function->isa<CallableFunction>()) {
                    callable_buffer = resource;
                } else if (function->isa<KernelFunction>()) {
                    kernel_buffer = resource;
                }
            }
        }
        expect(callable_buffer != nullptr);
        expect(kernel_buffer != nullptr);
        expect(origins.at(callable_buffer) == kernel_buffer)
            << "a complete equal-origin proof is independent of resource "
               "usage direction";

        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u))
            << "reads and writes of one proven kernel buffer must not force "
               "call-site specialization";
        expect(eq(legalized.inline_info.inlined_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(
                module.get());
        expect(dialect.succeeded())
            << (dialect.diagnostics.empty() ?
                    "unknown dialect failure" :
                    dialect.diagnostics.front().message);
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get());
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(
                   luisa::span{compiled.spv_bin},
                   spv::Op::OpFunctionCall) >= 2u)
            << "both read/write calls must remain outlined";
    };

    "spirv_pointer_legalization_specializes_conflicting_read_write_buffer_origins"_test = [] {
        Callable update = [](BufferUInt values, UInt index) noexcept {
            values.write(index, values.read(index) + 1u);
        };
        Kernel1D kernel = [&update](BufferUInt values_a,
                                    BufferUInt values_b) noexcept {
            update(values_a, 0u);
            update(values_b, 0u);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());

        lc::spirv::SpirvFunctionCallSiteList call_sites;
        auto analysis =
            lc::spirv::analyze_spirv_function_argument_usage(
                module.get(), nullptr, {}, &call_sites);
        auto origins = lc::spirv::
            analyze_spirv_unique_resource_origins_from_call_sites(
                analysis, luisa::span{call_sites});
        const ResourceArgument *callable_buffer = nullptr;
        for (auto *function : module->function_list()) {
            if (!function->isa<CallableFunction>()) { continue; }
            for (auto *argument : function->arguments()) {
                if (argument->is_resource() &&
                    argument->type()->is_buffer()) {
                    callable_buffer =
                        static_cast<ResourceArgument *>(argument);
                }
            }
        }
        expect(callable_buffer != nullptr);
        expect(!origins.contains(callable_buffer))
            << "two distinct kernel descriptors must reach the conflicting "
               "lattice element, independent of resource usage direction";

        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 2u));
        expect(eq(legalized.inline_info.inlined_call_count, 2u));
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get());
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(eq(count_opcode(
                      luisa::span{compiled.spv_bin},
                      spv::Op::OpFunctionCall),
                  0u))
            << "a conflicting read/write descriptor origin must retain the "
               "conservative call-site specialization fallback";
    };

    "spirv_pointer_legalization_proves_transitive_readonly_buffer_origin"_test = [] {
        Callable read = [](BufferUInt input, UInt index) noexcept {
            return input.read(index);
        };
        Callable relay = [&read](BufferUInt input,
                                 UInt index) noexcept {
            return read(input, index);
        };
        Kernel1D kernel = [&relay](BufferUInt input,
                                   BufferUInt output) noexcept {
            output.write(0u, relay(input, 1u));
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u))
            << "the unique kernel origin must propagate through every "
               "read-only callable forwarding edge";
        expect(eq(legalized.inline_info.inlined_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(
                module.get());
        expect(dialect.succeeded())
            << (dialect.diagnostics.empty() ?
                    "unknown dialect failure" :
                    dialect.diagnostics.front().message);
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get());
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(
                   luisa::span{compiled.spv_bin},
                   spv::Op::OpFunctionCall) >= 2u)
            << "both forwarding callables must remain outlined";
    };

    "spirv_pointer_legalization_excludes_dead_wrapper_resource_edges"_test = [] {
        Callable read = [](BufferUInt input, UInt index) noexcept {
            return input.read(index);
        };
        Callable inner = [&read](UInt &value,
                                 BufferUInt input) noexcept {
            value = read(input, 0u);
        };
        Callable outer = [&inner](UInt &value,
                                  BufferUInt input) noexcept {
            inner(value, input);
        };
        Kernel1D kernel = [&outer](BufferUInt input,
                                   BufferUInt output) noexcept {
            Var<std::array<uint32_t, 2u>> values;
            values[0u] = 3u;
            values[1u] = 5u;
            // The derived reference forces outer and then inner to be
            // specialized. Their old definitions must leave the reachable
            // resource-flow domain before the next fixed-point iteration.
            outer(values[1u], input);
            output.write(0u, values[1u]);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());

        // Reproduce the production coroutine shape: a wrapper that must be
        // specialized owns an orphan block which references another callable.
        // The orphan calls the same live resource helper through a formal with
        // no semantic incoming edge. Physical ownership keeps that callable
        // alive until the wrapper is removed, but neither the orphan nor its
        // unresolved resource edge belongs to the kernel-rooted codegen
        // closure used by pointer legalization.
        CallableFunction *outer_xir = nullptr;
        CallableFunction *inner_xir = nullptr;
        CallableFunction *read_xir = nullptr;
        KernelFunction *kernel_xir = nullptr;
        for (auto *function : module->function_list()) {
            if (function->isa<KernelFunction>()) {
                kernel_xir = static_cast<KernelFunction *>(function);
                break;
            }
        }
        auto first_callable = [](FunctionDefinition *definition) noexcept
            -> CallableFunction * {
            CallableFunction *result = nullptr;
            definition->traverse_instructions(
                [&](Instruction *instruction) noexcept {
                    if (result != nullptr ||
                        !instruction->isa<CallInst>()) {
                        return;
                    }
                    auto *callee = static_cast<CallInst *>(instruction)
                                       ->callee();
                    if (callee != nullptr &&
                        callee->isa<CallableFunction>()) {
                        result = static_cast<CallableFunction *>(callee);
                    }
                });
            return result;
        };
        expect(kernel_xir != nullptr);
        outer_xir = first_callable(kernel_xir);
        expect(outer_xir != nullptr);
        inner_xir = first_callable(outer_xir);
        expect(inner_xir != nullptr);
        read_xir = first_callable(inner_xir);
        expect(read_xir != nullptr);

        auto *buffer_type = Type::buffer(Type::of<uint32_t>());
        auto *orphan = module->create_callable(Type::of<void>());
        auto *orphan_input =
            orphan->create_resource_argument(buffer_type);
        auto *orphan_body = orphan->create_body_block();
        auto *zero = module->create_constant_zero(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(orphan_body);
        static_cast<void>(builder.call(
            Type::of<uint32_t>(), read_xir,
            {orphan_input, zero}));
        builder.return_void();

        ResourceArgument *outer_input = nullptr;
        for (auto *argument : outer_xir->arguments()) {
            if (argument->is_resource() &&
                argument->type() == buffer_type) {
                outer_input = static_cast<ResourceArgument *>(argument);
                break;
            }
        }
        expect(outer_input != nullptr);
        auto *disconnected = outer_xir->create_basic_block();
        builder.set_insertion_point(disconnected);
        static_cast<void>(builder.call(
            nullptr, orphan, {outer_input}));
        builder.return_void();
        expect(xir_verify_module(module.get()).succeeded());

        lc::spirv::SpirvFunctionCallSiteList live_call_sites;
        auto live_usage =
            lc::spirv::analyze_spirv_function_argument_usage(
                module.get(), nullptr,
                {.kernel_reachable_only = true},
                &live_call_sites);
        expect(eq(live_call_sites.size(), 3u))
            << "only kernel->outer->inner->read belongs to the semantic call graph";
        for (auto *call : live_call_sites) {
            expect(call->callee() != orphan)
                << "an orphan physical function operand must not enter the sparse call-site index";
        }
        auto live_origins = lc::spirv::
            analyze_spirv_unique_resource_origins_from_call_sites(
                live_usage, luisa::span{live_call_sites});
        expect(eq(live_origins.size(), 3u));
        expect(!live_origins.contains(orphan_input))
            << "the orphan formal must not receive an origin proof";

        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 2u))
            << "only the two derived-reference wrappers require specialization";
        expect(eq(legalized.inline_info.inlined_call_count, 2u));
        expect(eq(legalized.pruned_unreachable_callable_count, 1u))
            << "the orphan-only callable must be pruned after its wrapper is removed";
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(eq(legalized.argument_usage_analysis_count, 3u));
        expect(xir_verify_module(module.get()).succeeded());

        auto callable_count = size_t{0u};
        auto call_count = size_t{0u};
        for (auto *function : module->function_list()) {
            callable_count += function->isa<CallableFunction>() ? 1u : 0u;
            if (auto *definition = function->definition()) {
                definition->traverse_instructions(
                    [&](const Instruction *instruction) noexcept {
                        call_count += instruction->isa<CallInst>() ? 1u : 0u;
                    });
            }
        }
        expect(eq(callable_count, 1u));
        expect(eq(call_count, 1u))
            << "the uniquely rooted read helper must remain outlined";

        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(
                module.get());
        expect(dialect.succeeded())
            << (dialect.diagnostics.empty() ?
                    "unknown dialect failure" :
                    dialect.diagnostics.front().message);
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get());
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(
                   luisa::span{compiled.spv_bin},
                   spv::Op::OpFunctionCall) > 0u)
            << "dead wrappers must not force the live read helper inline";
    };

    "spirv_pointer_legalization_specializes_conflicting_readonly_buffer_origins"_test = [] {
        Callable read = [](BufferUInt input, UInt index) noexcept {
            return input.read(index);
        };
        Kernel1D kernel = [&read](BufferUInt input_a,
                                  BufferUInt input_b,
                                  BufferUInt output) noexcept {
            output.write(
                0u, read(input_a, 0u) + read(input_b, 0u));
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 2u))
            << "two distinct kernel descriptors are a conflicting origin, "
               "never a proof that the callable ABI can omit the resource";
        expect(eq(legalized.inline_info.inlined_call_count, 2u));
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get());
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(eq(count_opcode(
                      luisa::span{compiled.spv_bin},
                      spv::Op::OpFunctionCall),
                  0u))
            << "conflicting descriptor origins must retain the conservative "
               "call-site specialization fallback";
    };

    "spirv_pointer_legalization_outlines_unique_readonly_bindless_origin"_test = [] {
        Callable read = [](BindlessVar bindless,
                           UInt index) noexcept {
            return bindless.buffer<uint32_t>(0u).read(index);
        };
        Kernel1D kernel = [&read](BindlessVar bindless,
                                  BufferUInt output) noexcept {
            output.write(
                0u, read(bindless, 0u) + read(bindless, 1u));
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u))
            << "a uniquely rooted read-only bindless array must use its "
               "kernel descriptor and metadata without inlining";
        expect(eq(legalized.inline_info.inlined_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(
                module.get());
        expect(dialect.succeeded())
            << (dialect.diagnostics.empty() ?
                    "unknown dialect failure" :
                    dialect.diagnostics.front().message);
        lc::spirv::SpirvTargetFeatures features{
            .descriptor_indexing = true,
            .runtime_descriptor_array = true,
            .descriptor_binding_partially_bound = true,
            .storage_buffer_array_non_uniform_indexing = true,
            .descriptor_binding_storage_buffer_update_after_bind = true,
            .storage_buffer_array_dynamic_indexing = true};
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get(), features);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(
                   luisa::span{compiled.spv_bin},
                   spv::Op::OpFunctionCall) > 0u)
            << "the read-only bindless callable must remain outlined";
    };

    "spirv_pointer_legalization_outlines_unique_read_write_bindless_origin"_test = [] {
        Callable update = [](BindlessVar bindless,
                             UInt index) noexcept {
            auto buffer = bindless.buffer<uint32_t>(0u);
            buffer.write(index, buffer.read(index) + 1u);
        };
        Kernel1D kernel = [&update](BindlessVar bindless) noexcept {
            update(bindless, 0u);
            update(bindless, 1u);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u))
            << "a uniquely rooted read/write bindless array must use its "
               "kernel descriptor and metadata without inlining";
        expect(eq(legalized.inline_info.inlined_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(
                module.get());
        expect(dialect.succeeded())
            << (dialect.diagnostics.empty() ?
                    "unknown dialect failure" :
                    dialect.diagnostics.front().message);
        lc::spirv::SpirvTargetFeatures features{
            .descriptor_indexing = true,
            .runtime_descriptor_array = true,
            .descriptor_binding_partially_bound = true,
            .storage_buffer_array_non_uniform_indexing = true,
            .descriptor_binding_storage_buffer_update_after_bind = true,
            .storage_buffer_array_dynamic_indexing = true};
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get(), features);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(
                   luisa::span{compiled.spv_bin},
                   spv::Op::OpFunctionCall) >= 2u)
            << "both read/write bindless calls must remain outlined";
    };

    "spirv_pointer_legalization_outlines_unique_writable_accel_origin"_test = [] {
        Callable update = [](AccelVar accel) noexcept {
            accel.set_instance_user_id(0u, 19u);
        };
        Kernel1D kernel = [&update](AccelVar accel) noexcept {
            update(accel);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u))
            << "the writable instance buffer is a side channel of the same "
               "uniquely rooted accel resource";
        expect(eq(legalized.inline_info.inlined_call_count, 0u));
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        lc::spirv::SpirvTargetFeatures features{};
        features.ray_query = true;
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get(), features);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpFunctionCall) > 0u)
            << "the writable accel callable must remain outlined";
    };

    "spirv_pointer_legalization_outlines_unique_accel_instance_read_origin"_test = [] {
        Callable query = [](AccelVar accel) noexcept {
            return accel.instance_user_id(0u);
        };
        Kernel1D kernel = [&query](AccelVar accel,
                                   BufferUInt output) noexcept {
            output.write(0u, query(accel));
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u))
            << "the readable instance buffer is a side channel of the same "
               "uniquely rooted accel resource";
        expect(eq(legalized.inline_info.inlined_call_count, 0u));
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        lc::spirv::SpirvTargetFeatures features{};
        features.ray_query = true;
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get(), features);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpFunctionCall) > 0u)
            << "the accel instance-query callable must remain outlined";
    };

    "spirv_pointer_legalization_specializes_conflicting_accel_instance_origins"_test = [] {
        Callable query = [](AccelVar accel) noexcept {
            return accel.instance_user_id(0u);
        };
        Kernel1D kernel = [&query](AccelVar accel_a,
                                   AccelVar accel_b,
                                   BufferUInt output) noexcept {
            output.write(
                0u, query(accel_a) + query(accel_b));
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 2u))
            << "two accel instance buffers have conflicting kernel origins";
        expect(eq(legalized.inline_info.inlined_call_count, 2u));
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        lc::spirv::SpirvTargetFeatures features{};
        features.ray_query = true;
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get(), features);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(eq(count_opcode(luisa::span{compiled.spv_bin},
                               spv::Op::OpFunctionCall),
                  0u))
            << "conflicting accel side channels must retain call-site "
               "specialization";
    };

    "spirv_pointer_legalization_preserves_trace_only_accel_callable"_test = [] {
        Callable trace = [](AccelVar accel, Var<Ray> ray) noexcept -> Bool {
            return accel.intersect_any(ray, {});
        };
        Kernel1D kernel = [&trace](AccelVar accel,
                                   BufferUInt output) noexcept {
            auto ray = make_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f));
            output.write(0u, cast<uint32_t>(trace(accel, ray)));
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u))
            << "traversal-only accel callables need no instance-buffer binding";
        expect(eq(legalized.inline_info.inlined_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        lc::spirv::SpirvTargetFeatures features{};
        features.ray_query = true;
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get(), features);
        expect(validates(luisa::span{compiled.spv_bin}));
    };

    "spirv_pointer_legalization_outlines_unique_dual_role_texture_origin"_test = [] {
        Callable update = [](ImageFloat image) noexcept {
            auto coord = make_uint2(0u);
            image.write(coord, image.read(coord));
        };
        Kernel1D kernel = [&update](ImageFloat image) noexcept {
            update(image);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 0u))
            << "read and write texture descriptors are side channels of the "
               "same uniquely rooted texture resource";
        expect(eq(legalized.inline_info.inlined_call_count, 0u));
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        lc::spirv::SpirvTargetFeatures features{};
        features.storage_image_read_without_format = true;
        features.storage_image_write_without_format = true;
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get(), features);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpFunctionCall) > 0u)
            << "the dual-role texture callable must remain outlined";
    };

    "spirv_pointer_legalization_specializes_conflicting_dual_role_texture_origins"_test = [] {
        Callable update = [](ImageFloat image) noexcept {
            auto coord = make_uint2(0u);
            image.write(coord, image.read(coord));
        };
        Kernel1D kernel = [&update](ImageFloat image_a,
                                    ImageFloat image_b) noexcept {
            update(image_a);
            update(image_b);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto destructured =
            destructure_cfg_pass_run_on_module(module.get());
        expect(destructured.succeeded());
        auto legalized =
            lc::spirv::legalize_spirv_pointer_arguments(module.get());
        expect(legalized.succeeded()) << legalized.diagnostic;
        expect(eq(legalized.planned_pointer_call_count, 2u))
            << "two dual-role texture descriptor pairs have conflicting "
               "kernel origins";
        expect(eq(legalized.inline_info.inlined_call_count, 2u));
        expect(eq(legalized.remaining_pointer_call_count, 0u));
        expect(xir_verify_module(module.get()).succeeded());

        lc::spirv::SpirvTargetFeatures features{};
        features.storage_image_read_without_format = true;
        features.storage_image_write_without_format = true;
        auto compiled = compile_exact_xir(
            kernel.function()->function(), module.get(), features);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(eq(count_opcode(luisa::span{compiled.spv_bin},
                               spv::Op::OpFunctionCall),
                  0u))
            << "conflicting texture descriptor pairs must retain call-site "
               "specialization";
    };

    "spirv_pointer_switch_retry_failure_is_deterministic"_test = [] {
        Module module;
        auto *uint_type = Type::of<uint32_t>();
        auto *buffer_type = Type::buffer(uint_type);
        auto *callable = module.create_callable(Type::of<void>());
        auto *callable_buffer =
            callable->create_resource_argument(buffer_type);
        auto *selector = callable->create_value_argument(uint_type);
        auto *entry = callable->create_body_block();

        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.call(uint_type, ResourceReadOp::BUFFER_READ,
                     {callable_buffer, selector});
        auto *switch_inst = builder.switch_(selector);
        auto *recursive_case =
            switch_inst->create_case_block(0u);
        auto *default_block =
            switch_inst->create_default_block();
        auto *merge_block = switch_inst->create_merge_block();
        builder.set_insertion_point(recursive_case);
        builder.call(nullptr, callable,
                     {callable_buffer, selector});
        builder.br(merge_block);
        builder.set_insertion_point(default_block);
        builder.br(merge_block);
        builder.set_insertion_point(merge_block);
        builder.return_void();

        auto *kernel = module.create_kernel();
        auto *kernel_buffer =
            kernel->create_resource_argument(buffer_type);
        auto *kernel_selector =
            kernel->create_value_argument(uint_type);
        auto *kernel_body = kernel->create_body_block();
        builder.set_insertion_point(kernel_body);
        auto *kernel_call = builder.call(
            nullptr, callable,
            {kernel_buffer, kernel_selector});
        auto kernel_call_lock = kernel_call->lock_into<Instruction>();
        builder.return_void();

        auto result =
            lc::spirv::legalize_spirv_pointer_arguments(&module);
        expect(result.status ==
               lc::spirv::SpirvPointerLegalizationStatus::
                   INLINE_RETRY_FAILED);
        expect(eq(result.planned_pointer_call_count, 2u));
        expect(eq(result.destructured_switch_count, 0u))
            << "recursive-call preflight must reject the complete plan before mutation";
        expect(eq(result.inline_info.skipped_recursive_callable_count,
                  1u));
        expect(eq(result.remaining_pointer_call_count, 2u));
        expect(result.diagnostic ==
               "SPIR-V pointer-argument inline retry failed "
               "(remaining=2, structured=0, malformed=0, recursive=1).")
            << "failure diagnostics must be stable and actionable";
        expect(kernel_call_lock->is_linked());
        expect(entry->terminator()->isa<SwitchInst>())
            << "failed preflight must preserve the original structured CFG";
    };

    "spirv_pointer_switch_orphan_failure_is_atomic"_test = [] {
        Module module;
        auto *uint_type = Type::of<uint32_t>();
        auto fixture = make_pointer_switch_callable(
            module, PointerSwitchOrphan::UNTERMINATED);
        XIRBuilder builder;
        auto *kernel = module.create_kernel();
        auto *kernel_selector = kernel->create_value_argument(uint_type);
        builder.set_insertion_point(kernel->create_body_block());
        auto *kernel_value = builder.alloca_shared(uint_type);
        auto *call = builder.call(
            nullptr, fixture.callable,
            {kernel_value, kernel_selector});
        builder.return_void();

        auto result =
            lc::spirv::legalize_spirv_pointer_arguments(&module);
        expect(result.status ==
               lc::spirv::SpirvPointerLegalizationStatus::
                   DESTRUCTURE_FAILED);
        expect(eq(result.planned_pointer_call_count, 1u));
        expect(eq(result.blocking_function_count, 1u));
        expect(eq(result.destructured_blocking_function_count, 0u));
        expect(eq(result.destructured_switch_count, 0u));
        expect(eq(result.remaining_pointer_call_count, 1u));
        expect(result.diagnostic ==
               "SPIR-V pointer-argument fallback rejected 1 blocking "
               "function(s) during atomic destructure preflight; the module "
               "was left unchanged.");
        expect(fixture.entry->terminator() ==
               fixture.reachable_switch);
        expect(!fixture.orphan->is_terminated());
        expect(call->is_linked());
    };

    "spirv_pointer_switch_later_orphan_failure_rolls_back_all_functions"_test = [] {
        Module module;
        auto *uint_type = Type::of<uint32_t>();
        auto first = make_pointer_switch_callable(
            module, PointerSwitchOrphan::NONE);
        auto second = make_pointer_switch_callable(
            module, PointerSwitchOrphan::UNTERMINATED);

        XIRBuilder builder;
        auto *kernel = module.create_kernel();
        auto *kernel_selector = kernel->create_value_argument(uint_type);
        builder.set_insertion_point(kernel->create_body_block());
        auto *kernel_value = builder.alloca_shared(uint_type);
        auto *first_call = builder.call(
            nullptr, first.callable,
            {kernel_value, kernel_selector});
        auto *second_call = builder.call(
            nullptr, second.callable,
            {kernel_value, kernel_selector});
        builder.return_void();

        auto result =
            lc::spirv::legalize_spirv_pointer_arguments(&module);
        expect(result.status ==
               lc::spirv::SpirvPointerLegalizationStatus::
                   DESTRUCTURE_FAILED);
        expect(eq(result.planned_pointer_call_count, 2u));
        expect(eq(result.blocking_function_count, 2u));
        expect(eq(result.destructured_blocking_function_count, 0u));
        expect(eq(result.destructured_switch_count, 0u));
        expect(eq(result.remaining_pointer_call_count, 2u));
        expect(first.entry->terminator() ==
               first.reachable_switch);
        expect(second.entry->terminator() ==
               second.reachable_switch);
        expect(!second.orphan->is_terminated());
        expect(first_call->is_linked());
        expect(second_call->is_linked());
    };

    "spirv_pointer_switch_malformed_orphan_failure_is_atomic"_test = [] {
        Module module;
        auto *uint_type = Type::of<uint32_t>();
        auto fixture = make_pointer_switch_callable(
            module, PointerSwitchOrphan::MALFORMED_SWITCH);
        XIRBuilder builder;
        auto *kernel = module.create_kernel();
        auto *kernel_selector = kernel->create_value_argument(uint_type);
        builder.set_insertion_point(kernel->create_body_block());
        auto *kernel_value = builder.alloca_shared(uint_type);
        auto *call = builder.call(
            nullptr, fixture.callable,
            {kernel_value, kernel_selector});
        builder.return_void();

        auto result =
            lc::spirv::legalize_spirv_pointer_arguments(&module);
        expect(result.status ==
               lc::spirv::SpirvPointerLegalizationStatus::
                   DESTRUCTURE_FAILED);
        expect(eq(result.planned_pointer_call_count, 1u));
        expect(eq(result.blocking_function_count, 1u));
        expect(eq(result.destructured_blocking_function_count, 0u));
        expect(eq(result.destructured_switch_count, 0u));
        expect(eq(result.remaining_pointer_call_count, 1u));
        expect(result.diagnostic ==
               "SPIR-V pointer-argument fallback rejected 1 blocking "
               "function(s) during atomic destructure preflight; the module "
               "was left unchanged.");
        expect(fixture.entry->terminator() ==
               fixture.reachable_switch);
        expect(fixture.orphan->terminator() ==
               fixture.orphan_switch);
        expect(call->is_linked());
    };

    "spirv_pointer_switch_later_malformed_orphan_preserves_all_functions"_test = [] {
        Module module;
        auto *uint_type = Type::of<uint32_t>();
        auto first = make_pointer_switch_callable(
            module, PointerSwitchOrphan::NONE);
        auto second = make_pointer_switch_callable(
            module, PointerSwitchOrphan::MALFORMED_SWITCH);

        XIRBuilder builder;
        auto *kernel = module.create_kernel();
        auto *kernel_selector = kernel->create_value_argument(uint_type);
        builder.set_insertion_point(kernel->create_body_block());
        auto *kernel_value = builder.alloca_shared(uint_type);
        auto *first_call = builder.call(
            nullptr, first.callable,
            {kernel_value, kernel_selector});
        auto *second_call = builder.call(
            nullptr, second.callable,
            {kernel_value, kernel_selector});
        builder.return_void();

        auto result =
            lc::spirv::legalize_spirv_pointer_arguments(&module);
        expect(result.status ==
               lc::spirv::SpirvPointerLegalizationStatus::
                   DESTRUCTURE_FAILED);
        expect(eq(result.planned_pointer_call_count, 2u));
        expect(eq(result.blocking_function_count, 2u));
        expect(eq(result.destructured_blocking_function_count, 0u));
        expect(eq(result.destructured_switch_count, 0u));
        expect(eq(result.remaining_pointer_call_count, 2u));
        expect(first.entry->terminator() ==
               first.reachable_switch);
        expect(second.entry->terminator() ==
               second.reachable_switch);
        expect(second.orphan->terminator() ==
               second.orphan_switch);
        expect(first_call->is_linked());
        expect(second_call->is_linked());
    };
}
