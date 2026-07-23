// Test for the native SPIR-V callable-reference handoff.
// This test covers:
// - exact diagnostics for residual GEP and shared-allocation reference actuals
// - accepted direct local allocations and forwarded reference parameters
// - validator-backed opt0 codegen for every accepted pointer shape

#include "ut/ut.hpp"

#include <cstdlib>
#include <optional>
#include <string>

#include <spirv-tools/libspirv.hpp>

#include <luisa/ast/type.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/verifier.h>

#include "spirv_codegen/dialect.h"
#include "spirv_codegen/entry.h"
#include "spirv_codegen/pointer_legalization.h"

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

[[nodiscard]] CallableFunction *create_reference_writer(
    Module &module, XIRBuilder &builder) noexcept {
    auto *callable = module.create_callable(Type::of<void>());
    auto *reference =
        callable->create_reference_argument(Type::of<uint32_t>());
    builder.set_insertion_point(callable->create_body_block());
    builder.store(reference,
                  module.create_constant_one(Type::of<uint32_t>()));
    builder.return_void();
    return callable;
}

[[nodiscard]] const lc::spirv::SpirvXIRDialectDiagnostic *
find_instruction_diagnostic(
    const lc::spirv::SpirvXIRDialectValidationResult &validation,
    const Instruction *instruction) noexcept {
    for (auto &&diagnostic : validation.diagnostics) {
        if (diagnostic.instruction == instruction) { return &diagnostic; }
    }
    return nullptr;
}

[[nodiscard]] bool validates_vulkan_1_2(
    luisa::span<const uint32_t> words) noexcept {
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    return tools.Validate(words.data(), words.size());
}

[[nodiscard]] size_t count_opcode(
    luisa::span<const uint32_t> words, spv::Op expected) noexcept {
    auto count = size_t{0u};
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = words[offset] >> 16u;
        if (word_count == 0u ||
            word_count > words.size() - offset) {
            break;
        }
        auto opcode =
            static_cast<spv::Op>(words[offset] & 0xffffu);
        count += opcode == expected ? 1u : 0u;
        offset += word_count;
    }
    return count;
}

[[nodiscard]] lc::spirv::SpirvResult compile_opt0(
    luisa::compute::Function ast_kernel,
    const Module &module) {
    ScopedEnvironmentVariable disable_spirv_optimization{
        "LUISA_SPIRV_OPT_LEVEL", "0"};
    ScopedEnvironmentVariable clear_spirv_pass_override{
        "LUISA_SPIRV_OPT_PASSES", nullptr};
    return lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
        ast_kernel, &module,
        ShaderOption{.enable_cache = false});
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_callable_reference_handoff_rejects_shared_alloca"_test = [] {
        Module module;
        XIRBuilder builder;
        auto *callable = create_reference_writer(module, builder);
        auto *kernel = module.create_kernel();
        builder.set_insertion_point(kernel->create_body_block());
        auto *shared = builder.alloca_shared(Type::of<uint32_t>());
        auto *call = builder.call(nullptr, callable, {shared});
        builder.return_void();

        auto generic = xir_verify_module(&module);
        expect(generic.succeeded())
            << "shared references are valid generic XIR and must fail only at the SPIR-V ABI boundary";
        auto reference_actual =
            lc::spirv::validate_spirv_callable_reference_actual(shared);
        expect(reference_actual.status ==
               lc::spirv::SpirvCallableReferenceActualStatus::
                   SHARED_ALLOCATION);
        expect(reference_actual.diagnostic ==
               "a shared/workgroup allocation cannot be passed through a Function-storage callable parameter");

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        auto *diagnostic =
            find_instruction_diagnostic(validation, call);
        expect(diagnostic != nullptr);
        if (diagnostic != nullptr) {
            expect(diagnostic->message ==
                   "Native XIR-to-SPIR-V callable reference argument 0 for '<unnamed>' is unsupported: a shared/workgroup allocation cannot be passed through a Function-storage callable parameter; specialize this call before codegen.");
        }
    };

    "spirv_callable_reference_handoff_rejects_gep"_test = [] {
        Module module;
        XIRBuilder builder;
        auto *callable = create_reference_writer(module, builder);
        auto *kernel = module.create_kernel();
        builder.set_insertion_point(kernel->create_body_block());
        auto *array_type =
            Type::array(Type::of<uint32_t>(), 2u);
        auto *local = builder.alloca_local(array_type);
        auto *element = builder.gep(
            Type::of<uint32_t>(), local,
            {module.create_constant_zero(Type::of<uint32_t>())});
        auto *call = builder.call(nullptr, callable, {element});
        builder.return_void();

        auto generic = xir_verify_module(&module);
        expect(generic.succeeded())
            << "GEP reference actuals are valid generic XIR and must fail only at the SPIR-V ABI boundary";
        auto reference_actual =
            lc::spirv::validate_spirv_callable_reference_actual(element);
        expect(reference_actual.status ==
               lc::spirv::SpirvCallableReferenceActualStatus::
                   DERIVED_POINTER);
        expect(reference_actual.diagnostic ==
               "a GEP-derived pointer cannot be passed through a callable parameter without VariablePointers");

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        auto *diagnostic =
            find_instruction_diagnostic(validation, call);
        expect(diagnostic != nullptr);
        if (diagnostic != nullptr) {
            expect(diagnostic->message ==
                   "Native XIR-to-SPIR-V callable reference argument 0 for '<unnamed>' is unsupported: a GEP-derived pointer cannot be passed through a callable parameter without VariablePointers; specialize this call before codegen.");
        }
    };

    "spirv_callable_reference_handoff_accepts_local_alloca"_test = [] {
        Module module;
        XIRBuilder builder;
        auto *callable = create_reference_writer(module, builder);
        auto *xir_kernel = module.create_kernel();
        builder.set_insertion_point(xir_kernel->create_body_block());
        auto *local = builder.alloca_local(Type::of<uint32_t>());
        builder.store(local,
                      module.create_constant_zero(Type::of<uint32_t>()));
        builder.call(nullptr, callable, {local});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        expect(lc::spirv::validate_spirv_callable_reference_actual(local)
                   .succeeded());
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded());

        Kernel1D kernel = []() noexcept {};
        xir_kernel->set_block_size(
            kernel.function()->function().block_size());
        auto result = compile_opt0(
            kernel.function()->function(), module);
        expect(validates_vulkan_1_2(
            luisa::span{result.spv_bin}));
        expect(count_opcode(luisa::span{result.spv_bin},
                            spv::Op::OpFunctionCall) > 0u);
    };

    "spirv_callable_reference_handoff_accepts_forwarded_reference"_test = [] {
        Module module;
        XIRBuilder builder;
        auto *writer = create_reference_writer(module, builder);
        auto *relay = module.create_callable(Type::of<void>());
        auto *forwarded =
            relay->create_reference_argument(Type::of<uint32_t>());
        builder.set_insertion_point(relay->create_body_block());
        builder.call(nullptr, writer, {forwarded});
        builder.return_void();

        auto *xir_kernel = module.create_kernel();
        builder.set_insertion_point(xir_kernel->create_body_block());
        auto *local = builder.alloca_local(Type::of<uint32_t>());
        builder.store(local,
                      module.create_constant_zero(Type::of<uint32_t>()));
        builder.call(nullptr, relay, {local});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        expect(lc::spirv::validate_spirv_callable_reference_actual(
                   forwarded)
                   .succeeded());
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded());

        Kernel1D kernel = []() noexcept {};
        xir_kernel->set_block_size(
            kernel.function()->function().block_size());
        auto result = compile_opt0(
            kernel.function()->function(), module);
        expect(validates_vulkan_1_2(
            luisa::span{result.spv_bin}));
        expect(count_opcode(luisa::span{result.spv_bin},
                            spv::Op::OpFunctionCall) >= 2u);
    };
}
