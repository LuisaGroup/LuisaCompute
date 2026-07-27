#include "ut/ut.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

#include "spirv_codegen/arithmetic_support.h"
#include "spirv_codegen/call_graph_validation.h"
#include "spirv_codegen/control_flow_plan.h"
#include "spirv_codegen/dialect.h"

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;

namespace {

template<typename Enum>
void expect_complete_matrix(Enum last, size_t expected_unsupported,
                            size_t expected_semantic_no_ops = 0u) {
    using Underlying = std::underlying_type_t<Enum>;
    auto last_value = static_cast<Underlying>(last);
    size_t unsupported_count = 0u;
    size_t semantic_no_op_count = 0u;
    for (Underlying raw = 0; raw <= last_value; ++raw) {
        auto info = lc::spirv::spirv_xir_dialect_support(
            static_cast<Enum>(raw));
        expect(info.known())
            << "every current opcode must have an explicit SPIR-V dialect classification";
        unsupported_count +=
            info.support == lc::spirv::SpirvXIRDialectSupport::UNSUPPORTED;
        semantic_no_op_count +=
            info.support == lc::spirv::SpirvXIRDialectSupport::SEMANTIC_NO_OP;
        if (info.support != lc::spirv::SpirvXIRDialectSupport::SUPPORTED) {
            expect(!info.reason.empty())
                << "non-supported classifications require an exact reason";
        }
    }
    expect(unsupported_count == expected_unsupported);
    expect(semantic_no_op_count == expected_semantic_no_ops);

    // All matrices deliberately fail closed outside their classified range.
    auto unknown = lc::spirv::spirv_xir_dialect_support(
        static_cast<Enum>(last_value + 1));
    expect(unknown.support == lc::spirv::SpirvXIRDialectSupport::UNKNOWN);
    expect(!unknown.accepted());
}

template<typename Enum>
void expect_unsupported(Enum op, luisa::string_view reason_fragment) {
    auto info = lc::spirv::spirv_xir_dialect_support(op);
    expect(info.support == lc::spirv::SpirvXIRDialectSupport::UNSUPPORTED);
    expect(!info.accepted());
    expect(info.reason.find(reason_fragment) != luisa::string_view::npos)
        << "the rejection must explain the unsupported semantic boundary";
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

void expect_generic_xir_valid(
    const Module &module, const char *boundary) noexcept {
    auto verification = xir_verify_module(&module);
    expect(verification.succeeded()) << boundary;
}

void expect_generic_xir_invalid_at(
    const Module &module,
    const luisa::compute::xir::Function *expected_function,
    const BasicBlock *expected_block,
    const Instruction *expected_instruction,
    luisa::string_view expected_fragment) noexcept {
    auto verification = xir_verify_module(&module);
    expect(!verification.succeeded());
    expect(eq(verification.errors.size(), 1u));
    if (verification.errors.size() != 1u) { return; }
    auto &&error = verification.errors.front();
    expect(error.function == expected_function);
    expect(error.block == expected_block);
    expect(error.instruction == expected_instruction);
    expect(error.message.find(expected_fragment) != luisa::string::npos);
}

void expect_generic_xir_error_at(
    const Module &module,
    const luisa::compute::xir::Function *expected_function,
    const BasicBlock *expected_block,
    const Instruction *expected_instruction,
    luisa::string_view expected_fragment) noexcept {
    auto verification = xir_verify_module(&module);
    expect(!verification.succeeded());
    auto found = false;
    for (auto &&error : verification.errors) {
        found |= error.function == expected_function &&
                 error.block == expected_block &&
                 error.instruction == expected_instruction &&
                 error.message.find(expected_fragment) !=
                     luisa::string::npos;
    }
    expect(found)
        << "generic XIR verification must report the exact malformed memory access";
}

[[nodiscard]] const lc::spirv::SpirvXIRDialectDiagnostic *
expect_only_diagnostic_location(
    const lc::spirv::SpirvXIRDialectValidationResult &result,
    const luisa::compute::xir::Function *expected_function,
    const BasicBlock *expected_block,
    const Instruction *expected_instruction) noexcept {
    expect(!result.succeeded());
    expect(eq(result.diagnostics.size(), 1u));
    if (result.diagnostics.size() != 1u) { return nullptr; }
    auto &&diagnostic = result.diagnostics.front();
    expect(diagnostic.function == expected_function);
    expect(diagnostic.block == expected_block);
    expect(diagnostic.instruction == expected_instruction);
    return &diagnostic;
}

void expect_diagnostics_at(
    const lc::spirv::SpirvXIRDialectValidationResult &result,
    size_t expected_count,
    const luisa::compute::xir::Function *expected_function,
    const BasicBlock *expected_block,
    const Instruction *expected_instruction) noexcept {
    expect(!result.succeeded());
    expect(eq(result.diagnostics.size(), expected_count));
    for (auto &&diagnostic : result.diagnostics) {
        expect(diagnostic.function == expected_function);
        expect(diagnostic.block == expected_block);
        expect(diagnostic.instruction == expected_instruction);
    }
}

void expect_only_diagnostic_at(
    const lc::spirv::SpirvXIRDialectValidationResult &result,
    const luisa::compute::xir::Function *expected_function,
    const BasicBlock *expected_block,
    const Instruction *expected_instruction,
    luisa::string_view expected_fragment) noexcept {
    if (auto *diagnostic = expect_only_diagnostic_location(
            result, expected_function, expected_block,
            expected_instruction)) {
        expect(diagnostic->message.find(expected_fragment) !=
               luisa::string::npos);
    }
}

void expect_diagnostic_at(
    const lc::spirv::SpirvXIRDialectValidationResult &result,
    const luisa::compute::xir::Function *expected_function,
    const BasicBlock *expected_block,
    const Instruction *expected_instruction,
    luisa::string_view expected_fragment) noexcept {
    expect(!result.succeeded());
    auto found = false;
    for (auto &&diagnostic : result.diagnostics) {
        found |= diagnostic.function == expected_function &&
                 diagnostic.block == expected_block &&
                 diagnostic.instruction == expected_instruction &&
                 diagnostic.message.find(expected_fragment) !=
                     luisa::string::npos;
    }
    expect(found)
        << "SPIR-V dialect validation must report the exact malformed instruction";
}

void expect_only_diagnostic(
    const lc::spirv::SpirvXIRDialectValidationResult &result,
    const luisa::compute::xir::Function *expected_function,
    luisa::string_view expected) noexcept {
    if (auto *diagnostic = expect_only_diagnostic_location(
            result, expected_function, nullptr, nullptr)) {
        expect(diagnostic->message == expected);
    }
}

struct RayQueryTestContext {
    KernelFunction *kernel;
    ResourceArgument *accel;
    luisa::compute::xir::Constant *ray;
    luisa::compute::xir::Constant *mask;
    const Type *query_type;
};

[[nodiscard]] RayQueryTestContext make_ray_query_test_context(
    Module &module) noexcept {
    auto *kernel = module.create_kernel();
    return {
        .kernel = kernel,
        .accel = kernel->create_resource_argument(Type::of<Accel>()),
        .ray = module.create_constant_zero(Type::of<Ray>()),
        .mask = module.create_constant_one(Type::of<uint32_t>()),
        .query_type = Type::custom("LC_RayQueryAny"),
    };
}

[[nodiscard]] ResourceQueryInst *emit_ray_query_initializer(
    XIRBuilder &builder,
    const RayQueryTestContext &context) noexcept {
    return builder.call(
        context.query_type,
        ResourceQueryOp::RAY_TRACING_QUERY_ANY,
        {context.accel, context.ray, context.mask});
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_xir_dialect_matrix_is_complete_and_fails_closed"_test = [] {
        expect_complete_matrix(AllocaOp::SHARED, 0u);
        expect_complete_matrix(ArithmeticOp::EXTRACT, 0u);
        expect_complete_matrix(AtomicOp::FETCH_MAX, 0u);
        expect_complete_matrix(
            luisa::compute::xir::CastOp::BITWISE_CAST, 0u);
        expect_complete_matrix(
            ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR, 6u);
        expect_complete_matrix(ResourceReadOp::DEVICE_ADDRESS_READ, 1u);
        expect_complete_matrix(
            ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT, 3u);
        expect_complete_matrix(ThreadGroupOp::SYNCHRONIZE_BLOCK, 2u, 1u);
        expect_complete_matrix(
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, 0u);
        expect_complete_matrix(
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED, 0u);
        expect_complete_matrix(DerivedSpecialRegisterTag::DISPATCH_SIZE, 2u);
        expect_complete_matrix(
            DerivedInstructionTag::INDEXED_BRANCH, 14u, 1u);
    };

    "spirv_xir_kernel_abi_requires_exact_ast_xir_pairing"_test = [] {
        Kernel1D ast_kernel = [](BufferUInt, UInt) noexcept {};
        auto ast_function = ast_kernel.function()->function();
        auto make_kernel = [&](Module &module) {
            auto *kernel = module.create_kernel();
            kernel->set_block_size(ast_function.block_size());
            kernel->create_resource_argument(
                Type::buffer(Type::of<uint32_t>()));
            kernel->create_value_argument(Type::of<uint32_t>());
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            builder.return_void();
            return kernel;
        };

        Module valid_module;
        static_cast<void>(make_kernel(valid_module));
        expect(lc::spirv::validate_spirv_xir_kernel_abi(
                   ast_function, &valid_module)
                   .succeeded());

        Module no_kernel;
        expect(lc::spirv::validate_spirv_xir_kernel_abi(
                   ast_function, &no_kernel)
                   .status ==
               lc::spirv::SpirvXIRKernelABIStatus::
                   KERNEL_DEFINITION_COUNT_MISMATCH);

        Module two_kernels;
        static_cast<void>(make_kernel(two_kernels));
        static_cast<void>(make_kernel(two_kernels));
        expect(lc::spirv::validate_spirv_xir_kernel_abi(
                   ast_function, &two_kernels)
                   .status ==
               lc::spirv::SpirvXIRKernelABIStatus::
                   KERNEL_DEFINITION_COUNT_MISMATCH);

        Module wrong_block_size;
        auto *block_kernel = make_kernel(wrong_block_size);
        auto block_size = ast_function.block_size();
        auto alternative_block_size =
            block_size.x == 32u && block_size.y == 1u &&
                    block_size.z == 1u ?
                make_uint3(64u, 1u, 1u) :
                make_uint3(32u, 1u, 1u);
        expect(KernelFunction::is_valid_block_size(
            alternative_block_size));
        block_kernel->set_block_size(alternative_block_size);
        expect(lc::spirv::validate_spirv_xir_kernel_abi(
                   ast_function, &wrong_block_size)
                   .status ==
               lc::spirv::SpirvXIRKernelABIStatus::BLOCK_SIZE_MISMATCH);

        Module wrong_count;
        auto *count_kernel = wrong_count.create_kernel();
        count_kernel->set_block_size(ast_function.block_size());
        count_kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        XIRBuilder builder;
        builder.set_insertion_point(count_kernel->create_body_block());
        builder.return_void();
        expect(lc::spirv::validate_spirv_xir_kernel_abi(
                   ast_function, &wrong_count)
                   .status ==
               lc::spirv::SpirvXIRKernelABIStatus::ARGUMENT_COUNT_MISMATCH);

        Module wrong_type;
        auto *type_kernel = wrong_type.create_kernel();
        type_kernel->set_block_size(ast_function.block_size());
        type_kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        type_kernel->create_value_argument(Type::of<int32_t>());
        builder.set_insertion_point(type_kernel->create_body_block());
        builder.return_void();
        auto type_result = lc::spirv::validate_spirv_xir_kernel_abi(
            ast_function, &wrong_type);
        expect(type_result.status ==
               lc::spirv::SpirvXIRKernelABIStatus::ARGUMENT_TYPE_MISMATCH);
        expect(eq(type_result.argument_index, 1u));

        Module wrong_kind;
        auto *kind_kernel = wrong_kind.create_kernel();
        kind_kernel->set_block_size(ast_function.block_size());
        kind_kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        kind_kernel->create_reference_argument(Type::of<uint32_t>());
        builder.set_insertion_point(kind_kernel->create_body_block());
        builder.return_void();
        auto kind_result = lc::spirv::validate_spirv_xir_kernel_abi(
            ast_function, &wrong_kind);
        expect(kind_result.status ==
               lc::spirv::SpirvXIRKernelABIStatus::ARGUMENT_KIND_MISMATCH);
        expect(eq(kind_result.argument_index, 1u));
    };

    "spirv_xir_raw_conditional_branch_is_rejected_at_handoff"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *entry = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        builder.return_void();
        builder.set_insertion_point(false_block);
        builder.return_void();

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation, "raw ConditionalBranch"));
        expect(has_diagnostic(validation, "restructure_cfg"));
    };

    "spirv_xir_remaining_reg2mem_spill_is_rejected_at_handoff"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *alloca = builder.alloca_local(Type::of<int32_t>());
        alloca->create_metadata<Reg2MemSpillMD>()->set_kind(
            Reg2MemSpillKind::PHI);
        auto *zero = module.create_constant_zero(Type::of<int32_t>());
        builder.store(alloca, zero);
        builder.load(Type::of<int32_t>(), alloca);
        builder.return_void();

        expect_generic_xir_valid(
            module, "reg2mem memory form is valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, kernel, entry, alloca,
            "remaining phi reg2mem spill");
    };

    "spirv_xir_memory_access_requires_exact_lvalue_rvalue_types"_test = [] {
        {
            Module module;
            auto *kernel = module.create_kernel();
            auto *entry = kernel->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(entry);
            auto *declared_slot = builder.alloca_local(Type::of<float>());
            auto *wrong_slot = builder.alloca_local(Type::of<uint32_t>());
            auto *load = builder.load(Type::of<float>(), declared_slot);
            load->set_variable(wrong_slot);
            builder.return_void();

            expect_generic_xir_error_at(
                module, kernel, entry, load,
                "Load variable or result type is invalid");
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect_diagnostic_at(
                validation, kernel, entry, load,
                "type exactly matches the result");
        }
        {
            Module module;
            auto *kernel = module.create_kernel();
            auto *entry = kernel->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(entry);
            auto *slot = builder.alloca_local(Type::of<uint2>());
            auto *declared_value =
                module.create_constant_one(Type::of<uint2>());
            auto *wrong_value =
                module.create_constant_one(Type::of<uint32_t>());
            auto *store = builder.store(slot, declared_value);
            store->set_value(wrong_value);
            builder.return_void();

            expect_generic_xir_error_at(
                module, kernel, entry, store,
                "Store variable or value type is invalid");
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect_diagnostic_at(
                validation, kernel, entry, store,
                "rvalue of exactly the same type");
        }
        {
            Module module;
            auto *kernel = module.create_kernel();
            auto *entry = kernel->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(entry);
            auto *destination =
                builder.alloca_local(Type::of<uint32_t>());
            auto *source = builder.alloca_local(Type::of<uint32_t>());
            auto *store = builder.store(
                destination,
                module.create_constant_zero(Type::of<uint32_t>()));
            store->set_value(source);
            builder.return_void();

            expect_generic_xir_error_at(
                module, kernel, entry, store,
                "Store variable or value type is invalid");
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect_diagnostic_at(
                validation, kernel, entry, store,
                "an lvalue address and an rvalue");
        }
    };

    "spirv_xir_canonical_loop_prepare_conditional_is_accepted"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.cond_br(condition, body, merge);
        builder.set_insertion_point(body);
        builder.br(update);
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();

        auto prepare_plan = lc::spirv::plan_spirv_loop_prepare(loop);
        expect(prepare_plan.succeeded());
        expect(prepare_plan.kind ==
               lc::spirv::SpirvLoopPrepareKind::CONDITIONAL);
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded());
    };

    "spirv_xir_canonical_loop_prepare_branch_is_accepted"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.br(body);
        builder.set_insertion_point(body);
        builder.br(update);
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();

        auto prepare_plan = lc::spirv::plan_spirv_loop_prepare(loop);
        expect(prepare_plan.succeeded());
        expect(prepare_plan.kind ==
               lc::spirv::SpirvLoopPrepareKind::UNCONDITIONAL);
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded());
    };

    "spirv_xir_loop_prepare_classifier_rejects_malformed_role_operand"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *non_block =
            kernel->create_value_argument(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.br(body);
        builder.set_insertion_point(body);
        builder.br(update);
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();

        loop->set_operand_count(0u);
        auto missing = lc::spirv::plan_spirv_loop_prepare(loop);
        expect(!missing.succeeded());
        expect(missing.diagnostic.find("exactly one") !=
               luisa::string::npos);

        loop->set_operand_count(1u);
        loop->set_operand(LoopInst::operand_index_prepare_block, non_block);
        auto wrong_kind = lc::spirv::plan_spirv_loop_prepare(loop);
        expect(!wrong_kind.succeeded());
        expect(wrong_kind.diagnostic.find("non-null BasicBlock") !=
               luisa::string::npos);
    };

    "spirv_xir_loop_prepare_classifier_rejects_noncanonical_conditional_targets"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        auto *branch = builder.cond_br(condition, body, merge);
        builder.set_insertion_point(body);
        builder.br(update);
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();

        branch->set_operand(
            ConditionalBranchTerminatorInstruction::operand_index_true_target,
            merge);
        branch->set_operand(
            ConditionalBranchTerminatorInstruction::operand_index_false_target,
            body);
        auto reversed = lc::spirv::plan_spirv_loop_prepare(loop);
        expect(!reversed.succeeded());
        expect(reversed.diagnostic.find(
                   "ConditionalBranch(bool, Loop.body, Loop.merge)") !=
               luisa::string::npos);

        branch->set_operand(
            ConditionalBranchTerminatorInstruction::operand_index_true_target,
            body);
        branch->set_operand(
            ConditionalBranchTerminatorInstruction::operand_index_false_target,
            update);
        auto wrong_exit = lc::spirv::plan_spirv_loop_prepare(loop);
        expect(!wrong_exit.succeeded());
        expect(wrong_exit.diagnostic.find(
                   "ConditionalBranch(bool, Loop.body, Loop.merge)") !=
               luisa::string::npos);
    };

    "spirv_xir_dialect_validates_raw_structured_role_blocks"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.cond_br(condition, body, merge);
        builder.set_insertion_point(body);
        builder.break_(merge);
        // Loop.update is a raw structured role pointer. The ordinary CFG above
        // never reaches it, but the physical planner still owns its identity;
        // dialect validation must therefore inspect it rather than depend on
        // optional post-restructure inactive-payload cleanup.
        builder.set_insertion_point(update);
        auto *debug_break = builder.debug_break();
        builder.return_void();
        builder.set_insertion_point(merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module,
                   {.require_unique_merge_blocks = true,
                    .require_canonical_break_continue_targets = true})
                   .succeeded());
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(eq(validation.diagnostics.size(), 1u));
        if (!validation.diagnostics.empty()) {
            auto &&diagnostic = validation.diagnostics.front();
            expect(diagnostic.block == update);
            expect(diagnostic.instruction == debug_break);
            expect(diagnostic.message.find(
                       "no debug-break instruction contract") !=
                   luisa::string::npos);
        }
    };

    "spirv_xir_dialect_reports_foreign_raw_roles_nonfatally"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *local_update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.cond_br(condition, body, merge);
        builder.set_insertion_point(body);
        builder.br(local_update);
        builder.set_insertion_point(local_update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();

        auto *callable = module.create_callable(Type::of<void>());
        auto *foreign_update = callable->create_body_block();
        builder.set_insertion_point(foreign_update);
        builder.return_void();

        // Loop.update is a raw pointer rather than an ordinary operand. A
        // malformed foreign role must become a verifier diagnostic; dialect
        // validation is a reporting boundary and must never assert here.
        loop->set_update_block(foreign_update);
        auto call_graph =
            lc::spirv::validate_spirv_reachable_call_graph(&module);
        expect(!call_graph.succeeded());
        expect(call_graph.functions_post_order.empty())
            << "an invalid structural closure must not expose a partial emission order";
        expect(eq(call_graph.diagnostics.size(), 1u));
        if (call_graph.diagnostics.size() == 1u) {
            auto &&diagnostic = call_graph.diagnostics.front();
            expect(diagnostic.function == kernel);
            expect(diagnostic.block == loop->parent_block());
            expect(diagnostic.instruction == loop);
            expect(diagnostic.message.find(
                       "foreign structural block") !=
                   luisa::string::npos);
        }
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation,
                              "Loop has an invalid owned block"));
    };

    "spirv_xir_dialect_reports_missing_body_nonfatally"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto call_graph =
            lc::spirv::validate_spirv_reachable_call_graph(&module);
        expect(!call_graph.succeeded());
        expect(call_graph.functions_post_order.empty());
        expect(eq(call_graph.diagnostics.size(), 1u));
        if (call_graph.diagnostics.size() == 1u) {
            auto &&diagnostic = call_graph.diagnostics.front();
            expect(diagnostic.function == kernel);
            expect(diagnostic.block == nullptr);
            expect(diagnostic.instruction == nullptr);
            expect(diagnostic.message.find("body block") !=
                   luisa::string::npos);
        }
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation, "body block"));
    };

    "spirv_xir_call_graph_rejects_null_required_role_nonfatally"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        builder.set_insertion_point(true_block);
        builder.return_void();
        builder.set_insertion_point(false_block);
        builder.return_void();

        auto call_graph =
            lc::spirv::validate_spirv_reachable_call_graph(&module);
        expect(!call_graph.succeeded());
        expect(call_graph.functions_post_order.empty());
        expect(eq(call_graph.diagnostics.size(), 1u));
        if (call_graph.diagnostics.size() == 1u) {
            auto &&diagnostic = call_graph.diagnostics.front();
            expect(diagnostic.function == kernel);
            expect(diagnostic.block == selection->parent_block());
            expect(diagnostic.instruction == selection);
            expect(diagnostic.message.find("If") != luisa::string::npos);
            expect(diagnostic.message.find("non-null merge") !=
                   luisa::string::npos);
        }
    };

    "spirv_xir_call_graph_rejects_non_block_required_operand_nonfatally"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        auto *merge = selection->create_merge_block();
        selection->set_operand(
            ConditionalBranchTerminatorInstruction::
                operand_index_true_target,
            condition);
        builder.set_insertion_point(true_block);
        builder.return_void();
        builder.set_insertion_point(false_block);
        builder.return_void();
        builder.set_insertion_point(merge);
        builder.return_void();

        auto call_graph =
            lc::spirv::validate_spirv_reachable_call_graph(&module);
        expect(!call_graph.succeeded());
        expect(call_graph.functions_post_order.empty());
        expect(eq(call_graph.diagnostics.size(), 1u));
        if (call_graph.diagnostics.size() == 1u) {
            auto &&diagnostic = call_graph.diagnostics.front();
            expect(diagnostic.function == kernel);
            expect(diagnostic.block == selection->parent_block());
            expect(diagnostic.instruction == selection);
            expect(diagnostic.message.find("true target") !=
                   luisa::string::npos);
            expect(diagnostic.message.find("non-block") !=
                   luisa::string::npos);
        }
    };

    "spirv_xir_call_graph_failure_discards_completed_roots"_test = [] {
        Module module;
        auto *leaf = module.create_callable(Type::of<void>());
        XIRBuilder builder;
        builder.set_insertion_point(leaf->create_body_block());
        builder.return_void();

        auto *valid_kernel = module.create_kernel();
        builder.set_insertion_point(valid_kernel->create_body_block());
        builder.call(Type::of<void>(), leaf, {});
        builder.return_void();

        auto *invalid_kernel = module.create_kernel();
        auto call_graph =
            lc::spirv::validate_spirv_reachable_call_graph(&module);
        expect(!call_graph.succeeded());
        expect(call_graph.functions_post_order.empty())
            << "a failure in any root must discard every completed root and callee";
        expect(eq(call_graph.diagnostics.size(), 1u));
        if (call_graph.diagnostics.size() == 1u) {
            expect(call_graph.diagnostics.front().function == invalid_kernel);
            expect(call_graph.diagnostics.front().message.find("body block") !=
                   luisa::string::npos);
        }
    };

    "spirv_xir_call_graph_freezes_callee_before_caller_diamond"_test = [] {
        Module module;
        XIRBuilder builder;
        auto *leaf = module.create_callable(Type::of<void>());
        builder.set_insertion_point(leaf->create_body_block());
        builder.return_void();

        auto *left = module.create_callable(Type::of<void>());
        builder.set_insertion_point(left->create_body_block());
        builder.call(Type::of<void>(), leaf, {});
        builder.return_void();

        auto *right = module.create_callable(Type::of<void>());
        builder.set_insertion_point(right->create_body_block());
        builder.call(Type::of<void>(), leaf, {});
        builder.return_void();

        auto *kernel = module.create_kernel();
        builder.set_insertion_point(kernel->create_body_block());
        builder.call(Type::of<void>(), left, {});
        builder.call(Type::of<void>(), right, {});
        builder.return_void();

        auto call_graph =
            lc::spirv::validate_spirv_reachable_call_graph(&module);
        expect(call_graph.succeeded());
        expect(call_graph.diagnostics.empty());
        expect(eq(call_graph.functions_post_order.size(), 4u));
        if (call_graph.functions_post_order.size() == 4u) {
            auto position = [&](const luisa::compute::xir::Function *function) noexcept {
                for (auto i = size_t{0u};
                     i < call_graph.functions_post_order.size(); ++i) {
                    if (call_graph.functions_post_order[i] == function) {
                        return i;
                    }
                }
                return call_graph.functions_post_order.size();
            };
            auto leaf_position = position(leaf);
            auto left_position = position(left);
            auto right_position = position(right);
            auto kernel_position = position(kernel);
            expect(leaf_position < call_graph.functions_post_order.size());
            expect(left_position < call_graph.functions_post_order.size());
            expect(right_position < call_graph.functions_post_order.size());
            expect(kernel_position < call_graph.functions_post_order.size());
            expect(leaf_position < left_position);
            expect(leaf_position < right_position);
            expect(left_position < kernel_position);
            expect(right_position < kernel_position);
            expect(left_position != right_position)
                << "independent siblings must both appear exactly once; their "
                   "relative order is not part of the call-graph contract";
        }
    };

    "spirv_xir_dialect_rejects_every_planner_role_shape_nonfatally"_test = [] {
        {
            Module module;
            auto *kernel = module.create_kernel();
            auto *condition =
                kernel->create_value_argument(Type::of<bool>());
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            auto *selection = builder.if_(condition);
            auto *true_block = selection->create_true_block();
            auto *false_block = selection->create_false_block();
            builder.set_insertion_point(true_block);
            builder.return_void();
            builder.set_insertion_point(false_block);
            builder.return_void();
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect(!validation.succeeded());
            expect(has_diagnostic(validation, "If"));
            expect(has_diagnostic(validation, "non-null merge"));
        }
        {
            Module module;
            auto *kernel = module.create_kernel();
            auto *condition =
                kernel->create_value_argument(Type::of<bool>());
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            auto *loop = builder.loop();
            auto *prepare = loop->create_prepare_block();
            auto *body = loop->create_body_block();
            auto *unused_update = loop->create_update_block();
            auto *merge = loop->create_merge_block();
            loop->set_update_block(prepare);
            builder.set_insertion_point(prepare);
            builder.cond_br(condition, body, merge);
            builder.set_insertion_point(body);
            builder.br(prepare);
            builder.set_insertion_point(unused_update);
            builder.return_void();
            builder.set_insertion_point(merge);
            builder.return_void();
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect(!validation.succeeded());
            expect(has_diagnostic(validation, "distinct Loop owner"));
            expect(has_diagnostic(validation, "multiple Loop.prepare"));
        }
        {
            Module module;
            auto *kernel = module.create_kernel();
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            auto *loop = builder.simple_loop();
            auto *body = loop->create_body_block();
            auto *unused_merge = loop->create_merge_block();
            loop->set_merge_block(body);
            builder.set_insertion_point(body);
            builder.return_void();
            builder.set_insertion_point(unused_merge);
            builder.return_void();
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect(!validation.succeeded());
            expect(has_diagnostic(validation,
                                  "distinct SimpleLoop owner"));
        }
        {
            Module module;
            auto *kernel = module.create_kernel();
            auto *selector =
                kernel->create_value_argument(Type::of<uint32_t>());
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            auto *switch_inst = builder.switch_(selector);
            auto *case_block = switch_inst->create_case_block(0u);
            auto *default_block = switch_inst->create_default_block();
            builder.set_insertion_point(case_block);
            builder.return_void();
            builder.set_insertion_point(default_block);
            builder.return_void();
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect(!validation.succeeded());
            expect(has_diagnostic(validation, "Switch"));
            expect(has_diagnostic(validation, "non-null merge"));
        }
        {
            Module module;
            auto *kernel = module.create_kernel();
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            auto *loop = builder.loop();
            auto *prepare = loop->create_prepare_block();
            auto *body = loop->create_body_block();
            auto *update = loop->create_update_block();
            auto *merge = loop->create_merge_block();
            builder.set_insertion_point(prepare);
            builder.br(merge);
            builder.set_insertion_point(body);
            builder.br(update);
            builder.set_insertion_point(update);
            builder.br(prepare);
            builder.set_insertion_point(merge);
            builder.return_void();
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect(!validation.succeeded());
            expect(has_diagnostic(
                validation,
                "canonical unconditional Loop.prepare shape"));
            expect(has_diagnostic(validation,
                                  "Branch(Loop.body)"));
        }
        {
            Module module;
            auto *kernel = module.create_kernel();
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            auto *loop = builder.loop();
            auto *prepare = loop->create_prepare_block();
            auto *body = loop->create_body_block();
            auto *update = loop->create_update_block();
            auto *merge = loop->create_merge_block();
            builder.set_insertion_point(prepare);
            builder.return_void();
            builder.set_insertion_point(body);
            builder.br(update);
            builder.set_insertion_point(update);
            builder.br(prepare);
            builder.set_insertion_point(merge);
            builder.return_void();
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect(!validation.succeeded());
            expect(has_diagnostic(
                validation,
                "Branch(Loop.body) or ConditionalBranch"));
        }
        {
            Module module;
            auto *kernel = module.create_kernel();
            auto *condition =
                kernel->create_value_argument(Type::of<bool>());
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            auto *outer = builder.if_(condition);
            auto *outer_true = outer->create_true_block();
            auto *outer_false = outer->create_false_block();
            auto *shared_merge = outer->create_merge_block();
            builder.set_insertion_point(outer_true);
            auto *inner = builder.if_(condition);
            auto *inner_true = inner->create_true_block();
            auto *inner_false = inner->create_false_block();
            inner->set_merge_block(shared_merge);
            builder.set_insertion_point(inner_true);
            builder.br(shared_merge);
            builder.set_insertion_point(inner_false);
            builder.br(shared_merge);
            builder.set_insertion_point(outer_false);
            builder.br(shared_merge);
            builder.set_insertion_point(shared_merge);
            builder.return_void();
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect(!validation.succeeded());
            expect(has_diagnostic(validation,
                                  "exactly one owner"));
        }
    };

    "spirv_xir_true_orphan_loop_cannot_own_active_raw_branch"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *entry = kernel->create_body_block();
        auto *body = kernel->create_basic_block();
        auto *update = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *orphan_owner = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.cond_br(condition, body, merge);
        builder.set_insertion_point(body);
        builder.return_void();
        builder.set_insertion_point(update);
        builder.return_void();
        builder.set_insertion_point(merge);
        builder.return_void();
        builder.set_insertion_point(orphan_owner);
        auto *orphan_loop = builder.loop();
        orphan_loop->set_prepare_block(entry);
        orphan_loop->set_body_block(body);
        orphan_loop->set_update_block(update);
        orphan_loop->set_merge_block(merge);

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation, "raw ConditionalBranch"));
    };

    "spirv_xir_switch_planner_precondition_is_nonfatal"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *switch_inst = builder.switch_(selector);
        auto *self_cycle = switch_inst->create_case_block(0u);
        auto *default_block = switch_inst->create_default_block();
        auto *merge = switch_inst->create_merge_block();
        builder.set_insertion_point(self_cycle);
        builder.br(self_cycle);
        builder.set_insertion_point(default_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto physical = lc::spirv::ControlFlowPlan::
            validate_function_physical_loop_boundaries(kernel);
        expect(!physical.planning_succeeded());
        expect(physical.planning_diagnostic.find(
                   "cyclic Switch case construct targeting itself") !=
               luisa::string::npos);
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(
            validation,
            "cyclic Switch case construct targeting itself"));
        expect(has_diagnostic(
            validation,
            "control-flow planning precondition failed"));
    };

    "spirv_xir_true_orphans_do_not_change_active_structural_legality"_test = [] {
        {
            Module module;
            auto *kernel = module.create_kernel();
            auto *condition =
                kernel->create_value_argument(Type::of<bool>());
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            auto *active_loop = builder.loop();
            auto *prepare = active_loop->create_prepare_block();
            auto *body = active_loop->create_body_block();
            auto *update = active_loop->create_update_block();
            auto *merge = active_loop->create_merge_block();
            builder.set_insertion_point(prepare);
            builder.cond_br(condition, body, merge);
            builder.set_insertion_point(body);
            builder.br(update);
            builder.set_insertion_point(update);
            builder.br(prepare);
            builder.set_insertion_point(merge);
            builder.return_void();

            auto *orphan_owner = kernel->create_basic_block();
            auto *orphan_body = kernel->create_basic_block();
            auto *orphan_update = kernel->create_basic_block();
            auto *orphan_merge = kernel->create_basic_block();
            builder.set_insertion_point(orphan_owner);
            auto *orphan_loop = builder.loop();
            orphan_loop->set_prepare_block(prepare);
            orphan_loop->set_body_block(orphan_body);
            orphan_loop->set_update_block(orphan_update);
            orphan_loop->set_merge_block(orphan_merge);
            builder.set_insertion_point(orphan_body);
            builder.return_void();
            builder.set_insertion_point(orphan_update);
            builder.return_void();
            builder.set_insertion_point(orphan_merge);
            builder.return_void();

            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect(validation.succeeded())
                << "an orphan Loop sharing an active prepare must not create a second backend owner";
        }
        {
            Module module;
            auto *kernel = module.create_kernel();
            auto *condition =
                kernel->create_value_argument(Type::of<bool>());
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            auto *active = builder.if_(condition);
            auto *active_true = active->create_true_block();
            auto *active_false = active->create_false_block();
            auto *active_merge = active->create_merge_block();
            builder.set_insertion_point(active_true);
            builder.br(active_merge);
            builder.set_insertion_point(active_false);
            builder.br(active_merge);
            builder.set_insertion_point(active_merge);
            builder.return_void();

            auto *orphan_owner = kernel->create_basic_block();
            builder.set_insertion_point(orphan_owner);
            auto *orphan = builder.if_(condition);
            auto *orphan_true = orphan->create_true_block();
            auto *orphan_false = orphan->create_false_block();
            orphan->set_merge_block(active_merge);
            builder.set_insertion_point(orphan_true);
            builder.debug_break();
            builder.return_void();
            builder.set_insertion_point(orphan_false);
            builder.return_void();

            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect(validation.succeeded())
                << "orphan merge ownership and unsupported orphan payloads are outside the backend closure";
        }
    };

    "spirv_xir_disconnected_payload_rejects_cross_block_ssa"_test = [] {
        Module module;
        auto *function = module.create_callable(Type::of<int32_t>());
        auto *condition =
            function->create_value_argument(Type::of<bool>());
        auto *zero =
            module.create_constant_zero(Type::of<int32_t>());
        auto *one =
            module.create_constant_one(Type::of<int32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(function->create_body_block());
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        auto *dead_merge = selection->create_merge_block();
        builder.set_insertion_point(true_block);
        auto *value = builder.call(
            Type::of<int32_t>(), ArithmeticOp::BINARY_ADD,
            {zero, one});
        builder.return_(value);
        builder.set_insertion_point(false_block);
        builder.return_(zero);
        builder.set_insertion_point(dead_merge);
        builder.return_(value);

        auto *kernel = module.create_kernel();
        auto *true_value =
            module.create_constant_one(Type::of<bool>());
        builder.set_insertion_point(kernel->create_body_block());
        builder.call(Type::of<int32_t>(), function, {true_value});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded())
            << "generic unreachable-block dominance is intentionally broader than the backend policy";
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation,
                              "cross-block instruction value"));
    };

    "spirv_xir_disconnected_payload_rejects_structured_reentry"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        auto *dead_merge = selection->create_merge_block();
        builder.set_insertion_point(true_block);
        builder.return_void();
        builder.set_insertion_point(false_block);
        builder.return_void();
        builder.set_insertion_point(dead_merge);
        builder.br(true_block);

        expect(xir_verify_module(&module).succeeded())
            << "generic XIR does not make the disconnected source reachable";
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation,
                              "unplanned physical predecessor"));
        expect(has_diagnostic(validation,
                              "ordinary-unreachable"));
    };

    "spirv_xir_disconnected_payload_rejects_nested_structure"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *outer = builder.if_(condition);
        auto *outer_true = outer->create_true_block();
        auto *outer_false = outer->create_false_block();
        auto *dead_merge = outer->create_merge_block();
        builder.set_insertion_point(outer_true);
        builder.return_void();
        builder.set_insertion_point(outer_false);
        builder.return_void();

        builder.set_insertion_point(dead_merge);
        auto *nested = builder.if_(condition);
        auto *nested_true = nested->create_true_block();
        auto *nested_false = nested->create_false_block();
        auto *nested_merge = nested->create_merge_block();
        builder.set_insertion_point(nested_true);
        builder.return_void();
        builder.set_insertion_point(nested_false);
        builder.return_void();
        builder.set_insertion_point(nested_merge);
        builder.return_void();

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation,
                              "nested structured terminator"));
        expect(has_diagnostic(validation,
                              "ordinary-unreachable"));
    };

    "spirv_xir_disconnected_payload_rejects_ray_query_lifetime"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        auto *dead_merge = selection->create_merge_block();
        builder.set_insertion_point(true_block);
        builder.return_void();
        builder.set_insertion_point(false_block);
        builder.return_void();
        builder.set_insertion_point(dead_merge);
        auto *query_alloca =
            builder.alloca_local(Type::custom("LC_RayQueryAny"));
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the disconnected ray-query lifetime fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, kernel, dead_merge, query_alloca,
            "ray-query lifetime validation");
        expect(has_diagnostic(validation, "ordinary-unreachable"));
    };

    "spirv_xir_reachable_recursion_is_reported_nonfatally"_test = [] {
        Module module;
        auto *callable = module.create_callable(Type::of<void>());
        XIRBuilder builder;
        builder.set_insertion_point(callable->create_body_block());
        auto *recursive_call =
            builder.call(Type::of<void>(), callable, {});
        builder.return_void();

        auto *kernel = module.create_kernel();
        builder.set_insertion_point(kernel->create_body_block());
        builder.call(Type::of<void>(), callable, {});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the reachable-recursion fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, callable, recursive_call->parent_block(),
            recursive_call, "reachable recursive callable cycle");
    };

    "spirv_xir_unreachable_recursive_callable_is_outside_emission"_test = [] {
        Module module;
        auto *callable = module.create_callable(Type::of<void>());
        XIRBuilder builder;
        builder.set_insertion_point(callable->create_body_block());
        builder.call(Type::of<void>(), callable, {});
        builder.return_void();

        auto *kernel = module.create_kernel();
        builder.set_insertion_point(kernel->create_body_block());
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the unreachable-recursion fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded())
            << "a callable definition unreachable from the kernel is not emitted";
    };

    "spirv_xir_unreachable_backend_unsupported_callable_is_outside_native_dialect"_test = [] {
        Module module;
        auto *callable = module.create_callable(Type::of<void>());
        XIRBuilder builder;
        builder.set_insertion_point(callable->create_body_block());
        builder.debug_break();
        builder.return_void();

        auto *kernel = module.create_kernel();
        builder.set_insertion_point(kernel->create_body_block());
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the unused backend-unsupported callable fixture must remain valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded())
            << "native dialect checks must use the same reachable definition set as emission";
    };

    "spirv_xir_reachable_backend_unsupported_callable_is_rejected"_test = [] {
        Module module;
        auto *callable = module.create_callable(Type::of<void>());
        XIRBuilder builder;
        builder.set_insertion_point(callable->create_body_block());
        auto *debug_break = builder.debug_break();
        builder.return_void();

        auto *kernel = module.create_kernel();
        builder.set_insertion_point(kernel->create_body_block());
        builder.call(Type::of<void>(), callable, {});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the reachable backend-unsupported callable fixture must remain valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, callable, debug_break->parent_block(),
            debug_break, "no debug-break instruction contract");
    };

    "spirv_xir_generic_verification_remains_whole_module"_test = [] {
        Module module;
        auto *unused_callable =
            module.create_callable(Type::of<void>());
        auto *unterminated = unused_callable->create_body_block();

        auto *kernel = module.create_kernel();
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        builder.return_void();

        auto generic = xir_verify_module(&module);
        expect(!generic.succeeded());
        expect(eq(generic.errors.size(), 1u));
        if (generic.errors.size() == 1u) {
            expect(generic.errors.front().function == unused_callable);
            expect(generic.errors.front().block == unterminated);
            expect(generic.errors.front().instruction == nullptr);
            expect(generic.errors.front().message ==
                   "Basic block is not terminated.");
        }

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, unused_callable, unterminated, nullptr,
            "Basic block is not terminated");
    };

    "spirv_xir_ray_query_single_dominating_binding_is_accepted"_test = [] {
        Module module;
        auto context = make_ray_query_test_context(module);
        XIRBuilder builder;
        builder.set_insertion_point(
            context.kernel->create_body_block());
        auto *query_alloca =
            builder.alloca_local(context.query_type);
        auto *initializer =
            emit_ray_query_initializer(builder, context);
        builder.store(query_alloca, initializer);
        builder.call(
            Type::of<bool>(),
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
            {query_alloca});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the dominating ray-query binding fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded())
            << "one direct initializer dominating every use is the native ray-query lifetime contract";
    };

    "spirv_xir_ray_query_conditional_binding_is_reported_nonfatally"_test = [] {
        Module module;
        auto context = make_ray_query_test_context(module);
        auto *condition =
            context.kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(
            context.kernel->create_body_block());
        auto *query_alloca =
            builder.alloca_local(context.query_type);
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        auto *merge_block = selection->create_merge_block();
        builder.set_insertion_point(true_block);
        auto *initializer =
            emit_ray_query_initializer(builder, context);
        builder.store(query_alloca, initializer);
        builder.br(merge_block);
        builder.set_insertion_point(false_block);
        builder.br(merge_block);
        builder.set_insertion_point(merge_block);
        auto *query_use = builder.call(
            Type::of<bool>(),
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
            {query_alloca});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the conditional ray-query binding fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, context.kernel, merge_block, query_use,
            "initialization to dominate every use");
    };

    "spirv_xir_ray_query_copy_is_reported_nonfatally"_test = [] {
        Module module;
        auto context = make_ray_query_test_context(module);
        XIRBuilder builder;
        builder.set_insertion_point(
            context.kernel->create_body_block());
        auto *source = builder.alloca_local(context.query_type);
        auto *destination = builder.alloca_local(context.query_type);
        auto *initializer =
            emit_ray_query_initializer(builder, context);
        builder.store(source, initializer);
        auto *loaded = builder.load(context.query_type, source);
        auto *copy = builder.store(destination, loaded);
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the opaque ray-query copy fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_diagnostics_at(
            validation, 2u, context.kernel,
            copy->parent_block(), copy);
        expect(has_diagnostic(validation,
                              "cannot copy or rebind"));
        expect(has_diagnostic(validation,
                              "cannot initialize an opaque ray-query alloca from a copied query"));
    };

    "spirv_xir_ray_query_phi_materialization_is_reported_nonfatally"_test = [] {
        Module module;
        auto context = make_ray_query_test_context(module);
        auto *condition =
            context.kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(
            context.kernel->create_body_block());
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        auto *merge_block = selection->create_merge_block();
        builder.set_insertion_point(true_block);
        auto *true_query =
            emit_ray_query_initializer(builder, context);
        builder.br(merge_block);
        builder.set_insertion_point(false_block);
        auto *false_query =
            emit_ray_query_initializer(builder, context);
        builder.br(merge_block);
        builder.set_insertion_point(merge_block);
        auto *phi = builder.phi(
            context.query_type,
            {{true_query, true_block},
             {false_query, false_block}});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the opaque ray-query Phi fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, context.kernel, merge_block, phi,
            "cannot materialize an opaque ray-query value");
    };

    "spirv_xir_true_orphan_resource_use_does_not_pin_callable_descriptor"_test = [] {
        Module module;
        auto *callable = module.create_callable(Type::of<void>());
        auto *buffer = callable->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        XIRBuilder builder;
        builder.set_insertion_point(callable->create_body_block());
        builder.return_void();
        auto *orphan = callable->create_basic_block();
        builder.set_insertion_point(orphan);
        builder.call(Type::of<uint32_t>(),
                     ResourceQueryOp::BUFFER_SIZE, {buffer});
        builder.return_void();

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded());
    };

    "spirv_xir_direct_integer_texture_sampling_is_rejected_exactly"_test = [] {
        struct SampleCase {
            ResourceQueryOp op;
            size_t dimension;
        };
        constexpr std::array sample_cases{
            SampleCase{ResourceQueryOp::TEXTURE2D_SAMPLE, 2u},
            SampleCase{ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL, 2u},
            SampleCase{ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD, 2u},
            SampleCase{ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL, 2u},
            SampleCase{ResourceQueryOp::TEXTURE3D_SAMPLE, 3u},
            SampleCase{ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL, 3u},
            SampleCase{ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD, 3u},
            SampleCase{ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL, 3u}};
        for (auto *sampled_type :
             std::array{Type::of<int32_t>(), Type::of<uint32_t>()}) {
            for (auto sample_case : sample_cases) {
                Module module;
                auto *kernel = module.create_kernel();
                auto *texture = kernel->create_resource_argument(
                    Type::texture(sampled_type,
                                  sample_case.dimension));
                auto *coordinate = module.create_constant_zero(
                    Type::vector(Type::of<float>(),
                                 sample_case.dimension));
                auto *lod = module.create_constant_zero(
                    Type::of<float>());
                auto *selector = module.create_constant_zero(
                    Type::of<uint32_t>());
                XIRBuilder builder;
                auto *body = kernel->create_body_block();
                builder.set_insertion_point(body);
                ResourceQueryInst *sample = nullptr;
                switch (sample_case.op) {
                    case ResourceQueryOp::TEXTURE2D_SAMPLE:
                    case ResourceQueryOp::TEXTURE3D_SAMPLE:
                        sample = builder.call(
                            Type::of<float4>(), sample_case.op,
                            {texture, coordinate, selector, selector});
                        break;
                    case ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL:
                    case ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL:
                        sample = builder.call(
                            Type::of<float4>(), sample_case.op,
                            {texture, coordinate, lod, selector,
                             selector});
                        break;
                    case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD:
                    case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD:
                        sample = builder.call(
                            Type::of<float4>(), sample_case.op,
                            {texture, coordinate, coordinate,
                             coordinate, selector, selector});
                        break;
                    case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
                    case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
                        sample = builder.call(
                            Type::of<float4>(), sample_case.op,
                            {texture, coordinate, coordinate,
                             coordinate, lod, selector, selector});
                        break;
                    default: break;
                }
                builder.return_void();

                expect(sample != nullptr);
                expect_generic_xir_valid(
                    module,
                    "integer direct-texture sampling remains valid generic XIR and must be rejected by the native SPIR-V dialect");
                auto validation =
                    lc::spirv::validate_spirv_xir_codegen_dialect(
                        &module);
                auto *diagnostic = expect_only_diagnostic_location(
                    validation, kernel, body, sample);
                if (diagnostic != nullptr) {
                    expect(diagnostic->message == luisa::format(
                                                      "Native XIR-to-SPIR-V direct texture sampling '{}' requires a float32 texture because XIR defines every sampling result as float4; got sampled scalar type {}.",
                                                      to_string(sample_case.op),
                                                      sampled_type->description()));
                }
            }
        }
    };

    "spirv_xir_dialect_rejections_name_the_missing_semantics"_test = [] {
        for (auto op : std::array{
                 ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX,
                 ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT}) {
            expect_unsupported(op, "motion-key");
        }
        for (auto op : std::array{
                 ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR,
                 ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR,
                 ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR,
                 ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR}) {
            expect_unsupported(op, "motion-time");
        }
        expect_unsupported(ResourceReadOp::DEVICE_ADDRESS_READ,
                           "physical-storage-buffer");
        expect_unsupported(ResourceWriteOp::DEVICE_ADDRESS_WRITE,
                           "physical-storage-buffer");
        for (auto op : std::array{
                 ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX,
                 ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT}) {
            expect_unsupported(op, "motion-key");
        }
        for (auto op : std::array{
                 ThreadGroupOp::RASTER_QUAD_DDX,
                 ThreadGroupOp::RASTER_QUAD_DDY}) {
            expect_unsupported(op, "raster invocation model");
        }
    };

    "spirv_xir_shader_execution_reorder_is_an_explicit_semantic_no_op"_test = [] {
        auto info = lc::spirv::spirv_xir_dialect_support(
            ThreadGroupOp::SHADER_EXECUTION_REORDER);
        expect(info.support ==
               lc::spirv::SpirvXIRDialectSupport::SEMANTIC_NO_OP);
        expect(info.accepted());
        expect(info.reason.find("optimization-only") !=
               luisa::string_view::npos);

        Module module;
        auto kernel = module.create_kernel();
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.shader_execution_reorder();
        builder.return_void();
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded());
    };

    "spirv_xir_assume_is_an_explicit_semantic_no_op"_test = [] {
        auto info = lc::spirv::spirv_xir_dialect_support(
            DerivedInstructionTag::ASSUME);
        expect(info.support ==
               lc::spirv::SpirvXIRDialectSupport::SEMANTIC_NO_OP);
        expect(info.accepted());
        expect(info.reason.find("optimization-only") !=
               luisa::string_view::npos);

        Module module;
        auto kernel = module.create_kernel();
        auto condition = kernel->create_value_argument(Type::of<bool>());
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.assume_(condition, "value is in range");
        builder.return_void();
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded());
    };

    "spirv_xir_assert_is_only_accepted_when_release_assertions_are_disabled"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        auto condition = kernel->create_value_argument(Type::of<bool>());
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto assertion = builder.assert_(condition, "value is in range");
        builder.return_void();

        auto debug_validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            debug_validation, kernel, body, assertion,
            "device-side failure-reporting contract");

        auto release_validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(
                &module,
                {.release_assertions_are_no_op = true});
        expect(release_validation.succeeded());
    };

    "spirv_xir_indirect_dispatch_requires_the_specialized_kernel_argument"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        auto indirect = kernel->create_reference_argument(
            Type::custom("LC_IndirectDispatchBuffer"));
        auto index = module.create_constant_zero(Type::of<uint32_t>());
        auto count = module.create_constant_one(Type::of<uint32_t>());
        auto block_size = module.create_constant_one(Type::of<uint3>());
        auto dispatch_size = module.create_constant_one(Type::of<uint3>());
        auto kernel_id = module.create_constant_zero(Type::of<uint32_t>());
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.call(ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT,
                     {indirect, count});
        builder.call(ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL,
                     {indirect, index, block_size, dispatch_size, kernel_id});
        builder.return_void();

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded());

        Module ordinary_buffer_module;
        auto ordinary_kernel = ordinary_buffer_module.create_kernel();
        auto ordinary_buffer = ordinary_kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto ordinary_count = ordinary_buffer_module.create_constant_one(
            Type::of<uint32_t>());
        auto ordinary_body = ordinary_kernel->create_body_block();
        builder.set_insertion_point(ordinary_body);
        builder.call(ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT,
                     {ordinary_buffer, ordinary_count});
        builder.return_void();

        auto ordinary_validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(
                &ordinary_buffer_module);
        expect(!ordinary_validation.succeeded());
        expect(has_diagnostic(
            ordinary_validation,
            "requires the specialized LC_IndirectDispatchBuffer"));
        expect(has_diagnostic(ordinary_validation,
                              "ordinary buffers"));
    };

    "spirv_xir_device_address_queries_are_supported_without_pointer_dereference"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        auto buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *device_address = builder.call(
            Type::of<uint64_t>(),
            ResourceQueryOp::BUFFER_DEVICE_ADDRESS, {buffer});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the device-address fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(validation.succeeded());
        expect(device_address != nullptr);
        for (auto op : std::array{
                 ResourceQueryOp::BUFFER_DEVICE_ADDRESS,
                 ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS}) {
            auto support = lc::spirv::spirv_xir_dialect_support(op);
            expect(support.accepted());
            expect(support.support ==
                   lc::spirv::SpirvXIRDialectSupport::SUPPORTED);
        }
    };

    "spirv_xir_unknown_opcode_is_rejected_without_stringifying_it"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.call(Type::of<uint32_t>(),
                     static_cast<ResourceQueryOp>(999), {});
        builder.return_void();

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation,
                              "unknown resource query opcode 999"));
        expect(has_diagnostic(validation, "fail-closed"));
    };

    "spirv_xir_compute_derivative_is_rejected_before_emission"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        auto value = kernel->create_value_argument(Type::of<float>());
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *derivative =
            builder.raster_quad_ddx(Type::of<float>(), value);
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the compute-derivative fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, kernel, body, derivative,
            "raster_quad_ddx");
        expect(has_diagnostic(validation, "raster invocation model"));
    };

    "spirv_xir_subgroup_type_contract_accepts_each_supported_shape_family"_test = [] {
        auto expect_accepted = [](
                                   ThreadGroupOp op, const Type *result_type,
                                   std::initializer_list<const Type *> operand_types) {
            Module module;
            auto *kernel = module.create_kernel();
            luisa::vector<Value *> operands;
            operands.reserve(operand_types.size());
            for (auto *type : operand_types) {
                operands.emplace_back(
                    kernel->create_value_argument(type));
            }
            XIRBuilder builder;
            builder.set_insertion_point(kernel->create_body_block());
            builder.call(
                result_type, op,
                luisa::span<Value *const>{operands.data(), operands.size()});
            builder.return_void();

            expect_generic_xir_valid(
                module,
                "the accepted subgroup type fixture must remain valid generic XIR");
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect(validation.succeeded())
                << luisa::format(
                       "accepted subgroup operation '{}' was rejected",
                       luisa::compute::xir::to_string(op));
        };

        expect_accepted(
            ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE,
            Type::of<bool>(), {});
        expect_accepted(
            ThreadGroupOp::WARP_FIRST_ACTIVE_LANE,
            Type::of<uint32_t>(), {});
        expect_accepted(
            ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL,
            Type::of<bool2>(), {Type::of<float2>()});
        for (auto op : std::array{
                 ThreadGroupOp::WARP_ACTIVE_BIT_AND,
                 ThreadGroupOp::WARP_ACTIVE_BIT_OR,
                 ThreadGroupOp::WARP_ACTIVE_BIT_XOR}) {
            expect_accepted(
                op, Type::of<uint3>(), {Type::of<uint3>()});
        }
        for (auto op : std::array{
                 ThreadGroupOp::WARP_ACTIVE_MAX,
                 ThreadGroupOp::WARP_ACTIVE_MIN,
                 ThreadGroupOp::WARP_ACTIVE_PRODUCT,
                 ThreadGroupOp::WARP_ACTIVE_SUM,
                 ThreadGroupOp::WARP_PREFIX_SUM,
                 ThreadGroupOp::WARP_PREFIX_PRODUCT}) {
            expect_accepted(
                op, Type::of<float4>(), {Type::of<float4>()});
        }
        expect_accepted(
            ThreadGroupOp::WARP_ACTIVE_SUM,
            Type::of<int32_t>(), {Type::of<int32_t>()});
        for (auto op : std::array{
                 ThreadGroupOp::WARP_ACTIVE_COUNT_BITS,
                 ThreadGroupOp::WARP_PREFIX_COUNT_BITS}) {
            expect_accepted(
                op, Type::of<uint32_t>(), {Type::of<bool>()});
        }
        for (auto op : std::array{
                 ThreadGroupOp::WARP_ACTIVE_ALL,
                 ThreadGroupOp::WARP_ACTIVE_ANY}) {
            expect_accepted(
                op, Type::of<bool>(), {Type::of<bool>()});
        }
        expect_accepted(
            ThreadGroupOp::WARP_ACTIVE_BIT_MASK,
            Type::of<uint4>(), {Type::of<bool>()});
        expect_accepted(
            ThreadGroupOp::WARP_READ_LANE,
            Type::of<float3x3>(),
            {Type::of<float3x3>(), Type::of<uint32_t>()});
        expect_accepted(
            ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE,
            Type::of<float2x2>(), {Type::of<float2x2>()});
    };

    "spirv_xir_subgroup_type_contract_rejects_generic_invalid_shapes_at_handoff"_test = [] {
        auto expect_rejected = [](
                                   ThreadGroupOp op, const Type *result_type,
                                   std::initializer_list<const Type *> operand_types,
                                   luisa::string_view expected_fragment) {
            Module module;
            auto *kernel = module.create_kernel();
            luisa::vector<Value *> operands;
            operands.reserve(operand_types.size());
            for (auto *type : operand_types) {
                operands.emplace_back(
                    kernel->create_value_argument(type));
            }
            auto *body = kernel->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *instruction = builder.call(
                result_type, op,
                luisa::span<Value *const>{operands.data(), operands.size()});
            builder.return_void();

            expect_generic_xir_invalid_at(
                module, kernel, body, instruction,
                "Instruction operands or result type are invalid");
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect_diagnostics_at(
                validation, 2u, kernel, body, instruction);
            expect(has_diagnostic(
                validation,
                luisa::format(
                    "subgroup operation '{}' requires",
                    luisa::compute::xir::to_string(op))));
            expect(has_diagnostic(validation, expected_fragment));
            expect(has_diagnostic(
                validation,
                "Invalid XIR at the native SPIR-V handoff"));
        };

        expect_rejected(
            ThreadGroupOp::WARP_ACTIVE_SUM,
            Type::of<float2x2>(), {Type::of<float2x2>()},
            "numeric scalar/vector");
        expect_rejected(
            ThreadGroupOp::WARP_ACTIVE_BIT_AND,
            Type::of<float2x2>(), {Type::of<float2x2>()},
            "integer scalar/vector");
        expect_rejected(
            ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL,
            Type::of<bool>(), {Type::of<float2x2>()},
            "scalar/vector operand");
        expect_rejected(
            ThreadGroupOp::WARP_ACTIVE_BIT_XOR,
            Type::of<float>(), {Type::of<float>()},
            "integer scalar/vector");
        expect_rejected(
            ThreadGroupOp::WARP_ACTIVE_SUM,
            Type::of<bool>(), {Type::of<bool>()},
            "numeric scalar/vector");
        expect_rejected(
            ThreadGroupOp::WARP_ACTIVE_COUNT_BITS,
            Type::of<uint32_t>(), {Type::of<uint32_t>()},
            "one bool operand and a uint32 result");
        expect_rejected(
            ThreadGroupOp::WARP_ACTIVE_ANY,
            Type::of<bool>(), {Type::of<uint32_t>()},
            "one bool operand and a bool result");
        expect_rejected(
            ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL,
            Type::of<bool>(), {Type::of<float2>()},
            "bool result with the same shape");
        expect_rejected(
            ThreadGroupOp::WARP_READ_LANE,
            Type::of<float3>(),
            {Type::of<float3>(), Type::of<int32_t>()},
            "uint32 lane index");
        expect_rejected(
            ThreadGroupOp::WARP_READ_LANE,
            Type::of<float2>(),
            {Type::of<float3>(), Type::of<uint32_t>()},
            "value's result type");
    };

    "spirv_xir_subgroup_type_contract_rejects_backend_unsupported_aggregate_shuffles"_test = [] {
        auto expect_rejected = [](
                                   ThreadGroupOp op, const Type *value_type,
                                   std::initializer_list<const Type *> operand_types) {
            Module module;
            auto *kernel = module.create_kernel();
            luisa::vector<Value *> operands;
            operands.reserve(operand_types.size());
            for (auto *type : operand_types) {
                operands.emplace_back(
                    kernel->create_value_argument(type));
            }
            auto *body = kernel->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *instruction = builder.call(
                value_type, op,
                luisa::span<Value *const>{operands.data(), operands.size()});
            builder.return_void();

            expect_generic_xir_valid(
                module,
                "aggregate subgroup shuffles are valid generic XIR");
            auto validation =
                lc::spirv::validate_spirv_xir_codegen_dialect(&module);
            expect_only_diagnostic_at(
                validation, kernel, body, instruction,
                luisa::format(
                    "subgroup operation '{}' requires",
                    luisa::compute::xir::to_string(op)));
            expect(has_diagnostic(
                validation, "supported scalar/vector/matrix"));
        };

        auto *array_type = Type::array(Type::of<float>(), 2u);
        expect_rejected(
            ThreadGroupOp::WARP_READ_LANE, array_type,
            {array_type, Type::of<uint32_t>()});
        auto *structure_type = Type::structure(
            {Type::of<float>(), Type::of<uint32_t>()});
        expect_rejected(
            ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE,
            structure_type, {structure_type});
    };

    "spirv_xir_subgroup_float64_is_rejected_before_emission"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        auto value = kernel->create_value_argument(Type::of<double>());
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *reduction = builder.call(
            Type::of<double>(), ThreadGroupOp::WARP_ACTIVE_SUM,
            {value});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the float64 subgroup fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, kernel, body, reduction,
            "warp_active_sum");
        expect(has_diagnostic(
            validation, "supported numeric scalar/vector"));
    };

    "spirv_xir_fp8_comparison_is_rejected_before_emission"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        auto fp8 = Type::from("float8e4m3");
        auto lhs = kernel->create_value_argument(fp8);
        auto rhs = kernel->create_value_argument(fp8);
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
                     {lhs, rhs});
        builder.return_void();

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation, "binary_equal"));
        expect(has_diagnostic(validation, "does not support FP8"));
    };

    "spirv_fp8_transport_allowlist_excludes_matrix_instructions"_test = [] {
        // SPV_EXT_float8 admits composite and selection instructions, but
        // does not admit the matrix-instruction category that owns
        // OpTranspose. Keep the XIR classification exact even though Luisa's
        // current ordinary matrix type is float32-only.
        expect(lc::spirv::spirv_fp8_transport_op_supported(
            ArithmeticOp::SELECT));
        expect(lc::spirv::spirv_fp8_transport_op_supported(
            ArithmeticOp::AGGREGATE));
        expect(lc::spirv::spirv_fp8_transport_op_supported(
            ArithmeticOp::SHUFFLE));
        expect(lc::spirv::spirv_fp8_transport_op_supported(
            ArithmeticOp::INSERT));
        expect(lc::spirv::spirv_fp8_transport_op_supported(
            ArithmeticOp::EXTRACT));
        expect(!lc::spirv::spirv_fp8_transport_op_supported(
            ArithmeticOp::MATRIX_TRANSPOSE));
    };

    "spirv_xir_function_local_atomic_is_rejected_before_emission"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        auto one = module.create_constant_one(Type::of<int32_t>());
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto local = builder.alloca_local(Type::of<int32_t>());
        auto *atomic = builder.atomic_fetch_add(
            Type::of<int32_t>(), local, {}, one);
        builder.return_void();

        auto generic = xir_verify_module(&module);
        expect(!generic.succeeded())
            << "generic XIR atomics intentionally admit storage buffers and shared arrays, not local allocas";
        expect(eq(generic.errors.size(), 1u));
        if (generic.errors.size() == 1u) {
            auto &&error = generic.errors.front();
            expect(error.function == kernel);
            expect(error.block == body);
            expect(error.instruction == atomic);
            expect(error.message.find(
                       "Instruction operands or result type are invalid") !=
                   luisa::string::npos);
        }
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_diagnostics_at(
            validation, 2u, kernel, body, atomic);
        expect(has_diagnostic(validation,
                              "function-local allocation"));
        expect(has_diagnostic(validation, "Function storage class"));
        expect(has_diagnostic(validation,
                              "Invalid XIR at the native SPIR-V handoff"));
    };

    "spirv_xir_kernel_reference_argument_is_rejected_at_the_abi_boundary"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        kernel->create_reference_argument(Type::of<int32_t>());
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.return_void();

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation,
                              "unsupported reference argument"));
        expect(has_diagnostic(validation, "kernel ABI"));
    };

    "spirv_xir_used_callable_buffer_requires_specialization"_test = [] {
        Module module;
        auto callable = module.create_callable(Type::of<uint32_t>());
        auto buffer = callable->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto size = builder.call(Type::of<uint32_t>(),
                                 ResourceQueryOp::BUFFER_SIZE, {buffer});
        builder.return_(size);

        auto *kernel = module.create_kernel();
        auto *kernel_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        builder.set_insertion_point(kernel->create_body_block());
        static_cast<void>(builder.call(
            Type::of<uint32_t>(), callable, {kernel_buffer}));
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the callable-buffer specialization fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, callable, nullptr, nullptr,
            "buffer and bindless descriptors");
        expect(has_diagnostic(validation, "specialized at call sites"));
    };

    "spirv_xir_writable_callable_accel_requires_specialization"_test = [] {
        Module module;
        auto callable = module.create_callable(Type::of<void>());
        callable->set_name("write_accel");
        auto accel = callable->create_resource_argument(
            Type::from("accel"));
        auto instance_index = module.create_constant_zero(
            Type::of<uint32_t>());
        auto user_id = module.create_constant_one(
            Type::of<uint32_t>());
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.call(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID,
                     {accel, instance_index, user_id});
        builder.return_void();

        auto *kernel = module.create_kernel();
        auto *kernel_accel = kernel->create_resource_argument(
            Type::from("accel"));
        builder.set_insertion_point(kernel->create_body_block());
        builder.call(Type::of<void>(), callable, {kernel_accel});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the writable callable-accel fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic(
            validation, callable,
            "Native XIR-to-SPIR-V callable 'write_accel' retains an "
            "acceleration-structure resource argument that is writable or "
            "requires instance-buffer state; such acceleration-structure "
            "arguments must be specialized at call sites before codegen.");
    };

    "spirv_xir_callable_accel_instance_read_requires_specialization"_test = [] {
        Module module;
        auto callable = module.create_callable(Type::of<void>());
        callable->set_name("read_accel_instance");
        auto accel = callable->create_resource_argument(
            Type::from("accel"));
        auto instance_index = module.create_constant_zero(
            Type::of<uint32_t>());
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        static_cast<void>(builder.call(
            Type::of<uint32_t>(),
            ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID,
            {accel, instance_index}));
        builder.return_void();

        auto *kernel = module.create_kernel();
        auto *kernel_accel = kernel->create_resource_argument(
            Type::from("accel"));
        builder.set_insertion_point(kernel->create_body_block());
        builder.call(Type::of<void>(), callable, {kernel_accel});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the callable accel-instance read fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic(
            validation, callable,
            "Native XIR-to-SPIR-V callable 'read_accel_instance' retains an "
            "acceleration-structure resource argument that is writable or "
            "requires instance-buffer state; such acceleration-structure "
            "arguments must be specialized at call sites before codegen.");
    };

    "spirv_xir_dual_role_callable_texture_requires_specialization"_test = [] {
        Module module;
        auto callable = module.create_callable(Type::of<void>());
        callable->set_name("update_texture");
        auto texture = callable->create_resource_argument(
            Type::texture(Type::of<float>(), 2u));
        auto coord = module.create_constant_zero(Type::of<uint2>());
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto value = builder.call(Type::of<float4>(),
                                  ResourceReadOp::TEXTURE2D_READ,
                                  {texture, coord});
        builder.call(ResourceWriteOp::TEXTURE2D_WRITE,
                     {texture, coord, value});
        builder.return_void();

        auto *kernel = module.create_kernel();
        auto *kernel_texture = kernel->create_resource_argument(
            Type::texture(Type::of<float>(), 2u));
        builder.set_insertion_point(kernel->create_body_block());
        builder.call(Type::of<void>(), callable, {kernel_texture});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the dual-role callable-texture fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic(
            validation, callable,
            "Native XIR-to-SPIR-V callable 'update_texture' retains a texture "
            "resource argument used for both read and write; dual "
            "sampled/storage-image bindings must be specialized at call sites "
            "before codegen.");
    };

    "spirv_xir_cast_shape_is_rejected_before_emission"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        auto source_type = Type::vector(Type::of<int32_t>(), 3u);
        auto target_type = Type::vector(Type::of<float>(), 2u);
        auto value = kernel->create_value_argument(source_type);
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cast_(target_type,
                      luisa::compute::xir::CastOp::STATIC_CAST,
                      value);
        builder.return_void();

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation,
                              "static_cast requires equal scalar/vector dimensions"));
    };

    "spirv_xir_raster_special_register_is_rejected_before_emission"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        auto object_id = module.create_object_id();
        auto zero = module.create_constant_zero(Type::of<uint32_t>());
        auto body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *object_id_use = builder.call(
            Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD,
            {object_id, zero});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "the raster special-register fixture must be valid generic XIR");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, kernel, body, object_id_use,
            "special register 'object_id'");
        expect(has_diagnostic(validation, "raster-stage builtin"));
    };

    "spirv_xir_zero_length_array_is_rejected_at_the_type_boundary"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        kernel->create_value_argument(
            Type::array(Type::of<uint32_t>(), 0u));
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.return_void();

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation, "array<uint,0>"));
        expect(has_diagnostic(validation, "cannot represent"));
    };

    "spirv_xir_zero_stride_buffer_is_rejected_at_the_storage_boundary"_test = [] {
        std::array<const Type *, 0u> no_members{};
        auto empty = Type::structure(4u, luisa::span{no_members});
        expect(eq(empty->size(), size_t{0u}));

        Module module;
        auto *kernel = module.create_kernel();
        kernel->create_resource_argument(Type::buffer(empty));
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.return_void();

        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!validation.succeeded());
        expect(has_diagnostic(validation, "buffer<struct<4>>"));
        expect(has_diagnostic(validation, "cannot represent"));
    };

    "spirv_xir_bindless_buffer_size_rejects_constant_zero_stride"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *bindless = kernel->create_resource_argument(
            Type::from("bindless_array"));
        auto *zero = module.create_constant_zero(Type::of<uint32_t>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *size = builder.call(
            Type::of<uint32_t>(),
            ResourceQueryOp::BINDLESS_BUFFER_SIZE,
            {bindless, zero, zero});
        builder.return_void();

        expect_generic_xir_valid(
            module,
            "a zero bindless element stride is valid generic XIR and must be rejected by the native SPIR-V dialect");
        auto validation =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect_only_diagnostic_at(
            validation, kernel, body, size,
            "nonzero element stride");
        expect(has_diagnostic(validation, "division undefined"));
    };
}
