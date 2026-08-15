#include "hip_callable_abi.h"
#include "ut/ut.hpp"

#include <memory>
#include <string_view>

#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/SourceMgr.h>

using namespace luisa::compute::hip;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] std::unique_ptr<llvm::Module> parse_module(
    llvm::LLVMContext &context,
    std::string_view text) {
    llvm::SMDiagnostic diagnostic;
    return llvm::parseAssemblyString(text, diagnostic, context);
}

static auto suite = [] {
    "HIP callable ABI projects exactly the observed aggregate leaves"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-i64:64-n32:64"
            define private float @projected({ [4 x <4 x float>], i32 } %value) #0 {
            entry:
              %frozen = freeze { [4 x <4 x float>], i32 } %value
              %lane = extractvalue { [4 x <4 x float>], i32 } %frozen, 0, 2
              %x = extractelement <4 x float> %lane, i64 0
              %tag = extractvalue { [4 x <4 x float>], i32 } %value, 1
              %tag.float = uitofp i32 %tag to float
              %sum = fadd float %x, %tag.float
              ret float %sum
            }
            define float @caller({ [4 x <4 x float>], i32 } %actual) {
            entry:
              %result = call float @projected({ [4 x <4 x float>], i32 } %actual)
              ret float %result
            }
            attributes #0 = { noinline "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        auto stats = specialize_generated_callable_aggregate_arguments(*module);
        expect(stats.rewritten_function_count == 1u);
        // The original ABI is 80 B. Its selected float4 and i32 leaves occupy
        // 20 B, so this also checks layout-aware accounting.
        expect(stats.removed_aggregate_bytes == 60u);
        expect(!llvm::verifyModule(*module));

        auto *projected = module->getFunction("projected");
        expect(projected != nullptr);
        expect(projected->arg_size() == 2u);
        expect(projected->getFunctionType()->getParamType(0u)->isVectorTy());
        expect(projected->getFunctionType()->getParamType(1u)->isIntegerTy(32u));
        auto *caller = module->getFunction("caller");
        expect(caller != nullptr);
        auto call_count = 0u;
        for (auto &block : *caller) {
            for (auto &instruction : block) {
                if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
                    call_count++;
                    expect(call->getCalledFunction() == projected);
                    expect(call->arg_size() == 2u);
                }
            }
        }
        expect(call_count == 1u);
    };

    "HIP callable ABI rejects an incompletely modeled aggregate use"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-i64:64-n32:64"
            declare void @consume({ i32, i32 })
            define private void @opaque({ i32, i32 } %value) #0 {
            entry:
              call void @consume({ i32, i32 } %value)
              ret void
            }
            define void @caller({ i32, i32 } %actual) {
            entry:
              call void @opaque({ i32, i32 } %actual)
              ret void
            }
            attributes #0 = { noinline "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        auto stats = specialize_generated_callable_aggregate_arguments(*module);
        expect(stats.rewritten_function_count == 0u);
        expect(stats.removed_aggregate_bytes == 0u);
        expect(!llvm::verifyModule(*module));
        auto *opaque = module->getFunction("opaque");
        expect(opaque != nullptr);
        expect(opaque->arg_size() == 1u);
        expect(opaque->getFunctionType()->getParamType(0u)->isStructTy());
    };

    "HIP callable ABI removes an unused aggregate argument"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-i64:64-n32:64"
            define private i32 @unused({ i64, i64 } %value, i32 %kept) #0 {
            entry:
              ret i32 %kept
            }
            define i32 @caller({ i64, i64 } %actual) {
            entry:
              %result = call i32 @unused({ i64, i64 } %actual, i32 7)
              ret i32 %result
            }
            attributes #0 = { noinline "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        auto stats = specialize_generated_callable_aggregate_arguments(*module);
        expect(stats.rewritten_function_count == 1u);
        expect(stats.removed_aggregate_bytes == 16u);
        expect(!llvm::verifyModule(*module));
        auto *unused = module->getFunction("unused");
        expect(unused != nullptr);
        expect(unused->arg_size() == 1u);
        expect(unused->getFunctionType()->getParamType(0u)->isIntegerTy(32u));
    };

    "HIP callable ABI rejects a dead unmodeled aggregate chain"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-i64:64-n32:64"
            define private i32 @dead_chain({ { i32, i32 }, i32 } %value) #0 {
            entry:
              %dead = extractvalue { { i32, i32 }, i32 } %value, 0
              %used = extractvalue { { i32, i32 }, i32 } %value, 1
              ret i32 %used
            }
            define i32 @caller({ { i32, i32 }, i32 } %actual) {
            entry:
              %result = call i32 @dead_chain({ { i32, i32 }, i32 } %actual)
              ret i32 %result
            }
            attributes #0 = { noinline "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        auto stats = specialize_generated_callable_aggregate_arguments(*module);
        expect(stats.rewritten_function_count == 0u);
        expect(stats.removed_aggregate_bytes == 0u);
        expect(!llvm::verifyModule(*module));
        auto *dead_chain = module->getFunction("dead_chain");
        expect(dead_chain != nullptr);
        expect(dead_chain->arg_size() == 1u);
        expect(dead_chain->getFunctionType()->getParamType(0u)->isStructTy());
    };

    "HIP callable ABI keeps returns within the AMDGPU VGPR convention"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-p5:32:32-i64:64-n32:64-A5"
            define private [32 x i32] @fits([32 x i32] %value) #0 {
            entry:
              ret [32 x i32] %value
            }
            define [32 x i32] @caller([32 x i32] %actual) {
            entry:
              %result = call [32 x i32] @fits([32 x i32] %actual)
              ret [32 x i32] %result
            }
            attributes #0 = { noinline "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        auto stats = demote_generated_callable_large_returns(*module);
        expect(stats.rewritten_function_count == 0u);
        expect(stats.rewritten_call_count == 0u);
        expect(stats.shared_result_slot_count == 0u);
        expect(stats.demoted_return_bytes == 0u);
        expect(!llvm::verifyModule(*module));
        auto *fits = module->getFunction("fits");
        expect(fits != nullptr);
        expect(fits->getReturnType()->isArrayTy());
    };

    "HIP callable ABI shares one post-IPO large-return slot per caller"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-p5:32:32-i64:64-n32:64-A5"
            define private [33 x i32] @large_a([33 x i32] %value) #0 {
            entry:
              ret [33 x i32] %value
            }
            define private [33 x i32] @large_b([33 x i32] %value) #0 {
            entry:
              %head = extractvalue [33 x i32] %value, 0
              %next = add i32 %head, 1
              %result = insertvalue [33 x i32] %value, i32 %next, 0
              ret [33 x i32] %result
            }
            define i32 @caller([33 x i32] %actual) {
            entry:
              %a = call [33 x i32] @large_a([33 x i32] %actual)
              %b = call [33 x i32] @large_b([33 x i32] %a)
              %result = extractvalue [33 x i32] %b, 0
              ret i32 %result
            }
            attributes #0 = { "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        auto stats = demote_generated_callable_large_returns(*module);
        expect(stats.rewritten_function_count == 2u);
        expect(stats.rewritten_call_count == 2u);
        expect(stats.shared_result_slot_count == 1u);
        expect(stats.demoted_return_bytes == 264u);
        expect(!llvm::verifyModule(*module));

        auto *large_a = module->getFunction("large_a");
        auto *large_b = module->getFunction("large_b");
        expect(large_a != nullptr);
        expect(large_b != nullptr);
        expect(large_a->getReturnType()->isVoidTy());
        expect(large_b->getReturnType()->isVoidTy());
        expect(large_a->arg_size() == 2u);
        expect(large_b->arg_size() == 2u);
        expect(large_a->getFunctionType()
                   ->getParamType(0u)
                   ->getPointerAddressSpace() == 5u);

        auto *caller = module->getFunction("caller");
        expect(caller != nullptr);
        auto alloca_count = 0u;
        auto call_count = 0u;
        auto load_count = 0u;
        llvm::AllocaInst *shared_slot = nullptr;
        for (auto &instruction : caller->getEntryBlock()) {
            if (auto *alloca = llvm::dyn_cast<llvm::AllocaInst>(&instruction)) {
                alloca_count++;
                shared_slot = alloca;
            }
            if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
                call_count++;
                expect(call->getType()->isVoidTy());
                expect(call->arg_size() == 2u);
                expect(call->getArgOperand(0u) == shared_slot);
                auto *next = call->getNextNode();
                expect(next != nullptr);
                auto *load = llvm::dyn_cast_or_null<llvm::LoadInst>(next);
                expect(load != nullptr);
                if (load != nullptr) {
                    expect(load->getPointerOperand() == shared_slot);
                }
            }
            if (llvm::isa<llvm::LoadInst>(instruction)) { load_count++; }
        }
        expect(alloca_count == 1u);
        expect(call_count == 2u);
        expect(load_count == 2u);
    };

    "HIP callable ABI models packed half aggregate return boundaries"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-p5:32:32-i64:64-n32:64-A5"
            define private { [16 x i32], <32 x half> } @fits(
                { [16 x i32], <32 x half> } %value) #0 {
            entry:
              ret { [16 x i32], <32 x half> } %value
            }
            define private { [16 x i32], <34 x half> } @large(
                { [16 x i32], <34 x half> } %value) #0 {
            entry:
              ret { [16 x i32], <34 x half> } %value
            }
            define i32 @caller(
                { [16 x i32], <32 x half> } %fits.value,
                { [16 x i32], <34 x half> } %large.value) {
            entry:
              %fits.result = call { [16 x i32], <32 x half> } @fits(
                  { [16 x i32], <32 x half> } %fits.value)
              %large.result = call { [16 x i32], <34 x half> } @large(
                  { [16 x i32], <34 x half> } %large.value)
              %fits.head = extractvalue { [16 x i32], <32 x half> } %fits.result, 0, 0
              %large.head = extractvalue { [16 x i32], <34 x half> } %large.result, 0, 0
              %result = add i32 %fits.head, %large.head
              ret i32 %result
            }
            attributes #0 = { noinline "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        auto *original_large = module->getFunction("large");
        expect(original_large != nullptr);
        const auto expected_large_bytes =
            module->getDataLayout()
                .getTypeAllocSize(original_large->getReturnType())
                .getFixedValue();
        auto stats = demote_generated_callable_large_returns(*module);
        expect(stats.rewritten_function_count == 1u);
        expect(stats.rewritten_call_count == 1u);
        expect(stats.shared_result_slot_count == 1u);
        expect(stats.demoted_return_bytes == expected_large_bytes);
        expect(!llvm::verifyModule(*module));
        auto *fits = module->getFunction("fits");
        auto *large = module->getFunction("large");
        expect(fits != nullptr);
        expect(large != nullptr);
        expect(fits->getReturnType()->isStructTy());
        expect(large->getReturnType()->isVoidTy());
    };

    "HIP callable ABI removes invalidated returned and speculatable attributes"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-p5:32:32-i64:64-n32:64-A5"
            define private [33 x i32] @large(
                [33 x i32] returned %value) #0 {
            entry:
              ret [33 x i32] %value
            }
            define i32 @caller([33 x i32] %actual) {
            entry:
              %result = call [33 x i32] @large(
                  [33 x i32] returned %actual) #0
              %head = extractvalue [33 x i32] %result, 0
              ret i32 %head
            }
            attributes #0 = { noinline speculatable memory(none) "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        expect(!llvm::verifyModule(*module));
        auto stats = demote_generated_callable_large_returns(*module);
        expect(stats.rewritten_function_count == 1u);
        expect(stats.rewritten_call_count == 1u);
        expect(!llvm::verifyModule(*module));

        auto *large = module->getFunction("large");
        expect(large != nullptr);
        expect(!large->hasFnAttribute(llvm::Attribute::Speculatable));
        expect(!large->getAttributes().hasParamAttr(
            1u, llvm::Attribute::Returned));
        expect(large->getMemoryEffects().getModRef(
                   llvm::MemoryEffects::Location::ArgMem) ==
               llvm::ModRefInfo::Mod);
        auto *caller = module->getFunction("caller");
        expect(caller != nullptr);
        for (auto &instruction : caller->getEntryBlock()) {
            if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
                expect(!call->hasFnAttr(llvm::Attribute::Speculatable));
                expect(!call->getAttributes().hasParamAttr(
                    1u, llvm::Attribute::Returned));
            }
        }
    };

    "HIP callable ABI preserves FastCC and rejects unmodeled ABIs"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-p5:32:32-i64:64-n32:64-A5"
            define [33 x i32] @external_large(i32 %count) #0 {
            entry:
              ret [33 x i32] zeroinitializer
            }
            define private [33 x i32] @function_allocsize(i32 %count) #1 {
            entry:
              ret [33 x i32] zeroinitializer
            }
            define private [33 x i32] @call_allocsize(i32 %count) #0 {
            entry:
              ret [33 x i32] zeroinitializer
            }
            define private fastcc [33 x i32] @fastcc_large(i32 %count) #0 {
            entry:
              ret [33 x i32] zeroinitializer
            }
            define private coldcc [33 x i32] @coldcc_large(i32 %count) #0 {
            entry:
              ret [33 x i32] zeroinitializer
            }
            define private [33 x i32] @bundled_large(i32 %count) #0 {
            entry:
              ret [33 x i32] zeroinitializer
            }
            define private [33 x i32] @function_metadata(i32 %count) #0 !custom !0 {
            entry:
              ret [33 x i32] zeroinitializer
            }
            define private [33 x i32] @call_metadata(i32 %count) #0 {
            entry:
              ret [33 x i32] zeroinitializer
            }
            define private [33 x i32] @return_metadata(i32 %count) #0 {
            entry:
              ret [33 x i32] zeroinitializer, !custom !0
            }
            define i32 @caller(i32 %count) {
            entry:
              %a = call [33 x i32] @external_large(i32 %count)
              %b = call [33 x i32] @function_allocsize(i32 %count)
              %c = call [33 x i32] @call_allocsize(i32 %count) #2
              %d = call fastcc [33 x i32] @fastcc_large(i32 %count)
              %e = call coldcc [33 x i32] @coldcc_large(i32 %count)
              %f = call [33 x i32] @bundled_large(i32 %count) [ "deopt"(i32 %count) ]
              %g = call [33 x i32] @function_metadata(i32 %count)
              %h = call [33 x i32] @call_metadata(i32 %count), !custom !0
              %i = call [33 x i32] @return_metadata(i32 %count)
              %a.head = extractvalue [33 x i32] %a, 0
              %b.head = extractvalue [33 x i32] %b, 0
              %c.head = extractvalue [33 x i32] %c, 0
              %d.head = extractvalue [33 x i32] %d, 0
              %e.head = extractvalue [33 x i32] %e, 0
              %f.head = extractvalue [33 x i32] %f, 0
              %g.head = extractvalue [33 x i32] %g, 0
              %h.head = extractvalue [33 x i32] %h, 0
              %i.head = extractvalue [33 x i32] %i, 0
              %ab = add i32 %a.head, %b.head
              %abc = add i32 %ab, %c.head
              %abcd = add i32 %abc, %d.head
              %abcde = add i32 %abcd, %e.head
              %abcdef = add i32 %abcde, %f.head
              %abcdefg = add i32 %abcdef, %g.head
              %abcdefgh = add i32 %abcdefg, %h.head
              %result = add i32 %abcdefgh, %i.head
              ret i32 %result
            }
            attributes #0 = { noinline "luisa-generated-callable" }
            attributes #1 = { noinline allocsize(0) "luisa-generated-callable" }
            attributes #2 = { allocsize(0) }
            !0 = !{!"unmodeled semantic metadata"}
        )");
        expect(module != nullptr);
        expect(!llvm::verifyModule(*module));
        auto stats = demote_generated_callable_large_returns(*module);
        expect(stats.rewritten_function_count == 1u);
        expect(stats.rewritten_call_count == 1u);
        expect(stats.shared_result_slot_count == 1u);
        expect(stats.demoted_return_bytes == 132u);
        expect(!llvm::verifyModule(*module));
        expect(module->getFunction("external_large")
                   ->getReturnType()
                   ->isArrayTy());
        expect(module->getFunction("function_allocsize")
                   ->getReturnType()
                   ->isArrayTy());
        expect(module->getFunction("call_allocsize")
                   ->getReturnType()
                   ->isArrayTy());
        expect(module->getFunction("fastcc_large")
                   ->getReturnType()
                   ->isVoidTy());
        expect(module->getFunction("fastcc_large")->getCallingConv() ==
               llvm::CallingConv::Fast);
        auto fastcc_call_count = 0u;
        for (auto *user : module->getFunction("fastcc_large")->users()) {
            auto *call = llvm::dyn_cast<llvm::CallInst>(user);
            expect(call != nullptr);
            if (call != nullptr) {
                fastcc_call_count++;
                expect(call->getCallingConv() == llvm::CallingConv::Fast);
                expect(call->getType()->isVoidTy());
            }
        }
        expect(fastcc_call_count == 1u);
        expect(module->getFunction("coldcc_large")
                   ->getReturnType()
                   ->isArrayTy());
        expect(module->getFunction("bundled_large")
                   ->getReturnType()
                   ->isArrayTy());
        expect(module->getFunction("function_metadata")
                   ->getReturnType()
                   ->isArrayTy());
        expect(module->getFunction("call_metadata")
                   ->getReturnType()
                   ->isArrayTy());
        expect(module->getFunction("return_metadata")
                   ->getReturnType()
                   ->isArrayTy());
    };

    "HIP callable ABI rejects a musttail large-return use atomically"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-p5:32:32-i64:64-n32:64-A5"
            define private [33 x i32] @large([33 x i32] %value) #0 {
            entry:
              ret [33 x i32] %value
            }
            define [33 x i32] @caller([33 x i32] %actual) {
            entry:
              %result = musttail call [33 x i32] @large([33 x i32] %actual)
              ret [33 x i32] %result
            }
            attributes #0 = { noinline "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        expect(!llvm::verifyModule(*module));
        auto stats = demote_generated_callable_large_returns(*module);
        expect(stats.rewritten_function_count == 0u);
        expect(stats.rewritten_call_count == 0u);
        expect(!llvm::verifyModule(*module));
        auto *large = module->getFunction("large");
        expect(large != nullptr);
        expect(large->getReturnType()->isArrayTy());
    };

    "HIP callable ABI rejects a notail large-return use atomically"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-p5:32:32-i64:64-n32:64-A5"
            define private [33 x i32] @large([33 x i32] %value) #0 {
            entry:
              ret [33 x i32] %value
            }
            define i32 @caller([33 x i32] %actual) {
            entry:
              %result = notail call [33 x i32] @large([33 x i32] %actual)
              %head = extractvalue [33 x i32] %result, 0
              ret i32 %head
            }
            attributes #0 = { noinline "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        expect(!llvm::verifyModule(*module));
        auto stats = demote_generated_callable_large_returns(*module);
        expect(stats.rewritten_function_count == 0u);
        expect(stats.rewritten_call_count == 0u);
        expect(!llvm::verifyModule(*module));
        auto *large = module->getFunction("large");
        expect(large != nullptr);
        expect(large->getReturnType()->isArrayTy());
    };

    "HIP constant dispatcher specializes once per distinct identity"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            define private fastcc i32 @dispatch(
                ptr nonnull %state, i32 noundef %pipeline,
                i32 noundef %kind) {
            entry:
              switch i32 %pipeline, label %invalid [
                i32 0, label %pipeline.0
                i32 1, label %pipeline.1
              ]
            pipeline.0:
              %a = add i32 %kind, 10
              ret i32 %a
            pipeline.1:
              %b = add i32 %kind, 20
              ret i32 %b
            invalid:
              unreachable
            }
            define i32 @caller(ptr %state) {
            entry:
              %a = tail call fastcc i32 @dispatch(
                  ptr nonnull %state, i32 noundef 0, i32 noundef 1)
              %b = tail call fastcc i32 @dispatch(
                  ptr nonnull %state, i32 noundef 0, i32 noundef 2)
              %c = tail call fastcc i32 @dispatch(
                  ptr nonnull %state, i32 noundef 1, i32 noundef 1)
              %d = tail call fastcc i32 @dispatch(
                  ptr nonnull %state, i32 noundef 1, i32 noundef 2)
              %ab = add i32 %a, %b
              %cd = add i32 %c, %d
              %result = add i32 %ab, %cd
              ret i32 %result
            }
        )");
        expect(module != nullptr);
        auto *dispatch = module->getFunction("dispatch");
        expect(dispatch != nullptr);
        dispatch->getArg(1u)->addAttr(llvm::Attribute::get(
            context,
            llvm_constant_argument_specialization_attribute));

        auto stats =
            specialize_marked_constant_integer_arguments(*module);
        expect(stats.rewritten_function_count == 1u);
        expect(stats.cloned_function_count == 2u);
        expect(stats.rewritten_call_count == 4u);
        expect(!llvm::verifyModule(*module));
        expect(module->getFunction("dispatch") == nullptr);

        auto *zero = module->getFunction("dispatch.constant.0");
        auto *one = module->getFunction("dispatch.constant.1");
        expect(zero != nullptr && one != nullptr);
        for (auto *specialized : {zero, one}) {
            expect(specialized->arg_size() == 2u);
            expect(specialized->getArg(0u)->hasAttribute(
                llvm::Attribute::NonNull));
            expect(specialized->getArg(1u)->hasAttribute(
                llvm::Attribute::NoUndef));
            for (auto &block : *specialized) {
                expect(!llvm::isa<llvm::SwitchInst>(
                    block.getTerminator()));
                if (auto *branch = llvm::dyn_cast<llvm::BranchInst>(
                        block.getTerminator())) {
                    expect(!branch->isConditional());
                }
            }
        }

        auto *caller = module->getFunction("caller");
        auto rewritten_calls = 0u;
        for (auto &block : *caller) {
            for (auto &instruction : block) {
                if (auto *call =
                        llvm::dyn_cast<llvm::CallInst>(&instruction)) {
                    rewritten_calls++;
                    expect(call->getCalledFunction() == zero ||
                           call->getCalledFunction() == one);
                    expect(call->arg_size() == 2u);
                    expect(call->getTailCallKind() ==
                           llvm::CallInst::TCK_Tail);
                    expect(call->paramHasAttr(
                        0u, llvm::Attribute::NonNull));
                    expect(call->paramHasAttr(
                        1u, llvm::Attribute::NoUndef));
                }
            }
        }
        expect(rewritten_calls == 4u);
    };

    "HIP constant dispatcher specializes one Cartesian tuple"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            define private fastcc i32 @dispatch(
                ptr nonnull %state, i8 noundef %pipeline,
                i16 noundef %kind, i32 noundef %value) {
            entry:
              %pipeline.wide = zext i8 %pipeline to i32
              %kind.wide = zext i16 %kind to i32
              %pipeline.term = mul i32 %pipeline.wide, 1000
              %kind.term = mul i32 %kind.wide, 10
              %constant.term = add i32 %pipeline.term, %kind.term
              %result = add i32 %constant.term, %value
              ret i32 %result
            }
            define i32 @caller(ptr %state, i32 %value) {
            entry:
              %a = tail call fastcc i32 @dispatch(
                  ptr nonnull %state, i8 noundef 0,
                  i16 noundef 1, i32 noundef %value)
              %b = tail call fastcc i32 @dispatch(
                  ptr nonnull %state, i8 noundef 0,
                  i16 noundef 2, i32 noundef %value)
              %c = tail call fastcc i32 @dispatch(
                  ptr nonnull %state, i8 noundef 1,
                  i16 noundef 1, i32 noundef %value)
              %d = tail call fastcc i32 @dispatch(
                  ptr nonnull %state, i8 noundef 1,
                  i16 noundef 2, i32 noundef %value)
              %ab = add i32 %a, %b
              %cd = add i32 %c, %d
              %result = add i32 %ab, %cd
              ret i32 %result
            }
        )");
        expect(module != nullptr);
        auto *dispatch = module->getFunction("dispatch");
        expect(dispatch != nullptr);
        for (auto argument_index : {1u, 2u}) {
            dispatch->getArg(argument_index)->addAttr(llvm::Attribute::get(context, llvm_constant_argument_specialization_attribute));
        }

        auto stats =
            specialize_marked_constant_integer_arguments(*module);
        expect(stats.rewritten_function_count == 1u);
        expect(stats.cloned_function_count == 4u);
        expect(stats.merged_clone_count == 0u);
        expect(stats.rewritten_call_count == 4u);
        expect(!llvm::verifyModule(*module));
        expect(module->getFunction("dispatch") == nullptr);

        for (auto name : {
                 "dispatch.constant.0.1",
                 "dispatch.constant.0.2",
                 "dispatch.constant.1.1",
                 "dispatch.constant.1.2"}) {
            auto *specialized = module->getFunction(name);
            expect(specialized != nullptr);
            if (specialized == nullptr) { continue; }
            expect(specialized->arg_size() == 2u);
            expect(specialized->getArg(0u)->hasAttribute(
                llvm::Attribute::NonNull));
            expect(specialized->getArg(1u)->hasAttribute(
                llvm::Attribute::NoUndef));
        }

        auto rewritten_calls = 0u;
        for (auto &block : *module->getFunction("caller")) {
            for (auto &instruction : block) {
                if (auto *call =
                        llvm::dyn_cast<llvm::CallInst>(&instruction)) {
                    rewritten_calls++;
                    expect(call->arg_size() == 2u);
                    expect(call->getTailCallKind() ==
                           llvm::CallInst::TCK_Tail);
                    expect(call->paramHasAttr(
                        0u, llvm::Attribute::NonNull));
                    expect(call->paramHasAttr(
                        1u, llvm::Attribute::NoUndef));
                }
            }
        }
        expect(rewritten_calls == 4u);
    };

    "HIP constant dispatcher merges equal specialized bodies"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            define private fastcc i32 @dispatch(
                i32 %pipeline, i32 %value) unnamed_addr {
            entry:
              switch i32 %pipeline, label %invalid [
                i32 0, label %pipeline.0
                i32 1, label %pipeline.1
              ]
            pipeline.0:
              %a = add i32 %value, 7
              ret i32 %a
            pipeline.1:
              %b = add i32 %value, 7
              ret i32 %b
            invalid:
              unreachable
            }
            define i32 @caller(i32 %value) {
            entry:
              %a = call fastcc i32 @dispatch(i32 0, i32 %value)
              %b = call fastcc i32 @dispatch(i32 1, i32 %value)
              %result = add i32 %a, %b
              ret i32 %result
            }
        )");
        expect(module != nullptr);
        auto *dispatch = module->getFunction("dispatch");
        dispatch->getArg(0u)->addAttr(llvm::Attribute::get(
            context,
            llvm_constant_argument_specialization_attribute));
        auto stats =
            specialize_marked_constant_integer_arguments(*module);
        expect(stats.rewritten_function_count == 1u);
        expect(stats.cloned_function_count == 2u);
        expect(stats.merged_clone_count == 1u);
        expect(stats.rewritten_call_count == 2u);
        expect(!llvm::verifyModule(*module));

        llvm::Function *shared_target = nullptr;
        auto call_count = 0u;
        for (auto &block : *module->getFunction("caller")) {
            for (auto &instruction : block) {
                if (auto *call =
                        llvm::dyn_cast<llvm::CallInst>(&instruction)) {
                    call_count++;
                    if (shared_target == nullptr) {
                        shared_target = call->getCalledFunction();
                    } else {
                        expect(call->getCalledFunction() ==
                               shared_target);
                    }
                }
            }
        }
        expect(call_count == 2u);
        expect(shared_target != nullptr);
        expect(shared_target->arg_size() == 1u);
    };

    "HIP constant dispatcher rejects a dynamic identity atomically"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            define private i32 @dispatch(i32 %pipeline, i32 %value) {
            entry:
              %result = add i32 %pipeline, %value
              ret i32 %result
            }
            define i32 @caller(i32 %dynamic) {
            entry:
              %a = call i32 @dispatch(i32 0, i32 1)
              %b = call i32 @dispatch(i32 %dynamic, i32 2)
              %result = add i32 %a, %b
              ret i32 %result
            }
        )");
        expect(module != nullptr);
        auto *dispatch = module->getFunction("dispatch");
        dispatch->getArg(0u)->addAttr(llvm::Attribute::get(
            context,
            llvm_constant_argument_specialization_attribute));
        auto stats =
            specialize_marked_constant_integer_arguments(*module);
        expect(stats.rewritten_function_count == 0u);
        expect(stats.cloned_function_count == 0u);
        expect(stats.rewritten_call_count == 0u);
        expect(!llvm::verifyModule(*module));
        dispatch = module->getFunction("dispatch");
        expect(dispatch != nullptr);
        expect(!dispatch->getArg(0u)->hasAttribute(
            llvm_constant_argument_specialization_attribute));
        expect(dispatch->arg_size() == 2u);
    };

    "HIP constant dispatcher rejects a partly dynamic tuple atomically"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            define private i32 @dispatch(
                i32 %pipeline, i32 %kind, i32 %value) {
            entry:
              %a = add i32 %pipeline, %kind
              %result = add i32 %a, %value
              ret i32 %result
            }
            define i32 @caller(i32 %dynamic) {
            entry:
              %a = call i32 @dispatch(i32 0, i32 1, i32 1)
              %b = call i32 @dispatch(i32 1, i32 %dynamic, i32 2)
              %result = add i32 %a, %b
              ret i32 %result
            }
        )");
        expect(module != nullptr);
        auto *dispatch = module->getFunction("dispatch");
        for (auto argument_index : {0u, 1u}) {
            dispatch->getArg(argument_index)->addAttr(llvm::Attribute::get(context, llvm_constant_argument_specialization_attribute));
        }
        auto stats =
            specialize_marked_constant_integer_arguments(*module);
        expect(stats.rewritten_function_count == 0u);
        expect(stats.cloned_function_count == 0u);
        expect(stats.rewritten_call_count == 0u);
        expect(!llvm::verifyModule(*module));
        dispatch = module->getFunction("dispatch");
        expect(dispatch != nullptr);
        for (auto argument_index : {0u, 1u}) {
            expect(!dispatch->getArg(argument_index)->hasAttribute(llvm_constant_argument_specialization_attribute));
        }
        expect(dispatch->arg_size() == 3u);
    };
    return 0;
}();

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
