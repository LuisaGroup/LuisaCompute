#include "hip_callable_abi.h"
#include "ut/ut.hpp"

#include <memory>
#include <string_view>

#include <llvm/AsmParser/Parser.h>
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
    return 0;
}();

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
