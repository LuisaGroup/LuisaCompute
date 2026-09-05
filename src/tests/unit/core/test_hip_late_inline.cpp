#include "hip_callable_abi.h"
#include "ut/ut.hpp"

#include <memory>
#include <string>
#include <string_view>

#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/SourceMgr.h>

using namespace luisa::compute::hip;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] std::unique_ptr<llvm::Module> parse_module(
    llvm::LLVMContext &context, std::string_view text) {
    llvm::SMDiagnostic diagnostic;
    auto module = llvm::parseAssemblyString(text, diagnostic, context);
    if (!module) {
        diagnostic.print("test_hip_late_inline", llvm::errs());
    }
    return module;
}

template<typename Instruction>
[[nodiscard]] unsigned count_instructions(const llvm::Function &function) {
    auto count = 0u;
    for (auto &block : function) {
        for (auto &instruction : block) {
            count += llvm::isa<Instruction>(instruction);
        }
    }
    return count;
}

static auto suite = [] {
    "HIP late inlining promotes newly nonescaping scalar and vector state"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-p5:32:32-i64:64-n32:64-A5"
            define private i32 @wide(ptr %count, ptr %rgb, <32 x i32> %payload, i1 %pick) #0 {
            entry:
              %old = load i32, ptr %count
              %lane = extractelement <32 x i32> %payload, i32 0
              br i1 %pick, label %update, label %exit
            update:
              %next = add i32 %old, %lane
              store i32 %next, ptr %count
              store <4 x float> <float 1.0, float 2.0, float 3.0, float 0.0>, ptr %rgb
              br label %exit
            exit:
              %result = load i32, ptr %count
              ret i32 %result
            }
            define i32 @caller(i32 %seed, <32 x i32> %payload, i1 %pick) {
            entry:
              %count = alloca i32, align 4, addrspace(5)
              %rgb = alloca [4 x float], align 16, addrspace(5)
              store i32 %seed, ptr addrspace(5) %count
              store <4 x float> zeroinitializer, ptr addrspace(5) %rgb
              %generic.count = addrspacecast ptr addrspace(5) %count to ptr
              %generic.rgb = addrspacecast ptr addrspace(5) %rgb to ptr
              %result = call i32 @wide(ptr %generic.count, ptr %generic.rgb, <32 x i32> %payload, i1 %pick)
              %after = load i32, ptr addrspace(5) %count
              %color = load <4 x float>, ptr addrspace(5) %rgb
              %red = extractelement <4 x float> %color, i32 0
              %red.int = fptoui float %red to i32
              %sum = add i32 %after, %result
              %answer = add i32 %sum, %red.int
              ret i32 %answer
            }
            define i32 @untouched(i32 %seed) {
            entry:
              %slot = alloca i32, addrspace(5)
              store i32 %seed, ptr addrspace(5) %slot
              %result = load i32, ptr addrspace(5) %slot
              ret i32 %result
            }
            attributes #0 = { "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        if (!module) { return; }
        auto stats = inline_unique_oversized_generated_callables(*module);
        expect(stats.inlined_function_count == 1u);
        expect(module->getFunction("wide") == nullptr);
        expect(!llvm::verifyModule(*module));
        auto &caller = *module->getFunction("caller");
        expect(count_instructions<llvm::AllocaInst>(caller) == 0u);
        expect(count_instructions<llvm::LoadInst>(caller) == 0u);
        expect(count_instructions<llvm::StoreInst>(caller) == 0u);
        // The conditional update must retain its incoming initialized values,
        // including the RGB zero, as SSA alternatives rather than memory.
        expect(count_instructions<llvm::PHINode>(caller) >= 2u);
        auto preserved_zero = false;
        for (auto &block : caller) {
            for (auto &phi : block.phis()) {
                if (!phi.getType()->isFPOrFPVectorTy()) { continue; }
                for (auto &incoming : phi.incoming_values()) {
                    if (auto constant = llvm::dyn_cast<llvm::Constant>(
                            incoming.get())) {
                        preserved_zero |= constant->isNullValue();
                    }
                }
            }
        }
        expect(preserved_zero);
        expect(count_instructions<llvm::AllocaInst>(
                   *module->getFunction("untouched")) == 1u);
    };

    "HIP late inlining retains escaped and volatile private state"_test = [] {
        for (auto escaped : {false, true}) {
            auto ir = std::string{R"(
                target datalayout = "e-p:64:64-p5:32:32-i64:64-n32:64-A5"
                declare void @observe(ptr)
                define private i32 @wide(ptr %state, <32 x i32> %payload) #0 {
                entry:
                  %lane = extractelement <32 x i32> %payload, i32 0
            )"};
            ir += escaped ?
                      "store i32 %lane, ptr %state\ncall void @observe(ptr %state)\n" :
                      "store volatile i32 %lane, ptr %state\n";
            ir += R"(
                  %result = load i32, ptr %state
                  ret i32 %result
                }
                define i32 @caller(<32 x i32> %payload) {
                entry:
                  %state = alloca i32, addrspace(5)
                  %generic = addrspacecast ptr addrspace(5) %state to ptr
                  %result = call i32 @wide(ptr %generic, <32 x i32> %payload)
                  ret i32 %result
                }
                attributes #0 = { "luisa-generated-callable" }
            )";
            llvm::LLVMContext context;
            auto module = parse_module(context, ir);
            expect(module != nullptr);
            if (!module) { continue; }
            auto stats = inline_unique_oversized_generated_callables(*module);
            expect(stats.inlined_function_count == 1u);
            expect(!llvm::verifyModule(*module));
            auto &caller = *module->getFunction("caller");
            expect(count_instructions<llvm::AllocaInst>(caller) == 1u);
            expect(count_instructions<llvm::StoreInst>(caller) == 1u);
            if (escaped) {
                expect(!module->getFunction("observe")->use_empty());
            } else {
                auto volatile_stores = 0u;
                for (auto &block : caller) {
                    for (auto &instruction : block) {
                        if (auto store = llvm::dyn_cast<llvm::StoreInst>(
                                &instruction)) {
                            volatile_stores += store->isVolatile();
                        }
                    }
                }
                expect(volatile_stores == 1u);
            }
        }
    };

    "HIP late inlining cleans the surviving caller of a nested chain"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-p5:32:32-i64:64-n32:64-A5"
            define private i32 @leaf(ptr %state, <32 x i32> %payload) #0 {
            entry:
              %lane = extractelement <32 x i32> %payload, i32 0
              store i32 %lane, ptr %state
              ret i32 %lane
            }
            define private i32 @middle(ptr %state, <32 x i32> %payload) #0 {
            entry:
              %result = call i32 @leaf(ptr %state, <32 x i32> %payload)
              ret i32 %result
            }
            define i32 @caller(<32 x i32> %payload) {
            entry:
              %state = alloca i32, addrspace(5)
              %generic = addrspacecast ptr addrspace(5) %state to ptr
              %result = call i32 @middle(ptr %generic, <32 x i32> %payload)
              %after = load i32, ptr addrspace(5) %state
              %sum = add i32 %after, %result
              ret i32 %sum
            }
            attributes #0 = { "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        if (!module) { return; }
        auto stats = inline_unique_oversized_generated_callables(*module);
        expect(stats.inlined_function_count == 2u);
        expect(module->getFunction("leaf") == nullptr);
        expect(module->getFunction("middle") == nullptr);
        expect(!llvm::verifyModule(*module));
        auto &caller = *module->getFunction("caller");
        expect(count_instructions<llvm::AllocaInst>(caller) == 0u);
        expect(count_instructions<llvm::LoadInst>(caller) == 0u);
        expect(count_instructions<llvm::StoreInst>(caller) == 0u);
    };
    "HIP late inlining handles oversized returns independently of arguments"_test = [] {
        for (auto lanes : {32u, 33u}) {
            auto type = "[" + std::to_string(lanes) + " x i32]";
            auto ir =
                "target datalayout = \"e-p:64:64-p5:32:32-i64:64-n32:64-A5\"\n"
                "define private " + type + " @wide(i32 %seed) #0 {\n"
                "  %value = insertvalue " + type + " zeroinitializer, i32 %seed, 0\n"
                "  ret " + type + " %value\n}\n"
                "define i32 @caller(i32 %seed) {\n"
                "  %value = call " + type + " @wide(i32 %seed)\n"
                "  %result = extractvalue " + type + " %value, 0\n"
                "  ret i32 %result\n}\n"
                "attributes #0 = { \"luisa-generated-callable\" }\n";
            llvm::LLVMContext context;
            auto module = parse_module(context, ir);
            expect(module != nullptr);
            if (!module) { continue; }
            auto stats = inline_unique_oversized_generated_callables(*module);
            expect(stats.inlined_function_count == (lanes == 33u ? 1u : 0u));
            expect(stats.removed_return_locations == (lanes == 33u ? 33u : 0u));
            expect((module->getFunction("wide") == nullptr) == (lanes == 33u));
            expect(!llvm::verifyModule(*module));
        }
    };

    "HIP late inlining preserves shared self-recursive and address-taken boundaries"_test = [] {
        llvm::LLVMContext context;
        auto module = parse_module(context, R"(
            target datalayout = "e-p:64:64-p5:32:32-i64:64-n32:64-A5"
            @address = global ptr @address_taken
            define private i32 @address_taken([33 x i32] %value) #0 {
              %result = extractvalue [33 x i32] %value, 0
              ret i32 %result
            }
            define private i32 @recursive([33 x i32] %value) #0 {
              %result = call i32 @recursive([33 x i32] %value)
              ret i32 %result
            }
            define private i32 @shared([33 x i32] %value) #0 {
              %result = extractvalue [33 x i32] %value, 0
              ret i32 %result
            }
            define i32 @caller([33 x i32] %first, [33 x i32] %second) {
              %a = call i32 @shared([33 x i32] %first)
              %b = call i32 @shared([33 x i32] %second)
              %sum = add i32 %a, %b
              ret i32 %sum
            }
            attributes #0 = { "luisa-generated-callable" }
        )");
        expect(module != nullptr);
        if (!module) { return; }
        auto stats = inline_unique_oversized_generated_callables(*module);
        expect(stats.inlined_function_count == 0u);
        expect(!llvm::verifyModule(*module));
        for (auto name : {"address_taken", "recursive", "shared"}) {
            expect(module->getFunction(name) != nullptr);
        }
    };
    return 0;
}();

}// namespace
