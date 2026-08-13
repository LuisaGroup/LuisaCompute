#include "hip_callable_inline_graph.h"
#include "ut/ut.hpp"

#include <memory>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

using namespace luisa::compute::hip;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto generated_attribute = "luisa-generated-callable";

struct DispatchModule {
    std::unique_ptr<llvm::LLVMContext> context;
    std::unique_ptr<llvm::Module> module;
};

[[nodiscard]] llvm::Function *make_leaf(
    llvm::Module &module) {
    auto &context = module.getContext();
    auto *i32 = llvm::Type::getInt32Ty(context);
    auto *function = llvm::Function::Create(
        llvm::FunctionType::get(i32, {i32}, false),
        llvm::GlobalValue::PrivateLinkage,
        "",
        module);
    function->addFnAttr(generated_attribute);
    auto *entry = llvm::BasicBlock::Create(
        context, "", function);
    llvm::IRBuilder<> builder{entry};
    builder.CreateRet(function->getArg(0u));
    return function;
}

[[nodiscard]] DispatchModule make_dispatch_module(
    bool unreachable_default,
    bool multiple_calls_in_first_case) {
    auto result = DispatchModule{};
    result.context = std::make_unique<llvm::LLVMContext>();
    result.module = std::make_unique<llvm::Module>(
        "callable-inline-graph-test", *result.context);
    auto &module = *result.module;
    auto &context = *result.context;
    auto *i32 = llvm::Type::getInt32Ty(context);
    auto *leaf_a = make_leaf(module);
    auto *leaf_b = make_leaf(module);
    auto *common = unreachable_default ?
                       nullptr :
                       make_leaf(module);
    auto *dispatch = llvm::Function::Create(
        llvm::FunctionType::get(i32, {i32, i32}, false),
        llvm::GlobalValue::PrivateLinkage,
        "",
        module);
    dispatch->addFnAttr(generated_attribute);
    auto *entry = llvm::BasicBlock::Create(
        context, "", dispatch);
    auto *case_a = llvm::BasicBlock::Create(
        context, "", dispatch);
    auto *case_b = llvm::BasicBlock::Create(
        context, "", dispatch);
    auto *case_merge = llvm::BasicBlock::Create(
        context, "", dispatch);
    auto *default_block = llvm::BasicBlock::Create(
        context, "", dispatch);
    auto *all_merge = unreachable_default ?
                          case_merge :
                          llvm::BasicBlock::Create(
                              context, "", dispatch);

    auto *selector = dispatch->getArg(0u);
    auto *input = dispatch->getArg(1u);
    llvm::IRBuilder<> builder{entry};
    auto *switch_inst = builder.CreateSwitch(
        selector, default_block, 2u);
    switch_inst->addCase(
        llvm::ConstantInt::get(i32, 0u), case_a);
    switch_inst->addCase(
        llvm::ConstantInt::get(i32, 1u), case_b);

    builder.SetInsertPoint(case_a);
    auto *value_a = builder.CreateCall(leaf_a, {input});
    if (multiple_calls_in_first_case) {
        value_a = builder.CreateCall(leaf_b, {value_a});
    }
    builder.CreateBr(case_merge);

    builder.SetInsertPoint(case_b);
    auto *value_b = builder.CreateCall(leaf_b, {input});
    builder.CreateBr(case_merge);

    builder.SetInsertPoint(case_merge);
    auto *case_result = builder.CreatePHI(i32, 2u);
    case_result->addIncoming(value_a, case_a);
    case_result->addIncoming(value_b, case_b);
    if (unreachable_default) {
        builder.CreateRet(case_result);
        builder.SetInsertPoint(default_block);
        builder.CreateUnreachable();
    } else {
        builder.CreateBr(all_merge);
        builder.SetInsertPoint(default_block);
        builder.CreateBr(all_merge);
        builder.SetInsertPoint(all_merge);
        auto *selected = builder.CreatePHI(i32, 2u);
        selected->addIncoming(case_result, case_merge);
        selected->addIncoming(input, default_block);
        builder.CreateRet(builder.CreateCall(common, {selected}));
    }
    return result;
}

static auto suite = [] {
    "HIP callable graph recognizes valid cases with unreachable default"_test = [] {
        auto test_module = make_dispatch_module(true, false);
        auto graph = build_generated_callable_inline_graph(
            *test_module.module, generated_attribute);
        auto dispatch_index = graph.functions.size() - 1u;
        expect(graph.functions.size() == 3u);
        expect(dispatch_index < graph.nodes.size());
        if (dispatch_index >= graph.nodes.size()) { return; }
        const auto &dispatch = graph.nodes[dispatch_index];
        expect(dispatch.callees.size() == 2u);
        expect(dispatch.alternative_call_groups.size() == 1u);
        expect(dispatch.alternative_call_groups.front().size() == 2u);
    };

    "HIP callable graph rejects a successor without a unique frontier"_test = [] {
        auto test_module = make_dispatch_module(true, true);
        auto graph = build_generated_callable_inline_graph(
            *test_module.module, generated_attribute);
        auto dispatch_index = graph.functions.size() - 1u;
        expect(dispatch_index < graph.nodes.size());
        if (dispatch_index >= graph.nodes.size()) { return; }
        expect(graph.nodes[dispatch_index]
                   .alternative_call_groups.empty());
    };

    "HIP callable graph includes a reachable default in the merge proof"_test = [] {
        auto test_module = make_dispatch_module(false, false);
        auto graph = build_generated_callable_inline_graph(
            *test_module.module, generated_attribute);
        auto dispatch_index = graph.functions.size() - 1u;
        expect(dispatch_index < graph.nodes.size());
        if (dispatch_index >= graph.nodes.size()) { return; }
        const auto &dispatch = graph.nodes[dispatch_index];
        expect(dispatch.callees.size() == 3u);
        expect(dispatch.alternative_call_groups.size() == 1u);
        expect(dispatch.alternative_call_groups.front().size() == 2u);
        for (auto call_site : dispatch.alternative_call_groups.front()) {
            expect(dispatch.callees[call_site] < 2u);
        }
    };
    return 0;
}();

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
