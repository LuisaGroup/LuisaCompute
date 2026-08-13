#include "hip_callable_abi.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Local.h>

#include <luisa/core/logging.h>

#include <algorithm>
#include <map>
#include <vector>

namespace luisa::compute::hip {

namespace {

using AggregatePath = std::vector<unsigned>;

struct AggregateProjection {
    llvm::ExtractValueInst *terminal;
    AggregatePath path;
    llvm::SmallVector<llvm::FreezeInst *, 2> freezes;
};

struct AggregateArgumentPlan {
    llvm::Argument *argument;
    std::vector<AggregatePath> paths;
    std::vector<AggregateProjection> projections;
};

// The analysis domain for one aggregate argument is the finite lattice
//
//   unused < known leaf-path set < whole aggregate.
//
// collect_aggregate_projections computes the least upper bound of every use.
// freeze is transparent except that it is reproduced per selected leaf;
// extractvalue appends a path component. Any other operation immediately
// yields `whole aggregate`, making the transformation inapplicable. Rewriting
// therefore preserves exactly the observable leaf values and never guesses at
// a partially understood aggregate operation.

[[nodiscard]] bool collect_aggregate_projections(
    llvm::Value *value, AggregatePath path,
    llvm::SmallVector<llvm::FreezeInst *, 2> freezes,
    llvm::SmallPtrSetImpl<llvm::Value *> &active_values,
    std::vector<AggregateProjection> &projections) noexcept {
    // The root argument may be completely unused and can then be removed.
    // A dead aggregate instruction below the root is deliberately rejected:
    // it would otherwise remain as an unmodeled use after the rewrite. The
    // ordinary optimizer normally removes such instructions before this pass.
    if (value->use_empty()) {
        return path.empty() && freezes.empty();
    }
    if (!active_values.insert(value).second) { return false; }
    for (auto *user : value->users()) {
        if (auto *freeze = llvm::dyn_cast<llvm::FreezeInst>(user)) {
            auto nested_freezes = freezes;
            nested_freezes.emplace_back(freeze);
            if (!collect_aggregate_projections(
                    freeze, path, std::move(nested_freezes),
                    active_values, projections)) {
                active_values.erase(value);
                return false;
            }
            continue;
        }
        if (auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(user)) {
            if (extract->getAggregateOperand() != value) { return false; }
            auto nested_path = path;
            nested_path.insert(
                nested_path.end(), extract->idx_begin(),
                extract->idx_end());
            if (extract->getType()->isAggregateType()) {
                if (!collect_aggregate_projections(
                        extract, std::move(nested_path), freezes,
                        active_values, projections)) {
                    active_values.erase(value);
                    return false;
                }
            } else {
                projections.emplace_back(AggregateProjection{
                    .terminal = extract,
                    .path = std::move(nested_path),
                    .freezes = freezes});
            }
            continue;
        }
        active_values.erase(value);
        return false;
    }
    active_values.erase(value);
    return true;
}

[[nodiscard]] llvm::AttributeList remap_argument_attributes(
    llvm::LLVMContext &context, llvm::AttributeList attributes,
    llvm::ArrayRef<const AggregateArgumentPlan *> plans) noexcept {
    llvm::SmallVector<llvm::AttributeSet, 16> parameter_attributes;
    for (auto i = 0u; i < plans.size(); i++) {
        if (auto *plan = plans[i]) {
            parameter_attributes.resize(
                parameter_attributes.size() + plan->paths.size());
        } else {
            parameter_attributes.emplace_back(
                attributes.getParamAttrs(i));
        }
    }
    return llvm::AttributeList::get(
        context, attributes.getFnAttrs(), attributes.getRetAttrs(),
        parameter_attributes);
}

}// namespace

AggregateArgumentSpecializationStats
specialize_generated_callable_aggregate_arguments(
    llvm::Module &module,
    llvm::StringRef callable_attribute) noexcept {
    llvm::SmallVector<llvm::Function *, 32> functions;
    for (auto &function : module) {
        if (!function.isDeclaration() &&
            function.hasFnAttribute(callable_attribute) &&
            function.hasFnAttribute(llvm::Attribute::NoInline)) {
            functions.emplace_back(&function);
        }
    }

    auto stats = AggregateArgumentSpecializationStats{};
    const auto &data_layout = module.getDataLayout();
    for (auto *function : functions) {
        if (function->isVarArg() || function->hasAddressTaken() ||
            function->hasPersonalityFn() || function->hasPrefixData() ||
            function->hasPrologueData()) {
            continue;
        }

        std::vector<AggregateArgumentPlan> owned_plans;
        std::vector<const AggregateArgumentPlan *> plans(
            function->arg_size(), nullptr);
        auto argument_index = 0u;
        for (auto &argument : function->args()) {
            auto *type = argument.getType();
            if (!type->isAggregateType()) {
                argument_index++;
                continue;
            }
            auto projections = std::vector<AggregateProjection>{};
            llvm::SmallPtrSet<llvm::Value *, 8> active_values;
            if (!collect_aggregate_projections(
                    &argument, {}, {}, active_values, projections)) {
                argument_index++;
                continue;
            }
            auto paths = std::vector<AggregatePath>{};
            paths.reserve(projections.size());
            for (auto &projection : projections) {
                paths.emplace_back(projection.path);
            }
            std::sort(paths.begin(), paths.end());
            paths.erase(
                std::unique(paths.begin(), paths.end()), paths.end());

            auto original_size =
                data_layout.getTypeAllocSize(type).getFixedValue();
            auto projected_size = size_t{};
            for (auto &path : paths) {
                auto *projected_type =
                    llvm::ExtractValueInst::getIndexedType(type, path);
                LUISA_ASSERT(projected_type != nullptr &&
                             !projected_type->isAggregateType());
                projected_size += data_layout.getTypeAllocSize(
                    projected_type).getFixedValue();
            }
            if (projected_size >= original_size) {
                argument_index++;
                continue;
            }
            owned_plans.emplace_back(AggregateArgumentPlan{
                .argument = &argument,
                .paths = std::move(paths),
                .projections = std::move(projections)});
            plans[argument_index] = &owned_plans.back();
            argument_index++;
        }
        if (owned_plans.empty()) { continue; }

        // Rebind plan pointers after vector growth. The plans are indexed by
        // the stable original arguments, not by storage addresses.
        for (auto &plan : owned_plans) {
            plans[plan.argument->getArgNo()] = &plan;
        }

        llvm::SmallVector<llvm::CallInst *, 16> calls;
        auto supported_uses = true;
        for (auto *user : function->users()) {
            auto *call = llvm::dyn_cast<llvm::CallInst>(user);
            if (call == nullptr || call->getCalledOperand() != function ||
                call->isMustTailCall()) {
                supported_uses = false;
                break;
            }
            calls.emplace_back(call);
        }
        if (!supported_uses || calls.empty()) { continue; }

        for (auto &plan : owned_plans) {
            auto original_size = data_layout.getTypeAllocSize(
                plan.argument->getType()).getFixedValue();
            auto projected_size = size_t{};
            for (auto &path : plan.paths) {
                projected_size += data_layout.getTypeAllocSize(
                    llvm::ExtractValueInst::getIndexedType(
                        plan.argument->getType(), path)).getFixedValue();
            }
            stats.removed_aggregate_bytes +=
                original_size - projected_size;
        }

        llvm::SmallVector<llvm::Type *, 16> parameter_types;
        for (auto &argument : function->args()) {
            if (auto *plan = plans[argument.getArgNo()]) {
                for (auto &path : plan->paths) {
                    parameter_types.emplace_back(
                        llvm::ExtractValueInst::getIndexedType(
                            argument.getType(), path));
                }
            } else {
                parameter_types.emplace_back(argument.getType());
            }
        }
        auto *new_function_type = llvm::FunctionType::get(
            function->getReturnType(), parameter_types, false);
        auto old_name = function->getName().str();
        function->setName(old_name + ".aggregate.abi.old");
        auto *new_function = llvm::Function::Create(
            new_function_type, function->getLinkage(), old_name,
            &module);
        new_function->copyAttributesFrom(function);
        new_function->setAttributes(remap_argument_attributes(
            module.getContext(), function->getAttributes(), plans));
        new_function->setCallingConv(function->getCallingConv());
        new_function->copyMetadata(function, 0u);

        // Move the original CFG unchanged. Projected parameters dominate the
        // entire function, and any scalar freezes can be placed at the first
        // legal insertion point of the existing entry block. Avoiding a new
        // predecessor is important: an LLVM entry block is allowed to contain
        // zero-incoming phi nodes, which would become invalid if a synthetic
        // prologue branched to it.
        new_function->splice(new_function->end(), function);
        auto *entry = &new_function->getEntryBlock();
        llvm::IRBuilder<> builder{
            &*entry->getFirstInsertionPt()};
        auto new_argument = new_function->arg_begin();
        for (auto &old_argument : function->args()) {
            if (auto *plan = plans[old_argument.getArgNo()]) {
                std::map<AggregatePath, llvm::Value *> projected_values;
                for (auto &path : plan->paths) {
                    projected_values.emplace(path, new_argument++);
                }
                llvm::DenseMap<llvm::FreezeInst *,
                               std::map<AggregatePath, llvm::Value *>>
                    frozen_values;
                for (auto &projection : plan->projections) {
                    auto *value = projected_values.at(projection.path);
                    for (auto *freeze : projection.freezes) {
                        auto &values = frozen_values[freeze];
                        auto [iter, inserted] = values.try_emplace(
                            projection.path, nullptr);
                        if (inserted) {
                            iter->second = builder.CreateFreeze(value);
                        }
                        value = iter->second;
                    }
                    projection.terminal->replaceAllUsesWith(value);
                }
                for (auto &projection : plan->projections) {
                    llvm::RecursivelyDeleteTriviallyDeadInstructions(
                        projection.terminal);
                }
                LUISA_ASSERT(old_argument.use_empty(),
                             "Specialized aggregate argument still has uses.");
            } else {
                old_argument.replaceAllUsesWith(new_argument++);
            }
        }

        for (auto *call : calls) {
            llvm::IRBuilder<> call_builder{call};
            llvm::SmallVector<llvm::Value *, 16> arguments;
            for (auto i = 0u; i < call->arg_size(); i++) {
                auto *actual = call->getArgOperand(i);
                if (auto *plan = plans[i]) {
                    for (auto &path : plan->paths) {
                        arguments.emplace_back(
                            call_builder.CreateExtractValue(actual, path));
                    }
                } else {
                    arguments.emplace_back(actual);
                }
            }
            llvm::SmallVector<llvm::OperandBundleDef, 2> bundles;
            call->getOperandBundlesAsDefs(bundles);
            auto *new_call = call_builder.CreateCall(
                new_function_type, new_function, arguments, bundles);
            new_call->setCallingConv(call->getCallingConv());
            new_call->setTailCallKind(call->getTailCallKind());
            new_call->setAttributes(remap_argument_attributes(
                module.getContext(), call->getAttributes(), plans));
            new_call->setDebugLoc(call->getDebugLoc());
            new_call->copyMetadata(*call);
            new_call->setFastMathFlags(call->getFastMathFlags());
            new_call->takeName(call);
            call->replaceAllUsesWith(new_call);
            call->eraseFromParent();
        }
        LUISA_ASSERT(function->use_empty());
        function->eraseFromParent();
        stats.rewritten_function_count++;
    }
    return stats;
}

}// namespace luisa::compute::hip
