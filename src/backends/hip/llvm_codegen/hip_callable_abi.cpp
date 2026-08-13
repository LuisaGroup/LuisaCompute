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
#include <limits>
#include <map>
#include <optional>
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

struct LargeReturnPlan {
    llvm::Function *original;
    llvm::Function *replacement{};
    llvm::Type *return_type;
    size_t return_bytes;
};

// Luisa's retained generated callables use FastCC so LLVM can optimize their
// internal ABI. Some focused/runtime-generated modules retain the default C
// convention. Both conventions use RetCC_AMDGPU_Func for AMDGPU function
// returns, and the transform preserves the convention identically on the
// replacement function and every direct call. Other conventions may carry
// target- or language-specific ABI rules that are not modeled here.
[[nodiscard]] bool supported_large_return_calling_convention(
    llvm::CallingConv::ID convention) noexcept {
    return convention == llvm::CallingConv::C ||
           convention == llvm::CallingConv::Fast;
}

// RetCC_AMDGPU_Func exposes 32 32-bit VGPR return locations. This computes a
// conservative upper bound on the number of those locations occupied after
// the calling convention's aggregate decomposition. Aggregates are decomposed
// recursively without charging layout padding. Narrow scalar leaves consume
// one location because the convention may promote them; 16-bit vector pairs
// are legal packed return values, while wider vector leaves occupy one
// location per 32-bit chunk. Returning nullopt rejects scalable or unsized
// types rather than guessing.
[[nodiscard]] std::optional<size_t> amdgpu_return_vgpr_count(
    llvm::Type *type, const llvm::DataLayout &data_layout) noexcept {
    if (type->isVoidTy()) { return 0u; }
    if (auto *structure = llvm::dyn_cast<llvm::StructType>(type)) {
        if (structure->isOpaque()) { return std::nullopt; }
        auto count = size_t{};
        for (auto *element : structure->elements()) {
            auto element_count =
                amdgpu_return_vgpr_count(element, data_layout);
            if (!element_count ||
                *element_count >
                    std::numeric_limits<size_t>::max() - count) {
                return std::nullopt;
            }
            count += *element_count;
        }
        return count;
    }
    if (auto *array = llvm::dyn_cast<llvm::ArrayType>(type)) {
        auto element_count =
            amdgpu_return_vgpr_count(array->getElementType(), data_layout);
        if (!element_count ||
            (array->getNumElements() != 0u &&
             *element_count > std::numeric_limits<size_t>::max() /
                                  array->getNumElements())) {
            return std::nullopt;
        }
        return *element_count * array->getNumElements();
    }
    if (llvm::isa<llvm::ScalableVectorType>(type) || !type->isSized()) {
        return std::nullopt;
    }
    if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        auto *element = vector->getElementType();
        auto element_bits = data_layout.getTypeSizeInBits(element);
        if (element_bits.isScalable()) { return std::nullopt; }
        const auto bits = element_bits.getFixedValue();
        const auto lanes = vector->getNumElements();
        if (bits < 16u) {
            // RetCC_AMDGPU_Func has no packed sub-16-bit vector location, so
            // legalization may promote every lane independently.
            return lanes;
        }
        if (bits == 16u) { return (lanes + 1u) / 2u; }
        const auto locations_per_lane = (bits + 31u) / 32u;
        if (lanes != 0u &&
            locations_per_lane >
                std::numeric_limits<size_t>::max() / lanes) {
            return std::nullopt;
        }
        return locations_per_lane * lanes;
    }
    auto bits = data_layout.getTypeSizeInBits(type);
    if (bits.isScalable()) { return std::nullopt; }
    return std::max<size_t>(
        1u, (bits.getFixedValue() + 31u) / 32u);
}

[[nodiscard]] llvm::AttributeList prepend_result_pointer_attributes(
    llvm::LLVMContext &context, llvm::AttributeList attributes,
    size_t old_parameter_count) noexcept {
    // The replacement writes the result through its new parameter, so a
    // formerly speculatable call is no longer speculatable. A `returned`
    // parameter describes equality with the direct SSA return and is invalid
    // once that return becomes void. All other old parameters retain their
    // attributes at index + 1.
    auto function_attributes = attributes.getFnAttrs().removeAttribute(
        context, llvm::Attribute::Speculatable);
    llvm::SmallVector<llvm::AttributeSet, 16> parameter_attributes;
    parameter_attributes.reserve(old_parameter_count + 1u);
    parameter_attributes.emplace_back();
    for (auto parameter_index = size_t{0u};
         parameter_index < old_parameter_count; parameter_index++) {
        parameter_attributes.emplace_back(
            attributes.getParamAttrs(
                          static_cast<unsigned>(parameter_index))
                .removeAttribute(context, llvm::Attribute::Returned));
    }
    return llvm::AttributeList::get(
        context, function_attributes, llvm::AttributeSet{},
        parameter_attributes);
}

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
                                                 projected_type)
                                      .getFixedValue();
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
                                                plan.argument->getType())
                                     .getFixedValue();
            auto projected_size = size_t{};
            for (auto &path : plan.paths) {
                projected_size += data_layout.getTypeAllocSize(
                                                 llvm::ExtractValueInst::getIndexedType(
                                                     plan.argument->getType(), path))
                                      .getFixedValue();
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

LargeReturnDemotionStats demote_generated_callable_large_returns(
    llvm::Module &module,
    llvm::StringRef callable_attribute) noexcept {
    const auto &data_layout = module.getDataLayout();
    auto plans = std::vector<LargeReturnPlan>{};
    for (auto &function : module) {
        if (function.isDeclaration() ||
            !function.hasFnAttribute(callable_attribute) ||
            !function.hasLocalLinkage() || function.isVarArg() ||
            !supported_large_return_calling_convention(
                function.getCallingConv()) ||
            function.hasAddressTaken() || function.hasMetadata() ||
            function.hasComdat() || function.hasGC() ||
            function.hasPersonalityFn() || function.hasPrefixData() ||
            function.hasPrologueData() ||
            function.hasFnAttribute(llvm::Attribute::AllocSize)) {
            continue;
        }
        auto *return_type = function.getReturnType();
        auto return_vgprs =
            amdgpu_return_vgpr_count(return_type, data_layout);
        if (!return_vgprs ||
            *return_vgprs <= amdgpu_callable_return_vgpr_limit) {
            continue;
        }
        auto supported_returns = true;
        for (auto &block : function) {
            if (auto *return_instruction =
                    llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator());
                return_instruction != nullptr &&
                return_instruction->hasMetadataOtherThanDebugLoc()) {
                supported_returns = false;
                break;
            }
        }
        if (!supported_returns) { continue; }
        auto supported_uses = true;
        for (auto *user : function.users()) {
            auto *call = llvm::dyn_cast<llvm::CallInst>(user);
            if (call == nullptr || call->getCalledOperand() != &function ||
                call->getCallingConv() != function.getCallingConv() ||
                call->hasOperandBundles() ||
                call->getTailCallKind() != llvm::CallInst::TCK_None ||
                call->hasMetadataOtherThanDebugLoc() ||
                call->getFastMathFlags().any() ||
                call->hasFnAttr(llvm::Attribute::AllocSize)) {
                supported_uses = false;
                break;
            }
        }
        if (!supported_uses || function.use_empty()) { continue; }
        plans.emplace_back(LargeReturnPlan{
            .original = &function,
            .return_type = return_type,
            .return_bytes = static_cast<size_t>(
                data_layout.getTypeAllocSize(return_type).getFixedValue())});
    }

    auto stats = LargeReturnDemotionStats{};
    if (plans.empty()) { return stats; }
    const auto alloca_address_space = data_layout.getAllocaAddrSpace();

    // First create every replacement and move each body. Calls are rewritten
    // only after this phase, so a call nested in another transformed callable
    // automatically belongs to that callable's replacement function.
    for (auto &plan : plans) {
        auto *function = plan.original;
        llvm::SmallVector<llvm::Type *, 16> parameter_types;
        parameter_types.reserve(function->arg_size() + 1u);
        parameter_types.emplace_back(
            llvm::PointerType::get(module.getContext(),
                                   alloca_address_space));
        for (auto &argument : function->args()) {
            parameter_types.emplace_back(argument.getType());
        }
        auto *replacement_type = llvm::FunctionType::get(
            llvm::Type::getVoidTy(module.getContext()),
            parameter_types, false);
        auto old_name = function->getName().str();
        function->setName(old_name + ".large.return.abi.old");
        auto *replacement = llvm::Function::Create(
            replacement_type, function->getLinkage(), old_name, &module);
        plan.replacement = replacement;
        replacement->copyAttributesFrom(function);
        replacement->setAttributes(prepend_result_pointer_attributes(
            module.getContext(), function->getAttributes(),
            function->arg_size()));
        replacement->setMemoryEffects(
            function->getMemoryEffects() |
            llvm::MemoryEffects::argMemOnly(llvm::ModRefInfo::Mod));
        replacement->setCallingConv(function->getCallingConv());
        replacement->splice(replacement->end(), function);

        auto new_argument = std::next(replacement->arg_begin());
        for (auto &old_argument : function->args()) {
            old_argument.replaceAllUsesWith(new_argument);
            new_argument->takeName(&old_argument);
            ++new_argument;
        }

        auto *result_pointer = replacement->getArg(0u);
        result_pointer->setName("return.storage");
        llvm::SmallVector<llvm::ReturnInst *, 4> returns;
        for (auto &block : *replacement) {
            if (auto *return_instruction =
                    llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator())) {
                returns.emplace_back(return_instruction);
            }
        }
        for (auto *return_instruction : returns) {
            auto *return_value = return_instruction->getReturnValue();
            LUISA_ASSERT(return_value != nullptr,
                         "Large-return callable has a void return.");
            llvm::IRBuilder<> builder{return_instruction};
            auto *store = builder.CreateStore(
                return_value, result_pointer);
            store->setAlignment(
                data_layout.getPrefTypeAlign(plan.return_type));
            auto *void_return = builder.CreateRetVoid();
            void_return->setDebugLoc(return_instruction->getDebugLoc());
            return_instruction->eraseFromParent();
        }
        stats.demoted_return_bytes += plan.return_bytes;
    }

    // All calls of one exact result type in one caller reuse one slot. Every
    // result is loaded immediately after its defining call, so the slot's live
    // intervals are disjoint even when the calls themselves are not mutually
    // exclusive. Recursive invocations have distinct machine stack frames.
    llvm::DenseMap<llvm::Function *,
                   llvm::DenseMap<llvm::Type *, llvm::AllocaInst *>>
        caller_result_slots;
    for (auto &plan : plans) {
        auto *function = plan.original;
        auto *replacement = plan.replacement;
        llvm::SmallVector<llvm::CallInst *, 16> calls;
        for (auto *user : function->users()) {
            calls.emplace_back(llvm::cast<llvm::CallInst>(user));
        }
        for (auto *call : calls) {
            auto *caller = call->getFunction();
            auto &type_slots = caller_result_slots[caller];
            auto *&slot = type_slots[plan.return_type];
            if (slot == nullptr) {
                auto *entry = &caller->getEntryBlock();
                llvm::IRBuilder<> entry_builder{
                    &*entry->getFirstInsertionPt()};
                slot = entry_builder.CreateAlloca(
                    plan.return_type, alloca_address_space, nullptr,
                    "callable.return.storage");
                slot->setAlignment(
                    data_layout.getPrefTypeAlign(plan.return_type));
                stats.shared_result_slot_count++;
            }

            llvm::IRBuilder<> builder{call};
            llvm::SmallVector<llvm::Value *, 16> arguments;
            arguments.reserve(call->arg_size() + 1u);
            arguments.emplace_back(slot);
            for (auto &operand : call->args()) {
                arguments.emplace_back(operand.get());
            }
            llvm::SmallVector<llvm::OperandBundleDef, 2> bundles;
            call->getOperandBundlesAsDefs(bundles);
            auto *new_call = builder.CreateCall(
                replacement->getFunctionType(), replacement,
                arguments, bundles);
            new_call->setCallingConv(call->getCallingConv());
            new_call->setAttributes(prepend_result_pointer_attributes(
                module.getContext(), call->getAttributes(),
                call->arg_size()));
            new_call->setMemoryEffects(
                call->getMemoryEffects() |
                llvm::MemoryEffects::argMemOnly(
                    llvm::ModRefInfo::Mod));
            new_call->setDebugLoc(call->getDebugLoc());
            new_call->copyMetadata(*call);
            auto *load = builder.CreateLoad(
                plan.return_type, slot);
            load->setAlignment(
                data_layout.getPrefTypeAlign(plan.return_type));
            load->setDebugLoc(call->getDebugLoc());
            load->takeName(call);
            call->replaceAllUsesWith(load);
            call->eraseFromParent();
            stats.rewritten_call_count++;
        }
    }
    for (auto &plan : plans) {
        LUISA_ASSERT(plan.original->use_empty());
        plan.original->eraseFromParent();
        stats.rewritten_function_count++;
    }
    return stats;
}

}// namespace luisa::compute::hip
