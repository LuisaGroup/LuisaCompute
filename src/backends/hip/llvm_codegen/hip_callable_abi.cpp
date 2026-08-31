#include "hip_callable_abi.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/IPO/MergeFunctions.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>

#include <luisa/core/logging.h>

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <vector>

namespace luisa::compute::hip {

void finalize_hip_function_attributes(
    llvm::Function &function,
    llvm::StringRef target_cpu,
    llvm::StringRef target_features,
    llvm::StringRef max_vgpr_count) noexcept {
    const auto is_generated_callable =
        function.hasFnAttribute(llvm_generated_callable_attribute);
    if (is_generated_callable) {
        function.removeFnAttr(llvm::Attribute::AlwaysInline);
        function.removeFnAttr(llvm::Attribute::NoInline);
        function.removeFnAttr(llvm::Attribute::InlineHint);
    }
    function.removeFnAttr(llvm_generated_callable_attribute);
    if (function.isDeclaration()) { return; }

    function.removeFnAttr("target-cpu");
    function.removeFnAttr("target-features");
    function.removeFnAttr("amdgpu-num-vgpr");
    if (!target_cpu.empty()) {
        function.addFnAttr("target-cpu", target_cpu);
    }
    if (!target_features.empty()) {
        function.addFnAttr("target-features", target_features);
    }
    if (!max_vgpr_count.empty()) {
        function.addFnAttr("amdgpu-num-vgpr", max_vgpr_count);
    }
}

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

struct LargeArgumentPlan {
    llvm::Function *original;
    llvm::Function *replacement{};
    llvm::StructType *record_type;
    size_t direct_argument_count;
    size_t packed_argument_count;
    size_t record_bytes;
    size_t record_alignment;
};

struct CallerArgumentStorage {
    size_t bytes{};
    size_t alignment{1u};
    llvm::AllocaInst *slot{};
};

struct ConstantArgumentCallGroup {
    llvm::SmallVector<llvm::ConstantInt *, 2> values;
    llvm::SmallVector<llvm::CallInst *, 8> calls;
};

[[nodiscard]] llvm::AttributeList remove_parameter_attributes(
    llvm::LLVMContext &context, llvm::AttributeList attributes,
    llvm::ArrayRef<unsigned> removed_parameters,
    unsigned old_parameter_count) noexcept {
    llvm::SmallVector<llvm::AttributeSet, 16> parameter_attributes;
    parameter_attributes.reserve(
        old_parameter_count - removed_parameters.size());
    for (auto parameter_index = 0u;
         parameter_index < old_parameter_count; parameter_index++) {
        if (!std::binary_search(
                removed_parameters.begin(), removed_parameters.end(),
                parameter_index)) {
            parameter_attributes.emplace_back(
                attributes.getParamAttrs(parameter_index));
        }
    }
    return llvm::AttributeList::get(
        context, attributes.getFnAttrs(), attributes.getRetAttrs(),
        parameter_attributes);
}

void simplify_constant_argument_clone(
    llvm::Function &function) noexcept {
    // CloneFunction substitutes the formal in SSA but intentionally does not
    // run a pass pipeline. Iterate the local fixed point needed by this
    // transformation: fold pure instructions, fold constant terminators, then
    // delete unreachable alternatives. No interprocedural or alias fact is
    // introduced here.
    auto changed = false;
    do {
        changed = false;
        for (auto &block : function) {
            changed |= llvm::SimplifyInstructionsInBlock(&block);
        }
        for (auto &block : function) {
            changed |= llvm::ConstantFoldTerminator(
                &block, true);
        }
        changed |= llvm::removeUnreachableBlocks(function);
    } while (changed);
}

// Luisa's retained generated callables use FastCC so LLVM can optimize their
// internal ABI. Some focused/runtime-generated modules retain the default C
// convention. Both conventions use CC_AMDGPU_Func/RetCC_AMDGPU_Func, and the
// transforms preserve the convention identically on every replacement and
// direct call. Other conventions may carry target- or language-specific ABI
// rules that are not modeled here.
[[nodiscard]] bool supported_generated_callable_calling_convention(
    llvm::CallingConv::ID convention) noexcept {
    return convention == llvm::CallingConv::C ||
           convention == llvm::CallingConv::Fast;
}

// CC_AMDGPU_Func and RetCC_AMDGPU_Func decompose values into the same 32-bit
// VGPR location types. This computes a conservative upper bound on the number
// of locations occupied after aggregate decomposition, without charging
// layout padding. Narrow scalar leaves consume one location because the
// convention may promote them; 16-bit vector pairs are legal packed values,
// while wider vector leaves occupy one location per 32-bit chunk. Returning
// nullopt rejects scalable or unsized types rather than guessing.
[[nodiscard]] std::optional<size_t> amdgpu_value_vgpr_count(
    llvm::Type *type, const llvm::DataLayout &data_layout) noexcept {
    if (type->isVoidTy()) { return 0u; }
    if (auto *structure = llvm::dyn_cast<llvm::StructType>(type)) {
        if (structure->isOpaque()) { return std::nullopt; }
        auto count = size_t{};
        for (auto *element : structure->elements()) {
            auto element_count =
                amdgpu_value_vgpr_count(element, data_layout);
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
            amdgpu_value_vgpr_count(array->getElementType(), data_layout);
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
            // The fixed AMDGPU function ABI has no packed sub-16-bit vector
            // location, so
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

[[nodiscard]] llvm::AttributeList pack_suffix_argument_attributes(
    llvm::LLVMContext &context, llvm::AttributeList attributes,
    size_t direct_argument_count) noexcept {
    llvm::SmallVector<llvm::AttributeSet, 16> parameter_attributes;
    parameter_attributes.reserve(direct_argument_count + 1u);
    for (auto parameter_index = size_t{0u};
         parameter_index < direct_argument_count; parameter_index++) {
        parameter_attributes.emplace_back(
            attributes.getParamAttrs(
                static_cast<unsigned>(parameter_index)));
    }
    // The record contains copied values rather than preserving the ABI role of
    // any one packed formal. In particular, attributes such as noundef,
    // nonnull, returned, or dereferenceable belong to the loaded value, not to
    // the record pointer. IPO has already consumed those facts.
    parameter_attributes.emplace_back();
    return llvm::AttributeList::get(
        context, attributes.getFnAttrs(), attributes.getRetAttrs(),
        parameter_attributes);
}

[[nodiscard]] bool has_packed_argument_abi_attribute(
    llvm::AttributeList attributes, size_t first_packed,
    size_t parameter_count) noexcept {
    // These attributes change how an argument is physically passed or tie it
    // to another ABI entity. They cannot be represented by copying the SSA
    // value into an ordinary record. Optimization/validity facts (noundef,
    // nonnull, alignment, dereferenceability, noalias, readonly, ...) may be
    // dropped after IPO without changing any defined execution.
    constexpr llvm::Attribute::AttrKind abi_attributes[] = {
        llvm::Attribute::ByVal,
        llvm::Attribute::ByRef,
        llvm::Attribute::InAlloca,
        llvm::Attribute::Preallocated,
        llvm::Attribute::StructRet,
        llvm::Attribute::Nest,
        llvm::Attribute::SwiftSelf,
        llvm::Attribute::SwiftAsync};
    for (auto index = first_packed; index < parameter_count; index++) {
        for (auto attribute : abi_attributes) {
            if (attributes.hasParamAttr(
                    static_cast<unsigned>(index), attribute)) {
                return true;
            }
        }
    }
    return false;
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

ConstantArgumentSpecializationStats
specialize_marked_constant_integer_arguments(
    llvm::Module &module,
    llvm::StringRef argument_attribute) noexcept {
    llvm::SmallVector<llvm::Function *, 8> marked_functions;
    for (auto &function : module) {
        for (auto &argument : function.args()) {
            if (argument.hasAttribute(argument_attribute)) {
                marked_functions.emplace_back(&function);
                break;
            }
        }
    }

    auto stats = ConstantArgumentSpecializationStats{};
    llvm::SmallVector<llvm::Function *, 16> specialized_functions;
    for (auto *function : marked_functions) {
        llvm::SmallVector<unsigned, 2> marked_arguments;
        for (auto &argument : function->args()) {
            if (argument.hasAttribute(argument_attribute)) {
                marked_arguments.emplace_back(argument.getArgNo());
            }
        }
        // The marker is a codegen-internal analysis request, not a target ABI
        // attribute. Strip every occurrence before any fail-closed exit.
        for (auto argument_index : marked_arguments) {
            function->removeParamAttr(
                argument_index, argument_attribute);
        }
        if (marked_arguments.empty()) { continue; }
        const auto marked_arguments_are_supported = std::all_of(
            marked_arguments.begin(), marked_arguments.end(),
            [function](auto argument_index) noexcept {
                auto *argument = function->getArg(argument_index);
                return argument->getType()->isIntegerTy() &&
                       argument->getType()->getIntegerBitWidth() <= 64u;
            });
        if (function->isDeclaration() || function->isVarArg() ||
            !function->hasLocalLinkage() || function->hasAddressTaken() ||
            function->hasMetadata() || function->hasComdat() ||
            function->hasGC() || function->hasPersonalityFn() ||
            function->hasPrefixData() || function->hasPrologueData() ||
            function->hasFnAttribute(llvm::Attribute::AllocSize) ||
            !marked_arguments_are_supported) {
            continue;
        }

        llvm::SmallVector<llvm::CallInst *, 16> calls;
        auto supported_uses = true;
        for (auto *user : function->users()) {
            auto *call = llvm::dyn_cast<llvm::CallInst>(user);
            if (call == nullptr ||
                call->getCalledOperand() != function ||
                call->getFunction() == function ||
                call->getCallingConv() != function->getCallingConv() ||
                call->isMustTailCall() ||
                call->hasMetadataOtherThanDebugLoc() ||
                call->hasFnAttr(llvm::Attribute::AllocSize)) {
                supported_uses = false;
                break;
            }
            for (auto argument_index : marked_arguments) {
                if (!llvm::isa<llvm::ConstantInt>(
                        call->getArgOperand(argument_index))) {
                    supported_uses = false;
                    break;
                }
            }
            if (!supported_uses) { break; }
            calls.emplace_back(call);
        }
        if (!supported_uses || calls.empty()) { continue; }

        std::vector<ConstantArgumentCallGroup> groups;
        for (auto *call : calls) {
            llvm::SmallVector<llvm::ConstantInt *, 2> values;
            values.reserve(marked_arguments.size());
            for (auto argument_index : marked_arguments) {
                values.emplace_back(llvm::cast<llvm::ConstantInt>(
                    call->getArgOperand(argument_index)));
            }
            auto group = std::find_if(
                groups.begin(), groups.end(),
                [&values](const auto &candidate) noexcept {
                    return std::equal(
                        candidate.values.begin(),
                        candidate.values.end(), values.begin(),
                        values.end(),
                        [](auto *lhs, auto *rhs) noexcept {
                            return lhs->getValue() == rhs->getValue();
                        });
                });
            if (group == groups.end()) {
                groups.emplace_back(ConstantArgumentCallGroup{
                    .values = std::move(values)});
                group = std::prev(groups.end());
            }
            group->calls.emplace_back(call);
        }
        std::sort(
            groups.begin(), groups.end(),
            [](const auto &lhs, const auto &rhs) noexcept {
                return std::lexicographical_compare(
                    lhs.values.begin(), lhs.values.end(),
                    rhs.values.begin(), rhs.values.end(),
                    [](auto *lhs_value, auto *rhs_value) noexcept {
                        return lhs_value->getValue().ult(
                            rhs_value->getValue());
                    });
            });

        auto original_name = function->getName().str();
        for (auto &group : groups) {
            llvm::ValueToValueMapTy value_map;
            for (auto i = 0u; i < marked_arguments.size(); i++) {
                value_map[function->getArg(marked_arguments[i])] =
                    group.values[i];
            }
            auto *clone = llvm::CloneFunction(function, value_map);
            llvm::SmallString<64> clone_name{original_name};
            clone_name.append(".constant");
            for (auto *value : group.values) {
                clone_name.push_back('.');
                value->getValue().toString(
                    clone_name, 10u, false);
            }
            clone->setName(clone_name);
            simplify_constant_argument_clone(*clone);
            specialized_functions.emplace_back(clone);

            for (auto *call : group.calls) {
                llvm::IRBuilder<> builder{call};
                llvm::SmallVector<llvm::Value *, 16> arguments;
                arguments.reserve(
                    call->arg_size() - marked_arguments.size());
                for (auto actual_index = 0u;
                     actual_index < call->arg_size(); actual_index++) {
                    if (!std::binary_search(
                            marked_arguments.begin(),
                            marked_arguments.end(), actual_index)) {
                        arguments.emplace_back(
                            call->getArgOperand(actual_index));
                    }
                }
                llvm::SmallVector<llvm::OperandBundleDef, 2> bundles;
                call->getOperandBundlesAsDefs(bundles);
                auto *new_call = builder.CreateCall(
                    clone->getFunctionType(), clone,
                    arguments, bundles);
                new_call->setCallingConv(call->getCallingConv());
                new_call->setTailCallKind(call->getTailCallKind());
                new_call->setAttributes(remove_parameter_attributes(
                    module.getContext(), call->getAttributes(),
                    marked_arguments, call->arg_size()));
                new_call->setDebugLoc(call->getDebugLoc());
                new_call->copyMetadata(*call);
                new_call->setFastMathFlags(
                    call->getFastMathFlags());
                new_call->takeName(call);
                call->replaceAllUsesWith(new_call);
                call->eraseFromParent();
                stats.rewritten_call_count++;
            }
            stats.cloned_function_count++;
        }
        LUISA_ASSERT(
            function->use_empty(),
            "Constant-argument specialization left an original use.");
        function->eraseFromParent();
        stats.rewritten_function_count++;
    }
    if (specialized_functions.size() > 1u) {
        // Specialization can expose equality that the main IPO pipeline could
        // not see through the formerly dynamic parameter. Delegate semantic
        // equivalence (including attributes and constants) to LLVM's own
        // function comparator instead of inventing a backend hash.
        auto merged = llvm::MergeFunctionsPass::runOnFunctions(
            specialized_functions);
        stats.merged_clone_count = merged.size();
    }
    return stats;
}

AggregateArgumentSpecializationStats
specialize_generated_callable_aggregate_arguments(
    llvm::Module &module,
    llvm::StringRef callable_attribute) noexcept {
    llvm::SmallVector<llvm::Function *, 32> functions;
    for (auto &function : module) {
        if (!function.isDeclaration() &&
            function.hasFnAttribute(callable_attribute)) {
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
            !supported_generated_callable_calling_convention(
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
            amdgpu_value_vgpr_count(return_type, data_layout);
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

LargeArgumentDemotionStats demote_generated_callable_large_arguments(
    llvm::Module &module,
    llvm::StringRef callable_attribute) noexcept {
    const auto &data_layout = module.getDataLayout();
    const auto alloca_address_space = data_layout.getAllocaAddrSpace();
    auto *record_pointer_type = llvm::PointerType::get(
        module.getContext(), alloca_address_space);
    const auto record_pointer_locations =
        amdgpu_value_vgpr_count(record_pointer_type, data_layout);
    if (!record_pointer_locations ||
        *record_pointer_locations >
            amdgpu_callable_argument_vgpr_limit) {
        return {};
    }

    auto plans = std::vector<LargeArgumentPlan>{};
    for (auto &function : module) {
        if (function.isDeclaration() ||
            !function.hasFnAttribute(callable_attribute) ||
            !function.hasLocalLinkage() || function.isVarArg() ||
            !supported_generated_callable_calling_convention(
                function.getCallingConv()) ||
            function.hasAddressTaken() || function.hasMetadata() ||
            function.hasComdat() || function.hasGC() ||
            function.hasPersonalityFn() || function.hasPrefixData() ||
            function.hasPrologueData() ||
            function.hasFnAttribute(llvm::Attribute::AllocSize)) {
            continue;
        }

        llvm::SmallVector<size_t, 16> argument_locations;
        argument_locations.reserve(function.arg_size());
        auto total_locations = size_t{0u};
        auto modeled = true;
        for (auto &argument : function.args()) {
            auto locations = amdgpu_value_vgpr_count(
                argument.getType(), data_layout);
            if (!locations ||
                *locations >
                    std::numeric_limits<size_t>::max() -
                        total_locations) {
                modeled = false;
                break;
            }
            argument_locations.emplace_back(*locations);
            total_locations += *locations;
        }
        if (!modeled ||
            total_locations <=
                amdgpu_callable_argument_vgpr_limit) {
            continue;
        }

        // Keep the maximal ordered prefix that still leaves one ABI location
        // for the suffix-record pointer. Preserving order makes the model
        // independent of target register-assignment heuristics and prevents a
        // future parameter from silently changing the ABI of an earlier one.
        auto direct_argument_count = size_t{0u};
        auto direct_locations = size_t{0u};
        while (direct_argument_count < argument_locations.size()) {
            const auto locations =
                argument_locations[direct_argument_count];
            if (locations >
                amdgpu_callable_argument_vgpr_limit -
                    *record_pointer_locations - direct_locations) {
                break;
            }
            direct_locations += locations;
            direct_argument_count++;
        }
        LUISA_ASSERT(
            direct_argument_count < function.arg_size(),
            "Overflowing callable ABI must have a non-empty packed suffix.");

        if (has_packed_argument_abi_attribute(
                function.getAttributes(), direct_argument_count,
                function.arg_size())) {
            continue;
        }
        auto supported_uses = true;
        for (auto *user : function.users()) {
            auto *call = llvm::dyn_cast<llvm::CallInst>(user);
            if (call == nullptr ||
                call->getCalledOperand() != &function ||
                call->getCallingConv() != function.getCallingConv() ||
                call->isMustTailCall() || call->hasOperandBundles() ||
                call->hasMetadataOtherThanDebugLoc() ||
                call->hasFnAttr(llvm::Attribute::AllocSize) ||
                has_packed_argument_abi_attribute(
                    call->getAttributes(), direct_argument_count,
                    call->arg_size())) {
                supported_uses = false;
                break;
            }
        }
        if (!supported_uses || function.use_empty()) { continue; }

        llvm::SmallVector<llvm::Type *, 16> record_elements;
        record_elements.reserve(
            function.arg_size() - direct_argument_count);
        for (auto index = direct_argument_count;
             index < function.arg_size(); index++) {
            record_elements.emplace_back(
                function.getFunctionType()->getParamType(
                    static_cast<unsigned>(index)));
        }
        auto *record_type = llvm::StructType::get(
            module.getContext(), record_elements, false);
        plans.emplace_back(LargeArgumentPlan{
            .original = &function,
            .record_type = record_type,
            .direct_argument_count = direct_argument_count,
            .packed_argument_count = record_elements.size(),
            .record_bytes = static_cast<size_t>(
                data_layout.getTypeAllocSize(record_type)
                    .getFixedValue()),
            .record_alignment = static_cast<size_t>(
                data_layout.getPrefTypeAlign(record_type).value())});
    }

    auto stats = LargeArgumentDemotionStats{};
    if (plans.empty()) { return stats; }

    // Create every replacement before rewriting calls. A transformed caller
    // may itself be another retained callable; moving its CFG never invalidates
    // a direct call selected by a different plan.
    for (auto &plan : plans) {
        auto *function = plan.original;
        llvm::SmallVector<llvm::Type *, 16> parameter_types;
        parameter_types.reserve(plan.direct_argument_count + 1u);
        for (auto index = size_t{0u};
             index < plan.direct_argument_count; index++) {
            parameter_types.emplace_back(
                function->getFunctionType()->getParamType(
                    static_cast<unsigned>(index)));
        }
        parameter_types.emplace_back(record_pointer_type);
        auto *replacement_type = llvm::FunctionType::get(
            function->getReturnType(), parameter_types, false);
        auto old_name = function->getName().str();
        function->setName(old_name + ".large.arguments.abi.old");
        auto *replacement = llvm::Function::Create(
            replacement_type, function->getLinkage(), old_name, &module);
        plan.replacement = replacement;
        replacement->copyAttributesFrom(function);
        replacement->setAttributes(pack_suffix_argument_attributes(
            module.getContext(), function->getAttributes(),
            plan.direct_argument_count));
        replacement->setMemoryEffects(
            function->getMemoryEffects() |
            llvm::MemoryEffects::argMemOnly(
                llvm::ModRefInfo::Ref));
        replacement->setCallingConv(function->getCallingConv());
        replacement->splice(replacement->end(), function);

        auto replacement_argument = replacement->arg_begin();
        for (auto index = size_t{0u};
             index < plan.direct_argument_count; index++) {
            auto *old_argument = function->getArg(
                static_cast<unsigned>(index));
            old_argument->replaceAllUsesWith(replacement_argument);
            replacement_argument->takeName(old_argument);
            ++replacement_argument;
        }
        auto *record_pointer = &*replacement_argument;
        record_pointer->setName("argument.suffix.storage");
        auto *entry = &replacement->getEntryBlock();
        llvm::IRBuilder<> builder{&*entry->getFirstInsertionPt()};
        for (auto index = plan.direct_argument_count;
             index < function->arg_size(); index++) {
            auto field_index = static_cast<unsigned>(
                index - plan.direct_argument_count);
            auto *field = builder.CreateStructGEP(
                plan.record_type, record_pointer, field_index);
            auto *old_argument = function->getArg(
                static_cast<unsigned>(index));
            auto *load = builder.CreateLoad(
                old_argument->getType(), field,
                old_argument->getName() + ".packed");
            load->setAlignment(data_layout.getABITypeAlign(
                old_argument->getType()));
            old_argument->replaceAllUsesWith(load);
        }
    }

    // Every record in one caller can use the same max-sized, max-aligned byte
    // slot: all fields are written immediately before a synchronous call and
    // no operand evaluation occurs between the first store and that call. A
    // nested callee owns storage in its own machine frame, and recursive
    // invocations likewise receive distinct frames. This is interval coloring
    // with one color because the per-caller record live ranges are pairwise
    // disjoint by construction, independent of CFG path correlation.
    llvm::DenseMap<llvm::Function *, CallerArgumentStorage>
        caller_argument_storage;
    for (auto &plan : plans) {
        for (auto *user : plan.original->users()) {
            auto *call = llvm::cast<llvm::CallInst>(user);
            auto &storage = caller_argument_storage[call->getFunction()];
            storage.bytes = std::max(storage.bytes, plan.record_bytes);
            storage.alignment = std::max(
                storage.alignment, plan.record_alignment);
        }
    }
    for (auto &[caller, storage] : caller_argument_storage) {
        auto *entry = &caller->getEntryBlock();
        llvm::IRBuilder<> entry_builder{
            &*entry->getFirstInsertionPt()};
        auto *storage_type = llvm::ArrayType::get(
            llvm::Type::getInt8Ty(module.getContext()), storage.bytes);
        storage.slot = entry_builder.CreateAlloca(
            storage_type, alloca_address_space, nullptr,
            "callable.argument.storage");
        storage.slot->setAlignment(llvm::Align{storage.alignment});
        stats.shared_argument_slot_count++;
    }

    for (auto &plan : plans) {
        auto *function = plan.original;
        auto *replacement = plan.replacement;
        llvm::SmallVector<llvm::CallInst *, 16> calls;
        for (auto *user : function->users()) {
            calls.emplace_back(llvm::cast<llvm::CallInst>(user));
        }
        for (auto *call : calls) {
            auto *caller = call->getFunction();
            auto *slot = caller_argument_storage.find(caller)->second.slot;

            llvm::IRBuilder<> builder{call};
            for (auto index = plan.direct_argument_count;
                 index < call->arg_size(); index++) {
                auto field_index = static_cast<unsigned>(
                    index - plan.direct_argument_count);
                auto *field = builder.CreateStructGEP(
                    plan.record_type, slot, field_index);
                auto *store = builder.CreateStore(
                    call->getArgOperand(
                        static_cast<unsigned>(index)),
                    field);
                store->setAlignment(data_layout.getABITypeAlign(
                    call->getArgOperand(
                            static_cast<unsigned>(index))
                        ->getType()));
            }
            llvm::SmallVector<llvm::Value *, 16> arguments;
            arguments.reserve(plan.direct_argument_count + 1u);
            for (auto index = size_t{0u};
                 index < plan.direct_argument_count; index++) {
                arguments.emplace_back(call->getArgOperand(
                    static_cast<unsigned>(index)));
            }
            arguments.emplace_back(slot);
            auto *new_call = builder.CreateCall(
                replacement->getFunctionType(), replacement, arguments);
            new_call->setCallingConv(call->getCallingConv());
            if (call->getTailCallKind() ==
                llvm::CallInst::TCK_NoTail) {
                new_call->setTailCallKind(
                    llvm::CallInst::TCK_NoTail);
            }
            new_call->setAttributes(pack_suffix_argument_attributes(
                module.getContext(), call->getAttributes(),
                plan.direct_argument_count));
            new_call->setMemoryEffects(
                call->getMemoryEffects() |
                llvm::MemoryEffects::argMemOnly(
                    llvm::ModRefInfo::Ref));
            new_call->setDebugLoc(call->getDebugLoc());
            new_call->copyMetadata(*call);
            new_call->setFastMathFlags(call->getFastMathFlags());
            new_call->takeName(call);
            call->replaceAllUsesWith(new_call);
            call->eraseFromParent();
            stats.rewritten_call_count++;
        }
    }
    for (auto &plan : plans) {
        LUISA_ASSERT(plan.original->use_empty());
        plan.original->eraseFromParent();
        stats.rewritten_function_count++;
        stats.packed_argument_count += plan.packed_argument_count;
        stats.argument_record_bytes += plan.record_bytes;
    }
    return stats;
}

}// namespace luisa::compute::hip
