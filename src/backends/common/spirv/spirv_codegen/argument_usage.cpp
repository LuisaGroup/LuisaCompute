#include "argument_usage.h"
#include "bindless_usage.h"
#include "structural_closure.h"

#include <algorithm>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/type_traits.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/module.h>

namespace lc::spirv {

namespace xir = luisa::compute::xir;
using luisa::compute::Usage;

namespace {

[[nodiscard]] const SpirvFunctionArgumentAnalysis &
function_argument_analysis_of(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const xir::Function *function,
    const xir::Argument *argument) noexcept {
    LUISA_ASSERT(function != nullptr && argument != nullptr,
                 "SPIR-V argument analysis lookup requires a non-null "
                 "function and argument.");
    auto fit = analysis.find(function);
    LUISA_ASSERT(
        fit != analysis.end(),
        "SPIR-V argument analysis has no entry for the queried function.");
    auto index = size_t{0u};
    for (auto *candidate : function->arguments()) {
        if (candidate == argument) {
            LUISA_ASSERT(
                index < fit->second.size(),
                "SPIR-V argument analysis entry has {} slots for argument "
                "index {}.",
                fit->second.size(), index);
            return fit->second[index];
        }
        index++;
    }
    LUISA_ERROR_WITH_LOCATION(
        "SPIR-V argument analysis lookup received an argument not owned by "
        "the queried function.");
}

}// namespace

bool spirv_resource_query_requires_accel_traversal_descriptor(
    xir::ResourceQueryOp op) noexcept {
    switch (op) {
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
            return true;
        default: return false;
    }
}

bool spirv_resource_query_requires_accel_instance_buffer(
    xir::ResourceQueryOp op) noexcept {
    switch (op) {
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT:
            return true;
        default: return false;
    }
}

SpirvFunctionArgumentAnalysisMap
analyze_spirv_function_argument_usage(
    const xir::Module *module) noexcept {
    SpirvFunctionArgumentAnalysisMap analysis;
    if (module == nullptr) { return analysis; }
    luisa::unordered_map<
        const xir::Function *,
        luisa::unordered_map<const xir::Argument *, size_t>>
        indices;
    for (auto *function : module->function_list()) {
        if (!function->is_definition()) { continue; }
        auto index = size_t{0u};
        luisa::unordered_map<const xir::Argument *, size_t> function_indices;
        for (auto *argument : function->arguments()) {
            function_indices.emplace(argument, index++);
        }
        indices.emplace(function, std::move(function_indices));
        analysis.emplace(
            function,
            luisa::vector<SpirvFunctionArgumentAnalysis>(index));
    }
    auto add_usage = [&](const xir::Function *function,
                         const xir::Value *value,
                         Usage incoming) noexcept {
        if (function == nullptr || value == nullptr ||
            !value->isa<xir::Argument>()) {
            return false;
        }
        auto *argument = static_cast<const xir::Argument *>(value);
        if (argument->parent_function() != function) { return false; }
        auto fit = indices.find(function);
        if (fit == indices.end()) { return false; }
        auto ait = fit->second.find(argument);
        if (ait == fit->second.end()) { return false; }
        auto &slot = analysis.at(function)[ait->second].usage;
        auto merged = static_cast<Usage>(
            luisa::to_underlying(slot) |
            luisa::to_underlying(incoming));
        if (merged == slot) { return false; }
        slot = merged;
        return true;
    };
    auto require_accel_instance_buffer = [&](
                                             const xir::Function *function,
                                             const xir::Value *value) noexcept {
        if (function == nullptr || value == nullptr ||
            !value->isa<xir::Argument>()) {
            return false;
        }
        auto *argument = static_cast<const xir::Argument *>(value);
        if (argument->parent_function() != function ||
            argument->type() == nullptr || !argument->type()->is_accel()) {
            return false;
        }
        auto fit = indices.find(function);
        if (fit == indices.end()) { return false; }
        auto ait = fit->second.find(argument);
        if (ait == fit->second.end()) { return false; }
        auto &slot = analysis.at(function)[ait->second]
                         .requires_accel_instance_buffer;
        if (slot) { return false; }
        slot = true;
        return true;
    };
    auto require_accel_traversal_descriptor = [&](
                                                  const xir::Function *function,
                                                  const xir::Value *value) noexcept {
        if (function == nullptr || value == nullptr ||
            !value->isa<xir::Argument>()) {
            return false;
        }
        auto *argument = static_cast<const xir::Argument *>(value);
        if (argument->parent_function() != function ||
            argument->type() == nullptr || !argument->type()->is_accel()) {
            return false;
        }
        auto fit = indices.find(function);
        if (fit == indices.end()) { return false; }
        auto ait = fit->second.find(argument);
        if (ait == fit->second.end()) { return false; }
        auto &slot = analysis.at(function)[ait->second]
                         .requires_accel_traversal_descriptor;
        if (slot) { return false; }
        slot = true;
        return true;
    };
    auto require_bindless_buffer_metadata = [&](
                                                const xir::Function *function,
                                                const xir::Value *value) noexcept {
        if (function == nullptr || value == nullptr ||
            !value->isa<xir::Argument>()) {
            return false;
        }
        auto *argument = static_cast<const xir::Argument *>(value);
        if (argument->parent_function() != function ||
            argument->type() == nullptr ||
            !argument->type()->is_bindless_array()) {
            return false;
        }
        auto fit = indices.find(function);
        if (fit == indices.end()) { return false; }
        auto ait = fit->second.find(argument);
        if (ait == fit->second.end()) { return false; }
        auto &slot = analysis.at(function)[ait->second]
                         .requires_bindless_buffer_metadata;
        if (slot) { return false; }
        slot = true;
        return true;
    };
    auto require_buffer_device_address = [&](
                                             const xir::Function *function,
                                             const xir::Value *value) noexcept {
        if (function == nullptr || value == nullptr ||
            !value->isa<xir::Argument>()) {
            return false;
        }
        auto *argument = static_cast<const xir::Argument *>(value);
        if (argument->parent_function() != function ||
            argument->type() == nullptr ||
            (!argument->type()->is_buffer() &&
             !argument->type()->is_bindless_array())) {
            return false;
        }
        auto fit = indices.find(function);
        if (fit == indices.end()) { return false; }
        auto ait = fit->second.find(argument);
        if (ait == fit->second.end()) { return false; }
        auto &slot = analysis.at(function)[ait->second]
                         .requires_buffer_device_address;
        if (slot) { return false; }
        slot = true;
        return true;
    };
    auto require_buffer_coherence = [&](
                                        const xir::Function *function,
                                        const xir::Value *value) noexcept {
        if (function == nullptr || value == nullptr ||
            !value->isa<xir::Argument>()) {
            return false;
        }
        auto *argument = static_cast<const xir::Argument *>(value);
        if (argument->parent_function() != function ||
            argument->type() == nullptr ||
            !argument->type()->is_buffer()) {
            return false;
        }
        auto fit = indices.find(function);
        if (fit == indices.end()) { return false; }
        auto ait = fit->second.find(argument);
        if (ait == fit->second.end()) { return false; }
        auto &slot = analysis.at(function)[ait->second]
                         .requires_buffer_coherence;
        if (slot) { return false; }
        slot = true;
        return true;
    };
    auto traverse_definition = [](
                                   const xir::FunctionDefinition *definition,
                                   auto &&visit) noexcept {
        auto closure = plan_spirv_codegen_structural_closure(definition);
        if (!closure.succeeded()) { return; }
        for (auto *block : closure.blocks) {
            block->traverse_instructions(visit);
        }
    };
    for (auto *function : module->function_list()) {
        auto *definition = function->definition();
        if (definition == nullptr) { continue; }
        traverse_definition(
            definition,
            [&](const xir::Instruction *instruction) noexcept {
                switch (instruction->derived_instruction_tag()) {
                    case xir::DerivedInstructionTag::RESOURCE_QUERY: {
                        auto *query = static_cast<
                            const xir::ResourceQueryInst *>(instruction);
                        if (instruction->operand_count() != 0u) {
                            static_cast<void>(add_usage(
                                function, instruction->operand(0u),
                                Usage::READ));
                            if (spirv_resource_query_requires_accel_instance_buffer(
                                    query->op())) {
                                static_cast<void>(
                                    require_accel_instance_buffer(
                                        function,
                                        instruction->operand(0u)));
                            }
                            if (spirv_resource_query_requires_accel_traversal_descriptor(
                                    query->op())) {
                                static_cast<void>(
                                    require_accel_traversal_descriptor(
                                        function,
                                        instruction->operand(0u)));
                            }
                            if (spirv_bindless_resource_usage(query->op())
                                    .buffer_metadata) {
                                static_cast<void>(
                                    require_bindless_buffer_metadata(
                                        function,
                                        instruction->operand(0u)));
                            }
                            if (query->op() ==
                                    xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS ||
                                query->op() ==
                                    xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS) {
                                static_cast<void>(
                                    require_buffer_device_address(
                                        function,
                                        instruction->operand(0u)));
                            }
                        }
                        break;
                    }
                    case xir::DerivedInstructionTag::RESOURCE_READ: {
                        auto *read = static_cast<
                            const xir::ResourceReadInst *>(instruction);
                        if (instruction->operand_count() != 0u) {
                            static_cast<void>(add_usage(
                                function, instruction->operand(0u),
                                Usage::READ));
                            if (spirv_bindless_resource_usage(read->op())
                                    .buffer_metadata) {
                                static_cast<void>(
                                    require_bindless_buffer_metadata(
                                        function,
                                        instruction->operand(0u)));
                            }
                            if (read->op() ==
                                    xir::ResourceReadOp::BUFFER_VOLATILE_READ ||
                                read->op() ==
                                    xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ) {
                                static_cast<void>(
                                    require_buffer_coherence(
                                        function,
                                        instruction->operand(0u)));
                            }
                        }
                        break;
                    }
                    case xir::DerivedInstructionTag::RESOURCE_WRITE: {
                        auto *write = static_cast<
                            const xir::ResourceWriteInst *>(instruction);
                        if (instruction->operand_count() != 0u) {
                            static_cast<void>(add_usage(
                                function, instruction->operand(0u),
                                Usage::WRITE));
                            static_cast<void>(
                                require_accel_instance_buffer(
                                    function,
                                    instruction->operand(0u)));
                            if (spirv_bindless_resource_usage(write->op())
                                    .buffer_metadata) {
                                static_cast<void>(
                                    require_bindless_buffer_metadata(
                                        function,
                                        instruction->operand(0u)));
                            }
                            if (write->op() ==
                                    xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE ||
                                write->op() ==
                                    xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE) {
                                static_cast<void>(
                                    require_buffer_coherence(
                                        function,
                                        instruction->operand(0u)));
                            }
                        }
                        break;
                    }
                    case xir::DerivedInstructionTag::ATOMIC: {
                        if (instruction->operand_count() != 0u) {
                            static_cast<void>(add_usage(
                                function, instruction->operand(0u),
                                Usage::READ_WRITE));
                        }
                        break;
                    }
                    default: break;
                }
            });
    }
    for (auto changed = true; changed;) {
        changed = false;
        for (auto *function : module->function_list()) {
            auto *definition = function->definition();
            if (definition == nullptr) { continue; }
            traverse_definition(
                definition,
                [&](const xir::Instruction *instruction) noexcept {
                    if (!instruction->isa<xir::CallInst>()) { return; }
                    auto *call =
                        static_cast<const xir::CallInst *>(instruction);
                    if (call->operand_count() == 0u) { return; }
                    auto *callee_value = call->operand(
                        xir::CallInst::operand_index_callee);
                    auto *callee =
                        callee_value != nullptr &&
                                callee_value->isa<xir::Function>() ?
                            static_cast<const xir::Function *>(callee_value) :
                            nullptr;
                    auto analysis_iter = analysis.find(callee);
                    if (analysis_iter == analysis.end()) { return; }
                    auto count = std::min(
                        call->operand_count() -
                            xir::CallInst::operand_index_argument_offset,
                        analysis_iter->second.size());
                    for (auto i = 0u; i < count; ++i) {
                        auto incoming = analysis_iter->second[i];
                        auto *actual = call->operand(
                            xir::CallInst::operand_index_argument_offset + i);
                        if (incoming.usage != Usage::NONE) {
                            changed |= add_usage(
                                function, actual,
                                incoming.usage);
                        }
                        if (incoming.requires_accel_instance_buffer) {
                            changed |= require_accel_instance_buffer(
                                function, actual);
                        }
                        if (incoming.requires_accel_traversal_descriptor) {
                            changed |= require_accel_traversal_descriptor(
                                function, actual);
                        }
                        if (incoming.requires_bindless_buffer_metadata) {
                            changed |= require_bindless_buffer_metadata(
                                function, actual);
                        }
                        if (incoming.requires_buffer_device_address) {
                            changed |= require_buffer_device_address(
                                function, actual);
                        }
                        if (incoming.requires_buffer_coherence) {
                            changed |= require_buffer_coherence(
                                function, actual);
                        }
                    }
                });
        }
    }
    return analysis;
}

Usage spirv_function_argument_usage_of(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const xir::Function *function,
    const xir::Argument *argument) noexcept {
    return function_argument_analysis_of(
               analysis, function, argument)
        .usage;
}

bool spirv_function_argument_requires_accel_traversal_descriptor(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const xir::Function *function,
    const xir::Argument *argument) noexcept {
    return function_argument_analysis_of(
               analysis, function, argument)
        .requires_accel_traversal_descriptor;
}

bool spirv_function_argument_requires_accel_instance_buffer(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const xir::Function *function,
    const xir::Argument *argument) noexcept {
    return function_argument_analysis_of(
               analysis, function, argument)
        .requires_accel_instance_buffer;
}

bool spirv_function_argument_requires_bindless_buffer_metadata(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const xir::Function *function,
    const xir::Argument *argument) noexcept {
    return function_argument_analysis_of(
               analysis, function, argument)
        .requires_bindless_buffer_metadata;
}

bool spirv_function_argument_requires_buffer_device_address(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const xir::Function *function,
    const xir::Argument *argument) noexcept {
    return function_argument_analysis_of(
               analysis, function, argument)
        .requires_buffer_device_address;
}

bool spirv_function_argument_requires_buffer_coherence(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const xir::Function *function,
    const xir::Argument *argument) noexcept {
    return function_argument_analysis_of(
               analysis, function, argument)
        .requires_buffer_coherence;
}

}// namespace lc::spirv
