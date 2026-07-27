#include "ray_query_lifetime.h"
#include "structural_closure.h"

#include <utility>

#include <luisa/ast/type.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/dom_tree.h>

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;

namespace {

[[nodiscard]] bool is_ray_query_type(const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           (type->description() == "LC_RayQueryAll" ||
            type->description() == "LC_RayQueryAny");
}

[[nodiscard]] bool is_query_initializer(
    const xir::Value *value) noexcept {
    if (value == nullptr ||
        !value->isa<xir::ResourceQueryInst>()) {
        return false;
    }
    auto *query = static_cast<const xir::ResourceQueryInst *>(value);
    switch (query->op()) {
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
            return true;
        default: return false;
    }
}

}// namespace

SpirvRayQueryLifetimeValidationResult
validate_spirv_ray_query_lifetimes(
    const xir::FunctionDefinition *function) noexcept {
    SpirvRayQueryLifetimeValidationResult result;
    auto error = [&](const xir::BasicBlock *block,
                     const xir::Instruction *instruction,
                     luisa::string message) noexcept {
        result.diagnostics.emplace_back(
            SpirvRayQueryLifetimeDiagnostic{
                .block = block,
                .instruction = instruction,
                .message = std::move(message),
            });
    };
    if (function == nullptr) {
        error(nullptr, nullptr,
              "Native XIR-to-SPIR-V ray-query lifetime validation requires a non-null function definition.");
        return result;
    }
    auto function_name = function->name().value_or("<unnamed>");
    auto closure = plan_spirv_codegen_structural_closure(function);
    if (!closure.succeeded()) {
        error(nullptr, nullptr,
              luisa::format(
                  "Native XIR-to-SPIR-V cannot validate ray-query lifetimes in function '{}': its structural closure is invalid.",
                  function_name));
        return result;
    }

    luisa::unordered_set<const xir::BasicBlock *> active_blocks;
    active_blocks.reserve(closure.ordinary_block_count);
    for (auto i = size_t{0u};
         i < closure.ordinary_block_count; ++i) {
        active_blocks.emplace(closure.blocks[i]);
    }

    if (is_ray_query_type(function->type())) {
        error(nullptr, nullptr,
              luisa::format(
                  "Native XIR-to-SPIR-V callable '{}' cannot return opaque ray-query objects.",
                  function_name));
    }
    for (auto *argument : function->arguments()) {
        if (is_ray_query_type(argument->type()) &&
            !argument->is_reference()) {
            error(nullptr, nullptr,
                  luisa::format(
                      "Native XIR-to-SPIR-V callable '{}' requires every ray-query argument to be passed by reference.",
                      function_name));
        }
    }

    luisa::unordered_map<const xir::Instruction *, size_t>
        instruction_indices;
    luisa::unordered_map<const xir::Value *, const xir::StoreInst *>
        query_initializer_bindings;
    luisa::vector<const xir::ResourceQueryInst *> query_initializers;
    luisa::vector<const xir::AllocaInst *> query_allocas;
    for (auto i = size_t{0u};
         i < closure.ordinary_block_count; ++i) {
        auto *block = closure.blocks[i];
        auto instruction_index = size_t{0u};
        for (auto *instruction : block->instructions()) {
            instruction_indices.emplace(
                instruction, instruction_index++);
            if (instruction->isa<xir::StoreInst>()) {
                auto *store =
                    static_cast<const xir::StoreInst *>(instruction);
                auto *value = store->value();
                if (value != nullptr &&
                    is_ray_query_type(value->type())) {
                    auto *variable = store->variable();
                    if (variable == nullptr ||
                        !variable->isa<xir::AllocaInst>() ||
                        !is_ray_query_type(variable->type()) ||
                        !is_query_initializer(value)) {
                        error(
                            block, store,
                            luisa::format(
                                "Native XIR-to-SPIR-V callable '{}' cannot copy or rebind an opaque ray-query object. Only a direct query initializer may bind a local query alloca once.",
                                function_name));
                    } else if (!query_initializer_bindings
                                    .emplace(value, store)
                                    .second) {
                        error(
                            block, store,
                            luisa::format(
                                "Native XIR-to-SPIR-V callable '{}' cannot bind one opaque ray-query initializer to multiple allocas.",
                                function_name));
                    }
                }
            }
            if (!is_ray_query_type(instruction->type())) { continue; }
            if (instruction->isa<xir::AllocaInst>()) {
                auto *alloca =
                    static_cast<const xir::AllocaInst *>(instruction);
                if (!alloca->is_local()) {
                    error(
                        block, alloca,
                        luisa::format(
                            "Native XIR-to-SPIR-V callable '{}' cannot place an opaque ray-query object in shared memory.",
                            function_name));
                }
                query_allocas.emplace_back(alloca);
                continue;
            }
            if (instruction->isa<xir::LoadInst>()) { continue; }
            if (is_query_initializer(instruction)) {
                query_initializers.emplace_back(
                    static_cast<const xir::ResourceQueryInst *>(
                        instruction));
                continue;
            }
            error(
                block, instruction,
                luisa::format(
                    "Native XIR-to-SPIR-V callable '{}' cannot materialize an opaque ray-query value with instruction '{}'. Ray-query select, Phi, call-return, cast, and composite forms are unsupported.",
                    function_name,
                    xir::to_string(
                        instruction->derived_instruction_tag())));
        }
    }

    for (auto *initializer : query_initializers) {
        auto binding = query_initializer_bindings.find(initializer);
        if (binding == query_initializer_bindings.end()) { continue; }
        for (auto *use : initializer->use_list()) {
            auto *user = use->user();
            if (user != nullptr && user->isa<xir::Instruction>()) {
                auto *user_instruction =
                    static_cast<const xir::Instruction *>(user);
                if (!active_blocks.contains(
                        user_instruction->parent_block())) {
                    continue;
                }
            }
            if (user != binding->second) {
                error(
                    user != nullptr && user->isa<xir::Instruction>() ?
                        static_cast<const xir::Instruction *>(user)
                            ->parent_block() :
                        nullptr,
                    user != nullptr && user->isa<xir::Instruction>() ?
                        static_cast<const xir::Instruction *>(user) :
                        nullptr,
                    luisa::format(
                        "Native XIR-to-SPIR-V callable '{}' cannot both bind and alias an opaque ray-query initializer.",
                        function_name));
            }
        }
    }

    if (query_allocas.empty()) { return result; }
    auto dom = xir::compute_dom_tree(
        const_cast<xir::FunctionDefinition *>(function));
    auto dominates = [&](const xir::Instruction *definition,
                         const xir::Instruction *use) noexcept {
        auto *definition_block = const_cast<xir::BasicBlock *>(
            definition->parent_block());
        auto *use_block = const_cast<xir::BasicBlock *>(
            use->parent_block());
        if (!dom.contains(definition_block) ||
            !dom.contains(use_block)) {
            return false;
        }
        if (definition_block != use_block) {
            return dom.dominates(definition_block, use_block);
        }
        auto definition_index = instruction_indices.find(definition);
        auto use_index = instruction_indices.find(use);
        return definition_index != instruction_indices.end() &&
               use_index != instruction_indices.end() &&
               definition_index->second < use_index->second;
    };

    for (auto *alloca : query_allocas) {
        const xir::StoreInst *binding_store = nullptr;
        luisa::vector<const xir::Instruction *> object_uses;
        for (auto *use : alloca->use_list()) {
            auto *user = use->user();
            if (user == nullptr || !user->isa<xir::Instruction>()) {
                error(
                    alloca->parent_block(), alloca,
                    luisa::format(
                        "Native XIR-to-SPIR-V callable '{}' found a non-instruction use of an opaque ray-query alloca.",
                        function_name));
                continue;
            }
            auto *user_instruction =
                static_cast<const xir::Instruction *>(user);
            if (!active_blocks.contains(
                    user_instruction->parent_block())) {
                continue;
            }
            if (user_instruction->isa<xir::StoreInst>()) {
                auto *store = static_cast<const xir::StoreInst *>(
                    user_instruction);
                if (store->variable() != alloca) {
                    error(
                        user_instruction->parent_block(), store,
                        luisa::format(
                            "Native XIR-to-SPIR-V callable '{}' cannot copy an opaque ray-query alloca.",
                            function_name));
                    continue;
                }
                if (binding_store != nullptr) {
                    error(
                        user_instruction->parent_block(), store,
                        luisa::format(
                            "Native XIR-to-SPIR-V callable '{}' cannot reassign an opaque ray-query alloca.",
                            function_name));
                    continue;
                }
                binding_store = store;
                continue;
            }
            auto allowed_use =
                user_instruction->isa<xir::LoadInst>() ||
                user_instruction->isa<xir::CallInst>() ||
                user_instruction->isa<xir::RayQueryObjectReadInst>() ||
                user_instruction->isa<xir::RayQueryObjectWriteInst>() ||
                user_instruction->isa<xir::RayQueryDispatchInst>() ||
                user_instruction->isa<xir::RayQueryPipelineInst>();
            if (!allowed_use) {
                error(
                    user_instruction->parent_block(), user_instruction,
                    luisa::format(
                        "Native XIR-to-SPIR-V callable '{}' has unsupported '{}' use of an opaque ray-query alloca.",
                        function_name,
                        xir::to_string(
                            user_instruction->derived_instruction_tag())));
            }
            object_uses.emplace_back(user_instruction);
        }
        if (binding_store == nullptr) {
            if (!object_uses.empty()) {
                error(
                    object_uses.front()->parent_block(),
                    object_uses.front(),
                    luisa::format(
                        "Native XIR-to-SPIR-V callable '{}' uses an opaque ray-query alloca before initialization.",
                        function_name));
            }
            continue;
        }
        if (!is_query_initializer(binding_store->value())) {
            error(
                binding_store->parent_block(), binding_store,
                luisa::format(
                    "Native XIR-to-SPIR-V callable '{}' cannot initialize an opaque ray-query alloca from a copied query.",
                    function_name));
        }
        if (!dom.contains(const_cast<xir::BasicBlock *>(
                binding_store->parent_block()))) {
            error(
                binding_store->parent_block(), binding_store,
                luisa::format(
                    "Native XIR-to-SPIR-V callable '{}' has an unreachable ray-query initialization.",
                    function_name));
        }
        for (auto *use : object_uses) {
            if (!dominates(binding_store, use)) {
                error(
                    use->parent_block(), use,
                    luisa::format(
                        "Native XIR-to-SPIR-V callable '{}' requires the single ray-query initialization to dominate every use; conditional initialization is unsupported.",
                        function_name));
            }
        }
    }
    return result;
}

}// namespace lc::spirv
