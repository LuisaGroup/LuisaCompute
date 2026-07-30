#include "atomic_target_contract.h"

#include "structural_closure.h"

#include <luisa/ast/type.h>
#include <luisa/core/stl/format.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/gep.h>

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;

namespace {

[[nodiscard]] const xir::Value *root_address(
    const xir::Value *value) noexcept {
    while (value != nullptr && value->isa<xir::GEPInst>()) {
        value = static_cast<const xir::GEPInst *>(value)->base();
    }
    return value;
}

[[nodiscard]] luisa::string_view shared_float_atomic_feature(
    xir::AtomicOp op) noexcept {
    switch (op) {
        case xir::AtomicOp::EXCHANGE:
            return "shaderSharedFloat32Atomics";
        case xir::AtomicOp::FETCH_ADD:
        case xir::AtomicOp::FETCH_SUB:
            return "shaderSharedFloat32AtomicAdd";
        case xir::AtomicOp::FETCH_MIN:
        case xir::AtomicOp::FETCH_MAX:
            return "shaderSharedFloat32AtomicMinMax";
        default: return "<none>";
    }
}

}// namespace

SpirvAtomicTargetContractResult
validate_spirv_atomic_target_contract(
    luisa::span<const xir::Function *const> functions,
    const SpirvTargetFeatures &features) noexcept {
    SpirvAtomicTargetContractResult result;
    for (auto *function : functions) {
        if (function == nullptr || !function->is_definition()) {
            continue;
        }
        traverse_spirv_codegen_structural_instructions(
            function->definition(),
            [&](const xir::Instruction *instruction) noexcept {
                if (!instruction->isa<xir::AtomicInst>()) { return; }
                auto *atomic =
                    static_cast<const xir::AtomicInst *>(instruction);
                auto *leaf_type = atomic->type();
                auto *root = root_address(atomic->base());
                auto *alloca =
                    root != nullptr && root->isa<xir::AllocaInst>() ?
                        static_cast<const xir::AllocaInst *>(root) :
                        nullptr;
                auto shared = alloca != nullptr && alloca->is_shared();
                auto buffer = root != nullptr && root->type() != nullptr &&
                              root->type()->is_buffer();

                // All remaining cases are dialect invariants. Keep this
                // planner total for direct unit use without duplicating those
                // diagnostics or dereferencing malformed instructions.
                if (leaf_type == nullptr || (!shared && !buffer)) { return; }

                if (leaf_type->is_float32()) {
                    auto storage = shared ?
                                       SpirvFloatAtomicStorage::SHARED :
                                       SpirvFloatAtomicStorage::BUFFER;
                    auto implementation = plan_spirv_float_atomic(
                        atomic->op(), 32u, storage, features);
                    if (implementation ==
                        SpirvFloatAtomicImplementation::UNSUPPORTED_FEATURE) {
                        result.diagnostics.emplace_back(
                            SpirvAtomicTargetContractDiagnostic{
                                .function = function,
                                .instruction = atomic,
                                .message = luisa::format(
                                    "Native XIR-to-SPIR-V shared float32 atomic '{}' requires target feature '{}'.",
                                    xir::to_string(atomic->op()),
                                    shared_float_atomic_feature(
                                        atomic->op()))});
                    }
                    return;
                }

                if (!leaf_type->is_int64() &&
                    !leaf_type->is_uint64()) {
                    return;
                }
                auto supported = shared ?
                                     features.shader_shared_int64_atomics :
                                     features.shader_buffer_int64_atomics;
                if (!supported) {
                    result.diagnostics.emplace_back(
                        SpirvAtomicTargetContractDiagnostic{
                            .function = function,
                            .instruction = atomic,
                            .message = luisa::format(
                                "Native XIR-to-SPIR-V {} int64 atomic '{}' requires target feature '{}'.",
                                shared ? "shared" : "buffer",
                                xir::to_string(atomic->op()),
                                shared ?
                                    "shaderSharedInt64Atomics" :
                                    "shaderBufferInt64Atomics")});
                }
            });
    }
    return result;
}

}// namespace lc::spirv
