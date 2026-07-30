#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>

namespace luisa::compute::xir {
class FunctionDefinition;
class Instruction;
}// namespace luisa::compute::xir

namespace lc::spirv {

enum class SpirvCodegenStructuralClosureStatus : uint8_t {
    SUCCESS,
    NULL_FUNCTION,
    MISSING_BODY,
    UNOWNED_BLOCK,
};

struct SpirvCodegenStructuralClosure {
    // The prefix [0, ordinary_block_count) is reachable from the function body
    // by encoded CFG operands. The suffix contains blocks included only by raw
    // structured-role ownership, recursively closed over their operands/roles.
    luisa::vector<const luisa::compute::xir::BasicBlock *> blocks;
    size_t ordinary_block_count{0u};
    SpirvCodegenStructuralClosureStatus status{
        SpirvCodegenStructuralClosureStatus::SUCCESS};
    const luisa::compute::xir::BasicBlock *invalid_block{nullptr};
    // When a required encoded/raw structural role is malformed, retain its
    // owner so non-fatal callers can report the source of the bad edge rather
    // than only the (possibly null or non-block) target.
    const luisa::compute::xir::Instruction *invalid_instruction{nullptr};
    luisa::string_view invalid_role;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SpirvCodegenStructuralClosureStatus::SUCCESS;
    }
};

[[nodiscard]] SpirvCodegenStructuralClosure
plan_spirv_codegen_structural_closure(
    const luisa::compute::xir::FunctionDefinition *function) noexcept;

// Returns exactly the block identities that native SPIR-V codegen may emit:
// ordinary CFG reachability plus recursively owned structured role blocks.
// Function-owned blocks outside this closure are true orphans and must not
// affect the dialect, planner, or emitter boundary.
[[nodiscard]] luisa::vector<const luisa::compute::xir::BasicBlock *>
collect_spirv_codegen_structural_closure(
    const luisa::compute::xir::FunctionDefinition *function) noexcept;

template<typename Visit>
void traverse_spirv_codegen_structural_instructions(
    const luisa::compute::xir::FunctionDefinition *function,
    Visit &&visit) noexcept {
    for (auto *block :
         collect_spirv_codegen_structural_closure(function)) {
        // `visit` is shared by every block in the closure. Keep it as an
        // lvalue here: forwarding the same rvalue repeatedly could move state
        // out of a stateful visitor on the first block.
        block->traverse_instructions(visit);
    }
}

}// namespace lc::spirv
