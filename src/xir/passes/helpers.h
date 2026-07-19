#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/xir/op.h>
#include <luisa/xir/instruction.h>
#include <luisa/ast/type.h>
#include <luisa/xir/passes/dom_tree.h>

namespace luisa::compute::xir {

static constexpr uint32_t TERMINAL_TOKEN = 0xFFFFFFFFu;

class AllocaInst;
class PhiInst;
class Value;
class Instruction;
class XIRBuilder;
class FunctionDefinition;

struct InstructionCloneValueResolver;

[[nodiscard]] LUISA_XIR_API Value *trace_pointer_base_value(Value *pointer) noexcept;
[[nodiscard]] LUISA_XIR_API AllocaInst *trace_pointer_base_local_alloca_inst(Value *pointer) noexcept;
// Returns true if the function still contains structured control-flow
// instructions. CFG-only transforms use this as a hard mutation boundary.
[[nodiscard]] LUISA_XIR_API bool contains_structured_control_flow(FunctionDefinition *function) noexcept;
[[nodiscard]] LUISA_XIR_API bool remove_redundant_phi_instruction(PhiInst *phi) noexcept;
[[nodiscard]] LUISA_XIR_API bool simplify_phi_instruction(PhiInst *phi, const DomTree *dom_tree = nullptr) noexcept;
LUISA_XIR_API void lower_phi_node_to_local_variable(PhiInst *phi) noexcept;
LUISA_XIR_API void hoist_alloca_instructions_to_entry_block(FunctionDefinition *f) noexcept;

[[nodiscard]] bool eval_scalar_op(const Type *type, ArithmeticOp op,
                                  void *data,
                                  const void *op0_data,
                                  const void *op1_data,
                                  const void *op2_data) noexcept;
[[nodiscard]] bool eval_pow_int_op(const Type *result_type,
                                   const Type *exponent_type,
                                   void *data,
                                   const void *base_data,
                                   const void *exponent_data) noexcept;

enum struct MemoryScope : uint8_t {
    NONE,
    LOCAL,
    SHARED,
    GLOBAL,
};

enum struct MemoryEffects : uint8_t {
    NONE = 0u,
    READ = 1u,
    WRITE = 2u,
    READ_WRITE = 3u,
};

[[nodiscard]] constexpr MemoryEffects operator|(MemoryEffects a, MemoryEffects b) noexcept {
    return static_cast<MemoryEffects>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}
[[nodiscard]] constexpr bool has_read(MemoryEffects e) noexcept {
    return (static_cast<uint8_t>(e) & 1u) != 0u;
}
[[nodiscard]] constexpr bool has_write(MemoryEffects e) noexcept {
    return (static_cast<uint8_t>(e) & 2u) != 0u;
}

struct InstructionMemoryInfo {
    MemoryScope scope{MemoryScope::NONE};
    MemoryEffects effects{MemoryEffects::NONE};
    bool is_volatile{false};

    [[nodiscard]] constexpr bool is_pure() const noexcept {
        return effects == MemoryEffects::NONE && !is_volatile;
    }
    [[nodiscard]] constexpr bool reads_memory() const noexcept {
        return has_read(effects);
    }
    [[nodiscard]] constexpr bool writes_memory() const noexcept {
        return has_write(effects);
    }
    [[nodiscard]] constexpr bool is_removable_if_unused() const noexcept {
        return !has_write(effects) && !is_volatile;
    }
    [[nodiscard]] constexpr bool is_safe_to_value_number() const noexcept {
        return is_pure();
    }
};

[[nodiscard]] LUISA_XIR_API InstructionMemoryInfo get_memory_info(Instruction *inst) noexcept;

}// namespace luisa::compute::xir
