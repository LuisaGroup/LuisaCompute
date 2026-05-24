#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/xir/op.h>
#include <luisa/xir/instruction.h>
#include <luisa/ast/type.h>

namespace luisa::compute::xir {

class AllocaInst;
class PhiInst;
class Value;
class Instruction;
class XIRBuilder;
class FunctionDefinition;

struct InstructionCloneValueResolver;

[[nodiscard]] LUISA_XIR_API Value *trace_pointer_base_value(Value *pointer) noexcept;
[[nodiscard]] LUISA_XIR_API AllocaInst *trace_pointer_base_local_alloca_inst(Value *pointer) noexcept;
[[nodiscard]] LUISA_XIR_API bool remove_redundant_phi_instruction(PhiInst *phi) noexcept;
LUISA_XIR_API void lower_phi_node_to_local_variable(PhiInst *phi) noexcept;
LUISA_XIR_API void hoist_alloca_instructions_to_entry_block(FunctionDefinition *f) noexcept;

[[nodiscard]] bool eval_scalar_op(const Type *type, ArithmeticOp op,
                                  void *data,
                                  const void *op0_data,
                                  const void *op1_data,
                                  const void *op2_data) noexcept;

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

[[nodiscard]] inline InstructionMemoryInfo get_memory_info(Instruction *inst) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ARITHMETIC:
        case DerivedInstructionTag::CAST:
        case DerivedInstructionTag::GEP:
        case DerivedInstructionTag::PHI:
        case DerivedInstructionTag::CLOCK:
            return {MemoryScope::NONE, MemoryEffects::NONE, false};
        case DerivedInstructionTag::RESOURCE_QUERY:
            return {MemoryScope::GLOBAL, MemoryEffects::NONE, false};
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
            return {MemoryScope::LOCAL, MemoryEffects::READ, false};
        case DerivedInstructionTag::ALLOCA:
            return {MemoryScope::LOCAL, MemoryEffects::NONE, false};
        case DerivedInstructionTag::LOAD:
            return {MemoryScope::LOCAL, MemoryEffects::READ, false};
        case DerivedInstructionTag::STORE:
            return {MemoryScope::LOCAL, MemoryEffects::WRITE, false};
        case DerivedInstructionTag::RESOURCE_READ:
            return {MemoryScope::GLOBAL, MemoryEffects::READ, false};
        case DerivedInstructionTag::RESOURCE_WRITE:
            return {MemoryScope::GLOBAL, MemoryEffects::WRITE, false};
        case DerivedInstructionTag::ATOMIC:
            return {MemoryScope::GLOBAL, MemoryEffects::READ_WRITE, false};
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            return {MemoryScope::LOCAL, MemoryEffects::WRITE, false};
        case DerivedInstructionTag::THREAD_GROUP:
            return {MemoryScope::SHARED, MemoryEffects::READ_WRITE, true};
        case DerivedInstructionTag::CALL:
            return {MemoryScope::GLOBAL, MemoryEffects::READ_WRITE, false};
        case DerivedInstructionTag::PRINT:
        case DerivedInstructionTag::DEBUG_BREAK:
            return {MemoryScope::NONE, MemoryEffects::NONE, true};
        case DerivedInstructionTag::ASSERT:
        case DerivedInstructionTag::ASSUME:
            return {MemoryScope::NONE, MemoryEffects::NONE, true};
        case DerivedInstructionTag::AUTODIFF_SCOPE:
        case DerivedInstructionTag::AUTODIFF_INTRINSIC:
            return {MemoryScope::GLOBAL, MemoryEffects::READ_WRITE, true};
        default:
            return {MemoryScope::NONE, MemoryEffects::NONE, true};
    }
}

}// namespace luisa::compute::xir
