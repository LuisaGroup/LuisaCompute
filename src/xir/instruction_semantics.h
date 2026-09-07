#pragma once

#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/xir/instruction.h>

namespace luisa::compute::xir::detail {

[[nodiscard]] bool interchange_instruction_semantics_valid(
    DerivedInstructionTag tag, int64_t op, const Type *type,
    luisa::span<const Value *const> operands,
    BindlessResourceAccess bindless_access = {}) noexcept;

}// namespace luisa::compute::xir::detail
