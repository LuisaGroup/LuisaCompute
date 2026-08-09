#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LUISA_XIR_API ResourceQueryInst final : public InstructionOpMixin<ResourceQueryOp, DerivedInstruction<ResourceQueryInst, DerivedInstructionTag::RESOURCE_QUERY>> {
private:
    BindlessResourceAccess _bindless_access;

public:
    ResourceQueryInst(BasicBlock *parent_block,
                      const Type *type, ResourceQueryOp op,
                      luisa::span<Value *const> operands = {}) noexcept;
    ResourceQueryInst(BasicBlock *parent_block,
                      const Type *type, ResourceQueryOp op,
                      luisa::span<Value *const> operands,
                      BindlessResourceAccess bindless_access) noexcept;
    [[nodiscard]] auto bindless_access() const noexcept {
        return _bindless_access;
    }
    [[nodiscard]] ResourceQueryInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

class LUISA_XIR_API ResourceReadInst final : public InstructionOpMixin<ResourceReadOp, DerivedInstruction<ResourceReadInst, DerivedInstructionTag::RESOURCE_READ>> {
private:
    BindlessResourceAccess _bindless_access;

public:
    ResourceReadInst(BasicBlock *parent_block,
                     const Type *type, ResourceReadOp op,
                     luisa::span<Value *const> operands = {}) noexcept;
    ResourceReadInst(BasicBlock *parent_block,
                     const Type *type, ResourceReadOp op,
                     luisa::span<Value *const> operands,
                     BindlessResourceAccess bindless_access) noexcept;
    [[nodiscard]] auto bindless_access() const noexcept {
        return _bindless_access;
    }
    [[nodiscard]] ResourceReadInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

class LUISA_XIR_API ResourceWriteInst final : public InstructionOpMixin<ResourceWriteOp, DerivedInstruction<ResourceWriteInst, DerivedInstructionTag::RESOURCE_WRITE>> {
private:
    BindlessResourceAccess _bindless_access;

public:
    ResourceWriteInst(BasicBlock *parent_block, ResourceWriteOp op,
                      luisa::span<Value *const> operands = {}) noexcept;
    ResourceWriteInst(BasicBlock *parent_block, ResourceWriteOp op,
                      luisa::span<Value *const> operands,
                      BindlessResourceAccess bindless_access) noexcept;
    [[nodiscard]] auto bindless_access() const noexcept {
        return _bindless_access;
    }
    [[nodiscard]] ResourceWriteInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
