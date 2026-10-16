#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/resource.h>

namespace luisa::compute::xir {

ResourceQueryInst::ResourceQueryInst(BasicBlock *parent_block, const Type *type, ResourceQueryOp op,
                                     luisa::span<Value *const> operands) noexcept
    : ResourceQueryInst{parent_block, type, op, operands, {}} {}

ResourceQueryInst::ResourceQueryInst(BasicBlock *parent_block, const Type *type, ResourceQueryOp op,
                                     luisa::span<Value *const> operands,
                                     BindlessResourceAccess bindless_access) noexcept
    : Super{op, parent_block, type},
      _bindless_access{bindless_access} { set_operands(operands); }

ResourceQueryInst *ResourceQueryInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 8u> cloned_operands;
    cloned_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        cloned_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    return b.call(type(), op(), cloned_operands, bindless_access());
}

ResourceReadInst::ResourceReadInst(BasicBlock *parent_block, const Type *type, ResourceReadOp op,
                                   luisa::span<Value *const> operands) noexcept
    : ResourceReadInst{parent_block, type, op, operands, {}} {}

ResourceReadInst::ResourceReadInst(BasicBlock *parent_block, const Type *type, ResourceReadOp op,
                                   luisa::span<Value *const> operands,
                                   BindlessResourceAccess bindless_access) noexcept
    : Super{op, parent_block, type},
      _bindless_access{bindless_access} { set_operands(operands); }

ResourceReadInst *ResourceReadInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 8u> cloned_operands;
    cloned_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        cloned_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    return b.call(type(), op(), cloned_operands, bindless_access());
}

ResourceWriteInst::ResourceWriteInst(BasicBlock *parent_block, ResourceWriteOp op,
                                     luisa::span<Value *const> operands) noexcept
    : ResourceWriteInst{parent_block, op, operands, {}} {}

ResourceWriteInst::ResourceWriteInst(BasicBlock *parent_block, ResourceWriteOp op,
                                     luisa::span<Value *const> operands,
                                     BindlessResourceAccess bindless_access) noexcept
    : Super{op, parent_block, nullptr},
      _bindless_access{bindless_access} { set_operands(operands); }

ResourceWriteInst *ResourceWriteInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 8u> cloned_operands;
    cloned_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        cloned_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    return b.call(op(), cloned_operands, bindless_access());
}

}// namespace luisa::compute::xir
