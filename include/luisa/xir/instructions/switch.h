#pragma once

#include <luisa/xir/instructions/indexed_branch.h>

namespace luisa::compute::xir {

// Switch instruction:
//
// switch (value) {
//   case case_values[0]: { case_blocks[0] }
//   case case_values[1]: { case_blocks[1] }
//   ...
//   default: { default_block }
// }
// { merge_block }
//
// Note: this instruction must be the terminator of a basic block.
class LUISA_XIR_API SwitchInst final
    : public ControlFlowMergeMixin<
          DerivedInstruction<
              SwitchInst, DerivedInstructionTag::SWITCH,
              IndexedBranchTerminatorInstruction>> {

public:
    SwitchInst(BasicBlock *parent_block, Value *value) noexcept;

    [[nodiscard]] SwitchInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
