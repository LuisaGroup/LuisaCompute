#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/switch.h>

namespace luisa::compute::xir {

SwitchInst::SwitchInst(BasicBlock *parent_block, Value *value) noexcept
    : Super{parent_block, value} {}

SwitchInst *SwitchInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    auto resolved_value = resolver.resolve(value());
    auto cloned = b.switch_(resolved_value);
    auto resolved_default = resolver.resolve(default_block());
    LUISA_DEBUG_ASSERT(resolved_default == nullptr || resolved_default->isa<BasicBlock>(), "Invalid default block.");
    cloned->set_default_block(static_cast<BasicBlock *>(resolved_default));
    auto resolved_merge = resolver.resolve(merge_block());
    LUISA_DEBUG_ASSERT(resolved_merge == nullptr || resolved_merge->isa<BasicBlock>(), "Invalid merge block.");
    cloned->set_merge_block(static_cast<BasicBlock *>(resolved_merge));
    for (auto i = 0u; i < case_count(); i++) {
        auto resolved_case = resolver.resolve(case_block(i));
        LUISA_DEBUG_ASSERT(resolved_case == nullptr || resolved_case->isa<BasicBlock>(), "Invalid case block.");
        cloned->add_case(case_value(i), static_cast<BasicBlock *>(resolved_case));
    }
    return cloned;
}

}// namespace luisa::compute::xir
