#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/phi.h>

namespace luisa::compute::xir {

void PhiInst::set_incoming_count(size_t count) noexcept {
    set_operand_count(count * 2);
}

void PhiInst::set_incoming(size_t index, Value *value, BasicBlock *block) noexcept {
    LUISA_DEBUG_ASSERT(index < incoming_count(), "Phi incoming index out of range.");
    set_operand(index * 2 + 0, value);
    set_operand(index * 2 + 1, block);
}

void PhiInst::add_incoming(Value *value, BasicBlock *block) noexcept {
    add_operand(value);
    add_operand(block);
}

void PhiInst::insert_incoming(size_t index, Value *value, BasicBlock *block) noexcept {
    insert_operand(index * 2 + 0, value);
    insert_operand(index * 2 + 1, block);
}

void PhiInst::remove_incoming(size_t index) noexcept {
    if (index < incoming_count()) {
        remove_operand(index * 2 + 1);
        remove_operand(index * 2 + 0);
    }
}

size_t PhiInst::incoming_count() const noexcept {
    LUISA_DEBUG_ASSERT(operand_count() % 2 == 0, "Invalid phi operand count.");
    return operand_count() / 2;
}

PhiIncoming PhiInst::incoming(size_t index) noexcept {
    LUISA_DEBUG_ASSERT(index < incoming_count(), "Phi incoming index out of range.");
    auto value = operand(index * 2 + 0);
    auto block = static_cast<BasicBlock *>(operand(index * 2 + 1));
    return {value, block};
}

ConstPhiIncoming PhiInst::incoming(size_t index) const noexcept {
    LUISA_DEBUG_ASSERT(index < incoming_count(), "Phi incoming index out of range.");
    auto value = operand(index * 2 + 0);
    auto block = static_cast<const BasicBlock *>(operand(index * 2 + 1));
    return {value, block};
}

PhiIncomingUse PhiInst::incoming_use(size_t index) noexcept {
    LUISA_DEBUG_ASSERT(index < incoming_count(), "Phi incoming index out of range.");
    auto value = operand_use(index * 2 + 0);
    auto block = operand_use(index * 2 + 1);
    return {value, block};
}

ConstPhiIncomingUse PhiInst::incoming_use(size_t index) const noexcept {
    LUISA_DEBUG_ASSERT(index < incoming_count(), "Phi incoming index out of range.");
    auto value = operand_use(index * 2 + 0);
    auto block = operand_use(index * 2 + 1);
    return {value, block};
}

luisa::span<PhiIncomingUse> PhiInst::incoming_uses() noexcept {
    auto n = incoming_count();
    auto uses = operand_uses();
    return {reinterpret_cast<PhiIncomingUse *>(uses.data()), n};
}

luisa::span<const ConstPhiIncomingUse> PhiInst::incoming_uses() const noexcept {
    auto n = incoming_count();
    auto uses = operand_uses();
    return {reinterpret_cast<const ConstPhiIncomingUse *>(uses.data()), n};
}

}// namespace luisa::compute::xir
