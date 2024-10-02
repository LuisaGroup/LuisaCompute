#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/switch.h>

namespace luisa::compute::xir {

SwitchInst::SwitchInst(Pool *pool, Value *value, const Name *name) noexcept
    : Instruction{pool, nullptr, name} {
    auto merge_block = static_cast<Value *>(pool->create<BasicBlock>(this));
    auto default_block = static_cast<Value *>(pool->create<BasicBlock>(this));
    auto operands = std::array{value, merge_block, default_block};
    LUISA_DEBUG_ASSERT(operands[operand_index_value] == value, "Unexpected operand index.");
    LUISA_DEBUG_ASSERT(operands[operand_index_merge_block] == merge_block, "Unexpected operand index.");
    LUISA_DEBUG_ASSERT(operands[operand_index_default_block] == default_block, "Unexpected operand index.");
    set_operands(operands);
}

void SwitchInst::set_value(Value *value) noexcept {
    set_operand(operand_index_value, value);
}

void SwitchInst::set_case_count(size_t count) noexcept {
    _case_values.resize(count);
    set_operand_count(operand_index_case_block_offset + count);
}

size_t SwitchInst::case_count() const noexcept {
    LUISA_DEBUG_ASSERT(operand_count() == operand_index_case_block_offset + _case_values.size(),
                       "Invalid switch operand count.");
    return _case_values.size();
}

void SwitchInst::set_case_value(size_t index, case_value_type value) noexcept {
    LUISA_DEBUG_ASSERT(index < case_count(), "Switch case index out of range.");
    _case_values[index] = value;
}

BasicBlock *SwitchInst::add_case(case_value_type value) noexcept {
    _case_values.emplace_back(value);
    auto block = pool()->create<BasicBlock>(this);
    add_operand(block);
    return block;
}

BasicBlock *SwitchInst::insert_case(size_t index, case_value_type value) noexcept {
    LUISA_DEBUG_ASSERT(index <= case_count(), "Switch case index out of range.");
    _case_values.insert(_case_values.cbegin() + index, value);
    auto block = pool()->create<BasicBlock>(this);
    insert_operand(operand_index_case_block_offset + index, block);
    return block;
}

void SwitchInst::remove_case(size_t index) noexcept {
    if (index < case_count()) {
        _case_values.erase(_case_values.cbegin() + index);
        remove_operand(operand_index_case_block_offset + index);
    }
}

SwitchInst::case_value_type SwitchInst::case_value(size_t index) const noexcept {
    LUISA_DEBUG_ASSERT(index < case_count(), "Switch case index out of range.");
    return _case_values[index];
}

BasicBlock *SwitchInst::case_block(size_t index) noexcept {
    LUISA_DEBUG_ASSERT(index < case_count(), "Switch case index out of range.");
    return static_cast<BasicBlock *>(operand(operand_index_case_block_offset + index));
}

const BasicBlock *SwitchInst::case_block(size_t index) const noexcept {
    return const_cast<SwitchInst *>(this)->case_block(index);
}

luisa::span<const SwitchInst::case_value_type> SwitchInst::case_values() const noexcept {
    return _case_values;
}

luisa::span<Use *> SwitchInst::case_block_uses() noexcept {
    return operand_uses().subspan(operand_index_case_block_offset);
}

luisa::span<const Use *const> SwitchInst::case_block_uses() const noexcept {
    return const_cast<SwitchInst *>(this)->case_block_uses();
}

Value *SwitchInst::value() noexcept {
    return operand(operand_index_value);
}

const Value *SwitchInst::value() const noexcept {
    return operand(operand_index_value);
}

BasicBlock *SwitchInst::merge_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_merge_block));
}

const BasicBlock *SwitchInst::merge_block() const noexcept {
    return const_cast<SwitchInst *>(this)->merge_block();
}

BasicBlock *SwitchInst::default_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_default_block));
}

const BasicBlock *SwitchInst::default_block() const noexcept {
    return const_cast<SwitchInst *>(this)->default_block();
}

}// namespace luisa::compute::xir
