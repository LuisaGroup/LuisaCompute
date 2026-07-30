#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/indexed_branch.h>

namespace luisa::compute::xir {

IndexedBranchTerminatorInstruction::IndexedBranchTerminatorInstruction(
    BasicBlock *parent_block, Value *value) noexcept
    : TerminatorInstruction{parent_block} {
    auto default_block = static_cast<Value *>(nullptr);
    auto operands = std::array{value, default_block};
    LUISA_DEBUG_ASSERT(
        operands[operand_index_value] == value, "Unexpected operand index.");
    set_operands(operands);
}

IndexedBranchTerminatorInstruction::case_value_type
IndexedBranchTerminatorInstruction::canonicalize_case_value(
    const Type *selector_type, case_value_type value) noexcept {
    if (selector_type == nullptr) { return value; }
    uint32_t bit_width = 0u;
    switch (selector_type->tag()) {
        case Type::Tag::BOOL: bit_width = 1u; break;
        case Type::Tag::INT8:
        case Type::Tag::UINT8: bit_width = 8u; break;
        case Type::Tag::INT16:
        case Type::Tag::UINT16: bit_width = 16u; break;
        case Type::Tag::INT32:
        case Type::Tag::UINT32: bit_width = 32u; break;
        case Type::Tag::INT64:
        case Type::Tag::UINT64: bit_width = 64u; break;
        default: return value;
    }
    if (bit_width == 64u) { return value; }
    return value & ((case_value_type{1u} << bit_width) - 1u);
}

void IndexedBranchTerminatorInstruction::set_value(Value *value) noexcept {
    set_operand(operand_index_value, value);
    auto selector_type = value == nullptr ? nullptr : value->type();
    for (auto &case_value : _case_values) {
        case_value = canonicalize_case_value(selector_type, case_value);
    }
}

void IndexedBranchTerminatorInstruction::set_default_block(
    BasicBlock *block) noexcept {
    set_operand(operand_index_default_block, block);
}

BasicBlock *
IndexedBranchTerminatorInstruction::create_default_block(
    bool overwrite_existing) noexcept {
    LUISA_ASSERT(
        default_block() == nullptr || overwrite_existing,
        "Default block already exists.");
    auto new_block = parent_function()->create_basic_block();
    set_default_block(new_block);
    return new_block;
}

BasicBlock *IndexedBranchTerminatorInstruction::create_case_block(
    case_value_type value) noexcept {
    auto new_block = parent_function()->create_basic_block();
    add_case(value, new_block);
    return new_block;
}

void IndexedBranchTerminatorInstruction::set_case(
    size_t index, case_value_type value, BasicBlock *block) noexcept {
    set_case_value(index, value);
    set_case_block(index, block);
}

void IndexedBranchTerminatorInstruction::set_case_count(size_t count) noexcept {
    _case_values.resize(count);
    set_operand_count(operand_index_case_block_offset + count);
}

size_t IndexedBranchTerminatorInstruction::case_count() const noexcept {
    LUISA_DEBUG_ASSERT(
        operand_count() ==
            operand_index_case_block_offset + _case_values.size(),
        "Invalid indexed branch operand count.");
    return _case_values.size();
}

void IndexedBranchTerminatorInstruction::set_case_value(
    size_t index, case_value_type value) noexcept {
    LUISA_DEBUG_ASSERT(
        index < case_count(), "Indexed branch case index out of range.");
    _case_values[index] = canonicalize_case_value(
        this->value() == nullptr ? nullptr : this->value()->type(), value);
}

void IndexedBranchTerminatorInstruction::set_case_block(
    size_t index, BasicBlock *block) noexcept {
    set_operand(operand_index_case_block_offset + index, block);
}

void IndexedBranchTerminatorInstruction::add_case(
    case_value_type value, BasicBlock *block) noexcept {
    _case_values.emplace_back(canonicalize_case_value(
        this->value() == nullptr ? nullptr : this->value()->type(), value));
    add_operand(block);
}

void IndexedBranchTerminatorInstruction::insert_case(
    size_t index, case_value_type value, BasicBlock *block) noexcept {
    LUISA_DEBUG_ASSERT(
        index <= case_count(), "Indexed branch case index out of range.");
    _case_values.insert(
        _case_values.cbegin() + index,
        canonicalize_case_value(
            this->value() == nullptr ? nullptr : this->value()->type(), value));
    insert_operand(operand_index_case_block_offset + index, block);
}

void IndexedBranchTerminatorInstruction::remove_case(size_t index) noexcept {
    if (index < case_count()) {
        _case_values.erase(_case_values.cbegin() + index);
        remove_operand(operand_index_case_block_offset + index);
    }
}

IndexedBranchTerminatorInstruction::case_value_type
IndexedBranchTerminatorInstruction::case_value(size_t index) const noexcept {
    LUISA_DEBUG_ASSERT(
        index < case_count(), "Indexed branch case index out of range.");
    return _case_values[index];
}

BasicBlock *IndexedBranchTerminatorInstruction::case_block(
    size_t index) noexcept {
    LUISA_DEBUG_ASSERT(
        index < case_count(), "Indexed branch case index out of range.");
    return static_cast<BasicBlock *>(
        operand(operand_index_case_block_offset + index));
}

const BasicBlock *IndexedBranchTerminatorInstruction::case_block(
    size_t index) const noexcept {
    return const_cast<IndexedBranchTerminatorInstruction *>(this)->case_block(
        index);
}

luisa::span<const IndexedBranchTerminatorInstruction::case_value_type>
IndexedBranchTerminatorInstruction::case_values() const noexcept {
    return _case_values;
}

luisa::span<Use *>
IndexedBranchTerminatorInstruction::case_block_uses() noexcept {
    return operand_uses().subspan(operand_index_case_block_offset);
}

luisa::span<const Use *const>
IndexedBranchTerminatorInstruction::case_block_uses() const noexcept {
    return const_cast<IndexedBranchTerminatorInstruction *>(this)
        ->case_block_uses();
}

Value *IndexedBranchTerminatorInstruction::value() noexcept {
    return operand(operand_index_value);
}

const Value *IndexedBranchTerminatorInstruction::value() const noexcept {
    return operand(operand_index_value);
}

BasicBlock *IndexedBranchTerminatorInstruction::default_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_default_block));
}

const BasicBlock *
IndexedBranchTerminatorInstruction::default_block() const noexcept {
    return const_cast<IndexedBranchTerminatorInstruction *>(this)
        ->default_block();
}

IndexedBranchInst::IndexedBranchInst(
    BasicBlock *parent_block, Value *value) noexcept
    : Super{parent_block, value} {}

IndexedBranchInst *IndexedBranchInst::clone(
    XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    auto resolved_value = resolver.resolve(value());
    auto cloned = b.indexed_branch(resolved_value);
    auto resolved_default = resolver.resolve(default_block());
    LUISA_DEBUG_ASSERT(
        resolved_default == nullptr || resolved_default->isa<BasicBlock>(),
        "Invalid default block.");
    cloned->set_default_block(static_cast<BasicBlock *>(resolved_default));
    for (auto i = 0u; i < case_count(); i++) {
        auto resolved_case = resolver.resolve(case_block(i));
        LUISA_DEBUG_ASSERT(
            resolved_case == nullptr || resolved_case->isa<BasicBlock>(),
            "Invalid case block.");
        cloned->add_case(
            case_value(i), static_cast<BasicBlock *>(resolved_case));
    }
    return cloned;
}

}// namespace luisa::compute::xir
