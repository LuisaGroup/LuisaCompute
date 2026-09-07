#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class BasicBlock;
class Instruction;

namespace detail {

// Immutable dense view of one ray-query handler region. Scratch coordinates
// differ per captured alloca, but the CFG and textual instruction order do
// not change until every localization proof has completed.
class RayQueryHandlerGraph {

public:
    struct Block {
        BasicBlock *source{nullptr};
        luisa::vector<size_t> successors;
        size_t successor_count{0u};
        size_t external_successor_count{0u};
        size_t instruction_count{0u};
    };

    struct InstructionLocation {
        size_t block_id;
        size_t ordinal;
    };

private:
    struct PointerIdentityHash {
        [[nodiscard]] size_t operator()(const void *pointer) const noexcept {
            return static_cast<size_t>(
                reinterpret_cast<uintptr_t>(pointer));
        }
    };

    luisa::vector<Block> _blocks;
    luisa::unordered_map<const BasicBlock *, size_t, PointerIdentityHash>
        _block_ids;
    luisa::unordered_map<const Instruction *, InstructionLocation,
                         PointerIdentityHash>
        _instruction_locations;
    size_t _instruction_count{0u};
    bool _valid{true};

public:
    explicit RayQueryHandlerGraph(
        const luisa::unordered_set<BasicBlock *> &blocks) noexcept;

    [[nodiscard]] bool valid() const noexcept { return _valid; }
    [[nodiscard]] bool contains(const BasicBlock *block) const noexcept;
    [[nodiscard]] size_t block_id(const BasicBlock *block) const noexcept;
    [[nodiscard]] const Block &block(size_t id) const noexcept;
    [[nodiscard]] size_t size() const noexcept { return _blocks.size(); }
    [[nodiscard]] size_t instruction_count() const noexcept {
        return _instruction_count;
    }
    [[nodiscard]] const InstructionLocation *instruction_location(
        const Instruction *instruction) const noexcept;
    [[nodiscard]] size_t block_instruction_count(
        const BasicBlock *block) const noexcept;
    [[nodiscard]] bool verify(
        const luisa::unordered_set<BasicBlock *> &blocks) const noexcept;
};

}// namespace detail
}// namespace luisa::compute::xir
