#include "lower_ray_query_handler_graph.h"

#include <limits>

#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instruction.h>

namespace luisa::compute::xir::detail {

RayQueryHandlerGraph::RayQueryHandlerGraph(
    const luisa::unordered_set<BasicBlock *> &blocks) noexcept {
    _blocks.reserve(blocks.size());
    _block_ids.reserve(blocks.size());
    for (auto *block : blocks) {
        if (block == nullptr) {
            _valid = false;
            continue;
        }
        auto id = _blocks.size();
        auto [iter, inserted] = _block_ids.emplace(block, id);
        static_cast<void>(iter);
        if (!inserted) {
            _valid = false;
            continue;
        }
        _blocks.emplace_back(Block{.source = block});
    }
    for (auto block_id = 0u; block_id < _blocks.size(); ++block_id) {
        auto &record = _blocks[block_id];
        auto ordinal = size_t{0u};
        for (auto *instruction : record.source->instructions()) {
            auto [iter, inserted] = _instruction_locations.emplace(
                instruction,
                InstructionLocation{block_id, ordinal++});
            static_cast<void>(iter);
            _valid &= inserted;
        }
        record.instruction_count = ordinal;
        _instruction_count += ordinal;
        record.source->traverse_successors(
            false, [&](BasicBlock *successor) noexcept {
                ++record.successor_count;
                if (auto iter = _block_ids.find(successor);
                    iter != _block_ids.end()) {
                    record.successors.emplace_back(iter->second);
                } else {
                    ++record.external_successor_count;
                }
            });
    }
}

bool RayQueryHandlerGraph::contains(
    const BasicBlock *block) const noexcept {
    return _block_ids.contains(block);
}

size_t RayQueryHandlerGraph::block_id(
    const BasicBlock *block) const noexcept {
    if (auto iter = _block_ids.find(block); iter != _block_ids.end()) {
        return iter->second;
    }
    return std::numeric_limits<size_t>::max();
}

const RayQueryHandlerGraph::Block &RayQueryHandlerGraph::block(
    size_t id) const noexcept {
    LUISA_DEBUG_ASSERT(id < _blocks.size(), "Invalid handler block ID.");
    return _blocks[id];
}

const RayQueryHandlerGraph::InstructionLocation *
RayQueryHandlerGraph::instruction_location(
    const Instruction *instruction) const noexcept {
    if (auto iter = _instruction_locations.find(instruction);
        iter != _instruction_locations.end()) {
        return &iter->second;
    }
    return nullptr;
}

size_t RayQueryHandlerGraph::block_instruction_count(
    const BasicBlock *block) const noexcept {
    auto id = block_id(block);
    return id < _blocks.size() ? _blocks[id].instruction_count : 0u;
}

bool RayQueryHandlerGraph::verify(
    const luisa::unordered_set<BasicBlock *> &blocks) const noexcept {
    if (!_valid || blocks.size() != _blocks.size()) { return false; }
    auto instruction_count = size_t{0u};
    for (auto *source : blocks) {
        auto id = block_id(source);
        if (id >= _blocks.size()) { return false; }
        auto &record = _blocks[id];
        if (record.source != source) { return false; }
        auto ordinal = size_t{0u};
        for (auto *instruction : source->instructions()) {
            auto *location = instruction_location(instruction);
            if (location == nullptr || location->block_id != id ||
                location->ordinal != ordinal++) {
                return false;
            }
        }
        if (ordinal != record.instruction_count) { return false; }
        instruction_count += ordinal;
        auto successor_count = size_t{0u};
        auto external_successor_count = size_t{0u};
        luisa::vector<size_t> successors;
        source->traverse_successors(
            false, [&](BasicBlock *successor) noexcept {
                ++successor_count;
                auto successor_id = block_id(successor);
                if (successor_id < _blocks.size()) {
                    successors.emplace_back(successor_id);
                } else {
                    ++external_successor_count;
                }
            });
        if (successor_count != record.successor_count ||
            external_successor_count != record.external_successor_count ||
            successors != record.successors) {
            return false;
        }
    }
    return instruction_count == _instruction_count;
}

}// namespace luisa::compute::xir::detail
