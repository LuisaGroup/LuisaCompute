#include "restructure_cfg_loop_boundary.h"

#include "restructure_cfg_selection_merge.h"

#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/raster_discard.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/unreachable.h>

#include <cstdint>
#include <utility>

namespace luisa::compute::xir {
namespace {

template<typename Visitor>
void traverse_structured_successors(
    BasicBlock *block, Visitor &&visit) noexcept {
    if (block == nullptr || !block->is_terminated()) { return; }
    auto *terminator = block->terminator();
    for (auto *operand_use : terminator->operand_uses()) {
        auto *operand = operand_use->value();
        if (operand != nullptr && operand->isa<BasicBlock>()) {
            visit(static_cast<BasicBlock *>(operand));
        }
    }
    if (auto *merge = terminator->control_flow_merge();
        merge != nullptr && merge->merge_block() != nullptr) {
        visit(merge->merge_block());
    }
    if (terminator->isa<LoopInst>()) {
        auto *loop = static_cast<LoopInst *>(terminator);
        if (loop->body_block() != nullptr) {
            visit(loop->body_block());
        }
        if (loop->update_block() != nullptr) {
            visit(loop->update_block());
        }
    }
}

}// namespace

// The classification is a finite forward reachability problem over BREAK,
// CONTINUE, and INVALID facts. The reverse graph is immutable and stored as
// dense CSR. Each loop solve is restricted to a successor-closed induced
// region, except for its explicitly seeded boundaries, so its least fixed
// point equals the restriction of the whole-CFG solution.
class LoopBoundaryPathDataflow::Impl {
private:
    static constexpr auto break_bit = uint8_t{1u};
    static constexpr auto continue_bit = uint8_t{2u};
    static constexpr auto invalid_bit = uint8_t{4u};

    luisa::vector<BasicBlock *> _blocks;
    luisa::unordered_map<BasicBlock *, size_t> _block_ids;
    luisa::vector<size_t> _reverse_edge_offsets;
    luisa::vector<size_t> _reverse_edge_sources;
    luisa::vector<BasicBlock *> _canonical_targets;
    luisa::vector<uint8_t> _facts;
    luisa::vector<uint8_t> _terminal;
    luisa::vector<uint32_t> _visit_epochs;
    luisa::vector<uint32_t> _fact_epochs;
    uint32_t _visit_epoch{0u};
    luisa::vector<size_t> _work;
    luisa::vector<size_t> _region;
    luisa::vector<size_t> _active;
    size_t _edge_visit_count{0u};

private:
    void _begin_visit() noexcept {
        ++_visit_epoch;
        if (_visit_epoch == 0u) {
            std::fill(
                _visit_epochs.begin(),
                _visit_epochs.end(), 0u);
            std::fill(
                _fact_epochs.begin(),
                _fact_epochs.end(), 0u);
            _visit_epoch = 1u;
        }
        _work.clear();
        _region.clear();
        _active.clear();
        _edge_visit_count = 0u;
    }

public:
    explicit Impl(FunctionDefinition *definition) noexcept {
        if (definition == nullptr) { return; }
        for (auto *block : definition->basic_blocks()) {
            auto id = _blocks.size();
            _blocks.emplace_back(block);
            _block_ids.emplace(block, id);
        }
        _canonical_targets.reserve(_blocks.size());
        for (auto *block : _blocks) {
            _canonical_targets.emplace_back(
                detail::canonical_trivial_branch_chain_target(
                    block));
        }

        luisa::vector<std::pair<size_t, size_t>> reverse_edges;
        auto add_edge = [&](size_t source,
                            BasicBlock *target) noexcept {
            if (auto iter = _block_ids.find(target);
                iter != _block_ids.end()) {
                reverse_edges.emplace_back(
                    iter->second, source);
            }
        };
        for (auto source = size_t{0u};
             source < _blocks.size(); ++source) {
            auto *block = _blocks[source];
            if (!block->is_terminated()) { continue; }
            auto *terminator = block->terminator();
            if (terminator->isa<BreakInst>() ||
                terminator->isa<ContinueInst>() ||
                terminator->isa<ReturnInst>() ||
                terminator->isa<UnreachableInst>() ||
                terminator->isa<RasterDiscardInst>()) {
                continue;
            }
            if (terminator->isa<LoopInst>() ||
                terminator->isa<SimpleLoopInst>()) {
                if (auto *merge =
                        terminator->control_flow_merge();
                    merge != nullptr) {
                    add_edge(source, merge->merge_block());
                }
                continue;
            }
            traverse_structured_successors(
                block, [&](BasicBlock *successor) noexcept {
                    add_edge(source, successor);
                });
        }
        _reverse_edge_offsets.resize(
            _blocks.size() + 1u, 0u);
        for (auto [target, source] : reverse_edges) {
            static_cast<void>(source);
            ++_reverse_edge_offsets[target + 1u];
        }
        for (auto i = size_t{1u};
             i < _reverse_edge_offsets.size(); ++i) {
            _reverse_edge_offsets[i] +=
                _reverse_edge_offsets[i - 1u];
        }
        _reverse_edge_sources.resize(reverse_edges.size());
        auto cursors = _reverse_edge_offsets;
        for (auto [target, source] : reverse_edges) {
            _reverse_edge_sources[cursors[target]++] = source;
        }
        _facts.resize(_blocks.size(), uint8_t{0u});
        _terminal.resize(_blocks.size(), uint8_t{0u});
        _visit_epochs.resize(_blocks.size(), 0u);
        _fact_epochs.resize(_blocks.size(), 0u);
        _work.reserve(_blocks.size());
        _region.reserve(_blocks.size());
        _active.reserve(_blocks.size());
    }

    void evaluate(BasicBlock *body,
                  BasicBlock *continue_target,
                  BasicBlock *loop_entry,
                  BasicBlock *merge) noexcept {
        _begin_visit();
        auto enqueue_region = [&](BasicBlock *block,
                                  bool allow_loop_entry) noexcept {
            if (block == nullptr || block == merge ||
                (!allow_loop_entry && block == loop_entry)) {
                return;
            }
            auto iter = _block_ids.find(block);
            if (iter == _block_ids.end()) { return; }
            auto id = iter->second;
            if (_visit_epochs[id] == _visit_epoch) { return; }
            _visit_epochs[id] = _visit_epoch;
            _work.emplace_back(id);
        };
        enqueue_region(body, true);
        while (!_work.empty()) {
            auto id = _work.back();
            _work.pop_back();
            _region.emplace_back(id);
            traverse_structured_successors(
                _blocks[id], [&](BasicBlock *successor) noexcept {
                    enqueue_region(successor, false);
                });
        }

        auto activate_id = [&](size_t id) noexcept {
            if (_fact_epochs[id] == _visit_epoch) { return; }
            _fact_epochs[id] = _visit_epoch;
            _facts[id] = uint8_t{0u};
            _terminal[id] = uint8_t{0u};
            _active.emplace_back(id);
        };
        for (auto id : _region) { activate_id(id); }
        auto activate_block = [&](BasicBlock *block) noexcept {
            if (auto iter = _block_ids.find(block);
                iter != _block_ids.end()) {
                activate_id(iter->second);
            }
        };
        activate_block(continue_target);
        activate_block(loop_entry);
        activate_block(merge);

        auto *canonical_merge =
            detail::canonical_trivial_branch_chain_target(merge);
        for (auto id : _active) {
            auto *block = _blocks[id];
            auto fact = uint8_t{0u};
            if (block == merge ||
                _canonical_targets[id] == canonical_merge) {
                fact = break_bit;
            } else if (block == continue_target ||
                       block == loop_entry) {
                fact = continue_bit;
            } else if (!block->is_terminated()) {
                fact = invalid_bit;
            } else {
                auto *terminator = block->terminator();
                if (terminator->isa<BreakInst>()) {
                    fact = static_cast<BreakInst *>(terminator)
                                       ->target_block() == merge ?
                               break_bit :
                               invalid_bit;
                } else if (terminator->isa<ContinueInst>()) {
                    auto *target =
                        static_cast<ContinueInst *>(terminator)
                            ->target_block();
                    fact = target == continue_target ||
                                   target == loop_entry ?
                               continue_bit :
                               invalid_bit;
                } else if (terminator->isa<ReturnInst>() ||
                           terminator->isa<UnreachableInst>() ||
                           terminator->isa<RasterDiscardInst>()) {
                    fact = invalid_bit;
                } else if (terminator->isa<LoopInst>() ||
                           terminator->isa<SimpleLoopInst>()) {
                    auto *control_flow_merge =
                        terminator->control_flow_merge();
                    if (control_flow_merge == nullptr ||
                        control_flow_merge->merge_block() ==
                            nullptr) {
                        fact = invalid_bit;
                    }
                }
            }
            if (fact != 0u) {
                _facts[id] = fact;
                _terminal[id] = uint8_t{1u};
                _work.emplace_back(id);
            }
        }
        for (auto cursor = size_t{0u};
             cursor < _work.size(); ++cursor) {
            auto node = _work[cursor];
            auto fact = _facts[node];
            for (auto edge = _reverse_edge_offsets[node];
                 edge < _reverse_edge_offsets[node + 1u];
                 ++edge) {
                ++_edge_visit_count;
                auto predecessor =
                    _reverse_edge_sources[edge];
                if (_fact_epochs[predecessor] !=
                    _visit_epoch) {
                    continue;
                }
                if (_terminal[predecessor] != 0u) { continue; }
                auto combined = static_cast<uint8_t>(
                    _facts[predecessor] | fact);
                if (combined != _facts[predecessor]) {
                    _facts[predecessor] = combined;
                    _work.emplace_back(predecessor);
                }
            }
        }
    }

    [[nodiscard]] LoopBoundaryTargetKind classify(
        BasicBlock *target) const noexcept {
        auto iter = _block_ids.find(target);
        if (iter == _block_ids.end() ||
            _fact_epochs[iter->second] != _visit_epoch) {
            return LoopBoundaryTargetKind::NONE;
        }
        auto fact = _facts[iter->second];
        if ((fact & invalid_bit) != 0u) {
            return LoopBoundaryTargetKind::NONE;
        }
        switch (fact & (break_bit | continue_bit)) {
            case break_bit:
                return LoopBoundaryTargetKind::BREAK;
            case continue_bit:
                return LoopBoundaryTargetKind::CONTINUE;
            case break_bit | continue_bit:
                return LoopBoundaryTargetKind::MIXED;
            default:
                return LoopBoundaryTargetKind::NONE;
        }
    }

    [[nodiscard]] size_t region_size() const noexcept {
        return _region.size();
    }

    [[nodiscard]] BasicBlock *region_block(
        size_t index) const noexcept {
        return _blocks[_region[index]];
    }

    [[nodiscard]] size_t active_block_count() const noexcept {
        return _active.size();
    }

    [[nodiscard]] size_t edge_visit_count() const noexcept {
        return _edge_visit_count;
    }
};

LoopBoundaryPathDataflow::LoopBoundaryPathDataflow(
    FunctionDefinition *definition) noexcept
    : _impl{luisa::make_unique<Impl>(definition)} {}

LoopBoundaryPathDataflow::~LoopBoundaryPathDataflow() noexcept = default;

LoopBoundaryPathDataflow::LoopBoundaryPathDataflow(
    LoopBoundaryPathDataflow &&) noexcept = default;

LoopBoundaryPathDataflow &
LoopBoundaryPathDataflow::operator=(
    LoopBoundaryPathDataflow &&) noexcept = default;

void LoopBoundaryPathDataflow::evaluate(
    BasicBlock *body,
    BasicBlock *continue_target,
    BasicBlock *loop_entry,
    BasicBlock *merge) noexcept {
    _impl->evaluate(
        body, continue_target, loop_entry, merge);
}

LoopBoundaryTargetKind LoopBoundaryPathDataflow::classify(
    BasicBlock *target) const noexcept {
    return _impl->classify(target);
}

size_t LoopBoundaryPathDataflow::region_size() const noexcept {
    return _impl->region_size();
}

BasicBlock *LoopBoundaryPathDataflow::region_block(
    size_t index) const noexcept {
    return _impl->region_block(index);
}

size_t LoopBoundaryPathDataflow::active_block_count() const noexcept {
    return _impl->active_block_count();
}

size_t LoopBoundaryPathDataflow::edge_visit_count() const noexcept {
    return _impl->edge_visit_count();
}

}// namespace luisa::compute::xir
