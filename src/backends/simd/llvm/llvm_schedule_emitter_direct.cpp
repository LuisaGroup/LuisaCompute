#include "llvm_schedule_emitter.h"

#include <algorithm>

namespace luisa::compute::simd::detail {

bool ScheduleEmitter::_can_emit_direct_control_flow() const noexcept {
    std::vector<bool> covered_convergences(
        _source.convergence_points().size(), false);
    for (auto &&block : _source.blocks()) {
        auto supported = std::visit(
            [&](const auto &control) noexcept {
                using T = std::decay_t<decltype(control)>;
                if constexpr (std::is_same_v<
                                  T, schedule::SplitTerminator>) {
                    auto *condition = _source.value(control.condition);
                    if (condition != nullptr &&
                        schedule::is_uniform(condition->value_class)) {
                        return true;
                    }
                    auto diamond =
                        _find_predicated_memory_diamond(block);
                    if (diamond && control.convergence) {
                        covered_convergences[control.convergence->value] =
                            true;
                    }
                    return diamond.has_value();
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::SwitchTerminator>) {
                    auto *selector = _source.value(control.selector);
                    return selector != nullptr &&
                           schedule::is_uniform(
                               selector->value_class);
                } else {
                    return std::is_same_v<
                               T, schedule::BranchTerminator> ||
                           std::is_same_v<
                               T, schedule::ReturnTerminator> ||
                           std::is_same_v<
                               T, schedule::UnreachableTerminator>;
                }
            },
            block.terminator);
        if (!supported) { return false; }
    }
    return std::all_of(
        covered_convergences.begin(), covered_convergences.end(),
        [](bool covered) noexcept { return covered; });
}

[[nodiscard]] std::optional<std::vector<schedule::BlockId>>
ScheduleEmitter::_find_predicated_acyclic_order() const noexcept {
    static constexpr auto max_block_count = size_t{16u};
    static constexpr auto max_instruction_count = size_t{32u};
    if (!_is_surface_filter_handler_entry() ||
        (_width != 4u && _width != 8u && _width != 16u) ||
        _source.blocks().empty() ||
        _source.blocks().size() > max_block_count ||
        !_source.loops().empty()) {
        return std::nullopt;
    }
    auto instruction_count = size_t{0u};
    for (auto &&block : _source.blocks()) {
        instruction_count += block.instructions.size();
    }
    if (instruction_count > max_instruction_count) {
        return std::nullopt;
    }

    auto block_count = _source.blocks().size();
    if (_source.entry().value >= block_count) {
        return std::nullopt;
    }
    std::vector<std::vector<schedule::BlockId>> successors(block_count);
    std::vector<size_t> indegrees(block_count, 0u);
    std::vector<uint8_t> seen_blocks(block_count, uint8_t{0u});
    auto add_edge = [&](schedule::BlockId source,
                        const schedule::ControlEdge &edge) noexcept {
        if (source.value >= block_count ||
            edge.target.value >= block_count || edge.loop_back) {
            return false;
        }
        successors[source.value].emplace_back(edge.target);
        indegrees[edge.target.value]++;
        return true;
    };
    for (auto &&block : _source.blocks()) {
        if (block.id.value >= block_count ||
            seen_blocks[block.id.value] != 0u) {
            return std::nullopt;
        }
        seen_blocks[block.id.value] = 1u;
        auto supported = std::visit(
            [&](const auto &terminator) noexcept {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    return add_edge(block.id, terminator.edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SplitTerminator>) {
                    return add_edge(block.id, terminator.true_edge) &&
                           add_edge(block.id, terminator.false_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SwitchTerminator>) {
                    for (auto &&item : terminator.cases) {
                        if (!add_edge(block.id, item.edge)) {
                            return false;
                        }
                    }
                    return add_edge(block.id, terminator.default_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator>) {
                    auto *point = _source.convergence(
                        terminator.convergence);
                    if (point == nullptr) { return false; }
                    schedule::ControlEdge edge{point->target};
                    return add_edge(block.id, edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::ReturnTerminator>) {
                    return !terminator.value.has_value();
                } else {
                    return false;
                }
            },
            block.terminator);
        if (!supported) { return std::nullopt; }
    }

    // Only the entry may start without a predecessor. Kahn's algorithm below
    // then simultaneously proves that every block is reachable and that the
    // handler has no cycle hidden outside Schedule's natural-loop table.
    if (indegrees[_source.entry().value] != 0u) {
        return std::nullopt;
    }
    for (auto index = size_t{0u}; index < block_count; index++) {
        if (indegrees[index] == 0u &&
            index != _source.entry().value) {
            return std::nullopt;
        }
    }
    std::vector<schedule::BlockId> ready;
    ready.reserve(block_count);
    ready.emplace_back(_source.entry());
    std::vector<schedule::BlockId> order;
    order.reserve(block_count);
    for (auto cursor = size_t{0u}; cursor < ready.size(); cursor++) {
        auto block = ready[cursor];
        order.emplace_back(block);
        for (auto target : successors[block.value]) {
            if (--indegrees[target.value] == 0u) {
                ready.emplace_back(target);
            }
        }
    }
    if (order.size() != block_count) { return std::nullopt; }
    return order;
}

void ScheduleEmitter::_build_direct(::llvm::Value *initial_mask) {
    auto &context = _module.getContext();
    std::vector<bool> inlined_blocks(
        _source.blocks().size(), false);
    for (auto &&block : _source.blocks()) {
        if (auto diamond = _find_predicated_memory_diamond(block)) {
            inlined_blocks[diamond->true_block->id.value] = true;
            inlined_blocks[diamond->false_block->id.value] = true;
        }
    }
    std::vector<::llvm::BasicBlock *> blocks(
        _source.blocks().size(), nullptr);
    for (auto &&block : _source.blocks()) {
        if (!inlined_blocks[block.id.value]) {
            blocks[block.id.value] = ::llvm::BasicBlock::Create(
                context,
                "direct.schedule." + std::to_string(block.id.value),
                _entry);
        }
    }
    auto *activate = ::llvm::BasicBlock::Create(
        context, "direct.activate", _entry);
    auto *inactive = ::llvm::BasicBlock::Create(
        context, "direct.inactive", _entry);
    auto *active = _builder.CreateOrReduce(initial_mask);
    _builder.CreateCondBr(active, activate, inactive);

    _builder.SetInsertPoint(inactive);
    _builder.CreateRetVoid();

    _builder.SetInsertPoint(activate);
    _active_mask = _width == 1u ?
                       static_cast<::llvm::Value *>(
                           ::llvm::ConstantVector::getSplat(
                               ::llvm::ElementCount::getFixed(1u),
                               _builder.getTrue())) :
                       initial_mask;
    // With a statically row-aligned packet, any nonempty dispatch-edge mask
    // is a prefix and therefore contains lane zero. Direct control never
    // changes that mask, so keep the seed in a register constant instead of
    // repeating first-active extraction in hot memory loops.
    auto lane_zero_is_active =
        _entry_abi == ScheduleEntryABI::packet &&
        (_width == 1u ||
         (_static_block_size[0u] >= _width &&
          _static_block_size[0u] % _width == 0u));
    _seed_lane = lane_zero_is_active ?
                     static_cast<::llvm::Value *>(
                         _builder.getInt32(0u)) :
                     _safe_first_lane(_active_mask);
    _builder.CreateBr(blocks[_source.entry().value]);
    for (auto &&block : _source.blocks()) {
        if (inlined_blocks[block.id.value]) { continue; }
        _builder.SetInsertPoint(blocks[block.id.value]);
        _locals.clear();
        for (auto &&instruction : block.instructions) {
            _emit_instruction(instruction);
            if (_failed()) { return; }
        }
        _emit_direct_terminator(block, blocks);
        if (_failed()) { return; }
    }
}

void ScheduleEmitter::_build_predicated_acyclic(
    ::llvm::Value *initial_mask) {
    auto &context = _module.getContext();
    auto *mask_type = _layout.mask_type();
    auto *zero_mask = ::llvm::Constant::getNullValue(mask_type);
    auto block_count = _source.blocks().size();

    std::vector<::llvm::AllocaInst *> incoming_masks;
    incoming_masks.reserve(block_count);
    for (auto index = size_t{0u}; index < block_count; index++) {
        auto *mask = _builder.CreateAlloca(
            mask_type, nullptr,
            "predicated.acyclic.mask." + std::to_string(index));
        _builder.CreateStore(zero_mask, mask);
        incoming_masks.emplace_back(mask);
    }
    _builder.CreateStore(
        initial_mask, incoming_masks[_source.entry().value]);

    std::vector<::llvm::BasicBlock *> checks;
    std::vector<::llvm::BasicBlock *> bodies;
    checks.reserve(block_count);
    bodies.reserve(block_count);
    for (auto block : _predicated_acyclic_order) {
        checks.emplace_back(::llvm::BasicBlock::Create(
            context,
            "predicated.acyclic.check." +
                std::to_string(block.value),
            _entry));
        bodies.emplace_back(::llvm::BasicBlock::Create(
            context,
            "predicated.acyclic.body." +
                std::to_string(block.value),
            _entry));
    }
    auto *exit = ::llvm::BasicBlock::Create(
        context, "predicated.acyclic.exit", _entry);
    _builder.CreateBr(checks.front());

    auto merge_edge = [&](const schedule::ControlEdge &edge,
                          ::llvm::Value *mask) {
        _apply_assignments(edge.assignments, mask);
        if (_failed()) { return; }
        auto *slot = incoming_masks[edge.target.value];
        auto *previous = _builder.CreateLoad(mask_type, slot);
        _builder.CreateStore(_builder.CreateOr(previous, mask), slot);
    };

    for (auto order_index = size_t{0u};
         order_index < block_count; order_index++) {
        auto block_id = _predicated_acyclic_order[order_index];
        auto *block = _source.block(block_id);
        auto *next = order_index + 1u < block_count ?
                         checks[order_index + 1u] :
                         exit;

        _builder.SetInsertPoint(checks[order_index]);
        auto *block_mask = _builder.CreateLoad(
            mask_type, incoming_masks[block_id.value],
            "predicated.acyclic.active.mask");
        _builder.CreateCondBr(
            _builder.CreateOrReduce(block_mask),
            bodies[order_index], next);

        _builder.SetInsertPoint(bodies[order_index]);
        _active_mask = block_mask;
        _seed_lane = _safe_first_lane(_active_mask);
        _locals.clear();
        for (auto &&instruction : block->instructions) {
            _emit_instruction(instruction, nullptr, _active_mask);
            if (_failed()) { return; }
        }
        std::visit(
            [&](const auto &terminator) {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    merge_edge(terminator.edge, _active_mask);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SplitTerminator>) {
                    auto *condition_value = _source.value(
                        terminator.condition);
                    if (condition_value == nullptr) {
                        _fail("predicated acyclic split has an invalid condition");
                        return;
                    }
                    auto *condition = _as_lane_vector(
                        _load_value(terminator.condition),
                        *condition_value);
                    if (condition == nullptr) { return; }
                    auto *true_mask = _builder.CreateAnd(
                        _active_mask, condition);
                    auto *false_mask = _builder.CreateAnd(
                        _active_mask, _builder.CreateNot(condition));
                    merge_edge(terminator.true_edge, true_mask);
                    if (!_failed()) {
                        merge_edge(terminator.false_edge, false_mask);
                    }
                } else if constexpr (std::is_same_v<
                                         T, schedule::SwitchTerminator>) {
                    auto *selector_value = _source.value(
                        terminator.selector);
                    if (selector_value == nullptr) {
                        _fail("predicated acyclic switch has an invalid selector");
                        return;
                    }
                    auto *selector = _as_lane_vector(
                        _load_value(terminator.selector),
                        *selector_value);
                    if (selector == nullptr) { return; }
                    auto *remaining_mask = _active_mask;
                    auto *element_type = ::llvm::cast<
                        ::llvm::IntegerType>(
                        ::llvm::cast<::llvm::FixedVectorType>(
                            selector->getType())
                            ->getElementType());
                    for (auto &&item : terminator.cases) {
                        auto *label = _builder.CreateVectorSplat(
                            _width,
                            ::llvm::ConstantInt::get(
                                element_type, item.value));
                        auto *matches = _builder.CreateICmpEQ(
                            selector, label);
                        // Match Schedule's ordered switch semantics: a lane
                        // consumed by an earlier label cannot enter a later
                        // duplicate label.
                        auto *case_mask = _builder.CreateAnd(
                            remaining_mask, matches);
                        merge_edge(item.edge, case_mask);
                        if (_failed()) { return; }
                        remaining_mask = _builder.CreateAnd(
                            remaining_mask,
                            _builder.CreateNot(matches));
                    }
                    merge_edge(
                        terminator.default_edge, remaining_mask);
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator>) {
                    auto *point = _source.convergence(
                        terminator.convergence);
                    schedule::ControlEdge edge{point->target};
                    edge.assignments = terminator.assignments;
                    merge_edge(edge, _active_mask);
                } else if constexpr (!std::is_same_v<
                                         T, schedule::ReturnTerminator>) {
                    _fail("unsupported terminator reached predicated acyclic LLVM emission");
                }
            },
            block->terminator);
        if (_failed()) { return; }
        _builder.CreateBr(next);
    }

    _builder.SetInsertPoint(exit);
    _builder.CreateRetVoid();
}

}// namespace luisa::compute::simd::detail
