#include "llvm_schedule_emitter.h"

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
        _width == 1u ||
        (_static_block_size[0u] >= _width &&
         _static_block_size[0u] % _width == 0u);
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

}// namespace luisa::compute::simd::detail
