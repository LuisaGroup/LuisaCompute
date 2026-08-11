#include "xir_to_schedule.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <luisa/ast/type.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/thread_group.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/post_dom_tree.h>
#include <luisa/xir/special_register.h>

#include "../../../xir/passes/natural_loop.h"
#include "warp_uniformity.h"

namespace luisa::compute::simd::schedule {

const char *to_string(XIRToScheduleDiagnosticCode code) noexcept {
    switch (code) {
        case XIRToScheduleDiagnosticCode::invalid_source:
            return "invalid_source";
        case XIRToScheduleDiagnosticCode::invalid_warp_width:
            return "invalid_warp_width";
        case XIRToScheduleDiagnosticCode::malformed_cfg:
            return "malformed_cfg";
        case XIRToScheduleDiagnosticCode::structured_control_flow:
            return "structured_control_flow";
        case XIRToScheduleDiagnosticCode::irreducible_control_flow:
            return "irreducible_control_flow";
        case XIRToScheduleDiagnosticCode::unsupported_instruction:
            return "unsupported_instruction";
        case XIRToScheduleDiagnosticCode::unsupported_value:
            return "unsupported_value";
        case XIRToScheduleDiagnosticCode::invalid_phi:
            return "invalid_phi";
        case XIRToScheduleDiagnosticCode::schedule_verification:
            return "schedule_verification";
    }
    return "unknown";
}

namespace {

struct CFGEdge {
    const xir::BasicBlock *source{nullptr};
    const xir::BasicBlock *target{nullptr};
    friend bool operator==(CFGEdge, CFGEdge) noexcept = default;
};

struct CFGEdgeHash {
    [[nodiscard]] size_t operator()(CFGEdge edge) const noexcept {
        auto source = std::hash<const void *>{}(edge.source);
        auto target = std::hash<const void *>{}(edge.target);
        return source ^ (target + 0x9e3779b97f4a7c15ull +
                         (source << 6u) + (source >> 2u));
    }
};

[[nodiscard]] std::string copy_string(luisa::string_view value) {
    return {value.data(), value.size()};
}

[[nodiscard]] std::string value_name(
    const xir::Value *value, std::string fallback = {}) {
    if (value != nullptr) {
        if (auto name = value->name(); name && !name->empty()) {
            return copy_string(*name);
        }
    }
    return fallback;
}

[[nodiscard]] bool is_structured_control(
    xir::DerivedInstructionTag tag) noexcept {
    using Tag = xir::DerivedInstructionTag;
    switch (tag) {
        case Tag::IF:
        case Tag::SWITCH:
        case Tag::LOOP:
        case Tag::SIMPLE_LOOP:
        case Tag::BREAK:
        case Tag::CONTINUE: return true;
        default: return false;
    }
}

[[nodiscard]] bool is_supported_terminator(
    xir::DerivedInstructionTag tag) noexcept {
    using Tag = xir::DerivedInstructionTag;
    switch (tag) {
        case Tag::BRANCH:
        case Tag::CONDITIONAL_BRANCH:
        case Tag::INDEXED_BRANCH:
        case Tag::RETURN:
        case Tag::UNREACHABLE: return true;
        default: return false;
    }
}

[[nodiscard]] bool is_supported_non_terminator(
    const xir::Instruction *instruction) noexcept {
    using Tag = xir::DerivedInstructionTag;
    switch (instruction->derived_instruction_tag()) {
        case Tag::PHI:
        case Tag::ALLOCA:
        case Tag::LOAD:
        case Tag::STORE:
        case Tag::GEP:
        case Tag::ATOMIC:
        case Tag::ARITHMETIC:
        case Tag::RESOURCE_QUERY:
        case Tag::RESOURCE_READ:
        case Tag::RESOURCE_WRITE:
        case Tag::CAST:
        case Tag::PRINT:
        case Tag::CLOCK:
        case Tag::DEBUG_BREAK:
        case Tag::ASSERT:
        case Tag::ASSUME:
        case Tag::OUTLINE: return true;
        case Tag::THREAD_GROUP: {
            auto op = static_cast<const xir::ThreadGroupInst *>(instruction)
                          ->op();
            return op != xir::ThreadGroupOp::SYNCHRONIZE_BLOCK;
        }
        default: return false;
    }
}

[[nodiscard]] bool is_collective(xir::ThreadGroupOp op) noexcept {
    return op != xir::ThreadGroupOp::SHADER_EXECUTION_REORDER &&
           op != xir::ThreadGroupOp::SYNCHRONIZE_BLOCK;
}

class LoweringContext {

private:
    struct LoopRecord {
        const xir::NaturalLoop *source{nullptr};
        LoopId id{};
        size_t size{0u};
    };

    struct ConvergenceRecord {
        xir::BasicBlock *branch{nullptr};
        xir::BasicBlock *target{nullptr};
        ConvergenceId id{};
    };

    const xir::Function *_source{nullptr};
    xir::FunctionDefinition *_definition{nullptr};
    XIRToScheduleOptions _options{};
    XIRToScheduleResult _result{};
    std::optional<Function> _function{};
    std::vector<xir::BasicBlock *> _blocks{};
    std::unordered_map<const xir::BasicBlock *, size_t> _block_indices{};
    std::unordered_map<const xir::BasicBlock *, BlockId> _block_ids{};
    std::unordered_map<const xir::Value *, ValueId> _value_ids{};
    std::unordered_set<CFGEdge, CFGEdgeHash> _cfg_edges{};
    std::vector<std::vector<xir::BasicBlock *>> _predecessors{};
    std::vector<std::vector<xir::BasicBlock *>> _successors{};
    std::unordered_map<CFGEdge, LoopId, CFGEdgeHash> _loop_back_ids{};
    std::unordered_map<const xir::BasicBlock *, ConvergenceId>
        _convergence_by_branch{};
    std::vector<LoopRecord> _loops{};
    std::vector<ConvergenceRecord> _convergences{};
    WarpUniformityAnalysis _uniformity{};
    const xir::DomTree *_dom_tree{nullptr};
    uint32_t _next_collective_id{0u};
    uint32_t _next_external_value_id{0u};
    std::optional<ValueId> _active_mask{};

private:
    void _diagnose(
        XIRToScheduleDiagnosticCode code, std::string message,
        const xir::BasicBlock *block = nullptr,
        const xir::Instruction *instruction = nullptr) {
        _result.diagnostics.emplace_back(XIRToScheduleDiagnostic{
            .code = code,
            .message = std::move(message),
            .block = block,
            .instruction = instruction,
        });
    }

    [[nodiscard]] bool _failed() const noexcept {
        return !_result.diagnostics.empty();
    }

    [[nodiscard]] static std::string _instruction_name(
        const xir::Instruction *instruction) {
        return copy_string(
            xir::to_string(instruction->derived_instruction_tag()));
    }

    void _collect_and_preflight_cfg() {
        if (_source == nullptr || _source->definition() == nullptr ||
            _source->definition()->body_block() == nullptr) {
            _diagnose(
                XIRToScheduleDiagnosticCode::invalid_source,
                "XIR source must be a function definition with a body");
            return;
        }
        if (_options.logical_warp_width > 128u) {
            _diagnose(
                XIRToScheduleDiagnosticCode::invalid_warp_width,
                "logical warp width must be symbolic or at most 128");
            return;
        }
        _definition = const_cast<xir::FunctionDefinition *>(
            _source->definition());
        _definition->traverse_basic_blocks(
            [&](xir::BasicBlock *block) noexcept {
                _blocks.emplace_back(block);
            });
        if (_blocks.empty()) {
            _diagnose(XIRToScheduleDiagnosticCode::malformed_cfg,
                      "XIR function has no reachable basic blocks");
            return;
        }

        _block_indices.reserve(_blocks.size());
        for (auto i = size_t{0u}; i < _blocks.size(); i++) {
            _block_indices.emplace(_blocks[i], i);
        }

        _predecessors.resize(_blocks.size());
        _successors.resize(_blocks.size());
        auto edge_operand_count = size_t{0u};
        for (auto *source : _blocks) {
            if (source->is_terminated()) {
                edge_operand_count += source->terminator()->operand_count();
            }
        }
        _cfg_edges.reserve(edge_operand_count);
        for (auto *source : _blocks) {
            if (!source->is_terminated()) { continue; }
            for (auto *operand_use : source->terminator()->operand_uses()) {
                auto *operand = operand_use->value();
                if (operand == nullptr || !operand->isa<xir::BasicBlock>()) {
                    continue;
                }
                auto *target = static_cast<xir::BasicBlock *>(operand);
                auto target_iter = _block_indices.find(target);
                if (target_iter != _block_indices.end() &&
                    _cfg_edges.emplace(CFGEdge{source, target}).second) {
                    _predecessors[target_iter->second].emplace_back(source);
                    _successors[_block_indices.at(source)]
                        .emplace_back(target);
                }
            }
        }

        for (auto *block : _blocks) {
            if (!block->is_terminated() || block->terminator() == nullptr) {
                _diagnose(XIRToScheduleDiagnosticCode::malformed_cfg,
                          "reachable XIR block is not terminated", block);
                continue;
            }
            auto *terminator = block->terminator();
            auto terminator_tag = terminator->derived_instruction_tag();
            if (is_structured_control(terminator_tag)) {
                _diagnose(
                    XIRToScheduleDiagnosticCode::structured_control_flow,
                    "structured XIR terminator '" +
                        _instruction_name(terminator) +
                        "' must be lowered with destructure_cfg before SIMD scheduling",
                    block, terminator);
            } else if (!is_supported_terminator(terminator_tag)) {
                auto message = "XIR terminator '" +
                               _instruction_name(terminator) +
                               "' is not supported by the Phase 1 SIMD lowering";
                if (terminator_tag ==
                    xir::DerivedInstructionTag::RASTER_DISCARD) {
                    message += " (raster execution is not a CPU backend target)";
                }
                _diagnose(
                    XIRToScheduleDiagnosticCode::unsupported_instruction,
                    std::move(message), block, terminator);
            }

            for (auto *instruction : block->instructions()) {
                if (instruction == terminator) { continue; }
                if (instruction->is_terminator()) {
                    _diagnose(
                        XIRToScheduleDiagnosticCode::malformed_cfg,
                        "non-final terminator appears in an XIR basic block",
                        block, instruction);
                    continue;
                }
                if (is_supported_non_terminator(instruction)) { continue; }
                auto tag = instruction->derived_instruction_tag();
                auto message = "XIR instruction '" +
                               _instruction_name(instruction) +
                               "' is not supported by the Phase 1 SIMD lowering";
                if (tag == xir::DerivedInstructionTag::CALL) {
                    message += "; run the XIR inline pass first";
                } else if (tag ==
                               xir::DerivedInstructionTag::THREAD_GROUP) {
                    message +=
                        "; block barriers require cooperative-block scheduling";
                } else if (is_structured_control(tag)) {
                    message += "; run destructure_cfg first";
                }
                _diagnose(
                    is_structured_control(tag) ?
                        XIRToScheduleDiagnosticCode::structured_control_flow :
                        XIRToScheduleDiagnosticCode::unsupported_instruction,
                    std::move(message), block, instruction);
            }
        }
    }

    [[nodiscard]] bool _cfg_is_reducible(
        const luisa::vector<xir::NaturalLoop> &natural_loops) const {
        std::unordered_set<CFGEdge, CFGEdgeHash> removable_back_edges;
        auto back_edge_count = size_t{0u};
        for (auto &&loop : natural_loops) {
            back_edge_count += loop.back_edges.size();
        }
        removable_back_edges.reserve(back_edge_count);
        for (auto &&loop : natural_loops) {
            for (auto &&[source, target] : loop.back_edges) {
                removable_back_edges.emplace(CFGEdge{source, target});
            }
        }

        std::vector<uint32_t> indegree(_blocks.size(), 0u);
        for (auto edge : _cfg_edges) {
            if (!removable_back_edges.contains(edge)) {
                ++indegree[_block_indices.at(edge.target)];
            }
        }
        std::queue<size_t> ready;
        for (auto i = size_t{0u}; i < indegree.size(); i++) {
            if (indegree[i] == 0u) { ready.emplace(i); }
        }
        auto visited = size_t{0u};
        while (!ready.empty()) {
            auto index = ready.front();
            ready.pop();
            ++visited;
            auto *source = _blocks[index];
            for (auto *target : _successors[index]) {
                if (removable_back_edges.contains(
                        CFGEdge{source, target})) {
                    continue;
                }
                auto target_index = _block_indices.at(target);
                auto &degree = indegree[target_index];
                if (--degree == 0u) {
                    ready.emplace(target_index);
                }
            }
        }
        return visited == _blocks.size();
    }

    void _create_function_and_blocks() {
        auto function_name = value_name(
            _source,
            _source->isa<xir::KernelFunction>() ? "kernel" : "callable");
        _function.emplace(std::move(function_name),
                          _options.logical_warp_width);
        _block_ids.reserve(_blocks.size());
        for (auto *source_block : _blocks) {
            auto name = value_name(source_block);
            auto id = _function->add_block(std::move(name));
            _block_ids.emplace(source_block, id);
        }
        _function->set_entry(
            _block_ids.at(_definition->body_block()));
    }

    void _create_loops(
        const luisa::vector<xir::NaturalLoop> &natural_loops) {
        _loops.reserve(natural_loops.size());
        auto back_edge_count = size_t{0u};
        for (auto &&loop : natural_loops) {
            back_edge_count += loop.back_edges.size();
        }
        _loop_back_ids.reserve(back_edge_count);
        for (auto &&source_loop : natural_loops) {
            std::vector<BlockId> blocks;
            blocks.reserve(source_loop.body_blocks.size() + 1u);
            blocks.emplace_back(_block_ids.at(source_loop.header));
            for (auto *block : source_loop.body_blocks) {
                blocks.emplace_back(_block_ids.at(block));
            }
            std::sort(blocks.begin(), blocks.end(),
                      [](BlockId lhs, BlockId rhs) noexcept {
                          return lhs.value < rhs.value;
                      });
            blocks.erase(std::unique(blocks.begin(), blocks.end()),
                         blocks.end());
            std::vector<BlockId> exits;
            exits.reserve(source_loop.exit_blocks.size());
            for (auto *exit : source_loop.exit_blocks) {
                if (auto iter = _block_ids.find(exit);
                    iter != _block_ids.end()) {
                    exits.emplace_back(iter->second);
                }
            }
            std::sort(exits.begin(), exits.end(),
                      [](BlockId lhs, BlockId rhs) noexcept {
                          return lhs.value < rhs.value;
                      });
            exits.erase(std::unique(exits.begin(), exits.end()), exits.end());
            auto id = _function->add_loop(
                _block_ids.at(source_loop.header), std::move(blocks),
                std::move(exits));
            _loops.emplace_back(LoopRecord{
                .source = &source_loop,
                .id = id,
                .size = source_loop.body_blocks.size() + 1u,
            });
            for (auto &&[source, target] : source_loop.back_edges) {
                _loop_back_ids.emplace(CFGEdge{source, target}, id);
            }
        }

        // NaturalLoop already materializes loop membership. Index that output
        // once instead of testing every loop pair (which is quadratic for
        // deeply nested generated kernels).
        std::vector<std::vector<size_t>> containing_loops(_blocks.size());
        for (auto loop_index = size_t{0u};
             loop_index < _loops.size(); loop_index++) {
            auto add_membership = [&](const xir::BasicBlock *block) {
                containing_loops[_block_indices.at(block)]
                    .emplace_back(loop_index);
            };
            add_membership(_loops[loop_index].source->header);
            for (auto *block : _loops[loop_index].source->body_blocks) {
                add_membership(block);
            }
        }
        for (auto i = size_t{0u}; i < _loops.size(); i++) {
            std::optional<size_t> parent_index;
            auto header_index =
                _block_indices.at(_loops[i].source->header);
            for (auto candidate : containing_loops[header_index]) {
                if (_loops[candidate].size <= _loops[i].size) { continue; }
                if (!parent_index ||
                    _loops[candidate].size <
                        _loops[*parent_index].size) {
                    parent_index = candidate;
                }
            }
            if (parent_index) {
                _function->loop(_loops[i].id)->parent =
                    _loops[*parent_index].id;
            }
        }
    }

    [[nodiscard]] const xir::Value *_branch_selector(
        const xir::Instruction *terminator) const noexcept {
        using Tag = xir::DerivedInstructionTag;
        switch (terminator->derived_instruction_tag()) {
            case Tag::CONDITIONAL_BRANCH:
                return static_cast<const xir::ConditionalBranchInst *>(
                           terminator)
                    ->condition();
            case Tag::INDEXED_BRANCH:
                return static_cast<const xir::IndexedBranchInst *>(terminator)
                    ->value();
            default: return nullptr;
        }
    }

    void _create_convergences(const xir::PostDomTree &post_dom_tree) {
        _convergences.reserve(_blocks.size());
        _convergence_by_branch.reserve(_blocks.size());
        for (auto *block : _blocks) {
            auto *terminator = block->terminator();
            auto *selector = _branch_selector(terminator);
            if (selector == nullptr || _uniformity.is_uniform(selector)) {
                continue;
            }
            auto *target = post_dom_tree.immediate_post_dominator(block);
            if (target == nullptr || target == block ||
                !_block_ids.contains(target)) {
                continue;
            }
            auto id = _function->add_convergence(_block_ids.at(target));
            _convergences.emplace_back(ConvergenceRecord{
                .branch = block,
                .target = target,
                .id = id,
            });
            _convergence_by_branch.emplace(block, id);
        }

        std::vector<std::vector<ConvergenceId>> closes_at(_blocks.size());
        for (auto &&record : _convergences) {
            // A target outside the split's dominator subtree cannot dominate
            // any block in that subtree, so the branch-frame exit closes it.
            if (_dom_tree->dominates(record.branch, record.target)) {
                closes_at[_block_indices.at(record.target)]
                    .emplace_back(record.id);
            }
        }

        // Walk the dominator tree once. The active convergence stack is the
        // static analogue of the runtime token stack. Closing at merge
        // subtrees before opening a new split makes parent discovery O(B + C)
        // and diagnoses non-nested convergence instead of hiding it behind an
        // ancestor-chain scan.
        struct Frame {
            const xir::DomTreeNode *node{nullptr};
            size_t next_child{0u};
            size_t active_after_close{0u};
            std::vector<ConvergenceId> closed{};
            bool entered{false};
        };
        std::vector<ConvergenceId> active;
        std::vector<uint8_t> closing(_convergences.size(), 0u);
        std::vector<Frame> stack;
        if (_dom_tree->root() != nullptr) {
            stack.emplace_back(Frame{.node = _dom_tree->root()});
        }
        while (!stack.empty()) {
            auto &frame = stack.back();
            if (!frame.entered) {
                frame.entered = true;
                auto *block = frame.node->block();
                auto block_index = _block_indices.at(block);
                for (auto convergence : closes_at[block_index]) {
                    closing[convergence.value] = 1u;
                }
                auto closed_count = size_t{0u};
                while (!active.empty() &&
                       closing[active.back().value] != 0u) {
                    frame.closed.emplace_back(active.back());
                    active.pop_back();
                    ++closed_count;
                }
                if (closed_count != closes_at[block_index].size()) {
                    _diagnose(
                        XIRToScheduleDiagnosticCode::irreducible_control_flow,
                        "convergence scopes are not properly nested; normalize the CFG before SIMD scheduling",
                        block, block->terminator());
                }
                for (auto convergence : closes_at[block_index]) {
                    closing[convergence.value] = 0u;
                }
                frame.active_after_close = active.size();
                if (auto iter = _convergence_by_branch.find(block);
                    iter != _convergence_by_branch.end()) {
                    auto convergence = iter->second;
                    if (!active.empty()) {
                        auto parent = active.back();
                        _function->convergence(convergence)->parent = parent;
                    }
                    active.emplace_back(convergence);
                }
            }
            auto children = frame.node->children();
            if (frame.next_child < children.size()) {
                auto *child = children[frame.next_child++];
                stack.emplace_back(Frame{.node = child});
            } else {
                active.resize(frame.active_after_close);
                for (auto iter = frame.closed.rbegin();
                     iter != frame.closed.rend(); ++iter) {
                    active.emplace_back(*iter);
                }
                stack.pop_back();
            }
        }
    }

    void _create_values() {
        auto value_count = size_t{0u};
        for (auto *argument : _source->arguments()) {
            static_cast<void>(argument);
            ++value_count;
        }
        for (auto *block : _blocks) {
            for (auto *instruction : block->instructions()) {
                value_count += !instruction->is_terminator() &&
                                       instruction->type() != nullptr ?
                                   1u :
                                   0u;
            }
        }
        _value_ids.reserve(value_count);
        auto argument_index = uint32_t{0u};
        for (auto *argument : _source->arguments()) {
            auto index = argument_index++;
            auto id = _function->add_value(
                _uniformity.classify(argument), argument->type(),
                ValueOrigin::parameter, std::nullopt,
                value_name(argument,
                           "arg" + std::to_string(index)),
                ParameterValueMetadata{
                    .index = index,
                    .argument_tag = static_cast<uint32_t>(
                        argument->derived_argument_tag()),
                });
            _value_ids.emplace(argument, id);
        }

        for (auto *source_block : _blocks) {
            auto block = _block_ids.at(source_block);
            for (auto *instruction : source_block->instructions()) {
                if (instruction->is_terminator() ||
                    instruction->type() == nullptr) {
                    continue;
                }
                auto is_phi = instruction->isa<xir::PhiInst>();
                auto id = _function->add_value(
                    _uniformity.classify(instruction), instruction->type(),
                    is_phi ? ValueOrigin::state_slot :
                             ValueOrigin::instruction,
                    block, value_name(instruction));
                _value_ids.emplace(instruction, id);
            }
        }
    }

    [[nodiscard]] std::optional<ValueId> _map_value(
        const xir::Value *value, const xir::BasicBlock *block,
        const xir::Instruction *instruction) {
        if (value == nullptr) {
            _diagnose(XIRToScheduleDiagnosticCode::unsupported_value,
                      "XIR operand is null", block, instruction);
            return std::nullopt;
        }
        if (auto iter = _value_ids.find(value); iter != _value_ids.end()) {
            return iter->second;
        }
        using Tag = xir::DerivedValueTag;
        switch (value->derived_value_tag()) {
            case Tag::CONSTANT: {
                auto *constant = static_cast<const xir::Constant *>(value);
                if (value->type() == nullptr || constant->data() == nullptr ||
                    value->type()->size() == 0u) {
                    _diagnose(
                        XIRToScheduleDiagnosticCode::unsupported_value,
                        "XIR constant has no code-generatable payload", block,
                        instruction);
                    return std::nullopt;
                }
                std::vector<std::byte> bytes(value->type()->size());
                std::memcpy(bytes.data(), constant->data(), bytes.size());
                auto id = _function->add_value(
                    ValueClass::warp_uniform, value->type(),
                    ValueOrigin::constant,
                    std::nullopt,
                    value_name(value, "const" + std::to_string(
                                                 _next_external_value_id++)),
                    ConstantValueMetadata{.bytes = std::move(bytes)});
                _value_ids.emplace(value, id);
                return id;
            }
            case Tag::SPECIAL_REGISTER: {
                auto *special =
                    static_cast<const xir::SpecialRegister *>(value);
                auto id = _function->add_value(
                    _uniformity.classify(value), value->type(),
                    ValueOrigin::special_register, std::nullopt,
                    value_name(
                        value,
                        copy_string(xir::to_string(
                            special->derived_special_register_tag()))),
                    SpecialRegisterValueMetadata{
                        .tag = static_cast<uint32_t>(
                            special->derived_special_register_tag()),
                    });
                _value_ids.emplace(value, id);
                return id;
            }
            case Tag::ARGUMENT:
                _diagnose(
                    XIRToScheduleDiagnosticCode::unsupported_value,
                    "operand references an argument from another function",
                    block, instruction);
                return std::nullopt;
            case Tag::INSTRUCTION:
                _diagnose(
                    XIRToScheduleDiagnosticCode::unsupported_value,
                    "operand references an unavailable or void XIR instruction",
                    block, instruction);
                return std::nullopt;
            case Tag::UNDEFINED:
                _diagnose(
                    XIRToScheduleDiagnosticCode::unsupported_value,
                    "undefined XIR values must be eliminated before SIMD scheduling",
                    block, instruction);
                return std::nullopt;
            case Tag::FUNCTION:
            case Tag::BASIC_BLOCK:
                _diagnose(
                    XIRToScheduleDiagnosticCode::unsupported_value,
                    "control or function value used as a Schedule IR data operand",
                    block, instruction);
                return std::nullopt;
        }
        return std::nullopt;
    }

    [[nodiscard]] Opcode _opcode(
        const xir::Instruction *instruction) const noexcept {
        using Tag = xir::DerivedInstructionTag;
        switch (instruction->derived_instruction_tag()) {
            case Tag::ALLOCA: return Opcode::alloca;
            case Tag::LOAD: return Opcode::load;
            case Tag::STORE: return Opcode::store;
            case Tag::GEP: return Opcode::gep;
            case Tag::ATOMIC: return Opcode::atomic;
            case Tag::ARITHMETIC: return Opcode::arithmetic;
            case Tag::RESOURCE_QUERY: return Opcode::resource_query;
            case Tag::RESOURCE_READ: return Opcode::resource_read;
            case Tag::RESOURCE_WRITE: return Opcode::resource_write;
            case Tag::CAST: return Opcode::cast;
            case Tag::PRINT: return Opcode::print;
            case Tag::CLOCK: return Opcode::clock;
            case Tag::ASSERT: return Opcode::assert_;
            case Tag::THREAD_GROUP: {
                auto op = static_cast<const xir::ThreadGroupInst *>(instruction)
                              ->op();
                return is_collective(op) ? Opcode::warp_collective :
                                           Opcode::opaque;
            }
            case Tag::DEBUG_BREAK:
            case Tag::ASSUME:
            case Tag::OUTLINE: return Opcode::opaque;
            default: return Opcode::opaque;
        }
    }

    [[nodiscard]] std::optional<uint32_t> _source_op(
        const xir::Instruction *instruction) const noexcept {
        using Tag = xir::DerivedInstructionTag;
        switch (instruction->derived_instruction_tag()) {
            case Tag::ALLOCA:
                return static_cast<uint32_t>(
                    static_cast<const xir::AllocaInst *>(instruction)->op());
            case Tag::ATOMIC:
                return static_cast<uint32_t>(
                    static_cast<const xir::AtomicInst *>(instruction)->op());
            case Tag::ARITHMETIC:
                return static_cast<uint32_t>(
                    static_cast<const xir::ArithmeticInst *>(instruction)
                        ->op());
            case Tag::RESOURCE_QUERY:
                return static_cast<uint32_t>(
                    static_cast<const xir::ResourceQueryInst *>(instruction)
                        ->op());
            case Tag::RESOURCE_READ:
                return static_cast<uint32_t>(
                    static_cast<const xir::ResourceReadInst *>(instruction)
                        ->op());
            case Tag::RESOURCE_WRITE:
                return static_cast<uint32_t>(
                    static_cast<const xir::ResourceWriteInst *>(instruction)
                        ->op());
            case Tag::CAST:
                return static_cast<uint32_t>(
                    static_cast<const xir::CastInst *>(instruction)->op());
            case Tag::THREAD_GROUP:
                return static_cast<uint32_t>(
                    static_cast<const xir::ThreadGroupInst *>(instruction)
                        ->op());
            case Tag::DEBUG_BREAK:
            case Tag::ASSUME:
            case Tag::OUTLINE:
                return static_cast<uint32_t>(
                    instruction->derived_instruction_tag());
            default: return std::nullopt;
        }
    }

    [[nodiscard]] ValueId _get_active_mask() {
        if (!_active_mask) {
            _active_mask = _function->add_value(
                ValueClass::mask, nullptr,
                ValueOrigin::scheduler_builtin, std::nullopt,
                "active_mask",
                SchedulerBuiltinValueMetadata{
                    .builtin = SchedulerBuiltin::active_mask,
                });
        }
        return *_active_mask;
    }

    void _emit_instruction(
        const xir::Instruction *source_instruction,
        BasicBlock &target_block) {
        if (source_instruction->isa<xir::PhiInst>()) { return; }
        Instruction instruction{
            .opcode = _opcode(source_instruction),
            .source_op = _source_op(source_instruction),
        };
        if (source_instruction->type() != nullptr) {
            if (auto iter = _value_ids.find(source_instruction);
                iter != _value_ids.end()) {
                instruction.result = iter->second;
            } else {
                _diagnose(
                    XIRToScheduleDiagnosticCode::unsupported_value,
                    "result-producing XIR instruction has no Schedule IR value",
                    source_instruction->parent_block(), source_instruction);
            }
        }
        for (auto *operand_use : source_instruction->operand_uses()) {
            if (auto operand = _map_value(
                    operand_use->value(),
                    source_instruction->parent_block(), source_instruction)) {
                instruction.operands.emplace_back(*operand);
            }
        }
        if (instruction.opcode == Opcode::warp_collective) {
            instruction.collective_id = _next_collective_id++;
            instruction.participant_mask = _get_active_mask();
        }
        target_block.instructions.emplace_back(std::move(instruction));
    }

    [[nodiscard]] ControlEdge _edge(
        const xir::BasicBlock *target, const xir::BasicBlock *source,
        const xir::Instruction *terminator) {
        if (target == nullptr) {
            _diagnose(XIRToScheduleDiagnosticCode::malformed_cfg,
                      "XIR control edge has a null target", source,
                      terminator);
            return {};
        }
        if (auto iter = _block_ids.find(target); iter != _block_ids.end()) {
            return ControlEdge{iter->second};
        }
        _diagnose(
            XIRToScheduleDiagnosticCode::malformed_cfg,
            "XIR control edge targets a block outside the reachable function",
            source, terminator);
        return {};
    }

    void _emit_terminator(
        const xir::BasicBlock *source_block, BasicBlock &target_block) {
        auto *terminator = source_block->terminator();
        using Tag = xir::DerivedInstructionTag;
        switch (terminator->derived_instruction_tag()) {
            case Tag::BRANCH: {
                auto *branch = static_cast<const xir::BranchInst *>(terminator);
                target_block.terminator = BranchTerminator{
                    _edge(branch->target_block(), source_block, terminator)};
                break;
            }
            case Tag::CONDITIONAL_BRANCH: {
                auto *branch =
                    static_cast<const xir::ConditionalBranchInst *>(terminator);
                auto condition = _map_value(
                    branch->condition(), source_block, terminator);
                target_block.strategy =
                    condition && is_uniform(
                                     _function->value(*condition)
                                         ->value_class) ?
                        RegionStrategy::uniform_control :
                        RegionStrategy::cohort;
                target_block.terminator = SplitTerminator{
                    .condition = condition.value_or(ValueId{}),
                    .true_edge = _edge(
                        branch->true_block(), source_block, terminator),
                    .false_edge = _edge(
                        branch->false_block(), source_block, terminator),
                    .convergence = _convergence_by_branch.contains(source_block) ?
                        std::optional{
                            _convergence_by_branch.at(source_block)} :
                        std::nullopt,
                };
                break;
            }
            case Tag::INDEXED_BRANCH: {
                auto *branch =
                    static_cast<const xir::IndexedBranchInst *>(terminator);
                auto selector =
                    _map_value(branch->value(), source_block, terminator);
                SwitchTerminator schedule_switch{
                    .selector = selector.value_or(ValueId{}),
                    .default_edge = _edge(
                        branch->default_block(), source_block, terminator),
                    .convergence = _convergence_by_branch.contains(source_block) ?
                        std::optional{
                            _convergence_by_branch.at(source_block)} :
                        std::nullopt,
                };
                schedule_switch.cases.reserve(branch->case_count());
                for (auto i = size_t{0u}; i < branch->case_count(); i++) {
                    schedule_switch.cases.emplace_back(SwitchCase{
                        .value = branch->case_value(i),
                        .edge = _edge(
                            branch->case_block(i), source_block, terminator),
                    });
                }
                target_block.strategy =
                    selector && is_uniform(
                                    _function->value(*selector)
                                        ->value_class) ?
                        RegionStrategy::uniform_control :
                        RegionStrategy::cohort;
                target_block.terminator = std::move(schedule_switch);
                break;
            }
            case Tag::RETURN: {
                auto *return_inst =
                    static_cast<const xir::ReturnInst *>(terminator);
                std::optional<ValueId> value;
                if (return_inst->return_value() != nullptr) {
                    value = _map_value(
                        return_inst->return_value(), source_block, terminator);
                }
                target_block.terminator = ReturnTerminator{value};
                break;
            }
            case Tag::UNREACHABLE:
                target_block.terminator = UnreachableTerminator{};
                break;
            default:
                _diagnose(
                    XIRToScheduleDiagnosticCode::unsupported_instruction,
                    "unsupported terminator reached Schedule IR emission",
                    source_block, terminator);
                break;
        }
    }

    void _emit_blocks() {
        for (auto *source_block : _blocks) {
            auto *target_block =
                _function->block(_block_ids.at(source_block));
            for (auto *instruction : source_block->instructions()) {
                if (instruction->is_terminator()) { break; }
                _emit_instruction(instruction, *target_block);
            }
            _emit_terminator(source_block, *target_block);
        }
    }

    template<typename Visit>
    static void _traverse_edges(BasicBlock &block, Visit &&visit) {
        std::visit(
            [&](auto &terminator) {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<T, BranchTerminator>) {
                    visit(terminator.edge);
                } else if constexpr (std::is_same_v<T, SplitTerminator>) {
                    visit(terminator.true_edge);
                    visit(terminator.false_edge);
                } else if constexpr (std::is_same_v<T, SwitchTerminator>) {
                    for (auto &item : terminator.cases) {
                        visit(item.edge);
                    }
                    visit(terminator.default_edge);
                } else if constexpr (
                    std::is_same_v<T, BlockBarrierTerminator>) {
                    visit(terminator.resume_edge);
                }
            },
            block.terminator);
    }

    void _annotate_loop_backs() {
        std::unordered_set<CFGEdge, CFGEdgeHash> annotated;
        annotated.reserve(_loop_back_ids.size());
        for (auto *source : _blocks) {
            _traverse_edges(
                *_function->block(_block_ids.at(source)),
                [&](ControlEdge &edge) {
                    auto key = CFGEdge{source, _blocks[edge.target.value]};
                    if (auto iter = _loop_back_ids.find(key);
                        iter != _loop_back_ids.end()) {
                        edge.loop_back = iter->second;
                        annotated.emplace(key);
                    }
                });
        }
        for (auto &&[edge_key, loop] : _loop_back_ids) {
            static_cast<void>(loop);
            if (!annotated.contains(edge_key)) {
                _diagnose(
                    XIRToScheduleDiagnosticCode::malformed_cfg,
                    "natural-loop back-edge was lost during Schedule IR projection",
                    edge_key.source,
                    edge_key.source->terminator());
            }
        }
    }

    void _annotate_convergence_joins() {
        // A gate is active for exactly the part of its branch's dominator
        // subtree before the gate target. Indexing active gates by target lets
        // each CFG edge copy only the joins it actually emits. This avoids the
        // convergence-count times predecessor-count product that arises from
        // querying every gate independently.
        struct Frame {
            const xir::DomTreeNode *node{nullptr};
            size_t next_child{0u};
            std::vector<ConvergenceId> closed{};
            std::optional<ConvergenceId> opened{};
            bool entered{false};
        };
        std::vector<std::vector<ConvergenceId>> active_by_target(
            _blocks.size());
        std::vector<Frame> stack;
        if (_dom_tree->root() != nullptr) {
            stack.emplace_back(Frame{.node = _dom_tree->root()});
        }
        while (!stack.empty()) {
            auto &frame = stack.back();
            auto *source = frame.node->block();
            auto source_index = _block_indices.at(source);
            if (!frame.entered) {
                frame.entered = true;

                // Entering a gate target closes every active gate for the
                // whole dominated subtree. Restore them when leaving so DOM
                // sibling subtrees still observe their enclosing scopes.
                frame.closed =
                    std::move(active_by_target[source_index]);
                if (auto iter = _convergence_by_branch.find(source);
                    iter != _convergence_by_branch.end()) {
                    frame.opened = iter->second;
                    auto target =
                        _function->convergence(*frame.opened)->target;
                    active_by_target[target.value]
                        .emplace_back(*frame.opened);
                }

                _traverse_edges(
                    *_function->block(_block_ids.at(source)),
                    [&](ControlEdge &edge) {
                        auto &&active =
                            active_by_target[edge.target.value];
                        edge.joins.assign(
                            active.rbegin(), active.rend());
                    });
            }
            auto children = frame.node->children();
            if (frame.next_child < children.size()) {
                auto *child = children[frame.next_child++];
                stack.emplace_back(Frame{.node = child});
            } else {
                if (frame.opened) {
                    auto target =
                        _function->convergence(*frame.opened)->target;
                    auto &active = active_by_target[target.value];
                    if (active.empty() || active.back() != *frame.opened) {
                        _diagnose(
                            XIRToScheduleDiagnosticCode::irreducible_control_flow,
                            "convergence scopes are not properly nested; normalize the CFG before SIMD scheduling",
                            source, source->terminator());
                    } else {
                        active.pop_back();
                    }
                }
                active_by_target[source_index] =
                    std::move(frame.closed);
                stack.pop_back();
            }
        }
    }

    void _lower_phi_assignments() {
        std::unordered_map<
            CFGEdge, std::vector<EdgeAssignment>, CFGEdgeHash>
            assignments_by_edge;
        assignments_by_edge.reserve(_cfg_edges.size());
        for (auto *target_block : _blocks) {
            for (auto *instruction : target_block->instructions()) {
                if (!instruction->isa<xir::PhiInst>()) { continue; }
                auto *phi = static_cast<const xir::PhiInst *>(instruction);
                auto destination_iter = _value_ids.find(phi);
                if (destination_iter == _value_ids.end()) {
                    _diagnose(
                        XIRToScheduleDiagnosticCode::invalid_phi,
                        "XIR PHI has no Schedule IR state slot", target_block,
                        phi);
                    continue;
                }
                for (auto incoming_index = size_t{0u};
                     incoming_index < phi->incoming_count();
                     incoming_index++) {
                    auto incoming = phi->incoming(incoming_index);
                    auto source = _map_value(
                        incoming.value, target_block, phi);
                    auto edge_key = CFGEdge{incoming.block, target_block};
                    if (!source || !_block_ids.contains(incoming.block) ||
                        !_cfg_edges.contains(edge_key)) {
                        _diagnose(
                            XIRToScheduleDiagnosticCode::invalid_phi,
                            "XIR PHI incoming references an unavailable predecessor or value",
                            target_block, phi);
                        continue;
                    }
                    assignments_by_edge[edge_key].emplace_back(
                        EdgeAssignment{
                            .destination = destination_iter->second,
                            .source = *source,
                        });
                }
            }
        }
        for (auto *source : _blocks) {
            _traverse_edges(
                *_function->block(_block_ids.at(source)),
                [&](ControlEdge &edge) {
                    auto key = CFGEdge{source, _blocks[edge.target.value]};
                    if (auto iter = assignments_by_edge.find(key);
                        iter != assignments_by_edge.end()) {
                        edge.assignments = iter->second;
                    }
                });
        }
    }

    void _append_verification_diagnostics() {
        auto verification = verify(*_function);
        for (auto &&error : verification.errors) {
            const xir::BasicBlock *source_block = nullptr;
            if (error.block && error.block->value < _blocks.size()) {
                source_block = _blocks[error.block->value];
            }
            _diagnose(
                XIRToScheduleDiagnosticCode::schedule_verification,
                "Schedule IR verification failed: " + error.message,
                source_block);
        }
    }

public:
    LoweringContext(const xir::Function *source,
                    XIRToScheduleOptions options) noexcept
        : _source{source}, _options{options} {}

    [[nodiscard]] XIRToScheduleResult run() {
        _collect_and_preflight_cfg();
        if (_failed()) { return std::move(_result); }

        auto dom_tree = xir::compute_dom_tree(
            const_cast<xir::Function *>(_source));
        _dom_tree = &dom_tree;
        auto natural_loops =
            xir::discover_natural_loops(_definition, dom_tree);
        if (!_cfg_is_reducible(natural_loops)) {
            _diagnose(
                XIRToScheduleDiagnosticCode::irreducible_control_flow,
                "CFG remains cyclic after removing natural back-edges; run lower_irreducible_cfg before SIMD scheduling");
            return std::move(_result);
        }
        // SIMD reconvergence is conditional on scalar-lane termination. Use
        // post-dominance over terminating executions so a natural-loop
        // back-edge does not hide the common exit where lanes with different
        // trip counts must rendezvous. The default XIR analysis remains
        // conservative about genuinely infinite executions.
        auto post_dom_tree = xir::compute_post_dom_tree(
            const_cast<xir::Function *>(_source),
            {.account_for_infinite_paths = false});
        _uniformity.analyze(_source);

        _create_function_and_blocks();
        _create_loops(natural_loops);
        _create_convergences(post_dom_tree);
        _create_values();
        _emit_blocks();
        if (_failed()) { return std::move(_result); }
        _annotate_loop_backs();
        if (_failed()) { return std::move(_result); }
        _annotate_convergence_joins();
        _lower_phi_assignments();
        if (!_failed()) { _append_verification_diagnostics(); }
        if (!_failed()) { _result.function = std::move(*_function); }
        return std::move(_result);
    }
};

}// namespace

XIRToScheduleResult lower_xir_to_schedule(
    const xir::Function *source, XIRToScheduleOptions options) {
    return LoweringContext{source, options}.run();
}

}// namespace luisa::compute::simd::schedule
