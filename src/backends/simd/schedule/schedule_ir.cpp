#include "schedule_ir.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <type_traits>

namespace luisa::compute::simd::schedule {

Function::Function(std::string name, uint32_t logical_warp_width) noexcept
    : _name{std::move(name)}, _logical_warp_width{logical_warp_width} {}

ValueId Function::add_value(ValueClass value_class, const Type *type,
                            ValueOrigin origin,
                            std::optional<BlockId> defining_block,
                            std::string name, ValueMetadata metadata) {
    auto id = ValueId{static_cast<uint32_t>(_values.size())};
    _values.emplace_back(Value{
        .id = id,
        .value_class = value_class,
        .origin = origin,
        .type = type,
        .defining_block = defining_block,
        .name = std::move(name),
        .metadata = std::move(metadata),
    });
    return id;
}

BlockId Function::add_block(std::string name) {
    auto id = BlockId{static_cast<uint32_t>(_blocks.size())};
    _blocks.emplace_back(BasicBlock{
        .id = id,
        .name = std::move(name),
    });
    return id;
}

ConvergenceId Function::add_convergence(
    BlockId target, std::optional<ConvergenceId> parent) {
    auto id = ConvergenceId{
        static_cast<uint32_t>(_convergence_points.size())};
    _convergence_points.emplace_back(ConvergencePoint{
        .id = id,
        .target = target,
        .parent = parent,
    });
    return id;
}

LoopId Function::add_loop(BlockId header, std::vector<BlockId> blocks,
                          std::vector<BlockId> exits,
                          std::optional<LoopId> parent) {
    auto id = LoopId{static_cast<uint32_t>(_loops.size())};
    _loops.emplace_back(Loop{
        .id = id,
        .header = header,
        .blocks = std::move(blocks),
        .exits = std::move(exits),
        .parent = parent,
    });
    return id;
}

Value *Function::value(ValueId id) noexcept {
    return id.valid() && id.value < _values.size() ?
               &_values[id.value] :
               nullptr;
}

const Value *Function::value(ValueId id) const noexcept {
    return const_cast<Function *>(this)->value(id);
}

BasicBlock *Function::block(BlockId id) noexcept {
    return id.valid() && id.value < _blocks.size() ?
               &_blocks[id.value] :
               nullptr;
}

const BasicBlock *Function::block(BlockId id) const noexcept {
    return const_cast<Function *>(this)->block(id);
}

ConvergencePoint *Function::convergence(ConvergenceId id) noexcept {
    return id.valid() && id.value < _convergence_points.size() ?
               &_convergence_points[id.value] :
               nullptr;
}

const ConvergencePoint *Function::convergence(
    ConvergenceId id) const noexcept {
    return const_cast<Function *>(this)->convergence(id);
}

Loop *Function::loop(LoopId id) noexcept {
    return id.valid() && id.value < _loops.size() ?
               &_loops[id.value] :
               nullptr;
}

const Loop *Function::loop(LoopId id) const noexcept {
    return const_cast<Function *>(this)->loop(id);
}

const char *to_string(ValueClass value) noexcept {
    switch (value) {
        case ValueClass::warp_uniform: return "warp_uniform";
        case ValueClass::cohort_uniform: return "cohort_uniform";
        case ValueClass::varying: return "varying";
        case ValueClass::mask: return "mask";
        case ValueClass::token: return "token";
    }
    return "unknown";
}

const char *to_string(ValueOrigin value) noexcept {
    switch (value) {
        case ValueOrigin::parameter: return "param";
        case ValueOrigin::constant: return "constant";
        case ValueOrigin::special_register: return "sreg";
        case ValueOrigin::scheduler_builtin: return "builtin";
        case ValueOrigin::instruction: return "instruction";
        case ValueOrigin::state_slot: return "state";
    }
    return "unknown";
}

const char *to_string(RegionStrategy value) noexcept {
    switch (value) {
        case RegionStrategy::uniform_control: return "uniform";
        case RegionStrategy::predicated: return "predicated";
        case RegionStrategy::cohort: return "cohort";
    }
    return "unknown";
}

const char *to_string(Opcode value) noexcept {
    switch (value) {
        case Opcode::constant: return "constant";
        case Opcode::special_register: return "special_register";
        case Opcode::arithmetic: return "arithmetic";
        case Opcode::cast: return "cast";
        case Opcode::call: return "call";
        case Opcode::alloca: return "alloca";
        case Opcode::load: return "load";
        case Opcode::store: return "store";
        case Opcode::gep: return "gep";
        case Opcode::atomic: return "atomic";
        case Opcode::resource_query: return "resource_query";
        case Opcode::resource_read: return "resource_read";
        case Opcode::resource_write: return "resource_write";
        case Opcode::warp_collective: return "warp_collective";
        case Opcode::edge_copy: return "edge_copy";
        case Opcode::print: return "print";
        case Opcode::assert_: return "assert";
        case Opcode::clock: return "clock";
        case Opcode::opaque: return "opaque";
    }
    return "unknown";
}

namespace {

template<typename Id>
[[nodiscard]] bool id_in_range(Id id, size_t size) noexcept {
    return id.valid() && id.value < size;
}

void add_error(VerificationResult &result, std::string message,
               std::optional<BlockId> block = std::nullopt) {
    result.errors.emplace_back(VerificationError{
        .message = std::move(message),
        .block = block,
    });
}

}// namespace

VerificationResult verify(const Function &function) {
    VerificationResult result;
    auto valid_block = [&](BlockId id) noexcept {
        return id_in_range(id, function.blocks().size());
    };
    auto valid_value = [&](ValueId id) noexcept {
        return id_in_range(id, function.values().size());
    };
    auto valid_convergence = [&](ConvergenceId id) noexcept {
        return id_in_range(id, function.convergence_points().size());
    };
    auto valid_loop = [&](LoopId id) noexcept {
        return id_in_range(id, function.loops().size());
    };

    auto width = function.logical_warp_width();
    if (width > 128u) {
        add_error(result, "logical warp width must be symbolic or at most 128");
    }
    if (function.blocks().empty()) {
        add_error(result, "function has no basic blocks");
        return result;
    }
    if (!valid_block(function.entry())) {
        add_error(result, "function entry block is invalid");
    }

    for (auto i = size_t{0u}; i < function.values().size(); i++) {
        auto &&value = function.values()[i];
        if (value.id.value != i) {
            add_error(result, "value table contains a non-canonical ID");
        }
        if (value.defining_block && !valid_block(*value.defining_block)) {
            add_error(result, "value has an invalid defining block");
        }
        if (value.origin != ValueOrigin::instruction &&
            value.origin != ValueOrigin::state_slot &&
            value.defining_block) {
            add_error(result,
                      "external value must not have a defining block");
        }
        if ((value.origin == ValueOrigin::instruction ||
             value.origin == ValueOrigin::state_slot) &&
            !value.defining_block) {
            add_error(result,
                      "instruction or state value must have a defining block");
        }
        auto metadata_matches_origin = std::visit(
            [&](const auto &metadata) noexcept {
                using M = std::decay_t<decltype(metadata)>;
                if constexpr (std::is_same_v<M, std::monostate>) {
                    // Instructions and state slots have no external source
                    // payload. Dependency-light hand-authored fixtures may
                    // also omit metadata when they omit a concrete type.
                    return value.origin == ValueOrigin::instruction ||
                           value.origin == ValueOrigin::state_slot ||
                           value.type == nullptr;
                } else if constexpr (
                    std::is_same_v<M, ParameterValueMetadata>) {
                    return value.origin == ValueOrigin::parameter;
                } else if constexpr (
                    std::is_same_v<M, ConstantValueMetadata>) {
                    return value.origin == ValueOrigin::constant &&
                           !metadata.bytes.empty();
                } else if constexpr (
                    std::is_same_v<M, SpecialRegisterValueMetadata>) {
                    return value.origin == ValueOrigin::special_register;
                } else {
                    return value.origin == ValueOrigin::scheduler_builtin;
                }
            },
            value.metadata);
        if (!metadata_matches_origin) {
            add_error(result,
                      "value source metadata does not match its origin");
        }
    }

    std::vector<std::vector<ValueId>> state_slots_by_block(
        function.blocks().size());
    for (auto &&value : function.values()) {
        if (value.origin == ValueOrigin::state_slot &&
            value.defining_block && valid_block(*value.defining_block)) {
            state_slots_by_block[value.defining_block->value]
                .emplace_back(value.id);
        }
    }

    auto check_parent_cycles = [&](size_t count, auto &&parent_at,
                                   const char *message) {
        std::vector<uint8_t> color(count, 0u);
        std::vector<size_t> path;
        for (auto root = size_t{0u}; root < count; root++) {
            if (color[root] != 0u) { continue; }
            path.clear();
            auto current = std::optional<size_t>{root};
            while (current && color[*current] == 0u) {
                color[*current] = 1u;
                path.emplace_back(*current);
                current = parent_at(*current);
            }
            if (current && color[*current] == 1u) {
                add_error(result, message);
            }
            for (auto node : path) { color[node] = 2u; }
        }
    };

    for (auto i = size_t{0u}; i < function.convergence_points().size(); i++) {
        auto &&point = function.convergence_points()[i];
        if (point.id.value != i) {
            add_error(result,
                      "convergence table contains a non-canonical ID");
        }
        if (!valid_block(point.target)) {
            add_error(result, "convergence point has an invalid target");
        }
        if (point.parent && !valid_convergence(*point.parent)) {
            add_error(result, "convergence point has an invalid parent");
        }
    }
    check_parent_cycles(
        function.convergence_points().size(),
        [&](size_t index) -> std::optional<size_t> {
            auto parent = function.convergence_points()[index].parent;
            return parent && valid_convergence(*parent) ?
                       std::optional<size_t>{parent->value} :
                       std::nullopt;
        },
        "convergence parent graph contains a cycle");

    for (auto i = size_t{0u}; i < function.loops().size(); i++) {
        auto &&loop = function.loops()[i];
        if (loop.id.value != i) {
            add_error(result, "loop table contains a non-canonical ID");
        }
        if (!valid_block(loop.header)) {
            add_error(result, "loop has an invalid header");
        }
        auto contains_header = false;
        for (auto block : loop.blocks) {
            if (!valid_block(block)) {
                add_error(result, "loop has an invalid member block");
            }
            contains_header |= block == loop.header;
        }
        if (!contains_header) {
            add_error(result, "loop membership does not contain its header");
        }
        for (auto exit : loop.exits) {
            if (!valid_block(exit)) {
                add_error(result, "loop has an invalid exit");
            }
            if (loop.header == exit) {
                add_error(result,
                          "loop header and exits must be different");
            }
        }
        if (loop.parent && !valid_loop(*loop.parent)) {
            add_error(result, "loop has an invalid parent");
        }
    }
    check_parent_cycles(
        function.loops().size(),
        [&](size_t index) -> std::optional<size_t> {
            auto parent = function.loops()[index].parent;
            return parent && valid_loop(*parent) ?
                       std::optional<size_t>{parent->value} :
                       std::nullopt;
        },
        "loop parent graph contains a cycle");

    std::vector<uint32_t> definition_count(function.values().size(), 0u);
    std::vector<std::vector<BlockId>> successors(function.blocks().size());
    std::vector<uint32_t> assignment_marks(function.values().size(), 0u);
    std::vector<uint32_t> convergence_marks(
        function.convergence_points().size(), 0u);
    auto assignment_epoch = uint32_t{0u};
    auto convergence_epoch = uint32_t{0u};
    auto next_epoch = [](std::vector<uint32_t> &marks,
                         uint32_t &epoch) noexcept {
        if (++epoch == 0u) {
            std::fill(marks.begin(), marks.end(), 0u);
            epoch = 1u;
        }
        return epoch;
    };
    for (auto block_index = size_t{0u};
         block_index < function.blocks().size(); block_index++) {
        auto &&block = function.blocks()[block_index];
        if (block.id.value != block_index) {
            add_error(result, "block table contains a non-canonical ID",
                      block.id);
        }
        for (auto &&instruction : block.instructions) {
            for (auto operand : instruction.operands) {
                if (!valid_value(operand)) {
                    add_error(result, "instruction has an invalid operand",
                              block.id);
                }
            }
            if (instruction.result) {
                if (!valid_value(*instruction.result)) {
                    add_error(result, "instruction has an invalid result",
                              block.id);
                } else {
                    auto &count = definition_count[instruction.result->value];
                    ++count;
                    auto *value = function.value(*instruction.result);
                    if (!value->defining_block ||
                        *value->defining_block != block.id) {
                        add_error(
                            result,
                            "instruction result disagrees with its defining block",
                            block.id);
                    }
                }
            }
            if (instruction.opcode == Opcode::warp_collective &&
                !instruction.collective_id) {
                add_error(result,
                          "warp collective is missing a dynamic instance ID",
                          block.id);
            }
            if (instruction.opcode != Opcode::warp_collective &&
                instruction.collective_id) {
                add_error(result,
                          "non-collective instruction has a collective ID",
                          block.id);
            }
            if (instruction.opcode == Opcode::warp_collective) {
                if (!instruction.participant_mask ||
                    !valid_value(*instruction.participant_mask) ||
                    function.value(*instruction.participant_mask)
                            ->value_class != ValueClass::mask) {
                    add_error(result,
                              "warp collective has an invalid participant mask",
                              block.id);
                }
            } else if (instruction.participant_mask) {
                add_error(result,
                          "non-collective instruction has a participant mask",
                          block.id);
            }
            if (instruction.cohort_uniform_operand_index) {
                if (instruction.opcode != Opcode::resource_read) {
                    add_error(
                        result,
                        "non-resource-read instruction has a cohort-uniform operand annotation",
                        block.id);
                } else if (*instruction.cohort_uniform_operand_index >=
                           instruction.operands.size()) {
                    add_error(
                        result,
                        "cohort-uniform operand annotation is out of range",
                        block.id);
                }
            }
            if (instruction.lane_consecutive_operand_index) {
                if (instruction.opcode != Opcode::resource_read &&
                    instruction.opcode != Opcode::resource_write) {
                    add_error(
                        result,
                        "non-resource instruction has a lane-consecutive operand annotation",
                        block.id);
                } else if (*instruction.lane_consecutive_operand_index >=
                           instruction.operands.size()) {
                    add_error(
                        result,
                        "lane-consecutive operand annotation is out of range",
                        block.id);
                } else if (instruction.cohort_uniform_operand_index ==
                           instruction.lane_consecutive_operand_index) {
                    add_error(
                        result,
                        "one operand cannot be both cohort-uniform and lane-consecutive",
                        block.id);
                }
            }
        }

        auto check_assignments = [&](const auto &assignments,
                                     std::optional<BlockId> target) {
            auto epoch = next_epoch(assignment_marks, assignment_epoch);
            for (auto &&assignment : assignments) {
                if (!valid_value(assignment.destination) ||
                    !valid_value(assignment.source)) {
                    add_error(result, "edge assignment has an invalid value",
                              block.id);
                    continue;
                }
                auto *destination = function.value(assignment.destination);
                if (destination->origin != ValueOrigin::state_slot) {
                    add_error(
                        result,
                        "edge assignment destination is not a state slot",
                        block.id);
                    continue;
                }
                auto &mark = assignment_marks[assignment.destination.value];
                if (mark == epoch) {
                    add_error(result,
                              "edge assigns a state slot more than once",
                              block.id);
                } else {
                    mark = epoch;
                }
                if (target && destination->defining_block != target) {
                    add_error(result,
                              "edge assignment targets a state slot owned by another block",
                              block.id);
                }
            }
            if (!target || !valid_block(*target)) { return; }
            for (auto state_slot : state_slots_by_block[target->value]) {
                if (assignment_marks[state_slot.value] != epoch) {
                    add_error(result,
                              "incoming edge is missing a state-slot assignment",
                              block.id);
                }
            }
        };
        auto add_edge = [&](const ControlEdge &edge) {
            auto target = edge.target;
            if (!valid_block(target)) {
                add_error(result, "terminator has an invalid target", block.id);
            } else {
                successors[block_index].emplace_back(target);
            }
            auto join_epoch =
                next_epoch(convergence_marks, convergence_epoch);
            for (auto join_index = size_t{0u};
                 join_index < edge.joins.size(); join_index++) {
                auto join = edge.joins[join_index];
                if (!valid_convergence(join)) {
                    add_error(result, "edge has an invalid convergence join",
                              block.id);
                    continue;
                }
                auto &mark = convergence_marks[join.value];
                if (mark == join_epoch) {
                    add_error(result,
                              "edge repeats a convergence join",
                              block.id);
                } else {
                    mark = join_epoch;
                }
                if (function.convergence(join)->target != target) {
                    add_error(result,
                              "edge convergence join does not target its gate",
                              block.id);
                }
                if (join_index != 0u) {
                    auto inner = edge.joins[join_index - 1u];
                    if (valid_convergence(inner) &&
                        function.convergence(inner)->parent != join) {
                        add_error(result,
                                  "edge convergence joins are not ordered inner-to-outer",
                                  block.id);
                    }
                }
            }
            if (edge.loop_back) {
                if (!valid_loop(*edge.loop_back)) {
                    add_error(result, "edge has an invalid loop back-edge",
                              block.id);
                } else if (function.loop(*edge.loop_back)->header != target) {
                    add_error(result,
                              "loop back-edge does not target its header",
                              block.id);
                }
            }
            check_assignments(edge.assignments, target);
        };
        std::visit(
            [&](const auto &terminator) {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<T, std::monostate>) {
                    add_error(result, "basic block has no terminator", block.id);
                } else if constexpr (std::is_same_v<T, BranchTerminator>) {
                    add_edge(terminator.edge);
                } else if constexpr (std::is_same_v<T, SplitTerminator>) {
                    if (!valid_value(terminator.condition)) {
                        add_error(result, "split has an invalid condition",
                                  block.id);
                    }
                    add_edge(terminator.true_edge);
                    add_edge(terminator.false_edge);
                    if (terminator.convergence &&
                        !valid_convergence(*terminator.convergence)) {
                        add_error(result, "split has invalid convergence",
                                  block.id);
                    }
                } else if constexpr (std::is_same_v<T, SwitchTerminator>) {
                    if (!valid_value(terminator.selector)) {
                        add_error(result, "switch has an invalid selector",
                                  block.id);
                    }
                    for (auto &&item : terminator.cases) {
                        add_edge(item.edge);
                    }
                    add_edge(terminator.default_edge);
                    if (terminator.convergence &&
                        !valid_convergence(*terminator.convergence)) {
                        add_error(result, "switch has invalid convergence",
                                  block.id);
                    }
                } else if constexpr (std::is_same_v<T, JoinTerminator>) {
                    if (!valid_convergence(terminator.convergence)) {
                        add_error(result, "join has invalid convergence",
                                  block.id);
                    } else {
                        successors[block_index].emplace_back(
                            function.convergence(terminator.convergence)
                                ->target);
                    }
                    check_assignments(
                        terminator.assignments,
                        valid_convergence(terminator.convergence) ?
                            std::optional{function.convergence(
                                              terminator.convergence)
                                              ->target} :
                            std::nullopt);
                } else if constexpr (std::is_same_v<T, LoopBackTerminator>) {
                    if (!valid_loop(terminator.loop)) {
                        add_error(result, "loop back-edge has invalid loop",
                                  block.id);
                    } else {
                        successors[block_index].emplace_back(
                            function.loop(terminator.loop)->header);
                    }
                    check_assignments(
                        terminator.assignments,
                        valid_loop(terminator.loop) ?
                            std::optional{function.loop(terminator.loop)
                                              ->header} :
                            std::nullopt);
                } else if constexpr (
                    std::is_same_v<T, BlockBarrierTerminator>) {
                    add_edge(terminator.resume_edge);
                } else if constexpr (std::is_same_v<T, ReturnTerminator>) {
                    if (terminator.value && !valid_value(*terminator.value)) {
                        add_error(result, "return has an invalid value",
                                  block.id);
                    }
                }
            },
            block.terminator);
    }

    for (auto i = size_t{0u}; i < function.values().size(); i++) {
        auto &&value = function.values()[i];
        if (value.origin == ValueOrigin::instruction &&
            definition_count[i] != 1u) {
            add_error(result,
                      "instruction value must have exactly one definition");
        }
    }

    if (valid_block(function.entry())) {
        std::vector<bool> reached(function.blocks().size(), false);
        std::vector<BlockId> work{function.entry()};
        while (!work.empty()) {
            auto current = work.back();
            work.pop_back();
            if (reached[current.value]) { continue; }
            reached[current.value] = true;
            for (auto successor : successors[current.value]) {
                if (!reached[successor.value]) { work.emplace_back(successor); }
            }
        }
        for (auto i = size_t{0u}; i < reached.size(); i++) {
            if (!reached[i]) {
                add_error(result, "basic block is unreachable",
                          BlockId{static_cast<uint32_t>(i)});
            }
        }
    }
    return result;
}

std::string to_string(const Function &function) {
    std::ostringstream out;
    out << "schedule.func @" << function.name() << " warp=";
    if (function.logical_warp_width() == 0u) {
        out << "symbolic";
    } else {
        out << function.logical_warp_width();
    }
    out << " entry=bb" << function.entry().value << " {\n";
    for (auto &&value : function.values()) {
        out << "  value %" << value.id.value << ' '
            << to_string(value.value_class);
        if (!value.name.empty()) { out << " @" << value.name; }
        out << " origin=" << to_string(value.origin);
        if (value.defining_block) {
            out << (value.origin == ValueOrigin::state_slot ?
                        " home=bb" :
                        " def=bb")
                << value.defining_block->value;
        }
        std::visit(
            [&](const auto &metadata) {
                using M = std::decay_t<decltype(metadata)>;
                if constexpr (std::is_same_v<M, ParameterValueMetadata>) {
                    out << " arg=" << metadata.index
                        << " arg_tag=" << metadata.argument_tag;
                } else if constexpr (
                    std::is_same_v<M, ConstantValueMetadata>) {
                    out << " bytes=0x" << std::hex << std::setfill('0');
                    for (auto byte : metadata.bytes) {
                        out << std::setw(2)
                            << std::to_integer<uint32_t>(byte);
                    }
                    out << std::dec << std::setfill(' ');
                } else if constexpr (
                    std::is_same_v<M, SpecialRegisterValueMetadata>) {
                    out << " sreg_tag=" << metadata.tag;
                } else if constexpr (
                    std::is_same_v<M, SchedulerBuiltinValueMetadata>) {
                    out << " builtin="
                        << static_cast<uint32_t>(metadata.builtin);
                }
            },
            value.metadata);
        out << "\n";
    }
    for (auto &&point : function.convergence_points()) {
        out << "  convergence c" << point.id.value << " -> bb"
            << point.target.value;
        if (point.parent) { out << " parent=c" << point.parent->value; }
        out << "\n";
    }
    for (auto &&loop : function.loops()) {
        out << "  loop l" << loop.id.value << " header=bb"
            << loop.header.value << " blocks=[";
        for (auto i = size_t{0u}; i < loop.blocks.size(); i++) {
            if (i != 0u) { out << ", "; }
            out << "bb" << loop.blocks[i].value;
        }
        out << "] exits=[";
        for (auto i = size_t{0u}; i < loop.exits.size(); i++) {
            if (i != 0u) { out << ", "; }
            out << "bb" << loop.exits[i].value;
        }
        out << ']';
        if (loop.parent) { out << " parent=l" << loop.parent->value; }
        out << "\n";
    }
    for (auto &&block : function.blocks()) {
        out << "bb" << block.id.value;
        if (!block.name.empty()) { out << " @" << block.name; }
        out << " [" << to_string(block.strategy) << "]:\n";
        for (auto &&instruction : block.instructions) {
            out << "  ";
            if (instruction.result) {
                out << '%' << instruction.result->value << " = ";
            }
            out << to_string(instruction.opcode);
            if (instruction.source_op) {
                out << " op=" << *instruction.source_op;
            }
            if (instruction.collective_id) {
                out << " collective=" << *instruction.collective_id;
            }
            if (instruction.participant_mask) {
                out << " mask=%" << instruction.participant_mask->value;
            }
            if (instruction.cohort_uniform_operand_index) {
                out << " cohort_uniform_operand="
                    << *instruction.cohort_uniform_operand_index;
            }
            if (instruction.lane_consecutive_operand_index) {
                out << " lane_consecutive_operand="
                    << *instruction.lane_consecutive_operand_index;
            }
            for (auto operand : instruction.operands) {
                out << " %" << operand.value;
            }
            out << "\n";
        }
        auto print_assignments = [&](const auto &assignments) {
            if (assignments.empty()) { return; }
            out << " assign={";
            for (auto i = size_t{0u}; i < assignments.size(); i++) {
                if (i != 0u) { out << ", "; }
                out << '%' << assignments[i].destination.value << " <- %"
                    << assignments[i].source.value;
            }
            out << '}';
        };
        auto print_edge = [&](const ControlEdge &edge) {
            out << "bb" << edge.target.value;
            if (!edge.joins.empty()) {
                out << " joins=[";
                for (auto i = size_t{0u}; i < edge.joins.size(); i++) {
                    if (i != 0u) { out << ", "; }
                    out << 'c' << edge.joins[i].value;
                }
                out << ']';
            }
            if (edge.loop_back) {
                out << " loop_back=l" << edge.loop_back->value;
            }
            print_assignments(edge.assignments);
        };
        std::visit(
            [&](const auto &terminator) {
                using T = std::decay_t<decltype(terminator)>;
                out << "  ";
                if constexpr (std::is_same_v<T, std::monostate>) {
                    out << "<missing terminator>";
                } else if constexpr (std::is_same_v<T, BranchTerminator>) {
                    out << "branch ";
                    print_edge(terminator.edge);
                } else if constexpr (std::is_same_v<T, SplitTerminator>) {
                    out << "split %" << terminator.condition.value << ' ';
                    print_edge(terminator.true_edge);
                    out << ' ';
                    print_edge(terminator.false_edge);
                    if (terminator.convergence) {
                        out << " convergence=c"
                            << terminator.convergence->value;
                    }
                } else if constexpr (std::is_same_v<T, SwitchTerminator>) {
                    out << "switch %" << terminator.selector.value;
                    for (auto &&item : terminator.cases) {
                        out << " [" << item.value << ": ";
                        print_edge(item.edge);
                        out << ']';
                    }
                    out << " default=";
                    print_edge(terminator.default_edge);
                    if (terminator.convergence) {
                        out << " convergence=c"
                            << terminator.convergence->value;
                    }
                } else if constexpr (std::is_same_v<T, JoinTerminator>) {
                    out << "join c" << terminator.convergence.value;
                    print_assignments(terminator.assignments);
                } else if constexpr (std::is_same_v<T, LoopBackTerminator>) {
                    out << "loop_back l" << terminator.loop.value;
                    print_assignments(terminator.assignments);
                } else if constexpr (
                    std::is_same_v<T, BlockBarrierTerminator>) {
                    out << "block_barrier " << terminator.barrier_id
                        << " resume=";
                    print_edge(terminator.resume_edge);
                } else if constexpr (std::is_same_v<T, ReturnTerminator>) {
                    out << "return";
                    if (terminator.value) {
                        out << " %" << terminator.value->value;
                    }
                } else if constexpr (
                    std::is_same_v<T, UnreachableTerminator>) {
                    out << "unreachable";
                }
                out << "\n";
            },
            block.terminator);
    }
    out << "}\n";
    return out.str();
}

}// namespace luisa::compute::simd::schedule
