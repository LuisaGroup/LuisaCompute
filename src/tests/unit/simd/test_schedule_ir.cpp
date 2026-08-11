#include "schedule_ir.h"

#include <iostream>
#include <string_view>
#include <utility>

using namespace luisa::compute::simd::schedule;

namespace {

[[nodiscard]] bool check(bool condition, const char *expression,
                         const char *file, int line) noexcept {
    if (!condition) {
        std::cerr << file << ':' << line << ": check failed: "
                  << expression << '\n';
    }
    return condition;
}

#define CHECK(EXPR)                                                           \
    do {                                                                      \
        if (!check(static_cast<bool>(EXPR), #EXPR, __FILE__, __LINE__)) {     \
            return false;                                                     \
        }                                                                     \
    } while (false)

[[nodiscard]] bool contains_error(const VerificationResult &result,
                                  std::string_view needle) noexcept {
    for (auto &&error : result.errors) {
        if (error.message.find(needle) != std::string::npos) { return true; }
    }
    return false;
}

[[nodiscard]] Function make_diamond(uint32_t width) {
    Function function{"diamond", width};
    auto entry = function.add_block("entry");
    auto true_block = function.add_block("true");
    auto false_block = function.add_block("false");
    auto merge = function.add_block("merge");
    function.set_entry(entry);

    auto condition = function.add_value(
        ValueClass::varying, nullptr, ValueOrigin::parameter,
        std::nullopt, "condition");
    auto true_value = function.add_value(
        ValueClass::varying, nullptr, ValueOrigin::instruction,
        true_block, "true_value");
    auto false_value = function.add_value(
        ValueClass::varying, nullptr, ValueOrigin::instruction,
        false_block, "false_value");
    auto selected = function.add_value(
        ValueClass::varying, nullptr, ValueOrigin::state_slot,
        merge, "selected");
    auto convergence = function.add_convergence(merge);

    function.block(entry)->terminator = SplitTerminator{
        .condition = condition,
        .true_edge = ControlEdge{true_block},
        .false_edge = ControlEdge{false_block},
        .convergence = convergence,
    };
    function.block(true_block)->instructions.emplace_back(Instruction{
        .opcode = Opcode::constant,
        .result = true_value,
    });
    function.block(true_block)->terminator = JoinTerminator{
        .convergence = convergence,
        .assignments = {{selected, true_value}},
    };
    function.block(false_block)->instructions.emplace_back(Instruction{
        .opcode = Opcode::constant,
        .result = false_value,
    });
    function.block(false_block)->terminator = JoinTerminator{
        .convergence = convergence,
        .assignments = {{selected, false_value}},
    };
    function.block(merge)->terminator = ReturnTerminator{};
    return function;
}

[[nodiscard]] bool test_valid_diamond() {
    auto function = make_diamond(8u);
    auto result = verify(function);
    CHECK(result.succeeded());
    auto text = to_string(function);
    CHECK(text.find("schedule.func @diamond warp=8 entry=bb0") !=
          std::string::npos);
    CHECK(text.find("split %0 bb1 bb2 convergence=c0") !=
          std::string::npos);
    CHECK(text.find("join c0") != std::string::npos);
    CHECK(text.find("assign={%3 <- %1}") != std::string::npos);
    CHECK(text.find("convergence c0 -> bb3") != std::string::npos);
    return true;
}

[[nodiscard]] bool test_symbolic_width() {
    auto function = make_diamond(0u);
    CHECK(verify(function).succeeded());
    CHECK(to_string(function).find("warp=symbolic") != std::string::npos);
    return true;
}

[[nodiscard]] bool test_valid_loop() {
    Function function{"loop", 4u};
    auto entry = function.add_block("entry");
    auto header = function.add_block("header");
    auto body = function.add_block("body");
    auto exit = function.add_block("exit");
    function.set_entry(entry);
    auto condition = function.add_value(
        ValueClass::varying, nullptr, ValueOrigin::parameter,
        std::nullopt, "condition");
    auto initial = function.add_value(
        ValueClass::uniform, nullptr, ValueOrigin::parameter,
        std::nullopt, "initial");
    auto state = function.add_value(
        ValueClass::varying, nullptr, ValueOrigin::state_slot,
        header, "state");
    auto next = function.add_value(
        ValueClass::varying, nullptr, ValueOrigin::instruction,
        body, "next");
    auto loop = function.add_loop(header, exit);
    auto entry_edge = ControlEdge{header};
    entry_edge.assignments.emplace_back(EdgeAssignment{state, initial});
    function.block(entry)->terminator =
        BranchTerminator{std::move(entry_edge)};
    function.block(header)->terminator = SplitTerminator{
        .condition = condition,
        .true_edge = ControlEdge{body},
        .false_edge = ControlEdge{exit},
    };
    function.block(body)->instructions.emplace_back(Instruction{
        .opcode = Opcode::arithmetic,
        .result = next,
        .operands = {state},
    });
    function.block(body)->terminator = LoopBackTerminator{
        .loop = loop,
        .assignments = {{state, next}},
    };
    function.block(exit)->terminator = ReturnTerminator{};
    auto result = verify(function);
    CHECK(result.succeeded());
    auto text = to_string(function);
    CHECK(text.find("loop l0 header=bb1 exits=[bb3]") !=
          std::string::npos);
    CHECK(text.find("loop_back l0") != std::string::npos);
    return true;
}

[[nodiscard]] bool test_missing_state_assignment() {
    auto function = make_diamond(8u);
    auto &terminator = function.block(BlockId{1u})->terminator;
    std::get<JoinTerminator>(terminator).assignments.clear();
    auto result = verify(function);
    CHECK(!result.succeeded());
    CHECK(contains_error(result, "missing a state-slot assignment"));
    return true;
}

[[nodiscard]] bool test_control_edge_join_order() {
    Function function{"joined_edge", 8u};
    auto entry = function.add_block("entry");
    auto merge = function.add_block("merge");
    function.set_entry(entry);
    auto outer = function.add_convergence(merge);
    auto inner = function.add_convergence(merge, outer);
    auto edge = ControlEdge{merge};
    edge.joins = {inner, outer};
    function.block(entry)->terminator = BranchTerminator{std::move(edge)};
    function.block(merge)->terminator = ReturnTerminator{};
    CHECK(verify(function).succeeded());
    CHECK(to_string(function).find("joins=[c1, c0]") !=
          std::string::npos);

    auto &bad_edge = std::get<BranchTerminator>(
                         function.block(entry)->terminator)
                         .edge;
    bad_edge.joins = {outer, inner};
    auto result = verify(function);
    CHECK(!result.succeeded());
    CHECK(contains_error(result, "inner-to-outer"));
    return true;
}

[[nodiscard]] bool test_missing_terminator() {
    Function function{"missing_terminator"};
    auto entry = function.add_block("entry");
    function.set_entry(entry);
    auto result = verify(function);
    CHECK(!result.succeeded());
    CHECK(contains_error(result, "no terminator"));
    return true;
}

[[nodiscard]] bool test_invalid_target() {
    Function function{"invalid_target"};
    auto entry = function.add_block("entry");
    function.set_entry(entry);
    function.block(entry)->terminator = BranchTerminator{BlockId{99u}};
    auto result = verify(function);
    CHECK(!result.succeeded());
    CHECK(contains_error(result, "invalid target"));
    return true;
}

[[nodiscard]] bool test_duplicate_definition() {
    Function function{"duplicate_definition"};
    auto entry = function.add_block("entry");
    function.set_entry(entry);
    auto value = function.add_value(
        ValueClass::varying, nullptr, ValueOrigin::instruction,
        entry, "duplicate");
    function.block(entry)->instructions.emplace_back(Instruction{
        .opcode = Opcode::constant,
        .result = value,
    });
    function.block(entry)->instructions.emplace_back(Instruction{
        .opcode = Opcode::constant,
        .result = value,
    });
    function.block(entry)->terminator = ReturnTerminator{};
    auto result = verify(function);
    CHECK(!result.succeeded());
    CHECK(contains_error(result, "exactly one definition"));
    return true;
}

[[nodiscard]] bool test_collective_instance_required() {
    Function function{"collective"};
    auto entry = function.add_block("entry");
    function.set_entry(entry);
    auto value = function.add_value(
        ValueClass::varying, nullptr, ValueOrigin::instruction,
        entry, "sum");
    auto mask = function.add_value(
        ValueClass::mask, nullptr, ValueOrigin::scheduler_builtin,
        std::nullopt, "active_mask");
    function.block(entry)->instructions.emplace_back(Instruction{
        .opcode = Opcode::warp_collective,
        .result = value,
        .participant_mask = mask,
    });
    function.block(entry)->terminator = ReturnTerminator{};
    auto result = verify(function);
    CHECK(!result.succeeded());
    CHECK(contains_error(result, "dynamic instance ID"));
    function.block(entry)->instructions.front().collective_id = 0u;
    CHECK(verify(function).succeeded());
    return true;
}

[[nodiscard]] bool test_invalid_hierarchies_do_not_crash() {
    Function function{"invalid_hierarchies"};
    auto entry = function.add_block("entry");
    function.set_entry(entry);
    static_cast<void>(function.add_convergence(
        entry, ConvergenceId{99u}));
    static_cast<void>(function.add_loop(entry, BlockId{99u}, LoopId{99u}));
    function.block(entry)->terminator = ReturnTerminator{};
    auto result = verify(function);
    CHECK(!result.succeeded());
    CHECK(contains_error(result, "invalid parent"));
    CHECK(contains_error(result, "invalid exit"));
    return true;
}

[[nodiscard]] bool test_invalid_width() {
    auto function = make_diamond(129u);
    auto result = verify(function);
    CHECK(!result.succeeded());
    CHECK(contains_error(result, "at most 128"));
    return true;
}

[[nodiscard]] bool test_large_indexed_verification() {
    constexpr auto block_count = 4096u;
    Function function{"large_linear_cfg", 8u};
    std::vector<BlockId> blocks;
    std::vector<ValueId> states;
    blocks.reserve(block_count);
    states.reserve(block_count - 1u);
    for (auto i = 0u; i < block_count; i++) {
        blocks.emplace_back(function.add_block());
    }
    function.set_entry(blocks.front());
    auto seed = function.add_value(
        ValueClass::uniform, nullptr, ValueOrigin::parameter,
        std::nullopt, "seed");
    for (auto i = 1u; i < block_count; i++) {
        states.emplace_back(function.add_value(
            ValueClass::varying, nullptr, ValueOrigin::state_slot,
            blocks[i]));
    }
    for (auto i = 0u; i + 1u < block_count; i++) {
        auto edge = ControlEdge{blocks[i + 1u]};
        edge.assignments.emplace_back(EdgeAssignment{
            .destination = states[i],
            .source = i == 0u ? seed : states[i - 1u],
        });
        function.block(blocks[i])->terminator =
            BranchTerminator{std::move(edge)};
    }
    function.block(blocks.back())->terminator = ReturnTerminator{};

    std::optional<ConvergenceId> parent;
    for (auto i = 0u; i < block_count; i++) {
        parent = function.add_convergence(blocks.back(), parent);
    }
    CHECK(verify(function).succeeded());
    return true;
}

}// namespace

int main() {
    struct Test {
        std::string_view name;
        bool (*run)();
    };
    constexpr Test tests[]{
        {"valid diamond", &test_valid_diamond},
        {"symbolic width", &test_symbolic_width},
        {"valid loop", &test_valid_loop},
        {"missing state assignment", &test_missing_state_assignment},
        {"control edge join order", &test_control_edge_join_order},
        {"missing terminator", &test_missing_terminator},
        {"invalid target", &test_invalid_target},
        {"duplicate definition", &test_duplicate_definition},
        {"collective instance", &test_collective_instance_required},
        {"invalid hierarchies", &test_invalid_hierarchies_do_not_crash},
        {"invalid width", &test_invalid_width},
        {"large indexed verification", &test_large_indexed_verification},
    };
    auto failures = 0u;
    for (auto test : tests) {
        if (test.run()) {
            std::cout << "[pass] " << test.name << '\n';
        } else {
            std::cerr << "[fail] " << test.name << '\n';
            ++failures;
        }
    }
    return failures == 0u ? 0 : 1;
}
