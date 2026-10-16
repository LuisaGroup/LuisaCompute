#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <luisa/core/stl/optional.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::simd::schedule {

inline constexpr auto invalid_schedule_id =
    std::numeric_limits<uint32_t>::max();

template<typename Tag>
struct Id {
    uint32_t value{invalid_schedule_id};
    [[nodiscard]] constexpr bool valid() const noexcept {
        return value != invalid_schedule_id;
    }
    friend constexpr bool operator==(Id, Id) noexcept = default;
};

struct BlockIdTag;
struct ValueIdTag;
struct ConvergenceIdTag;
struct LoopIdTag;

using BlockId = Id<BlockIdTag>;
using ValueId = Id<ValueIdTag>;
using ConvergenceId = Id<ConvergenceIdTag>;
using LoopId = Id<LoopIdTag>;

enum struct ValueClass {
    // Stable across every lane and every dynamic cohort in one logical warp.
    // These values may live in one scalar warp-state slot.
    warp_uniform,
    // Scalar while a dynamic cohort executes, but different paths or loop
    // epochs may observe different values. These values require lane-wise
    // state if they survive a scheduler suspension point.
    cohort_uniform,
    varying,
    mask,
    token,
};

[[nodiscard]] constexpr bool is_uniform(ValueClass value) noexcept {
    return value == ValueClass::warp_uniform ||
           value == ValueClass::cohort_uniform;
}

enum struct ValueOrigin {
    parameter,
    constant,
    special_register,
    scheduler_builtin,
    instruction,
    state_slot,
};

// Source metadata is copied into Schedule IR so later code generation never
// has to retain or query the source XIR module. Enum-like fields intentionally
// use their stable numeric value to keep this dependency-light dialect free of
// XIR headers.
struct ParameterValueMetadata {
    uint32_t index{0u};
    uint32_t argument_tag{0u};
};

struct ConstantValueMetadata {
    std::vector<std::byte> bytes{};
};

struct SpecialRegisterValueMetadata {
    uint32_t tag{0u};
};

enum struct SchedulerBuiltin : uint32_t {
    active_mask,
};

struct SchedulerBuiltinValueMetadata {
    SchedulerBuiltin builtin{SchedulerBuiltin::active_mask};
};

using ValueMetadata = std::variant<
    std::monostate,
    ParameterValueMetadata,
    ConstantValueMetadata,
    SpecialRegisterValueMetadata,
    SchedulerBuiltinValueMetadata>;

enum struct RegionStrategy {
    uniform_control,
    predicated,
    cohort,
};

enum struct Opcode {
    constant,
    special_register,
    arithmetic,
    cast,
    call,
    alloca,
    load,
    store,
    gep,
    atomic,
    resource_query,
    resource_read,
    resource_write,
    ray_query_read,
    ray_query_write,
    ray_query_pipeline,
    warp_collective,
    edge_copy,
    print,
    assert_,
    clock,
    opaque,
};

struct Value {
    ValueId id{};
    ValueClass value_class{ValueClass::varying};
    ValueOrigin origin{ValueOrigin::instruction};
    const Type *type{nullptr};
    luisa::optional<BlockId> defining_block{};
    std::string name{};
    ValueMetadata metadata{};
};

enum struct StridedMmaVectorization : uint8_t {
    output,
    contraction,
};

// Owned semantic descriptor for the only admitted private matrix leaf call.
// No source XIR object or runtime callback survives this boundary.
struct StridedMmaMetadata {
    std::vector<uint64_t> output_extents{};
    std::vector<uint64_t> lhs_output_strides{};
    std::vector<uint64_t> rhs_output_strides{};
    uint64_t contraction_extent{0u};
    uint64_t lhs_contraction_stride{0u};
    uint64_t rhs_contraction_stride{0u};
    uint32_t vector_width{1u};
    bool allow_reassociation{false};
    StridedMmaVectorization vectorization{StridedMmaVectorization::output};
};

// Owned call-site copy semantics; no XIR lifetime or runtime callback ABI.
struct ContiguousCopyMetadata {
    uint64_t element_count{0u};
    uint32_t vector_width{1u};
};

struct Instruction {
    Opcode opcode{Opcode::opaque};
    luisa::optional<ValueId> result{};
    std::vector<ValueId> operands{};
    // Preserves the exact XIR opcode during the first lowering. Schedule IR
    // owns execution class and scheduling; target codegen still needs the
    // source operation within broad categories such as arithmetic or atomic.
    luisa::optional<uint32_t> source_op{};
    // Exact user-visible payload for side-effecting debug operations. An
    // optional is required because an empty print format or assertion message
    // is still distinct from an instruction that carries no message.
    luisa::optional<std::string> message{};
    luisa::optional<uint32_t> collective_id{};
    luisa::optional<ValueId> participant_mask{};
    // One operand may be lane-equal only at this instruction's dynamic
    // continuation even when its backing state is varying across loop exits.
    // This is a use-site fact, not a global ValueClass refinement.
    // For a GEP index, users must establish the same dynamic epoch before
    // using equality to realize a contiguous private access.
    luisa::optional<uint32_t> cohort_uniform_operand_index{};
    // One integer operand may be proven to increase by exactly one between
    // adjacent physical packet lanes. The proof is use-site-local and relies
    // on static block geometry, so it must not change the ValueClass of the
    // backing SSA value.
    luisa::optional<uint32_t> lane_consecutive_operand_index{};
    // void(lhs, rhs, seed, output), all fixed-array<float> local references.
    luisa::optional<StridedMmaMetadata> strided_mma{};
    // void(buffer<float>, uint64 element_offset, local array<float> reference).
    luisa::optional<ContiguousCopyMetadata> contiguous_copy{};
};

struct EdgeAssignment {
    ValueId destination{};
    ValueId source{};
};

struct ControlEdge {
    BlockId target{};
    // Statically provable gates reached before entering target, ordered from
    // the innermost scope to the outermost. A block with entries from both
    // inside and outside a scope may need an additional context-dependent
    // arrival; codegen therefore validates the current dynamic token against
    // the target block even when that gate is absent from this hint list.
    std::vector<ConvergenceId> joins{};
    luisa::optional<LoopId> loop_back{};
    std::vector<EdgeAssignment> assignments{};

    constexpr ControlEdge() noexcept = default;
    constexpr ControlEdge(BlockId target) noexcept : target{target} {}
};

struct BranchTerminator {
    ControlEdge edge{};
};

struct SplitTerminator {
    ValueId condition{};
    ControlEdge true_edge{};
    ControlEdge false_edge{};
    luisa::optional<ConvergenceId> convergence{};
    // The backing condition remains varying across loop exits, but all active
    // lanes in this dynamic continuation are proven to agree on it.
    bool cohort_uniform_condition{false};
};

struct SwitchCase {
    uint64_t value{0u};
    ControlEdge edge{};
};

struct SwitchTerminator {
    ValueId selector{};
    std::vector<SwitchCase> cases{};
    ControlEdge default_edge{};
    luisa::optional<ConvergenceId> convergence{};
};

// Arriving lanes park at the convergence gate. Once its expected live mask
// has arrived or terminated, the scheduler resumes the gate's target block.
struct JoinTerminator {
    ConvergenceId convergence{};
    std::vector<EdgeAssignment> assignments{};
};

// A loop back-edge advances this loop's dynamic epoch before resuming header.
struct LoopBackTerminator {
    LoopId loop{};
    std::vector<EdgeAssignment> assignments{};
};

struct BlockBarrierTerminator {
    uint32_t barrier_id{0u};
    ControlEdge resume_edge{};
};

struct ReturnTerminator {
    luisa::optional<ValueId> value{};
};

struct UnreachableTerminator {};

using Terminator = std::variant<
    std::monostate,
    BranchTerminator,
    SplitTerminator,
    SwitchTerminator,
    JoinTerminator,
    LoopBackTerminator,
    BlockBarrierTerminator,
    ReturnTerminator,
    UnreachableTerminator>;

struct BasicBlock {
    BlockId id{};
    std::string name{};
    RegionStrategy strategy{RegionStrategy::cohort};
    std::vector<Instruction> instructions{};
    Terminator terminator{};
};

struct ConvergencePoint {
    ConvergenceId id{};
    BlockId target{};
    luisa::optional<ConvergenceId> parent{};
};

struct Loop {
    LoopId id{};
    BlockId header{};
    // Blocks whose dynamic instances are distinguished by this loop's epoch.
    // Exit blocks are deliberately excluded so lanes that took different
    // iteration counts may reconverge after leaving the loop.
    std::vector<BlockId> blocks{};
    // Intentional infinite loops have an empty exit set.
    std::vector<BlockId> exits{};
    // A proven finite upper bound. Additional early exits may make the
    // dynamic trip count smaller; no execution may take more iterations.
    luisa::optional<uint64_t> max_trip_count{};
    luisa::optional<LoopId> parent{};
};

class Function {

private:
    std::string _name;
    // Zero denotes symbolic width before LLVM specialization.
    uint32_t _logical_warp_width{0u};
    BlockId _entry{};
    std::vector<Value> _values;
    std::vector<BasicBlock> _blocks;
    std::vector<ConvergencePoint> _convergence_points;
    std::vector<Loop> _loops;

public:
    explicit Function(std::string name = {},
                      uint32_t logical_warp_width = 0u) noexcept;

    [[nodiscard]] ValueId add_value(
        ValueClass value_class, const Type *type = nullptr,
        ValueOrigin origin = ValueOrigin::instruction,
        luisa::optional<BlockId> defining_block = luisa::nullopt,
        std::string name = {}, ValueMetadata metadata = {});
    [[nodiscard]] BlockId add_block(std::string name = {});
    [[nodiscard]] ConvergenceId add_convergence(
        BlockId target,
        luisa::optional<ConvergenceId> parent = luisa::nullopt);
    [[nodiscard]] LoopId add_loop(
        BlockId header, std::vector<BlockId> blocks,
        std::vector<BlockId> exits,
        luisa::optional<LoopId> parent = luisa::nullopt,
        luisa::optional<uint64_t> max_trip_count = luisa::nullopt);
    [[nodiscard]] LoopId add_loop(
        BlockId header, BlockId exit,
        luisa::optional<LoopId> parent = luisa::nullopt) {
        return add_loop(header, std::vector<BlockId>{header},
                        std::vector<BlockId>{exit}, parent);
    }

    void set_entry(BlockId entry) noexcept { _entry = entry; }

    [[nodiscard]] const std::string &name() const noexcept { return _name; }
    [[nodiscard]] uint32_t logical_warp_width() const noexcept {
        return _logical_warp_width;
    }
    [[nodiscard]] BlockId entry() const noexcept { return _entry; }
    [[nodiscard]] const std::vector<Value> &values() const noexcept {
        return _values;
    }
    [[nodiscard]] std::vector<Value> &values() noexcept { return _values; }
    [[nodiscard]] const std::vector<BasicBlock> &blocks() const noexcept {
        return _blocks;
    }
    [[nodiscard]] std::vector<BasicBlock> &blocks() noexcept { return _blocks; }
    [[nodiscard]] const std::vector<ConvergencePoint> &convergence_points()
        const noexcept {
        return _convergence_points;
    }
    [[nodiscard]] const std::vector<Loop> &loops() const noexcept {
        return _loops;
    }

    [[nodiscard]] Value *value(ValueId id) noexcept;
    [[nodiscard]] const Value *value(ValueId id) const noexcept;
    [[nodiscard]] BasicBlock *block(BlockId id) noexcept;
    [[nodiscard]] const BasicBlock *block(BlockId id) const noexcept;
    [[nodiscard]] ConvergencePoint *convergence(
        ConvergenceId id) noexcept;
    [[nodiscard]] const ConvergencePoint *convergence(
        ConvergenceId id) const noexcept;
    [[nodiscard]] Loop *loop(LoopId id) noexcept;
    [[nodiscard]] const Loop *loop(LoopId id) const noexcept;
};

struct VerificationError {
    std::string message;
    luisa::optional<BlockId> block{};
};

struct VerificationResult {
    std::vector<VerificationError> errors;
    [[nodiscard]] bool succeeded() const noexcept { return errors.empty(); }
};

[[nodiscard]] VerificationResult verify(const Function &function);
[[nodiscard]] std::string to_string(const Function &function);

[[nodiscard]] const char *to_string(ValueClass value) noexcept;
[[nodiscard]] const char *to_string(ValueOrigin value) noexcept;
[[nodiscard]] const char *to_string(RegionStrategy value) noexcept;
[[nodiscard]] const char *to_string(Opcode value) noexcept;

}// namespace luisa::compute::simd::schedule
