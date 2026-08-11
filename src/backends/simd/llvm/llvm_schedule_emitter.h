#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/ADT/APInt.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/ast/type.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/op.h>
#include <luisa/xir/special_register.h>

#include "llvm_schedule_codegen.h"
#include "llvm_value_layout.h"
#include "llvm_warp_collectives.h"
#include "../schedule/schedule_ir.h"

namespace luisa::compute::simd::detail {

// Private implementation shared by the focused Schedule IR lowering units.
// Keep the public codegen facade independent of these implementation details.
class ScheduleEmitter {

private:
    ::llvm::Module &_module;
    const schedule::Function &_source;
    uint32_t _width;
    std::string _entry_name;
    bool _enable_fast_math;
    LLVMScheduleCodegenResult _result{};
    LLVMValueLayout _layout;
    LLVMWarpCollectives _collectives;
    ::llvm::IRBuilder<> _builder;
    ::llvm::Function *_entry{nullptr};
    ::llvm::Value *_argument_buffer{nullptr};
    ::llvm::Value *_return_buffer{nullptr};
    ::llvm::Value *_launch_config{nullptr};
    ::llvm::Value *_active_lane_count{nullptr};
    ::llvm::BasicBlock *_scheduler_loop{nullptr};
    ::llvm::AllocaInst *_live_mask{nullptr};
    ::llvm::AllocaInst *_runnable_mask{nullptr};
    ::llvm::AllocaInst *_pc_state{nullptr};
    ::llvm::AllocaInst *_token_state{nullptr};
    ::llvm::AllocaInst *_frame_active{nullptr};
    ::llvm::AllocaInst *_frame_static_id{nullptr};
    ::llvm::AllocaInst *_frame_parent_token{nullptr};
    ::llvm::AllocaInst *_frame_expected{nullptr};
    ::llvm::AllocaInst *_frame_arrived{nullptr};
    ::llvm::Constant *_convergence_targets{nullptr};
    std::vector<uint32_t> _target_convergence_depths{};
    std::vector<::llvm::AllocaInst *> _loop_epochs{};
    std::vector<std::vector<schedule::LoopId>> _block_loops{};
    std::vector<::llvm::AllocaInst *> _state_slots{};
    std::vector<uint8_t> _spilled_instruction_values{};
    std::vector<uint8_t> _local_lvalue_values{};
    std::vector<::llvm::Value *> _local_allocations{};
    std::vector<::llvm::Value *> _external_values{};
    std::vector<size_t> _parameter_offsets{};
    std::unordered_map<uint32_t, ::llvm::Value *> _locals{};
    ::llvm::Value *_active_mask{nullptr};
    ::llvm::Value *_seed_lane{nullptr};
    ::llvm::Value *_linear_thread_indices{nullptr};
    std::array<::llvm::Value *, 3u> _block_id{};
    std::array<::llvm::Value *, 3u> _dispatch_size{};
    std::array<::llvm::Value *, 3u> _block_size{};
    std::array<::llvm::Value *, 3u> _thread_id{};
    std::array<::llvm::Value *, 3u> _dispatch_id{};

    using UnaryLeaf = std::function<::llvm::Value *(
        ::llvm::Value *, const Type *)>;
    using BinaryLeaf = std::function<::llvm::Value *(
        ::llvm::Value *, ::llvm::Value *, const Type *, const Type *)>;
    using TernaryLeaf = std::function<::llvm::Value *(
        ::llvm::Value *, ::llvm::Value *, ::llvm::Value *,
        const Type *, const Type *, const Type *)>;

    struct BindlessBufferLanes {
        ::llvm::Value *data{nullptr};
        ::llvm::Value *size_bytes{nullptr};
    };

private:
    void _fail(std::string message);
    [[nodiscard]] bool _failed() const noexcept;
    [[nodiscard]] static size_t _align_up(size_t value,
                                          size_t alignment) noexcept;
    [[nodiscard]] static bool _is_scalar_data(const Type *type) noexcept;
    [[nodiscard]] static bool _is_data(const Type *type) noexcept;
    [[nodiscard]] static size_t _abi_size(const Type *type) noexcept;
    [[nodiscard]] static size_t _abi_alignment(const Type *type) noexcept;
    [[nodiscard]] static uint32_t _child_count(const Type *type) noexcept;
    [[nodiscard]] static const Type *_child_type(
        const Type *type, uint32_t index) noexcept;
    [[nodiscard]] static size_t _child_offset(
        const Type *type, uint32_t index) noexcept;
    [[nodiscard]] ::llvm::Type *_data_type(
        const Type *type, bool varying);
    [[nodiscard]] ::llvm::Value *_extract_child(
        ::llvm::Value *aggregate, const Type *type, uint32_t index,
        bool varying);
    [[nodiscard]] ::llvm::Value *_insert_child(
        ::llvm::Value *aggregate, ::llvm::Value *child,
        const Type *type, uint32_t index, bool varying);
    [[nodiscard]] ::llvm::Value *_assemble(
        const Type *type, bool varying,
        const std::function<::llvm::Value *(uint32_t)> &child);
    [[nodiscard]] ::llvm::Value *_splat_data(
        ::llvm::Value *value, const Type *type);
    [[nodiscard]] ::llvm::Value *_extract_lane(
        ::llvm::Value *value, const Type *type, ::llvm::Value *lane);
    [[nodiscard]] ::llvm::Value *_masked_merge(
        ::llvm::Value *new_value, ::llvm::Value *old_value,
        const Type *type, ::llvm::Value *mask);
    [[nodiscard]] ::llvm::StructType *_local_handle_type();
    [[nodiscard]] ::llvm::Value *_local_handle(
        ::llvm::Value *base, ::llvm::Value *offsets);
    [[nodiscard]] static ::llvm::Value *_local_base(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *handle);
    [[nodiscard]] static ::llvm::Value *_local_offsets(
        ::llvm::IRBuilder<> &builder, ::llvm::Value *handle);
    [[nodiscard]] ::llvm::Value *_merge_local_handles(
        ::llvm::Value *new_handle, ::llvm::Value *old_handle,
        ::llvm::Value *mask);
    [[nodiscard]] bool _is_local_lvalue(
        schedule::ValueId id) const noexcept;
    static void _for_each_assignment(
        const schedule::BasicBlock &block,
        const std::function<void(schedule::EdgeAssignment)> &visit);
    void _analyze_local_lvalues();
    void _preflight_edge(const schedule::ControlEdge &edge,
                         bool split_edge);
    void _preflight();
    [[nodiscard]] ::llvm::Constant *_scalar_constant(
        const Type *type, const std::byte *bytes);
    [[nodiscard]] ::llvm::Value *_lane_ids();
    [[nodiscard]] ::llvm::Constant *_constant_data(
        const Type *type, const std::byte *bytes, size_t offset = 0u);
    [[nodiscard]] ::llvm::Value *_byte_pointer(
        ::llvm::Value *base, size_t offset);
    [[nodiscard]] ::llvm::Value *_load_uniform_data(
        ::llvm::Value *base, const Type *type, size_t offset = 0u);
    [[nodiscard]] ::llvm::Value *_load_buffer_view(
        ::llvm::Value *base);
    [[nodiscard]] ::llvm::Value *_load_texture_view(
        ::llvm::Value *base);
    [[nodiscard]] ::llvm::Value *_load_bindless_view(
        ::llvm::Value *base);
    [[nodiscard]] ::llvm::Value *_load_launch_u32(size_t offset);
    void _ensure_launch_vectors();
    [[nodiscard]] ::llvm::Value *_triplet(
        const Type *type, const std::array<::llvm::Value *, 3u> &values,
        bool varying);
    [[nodiscard]] ::llvm::Value *_special_register(
        const schedule::Value &value,
        xir::DerivedSpecialRegisterTag tag);
    void _create_external_values();
    [[nodiscard]] ::llvm::Value *_load_value(schedule::ValueId id);
    [[nodiscard]] ::llvm::Value *_as_lane_vector(
        ::llvm::Value *value, const schedule::Value &schedule_value);
    [[nodiscard]] ::llvm::Value *_select_data(
        ::llvm::Value *condition, ::llvm::Value *true_value,
        ::llvm::Value *false_value, const Type *type, bool varying);
    [[nodiscard]] ::llvm::Value *_componentwise_unary(
        const Type *result_type, ::llvm::Value *operand,
        const Type *operand_type, bool varying, const UnaryLeaf &leaf);
    [[nodiscard]] ::llvm::Value *_componentwise_binary(
        const Type *result_type, ::llvm::Value *lhs, const Type *lhs_type,
        ::llvm::Value *rhs, const Type *rhs_type, bool varying,
        const BinaryLeaf &leaf);
    [[nodiscard]] ::llvm::Value *_componentwise_ternary(
        const Type *result_type,
        ::llvm::Value *a, const Type *a_type,
        ::llvm::Value *b, const Type *b_type,
        ::llvm::Value *c, const Type *c_type, bool varying,
        const TernaryLeaf &leaf);
    [[nodiscard]] ::llvm::Value *_componentwise_varying_to_uniform(
        const Type *result_type, ::llvm::Value *operand,
        const Type *operand_type, const UnaryLeaf &leaf);
    [[nodiscard]] static std::optional<uint64_t> _constant_index(
        ::llvm::Value *value) noexcept;
    [[nodiscard]] ::llvm::Value *_index_constant_like(
        ::llvm::Value *index, uint64_t value);
    [[nodiscard]] ::llvm::Value *_extract_indexed(
        ::llvm::Value *aggregate, const Type *type,
        const std::vector<::llvm::Value *> &indices, size_t depth,
        bool varying);
    [[nodiscard]] ::llvm::Value *_insert_indexed(
        ::llvm::Value *aggregate, const Type *type,
        ::llvm::Value *replacement,
        const std::vector<::llvm::Value *> &indices, size_t depth,
        bool varying);
    [[nodiscard]] ::llvm::Value *_aggregate_operation(
        const schedule::Value &result,
        const schedule::Instruction &instruction,
        const std::vector<::llvm::Value *> &operands, bool varying);
    [[nodiscard]] ::llvm::Value *_arithmetic(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_cast(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_lane_offsets(
        ::llvm::Value *index, uint64_t stride);
    [[nodiscard]] std::optional<uint64_t> _constant_aggregate_index(
        schedule::ValueId id) const noexcept;
    [[nodiscard]] bool _advance_aggregate_offset(
        ::llvm::Value *&offsets, const Type *&current_type,
        schedule::ValueId index_id);
    [[nodiscard]] ::llvm::Value *_local_alloca(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_local_gep(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_local_load(
        const schedule::Instruction &instruction);
    void _local_store(const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_atomic(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_leaf_pointers(
        ::llvm::Value *base, ::llvm::Value *offsets,
        size_t leaf_offset);
    [[nodiscard]] ::llvm::AllocaInst *_entry_scratch(
        ::llvm::Type *type, std::string_view name);
    [[nodiscard]] ::llvm::Value *_texture_read(
        const schedule::Instruction &instruction);
    void _texture_write(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_gather_data(
        ::llvm::Value *base, ::llvm::Value *offsets,
        const Type *type, size_t leaf_offset = 0u);
    void _scatter_data(
        ::llvm::Value *base, ::llvm::Value *offsets,
        const Type *type, ::llvm::Value *value,
        size_t leaf_offset = 0u);
    [[nodiscard]] BindlessBufferLanes _bindless_buffer_lanes(
        schedule::ValueId bindless_id, schedule::ValueId slot_id);
    [[nodiscard]] ::llvm::Value *_bindless_access_offsets(
        const BindlessBufferLanes &buffer, ::llvm::Value *index,
        uint64_t stride, size_t access_size);
    [[nodiscard]] ::llvm::Value *_bindless_resource_read(
        const schedule::Instruction &instruction);
    void _bindless_resource_write(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_bindless_resource_query(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_resource_read(
        const schedule::Instruction &instruction);
    void _resource_write(const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_resource_query(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_collective(
        const schedule::Instruction &instruction);
    void _emit_instruction(const schedule::Instruction &instruction);
    void _assign(schedule::EdgeAssignment assignment,
                 ::llvm::Value *mask);
    void _apply_assignments(
        const std::vector<schedule::EdgeAssignment> &assignments,
        ::llvm::Value *mask);
    [[nodiscard]] ::llvm::Value *_zero_mask() noexcept;
    [[nodiscard]] ::llvm::Value *_splat(::llvm::Value *scalar);
    [[nodiscard]] ::llvm::Value *_safe_first_lane(::llvm::Value *mask);
    void _masked_write(::llvm::AllocaInst *slot, ::llvm::Value *value,
                       ::llvm::Value *mask);
    [[nodiscard]] ::llvm::Value *_frame_mask_pointer(
        ::llvm::AllocaInst *frames, ::llvm::Value *index);
    void _trap_if(::llvm::Value *condition, std::string_view label);
    [[nodiscard]] ::llvm::Value *_current_token(::llvm::Value *mask);
    void _declare_convergence(schedule::ConvergenceId convergence,
                              ::llvm::Value *divergent);
    void _advance_loop_epoch(schedule::LoopId loop, ::llvm::Value *mask);
    [[nodiscard]] ::llvm::Value *_arrive_at_convergence_target(
        ::llvm::Value *target, ::llvm::Value *flow,
        ::llvm::Value **matched);
    [[nodiscard]] ::llvm::Value *_cascade_at_convergence_target(
        ::llvm::Value *target, ::llvm::Value *flow);
    void _resume(schedule::BlockId target, ::llvm::Value *mask);
    void _route_edge(const schedule::ControlEdge &edge,
                     ::llvm::Value *mask);
    void _emit_arrival(const schedule::ControlEdge &edge,
                       ::llvm::Value *mask);
    void _emit_terminator(const schedule::Terminator &terminator);
    void _emit_scalar_terminator(
        const schedule::Terminator &terminator,
        const std::vector<::llvm::BasicBlock *> &blocks);
    void _find_instruction_spills();
    void _allocate_state();
    void _build_scalar(::llvm::Value *initial_mask);
    void _build();

public:
    ScheduleEmitter(::llvm::Module &module,
                    const schedule::Function &source, uint32_t width,
                    std::string_view entry_name,
                    bool enable_fast_math);
    [[nodiscard]] LLVMScheduleCodegenResult run();
};

}// namespace luisa::compute::simd::detail
