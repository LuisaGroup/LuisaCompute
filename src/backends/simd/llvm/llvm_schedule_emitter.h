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

enum class ScheduleEntryABI {
    packet,
    ray_query_handler,
};

// Private implementation shared by the focused Schedule IR lowering units.
// Keep the public codegen facade independent of these implementation details.
class ScheduleEmitter {

private:
    ::llvm::Module &_module;
    const schedule::Function &_source;
    uint32_t _width;
    std::string _entry_name;
    bool _enable_fast_math;
    std::array<uint32_t, 3u> _static_block_size{};
    bool _enable_uniform_buffer_broadcast{true};
    bool _enable_lane_affine_buffer{true};
    bool _enable_paired_leaf_gather{false};
    uint32_t _dispatch_worker_count{1u};
    bool _enable_native_predicated_loop{true};
    bool _enable_runtime_packet_geometry{false};
    bool _enable_linear_1d_packet_tail_narrowing{false};
    ScheduleEntryABI _entry_abi{ScheduleEntryABI::packet};
    std::span<const LLVMSIMDRayQueryPipelineHandlers>
        _ray_query_pipeline_handlers{};
    size_t _print_format_id_base{0u};
    bool _use_scalar_frame_metadata{false};
    bool _direct_control_flow{false};
    bool _has_block_barrier{false};
    bool _has_shared_memory{false};
    bool _cooperative_block{false};
    size_t _shared_memory_size{0u};
    LLVMScheduleCodegenResult _result{};
    LLVMValueLayout _layout;
    LLVMWarpCollectives _collectives;
    ::llvm::IRBuilder<> _builder;
    ::llvm::Function *_entry{nullptr};
    ::llvm::Value *_argument_buffer{nullptr};
    ::llvm::Value *_return_buffer{nullptr};
    ::llvm::Value *_launch_config{nullptr};
    ::llvm::Value *_active_lane_count{nullptr};
    ::llvm::Value *_handler_active_mask_bits{nullptr};
    ::llvm::BasicBlock *_scheduler_loop{nullptr};
    ::llvm::BasicBlock *_scheduler_dispatch_route{nullptr};
    ::llvm::PHINode *_scheduler_dispatch_pc{nullptr};
    ::llvm::AllocaInst *_live_mask{nullptr};
    ::llvm::AllocaInst *_runnable_mask{nullptr};
    ::llvm::AllocaInst *_ready_count{nullptr};
    ::llvm::AllocaInst *_ready_masks{nullptr};
    ::llvm::AllocaInst *_ready_targets{nullptr};
    ::llvm::AllocaInst *_ready_tokens{nullptr};
    ::llvm::AllocaInst *_current_mask{nullptr};
    ::llvm::AllocaInst *_current_token{nullptr};
    ::llvm::AllocaInst *_frame_active{nullptr};
    ::llvm::AllocaInst *_frame_static_id{nullptr};
    ::llvm::AllocaInst *_frame_parent_token{nullptr};
    ::llvm::AllocaInst *_frame_expected{nullptr};
    ::llvm::AllocaInst *_frame_arrived{nullptr};
    ::llvm::Constant *_convergence_targets{nullptr};
    std::vector<uint32_t> _target_convergence_depths{};
    std::vector<::llvm::AllocaInst *> _state_slots{};
    std::vector<uint8_t> _spilled_instruction_values{};
    std::vector<uint8_t> _local_lvalue_values{};
    std::vector<uint8_t> _shared_lvalue_values{};
    std::vector<::llvm::Value *> _local_allocations{};
    std::vector<uint32_t> _ray_query_scratch_slots{};
    std::vector<::llvm::AllocaInst *> _ray_query_scratch_storage{};
    std::vector<uint32_t> _ray_query_status_slots{};
    std::vector<::llvm::AllocaInst *> _ray_query_status_storage{};
    std::vector<::llvm::AllocaInst *> _ray_query_status_callback_storage{};
    std::vector<::llvm::AllocaInst *> _ray_query_pipeline_callback_storage{};
    std::vector<::llvm::AllocaInst *>
        _ray_query_surface_filter_pipeline_callback_storage{};
    std::vector<::llvm::AllocaInst *> _ray_query_state_handle_storage{};
    // Dense indices exist only for natural loops enclosing at least one
    // static block barrier. Each slot is a per-lane epoch vector retained in
    // the packet coroutine frame across suspensions.
    std::vector<int32_t> _cooperative_loop_epoch_indices{};
    std::vector<::llvm::AllocaInst *> _cooperative_loop_epochs{};
    std::vector<::llvm::Value *> _external_values{};
    std::vector<const schedule::Value *> _parameters{};
    std::vector<size_t> _parameter_offsets{};
    std::unordered_map<uint32_t, ::llvm::Value *> _locals{};
    std::unordered_map<const schedule::Instruction *, uint64_t>
        _print_format_ids{};
    std::unordered_map<const schedule::Instruction *, ::llvm::AllocaInst *>
        _print_argument_storage{};
    ::llvm::Value *_active_mask{nullptr};
    ::llvm::Value *_seed_lane{nullptr};
    ::llvm::Value *_coroutine_token{nullptr};
    ::llvm::Value *_coroutine_handle{nullptr};
    ::llvm::Value *_packet_index{nullptr};
    ::llvm::Value *_packet_participating{nullptr};
    ::llvm::BasicBlock *_coroutine_final{nullptr};
    ::llvm::BasicBlock *_coroutine_cleanup{nullptr};
    ::llvm::BasicBlock *_coroutine_suspend{nullptr};
    ::llvm::Value *_linear_thread_indices{nullptr};
    std::array<::llvm::Value *, 3u> _block_id{};
    std::array<::llvm::Value *, 3u> _dispatch_size{};
    std::array<::llvm::Value *, 3u> _block_size{};
    std::array<::llvm::Value *, 3u> _thread_id{};
    std::array<::llvm::Value *, 3u> _dispatch_id{};
    std::vector<::llvm::BasicBlock *> _schedule_blocks{};

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

    struct BindlessArrayLanes {
        ::llvm::Value *view{nullptr};
        ::llvm::Value *slots{nullptr};
        ::llvm::Value *slot_count{nullptr};
        ::llvm::Value *slot_indices{nullptr};
    };

    struct AccelInstanceAddress {
        ::llvm::Value *data{nullptr};
        ::llvm::Value *offsets{nullptr};
        ::llvm::Value *scalar{nullptr};
    };

    struct AccelMotionAddress {
        AccelInstanceAddress instance{};
        ::llvm::Value *frame{nullptr};
    };

    struct PredicatedMemoryDiamond {
        const schedule::BasicBlock *true_block{nullptr};
        const schedule::BasicBlock *false_block{nullptr};
        schedule::BlockId merge{};
        size_t instruction_count{0u};
    };

    struct GuardedPredicatedMathDiamond {
        std::vector<const schedule::BasicBlock *> true_blocks{};
        std::vector<const schedule::BasicBlock *> false_blocks{};
        schedule::BlockId merge{};
        bool two_sided{false};
        size_t instruction_count{0u};
    };

    struct NestedPredicatedRegion {
        const schedule::BasicBlock *nested_split_block{nullptr};
        GuardedPredicatedMathDiamond nested_diamond{};
        const schedule::BasicBlock *nested_merge_block{nullptr};
        const schedule::BasicBlock *other_block{nullptr};
        schedule::BlockId merge{};
        bool nested_on_true{false};
        size_t instruction_count{0u};
    };

    struct ChainedPredicatedRegion {
        struct Continuation {
            std::vector<const schedule::BasicBlock *> blocks{};
            GuardedPredicatedMathDiamond diamond{};
        };
        struct NestedContinuation {
            std::vector<const schedule::BasicBlock *> blocks{};
            NestedPredicatedRegion region{};
        };
        GuardedPredicatedMathDiamond first_diamond{};
        std::vector<Continuation> continuations{};
        std::optional<NestedContinuation> nested_continuation{};
        std::vector<const schedule::BasicBlock *> terminal_blocks{};
        std::vector<const schedule::BasicBlock *> inlined_blocks{};
        schedule::BlockId merge{};
        size_t instruction_count{0u};
        size_t terminal_instruction_count{0u};
    };

    struct PredicatedLoop {
        const schedule::Loop *loop{nullptr};
        schedule::ConvergenceId convergence{};
        std::vector<schedule::BlockId> exits{};
        std::vector<const schedule::BasicBlock *> order{};
        size_t instruction_count{0u};
        uint32_t batch_iteration_count{16u};
    };

    struct StructuredEarlyExitLoop {
        struct ExitTail {
            schedule::BlockId entry{};
            std::vector<const schedule::BasicBlock *> blocks{};
        };
        const schedule::Loop *loop{nullptr};
        const schedule::BasicBlock *header{nullptr};
        schedule::BlockId common_exit{};
        schedule::ValueId induction{};
        std::vector<schedule::ValueId> cohort_uniform_values{};
        std::vector<ExitTail> exit_tails{};
        std::vector<const schedule::BasicBlock *> emitted_blocks{};
        std::vector<const schedule::BasicBlock *> absorbed_blocks{};
        size_t instruction_count{0u};
    };

    struct CoherentAllOnRegion {
        std::vector<const schedule::BasicBlock *> blocks{};
        size_t instruction_count{0u};
        size_t weighted_cost{0u};
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
    [[nodiscard]] ::llvm::StructType *_buffer_view_type();
    [[nodiscard]] ::llvm::StructType *_texture_view_type();
    [[nodiscard]] ::llvm::StructType *_bindless_view_type();
    [[nodiscard]] ::llvm::StructType *_accel_view_type();
    [[nodiscard]] ::llvm::Type *_handler_parameter_type(
        const schedule::Value &value);
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
    [[nodiscard]] bool _is_shared_lvalue(
        schedule::ValueId id) const noexcept;
    static void _for_each_assignment(
        const schedule::BasicBlock &block,
        const std::function<void(schedule::EdgeAssignment)> &visit);
    void _analyze_local_lvalues();
    void _analyze_ray_query_scratch();
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
    [[nodiscard]] ::llvm::Value *_load_accel_view(
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
        const schedule::Instruction &instruction,
        ::llvm::Value *operand_sanitization_mask = nullptr);
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
    [[nodiscard]] ::llvm::Value *_atomic_lanes(
        xir::AtomicOp op, const schedule::Value &result,
        ::llvm::Value *base, ::llvm::Value *offsets,
        const std::vector<::llvm::Value *> &values);
    [[nodiscard]] ::llvm::Value *_shared_atomic(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_atomic(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_leaf_pointers(
        ::llvm::Value *base, ::llvm::Value *offsets,
        size_t leaf_offset);
    [[nodiscard]] ::llvm::AllocaInst *_entry_scratch(
        ::llvm::Type *type, std::string_view name);
    [[nodiscard]] ::llvm::Value *_texture_read(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_direct_texture_sample(
        const schedule::Instruction &instruction);
    void _texture_write(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_gather_data(
        ::llvm::Value *base, ::llvm::Value *offsets,
        const Type *type, size_t leaf_offset = 0u);
    [[nodiscard]] ::llvm::Value *_gather_paired_vector_data(
        ::llvm::Value *base, ::llvm::Value *offsets,
        const Type *type);
    [[nodiscard]] static uint32_t _vector_storage_component_count(
        const Type *type) noexcept;
    [[nodiscard]] ::llvm::Value *_lane_consecutive_address(
        ::llvm::Value *base, ::llvm::Value *index,
        uint64_t stride, ::llvm::Value *seed_lane,
        ::llvm::Value *seed_mask);
    [[nodiscard]] ::llvm::Value *_expand_lane_mask(
        ::llvm::Value *mask, uint32_t component_count,
        uint32_t storage_component_count);
    [[nodiscard]] ::llvm::Value *_interleave_vector_data(
        ::llvm::Value *value, const Type *type);
    [[nodiscard]] ::llvm::Value *_deinterleave_vector_data(
        ::llvm::Value *value, const Type *type);
    [[nodiscard]] ::llvm::Value *_load_contiguous_data(
        ::llvm::Value *base, ::llvm::Value *index,
        const Type *type, ::llvm::Value *seed_lane,
        ::llvm::Value *seed_mask);
    [[nodiscard]] ::llvm::Value *_load_contiguous_vector_data(
        ::llvm::Value *base, ::llvm::Value *index,
        const Type *type, ::llvm::Value *seed_lane,
        ::llvm::Value *seed_mask);
    void _scatter_data(
        ::llvm::Value *base, ::llvm::Value *offsets,
        const Type *type, ::llvm::Value *value,
        size_t leaf_offset = 0u);
    void _store_contiguous_data(
        ::llvm::Value *base, ::llvm::Value *index,
        const Type *type, ::llvm::Value *value);
    void _store_contiguous_vector_data(
        ::llvm::Value *base, ::llvm::Value *index,
        const Type *type, ::llvm::Value *value);
    [[nodiscard]] BindlessArrayLanes _bindless_array_lanes(
        schedule::ValueId bindless_id, schedule::ValueId slot_id);
    [[nodiscard]] ::llvm::Value *_bindless_callback_mask(
        bool varying_result);
    [[nodiscard]] BindlessBufferLanes _bindless_buffer_lanes(
        schedule::ValueId bindless_id, schedule::ValueId slot_id);
    [[nodiscard]] ::llvm::Value *_bindless_access_offsets(
        const BindlessBufferLanes &buffer, ::llvm::Value *index,
        uint64_t stride, size_t access_size);
    [[nodiscard]] ::llvm::Value *_bindless_resource_read(
        const schedule::Instruction &instruction);
    void _bindless_resource_write(
        const schedule::Instruction &instruction);
    void _indirect_dispatch_write(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_bindless_resource_query(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_bindless_texture_read(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_bindless_texture_query(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_accel_query(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_ray_query_create(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_ray_query_state_handles(
        schedule::ValueId object_id);
    [[nodiscard]] ::llvm::AllocaInst *_ray_query_status_slot(
        schedule::ValueId object_id) const noexcept;
    [[nodiscard]] ::llvm::AllocaInst *_ray_query_state_handle_slot(
        schedule::ValueId object_id) const noexcept;
    [[nodiscard]] ::llvm::Value *_ray_query_status_mask(
        schedule::ValueId object_id, uint32_t shift);
    void _ray_query_update_status(
        schedule::ValueId object_id, ::llvm::Value *status);
    [[nodiscard]] ::llvm::Value *_ray_query_read(
        const schedule::Instruction &instruction);
    void _ray_query_write(
        const schedule::Instruction &instruction);
    void _ray_query_pipeline(
        const schedule::Instruction &instruction);
    [[nodiscard]] AccelInstanceAddress _accel_instance_address(
        ::llvm::Value *accel, schedule::ValueId index_id,
        bool varying);
    [[nodiscard]] ::llvm::Value *_accel_instance_query(
        const schedule::Instruction &instruction);
    void _accel_instance_write(
        const schedule::Instruction &instruction);
    [[nodiscard]] AccelMotionAddress _accel_motion_address(
        ::llvm::Value *accel, schedule::ValueId instance_id,
        schedule::ValueId keyframe_id, bool varying,
        uint32_t expected_mode);
    [[nodiscard]] ::llvm::Value *_accel_motion_query(
        const schedule::Instruction &instruction);
    void _accel_motion_write(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_resource_read(
        const schedule::Instruction &instruction,
        ::llvm::Value *lane_affine_seed = nullptr,
        ::llvm::Value *operand_sanitization_mask = nullptr);
    void _resource_write(const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_resource_query(
        const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_collective(
        const schedule::Instruction &instruction);
    void _print(const schedule::Instruction &instruction);
    void _assert(const schedule::Instruction &instruction);
    [[nodiscard]] ::llvm::Value *_clock(
        const schedule::Instruction &instruction);
    void _store_debug_data(
        ::llvm::Value *base, const Type *type,
        ::llvm::Value *value, size_t offset = 0u);
    void _emit_instruction(
        const schedule::Instruction &instruction,
        ::llvm::Value *lane_affine_seed = nullptr,
        ::llvm::Value *operand_sanitization_mask = nullptr);
    void _assign(schedule::EdgeAssignment assignment,
                 ::llvm::Value *mask);
    void _apply_assignments(
        const std::vector<schedule::EdgeAssignment> &assignments,
        ::llvm::Value *mask);
    [[nodiscard]] ::llvm::Value *_zero_mask() noexcept;
    [[nodiscard]] ::llvm::Value *_splat(::llvm::Value *scalar);
    [[nodiscard]] ::llvm::Value *_safe_first_lane(::llvm::Value *mask);
    [[nodiscard]] ::llvm::Value *_frame_bit(::llvm::Value *index);
    [[nodiscard]] ::llvm::Value *_frame_is_active(
        ::llvm::Value *active_bits, ::llvm::Value *index);
    [[nodiscard]] ::llvm::Value *_frame_mask_pointer(
        ::llvm::AllocaInst *frames, ::llvm::Value *index);
    [[nodiscard]] ::llvm::Value *_load_frame_metadata(
        ::llvm::AllocaInst *frames, ::llvm::Value *index);
    void _store_frame_metadata(
        ::llvm::AllocaInst *frames, ::llvm::Value *index,
        ::llvm::Value *value);
    [[nodiscard]] ::llvm::Value *_load_convergence_target(
        ::llvm::Value *static_id);
    [[nodiscard]] ::llvm::Value *_ready_element_pointer(
        ::llvm::AllocaInst *array, ::llvm::Value *index);
    void _trap_if(::llvm::Value *condition, std::string_view label);
    void _declare_convergence(schedule::ConvergenceId convergence,
                              ::llvm::Value *divergent);
    [[nodiscard]] ::llvm::Value *_arrive_at_convergence_target(
        ::llvm::Value *target, ::llvm::Value *flow,
        ::llvm::Value **matched);
    [[nodiscard]] ::llvm::Value *_cascade_at_convergence_target(
        ::llvm::Value *target, ::llvm::Value *flow);
    void _resume(::llvm::Value *target, ::llvm::Value *mask,
                 ::llvm::Value *token);
    void _resume(::llvm::Value *target, ::llvm::Value *mask);
    void _resume(schedule::BlockId target, ::llvm::Value *mask);
    [[nodiscard]] ::llvm::Value *_route_edge(
        const schedule::ControlEdge &edge, ::llvm::Value *mask);
    void _advance_cooperative_loop_epoch(
        schedule::LoopId loop, ::llvm::Value *mask);
    void _continue_at(
        schedule::BlockId target, ::llvm::Value *mask);
    void _emit_arrival(const schedule::ControlEdge &edge,
                       ::llvm::Value *mask);
    void _begin_cooperative_coroutine();
    [[nodiscard]] ::llvm::Value *_cooperative_barrier_slot();
    [[nodiscard]] ::llvm::Value *_cooperative_loop_epoch_slot(
        uint32_t loop_epoch_index);
    void _publish_cooperative_loop_epochs(uint32_t barrier_id);
    void _initialize_cooperative_packet(::llvm::Value *initial_mask);
    void _emit_block_barrier(
        const schedule::BlockBarrierTerminator &barrier);
    void _finish_entry();
    [[nodiscard]] std::optional<PredicatedMemoryDiamond>
    _find_predicated_memory_diamond(
        const schedule::BasicBlock &block) const noexcept;
    void _emit_predicated_memory_diamond(
        const schedule::BasicBlock &block,
        const schedule::SplitTerminator &control,
        const PredicatedMemoryDiamond &diamond,
        const std::vector<::llvm::BasicBlock *> *direct_blocks);
    [[nodiscard]] const schedule::Loop *_innermost_loop_containing(
        schedule::BlockId block) const noexcept;
    [[nodiscard]] std::optional<GuardedPredicatedMathDiamond>
    _find_guarded_predicated_math_diamond(
        const schedule::BasicBlock &block) const noexcept;
    void _emit_guarded_predicated_math_diamond(
        const schedule::SplitTerminator &control,
        const GuardedPredicatedMathDiamond &diamond,
        bool continue_at_merge = true);
    [[nodiscard]] std::optional<NestedPredicatedRegion>
    _find_nested_predicated_region(
        const schedule::BasicBlock &block) const noexcept;
    void _emit_nested_predicated_region(
        const schedule::SplitTerminator &control,
        const NestedPredicatedRegion &region,
        bool continue_at_merge = true);
    [[nodiscard]] std::optional<ChainedPredicatedRegion>
    _find_chained_predicated_region(
        const schedule::BasicBlock &block) const noexcept;
    void _emit_chained_predicated_region(
        const schedule::SplitTerminator &control,
        const ChainedPredicatedRegion &region,
        bool continue_at_merge = true);
    [[nodiscard]] std::optional<PredicatedLoop>
    _find_predicated_loop(
        const schedule::BasicBlock &header) const noexcept;
    void _emit_predicated_loop(const PredicatedLoop &loop);
    [[nodiscard]] std::optional<StructuredEarlyExitLoop>
    _find_structured_early_exit_loop(
        const schedule::BasicBlock &header) const noexcept;
    void _emit_structured_early_exit_loop(
        const StructuredEarlyExitLoop &loop);
    [[nodiscard]] std::optional<CoherentAllOnRegion>
    _find_coherent_all_on_region(
        const schedule::SplitTerminator &control,
        const schedule::ControlEdge &entry_edge) const noexcept;
    void _emit_coherent_all_on_region(
        const schedule::ControlEdge &entry_edge,
        const CoherentAllOnRegion &region);
    void _emit_terminator(
        const schedule::BasicBlock &block,
        bool allow_all_on_region_versioning = true);
    void _emit_direct_terminator(
        const schedule::BasicBlock &block,
        const std::vector<::llvm::BasicBlock *> &blocks);
    [[nodiscard]] bool _can_emit_direct_control_flow() const noexcept;
    void _find_instruction_spills();
    void _coalesce_state_slots();
    void _allocate_state();
    void _partition_state_residency();
    void _build_direct(::llvm::Value *initial_mask);
    void _build();

public:
    ScheduleEmitter(::llvm::Module &module,
                    const schedule::Function &source, uint32_t width,
                    std::string_view entry_name,
                    bool enable_fast_math,
                    std::array<uint32_t, 3u> static_block_size,
                    bool enable_uniform_buffer_broadcast,
                    bool enable_lane_affine_buffer,
                    bool enable_paired_leaf_gather,
                    uint32_t dispatch_worker_count,
                    bool enable_native_predicated_loop,
                    bool enable_runtime_packet_geometry,
                    bool enable_linear_1d_packet_tail_narrowing,
                    ScheduleEntryABI entry_abi = ScheduleEntryABI::packet,
                    std::span<const LLVMSIMDRayQueryPipelineHandlers>
                        ray_query_pipeline_handlers = {},
                    size_t print_format_id_base = 0u);
    [[nodiscard]] LLVMScheduleCodegenResult run();
};

}// namespace luisa::compute::simd::detail
