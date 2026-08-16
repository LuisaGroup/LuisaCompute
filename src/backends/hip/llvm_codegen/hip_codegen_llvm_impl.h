//
// Created by mike on 3/18/26.
//

#pragma once

#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Target/TargetMachine.h>

#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/ray_query.h>

#include "hip_codegen_llvm.h"
#include "hip_callable_abi.h"

namespace luisa::compute::hip {

class HIPCodegenLLVMImpl {

public:
    static constexpr auto amdgpu_target_triple = "amdgcn-amd-amdhsa";
    static constexpr auto amdgpu_address_space_global = 1u;
    static constexpr auto amdgpu_address_space_shared = 3u;
    static constexpr auto amdgpu_address_space_constant = 4u;
    static constexpr auto amdgpu_address_space_local = 5u;
    struct LLVMTypeInfo {
        llvm::Type *mem_type;
        llvm::Type *reg_type;
        luisa::vector<size_t> member_indices;
        luisa::vector<size_t> member_offsets;
    };

    struct KernelArgumentStruct {
        static constexpr auto argument_alignment = 16u;
        llvm::StructType *llvm_type;
        std::vector<size_t> argument_indices;
        size_t print_buffer_index{0};
        bool has_print_buffer{false};
        size_t dispatch_size_and_kernel_id_index;
        size_t rt_global_stack_buffer_index{0};
        bool has_rt_global_stack_buffer{false};
    };

    using IB = llvm::IRBuilder<>;

    struct FunctionContext {
        llvm::Function *llvm_func;
        llvm::BasicBlock *llvm_alloca_block;
        llvm::BasicBlock *llvm_entry_block;
        llvm::Value *llvm_dispatch_size{nullptr};
        llvm::Value *llvm_kernel_id{nullptr};
        llvm::Value *llvm_print_buffer_capacity{nullptr};
        llvm::Value *llvm_print_buffer_content{nullptr};
        // RT global stack buffer fields (only set for RT-enabled kernels/callables)
        llvm::Value *llvm_rt_stack_size{nullptr};
        llvm::Value *llvm_rt_stack_count{nullptr};
        llvm::Value *llvm_rt_stack_data{nullptr};
        llvm::Value *llvm_rq_state{nullptr};// Ray query per-thread state pointer (alloca)
        // All function-local query objects share llvm_rq_state. This bit is
        // therefore a property of the complete FunctionContext, never of an
        // individual RayQueryPipelineInst lowered within it.
        bool llvm_rq_state_uses_resumable_abi{false};
        llvm::DenseMap<const xir::Value *, llvm::Value *> local_values;
        // A logical XIR block may lower to a single-entry/single-exit LLVM
        // region (for example, a floating-point atomic RMW expands to a CAS
        // loop). Branch targets use the region entry stored in local_values;
        // PHI incoming edges must use the region exit stored here.
        llvm::DenseMap<const xir::BasicBlock *, llvm::BasicBlock *>
            llvm_exit_blocks;
        std::vector<const xir::PhiInst *> pending_phi_nodes;

        explicit FunctionContext(llvm::Function *f) noexcept;

        template<typename T = llvm::Value>
            requires std::derived_from<T, llvm::Value>
        [[nodiscard]] T *get_local_value(const xir::Value *v) const noexcept {
            LUISA_DEBUG_ASSERT(v != nullptr, "Value is null.");
            auto iter = local_values.find(v);
            LUISA_DEBUG_ASSERT(iter != local_values.end() && llvm::isa<T>(iter->second), "Local value not found.");
            return llvm::cast<T>(iter->second);
        }
    };

    struct RayTracingAnalysis {
        bool uses_ray_tracing = false;
        bool uses_ray_query = false;
        bool uses_motion_blur = false;
        bool uses_static_trace = false;
        bool uses_motion_ray_query = false;
        // A RayQueryPipelineInst is the semantic boundary produced by
        // lower_ray_query_loop for Query::trace(): candidate handlers execute
        // synchronously and the traversal state cannot escape between them.
        // Explicit loop/dispatch/proceed instructions retain resumable query
        // semantics and therefore disqualify the module-wide compact ABI.
        bool uses_ray_query_pipeline = false;
        bool uses_resumable_ray_query_control = false;
        // A kernel-reachable device opacity write makes the packed HIPRT
        // instance-node copy stale until the next scene build. Such kernels
        // must retain exact per-candidate reads from CodegenInstance.
        bool writes_instance_opacity = false;
        // Hardware LDS traversal stacks are lane-local but not reentrant. A
        // trace issued by a candidate handler would overwrite the suspended
        // outer frontier, so such pipelines must retain the software path.
        bool ray_query_pipeline_handler_uses_ray_tracing = false;
    };

    struct PrintInfo {
        const Type *type;
        uint32_t index;
    };

    // A synchronous RayQuery callback ABI is initially emitted as the product
    // QueryIdentity x UserEnvironment. QueryIdentity is intrinsic traversal
    // state and is transported separately by the native wrapper. Once every
    // Callable body has been translated, unused fields can be projected from
    // UserEnvironment exactly. Keep the construction sites here until that
    // proof is available; the finalizer rewrites both producer stores and
    // consumer loads to the same compact product type.
    struct RayQueryPipelineContext {
        uint32_t pipeline_index;
        const xir::Function *parent_function;
        llvm::AllocaInst *storage;
        llvm::Value *generic_storage;
        llvm::CallInst *trace_call;
        llvm::Function *on_surface;
        llvm::Function *on_procedural;
        // True exactly when the parent function can observe query state after
        // the synchronous pipeline.
        bool post_state_observed;
        // True when either handler observes committed state or either public
        // ray representation. Such a handler requires an observable query
        // transaction instead of the candidate-only
        // {candidate, commit, terminate} action product.
        bool full_candidate_state_observed;
        // True when the XIR handler pair and the query's only observable
        // post-state form a closest-hit reduction. This permits HIPRT to run
        // the handlers as native intersection/filter callbacks and return one
        // final hit, instead of exposing a resumable candidate frontier.
        bool native_closest_reduction;
        // True when one handler domain observes both the immutable world ray
        // and the candidate-dependent object ray. The compact synchronous ABI
        // has one ray field, so this observation product requires the exact
        // resumable representation with two simultaneously live rays.
        bool distinct_ray_states_required;
        luisa::vector<llvm::StoreInst *> stores;
        luisa::vector<llvm::LoadInst *> loads;
        // The compact candidate transaction decodes the same projected user
        // environment but constructs query identity locally. Index zero is
        // therefore null; every other entry mirrors `loads`.
        luisa::vector<llvm::LoadInst *> compact_loads;
    };

    struct RayQueryPipelineProjectionInfo {
        size_t maximum_context_bytes{0u};
        size_t maximum_budget_constrained_context_bytes{0u};
        size_t oversized_compact_handler_only_pipeline_count{0u};
        luisa::vector<const xir::Function *>
            exact_state_required_functions;
        luisa::vector<const xir::Function *>
            oversized_budget_constrained_state_functions;
    };

    static constexpr auto llvm_buffer_type_ptr_index = 0;
    static constexpr auto llvm_buffer_type_size_index = 1;

    static constexpr auto llvm_texture_type_handle_index = 0;
    static constexpr auto llvm_texture_type_descriptor_index = 1;

    static constexpr auto llvm_bindless_array_type_slots_index = 0;
    static constexpr auto llvm_bindless_array_type_size_index = 1;
    static constexpr auto llvm_bindless_array_type_samplers_index = 2;

    static constexpr auto llvm_bindless_array_slot_type_buffer_ptr_index = 0;
    static constexpr auto llvm_bindless_array_slot_type_buffer_size_index = 1;
    static constexpr auto llvm_bindless_array_slot_type_texture2d_handle_index = 2;
    static constexpr auto llvm_bindless_array_slot_type_texture2d_levels_index = 3;
    static constexpr auto llvm_bindless_array_slot_type_texture2d_size_index = 4;
    static constexpr auto llvm_bindless_array_slot_type_texture3d_handle_index = 5;
    static constexpr auto llvm_bindless_array_slot_type_texture3d_levels_index = 6;
    static constexpr auto llvm_bindless_array_slot_type_texture3d_size_xy_index = 7;
    static constexpr auto llvm_bindless_array_slot_type_texture3d_size_z_index = 8;

    static constexpr auto llvm_accel_type_handle_index = 0;
    static constexpr auto llvm_accel_type_instances_index = 1;

    static constexpr auto llvm_accel_instance_type_affine_index = 0;
    static constexpr auto llvm_accel_instance_type_user_id_index = 1;
    static constexpr auto llvm_accel_instance_type_sbt_offset_index = 2;
    static constexpr auto llvm_accel_instance_type_mask_index = 3;
    static constexpr auto llvm_accel_instance_type_flags_index = 4;
    static constexpr auto llvm_accel_instance_type_handle_index = 5;
    static constexpr auto llvm_accel_instance_type_motion_data_index = 6;
    static constexpr auto llvm_accel_instance_visibility_mask_bits = 0xffu;
    static constexpr auto llvm_accel_instance_packed_opacity_bit = 1u << 31u;

    static constexpr auto llvm_ray_type_origin_index = 0;
    static constexpr auto llvm_ray_type_t_min_index = 1;
    static constexpr auto llvm_ray_type_direction_index = 2;
    static constexpr auto llvm_ray_type_t_max_index = 3;

    static constexpr auto llvm_surface_hit_type_inst_id_index = 0;
    static constexpr auto llvm_surface_hit_type_prim_id_index = 1;
    static constexpr auto llvm_surface_hit_type_bary_index = 2;
    static constexpr auto llvm_surface_hit_type_t_index = 3;

    static constexpr auto llvm_procedural_hit_type_inst_id_index = 0;
    static constexpr auto llvm_procedural_hit_type_prim_id_index = 1;

    static constexpr auto llvm_committed_hit_type_inst_id_index = 0;
    static constexpr auto llvm_committed_hit_type_prim_id_index = 1;
    static constexpr auto llvm_committed_hit_type_bary_index = 2;
    static constexpr auto llvm_committed_hit_type_hit_kind_index = 3;
    static constexpr auto llvm_committed_hit_type_t_index = 4;

    // A HIP RayQuery value is an opaque token whose complete semantics are the
    // identity of its per-invocation traversal state. AMDGPU private pointers
    // are 32-bit in the target data layout, so no wider source-layout surrogate
    // is required inside generated LLVM.
    static constexpr auto llvm_ray_query_state_address_bits = 32u;

    static constexpr auto llvm_ray_query_state_surface_terminated = 0;
    static constexpr auto llvm_ray_query_state_surface_candidate = 1;
    static constexpr auto llvm_ray_query_state_procedural_candidate = 2;
    static constexpr auto llvm_ray_query_state_custom_candidate = 3;

    // Sound over-approximate observations made by synchronous candidate
    // handler call graphs. These bits match the native wrapper ABI.
    static constexpr auto llvm_ray_query_observes_committed_hit = 1u << 0u;
    static constexpr auto llvm_ray_query_observes_world_ray = 1u << 1u;
    static constexpr auto llvm_ray_query_observes_object_ray = 1u << 2u;
    static constexpr auto llvm_ray_query_handler_observation_mask =
        llvm_ray_query_observes_committed_hit |
        llvm_ray_query_observes_world_ray |
        llvm_ray_query_observes_object_ray;
    // One trace argument carries two independent handler observations. The
    // separation is semantic, not merely an optimization: a procedural object
    // ray must not make an unrelated surface callback project that ray into
    // the public query state.
    static constexpr auto llvm_ray_query_procedural_observation_shift = 8u;

    static constexpr std::string_view llvm_ray_query_intrinsic_name_world_space_ray = "luisa_ray_query_world_space_ray";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_procedural_candidate_hit = "luisa_ray_query_procedural_candidate_hit";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_surface_candidate_hit = "luisa_ray_query_surface_candidate_hit";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_committed_hit = "luisa_ray_query_committed_hit";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_is_surface_candidate = "luisa_ray_query_is_surface_candidate";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_is_procedural_candidate = "luisa_ray_query_is_procedural_candidate";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_is_terminated = "luisa_ray_query_is_terminated";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_commit_surface_hit = "luisa_ray_query_commit_surface_hit";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_commit_procedural_hit = "luisa_ray_query_commit_procedural_hit";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_state = "luisa_ray_query_state";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_initialize = "luisa_ray_query_initialize";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_spawn = "luisa_ray_query_spawn";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_proceed = "luisa_ray_query_proceed";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_advance = "luisa_ray_query_advance";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_dispatch = "luisa_ray_query_dispatch";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_terminate = "luisa_ray_query_terminate";

    // Scalar-returning accessors (avoid AMDGPU addrspace aliasing with output pointers)
    static constexpr std::string_view llvm_ray_query_intrinsic_name_candidate_inst_id = "luisa_ray_query_candidate_inst_id";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_candidate_prim_id = "luisa_ray_query_candidate_prim_id";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_candidate_bary_u = "luisa_ray_query_candidate_bary_u";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_candidate_bary_v = "luisa_ray_query_candidate_bary_v";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_candidate_hit_t = "luisa_ray_query_candidate_hit_t";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_committed_inst_id = "luisa_ray_query_committed_inst_id";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_committed_prim_id = "luisa_ray_query_committed_prim_id";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_committed_bary_u = "luisa_ray_query_committed_bary_u";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_committed_bary_v = "luisa_ray_query_committed_bary_v";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_committed_hit_kind = "luisa_ray_query_committed_hit_kind";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_committed_hit_t = "luisa_ray_query_committed_hit_t";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_ray_origin_x = "luisa_ray_query_ray_origin_x";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_ray_origin_y = "luisa_ray_query_ray_origin_y";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_ray_origin_z = "luisa_ray_query_ray_origin_z";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_ray_tmin = "luisa_ray_query_ray_tmin";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_ray_direction_x = "luisa_ray_query_ray_direction_x";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_ray_direction_y = "luisa_ray_query_ray_direction_y";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_ray_direction_z = "luisa_ray_query_ray_direction_z";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_ray_tmax = "luisa_ray_query_ray_tmax";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_object_ray_origin_x = "luisa_ray_query_object_ray_origin_x";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_object_ray_origin_y = "luisa_ray_query_object_ray_origin_y";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_object_ray_origin_z = "luisa_ray_query_object_ray_origin_z";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_object_ray_tmin = "luisa_ray_query_object_ray_tmin";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_object_ray_direction_x = "luisa_ray_query_object_ray_direction_x";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_object_ray_direction_y = "luisa_ray_query_object_ray_direction_y";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_object_ray_direction_z = "luisa_ray_query_object_ray_direction_z";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_object_ray_tmax = "luisa_ray_query_object_ray_tmax";

private:
    HIPCodegenLLVMConfig _config;
    llvm::TargetMachine *_target_machine{nullptr};
    std::unique_ptr<llvm::DataLayout> _data_layout;
    llvm::LLVMContext _llvm_context;
    std::unique_ptr<llvm::Module> _llvm_module;
    bool _supports_hardware_rt_stack{false};
    bool _uses_hardware_rt_stack{false};
    bool _uses_synchronous_ray_query_pipeline{false};
    bool _uses_mixed_ray_query_pipeline{false};
    bool _uses_iterative_synchronous_ray_query_pipeline{false};
    bool _uses_resumable_hardware_ray_query_pipeline{false};
    bool _uses_native_closest_ray_query_pipeline{false};
    bool _uses_static_global_rt_stack{false};
    bool _requires_global_rt_stack{false};
    luisa::vector<const xir::Function *>
        _retry_with_resumable_ray_query_state_functions;

    RayTracingAnalysis _rt_analysis;

    llvm::Type *_llvm_buffer_type{nullptr};
    llvm::Type *_llvm_print_buffer_type{nullptr};
    llvm::Type *_llvm_texture_type{nullptr};
    llvm::Type *_llvm_bindless_array_type{nullptr};
    llvm::Type *_llvm_bindless_array_slot_type{nullptr};
    llvm::Type *_llvm_accel_type{nullptr};
    llvm::Type *_llvm_accel_instance_type{nullptr};
    llvm::Type *_llvm_ray_type{nullptr};
    llvm::Type *_llvm_surface_hit_type{nullptr};
    llvm::Type *_llvm_procedural_hit_type{nullptr};
    llvm::Type *_llvm_committed_hit_type{nullptr};
    llvm::Type *_llvm_ray_query_type{nullptr};
    llvm::DenseMap<const Type *, luisa::unique_ptr<LLVMTypeInfo>> _xir_to_llvm_type;
    llvm::DenseMap<const xir::Value *, llvm::Constant *> _xir_to_llvm_global;
    llvm::DenseMap<const xir::KernelFunction *, luisa::unique_ptr<KernelArgumentStruct>> _kernel_arg_struct_types;
    luisa::unordered_map<const xir::PrintInst *, PrintInfo> _print_info;
    luisa::vector<std::pair<luisa::string, const Type *>> _print_formats;
    size_t _ray_query_pipeline_count{0u};
    llvm::Function *_llvm_ray_query_pipeline_dispatch{nullptr};
    llvm::SwitchInst *_llvm_ray_query_pipeline_switch{nullptr};
    llvm::Function *_llvm_ray_query_pipeline_compact_dispatch{nullptr};
    llvm::SwitchInst *_llvm_ray_query_pipeline_compact_switch{nullptr};
    llvm::Value *_llvm_ray_query_pipeline_compact_query{nullptr};
    llvm::BasicBlock *_llvm_ray_query_pipeline_compact_finish{nullptr};
    luisa::vector<RayQueryPipelineContext>
        _llvm_ray_query_pipeline_contexts;

    template<typename T = llvm::Value>
        requires std::derived_from<T, llvm::Value>
    [[nodiscard]] T *_get_llvm_value(IB &b, const FunctionContext &func_ctx, const xir::Value *v) noexcept {
        LUISA_DEBUG_ASSERT(v != nullptr, "Value is null.");
        auto checked_llvm_value = [](llvm::Value *llvm_v) noexcept {
            LUISA_DEBUG_ASSERT(llvm::isa<T>(llvm_v), "LLVM value type mismatch.");
            return static_cast<T *>(llvm_v);
        };
        switch (v->derived_value_tag()) {
            case xir::DerivedValueTag::UNDEFINED: return checked_llvm_value(llvm::UndefValue::get(_get_llvm_type(v->type())->reg_type));
            case xir::DerivedValueTag::FUNCTION: return checked_llvm_value(_get_or_declare_llvm_function(static_cast<const xir::Function *>(v)));
            case xir::DerivedValueTag::BASIC_BLOCK: return func_ctx.get_local_value<T>(v);
            case xir::DerivedValueTag::INSTRUCTION: return func_ctx.get_local_value<T>(v);
            case xir::DerivedValueTag::CONSTANT: return checked_llvm_value(_get_llvm_constant(b, static_cast<const xir::Constant *>(v)));
            case xir::DerivedValueTag::ARGUMENT: return func_ctx.get_local_value<T>(v);
            case xir::DerivedValueTag::SPECIAL_REGISTER: {
                auto sreg_tag = static_cast<const xir::SpecialRegister *>(v)->derived_special_register_tag();
                return checked_llvm_value(_read_special_register(b, func_ctx, sreg_tag));
            }
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported value type.");
    }

    template<typename T, typename... Extra>
    [[nodiscard]] auto _to_string(T *llvm_object, Extra &&...extra) const noexcept {
        std::string str;
        llvm::raw_string_ostream os{str};
        llvm_object->print(os, std::forward<Extra>(extra)...);
        os.flush();
        return str;
    }

private:
    void _initialize() noexcept;
    void _analyze_ray_tracing_usage(const xir::Module &module) noexcept;
    void _analyze_ray_tracing_in_function(
        const xir::Function *function,
        llvm::DenseSet<const xir::Function *> &visited) noexcept;
    [[nodiscard]] bool _function_uses_resumable_ray_query_state(
        const xir::Function *function) const noexcept;
    void _link_native_include() noexcept;
    void _specialize_oclc_options() noexcept;
    void _link_ockl_if_needed() noexcept;
    void _postprocess_rt_kernel() noexcept;
    [[nodiscard]] RayQueryPipelineProjectionInfo
    _finalize_ray_query_pipeline_contexts() noexcept;
    void _run_optimization_passes() noexcept;
    void _dump_module(const std::filesystem::path &path) const noexcept;
    [[nodiscard]] luisa::string _generate_code() const noexcept;

    [[nodiscard]] static size_t _get_type_alignment(const Type *type) noexcept;
    void _collect_print_info(const xir::Module &xir_module) noexcept;
    [[nodiscard]] const LLVMTypeInfo *_get_llvm_type(const Type *type) noexcept;
    [[nodiscard]] const KernelArgumentStruct *_get_kernel_argument_struct(const xir::KernelFunction *func) noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_buffer_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_print_buffer_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_texture_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_bindless_array_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_bindless_array_slot_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_accel_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_accel_instance_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_ray_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_surface_hit_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_procedural_hit_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_committed_hit_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_ray_query_type() noexcept;
    [[nodiscard]] std::pair<llvm::Value *, const Type *>
    _lower_access_chain_address(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_ptr,
                                const Type *type, luisa::span<const xir::Use *const> index_uses) noexcept;

    [[nodiscard]] llvm::Function *_get_or_declare_llvm_function(const xir::Function *func) noexcept;
    [[nodiscard]] llvm::Function *_declare_llvm_kernel_function(const xir::KernelFunction *func) noexcept;
    [[nodiscard]] llvm::Function *_declare_llvm_callable_function(const xir::CallableFunction *func) noexcept;
    [[nodiscard]] llvm::Function *_declare_llvm_external_function(const xir::ExternalFunction *func) noexcept;
    [[nodiscard]] llvm::Function *_translate_function(const xir::FunctionDefinition *func) noexcept;
    [[nodiscard]] llvm::Function *_translate_kernel_function(const xir::KernelFunction *func) noexcept;
    [[nodiscard]] llvm::Function *_translate_callable_function(const xir::CallableFunction *func) noexcept;
    [[nodiscard]] llvm::BasicBlock *_translate_function_definition(FunctionContext &func_ctx, const xir::FunctionDefinition *f) noexcept;
    static void _mark_llvm_function_as_pure(llvm::Function *func) noexcept;
    [[nodiscard]] llvm::Function *_get_assert_function() noexcept;
    [[nodiscard]] llvm::Function *_get_vprintf_function() noexcept;
    [[nodiscard]] llvm::Function *_get_texture2d_read_function(llvm::VectorType *llvm_value_type) noexcept;
    [[nodiscard]] llvm::Function *_get_texture2d_write_function(llvm::VectorType *llvm_value_type) noexcept;
    [[nodiscard]] llvm::Function *_get_texture3d_read_function(llvm::VectorType *llvm_value_type) noexcept;
    [[nodiscard]] llvm::Function *_get_texture3d_write_function(llvm::VectorType *llvm_value_type) noexcept;
    [[nodiscard]] llvm::InlineAsm *_get_inline_asm(std::string_view asm_string, std::string_view constraints, bool has_side_effects) noexcept;

    [[nodiscard]] llvm::Value *_get_llvm_literal(IB &b, const Type *type, const void *data) noexcept;
    [[nodiscard]] llvm::Value *_get_llvm_constant(IB &b, const xir::Constant *c, bool load_global = true) noexcept;

    [[nodiscard]] llvm::Value *_read_special_register(IB &b, const FunctionContext &func_ctx, xir::DerivedSpecialRegisterTag tag) noexcept;
    [[nodiscard]] llvm::Value *_read_block_id(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_block_size(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_thread_id(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_dispatch_size(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_dispatch_id(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_warp_size(IB &b, const FunctionContext &func_ctx) const noexcept;
    [[nodiscard]] llvm::Value *_read_warp_lane_id(IB &b, const FunctionContext &func_ctx) const noexcept;
    [[nodiscard]] llvm::Value *_read_warp_active_lane_mask(IB &b) const noexcept;
    [[nodiscard]] llvm::Value *_read_warp_prefix_lane_mask(IB &b, const FunctionContext &func_ctx) const noexcept;
    [[nodiscard]] static llvm::Value *_read_kernel_id(IB &b, const FunctionContext &func_ctx) noexcept;

    [[nodiscard]] llvm::Value *_convert_llvm_reg_value_to_mem(IB &b, llvm::Value *reg_v, const Type *type) noexcept;
    [[nodiscard]] llvm::Value *_convert_llvm_mem_value_to_reg(IB &b, llvm::Value *mem_v, const Type *type) noexcept;
    [[nodiscard]] llvm::Value *_bitwise_cast(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_static_cast(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_static_cast_scalar_to_scalar(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_static_cast_scalar_to_vector(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_static_cast_vector_to_vector(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_texel_cast(IB &b, llvm::Value *llvm_src, llvm::Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_unpack_r10g10b10a2(
        IB &b, llvm::Value *packed, llvm::VectorType *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_pack_r10g10b10a2(
        IB &b, llvm::Value *value) noexcept;
    [[nodiscard]] llvm::Value *_safe_fp_cast(IB &b, llvm::Value *llvm_src, llvm::Type *dst_type, const llvm::Twine &name = "") const noexcept;

    [[nodiscard]] static llvm::Value *_create_llvm_vector(IB &b, llvm::ArrayRef<llvm::Value *> elems) noexcept;
    void _translate_instruction(IB &b, FunctionContext &func_ctx, const xir::Instruction *inst) noexcept;

    void _translate_if_inst(IB &b, const FunctionContext &func_ctx, const xir::IfInst *inst) noexcept;
    void _translate_switch_inst(IB &b, const FunctionContext &func_ctx, const xir::SwitchInst *inst) noexcept;
    static void _translate_loop_inst(IB &b, const FunctionContext &func_ctx, const xir::LoopInst *inst) noexcept;
    static void _translate_simple_loop_inst(IB &b, const FunctionContext &func_ctx, const xir::SimpleLoopInst *inst) noexcept;
    static void _translate_branch_inst(IB &b, const FunctionContext &func_ctx, const xir::BranchInst *inst) noexcept;
    void _translate_conditional_branch_inst(IB &b, const FunctionContext &func_ctx, const xir::ConditionalBranchInst *inst) noexcept;
    static void _translate_unreachable_inst(IB &b, FunctionContext &func_ctx, const xir::UnreachableInst *inst) noexcept;
    static void _translate_break_inst(IB &b, const FunctionContext &func_ctx, const xir::BreakInst *inst) noexcept;
    static void _translate_continue_inst(IB &b, const FunctionContext &func_ctx, const xir::ContinueInst *inst) noexcept;
    void _translate_return_inst(IB &b, const FunctionContext &func_ctx, const xir::ReturnInst *inst) noexcept;

    [[nodiscard]] llvm::PHINode *_translate_phi_inst(IB &b, FunctionContext &func_ctx, const xir::PhiInst *inst) noexcept;
    void _finalize_pending_phi_nodes(const FunctionContext &func_ctx, const luisa::unordered_set<const xir::BasicBlock *> &translated_blocks) noexcept;

    [[nodiscard]] llvm::Value *_translate_alloca_inst(IB &b, FunctionContext &func_ctx, const xir::AllocaInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_load_inst(IB &b, const FunctionContext &func_ctx, const xir::LoadInst *inst) noexcept;
    void _translate_store_inst(IB &b, const FunctionContext &func_ctx, const xir::StoreInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_gep_inst(IB &b, FunctionContext &func_ctx, const xir::GEPInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_load_llvm_value(IB &b, llvm::Value *llvm_ptr, const Type *type, bool is_volatile = false) noexcept;
    void _store_llvm_value(IB &b, llvm::Value *llvm_ptr, llvm::Value *llvm_value, const Type *type, bool is_volatile = false) noexcept;
    [[nodiscard]] static llvm::Value *_create_temp_in_alloca_block(const FunctionContext &func_ctx, llvm::Type *t, size_t align = 0) noexcept;

    [[nodiscard]] llvm::Value *_translate_atomic_inst(IB &b, FunctionContext &func_ctx, const xir::AtomicInst *inst) noexcept;

    [[nodiscard]] llvm::Value *_translate_arithmetic_inst(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_call_ocml_unary_op(IB &b, llvm::StringRef op_name, llvm::Value *llvm_value) const noexcept;
    [[nodiscard]] llvm::Value *_call_ocml_binary_op(IB &b, llvm::StringRef op_name, llvm::Value *llvm_lhs, llvm::Value *llvm_rhs) noexcept;
    [[nodiscard]] llvm::Value *_translate_outer_product(IB &b, llvm::Value *lhs, llvm::Value *rhs) noexcept;
    [[nodiscard]] llvm::Value *_translate_matrix_multiply(IB &b, llvm::Value *lhs, llvm::Value *rhs) noexcept;
    [[nodiscard]] llvm::Value *_translate_matrix_determinant(IB &b, llvm::Value *m) noexcept;
    [[nodiscard]] static llvm::Value *_translate_matrix_transpose(IB &b, llvm::Value *m) noexcept;
    [[nodiscard]] llvm::Value *_translate_matrix_inverse(IB &b, llvm::Value *m) noexcept;
    [[nodiscard]] llvm::Value *_translate_aggregate(IB &b, const FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_shuffle(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_insert(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_extract(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept;

    [[nodiscard]] llvm::Value *_translate_thread_group_inst(IB &b, FunctionContext &func_ctx, const xir::ThreadGroupInst *inst) noexcept;

    [[nodiscard]] llvm::Value *_translate_resource_query_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceQueryInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_resource_read_inst(IB &b, const FunctionContext &func_ctx, const xir::ResourceReadInst *inst) noexcept;
    void _translate_resource_write_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceWriteInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_get_buffer_element_pointer(IB &b, llvm::Value *buffer, llvm::Value *index, size_t index_stride, size_t element_size) noexcept;
    [[nodiscard]] llvm::Value *_get_direct_texture_descriptor_pointer(IB &b, llvm::Value *texture) noexcept;
    [[nodiscard]] llvm::Value *_get_direct_texture_base_level(IB &b, llvm::Value *texture) noexcept;
    [[nodiscard]] llvm::Value *_get_direct_texture_storage(IB &b, llvm::Value *texture) noexcept;
    [[nodiscard]] llvm::Value *_sample_packed_r10g10b10a2(
        IB &b, llvm::Value *resource, llvm::Value *coord,
        llvm::ArrayRef<llvm::Value *> sizes, llvm::Value *filter,
        llvm::Value *address) noexcept;
    [[nodiscard]] llvm::Value *_sample_texture_level(
        IB &b, bool is_2d, llvm::Value *resource, llvm::Value *sampler,
        llvm::Value *coord, llvm::ArrayRef<llvm::Value *> sizes,
        llvm::Value *filter, llvm::Value *address,
        llvm::Value *is_packed_r10g10b10a2) noexcept;
    [[nodiscard]] llvm::Value *_get_bindless_array_slot_pointer(IB &b, llvm::Value *bindless_array, llvm::Value *slot_index) noexcept;
    [[nodiscard]] llvm::Value *_get_bindless_array_texture_storage(
        IB &b, llvm::Value *bindless_array,
        llvm::Value *slot_index, int dim) noexcept;
    [[nodiscard]] llvm::Value *_get_bindless_array_texture_handle(IB &b, llvm::Value *bindless_array,
                                                                  llvm::Value *slot_index, int dim,
                                                                  llvm::Value *level = nullptr) noexcept;
    [[nodiscard]] llvm::Value *_get_accel_instance_pointer(IB &b, llvm::Value *accel, llvm::Value *instance_index) noexcept;
    [[nodiscard]] llvm::Value *_get_accel_instance_motion_frame(
        IB &b, llvm::Value *accel, llvm::Value *instance_index,
        llvm::Value *key_index, AccelMotionMode expected_mode) noexcept;
    [[nodiscard]] llvm::Value *_load_accel_affine_matrix(IB &b, llvm::Value *affine_ptr) noexcept;
    static void _store_accel_affine_matrix(IB &b, llvm::Value *affine_ptr, llvm::Value *matrix) noexcept;
    void _set_accel_instance_opacity(IB &b, llvm::Value *accel, llvm::Value *instance_index, llvm::Value *is_opaque) noexcept;
    [[nodiscard]] llvm::Value *_accel_trace_closest(
        IB &b, const FunctionContext &func_ctx, llvm::Value *accel,
        llvm::Value *ray, llvm::Value *time, llvm::Value *mask) noexcept;
    [[nodiscard]] llvm::Value *_accel_trace_any(
        IB &b, const FunctionContext &func_ctx, llvm::Value *accel,
        llvm::Value *ray, llvm::Value *time, llvm::Value *mask) noexcept;

    void _translate_ray_query_loop_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryLoopInst *inst) noexcept;
    void _translate_ray_query_dispatch_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryDispatchInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_ray_query_object_read_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectReadInst *inst) noexcept;
    void _translate_ray_query_object_write_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectWriteInst *inst) noexcept;
    void _translate_ray_query_pipeline_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryPipelineInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_get_ray_query_state_pointer(IB &b, const FunctionContext &func_ctx, const xir::Value *query_object) noexcept;
    [[nodiscard]] llvm::Value *_advance_ray_query(IB &b, llvm::Value *llvm_state_ptr) noexcept;
    [[nodiscard]] llvm::Value *_call_ray_query_intrinsic(IB &b, llvm::Value *llvm_state_ptr, llvm::StringRef name, llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args, bool use_pipeline_abi) noexcept;
    [[nodiscard]] llvm::Value *_call_ray_query_intrinsic(IB &b, llvm::Value *llvm_state_ptr, llvm::StringRef name, llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args) noexcept;
    [[nodiscard]] llvm::Value *_call_ray_query_intrinsic(IB &b, FunctionContext &func_ctx, llvm::StringRef name, llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args) noexcept;
    [[nodiscard]] static llvm::Value *_create_opaque_float_barrier(IB &b, llvm::Value *val, const llvm::Twine &name) noexcept;

    [[nodiscard]] llvm::Value *_translate_cast_inst(IB &b, FunctionContext &func_ctx, const xir::CastInst *inst) noexcept;

    void _translate_print_inst(IB &b, FunctionContext &func_ctx, const xir::PrintInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_clock_inst(IB &b, FunctionContext &func_ctx, const xir::ClockInst *inst) noexcept;
    void _translate_debug_break_inst(IB &b, FunctionContext &func_ctx, const xir::DebugBreakInst *inst) noexcept;
    void _translate_assert_inst(IB &b, FunctionContext &func_ctx, const xir::AssertInst *inst) noexcept;
    void _translate_assume_inst(IB &b, FunctionContext &func_ctx, const xir::AssumeInst *inst) noexcept;
    void _create_assertion_with_message(IB &b, llvm::Value *cond, luisa::string_view message) noexcept;

    [[nodiscard]] llvm::Value *_translate_call_inst(IB &b, FunctionContext &func_ctx, const xir::CallInst *inst) noexcept;
    void _translate_outline_inst(IB &b, FunctionContext &func_ctx, const xir::OutlineInst *inst) noexcept;

public:
    explicit HIPCodegenLLVMImpl(HIPCodegenLLVMConfig config) noexcept;
    [[nodiscard]] luisa::string generate(const xir::Module &xir_module) noexcept;
    [[nodiscard]] bool requires_global_rt_stack() const noexcept {
        return _requires_global_rt_stack;
    }
    [[nodiscard]] bool uses_static_global_rt_stack() const noexcept {
        return _uses_static_global_rt_stack;
    }
    [[nodiscard]] const luisa::vector<const xir::Function *> &
    retry_with_resumable_ray_query_state_functions() const noexcept {
        return _retry_with_resumable_ray_query_state_functions;
    }
    [[nodiscard]] luisa::vector<std::pair<luisa::string, luisa::string>>
    take_print_formats() && noexcept;
};

}// namespace luisa::compute::hip
