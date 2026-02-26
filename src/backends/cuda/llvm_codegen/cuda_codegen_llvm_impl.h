//
// Created by mike on 9/19/25.
//

#pragma once

#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Target/TargetMachine.h>

#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>

#include "cuda_codegen_llvm_config.h"

namespace luisa::compute::cuda {

struct CUDACodegenLLVMConfig;

class CUDACodegenLLVMImpl {

public:
    static constexpr auto nvptx_target_triple = "nvptx64-nvidia-cuda";
    static constexpr auto nvptx_address_space_global = 1u;
    static constexpr auto nvptx_address_space_shared = 3u;
    static constexpr auto nvptx_address_space_constant = 4u;
    static constexpr auto nvptx_address_space_local = 5u;

    // This struct holds information about the LLVM types mapped from XIR types
    // mem_type is the type that has the same size and smaller or equal alignment as the XIR type for correct memory layout
    // reg_type is the type used in registers, which may be different from mem_type for i1/i64/f64 vector types and aggregate types that contain vector types
    struct LLVMTypeInfo {
        llvm::Type *mem_type;                // The LLVM type used in memory (with proper alignment)
        llvm::Type *reg_type;                // The LLVM type used in registers
        luisa::vector<size_t> member_indices;// For struct type, the mapping from member index to LLVM struct field index
        luisa::vector<size_t> member_offsets;// For struct type, the byte offset of each member
    };

    struct KernelArgumentStruct {
        static constexpr auto argument_alignment = 16u;// all arguments are aligned to 16 bytes
        llvm::StructType *llvm_type;
        std::vector<size_t> argument_indices;
        size_t dispatch_size_and_kernel_id_index;
    };

    using IB = llvm::IRBuilder<>;

    struct FunctionContext {

        llvm::Function *llvm_func;
        llvm::BasicBlock *llvm_alloca_block;
        llvm::BasicBlock *llvm_entry_block;
        llvm::Value *llvm_dispatch_size{nullptr};
        llvm::Value *llvm_kernel_id{nullptr};
        llvm::DenseMap<const xir::Value *, llvm::Value *> local_values;
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

    static constexpr auto nvvm_required_arch_exp2_f16 = 80;
    static constexpr auto nvvm_required_arch_tanh_f16 = 75;
    static constexpr auto nvvm_required_arch_match_all = 75;
    static constexpr auto nvvm_required_arch_redux_i32 = 80;
    static constexpr auto nvvm_required_arch_redux_f32 = 101;

    struct RayTracingAnalysis {
        bool uses_ray_tracing = false;
        bool uses_ray_query = false;
        bool uses_motion_blur = false;
        CurveBasisSet curve_basis_set = CurveBasisSet::make_none();
    };

    static constexpr auto llvm_buffer_type_ptr_index = 0;
    static constexpr auto llvm_buffer_type_size_index = 1;

    static constexpr auto llvm_texture_type_handle_index = 0;
    static constexpr auto llvm_texture_type_storage_index = 1;

    static constexpr auto llvm_bindless_array_type_slots_index = 0;
    static constexpr auto llvm_bindless_array_type_size_index = 1;

    static constexpr auto llvm_bindless_array_slot_type_buffer_ptr_index = 0;
    static constexpr auto llvm_bindless_array_slot_type_buffer_size_index = 1;
    static constexpr auto llvm_bindless_array_slot_type_texture2d_handle_index = 2;
    static constexpr auto llvm_bindless_array_slot_type_texture3d_handle_index = 3;

    static constexpr auto llvm_accel_type_handle_index = 0;
    static constexpr auto llvm_accel_type_instances_index = 1;

    static constexpr auto llvm_accel_instance_type_affine_index = 0;
    static constexpr auto llvm_accel_instance_type_user_id_index = 1;
    static constexpr auto llvm_accel_instance_type_sbt_offset_index = 2;
    static constexpr auto llvm_accel_instance_type_mask_index = 3;
    static constexpr auto llvm_accel_instance_type_flags_index = 4;
    static constexpr auto llvm_accel_instance_type_handle_index = 5;

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

    static constexpr auto llvm_ray_query_type_accel_index = 0;
    static constexpr auto llvm_ray_query_type_ray_index = 1;
    static constexpr auto llvm_ray_query_type_time_index = 2;
    static constexpr auto llvm_ray_query_type_mask_index = 3;
    static constexpr auto llvm_ray_query_type_flags_index = 4;

    static constexpr auto llvm_ray_query_state_surface_terminated = 0;
    static constexpr auto llvm_ray_query_state_surface_candidate = 1;
    static constexpr auto llvm_ray_query_state_procedural_candidate = 2;

    static constexpr std::string_view llvm_ray_query_intrinsic_name_world_space_ray = "luisa.ray.query.world.space.ray";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_procedural_candidate_hit = "luisa.ray.query.procedural.candidate.hit";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_surface_candidate_hit = "luisa.ray.query.surface.candidate.hit";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_committed_hit = "luisa.ray.query.committed.hit";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_is_surface_candidate = "luisa.ray.query.is.surface.candidate";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_is_procedural_candidate = "luisa.ray.query.is.procedural.candidate";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_is_terminated = "luisa.ray.query.is.terminated";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_commit_surface_hit = "luisa.ray.query.commit.surface.hit";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_commit_procedural_hit = "luisa.ray.query.commit.procedural.hit";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_state = "luisa.ray.query.state";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_initialize = "luisa.ray.query.initialize";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_spawn = "luisa.ray.query.spawn";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_proceed = "luisa.ray.query.proceed";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_dispatch = "luisa.ray.query.dispatch";
    static constexpr std::string_view llvm_ray_query_intrinsic_name_terminate = "luisa.ray.query.terminate";

private:
    CUDACodegenLLVMConfig _config;
    llvm::TargetMachine *_target_machine{nullptr};
    std::unique_ptr<llvm::DataLayout> _data_layout;
    llvm::LLVMContext _llvm_context;
    std::unique_ptr<llvm::Module> _llvm_module;

    RayTracingAnalysis _rt_analysis;

    llvm::Type *_llvm_buffer_type{nullptr};             // { ptr, i64 size } as defined in cuda_buffer.h
    llvm::Type *_llvm_texture_type{nullptr};            // { i64 handle, i64 storage } as defined in cuda_texture.h
    llvm::Type *_llvm_bindless_array_type{nullptr};     // { ptr slots, i64 capacity } as defined in cuda_bindless_array.h
    llvm::Type *_llvm_bindless_array_slot_type{nullptr};// { i64 buffer, i64 size, i64 tex2d, i64 tex3d } as defined in cuda_bindless_array.h
    llvm::Type *_llvm_accel_type{nullptr};              // { i64 handle, ptr instances } as defined in cuda_accel.h
    llvm::Type *_llvm_accel_instance_type{nullptr};     // { [ <4 x float> x 3 ] affine, i32 user_id, i32 sbt_offset, i32 mask, i32 flags, i64 handle, i32 padding } as defined in optix_api.h
    llvm::Type *_llvm_ray_type{nullptr};                // { [3 x float] origin, float t_min, [3 x float] direction, float t_max }
    llvm::Type *_llvm_surface_hit_type{nullptr};        // { i32 inst_id, i32 prim_id, <2 x float> bary, float t }
    llvm::Type *_llvm_procedural_hit_type{nullptr};     // { i32 inst_id, i32 prim_id }
    llvm::Type *_llvm_committed_hit_type{nullptr};      // { i32 inst_id, i32 prim_id, <2 x float> bary, i32 hit_kind, float t }
    llvm::DenseMap<const Type *, luisa::unique_ptr<LLVMTypeInfo>> _xir_to_llvm_type;
    llvm::DenseMap<const xir::Value *, llvm::Constant *> _xir_to_llvm_global;
    llvm::DenseMap<const xir::KernelFunction *, luisa::unique_ptr<KernelArgumentStruct>> _kernel_arg_struct_types;

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
    [[nodiscard]] static const llvm::Target *_get_nvptx_target() noexcept;
    void _initialize() noexcept;
    using LLVMModulePassManagerCallback = llvm::function_ref<void(llvm::ModulePassManager &)>;
    void _run_optimization_passes(LLVMModulePassManagerCallback callback = {}) noexcept;
    void _dump_module(const std::filesystem::path &path) const noexcept;
    [[nodiscard]] luisa::string _generate_ptx() const noexcept;

    /* the following methods are defined in cuda_codegen_llvm_impl_analysis.cpp */
    void _analyze_ray_tracing_usage(const xir::Module &module) noexcept;
    void _analyze_ray_tracing_in_function(const xir::Function *f, llvm::DenseSet<const xir::Function *> &visited) noexcept;

    /* the following methods are defined in cuda_codegen_llvm_impl_type.cpp */
    [[nodiscard]] static size_t _get_type_alignment(const Type *type) noexcept;
    [[nodiscard]] const LLVMTypeInfo *_get_llvm_type(const Type *type) noexcept;
    [[nodiscard]] const KernelArgumentStruct *_get_kernel_argument_struct(const xir::KernelFunction *func) noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_buffer_type() noexcept;
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

    /* the following methods are defined in cuda_codegen_llvm_impl_func.cpp */
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

    /* the following methods are defined in cuda_codegen_llvm_impl_const.cpp */
    [[nodiscard]] llvm::Value *_get_llvm_literal(IB &b, const Type *type, const void *data) noexcept;
    [[nodiscard]] llvm::Value *_get_llvm_constant(IB &b, const xir::Constant *c, bool load_global = true) noexcept;

    /* the following methods are defined in cuda_codegen_llvm_impl_sreg.cpp */
    [[nodiscard]] llvm::Value *_read_special_register(IB &b, const FunctionContext &func_ctx, xir::DerivedSpecialRegisterTag tag) noexcept;
    [[nodiscard]] llvm::Value *_read_block_id(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_block_size(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_thread_id(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_dispatch_size(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_dispatch_id(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_warp_size(IB &b, const FunctionContext &func_ctx) const noexcept;
    [[nodiscard]] llvm::Value *_read_warp_lane_id(IB &b, const FunctionContext &func_ctx) const noexcept;
    [[nodiscard]] static llvm::Value *_read_kernel_id(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_warp_active_lane_mask(IB &b) const noexcept;
    [[nodiscard]] llvm::Value *_read_warp_prefix_lane_mask(IB &b) const noexcept;

    /* the following methods are defined in cuda_codegen_llvm_impl_cast.cpp */
    [[nodiscard]] llvm::Value *_convert_llvm_reg_value_to_mem(IB &b, llvm::Value *reg_v, const Type *type) noexcept;
    [[nodiscard]] llvm::Value *_convert_llvm_mem_value_to_reg(IB &b, llvm::Value *mem_v, const Type *type) noexcept;
    [[nodiscard]] llvm::Value *_bitwise_cast(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_static_cast(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_static_cast_scalar_to_scalar(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_static_cast_scalar_to_vector(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_static_cast_vector_to_vector(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_texel_cast(IB &b, llvm::Value *llvm_src, llvm::Type *dst_type) noexcept;
    [[nodiscard]] llvm::Value *_safe_fp_cast(IB &b, llvm::Value *llvm_src, llvm::Type *dst_type, const llvm::Twine &name = "") const noexcept;// workaround OptiX's silly bug in fp cast

    /* the following methods are defined in cuda_codegen_llvm_impl_inst.cpp, if not otherwise specified */
    [[nodiscard]] static llvm::Value *_create_llvm_vector(IB &b, llvm::ArrayRef<llvm::Value *> elems) noexcept;
    void _translate_instruction(IB &b, FunctionContext &func_ctx, const xir::Instruction *inst) noexcept;

    // control flow instructions: if, switch, loop, simple_loop, branch, conditional_branch, unreachable, break, continue, return, raster_discard, defined in cuda_codegen_llvm_impl_cflow.cpp
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
    static void _translate_raster_discard_inst(IB &b, FunctionContext &func_ctx, const xir::RasterDiscardInst *inst) noexcept;

    // PHI nodes, defined in cuda_codegen_llvm_impl_phi.cpp
    [[nodiscard]] llvm::PHINode *_translate_phi_inst(IB &b, FunctionContext &func_ctx, const xir::PhiInst *inst) noexcept;
    void _finalize_pending_phi_nodes(const FunctionContext &func_ctx) noexcept;

    // variable instructions: alloca, load, store, gep, defined in cuda_codegen_llvm_impl_var.cpp
    [[nodiscard]] llvm::Value *_translate_alloca_inst(IB &b, FunctionContext &func_ctx, const xir::AllocaInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_load_inst(IB &b, const FunctionContext &func_ctx, const xir::LoadInst *inst) noexcept;
    void _translate_store_inst(IB &b, const FunctionContext &func_ctx, const xir::StoreInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_gep_inst(IB &b, FunctionContext &func_ctx, const xir::GEPInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_load_llvm_value(IB &b, llvm::Value *llvm_ptr, const Type *type) noexcept;
    void _store_llvm_value(IB &b, llvm::Value *llvm_ptr, llvm::Value *llvm_value, const Type *type) noexcept;
    [[nodiscard]] static llvm::Value *_create_temp_in_alloca_block(const FunctionContext &func_ctx, llvm::Type *t, size_t align = 0) noexcept;

    // atomic instructions, defined in cuda_codegen_llvm_impl_atomic.cpp
    [[nodiscard]] llvm::Value *_translate_atomic_inst(IB &b, FunctionContext &func_ctx, const xir::AtomicInst *inst) noexcept;

    // arithmetic instructions, defined in cuda_codegen_llvm_impl_arith.cpp
    [[nodiscard]] llvm::Value *_translate_arithmetic_inst(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_call_libdevice_unary_op(IB &b, llvm::StringRef op_name, llvm::Value *llvm_value) const noexcept;
    [[nodiscard]] llvm::Value *_call_libdevice_binary_op(IB &b, llvm::StringRef op_name, llvm::Value *llvm_lhs, llvm::Value *llvm_rhs) noexcept;
    [[nodiscard]] llvm::Value *_translate_outer_product(IB &b, llvm::Value *lhs, llvm::Value *rhs) noexcept;
    [[nodiscard]] llvm::Value *_translate_matrix_multiply(IB &b, llvm::Value *lhs, llvm::Value *rhs) noexcept;
    [[nodiscard]] llvm::Value *_translate_matrix_determinant(IB &b, llvm::Value *m) noexcept;
    [[nodiscard]] static llvm::Value *_translate_matrix_transpose(IB &b, llvm::Value *m) noexcept;
    [[nodiscard]] llvm::Value *_translate_matrix_inverse(IB &b, llvm::Value *m) noexcept;
    [[nodiscard]] llvm::Value *_translate_aggregate(IB &b, const FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_shuffle(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_insert(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_extract(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept;

    // thread group instructions, defined in cuda_codegen_llvm_impl_cta.cpp
    [[nodiscard]] llvm::Value *_translate_thread_group_inst(IB &b, FunctionContext &func_ctx, const xir::ThreadGroupInst *inst) noexcept;

    // resource instructions: resource_query, resource_read, resource_write, defined in cuda_codegen_llvm_impl_resource.cpp
    [[nodiscard]] llvm::Value *_translate_resource_query_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceQueryInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_resource_read_inst(IB &b, const FunctionContext &func_ctx, const xir::ResourceReadInst *inst) noexcept;
    void _translate_resource_write_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceWriteInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_get_buffer_element_pointer(IB &b, llvm::Value *buffer, llvm::Value *index, size_t index_stride, size_t element_size) noexcept;
    [[nodiscard]] llvm::Value *_get_bindless_array_slot_pointer(IB &b, llvm::Value *bindless_array, llvm::Value *slot_index) noexcept;
    [[nodiscard]] llvm::Value *_get_bindless_array_texture_handle(IB &b, llvm::Value *bindless_array, llvm::Value *slot_index, int dim) noexcept;
    [[nodiscard]] llvm::Value *_get_accel_instance_pointer(IB &b, llvm::Value *accel, llvm::Value *instance_index) noexcept;
    [[nodiscard]] llvm::Value *_get_accel_instance_motion_data(IB &b, llvm::Value *accel, llvm::Value *instance_index, llvm::Value *key_index, size_t stride) noexcept;
    void _set_accel_instance_opacity(IB &b, llvm::Value *accel, llvm::Value *instance_index, llvm::Value *is_opaque) noexcept;
    [[nodiscard]] llvm::Value *_load_accel_affine_matrix(IB &b, llvm::Value *affine_ptr) noexcept;
    static void _store_accel_affine_matrix(IB &b, llvm::Value *affine_ptr, llvm::Value *matrix) noexcept;
    [[nodiscard]] llvm::Value *_accel_trace_closest(IB &b, uint32_t flags, llvm::Value *accel, llvm::Value *ray, llvm::Value *time, llvm::Value *mask) noexcept;
    [[nodiscard]] llvm::Value *_accel_trace_any(IB &b, uint32_t flags, llvm::Value *accel, llvm::Value *ray, llvm::Value *time, llvm::Value *mask) noexcept;
    void _call_optix_trace(IB &b, uint32_t payload_type, uint32_t sbt_offset, uint32_t flags,
                           llvm::Value *accel, llvm::Value *ray, llvm::Value *time, llvm::Value *mask,
                           llvm::ArrayRef<llvm::Value *> registers) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_undef(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_hit_object_is_hit(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_hit_object_triangle_barycentrics(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_hit_object_curve_parameter(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_hit_object_instance_index(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_hit_object_primitive_index(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_hit_object_ray_t_max(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_hit_object_hit_kind(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_read_instance_index(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_read_primitive_index(IB &b) noexcept;
    void _call_optix_hit_object_reset(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_get_triangle_barycentrics(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_get_curve_parameter(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_get_hit_distance(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_get_hit_kind(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_get_world_space_ray(IB &b) noexcept;
    [[nodiscard]] llvm::Value *_call_optix_get_payload(IB &b, uint32_t index) noexcept;
    void _call_optix_set_payload_types(IB &b, uint32_t types) noexcept;
    void _call_optix_report_intersection(IB &b, llvm::Value *hit_kind, llvm::Value *t) noexcept;
    void _call_optix_ignore_intersection(IB &b) noexcept;
    void _call_optix_terminate_ray(IB &b) noexcept;

    // ray query instructions: ray_query_loop, ray_query_dispatch, ray_query_object_read, ray_query_object_write, ray_query_pipeline, defined in cuda_codegen_llvm_impl_rq.cpp
    void _translate_ray_query_loop_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryLoopInst *inst) noexcept;
    void _translate_ray_query_dispatch_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryDispatchInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_ray_query_object_read_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectReadInst *inst) noexcept;
    void _translate_ray_query_object_write_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectWriteInst *inst) noexcept;
    void _translate_ray_query_pipeline_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryPipelineInst *inst) noexcept;
    llvm::Value *_call_ray_query_intrinsic(IB &b, llvm::StringRef name, llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args) noexcept;
    void _lower_ray_query_intrinsics(llvm::Function *f) noexcept;
    void _materialize_ray_query_loops() noexcept;

    // autodiff instructions: autodiff_scope, autodiff_intrinsic, defined in cuda_codegen_llvm_impl_autodiff.cpp
    void _translate_autodiff_scope_inst(IB &b, FunctionContext &func_ctx, const xir::AutodiffScopeInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_autodiff_intrinsic_inst(IB &b, FunctionContext &func_ctx, const xir::AutodiffIntrinsicInst *inst) noexcept;

    // cast instruction, defined in cuda_codegen_llvm_impl_cast.cpp
    [[nodiscard]] llvm::Value *_translate_cast_inst(IB &b, FunctionContext &func_ctx, const xir::CastInst *inst) noexcept;

    // debug and profile instructions: print, clock, debug_break, assert, assume, defined in cuda_codegen_llvm_impl_debug.cpp
    void _translate_print_inst(IB &b, FunctionContext &func_ctx, const xir::PrintInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_clock_inst(IB &b, FunctionContext &func_ctx, const xir::ClockInst *inst) noexcept;
    void _translate_debug_break_inst(IB &b, FunctionContext &func_ctx, const xir::DebugBreakInst *inst) noexcept;
    void _translate_assert_inst(IB &b, FunctionContext &func_ctx, const xir::AssertInst *inst) noexcept;
    void _translate_assume_inst(IB &b, FunctionContext &func_ctx, const xir::AssumeInst *inst) noexcept;
    void _create_assertion_with_message(IB &b, llvm::Value *cond, luisa::string_view message) noexcept;

    // call and outline instruction, defined in cuda_codegen_llvm_impl_func.cpp
    [[nodiscard]] llvm::Value *_translate_call_inst(IB &b, FunctionContext &func_ctx, const xir::CallInst *inst) noexcept;
    void _translate_outline_inst(IB &b, FunctionContext &func_ctx, const xir::OutlineInst *inst) noexcept;

public:
    explicit CUDACodegenLLVMImpl(CUDACodegenLLVMConfig config) noexcept;
    [[nodiscard]] luisa::string generate(const xir::Module &xir_module) noexcept;
};

}// namespace luisa::compute::cuda
