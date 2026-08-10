#pragma once

#include "../../storage_buffer_metadata.h"

#include <luisa/core/binary_io.h>
#include <luisa/ast/function.h>
#include <luisa/vstl/common.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/core/string_scratch.h>
#include <SPIRV/SpvBuilder.h>
#include <luisa/runtime/rhi/resource.h>
#include "property.h"
#include <luisa/xir/op.h>
#include <luisa/ast/usage.h>
#include <luisa/xir/passes/uniformity_analysis.h>
#include "aggregate_index.h"
#include "atomic_buffer_plan.h"
#include "argument_usage.h"
#include "bindless_usage.h"
#include "control_flow_plan.h"
#include "kernel_argument_role.h"
#include "runtime_target_plan.h"
#include "target_features.h"

namespace lc::spirv {
using namespace luisa;
using namespace luisa::compute;

struct SpirvKernelArgumentLayoutPlan;

struct SpirvResult {
    using Properties = vstd::vector<Property>;
    std::vector<uint32_t> spv_bin;
    Properties properties;
    vstd::vector<std::pair<Variable, Usage>> argument_usages;
    // Parallel to argument_usages. Native Vulkan persists these exact roles
    // per argument so optional accel descriptors are never inferred by
    // greedily inspecting the following argument's property.
    vstd::vector<SpirvKernelArgumentRoleMask> argument_roles;
    vstd::vector<std::pair<vstd::string, luisa::compute::Type const *>> printers;
    luisa::vector<std::byte> constant_ubo_data;
    SpirvTargetFeatureMask required_target_features{};
    bool useTex2DBindless{false};
    bool useTex3DBindless{false};
    // Global unbounded buffer heap only. Per-array metadata is represented by
    // SPIRVBindlessBufferMetadata properties and may exist without this heap.
    bool useBufferBindless{false};
};
class SpirvCodegenEntry {

public:
    struct PrintInfo {
        const Type *type;
        size_t index;
    };
    using PrintFormatVector = luisa::vector<std::pair<luisa::string, const Type *>>;

private:
    StringScratch &_scratch;
    luisa::unique_ptr<spv::Builder> _builder_ptr;
    spv::SpvBuildLogger _logger;
    spv::Builder &_builder;// reference to *_builder_ptr

    luisa::unordered_map<const Type *, spv::Id> _type_map;
    luisa::unordered_map<const Type *, spv::Id> _sampled_image_type_map;
    luisa::unordered_map<const Type *, spv::Id> _storage_image_type_map;
    luisa::unordered_map<const xir::Value *, spv::Id> _value_map;
    luisa::unordered_map<const xir::Function *, spv::Function *> _function_map;
    // Immutable logical block entries. Once a function plan is bound, entries
    // are never replaced even if an instruction creates additional blocks.
    luisa::unordered_map<const xir::BasicBlock *, spv::Block *> _block_map;
    luisa::unordered_map<const xir::BasicBlock *, spv::Block *> _block_tail;
    luisa::unordered_set<const xir::BasicBlock *> _emitted_blocks;
    luisa::unordered_set<spv::Block *> _registered_physical_blocks;
    luisa::vector<spv::Block *> _synthetic_blocks;
    luisa::unique_ptr<ControlFlowPlan> _control_flow_plan;
    const xir::BasicBlock *_current_xir_block{nullptr};

    static constexpr size_t invalid_phi_node_index = ~size_t{0u};
    struct DeferredPhiNodeIncoming {
        size_t order{0u};
        const xir::Value *logical_value{nullptr};
        const xir::BasicBlock *logical_predecessor{nullptr};
        size_t child_node_index{invalid_phi_node_index};
        spv::Id resolved_value{spv::NoResult};
        spv::Id resolved_predecessor{spv::NoResult};
        bool resolved{false};
    };
    struct DeferredPhiNode {
        spv::Instruction *instruction{nullptr};
        spv::Block *block{nullptr};
        luisa::vector<DeferredPhiNodeIncoming> incomings;
    };
    struct DeferredPhi {
        luisa::vector<DeferredPhiNode> nodes;
        luisa::unordered_map<size_t, size_t> forwarding_node_indices;
        size_t result_node_index{invalid_phi_node_index};
    };
    struct DeferredPhiIncomingRef {
        const xir::PhiInst *phi{nullptr};
        size_t node_index{invalid_phi_node_index};
        size_t incoming_index{invalid_phi_node_index};
    };
    luisa::unordered_map<const xir::PhiInst *, DeferredPhi> _deferred_phis;
    luisa::unordered_map<const xir::BasicBlock *, luisa::vector<DeferredPhiIncomingRef>>
        _deferred_phi_incomings_by_predecessor;

    luisa::unordered_map<const xir::PrintInst *, PrintInfo> _print_info;
    PrintFormatVector _print_formats;
    luisa::vector<const xir::Instruction *> _control_flow_stack;
    bool _allow_indirect_dispatch;
    bool _requires_printing{false};
    SpirvResult::Properties _properties;
    luisa::vector<spv::Id> _property_ids;
    static constexpr size_t invalid_resource_property_index = ~size_t{0u};
    // A logical kernel resource may require multiple Vulkan descriptors. The
    // indices below refer to `_properties`; `_kernel_resource_property_id`
    // owns the separate `_property_ids` indexing convention.
    struct KernelResourceBinding {
        Type::Tag type_tag{};
        Usage usage{Usage::NONE};
        bool requires_accel_traversal_descriptor{false};
        bool requires_accel_instance_buffer{false};
        bool requires_bindless_buffer_metadata{false};
        bool requires_buffer_coherence{false};
        size_t read_property_index{invalid_resource_property_index};
        size_t write_property_index{invalid_resource_property_index};
        size_t accel_instance_property_index{invalid_resource_property_index};
        size_t bindless_buffer_metadata_property_index{
            invalid_resource_property_index};
    };
    luisa::vector<KernelResourceBinding> _kernel_resource_bindings;
    bool _has_argument_buffer{false};
    spv::Id _argument_buffer_id{spv::NoResult};
    spv::Id _indirect_dispatch_buffer_id{spv::NoResult};
    struct DispatchMetadataState {
        spv::Id packed{spv::NoResult};
        spv::Id dispatch_size{spv::NoResult};
        spv::Id kernel_id{spv::NoResult};
    } _dispatch_metadata;
    // A surviving callable that directly reads dispatch metadata, or forwards
    // it to another callable, receives one backend-private trailing uint4
    // parameter. This is explicit SSA; kernel-local IDs never cross functions.
    luisa::unordered_set<const xir::Function *>
        _functions_requiring_dispatch_metadata;
    size_t _buffer_metadata_offset{0u};
    luisa::unordered_map<spv::Id, uint32_t> _direct_buffer_metadata_indices;
    // A bound direct byte-buffer view has a compile-time logical offset. The
    // Vulkan descriptor base is at least uint32-aligned, so gcd(offset, 4)
    // also divides its runtime descriptor-relative bias. Unbound resources
    // deliberately have no entry: their view offset remains runtime data.
    luisa::unordered_map<const xir::Argument *, size_t>
        _bound_direct_buffer_bias_alignments;
    luisa::unordered_map<spv::Id, spv::Id> _bindless_buffer_metadata_ids;
    bool _use_tex2d_bindless{false};
    bool _use_tex3d_bindless{false};
    bool _use_buffer_bindless{false};
    bool _use_buffer_bindless_metadata{false};
    bool _enable_fast_math{false};
    bool _enable_debug_info{false};
    SpirvTargetFeatures _target_features{};
    SpirvRuntimeTargetPlan _runtime_target_plan{};
    bool _runtime_target_plan_installed{false};
    SpirvTargetFeatureMask _required_target_features{};
    bool _uses_float8{false};
    bool _uses_8bit_storage_buffer{false};
    bool _uses_8bit_uniform_storage{false};
    bool _uses_8bit_push_constant{false};
    bool _uses_16bit_storage_buffer{false};
    bool _uses_16bit_uniform_storage{false};
    bool _uses_16bit_push_constant{false};
    spv::Id _buffer_heap_id{spv::NoResult};
    spv::Id _tex2d_heap_id{spv::NoResult};
    spv::Id _tex3d_heap_id{spv::NoResult};
    spv::Id _glsl450{spv::NoResult};
    spv::Instruction *_entry_point_inst{nullptr};
    luisa::vector<spv::Id> _deferred_entry_point_interface_ids;
    luisa::unordered_set<spv::Id> _entry_point_interface_ids;
    spv::Id _global_invocation_id_var{spv::NoResult};
    luisa::uint3 _kernel_block_size{0u};
    luisa::unordered_map<spv::BuiltIn, spv::Id> _builtin_var_map;
    luisa::unordered_map<spv::Id, bool> _is_storage_image_map;
    struct RayQueryState {
        spv::Id initial_ray{spv::NoResult};
        spv::Id proceed_state{spv::NoResult};
    };
    luisa::unordered_map<spv::Id, RayQueryState> _ray_query_states;
    luisa::unordered_map<const xir::Function *, luisa::vector<bool>> _callable_arg_used;
    SpirvFunctionArgumentAnalysisMap _function_argument_usage;
    SpirvReadonlyResourceOriginMap
        _readonly_resource_origins;
    luisa::unordered_set<const Type *> _needs_atomic_buffer_types;
    luisa::unordered_map<const Type *, SpirvAtomicBufferStoragePlan>
        _atomic_buffer_storage_plans;
    bool _atomic_buffer_plan_installed{false};
    luisa::unordered_map<const Type *, spv::Id> _laid_out_type_map;
    luisa::compute::xir::UniformityAnalysis _uniformity;

    spv::Id _constant_ubo_var{spv::NoResult};
    // Constant identity, rather than a 64-bit content hash, owns the mapping.
    // Hash collisions must never redirect one XIR constant to another UBO
    // member with different bytes.
    luisa::unordered_map<const xir::Constant *, uint32_t>
        _ubo_constant_member_indices;
    luisa::vector<std::byte> _constant_ubo_data;
    luisa::vector<const xir::Constant *> _ubo_array_constants;
    bool _has_constant_ubo{false};

private:
    enum class BufferIndexUnit : uint8_t {
        ELEMENT,
        BYTE
    };

    struct BindlessBufferBinding {
        spv::Id buffer{spv::NoResult};
        spv::Id slot_index{spv::NoResult};
    };

    struct BindlessTextureBinding {
        spv::Id image{spv::NoResult};
        spv::Id packed{spv::NoResult};
        bool nonuniform{false};
    };

    struct InstructionUsageAnalysis {
        luisa::unordered_set<const Type *> used_types;
        luisa::unordered_set<const xir::Constant *> used_constants;
        luisa::vector<const xir::Function *> used_functions_post_order;
        SpirvBindlessResourceUsage bindless_resources;
    };
    void _analyze_instruction_usage(
        const xir::Function *function,
        InstructionUsageAnalysis &analysis) noexcept;
    [[nodiscard]] InstructionUsageAnalysis _analyze_module_usage(const xir::Module *module) noexcept;
    void _install_atomic_buffer_plan(
        const SpirvAtomicBufferModulePlan &plan) noexcept;
    void _install_runtime_target_plan(
        const SpirvRuntimeTargetPlan &plan) noexcept;
    [[nodiscard]] bool _buffer_uses_word_storage(const Type *type) noexcept;

    spv::Id _convert_type(const Type *type, Usage usage) noexcept;
    spv::Id _convert_laid_out_type(const Type *type) noexcept;
    void _mark_8bit_storage_usage(const Type *type, spv::StorageClass storage) noexcept;
    void _mark_16bit_storage_usage(const Type *type, spv::StorageClass storage) noexcept;
    spv::Id _emit_literal(const Type *type, const void *data) noexcept;
    spv::Id _emit_constant(const xir::Constant *c) noexcept;
    spv::Id _emit_alloca(const xir::AllocaInst *alloca) noexcept;
    spv::Id _emit_value(const xir::Value *value) noexcept;
    [[nodiscard]] spv::Block *_emit_dispatch_metadata_prologue(
        spv::Function *function) noexcept;
    void _set_dispatch_metadata(spv::Id packed) noexcept;
    void _analyze_dispatch_metadata_requirements(
        const InstructionUsageAnalysis &analysis) noexcept;
    [[nodiscard]] static bool _is_indirect_dispatch_type(
        const Type *type) noexcept;
    [[nodiscard]] static bool _is_kernel_resource_argument(
        const xir::Argument *argument) noexcept;
    void generate_binding(
        Function kernel,
        luisa::span<const std::pair<Variable, Usage>> argument_usages,
        const xir::KernelFunction *xir_kernel);
    [[nodiscard]] spv::Block *_xir_block_entry(const xir::BasicBlock *bb) const noexcept;
    [[nodiscard]] spv::Block *_physical_block(ControlFlowPlan::Target target) const noexcept;
    [[nodiscard]] spv::Block *_create_physical_block(spv::Function *function = nullptr) noexcept;
    void _register_physical_block(spv::Block *block, bool already_appended) noexcept;
    void _set_current_tail(spv::Block *block) noexcept;
    void _register_entry_point_interface(spv::Id id) noexcept;
    void _require_target_feature(SpirvTargetFeatureMask feature,
                                 bool supported) noexcept;
    void _require_sampled_image_array_indexing(bool nonuniform) noexcept;
    void _require_storage_buffer_array_indexing(bool nonuniform) noexcept;
    void _require_subgroup_type(const Type *type,
                                luisa::string_view operation) noexcept;

    void _predeclare_allocas(const xir::FunctionDefinition *def) noexcept;
    void _validate_ray_query_lifetimes(const xir::FunctionDefinition *def) const noexcept;
    void _predeclare_phis() noexcept;
    void _resolve_phi_incomings_from_predecessor(const xir::BasicBlock *predecessor) noexcept;
    void _finalize_phis() noexcept;
    void _emit_phi_inst(const xir::PhiInst *instruction) noexcept;
    void _prepare_control_flow_plan(const xir::FunctionDefinition *def) noexcept;
    void _emit_kernel(
        const xir::KernelFunction *kernel,
        const SpirvKernelArgumentLayoutPlan &argument_layout) noexcept;
    void _emit_callable(const xir::CallableFunction *callable, const xir::Module *module) noexcept;
    void _reset_function_codegen_state() noexcept;
    void _emit_function_blocks(const xir::FunctionDefinition *def) noexcept;
    void _analyze_function_argument_usage(const xir::Module *module) noexcept;
    Usage _function_argument_usage_of(const xir::Function *function,
                                      const xir::Argument *argument) const noexcept;
    Usage _resource_argument_binding_usage(const xir::Argument *argument) const noexcept;
    bool _emit_block(const xir::BasicBlock *bb) noexcept;
    void _emit_instruction(const xir::Instruction *inst) noexcept;
    [[nodiscard]] spv::Block *_resolve_branch_target(const xir::BasicBlock *bb) const noexcept;

    void _emit_if_inst(const xir::IfInst *inst) noexcept;
    void _emit_loop_inst(const xir::LoopInst *inst) noexcept;
    void _emit_simple_loop_inst(const xir::SimpleLoopInst *inst) noexcept;
    void _emit_switch_inst(const xir::SwitchInst *inst) noexcept;
    void _emit_loop_merge(
        const ControlFlowPlan::LoopRegion &region) noexcept;
    void _emit_branch_inst(const xir::BranchInst *inst) noexcept;
    void _emit_conditional_branch_inst(const xir::ConditionalBranchInst *inst) noexcept;
    void _emit_arithmetic_inst(const xir::ArithmeticInst *inst) noexcept;
    void _emit_atomic_inst(const xir::AtomicInst *inst) noexcept;
    spv::Id _emit_float_atomic_cas_loop(
        spv::Id ptr, spv::Id val, spv::Id float_type, xir::AtomicOp op,
        spv::Id scope, spv::Id load_semantics,
        spv::Id equal_semantics, spv::Id unequal_semantics) noexcept;
    void _emit_ray_query_traversal_to_completion(
        spv::Id ray_query) noexcept;
    void _emit_resource_query_inst(const xir::ResourceQueryInst *inst) noexcept;
    void _emit_resource_read_inst(const xir::ResourceReadInst *inst) noexcept;
    void _emit_resource_write_inst(const xir::ResourceWriteInst *inst) noexcept;
    [[nodiscard]] bool _bindless_index_is_nonuniform(
        const xir::Value *index,
        xir::BindlessResourceAccess access) const noexcept;
    [[nodiscard]] spv::Id _load_bindless_slot_word(
        spv::Id bindless_array, spv::Id slot_index,
        uint32_t stride_words, uint32_t field_word,
        bool nonuniform) noexcept;
    [[nodiscard]] BindlessBufferBinding _load_bindless_buffer_binding(
        spv::Id bindless_array, const xir::Value *slot,
        xir::BindlessResourceAccess access) noexcept;
    [[nodiscard]] BindlessTextureBinding _load_bindless_texture_binding(
        spv::Id bindless_array, const xir::Value *slot,
        bool is_2d, xir::BindlessResourceAccess access) noexcept;
    spv::Id _emit_buffer_read(spv::Id buffer, spv::Id index, const Type *read_type, const Type *buffer_type, BufferIndexUnit index_unit, spv::MemoryAccessMask memory_access = spv::MemoryAccessMask::MaskNone, size_t byte_index_alignment = 1u) noexcept;
    spv::Id _emit_buffer_read_impl(spv::Id buffer, spv::Id byte_offset, const Type *elem_type, size_t byte_alignment, spv::MemoryAccessMask memory_access = spv::MemoryAccessMask::MaskNone) noexcept;
    void _emit_buffer_write(spv::Id buffer, spv::Id index, spv::Id value, const Type *value_type, const Type *buffer_type, BufferIndexUnit index_unit, spv::MemoryAccessMask memory_access = spv::MemoryAccessMask::MaskNone, size_t byte_index_alignment = 1u) noexcept;
    void _emit_buffer_write_impl(spv::Id buffer, spv::Id byte_offset, spv::Id value, const Type *elem_type, size_t byte_alignment, spv::MemoryAccessMask memory_access = spv::MemoryAccessMask::MaskNone) noexcept;
    void _emit_buffer_write_word_masked(spv::Id buffer, spv::Id word_index, spv::Id value, spv::Id mask) noexcept;
    [[nodiscard]] spv::Id _load_direct_buffer_metadata(
        spv::Id buffer, StorageBufferMetadataField field,
        spv::Id target_type) noexcept;
    [[nodiscard]] spv::Id _load_bindless_buffer_metadata(
        spv::Id bindless_array, spv::Id slot_index,
        StorageBufferMetadataField field,
        spv::Id target_type) noexcept;
    [[nodiscard]] size_t _direct_buffer_bias_alignment(
        const xir::Value *resource) const noexcept;
    [[nodiscard]] spv::Id _add_direct_buffer_bias(
        spv::Id buffer, spv::Id byte_offset) noexcept;
    [[nodiscard]] spv::Id _add_bindless_buffer_bias(
        spv::Id bindless_array, spv::Id slot_index,
        spv::Id byte_offset, xir::BindlessResourceAccess access) noexcept;
    void _emit_thread_group_inst(const xir::ThreadGroupInst *inst) noexcept;
    void _emit_ray_query_object_read_inst(const xir::RayQueryObjectReadInst *inst) noexcept;
    void _emit_ray_query_object_write_inst(const xir::RayQueryObjectWriteInst *inst) noexcept;
    void _emit_ray_query_loop_inst(const xir::RayQueryLoopInst *inst) noexcept;
    void _emit_ray_query_dispatch_inst(const xir::RayQueryDispatchInst *inst) noexcept;
    [[nodiscard]] static bool _is_ray_query_type(const Type *type) noexcept;
    [[nodiscard]] static const Type *_ray_query_initial_ray_type() noexcept;
    [[nodiscard]] const RayQueryState &_ray_query_state(spv::Id query_object) const noexcept;
    [[nodiscard]] const KernelResourceBinding &_kernel_resource_binding(
        const xir::Argument *argument) const noexcept;
    [[nodiscard]] spv::Id _kernel_resource_property_id(
        size_t property_index) const noexcept;
    spv::Id _resolve_resource_argument(const xir::Argument *arg) noexcept;
    spv::Id _resolve_writable_resource(const xir::Value *resource) noexcept;
    spv::Id _resolve_accel_instance_buffer(const xir::Value *accel) noexcept;
    spv::Id _load_texture(spv::Id tex_var) noexcept;
    [[nodiscard]] std::vector<spv::Id> _emit_aggregate_access_indices(
        const SpirvAggregateIndexPlan &plan) noexcept;
    spv::Id _create_access_chain(spv::StorageClass storage, spv::Id base, const std::vector<spv::Id> &indices, bool nonuniform = false) noexcept;
    spv::Id _ensure_type(spv::Id value, spv::Id target_type) noexcept;

public:
    SpirvCodegenEntry(StringScratch &scratch, bool allow_indirect) noexcept;
    ~SpirvCodegenEntry() noexcept;
    void emit(const xir::Module *module, luisa::span<const Function::Binding> bindings,
              luisa::string_view device_lib, luisa::string_view native_include) noexcept;
    [[nodiscard]] auto move_print_formats() && noexcept { return std::move(_print_formats); }
    static SpirvResult compile_spirv(
        Function kernel, const ShaderOption &option,
        SpirvTargetFeatures target_features = {});
    // Direct native-XIR entry used by backend conformance tests and tools that
    // already own a legalized module. The AST function still defines the
    // external kernel ABI and descriptor bindings.
    static SpirvResult compile_spirv_xir(
        Function kernel, const xir::Module *module,
        const ShaderOption &option,
        SpirvTargetFeatures target_features = {});

private:
    [[nodiscard]] vstd::vector<std::pair<Variable, Usage>>
    _collect_kernel_argument_usages(Function kernel, const xir::Module *module) const noexcept;
    [[nodiscard]] vstd::vector<SpirvKernelArgumentRoleMask>
    _collect_kernel_argument_roles(Function kernel, const xir::Module *module) const noexcept;
};

}// namespace lc::spirv
