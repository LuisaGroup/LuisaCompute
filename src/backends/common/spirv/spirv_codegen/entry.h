#pragma once

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

namespace lc::spirv {
using namespace luisa;
using namespace luisa::compute;
struct SpirvResult {
    using Properties = vstd::vector<Property>;
    std::vector<uint32_t> spv_bin;
    Properties properties;
    vstd::vector<std::pair<vstd::string, luisa::compute::Type const *>> printers;
    luisa::vector<std::byte> constant_ubo_data;
    bool useTex2DBindless;
    bool useTex3DBindless;
    bool useBufferBindless;
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
    spv::Builder &_builder; // reference to *_builder_ptr

    luisa::unordered_map<const Type *, spv::Id> _type_map;
    luisa::unordered_map<const xir::Value *, spv::Id> _value_map;
    luisa::unordered_map<const xir::Function *, spv::Function *> _function_map;
    luisa::unordered_map<const xir::BasicBlock *, spv::Block *> _block_map;
    luisa::unordered_map<const xir::BasicBlock *, std::pair<spv::Block *, spv::Block *>> _loop_header_info;
    luisa::unordered_map<const xir::BasicBlock *, spv::Block *> _loop_header_redirect;
    luisa::unordered_set<const xir::BasicBlock *> _emitted_blocks;
    luisa::unordered_set<spv::Id> _used_merge_blocks;

    luisa::unordered_map<const xir::PrintInst *, PrintInfo> _print_info;
    PrintFormatVector _print_formats;
    luisa::vector<const xir::Instruction *> _control_flow_stack;
    bool _allow_indirect_dispatch;
    bool _requires_printing{false};
    SpirvResult::Properties _properties;
    luisa::vector<spv::Id> _property_ids;
    bool _use_tex2d_bindless{false};
    bool _use_tex3d_bindless{false};
    bool _use_buffer_bindless{false};
    bool _use_native_float_atomics{true};
    bool _uses_int8{false};
    bool _uses_float8{false};
    bool _uses_8bit_storage_buffer{false};
    bool _uses_8bit_uniform_storage{false};
    bool _uses_8bit_push_constant{false};
    spv::Id _buffer_heap_id{spv::NoResult};
    spv::Id _tex2d_heap_id{spv::NoResult};
    spv::Id _tex3d_heap_id{spv::NoResult};
    spv::Id _glsl450{spv::NoResult};
    spv::Instruction *_entry_point_inst{nullptr};
    spv::Id _global_invocation_id_var{spv::NoResult};
    luisa::unordered_map<spv::BuiltIn, spv::Id> _builtin_var_map;
    luisa::unordered_map<spv::Id, bool> _is_storage_image_map;
    luisa::unordered_map<spv::Id, spv::Id> _accel_instance_buffer_map;
    luisa::unordered_map<spv::Id, spv::Id> _rq_proceed_result;// rq object SSA id -> last OpRayQueryProceedKHR result SSA id
    luisa::unordered_map<const xir::Function *, luisa::vector<bool>> _callable_arg_used;
    luisa::unordered_set<const Type *> _needs_atomic_buffer_types;
    luisa::unordered_map<const Type *, spv::Id> _laid_out_type_map;
    luisa::compute::xir::UniformityAnalysis _uniformity;

    spv::Id _constant_ubo_var{spv::NoResult};
    luisa::unordered_map<uint64_t, uint32_t> _ubo_constant_member_by_hash;
    luisa::vector<std::byte> _constant_ubo_data;
    luisa::vector<const xir::Constant *> _ubo_array_constants;
    bool _has_constant_ubo{false};

private:
    struct InstructionUsageAnalysis {
        luisa::unordered_set<const Type *> used_types;
        luisa::unordered_set<const xir::Constant *> used_constants;
        luisa::vector<const xir::Function *> used_functions_post_order;
    };
    void _analyze_instruction_usage(const xir::Function *f, InstructionUsageAnalysis &analysis,
                                    luisa::unordered_set<const xir::Function *> &visited) noexcept;
    [[nodiscard]] InstructionUsageAnalysis _analyze_module_usage(const xir::Module *module) noexcept;
    void _mark_atomic_buffer_types(const InstructionUsageAnalysis &analysis) noexcept;

    spv::Id _convert_type(const Type *type, Usage usage) noexcept;
    spv::Id _convert_laid_out_type(const Type *type) noexcept;
    [[nodiscard]] bool _type_contains_bool(const Type *type) noexcept;
    void _mark_8bit_storage_usage(const Type *type, spv::StorageClass storage) noexcept;
    spv::Id _emit_literal(const Type *type, const void *data) noexcept;
    spv::Id _emit_constant(const xir::Constant *c) noexcept;
    spv::Id _emit_value(const xir::Value *value) noexcept;
    spv::Block *_get_or_create_block(const xir::BasicBlock *bb) noexcept;

    void _emit_kernel(const xir::KernelFunction *kernel) noexcept;
    void _emit_callable(const xir::CallableFunction *callable, const xir::Module *module) noexcept;
    void _emit_block(const xir::BasicBlock *bb) noexcept;
    void _emit_instruction(const xir::Instruction *inst) noexcept;

    void _emit_if_inst(const xir::IfInst *inst) noexcept;
    void _emit_loop_inst(const xir::LoopInst *inst) noexcept;
    void _emit_simple_loop_inst(const xir::SimpleLoopInst *inst) noexcept;
    void _emit_switch_inst(const xir::SwitchInst *inst) noexcept;
    void _emit_branch_inst(const xir::BranchInst *inst) noexcept;
    void _emit_conditional_branch_inst(const xir::ConditionalBranchInst *inst) noexcept;
    void _emit_arithmetic_inst(const xir::ArithmeticInst *inst) noexcept;
    void _emit_atomic_inst(const xir::AtomicInst *inst) noexcept;
    spv::Id _emit_float_atomic_cas_loop(spv::Id ptr, spv::Id val, spv::Id float_type, xir::AtomicOp op) noexcept;
    spv::Id _emit_float_compare_exchange_cas_loop(spv::Id ptr, spv::Id expected, spv::Id desired, spv::Id float_type) noexcept;
    void _emit_resource_query_inst(const xir::ResourceQueryInst *inst) noexcept;
    void _emit_resource_read_inst(const xir::ResourceReadInst *inst) noexcept;
    void _emit_resource_write_inst(const xir::ResourceWriteInst *inst) noexcept;
    spv::Id _emit_buffer_read(spv::Id buffer, spv::Id index, const Type *read_type, const Type *buffer_type, bool index_is_word_offset = false) noexcept;
    spv::Id _emit_buffer_read_impl(spv::Id buffer, spv::Id word_offset, const Type *elem_type) noexcept;
    void _emit_buffer_write(spv::Id buffer, spv::Id index, spv::Id value, const Type *value_type, const Type *buffer_type, bool index_is_word_offset = false) noexcept;
    void _emit_buffer_write_impl(spv::Id buffer, spv::Id word_offset, spv::Id value, const Type *elem_type) noexcept;
    void _emit_thread_group_inst(const xir::ThreadGroupInst *inst) noexcept;
    void _emit_ray_query_object_read_inst(const xir::RayQueryObjectReadInst *inst) noexcept;
    void _emit_ray_query_object_write_inst(const xir::RayQueryObjectWriteInst *inst) noexcept;
    void _emit_ray_query_loop_inst(const xir::RayQueryLoopInst *inst) noexcept;
    void _emit_ray_query_dispatch_inst(const xir::RayQueryDispatchInst *inst) noexcept;
    spv::Id _resolve_resource_argument(const xir::Argument *arg) noexcept;
    spv::Id _resolve_accel_instance_buffer(const xir::Argument *arg) noexcept;
    size_t _get_resource_property_base(const xir::Function *func) const noexcept;
    spv::Id _load_texture(spv::Id tex_var) noexcept;
    spv::Id _create_access_chain(spv::StorageClass storage, spv::Id base, const std::vector<spv::Id> &indices, bool nonuniform = false) noexcept;
    spv::Id _ensure_type(spv::Id value, spv::Id target_type) noexcept;

public:
    SpirvCodegenEntry(StringScratch &scratch, bool allow_indirect) noexcept;
    ~SpirvCodegenEntry() noexcept;
    void emit(const xir::Module *module, luisa::span<const Function::Binding> bindings,
              luisa::string_view device_lib, luisa::string_view native_include) noexcept;
    [[nodiscard]] auto move_print_formats() && noexcept { return std::move(_print_formats); }
    void generate_binding(Function kernel);

    static SpirvResult compile_spirv(Function kernel, const ShaderOption &option, bool use_native_float_atomics = true);
};

}// namespace lc::spirv
