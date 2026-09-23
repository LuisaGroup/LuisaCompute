#pragma once

#ifdef LUISA_ENABLE_XIR

#include <luisa/core/string_scratch.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/lex_scope_analysis.h>

namespace luisa::compute::cuda {

class CUDACodegenXIR final {

public:
    struct PrintInfo {
        const Type *type;
        size_t index;
    };
    using PrintFormatVector = luisa::vector<std::pair<luisa::string, const Type *>>;

private:
    StringScratch &_scratch;
    luisa::unordered_map<const xir::PrintInst *, PrintInfo> _print_info;
    PrintFormatVector _print_formats;
    luisa::vector<const xir::Instruction *> _control_flow_stack;
    bool _allow_indirect_dispatch;
    bool _requires_printing{false};
    bool _requires_optix{false};
    // Compile ray tracing for the software fallback (see CUDACodegenAST): the
    // emitted traversal is `cuda_device_fallback_rtx.h` instead of OptiX, the
    // kernel is an ordinary `kernel_main`, and the operations the fallback does
    // not implement are reported instead of being emitted.
    bool _fallback_rtx{false};

private:
    const Type *_ray_type;
    const Type *_triangle_hit_type;
    const Type *_procedural_hit_type;
    const Type *_committed_hit_type;
    const Type *_ray_query_all_type;
    const Type *_ray_query_any_type;
    const Type *_indirect_buffer_type;
    const Type *_motion_srt_type;

private:
    xir::LexScopeInfo _lex_scope_info;
    luisa::unordered_map<const xir::Value *, size_t> _local_value_indices;
    luisa::unordered_map<const xir::Value *, size_t> _global_value_indices;

private:
    struct InstructionUsageAnalysis {
        CurveBasisSet required_curve_bases;
        bool requires_raytracing_closest = false;
        bool requires_raytracing_any = false;
        bool requires_raytracing_query = false;
        luisa::unordered_set<const Type *> used_types;
        luisa::unordered_set<const xir::Constant *> used_constants;
        luisa::vector<const xir::RayQueryPipelineInst *> ray_query_pipelines;
        luisa::vector<const xir::Function *> used_functions_post_order;
    };
    void _analyze_instruction_usage(const xir::Function *f, InstructionUsageAnalysis &analysis,
                                    luisa::unordered_set<const xir::Function *> &visited) noexcept;

private:
    [[nodiscard]] static bool _should_emit_global_constant(const xir::Constant *c) noexcept;
    [[nodiscard]] bool _is_ray_query_callback_function(const xir::CallableFunction *f) const noexcept;
    [[nodiscard]] bool _is_builtin_type(const Type *t) const noexcept;
    [[nodiscard]] const xir::Instruction *_find_innermost_loop() const noexcept;
    void _emit_type_name(const Type *type) noexcept;
    void _emit_type_definition(const Type *type, luisa::unordered_set<const Type *> &defined_types) noexcept;
    void _emit_type_definitions(luisa::unordered_set<const Type *> used_types) noexcept;
    void _emit_kernel_params_struct(const xir::KernelFunction *kernel) noexcept;
    void _emit_value_name(const xir::Value *value, bool is_use = true) noexcept;
    void _emit_global_constants(luisa::unordered_set<const xir::Constant *> used_constants) noexcept;
    void _emit_function_definition(const xir::FunctionDefinition *def, luisa::span<const Function::Binding> bindings) noexcept;
    void _emit_kernel_definition(const xir::KernelFunction *kernel, luisa::span<const Function::Binding> bindings) noexcept;
    void _emit_hoisted_lexical_scope_breakers() noexcept;
    void _emit_callable_definition(const xir::CallableFunction *callable) noexcept;
    void _emit_ray_query_callback_definition(const xir::CallableFunction *callable) noexcept;
    void _emit_instructions(const xir::InstructionList &inst_list, int indent) noexcept;
    void _emit_metadata(const xir::MetadataList &md_list, int indent) const noexcept;
    void _emit_indent(int indent) const noexcept;
    void _emit_access_chain(const Type *base_type, luisa::span<const xir::Use *const> chain) noexcept;
    void _emit_result_value_eq(const xir::Instruction *inst) noexcept;
    void _emit_ray_query_pipeline_inst(const xir::RayQueryPipelineInst *inst, int indent) noexcept;
    void _emit_if_inst(const xir::IfInst *inst, int indent) noexcept;
    void _emit_switch_inst(const xir::SwitchInst *inst, int indent) noexcept;
    void _emit_loop_inst(const xir::LoopInst *inst, int indent) noexcept;
    void _emit_simple_loop_inst(const xir::SimpleLoopInst *inst, int indent) noexcept;
    void _emit_atomic_inst(const xir::AtomicInst *inst) noexcept;
    void _emit_arithmetic_inst(const xir::ArithmeticInst *inst, int indent) noexcept;
    void _emit_thread_group_inst(const xir::ThreadGroupInst *inst) noexcept;
    void _emit_resource_query_inst(const xir::ResourceQueryInst *inst) noexcept;
    void _emit_resource_read_inst(const xir::ResourceReadInst *inst) noexcept;
    void _emit_resource_write_inst(const xir::ResourceWriteInst *inst) noexcept;
    void _emit_ray_query_object_read_inst(const xir::RayQueryObjectReadInst *inst) noexcept;
    void _emit_ray_query_object_write_inst(const xir::RayQueryObjectWriteInst *inst) noexcept;
    void _emit_branch_inst(const xir::BranchInst *inst) const noexcept;
    void _emit_conditional_branch_inst(const xir::ConditionalBranchInst *inst) noexcept;
    void _emit_operand_list(luisa::span<const xir::Use *const> operands) noexcept;
    void _emit_intrinsic_call(luisa::string_view name, const xir::Instruction *inst) noexcept;

    // ray query pipelines
    struct RayQueryPipelineArgument {
        enum struct Tag : uint8_t {
            CONTEXT_CAPTURE,
            KERNEL_PARAM,
        };
        Tag tag;
        bool is_pointer;
        int mapped_index;
    };
    struct RayQueryPipelineInfo {
        uint32_t index;
        bool any_context_capture{false};
        luisa::vector<RayQueryPipelineArgument> args;
    };
    luisa::unordered_map<const xir::RayQueryPipelineInst *, RayQueryPipelineInfo> _ray_query_pipeline_info;
    [[nodiscard]] int _find_ray_query_captured_kernel_param_index(const xir::Value *capture) const noexcept;
    void _preprocess_ray_query_pipelines(luisa::span<const xir::RayQueryPipelineInst *const> pipelines) noexcept;
    void _postprocess_ray_query_pipelines(luisa::span<const xir::RayQueryPipelineInst *const> pipelines,
                                          luisa::span<const Function::Binding> bindings) noexcept;

    template<typename F>
    void _with_control_flow(const xir::Instruction *inst, F &&f) noexcept {
        _control_flow_stack.emplace_back(inst);
        f();
        assert(!_control_flow_stack.empty() && _control_flow_stack.back() == inst && "Control flow stack mismatch.");
        _control_flow_stack.pop_back();
    }

    template<typename... Args>
    void _emit_with_template(const xir::Instruction *inst, Args... args) noexcept {
        auto do_emit = [&]<typename T>(T v) noexcept {
            if constexpr (luisa::is_integral_v<T>) {
                _emit_value_name(inst->operand(v));
            } else {
                _scratch << v;
            }
        };
        _emit_result_value_eq(inst);
        (do_emit(args), ...);
        _scratch << ";";
    }

public:
    CUDACodegenXIR(StringScratch &scratch, bool allow_indirect,
                   bool fallback_rtx = false) noexcept;
    ~CUDACodegenXIR() noexcept;
    void emit(const xir::Module *module, luisa::span<const Function::Binding> bindings,
              luisa::string_view device_lib, luisa::string_view native_include) noexcept;
    [[nodiscard]] auto move_print_formats() && noexcept { return std::move(_print_formats); }
};

}// namespace luisa::compute::cuda

#endif
