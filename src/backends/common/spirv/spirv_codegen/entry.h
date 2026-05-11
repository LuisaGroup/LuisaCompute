#pragma once

#include <luisa/core/binary_io.h>
#include <luisa/ast/function.h>
#include <luisa/vstl/common.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/ast/function.h>
#include <luisa/core/string_scratch.h>
#include <SPIRV/SpvBuilder.h>
#include <luisa/runtime/rhi/resource.h>
#include "property.h"

namespace lc::spirv {
using namespace luisa;
using namespace luisa::compute;
struct SpirvResult {
    using Properties = vstd::vector<Property>;
    luisa::BinaryBlob spv_bin;
    Properties properties;
    vstd::vector<std::pair<vstd::string, luisa::compute::Type const *>> printers;
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
    spv::Builder _builder;
    spv::SpvBuildLogger _logger;

    luisa::unordered_map<const Type *, spv::Id> _type_map;
    luisa::unordered_map<const xir::Value *, spv::Id> _value_map;
    luisa::unordered_map<const xir::Function *, spv::Function *> _function_map;
    luisa::unordered_map<const xir::BasicBlock *, spv::Block *> _block_map;
    luisa::unordered_map<const xir::BasicBlock *, std::pair<spv::Block *, spv::Block *>> _loop_header_info;
    luisa::unordered_set<const xir::BasicBlock *> _emitted_blocks;

    luisa::unordered_map<const xir::PrintInst *, PrintInfo> _print_info;
    PrintFormatVector _print_formats;
    luisa::vector<const xir::Instruction *> _control_flow_stack;
    bool _allow_indirect_dispatch;
    bool _requires_printing{false};

private:
    struct InstructionUsageAnalysis {
        luisa::unordered_set<const Type *> used_types;
        luisa::unordered_set<const xir::Constant *> used_constants;
        luisa::vector<const xir::Function *> used_functions_post_order;
    };
    void _analyze_instruction_usage(const xir::Function *f, InstructionUsageAnalysis &analysis,
                                    luisa::unordered_set<const xir::Function *> &visited) noexcept;

    spv::Id _convert_type(const Type *type) noexcept;
    spv::Id _emit_constant(const xir::Constant *c) noexcept;
    spv::Id _emit_value(const xir::Value *value) noexcept;
    spv::Block *_get_or_create_block(const xir::BasicBlock *bb) noexcept;

    void _emit_kernel(const xir::KernelFunction *kernel) noexcept;
    void _emit_callable(const xir::CallableFunction *callable) noexcept;
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
    void _emit_resource_query_inst(const xir::ResourceQueryInst *inst) noexcept;
    void _emit_resource_read_inst(const xir::ResourceReadInst *inst) noexcept;
    void _emit_resource_write_inst(const xir::ResourceWriteInst *inst) noexcept;
    void _emit_thread_group_inst(const xir::ThreadGroupInst *inst) noexcept;

public:
    SpirvCodegenEntry(StringScratch &scratch, bool allow_indirect) noexcept;
    ~SpirvCodegenEntry() noexcept;
    void emit(const xir::Module *module, luisa::span<const Function::Binding> bindings,
              luisa::string_view device_lib, luisa::string_view native_include) noexcept;
    [[nodiscard]] auto move_print_formats() && noexcept { return std::move(_print_formats); }
    static SpirvResult compile_spirv(Function kernel, const ShaderOption &option);
};

}// namespace lc::spirv
