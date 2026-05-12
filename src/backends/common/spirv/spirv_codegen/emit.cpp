#include "entry.h"

#include <sstream>
#include <luisa/core/logging.h>
#include <luisa/ast/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/function.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/thread_group.h>
#include <luisa/xir/instructions/print.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/instructions/assume.h>
#include <luisa/xir/instructions/assert.h>
#include <luisa/xir/instructions/clock.h>
#include <luisa/xir/instructions/debug_break.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/autodiff.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/raster_discard.h>
#include <SPIRV/disassemble.h>

namespace lc::spirv {

SpirvCodegenEntry::SpirvCodegenEntry(StringScratch &scratch, bool allow_indirect) noexcept
    : _scratch{scratch},
      _builder{spv::Spv_1_5, 0, &_logger},
      _allow_indirect_dispatch{allow_indirect} {
    _builder.setSource(spv::SourceLanguage::Unknown, 0);
    _builder.setMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
    _builder.addCapability(spv::Capability::Shader);
    _glsl450 = _builder.import("GLSL.std.450");
}

SpirvCodegenEntry::~SpirvCodegenEntry() noexcept = default;

void SpirvCodegenEntry::_analyze_instruction_usage(
    const xir::Function *f, InstructionUsageAnalysis &analysis,
    luisa::unordered_set<const xir::Function *> &visited) noexcept {
    if (!visited.emplace(f).second) { return; }
    for (auto arg : f->arguments()) {
        LUISA_ASSERT(arg != nullptr, "Function argument is null.");
        analysis.used_types.emplace(arg->type());
    }
    analysis.used_types.emplace(f->type());
    if (auto def = f->definition()) {
        def->traverse_instructions([&](const xir::Instruction *inst) noexcept {
            switch (inst->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::PRINT: {
                    _requires_printing = true;
                    auto print = static_cast<const xir::PrintInst *>(inst);
                    auto [iter, success] = _print_info.emplace(print, PrintInfo{nullptr, _print_formats.size()});
                    LUISA_ASSERT(success, "Print info already exists.");
                    luisa::vector<const Type *> arg_types;
                    arg_types.reserve(print->operand_count() + 2u);
                    arg_types.emplace_back(Type::of<uint>());// arg size
                    arg_types.emplace_back(Type::of<uint>());// fmt id
                    for (auto op_use : print->operand_uses()) {
                        LUISA_ASSERT(op_use->value() != nullptr, "Print operand use is null.");
                        arg_types.emplace_back(op_use->value()->type());
                    }
                    auto s = Type::structure(arg_types);
                    analysis.used_types.emplace(s);
                    iter->second.type = s;
                    _print_formats.emplace_back(print->format(), s);
                    break;
                }
                default: break;
            }
            analysis.used_types.emplace(inst->type());
            for (auto op_use : inst->operand_uses()) {
                if (auto value = op_use->value()) {
                    analysis.used_types.emplace(value->type());
                    switch (value->derived_value_tag()) {
                        case xir::DerivedValueTag::FUNCTION:
                            _analyze_instruction_usage(static_cast<const xir::Function *>(value), analysis, visited);
                            break;
                        case xir::DerivedValueTag::CONSTANT:
                            analysis.used_constants.emplace(static_cast<const xir::Constant *>(value));
                            break;
                        default: break;
                    }
                }
            }
        });
    }
    analysis.used_functions_post_order.emplace_back(f);
}

namespace {

[[nodiscard]] spv::Id spirv_codegen_emit_scalar_constant(spv::Builder &builder, const Type *type, const void *data) noexcept {
    switch (type->tag()) {
        case Type::Tag::BOOL: return builder.makeBoolConstant(*static_cast<const bool *>(data));
        case Type::Tag::INT8: return builder.makeInt8Constant(*static_cast<const int8_t *>(data));
        case Type::Tag::UINT8: return builder.makeUint8Constant(*static_cast<const uint8_t *>(data));
        case Type::Tag::INT16: return builder.makeInt16Constant(*static_cast<const int16_t *>(data));
        case Type::Tag::UINT16: return builder.makeUint16Constant(*static_cast<const uint16_t *>(data));
        case Type::Tag::INT32: return builder.makeIntConstant(*static_cast<const int32_t *>(data));
        case Type::Tag::UINT32: return builder.makeUintConstant(*static_cast<const uint32_t *>(data));
        case Type::Tag::INT64: return builder.makeInt64Constant(*static_cast<const int64_t *>(data));
        case Type::Tag::UINT64: return builder.makeUint64Constant(*static_cast<const uint64_t *>(data));
        case Type::Tag::FLOAT16: return builder.makeFloat16Constant(*static_cast<const half *>(data));
        case Type::Tag::FLOAT32: return builder.makeFloatConstant(*static_cast<const float *>(data));
        case Type::Tag::FLOAT64: return builder.makeDoubleConstant(*static_cast<const double *>(data));
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid scalar constant type {}.", type->description());
}

}// namespace

spv::Id SpirvCodegenEntry::_emit_constant(const xir::Constant *c) noexcept {
    if (auto it = _value_map.find(c); it != _value_map.end()) { return it->second; }
    auto type = c->type();
    auto spv_type = _convert_type(type, Usage::READ);
    spv::Id id = spv::NoResult;
    switch (type->tag()) {
        case Type::Tag::BOOL:
        case Type::Tag::INT8:
        case Type::Tag::UINT8:
        case Type::Tag::INT16:
        case Type::Tag::UINT16:
        case Type::Tag::INT32:
        case Type::Tag::UINT32:
        case Type::Tag::INT64:
        case Type::Tag::UINT64:
        case Type::Tag::FLOAT16:
        case Type::Tag::FLOAT32:
        case Type::Tag::FLOAT64:
            id = spirv_codegen_emit_scalar_constant(_builder, type, c->data());
            break;
        case Type::Tag::VECTOR: {
            auto elem_type = type->element();
            auto dim = type->dimension();
            auto elem_stride = elem_type->size();
            std::vector<spv::Id> comps;
            comps.reserve(dim);
            for (auto i = 0u; i < dim; ++i) {
                auto elem_data = static_cast<const std::byte *>(c->data()) + i * elem_stride;
                comps.emplace_back(spirv_codegen_emit_scalar_constant(_builder, elem_type, elem_data));
            }
            id = _builder.makeCompositeConstant(spv_type, comps);
            break;
        }
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V constant emission for type {}.", type->description());
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit constant.");
    _value_map.emplace(c, id);
    return id;
}

spv::Id SpirvCodegenEntry::_emit_value(const xir::Value *value) noexcept {
    if (auto it = _value_map.find(value); it != _value_map.end()) { return it->second; }
    spv::Id id = spv::NoResult;
    switch (value->derived_value_tag()) {
        case xir::DerivedValueTag::CONSTANT:
            id = _emit_constant(static_cast<const xir::Constant *>(value));
            break;
        case xir::DerivedValueTag::UNDEFINED:
            id = _builder.createUndefined(_convert_type(value->type(), Usage::READ));
            break;
        case xir::DerivedValueTag::SPECIAL_REGISTER: {
            auto reg = static_cast<const xir::SpecialRegister *>(value);
            spv::BuiltIn builtin;
            switch (reg->derived_special_register_tag()) {
                case xir::DerivedSpecialRegisterTag::THREAD_ID: builtin = spv::BuiltIn::LocalInvocationId; break;
                case xir::DerivedSpecialRegisterTag::BLOCK_ID: builtin = spv::BuiltIn::WorkgroupId; break;
                case xir::DerivedSpecialRegisterTag::DISPATCH_ID: builtin = spv::BuiltIn::GlobalInvocationId; break;
                case xir::DerivedSpecialRegisterTag::BLOCK_SIZE: builtin = spv::BuiltIn::WorkgroupSize; break;
                case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE: builtin = spv::BuiltIn::NumWorkgroups; break;
                case xir::DerivedSpecialRegisterTag::WARP_SIZE: builtin = spv::BuiltIn::SubgroupSize; break;
                case xir::DerivedSpecialRegisterTag::WARP_LANE_ID: builtin = spv::BuiltIn::SubgroupLocalInvocationId; break;
                default:
                    LUISA_NOT_IMPLEMENTED("SPIR-V special register {}.", xir::to_string(reg->derived_special_register_tag()));
            }
            auto var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Input,
                                               _convert_type(reg->type(), Usage::READ), "sr");
            _builder.addDecoration(var, spv::Decoration::BuiltIn, (int)builtin);
            id = _builder.createLoad(var, spv::NoPrecision);
            break;
        }
        case xir::DerivedValueTag::ARGUMENT: {
            auto arg = static_cast<const xir::Argument *>(value);
            if (arg->is_resource()) {
                id = _resolve_resource_argument(arg);
                break;
            }
            [[fallthrough]];
        }
        case xir::DerivedValueTag::FUNCTION:
        case xir::DerivedValueTag::BASIC_BLOCK:
        case xir::DerivedValueTag::INSTRUCTION:
            LUISA_ERROR_WITH_LOCATION("SPIR-V value {} should have been pre-mapped.", xir::to_string(value->derived_value_tag()));
            break;
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit value.");
    _value_map.emplace(value, id);
    return id;
}

spv::Block *SpirvCodegenEntry::_get_or_create_block(const xir::BasicBlock *bb) noexcept {
    if (bb == nullptr) { return nullptr; }
    if (auto it = _block_map.find(bb); it != _block_map.end()) { return it->second; }
    auto block = &_builder.makeNewBlock();
    _block_map.emplace(bb, block);
    return block;
}

void SpirvCodegenEntry::_emit_block(const xir::BasicBlock *bb) noexcept {
    if (bb == nullptr) { return; }
    if (!_emitted_blocks.emplace(bb).second) { return; }
    auto spv_block = _get_or_create_block(bb);
    _builder.setBuildPoint(spv_block);
    if (auto it = _loop_header_info.find(bb); it != _loop_header_info.end()) {
        _builder.createLoopMerge(it->second.first, it->second.second,
                                 spv::LoopControlMask::MaskNone, {});
    }
    for (auto inst : bb->instructions()) {
        _emit_instruction(inst);
    }
}

void SpirvCodegenEntry::_emit_kernel(const xir::KernelFunction *kernel) noexcept {
    auto ret_type = _builder.makeVoidType();
    std::vector<spv::Id> param_types;
    luisa::vector<const xir::Argument *> value_args;
    for (auto arg : kernel->arguments()) {
        if (arg->is_resource()) { continue; }
        param_types.emplace_back(_convert_type(arg->type(), Usage::READ));
        value_args.push_back(arg);
    }
    spv::Block *entry = nullptr;
    auto func = _builder.makeFunctionEntry(spv::NoPrecision, ret_type, "main",
                                           spv::LinkageType::Max, param_types, {}, &entry);
    _value_map.emplace(kernel, func->getId());
    _function_map.emplace(kernel, func);

    auto i = 0u;
    for (auto arg : value_args) {
        _value_map.emplace(arg, func->getParamId(i));
        ++i;
    }

    _builder.addEntryPoint(spv::ExecutionModel::GLCompute, func, "main");
    auto bs = kernel->block_size();
    _builder.addExecutionMode(func, spv::ExecutionMode::LocalSize,
                              static_cast<int>(bs.x),
                              static_cast<int>(bs.y),
                              static_cast<int>(bs.z));

    _builder.enterFunction(func);
    _builder.setBuildPoint(entry);
    _emit_block(kernel->body_block());

    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.makeReturn(false);
    }
    _builder.leaveFunction();
}

void SpirvCodegenEntry::_emit_callable(const xir::CallableFunction *callable) noexcept {
    auto ret_type = _convert_type(callable->type(), Usage::READ);
    std::vector<spv::Id> param_types;
    luisa::vector<const xir::Argument *> value_args;
    for (auto arg : callable->arguments()) {
        if (arg->is_resource()) { continue; }
        param_types.emplace_back(_convert_type(arg->type(), Usage::READ));
        value_args.push_back(arg);
    }
    spv::Block *entry = nullptr;
    auto func = _builder.makeFunctionEntry(spv::NoPrecision, ret_type,
                                           luisa::string{callable->name().value_or("callable")}.c_str(),
                                           spv::LinkageType::Max,
                                           param_types, {}, &entry);
    _value_map.emplace(callable, func->getId());
    _function_map.emplace(callable, func);

    auto i = 0u;
    for (auto arg : value_args) {
        _value_map.emplace(arg, func->getParamId(i));
        ++i;
    }

    _builder.enterFunction(func);
    _builder.setBuildPoint(entry);
    _emit_block(callable->body_block());

    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.makeReturn(false);
    }
    _builder.leaveFunction();
}

void SpirvCodegenEntry::emit(const xir::Module *module,
                             luisa::span<const Function::Binding> bindings,
                             luisa::string_view device_lib,
                             luisa::string_view native_include) noexcept {
    auto kernel = [module] {
        const xir::KernelFunction *k = nullptr;
        for (auto f : module->function_list()) {
            if (f->isa<xir::KernelFunction>()) {
                LUISA_ASSERT(k == nullptr,
                             "SPIR-V codegen: expected exactly one kernel function.");
                k = static_cast<const xir::KernelFunction *>(f);
            }
        }
        LUISA_ASSERT(k != nullptr, "SPIR-V codegen: kernel function not found in module.");
        return k;
    }();

    auto analysis = [this, kernel] {
        InstructionUsageAnalysis analysis;
        analysis.used_types.reserve(Type::count());
        analysis.used_constants.reserve(64u);
        analysis.used_functions_post_order.reserve(64u);
        luisa::unordered_set<const xir::Function *> visited;
        visited.reserve(64u);
        _analyze_instruction_usage(kernel, analysis, visited);
        LUISA_ASSERT(!analysis.used_functions_post_order.empty() &&
                         analysis.used_functions_post_order.back() == kernel,
                     "SPIR-V codegen: kernel function not found in post-order traversal.");
        return analysis;
    }();

    for (auto type : analysis.used_types) {
        if (type != nullptr) { _convert_type(type, Usage::READ); }
    }

    for (auto c : analysis.used_constants) {
        _emit_constant(c);
    }

    for (auto f : analysis.used_functions_post_order) {
        if (auto def = f->definition()) {
            switch (f->derived_function_tag()) {
                case xir::DerivedFunctionTag::KERNEL:
                    _emit_kernel(static_cast<const xir::KernelFunction *>(f));
                    break;
                case xir::DerivedFunctionTag::CALLABLE:
                    _emit_callable(static_cast<const xir::CallableFunction *>(f));
                    break;
                default:
                    LUISA_NOT_IMPLEMENTED("External function in SPIR-V codegen.");
            }
        } else {
            LUISA_NOT_IMPLEMENTED("External function in SPIR-V codegen.");
        }
    }

    _builder.postProcess(false);

    std::vector<unsigned int> spirv;
    _builder.dump(spirv);
    std::ostringstream oss;
    spv::Disassemble(oss, spirv);
    _scratch << oss.str();
}

}// namespace lc::spirv