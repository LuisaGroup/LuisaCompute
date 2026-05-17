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
      _builder_ptr{std::make_unique<spv::Builder>(spv::Spv_1_5, 0, &_logger)},
      _builder{*_builder_ptr},
      _allow_indirect_dispatch{allow_indirect} {
    _builder.setSource(spv::SourceLanguage::Unknown, 0);
    _builder.setMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
    _builder.addCapability(spv::Capability::Shader);
    _glsl450 = _builder.import("GLSL.std.450");
}

SpirvCodegenEntry::~SpirvCodegenEntry() noexcept {
    _is_storage_image_map.clear();
    _type_map.clear();
    _value_map.clear();
    _function_map.clear();
    _block_map.clear();
    _loop_header_info.clear();
    _loop_header_redirect.clear();
    _emitted_blocks.clear();
    _print_info.clear();
    _print_formats.clear();
    _control_flow_stack.clear();
    _properties.clear();
    _property_ids.clear();
    _entry_point_inst = nullptr;
    _global_invocation_id_var = spv::NoResult;
}

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

spv::Id SpirvCodegenEntry::_emit_literal(const Type *type, const void *data) noexcept {
    auto spv_type = _convert_type(type, Usage::READ);
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
            return spirv_codegen_emit_scalar_constant(_builder, type, data);
        case Type::Tag::VECTOR: {
            auto elem_type = type->element();
            auto dim = type->dimension();
            auto elem_stride = elem_type->size();
            std::vector<spv::Id> comps;
            comps.reserve(dim);
            for (uint i = 0u; i < dim; ++i) {
                auto elem_data = static_cast<const std::byte *>(data) + i * elem_stride;
                comps.emplace_back(_emit_literal(elem_type, elem_data));
            }
            return _builder.makeCompositeConstant(spv_type, comps);
        }
        case Type::Tag::MATRIX: {
            auto elem_type = type->element();
            auto dim = type->dimension();
            auto col_type = Type::vector(elem_type, dim);
            auto col_stride = col_type->size();
            std::vector<spv::Id> cols;
            cols.reserve(dim);
            for (uint i = 0u; i < dim; ++i) {
                auto col_data = static_cast<const std::byte *>(data) + i * col_stride;
                cols.emplace_back(_emit_literal(col_type, col_data));
            }
            return _builder.makeCompositeConstant(spv_type, cols);
        }
        case Type::Tag::ARRAY: {
            auto elem_type = type->element();
            auto dim = type->dimension();
            auto elem_stride = elem_type->size();
            std::vector<spv::Id> elems;
            elems.reserve(dim);
            for (uint i = 0u; i < dim; ++i) {
                auto elem_data = static_cast<const std::byte *>(data) + i * elem_stride;
                elems.emplace_back(_emit_literal(elem_type, elem_data));
            }
            return _builder.makeCompositeConstant(spv_type, elems);
        }
        case Type::Tag::STRUCTURE: {
            auto members = type->members();
            std::vector<spv::Id> member_values;
            member_values.reserve(members.size());
            auto offset = 0u;
            for (auto member : members) {
                offset = luisa::align(offset, member->alignment());
                auto member_data = static_cast<const std::byte *>(data) + offset;
                member_values.emplace_back(_emit_literal(member, member_data));
                offset += member->size();
            }
            return _builder.makeCompositeConstant(spv_type, member_values);
        }
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V constant emission for type {}.", type->description());
    }
}

spv::Id SpirvCodegenEntry::_emit_constant(const xir::Constant *c) noexcept {
    if (auto it = _value_map.find(c); it != _value_map.end()) { return it->second; }
    auto id = _emit_literal(c->type(), c->data());
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
            auto tag = reg->derived_special_register_tag();
            if (tag == xir::DerivedSpecialRegisterTag::DISPATCH_SIZE) {
                // Load dispatch size from push constant (dsp_c.v.xyz)
                auto push_const = _property_ids[0];
                auto uint_type = _builder.makeUintType(32);
                auto ptr = _create_access_chain(spv::StorageClass::PushConstant, push_const,
                                                {_builder.makeUintConstant(0u)});
                auto loaded = _builder.createLoad(ptr, spv::NoPrecision);
                auto x = _builder.createCompositeExtract(loaded, uint_type, 0);
                auto y = _builder.createCompositeExtract(loaded, uint_type, 1);
                auto z = _builder.createCompositeExtract(loaded, uint_type, 2);
                auto uint3_type = _builder.makeVectorType(uint_type, 3);
                id = _builder.createCompositeConstruct(uint3_type, {x, y, z});
                break;
            }
            spv::BuiltIn builtin;
            switch (tag) {
                case xir::DerivedSpecialRegisterTag::THREAD_ID: builtin = spv::BuiltIn::LocalInvocationId; break;
                case xir::DerivedSpecialRegisterTag::BLOCK_ID: builtin = spv::BuiltIn::WorkgroupId; break;
                case xir::DerivedSpecialRegisterTag::DISPATCH_ID: builtin = spv::BuiltIn::GlobalInvocationId; break;
                case xir::DerivedSpecialRegisterTag::BLOCK_SIZE: builtin = spv::BuiltIn::WorkgroupSize; break;
                case xir::DerivedSpecialRegisterTag::WARP_SIZE: builtin = spv::BuiltIn::SubgroupSize; break;
                case xir::DerivedSpecialRegisterTag::WARP_LANE_ID: builtin = spv::BuiltIn::SubgroupLocalInvocationId; break;
                default:
                    LUISA_NOT_IMPLEMENTED("SPIR-V special register {}.", xir::to_string(reg->derived_special_register_tag()));
            }
            spv::Id var;
            if (builtin == spv::BuiltIn::GlobalInvocationId && _global_invocation_id_var != spv::NoResult) {
                var = _global_invocation_id_var;
            } else if (auto it = _builtin_var_map.find(builtin); it != _builtin_var_map.end()) {
                var = it->second;
            } else {
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Input,
                                              _convert_type(reg->type(), Usage::READ), "sr");
                _builder.addDecoration(var, spv::Decoration::BuiltIn, static_cast<int32_t>(builtin));
                if (_entry_point_inst != nullptr) {
                    _entry_point_inst->addIdOperand(var);
                }
                if (builtin == spv::BuiltIn::GlobalInvocationId) {
                    _global_invocation_id_var = var;
                }
                _builtin_var_map.emplace(builtin, var);
            }
            id = _builder.createLoad(var, spv::NoPrecision);
            break;
        }
        case xir::DerivedValueTag::ARGUMENT: {
            auto arg = static_cast<const xir::Argument *>(value);
            if (auto it = _value_map.find(arg); it != _value_map.end()) {
                id = it->second;
                break;
            }
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
    // Do not cache special registers (builtins) because their load instructions
    // must dominate all uses. Caching could place the load inside a loop body
    // and reuse it after the loop, violating SPIR-V dominance rules.
    // Also do not cache Undefined values because OpUndef is added to the
    // current block and must dominate all uses within the same function.
    if (value->derived_value_tag() != xir::DerivedValueTag::SPECIAL_REGISTER &&
        value->derived_value_tag() != xir::DerivedValueTag::UNDEFINED) {
        _value_map.emplace(value, id);
    }
    return id;
}

spv::Id SpirvCodegenEntry::_create_access_chain(spv::StorageClass storage, spv::Id base, const std::vector<spv::Id> &indices, bool nonuniform) noexcept {
    auto old_access_chain = _builder.getAccessChain();
    auto new_access_chain = old_access_chain;
    new_access_chain.base = base;
    new_access_chain.indexChain = indices;
    new_access_chain.swizzle.clear();
    new_access_chain.component = spv::NoResult;
    new_access_chain.descHeapInfo.descHeapBaseTy = spv::NoResult;
    new_access_chain.descHeapInfo.descHeapStorageClass = spv::StorageClass::Max;
    new_access_chain.descHeapInfo.descHeapBaseArrayStride = 0;
    new_access_chain.descHeapInfo.descHeapInstId.clear();
    new_access_chain.descHeapInfo.structRsrcTyOffsetCount = 0;
    new_access_chain.descHeapInfo.structRsrcTyFirstArrIndex = 0;
    new_access_chain.descHeapInfo.structRemappedBase = spv::NoResult;
    if (nonuniform) {
        new_access_chain.coherentFlags.nonUniform = 1;
        for (auto idx : indices) {
            _builder.addDecoration(idx, spv::Decoration::NonUniformEXT);
        }
    }
    _builder.setAccessChain(new_access_chain);
    auto id = _builder.createAccessChain(storage, base, indices);
    _builder.setAccessChain(old_access_chain);
    if (nonuniform) {
        _builder.addDecoration(id, spv::Decoration::NonUniformEXT);
    }
    return id;
}

spv::Block *SpirvCodegenEntry::_get_or_create_block(const xir::BasicBlock *bb) noexcept {
    if (bb == nullptr) { return nullptr; }
    if (auto it = _block_map.find(bb); it != _block_map.end()) { return it->second; }
    auto block = &_builder.makeNewBlock();
    // makeNewBlock already adds the block to the function, no need to add again
    _block_map.emplace(bb, block);
    return block;
}

void SpirvCodegenEntry::_emit_block(const xir::BasicBlock *bb) noexcept {
    if (bb == nullptr) { return; }
    if (!_emitted_blocks.emplace(bb).second) { return; }
    auto spv_block = _get_or_create_block(bb);
    _builder.setBuildPoint(spv_block);
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
    // Entry point must have no parameters in SPIR-V
    auto func = _builder.makeFunctionEntry(spv::NoPrecision, ret_type, "main",
                                           spv::LinkageType::Max, {}, {}, &entry);
    _value_map.emplace(kernel, func->getId());
    _function_map.emplace(kernel, func);
    _block_map.emplace(kernel->body_block(), entry);

    // Load non-resource arguments from the cbuffer (StructuredBuffer at property index 2)
    if (!value_args.empty()) {
        bool cbuffer_non_empty = false;
        for (auto arg : kernel->arguments()) {
            if (!arg->is_resource()) {
                cbuffer_non_empty = true;
                break;
            }
        }
        if (cbuffer_non_empty && _property_ids.size() > 2) {
            auto cbuffer_id = _property_ids[2];
            auto uint_type = _builder.makeUintType(32);
            auto bool_type = _builder.makeBoolType();
            size_t offset = 0;
            for (auto arg : value_args) {
                auto align = arg->type()->alignment();
                offset = (offset + align - 1) & ~(align - 1);
                auto word_offset = _builder.makeUintConstant(static_cast<uint32_t>(offset / 4));
                spv::Id loaded;
                auto byte_in_word = offset % 4;
                auto type_size = arg->type()->size();
                if (byte_in_word != 0 || type_size < 4) {
                    // Sub-word type: read the whole word and extract the relevant byte(s)
                    auto ptr = _create_access_chain(spv::StorageClass::StorageBuffer, cbuffer_id,
                                                    {_builder.makeUintConstant(0u), word_offset});
                    auto raw = _builder.createLoad(ptr, spv::NoPrecision);
                    if (byte_in_word != 0) {
                        raw = _builder.createBinOp(spv::Op::OpShiftRightLogical, uint_type, raw,
                                                   _builder.makeUintConstant(static_cast<uint32_t>(byte_in_word * 8)));
                    }
                    if (arg->type()->is_bool()) {
                        auto masked = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, raw, _builder.makeUintConstant(0xFFu));
                        loaded = _builder.createBinOp(spv::Op::OpINotEqual, bool_type, masked, _builder.makeUintConstant(0u));
                    } else {
                        auto bit_width = static_cast<int32_t>(type_size * 8);
                        auto mask = (1u << bit_width) - 1u;
                        auto masked = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, raw, _builder.makeUintConstant(mask));
                        auto trunc_type = _builder.makeIntegerType(bit_width, arg->type()->is_int());
                        auto truncated = _builder.createUnaryOp(spv::Op::OpUConvert, trunc_type, masked);
                        auto spv_type = _convert_type(arg->type(), Usage::READ);
                        if (trunc_type != spv_type) {
                            loaded = _builder.createUnaryOp(spv::Op::OpBitcast, spv_type, truncated);
                        } else {
                            loaded = truncated;
                        }
                    }
                } else {
                    loaded = _emit_buffer_read_impl(cbuffer_id, word_offset, arg->type());
                }
                _value_map.emplace(arg, loaded);
                offset += type_size;
            }
        } else {
            // Fallback: create undefined values if cbuffer is not available
            for (auto arg : value_args) {
                auto type = _convert_type(arg->type(), Usage::READ);
                _value_map.emplace(arg, _builder.createUndefined(type));
            }
        }
    }

    _entry_point_inst = _builder.addEntryPoint(spv::ExecutionModel::GLCompute, func, "main");
    for (auto id : _property_ids) {
        if (id != spv::NoResult) {
            _entry_point_inst->addIdOperand(id);
        }
    }
    auto bs = kernel->block_size();
    _builder.addExecutionMode(func, spv::ExecutionMode::LocalSize,
                              static_cast<int32_t>(bs.x),
                              static_cast<int32_t>(bs.y),
                              static_cast<int32_t>(bs.z));

    _builder.enterFunction(func);
    _builder.setBuildPoint(entry);

    // Add dispatch bounds check to prevent extra threads in the last workgroup
    // from executing the kernel body.
    {
        auto &function = *func;
        auto uint_type = _builder.makeUintType(32);
        auto bool_type = _builder.makeBoolType();

        // Load dispatch size from push constant
        auto push_const = _property_ids[0];
        auto dsp_size_ptr = _create_access_chain(spv::StorageClass::PushConstant, push_const,
                                                 {_builder.makeUintConstant(0u)});
        auto dsp_size_loaded = _builder.createLoad(dsp_size_ptr, spv::NoPrecision);

        // Load dispatch_id (GlobalInvocationId) - reuse cached builtin if available
        if (_global_invocation_id_var == spv::NoResult) {
            auto uint3_type = _builder.makeVectorType(uint_type, 3);
            _global_invocation_id_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Input,
                                                                uint3_type, "dispatch_id");
            _builder.addDecoration(_global_invocation_id_var, spv::Decoration::BuiltIn, static_cast<int32_t>(spv::BuiltIn::GlobalInvocationId));
            if (_entry_point_inst != nullptr) {
                _entry_point_inst->addIdOperand(_global_invocation_id_var);
            }
        }
        auto dispatch_id = _builder.createLoad(_global_invocation_id_var, spv::NoPrecision);

        // Compare all three components of dispatch_id against dispatch_size
        auto dsp_x = _builder.createCompositeExtract(dsp_size_loaded, uint_type, 0);
        auto dsp_y = _builder.createCompositeExtract(dsp_size_loaded, uint_type, 1);
        auto dsp_z = _builder.createCompositeExtract(dsp_size_loaded, uint_type, 2);
        auto id_x = _builder.createCompositeExtract(dispatch_id, uint_type, 0);
        auto id_y = _builder.createCompositeExtract(dispatch_id, uint_type, 1);
        auto id_z = _builder.createCompositeExtract(dispatch_id, uint_type, 2);
        auto cmp_x = _builder.createBinOp(spv::Op::OpUGreaterThanEqual, bool_type, id_x, dsp_x);
        auto cmp_y = _builder.createBinOp(spv::Op::OpUGreaterThanEqual, bool_type, id_y, dsp_y);
        auto cmp_z = _builder.createBinOp(spv::Op::OpUGreaterThanEqual, bool_type, id_z, dsp_z);
        auto cmp = _builder.createBinOp(spv::Op::OpLogicalOr, bool_type, cmp_x, cmp_y);
        cmp = _builder.createBinOp(spv::Op::OpLogicalOr, bool_type, cmp, cmp_z);

        // Create return block and body block
        auto return_block = new spv::Block(_builder.getUniqueId(), function);
        auto body_block = new spv::Block(_builder.getUniqueId(), function);

        // Selection merge
        auto selection_merge = new spv::Instruction(spv::Op::OpSelectionMerge);
        selection_merge->reserveOperands(2);
        selection_merge->addIdOperand(body_block->getId());
        selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
        entry->addInstruction(std::unique_ptr<spv::Instruction>(selection_merge));

        // Branch conditional
        _builder.createConditionalBranch(cmp, return_block, body_block);

        // Return block
        function.addBlock(return_block);
        _builder.setBuildPoint(return_block);
        _builder.makeReturn(false);

        // Body block
        function.addBlock(body_block);
        _builder.setBuildPoint(body_block);

        // Update block map so XIR body block maps to body_block instead of entry
        _block_map[kernel->body_block()] = body_block;
    }

    _emit_block(kernel->body_block());

    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.makeReturn(false);
    }
    _builder.leaveFunction();
}

void SpirvCodegenEntry::_emit_callable(const xir::CallableFunction *callable, const xir::Module *module) noexcept {
    auto ret_type = _convert_type(callable->type(), Usage::READ);
    std::vector<spv::Id> param_types;
    luisa::vector<const xir::Argument *> emitted_args;
    luisa::vector<bool> arg_used;
    for (auto arg : callable->arguments()) {
        bool used = !arg->use_list().empty();
        arg_used.push_back(used);
        if (!used && arg->is_resource()) {
            // Skip unused resource arguments to avoid type mismatches
            // between kernel globals (which may be arrays or have different
            // sampled/storage qualifiers) and callable parameters.
            continue;
        }
        if (arg->is_resource()) {
            auto type = arg->type();
            spv::Id pointee_type = spv::NoResult;
            spv::StorageClass storage = spv::StorageClass::Max;
            switch (type->tag()) {
                case Type::Tag::BUFFER:
                case Type::Tag::BINDLESS_ARRAY:
                    pointee_type = _convert_type(type, Usage::READ);
                    storage = spv::StorageClass::StorageBuffer;
                    _builder.addIncorporatedExtension("SPV_KHR_variable_pointers", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::VariablePointersStorageBuffer);
                    break;
                case Type::Tag::ACCEL:
                case Type::Tag::TEXTURE:
                    pointee_type = _convert_type(type, Usage::READ);
                    storage = spv::StorageClass::UniformConstant;
                    break;
                default:
                    LUISA_NOT_IMPLEMENTED("SPIR-V callable resource argument type {}", type->description());
            }
            param_types.emplace_back(_builder.makePointer(storage, pointee_type));
        } else if (arg->is_reference()) {
            auto type = _convert_type(arg->type(), Usage::READ);
            param_types.emplace_back(_builder.makePointer(spv::StorageClass::Function, type));
        } else {
            param_types.emplace_back(_convert_type(arg->type(), Usage::READ));
        }
        emitted_args.push_back(arg);
    }
    _callable_arg_used.emplace(callable, std::move(arg_used));
    spv::Block *entry = nullptr;
    auto func = _builder.makeFunctionEntry(spv::NoPrecision, ret_type,
                                           luisa::string{callable->name().value_or("callable")}.c_str(),
                                           spv::LinkageType::Max,
                                           param_types, {}, &entry);
    _value_map.emplace(callable, func->getId());
    _function_map.emplace(callable, func);

    int32_t i = 0;
    for (auto arg : emitted_args) {
        auto param_id = func->getParamId(i);
        _value_map.emplace(arg, param_id);
        if (arg->type()->tag() == Type::Tag::TEXTURE) {
            // Callable texture parameters are always created as sampled images
            // (Usage::READ in _convert_type), so they are not storage images.
            _is_storage_image_map.emplace(param_id, false);
        }
        ++i;
    }

    _builder.enterFunction(func);
    _builder.setBuildPoint(entry);
    _block_map.emplace(callable->body_block(), entry);
    _emit_block(callable->body_block());
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

    // Detect buffers that need atomic access (must stay as uint32 arrays)
    {
        for (auto f : analysis.used_functions_post_order) {
            if (auto def = f->definition()) {
                def->traverse_instructions([&](const xir::Instruction *inst) noexcept {
                    if (inst->isa<xir::AtomicInst>()) {
                        auto atomic = static_cast<const xir::AtomicInst *>(inst);
                        auto base = atomic->operand(0);
                        if (base != nullptr && base->type() != nullptr && base->type()->is_buffer()) {
                            _needs_atomic_buffer_types.emplace(base->type());
                        }
                    }
                });
            }
        }
    }

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
                    _emit_callable(static_cast<const xir::CallableFunction *>(f), module);
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