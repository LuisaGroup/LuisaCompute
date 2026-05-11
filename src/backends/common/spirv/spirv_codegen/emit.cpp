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

spv::Id SpirvCodegenEntry::_convert_type(const Type *type) noexcept {
    if (type == nullptr) { return _builder.makeVoidType(); }
    if (auto it = _type_map.find(type); it != _type_map.end()) { return it->second; }
    spv::Id id = spv::NoResult;
    switch (type->tag()) {
        case Type::Tag::BOOL: id = _builder.makeBoolType(); break;
        case Type::Tag::FLOAT16: id = _builder.makeFloatType(16); break;
        case Type::Tag::FLOAT32: id = _builder.makeFloatType(32); break;
        case Type::Tag::FLOAT64: id = _builder.makeFloatType(64); break;
        case Type::Tag::INT8: id = _builder.makeIntType(8); break;
        case Type::Tag::UINT8: id = _builder.makeUintType(8); break;
        case Type::Tag::INT16: id = _builder.makeIntType(16); break;
        case Type::Tag::UINT16: id = _builder.makeUintType(16); break;
        case Type::Tag::INT32: id = _builder.makeIntType(32); break;
        case Type::Tag::UINT32: id = _builder.makeUintType(32); break;
        case Type::Tag::INT64: id = _builder.makeIntType(64); break;
        case Type::Tag::UINT64: id = _builder.makeUintType(64); break;
        case Type::Tag::VECTOR:
            id = _builder.makeVectorType(_convert_type(type->element()), static_cast<int>(type->dimension()));
            break;
        case Type::Tag::MATRIX:
            id = _builder.makeMatrixType(_convert_type(type->element()),
                                         static_cast<int>(type->dimension()),
                                         static_cast<int>(type->dimension()));
            break;
        case Type::Tag::ARRAY: {
            auto elem_type = _convert_type(type->element());
            auto size_id = _builder.makeUintConstant(static_cast<unsigned>(type->dimension()));
            id = _builder.makeArrayType(elem_type, size_id, 0);
            break;
        }
        case Type::Tag::STRUCTURE: {
            auto members = type->members();
            std::vector<spv::Id> member_types;
            member_types.reserve(members.size());
            for (auto m : members) { member_types.emplace_back(_convert_type(m)); }
            std::vector<spv::StructMemberDebugInfo> member_debug;
            id = _builder.makeStructType(member_types, member_debug, "Struct", false);
            break;
        }
        case Type::Tag::BUFFER:
        case Type::Tag::TEXTURE:
        case Type::Tag::BINDLESS_ARRAY:
        case Type::Tag::ACCEL:
        case Type::Tag::CUSTOM:
            LUISA_NOT_IMPLEMENTED("SPIR-V type conversion for resource/custom type {}.", type->description());
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to convert type {}.", type->description());
    _type_map.emplace(type, id);
    return id;
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
    auto spv_type = _convert_type(type);
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
            id = _builder.createUndefined(_convert_type(value->type()));
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
                                               _convert_type(reg->type()), "sr");
            _builder.addDecoration(var, spv::Decoration::BuiltIn, (int)builtin);
            id = _builder.createLoad(var, spv::NoPrecision);
            break;
        }
        case xir::DerivedValueTag::FUNCTION:
        case xir::DerivedValueTag::ARGUMENT:
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

void SpirvCodegenEntry::_emit_instruction(const xir::Instruction *inst) noexcept {
    auto set_result = [&](spv::Id id) noexcept {
        if (inst->type() != nullptr) {
            _value_map.emplace(inst, id);
        }
    };
    switch (inst->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::ALLOCA: {
            auto alloca = static_cast<const xir::AllocaInst *>(inst);
            if (alloca->is_shared()) {
                LUISA_NOT_IMPLEMENTED("SPIR-V shared alloca.");
            }
            auto type = _convert_type(alloca->type());
            auto var = _builder.createVariable(spv::NoPrecision,
                                               spv::StorageClass::Function,
                                               type, "alloca");
            set_result(var);
            break;
        }
        case xir::DerivedInstructionTag::LOAD: {
            auto load = static_cast<const xir::LoadInst *>(inst);
            auto ptr = _emit_value(load->variable());
            auto id = _builder.createLoad(ptr, spv::NoPrecision);
            set_result(id);
            break;
        }
        case xir::DerivedInstructionTag::STORE: {
            auto store = static_cast<const xir::StoreInst *>(inst);
            auto ptr = _emit_value(store->variable());
            auto val = _emit_value(store->value());
            _builder.createStore(val, ptr);
            break;
        }
        case xir::DerivedInstructionTag::GEP: {
            auto gep = static_cast<const xir::GEPInst *>(inst);
            auto base = _emit_value(gep->base());
            std::vector<spv::Id> indices;
            for (auto index_use : gep->index_uses()) {
                indices.emplace_back(_emit_value(index_use->value()));
            }
            auto storage = _builder.getStorageClass(base);
            auto id = _builder.createAccessChain(storage, base, indices);
            set_result(id);
            break;
        }
        case xir::DerivedInstructionTag::ARITHMETIC:
            _emit_arithmetic_inst(static_cast<const xir::ArithmeticInst *>(inst));
            break;
        case xir::DerivedInstructionTag::CALL: {
            auto call = static_cast<const xir::CallInst *>(inst);
            auto callee_func = _function_map.at(call->callee());
            std::vector<spv::Id> args;
            for (auto arg_use : call->argument_uses()) {
                args.emplace_back(_emit_value(arg_use->value()));
            }
            auto id = _builder.createFunctionCall(callee_func, args);
            set_result(id);
            break;
        }
        case xir::DerivedInstructionTag::CAST: {
            auto cast = static_cast<const xir::CastInst *>(inst);
            auto val = _emit_value(cast->value());
            auto from = cast->value()->type();
            auto to = cast->type();
            auto spv_to = _convert_type(to);
            spv::Id id = spv::NoResult;
            if (cast->op() == xir::CastOp::BITWISE_CAST) {
                id = _builder.createUnaryOp(spv::Op::OpBitcast, spv_to, val);
            } else {
                if (from == to) {
                    id = val;
                } else if (from->is_bool() && to->is_scalar()) {
                    spv::Id zero = spv::NoResult;
                    spv::Id one = spv::NoResult;
                    if (to->is_int32()) {
                        zero = _builder.makeIntConstant(0);
                        one = _builder.makeIntConstant(1);
                    } else if (to->is_uint32()) {
                        zero = _builder.makeUintConstant(0);
                        one = _builder.makeUintConstant(1);
                    } else if (to->is_float32()) {
                        zero = _builder.makeFloatConstant(0.0f);
                        one = _builder.makeFloatConstant(1.0f);
                    } else {
                        LUISA_NOT_IMPLEMENTED("SPIR-V bool-to-scalar cast for {}.", to->description());
                    }
                    id = _builder.createTriOp(spv::Op::OpSelect, spv_to, val, one, zero);
                } else if (to->is_bool() && from->is_scalar()) {
                    spv::Id zero = spv::NoResult;
                    if (from->is_int32()) {
                        zero = _builder.makeIntConstant(0);
                    } else if (from->is_uint32()) {
                        zero = _builder.makeUintConstant(0);
                    } else if (from->is_float32()) {
                        zero = _builder.makeFloatConstant(0.0f);
                    } else {
                        LUISA_NOT_IMPLEMENTED("SPIR-V scalar-to-bool cast for {}.", from->description());
                    }
                    if (from->is_float()) {
                        id = _builder.createBinOp(spv::Op::OpFOrdNotEqual, spv_to, val, zero);
                    } else {
                        id = _builder.createBinOp(spv::Op::OpINotEqual, spv_to, val, zero);
                    }
                } else if (from->is_float() && to->is_int()) {
                    id = _builder.createUnaryOp(spv::Op::OpConvertFToS, spv_to, val);
                } else if (from->is_float() && to->is_uint()) {
                    id = _builder.createUnaryOp(spv::Op::OpConvertFToU, spv_to, val);
                } else if (from->is_int() && to->is_float()) {
                    id = _builder.createUnaryOp(spv::Op::OpConvertSToF, spv_to, val);
                } else if (from->is_uint() && to->is_float()) {
                    id = _builder.createUnaryOp(spv::Op::OpConvertUToF, spv_to, val);
                } else if (from->is_float() && to->is_float()) {
                    id = _builder.createUnaryOp(spv::Op::OpFConvert, spv_to, val);
                } else if ((from->is_int() || from->is_uint()) && (to->is_int() || to->is_uint())) {
                    if (from->size() == to->size()) {
                        id = val;
                    } else if (from->is_int() && to->is_int()) {
                        id = _builder.createUnaryOp(spv::Op::OpSConvert, spv_to, val);
                    } else if (from->is_uint() && to->is_uint()) {
                        id = _builder.createUnaryOp(spv::Op::OpUConvert, spv_to, val);
                    } else {
                        id = _builder.createUnaryOp(spv::Op::OpBitcast, spv_to, val);
                    }
                } else {
                    LUISA_NOT_IMPLEMENTED("SPIR-V static cast from {} to {}.", from->description(), to->description());
                }
            }
            set_result(id);
            break;
        }
        case xir::DerivedInstructionTag::IF: _emit_if_inst(static_cast<const xir::IfInst *>(inst)); break;
        case xir::DerivedInstructionTag::LOOP: _emit_loop_inst(static_cast<const xir::LoopInst *>(inst)); break;
        case xir::DerivedInstructionTag::SIMPLE_LOOP: _emit_simple_loop_inst(static_cast<const xir::SimpleLoopInst *>(inst)); break;
        case xir::DerivedInstructionTag::SWITCH: _emit_switch_inst(static_cast<const xir::SwitchInst *>(inst)); break;
        case xir::DerivedInstructionTag::BRANCH: _emit_branch_inst(static_cast<const xir::BranchInst *>(inst)); break;
        case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: _emit_conditional_branch_inst(static_cast<const xir::ConditionalBranchInst *>(inst)); break;
        case xir::DerivedInstructionTag::BREAK: {
            auto br = static_cast<const xir::BreakInst *>(inst);
            _builder.createBranch(false, _get_or_create_block(br->target_block()));
            break;
        }
        case xir::DerivedInstructionTag::CONTINUE: {
            auto cont = static_cast<const xir::ContinueInst *>(inst);
            _builder.createBranch(false, _get_or_create_block(cont->target_block()));
            break;
        }
        case xir::DerivedInstructionTag::RETURN: {
            auto ret = static_cast<const xir::ReturnInst *>(inst);
            if (ret->return_value()) {
                _builder.makeReturn(false, _emit_value(ret->return_value()));
            } else {
                _builder.makeReturn(false);
            }
            break;
        }
        case xir::DerivedInstructionTag::UNREACHABLE:
            _builder.createNoResultOp(spv::Op::OpUnreachable);
            break;
        case xir::DerivedInstructionTag::ATOMIC: _emit_atomic_inst(static_cast<const xir::AtomicInst *>(inst)); break;
        case xir::DerivedInstructionTag::RESOURCE_QUERY: _emit_resource_query_inst(static_cast<const xir::ResourceQueryInst *>(inst)); break;
        case xir::DerivedInstructionTag::RESOURCE_READ: _emit_resource_read_inst(static_cast<const xir::ResourceReadInst *>(inst)); break;
        case xir::DerivedInstructionTag::RESOURCE_WRITE: _emit_resource_write_inst(static_cast<const xir::ResourceWriteInst *>(inst)); break;
        case xir::DerivedInstructionTag::THREAD_GROUP: _emit_thread_group_inst(static_cast<const xir::ThreadGroupInst *>(inst)); break;
        case xir::DerivedInstructionTag::PHI:
            LUISA_ERROR_WITH_LOCATION("Phi instructions should be eliminated before SPIR-V codegen.");
        case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
        case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
        case xir::DerivedInstructionTag::AUTODIFF_SCOPE:
        case xir::DerivedInstructionTag::AUTODIFF_INTRINSIC:
            LUISA_ERROR_WITH_LOCATION("Instruction {} should be eliminated before SPIR-V codegen.",
                                      xir::to_string(inst->derived_instruction_tag()));
        case xir::DerivedInstructionTag::PRINT:
        case xir::DerivedInstructionTag::CLOCK:
        case xir::DerivedInstructionTag::ASSERT:
        case xir::DerivedInstructionTag::ASSUME:
        case xir::DerivedInstructionTag::DEBUG_BREAK:
        case xir::DerivedInstructionTag::OUTLINE:
        case xir::DerivedInstructionTag::RASTER_DISCARD:
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
        case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE:
            LUISA_NOT_IMPLEMENTED("SPIR-V codegen for instruction {}.", xir::to_string(inst->derived_instruction_tag()));
    }
}

void SpirvCodegenEntry::_emit_arithmetic_inst(const xir::ArithmeticInst *inst) noexcept {
    auto type = _convert_type(inst->type());
    auto is_float = inst->type()->is_float();
    auto is_signed_int = inst->type()->is_int();
    auto is_bool = inst->type()->is_bool();

    auto operand = [&](size_t i) noexcept { return _emit_value(inst->operand(i)); };
    spv::Id id = spv::NoResult;

    switch (inst->op()) {
        case xir::ArithmeticOp::UNARY_MINUS:
            if (is_float)
                id = _builder.createUnaryOp(spv::Op::OpFNegate, type, operand(0));
            else
                id = _builder.createUnaryOp(spv::Op::OpSNegate, type, operand(0));
            break;
        case xir::ArithmeticOp::UNARY_BIT_NOT:
            if (is_bool)
                id = _builder.createUnaryOp(spv::Op::OpLogicalNot, type, operand(0));
            else
                id = _builder.createUnaryOp(spv::Op::OpNot, type, operand(0));
            break;
        case xir::ArithmeticOp::BINARY_ADD:
            if (is_float)
                id = _builder.createBinOp(spv::Op::OpFAdd, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpIAdd, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_SUB:
            if (is_float)
                id = _builder.createBinOp(spv::Op::OpFSub, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpISub, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_MUL:
            if (is_float)
                id = _builder.createBinOp(spv::Op::OpFMul, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpIMul, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_DIV:
            if (is_float)
                id = _builder.createBinOp(spv::Op::OpFDiv, type, operand(0), operand(1));
            else if (is_signed_int)
                id = _builder.createBinOp(spv::Op::OpSDiv, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpUDiv, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_MOD:
            if (is_float)
                id = _builder.createBinOp(spv::Op::OpFMod, type, operand(0), operand(1));
            else if (is_signed_int)
                id = _builder.createBinOp(spv::Op::OpSMod, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpUMod, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_BIT_AND:
            if (is_bool)
                id = _builder.createBinOp(spv::Op::OpLogicalAnd, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpBitwiseAnd, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_BIT_OR:
            if (is_bool)
                id = _builder.createBinOp(spv::Op::OpLogicalOr, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpBitwiseOr, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_BIT_XOR:
            if (is_bool)
                id = _builder.createBinOp(spv::Op::OpLogicalNotEqual, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpBitwiseXor, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_SHIFT_LEFT:
            id = _builder.createBinOp(spv::Op::OpShiftLeftLogical, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_SHIFT_RIGHT:
            if (is_signed_int)
                id = _builder.createBinOp(spv::Op::OpShiftRightArithmetic, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpShiftRightLogical, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_LESS:
            if (is_float)
                id = _builder.createBinOp(spv::Op::OpFOrdLessThan, type, operand(0), operand(1));
            else if (is_signed_int)
                id = _builder.createBinOp(spv::Op::OpSLessThan, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpULessThan, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_GREATER:
            if (is_float)
                id = _builder.createBinOp(spv::Op::OpFOrdGreaterThan, type, operand(0), operand(1));
            else if (is_signed_int)
                id = _builder.createBinOp(spv::Op::OpSGreaterThan, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpUGreaterThan, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_LESS_EQUAL:
            if (is_float)
                id = _builder.createBinOp(spv::Op::OpFOrdLessThanEqual, type, operand(0), operand(1));
            else if (is_signed_int)
                id = _builder.createBinOp(spv::Op::OpSLessThanEqual, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpULessThanEqual, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
            if (is_float)
                id = _builder.createBinOp(spv::Op::OpFOrdGreaterThanEqual, type, operand(0), operand(1));
            else if (is_signed_int)
                id = _builder.createBinOp(spv::Op::OpSGreaterThanEqual, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpUGreaterThanEqual, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_EQUAL:
            if (is_float)
                id = _builder.createBinOp(spv::Op::OpFOrdEqual, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpIEqual, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::BINARY_NOT_EQUAL:
            if (is_float)
                id = _builder.createBinOp(spv::Op::OpFOrdNotEqual, type, operand(0), operand(1));
            else
                id = _builder.createBinOp(spv::Op::OpINotEqual, type, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::SELECT:
            id = _builder.createTriOp(spv::Op::OpSelect, type, operand(0), operand(1), operand(2));
            break;
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V arithmetic op {}.", xir::to_string(inst->op()));
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit arithmetic op.");
    if (inst->type() != nullptr) {
        _value_map.emplace(inst, id);
    }
}

void SpirvCodegenEntry::_emit_atomic_inst(const xir::AtomicInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED("SPIR-V atomic instruction {}.", xir::to_string(inst->op()));
}

void SpirvCodegenEntry::_emit_resource_query_inst(const xir::ResourceQueryInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED("SPIR-V resource query instruction {}.", xir::to_string(inst->op()));
}

void SpirvCodegenEntry::_emit_resource_read_inst(const xir::ResourceReadInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED("SPIR-V resource read instruction {}.", xir::to_string(inst->op()));
}

void SpirvCodegenEntry::_emit_resource_write_inst(const xir::ResourceWriteInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED("SPIR-V resource write instruction {}.", xir::to_string(inst->op()));
}

void SpirvCodegenEntry::_emit_thread_group_inst(const xir::ThreadGroupInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED("SPIR-V thread group instruction {}.", xir::to_string(inst->op()));
}

void SpirvCodegenEntry::_emit_if_inst(const xir::IfInst *inst) noexcept {
    auto cond = _emit_value(inst->condition());
    auto &function = _builder.getBuildPoint()->getParent();
    auto true_block = new spv::Block(_builder.getUniqueId(), function);
    auto false_block = new spv::Block(_builder.getUniqueId(), function);
    auto merge_block = new spv::Block(_builder.getUniqueId(), function);
    _block_map[inst->true_block()] = true_block;
    _block_map[inst->false_block()] = false_block;
    _block_map[inst->merge_block()] = merge_block;
    auto selection_merge = new spv::Instruction(spv::Op::OpSelectionMerge);
    selection_merge->reserveOperands(2);
    selection_merge->addIdOperand(merge_block->getId());
    selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
    _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(selection_merge));
    _builder.createConditionalBranch(cond, true_block, false_block);
    function.addBlock(true_block);
    _builder.setBuildPoint(true_block);
    _emit_block(inst->true_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, merge_block);
    }
    function.addBlock(false_block);
    _builder.setBuildPoint(false_block);
    _emit_block(inst->false_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, merge_block);
    }
    function.addBlock(merge_block);
    _builder.setBuildPoint(merge_block);
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_loop_inst(const xir::LoopInst *inst) noexcept {
    auto prepare = _get_or_create_block(inst->prepare_block());
    auto body = _get_or_create_block(inst->body_block());
    auto update = _get_or_create_block(inst->update_block());
    auto merge = _get_or_create_block(inst->merge_block());
    _loop_header_info.emplace(inst->prepare_block(), std::make_pair(merge, update));
    _builder.createBranch(false, prepare);
    _emit_block(inst->prepare_block());
    _emit_block(inst->body_block());
    _emit_block(inst->update_block());
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_simple_loop_inst(const xir::SimpleLoopInst *inst) noexcept {
    auto body = _get_or_create_block(inst->body_block());
    auto merge = _get_or_create_block(inst->merge_block());
    _loop_header_info.emplace(inst->body_block(), std::make_pair(merge, body));
    _builder.createBranch(false, body);
    _emit_block(inst->body_block());
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_switch_inst(const xir::SwitchInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED("SPIR-V switch instruction.");
}

void SpirvCodegenEntry::_emit_branch_inst(const xir::BranchInst *inst) noexcept {
    _builder.createBranch(false, _get_or_create_block(inst->target_block()));
    _emit_block(inst->target_block());
}

void SpirvCodegenEntry::_emit_conditional_branch_inst(const xir::ConditionalBranchInst *inst) noexcept {
    auto cond = _emit_value(inst->condition());
    auto true_block = _get_or_create_block(inst->true_block());
    auto false_block = _get_or_create_block(inst->false_block());
    _builder.createConditionalBranch(cond, true_block, false_block);
    _emit_block(inst->true_block());
    _emit_block(inst->false_block());
}

void SpirvCodegenEntry::_emit_kernel(const xir::KernelFunction *kernel) noexcept {
    auto ret_type = _builder.makeVoidType();
    std::vector<spv::Id> param_types;
    for (auto arg : kernel->arguments()) {
        param_types.emplace_back(_convert_type(arg->type()));
    }
    spv::Block *entry = nullptr;
    auto func = _builder.makeFunctionEntry(spv::NoPrecision, ret_type, "main",
                                           spv::LinkageType::Max, param_types, {}, &entry);
    _value_map.emplace(kernel, func->getId());
    _function_map.emplace(kernel, func);

    auto i = 0u;
    for (auto arg : kernel->arguments()) {
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
    auto ret_type = _convert_type(callable->type());
    std::vector<spv::Id> param_types;
    for (auto arg : callable->arguments()) {
        param_types.emplace_back(_convert_type(arg->type()));
    }
    spv::Block *entry = nullptr;
    auto func = _builder.makeFunctionEntry(spv::NoPrecision, ret_type,
                                           luisa::string{callable->name().value_or("callable")}.c_str(),
                                           spv::LinkageType::Max,
                                           param_types, {}, &entry);
    _value_map.emplace(callable, func->getId());
    _function_map.emplace(callable, func);

    auto i = 0u;
    for (auto arg : callable->arguments()) {
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
        if (type != nullptr) { _convert_type(type); }
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

}