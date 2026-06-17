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
      _builder_ptr{luisa::make_unique<spv::Builder>(spv::Spv_1_5, 0, &_logger)},
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
    _sampled_image_type_map.clear();
    _storage_image_type_map.clear();
    _value_map.clear();
    _function_map.clear();
    _block_map.clear();
    _loop_header_info.clear();
    _loop_header_redirect.clear();
    _emitted_blocks.clear();
    _used_merge_blocks.clear();
    _pending_blocks.clear();
    _added_blocks.clear();
    _emitting_values.clear();
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
        case Type::Tag::FLOAT8_E4M3: return builder.makeFloatE4M3Constant(*static_cast<const float *>(data));
        case Type::Tag::FLOAT8_E5M2: return builder.makeFloatE5M2Constant(*static_cast<const float *>(data));
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
        case Type::Tag::FLOAT8_E4M3:
        case Type::Tag::FLOAT8_E5M2:
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
    if (auto ubo_it = _ubo_constant_member_by_hash.find(c->hash());
        ubo_it != _ubo_constant_member_by_hash.end()) {
        auto ptr = _create_access_chain(spv::StorageClass::Uniform, _constant_ubo_var,
                                        {_builder.makeUintConstant(ubo_it->second)});
        return _builder.createLoad(ptr, spv::NoPrecision);
    }
    auto id = _emit_literal(c->type(), c->data());
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit constant.");
    _value_map.emplace(c, id);
    return id;
}

spv::Id SpirvCodegenEntry::_emit_alloca(const xir::AllocaInst *alloca) noexcept {
    if (auto iter = _value_map.find(alloca); iter != _value_map.end()) { return iter->second; }
    auto type = _convert_type(alloca->type(), Usage::READ);
    auto storage = alloca->is_shared() ? spv::StorageClass::Workgroup : spv::StorageClass::Function;
    auto var = _builder.createVariable(spv::NoPrecision, storage, type, "alloca");
    if (storage == spv::StorageClass::Workgroup && _entry_point_inst != nullptr) {
        _entry_point_inst->addIdOperand(var);
    }
    _value_map.emplace(alloca, var);
    return var;
}

spv::Id SpirvCodegenEntry::_emit_value(const xir::Value *value) noexcept {
    if (auto it = _value_map.find(value); it != _value_map.end()) { return it->second; }
    spv::Id id = spv::NoResult;
    switch (value->derived_value_tag()) {
        case xir::DerivedValueTag::CONSTANT:
            id = _emit_constant(static_cast<const xir::Constant *>(value));
            break;
        case xir::DerivedValueTag::UNDEFINED: {
            auto spv_type = _convert_type(value->type(), Usage::READ);
            if (_builder.isPointerType(spv_type)) {
                id = _builder.createUndefined(spv_type);
            } else {
                id = _builder.makeNullConstant(spv_type);
            }
            break;
        }
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
            if (builtin == spv::BuiltIn::SubgroupSize ||
                builtin == spv::BuiltIn::SubgroupLocalInvocationId) {
                _builder.addCapability(spv::Capability::GroupNonUniform);
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
        case xir::DerivedValueTag::INSTRUCTION: {
            auto *inst = static_cast<const xir::Instruction *>(value);
            // Guard against recursive cycles
            if (!_emitting_values.emplace(value).second) {
                // Cycle detected: create OpUndef as placeholder
                auto spv_type = _convert_type(value->type(), Usage::READ);
                id = _builder.createUndefined(spv_type);
                _value_map.emplace(value, id);
                break;
            }
            // Try to emit the parent block if it hasn't been emitted
            if (auto *parent = inst->parent_block();
                parent != nullptr && !_emitted_blocks.contains(parent)) {
                auto *saved_bp = _builder.getBuildPoint();
                _emit_block(parent);
                if (saved_bp != nullptr && _builder.getBuildPoint() != saved_bp) {
                    _builder.setBuildPoint(saved_bp);
                }
            }
            // If still not mapped, the parent block may be in the middle of being
            // emitted (a forward reference inside the same block).  Emit just this
            // instruction into the current block, but only if the current block is
            // the parent and hasn't been terminated yet.  Otherwise we would append
            // instructions after a terminator, which violates SPIR-V structural
            // rules.
            if (auto it = _value_map.find(value); it == _value_map.end()) {
                auto *saved_bp = _builder.getBuildPoint();
                bool can_direct_emit = false;
                if (auto *parent = inst->parent_block()) {
                    if (auto *parent_block = _get_or_create_block(parent)) {
                        can_direct_emit = (_builder.getBuildPoint() == parent_block) &&
                                          !parent_block->isTerminated();
                    }
                }
                if (can_direct_emit) {
                    LUISA_VERBOSE("_emit_value direct emit for XIR inst {} into block {}",
                                  reinterpret_cast<uintptr_t>(inst), _builder.getBuildPoint()->getId());
                    _emit_instruction(inst);
                } else {
                    LUISA_VERBOSE("_emit_value skipping direct emit for XIR inst {} (buildpoint={} parent={})",
                                  reinterpret_cast<uintptr_t>(inst),
                                  _builder.getBuildPoint() ? _builder.getBuildPoint()->getId() : 0,
                                  reinterpret_cast<uintptr_t>(inst->parent_block()));
                }
                if (saved_bp != nullptr) {
                    _builder.setBuildPoint(saved_bp);
                }
            }
            _emitting_values.erase(value);
            if (auto it = _value_map.find(value); it != _value_map.end()) {
                id = it->second;
                break;
            }
            [[fallthrough]];
        }
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

void SpirvCodegenEntry::_predeclare_allocas(const xir::FunctionDefinition *def) noexcept {
    for (auto *bb : def->basic_blocks()) {
        for (auto *inst : bb->instructions()) {
            if (inst->isa<xir::AllocaInst>()) {
                _emit_alloca(static_cast<const xir::AllocaInst *>(inst));
            }
        }
    }
}

spv::Block *SpirvCodegenEntry::_get_or_create_block(const xir::BasicBlock *bb) noexcept {
    if (bb == nullptr) { return nullptr; }
    if (auto it = _block_map.find(bb); it != _block_map.end()) { return it->second; }
    auto &function = _builder.getBuildPoint()->getParent();
    auto block = new spv::Block(_builder.getUniqueId(), function);
    _block_map.emplace(bb, block);
    return block;
}

void SpirvCodegenEntry::_emit_block(const xir::BasicBlock *bb, spv::Block *override_spv_block) noexcept {
    if (bb == nullptr) { return; }
    if (!_emitted_blocks.emplace(bb).second) { return; }
    auto spv_block = override_spv_block != nullptr ? override_spv_block : _get_or_create_block(bb);
    // If an override block is used, make sure the XIR block maps to it so that
    // later references (e.g., from _emit_value) resolve to the block that
    // actually contains the emitted instructions.
    if (override_spv_block != nullptr) {
        _block_map[bb] = override_spv_block;
    }
    // If the chosen SPIR-V block is already terminated, we cannot append more
    // instructions to it.  This can happen when an XIR block is mapped to a
    // SPIR-V block that was previously used as a merge target.  Create a fresh
    // block for the remaining instructions so they stay inside a valid block.
    if (spv_block->isTerminated()) {
        LUISA_VERBOSE("_emit_block: block {} is already terminated; creating fresh block for XIR block {}",
                      spv_block->getId(), reinterpret_cast<uintptr_t>(bb));
        auto &function = spv_block->getParent();
        spv_block = new spv::Block(_builder.getUniqueId(), function);
        _block_map[bb] = spv_block;
    }
    if (!_added_blocks.contains(spv_block)) {
        spv_block->getParent().addBlock(spv_block);
        _added_blocks.emplace(spv_block);
    }
    _builder.setBuildPoint(spv_block);
    for (auto inst : bb->instructions()) {
        _emit_instruction(inst);
    }
}

void SpirvCodegenEntry::_pre_register_merge_blocks(const xir::FunctionDefinition *def) noexcept {
    for (auto *bb : def->basic_blocks()) {
        if (!bb->is_terminated()) { continue; }
        const xir::BasicBlock *merge = nullptr;
        switch (bb->terminator()->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::IF:
                merge = static_cast<const xir::IfInst *>(bb->terminator())->merge_block();
                break;
            case xir::DerivedInstructionTag::LOOP:
                merge = static_cast<const xir::LoopInst *>(bb->terminator())->merge_block();
                break;
            case xir::DerivedInstructionTag::SIMPLE_LOOP:
                merge = static_cast<const xir::SimpleLoopInst *>(bb->terminator())->merge_block();
                break;
            case xir::DerivedInstructionTag::SWITCH:
                merge = static_cast<const xir::SwitchInst *>(bb->terminator())->merge_block();
                break;
            default: break;
        }
        if (merge != nullptr) {
            _get_or_create_block(merge);
        }
    }
}
void SpirvCodegenEntry::_reset_function_codegen_state() noexcept {
    _emitted_blocks.clear();
    _pending_blocks.clear();
    _loop_header_redirect.clear();
    _loop_header_info.clear();
    _used_merge_blocks.clear();
    _outer_merge_stack.clear();
    _block_map.clear();
    _loop_boundary_stack.clear();
    _added_blocks.clear();
    _emitting_values.clear();
    _rq_proceed_result.clear();
    _dom_tree.reset();
}

void SpirvCodegenEntry::_emit_function_blocks(const xir::FunctionDefinition *def) noexcept {
    luisa::vector<const xir::BasicBlock *> blocks;
    blocks.reserve(64u);
    def->traverse_basic_blocks(xir::BasicBlockTraversalOrder::REVERSE_POST_ORDER,
                               [&](const xir::BasicBlock *bb) noexcept {
                                   blocks.emplace_back(bb);
                               });
    for (auto bb : blocks) {
        _emit_block(bb);
    }
}

void SpirvCodegenEntry::_emit_kernel(const xir::KernelFunction *kernel) noexcept {
    _reset_function_codegen_state();
    _dom_tree = luisa::make_unique<xir::PostDomTree>(xir::compute_post_dom_tree(const_cast<xir::KernelFunction *>(kernel)));
    _uniformity.analyze(kernel);
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
    _added_blocks.emplace(entry);
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
                _mark_8bit_storage_usage(arg->type(), _builder.getStorageClass(cbuffer_id));
                auto align = arg->type()->alignment();
                offset = (offset + align - 1) & ~(align - 1);
                auto word_offset = _builder.makeUintConstant(static_cast<uint32_t>(offset / 4));
                spv::Id loaded;
                auto byte_in_word = offset % 4;
                auto type_size = arg->type()->size();
                if (byte_in_word != 0 || type_size < 4) {
                    // Sub-word type: read the whole word and extract the relevant byte(s)
                    auto ptr = _create_access_chain(_builder.getStorageClass(cbuffer_id), cbuffer_id,
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
            // Fallback: create null values if cbuffer is not available
            for (auto arg : value_args) {
                auto type = _convert_type(arg->type(), Usage::READ);
                _value_map.emplace(arg, _builder.makeNullConstant(type));
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
        _added_blocks.emplace(return_block);
        _builder.setBuildPoint(return_block);
        _builder.makeReturn(false);

        // Body block
        function.addBlock(body_block);
        _added_blocks.emplace(body_block);
        _builder.setBuildPoint(body_block);

        // Update block map so XIR body block maps to body_block instead of entry
        _block_map[kernel->body_block()] = body_block;
        // Track dispatch bounds check body_block as a used merge so nested
        // constructs that would reuse it create synthetic merges instead.
        _used_merge_blocks.emplace(body_block->getId());
    }

    _pre_register_merge_blocks(kernel);
    _predeclare_allocas(kernel);
    _emit_function_blocks(kernel);

    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.makeReturn(false);
    }
    _builder.leaveFunction();
}

Usage SpirvCodegenEntry::_resource_argument_binding_usage(const xir::Argument *argument) const noexcept {
    if (argument == nullptr || !argument->is_resource()) { return Usage::NONE; }
    auto func = argument->parent_function();
    if (func == nullptr || func->derived_function_tag() != xir::DerivedFunctionTag::KERNEL) { return Usage::NONE; }
    auto prop_index = _get_resource_property_base(func);
    auto found = false;
    for (auto arg : func->arguments()) {
        if (!arg->is_resource()) { continue; }
        if (arg == argument) {
            found = true;
            break;
        }
        ++prop_index;
        if (arg->type()->tag() == Type::Tag::ACCEL) { ++prop_index; }
    }
    if (!found || prop_index == 0u || prop_index > _properties.size()) { return Usage::NONE; }
    auto prop_idx = prop_index - 1u;
    auto usage = Usage::READ;
    switch (_properties[prop_idx].type) {
        case ShaderVariableType::RWStructuredBuffer:
        case ShaderVariableType::UAVBufferHeap:
        case ShaderVariableType::UAVTextureHeap:
            usage = Usage::READ_WRITE;
            break;
        default:
            break;
    }
    return usage;
}

void SpirvCodegenEntry::_analyze_function_argument_usage(const xir::Module *module) noexcept {
    _function_argument_usage.clear();
    auto merge_usage = [](Usage lhs, Usage rhs) noexcept {
        return static_cast<Usage>(luisa::to_underlying(lhs) | luisa::to_underlying(rhs));
    };
    luisa::unordered_map<const xir::Function *, luisa::unordered_map<const xir::Argument *, size_t>> arg_indices;
    for (auto function : module->function_list()) {
        if (!function->is_definition()) { continue; }
        auto count = 0u;
        luisa::unordered_map<const xir::Argument *, size_t> indices;
        for (auto arg : function->arguments()) {
            indices.emplace(arg, count++);
        }
        _function_argument_usage.emplace(function, luisa::vector<Usage>(count, Usage::NONE));
        arg_indices.emplace(function, std::move(indices));
    }
    auto add_usage = [&](const xir::Function *function, const xir::Value *value, Usage usage) noexcept {
        if (value == nullptr || value->derived_value_tag() != xir::DerivedValueTag::ARGUMENT) { return false; }
        auto *arg = static_cast<const xir::Argument *>(value);
        if (arg->parent_function() != function) { return false; }
        auto fit = arg_indices.find(function);
        if (fit == arg_indices.end()) { return false; }
        auto ait = fit->second.find(arg);
        if (ait == fit->second.end()) { return false; }
        auto &slot = _function_argument_usage.at(function)[ait->second];
        auto merged = merge_usage(slot, usage);
        if (merged == slot) { return false; }
        slot = merged;
        return true;
    };
    for (auto function : module->function_list()) {
        if (!function->is_definition()) { continue; }
        auto def = function->definition();
        def->traverse_instructions([&](const xir::Instruction *inst) noexcept {
            switch (inst->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::RESOURCE_READ: {
                    auto read = static_cast<const xir::ResourceReadInst *>(inst);
                    if (read->operand_count() > 0u) {
                        static_cast<void>(add_usage(function, read->operand(0u), Usage::READ));
                    }
                    break;
                }
                case xir::DerivedInstructionTag::RESOURCE_WRITE: {
                    auto write = static_cast<const xir::ResourceWriteInst *>(inst);
                    if (write->operand_count() > 0u) {
                        static_cast<void>(add_usage(function, write->operand(0u), Usage::WRITE));
                    }
                    break;
                }
                default: break;
            }
        });
    }
    for (auto function : module->function_list()) {
        if (!function->is_definition()) { continue; }
        for (auto arg : function->arguments()) {
            if (auto usage = _resource_argument_binding_usage(arg); usage != Usage::NONE) {
                static_cast<void>(add_usage(function, arg, usage));
            }
        }
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto function : module->function_list()) {
            if (!function->is_definition()) { continue; }
            auto def = function->definition();
            def->traverse_instructions([&](const xir::Instruction *inst) noexcept {
                if (inst->derived_instruction_tag() != xir::DerivedInstructionTag::CALL) { return; }
                auto call = static_cast<const xir::CallInst *>(inst);
                auto callee = call->callee();
                if (callee == nullptr || !callee->is_definition()) { return; }
                auto cit = _function_argument_usage.find(callee);
                if (cit == _function_argument_usage.end()) { return; }
                auto index = 0u;
                for (auto callee_arg : callee->arguments()) {
                    if (index >= call->argument_count() || index >= cit->second.size()) { break; }
                    auto callee_usage = cit->second[index];
                    if (callee_usage != Usage::NONE) {
                        changed |= add_usage(function, call->argument(index), callee_usage);
                    }
                    auto value = call->argument(index);
                    if (value != nullptr && value->derived_value_tag() == xir::DerivedValueTag::ARGUMENT) {
                        auto *caller_arg = static_cast<const xir::Argument *>(value);
                        if (caller_arg->parent_function() == function) {
                            auto fit = arg_indices.find(function);
                            if (fit != arg_indices.end()) {
                                auto ait = fit->second.find(caller_arg);
                                if (ait != fit->second.end()) {
                                auto caller_usage = _function_argument_usage.at(function)[ait->second];
                                changed |= add_usage(callee, callee_arg, caller_usage);
                                }
                            }
                        }
                    }
                    index++;
                }
            });
        }
    }
    for (auto &[function, usages] : _function_argument_usage) {
        for (auto &usage : usages) {
            if (usage == Usage::NONE) { usage = Usage::READ; }
        }
    }
}

Usage SpirvCodegenEntry::_function_argument_usage_of(
    const xir::Function *function,
    const xir::Argument *argument) const noexcept {
    auto fit = _function_argument_usage.find(function);
    if (fit == _function_argument_usage.end()) { return Usage::READ; }
    auto index = 0u;
    for (auto arg : function->arguments()) {
        if (arg == argument) {
            return index < fit->second.size() ? fit->second[index] : Usage::READ;
        }
        index++;
    }
    return Usage::READ;
}

void SpirvCodegenEntry::_emit_callable(const xir::CallableFunction *callable, const xir::Module *module) noexcept {
    _reset_function_codegen_state();
    _dom_tree = luisa::make_unique<xir::PostDomTree>(xir::compute_post_dom_tree(const_cast<xir::CallableFunction *>(callable)));
    _uniformity.analyze(callable);
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
        auto usage = _function_argument_usage_of(callable, arg);
        if (arg->is_resource()) {
            auto type = arg->type();
            spv::Id pointee_type = spv::NoResult;
            spv::StorageClass storage = spv::StorageClass::Max;
            switch (type->tag()) {
                case Type::Tag::BUFFER:
                    pointee_type = _convert_type(type, usage);
                    storage = spv::StorageClass::StorageBuffer;
                    _builder.addIncorporatedExtension("SPV_KHR_variable_pointers", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::VariablePointersStorageBuffer);
                    break;
                case Type::Tag::BINDLESS_ARRAY:
                    pointee_type = _convert_type(type, usage);
                    storage = spv::StorageClass::StorageBuffer;
                    _builder.addIncorporatedExtension("SPV_KHR_variable_pointers", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::VariablePointersStorageBuffer);
                    break;
                case Type::Tag::ACCEL:
                case Type::Tag::TEXTURE:
                    pointee_type = _convert_type(type, usage);
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
    _added_blocks.emplace(entry);

    int32_t i = 0;
    for (auto arg : emitted_args) {
        auto param_id = func->getParamId(i);
        _value_map.emplace(arg, param_id);
        if (arg->type()->tag() == Type::Tag::TEXTURE) {
            auto usage = _function_argument_usage_of(callable, arg);
            _is_storage_image_map.emplace(
                param_id,
                (luisa::to_underlying(usage) & luisa::to_underlying(Usage::WRITE)) != 0u);
        }
        ++i;
    }

    _builder.enterFunction(func);
    _builder.setBuildPoint(entry);
    _block_map.emplace(callable->body_block(), entry);
    _pre_register_merge_blocks(callable);
    _predeclare_allocas(callable);
    _emit_function_blocks(callable);
    _builder.leaveFunction();
}

void SpirvCodegenEntry::emit(const xir::Module *module,
                             luisa::span<const Function::Binding> bindings,
                             luisa::string_view device_lib,
                             luisa::string_view native_include) noexcept {
    _print_info.clear();
    _print_formats.clear();
    _requires_printing = false;
    auto analysis = _analyze_module_usage(module);
    _mark_atomic_buffer_types(analysis);
    _analyze_function_argument_usage(module);

    for (auto type : analysis.used_types) {
        if (type != nullptr) { _convert_type(type, Usage::READ); }
    }

    for (auto c : analysis.used_constants) {
        if (_ubo_constant_member_by_hash.contains(c->hash())) { continue; }
        _emit_constant(c);
    }

    if (_uses_int8) {
        _builder.addCapability(spv::Capability::Int8);
    }
    if (_uses_8bit_storage_buffer || _uses_8bit_uniform_storage || _uses_8bit_push_constant) {
        _builder.addExtension(spv::E_SPV_KHR_8bit_storage);
    }
    if (_uses_8bit_storage_buffer) {
        _builder.addCapability(spv::Capability::StorageBuffer8BitAccess);
    }
    if (_uses_8bit_uniform_storage) {
        _builder.addCapability(spv::Capability::UniformAndStorageBuffer8BitAccess);
    }
    if (_uses_8bit_push_constant) {
        _builder.addCapability(spv::Capability::StoragePushConstant8);
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

    std::vector<uint32_t> spirv;
    _builder.dump(spirv);
    std::ostringstream oss;
    spv::Disassemble(oss, spirv);
    _scratch << oss.str();
}

SpirvCodegenEntry::InstructionUsageAnalysis SpirvCodegenEntry::_analyze_module_usage(const xir::Module *module) noexcept {
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
}

void SpirvCodegenEntry::_mark_atomic_buffer_types(const InstructionUsageAnalysis &analysis) noexcept {
    for (auto f : analysis.used_functions_post_order) {
        if (auto def = f->definition()) {
            def->traverse_instructions([&](const xir::Instruction *inst) noexcept {
                if (inst->isa<xir::AtomicInst>()) {
                    auto atomic = static_cast<const xir::AtomicInst *>(inst);
                    auto base = atomic->base();
                    if (base != nullptr && base->type() != nullptr && base->type()->is_buffer()) {
                        _needs_atomic_buffer_types.emplace(base->type());
                    }
                }
            });
        }
    }
}

}// namespace lc::spirv
