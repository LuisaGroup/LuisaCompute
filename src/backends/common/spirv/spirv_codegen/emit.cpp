#include "entry.h"
#include "call_graph_validation.h"
#include "dialect.h"
#include "ray_query_lifetime.h"
#include "argument_usage.h"
#include "instruction_layout.h"
#include "kernel_argument_layout.h"
#include "structural_closure.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <sstream>
#include <type_traits>
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
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/runtime/rtx/ray.h>
#include <SPIRV/disassemble.h>
#include "../../indirect_dispatch_layout.h"

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
    _block_tail.clear();
    _emitted_blocks.clear();
    _registered_physical_blocks.clear();
    _synthetic_blocks.clear();
    _control_flow_plan.reset();
    _current_xir_block = nullptr;
    _deferred_phis.clear();
    _deferred_phi_incomings_by_predecessor.clear();
    _print_info.clear();
    _print_formats.clear();
    _control_flow_stack.clear();
    _properties.clear();
    _property_ids.clear();
    _entry_point_inst = nullptr;
    _deferred_entry_point_interface_ids.clear();
    _entry_point_interface_ids.clear();
    _global_invocation_id_var = spv::NoResult;
    _dispatch_metadata = {};
    _functions_requiring_dispatch_metadata.clear();
    _readonly_resource_origins.clear();
}

bool SpirvCodegenEntry::_is_indirect_dispatch_type(
    const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           type->description() == "LC_IndirectDispatchBuffer"sv;
}

bool SpirvCodegenEntry::_is_kernel_resource_argument(
    const xir::Argument *argument) noexcept {
    return argument != nullptr &&
           (argument->is_resource() ||
            _is_indirect_dispatch_type(argument->type()));
}

namespace {

[[nodiscard]] std::vector<uint8_t>
spirv_codegen_canonicalized_structure_index_operands(
    const xir::Instruction *instruction) noexcept {
    std::vector<uint8_t> canonicalized(
        instruction->operand_count(), uint8_t{0u});
    auto mark_structure_indices = [&](const Type *aggregate_type,
                                      size_t first_index,
                                      size_t index_count) noexcept {
        if (aggregate_type == nullptr ||
            first_index > instruction->operand_count() ||
            index_count > instruction->operand_count() - first_index) {
            return;
        }
        luisa::vector<const xir::Value *> indices;
        indices.reserve(index_count);
        for (auto i = 0u; i < index_count; ++i) {
            indices.emplace_back(instruction->operand(first_index + i));
        }
        auto plan = plan_spirv_aggregate_indices(
            aggregate_type, luisa::span{indices});
        if (!plan) { return; }
        LUISA_ASSERT(plan.steps.size() == index_count,
                     "SPIR-V aggregate index analysis produced {} steps "
                     "for {} operands.",
                     plan.steps.size(), index_count);
        for (auto i = 0u; i < plan.steps.size(); ++i) {
            if (plan.steps[i].kind ==
                SpirvAggregateIndexKind::STRUCTURE_MEMBER) {
                canonicalized[first_index + i] = 1u;
            }
        }
    };

    switch (instruction->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::GEP: {
            auto gep = static_cast<const xir::GEPInst *>(instruction);
            auto base = gep->base();
            mark_structure_indices(
                base == nullptr ? nullptr : base->type(),
                xir::GEPInst::operand_index_index_offset,
                gep->index_count());
            break;
        }
        case xir::DerivedInstructionTag::ATOMIC: {
            auto atomic = static_cast<const xir::AtomicInst *>(instruction);
            auto base = atomic->base();
            mark_structure_indices(
                base == nullptr ? nullptr : base->type(),
                1u, atomic->index_count());
            break;
        }
        case xir::DerivedInstructionTag::ARITHMETIC: {
            auto arithmetic =
                static_cast<const xir::ArithmeticInst *>(instruction);
            auto first_index = [&]() noexcept -> size_t {
                switch (arithmetic->op()) {
                    case xir::ArithmeticOp::INSERT: return 2u;
                    case xir::ArithmeticOp::EXTRACT: return 1u;
                    default: return instruction->operand_count();
                }
            }();
            if (first_index < instruction->operand_count()) {
                auto aggregate = instruction->operand(0u);
                mark_structure_indices(
                    aggregate == nullptr ? nullptr : aggregate->type(),
                    first_index,
                    instruction->operand_count() - first_index);
            }
            break;
        }
        default: break;
    }
    return canonicalized;
}

}// namespace

void SpirvCodegenEntry::_analyze_instruction_usage(
    const xir::Function *function,
    InstructionUsageAnalysis &analysis) noexcept {
    LUISA_ASSERT(function != nullptr,
                 "SPIR-V instruction-usage analysis received a null function.");
    for (auto arg : function->arguments()) {
        LUISA_ASSERT(arg != nullptr, "Function argument is null.");
        analysis.used_types.emplace(arg->type());
    }
    analysis.used_types.emplace(function->type());
    if (auto def = function->definition()) {
        traverse_spirv_codegen_structural_instructions(
            def, [&](const xir::Instruction *inst) noexcept {
                switch (inst->derived_instruction_tag()) {
                    case xir::DerivedInstructionTag::PRINT: {
                        _requires_printing = true;
                        auto print = static_cast<const xir::PrintInst *>(inst);
                        auto [iter, success] = _print_info.emplace(print, PrintInfo{nullptr, _print_formats.size()});
                        LUISA_ASSERT(success, "Print info already exists.");
                        luisa::vector<const Type *> arg_types;
                        arg_types.reserve(print->operand_count() + 2u);
                        arg_types.emplace_back(Type::of<uint32_t>());// arg size
                        arg_types.emplace_back(Type::of<uint32_t>());// fmt id
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
                    case xir::DerivedInstructionTag::RESOURCE_QUERY: {
                        auto resource = static_cast<
                            const xir::ResourceQueryInst *>(inst);
                        analysis.bindless_resources.merge(
                            spirv_bindless_resource_usage(
                                resource->op(), resource->bindless_access()));
                        break;
                    }
                    case xir::DerivedInstructionTag::RESOURCE_READ: {
                        auto resource = static_cast<
                            const xir::ResourceReadInst *>(inst);
                        analysis.bindless_resources.merge(
                            spirv_bindless_resource_usage(
                                resource->op(), resource->bindless_access()));
                        break;
                    }
                    case xir::DerivedInstructionTag::RESOURCE_WRITE: {
                        auto resource = static_cast<
                            const xir::ResourceWriteInst *>(inst);
                        analysis.bindless_resources.merge(
                            spirv_bindless_resource_usage(
                                resource->op(), resource->bindless_access()));
                        break;
                    }
                    default: break;
                }
                analysis.used_types.emplace(inst->type());
                auto canonicalized_structure_indices =
                    spirv_codegen_canonicalized_structure_index_operands(inst);
                auto operand_index = 0u;
                for (auto op_use : inst->operand_uses()) {
                    if (canonicalized_structure_indices[operand_index++]) {
                        // Structure-member indices are re-materialized as u32
                        // constants at the SPIR-V boundary. Do not emit their
                        // source-width XIR constants, or require capabilities for
                        // integer types that are absent from the generated module.
                        continue;
                    }
                    if (auto value = op_use->value()) {
                        analysis.used_types.emplace(value->type());
                        switch (value->derived_value_tag()) {
                            case xir::DerivedValueTag::CONSTANT:
                                analysis.used_constants.emplace(static_cast<const xir::Constant *>(value));
                                break;
                            default: break;
                        }
                    }
                }
            });
    }
}

namespace {

template<typename UInt>
[[nodiscard]] uint64_t spirv_codegen_load_constant_bits(const void *data) noexcept {
    UInt bits{};
    std::memcpy(&bits, data, sizeof(bits));
    return static_cast<uint64_t>(bits);
}

void spirv_codegen_add_narrow_constant_capabilities(
    spv::Builder &builder, const Type *type) noexcept {
    if (type == nullptr) { return; }
    switch (type->tag()) {
        case Type::Tag::INT8:
        case Type::Tag::UINT8:
            builder.addCapability(spv::Capability::Int8);
            break;
        case Type::Tag::INT16:
        case Type::Tag::UINT16:
            builder.addCapability(spv::Capability::Int16);
            break;
        case Type::Tag::FLOAT16:
            builder.addCapability(spv::Capability::Float16);
            break;
        case Type::Tag::VECTOR:
        case Type::Tag::MATRIX:
        case Type::Tag::ARRAY:
            spirv_codegen_add_narrow_constant_capabilities(
                builder, type->element());
            break;
        case Type::Tag::STRUCTURE:
            for (auto *member : type->members()) {
                spirv_codegen_add_narrow_constant_capabilities(
                    builder, member);
            }
            break;
        default: break;
    }
}

[[nodiscard]] spv::Id spirv_codegen_emit_scalar_constant(
    spv::Builder &builder, spv::Id spv_type,
    const Type *type, const void *data) noexcept {
    // Storage-only capabilities permit narrow loads, stores, and width-only
    // conversions, but SPIR-V explicitly forbids forming narrow constants
    // without the corresponding arithmetic capability. Do this at the
    // constant boundary rather than making every narrow storage type require
    // shaderInt8/shaderInt16/shaderFloat16.
    spirv_codegen_add_narrow_constant_capabilities(builder, type);
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
        case Type::Tag::FLOAT8_E4M3:
        case Type::Tag::FLOAT8_E5M2:
            return builder.makeFpConstantFromBits(
                spv_type, spirv_codegen_load_constant_bits<uint8_t>(data));
        case Type::Tag::FLOAT16:
            return builder.makeFpConstantFromBits(
                spv_type, spirv_codegen_load_constant_bits<uint16_t>(data));
        case Type::Tag::FLOAT32:
            return builder.makeFpConstantFromBits(
                spv_type, spirv_codegen_load_constant_bits<uint32_t>(data));
        case Type::Tag::FLOAT64:
            return builder.makeFpConstantFromBits(
                spv_type, spirv_codegen_load_constant_bits<uint64_t>(data));
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
            return spirv_codegen_emit_scalar_constant(_builder, spv_type, type, data);
        case Type::Tag::VECTOR: {
            auto elem_type = type->element();
            auto dim = type->dimension();
            auto elem_stride = elem_type->size();
            std::vector<spv::Id> comps;
            comps.reserve(dim);
            for (uint32_t i = 0u; i < dim; ++i) {
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
            for (uint32_t i = 0u; i < dim; ++i) {
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
            for (uint32_t i = 0u; i < dim; ++i) {
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
    if (auto ubo_it = _ubo_constant_member_indices.find(c);
        ubo_it != _ubo_constant_member_indices.end()) {
        auto ptr = _create_access_chain(
            spv::StorageClass::Uniform, _constant_ubo_var,
            std::vector<spv::Id>{_builder.makeUintConstant(ubo_it->second)});
        auto loaded = _builder.createLoad(ptr, spv::NoPrecision);
        auto logical_type = _convert_type(c->type(), Usage::READ);
        return _builder.getTypeId(loaded) == logical_type ?
                   loaded :
                   _builder.createUnaryOp(
                       spv::Op::OpCopyLogical, logical_type, loaded);
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
    if (storage == spv::StorageClass::Workgroup) {
        _register_entry_point_interface(var);
    }
    _value_map.emplace(alloca, var);
    return var;
}

void SpirvCodegenEntry::_register_entry_point_interface(spv::Id id) noexcept {
    if (id == spv::NoResult) { return; }
    if (_entry_point_inst == nullptr) {
        if (std::find(_deferred_entry_point_interface_ids.begin(),
                      _deferred_entry_point_interface_ids.end(), id) ==
            _deferred_entry_point_interface_ids.end()) {
            _deferred_entry_point_interface_ids.emplace_back(id);
        }
        return;
    }
    if (_entry_point_interface_ids.emplace(id).second) {
        // OpEntryPoint has a five-word fixed prefix for GLCompute, the entry
        // function ID, and the nul-terminated literal "main". The final
        // interface set depends on binding and target plans that intentionally
        // live after the pure-XIR dialect check, so enforce its physical limit
        // at the exact registration boundary.
        auto layout = plan_spirv_variadic_instruction(
            "OpEntryPoint", 5u,
            _entry_point_interface_ids.size());
        if (!layout) {
            LUISA_ERROR_WITH_LOCATION("{}", layout.diagnostic);
        }
        _entry_point_inst->addIdOperand(id);
    }
}

void SpirvCodegenEntry::_set_dispatch_metadata(spv::Id packed) noexcept {
    LUISA_ASSERT(
        packed != spv::NoResult &&
            _dispatch_metadata.packed == spv::NoResult,
        "SPIR-V dispatch metadata was initialized more than once in one function.");
    auto uint_type = _builder.makeUintType(32u);
    auto uint3_type = _builder.makeVectorType(uint_type, 3u);
    _dispatch_metadata.packed = packed;
    auto x = _builder.createCompositeExtract(packed, uint_type, 0u);
    auto y = _builder.createCompositeExtract(packed, uint_type, 1u);
    auto z = _builder.createCompositeExtract(packed, uint_type, 2u);
    _dispatch_metadata.dispatch_size =
        _builder.createCompositeConstruct(uint3_type, {x, y, z});
    _dispatch_metadata.kernel_id =
        _builder.createCompositeExtract(packed, uint_type, 3u);
}

spv::Block *SpirvCodegenEntry::_emit_dispatch_metadata_prologue(
    spv::Function *function) noexcept {
    LUISA_ASSERT(_allow_indirect_dispatch &&
                     _indirect_dispatch_buffer_id != spv::NoResult &&
                     function != nullptr &&
                     _builder.getBuildPoint() != nullptr &&
                     _dispatch_metadata.packed == spv::NoResult,
                 "SPIR-V indirect-dispatch metadata binding was not emitted.");
    auto uint_type = _builder.makeUintType(32u);
    auto uint4_type = _builder.makeVectorType(uint_type, 4u);
    auto bool_type = _builder.makeBoolType();
    auto push_const = _property_ids[0];
    auto control_ptr = _create_access_chain(
        spv::StorageClass::PushConstant, push_const,
        {_builder.makeUintConstant(1u)});
    auto control = _builder.createLoad(control_ptr, spv::NoPrecision);
    auto mode = _builder.createCompositeExtract(control, uint_type, 0u);
    auto is_indirect = _builder.createBinOp(
        spv::Op::OpINotEqual, bool_type, mode,
        _builder.makeUintConstant(
            static_cast<uint32_t>(IndirectDispatchMode::DIRECT)));

    auto *header = _builder.getBuildPoint();
    auto *direct_block = _create_physical_block(function);
    auto *indirect_block = _create_physical_block(function);
    auto *merge_block = _create_physical_block(function);
    auto selection_merge =
        std::make_unique<spv::Instruction>(spv::Op::OpSelectionMerge);
    selection_merge->reserveOperands(2u);
    selection_merge->addIdOperand(merge_block->getId());
    selection_merge->addImmediateOperand(
        spv::SelectionControlMask::MaskNone);
    header->addInstruction(std::move(selection_merge));
    _builder.createConditionalBranch(
        is_indirect, indirect_block, direct_block);

    // OpSelect is eager. Keep the source loads exclusively in the indirect
    // arm so an ordinary dispatch has no dynamic dependency on the statically
    // required dummy descriptor.
    _builder.setBuildPoint(direct_block);
    auto direct_ptr = _create_access_chain(
        spv::StorageClass::PushConstant, push_const,
        {_builder.makeUintConstant(0u)});
    auto direct = _builder.createLoad(direct_ptr, spv::NoPrecision);
    _builder.createBranch(false, merge_block);

    _builder.setBuildPoint(indirect_block);
    auto record_index =
        _builder.createCompositeExtract(control, uint_type, 1u);
    auto record_word = _builder.createBinOp(
        spv::Op::OpIMul, uint_type, record_index,
        _builder.makeUintConstant(
            IndirectDispatchLayout::record_word_count));
    record_word = _builder.createBinOp(
        spv::Op::OpIAdd, uint_type, record_word,
        _builder.makeUintConstant(
            IndirectDispatchLayout::header_word_count));
    std::vector<spv::Id> components;
    components.reserve(4u);
    for (auto component = 0u; component < 4u; ++component) {
        auto word = component == 0u ? record_word :
                                      _builder.createBinOp(
                                          spv::Op::OpIAdd, uint_type,
                                          record_word,
                                          _builder.makeUintConstant(component));
        auto ptr = _create_access_chain(
            spv::StorageClass::StorageBuffer,
            _indirect_dispatch_buffer_id,
            {_builder.makeUintConstant(0u), word});
        components.emplace_back(
            _builder.createLoad(ptr, spv::NoPrecision));
    }
    auto indirect =
        _builder.createCompositeConstruct(uint4_type, components);
    _builder.createBranch(false, merge_block);

    _builder.setBuildPoint(merge_block);
    auto phi = std::make_unique<spv::Instruction>(
        _builder.getUniqueId(), uint4_type, spv::Op::OpPhi);
    auto packed = phi->getResultId();
    phi->reserveOperands(4u);
    phi->addIdOperand(direct);
    phi->addIdOperand(direct_block->getId());
    phi->addIdOperand(indirect);
    phi->addIdOperand(indirect_block->getId());
    merge_block->addInstruction(std::move(phi));
    _set_dispatch_metadata(packed);
    return merge_block;
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
                spirv_codegen_add_narrow_constant_capabilities(
                    _builder, value->type());
                id = _builder.makeNullConstant(spv_type);
            }
            break;
        }
        case xir::DerivedValueTag::SPECIAL_REGISTER: {
            auto reg = static_cast<const xir::SpecialRegister *>(value);
            auto tag = reg->derived_special_register_tag();
            if (tag == xir::DerivedSpecialRegisterTag::DISPATCH_SIZE ||
                tag == xir::DerivedSpecialRegisterTag::KERNEL_ID) {
                // These IDs are materialized in the function prologue (or
                // from a callable's hidden uint4 parameter), so they dominate
                // every XIR use without creating memory access or CFG here.
                LUISA_ASSERT(
                    _dispatch_metadata.packed != spv::NoResult,
                    "SPIR-V dispatch metadata was requested outside a kernel "
                    "prologue or a callable with a hidden metadata parameter.");
                id = tag == xir::DerivedSpecialRegisterTag::KERNEL_ID ?
                         _dispatch_metadata.kernel_id :
                         _dispatch_metadata.dispatch_size;
                break;
            }
            if (tag == xir::DerivedSpecialRegisterTag::BLOCK_SIZE) {
                LUISA_ASSERT(_kernel_block_size.x != 0u && _kernel_block_size.y != 0u && _kernel_block_size.z != 0u,
                             "SPIR-V BLOCK_SIZE requested before the kernel block size was planned.");
                auto uint_type = _builder.makeUintType(32);
                auto uint3_type = _builder.makeVectorType(uint_type, 3);
                id = _builder.makeCompositeConstant(
                    uint3_type,
                    {_builder.makeUintConstant(_kernel_block_size.x),
                     _builder.makeUintConstant(_kernel_block_size.y),
                     _builder.makeUintConstant(_kernel_block_size.z)});
                break;
            }
            spv::BuiltIn builtin;
            switch (tag) {
                case xir::DerivedSpecialRegisterTag::THREAD_ID: builtin = spv::BuiltIn::LocalInvocationId; break;
                case xir::DerivedSpecialRegisterTag::BLOCK_ID: builtin = spv::BuiltIn::WorkgroupId; break;
                case xir::DerivedSpecialRegisterTag::DISPATCH_ID: builtin = spv::BuiltIn::GlobalInvocationId; break;
                case xir::DerivedSpecialRegisterTag::WARP_SIZE: builtin = spv::BuiltIn::SubgroupSize; break;
                case xir::DerivedSpecialRegisterTag::WARP_LANE_ID: builtin = spv::BuiltIn::SubgroupLocalInvocationId; break;
                default:
                    LUISA_NOT_IMPLEMENTED("SPIR-V special register {}.", xir::to_string(reg->derived_special_register_tag()));
            }
            if (builtin == spv::BuiltIn::SubgroupSize ||
                builtin == spv::BuiltIn::SubgroupLocalInvocationId) {
                _require_target_feature(
                    target_feature::subgroup_basic,
                    _target_features.subgroup_basic);
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
                _register_entry_point_interface(var);
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
            if (_is_kernel_resource_argument(arg)) {
                id = _resolve_resource_argument(arg);
                break;
            }
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V value scheduling error: non-resource argument {} was not mapped by its function prologue.",
                static_cast<const void *>(arg));
        }
        case xir::DerivedValueTag::FUNCTION:
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V value scheduling error: function {} was not predeclared.",
                static_cast<const void *>(value));
        case xir::DerivedValueTag::BASIC_BLOCK:
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V value scheduling error: BasicBlock used as an ordinary SSA value.");
        case xir::DerivedValueTag::INSTRUCTION: {
            auto *inst = static_cast<const xir::Instruction *>(value);
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V value scheduling error: instruction {} in XIR block {} was used before its definition was emitted. "
                "The control-flow plan never emits values recursively and never substitutes OpUndef.",
                static_cast<const void *>(inst), static_cast<const void *>(inst->parent_block()));
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
        LUISA_ASSERT(!indices.empty(),
                     "A non-uniform SPIR-V access chain must have an index.");
        new_access_chain.coherentFlags.nonUniform = 1;
        // Only the varying descriptor-array index is non-uniform. Prefix
        // indices select enclosing structs/arrays and are commonly interned
        // constants; decorating them would contaminate every use of the same
        // SPIR-V constant throughout the module.
        _builder.addDecoration(indices.back(), spv::Decoration::NonUniformEXT);
    }
    _builder.setAccessChain(new_access_chain);
    auto id = _builder.createAccessChain(storage, base, indices);
    _builder.setAccessChain(old_access_chain);
    if (nonuniform) {
        _builder.addDecoration(id, spv::Decoration::NonUniformEXT);
    }
    return id;
}

void SpirvCodegenEntry::_mark_16bit_storage_usage(const Type *type, spv::StorageClass storage) noexcept {
    if (type == nullptr) { return; }
    switch (type->tag()) {
        case Type::Tag::INT16:
        case Type::Tag::UINT16:
        case Type::Tag::FLOAT16:
            switch (storage) {
                case spv::StorageClass::StorageBuffer: _uses_16bit_storage_buffer = true; break;
                case spv::StorageClass::Uniform: _uses_16bit_uniform_storage = true; break;
                case spv::StorageClass::PushConstant: _uses_16bit_push_constant = true; break;
                default: break;
            }
            break;
        case Type::Tag::STRUCTURE:
            for (auto *member : type->members()) { _mark_16bit_storage_usage(member, storage); }
            break;
        case Type::Tag::ARRAY:
        case Type::Tag::VECTOR:
        case Type::Tag::MATRIX:
            _mark_16bit_storage_usage(type->element(), storage);
            break;
        case Type::Tag::BUFFER:
            _mark_16bit_storage_usage(type->element(), storage);
            break;
        default: break;
    }
}

void SpirvCodegenEntry::_predeclare_allocas(const xir::FunctionDefinition *def) noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr && def != nullptr,
                 "SPIR-V alloca predeclaration requires a function plan.");
    for (auto &block_plan : _control_flow_plan->blocks()) {
        auto *bb = block_plan.block;
        for (auto *inst : bb->instructions()) {
            if (inst->isa<xir::AllocaInst>()) {
                _emit_alloca(static_cast<const xir::AllocaInst *>(inst));
            }
        }
    }
}

void SpirvCodegenEntry::_predeclare_phis() noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr,
                 "SPIR-V Phi predeclaration requires a control-flow plan.");
    for (auto &phi_plan : _control_flow_plan->phi_plans()) {
        auto *phi = phi_plan.instruction;
        DeferredPhi deferred;

        // Allocate one auxiliary OpPhi per synthetic forwarding block used by
        // this logical Phi. The first occurrence order is deterministic because
        // both XIR incomings and their forwarding paths are frozen by the plan.
        for (auto &incoming : phi_plan.incomings) {
            for (auto synthetic_index : incoming.forwarding_synthetic_indices) {
                if (deferred.forwarding_node_indices.contains(synthetic_index)) { continue; }
                auto node_index = deferred.nodes.size();
                deferred.forwarding_node_indices.emplace(synthetic_index, node_index);
                deferred.nodes.emplace_back(DeferredPhiNode{
                    .block = _synthetic_blocks.at(synthetic_index)});
            }
        }
        deferred.result_node_index = deferred.nodes.size();
        deferred.nodes.emplace_back(DeferredPhiNode{
            .block = _physical_block(phi_plan.result_target)});

        auto add_forwarded_incoming = [&](size_t parent_node_index,
                                          size_t child_node_index,
                                          size_t order) noexcept {
            auto &parent = deferred.nodes[parent_node_index];
            for (auto &incoming : parent.incomings) {
                if (incoming.child_node_index == child_node_index) { return; }
            }
            parent.incomings.emplace_back(DeferredPhiNodeIncoming{
                .order = order,
                .child_node_index = child_node_index});
        };
        for (size_t incoming_index = 0u;
             incoming_index < phi_plan.incomings.size(); ++incoming_index) {
            auto &incoming_plan = phi_plan.incomings[incoming_index];
            auto first_node_index = deferred.result_node_index;
            if (!incoming_plan.forwarding_synthetic_indices.empty()) {
                first_node_index = deferred.forwarding_node_indices.at(
                    incoming_plan.forwarding_synthetic_indices.front());
            }
            deferred.nodes[first_node_index].incomings.emplace_back(
                DeferredPhiNodeIncoming{
                    .order = incoming_index,
                    .logical_value = incoming_plan.value,
                    .logical_predecessor = incoming_plan.predecessor});

            for (size_t path_index = 1u;
                 path_index < incoming_plan.forwarding_synthetic_indices.size(); ++path_index) {
                auto child_node_index = deferred.forwarding_node_indices.at(
                    incoming_plan.forwarding_synthetic_indices[path_index - 1u]);
                auto parent_node_index = deferred.forwarding_node_indices.at(
                    incoming_plan.forwarding_synthetic_indices[path_index]);
                add_forwarded_incoming(parent_node_index, child_node_index, incoming_index);
            }
            if (!incoming_plan.forwarding_synthetic_indices.empty()) {
                auto child_node_index = deferred.forwarding_node_indices.at(
                    incoming_plan.forwarding_synthetic_indices.back());
                add_forwarded_incoming(
                    deferred.result_node_index, child_node_index, incoming_index);
            }
        }

        auto phi_type = _convert_type(phi->type(), Usage::READ);
        for (auto &node : deferred.nodes) {
            auto layout = plan_spirv_phi_instruction(node.incomings.size());
            if (!layout) {
                LUISA_ERROR_WITH_LOCATION("{}", layout.diagnostic);
            }
            std::stable_sort(
                node.incomings.begin(), node.incomings.end(),
                [](auto &lhs, auto &rhs) noexcept { return lhs.order < rhs.order; });
            auto instruction = std::make_unique<spv::Instruction>(
                _builder.getUniqueId(), phi_type, spv::Op::OpPhi);
            instruction->reserveOperands(layout.operand_word_count);
            node.instruction = instruction.get();
            node.block->addInstruction(std::move(instruction));
        }
        auto result_id = deferred.nodes.at(deferred.result_node_index)
                             .instruction->getResultId();
        auto [deferred_iter, inserted] = _deferred_phis.emplace(phi, std::move(deferred));
        LUISA_ASSERT(inserted, "SPIR-V Phi was predeclared more than once.");
        LUISA_ASSERT(_value_map.emplace(phi, result_id).second,
                     "SPIR-V Phi result ID was mapped more than once.");

        auto &stored = deferred_iter->second;
        for (size_t node_index = 0u; node_index < stored.nodes.size(); ++node_index) {
            auto &node = stored.nodes[node_index];
            for (size_t incoming_index = 0u;
                 incoming_index < node.incomings.size(); ++incoming_index) {
                auto &incoming = node.incomings[incoming_index];
                if (incoming.logical_predecessor == nullptr) { continue; }
                _deferred_phi_incomings_by_predecessor[incoming.logical_predecessor]
                    .emplace_back(DeferredPhiIncomingRef{
                        .phi = phi,
                        .node_index = node_index,
                        .incoming_index = incoming_index});
            }
        }
    }
}

void SpirvCodegenEntry::_resolve_phi_incomings_from_predecessor(
    const xir::BasicBlock *predecessor) noexcept {
    auto iter = _deferred_phi_incomings_by_predecessor.find(predecessor);
    if (iter == _deferred_phi_incomings_by_predecessor.end()) { return; }
    LUISA_ASSERT(_current_xir_block == predecessor && _builder.getBuildPoint() != nullptr,
                 "SPIR-V Phi incoming resolution requires the active predecessor tail.");
    for (auto &ref : iter->second) {
        auto &deferred = _deferred_phis.at(ref.phi);
        auto &incoming = deferred.nodes.at(ref.node_index).incomings.at(ref.incoming_index);
        LUISA_ASSERT(!incoming.resolved && incoming.logical_value != nullptr &&
                         incoming.logical_predecessor == predecessor,
                     "SPIR-V Phi incoming was resolved more than once or has no logical value.");
        incoming.resolved_value = _emit_value(incoming.logical_value);
        auto *physical_predecessor = _builder.getBuildPoint();
        LUISA_ASSERT(physical_predecessor != nullptr && !physical_predecessor->isTerminated(),
                     "SPIR-V Phi incoming must be materialized before its physical terminator.");
        incoming.resolved_predecessor = physical_predecessor->getId();
        incoming.resolved = true;
    }
}

void SpirvCodegenEntry::_finalize_phis() noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr,
                 "SPIR-V Phi finalization requires a control-flow plan.");
    for (auto &phi_plan : _control_flow_plan->phi_plans()) {
        auto &deferred = _deferred_phis.at(phi_plan.instruction);
        for (auto &node : deferred.nodes) {
            auto layout = plan_spirv_phi_instruction(node.incomings.size());
            LUISA_ASSERT(layout.succeeded(), "{}", layout.diagnostic);
            node.instruction->clearOperands();
            luisa::unordered_set<spv::Id> planned_predecessors;
            for (auto &incoming : node.incomings) {
                auto value = incoming.resolved_value;
                auto predecessor = incoming.resolved_predecessor;
                if (incoming.child_node_index != invalid_phi_node_index) {
                    auto &child = deferred.nodes.at(incoming.child_node_index);
                    value = child.instruction->getResultId();
                    predecessor = child.block->getId();
                } else {
                    LUISA_ASSERT(incoming.resolved,
                                 "SPIR-V Phi incoming edge was never emitted.");
                }
                LUISA_ASSERT(value != spv::NoResult && predecessor != spv::NoResult &&
                                 planned_predecessors.emplace(predecessor).second,
                             "SPIR-V Phi has an invalid or duplicate physical predecessor.");
                node.instruction->addIdOperand(value);
                node.instruction->addIdOperand(predecessor);
            }
            luisa::unordered_set<spv::Id> actual_predecessors;
            for (auto *predecessor : node.block->getPredecessors()) {
                actual_predecessors.emplace(predecessor->getId());
            }
            LUISA_ASSERT(actual_predecessors.size() == planned_predecessors.size(),
                         "SPIR-V Phi physical predecessor count mismatch.");
            for (auto predecessor : actual_predecessors) {
                LUISA_ASSERT(planned_predecessors.contains(predecessor),
                             "SPIR-V Phi is missing a physical predecessor.");
            }
        }
    }
}

void SpirvCodegenEntry::_emit_phi_inst(const xir::PhiInst *instruction) noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr &&
                     _deferred_phis.contains(instruction) &&
                     _value_map.contains(instruction),
                 "SPIR-V Phi instruction was not predeclared by the control-flow plan.");
}

spv::Block *SpirvCodegenEntry::_xir_block_entry(const xir::BasicBlock *bb) const noexcept {
    LUISA_ASSERT(bb != nullptr, "SPIR-V physical block lookup received a null XIR block.");
    auto iter = _block_map.find(bb);
    LUISA_ASSERT(iter != _block_map.end() && iter->second != nullptr,
                 "SPIR-V physical block lookup occurred before the function plan was bound.");
    return iter->second;
}

spv::Block *SpirvCodegenEntry::_physical_block(ControlFlowPlan::Target target) const noexcept {
    switch (target.kind) {
        case ControlFlowPlan::Target::Kind::XIR_BLOCK:
            return _xir_block_entry(target.xir_block);
        case ControlFlowPlan::Target::Kind::SYNTHETIC_BLOCK:
            LUISA_ASSERT(target.synthetic_index < _synthetic_blocks.size(),
                         "SPIR-V synthetic block index is out of range.");
            return _synthetic_blocks[target.synthetic_index];
    }
    LUISA_ERROR_WITH_LOCATION("Invalid SPIR-V control-flow target kind.");
}

void SpirvCodegenEntry::_register_physical_block(spv::Block *block, bool already_appended) noexcept {
    LUISA_ASSERT(block != nullptr, "Cannot register a null SPIR-V physical block.");
    auto [_, inserted] = _registered_physical_blocks.emplace(block);
    LUISA_ASSERT(inserted, "SPIR-V physical block {} was registered more than once.", block->getId());
    if (!already_appended) {
        block->getParent().addBlock(block);
    }
}

spv::Block *SpirvCodegenEntry::_create_physical_block(spv::Function *function) noexcept {
    if (function == nullptr) {
        auto *build_point = _builder.getBuildPoint();
        LUISA_ASSERT(build_point != nullptr, "SPIR-V physical block creation requires an active function.");
        function = &build_point->getParent();
    }
    auto *block = new spv::Block(_builder.getUniqueId(), *function);
    _register_physical_block(block, false);
    return block;
}

void SpirvCodegenEntry::_set_current_tail(spv::Block *block) noexcept {
    LUISA_ASSERT(block != nullptr, "Cannot set a null SPIR-V physical tail.");
    LUISA_ASSERT(_registered_physical_blocks.contains(block),
                 "SPIR-V physical tail {} is not centrally registered.", block->getId());
    _builder.setBuildPoint(block);
    if (_current_xir_block != nullptr) {
        _block_tail[_current_xir_block] = block;
    }
}

void SpirvCodegenEntry::_prepare_control_flow_plan(const xir::FunctionDefinition *def) noexcept {
    LUISA_ASSERT(def != nullptr && def->body_block() != nullptr,
                 "SPIR-V function planning requires a body block.");
    LUISA_ASSERT(_block_map.size() == 1u && _block_map.contains(def->body_block()),
                 "SPIR-V function planning requires exactly one pre-bound function entry.");
    _control_flow_plan = luisa::make_unique<ControlFlowPlan>(ControlFlowPlan::create(def));
    auto *function = &_block_map.at(def->body_block())->getParent();
    for (auto &block_plan : _control_flow_plan->blocks()) {
        auto *block = block_plan.block;
        if (block == def->body_block()) {
            _block_tail.emplace(block, _block_map.at(block));
            continue;
        }
        auto *physical = _create_physical_block(function);
        auto [_, inserted] = _block_map.emplace(block, physical);
        LUISA_ASSERT(inserted, "SPIR-V logical block entry was bound more than once.");
        _block_tail.emplace(block, physical);
    }
    _synthetic_blocks.reserve(_control_flow_plan->synthetic_blocks().size());
    for (auto &&synthetic : _control_flow_plan->synthetic_blocks()) {
        static_cast<void>(synthetic);
        _synthetic_blocks.emplace_back(_create_physical_block(function));
    }
    _predeclare_phis();
    auto *saved_build_point = _builder.getBuildPoint();
    for (auto &&synthetic : _control_flow_plan->synthetic_blocks()) {
        if (synthetic.kind !=
                ControlFlowPlan::SyntheticBlockKind::EDGE_TRAMPOLINE &&
            synthetic.kind !=
                ControlFlowPlan::SyntheticBlockKind::SWITCH_CONTINUE) {
            continue;
        }
        auto *block = _synthetic_blocks.at(synthetic.ordinal);
        _builder.setBuildPoint(block);
        _builder.createBranch(false, _physical_block(synthetic.continuation));
    }
    if (saved_build_point != nullptr) { _builder.setBuildPoint(saved_build_point); }
}

bool SpirvCodegenEntry::_emit_block(const xir::BasicBlock *bb) noexcept {
    if (bb == nullptr || !_emitted_blocks.emplace(bb).second) { return false; }
    auto *entry = _xir_block_entry(bb);
    auto tail_iter = _block_tail.find(bb);
    LUISA_ASSERT(tail_iter != _block_tail.end() && tail_iter->second != nullptr,
                 "SPIR-V XIR block has no physical tail.");
    auto *tail = tail_iter->second;
    LUISA_ASSERT(!tail->isTerminated(),
                 "SPIR-V XIR block entry/tail {} was terminated before its instructions were emitted.",
                 entry->getId());
    auto *saved_xir_block = _current_xir_block;
    _current_xir_block = bb;
    _builder.setBuildPoint(tail);
    if (_control_flow_plan->block(bb).physically_pruned) {
        auto first = bb->instructions().begin();
        LUISA_ASSERT(
            first != bb->instructions().end() &&
                *first == bb->terminator() &&
                (bb->terminator()->isa<xir::BranchInst>() ||
                 bb->terminator()->isa<xir::BreakInst>() ||
                 bb->terminator()->isa<xir::ContinueInst>()),
            "SPIR-V physically pruned block is not an empty loop-boundary "
            "proxy.");
        _builder.createNoResultOp(spv::Op::OpUnreachable);
        _current_xir_block = saved_xir_block;
        return true;
    }
    for (auto *inst : bb->instructions()) {
        LUISA_ASSERT(_builder.getBuildPoint() != nullptr && !_builder.getBuildPoint()->isTerminated(),
                     "SPIR-V attempted to emit instruction {} after a physical block terminator.",
                     xir::to_string(inst->derived_instruction_tag()));
        if (inst->is_terminator()) {
            _resolve_phi_incomings_from_predecessor(bb);
        }
        _emit_instruction(inst);
        LUISA_ASSERT(_builder.getBuildPoint() != nullptr,
                     "SPIR-V instruction emission cleared the physical build point.");
        _block_tail[bb] = _builder.getBuildPoint();
    }
    _current_xir_block = saved_xir_block;
    return true;
}

bool SpirvCodegenEntry::_is_ray_query_type(const Type *type) noexcept {
    return type == Type::custom("LC_RayQueryAll") ||
           type == Type::custom("LC_RayQueryAny");
}

const Type *SpirvCodegenEntry::_ray_query_initial_ray_type() noexcept {
    return Type::of<Ray>();
}

const SpirvCodegenEntry::RayQueryState &
SpirvCodegenEntry::_ray_query_state(spv::Id query_object) const noexcept {
    if (auto iter = _ray_query_states.find(query_object);
        iter != _ray_query_states.end()) {
        return iter->second;
    }
    LUISA_ERROR_WITH_LOCATION(
        "SPIR-V ray-query object {} has no initialization state. "
        "Query copies, selects, Phi nodes, and uses before initialization are unsupported.",
        query_object);
}

void SpirvCodegenEntry::_validate_ray_query_lifetimes(
    const xir::FunctionDefinition *def) const noexcept {
    LUISA_ASSERT(def != nullptr, "SPIR-V ray-query lifetime validation requires a function.");
    LUISA_ASSERT(_control_flow_plan != nullptr &&
                     _control_flow_plan->function() == def,
                 "SPIR-V ray-query lifetime validation requires the frozen "
                 "control-flow plan for the same function.");
    auto result = validate_spirv_ray_query_lifetimes(def);
    LUISA_ASSERT(
        result.succeeded(),
        "SPIR-V dialect validation failed to reject an invalid ray-query lifetime in function '{}': {} ({} diagnostic(s) total).",
        def->name().value_or("<unnamed>"),
        result.diagnostics.empty() ?
            luisa::string_view{"<missing diagnostic>"} :
            luisa::string_view{result.diagnostics.front().message},
        result.diagnostics.size());
}

void SpirvCodegenEntry::_reset_function_codegen_state() noexcept {
    _emitted_blocks.clear();
    _block_map.clear();
    _block_tail.clear();
    _registered_physical_blocks.clear();
    _synthetic_blocks.clear();
    _control_flow_plan.reset();
    _current_xir_block = nullptr;
    _deferred_phis.clear();
    _deferred_phi_incomings_by_predecessor.clear();
    _ray_query_states.clear();
    _dispatch_metadata = {};
}

void SpirvCodegenEntry::_emit_function_blocks(const xir::FunctionDefinition *def) noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr, "SPIR-V function emission requires a control-flow plan.");
    for (auto &block_plan : _control_flow_plan->blocks()) {
        LUISA_ASSERT(_emit_block(block_plan.block),
                     "SPIR-V control-flow schedule attempted to emit a block more than once.");
    }
    LUISA_ASSERT(_emitted_blocks.size() == _control_flow_plan->blocks().size(),
                 "SPIR-V control-flow schedule did not emit every planned XIR block.");
    _finalize_phis();
}

void SpirvCodegenEntry::_emit_kernel(
    const xir::KernelFunction *kernel,
    const SpirvKernelArgumentLayoutPlan &argument_layout) noexcept {
    _reset_function_codegen_state();
    auto uniformity_blocks =
        collect_spirv_codegen_structural_closure(kernel);
    _uniformity.analyze(
        kernel, luisa::span<const xir::BasicBlock *const>{
                    uniformity_blocks.data(), uniformity_blocks.size()});
    auto ret_type = _builder.makeVoidType();
    LUISA_ASSERT(
        argument_layout.succeeded(),
        "SPIR-V kernel emission requires the validated module-wide argument "
        "layout plan.");
    LUISA_ASSERT(
        _buffer_metadata_offset ==
            argument_layout.buffer_metadata_offset,
        "SPIR-V kernel emission observed a direct-buffer metadata offset "
        "different from the layout frozen before callable emission.");
    spv::Block *entry = nullptr;
    // Entry point must have no parameters in SPIR-V
    auto func = _builder.makeFunctionEntry(spv::NoPrecision, ret_type, "main",
                                           spv::LinkageType::Max, {}, {}, &entry);
    _value_map.emplace(kernel, func->getId());
    _function_map.emplace(kernel, func);
    _register_physical_block(entry, true);

    // Load non-resource arguments from the frozen, dialect-validated host ABI
    // plan. Direct buffer-view metadata begins at the plan's final aligned
    // offset; emission never recalculates that boundary independently.
    if (!argument_layout.value_arguments.empty()) {
        LUISA_ASSERT(
            _has_argument_buffer &&
                _argument_buffer_id != spv::NoResult,
            "SPIR-V kernel value arguments require a generated argument "
            "buffer at the AST/XIR ABI handoff.");
        for (auto &&placement :
             argument_layout.value_arguments) {
            auto *arg = placement.argument;
            auto align = arg->type()->alignment();
            auto byte_offset = _builder.makeUintConstant(
                placement.byte_offset);
            auto loaded = _emit_buffer_read_impl(
                _argument_buffer_id, byte_offset, arg->type(), align);
            _value_map.emplace(arg, loaded);
        }
    }
    _entry_point_inst = _builder.addEntryPoint(spv::ExecutionModel::GLCompute, func, "main");
    for (auto id : _property_ids) {
        _register_entry_point_interface(id);
    }
    for (auto id : _deferred_entry_point_interface_ids) {
        _register_entry_point_interface(id);
    }
    _deferred_entry_point_interface_ids.clear();
    auto bs = kernel->block_size();
    _builder.addExecutionMode(func, spv::ExecutionMode::LocalSize,
                              static_cast<int32_t>(bs.x),
                              static_cast<int32_t>(bs.y),
                              static_cast<int32_t>(bs.z));

    _builder.enterFunction(func);
    _builder.setBuildPoint(entry);
    auto *metadata_merge =
        _emit_dispatch_metadata_prologue(func);
    LUISA_ASSERT(
        metadata_merge == _builder.getBuildPoint() &&
            _dispatch_metadata.dispatch_size != spv::NoResult,
        "SPIR-V kernel metadata prologue did not establish its merge state.");

    // Add dispatch bounds check to prevent extra threads in the last workgroup
    // from executing the kernel body.
    {
        auto &function = *func;
        auto uint_type = _builder.makeUintType(32);
        auto bool_type = _builder.makeBoolType();

        // Resolve either the direct push payload or the selected GPU-authored
        // logical dispatch record. Physical workgroup counts are deliberately
        // kept out of this bounds-check contract.
        auto dsp_size_loaded = _dispatch_metadata.dispatch_size;

        // Load dispatch_id (GlobalInvocationId) - reuse cached builtin if available
        if (_global_invocation_id_var == spv::NoResult) {
            auto uint3_type = _builder.makeVectorType(uint_type, 3);
            _global_invocation_id_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Input,
                                                                uint3_type, "dispatch_id");
            _builder.addDecoration(_global_invocation_id_var, spv::Decoration::BuiltIn, static_cast<int32_t>(spv::BuiltIn::GlobalInvocationId));
            _register_entry_point_interface(_global_invocation_id_var);
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
        auto return_block = _create_physical_block(&function);
        auto body_block = _create_physical_block(&function);

        // Selection merge
        auto selection_merge = new spv::Instruction(spv::Op::OpSelectionMerge);
        selection_merge->reserveOperands(2);
        selection_merge->addIdOperand(body_block->getId());
        selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
        metadata_merge->addInstruction(
            std::unique_ptr<spv::Instruction>(selection_merge));

        // Branch conditional
        _builder.createConditionalBranch(cmp, return_block, body_block);

        // Return block
        _builder.setBuildPoint(return_block);
        _builder.makeReturn(false);

        // Body block
        _builder.setBuildPoint(body_block);

        // Bind the logical body entry exactly once, after the dispatch prologue
        // has selected its physical continuation.
        auto [_, inserted] = _block_map.emplace(kernel->body_block(), body_block);
        LUISA_ASSERT(inserted, "SPIR-V kernel body entry was bound more than once.");
    }

    _prepare_control_flow_plan(kernel);
    _validate_ray_query_lifetimes(kernel);
    _predeclare_allocas(kernel);
    _emit_function_blocks(kernel);

    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.makeReturn(false);
    }
    _builder.leaveFunction();
}

Usage SpirvCodegenEntry::_resource_argument_binding_usage(const xir::Argument *argument) const noexcept {
    if (!_is_kernel_resource_argument(argument)) { return Usage::NONE; }
    auto func = argument->parent_function();
    if (func == nullptr || func->derived_function_tag() != xir::DerivedFunctionTag::KERNEL) { return Usage::NONE; }
    // The first usage analysis runs before descriptor planning so its result
    // can drive that planning. The second run folds the resulting binding
    // contract back into the final function-argument classification.
    if (_kernel_resource_bindings.empty()) { return Usage::NONE; }
    return _kernel_resource_binding(argument).usage;
}

void SpirvCodegenEntry::_analyze_function_argument_usage(const xir::Module *module) noexcept {
    _function_argument_usage =
        analyze_spirv_function_argument_usage(module);
    auto merge_usage = [](Usage lhs, Usage rhs) noexcept {
        return static_cast<Usage>(luisa::to_underlying(lhs) | luisa::to_underlying(rhs));
    };
    for (auto *function : module->function_list()) {
        auto fit = _function_argument_usage.find(function);
        if (fit == _function_argument_usage.end()) { continue; }
        auto index = size_t{0u};
        for (auto *argument : function->arguments()) {
            if (auto binding_usage =
                    _resource_argument_binding_usage(argument);
                binding_usage != Usage::NONE) {
                fit->second[index].usage = merge_usage(
                    fit->second[index].usage, binding_usage);
            }
            index++;
        }
    }
}

Usage SpirvCodegenEntry::_function_argument_usage_of(
    const xir::Function *function,
    const xir::Argument *argument) const noexcept {
    auto usage = spirv_function_argument_usage_of(
        _function_argument_usage, function, argument);
    return usage == Usage::NONE ? Usage::READ : usage;
}

void SpirvCodegenEntry::_emit_callable(const xir::CallableFunction *callable, const xir::Module *module) noexcept {
    _reset_function_codegen_state();
    auto uniformity_blocks =
        collect_spirv_codegen_structural_closure(callable);
    _uniformity.analyze(
        callable, luisa::span<const xir::BasicBlock *const>{
                      uniformity_blocks.data(), uniformity_blocks.size()});
    auto ret_type = _convert_type(callable->type(), Usage::READ);
    std::vector<spv::Id> param_types;
    luisa::vector<const xir::Argument *> emitted_args;
    luisa::vector<bool> arg_used;
    for (auto arg : callable->arguments()) {
        auto analyzed_usage = spirv_function_argument_usage_of(
            _function_argument_usage, callable, arg);
        bool used = analyzed_usage != Usage::NONE;
        auto module_specialized =
            arg->is_resource() &&
            _readonly_resource_origins.contains(arg);
        arg_used.push_back(used && !module_specialized);
        if ((!used || module_specialized) &&
            _is_kernel_resource_argument(arg)) {
            // Skip unused resource arguments to avoid type mismatches
            // between kernel globals (which may be arrays or have different
            // sampled/storage qualifiers) and callable parameters. A
            // module-specialized read-only resource is skipped for the same
            // ABI reason and resolved to its unique kernel binding at each
            // use inside the callable.
            continue;
        }
        auto usage = _function_argument_usage_of(callable, arg);
        if (_is_indirect_dispatch_type(arg->type())) {
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V legalization left an indirect-dispatch buffer as a "
                "callable parameter. It must be specialized at the call site "
                "before code generation.");
        } else if (arg->is_resource()) {
            auto type = arg->type();
            spv::Id pointee_type = spv::NoResult;
            spv::StorageClass storage = spv::StorageClass::Max;
            switch (type->tag()) {
                case Type::Tag::BUFFER:
                case Type::Tag::BINDLESS_ARRAY:
                    LUISA_ERROR_WITH_LOCATION(
                        "SPIR-V legalization left a callable {} resource "
                        "argument. Buffer and bindless resources must be "
                        "specialized at the call site before code generation.",
                        type->description());
                    break;
                case Type::Tag::ACCEL: {
                    auto writes = (luisa::to_underlying(usage) &
                                   luisa::to_underlying(Usage::WRITE)) != 0u;
                    LUISA_ASSERT(!writes,
                                 "SPIR-V callable accel writes require the "
                                 "separate instance-buffer descriptor and must "
                                 "be specialized at the call site.");
                    pointee_type = _convert_type(type, usage);
                    storage = spv::StorageClass::UniformConstant;
                    break;
                }
                case Type::Tag::TEXTURE: {
                    auto usage_bits = luisa::to_underlying(usage);
                    auto reads = (usage_bits &
                                  luisa::to_underlying(Usage::READ)) != 0u;
                    auto writes = (usage_bits &
                                   luisa::to_underlying(Usage::WRITE)) != 0u;
                    LUISA_ASSERT(!(reads && writes),
                                 "SPIR-V callable texture argument is both "
                                 "read and written. Dual sampled/storage-image "
                                 "bindings must be specialized at the call site.");
                    pointee_type = _convert_type(type, usage);
                    storage = spv::StorageClass::UniformConstant;
                    break;
                }
                default:
                    LUISA_NOT_IMPLEMENTED("SPIR-V callable resource argument type {}", type->description());
            }
            param_types.emplace_back(_builder.makePointer(storage, pointee_type));
        } else if (arg->is_reference()) {
            auto type = _convert_type(arg->type(), Usage::READ);
            param_types.emplace_back(_builder.makePointer(spv::StorageClass::Function, type));
            if (_is_ray_query_type(arg->type())) {
                // OpTypeRayQueryKHR is opaque and exposes no initialized TMax.
                // Carry the immutable initialization ray and mutable proceed
                // state beside the query pointer across callable boundaries.
                param_types.emplace_back(
                    _convert_type(_ray_query_initial_ray_type(), Usage::READ));
                param_types.emplace_back(_builder.makePointer(
                    spv::StorageClass::Function, _builder.makeBoolType()));
            }
        } else {
            param_types.emplace_back(_convert_type(arg->type(), Usage::READ));
        }
        emitted_args.push_back(arg);
    }
    auto requires_dispatch_metadata =
        _functions_requiring_dispatch_metadata.contains(callable);
    if (requires_dispatch_metadata) {
        auto uint_type = _builder.makeUintType(32u);
        param_types.emplace_back(
            _builder.makeVectorType(uint_type, 4u));
    }
    _callable_arg_used.emplace(callable, std::move(arg_used));
    spv::Block *entry = nullptr;
    auto func = _builder.makeFunctionEntry(spv::NoPrecision, ret_type,
                                           luisa::string{callable->name().value_or("callable")}.c_str(),
                                           spv::LinkageType::Max,
                                           param_types, {}, &entry);
    _value_map.emplace(callable, func->getId());
    _function_map.emplace(callable, func);
    _register_physical_block(entry, true);

    int32_t i = 0;
    for (auto arg : emitted_args) {
        auto param_id = func->getParamId(i++);
        _value_map.emplace(arg, param_id);
        if (arg->is_reference() && _is_ray_query_type(arg->type())) {
            auto initial_ray = func->getParamId(i++);
            auto proceed_state = func->getParamId(i++);
            auto [_, inserted] = _ray_query_states.emplace(
                param_id, RayQueryState{initial_ray, proceed_state});
            LUISA_ASSERT(inserted,
                         "SPIR-V callable ray-query parameter state was registered twice.");
        }
        if (arg->type()->tag() == Type::Tag::TEXTURE) {
            auto usage = _function_argument_usage_of(callable, arg);
            _is_storage_image_map.emplace(
                param_id,
                (luisa::to_underlying(usage) & luisa::to_underlying(Usage::WRITE)) != 0u);
        }
    }
    auto dispatch_metadata_param = spv::NoResult;
    if (requires_dispatch_metadata) {
        dispatch_metadata_param = func->getParamId(i++);
    }
    LUISA_ASSERT(
        static_cast<size_t>(i) == param_types.size(),
        "SPIR-V callable parameter planning produced {} IDs for {} types.",
        i, param_types.size());

    _builder.enterFunction(func);
    _builder.setBuildPoint(entry);
    if (requires_dispatch_metadata) {
        _set_dispatch_metadata(dispatch_metadata_param);
    }
    _block_map.emplace(callable->body_block(), entry);
    _prepare_control_flow_plan(callable);
    _validate_ray_query_lifetimes(callable);
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
    _direct_buffer_metadata_indices.clear();
    _bound_direct_buffer_bias_alignments.clear();
    _bindless_buffer_metadata_ids.clear();
    LUISA_ASSERT(
        _atomic_buffer_plan_installed,
        "SPIR-V emission requires a validated module-wide atomic-buffer "
        "representation plan before any resource type is converted.");
    LUISA_ASSERT(
        _runtime_target_plan_installed,
        "SPIR-V emission requires a validated runtime target plan before "
        "any descriptor or opaque type is converted.");
    auto analysis = _analyze_module_usage(module);
    _analyze_dispatch_metadata_requirements(analysis);
    _analyze_function_argument_usage(module);
    _readonly_resource_origins =
        analyze_spirv_readonly_resource_origins(
            module, _function_argument_usage);
    LUISA_ASSERT(!analysis.used_functions_post_order.empty() &&
                     analysis.used_functions_post_order.back()->isa<xir::KernelFunction>(),
                 "SPIR-V module plan requires the kernel to be last in callable post-order.");
    auto *kernel = static_cast<const xir::KernelFunction *>(
        analysis.used_functions_post_order.back());
    {
        luisa::vector<const xir::Argument *> xir_arguments;
        for (auto *argument : kernel->arguments()) {
            xir_arguments.emplace_back(argument);
        }
        LUISA_ASSERT(
            bindings.size() <= xir_arguments.size(),
            "SPIR-V kernel has {} bound AST arguments but only {} XIR arguments.",
            bindings.size(), xir_arguments.size());
        for (auto i = 0u; i < bindings.size(); ++i) {
            auto *argument = xir_arguments[i];
            luisa::visit(
                [&](auto &&binding) noexcept {
                    using Binding = std::remove_cvref_t<decltype(binding)>;
                    if constexpr (std::is_same_v<
                                      Binding,
                                      Function::BufferBinding>) {
                        if (argument->type() != nullptr &&
                            argument->type()->is_buffer() &&
                            argument->type()->element() == nullptr) {
                            // storage_buffer_descriptor_range() chooses a
                            // descriptor base divisible by four. Therefore
                            // bias = logical_offset - descriptor_offset has
                            // every power-of-two divisor shared by offset and
                            // four. Offset zero denotes the strongest fact.
                            _bound_direct_buffer_bias_alignments.emplace(
                                argument,
                                std::gcd(binding.offset, size_t{4u}));
                        }
                    }
                },
                bindings[i]);
        }
    }
    auto argument_layout =
        plan_spirv_kernel_argument_layout(kernel);
    LUISA_ASSERT(
        argument_layout.succeeded(),
        "SPIR-V dialect validation failed to reject an invalid kernel "
        "argument-block layout: {}",
        argument_layout.diagnostic);
    // Callables are emitted before their kernel in post-order. Any callable
    // specialized to a direct kernel buffer may load that buffer view's
    // offset, size, or address from the shared argument block, so the complete
    // kernel ABI must be frozen before the first function is emitted.
    _buffer_metadata_offset =
        argument_layout.buffer_metadata_offset;
    _kernel_block_size = kernel->block_size();

    // Storage capabilities must be fixed before function emission. Buffer
    // resources are StorageBuffer-backed; non-resource kernel arguments use the
    // generated cbuffer, and array constants use the constant UBO.
    for (auto *function : analysis.used_functions_post_order) {
        for (auto *argument : function->arguments()) {
            if (argument->type()->is_buffer()) {
                if (auto *element = argument->type()->element();
                    element != nullptr && !_buffer_uses_word_storage(argument->type())) {
                    _mark_16bit_storage_usage(element, spv::StorageClass::StorageBuffer);
                }
            }
        }
    }
    for (auto *constant : _ubo_array_constants) {
        _mark_8bit_storage_usage(constant->type(), spv::StorageClass::Uniform);
        _mark_16bit_storage_usage(constant->type(), spv::StorageClass::Uniform);
    }

    luisa::vector<const Type *> ordered_types;
    ordered_types.reserve(analysis.used_types.size());
    for (auto *type : analysis.used_types) { ordered_types.emplace_back(type); }
    std::sort(ordered_types.begin(), ordered_types.end(), [](const Type *lhs, const Type *rhs) noexcept {
        if (lhs == nullptr || rhs == nullptr) { return lhs == nullptr && rhs != nullptr; }
        return lhs->description() < rhs->description();
    });
    for (auto type : ordered_types) {
        if (type != nullptr && !_is_indirect_dispatch_type(type) &&
            !type->is_accel()) {
            // Accel is an opaque semantic resource, not a value type. Eagerly
            // converting every operand type would manufacture RayQueryKHR for
            // instance-buffer-only metadata access. Traversal bindings and
            // callable parameters convert it at their exact use sites.
            _convert_type(type, Usage::READ);
        }
    }

    luisa::vector<const xir::Constant *> ordered_constants;
    ordered_constants.reserve(analysis.used_constants.size());
    for (auto *constant : analysis.used_constants) { ordered_constants.emplace_back(constant); }
    std::sort(ordered_constants.begin(), ordered_constants.end(), [](const xir::Constant *lhs, const xir::Constant *rhs) noexcept {
        auto lhs_type = lhs->type();
        auto rhs_type = rhs->type();
        if (lhs_type->description() != rhs_type->description()) {
            return lhs_type->description() < rhs_type->description();
        }
        if (lhs->hash() != rhs->hash()) { return lhs->hash() < rhs->hash(); }
        return std::memcmp(lhs->data(), rhs->data(), lhs_type->size()) < 0;
    });
    for (auto c : ordered_constants) {
        if (_ubo_constant_member_indices.contains(c)) { continue; }
        _emit_constant(c);
    }

    if (_uses_8bit_storage_buffer || _uses_8bit_uniform_storage || _uses_8bit_push_constant) {
        _builder.addExtension(spv::E_SPV_KHR_8bit_storage);
    }
    if (_uses_8bit_storage_buffer) {
        _require_target_feature(
            target_feature::storage_buffer_8bit_access,
            _target_features.storage_buffer_8bit_access);
        _builder.addCapability(spv::Capability::StorageBuffer8BitAccess);
    }
    if (_uses_8bit_uniform_storage) {
        _require_target_feature(
            target_feature::uniform_storage_buffer_8bit_access,
            _target_features.uniform_storage_buffer_8bit_access);
        _builder.addCapability(spv::Capability::UniformAndStorageBuffer8BitAccess);
    }
    if (_uses_8bit_push_constant) {
        LUISA_ERROR(
            "Vulkan XIR-to-SPIR-V codegen does not enable "
            "storagePushConstant8.");
    }
    if (_uses_16bit_storage_buffer || _uses_16bit_uniform_storage || _uses_16bit_push_constant) {
        _builder.addExtension(spv::E_SPV_KHR_16bit_storage);
    }
    if (_uses_16bit_storage_buffer) {
        _require_target_feature(
            target_feature::storage_buffer_16bit_access,
            _target_features.storage_buffer_16bit_access);
        _builder.addCapability(spv::Capability::StorageBuffer16BitAccess);
    }
    if (_uses_16bit_uniform_storage) {
        _require_target_feature(
            target_feature::uniform_storage_buffer_16bit_access,
            _target_features.uniform_storage_buffer_16bit_access);
        _builder.addCapability(spv::Capability::UniformAndStorageBuffer16BitAccess);
    }
    if (_uses_16bit_push_constant) {
        LUISA_ERROR(
            "Vulkan XIR-to-SPIR-V codegen does not enable "
            "storagePushConstant16.");
    }

    for (auto f : analysis.used_functions_post_order) {
        if (auto def = f->definition()) {
            switch (f->derived_function_tag()) {
                case xir::DerivedFunctionTag::KERNEL:
                    LUISA_ASSERT(
                        f == kernel,
                        "SPIR-V module plan contains more than one kernel.");
                    _emit_kernel(kernel, argument_layout);
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
    luisa::string ext_inst_diagnostic;
    LUISA_ASSERT(
        validate_spirv_no_redundant_glsl_ext_inst(spirv, &ext_inst_diagnostic),
        "SPIR-V dialect validation failed: {}",
        ext_inst_diagnostic.empty() ? "unknown redundant GLSL.std.450 ExtInst" :
                                      luisa::string_view{ext_inst_diagnostic});
    std::ostringstream oss;
    spv::Disassemble(oss, spirv);
    _scratch << oss.str();
}

SpirvCodegenEntry::InstructionUsageAnalysis SpirvCodegenEntry::_analyze_module_usage(const xir::Module *module) noexcept {
    auto call_graph = validate_spirv_reachable_call_graph(module);
    LUISA_ASSERT(
        call_graph.succeeded(),
        "SPIR-V dialect validation failed to reject an invalid reachable call graph: {} ({} diagnostic(s) total).",
        call_graph.diagnostics.empty() ?
            luisa::string_view{"<missing diagnostic>"} :
            luisa::string_view{call_graph.diagnostics.front().message},
        call_graph.diagnostics.size());
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
    analysis.used_functions_post_order =
        std::move(call_graph.functions_post_order);
    for (auto *function : analysis.used_functions_post_order) {
        _analyze_instruction_usage(function, analysis);
    }
    LUISA_ASSERT(!analysis.used_functions_post_order.empty() &&
                     analysis.used_functions_post_order.back() == kernel,
                 "SPIR-V codegen: kernel function not found in post-order traversal.");
    return analysis;
}

void SpirvCodegenEntry::_analyze_dispatch_metadata_requirements(
    const InstructionUsageAnalysis &analysis) noexcept {
    _functions_requiring_dispatch_metadata.clear();
    // The traversal is callee-before-caller and recursion was rejected while
    // building it. One pass therefore propagates the hidden metadata parameter
    // transitively through every surviving callable chain.
    for (auto *function : analysis.used_functions_post_order) {
        auto requires_metadata =
            function->isa<xir::KernelFunction>();
        if (auto *definition = function->definition()) {
            traverse_spirv_codegen_structural_instructions(
                definition,
                [&](const xir::Instruction *instruction) noexcept {
                    for (auto *operand_use : instruction->operand_uses()) {
                        auto *operand = operand_use->value();
                        if (operand != nullptr &&
                            operand->isa<xir::SpecialRegister>()) {
                            auto tag = static_cast<const xir::SpecialRegister *>(
                                           operand)
                                           ->derived_special_register_tag();
                            requires_metadata |=
                                tag == xir::DerivedSpecialRegisterTag::DISPATCH_SIZE ||
                                tag == xir::DerivedSpecialRegisterTag::KERNEL_ID;
                        }
                    }
                    if (instruction->isa<xir::CallInst>()) {
                        auto *callee = static_cast<const xir::CallInst *>(
                                           instruction)
                                           ->callee();
                        requires_metadata |=
                            callee != nullptr &&
                            _functions_requiring_dispatch_metadata.contains(
                                callee);
                    }
                });
        }
        if (requires_metadata) {
            _functions_requiring_dispatch_metadata.emplace(function);
        }
    }
}

void SpirvCodegenEntry::_install_atomic_buffer_plan(
    const SpirvAtomicBufferModulePlan &plan) noexcept {
    LUISA_ASSERT(
        plan.succeeded(),
        "Cannot install an invalid SPIR-V atomic-buffer plan: {}",
        plan.diagnostics.empty() ?
            luisa::string_view{"<missing diagnostic>"} :
            luisa::string_view{plan.diagnostics.front().message});
    _needs_atomic_buffer_types.clear();
    _atomic_buffer_storage_plans.clear();
    _needs_atomic_buffer_types.reserve(plan.assignments.size());
    _atomic_buffer_storage_plans.reserve(plan.assignments.size());
    for (auto &&assignment : plan.assignments) {
        LUISA_ASSERT(
            assignment.buffer_type != nullptr &&
                assignment.storage !=
                    SpirvAtomicBufferStoragePlan::CONFLICT,
            "SPIR-V atomic-buffer plan contains an invalid assignment.");
        _needs_atomic_buffer_types.emplace(
            assignment.buffer_type);
        auto [iter, inserted] =
            _atomic_buffer_storage_plans.emplace(
                assignment.buffer_type, assignment.storage);
        LUISA_ASSERT(
            inserted || iter->second == assignment.storage,
            "SPIR-V atomic-buffer plan contains inconsistent duplicate "
            "assignments for Buffer<{}>.",
            assignment.buffer_type->element()->description());
    }
    _atomic_buffer_plan_installed = true;
}

void SpirvCodegenEntry::_install_runtime_target_plan(
    const SpirvRuntimeTargetPlan &plan) noexcept {
    LUISA_ASSERT(
        (plan.required_features & ~target_feature::known_mask) == 0u,
        "SPIR-V runtime target plan contains unknown feature bits "
        "0x{:016x}.",
        plan.required_features & ~target_feature::known_mask);
    LUISA_ASSERT(
        (plan.required_features & ~_target_features.enabled_mask()) == 0u,
        "Cannot install a SPIR-V runtime target plan with unavailable "
        "feature bits 0x{:016x}.",
        plan.required_features & ~_target_features.enabled_mask());
    _runtime_target_plan = plan;
    _runtime_target_plan_installed = true;
    // These flags describe descriptor-layout or Vulkan runtime semantics and
    // cannot be reconstructed by scanning optimized OpCapability records.
    _required_target_features |= plan.required_features;
}

}// namespace lc::spirv
