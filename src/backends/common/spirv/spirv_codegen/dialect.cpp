#include "dialect.h"
#include "aggregate_index.h"
#include "arithmetic_support.h"
#include "atomic_buffer_plan.h"
#include "argument_usage.h"
#include "call_graph_validation.h"
#include "control_flow_plan.h"
#include "instruction_layout.h"
#include "kernel_argument_layout.h"
#include "pointer_legalization.h"
#include "ray_query_lifetime.h"
#include "structural_closure.h"
#include "texture_sampling.h"

#include <type_traits>
#include <utility>

#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/thread_group.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/verifier.h>

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;

namespace {

using Support = SpirvXIRDialectSupport;
using SupportInfo = SpirvXIRDialectOpSupport;

[[nodiscard]] constexpr SupportInfo supported() noexcept {
    return {Support::SUPPORTED, {}};
}

[[nodiscard]] constexpr SupportInfo semantic_no_op(
    luisa::string_view reason) noexcept {
    return {Support::SEMANTIC_NO_OP, reason};
}

[[nodiscard]] constexpr SupportInfo unsupported(
    luisa::string_view reason) noexcept {
    return {Support::UNSUPPORTED, reason};
}

[[nodiscard]] constexpr SupportInfo unknown() noexcept {
    return {Support::UNKNOWN,
            "the opcode is not classified by the native SPIR-V dialect"};
}

template<typename Enum>
[[nodiscard]] constexpr auto enum_value(Enum value) noexcept {
    return static_cast<std::underlying_type_t<Enum>>(value);
}

[[nodiscard]] constexpr bool usage_contains(Usage usage,
                                            Usage expected) noexcept {
    return (static_cast<uint32_t>(usage) &
            static_cast<uint32_t>(expected)) != 0u;
}

[[nodiscard]] bool is_ray_query_type(const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           (type->description() == "LC_RayQueryAll" ||
            type->description() == "LC_RayQueryAny");
}

[[nodiscard]] bool is_indirect_dispatch_type(const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           type->description() == "LC_IndirectDispatchBuffer";
}

[[nodiscard]] bool is_spirv_plain_data_type(const Type *type) noexcept {
    if (type == nullptr) { return false; }
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
        case Type::Tag::FLOAT8_E5M2: return true;
        case Type::Tag::VECTOR:
            return type->element() != nullptr &&
                   type->element()->is_scalar();
        case Type::Tag::MATRIX:
            return type->element() != nullptr &&
                   type->element()->is_float32();
        case Type::Tag::ARRAY:
            // OpTypeArray requires a strictly positive length. XIR keeps
            // zero-length arrays representable for host-side type semantics,
            // so reject them explicitly at this backend boundary.
            return type->dimension() != 0u &&
                   is_spirv_plain_data_type(type->element());
        case Type::Tag::STRUCTURE:
            for (auto member : type->members()) {
                if (!is_spirv_plain_data_type(member)) { return false; }
            }
            return true;
        case Type::Tag::BUFFER:
        case Type::Tag::TEXTURE:
        case Type::Tag::BINDLESS_ARRAY:
        case Type::Tag::ACCEL:
        case Type::Tag::COOPERATIVE_VECTOR:
        case Type::Tag::COOPERATIVE_VECTOR_REF:
        case Type::Tag::COOPERATIVE_MATRIX_REF:
        case Type::Tag::CUSTOM: return false;
    }
    return false;
}

[[nodiscard]] bool is_spirv_value_type(const Type *type) noexcept {
    return is_spirv_plain_data_type(type) || is_ray_query_type(type);
}

[[nodiscard]] bool is_spirv_storage_layout_type(
    const Type *type) noexcept {
    if (!is_spirv_plain_data_type(type)) { return false; }
    if (type->is_array()) {
        // ArrayStride is an unsigned literal but must be strictly positive.
        // In particular, an array of empty structures has a positive element
        // count while its host stride is still zero.
        return type->element() != nullptr &&
               type->element()->size() != 0u &&
               is_spirv_storage_layout_type(type->element());
    }
    if (type->is_structure()) {
        for (auto member : type->members()) {
            if (!is_spirv_storage_layout_type(member)) { return false; }
        }
    }
    return true;
}

[[nodiscard]] bool is_spirv_resource_type(const Type *type) noexcept {
    if (type == nullptr) { return false; }
    switch (type->tag()) {
        case Type::Tag::BUFFER:
            return type->element() == nullptr ||
                   (type->element()->size() != 0u &&
                    is_spirv_storage_layout_type(type->element()));
        case Type::Tag::TEXTURE: {
            auto element = type->element();
            if (element != nullptr && element->is_vector()) {
                element = element->element();
            }
            return (type->dimension() == 2u || type->dimension() == 3u) &&
                   element != nullptr &&
                   (element->is_float32() || element->is_int32() ||
                    element->is_uint32());
        }
        case Type::Tag::BINDLESS_ARRAY:
        case Type::Tag::ACCEL: return true;
        default: return false;
    }
}

[[nodiscard]] const Type *subgroup_scalar_type(const Type *type) noexcept {
    while (type != nullptr && (type->is_vector() || type->is_matrix())) {
        type = type->element();
    }
    return type != nullptr && type->is_scalar() ? type : nullptr;
}

[[nodiscard]] bool subgroup_value_type_supported(const Type *type) noexcept {
    auto scalar = subgroup_scalar_type(type);
    return scalar != nullptr && !scalar->is_float8() &&
           !scalar->is_float64();
}

[[nodiscard]] bool type_contains_float8(const Type *type) noexcept {
    if (type == nullptr) { return false; }
    if (type->is_float8()) { return true; }
    if (type->is_vector() || type->is_matrix() || type->is_array()) {
        return type_contains_float8(type->element());
    }
    if (type->is_structure()) {
        for (auto member : type->members()) {
            if (type_contains_float8(member)) { return true; }
        }
    }
    return false;
}

[[nodiscard]] const xir::Value *root_address(
    const xir::Value *value) noexcept {
    while (value != nullptr && value->isa<xir::GEPInst>()) {
        value = static_cast<const xir::GEPInst *>(value)->base();
    }
    return value;
}

[[nodiscard]] luisa::string support_diagnostic(
    luisa::string_view family, luisa::string_view name,
    int64_t numeric_opcode, SupportInfo info) noexcept {
    if (info.support == Support::UNKNOWN) {
        return luisa::format(
            "Native XIR-to-SPIR-V rejected unknown {} opcode {} ('{}'); "
            "the dialect support matrix is fail-closed.",
            family, numeric_opcode, name);
    }
    return luisa::format(
        "Native XIR-to-SPIR-V does not support {} '{}': {}.",
        family, name, info.reason);
}

class DialectValidator {
private:
    SpirvXIRDialectValidationOptions _options;
    SpirvXIRDialectValidationResult _result;
    luisa::unordered_map<
        const xir::BasicBlock *,
        luisa::vector<const xir::LoopInst *>>
        _active_loop_prepare_owners;
    luisa::unordered_set<const Type *> _validated_type_layouts;
    luisa::unordered_set<const xir::Constant *>
        _validated_composite_constants;

private:
    void _error(const xir::Function *function,
                const xir::BasicBlock *block,
                const xir::Instruction *instruction,
                luisa::string message) noexcept {
        _result.diagnostics.emplace_back(SpirvXIRDialectDiagnostic{
            .function = function,
            .block = block,
            .instruction = instruction,
            .message = std::move(message),
        });
    }

    template<typename Enum>
    bool _require_supported(const xir::Function *function,
                            const xir::BasicBlock *block,
                            const xir::Instruction *instruction,
                            luisa::string_view family,
                            Enum op) noexcept {
        auto info = spirv_xir_dialect_support(op);
        if (info.accepted()) { return true; }
        auto name = info.known() ?
                        xir::to_string(op) :
                        luisa::string_view{"<unknown>"};
        _error(function, block, instruction,
               support_diagnostic(
                   family, name,
                   static_cast<int64_t>(enum_value(op)), info));
        return false;
    }

    void _validate_type_instruction_layout(
        const xir::Function *function,
        const xir::BasicBlock *block,
        const xir::Instruction *instruction,
        const Type *type) noexcept {
        if (type == nullptr ||
            !_validated_type_layouts.emplace(type).second) {
            return;
        }
        if (type->is_structure()) {
            // OpTypeStruct = header + result ID + one type ID per member.
            auto layout = plan_spirv_variadic_instruction(
                "OpTypeStruct", 2u, type->members().size());
            if (!layout) {
                _error(function, block, instruction,
                       std::move(layout.diagnostic));
            }
            for (auto *member : type->members()) {
                _validate_type_instruction_layout(
                    function, block, instruction, member);
            }
        } else if (type->is_vector() || type->is_matrix() ||
                   type->is_array() || type->is_buffer() ||
                   type->is_texture()) {
            _validate_type_instruction_layout(
                function, block, instruction, type->element());
        }
    }

    void _validate_composite_materialization_layout(
        const xir::Function *function,
        const xir::BasicBlock *block,
        const xir::Instruction *instruction,
        const Type *type,
        luisa::string_view opcode) noexcept {
        if (type == nullptr) { return; }
        auto constituent_count = size_t{0u};
        if (type->is_structure()) {
            constituent_count = type->members().size();
        } else if (type->is_vector() || type->is_matrix() ||
                   type->is_array()) {
            constituent_count = type->dimension();
        } else {
            return;
        }
        // OpConstantComposite and OpCompositeConstruct both consist of the
        // header, result type/result IDs, and one ID per constituent.
        auto layout = plan_spirv_variadic_instruction(
            opcode, 3u, constituent_count);
        if (!layout) {
            _error(function, block, instruction,
                   std::move(layout.diagnostic));
        }
        if (type->is_structure()) {
            for (auto *member : type->members()) {
                _validate_composite_materialization_layout(
                    function, block, instruction, member, opcode);
            }
        } else {
            _validate_composite_materialization_layout(
                function, block, instruction, type->element(), opcode);
        }
    }

    void _validate_variadic_instruction_layout(
        const xir::Function *function,
        const xir::BasicBlock *block,
        const xir::Instruction *instruction,
        luisa::string_view opcode,
        size_t fixed_word_count,
        size_t item_count) noexcept {
        auto layout = plan_spirv_variadic_instruction(
            opcode, fixed_word_count, item_count);
        if (!layout) {
            _error(function, block, instruction,
                   std::move(layout.diagnostic));
        }
    }

    void _validate_aggregate_indices(
        const xir::Function *function, const xir::BasicBlock *block,
        const xir::Instruction *instruction, const Type *aggregate_type,
        size_t first_index, size_t index_count, const Type *expected_type,
        luisa::string_view operation) noexcept {
        if (first_index > instruction->operand_count() ||
            index_count > instruction->operand_count() - first_index) {
            return;
        }
        luisa::vector<const xir::Value *> indices;
        indices.reserve(index_count);
        for (auto i = 0u; i < index_count; ++i) {
            indices.emplace_back(
                instruction->operand(first_index + i));
        }
        auto plan = plan_spirv_aggregate_indices(
            aggregate_type, luisa::span{indices});
        if (!plan) {
            _error(function, block, instruction,
                   luisa::format(
                       "Native XIR-to-SPIR-V rejected {} aggregate indices: {}",
                       operation, plan.diagnostic));
            return;
        }
        if (expected_type != nullptr &&
            plan.indexed_type != expected_type) {
            _error(function, block, instruction,
                   luisa::format(
                       "Native XIR-to-SPIR-V {} aggregate indices reach {}, not the declared {}.",
                       operation,
                       plan.indexed_type == nullptr ? "<null>" :
                                                      plan.indexed_type->description(),
                       expected_type->description()));
        }
    }

    void _validate_cast(const xir::Function *function,
                        const xir::BasicBlock *block,
                        const xir::CastInst *cast) noexcept {
        if (!_require_supported(function, block, cast, "cast", cast->op())) {
            return;
        }
        if (cast->operand_count() != 1u || cast->value() == nullptr ||
            cast->value()->type() == nullptr || cast->type() == nullptr) {
            _error(function, block, cast,
                   luisa::format(
                       "Native XIR-to-SPIR-V cast '{}' requires one typed operand "
                       "and a typed result.",
                       xir::to_string(cast->op())));
            return;
        }
        auto source = cast->value()->type();
        auto target = cast->type();
        if (!source->is_scalar_or_vector() ||
            !target->is_scalar_or_vector()) {
            _error(function, block, cast,
                   luisa::format(
                       "Native XIR-to-SPIR-V cast '{}' only supports scalar or "
                       "vector shapes, got {} -> {}.",
                       xir::to_string(cast->op()), source->description(),
                       target->description()));
            return;
        }
        if (cast->op() == xir::CastOp::STATIC_CAST) {
            if (source->dimension() != target->dimension()) {
                _error(function, block, cast,
                       luisa::format(
                           "Native XIR-to-SPIR-V static_cast requires equal scalar/vector "
                           "dimensions, got {} -> {}.",
                           source->description(), target->description()));
            }
            return;
        }
        if (source->is_bool_or_bool_vector() ||
            target->is_bool_or_bool_vector()) {
            _error(function, block, cast,
                   luisa::format(
                       "Native XIR-to-SPIR-V bitwise_cast cannot reinterpret logical "
                       "boolean values, got {} -> {}.",
                       source->description(), target->description()));
            return;
        }
        auto logical_width = [](const Type *type) noexcept {
            return type->is_vector() ?
                       type->element()->size() * type->dimension() :
                       type->size();
        };
        if (logical_width(source) != logical_width(target)) {
            _error(function, block, cast,
                   luisa::format(
                       "Native XIR-to-SPIR-V bitwise_cast requires equal logical bit "
                       "widths, got {} -> {}.",
                       source->description(), target->description()));
        }
    }

    void _validate_arithmetic(const xir::Function *function,
                              const xir::BasicBlock *block,
                              const xir::ArithmeticInst *inst) noexcept {
        if (!_require_supported(function, block, inst, "arithmetic operation",
                                inst->op())) {
            return;
        }
        // The emitter deliberately rejects arithmetic results in an FP8
        // scalar/vector/matrix. Comparisons and classification operations need
        // an operand check as well because their boolean result would otherwise
        // bypass that result-side guard and reach unsupported FP8 arithmetic.
        auto fp8_value_type = [](const Type *type) noexcept {
            return type != nullptr &&
                   (type->is_scalar() || type->is_vector() ||
                    type->is_matrix()) &&
                   type_contains_float8(type);
        };
        auto uses_fp8_arithmetic = fp8_value_type(inst->type());
        switch (inst->op()) {
            case xir::ArithmeticOp::BINARY_LESS:
            case xir::ArithmeticOp::BINARY_GREATER:
            case xir::ArithmeticOp::BINARY_LESS_EQUAL:
            case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
            case xir::ArithmeticOp::BINARY_EQUAL:
            case xir::ArithmeticOp::BINARY_NOT_EQUAL:
            case xir::ArithmeticOp::ISINF:
            case xir::ArithmeticOp::ISNAN:
                for (auto operand_use : inst->operand_uses()) {
                    auto operand = operand_use->value();
                    uses_fp8_arithmetic |=
                        operand != nullptr &&
                        fp8_value_type(operand->type());
                }
                break;
            default: break;
        }
        if (uses_fp8_arithmetic) {
            _error(function, block, inst,
                   luisa::format(
                       "Native XIR-to-SPIR-V arithmetic '{}' does not support FP8 "
                       "operands or results; widen to float16/float32 before "
                       "computing.",
                       xir::to_string(inst->op())));
        }
        if (spirv_glsl_transcendental_rejects_float64(inst->op())) {
            auto is_float64_value = [](const Type *type) noexcept {
                auto *scalar = subgroup_scalar_type(type);
                return scalar != nullptr && scalar->is_float64();
            };
            auto uses_float64 = is_float64_value(inst->type());
            for (auto *operand_use : inst->operand_uses()) {
                auto *operand = operand_use->value();
                uses_float64 |= operand != nullptr &&
                                is_float64_value(operand->type());
            }
            if (uses_float64) {
                _error(
                    function, block, inst,
                    luisa::format(
                        "Native XIR-to-SPIR-V arithmetic '{}' does not support float64 operands or results: GLSL.std.450 defines this transcendental operation only for float16/float32.",
                        xir::to_string(inst->op())));
            }
        }
        if (inst->op() == xir::ArithmeticOp::EXTRACT &&
            inst->operand_count() >= 2u && inst->operand(0) != nullptr) {
            _validate_aggregate_indices(
                function, block, inst, inst->operand(0)->type(), 1u,
                inst->operand_count() - 1u, inst->type(), "extract");
        } else if (inst->op() == xir::ArithmeticOp::INSERT &&
                   inst->operand_count() >= 3u &&
                   inst->operand(0) != nullptr &&
                   inst->operand(1) != nullptr) {
            _validate_aggregate_indices(
                function, block, inst, inst->operand(0)->type(), 2u,
                inst->operand_count() - 2u,
                inst->operand(1)->type(), "insert");
        }
    }

    void _validate_atomic(const xir::Function *function,
                          const xir::BasicBlock *block,
                          const xir::AtomicInst *inst) noexcept {
        if (!_require_supported(function, block, inst, "atomic operation",
                                inst->op())) {
            return;
        }
        if (inst->operand_count() == 0u || inst->base() == nullptr) {
            _error(function, block, inst,
                   luisa::format(
                       "Native XIR-to-SPIR-V atomic '{}' requires a non-null "
                       "address base before storage-class validation.",
                       xir::to_string(inst->op())));
            return;
        }
        auto minimum_operand_count = 1u + inst->value_count();
        if (inst->operand_count() < minimum_operand_count) {
            _error(function, block, inst,
                   luisa::format(
                       "Native XIR-to-SPIR-V atomic '{}' has {} operands; at least {} are required before aggregate indices can be planned.",
                       xir::to_string(inst->op()),
                       inst->operand_count(), minimum_operand_count));
            return;
        }
        _validate_aggregate_indices(
            function, block, inst, inst->base()->type(), 1u,
            inst->index_count(), inst->type(), "atomic");
        // Typed buffer variables add the wrapper-structure index before the
        // XIR-controlled aggregate sequence. Word-storage atomics collapse
        // the sequence to one word index, so this is a safe fail-closed upper
        // bound before the module-wide atomic representation plan runs.
        auto *base_type = inst->base()->type();
        auto access_fixed_word_count =
            base_type != nullptr && base_type->is_buffer() ? 5u : 4u;
        _validate_variadic_instruction_layout(
            function, block, inst, "OpAccessChain",
            access_fixed_word_count, inst->index_count());
        auto root = root_address(inst->base());
        auto alloca = root != nullptr && root->isa<xir::AllocaInst>() ?
                          static_cast<const xir::AllocaInst *>(root) :
                          nullptr;
        if (alloca != nullptr && !alloca->is_shared()) {
            _error(function, block, inst,
                   "Native XIR-to-SPIR-V atomics cannot target a function-local "
                   "allocation: SPIR-V atomic instructions do not admit the "
                   "Function storage class. Use an ordinary load/store for "
                   "thread-local state.");
            return;
        }
        if (alloca == nullptr &&
            (root == nullptr || root->type() == nullptr ||
             !root->type()->is_buffer())) {
            _error(function, block, inst,
                   "Native XIR-to-SPIR-V atomics require either a shared "
                   "allocation or a typed storage-buffer argument; this address "
                   "has no representable atomic storage class.");
            return;
        }
        if (inst->type() != nullptr && inst->type()->is_float32() &&
            inst->op() == xir::AtomicOp::COMPARE_EXCHANGE &&
            inst->operand_count() != 0u) {
            if (alloca != nullptr && alloca->is_shared()) {
                _error(function, block, inst,
                       "Native XIR-to-SPIR-V cannot implement float32 compare_exchange "
                       "on shared storage: the allocation has no integer-word "
                       "representation for core OpAtomicCompareExchange.");
            }
        }
    }

    void _validate_resource_query(
        const xir::Function *function, const xir::BasicBlock *block,
        const xir::ResourceQueryInst *inst) noexcept {
        if (!_require_supported(function, block, inst, "resource query",
                                inst->op())) {
            return;
        }
        if (inst->op() == xir::ResourceQueryOp::BUFFER_SIZE &&
            inst->operand_count() != 0u && inst->operand(0) != nullptr) {
            auto buffer = inst->operand(0)->type();
            auto element = buffer != nullptr && buffer->is_buffer() ?
                               buffer->element() :
                               nullptr;
            if (!is_spirv_plain_data_type(element) || element->size() == 0u) {
                _error(function, block, inst,
                       luisa::format(
                           "Native XIR-to-SPIR-V buffer_size requires a sized, "
                           "SPIR-V-representable element type, got {}.",
                           element == nullptr ? "<null>" :
                                                element->description()));
            }
        }
        if (inst->op() == xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE &&
            inst->operand_count() >= 3u) {
            uint64_t stride = 0u;
            if (xir::try_decode_constant_nonnegative_integer(
                    inst->operand(2), stride) &&
                stride == 0u) {
                _error(
                    function, block, inst,
                    "Native XIR-to-SPIR-V bindless_buffer_size requires a "
                    "nonzero element stride; a zero stride would make its "
                    "byte-size division undefined.");
            }
        }
        auto sample_info =
            spirv_texture_sample_op_info(inst->op());
        if (sample_info.valid && sample_info.direct &&
            inst->operand_count() != 0u && inst->operand(0) != nullptr) {
            auto *texture_type = inst->operand(0)->type();
            auto *sampled_type =
                texture_type != nullptr && texture_type->is_texture() ?
                    texture_type->element() :
                    nullptr;
            if (sampled_type != nullptr && sampled_type->is_vector()) {
                sampled_type = sampled_type->element();
            }
            if (sampled_type == nullptr || !sampled_type->is_float32()) {
                _error(
                    function, block, inst,
                    luisa::format(
                        "Native XIR-to-SPIR-V direct texture sampling '{}' requires a float32 texture because XIR defines every sampling result as float4; got sampled scalar type {}.",
                        xir::to_string(inst->op()),
                        sampled_type == nullptr ? "<null>" :
                                                  sampled_type->description()));
            }
        }
        if (sample_info.valid && sample_info.sampler_operands &&
            inst->operand_count() >= 2u) {
            auto validate_selector =
                [&](const xir::Value *value,
                    luisa::string_view name) noexcept {
                    if (value == nullptr ||
                        !spirv_sampler_selector_type_supported(
                            value->type())) {
                        _error(
                            function, block, inst,
                            luisa::format(
                                "Native XIR-to-SPIR-V texture sampler {} selector must be uint32, got {}.",
                                name,
                                value == nullptr || value->type() == nullptr ?
                                    "<null>" :
                                    value->type()->description()));
                        return;
                    }
                    auto decoded =
                        decode_spirv_sampler_selector_constant(value);
                    if (!decoded) {
                        _error(
                            function, block, inst,
                            luisa::format(
                                "Native XIR-to-SPIR-V texture sampler {} selector is invalid: {}",
                                name, decoded.diagnostic));
                    } else if (decoded.value &&
                               *decoded.value >= 4u) {
                        _error(
                            function, block, inst,
                            luisa::format(
                                "Native XIR-to-SPIR-V texture sampler {} selector {} is outside [0, 4).",
                                name, *decoded.value));
                    }
                };
            validate_selector(
                inst->operand(inst->operand_count() - 2u),
                "filter");
            validate_selector(
                inst->operand(inst->operand_count() - 1u),
                "address");
        }
    }

    void _validate_resource_read(
        const xir::Function *function, const xir::BasicBlock *block,
        const xir::ResourceReadInst *inst) noexcept {
        if (!_require_supported(function, block, inst, "resource read",
                                inst->op())) {
            return;
        }
        switch (inst->op()) {
            case xir::ResourceReadOp::BUFFER_READ:
            case xir::ResourceReadOp::BUFFER_VOLATILE_READ:
            case xir::ResourceReadOp::BYTE_BUFFER_READ:
            case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ:
            case xir::ResourceReadOp::BINDLESS_BUFFER_READ:
            case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ:
                if (!is_spirv_plain_data_type(inst->type())) {
                    _error(function, block, inst,
                           luisa::format(
                               "Native XIR-to-SPIR-V resource read '{}' cannot "
                               "materialize payload type {}.",
                               xir::to_string(inst->op()),
                               inst->type() == nullptr ? "<null>" :
                                                         inst->type()->description()));
                }
                _validate_composite_materialization_layout(
                    function, block, inst, inst->type(),
                    "OpCompositeConstruct");
                break;
            case xir::ResourceReadOp::TEXTURE2D_READ:
            case xir::ResourceReadOp::TEXTURE3D_READ:
            case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ:
            case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ:
            case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL:
            case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL:
            case xir::ResourceReadOp::DEVICE_ADDRESS_READ: break;
        }
    }

    void _validate_resource_write(
        const xir::Function *function, const xir::BasicBlock *block,
        const xir::ResourceWriteInst *inst) noexcept {
        if (!_require_supported(function, block, inst, "resource write",
                                inst->op())) {
            return;
        }
        if (inst->op() ==
                xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL ||
            inst->op() ==
                xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT) {
            auto expected_count =
                inst->op() ==
                        xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL ?
                    5u :
                    2u;
            if (inst->operand_count() != expected_count) {
                _error(function, block, inst,
                       luisa::format(
                           "Native XIR-to-SPIR-V indirect-dispatch write '{}' "
                           "requires {} operands, got {}.",
                           xir::to_string(inst->op()), expected_count,
                           inst->operand_count()));
                return;
            }
            auto base = inst->operand(0);
            auto argument =
                base != nullptr && base->isa<xir::Argument>() ?
                    static_cast<const xir::Argument *>(base) :
                    nullptr;
            if (argument == nullptr || !argument->is_reference() ||
                !is_indirect_dispatch_type(argument->type()) ||
                argument->parent_function() != function ||
                function->derived_function_tag() !=
                    xir::DerivedFunctionTag::KERNEL) {
                _error(function, block, inst,
                       luisa::format(
                           "Native XIR-to-SPIR-V indirect-dispatch write '{}' "
                           "requires the specialized LC_IndirectDispatchBuffer "
                           "reference argument of the containing kernel; ordinary "
                           "buffers, callable parameters, and local custom values "
                           "are not bound to the writable dispatch-record SSBO.",
                           xir::to_string(inst->op())));
                return;
            }
            auto uint_type = Type::of<uint32_t>();
            auto uint3_type = Type::of<uint3>();
            auto types_valid = inst->operand(1) != nullptr &&
                               inst->operand(1)->type() == uint_type;
            if (inst->op() ==
                xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL) {
                types_valid =
                    types_valid && inst->operand(2) != nullptr &&
                    inst->operand(2)->type() == uint3_type &&
                    inst->operand(3) != nullptr &&
                    inst->operand(3)->type() == uint3_type &&
                    inst->operand(4) != nullptr &&
                    inst->operand(4)->type() == uint_type;
            }
            if (!types_valid) {
                _error(function, block, inst,
                       inst->op() ==
                               xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL ?
                           "Native XIR-to-SPIR-V indirect_dispatch_set_kernel "
                           "requires (LC_IndirectDispatchBuffer, uint, uint3, "
                           "uint3, uint)." :
                           "Native XIR-to-SPIR-V indirect_dispatch_set_count "
                           "requires (LC_IndirectDispatchBuffer, uint).");
            }
            return;
        }
        size_t payload_index = 0u;
        switch (inst->op()) {
            case xir::ResourceWriteOp::BUFFER_WRITE:
            case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE:
            case xir::ResourceWriteOp::BYTE_BUFFER_WRITE:
            case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE:
            case xir::ResourceWriteOp::TEXTURE2D_WRITE:
            case xir::ResourceWriteOp::TEXTURE3D_WRITE:
                payload_index = 2u;
                break;
            case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE:
            case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE:
                payload_index = 3u;
                break;
            case xir::ResourceWriteOp::DEVICE_ADDRESS_WRITE:
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK:
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY:
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID:
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT:
            case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL:
            case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: break;
        }
        if (payload_index != 0u && inst->operand_count() > payload_index) {
            auto payload = inst->operand(payload_index);
            if (payload == nullptr ||
                !is_spirv_plain_data_type(payload->type())) {
                _error(function, block, inst,
                       luisa::format(
                           "Native XIR-to-SPIR-V resource write '{}' cannot store "
                           "payload type {}.",
                           xir::to_string(inst->op()),
                           payload == nullptr || payload->type() == nullptr ?
                               "<null>" :
                               payload->type()->description()));
            }
        }
    }

    void _validate_thread_group(
        const xir::Function *function, const xir::BasicBlock *block,
        const xir::ThreadGroupInst *inst) noexcept {
        if (!_require_supported(function, block, inst,
                                "thread-group operation", inst->op())) {
            return;
        }
        auto operand_type = [inst](size_t index) noexcept {
            if (index >= inst->operand_count()) {
                return static_cast<const Type *>(nullptr);
            }
            auto *operand = inst->operand(index);
            return operand == nullptr ? nullptr : operand->type();
        };
        auto is_scalar_or_vector = [](const Type *type) noexcept {
            return type != nullptr &&
                   (type->is_scalar() || type->is_vector());
        };
        auto is_integer_scalar_or_vector =
            [is_scalar_or_vector](const Type *type) noexcept {
                if (!is_scalar_or_vector(type)) { return false; }
                auto *scalar = type->is_vector() ? type->element() : type;
                return scalar != nullptr &&
                       (scalar->is_int() || scalar->is_uint());
            };
        auto is_numeric_scalar_or_vector =
            [is_scalar_or_vector](const Type *type) noexcept {
                if (!is_scalar_or_vector(type)) { return false; }
                auto *scalar = type->is_vector() ? type->element() : type;
                return scalar != nullptr &&
                       (scalar->is_int() || scalar->is_uint() ||
                        scalar->is_float());
            };
        auto is_shuffle_value = [](const Type *type) noexcept {
            return type != nullptr &&
                   (type->is_scalar() || type->is_vector() ||
                    type->is_matrix()) &&
                   subgroup_value_type_supported(type);
        };
        auto result_type = inst->type();
        auto value_type = operand_type(0u);
        auto valid = false;
        auto expected = luisa::string_view{};
        switch (inst->op()) {
            case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER:
                expected = "void with either no operands or two uint32 operands";
                valid = result_type == nullptr &&
                        (inst->operand_count() == 0u ||
                         (inst->operand_count() == 2u &&
                          value_type == Type::of<uint32_t>() &&
                          operand_type(1u) == Type::of<uint32_t>()));
                break;
            case xir::ThreadGroupOp::RASTER_QUAD_DDX:
            case xir::ThreadGroupOp::RASTER_QUAD_DDY:
                // Rejected by the support matrix above for compute entry points.
                return;
            case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE:
                expected = "no operands and a bool result";
                valid = inst->operand_count() == 0u &&
                        result_type == Type::of<bool>();
                break;
            case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE:
                expected = "no operands and a uint32 result";
                valid = inst->operand_count() == 0u &&
                        result_type == Type::of<uint32_t>();
                break;
            case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: {
                expected = "one supported scalar/vector operand and a bool result with the same shape";
                auto result_shape_matches =
                    value_type != nullptr &&
                    ((value_type->is_scalar() &&
                      result_type == Type::of<bool>()) ||
                     (value_type->is_vector() &&
                      result_type == Type::vector(
                                         Type::of<bool>(),
                                         value_type->dimension())));
                valid = inst->operand_count() == 1u &&
                        is_scalar_or_vector(value_type) &&
                        subgroup_value_type_supported(value_type) &&
                        result_shape_matches;
                break;
            }
            case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND:
            case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR:
            case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR:
                expected = "one integer scalar/vector operand and the same result type";
                valid = inst->operand_count() == 1u &&
                        is_integer_scalar_or_vector(value_type) &&
                        result_type == value_type;
                break;
            case xir::ThreadGroupOp::WARP_ACTIVE_MAX:
            case xir::ThreadGroupOp::WARP_ACTIVE_MIN:
            case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT:
            case xir::ThreadGroupOp::WARP_ACTIVE_SUM:
            case xir::ThreadGroupOp::WARP_PREFIX_SUM:
            case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT:
                expected = "one supported numeric scalar/vector operand and the same result type";
                valid = inst->operand_count() == 1u &&
                        is_numeric_scalar_or_vector(value_type) &&
                        subgroup_value_type_supported(value_type) &&
                        result_type == value_type;
                break;
            case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS:
            case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS:
                expected = "one bool operand and a uint32 result";
                valid = inst->operand_count() == 1u &&
                        value_type == Type::of<bool>() &&
                        result_type == Type::of<uint32_t>();
                break;
            case xir::ThreadGroupOp::WARP_ACTIVE_ALL:
            case xir::ThreadGroupOp::WARP_ACTIVE_ANY:
                expected = "one bool operand and a bool result";
                valid = inst->operand_count() == 1u &&
                        value_type == Type::of<bool>() &&
                        result_type == Type::of<bool>();
                break;
            case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK:
                expected = "one bool operand and a uint32x4 result";
                valid = inst->operand_count() == 1u &&
                        value_type == Type::of<bool>() &&
                        result_type == Type::of<uint4>();
                break;
            case xir::ThreadGroupOp::WARP_READ_LANE:
                expected = "a supported scalar/vector/matrix value, a uint32 lane index, and the value's result type";
                valid = inst->operand_count() == 2u &&
                        is_shuffle_value(value_type) &&
                        operand_type(1u) == Type::of<uint32_t>() &&
                        result_type == value_type;
                break;
            case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE:
                expected = "one supported scalar/vector/matrix operand and the same result type";
                valid = inst->operand_count() == 1u &&
                        is_shuffle_value(value_type) &&
                        result_type == value_type;
                break;
            case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK:
                expected = "no operands and no result";
                valid = inst->operand_count() == 0u && result_type == nullptr;
                break;
        }
        if (!valid) {
            _error(
                function, block, inst,
                luisa::format(
                    "Native XIR-to-SPIR-V subgroup operation '{}' requires {}.",
                    xir::to_string(inst->op()), expected));
        }
    }

    void _validate_special_register(
        const xir::Function *function, const xir::BasicBlock *block,
        const xir::Instruction *instruction,
        const xir::SpecialRegister *reg) noexcept {
        auto info = spirv_xir_dialect_support(
            reg->derived_special_register_tag());
        if (info.accepted()) { return; }
        auto name = info.known() ?
                        xir::to_string(reg->derived_special_register_tag()) :
                        luisa::string_view{"<unknown>"};
        _error(function, block, instruction,
               support_diagnostic(
                   "special register", name,
                   static_cast<int64_t>(enum_value(
                       reg->derived_special_register_tag())),
                   info));
    }

    void _validate_structural_closure(
        const xir::Function *function,
        const SpirvCodegenStructuralClosure &closure) noexcept {
        _active_loop_prepare_owners.clear();
        if (!closure.succeeded()) {
            switch (closure.status) {
                case SpirvCodegenStructuralClosureStatus::NULL_FUNCTION:
                    _error(
                        function, nullptr, nullptr,
                        "Native XIR-to-SPIR-V cannot validate a null function definition.");
                    break;
                case SpirvCodegenStructuralClosureStatus::MISSING_BODY:
                    _error(
                        function, nullptr, nullptr,
                        "Native XIR-to-SPIR-V requires every function definition to have a body block.");
                    break;
                case SpirvCodegenStructuralClosureStatus::UNOWNED_BLOCK:
                    _error(
                        function, closure.invalid_block, nullptr,
                        "Native XIR-to-SPIR-V structural closure references a null or foreign block that is not owned by the function.");
                    break;
                case SpirvCodegenStructuralClosureStatus::SUCCESS: break;
            }
            return;
        }

        luisa::unordered_set<const xir::BasicBlock *> active_blocks;
        active_blocks.reserve(closure.blocks.size());
        for (auto *block : closure.blocks) {
            active_blocks.emplace(block);
        }
        luisa::unordered_set<const xir::BasicBlock *> ordinary_blocks;
        ordinary_blocks.reserve(closure.ordinary_block_count);
        for (auto i = size_t{0u}; i < closure.ordinary_block_count; ++i) {
            ordinary_blocks.emplace(closure.blocks[i]);
        }

        luisa::unordered_map<
            const xir::BasicBlock *, const xir::Instruction *>
            merge_owners;
        luisa::unordered_map<const xir::BasicBlock *, uint32_t>
            loop_boundary_role_counts;
        auto require_role = [&](const xir::BasicBlock *owner,
                                const xir::Instruction *instruction,
                                const xir::BasicBlock *role,
                                luisa::string_view construct,
                                luisa::string_view role_name) noexcept {
            if (role == nullptr) {
                _error(
                    function, owner, instruction,
                    luisa::format(
                        "Native XIR-to-SPIR-V requires {} to have a non-null {} block.",
                        construct, role_name));
                return false;
            }
            if (!active_blocks.contains(role)) {
                _error(
                    function, owner, instruction,
                    luisa::format(
                        "Native XIR-to-SPIR-V requires the {} block of {} to belong to its active structural closure.",
                        role_name, construct));
                return false;
            }
            return true;
        };
        auto register_merge = [&](const xir::BasicBlock *owner,
                                  const xir::Instruction *instruction,
                                  const xir::BasicBlock *merge,
                                  luisa::string_view construct) noexcept {
            if (!require_role(
                    owner, instruction, merge, construct, "merge")) {
                return;
            }
            if (merge == owner) {
                _error(
                    function, owner, instruction,
                    luisa::format(
                        "Native XIR-to-SPIR-V rejects {} whose header/owner is also its merge block.",
                        construct));
            }
            if (auto [iter, inserted] =
                    merge_owners.emplace(merge, instruction);
                !inserted && iter->second != instruction) {
                _error(
                    function, owner, instruction,
                    "Native XIR-to-SPIR-V requires each active structured merge block to have exactly one owner.");
            }
        };
        auto reject_disconnected_structured_owner =
            [&](const xir::BasicBlock *owner,
                const xir::Instruction *instruction) noexcept {
                if (!ordinary_blocks.contains(owner)) {
                    _error(
                        function, owner, instruction,
                        "Native XIR-to-SPIR-V rejects a nested structured terminator in an ordinary-unreachable active role block; disconnected structured payloads have no backend dominance scope.");
                }
            };
        auto block_operand = [](const xir::Instruction *instruction,
                                size_t index) noexcept
            -> const xir::BasicBlock * {
            if (index >= instruction->operand_count()) { return nullptr; }
            auto *value = instruction->operand(index);
            return value != nullptr && value->isa<xir::BasicBlock>() ?
                       static_cast<const xir::BasicBlock *>(value) :
                       nullptr;
        };

        auto *entry = function->definition()->body_block();
        auto entry_predecessor_count = size_t{0u};
        for (auto *block : closure.blocks) {
            if (!block->is_terminated()) { continue; }
            auto *terminator = block->terminator();
            for (auto *operand_use : terminator->operand_uses()) {
                if (operand_use->value() == entry) {
                    entry_predecessor_count++;
                }
            }
            switch (terminator->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::IF: {
                    auto *instruction =
                        static_cast<const xir::IfInst *>(terminator);
                    reject_disconnected_structured_owner(block, instruction);
                    register_merge(
                        block, instruction, instruction->merge_block(), "If");
                    break;
                }
                case xir::DerivedInstructionTag::LOOP: {
                    auto *instruction =
                        static_cast<const xir::LoopInst *>(terminator);
                    reject_disconnected_structured_owner(block, instruction);
                    auto *prepare = block_operand(
                        instruction,
                        xir::LoopInst::operand_index_prepare_block);
                    auto *body = instruction->body_block();
                    auto *update = instruction->update_block();
                    auto *merge = instruction->merge_block();
                    auto roles_valid =
                        require_role(block, instruction, prepare, "Loop", "prepare") &
                        require_role(block, instruction, body, "Loop", "body") &
                        require_role(block, instruction, update, "Loop", "update") &
                        require_role(block, instruction, merge, "Loop", "merge");
                    if (roles_valid) {
                        luisa::unordered_set<const xir::BasicBlock *> roles;
                        roles.emplace(block);
                        roles.emplace(prepare);
                        roles.emplace(body);
                        roles.emplace(update);
                        roles.emplace(merge);
                        if (roles.size() != 5u) {
                            _error(
                                function, block, instruction,
                                "Native XIR-to-SPIR-V requires distinct Loop owner, prepare, body, update, and merge blocks.");
                        }
                        _active_loop_prepare_owners[prepare].emplace_back(
                            instruction);
                        loop_boundary_role_counts[prepare]++;
                        loop_boundary_role_counts[update]++;
                        if (auto [iter, inserted] =
                                merge_owners.emplace(merge, instruction);
                            !inserted && iter->second != instruction) {
                            _error(
                                function, block, instruction,
                                "Native XIR-to-SPIR-V requires each active structured merge block to have exactly one owner.");
                        }
                        auto prepare_plan =
                            plan_spirv_loop_prepare(instruction);
                        if (!prepare_plan.succeeded()) {
                            _error(
                                function, prepare,
                                prepare->is_terminated() ?
                                    prepare->terminator() :
                                    nullptr,
                                std::move(prepare_plan.diagnostic));
                        }
                    }
                    break;
                }
                case xir::DerivedInstructionTag::SIMPLE_LOOP: {
                    auto *instruction =
                        static_cast<const xir::SimpleLoopInst *>(terminator);
                    reject_disconnected_structured_owner(block, instruction);
                    auto *body = block_operand(
                        instruction,
                        xir::SimpleLoopInst::operand_index_body_block);
                    auto *merge = instruction->merge_block();
                    auto roles_valid =
                        require_role(block, instruction, body,
                                     "SimpleLoop", "body") &
                        require_role(block, instruction, merge,
                                     "SimpleLoop", "merge");
                    if (roles_valid) {
                        if (block == body || block == merge || body == merge) {
                            _error(
                                function, block, instruction,
                                "Native XIR-to-SPIR-V requires distinct SimpleLoop owner, body, and merge blocks.");
                        }
                        loop_boundary_role_counts[body]++;
                        if (auto [iter, inserted] =
                                merge_owners.emplace(merge, instruction);
                            !inserted && iter->second != instruction) {
                            _error(
                                function, block, instruction,
                                "Native XIR-to-SPIR-V requires each active structured merge block to have exactly one owner.");
                        }
                    }
                    break;
                }
                case xir::DerivedInstructionTag::SWITCH: {
                    auto *instruction =
                        static_cast<const xir::SwitchInst *>(terminator);
                    reject_disconnected_structured_owner(block, instruction);
                    register_merge(
                        block, instruction, instruction->merge_block(),
                        "Switch");
                    break;
                }
                case xir::DerivedInstructionTag::RAY_QUERY_LOOP: {
                    auto *instruction =
                        static_cast<const xir::RayQueryLoopInst *>(terminator);
                    reject_disconnected_structured_owner(block, instruction);
                    register_merge(
                        block, instruction, instruction->merge_block(),
                        "RayQueryLoop");
                    break;
                }
                default: break;
            }
        }
        for (auto [block, count] : loop_boundary_role_counts) {
            if (count > 1u) {
                _error(
                    function, block,
                    block->is_terminated() ? block->terminator() : nullptr,
                    "Native XIR-to-SPIR-V rejects a block shared by multiple Loop.prepare, Loop.update, or SimpleLoop.body boundary roles.");
            }
        }
        if (entry_predecessor_count != 0u) {
            _error(
                function, entry, nullptr,
                "Native XIR-to-SPIR-V requires the function body entry to have no logical predecessors.");
        }
        for (auto *instruction : entry->instructions()) {
            if (instruction->isa<xir::PhiInst>()) {
                _error(
                    function, entry, instruction,
                    "Native XIR-to-SPIR-V requires the function body entry to contain no Phi instructions.");
            }
        }

        // Blocks included only through a raw structured role are emitted as
        // disconnected structural-label payloads. Keep that policy explicit:
        // they may contain flat code, but no Phi, opaque ray-query state, or
        // instruction-valued dependency can cross a block boundary. This
        // avoids assigning an invented dominance/lifetime relation to a block
        // outside the ordinary CFG.
        for (auto i = closure.ordinary_block_count;
             i < closure.blocks.size(); ++i) {
            auto *block = closure.blocks[i];
            luisa::unordered_set<const xir::Instruction *> seen;
            for (auto *instruction : block->instructions()) {
                if (instruction->isa<xir::PhiInst>()) {
                    _error(
                        function, block, instruction,
                        "Native XIR-to-SPIR-V rejects Phi in an ordinary-unreachable active role block; disconnected payloads have no predecessor contract.");
                }
                if (instruction->isa<xir::BreakInst>() ||
                    instruction->isa<xir::ContinueInst>()) {
                    _error(
                        function, block, instruction,
                        "Native XIR-to-SPIR-V rejects Break or Continue in an ordinary-unreachable active role block; disconnected payloads have no enclosing structured scope.");
                }
                auto has_ray_query_value =
                    is_ray_query_type(instruction->type());
                for (auto *operand_use : instruction->operand_uses()) {
                    auto *operand = operand_use->value();
                    has_ray_query_value |=
                        operand != nullptr &&
                        is_ray_query_type(operand->type());
                }
                switch (instruction->derived_instruction_tag()) {
                    case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
                    case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
                    case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
                    case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
                    case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE:
                        has_ray_query_value = true;
                        break;
                    default: break;
                }
                if (has_ray_query_value) {
                    _error(
                        function, block, instruction,
                        "Native XIR-to-SPIR-V rejects opaque ray-query construction or use in an ordinary-unreachable active role block; ray-query lifetime validation requires ordinary CFG dominance.");
                }
                for (auto *operand_use : instruction->operand_uses()) {
                    auto *operand = operand_use->value();
                    if (operand == nullptr ||
                        !operand->isa<xir::Instruction>()) {
                        continue;
                    }
                    auto *definition =
                        static_cast<const xir::Instruction *>(operand);
                    if (definition->parent_block() != block) {
                        _error(
                            function, block, instruction,
                            "Native XIR-to-SPIR-V rejects a cross-block instruction value in an ordinary-unreachable active role block; disconnected payloads may use only arguments, constants, global-like values, and earlier definitions in the same block.");
                    } else if (!seen.contains(definition)) {
                        _error(
                            function, block, instruction,
                            "Native XIR-to-SPIR-V rejects a forward or cyclic same-block instruction value in an ordinary-unreachable active role block.");
                    }
                }
                seen.emplace(instruction);
            }
            if (block->is_terminated()) {
                auto *terminator = block->terminator();
                if (!terminator->isa<xir::ReturnInst>() &&
                    !terminator->isa<xir::UnreachableInst>()) {
                    _error(
                        function, block, terminator,
                        "Native XIR-to-SPIR-V requires an ordinary-unreachable active role block to end in Return or Unreachable; a disconnected branch could create an unplanned physical predecessor or structured nonlocal entry.");
                }
            }
        }
    }

    void _validate_conditional_branch(
        const xir::Function *function, const xir::BasicBlock *block,
        const xir::ConditionalBranchInst *branch) noexcept {
        auto owner_count = size_t{0u};
        if (auto iter = _active_loop_prepare_owners.find(block);
            iter != _active_loop_prepare_owners.end()) {
            owner_count = iter->second.size();
        }
        if (owner_count == 1u) { return; }
        _error(
            function, block, branch,
            owner_count == 0u ?
                "Native XIR-to-SPIR-V rejects raw ConditionalBranch "
                "outside canonical Loop.prepare; restructure_cfg must "
                "convert it to IfInst before codegen." :
                "Native XIR-to-SPIR-V rejects a ConditionalBranch in a "
                "prepare block shared by multiple LoopInst constructs.");
    }

    void _validate_instruction(const xir::Function *function,
                               const xir::Instruction *inst) noexcept {
        auto block = inst->parent_block();
        if (inst->find_metadata<xir::Reg2MemSpillMD>() != nullptr &&
            !inst->isa<xir::AllocaInst>()) {
            _error(
                function, block, inst,
                "Native XIR-to-SPIR-V rejected reg2mem spill metadata "
                "attached to a non-alloca instruction.");
        }
        auto release_assertion_no_op =
            inst->derived_instruction_tag() ==
                xir::DerivedInstructionTag::ASSERT &&
            _options.release_assertions_are_no_op;
        if (!release_assertion_no_op &&
            !_require_supported(function, block, inst, "instruction kind",
                                inst->derived_instruction_tag())) {
            return;
        }
        _validate_type_instruction_layout(
            function, block, inst, inst->type());
        for (auto *operand_use : inst->operand_uses()) {
            auto *operand = operand_use->value();
            _validate_type_instruction_layout(
                function, block, inst,
                operand == nullptr ? nullptr : operand->type());
            if (operand != nullptr &&
                operand->isa<xir::Constant>()) {
                auto *constant =
                    static_cast<const xir::Constant *>(operand);
                if (_validated_composite_constants.emplace(
                                                      constant)
                        .second) {
                    _validate_composite_materialization_layout(
                        function, block, inst, constant->type(),
                        "OpConstantComposite");
                }
            }
        }
        if (inst->type() != nullptr &&
            !is_spirv_value_type(inst->type())) {
            _error(function, block, inst,
                   luisa::format(
                       "Native XIR-to-SPIR-V cannot represent result/storage "
                       "type {} for instruction '{}'.",
                       inst->type()->description(),
                       xir::to_string(inst->derived_instruction_tag())));
        }
        for (auto operand_use : inst->operand_uses()) {
            auto operand = operand_use->value();
            if (operand != nullptr && operand->isa<xir::SpecialRegister>()) {
                _validate_special_register(
                    function, block, inst,
                    static_cast<const xir::SpecialRegister *>(operand));
            }
        }
        switch (inst->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::ALLOCA: {
                auto *alloca = static_cast<const xir::AllocaInst *>(inst);
                static_cast<void>(_require_supported(
                    function, block, inst, "allocation operation",
                    alloca->op()));
                if (auto *spill =
                        alloca->find_metadata<xir::Reg2MemSpillMD>()) {
                    _error(
                        function, block, inst,
                        luisa::format(
                            "Native XIR-to-SPIR-V rejected a remaining {} "
                            "reg2mem spill; post-restructure mem2reg must "
                            "recover SSA before codegen.",
                            xir::to_string(spill->kind())));
                }
                break;
            }
            case xir::DerivedInstructionTag::ARITHMETIC:
                _validate_arithmetic(
                    function, block,
                    static_cast<const xir::ArithmeticInst *>(inst));
                if (auto *arithmetic =
                        static_cast<const xir::ArithmeticInst *>(inst);
                    arithmetic->op() == xir::ArithmeticOp::AGGREGATE) {
                    // OpCompositeConstruct = header + result type/result IDs
                    // + one ID per constituent.
                    auto layout = plan_spirv_variadic_instruction(
                        "OpCompositeConstruct", 3u,
                        arithmetic->operand_count());
                    if (!layout) {
                        _error(function, block, inst,
                               std::move(layout.diagnostic));
                    }
                } else if (arithmetic->op() ==
                               xir::ArithmeticOp::INSERT &&
                           arithmetic->operand_count() >= 2u) {
                    // OpCompositeInsert has object/composite IDs before the
                    // literal index path. Its limit is one word stricter than
                    // the dynamic Function-storage OpAccessChain fallback.
                    _validate_variadic_instruction_layout(
                        function, block, inst, "OpCompositeInsert", 5u,
                        arithmetic->operand_count() - 2u);
                } else if (arithmetic->op() ==
                               xir::ArithmeticOp::EXTRACT &&
                           arithmetic->operand_count() >= 1u) {
                    // OpCompositeExtract and the dynamic OpAccessChain path
                    // have the same four-word fixed prefix.
                    _validate_variadic_instruction_layout(
                        function, block, inst,
                        "OpCompositeExtract/OpAccessChain", 4u,
                        arithmetic->operand_count() - 1u);
                }
                break;
            case xir::DerivedInstructionTag::ATOMIC:
                _validate_atomic(
                    function, block,
                    static_cast<const xir::AtomicInst *>(inst));
                break;
            case xir::DerivedInstructionTag::CAST:
                _validate_cast(function, block,
                               static_cast<const xir::CastInst *>(inst));
                break;
            case xir::DerivedInstructionTag::RESOURCE_QUERY:
                _validate_resource_query(
                    function, block,
                    static_cast<const xir::ResourceQueryInst *>(inst));
                break;
            case xir::DerivedInstructionTag::RESOURCE_READ:
                _validate_resource_read(
                    function, block,
                    static_cast<const xir::ResourceReadInst *>(inst));
                break;
            case xir::DerivedInstructionTag::RESOURCE_WRITE:
                _validate_resource_write(
                    function, block,
                    static_cast<const xir::ResourceWriteInst *>(inst));
                break;
            case xir::DerivedInstructionTag::THREAD_GROUP:
                _validate_thread_group(
                    function, block,
                    static_cast<const xir::ThreadGroupInst *>(inst));
                break;
            case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
                static_cast<void>(_require_supported(
                    function, block, inst, "ray-query object read",
                    static_cast<const xir::RayQueryObjectReadInst *>(inst)->op()));
                break;
            case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
                static_cast<void>(_require_supported(
                    function, block, inst, "ray-query object write",
                    static_cast<const xir::RayQueryObjectWriteInst *>(inst)->op()));
                break;
            case xir::DerivedInstructionTag::CALL: {
                auto call = static_cast<const xir::CallInst *>(inst);
                if (call->operand_count() == 0u) {
                    _error(function, block, inst,
                           "Native XIR-to-SPIR-V call has no callee operand.");
                    break;
                }
                auto *callee_value = call->operand(
                    xir::CallInst::operand_index_callee);
                auto *callee =
                    callee_value != nullptr &&
                            callee_value->isa<xir::Function>() ?
                        static_cast<const xir::Function *>(callee_value) :
                        nullptr;
                if (callee == nullptr || !callee->is_definition()) {
                    auto callee_name = callee == nullptr ?
                                           luisa::string_view{"<null>"} :
                                           callee->name().value_or(
                                               "<unnamed>");
                    _error(function, block, inst,
                           luisa::format(
                               "Native XIR-to-SPIR-V cannot call external function '{}': "
                               "the native path has no external-module linker.",
                               callee_name));
                } else if (call->argument_count() ==
                           callee->arguments().count_size()) {
                    auto argument_index = size_t{0u};
                    for (auto *formal : callee->arguments()) {
                        auto *actual = call->argument(argument_index);
                        if (formal != nullptr && formal->is_reference()) {
                            auto reference_actual =
                                validate_spirv_callable_reference_actual(
                                    actual);
                            if (!reference_actual) {
                                _error(
                                    function, block, inst,
                                    luisa::format(
                                        "Native XIR-to-SPIR-V callable reference argument {} for '{}' is unsupported: {}; specialize this call before codegen.",
                                        argument_index,
                                        callee->name().value_or("<unnamed>"),
                                        reference_actual.diagnostic));
                            }
                        }
                        argument_index++;
                    }
                }
                break;
            }
            case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
            case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
            case xir::DerivedInstructionTag::AUTODIFF_SCOPE:
            case xir::DerivedInstructionTag::AUTODIFF_INTRINSIC:
            case xir::DerivedInstructionTag::RASTER_DISCARD:
            case xir::DerivedInstructionTag::CORO_SUSPEND:
            case xir::DerivedInstructionTag::CORO_RESUME:
            case xir::DerivedInstructionTag::CORO_TERMINATE:
            case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE:
            case xir::DerivedInstructionTag::PRINT:
            case xir::DerivedInstructionTag::CLOCK:
            case xir::DerivedInstructionTag::DEBUG_BREAK:
            case xir::DerivedInstructionTag::ASSERT:
            case xir::DerivedInstructionTag::ASSUME:
            case xir::DerivedInstructionTag::OUTLINE:
                break;
            case xir::DerivedInstructionTag::IF:
            case xir::DerivedInstructionTag::LOOP:
            case xir::DerivedInstructionTag::SIMPLE_LOOP:
            case xir::DerivedInstructionTag::BRANCH: break;
            case xir::DerivedInstructionTag::CONDITIONAL_BRANCH:
                _validate_conditional_branch(
                    function, block,
                    static_cast<const xir::ConditionalBranchInst *>(inst));
                break;
            case xir::DerivedInstructionTag::UNREACHABLE:
            case xir::DerivedInstructionTag::BREAK:
            case xir::DerivedInstructionTag::CONTINUE:
            case xir::DerivedInstructionTag::RETURN:
            case xir::DerivedInstructionTag::LOAD:
            case xir::DerivedInstructionTag::STORE: break;
            case xir::DerivedInstructionTag::SWITCH: {
                auto switch_inst = static_cast<const xir::SwitchInst *>(inst);
                auto selector = switch_inst->operand_count() >
                                        xir::SwitchInst::operand_index_value ?
                                    switch_inst->operand(
                                        xir::SwitchInst::operand_index_value) :
                                    nullptr;
                auto layout = plan_spirv_switch_instruction(
                    selector == nullptr ? nullptr : selector->type(),
                    switch_inst->case_count());
                if (!layout) {
                    _error(function, block, inst, std::move(layout.diagnostic));
                }
                break;
            }
            case xir::DerivedInstructionTag::PHI: {
                auto phi = static_cast<const xir::PhiInst *>(inst);
                auto layout = plan_spirv_phi_instruction(
                    phi->incoming_count());
                if (!layout) {
                    _error(function, block, inst, std::move(layout.diagnostic));
                }
                break;
            }
            case xir::DerivedInstructionTag::GEP: {
                auto gep = static_cast<const xir::GEPInst *>(inst);
                if (gep->operand_count() >= 2u && gep->base() != nullptr) {
                    _validate_aggregate_indices(
                        function, block, gep, gep->base()->type(), 1u,
                        gep->index_count(), gep->type(), "GEP");
                    _validate_variadic_instruction_layout(
                        function, block, inst, "OpAccessChain", 4u,
                        gep->index_count());
                }
                break;
            }
        }
    }

public:
    explicit DialectValidator(
        SpirvXIRDialectValidationOptions options) noexcept
        : _options{options} {}

    [[nodiscard]] SpirvXIRDialectValidationResult validate(
        const xir::Module *module) noexcept {
        if (module == nullptr) {
            _error(nullptr, nullptr, nullptr,
                   "Cannot validate a null XIR module for native SPIR-V codegen.");
            return std::move(_result);
        }

        // Freeze the exact set of definitions that emission will consume
        // before applying any backend-specific rule. Generic XIR verification
        // below deliberately remains whole-module, but an unused callable is
        // not part of the native SPIR-V dialect merely because it shares the
        // same Module container as the kernel.
        auto call_graph =
            validate_spirv_reachable_call_graph(module);
        if (call_graph.succeeded()) {
            auto argument_analysis =
                analyze_spirv_function_argument_usage(module);
            // Match the emitter's hidden dispatch-metadata parameter
            // propagation exactly. The call graph is callee-before-caller.
            luisa::unordered_set<const xir::Function *>
                functions_requiring_dispatch_metadata;
            for (auto *function : call_graph.functions_post_order) {
                auto requires_metadata =
                    function->derived_function_tag() ==
                    xir::DerivedFunctionTag::KERNEL;
                if (auto *definition = function->definition()) {
                    traverse_spirv_codegen_structural_instructions(
                        definition,
                        [&](const xir::Instruction *instruction) noexcept {
                            for (auto *operand_use :
                                 instruction->operand_uses()) {
                                auto *operand = operand_use->value();
                                if (operand == nullptr ||
                                    !operand->isa<xir::SpecialRegister>()) {
                                    continue;
                                }
                                auto tag = static_cast<
                                               const xir::SpecialRegister *>(operand)
                                               ->derived_special_register_tag();
                                requires_metadata |=
                                    tag == xir::DerivedSpecialRegisterTag::DISPATCH_SIZE ||
                                    tag == xir::DerivedSpecialRegisterTag::KERNEL_ID;
                            }
                            if (instruction->isa<xir::CallInst>()) {
                                auto *callee = static_cast<
                                                   const xir::CallInst *>(instruction)
                                                   ->callee();
                                requires_metadata |=
                                    callee != nullptr &&
                                    functions_requiring_dispatch_metadata
                                        .contains(callee);
                            }
                        });
                }
                if (requires_metadata) {
                    functions_requiring_dispatch_metadata.emplace(
                        function);
                }
            }
            for (auto *function :
                 call_graph.functions_post_order) {
                auto function_name = function->name().value_or("<unnamed>");
                if (function->derived_function_tag() ==
                    xir::DerivedFunctionTag::CALLABLE) {
                    // OpName = header + target ID + nul-terminated string.
                    auto name_word_count =
                        function_name.size() / 4u + 1u;
                    _validate_variadic_instruction_layout(
                        function, nullptr, nullptr, "OpName", 2u,
                        name_word_count);
                }
                if (auto return_type = function->type(); return_type != nullptr) {
                    _validate_type_instruction_layout(
                        function, nullptr, nullptr, return_type);
                    if (is_ray_query_type(return_type)) {
                        _error(function, nullptr, nullptr,
                               luisa::format(
                                   "Native XIR-to-SPIR-V cannot return opaque ray-query "
                                   "objects from function '{}'; query state is only "
                                   "represented locally or through a callable reference "
                                   "argument with explicit side-channel state.",
                                   function_name));
                    } else if (!is_spirv_value_type(return_type)) {
                        _error(function, nullptr, nullptr,
                               luisa::format(
                                   "Native XIR-to-SPIR-V cannot represent return type {} "
                                   "in function '{}'.",
                                   return_type->description(), function_name));
                    }
                }
                for (auto argument : function->arguments()) {
                    auto type = argument->type();
                    _validate_type_instruction_layout(
                        function, nullptr, nullptr, type);
                    auto indirect_dispatch = is_indirect_dispatch_type(type);
                    auto ray_query = is_ray_query_type(type);
                    auto kernel = function->derived_function_tag() ==
                                  xir::DerivedFunctionTag::KERNEL;
                    auto callable = function->derived_function_tag() ==
                                    xir::DerivedFunctionTag::CALLABLE;
                    if (indirect_dispatch) {
                        if (argument->is_reference() && kernel) { continue; }
                        _error(function, nullptr, nullptr,
                               luisa::format(
                                   "Native XIR-to-SPIR-V only represents "
                                   "LC_IndirectDispatchBuffer as a reference argument "
                                   "of a kernel, not in function '{}'.",
                                   function_name));
                        continue;
                    }
                    if (ray_query) {
                        if (argument->is_reference() && callable) { continue; }
                        _error(function, nullptr, nullptr,
                               luisa::format(
                                   "Native XIR-to-SPIR-V only represents opaque ray-query "
                                   "arguments as callable references with explicit "
                                   "side-channel state, not in function '{}'.",
                                   function_name));
                        continue;
                    }
                    if (kernel && argument->is_reference()) {
                        _error(function, nullptr, nullptr,
                               luisa::format(
                                   "Native XIR-to-SPIR-V kernel '{}' has an unsupported "
                                   "reference argument of type {}; the kernel ABI only "
                                   "represents value payloads, resources, and the "
                                   "specialized indirect-dispatch buffer.",
                                   function_name,
                                   type == nullptr ? "<null>" :
                                                     type->description()));
                        continue;
                    }
                    if (callable && argument->is_resource() &&
                        type != nullptr) {
                        auto argument_usage =
                            spirv_function_argument_usage_of(
                                argument_analysis, function, argument);
                        if ((type->is_buffer() ||
                             type->is_bindless_array()) &&
                            argument_usage != Usage::NONE) {
                            _error(function, nullptr, nullptr,
                                   luisa::format(
                                       "Native XIR-to-SPIR-V callable '{}' retains a used {} "
                                       "resource argument; buffer and bindless descriptors "
                                       "must be specialized at call sites before codegen.",
                                       function_name, type->description()));
                            continue;
                        }
                        if (type->is_accel() &&
                            (usage_contains(argument_usage, Usage::WRITE) ||
                             spirv_function_argument_requires_accel_instance_buffer(
                                 argument_analysis, function, argument))) {
                            _error(function, nullptr, nullptr,
                                   luisa::format(
                                       "Native XIR-to-SPIR-V callable '{}' retains an "
                                       "acceleration-structure resource argument that is "
                                       "writable or requires instance-buffer state; such "
                                       "acceleration-structure arguments must be specialized "
                                       "at call sites before codegen.",
                                       function_name));
                            continue;
                        }
                        if (type->is_texture() &&
                            usage_contains(argument_usage, Usage::READ) &&
                            usage_contains(argument_usage, Usage::WRITE)) {
                            _error(function, nullptr, nullptr,
                                   luisa::format(
                                       "Native XIR-to-SPIR-V callable '{}' retains a texture "
                                       "resource argument used for both read and write; dual "
                                       "sampled/storage-image bindings must be specialized at "
                                       "call sites before codegen.",
                                       function_name));
                            continue;
                        }
                    }
                    auto supported_type = argument->is_resource() ?
                                              is_spirv_resource_type(type) :
                                              is_spirv_value_type(type);
                    if (!supported_type) {
                        _error(function, nullptr, nullptr,
                               luisa::format(
                                   "Native XIR-to-SPIR-V cannot represent {} argument "
                                   "type {} in function '{}'.",
                                   argument->is_resource() ? "resource" :
                                                             "value/reference",
                                   type == nullptr ? "<null>" :
                                                     type->description(),
                                   function_name));
                    }
                }
                if (function->derived_function_tag() ==
                    xir::DerivedFunctionTag::CALLABLE) {
                    auto emitted_parameter_count =
                        functions_requiring_dispatch_metadata.contains(
                            function) ?
                            size_t{1u} :
                            size_t{0u};
                    for (auto *argument : function->arguments()) {
                        auto usage = spirv_function_argument_usage_of(
                            argument_analysis, function, argument,
                            Usage::NONE);
                        if (argument->is_resource() &&
                            usage == Usage::NONE) {
                            continue;
                        }
                        emitted_parameter_count++;
                        if (argument->is_reference() &&
                            is_ray_query_type(argument->type())) {
                            // Query pointer plus immutable ray and mutable
                            // proceed-state side channels.
                            emitted_parameter_count += 2u;
                        }
                    }
                    // OpTypeFunction = header/result/return type + params.
                    _validate_variadic_instruction_layout(
                        function, nullptr, nullptr, "OpTypeFunction", 3u,
                        emitted_parameter_count);
                    // Every reachable callable has at least one call site;
                    // OpFunctionCall adds the callee ID to the typed result
                    // prefix and therefore has the stricter limit.
                    _validate_variadic_instruction_layout(
                        function, nullptr, nullptr, "OpFunctionCall", 4u,
                        emitted_parameter_count);
                }
                if (function->derived_function_tag() ==
                    xir::DerivedFunctionTag::KERNEL) {
                    for (auto *argument : function->arguments()) {
                        if (!argument->is_resource() &&
                            !argument->is_reference()) {
                            _validate_composite_materialization_layout(
                                function, nullptr, nullptr,
                                argument->type(),
                                "OpCompositeConstruct");
                        }
                    }
                    auto argument_layout =
                        plan_spirv_kernel_argument_layout(
                            static_cast<const xir::KernelFunction *>(
                                function));
                    if (!argument_layout) {
                        _error(function, nullptr, nullptr,
                               std::move(argument_layout.diagnostic));
                    }
                }
                if (auto *definition = function->definition()) {
                    auto closure =
                        plan_spirv_codegen_structural_closure(definition);
                    _validate_structural_closure(function, closure);
                    if (closure.succeeded()) {
                        for (auto *block : closure.blocks) {
                            block->traverse_instructions(
                                [&](const xir::Instruction *inst) noexcept {
                                    _validate_instruction(function, inst);
                                });
                        }
                    }
                }
            }
        }

        // Ordinary XIR verification remains part of this one handoff contract.
        // Dialect diagnostics are collected first so a recognized-but-
        // unsupported opcode gets the exact backend boundary message instead
        // of only a generic instruction-shape error.
        auto verification = xir::xir_verify_module(
            module,
            {.require_unique_merge_blocks = false,
             .require_canonical_break_continue_targets = true});
        for (auto &&error : verification.errors) {
            // Generic validity is a Module contract, independent of native
            // reachability. In particular, invalid orphan blocks and invalid
            // unused callable definitions remain invalid XIR even though they
            // have no physical SPIR-V counterpart.
            _error(error.function, error.block, error.instruction,
                   luisa::format(
                       "Invalid XIR at the native SPIR-V handoff: {}",
                       error.message));
        }

        // Usage analysis starts at the kernel and recursively visits every
        // function operand in the active structural closure. Preflight that
        // exact graph here so recursion remains a non-fatal dialect
        // diagnostic instead of reaching the emitter's traversal guard.
        for (auto &&diagnostic : call_graph.diagnostics) {
            _error(diagnostic.function, diagnostic.block,
                   diagnostic.instruction,
                   std::move(diagnostic.message));
        }

        // Pointer types are module-wide in SPIR-V Logical addressing. Plan
        // every atomic buffer across the exact kernel-reachable call graph
        // before emission chooses a typed or uint32-word representation.
        if (_result.diagnostics.empty()) {
            auto atomic_buffers = plan_spirv_atomic_buffers(
                luisa::span{
                    call_graph.functions_post_order.data(),
                    call_graph.functions_post_order.size()});
            for (auto &&diagnostic : atomic_buffers.diagnostics) {
                _error(
                    diagnostic.function,
                    diagnostic.instruction == nullptr ?
                        nullptr :
                        diagnostic.instruction->parent_block(),
                    diagnostic.instruction,
                    std::move(diagnostic.message));
            }
        }

        // The planner's final physical graph can reject a logically valid XIR
        // loop (for example, an explicit Loop whose update role is never
        // reached and therefore contributes no backedge). Query the exact
        // planner verdict here, while validation is still a non-fatal
        // reporting boundary. Only functions that passed all preceding
        // backend/generic checks reach this query; those checks are the
        // planner's structural precondition gate.
        auto has_module_error = false;
        for (auto &&diagnostic : _result.diagnostics) {
            has_module_error |= diagnostic.function == nullptr;
        }
        if (!has_module_error) {
            for (auto *function :
                 call_graph.functions_post_order) {
                auto *definition = function->definition();
                if (definition == nullptr) { continue; }
                auto has_function_error = false;
                for (auto &&diagnostic : _result.diagnostics) {
                    if (diagnostic.function == function) {
                        has_function_error = true;
                        break;
                    }
                }
                if (has_function_error) { continue; }
                auto ray_query_lifetimes =
                    validate_spirv_ray_query_lifetimes(definition);
                for (auto &&diagnostic :
                     ray_query_lifetimes.diagnostics) {
                    _error(function, diagnostic.block,
                           diagnostic.instruction,
                           std::move(diagnostic.message));
                }
                if (!ray_query_lifetimes.succeeded()) { continue; }
                auto physical = ControlFlowPlan::
                    validate_function_physical_loop_boundaries(definition);
                if (!physical.planning_succeeded()) {
                    _error(
                        function, nullptr, nullptr,
                        luisa::format(
                            "Native XIR-to-SPIR-V control-flow planning precondition failed: {}",
                            physical.planning_diagnostic));
                    continue;
                }
                for (auto loop_index = size_t{0u};
                     loop_index < physical.loops.size(); ++loop_index) {
                    auto &&loop = physical.loops[loop_index];
                    if (loop.succeeded()) { continue; }
                    _error(
                        function, nullptr, nullptr,
                        luisa::format(
                            "Native XIR-to-SPIR-V physical loop {} has {} entry edge(s) and {} backedge(s); the unique backedge must be dominated by its continue target and must not be dominated by its merge target (continue={}, merge={}).",
                            loop_index, loop.entry_edge_count,
                            loop.backedge_edge_count,
                            loop.backedge_dominated_by_continue_target,
                            loop.backedge_dominated_by_merge_target));
                }
            }
        }
        return std::move(_result);
    }
};

}// namespace

SpirvXIRKernelABIValidationResult validate_spirv_xir_kernel_abi(
    luisa::compute::Function ast_kernel,
    const xir::Module *module) noexcept {
    auto result = SpirvXIRKernelABIValidationResult{};
    auto fail = [&](SpirvXIRKernelABIStatus status,
                    luisa::string diagnostic,
                    size_t argument_index = ~size_t{0u}) noexcept {
        result.status = status;
        result.argument_index = argument_index;
        result.diagnostic = std::move(diagnostic);
        return result;
    };
    if (module == nullptr) {
        return fail(SpirvXIRKernelABIStatus::NULL_MODULE,
                    "native SPIR-V kernel ABI received a null XIR module");
    }
    if (ast_kernel.tag() != luisa::compute::Function::Tag::KERNEL) {
        return fail(
            SpirvXIRKernelABIStatus::AST_FUNCTION_IS_NOT_KERNEL,
            "native SPIR-V kernel ABI received a non-kernel AST function");
    }
    const xir::KernelFunction *xir_kernel = nullptr;
    auto kernel_count = size_t{0u};
    for (auto *function : module->function_list()) {
        if (function->derived_function_tag() !=
            xir::DerivedFunctionTag::KERNEL) {
            continue;
        }
        xir_kernel = static_cast<const xir::KernelFunction *>(function);
        kernel_count++;
    }
    if (kernel_count != 1u) {
        return fail(
            SpirvXIRKernelABIStatus::KERNEL_DEFINITION_COUNT_MISMATCH,
            luisa::format(
                "native SPIR-V kernel ABI requires exactly one XIR kernel; found {}",
                kernel_count));
    }
    auto ast_block_size = ast_kernel.block_size();
    auto xir_block_size = xir_kernel->block_size();
    if (xir_block_size.x != ast_block_size.x ||
        xir_block_size.y != ast_block_size.y ||
        xir_block_size.z != ast_block_size.z) {
        auto ast = ast_block_size;
        auto xir = xir_block_size;
        return fail(
            SpirvXIRKernelABIStatus::BLOCK_SIZE_MISMATCH,
            luisa::format(
                "native SPIR-V kernel block size differs between AST "
                "({},{},{}) and XIR ({},{},{})",
                ast.x, ast.y, ast.z, xir.x, xir.y, xir.z));
    }
    auto ast_arguments = ast_kernel.arguments();
    luisa::vector<const xir::Argument *> xir_arguments;
    for (auto *argument : xir_kernel->arguments()) {
        xir_arguments.emplace_back(argument);
    }
    if (ast_arguments.size() != xir_arguments.size()) {
        return fail(
            SpirvXIRKernelABIStatus::ARGUMENT_COUNT_MISMATCH,
            luisa::format(
                "native SPIR-V kernel argument count differs between AST ({}) and XIR ({})",
                ast_arguments.size(), xir_arguments.size()));
    }
    for (size_t i = 0u; i < ast_arguments.size(); ++i) {
        auto ast_argument = ast_arguments[i];
        auto *xir_argument = xir_arguments[i];
        auto *ast_type = ast_argument.type();
        auto *xir_type = xir_argument == nullptr ? nullptr :
                                                   xir_argument->type();
        if (ast_type == nullptr || xir_type == nullptr ||
            ast_type != xir_type) {
            return fail(
                SpirvXIRKernelABIStatus::ARGUMENT_TYPE_MISMATCH,
                luisa::format(
                    "native SPIR-V kernel argument {} type differs between AST ({}) and XIR ({})",
                    i,
                    ast_type == nullptr ? "<null>" :
                                          ast_type->description(),
                    xir_type == nullptr ? "<null>" :
                                          xir_type->description()),
                i);
        }
        auto expected_kind = [&]() noexcept {
            if (ast_type->is_resource()) {
                return xir::DerivedArgumentTag::RESOURCE;
            }
            if (ast_argument.is_reference() ||
                ast_type->is_custom()) {
                return xir::DerivedArgumentTag::REFERENCE;
            }
            return xir::DerivedArgumentTag::VALUE;
        }();
        if (xir_argument->derived_argument_tag() != expected_kind) {
            return fail(
                SpirvXIRKernelABIStatus::ARGUMENT_KIND_MISMATCH,
                luisa::format(
                    "native SPIR-V kernel argument {} kind differs between AST ({}) and XIR ({})",
                    i, static_cast<uint32_t>(expected_kind),
                    static_cast<uint32_t>(
                        xir_argument->derived_argument_tag())),
                i);
        }
    }
    return result;
}

// The runtime fallback below rejects forged/out-of-range enum values. For real
// enum growth, make an omitted classification a compile error on the compilers
// used by this project so an appended operation cannot silently inherit that
// fallback and masquerade as an intentionally unknown wire value.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch-enum"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch-enum"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(error : 4062)
#endif

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::AllocaOp op) noexcept {
    switch (op) {
        case xir::AllocaOp::LOCAL:
        case xir::AllocaOp::SHARED: return supported();
    }
    return unknown();
}

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::ArithmeticOp op) noexcept {
    switch (op) {
        case xir::ArithmeticOp::UNARY_MINUS:
        case xir::ArithmeticOp::UNARY_BIT_NOT:
        case xir::ArithmeticOp::BINARY_ADD:
        case xir::ArithmeticOp::BINARY_SUB:
        case xir::ArithmeticOp::BINARY_MUL:
        case xir::ArithmeticOp::BINARY_DIV:
        case xir::ArithmeticOp::BINARY_MOD:
        case xir::ArithmeticOp::BINARY_BIT_AND:
        case xir::ArithmeticOp::BINARY_BIT_OR:
        case xir::ArithmeticOp::BINARY_BIT_XOR:
        case xir::ArithmeticOp::BINARY_SHIFT_LEFT:
        case xir::ArithmeticOp::BINARY_SHIFT_RIGHT:
        case xir::ArithmeticOp::BINARY_ROTATE_LEFT:
        case xir::ArithmeticOp::BINARY_ROTATE_RIGHT:
        case xir::ArithmeticOp::BINARY_LESS:
        case xir::ArithmeticOp::BINARY_GREATER:
        case xir::ArithmeticOp::BINARY_LESS_EQUAL:
        case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
        case xir::ArithmeticOp::BINARY_EQUAL:
        case xir::ArithmeticOp::BINARY_NOT_EQUAL:
        case xir::ArithmeticOp::ALL:
        case xir::ArithmeticOp::ANY:
        case xir::ArithmeticOp::SELECT:
        case xir::ArithmeticOp::CLAMP:
        case xir::ArithmeticOp::SATURATE:
        case xir::ArithmeticOp::LERP:
        case xir::ArithmeticOp::SMOOTHSTEP:
        case xir::ArithmeticOp::STEP:
        case xir::ArithmeticOp::ABS:
        case xir::ArithmeticOp::MIN:
        case xir::ArithmeticOp::MAX:
        case xir::ArithmeticOp::CLZ:
        case xir::ArithmeticOp::CTZ:
        case xir::ArithmeticOp::POPCOUNT:
        case xir::ArithmeticOp::REVERSE:
        case xir::ArithmeticOp::ISINF:
        case xir::ArithmeticOp::ISNAN:
        case xir::ArithmeticOp::ACOS:
        case xir::ArithmeticOp::ACOSH:
        case xir::ArithmeticOp::ASIN:
        case xir::ArithmeticOp::ASINH:
        case xir::ArithmeticOp::ATAN:
        case xir::ArithmeticOp::ATAN2:
        case xir::ArithmeticOp::ATANH:
        case xir::ArithmeticOp::COS:
        case xir::ArithmeticOp::COSH:
        case xir::ArithmeticOp::SIN:
        case xir::ArithmeticOp::SINH:
        case xir::ArithmeticOp::TAN:
        case xir::ArithmeticOp::TANH:
        case xir::ArithmeticOp::EXP:
        case xir::ArithmeticOp::EXP2:
        case xir::ArithmeticOp::EXP10:
        case xir::ArithmeticOp::LOG:
        case xir::ArithmeticOp::LOG2:
        case xir::ArithmeticOp::LOG10:
        case xir::ArithmeticOp::POW:
        case xir::ArithmeticOp::POW_INT:
        case xir::ArithmeticOp::SQRT:
        case xir::ArithmeticOp::RSQRT:
        case xir::ArithmeticOp::CEIL:
        case xir::ArithmeticOp::FLOOR:
        case xir::ArithmeticOp::FRACT:
        case xir::ArithmeticOp::TRUNC:
        case xir::ArithmeticOp::ROUND:
        case xir::ArithmeticOp::RINT:
        case xir::ArithmeticOp::FMA:
        case xir::ArithmeticOp::COPYSIGN:
        case xir::ArithmeticOp::CROSS:
        case xir::ArithmeticOp::DOT:
        case xir::ArithmeticOp::LENGTH:
        case xir::ArithmeticOp::LENGTH_SQUARED:
        case xir::ArithmeticOp::NORMALIZE:
        case xir::ArithmeticOp::FACEFORWARD:
        case xir::ArithmeticOp::REFLECT:
        case xir::ArithmeticOp::REDUCE_SUM:
        case xir::ArithmeticOp::REDUCE_PRODUCT:
        case xir::ArithmeticOp::REDUCE_MIN:
        case xir::ArithmeticOp::REDUCE_MAX:
        case xir::ArithmeticOp::OUTER_PRODUCT:
        case xir::ArithmeticOp::MATRIX_COMP_NEG:
        case xir::ArithmeticOp::MATRIX_COMP_ADD:
        case xir::ArithmeticOp::MATRIX_COMP_SUB:
        case xir::ArithmeticOp::MATRIX_COMP_MUL:
        case xir::ArithmeticOp::MATRIX_COMP_DIV:
        case xir::ArithmeticOp::MATRIX_LINALG_MUL:
        case xir::ArithmeticOp::MATRIX_DETERMINANT:
        case xir::ArithmeticOp::MATRIX_TRANSPOSE:
        case xir::ArithmeticOp::MATRIX_INVERSE:
        case xir::ArithmeticOp::AGGREGATE:
        case xir::ArithmeticOp::SHUFFLE:
        case xir::ArithmeticOp::INSERT:
        case xir::ArithmeticOp::EXTRACT: return supported();
    }
    return unknown();
}

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::AtomicOp op) noexcept {
    switch (op) {
        case xir::AtomicOp::EXCHANGE:
        case xir::AtomicOp::COMPARE_EXCHANGE:
        case xir::AtomicOp::FETCH_ADD:
        case xir::AtomicOp::FETCH_SUB:
        case xir::AtomicOp::FETCH_AND:
        case xir::AtomicOp::FETCH_OR:
        case xir::AtomicOp::FETCH_XOR:
        case xir::AtomicOp::FETCH_MIN:
        case xir::AtomicOp::FETCH_MAX: return supported();
    }
    return unknown();
}

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::CastOp op) noexcept {
    switch (op) {
        case xir::CastOp::STATIC_CAST:
        case xir::CastOp::BITWISE_CAST: return supported();
    }
    return unknown();
}

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::ResourceQueryOp op) noexcept {
    switch (op) {
        case xir::ResourceQueryOp::BUFFER_SIZE:
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE:
        case xir::ResourceQueryOp::TEXTURE2D_SIZE:
        case xir::ResourceQueryOp::TEXTURE3D_SIZE:
        case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE:
        case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL:
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE:
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: return supported();
        case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS:
        case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS:
            return supported();
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT:
            return unsupported(
                "the Vulkan acceleration-structure instance ABI has no native "
                "motion-key representation in this code generator");
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
            return unsupported(
                "OpRayQueryInitializeKHR cannot represent the XIR motion-time operand");
    }
    return unknown();
}

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::ResourceReadOp op) noexcept {
    switch (op) {
        case xir::ResourceReadOp::BUFFER_READ:
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ:
        case xir::ResourceReadOp::BYTE_BUFFER_READ:
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ:
        case xir::ResourceReadOp::TEXTURE2D_READ:
        case xir::ResourceReadOp::TEXTURE3D_READ:
        case xir::ResourceReadOp::BINDLESS_BUFFER_READ:
        case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ:
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ:
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ:
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL:
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: return supported();
        case xir::ResourceReadOp::DEVICE_ADDRESS_READ:
            return unsupported(
                "physical-storage-buffer loads are not implemented");
    }
    return unknown();
}

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::ResourceWriteOp op) noexcept {
    switch (op) {
        case xir::ResourceWriteOp::BUFFER_WRITE:
        case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE:
        case xir::ResourceWriteOp::BYTE_BUFFER_WRITE:
        case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE:
        case xir::ResourceWriteOp::TEXTURE2D_WRITE:
        case xir::ResourceWriteOp::TEXTURE3D_WRITE:
        case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE:
        case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE:
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK:
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY:
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID:
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL:
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT:
            return supported();
        case xir::ResourceWriteOp::DEVICE_ADDRESS_WRITE:
            return unsupported(
                "physical-storage-buffer stores are not implemented");
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT:
            return unsupported(
                "the Vulkan acceleration-structure instance ABI has no native "
                "motion-key representation in this code generator");
    }
    return unknown();
}

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::ThreadGroupOp op) noexcept {
    switch (op) {
        case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER:
            return semantic_no_op(
                "shader execution reordering is an optimization-only scheduling "
                "hint and may be ignored without changing defined shader results");
        case xir::ThreadGroupOp::RASTER_QUAD_DDX:
        case xir::ThreadGroupOp::RASTER_QUAD_DDY:
            return unsupported(
                "quad derivatives require a raster invocation model, while this "
                "native path emits GLCompute entry points");
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE:
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE:
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL:
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND:
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR:
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR:
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS:
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX:
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN:
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT:
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM:
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL:
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY:
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK:
        case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS:
        case xir::ThreadGroupOp::WARP_PREFIX_SUM:
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT:
        case xir::ThreadGroupOp::WARP_READ_LANE:
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE:
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK: return supported();
    }
    return unknown();
}

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::RayQueryObjectReadOp op) noexcept {
    switch (op) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT:
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT:
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT:
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE:
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE:
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED:
            return supported();
    }
    return unknown();
}

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::RayQueryObjectWriteOp op) noexcept {
    switch (op) {
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE:
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL:
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE:
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED:
            return supported();
    }
    return unknown();
}

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::DerivedSpecialRegisterTag tag) noexcept {
    switch (tag) {
        case xir::DerivedSpecialRegisterTag::THREAD_ID:
        case xir::DerivedSpecialRegisterTag::BLOCK_ID:
        case xir::DerivedSpecialRegisterTag::WARP_LANE_ID:
        case xir::DerivedSpecialRegisterTag::DISPATCH_ID:
        case xir::DerivedSpecialRegisterTag::KERNEL_ID:
        case xir::DerivedSpecialRegisterTag::BLOCK_SIZE:
        case xir::DerivedSpecialRegisterTag::WARP_SIZE:
        case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE: return supported();
        case xir::DerivedSpecialRegisterTag::RASTER_OBJECT_ID:
        case xir::DerivedSpecialRegisterTag::RASTER_BARYCENTRICS:
            return unsupported(
                "the native code generator emits compute entry points and has no "
                "raster-stage builtin for this value");
    }
    return unknown();
}

SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::DerivedInstructionTag tag) noexcept {
    switch (tag) {
        case xir::DerivedInstructionTag::IF:
        case xir::DerivedInstructionTag::SWITCH:
        case xir::DerivedInstructionTag::LOOP:
        case xir::DerivedInstructionTag::SIMPLE_LOOP:
        case xir::DerivedInstructionTag::BRANCH:
        case xir::DerivedInstructionTag::CONDITIONAL_BRANCH:
        case xir::DerivedInstructionTag::UNREACHABLE:
        case xir::DerivedInstructionTag::BREAK:
        case xir::DerivedInstructionTag::CONTINUE:
        case xir::DerivedInstructionTag::RETURN:
        case xir::DerivedInstructionTag::PHI:
        case xir::DerivedInstructionTag::ALLOCA:
        case xir::DerivedInstructionTag::LOAD:
        case xir::DerivedInstructionTag::STORE:
        case xir::DerivedInstructionTag::GEP:
        case xir::DerivedInstructionTag::ATOMIC:
        case xir::DerivedInstructionTag::ARITHMETIC:
        case xir::DerivedInstructionTag::THREAD_GROUP:
        case xir::DerivedInstructionTag::RESOURCE_QUERY:
        case xir::DerivedInstructionTag::RESOURCE_READ:
        case xir::DerivedInstructionTag::RESOURCE_WRITE:
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
        case xir::DerivedInstructionTag::CALL:
        case xir::DerivedInstructionTag::CAST: return supported();
        case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
        case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
            return unsupported(
                "ray-query structured instructions must be lowered to ordinary "
                "control flow before the native codegen handoff");
        case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE:
            return unsupported(
                "ray-query pipelines must be lowered to ordinary ray-query "
                "control flow before the native codegen handoff");
        case xir::DerivedInstructionTag::AUTODIFF_SCOPE:
        case xir::DerivedInstructionTag::AUTODIFF_INTRINSIC:
            return unsupported(
                "automatic-differentiation instructions must be lowered before "
                "the native codegen handoff");
        case xir::DerivedInstructionTag::RASTER_DISCARD:
            return unsupported(
                "the native code generator emits compute entry points and has "
                "no raster-discard execution mode");
        case xir::DerivedInstructionTag::CORO_SUSPEND:
        case xir::DerivedInstructionTag::CORO_RESUME:
        case xir::DerivedInstructionTag::CORO_TERMINATE:
            return unsupported(
                "SPIR-V coroutine state-machine lowering is not implemented");
        case xir::DerivedInstructionTag::PRINT:
            return unsupported(
                "printing is not implemented and dropping its side effect is "
                "forbidden");
        case xir::DerivedInstructionTag::CLOCK: return supported();
        case xir::DerivedInstructionTag::DEBUG_BREAK:
            return unsupported(
                "the native path has no debug-break instruction contract");
        case xir::DerivedInstructionTag::ASSERT:
            return unsupported(
                "assertion intrinsics require a device-side failure-reporting "
                "contract before native SPIR-V codegen");
        case xir::DerivedInstructionTag::ASSUME:
            return semantic_no_op(
                "assumptions are optimization-only hints whose false condition "
                "already has undefined behavior, so ignoring them preserves all "
                "defined shader results");
        case xir::DerivedInstructionTag::OUTLINE:
            return unsupported(
                "outline regions must be lowered before the native codegen "
                "handoff");
    }
    return unknown();
}

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

SpirvXIRDialectValidationResult
validate_spirv_xir_codegen_dialect(
    const xir::Module *module,
    SpirvXIRDialectValidationOptions options) noexcept {
    return DialectValidator{options}.validate(module);
}

}// namespace lc::spirv
