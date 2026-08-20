#include <algorithm>
#include <type_traits>

#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/autodiff.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/outline.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/thread_group.h>
#include <luisa/xir/module.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/verifier.h>

#include "instruction_semantics.h"
#include "verifier_dom_tree.h"

namespace luisa::compute::xir {

namespace detail {

using BlockSet = VerifierBlockSet;
using BlockAdjacency = VerifierBlockAdjacency;

[[nodiscard]] bool typed_value_operand_valid(const Value *value) noexcept {
    return value != nullptr && value->type() != nullptr &&
           !value->isa<BasicBlock>() && !value->isa<Function>() &&
           !value->type()->is_resource();
}

[[nodiscard]] bool rvalue_operand_valid(const Value *value) noexcept {
    return typed_value_operand_valid(value) && !value->is_lvalue();
}

[[nodiscard]] bool data_operand_valid(const Value *value) noexcept {
    return rvalue_operand_valid(value) && !value->type()->is_custom();
}

[[nodiscard]] bool argument_matches(
    const Argument *formal, const Value *actual) noexcept {
    if (formal == nullptr || actual == nullptr ||
        actual->type() != formal->type()) {
        return false;
    }
    if (formal->is_resource()) {
        return actual->isa<ResourceArgument>() && !actual->is_lvalue();
    }
    if (formal->is_reference()) {
        return typed_value_operand_valid(actual) && actual->is_lvalue();
    }
    return rvalue_operand_valid(actual);
}

[[nodiscard]] bool argument_kind_matches_type(
    const Argument *argument) noexcept {
    if (argument == nullptr || argument->type() == nullptr) { return false; }
    if (argument->is_value()) {
        return !argument->type()->is_resource() &&
               !argument->type()->is_custom();
    }
    if (argument->is_reference()) {
        return !argument->type()->is_resource();
    }
    return argument->is_resource() && argument->type()->is_resource();
}

[[nodiscard]] bool ray_query_type(const Type *type) noexcept {
    return type == Type::custom("LC_RayQueryAll") ||
           type == Type::custom("LC_RayQueryAny");
}

[[nodiscard]] bool ray_query_object_valid(const Value *value) noexcept {
    return typed_value_operand_valid(value) && value->is_lvalue() &&
           ray_query_type(value->type());
}

template<typename IndexAt>
[[nodiscard]] const Type *aggregate_indexed_type(
    const Type *base_type, size_t index_count, IndexAt &&index_at) noexcept {
    auto current = base_type;
    for (auto i = 0u; i < index_count; i++) {
        auto index = index_at(i);
        if (!data_operand_valid(index) ||
            (!index->type()->is_int() && !index->type()->is_uint()) ||
            current == nullptr) {
            return nullptr;
        }
        switch (current->tag()) {
            case Type::Tag::ARRAY:
            case Type::Tag::VECTOR: {
                uint64_t constant_index = 0u;
                if (index->template isa<Constant>() &&
                    (!try_decode_constant_nonnegative_integer(index, constant_index) ||
                     constant_index >= current->dimension())) {
                    return nullptr;
                }
                current = current->element();
                break;
            }
            case Type::Tag::MATRIX: {
                uint64_t constant_index = 0u;
                if (index->template isa<Constant>() &&
                    (!try_decode_constant_nonnegative_integer(index, constant_index) ||
                     constant_index >= current->dimension())) {
                    return nullptr;
                }
                current = Type::vector(current->element(), current->dimension());
                break;
            }
            case Type::Tag::STRUCTURE: {
                uint64_t member_index = 0u;
                if (!try_decode_constant_nonnegative_integer(index, member_index) ||
                    member_index >= current->members().size()) {
                    return nullptr;
                }
                current = current->members()[member_index];
                break;
            }
            case Type::Tag::COOPERATIVE_VECTOR: {
                current = current->element();
                break;
            }
            default: return nullptr;
        }
    }
    return current;
}

[[nodiscard]] const Type *gep_indexed_type(const GEPInst *gep) noexcept {
    return aggregate_indexed_type(
        gep->base() == nullptr ? nullptr : gep->base()->type(),
        gep->index_count(),
        [gep](size_t i) noexcept { return gep->index(i); });
}

[[nodiscard]] size_t logical_register_width(const Type *type) noexcept {
    if (type == nullptr || !type->is_scalar_or_vector()) { return 0u; }
    if (type->is_vector()) {
        return type->element()->size() * type->dimension();
    }
    return type->size();
}

[[nodiscard]] bool cast_types_valid(const CastInst *cast) noexcept {
    if (cast->type() == nullptr || cast->type()->is_resource() ||
        cast->type()->is_custom() || !data_operand_valid(cast->value())) {
        return false;
    }
    auto source = cast->value()->type();
    auto target = cast->type();
    switch (cast->op()) {
        case CastOp::STATIC_CAST:
            return source->is_scalar_or_vector() && target->is_scalar_or_vector() &&
                   source->dimension() == target->dimension();
        case CastOp::BITWISE_CAST:
            return source->is_scalar_or_vector() && target->is_scalar_or_vector() &&
                   !source->is_bool_or_bool_vector() &&
                   !target->is_bool_or_bool_vector() &&
                   logical_register_width(source) == logical_register_width(target);
    }
    return false;
}

[[nodiscard]] bool scalar_or_vector_integer(const Type *type) noexcept {
    return type != nullptr &&
           (type->is_int_or_int_vector() || type->is_uint_or_uint_vector());
}

[[nodiscard]] bool scalar_or_vector_uint32(const Type *type) noexcept {
    return type != nullptr &&
           (type->is_uint32() ||
            (type->is_vector() && type->element()->is_uint32()));
}

[[nodiscard]] bool scalar_or_vector_numeric(const Type *type) noexcept {
    return scalar_or_vector_integer(type) ||
           (type != nullptr && type->is_float_or_float_vector());
}

[[nodiscard]] bool scalar_or_vector_bitwise(const Type *type) noexcept {
    return scalar_or_vector_integer(type) ||
           (type != nullptr && type->is_bool_or_bool_vector());
}

[[nodiscard]] bool same_scalar_or_vector_shape(
    const Type *lhs, const Type *rhs) noexcept {
    return lhs != nullptr && rhs != nullptr && lhs->is_scalar_or_vector() &&
           rhs->is_scalar_or_vector() && lhs->dimension() == rhs->dimension();
}

[[nodiscard]] bool boolean_shape_for(
    const Type *boolean_type, const Type *value_type) noexcept {
    if (boolean_type == nullptr || value_type == nullptr) { return false; }
    if (value_type->is_scalar()) { return boolean_type->is_bool(); }
    return value_type->is_vector() &&
           boolean_type == Type::vector(Type::of<bool>(), value_type->dimension());
}

[[nodiscard]] bool arithmetic_operand_count_valid(
    ArithmeticOp op, size_t count) noexcept {
    switch (op) {
        case ArithmeticOp::UNARY_MINUS:
        case ArithmeticOp::UNARY_BIT_NOT:
        case ArithmeticOp::ALL:
        case ArithmeticOp::ANY:
        case ArithmeticOp::SATURATE:
        case ArithmeticOp::ABS:
        case ArithmeticOp::CLZ:
        case ArithmeticOp::CTZ:
        case ArithmeticOp::POPCOUNT:
        case ArithmeticOp::REVERSE:
        case ArithmeticOp::ISINF:
        case ArithmeticOp::ISNAN:
        case ArithmeticOp::ACOS:
        case ArithmeticOp::ACOSH:
        case ArithmeticOp::ASIN:
        case ArithmeticOp::ASINH:
        case ArithmeticOp::ATAN:
        case ArithmeticOp::ATANH:
        case ArithmeticOp::COS:
        case ArithmeticOp::COSH:
        case ArithmeticOp::SIN:
        case ArithmeticOp::SINH:
        case ArithmeticOp::TAN:
        case ArithmeticOp::TANH:
        case ArithmeticOp::EXP:
        case ArithmeticOp::EXP2:
        case ArithmeticOp::EXP10:
        case ArithmeticOp::LOG:
        case ArithmeticOp::LOG2:
        case ArithmeticOp::LOG10:
        case ArithmeticOp::SQRT:
        case ArithmeticOp::RSQRT:
        case ArithmeticOp::CEIL:
        case ArithmeticOp::FLOOR:
        case ArithmeticOp::FRACT:
        case ArithmeticOp::TRUNC:
        case ArithmeticOp::ROUND:
        case ArithmeticOp::RINT:
        case ArithmeticOp::LENGTH:
        case ArithmeticOp::LENGTH_SQUARED:
        case ArithmeticOp::NORMALIZE:
        case ArithmeticOp::REDUCE_SUM:
        case ArithmeticOp::REDUCE_PRODUCT:
        case ArithmeticOp::REDUCE_MIN:
        case ArithmeticOp::REDUCE_MAX:
        case ArithmeticOp::MATRIX_COMP_NEG:
        case ArithmeticOp::MATRIX_DETERMINANT:
        case ArithmeticOp::MATRIX_TRANSPOSE:
        case ArithmeticOp::MATRIX_INVERSE: return count == 1u;

        case ArithmeticOp::BINARY_ADD:
        case ArithmeticOp::BINARY_SUB:
        case ArithmeticOp::BINARY_MUL:
        case ArithmeticOp::BINARY_DIV:
        case ArithmeticOp::BINARY_MOD:
        case ArithmeticOp::BINARY_BIT_AND:
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
        case ArithmeticOp::BINARY_SHIFT_LEFT:
        case ArithmeticOp::BINARY_SHIFT_RIGHT:
        case ArithmeticOp::BINARY_ROTATE_LEFT:
        case ArithmeticOp::BINARY_ROTATE_RIGHT:
        case ArithmeticOp::BINARY_LESS:
        case ArithmeticOp::BINARY_GREATER:
        case ArithmeticOp::BINARY_LESS_EQUAL:
        case ArithmeticOp::BINARY_GREATER_EQUAL:
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL:
        case ArithmeticOp::STEP:
        case ArithmeticOp::MIN:
        case ArithmeticOp::MAX:
        case ArithmeticOp::ATAN2:
        case ArithmeticOp::POW:
        case ArithmeticOp::POW_INT:
        case ArithmeticOp::COPYSIGN:
        case ArithmeticOp::CROSS:
        case ArithmeticOp::DOT:
        case ArithmeticOp::REFLECT:
        case ArithmeticOp::OUTER_PRODUCT:
        case ArithmeticOp::MATRIX_COMP_ADD:
        case ArithmeticOp::MATRIX_COMP_SUB:
        case ArithmeticOp::MATRIX_COMP_MUL:
        case ArithmeticOp::MATRIX_COMP_DIV:
        case ArithmeticOp::MATRIX_LINALG_MUL: return count == 2u;

        case ArithmeticOp::SELECT:
        case ArithmeticOp::CLAMP:
        case ArithmeticOp::LERP:
        case ArithmeticOp::SMOOTHSTEP:
        case ArithmeticOp::FMA:
        case ArithmeticOp::FACEFORWARD: return count == 3u;

        case ArithmeticOp::AGGREGATE: return count > 0u;
        case ArithmeticOp::SHUFFLE:
        case ArithmeticOp::EXTRACT: return count >= 2u;
        case ArithmeticOp::INSERT: return count >= 3u;
    }
    return false;
}

[[nodiscard]] bool resource_query_operand_count_valid(
    ResourceQueryOp op, size_t count) noexcept {
    switch (op) {
        case ResourceQueryOp::BUFFER_SIZE:
        case ResourceQueryOp::BYTE_BUFFER_SIZE:
        case ResourceQueryOp::TEXTURE2D_SIZE:
        case ResourceQueryOp::TEXTURE3D_SIZE:
        case ResourceQueryOp::BUFFER_DEVICE_ADDRESS: return count == 1u;
        case ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE:
        case ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: return count == 2u;
        case ResourceQueryOp::BINDLESS_BUFFER_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE:
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY:
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL:
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT: return count == 3u;
        case ResourceQueryOp::TEXTURE2D_SAMPLE:
        case ResourceQueryOp::TEXTURE3D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: return count == 4u;
        case ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: return count == 5u;
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: return count == 6u;
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: return count == 7u;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: return count == 8u;
    }
    return false;
}

[[nodiscard]] bool resource_read_operand_count_valid(
    ResourceReadOp op, size_t count) noexcept {
    switch (op) {
        case ResourceReadOp::BUFFER_READ:
        case ResourceReadOp::BUFFER_VOLATILE_READ:
        case ResourceReadOp::BYTE_BUFFER_READ:
        case ResourceReadOp::BYTE_BUFFER_VOLATILE_READ:
        case ResourceReadOp::TEXTURE2D_READ:
        case ResourceReadOp::TEXTURE3D_READ: return count == 2u;
        case ResourceReadOp::BINDLESS_BUFFER_READ:
        case ResourceReadOp::BINDLESS_BYTE_BUFFER_READ:
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ:
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ: return count == 3u;
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL:
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: return count == 4u;
        case ResourceReadOp::DEVICE_ADDRESS_READ:
        case ResourceReadOp::COOPERATIVE_VECTOR_SPLAT:
        case ResourceReadOp::COOPERATIVE_VECTOR_CAST: return count == 1u;
        case ResourceReadOp::COOPERATIVE_VECTOR_LOAD:
        case ResourceReadOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD: return count == 2u;
        case ResourceReadOp::BINDLESS_COOPERATIVE_VECTOR_LOAD: return count == 3u;
        case ResourceReadOp::COOPERATIVE_MUL: return count == 4u;
        case ResourceReadOp::BINDLESS_COOPERATIVE_MUL: return count == 5u;
        case ResourceReadOp::COOPERATIVE_MUL_ADD: return count == 7u;
        case ResourceReadOp::BINDLESS_COOPERATIVE_MUL_ADD: return count == 8u;
        // future cooperative-vector element-wise operations
        case ResourceReadOp::COOPERATIVE_VECTOR_DOT:
        case ResourceReadOp::COOPERATIVE_VECTOR_POW:
        case ResourceReadOp::COOPERATIVE_VECTOR_STEP:
        case ResourceReadOp::COOPERATIVE_VECTOR_ADD:
        case ResourceReadOp::COOPERATIVE_VECTOR_SUB:
        case ResourceReadOp::COOPERATIVE_VECTOR_MUL:
        case ResourceReadOp::COOPERATIVE_VECTOR_DIV:
        case ResourceReadOp::COOPERATIVE_VECTOR_LESS:
        case ResourceReadOp::COOPERATIVE_VECTOR_LESS_EQUAL:
        case ResourceReadOp::COOPERATIVE_VECTOR_GREATER:
        case ResourceReadOp::COOPERATIVE_VECTOR_GREATER_EQUAL:
        case ResourceReadOp::COOPERATIVE_VECTOR_EQUAL:
        case ResourceReadOp::COOPERATIVE_VECTOR_NOT_EQUAL: return count == 2u;
        case ResourceReadOp::COOPERATIVE_VECTOR_MIX:
        case ResourceReadOp::COOPERATIVE_VECTOR_LERP:
        case ResourceReadOp::COOPERATIVE_VECTOR_SMOOTHSTEP: return count == 3u;
        case ResourceReadOp::COOPERATIVE_VECTOR_ABS:
        case ResourceReadOp::COOPERATIVE_VECTOR_SIGN:
        case ResourceReadOp::COOPERATIVE_VECTOR_FLOOR:
        case ResourceReadOp::COOPERATIVE_VECTOR_CEIL:
        case ResourceReadOp::COOPERATIVE_VECTOR_FRACT:
        case ResourceReadOp::COOPERATIVE_VECTOR_TRUNC:
        case ResourceReadOp::COOPERATIVE_VECTOR_ROUND:
        case ResourceReadOp::COOPERATIVE_VECTOR_RINT:
        case ResourceReadOp::COOPERATIVE_VECTOR_SQRT:
        case ResourceReadOp::COOPERATIVE_VECTOR_RSQRT:
        case ResourceReadOp::COOPERATIVE_VECTOR_EXP2:
        case ResourceReadOp::COOPERATIVE_VECTOR_EXP10:
        case ResourceReadOp::COOPERATIVE_VECTOR_LOG2:
        case ResourceReadOp::COOPERATIVE_VECTOR_LOG10:
        case ResourceReadOp::COOPERATIVE_VECTOR_SATURATE:
        case ResourceReadOp::COOPERATIVE_VECTOR_ISINF:
        case ResourceReadOp::COOPERATIVE_VECTOR_ISNAN:
        case ResourceReadOp::COOPERATIVE_VECTOR_SIN:
        case ResourceReadOp::COOPERATIVE_VECTOR_COS:
        case ResourceReadOp::COOPERATIVE_VECTOR_TAN:
        case ResourceReadOp::COOPERATIVE_VECTOR_ASIN:
        case ResourceReadOp::COOPERATIVE_VECTOR_ACOS:
        case ResourceReadOp::COOPERATIVE_VECTOR_SINH:
        case ResourceReadOp::COOPERATIVE_VECTOR_COSH:
        case ResourceReadOp::COOPERATIVE_VECTOR_ASINH:
        case ResourceReadOp::COOPERATIVE_VECTOR_ACOSH:
        case ResourceReadOp::COOPERATIVE_VECTOR_ATANH: return count == 1u;
    }
    return false;
}

[[nodiscard]] bool resource_write_operand_count_valid(
    ResourceWriteOp op, size_t count) noexcept {
    switch (op) {
        case ResourceWriteOp::BUFFER_WRITE:
        case ResourceWriteOp::BUFFER_VOLATILE_WRITE:
        case ResourceWriteOp::BYTE_BUFFER_WRITE:
        case ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE:
        case ResourceWriteOp::TEXTURE2D_WRITE:
        case ResourceWriteOp::TEXTURE3D_WRITE:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: return count == 3u;
        case ResourceWriteOp::BINDLESS_BUFFER_WRITE:
        case ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: return count == 4u;
        case ResourceWriteOp::DEVICE_ADDRESS_WRITE:
        case ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: return count == 2u;
        case ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL: return count == 5u;
        case ResourceWriteOp::COOPERATIVE_VECTOR_ACCUMULATE:
        case ResourceWriteOp::COOPERATIVE_VECTOR_STORE:
        case ResourceWriteOp::COOPERATIVE_VECTOR_WORKGROUP_STORE: return count == 3u;
        case ResourceWriteOp::BINDLESS_COOPERATIVE_VECTOR_STORE: return count == 4u;
        case ResourceWriteOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE: return count == 5u;
    }
    return false;
}

[[nodiscard]] bool thread_group_operand_count_valid(
    ThreadGroupOp op, size_t count) noexcept {
    switch (op) {
        case ThreadGroupOp::SHADER_EXECUTION_REORDER:
            return count == 0u || count == 2u;
        case ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE:
        case ThreadGroupOp::WARP_FIRST_ACTIVE_LANE:
        case ThreadGroupOp::SYNCHRONIZE_BLOCK: return count == 0u;
        case ThreadGroupOp::WARP_READ_LANE: return count == 2u;
        case ThreadGroupOp::RASTER_QUAD_DDX:
        case ThreadGroupOp::RASTER_QUAD_DDY:
        case ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL:
        case ThreadGroupOp::WARP_ACTIVE_BIT_AND:
        case ThreadGroupOp::WARP_ACTIVE_BIT_OR:
        case ThreadGroupOp::WARP_ACTIVE_BIT_XOR:
        case ThreadGroupOp::WARP_ACTIVE_COUNT_BITS:
        case ThreadGroupOp::WARP_ACTIVE_MAX:
        case ThreadGroupOp::WARP_ACTIVE_MIN:
        case ThreadGroupOp::WARP_ACTIVE_PRODUCT:
        case ThreadGroupOp::WARP_ACTIVE_SUM:
        case ThreadGroupOp::WARP_ACTIVE_ALL:
        case ThreadGroupOp::WARP_ACTIVE_ANY:
        case ThreadGroupOp::WARP_ACTIVE_BIT_MASK:
        case ThreadGroupOp::WARP_PREFIX_COUNT_BITS:
        case ThreadGroupOp::WARP_PREFIX_SUM:
        case ThreadGroupOp::WARP_PREFIX_PRODUCT:
        case ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: return count == 1u;
    }
    return false;
}

[[nodiscard]] bool ray_query_object_write_operand_count_valid(
    RayQueryObjectWriteOp op, size_t count) noexcept {
    return count ==
           (op == RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL ?
                2u :
                1u);
}

[[nodiscard]] bool autodiff_intrinsic_operand_count_valid(
    AutodiffIntrinsicOp op, size_t count) noexcept {
    switch (op) {
        case AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT:
        case AutodiffIntrinsicOp::AUTODIFF_GRADIENT:
        case AutodiffIntrinsicOp::AUTODIFF_DETACH: return count == 1u;
        case AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER:
        case AutodiffIntrinsicOp::AUTODIFF_ACCUMULATE_GRADIENT:
        case AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT: return count == 2u;
        case AutodiffIntrinsicOp::AUTODIFF_BACKWARD: return count == 0u;
        case AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT: return count >= 2u;
    }
    return false;
}

template<typename Enum>
[[nodiscard]] bool enum_value_between(
    Enum value, Enum first, Enum last) noexcept {
    using Underlying = std::underlying_type_t<Enum>;
    auto underlying = static_cast<Underlying>(value);
    return underlying >= static_cast<Underlying>(first) &&
           underlying <= static_cast<Underlying>(last);
}

[[nodiscard]] bool instruction_opcode_valid(
    const Instruction *instruction,
    DerivedInstructionTag tag) noexcept {
    switch (tag) {
        case DerivedInstructionTag::ALLOCA:
            return enum_value_between(
                static_cast<const AllocaInst *>(instruction)->op(),
                AllocaOp::LOCAL, AllocaOp::SHARED);
        case DerivedInstructionTag::ATOMIC:
            return enum_value_between(
                static_cast<const AtomicInst *>(instruction)->op(),
                AtomicOp::EXCHANGE, AtomicOp::FETCH_MAX);
        case DerivedInstructionTag::ARITHMETIC:
            return enum_value_between(
                static_cast<const ArithmeticInst *>(instruction)->op(),
                ArithmeticOp::UNARY_MINUS, ArithmeticOp::EXTRACT);
        case DerivedInstructionTag::THREAD_GROUP:
            return enum_value_between(
                static_cast<const ThreadGroupInst *>(instruction)->op(),
                ThreadGroupOp::SHADER_EXECUTION_REORDER,
                ThreadGroupOp::SYNCHRONIZE_BLOCK);
        case DerivedInstructionTag::RESOURCE_QUERY:
            return enum_value_between(
                static_cast<const ResourceQueryInst *>(instruction)->op(),
                ResourceQueryOp::BUFFER_SIZE,
                ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR);
 case DerivedInstructionTag::RESOURCE_READ:
 return enum_value_between(
 static_cast<const ResourceReadInst *>(instruction)->op(),
 ResourceReadOp::BUFFER_READ,
 ResourceReadOp::COOPERATIVE_VECTOR_NOT_EQUAL);
        case DerivedInstructionTag::RESOURCE_WRITE:
            return enum_value_between(
                static_cast<const ResourceWriteInst *>(instruction)->op(),
                ResourceWriteOp::BUFFER_WRITE,
                ResourceWriteOp::COOPERATIVE_VECTOR_WORKGROUP_STORE);
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
            return enum_value_between(
                static_cast<const RayQueryObjectReadInst *>(instruction)->op(),
                RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY,
                RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED);
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            return enum_value_between(
                static_cast<const RayQueryObjectWriteInst *>(instruction)->op(),
                RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE,
                RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED);
        case DerivedInstructionTag::AUTODIFF_INTRINSIC:
            return enum_value_between(
                static_cast<const AutodiffIntrinsicInst *>(instruction)->op(),
                AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT,
                AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT);
        case DerivedInstructionTag::CAST:
            return enum_value_between(
                static_cast<const CastInst *>(instruction)->op(),
                CastOp::STATIC_CAST, CastOp::BITWISE_CAST);
        default: return true;
    }
}

[[nodiscard]] bool instruction_operand_shape_valid(
    const Instruction *instruction,
    DerivedInstructionTag tag) noexcept {
    auto count = instruction->operand_count();
    switch (tag) {
        case DerivedInstructionTag::IF:
        case DerivedInstructionTag::CONDITIONAL_BRANCH: return count == 3u;
        case DerivedInstructionTag::SWITCH:
        case DerivedInstructionTag::INDEXED_BRANCH:
            return count >= 2u &&
                   count ==
                       static_cast<
                           const IndexedBranchTerminatorInstruction *>(
                           instruction)
                               ->case_count() +
                           IndexedBranchTerminatorInstruction::
                               operand_index_case_block_offset;
        case DerivedInstructionTag::LOOP:
        case DerivedInstructionTag::SIMPLE_LOOP:
        case DerivedInstructionTag::BRANCH:
        case DerivedInstructionTag::CORO_RESUME:
        case DerivedInstructionTag::RETURN:
        case DerivedInstructionTag::LOAD:
        case DerivedInstructionTag::RAY_QUERY_LOOP:
        case DerivedInstructionTag::AUTODIFF_SCOPE:
        case DerivedInstructionTag::BREAK:
        case DerivedInstructionTag::CONTINUE:
        case DerivedInstructionTag::CAST:
        case DerivedInstructionTag::ASSERT:
        case DerivedInstructionTag::ASSUME:
        case DerivedInstructionTag::OUTLINE: return count == 1u;
        case DerivedInstructionTag::CORO_SUSPEND: {
            auto *suspend =
                static_cast<const CoroSuspendInst *>(instruction);
            return count ==
                   CoroSuspendInst::operand_index_frame_export_offset +
                       suspend->frame_export_count();
        }
        case DerivedInstructionTag::UNREACHABLE:
        case DerivedInstructionTag::RASTER_DISCARD:
        case DerivedInstructionTag::CORO_TERMINATE:
        case DerivedInstructionTag::ALLOCA:
        case DerivedInstructionTag::CLOCK: return count == 0u;
        case DerivedInstructionTag::PHI:
            return count == static_cast<const PhiInst *>(instruction)->incoming_count();
        case DerivedInstructionTag::STORE: return count == 2u;
        case DerivedInstructionTag::GEP: return count >= 2u;
        case DerivedInstructionTag::ATOMIC: {
            auto inst = static_cast<const AtomicInst *>(instruction);
            return count >= 1u + atomic_op_value_count(inst->op());
        }
        case DerivedInstructionTag::ARITHMETIC: {
            auto inst = static_cast<const ArithmeticInst *>(instruction);
            return arithmetic_operand_count_valid(inst->op(), count);
        }
        case DerivedInstructionTag::THREAD_GROUP: {
            auto inst = static_cast<const ThreadGroupInst *>(instruction);
            return thread_group_operand_count_valid(inst->op(), count);
        }
        case DerivedInstructionTag::RESOURCE_QUERY: {
            auto inst = static_cast<const ResourceQueryInst *>(instruction);
            return resource_query_operand_count_valid(inst->op(), count);
        }
        case DerivedInstructionTag::RESOURCE_READ: {
            auto inst = static_cast<const ResourceReadInst *>(instruction);
            return resource_read_operand_count_valid(inst->op(), count);
        }
        case DerivedInstructionTag::RESOURCE_WRITE: {
            auto inst = static_cast<const ResourceWriteInst *>(instruction);
            return resource_write_operand_count_valid(inst->op(), count);
        }
        case DerivedInstructionTag::RAY_QUERY_DISPATCH: return count == 4u;
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: return count == 1u;
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: {
            auto inst = static_cast<const RayQueryObjectWriteInst *>(instruction);
            return ray_query_object_write_operand_count_valid(inst->op(), count);
        }
        case DerivedInstructionTag::RAY_QUERY_PIPELINE: return count >= 3u;
        case DerivedInstructionTag::AUTODIFF_INTRINSIC: {
            auto inst = static_cast<const AutodiffIntrinsicInst *>(instruction);
            return autodiff_intrinsic_operand_count_valid(inst->op(), count);
        }
        case DerivedInstructionTag::CALL: return count >= 1u;
        case DerivedInstructionTag::PRINT:
        case DerivedInstructionTag::DEBUG_BREAK: return true;
    }
    return false;
}

[[nodiscard]] int64_t instruction_opcode(
    const Instruction *instruction,
    DerivedInstructionTag tag) noexcept {
    switch (tag) {
        case DerivedInstructionTag::ALLOCA:
            return static_cast<int64_t>(
                static_cast<const AllocaInst *>(instruction)->op());
        case DerivedInstructionTag::ATOMIC:
            return static_cast<int64_t>(
                static_cast<const AtomicInst *>(instruction)->op());
        case DerivedInstructionTag::ARITHMETIC:
            return static_cast<int64_t>(
                static_cast<const ArithmeticInst *>(instruction)->op());
        case DerivedInstructionTag::THREAD_GROUP:
            return static_cast<int64_t>(
                static_cast<const ThreadGroupInst *>(instruction)->op());
        case DerivedInstructionTag::RESOURCE_QUERY:
            return static_cast<int64_t>(
                static_cast<const ResourceQueryInst *>(instruction)->op());
        case DerivedInstructionTag::RESOURCE_READ:
            return static_cast<int64_t>(
                static_cast<const ResourceReadInst *>(instruction)->op());
        case DerivedInstructionTag::RESOURCE_WRITE:
            return static_cast<int64_t>(
                static_cast<const ResourceWriteInst *>(instruction)->op());
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
            return static_cast<int64_t>(
                static_cast<const RayQueryObjectReadInst *>(instruction)->op());
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            return static_cast<int64_t>(
                static_cast<const RayQueryObjectWriteInst *>(instruction)->op());
        case DerivedInstructionTag::AUTODIFF_INTRINSIC:
            return static_cast<int64_t>(
                static_cast<const AutodiffIntrinsicInst *>(instruction)->op());
        case DerivedInstructionTag::CAST:
            return static_cast<int64_t>(
                static_cast<const CastInst *>(instruction)->op());
        default: return -1;
    }
}

[[nodiscard]] bool instruction_semantics_valid(
    const Instruction *instruction,
    DerivedInstructionTag tag,
    luisa::vector<const Value *> &operands) noexcept {
    operands.clear();
    operands.reserve(instruction->operand_count());
    for (auto *operand_use : instruction->operand_uses()) {
        operands.emplace_back(operand_use->value());
    }
    if (tag == DerivedInstructionTag::CORO_SUSPEND) {
        auto *suspend =
            static_cast<const CoroSuspendInst *>(instruction);
        luisa::unordered_set<luisa::string_view> names;
        if (suspend->frame_export_count() +
                CoroSuspendInst::operand_index_frame_export_offset !=
            operands.size()) {
            return false;
        }
        for (size_t i = 0u;
             i < suspend->frame_export_count(); ++i) {
            auto &name = suspend->frame_export_name(i);
            auto *value = suspend->frame_export_value(i);
            if (name.empty() || !names.emplace(name).second ||
                !data_operand_valid(value) ||
                !value->type()->is_basic()) {
                return false;
            }
        }
    }
    auto bindless_access = [&]() noexcept {
        switch (tag) {
            case DerivedInstructionTag::RESOURCE_QUERY:
                return static_cast<const ResourceQueryInst *>(instruction)
                    ->bindless_access();
            case DerivedInstructionTag::RESOURCE_READ:
                return static_cast<const ResourceReadInst *>(instruction)
                    ->bindless_access();
            case DerivedInstructionTag::RESOURCE_WRITE:
                return static_cast<const ResourceWriteInst *>(instruction)
                    ->bindless_access();
            default: return BindlessResourceAccess{};
        }
    }();
    return interchange_instruction_semantics_valid(
        tag, instruction_opcode(instruction, tag),
        instruction->type(), operands, bindless_access);
}

template<typename OperandSpan>
[[nodiscard]] bool arithmetic_operand_types_valid(
    ArithmeticOp op, const Type *result, const OperandSpan &operands) noexcept {
    if (result == nullptr || result->is_resource() || result->is_custom()) { return false; }
    for (auto operand : operands) {
        if (!data_operand_valid(operand)) { return false; }
    }
    auto all_are = [&](const Type *type) noexcept {
        return std::all_of(operands.begin(), operands.end(),
                           [type](auto operand) noexcept { return operand->type() == type; });
    };
    auto same_or_element = [](const Type *candidate, const Type *type) noexcept {
        return candidate == type ||
               (type != nullptr && type->is_vector() && candidate == type->element());
    };
    switch (op) {
        case ArithmeticOp::UNARY_MINUS:
            return operands[0]->type() == result && scalar_or_vector_numeric(result);
        case ArithmeticOp::UNARY_BIT_NOT:
            return operands[0]->type() == result && scalar_or_vector_bitwise(result);

        case ArithmeticOp::BINARY_ADD:
        case ArithmeticOp::BINARY_SUB:
        case ArithmeticOp::BINARY_MUL:
        case ArithmeticOp::BINARY_DIV:
        case ArithmeticOp::BINARY_MOD:
            return all_are(result) && scalar_or_vector_numeric(result);
        case ArithmeticOp::BINARY_BIT_AND:
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
            return all_are(result) && scalar_or_vector_bitwise(result);
        case ArithmeticOp::BINARY_SHIFT_LEFT:
        case ArithmeticOp::BINARY_SHIFT_RIGHT:
        case ArithmeticOp::BINARY_ROTATE_LEFT:
        case ArithmeticOp::BINARY_ROTATE_RIGHT:
            return operands[0]->type() == result && scalar_or_vector_integer(result) &&
                   scalar_or_vector_integer(operands[1]->type()) &&
                   same_scalar_or_vector_shape(result, operands[1]->type());

        case ArithmeticOp::BINARY_LESS:
        case ArithmeticOp::BINARY_GREATER:
        case ArithmeticOp::BINARY_LESS_EQUAL:
        case ArithmeticOp::BINARY_GREATER_EQUAL:
            return operands[0]->type() == operands[1]->type() &&
                   scalar_or_vector_numeric(operands[0]->type()) &&
                   boolean_shape_for(result, operands[0]->type());
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL:
            return operands[0]->type() == operands[1]->type() &&
                   (scalar_or_vector_numeric(operands[0]->type()) ||
                    operands[0]->type()->is_bool_or_bool_vector()) &&
                   boolean_shape_for(result, operands[0]->type());

        case ArithmeticOp::ALL:
        case ArithmeticOp::ANY:
            return result->is_bool() && operands[0]->type()->is_bool_vector();
        case ArithmeticOp::SELECT: {
            auto condition = operands[2]->type();
            auto condition_valid = condition->is_bool() ||
                                   (result->is_vector() &&
                                    condition == Type::vector(Type::of<bool>(), result->dimension()));
            return operands[0]->type() == result && operands[1]->type() == result &&
                   condition_valid;
        }
        case ArithmeticOp::CLAMP:
            return all_are(result) && scalar_or_vector_numeric(result);
        case ArithmeticOp::SATURATE:
            return operands[0]->type() == result && result->is_float_or_float_vector();
        case ArithmeticOp::LERP:
            return operands[0]->type() == result && operands[1]->type() == result &&
                   result->is_float_or_float_vector() &&
                   same_or_element(operands[2]->type(), result);
        case ArithmeticOp::SMOOTHSTEP:
            return operands[2]->type() == result && result->is_float_or_float_vector() &&
                   same_or_element(operands[0]->type(), result) &&
                   same_or_element(operands[1]->type(), result);
        case ArithmeticOp::STEP:
            return operands[1]->type() == result && result->is_float_or_float_vector() &&
                   same_or_element(operands[0]->type(), result);

        case ArithmeticOp::ABS:
            return operands[0]->type() == result && scalar_or_vector_numeric(result);
        case ArithmeticOp::MIN:
        case ArithmeticOp::MAX:
            return all_are(result) && scalar_or_vector_numeric(result);
        case ArithmeticOp::CLZ:
        case ArithmeticOp::CTZ:
        case ArithmeticOp::POPCOUNT:
        case ArithmeticOp::REVERSE:
            return operands[0]->type() == result &&
                   scalar_or_vector_uint32(result);
        case ArithmeticOp::ISINF:
        case ArithmeticOp::ISNAN:
            return operands[0]->type()->is_float_or_float_vector() &&
                   boolean_shape_for(result, operands[0]->type());

        case ArithmeticOp::ACOS:
        case ArithmeticOp::ACOSH:
        case ArithmeticOp::ASIN:
        case ArithmeticOp::ASINH:
        case ArithmeticOp::ATAN:
        case ArithmeticOp::ATANH:
        case ArithmeticOp::COS:
        case ArithmeticOp::COSH:
        case ArithmeticOp::SIN:
        case ArithmeticOp::SINH:
        case ArithmeticOp::TAN:
        case ArithmeticOp::TANH:
        case ArithmeticOp::EXP:
        case ArithmeticOp::EXP2:
        case ArithmeticOp::EXP10:
        case ArithmeticOp::LOG:
        case ArithmeticOp::LOG2:
        case ArithmeticOp::LOG10:
        case ArithmeticOp::SQRT:
        case ArithmeticOp::RSQRT:
        case ArithmeticOp::CEIL:
        case ArithmeticOp::FLOOR:
        case ArithmeticOp::FRACT:
        case ArithmeticOp::TRUNC:
        case ArithmeticOp::ROUND:
        case ArithmeticOp::RINT:
            return operands[0]->type() == result && result->is_float_or_float_vector();
        case ArithmeticOp::ATAN2:
        case ArithmeticOp::POW:
        case ArithmeticOp::COPYSIGN:
            return all_are(result) && result->is_float_or_float_vector();
        case ArithmeticOp::POW_INT:
            return operands[0]->type() == result && result->is_float_or_float_vector() &&
                   scalar_or_vector_integer(operands[1]->type()) &&
                   (same_scalar_or_vector_shape(result, operands[1]->type()) ||
                    operands[1]->type()->is_scalar());
        case ArithmeticOp::FMA:
            return all_are(result) && result->is_float_or_float_vector();

        case ArithmeticOp::CROSS:
            return all_are(result) && result->is_float_vector() && result->dimension() == 3u;
        case ArithmeticOp::DOT:
            return operands[0]->type() == operands[1]->type() &&
                   operands[0]->type()->is_float_vector() &&
                   result == operands[0]->type()->element();
        case ArithmeticOp::LENGTH:
        case ArithmeticOp::LENGTH_SQUARED:
            return operands[0]->type()->is_float_vector() &&
                   result == operands[0]->type()->element();
        case ArithmeticOp::NORMALIZE:
            return operands[0]->type() == result && result->is_float_vector();
        case ArithmeticOp::FACEFORWARD:
            return all_are(result) && result->is_float_vector();
        case ArithmeticOp::REFLECT:
            return all_are(result) && result->is_float_vector();
        case ArithmeticOp::REDUCE_SUM:
        case ArithmeticOp::REDUCE_PRODUCT:
        case ArithmeticOp::REDUCE_MIN:
        case ArithmeticOp::REDUCE_MAX:
            return scalar_or_vector_numeric(operands[0]->type()) &&
                   operands[0]->type()->is_vector() &&
                   result == operands[0]->type()->element();

        case ArithmeticOp::OUTER_PRODUCT: {
            auto lhs = operands[0]->type();
            auto rhs = operands[1]->type();
            if (lhs->is_float_vector() && rhs == lhs) {
                return result == Type::matrix(lhs->dimension());
            }
            return result->is_matrix() && lhs == result && rhs == result;
        }
        case ArithmeticOp::MATRIX_COMP_NEG:
            return result->is_matrix() && operands[0]->type() == result;
        case ArithmeticOp::MATRIX_COMP_ADD:
        case ArithmeticOp::MATRIX_COMP_SUB:
        case ArithmeticOp::MATRIX_COMP_MUL:
        case ArithmeticOp::MATRIX_COMP_DIV: {
            if (!result->is_matrix()) { return false; }
            auto element = result->element();
            auto lhs = operands[0]->type();
            auto rhs = operands[1]->type();
            return (lhs == result || lhs == element) &&
                   (rhs == result || rhs == element) &&
                   (lhs == result || rhs == result);
        }
        case ArithmeticOp::MATRIX_LINALG_MUL: {
            auto lhs = operands[0]->type();
            auto rhs = operands[1]->type();
            if (lhs->is_matrix() && rhs->is_matrix()) {
                return lhs == rhs && result == lhs;
            }
            if (lhs->is_matrix() && rhs->is_float_vector() &&
                lhs->dimension() == rhs->dimension()) {
                return result == rhs;
            }
            return lhs->is_float_vector() && rhs->is_matrix() &&
                   lhs->dimension() == rhs->dimension() && result == lhs;
        }
        case ArithmeticOp::MATRIX_DETERMINANT:
            return operands[0]->type()->is_matrix() &&
                   result == operands[0]->type()->element();
        case ArithmeticOp::MATRIX_TRANSPOSE:
        case ArithmeticOp::MATRIX_INVERSE:
            return result->is_matrix() && operands[0]->type() == result;

        case ArithmeticOp::AGGREGATE:
            if (result->is_vector() || result->is_array() ||
                result->is_cooperative_vector()) {
                return operands.size() == result->dimension() && all_are(result->element());
            }
            if (result->is_matrix()) {
                return operands.size() == result->dimension() &&
                       all_are(Type::vector(result->element(), result->dimension()));
            }
            if (result->is_structure()) {
                if (operands.size() != result->members().size()) { return false; }
                for (auto i = 0u; i < operands.size(); i++) {
                    if (operands[i]->type() != result->members()[i]) { return false; }
                }
                return true;
            }
            return false;
        case ArithmeticOp::SHUFFLE:
            if (!result->is_vector() || !operands[0]->type()->is_vector() ||
                result->element() != operands[0]->type()->element() ||
                operands.size() != result->dimension() + 1u) {
                return false;
            }
            for (auto index : operands.subspan(1u)) {
                if (index->type() == nullptr ||
                    (!index->type()->is_int() && !index->type()->is_uint())) {
                    return false;
                }
                uint64_t constant_index = 0u;
                if (index->template isa<Constant>() &&
                    (!try_decode_constant_nonnegative_integer(index, constant_index) ||
                     constant_index >= operands[0]->type()->dimension())) {
                    return false;
                }
            }
            return true;
        case ArithmeticOp::EXTRACT:
            return aggregate_indexed_type(
                       operands[0]->type(), operands.size() - 1u,
                       [operands](size_t i) noexcept { return operands[i + 1u]; }) == result;
        case ArithmeticOp::INSERT:
            return operands[0]->type() == result &&
                   aggregate_indexed_type(
                       result, operands.size() - 2u,
                       [operands](size_t i) noexcept { return operands[i + 2u]; }) == operands[1]->type();
    }
    return false;
}

[[nodiscard]] bool arithmetic_types_valid(const ArithmeticInst *inst) noexcept {
    if (!arithmetic_operand_count_valid(inst->op(), inst->operand_count())) { return false; }
    luisa::vector<const Value *> operands;
    operands.reserve(inst->operand_count());
    for (auto operand_use : inst->operand_uses()) {
        operands.emplace_back(operand_use->value());
    }
    return arithmetic_operand_types_valid(
        inst->op(), inst->type(), luisa::span{operands});
}

class XIRVerifier {

private:
    const XIRVerificationOptions &_options;
    XIRVerificationResult &_result;

private:
    void _error(const Function *function, const BasicBlock *block,
                const Instruction *instruction, luisa::string message) noexcept {
        _result.errors.emplace_back(XIRVerificationError{
            .function = function,
            .block = block,
            .instruction = instruction,
            .message = std::move(message),
        });
    }

    [[nodiscard]] bool _use_list_contains(const Value *value,
                                          const Use *use) noexcept {
        // UseList maintains the physical intrusive-list owner independently
        // from Use::value(). Exact linkage is therefore one identity check,
        // with no scan and no probabilistic hash relation.
        ++_result.statistics.use_list_owner_checks;
        return value->use_list().contains(use);
    }

public:
    XIRVerifier(const XIRVerificationOptions &options,
                XIRVerificationResult &result) noexcept
        : _options{options}, _result{result} {}

    void verify(const Function *function) noexcept {
        if (function == nullptr) {
            _error(nullptr, nullptr, nullptr, "Function is null.");
            return;
        }
        auto *module = function->parent_module();
        if (function->isa<KernelFunction>()) {
            auto block_size =
                static_cast<const KernelFunction *>(function)->block_size();
            if (function->type() != nullptr ||
                !KernelFunction::is_valid_block_size(block_size)) {
                _error(function, nullptr, nullptr,
                       "Kernel return type or block size is invalid.");
            }
        }
        for (auto *argument : function->arguments()) {
            if (argument->parent_function() != function ||
                !argument_kind_matches_type(argument)) {
                _error(function, nullptr, nullptr,
                       "Function argument ownership or type is invalid.");
            }
        }
        auto *definition = function->definition();
        if (definition == nullptr) { return; }
        if (definition->body_block() == nullptr) {
            _error(function, nullptr, nullptr, "Function definition has no body block.");
            return;
        }

        luisa::vector<const BasicBlock *> blocks;
        BlockSet block_set;
        for (auto *block : definition->basic_blocks()) {
            blocks.emplace_back(block);
            block_set.emplace(block);
            if (block->parent_function() != function) {
                _error(function, block, nullptr, "Basic block has the wrong parent function.");
            }
        }
        if (!block_set.contains(definition->body_block())) {
            _error(function, definition->body_block(), nullptr,
                   "Function body block is not owned by the function.");
            return;
        }
        struct InstructionFacts {
            const BasicBlock *block;
            size_t order;
            DerivedInstructionTag tag;
            bool operand_shape_valid;
        };
        // Verification is read-only. Classify every instruction exactly once
        // and share the immutable facts between structural discovery and the
        // detailed validation pass instead of repeating virtual tag dispatch
        // and opcode/shape checks.
        DensePointerMap<const Instruction *, InstructionFacts>
            instruction_facts;
        BlockAdjacency successors;
        BlockAdjacency predecessors;
        DensePointerMap<const BasicBlock *, const Instruction *> merge_owners;
        luisa::vector<const Value *> semantics_operands;

        for (auto *block : blocks) {
            auto terminated = block->is_terminated();
            if (_options.require_terminated_blocks && !terminated) {
                _error(function, block, nullptr, "Basic block is not terminated.");
            }
            auto saw_terminator = false;
            auto saw_non_phi = false;
            size_t order = 0u;
            for (auto *instruction : block->instructions()) {
                ++_result.statistics.instruction_tag_queries;
                auto tag = instruction->derived_instruction_tag();
                auto opcode_valid =
                    instruction_opcode_valid(instruction, tag);
                auto operand_shape_valid =
                    opcode_valid &&
                    instruction_operand_shape_valid(instruction, tag);
                instruction_facts.emplace(
                    instruction,
                    InstructionFacts{
                        .block = block,
                        .order = order++,
                        .tag = tag,
                        .operand_shape_valid = operand_shape_valid});
                if (!opcode_valid) {
                    _error(
                        function, block, instruction,
                        luisa::format(
                            "Instruction opcode is invalid. Operation: '{}'.",
                            to_string(tag)));
                } else if (!operand_shape_valid) {
                    _error(
                        function, block, instruction,
                        luisa::format(
                            "Instruction operand count is invalid. Operation: '{}'.",
                            to_string(tag)));
                } else if (!instruction_semantics_valid(
                               instruction, tag,
                               semantics_operands)) {
                    _error(
                        function, block, instruction,
                        luisa::format(
                            "Instruction operands or result type are invalid. Operation: '{}'.",
                            to_string(tag)));
                }
                if (instruction->parent_block() != block) {
                    _error(function, block, instruction,
                           "Instruction has the wrong parent block.");
                }
                if (saw_terminator) {
                    _error(function, block, instruction,
                           "Instruction appears after a terminator.");
                }
                if (tag == DerivedInstructionTag::PHI) {
                    if (saw_non_phi) {
                        _error(function, block, instruction,
                               "PHI instruction does not precede non-PHI instructions.");
                    }
                } else {
                    saw_non_phi = true;
                }
                saw_terminator |= instruction->is_terminator();
            }
            if (!terminated) { continue; }
            auto *terminator = block->terminator();
            auto terminator_facts = instruction_facts.find(terminator);
            if (terminator_facts == instruction_facts.end()) {
                _error(function, block, terminator,
                       "Terminator was not classified by the verifier.");
                continue;
            }
            if (!terminator_facts->second.operand_shape_valid) {
                continue;
            }
            auto add_successor = [&](size_t operand_index) noexcept {
                auto *operand = terminator->operand(operand_index);
                if (operand != nullptr && operand->isa<BasicBlock>()) {
                    auto *target = static_cast<const BasicBlock *>(operand);
                    successors[block].emplace(target);
                    predecessors[target].emplace(block);
                }
            };
            switch (terminator_facts->second.tag) {
                case DerivedInstructionTag::IF:
                case DerivedInstructionTag::CONDITIONAL_BRANCH:
                    add_successor(ConditionalBranchTerminatorInstruction::operand_index_true_target);
                    add_successor(ConditionalBranchTerminatorInstruction::operand_index_false_target);
                    break;
                case DerivedInstructionTag::SWITCH:
                case DerivedInstructionTag::INDEXED_BRANCH:
                    for (auto i = IndexedBranchTerminatorInstruction::
                             operand_index_default_block;
                         i < terminator->operand_count(); i++) {
                        add_successor(i);
                    }
                    break;
                case DerivedInstructionTag::LOOP:
                case DerivedInstructionTag::SIMPLE_LOOP:
                case DerivedInstructionTag::BRANCH:
                case DerivedInstructionTag::BREAK:
                case DerivedInstructionTag::CONTINUE:
                case DerivedInstructionTag::RAY_QUERY_LOOP:
                case DerivedInstructionTag::AUTODIFF_SCOPE:
                case DerivedInstructionTag::OUTLINE: add_successor(0u); break;
                case DerivedInstructionTag::RAY_QUERY_DISPATCH:
                    add_successor(RayQueryDispatchInst::operand_index_exit_block);
                    add_successor(RayQueryDispatchInst::operand_index_on_surface_candidate_block);
                    add_successor(RayQueryDispatchInst::operand_index_on_procedural_candidate_block);
                    break;
                default: break;
            }
        }

        BlockSet reachable;
        luisa::vector<const BasicBlock *> worklist{definition->body_block()};
        while (!worklist.empty()) {
            auto *block = worklist.back();
            worklist.pop_back();
            if (!reachable.emplace(block).second) { continue; }
            if (auto iter = successors.find(block); iter != successors.end()) {
                for (auto *successor : iter->second) {
                    if (block_set.contains(successor)) { worklist.emplace_back(successor); }
                }
            }
        }
        if (_options.require_reachable_blocks) {
            for (auto *block : blocks) {
                if (!reachable.contains(block)) {
                    _error(function, block, nullptr, "Basic block is unreachable.");
                }
            }
        }

        VerifierSparseDomTree dominators{
            definition->body_block(),
            successors,
            predecessors,
            reachable};
        _result.statistics.dominance_tree_nodes +=
            dominators.size();
        _result.statistics.dominance_tree_edges +=
            dominators.tree_edge_count();
        _result.statistics.dominance_cfg_edges +=
            dominators.cfg_edge_count();
        _result.statistics.dominance_fixed_point_iterations +=
            dominators.fixed_point_iteration_count();

        auto block_dominates = [&](const BasicBlock *definition_block,
                                   const BasicBlock *use_block) noexcept {
            ++_result.statistics.dominance_queries;
            if (!reachable.contains(use_block)) { return true; }
            if (!reachable.contains(definition_block)) { return false; }
            return dominators.dominates(
                definition_block, use_block);
        };
        auto is_owned_block = [&](const Value *value) noexcept {
            if (value == nullptr || !value->isa<BasicBlock>()) { return false; }
            auto target = static_cast<const BasicBlock *>(value);
            return block_set.contains(target) && target->parent_function() == function;
        };

        struct BreakContinueScope {
            const BasicBlock *parent;
            const BasicBlock *merge;
            const BasicBlock *continue_target;
        };
        luisa::vector<BreakContinueScope> break_continue_scopes;
        if (_options.require_canonical_break_continue_targets) {
            for (auto *block : blocks) {
                if (!block->is_terminated()) { continue; }
                auto *terminator = block->terminator();
                auto facts = instruction_facts.find(terminator);
                if (facts == instruction_facts.end()) {
                    _error(function, block, terminator,
                           "Terminator was not classified by the verifier.");
                    continue;
                }
                auto tag = facts->second.tag;
                if (tag == DerivedInstructionTag::LOOP) {
                    auto *loop = static_cast<const LoopInst *>(terminator);
                    break_continue_scopes.emplace_back(BreakContinueScope{
                        .parent = block,
                        .merge = loop->merge_block(),
                        .continue_target = loop->update_block(),
                    });
                } else if (tag == DerivedInstructionTag::SIMPLE_LOOP) {
                    auto *loop = static_cast<const SimpleLoopInst *>(terminator);
                    break_continue_scopes.emplace_back(BreakContinueScope{
                        .parent = block,
                        .merge = loop->merge_block(),
                        .continue_target = loop->body_block(),
                    });
                } else if (tag == DerivedInstructionTag::SWITCH) {
                    auto *switch_inst = static_cast<const SwitchInst *>(terminator);
                    break_continue_scopes.emplace_back(BreakContinueScope{
                        .parent = block,
                        .merge = switch_inst->merge_block(),
                        .continue_target = nullptr,
                    });
                }
            }
        }

        auto nearest_break_continue_target =
            [&](const BasicBlock *block, bool is_continue) noexcept {
                struct Result {
                    const BasicBlock *target{nullptr};
                    bool found{false};
                    bool ambiguous{false};
                } result;
                size_t best_depth = 0u;
                for (auto &&scope : break_continue_scopes) {
                    if (is_continue && scope.continue_target == nullptr) { continue; }
                    if (scope.parent == block || !reachable.contains(scope.parent) ||
                        !block_dominates(scope.parent, block)) {
                        continue;
                    }
                    if (scope.merge == block ||
                        (scope.merge != nullptr && reachable.contains(scope.merge) &&
                         block_dominates(scope.merge, block))) {
                        continue;
                    }
                    auto depth = dominators.depth(scope.parent);
                    auto *target = is_continue ? scope.continue_target : scope.merge;
                    if (!result.found || depth > best_depth) {
                        result = {.target = target, .found = true, .ambiguous = false};
                        best_depth = depth;
                    } else if (depth == best_depth && result.target != target) {
                        result.ambiguous = true;
                    }
                }
                return result;
            };

        for (auto *block : blocks) {
            for (auto *instruction : block->instructions()) {
                auto facts = instruction_facts.find(instruction);
                if (facts == instruction_facts.end()) {
                    _error(function, block, instruction,
                           "Instruction was not classified by the verifier.");
                    continue;
                }
                auto tag = facts->second.tag;
                if (_options.require_no_phi &&
                    tag == DerivedInstructionTag::PHI) {
                    _error(function, block, instruction, "PHI instruction is not allowed.");
                }
                if (_options.require_no_unstructured_control_flow &&
                    (tag == DerivedInstructionTag::CONDITIONAL_BRANCH ||
                     tag == DerivedInstructionTag::INDEXED_BRANCH)) {
                    _error(function, block, instruction,
                           "Unstructured control flow is not allowed.");
                }
                if (!facts->second.operand_shape_valid) { continue; }

                if (auto *merge = instruction->control_flow_merge()) {
                    auto *merge_block = merge->merge_block();
                    auto allows_null_merge =
                        tag == DerivedInstructionTag::IF;
                    if ((merge_block == nullptr && !allows_null_merge) ||
                        (merge_block != nullptr &&
                         (!block_set.contains(merge_block) ||
                          merge_block->parent_function() != function))) {
                        _error(function, block, instruction,
                               "Structured control flow has an invalid merge block.");
                    } else if (merge_block != nullptr &&
                               _options.require_unique_merge_blocks) {
                        if (auto iter = merge_owners.find(merge_block);
                            iter != merge_owners.end() && iter->second != instruction) {
                            _error(function, block, instruction,
                                   "Structured merge block is owned by multiple instructions.");
                        } else {
                            merge_owners.emplace(merge_block, instruction);
                        }
                    }
                }

                if (tag == DerivedInstructionTag::LOOP) {
                    auto *loop = static_cast<const LoopInst *>(instruction);
                    if (!is_owned_block(loop->operand(LoopInst::operand_index_prepare_block))) {
                        _error(function, block, instruction,
                               "Loop has an invalid owned block.");
                    }
                    for (auto *owned : {loop->body_block(), loop->update_block()}) {
                        if (owned == nullptr || !block_set.contains(owned) ||
                            owned->parent_function() != function) {
                            _error(function, block, instruction,
                                   "Loop has an invalid owned block.");
                        }
                    }
                } else if (tag == DerivedInstructionTag::SIMPLE_LOOP) {
                    auto *loop = static_cast<const SimpleLoopInst *>(instruction);
                    if (!is_owned_block(loop->operand(
                            SimpleLoopInst::operand_index_body_block))) {
                        _error(function, block, instruction,
                               "Simple loop has an invalid body block.");
                    }
                } else if (tag == DerivedInstructionTag::AUTODIFF_SCOPE) {
                    auto *scope = static_cast<const AutodiffScopeInst *>(instruction);
                    if (!is_owned_block(scope->operand(
                            AutodiffScopeInst::operand_index_entry_block))) {
                        _error(function, block, instruction,
                               "Autodiff scope has an invalid entry block.");
                    }
                } else if (tag == DerivedInstructionTag::OUTLINE) {
                    auto *outline = static_cast<const OutlineInst *>(instruction);
                    if (!is_owned_block(outline->operand(
                            BranchTerminatorInstruction::operand_index_target))) {
                        _error(function, block, instruction,
                               "Outline instruction has an invalid target block.");
                    }
                } else if (tag == DerivedInstructionTag::RAY_QUERY_LOOP) {
                    auto *loop = static_cast<const RayQueryLoopInst *>(instruction);
                    if (!is_owned_block(loop->operand(
                            RayQueryLoopInst::operand_index_dispatch_block))) {
                        _error(function, block, instruction,
                               "Ray-query loop has an invalid dispatch block.");
                    }
                } else if (tag == DerivedInstructionTag::RAY_QUERY_DISPATCH) {
                    auto *dispatch = static_cast<const RayQueryDispatchInst *>(instruction);
                    auto operand_count = dispatch->operand_count();
                    auto operand_at = [dispatch, operand_count](size_t index) noexcept {
                        return index < operand_count ? dispatch->operand(index) : nullptr;
                    };
                    if (operand_count != 4u ||
                        !ray_query_object_valid(operand_at(
                            RayQueryDispatchInst::operand_index_query_object)) ||
                        !is_owned_block(operand_at(
                            RayQueryDispatchInst::operand_index_exit_block)) ||
                        !is_owned_block(operand_at(
                            RayQueryDispatchInst::operand_index_on_surface_candidate_block)) ||
                        !is_owned_block(operand_at(
                            RayQueryDispatchInst::operand_index_on_procedural_candidate_block))) {
                        _error(function, block, instruction,
                               "Ray-query dispatch operands are invalid.");
                    }
                } else if (
                    tag == DerivedInstructionTag::SWITCH ||
                    tag == DerivedInstructionTag::INDEXED_BRANCH) {
                    auto *indexed_branch = static_cast<
                        const IndexedBranchTerminatorInstruction *>(
                        instruction);
                    auto is_switch =
                        tag == DerivedInstructionTag::SWITCH;
                    auto operand_count = indexed_branch->operand_count();
                    auto expected_operand_count =
                        indexed_branch->case_count() +
                        IndexedBranchTerminatorInstruction::
                            operand_index_case_block_offset;
                    auto *selector =
                        operand_count >
                                IndexedBranchTerminatorInstruction::
                                    operand_index_value ?
                            indexed_branch->operand(
                                IndexedBranchTerminatorInstruction::
                                    operand_index_value) :
                            nullptr;
                    auto *default_block =
                        operand_count >
                                IndexedBranchTerminatorInstruction::
                                    operand_index_default_block ?
                            indexed_branch->operand(
                                IndexedBranchTerminatorInstruction::
                                    operand_index_default_block) :
                            nullptr;
                    auto *selector_type = selector == nullptr ? nullptr : selector->type();
                    if (selector == nullptr || !is_owned_block(default_block) ||
                        operand_count != expected_operand_count) {
                        _error(function, block, instruction,
                               is_switch ?
                                   "Switch value or default block is invalid." :
                                   "Indexed branch value or default block is invalid.");
                    } else if (!data_operand_valid(selector) || selector_type == nullptr ||
                               !selector_type->is_scalar() ||
                               (!selector_type->is_bool() &&
                                !scalar_or_vector_integer(selector_type))) {
                        _error(function, block, instruction,
                               is_switch ?
                                   "Switch selector is not an integer/bool scalar rvalue." :
                                   "Indexed branch selector is not an integer/bool scalar rvalue.");
                    }
                    luisa::unordered_set<
                        IndexedBranchTerminatorInstruction::case_value_type>
                        case_values;
                    for (auto i = 0u; i < indexed_branch->case_count(); i++) {
                        auto operand_index =
                            IndexedBranchTerminatorInstruction::
                                operand_index_case_block_offset +
                            i;
                        auto *case_block = operand_index < operand_count ?
                                               indexed_branch->operand(operand_index) :
                                               nullptr;
                        if (!is_owned_block(case_block)) {
                            _error(function, block, instruction,
                                   is_switch ?
                                       "Switch case block is invalid." :
                                       "Indexed branch case block is invalid.");
                        }
                        auto case_value = indexed_branch->case_value(i);
                        auto canonical_value =
                            IndexedBranchTerminatorInstruction::
                                canonicalize_case_value(
                                    selector_type, case_value);
                        if (case_value != canonical_value) {
                            _error(function, block, instruction,
                                   is_switch ?
                                       "Switch case value is outside the selector bit width." :
                                       "Indexed branch case value is outside the selector bit width.");
                        }
                        if (!case_values.emplace(canonical_value).second) {
                            _error(function, block, instruction,
                                   is_switch ?
                                       "Switch case values alias after selector-width normalization." :
                                       "Indexed branch case values alias after selector-width normalization.");
                        }
                    }
                }

                for (size_t operand_index = 0u;
                     operand_index < instruction->operand_count(); ++operand_index) {
                    auto *operand_use = instruction->operand_use(operand_index);
                    auto *operand = operand_use->value();
                    if (operand != nullptr) {
                        if (operand_use->user() != instruction ||
                            !_use_list_contains(operand, operand_use)) {
                            _error(function, block, instruction,
                                   "Operand use-list linkage is inconsistent.");
                        }
                        switch (operand->derived_value_tag()) {
                            case DerivedValueTag::BASIC_BLOCK: {
                                auto *target = static_cast<const BasicBlock *>(operand);
                                if (!block_set.contains(target) ||
                                    target->parent_function() != function) {
                                    _error(function, block, instruction,
                                           "Instruction references a block from another function.");
                                }
                                break;
                            }
                            case DerivedValueTag::ARGUMENT: {
                                auto *argument = static_cast<const Argument *>(operand);
                                if (argument->parent_function() != function) {
                                    _error(function, block, instruction,
                                           "Instruction references an argument from another function.");
                                }
                                break;
                            }
                            case DerivedValueTag::INSTRUCTION: {
                                auto *definition_instruction = static_cast<const Instruction *>(operand);
                                auto definition_iter =
                                    instruction_facts.find(
                                        definition_instruction);
                                if (definition_iter == instruction_facts.end()) {
                                    _error(function, block, instruction,
                                           "Instruction references a definition from another function.");
                                    break;
                                }
                                if (tag == DerivedInstructionTag::PHI) { break; }
                                auto *definition_block =
                                    definition_iter->second.block;
                                if (definition_block == block) {
                                    if (definition_iter->second.order >=
                                        facts->second.order) {
                                        _error(function, block, instruction,
                                               "Instruction operand does not precede its use.");
                                    }
                                } else if (!block_dominates(definition_block, block)) {
                                    _error(function, block, instruction,
                                           "Instruction operand does not dominate its use.");
                                }
                                break;
                            }
                            case DerivedValueTag::FUNCTION: {
                                auto *callee = static_cast<const Function *>(operand);
                                if (callee->parent_module() != module) {
                                    _error(function, block, instruction,
                                           "Instruction references a function from another module.");
                                }
                                break;
                            }
                            case DerivedValueTag::CONSTANT: {
                                auto *constant = static_cast<const Constant *>(operand);
                                if (constant->parent_module() != module) {
                                    _error(function, block, instruction,
                                           "Instruction references a constant from another module.");
                                }
                                break;
                            }
                            case DerivedValueTag::UNDEFINED: {
                                auto *undefined = static_cast<const Undefined *>(operand);
                                if (undefined->parent_module() != module) {
                                    _error(function, block, instruction,
                                           "Instruction references undef from another module.");
                                }
                                break;
                            }
                            case DerivedValueTag::SPECIAL_REGISTER: {
                                auto *special = static_cast<const SpecialRegister *>(operand);
                                if (special->parent_module() != module) {
                                    _error(function, block, instruction,
                                           "Instruction references a special register from another module.");
                                }
                                break;
                            }
                        }
                    }
                }

                if (tag == DerivedInstructionTag::IF ||
                    tag == DerivedInstructionTag::CONDITIONAL_BRANCH) {
                    auto *condition = instruction->operand_count() >
                                              ConditionalBranchTerminatorInstruction::operand_index_condition ?
                                          instruction->operand(
                                              ConditionalBranchTerminatorInstruction::operand_index_condition) :
                                          nullptr;
                    if (!data_operand_valid(condition) ||
                        !condition->type()->is_bool()) {
                        _error(function, block, instruction,
                               "Conditional branch condition is not a boolean rvalue.");
                    }
                    auto *true_target = instruction->operand_count() >
                                                ConditionalBranchTerminatorInstruction::operand_index_true_target ?
                                            instruction->operand(
                                                ConditionalBranchTerminatorInstruction::operand_index_true_target) :
                                            nullptr;
                    auto *false_target = instruction->operand_count() >
                                                 ConditionalBranchTerminatorInstruction::operand_index_false_target ?
                                             instruction->operand(
                                                 ConditionalBranchTerminatorInstruction::operand_index_false_target) :
                                             nullptr;
                    if (!is_owned_block(true_target) || !is_owned_block(false_target)) {
                        _error(function, block, instruction,
                               "Conditional branch has an invalid target.");
                    }
                } else if (tag == DerivedInstructionTag::BRANCH ||
                           tag == DerivedInstructionTag::BREAK ||
                           tag == DerivedInstructionTag::CONTINUE) {
                    auto *target = instruction->operand_count() >
                                           BranchTerminatorInstruction::operand_index_target ?
                                       instruction->operand(
                                           BranchTerminatorInstruction::operand_index_target) :
                                       nullptr;
                    if (!is_owned_block(target)) {
                        _error(function, block, instruction, "Branch has an invalid target.");
                    }
                    if (_options.require_canonical_break_continue_targets &&
                        reachable.contains(block) &&
                        (tag == DerivedInstructionTag::BREAK ||
                         tag == DerivedInstructionTag::CONTINUE)) {
                        auto is_continue = tag == DerivedInstructionTag::CONTINUE;
                        auto expected = nearest_break_continue_target(block, is_continue);
                        if (!expected.found || expected.ambiguous || target != expected.target) {
                            _error(
                                function, block, instruction,
                                is_continue ?
                                    "Continue target is not the nearest enclosing structured loop target." :
                                    "Break target is not the nearest enclosing structured break target.");
                        }
                    }
                }

                if (tag == DerivedInstructionTag::LOAD) {
                    auto *load = static_cast<const LoadInst *>(instruction);
                    if (!typed_value_operand_valid(load->variable()) ||
                        !load->variable()->is_lvalue() ||
                        load->type() != load->variable()->type()) {
                        _error(function, block, instruction,
                               "Load variable or result type is invalid.");
                    }
                } else if (tag == DerivedInstructionTag::STORE) {
                    auto *store = static_cast<const StoreInst *>(instruction);
                    if (!typed_value_operand_valid(store->variable()) ||
                        !store->variable()->is_lvalue() ||
                        !rvalue_operand_valid(store->value()) ||
                        store->variable()->type() != store->value()->type()) {
                        _error(function, block, instruction,
                               "Store variable or value type is invalid.");
                    }
                } else if (tag == DerivedInstructionTag::GEP) {
                    auto *gep = static_cast<const GEPInst *>(instruction);
                    if (!typed_value_operand_valid(gep->base()) ||
                        !gep->base()->is_lvalue() ||
                        gep->type() == nullptr || gep->index_count() == 0u ||
                        gep_indexed_type(gep) != gep->type()) {
                        _error(function, block, instruction, "GEP is invalid.");
                    }
                } else if (tag == DerivedInstructionTag::CAST) {
                    auto *cast = static_cast<const CastInst *>(instruction);
                    if (!cast_types_valid(cast)) {
                        _error(function, block, instruction,
                               "Cast operands or result type are invalid.");
                    }
                } else if (tag == DerivedInstructionTag::ARITHMETIC) {
                    auto *arithmetic = static_cast<const ArithmeticInst *>(instruction);
                    if (!arithmetic_types_valid(arithmetic)) {
                        _error(
                            function, block, instruction,
                            luisa::format(
                                "Arithmetic operands or result type are invalid. Operation: '{}'.",
                                to_string(arithmetic->op())));
                    }
                } else if (tag == DerivedInstructionTag::CALL) {
                    auto *call = static_cast<const CallInst *>(instruction);
                    auto *callee_value = call->operand_count() > CallInst::operand_index_callee ?
                                             call->operand(CallInst::operand_index_callee) :
                                             nullptr;
                    auto *callee = callee_value != nullptr && callee_value->isa<Function>() ?
                                       static_cast<const Function *>(callee_value) :
                                       nullptr;
                    if (callee == nullptr || call->type() != callee->type() ||
                        call->argument_count() != callee->arguments().count_size()) {
                        _error(function, block, instruction,
                               "Call result type or argument count is invalid.");
                    } else {
                        size_t index = 0u;
                        for (auto *argument : callee->arguments()) {
                            auto *value = call->argument(index++);
                            if (!argument_matches(argument, value)) {
                                _error(function, block, instruction,
                                       "Call argument type or value category is invalid.");
                            }
                        }
                    }
                } else if (tag == DerivedInstructionTag::RETURN) {
                    auto *return_inst = static_cast<const ReturnInst *>(instruction);
                    auto *return_value = return_inst->return_value();
                    if ((function->type() == nullptr) != (return_value == nullptr) ||
                        (return_value != nullptr &&
                         (!rvalue_operand_valid(return_value) ||
                          return_value->type() != function->type()))) {
                        _error(function, block, instruction,
                               "Return value does not match the function return type.");
                    }
                }

                if (tag == DerivedInstructionTag::PHI) {
                    auto *phi = static_cast<const PhiInst *>(instruction);
                    BlockSet incoming_blocks;
                    for (size_t i = 0u; i < phi->incoming_count(); ++i) {
                        auto incoming = phi->incoming(i);
                        if (!rvalue_operand_valid(incoming.value) ||
                            incoming.value->type() != phi->type() ||
                            incoming.block == nullptr ||
                            !predecessors[block].contains(incoming.block) ||
                            !incoming_blocks.emplace(incoming.block).second) {
                            _error(function, block, instruction,
                                   "PHI incoming edge or value is invalid.");
                            continue;
                        }
                        if (incoming.value->isa<Instruction>()) {
                            auto *incoming_instruction = static_cast<const Instruction *>(incoming.value);
                            auto definition_iter =
                                instruction_facts.find(
                                    incoming_instruction);
                            if (definition_iter == instruction_facts.end()) {
                                _error(function, block, instruction,
                                       "PHI references a definition from another function.");
                            } else if (definition_iter->second.block == incoming.block &&
                                       incoming.block->is_terminated()) {
                                auto terminator_iter = instruction_facts.find(
                                    incoming.block->terminator());
                                if (terminator_iter == instruction_facts.end()) {
                                    _error(function, block, instruction,
                                           "PHI predecessor terminator was not classified.");
                                } else if (definition_iter->second.order >=
                                           terminator_iter->second.order) {
                                    _error(function, block, instruction,
                                           "PHI incoming value does not precede the incoming edge.");
                                }
                            } else if (!block_dominates(definition_iter->second.block,
                                                        incoming.block)) {
                                _error(function, block, instruction,
                                       "PHI incoming value does not dominate the incoming edge.");
                            }
                        }
                    }
                    if (incoming_blocks.size() != predecessors[block].size()) {
                        _error(function, block, instruction,
                               "PHI incoming blocks do not match CFG predecessors.");
                    }
                }
            }
        }
    }
};

}// namespace detail

XIRVerificationResult xir_verify_function(
    const Function *function,
    const XIRVerificationOptions &options) noexcept {
    XIRVerificationResult result;
    detail::XIRVerifier verifier{options, result};
    verifier.verify(function);
    return result;
}

XIRVerificationResult xir_verify_functions(
    luisa::span<const Function *const> functions,
    const XIRVerificationOptions &options) noexcept {
    XIRVerificationResult result;
    detail::XIRVerifier verifier{options, result};
    for (auto *function : functions) {
        verifier.verify(function);
    }
    return result;
}

XIRVerificationResult xir_verify_module(
    const Module *module,
    const XIRVerificationOptions &options) noexcept {
    XIRVerificationResult result;
    if (module == nullptr) {
        result.errors.emplace_back(XIRVerificationError{
            .message = "Module is null.",
        });
        return result;
    }
    detail::XIRVerifier verifier{options, result};
    for (auto *function : module->function_list()) {
        verifier.verify(function);
    }
    return result;
}

}// namespace luisa::compute::xir
