#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal {

namespace detail {

[[nodiscard]] bool supported_type(const Type *type, luisa::string &reason) noexcept {
    if (type == nullptr) { return true; }
    switch (type->tag()) {
        case Type::Tag::BOOL: [[fallthrough]];
        case Type::Tag::INT8: [[fallthrough]];
        case Type::Tag::UINT8: [[fallthrough]];
        case Type::Tag::INT16: [[fallthrough]];
        case Type::Tag::UINT16: [[fallthrough]];
        case Type::Tag::INT32: [[fallthrough]];
        case Type::Tag::UINT32: [[fallthrough]];
        case Type::Tag::INT64: [[fallthrough]];
        case Type::Tag::UINT64: [[fallthrough]];
        case Type::Tag::FLOAT16: [[fallthrough]];
        case Type::Tag::FLOAT32: return true;
        case Type::Tag::FLOAT64: [[fallthrough]];
        case Type::Tag::FLOAT8_E4M3: [[fallthrough]];
        case Type::Tag::FLOAT8_E5M2: [[fallthrough]];
        case Type::Tag::INT4: [[fallthrough]];
        case Type::Tag::FP4_E2M1:
            reason = "unsupported scalar type '" + luisa::string{type->description()} + "'";
            return false;
        case Type::Tag::VECTOR: [[fallthrough]];
        case Type::Tag::MATRIX: [[fallthrough]];
        case Type::Tag::ARRAY: [[fallthrough]];
        case Type::Tag::COOPERATIVE_VECTOR:
            return supported_type(type->element(), reason);
        case Type::Tag::COOPERATIVE_VECTOR_REF: [[fallthrough]];
        case Type::Tag::COOPERATIVE_MATRIX_REF:
            return true;
        case Type::Tag::STRUCTURE:
            for (auto member : type->members()) {
                if (!supported_type(member, reason)) { return false; }
            }
            return true;
        case Type::Tag::BUFFER:
            return supported_type(type->element(), reason);
        case Type::Tag::BINDLESS_ARRAY:
            return true;
        case Type::Tag::ACCEL:
            return true;
        case Type::Tag::TEXTURE:
            if ((type->dimension() == 2u || type->dimension() == 3u) &&
                (type->element()->is_int32() ||
                 type->element()->is_uint32() ||
                 type->element()->is_float32())) {
                return true;
            }
            reason = "unsupported texture type '" + luisa::string{type->description()} + "'";
            return false;
        case Type::Tag::CUSTOM:
            if (is_indirect_dispatch_buffer_type(type) ||
                is_ray_query_type(type)) {
                return true;
            }
            reason = "unsupported custom type '" + luisa::string{type->description()} + "'";
            return false;
        default:
            reason = "unsupported type '" + luisa::string{type->description()} + "'";
            return false;
    }
}

[[nodiscard]] bool supported_ray_payload_capture(
    const xir::Value *value, luisa::string &reason) noexcept {
    if (value == nullptr || value->type() == nullptr) {
        reason = "ray-query pipeline has a null captured argument";
        return false;
    }
    if (value->is_lvalue() &&
        (!value->isa<xir::AllocaInst>() ||
         !static_cast<const xir::AllocaInst *>(value)->is_local())) {
        reason = "ray-query payload reference capture is not a local allocation";
        return false;
    }
    auto supported_payload_type = [&](auto &&self,
                                      const Type *type) noexcept -> bool {
        switch (type->tag()) {
            case Type::Tag::BOOL: [[fallthrough]];
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::INT64: [[fallthrough]];
            case Type::Tag::UINT64: [[fallthrough]];
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: return true;
            case Type::Tag::VECTOR: [[fallthrough]];
            case Type::Tag::MATRIX: [[fallthrough]];
            case Type::Tag::ARRAY:
                return self(self, type->element());
            case Type::Tag::STRUCTURE:
                for (auto member : type->members()) {
                    if (!self(self, member)) { return false; }
                }
                return true;
            case Type::Tag::BUFFER:
                return type->element() == nullptr ||
                       self(self, type->element());
            case Type::Tag::BINDLESS_ARRAY: return true;
            default: return false;
        }
    };
    if (!supported_payload_type(supported_payload_type, value->type())) {
        reason = "ray-query payload cannot represent captured type '" +
                 luisa::string{value->type()->description()} + "'";
        return false;
    }
    return true;
}

[[nodiscard]] bool supported_print_type(const Type *type) noexcept {
    if (type == nullptr) { return false; }
    if (type->is_scalar()) { return type->tag() != Type::Tag::FLOAT64; }
    if (type->is_vector() || type->is_matrix() || type->is_array()) {
        return supported_print_type(type->element());
    }
    if (type->is_structure()) {
        for (auto member : type->members()) {
            if (!supported_print_type(member)) { return false; }
        }
        return true;
    }
    return false;
}

[[nodiscard]] bool is_float3_storage_type(const Type *type) noexcept {
    return type != nullptr &&
           (type->is_array() || type->is_vector()) &&
           type->dimension() == 3u &&
           type->element()->is_float32();
}

[[nodiscard]] bool is_ray_type(const Type *type) noexcept {
    if (type == nullptr || !type->is_structure() ||
        type->members().size() != 4u) {
        return false;
    }
    auto members = type->members();
    return is_float3_storage_type(members[0u]) &&
           members[1u]->is_float32() &&
           is_float3_storage_type(members[2u]) &&
           members[3u]->is_float32();
}

[[nodiscard]] bool is_triangle_hit_type(const Type *type) noexcept {
    if (type == nullptr || !type->is_structure() ||
        type->members().size() != 4u) {
        return false;
    }
    auto members = type->members();
    return members[0u]->is_uint32() && members[1u]->is_uint32() &&
           members[2u]->is_float32_vector() &&
           members[2u]->dimension() == 2u &&
           members[3u]->is_float32();
}

[[nodiscard]] bool is_procedural_hit_type(const Type *type) noexcept {
    if (type == nullptr || !type->is_structure() ||
        type->members().size() != 2u) {
        return false;
    }
    auto members = type->members();
    return members[0u]->is_uint32() && members[1u]->is_uint32();
}

[[nodiscard]] bool is_committed_hit_type(const Type *type) noexcept {
    if (type == nullptr || !type->is_structure() ||
        type->members().size() != 5u) {
        return false;
    }
    auto members = type->members();
    return members[0u]->is_uint32() && members[1u]->is_uint32() &&
           members[2u]->is_float32_vector() &&
           members[2u]->dimension() == 2u &&
           members[3u]->is_uint32() && members[4u]->is_float32();
}

[[nodiscard]] bool is_float4x4_type(const Type *type) noexcept {
    return type != nullptr && type->is_matrix() &&
           type->element()->is_float32() && type->dimension() == 4u;
}

[[nodiscard]] bool supported_texture_usage(
    const xir::Value *texture, luisa::string &reason) noexcept {
    auto samples = false;
    for (auto use : texture->use_list()) {
        auto user = use->user();
        if (user == nullptr) { continue; }
        if (user->isa<xir::ResourceReadInst>() ||
            user->isa<xir::ResourceWriteInst>()) {
            continue;
        } else if (user->isa<xir::ResourceQueryInst>()) {
            auto query = static_cast<const xir::ResourceQueryInst *>(user);
            if (is_direct_texture_sample(query->op())) {
                samples = true;
            } else if (query->op() != xir::ResourceQueryOp::TEXTURE2D_SIZE &&
                       query->op() != xir::ResourceQueryOp::TEXTURE3D_SIZE) {
                reason = "texture is used by unsupported query '" +
                         luisa::string{xir::to_string(query->op())} + "'";
                return false;
            }
        } else {
            reason = "texture use was not normalized to a direct resource operation";
            return false;
        }
    }
    if (samples && !texture->type()->element()->is_float32()) {
        reason = "AIR direct texture sampling requires a float texture";
        return false;
    }
    return true;
}

[[nodiscard]] bool supported_arithmetic(xir::ArithmeticOp op) noexcept {
    switch (op) {
        case xir::ArithmeticOp::UNARY_MINUS: [[fallthrough]];
        case xir::ArithmeticOp::UNARY_BIT_NOT: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_ADD: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_SUB: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_MUL: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_DIV: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_MOD: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_BIT_AND: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_BIT_OR: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_BIT_XOR: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_SHIFT_LEFT: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_SHIFT_RIGHT: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_ROTATE_LEFT: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_ROTATE_RIGHT: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_LESS: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_GREATER: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_LESS_EQUAL: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_GREATER_EQUAL: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_EQUAL: [[fallthrough]];
        case xir::ArithmeticOp::BINARY_NOT_EQUAL: [[fallthrough]];
        case xir::ArithmeticOp::ALL: [[fallthrough]];
        case xir::ArithmeticOp::ANY: [[fallthrough]];
        case xir::ArithmeticOp::SELECT: [[fallthrough]];
        case xir::ArithmeticOp::CLAMP: [[fallthrough]];
        case xir::ArithmeticOp::SATURATE: [[fallthrough]];
        case xir::ArithmeticOp::LERP: [[fallthrough]];
        case xir::ArithmeticOp::SMOOTHSTEP: [[fallthrough]];
        case xir::ArithmeticOp::STEP: [[fallthrough]];
        case xir::ArithmeticOp::ABS: [[fallthrough]];
        case xir::ArithmeticOp::MIN: [[fallthrough]];
        case xir::ArithmeticOp::MAX: [[fallthrough]];
        case xir::ArithmeticOp::CLZ: [[fallthrough]];
        case xir::ArithmeticOp::CTZ: [[fallthrough]];
        case xir::ArithmeticOp::POPCOUNT: [[fallthrough]];
        case xir::ArithmeticOp::REVERSE: [[fallthrough]];
        case xir::ArithmeticOp::ISINF: [[fallthrough]];
        case xir::ArithmeticOp::ISNAN: [[fallthrough]];
        case xir::ArithmeticOp::ACOS: [[fallthrough]];
        case xir::ArithmeticOp::ACOSH: [[fallthrough]];
        case xir::ArithmeticOp::ASIN: [[fallthrough]];
        case xir::ArithmeticOp::ASINH: [[fallthrough]];
        case xir::ArithmeticOp::ATAN: [[fallthrough]];
        case xir::ArithmeticOp::ATAN2: [[fallthrough]];
        case xir::ArithmeticOp::ATANH: [[fallthrough]];
        case xir::ArithmeticOp::COS: [[fallthrough]];
        case xir::ArithmeticOp::COSH: [[fallthrough]];
        case xir::ArithmeticOp::SIN: [[fallthrough]];
        case xir::ArithmeticOp::SINH: [[fallthrough]];
        case xir::ArithmeticOp::TAN: [[fallthrough]];
        case xir::ArithmeticOp::TANH: [[fallthrough]];
        case xir::ArithmeticOp::EXP: [[fallthrough]];
        case xir::ArithmeticOp::EXP2: [[fallthrough]];
        case xir::ArithmeticOp::EXP10: [[fallthrough]];
        case xir::ArithmeticOp::LOG: [[fallthrough]];
        case xir::ArithmeticOp::LOG2: [[fallthrough]];
        case xir::ArithmeticOp::LOG10: [[fallthrough]];
        case xir::ArithmeticOp::POW: [[fallthrough]];
        case xir::ArithmeticOp::POW_INT: [[fallthrough]];
        case xir::ArithmeticOp::SQRT: [[fallthrough]];
        case xir::ArithmeticOp::RSQRT: [[fallthrough]];
        case xir::ArithmeticOp::CEIL: [[fallthrough]];
        case xir::ArithmeticOp::FLOOR: [[fallthrough]];
        case xir::ArithmeticOp::FRACT: [[fallthrough]];
        case xir::ArithmeticOp::TRUNC: [[fallthrough]];
        case xir::ArithmeticOp::ROUND: [[fallthrough]];
        case xir::ArithmeticOp::RINT: [[fallthrough]];
        case xir::ArithmeticOp::FMA: [[fallthrough]];
        case xir::ArithmeticOp::COPYSIGN: [[fallthrough]];
        case xir::ArithmeticOp::CROSS: [[fallthrough]];
        case xir::ArithmeticOp::DOT: [[fallthrough]];
        case xir::ArithmeticOp::LENGTH: [[fallthrough]];
        case xir::ArithmeticOp::LENGTH_SQUARED: [[fallthrough]];
        case xir::ArithmeticOp::NORMALIZE: [[fallthrough]];
        case xir::ArithmeticOp::FACEFORWARD: [[fallthrough]];
        case xir::ArithmeticOp::REFLECT: [[fallthrough]];
        case xir::ArithmeticOp::REDUCE_SUM: [[fallthrough]];
        case xir::ArithmeticOp::REDUCE_PRODUCT: [[fallthrough]];
        case xir::ArithmeticOp::REDUCE_MIN: [[fallthrough]];
        case xir::ArithmeticOp::REDUCE_MAX: [[fallthrough]];
        case xir::ArithmeticOp::OUTER_PRODUCT: [[fallthrough]];
        case xir::ArithmeticOp::MATRIX_COMP_NEG: [[fallthrough]];
        case xir::ArithmeticOp::MATRIX_COMP_ADD: [[fallthrough]];
        case xir::ArithmeticOp::MATRIX_COMP_SUB: [[fallthrough]];
        case xir::ArithmeticOp::MATRIX_COMP_MUL: [[fallthrough]];
        case xir::ArithmeticOp::MATRIX_COMP_DIV: [[fallthrough]];
        case xir::ArithmeticOp::MATRIX_LINALG_MUL: [[fallthrough]];
        case xir::ArithmeticOp::MATRIX_DETERMINANT: [[fallthrough]];
        case xir::ArithmeticOp::MATRIX_TRANSPOSE: [[fallthrough]];
        case xir::ArithmeticOp::MATRIX_INVERSE: [[fallthrough]];
        case xir::ArithmeticOp::AGGREGATE: [[fallthrough]];
        case xir::ArithmeticOp::SHUFFLE: [[fallthrough]];
        case xir::ArithmeticOp::INSERT: [[fallthrough]];
        case xir::ArithmeticOp::EXTRACT: return true;
        default: return false;
    }
}

[[nodiscard]] bool supported_simd_numeric_type(const Type *type) noexcept {
    if (type == nullptr || !type->is_scalar_or_vector()) { return false; }
    auto scalar = type->is_vector() ? type->element() : type;
    return scalar->is_int8() || scalar->is_uint8() ||
           scalar->is_int16() || scalar->is_uint16() ||
           scalar->is_int32() || scalar->is_uint32() ||
           scalar->is_float16() || scalar->is_float32();
}

[[nodiscard]] bool supported_simd_integer_type(const Type *type) noexcept {
    return supported_simd_numeric_type(type) &&
           (type->is_int_or_int_vector() || type->is_uint_or_uint_vector());
}

[[nodiscard]] bool supported_simd_shuffle_type(const Type *type) noexcept {
    if (type == nullptr || type->is_resource()) { return false; }
    if (type->is_scalar()) {
        return type->is_bool() || type->is_int8() || type->is_uint8() ||
               type->is_int16() || type->is_uint16() ||
               type->is_int32() || type->is_uint32() ||
               type->is_int64() || type->is_uint64() ||
               type->is_float16() || type->is_float32();
    }
    if (type->is_vector() || type->is_matrix() || type->is_array()) {
        return supported_simd_shuffle_type(type->element());
    }
    if (!type->is_structure()) { return false; }
    for (auto member : type->members()) {
        if (!supported_simd_shuffle_type(member)) { return false; }
    }
    return true;
}

[[nodiscard]] bool supported_thread_group(
    const xir::ThreadGroupInst *group, MetalAIRProgram program,
    luisa::string &reason) noexcept {
    auto reject = [&](luisa::string_view detail) noexcept {
        reason = "unsupported thread-group operation '" +
                 luisa::string{xir::to_string(group->op())} + "': " +
                 luisa::string{detail};
        return false;
    };
    auto one_bool_operand = [&]() noexcept {
        return group->operand_count() == 1u && group->operand(0u)->type()->is_bool();
    };
    if (program != MetalAIRProgram::COMPUTE) {
        auto depth_mode = air_raster_depth_mode(group->op());
        if (depth_mode != AIRRasterDepthMode::NONE) {
            return program == MetalAIRProgram::RASTER_FRAGMENT &&
                           group->type() == nullptr &&
                           group->operand_count() == 1u &&
                           group->operand(0u)->type() != nullptr &&
                           group->operand(0u)->type()->is_float32() ?
                       true :
                       reject("depth output requires one f32 scalar and no result in a fragment stage");
        }
        if (group->op() != xir::ThreadGroupOp::RASTER_QUAD_DDX &&
            group->op() != xir::ThreadGroupOp::RASTER_QUAD_DDY) {
            return reject("compute thread-group operation used by a raster stage");
        }
        auto type = group->operand_count() == 1u ?
                        group->operand(0u)->type() :
                        nullptr;
        auto element = type != nullptr && type->is_vector() ?
                           type->element() :
                           type;
        return program == MetalAIRProgram::RASTER_FRAGMENT &&
                       type != nullptr && type->is_scalar_or_vector() &&
                       element != nullptr &&
                       (element->is_float16() || element->is_float32()) ?
                   true :
                   reject("derivatives require one f16/f32 scalar or vector in a fragment stage");
    }
    switch (group->op()) {
        case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER:
            return group->operand_count() <= 2u || reject("expected at most hint and hint-bit operands");
        case xir::ThreadGroupOp::RASTER_QUAD_DDX: [[fallthrough]];
        case xir::ThreadGroupOp::RASTER_QUAD_DDY:
            return reject("raster-stage AIR generation is not enabled yet");
        case xir::ThreadGroupOp::RASTER_SET_Z_DEPTH: [[fallthrough]];
        case xir::ThreadGroupOp::RASTER_SET_Z_DEPTH_GREATER_EQUAL: [[fallthrough]];
        case xir::ThreadGroupOp::RASTER_SET_Z_DEPTH_LESS_EQUAL:
            return reject("fragment depth output used by a compute program");
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE:
            return group->operand_count() == 0u || reject("expected no operands");
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: {
            if (group->operand_count() != 1u ||
                !group->operand(0u)->type()->is_scalar_or_vector()) {
                return reject("expected one scalar/vector operand");
            }
            return supported_type(group->operand(0u)->type(), reason);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR:
            return group->operand_count() == 1u &&
                           supported_simd_integer_type(group->operand(0u)->type()) ?
                       true :
                       reject("expected one 8/16/32-bit integer scalar/vector operand");
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_PREFIX_SUM: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT:
            return group->operand_count() == 1u &&
                           supported_simd_numeric_type(group->operand(0u)->type()) ?
                       true :
                       reject("expected one 8/16/32-bit integer or 16/32-bit float scalar/vector operand");
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS:
            return one_bool_operand() || reject("expected one bool predicate operand");
        case xir::ThreadGroupOp::WARP_READ_LANE: {
            if (group->operand_count() != 2u) { return reject("expected value and lane operands"); }
            auto lane = group->operand(1u)->type();
            if (!lane->is_int32() && !lane->is_uint32()) {
                return reject("lane operand must be int or uint");
            }
            return supported_simd_shuffle_type(group->operand(0u)->type()) ||
                   reject("value contains a type that AIR SIMD shuffle cannot legalize");
        }
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE:
            if (group->operand_count() != 1u) { return reject("expected one value operand"); }
            return supported_simd_shuffle_type(group->operand(0u)->type()) ||
                   reject("value contains a type that AIR SIMD broadcast cannot legalize");
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK:
            return group->operand_count() == 0u || reject("expected no operands");
    }
    return reject("unknown operation");
}

[[nodiscard]] bool supported_instruction(
    const xir::Instruction *instruction,
    const MetalCodegenLLVMConfig &config,
    luisa::string &reason) noexcept {
    auto program = config.program;
    auto has_ray_query_operand = false;
    for (auto operand_use : instruction->operand_uses()) {
        auto operand = operand_use->value();
        if (!supported_type(operand == nullptr ? nullptr : operand->type(), reason)) {
            return false;
        }
        if (operand != nullptr &&
            operand->derived_value_tag() == xir::DerivedValueTag::SPECIAL_REGISTER) {
            auto special = static_cast<const xir::SpecialRegister *>(operand);
            auto tag = special->derived_special_register_tag();
            if (program == MetalAIRProgram::COMPUTE &&
                (tag == xir::DerivedSpecialRegisterTag::RASTER_OBJECT_ID ||
                 tag == xir::DerivedSpecialRegisterTag::RASTER_BARYCENTRICS ||
                 tag == xir::DerivedSpecialRegisterTag::RASTER_FRONT_FACING ||
                 tag == xir::DerivedSpecialRegisterTag::RASTER_BASE_INSTANCE)) {
                reason = "raster special register '" + luisa::string{xir::to_string(tag)} +
                         "' requires raster-stage AIR generation";
                return false;
            }
            if (program != MetalAIRProgram::COMPUTE) {
                auto supported_raster_special =
                    tag == xir::DerivedSpecialRegisterTag::KERNEL_ID ||
                    tag == xir::DerivedSpecialRegisterTag::RASTER_OBJECT_ID ||
                    (tag == xir::DerivedSpecialRegisterTag::RASTER_BARYCENTRICS &&
                     program == MetalAIRProgram::RASTER_FRAGMENT) ||
                    (tag == xir::DerivedSpecialRegisterTag::RASTER_FRONT_FACING &&
                     program == MetalAIRProgram::RASTER_FRAGMENT) ||
                    (tag == xir::DerivedSpecialRegisterTag::RASTER_BASE_INSTANCE &&
                     program == MetalAIRProgram::RASTER_VERTEX);
                if (!supported_raster_special) {
                    reason = "special register '" +
                             luisa::string{xir::to_string(tag)} +
                             "' is invalid in this raster stage";
                    return false;
                }
            }
        }
        has_ray_query_operand |=
            operand != nullptr && is_ray_query_type(operand->type());
    }
    if (has_ray_query_operand &&
        instruction->derived_instruction_tag() != xir::DerivedInstructionTag::STORE &&
        instruction->derived_instruction_tag() != xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ &&
        instruction->derived_instruction_tag() != xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE &&
        instruction->derived_instruction_tag() != xir::DerivedInstructionTag::RAY_QUERY_PIPELINE) {
        reason = "ray-query object escaped its initialization/read/write operations";
        return false;
    }
    switch (instruction->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::IF: [[fallthrough]];
        case xir::DerivedInstructionTag::SWITCH: [[fallthrough]];
        case xir::DerivedInstructionTag::INDEXED_BRANCH: [[fallthrough]];
        case xir::DerivedInstructionTag::LOOP: [[fallthrough]];
        case xir::DerivedInstructionTag::SIMPLE_LOOP: [[fallthrough]];
        case xir::DerivedInstructionTag::BRANCH: [[fallthrough]];
        case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: [[fallthrough]];
        case xir::DerivedInstructionTag::UNREACHABLE: [[fallthrough]];
        case xir::DerivedInstructionTag::BREAK: [[fallthrough]];
        case xir::DerivedInstructionTag::CONTINUE: [[fallthrough]];
        case xir::DerivedInstructionTag::RETURN: [[fallthrough]];
        case xir::DerivedInstructionTag::PHI: [[fallthrough]];
        case xir::DerivedInstructionTag::GEP: [[fallthrough]];
        case xir::DerivedInstructionTag::CAST: [[fallthrough]];
        case xir::DerivedInstructionTag::ASSERT: [[fallthrough]];
        case xir::DerivedInstructionTag::ASSUME: break;
        case xir::DerivedInstructionTag::RASTER_DISCARD:
            if (program != MetalAIRProgram::RASTER_FRAGMENT) {
                reason = "raster discard is only valid in a fragment stage";
                return false;
            }
            break;
        case xir::DerivedInstructionTag::ALLOCA: {
            auto allocation = static_cast<const xir::AllocaInst *>(instruction);
            if (is_ray_query_type(allocation->type())) {
                if (!allocation->is_local() ||
                    allocation->parent_function()->definition() == nullptr ||
                    allocation->parent_block() !=
                        allocation->parent_function()->definition()->body_block()) {
                    reason = "ray-query object is not a function-entry local allocation";
                    return false;
                }
                auto initialization_count = 0u;
                for (auto use : allocation->use_list()) {
                    auto user = use->user();
                    if (user == nullptr) {
                        reason = "ray-query object has a null use";
                        return false;
                    }
                    if (user->isa<xir::StoreInst>()) {
                        auto store = static_cast<const xir::StoreInst *>(user);
                        if (store->variable() != allocation ||
                            !store->value()->isa<xir::ResourceQueryInst>() ||
                            store->value()->type() != allocation->type()) {
                            reason = "ray-query object has an invalid initialization store";
                            return false;
                        }
                        initialization_count++;
                    } else if (user->isa<xir::RayQueryObjectReadInst>()) {
                        if (static_cast<const xir::RayQueryObjectReadInst *>(user)->operand(0u) != allocation) {
                            reason = "ray-query object escaped a read operand";
                            return false;
                        }
                    } else if (user->isa<xir::RayQueryObjectWriteInst>()) {
                        if (static_cast<const xir::RayQueryObjectWriteInst *>(user)->operand(0u) != allocation) {
                            reason = "ray-query object escaped a write operand";
                            return false;
                        }
                    } else if (user->isa<xir::RayQueryPipelineInst>()) {
                        if (static_cast<const xir::RayQueryPipelineInst *>(user)
                                ->query_object() != allocation) {
                            reason = "ray-query object escaped a pipeline operand";
                            return false;
                        }
                    } else {
                        reason = "ray-query object has an unsupported use";
                        return false;
                    }
                }
                if (initialization_count != 1u) {
                    reason = "ray-query object must have exactly one initialization store";
                    return false;
                }
            }
            break;
        }
        case xir::DerivedInstructionTag::LOAD:
            if (is_ray_query_type(instruction->type())) {
                reason = "ray-query objects cannot be loaded as ordinary values";
                return false;
            }
            break;
        case xir::DerivedInstructionTag::STORE: {
            auto store = static_cast<const xir::StoreInst *>(instruction);
            if (!is_ray_query_type(store->value()->type())) { break; }
            if (!store->variable()->isa<xir::AllocaInst>() ||
                store->variable()->type() != store->value()->type() ||
                !static_cast<const xir::AllocaInst *>(store->variable())->is_local() ||
                !store->value()->isa<xir::ResourceQueryInst>()) {
                reason = "invalid ray-query initialization store";
                return false;
            }
            auto query = static_cast<const xir::ResourceQueryInst *>(store->value());
            if (query->op() != xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL &&
                query->op() != xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY &&
                query->op() != xir::ResourceQueryOp::
                                   RAY_TRACING_QUERY_ALL_MOTION_BLUR &&
                query->op() != xir::ResourceQueryOp::
                                   RAY_TRACING_QUERY_ANY_MOTION_BLUR) {
                reason = "ray-query object was initialized by a non-query operation";
                return false;
            }
            break;
        }
        case xir::DerivedInstructionTag::ARITHMETIC: {
            auto arithmetic = static_cast<const xir::ArithmeticInst *>(instruction);
            if (supported_arithmetic(arithmetic->op())) { break; }
            reason = "unsupported arithmetic operation '" + luisa::string{xir::to_string(arithmetic->op())} + "'";
            return false;
        }
        case xir::DerivedInstructionTag::CALL: break;
        case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE: {
            auto pipeline = static_cast<const xir::RayQueryPipelineInst *>(
                instruction);
            if (pipeline->query_object() == nullptr ||
                !pipeline->query_object()->isa<xir::AllocaInst>() ||
                !is_ray_query_type(pipeline->query_object()->type())) {
                reason =
                    "Metal AIR ray-query pipeline requires a local query object";
                return false;
            }
            for (auto captured : pipeline->captured_argument_uses()) {
                if (!supported_ray_payload_capture(
                        captured->value(), reason)) {
                    return false;
                }
            }
            auto valid_handler = [&](const xir::Function *handler) noexcept {
                if (handler == nullptr ||
                    !handler->isa<xir::CallableFunction>() ||
                    handler->type() != nullptr ||
                    handler->arguments().count_size() !=
                        pipeline->captured_argument_count() + 1u) {
                    return false;
                }
                auto argument = handler->arguments().begin();
                if (!argument->is_reference() ||
                    argument->type() != pipeline->query_object()->type()) {
                    return false;
                }
                ++argument;
                for (auto captured : pipeline->captured_argument_uses()) {
                    if (argument == handler->arguments().end() ||
                        argument->type() != captured->value()->type() ||
                        argument->is_reference() !=
                            captured->value()->is_lvalue()) {
                        return false;
                    }
                    ++argument;
                }
                return argument == handler->arguments().end();
            };
            if (!valid_handler(pipeline->on_surface_function()) ||
                !valid_handler(pipeline->on_procedural_function())) {
                reason = "Metal AIR ray-query pipeline has an invalid handler ABI";
                return false;
            }
            auto procedural = pipeline->on_procedural_function();
            auto procedural_empty = true;
            procedural->definition()->traverse_instructions(
                [&procedural_empty](const xir::Instruction *candidate) noexcept {
                    procedural_empty &= candidate->isa<xir::ReturnInst>();
                });
            if (!procedural_empty) {
                reason =
                    "Metal AIR loop-to-IFT currently requires an empty procedural handler";
                return false;
            }
            auto config_for_pipeline = AIRRayTracingConfig{};
            auto query_object = pipeline->query_object();
            for (auto use : query_object->use_list()) {
                auto user = use->user();
                if (user == nullptr || !user->isa<xir::StoreInst>()) {
                    continue;
                }
                auto store = static_cast<const xir::StoreInst *>(user);
                if (store->variable() != query_object ||
                    !store->value()->isa<xir::ResourceQueryInst>()) {
                    continue;
                }
                auto constructor =
                    static_cast<const xir::ResourceQueryInst *>(store->value());
                auto basis = constructor->find_metadata<xir::CurveBasisMD>();
                config_for_pipeline.curves =
                    basis != nullptr && basis->curve_basis_set().any();
            }
            if (config_for_pipeline.curves) {
                reason =
                    "Metal AIR loop-to-IFT currently requires triangle-only acceleration structures";
                return false;
            }
            break;
        }
        case xir::DerivedInstructionTag::ATOMIC: {
            auto atomic = static_cast<const xir::AtomicInst *>(instruction);
            auto type = atomic->type();
            if (type == nullptr ||
                (type->tag() != Type::Tag::INT32 &&
                 type->tag() != Type::Tag::UINT32 &&
                 type->tag() != Type::Tag::FLOAT32)) {
                reason = "unsupported atomic value type '" +
                         luisa::string{type == nullptr ? "void" : type->description()} + "'";
                return false;
            }
            if (type->is_float() &&
                (atomic->op() == xir::AtomicOp::FETCH_AND ||
                 atomic->op() == xir::AtomicOp::FETCH_OR ||
                 atomic->op() == xir::AtomicOp::FETCH_XOR)) {
                reason = "unsupported floating-point atomic operation '" +
                         luisa::string{xir::to_string(atomic->op())} + "'";
                return false;
            }
            auto base = atomic->base();
            if (base == nullptr || base->type() == nullptr) {
                reason = "atomic instruction has no base";
                return false;
            }
            if (base->type()->is_buffer()) {
                if (atomic->index_uses().empty()) {
                    reason = "buffer atomic instruction has no element index";
                    return false;
                }
            } else if (!base->isa<xir::AllocaInst>() ||
                       !static_cast<const xir::AllocaInst *>(base)->is_shared()) {
                reason = "atomic base is neither a buffer nor shared memory";
                return false;
            }
            break;
        }
        case xir::DerivedInstructionTag::THREAD_GROUP: {
            auto group = static_cast<const xir::ThreadGroupInst *>(instruction);
            if (!supported_thread_group(group, program, reason)) { return false; }
            break;
        }
        case xir::DerivedInstructionTag::DEBUG_BREAK: {
            for (auto operand_use : instruction->operand_uses()) {
                auto operand = operand_use->value();
                if (operand == nullptr || operand->is_lvalue() ||
                    !supported_print_type(operand->type())) {
                    reason = "debug-break watch is not a supported data rvalue";
                    return false;
                }
            }
            break;
        }
        case xir::DerivedInstructionTag::PRINT: {
            for (auto operand_use : instruction->operand_uses()) {
                auto operand = operand_use->value();
                if (operand == nullptr ||
                    !supported_print_type(operand->type())) {
                    reason = "printer operand is not a plain scalar or aggregate";
                    return false;
                }
            }
            break;
        }
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ: {
            auto read = static_cast<const xir::RayQueryObjectReadInst *>(instruction);
            auto query_operand = read->operand_count() == 1u ?
                                     read->operand(0u) :
                                     nullptr;
            auto valid_query_operand =
                query_operand != nullptr &&
                is_ray_query_type(query_operand->type()) &&
                (query_operand->isa<xir::AllocaInst>() ||
                 (query_operand->isa<xir::Argument>() &&
                  static_cast<const xir::Argument *>(query_operand)
                      ->is_reference()));
            if (read->operand_count() != 1u ||
                !valid_query_operand) {
                reason = "ray-query read does not target a local query object";
                return false;
            }
            auto valid_result = [&]() noexcept {
                switch (read->op()) {
                    case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
                    case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY:
                        return is_ray_type(read->type());
                    case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT:
                        return is_procedural_hit_type(read->type());
                    case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT:
                        return is_triangle_hit_type(read->type());
                    case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT:
                        return is_committed_hit_type(read->type());
                    case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE: [[fallthrough]];
                    case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE: [[fallthrough]];
                    case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED:
                        return read->type() != nullptr && read->type()->is_bool();
                }
                return false;
            }();
            if (!valid_result) {
                reason = "invalid ray-query read result type";
                return false;
            }
            if (read->op() == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED) {
                auto previous = read->prev();
                if (previous == nullptr || previous->is_sentinel() ||
                    !previous->isa<xir::RayQueryObjectWriteInst>()) {
                    reason = "ray-query termination read does not immediately follow proceed";
                    return false;
                }
                auto proceed = static_cast<const xir::RayQueryObjectWriteInst *>(previous);
                if (proceed->op() != xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED ||
                    proceed->operand_count() != 1u ||
                    proceed->operand(0u) != read->operand(0u)) {
                    reason = "ray-query termination read does not match its proceed operation";
                    return false;
                }
            }
            break;
        }
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: {
            auto write = static_cast<const xir::RayQueryObjectWriteInst *>(instruction);
            auto expected_operands =
                write->op() == xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL ?
                    2u :
                    1u;
            auto query_operand = write->operand_count() >= 1u ?
                                     write->operand(0u) :
                                     nullptr;
            auto valid_query_operand =
                query_operand != nullptr &&
                is_ray_query_type(query_operand->type()) &&
                (query_operand->isa<xir::AllocaInst>() ||
                 (query_operand->isa<xir::Argument>() &&
                  static_cast<const xir::Argument *>(query_operand)
                      ->is_reference()));
            if (write->operand_count() != expected_operands ||
                !valid_query_operand ||
                (expected_operands == 2u &&
                 (write->operand(1u)->type() == nullptr ||
                  !write->operand(1u)->type()->is_float32()))) {
                reason = "invalid ray-query write operands";
                return false;
            }
            break;
        }
        case xir::DerivedInstructionTag::RESOURCE_QUERY: {
            auto query = static_cast<const xir::ResourceQueryInst *>(instruction);
            auto is_static_ray_query =
                query->op() == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL ||
                query->op() == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY;
            auto is_motion_ray_query =
                query->op() == xir::ResourceQueryOp::
                                   RAY_TRACING_QUERY_ALL_MOTION_BLUR ||
                query->op() == xir::ResourceQueryOp::
                                   RAY_TRACING_QUERY_ANY_MOTION_BLUR;
            if (is_static_ray_query || is_motion_ray_query) {
                auto is_any =
                    query->op() == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY ||
                    query->op() == xir::ResourceQueryOp::
                                       RAY_TRACING_QUERY_ANY_MOTION_BLUR;
                auto expected_type =
                    is_any ?
                        ray_query_any_type_name :
                        ray_query_all_type_name;
                auto mask_index = is_motion_ray_query ? 3u : 2u;
                if (query->operand_count() !=
                        (is_motion_ray_query ? 4u : 3u) ||
                    query->operand(0u)->type() == nullptr ||
                    !query->operand(0u)->type()->is_accel() ||
                    !is_ray_type(query->operand(1u)->type()) ||
                    (is_motion_ray_query &&
                     (query->operand(2u)->type() == nullptr ||
                      !query->operand(2u)->type()->is_float32())) ||
                    query->operand(mask_index)->type() == nullptr ||
                    !query->operand(mask_index)->type()->is_uint32() ||
                    !is_ray_query_type(query->type()) ||
                    query->type()->description() != expected_type) {
                    reason = "invalid acceleration ray-query operands or result type";
                    return false;
                }
                auto initialization_count = 0u;
                auto consumed_by_pipeline = false;
                for (auto use : query->use_list()) {
                    auto user = use->user();
                    if (user == nullptr || !user->isa<xir::StoreInst>()) {
                        reason = "ray-query construction escaped its initialization store";
                        return false;
                    }
                    auto store = static_cast<const xir::StoreInst *>(user);
                    if (store->value() != query ||
                        !store->variable()->isa<xir::AllocaInst>() ||
                        store->variable()->type() != query->type()) {
                        reason = "invalid ray-query construction store";
                        return false;
                    }
                    initialization_count++;
                    for (auto object_use : store->variable()->use_list()) {
                        auto object_user = object_use->user();
                        consumed_by_pipeline |=
                            object_user != nullptr &&
                            object_user->isa<xir::RayQueryPipelineInst>() &&
                            static_cast<const xir::RayQueryPipelineInst *>(
                                object_user)
                                    ->query_object() == store->variable();
                    }
                }
                if (initialization_count != 1u) {
                    reason = "ray-query construction must have exactly one initialization store";
                    return false;
                }
                if (!consumed_by_pipeline && is_motion_ray_query) {
                    reason =
                        "Metal 4 intersection_query does not accept motion "
                        "acceleration structures or a ray-time operand; the "
                        "query was not eligible for pipeline outlining";
                    return false;
                }
                if (!consumed_by_pipeline &&
                    config.enable_extended_accel_limits) {
                    reason =
                        "Metal 4 intersection_query does not accept the "
                        "extended_limits tag; the query was not eligible for "
                        "pipeline outlining";
                    return false;
                }
                break;
            }
            if (query->op() == xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM ||
                query->op() == xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID ||
                query->op() == xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK) {
                auto valid_result = query->op() == xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM ?
                                        is_float4x4_type(query->type()) :
                                        query->type() != nullptr && query->type()->is_uint32();
                if (query->operand_count() != 2u ||
                    query->operand(0u)->type() == nullptr ||
                    !query->operand(0u)->type()->is_accel() ||
                    query->operand(1u)->type() == nullptr ||
                    !query->operand(1u)->type()->is_uint32() ||
                    !valid_result) {
                    reason = "invalid acceleration instance-query operands or result type";
                    return false;
                }
                break;
            }
            if (query->op() == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST ||
                query->op() == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY ||
                query->op() == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR ||
                query->op() == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR) {
                auto motion =
                    query->op() == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR ||
                    query->op() == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR;
                auto closest =
                    query->op() == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST ||
                    query->op() == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR;
                auto valid_result = closest ?
                                        is_triangle_hit_type(query->type()) :
                                        query->type() != nullptr && query->type()->is_bool();
                auto mask_index = motion ? 3u : 2u;
                if (query->operand_count() != (motion ? 4u : 3u) ||
                    query->operand(0u)->type() == nullptr ||
                    !query->operand(0u)->type()->is_accel() ||
                    !is_ray_type(query->operand(1u)->type()) ||
                    (motion &&
                     (query->operand(2u)->type() == nullptr ||
                      !query->operand(2u)->type()->is_float32())) ||
                    query->operand(mask_index)->type() == nullptr ||
                    !query->operand(mask_index)->type()->is_uint32() ||
                    !valid_result) {
                    reason = "invalid acceleration trace operands or result type";
                    return false;
                }
                break;
            }
            if (is_direct_texture_sample(query->op())) {
                auto expected_operands = query->op() == xir::ResourceQueryOp::TEXTURE2D_SAMPLE ||
                                                 query->op() == xir::ResourceQueryOp::TEXTURE3D_SAMPLE ?
                                             4u :
                                         query->op() == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL ||
                                                 query->op() == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL ?
                                             5u :
                                         query->op() == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD ||
                                                 query->op() == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD ?
                                             6u :
                                             7u;
                if (query->operand_count() != expected_operands ||
                    query->operand(0u)->type() == nullptr ||
                    !query->operand(0u)->type()->is_texture() ||
                    !query->operand(0u)->type()->element()->is_float32() ||
                    query->type() == nullptr || !query->type()->is_float32_vector() ||
                    query->type()->dimension() != 4u) {
                    reason = "invalid direct texture-sample operands or result type";
                    return false;
                }
                break;
            }
            if (is_bindless_texture_sample(query->op())) {
                if (query->operand_count() !=
                        bindless_texture_sample_operand_count(query->op()) ||
                    query->operand(0u)->type() == nullptr ||
                    !query->operand(0u)->type()->is_bindless_array() ||
                    query->type() == nullptr || !query->type()->is_float32_vector() ||
                    query->type()->dimension() != 4u) {
                    reason = "invalid bindless texture-sample operands or result type";
                    return false;
                }
                break;
            }
            if (query->op() == xir::ResourceQueryOp::BUFFER_SIZE ||
                query->op() == xir::ResourceQueryOp::BYTE_BUFFER_SIZE ||
                query->op() == xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS ||
                query->op() == xir::ResourceQueryOp::TEXTURE2D_SIZE ||
                query->op() == xir::ResourceQueryOp::TEXTURE3D_SIZE ||
                query->op() == xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE ||
                query->op() == xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE ||
                query->op() == xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS ||
                query->op() == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE ||
                query->op() == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE ||
                query->op() == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL ||
                query->op() == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL) { break; }
            reason = "unsupported resource query '" + luisa::string{xir::to_string(query->op())} + "'";
            return false;
        }
        case xir::DerivedInstructionTag::RESOURCE_READ: {
            auto read = static_cast<const xir::ResourceReadInst *>(instruction);
            if (read->op() == xir::ResourceReadOp::BUFFER_READ ||
                read->op() == xir::ResourceReadOp::BUFFER_VOLATILE_READ ||
                read->op() == xir::ResourceReadOp::BYTE_BUFFER_READ ||
                read->op() == xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ ||
                read->op() == xir::ResourceReadOp::DEVICE_ADDRESS_READ ||
                read->op() == xir::ResourceReadOp::TEXTURE2D_READ ||
                read->op() == xir::ResourceReadOp::TEXTURE3D_READ ||
                read->op() == xir::ResourceReadOp::BINDLESS_BUFFER_READ ||
                read->op() == xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ ||
                read->op() == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ ||
                read->op() == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ ||
                read->op() == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL ||
                read->op() == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL ||
                read->op() == xir::ResourceReadOp::COOPERATIVE_VECTOR_LOAD ||
                read->op() == xir::ResourceReadOp::BINDLESS_COOPERATIVE_VECTOR_LOAD ||
                read->op() == xir::ResourceReadOp::COOPERATIVE_VECTOR_SPLAT ||
                read->op() == xir::ResourceReadOp::COOPERATIVE_VECTOR_CAST ||
                read->op() == xir::ResourceReadOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD) { break; }
            reason = "unsupported resource read '" + luisa::string{xir::to_string(read->op())} + "'";
            return false;
        }
        case xir::DerivedInstructionTag::RESOURCE_WRITE: {
            auto write = static_cast<const xir::ResourceWriteInst *>(instruction);
            if (write->op() == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM ||
                write->op() == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK ||
                write->op() == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY ||
                write->op() == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID) {
                auto value_type = write->operand_count() == 3u ?
                                      write->operand(2u)->type() :
                                      nullptr;
                auto valid_value = write->op() == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM ?
                                       is_float4x4_type(value_type) :
                                   write->op() == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY ?
                                       value_type != nullptr && value_type->is_bool() :
                                       value_type != nullptr && value_type->is_uint32();
                if (write->operand_count() != 3u ||
                    write->operand(0u)->type() == nullptr ||
                    !write->operand(0u)->type()->is_accel() ||
                    write->operand(1u)->type() == nullptr ||
                    !write->operand(1u)->type()->is_uint32() ||
                    !valid_value) {
                    reason = "invalid acceleration instance-write operands";
                    return false;
                }
                break;
            }
            if (write->op() == xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT) {
                if (write->operand_count() != 2u ||
                    !is_indirect_dispatch_buffer_type(write->operand(0u)->type()) ||
                    write->operand(1u)->type() == nullptr ||
                    !write->operand(1u)->type()->is_uint32()) {
                    reason = "invalid indirect-dispatch count operands";
                    return false;
                }
                break;
            }
            if (write->op() == xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL) {
                auto is_uint3 = [](const Type *type) noexcept {
                    return type != nullptr && type->is_uint32_vector() &&
                           type->dimension() == 3u;
                };
                if (write->operand_count() != 5u ||
                    !is_indirect_dispatch_buffer_type(write->operand(0u)->type()) ||
                    write->operand(1u)->type() == nullptr ||
                    !write->operand(1u)->type()->is_uint32() ||
                    !is_uint3(write->operand(2u)->type()) ||
                    !is_uint3(write->operand(3u)->type()) ||
                    write->operand(4u)->type() == nullptr ||
                    !write->operand(4u)->type()->is_uint32()) {
                    reason = "invalid indirect-dispatch kernel operands";
                    return false;
                }
                break;
            }
            if (write->op() == xir::ResourceWriteOp::BUFFER_WRITE ||
                write->op() == xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE ||
                write->op() == xir::ResourceWriteOp::BYTE_BUFFER_WRITE ||
                write->op() == xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE ||
                write->op() == xir::ResourceWriteOp::DEVICE_ADDRESS_WRITE ||
                write->op() == xir::ResourceWriteOp::TEXTURE2D_WRITE ||
                write->op() == xir::ResourceWriteOp::TEXTURE3D_WRITE ||
                write->op() == xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE ||
                write->op() == xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE ||
                write->op() == xir::ResourceWriteOp::COOPERATIVE_VECTOR_STORE ||
                write->op() == xir::ResourceWriteOp::BINDLESS_COOPERATIVE_VECTOR_STORE ||
                write->op() == xir::ResourceWriteOp::COOPERATIVE_VECTOR_WORKGROUP_STORE) { break; }
            if (write->op() == xir::ResourceWriteOp::COOPERATIVE_VECTOR_ACCUMULATE) {
                auto value_type = write->operand_count() == 3u ?
                                      write->operand(2u)->type() :
                                      nullptr;
                auto element = value_type != nullptr &&
                                       value_type->is_cooperative_vector() ?
                                   value_type->element() :
                                   nullptr;
                if (element == nullptr ||
                    (!element->is_int32() && !element->is_uint32() &&
                     !element->is_float32())) {
                    reason = "Metal cooperative-vector accumulation requires i32, u32, or f32 elements";
                    return false;
                }
                break;
            }
            reason = "unsupported resource write '" + luisa::string{xir::to_string(write->op())} + "'";
            return false;
        }
        default:
            reason = "unsupported instruction '" + luisa::string{xir::to_string(instruction->derived_instruction_tag())} + "'";
            return false;
    }
    return supported_type(instruction->type(), reason);
}

}// namespace detail

bool luisa_compute_metal_codegen_llvm_supported(
    const xir::Module &xir_module,
    luisa::string *reason) noexcept {
    return luisa_compute_metal_codegen_llvm_supported(
        xir_module, MetalCodegenLLVMConfig{}, reason);
}

bool luisa_compute_metal_codegen_llvm_supported(
    const xir::Module &xir_module,
    const MetalCodegenLLVMConfig &config,
    luisa::string *reason) noexcept {
    luisa::string local_reason;
    auto fail = [&](luisa::string message) noexcept {
        local_reason = std::move(message);
        if (reason != nullptr) { *reason = local_reason; }
        return false;
    };
    auto kernel_count = 0u;
    auto raster_stage_count = 0u;
    const xir::RasterStageFunction *raster_stage = nullptr;
    auto raster_depth_mode = detail::AIRRasterDepthMode::NONE;
    llvm::DenseSet<const xir::Function *> ray_pipeline_handlers;
    for (auto function : xir_module.function_list()) {
        if (auto definition = function->definition()) {
            definition->traverse_instructions(
            [&ray_pipeline_handlers](const xir::Instruction *instruction) noexcept {
                if (!instruction->isa<xir::RayQueryPipelineInst>()) { return; }
                auto pipeline = static_cast<const xir::RayQueryPipelineInst *>(
                    instruction);
                if (auto handler = pipeline->on_surface_function()) {
                    ray_pipeline_handlers.insert(handler);
                }
                if (auto handler = pipeline->on_procedural_function()) {
                    ray_pipeline_handlers.insert(handler);
                }
            });
        }
    }
    for (auto function : xir_module.function_list()) {
        if (function->derived_function_tag() == xir::DerivedFunctionTag::EXTERNAL &&
            (!function->name().has_value() || function->name()->empty())) {
            return fail("module contains an unnamed external function");
        }
        if (!detail::supported_type(function->type(), local_reason)) { return fail(std::move(local_reason)); }
        if (detail::is_ray_query_type(function->type())) {
            return fail("function returns an escaping ray-query object");
        }
        if (function->derived_function_tag() == xir::DerivedFunctionTag::KERNEL) { kernel_count++; }
        if (function->derived_function_tag() == xir::DerivedFunctionTag::RASTER_STAGE) {
            raster_stage_count++;
            raster_stage = static_cast<const xir::RasterStageFunction *>(function);
        }
        auto argument_index = 0u;
        for (auto argument : function->arguments()) {
            if (!detail::supported_type(argument->type(), local_reason)) { return fail(std::move(local_reason)); }
            if (detail::is_ray_query_type(argument->type())) {
                if (!ray_pipeline_handlers.contains(function) ||
                    argument_index != 0u || !argument->is_reference()) {
                    return fail("function has an escaping ray-query argument");
                }
            }
            if (argument->type()->is_texture() &&
                !detail::supported_texture_usage(argument, local_reason)) {
                return fail(std::move(local_reason));
            }
            if ((function->derived_function_tag() == xir::DerivedFunctionTag::KERNEL ||
                 function->derived_function_tag() == xir::DerivedFunctionTag::RASTER_STAGE) &&
                argument->is_reference() &&
                !detail::is_indirect_dispatch_buffer_type(argument->type())) {
                return fail("entry function has an unsupported reference argument");
            }
            argument_index++;
        }
        if (auto definition = function->definition()) {
            auto supported = true;
            definition->traverse_instructions([&](const xir::Instruction *instruction) noexcept {
                if (!supported) { return; }
                if (instruction->isa<xir::ThreadGroupInst>()) {
                    auto group = static_cast<const xir::ThreadGroupInst *>(instruction);
                    auto depth_mode = detail::air_raster_depth_mode(group->op());
                    if (depth_mode != detail::AIRRasterDepthMode::NONE) {
                        if (function->derived_function_tag() !=
                                xir::DerivedFunctionTag::RASTER_STAGE ||
                            static_cast<const xir::RasterStageFunction *>(function)->stage() !=
                                xir::RasterStage::FRAGMENT) {
                            local_reason =
                                "fragment depth output remained outside the fragment entry after inlining";
                            supported = false;
                            return;
                        }
                        if (raster_depth_mode != detail::AIRRasterDepthMode::NONE &&
                            raster_depth_mode != depth_mode) {
                            local_reason =
                                "fragment stage mixes incompatible shader-depth qualifiers";
                            supported = false;
                            return;
                        }
                        raster_depth_mode = depth_mode;
                    }
                }
                if (config.program != MetalAIRProgram::COMPUTE &&
                    instruction->isa<xir::DebugBreakInst>()) {
                    local_reason = "raster AIR does not support debug-break state";
                    supported = false;
                } else if (!detail::supported_instruction(
                               instruction, config, local_reason)) {
                    supported = false;
                }
            });
            if (!supported) { return fail(std::move(local_reason)); }
        }
    }
    if (config.program == MetalAIRProgram::COMPUTE) {
        if (kernel_count != 1u || raster_stage_count != 0u) {
            return fail("compute module must contain exactly one kernel and no raster stage");
        }
    } else {
        if (kernel_count != 0u || raster_stage_count != 1u || raster_stage == nullptr) {
            return fail("raster module must contain exactly one raster stage and no kernel");
        }
        auto expected_stage = config.program == MetalAIRProgram::RASTER_VERTEX ?
                                  xir::RasterStage::VERTEX :
                                  xir::RasterStage::FRAGMENT;
        if (raster_stage->stage() != expected_stage) {
            return fail("raster XIR stage identity does not match Metal AIR program mode");
        }
        luisa::vector<const xir::Argument *> stage_arguments;
        for (auto argument : raster_stage->arguments()) {
            stage_arguments.emplace_back(argument);
        }
        if (stage_arguments.empty()) {
            return fail("raster stage has no payload argument");
        }
        auto is_float4 = [](const Type *type) noexcept {
            return type != nullptr && type->is_float32_vector() &&
                   type->dimension() == 4u;
        };
        auto is_stage_value = [](const Type *type) noexcept {
            if (type == nullptr || !type->is_scalar_or_vector()) { return false; }
            auto element = type->is_vector() ? type->element() : type;
            return (element->is_float16() || element->is_float32() ||
                    element->is_int32() || element->is_uint32()) &&
                   (!type->is_vector() || type->dimension() <= 4u);
        };
        auto is_vertex_payload = [&](const Type *type) noexcept {
            if (is_float4(type)) { return true; }
            if (type == nullptr || !type->is_structure() ||
                type->members().empty() || !is_float4(type->members().front())) {
                return false;
            }
            for (auto member : type->members().subspan(1u)) {
                if (!is_stage_value(member)) { return false; }
            }
            return true;
        };
        if (config.program == MetalAIRProgram::RASTER_VERTEX) {
            auto app_data = stage_arguments.front()->type();
            if (app_data == nullptr || !app_data->is_structure() ||
                app_data->members().size() != 7u ||
                !app_data->members()[0u]->is_float32_vector() ||
                app_data->members()[0u]->dimension() != 3u ||
                !app_data->members()[1u]->is_float32_vector() ||
                app_data->members()[1u]->dimension() != 3u ||
                !app_data->members()[2u]->is_float32_vector() ||
                app_data->members()[2u]->dimension() != 4u ||
                !app_data->members()[3u]->is_float32_vector() ||
                app_data->members()[3u]->dimension() != 4u ||
                !app_data->members()[4u]->is_array() ||
                app_data->members()[4u]->dimension() != 4u ||
                !app_data->members()[4u]->element()->is_float32_vector() ||
                app_data->members()[4u]->element()->dimension() != 2u ||
                !app_data->members()[5u]->is_uint32() ||
                !app_data->members()[6u]->is_uint32()) {
                return fail("vertex payload is not the fixed Luisa AppData layout");
            }
            if (!is_vertex_payload(raster_stage->type())) {
                return fail("vertex return must be float4 or a structure beginning with float4");
            }
            if (!detail::validate_raster_interpolation(
                    raster_stage->type(), local_reason)) {
                return fail(std::move(local_reason));
            }
            std::array<bool, kVertexAttributeCount> seen{};
            for (auto attribute : config.raster.vertex_attributes) {
                auto semantic = static_cast<size_t>(attribute.semantic);
                if (semantic >= seen.size() || seen[semantic]) {
                    return fail("vertex attribute semantic is invalid or duplicated");
                }
                seen[semantic] = true;
                if (is_block_compressed(attribute.format)) {
                    return fail("block-compressed vertex attributes are unsupported");
                }
                if (attribute.format == PixelFormat::R10G10B10A2UInt ||
                    attribute.format == PixelFormat::R11G11B10F ||
                    attribute.format == PixelFormat::RGBA8SRGB) {
                    return fail("vertex attribute has no semantics-preserving Metal vertex format");
                }
            }
        } else {
            if (!is_vertex_payload(stage_arguments.front()->type())) {
                return fail("fragment payload must match a vertex float4 output shape");
            }
            if (!detail::validate_raster_interpolation(
                    stage_arguments.front()->type(), local_reason)) {
                return fail(std::move(local_reason));
            }
            auto color_type = raster_stage->type();
            if (color_type == nullptr) {
                if (raster_depth_mode == detail::AIRRasterDepthMode::NONE) {
                    return fail("void fragment stage must write shader depth");
                }
            } else if (color_type->is_structure()) {
                if (color_type->members().empty()) {
                    return fail("fragment color-target structure is empty");
                }
                if (color_type->members().size() > 8u) {
                    return fail("fragment stage returns more than 8 color targets");
                }
                for (auto member : color_type->members()) {
                    if (!is_stage_value(member)) {
                        return fail("fragment color target is not a supported scalar/vector");
                    }
                }
            } else if (!is_stage_value(color_type)) {
                return fail("fragment return is not a supported scalar/vector color target");
            }
        }
        auto stage_root_count = stage_arguments.size() - 1u;
        if (config.raster.stage_root_argument_offset >
                config.raster.root_arguments.size() ||
            stage_root_count >
                config.raster.root_arguments.size() -
                    config.raster.stage_root_argument_offset) {
            return fail("raster stage root-argument range is out of bounds");
        }
        for (auto i = 0u; i < stage_root_count; i++) {
            if (config.raster.root_arguments[config.raster.stage_root_argument_offset + i] !=
                stage_arguments[i + 1u]) {
                return fail("raster stage root arguments do not preserve AST order");
            }
        }
        for (auto argument : config.raster.root_arguments) {
            if (argument == nullptr ||
                !detail::supported_type(argument->type(), local_reason)) {
                return fail(local_reason.empty() ?
                                "raster root argument is invalid" :
                                std::move(local_reason));
            }
        }
    }
    if (reason != nullptr) { reason->clear(); }
    return true;
}

}// namespace luisa::compute::metal
