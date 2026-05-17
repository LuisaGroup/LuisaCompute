#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/constant.h>

#include <cmath>
#include <algorithm>

namespace luisa::compute::xir {

namespace detail {

// Helper to compute a scalar result. Returns true if foldable.
// data: output buffer; op0_data/op1_data/op2_data: inputs (may be nullptr if not used).
[[nodiscard]] static bool eval_scalar_op(const Type *type, ArithmeticOp op,
                                         void *data,
                                         const void *op0_data,
                                         const void *op1_data,
                                         const void *op2_data) noexcept {

    auto tag = type->tag();

    switch (op) {
        case ArithmeticOp::UNARY_MINUS: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = -*static_cast<const float *>(op0_data);
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = -*static_cast<const double *>(op0_data);
                    return true;
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = -(*static_cast<const int32_t *>(op0_data));
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = static_cast<uint32_t>(-static_cast<int32_t>(*static_cast<const uint32_t *>(op0_data)));
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::UNARY_BIT_NOT: {
            switch (tag) {
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = ~*static_cast<const int32_t *>(op0_data);
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = ~*static_cast<const uint32_t *>(op0_data);
                    return true;
                case Type::Tag::BOOL:
                    *static_cast<bool *>(data) = !*static_cast<const bool *>(op0_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::BINARY_ADD: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = *static_cast<const float *>(op0_data) + *static_cast<const float *>(op1_data);
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = *static_cast<const double *>(op0_data) + *static_cast<const double *>(op1_data);
                    return true;
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = *static_cast<const int32_t *>(op0_data) + *static_cast<const int32_t *>(op1_data);
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = *static_cast<const uint32_t *>(op0_data) + *static_cast<const uint32_t *>(op1_data);
                    return true;
                case Type::Tag::BOOL:
                    *static_cast<bool *>(data) = *static_cast<const bool *>(op0_data) || *static_cast<const bool *>(op1_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::BINARY_SUB: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = *static_cast<const float *>(op0_data) - *static_cast<const float *>(op1_data);
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = *static_cast<const double *>(op0_data) - *static_cast<const double *>(op1_data);
                    return true;
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = *static_cast<const int32_t *>(op0_data) - *static_cast<const int32_t *>(op1_data);
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = *static_cast<const uint32_t *>(op0_data) - *static_cast<const uint32_t *>(op1_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::BINARY_MUL: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = *static_cast<const float *>(op0_data) * *static_cast<const float *>(op1_data);
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = *static_cast<const double *>(op0_data) * *static_cast<const double *>(op1_data);
                    return true;
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = *static_cast<const int32_t *>(op0_data) * *static_cast<const int32_t *>(op1_data);
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = *static_cast<const uint32_t *>(op0_data) * *static_cast<const uint32_t *>(op1_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::BINARY_DIV: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = *static_cast<const float *>(op0_data) / *static_cast<const float *>(op1_data);
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = *static_cast<const double *>(op0_data) / *static_cast<const double *>(op1_data);
                    return true;
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = *static_cast<const int32_t *>(op0_data) / *static_cast<const int32_t *>(op1_data);
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = *static_cast<const uint32_t *>(op0_data) / *static_cast<const uint32_t *>(op1_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::BINARY_MOD: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = std::fmod(*static_cast<const float *>(op0_data), *static_cast<const float *>(op1_data));
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = std::fmod(*static_cast<const double *>(op0_data), *static_cast<const double *>(op1_data));
                    return true;
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = *static_cast<const int32_t *>(op0_data) % *static_cast<const int32_t *>(op1_data);
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = *static_cast<const uint32_t *>(op0_data) % *static_cast<const uint32_t *>(op1_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::BINARY_BIT_AND: {
            switch (tag) {
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = *static_cast<const int32_t *>(op0_data) & *static_cast<const int32_t *>(op1_data);
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = *static_cast<const uint32_t *>(op0_data) & *static_cast<const uint32_t *>(op1_data);
                    return true;
                case Type::Tag::BOOL:
                    *static_cast<bool *>(data) = *static_cast<const bool *>(op0_data) && *static_cast<const bool *>(op1_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::BINARY_BIT_OR: {
            switch (tag) {
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = *static_cast<const int32_t *>(op0_data) | *static_cast<const int32_t *>(op1_data);
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = *static_cast<const uint32_t *>(op0_data) | *static_cast<const uint32_t *>(op1_data);
                    return true;
                case Type::Tag::BOOL:
                    *static_cast<bool *>(data) = *static_cast<const bool *>(op0_data) || *static_cast<const bool *>(op1_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::BINARY_BIT_XOR: {
            switch (tag) {
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = *static_cast<const int32_t *>(op0_data) ^ *static_cast<const int32_t *>(op1_data);
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = *static_cast<const uint32_t *>(op0_data) ^ *static_cast<const uint32_t *>(op1_data);
                    return true;
                case Type::Tag::BOOL:
                    *static_cast<bool *>(data) = *static_cast<const bool *>(op0_data) != *static_cast<const bool *>(op1_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::BINARY_SHIFT_LEFT: {
            switch (tag) {
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = *static_cast<const int32_t *>(op0_data) << *static_cast<const int32_t *>(op1_data);
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = *static_cast<const uint32_t *>(op0_data) << *static_cast<const uint32_t *>(op1_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::BINARY_SHIFT_RIGHT: {
            switch (tag) {
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = *static_cast<const int32_t *>(op0_data) >> *static_cast<const int32_t *>(op1_data);
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = *static_cast<const uint32_t *>(op0_data) >> *static_cast<const uint32_t *>(op1_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::BINARY_LESS: {
            switch (tag) {
                case Type::Tag::BOOL: {
                    auto op_elem_type = type->element();// not used for scalar - check operand type
                    // Result is bool, operand is int or float. Check op0's scalar tag.
                    // For simplicity, we handle float and int32 cases
                }
                default: return false;
            }
            return false;
        }
        case ArithmeticOp::BINARY_GREATER: {
            return false;// handled in caller
        }
        case ArithmeticOp::BINARY_LESS_EQUAL: {
            return false;// handled in caller
        }
        case ArithmeticOp::BINARY_GREATER_EQUAL: {
            return false;// handled in caller
        }
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL: {
            return false;// handled in caller
        }

        case ArithmeticOp::ABS: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = std::abs(*static_cast<const float *>(op0_data));
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = std::abs(*static_cast<const double *>(op0_data));
                    return true;
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = std::abs(*static_cast<const int32_t *>(op0_data));
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = *static_cast<const uint32_t *>(op0_data);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::MIN: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = std::min(*static_cast<const float *>(op0_data), *static_cast<const float *>(op1_data));
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = std::min(*static_cast<const double *>(op0_data), *static_cast<const double *>(op1_data));
                    return true;
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = std::min(*static_cast<const int32_t *>(op0_data), *static_cast<const int32_t *>(op1_data));
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = std::min(*static_cast<const uint32_t *>(op0_data), *static_cast<const uint32_t *>(op1_data));
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::MAX: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = std::max(*static_cast<const float *>(op0_data), *static_cast<const float *>(op1_data));
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = std::max(*static_cast<const double *>(op0_data), *static_cast<const double *>(op1_data));
                    return true;
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = std::max(*static_cast<const int32_t *>(op0_data), *static_cast<const int32_t *>(op1_data));
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = std::max(*static_cast<const uint32_t *>(op0_data), *static_cast<const uint32_t *>(op1_data));
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::CLAMP: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = std::clamp(*static_cast<const float *>(op0_data), *static_cast<const float *>(op1_data), *static_cast<const float *>(op2_data));
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = std::clamp(*static_cast<const double *>(op0_data), *static_cast<const double *>(op1_data), *static_cast<const double *>(op2_data));
                    return true;
                case Type::Tag::INT32:
                    *static_cast<int32_t *>(data) = std::clamp(*static_cast<const int32_t *>(op0_data), *static_cast<const int32_t *>(op1_data), *static_cast<const int32_t *>(op2_data));
                    return true;
                case Type::Tag::UINT32:
                    *static_cast<uint32_t *>(data) = std::clamp(*static_cast<const uint32_t *>(op0_data), *static_cast<const uint32_t *>(op1_data), *static_cast<const uint32_t *>(op2_data));
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::SATURATE: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = std::clamp(*static_cast<const float *>(op0_data), 0.0f, 1.0f);
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = std::clamp(*static_cast<const double *>(op0_data), 0.0, 1.0);
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::LERP: {
            switch (tag) {
                case Type::Tag::FLOAT32: {
                    auto x = *static_cast<const float *>(op0_data);
                    auto y = *static_cast<const float *>(op1_data);
                    auto s = *static_cast<const float *>(op2_data);
                    *static_cast<float *>(data) = x * (1.0f - s) + y * s;
                    return true;
                }
                case Type::Tag::FLOAT64: {
                    auto x = *static_cast<const double *>(op0_data);
                    auto y = *static_cast<const double *>(op1_data);
                    auto s = *static_cast<const double *>(op2_data);
                    *static_cast<double *>(data) = x * (1.0 - s) + y * s;
                    return true;
                }
                default: return false;
            }
        }
        case ArithmeticOp::STEP: {
            switch (tag) {
                case Type::Tag::FLOAT32:
                    *static_cast<float *>(data) = (*static_cast<const float *>(op0_data) >= *static_cast<const float *>(op1_data)) ? 1.0f : 0.0f;
                    return true;
                case Type::Tag::FLOAT64:
                    *static_cast<double *>(data) = (*static_cast<const double *>(op0_data) >= *static_cast<const double *>(op1_data)) ? 1.0 : 0.0;
                    return true;
                default: return false;
            }
        }
        case ArithmeticOp::SMOOTHSTEP: {
            switch (tag) {
                case Type::Tag::FLOAT32: {
                    auto edge0 = *static_cast<const float *>(op0_data);
                    auto edge1 = *static_cast<const float *>(op1_data);
                    auto x = *static_cast<const float *>(op2_data);
                    auto t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
                    *static_cast<float *>(data) = t * t * (3.0f - 2.0f * t);
                    return true;
                }
                case Type::Tag::FLOAT64: {
                    auto edge0 = *static_cast<const double *>(op0_data);
                    auto edge1 = *static_cast<const double *>(op1_data);
                    auto x = *static_cast<const double *>(op2_data);
                    auto t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
                    *static_cast<double *>(data) = t * t * (3.0 - 2.0 * t);
                    return true;
                }
                default: return false;
            }
        }

        // Float unary math
        case ArithmeticOp::ACOS:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::acos(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::acos(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::ACOSH:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::acosh(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::acosh(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::ASIN:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::asin(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::asin(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::ASINH:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::asinh(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::asinh(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::ATAN:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::atan(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::atan(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::ATANH:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::atanh(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::atanh(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::COS:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::cos(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::cos(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::COSH:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::cosh(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::cosh(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::SIN:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::sin(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::sin(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::SINH:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::sinh(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::sinh(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::TAN:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::tan(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::tan(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::TANH:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::tanh(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::tanh(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::EXP:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::exp(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::exp(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::EXP2:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::exp2(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::exp2(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::LOG:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::log(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::log(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::LOG2:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::log2(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::log2(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::SQRT:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::sqrt(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::sqrt(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::RSQRT:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = 1.0f / std::sqrt(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = 1.0 / std::sqrt(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::CEIL:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::ceil(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::ceil(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::FLOOR:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::floor(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::floor(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::TRUNC:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::trunc(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::trunc(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::RINT:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::rint(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::rint(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::FRACT:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = *static_cast<const float *>(op0_data) - std::floor(*static_cast<const float *>(op0_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = *static_cast<const double *>(op0_data) - std::floor(*static_cast<const double *>(op0_data)); return true; }
            return false;
        case ArithmeticOp::ROUND:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::trunc(*static_cast<const float *>(op0_data) + std::copysign(0.5f, *static_cast<const float *>(op0_data))); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::trunc(*static_cast<const double *>(op0_data) + std::copysign(0.5, *static_cast<const double *>(op0_data))); return true; }
            return false;
        case ArithmeticOp::ATAN2:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::atan2(*static_cast<const float *>(op0_data), *static_cast<const float *>(op1_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::atan2(*static_cast<const double *>(op0_data), *static_cast<const double *>(op1_data)); return true; }
            return false;
        case ArithmeticOp::POW:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::pow(*static_cast<const float *>(op0_data), *static_cast<const float *>(op1_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::pow(*static_cast<const double *>(op0_data), *static_cast<const double *>(op1_data)); return true; }
            return false;
        case ArithmeticOp::POW_INT:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::pow(*static_cast<const float *>(op0_data), static_cast<float>(*static_cast<const int32_t *>(op1_data))); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::pow(*static_cast<const double *>(op0_data), static_cast<double>(*static_cast<const int32_t *>(op1_data))); return true; }
            return false;
        case ArithmeticOp::FMA:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::fma(*static_cast<const float *>(op0_data), *static_cast<const float *>(op1_data), *static_cast<const float *>(op2_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::fma(*static_cast<const double *>(op0_data), *static_cast<const double *>(op1_data), *static_cast<const double *>(op2_data)); return true; }
            return false;
        case ArithmeticOp::COPYSIGN:
            if (tag == Type::Tag::FLOAT32) { *static_cast<float *>(data) = std::copysign(*static_cast<const float *>(op0_data), *static_cast<const float *>(op1_data)); return true; }
            if (tag == Type::Tag::FLOAT64) { *static_cast<double *>(data) = std::copysign(*static_cast<const double *>(op0_data), *static_cast<const double *>(op1_data)); return true; }
            return false;

        case ArithmeticOp::ISINF: [[fallthrough]];
        case ArithmeticOp::ISNAN:
            // Handled in try_fold_scalar/try_fold_vector
            return false;

        default:
            return false;
    }
}

// Try to fold a scalar arithmetic instruction
[[nodiscard]] static Constant *try_fold_scalar(Module *module, const ArithmeticInst *inst) noexcept {
    auto type = inst->type();
    auto op = inst->op();

    // All operands must be Constant
    for (size_t i = 0; i < inst->operand_count(); ++i) {
        if (!inst->operand(i)->isa<Constant>()) { return nullptr; }
    }

    // ISINF/ISNAN: operand is float, result is bool
    if (op == ArithmeticOp::ISINF || op == ArithmeticOp::ISNAN) {
        auto op_type = inst->operand(0)->type();
        auto op_data = static_cast<const Constant *>(inst->operand(0))->data();
        bool result = false;
        if (op_type->is_float32()) {
            auto v = *static_cast<const float *>(op_data);
            result = (op == ArithmeticOp::ISINF) ? std::isinf(v) : std::isnan(v);
        } else if (op_type->is_float64()) {
            auto v = *static_cast<const double *>(op_data);
            result = (op == ArithmeticOp::ISINF) ? std::isinf(v) : std::isnan(v);
        } else {
            return nullptr;
        }
        return module->create_constant(Type::of<bool>(), &result);
    }

    // Comparisons: operand types may differ from result type (bool)
    if (op == ArithmeticOp::BINARY_LESS || op == ArithmeticOp::BINARY_GREATER ||
        op == ArithmeticOp::BINARY_LESS_EQUAL || op == ArithmeticOp::BINARY_GREATER_EQUAL ||
        op == ArithmeticOp::BINARY_EQUAL || op == ArithmeticOp::BINARY_NOT_EQUAL) {
        auto op_type = inst->operand(0)->type();
        auto op0_data = static_cast<const Constant *>(inst->operand(0))->data();
        auto op1_data = static_cast<const Constant *>(inst->operand(1))->data();
        bool result = false;
        if (op_type->is_float32()) {
            auto a = *static_cast<const float *>(op0_data);
            auto b = *static_cast<const float *>(op1_data);
            switch (op) {
                case ArithmeticOp::BINARY_LESS: result = a < b; break;
                case ArithmeticOp::BINARY_GREATER: result = a > b; break;
                case ArithmeticOp::BINARY_LESS_EQUAL: result = a <= b; break;
                case ArithmeticOp::BINARY_GREATER_EQUAL: result = a >= b; break;
                case ArithmeticOp::BINARY_EQUAL: result = a == b; break;
                case ArithmeticOp::BINARY_NOT_EQUAL: result = a != b; break;
                default: break;
            }
        } else if (op_type->is_float64()) {
            auto a = *static_cast<const double *>(op0_data);
            auto b = *static_cast<const double *>(op1_data);
            switch (op) {
                case ArithmeticOp::BINARY_LESS: result = a < b; break;
                case ArithmeticOp::BINARY_GREATER: result = a > b; break;
                case ArithmeticOp::BINARY_LESS_EQUAL: result = a <= b; break;
                case ArithmeticOp::BINARY_GREATER_EQUAL: result = a >= b; break;
                case ArithmeticOp::BINARY_EQUAL: result = a == b; break;
                case ArithmeticOp::BINARY_NOT_EQUAL: result = a != b; break;
                default: break;
            }
        } else if (op_type->is_int32()) {
            auto a = *static_cast<const int32_t *>(op0_data);
            auto b = *static_cast<const int32_t *>(op1_data);
            switch (op) {
                case ArithmeticOp::BINARY_LESS: result = a < b; break;
                case ArithmeticOp::BINARY_GREATER: result = a > b; break;
                case ArithmeticOp::BINARY_LESS_EQUAL: result = a <= b; break;
                case ArithmeticOp::BINARY_GREATER_EQUAL: result = a >= b; break;
                case ArithmeticOp::BINARY_EQUAL: result = a == b; break;
                case ArithmeticOp::BINARY_NOT_EQUAL: result = a != b; break;
                default: break;
            }
        } else if (op_type->is_uint32()) {
            auto a = *static_cast<const uint32_t *>(op0_data);
            auto b = *static_cast<const uint32_t *>(op1_data);
            switch (op) {
                case ArithmeticOp::BINARY_LESS: result = a < b; break;
                case ArithmeticOp::BINARY_GREATER: result = a > b; break;
                case ArithmeticOp::BINARY_LESS_EQUAL: result = a <= b; break;
                case ArithmeticOp::BINARY_GREATER_EQUAL: result = a >= b; break;
                case ArithmeticOp::BINARY_EQUAL: result = a == b; break;
                case ArithmeticOp::BINARY_NOT_EQUAL: result = a != b; break;
                default: break;
            }
        } else {
            return nullptr;
        }
        return module->create_constant(Type::of<bool>(), &result);
    }

    auto get_data = [&](size_t i) -> const void * {
        if (i >= inst->operand_count()) return nullptr;
        return static_cast<const Constant *>(inst->operand(i))->data();
    };

    auto size = type->size();
    luisa::vector<std::byte> result_data(size);
    std::memset(result_data.data(), 0, size);

    bool ok = eval_scalar_op(type, op,
                             result_data.data(),
                             get_data(0),
                             get_data(1),
                             get_data(2));
    if (!ok) { return nullptr; }
    return module->create_constant(type, result_data.data());
}

// Try to fold a vector arithmetic instruction element-wise
[[nodiscard]] static Constant *try_fold_vector(Module *module, const ArithmeticInst *inst) noexcept {
    auto type = inst->type();
    auto elem_type = type->element();
    auto dim = type->dimension();
    auto op = inst->op();

    // All operands must be Constant
    for (size_t i = 0; i < inst->operand_count(); ++i) {
        if (!inst->operand(i)->isa<Constant>()) { return nullptr; }
    }

    // ISINF/ISNAN: operand is float vector, result is bool vector
    if (op == ArithmeticOp::ISINF || op == ArithmeticOp::ISNAN) {
        auto op_type = inst->operand(0)->type();
        auto op_data = static_cast<const Constant *>(inst->operand(0))->data();
        auto op_elem_type = op_type->element();
        auto op_elem_size = op_elem_type->size();
        luisa::vector<std::byte> result_data(type->size());
        for (uint32_t i = 0; i < dim; ++i) {
            auto elem_data = static_cast<const std::byte *>(op_data) + i * op_elem_size;
            bool v = false;
            if (op_elem_type->is_float32()) {
                auto f = *static_cast<const float *>(static_cast<const void *>(elem_data));
                v = (op == ArithmeticOp::ISINF) ? std::isinf(f) : std::isnan(f);
            } else if (op_elem_type->is_float64()) {
                auto d = *static_cast<const double *>(static_cast<const void *>(elem_data));
                v = (op == ArithmeticOp::ISINF) ? std::isinf(d) : std::isnan(d);
            } else {
                return nullptr;
            }
            std::memcpy(result_data.data() + i, &v, 1);
        }
        return module->create_constant(type, result_data.data());
    }

    // Comparison ops: operand is float/int vector, result is bool vector
    if (op == ArithmeticOp::BINARY_LESS || op == ArithmeticOp::BINARY_GREATER ||
        op == ArithmeticOp::BINARY_LESS_EQUAL || op == ArithmeticOp::BINARY_GREATER_EQUAL ||
        op == ArithmeticOp::BINARY_EQUAL || op == ArithmeticOp::BINARY_NOT_EQUAL) {
        auto op_type = inst->operand(0)->type();
        auto op_elem_type = op_type->element();
        auto op_elem_size = op_elem_type->size();
        auto op0_data = static_cast<const Constant *>(inst->operand(0))->data();
        auto op1 = inst->operand(1);
        auto op1_type = op1->type();
        auto op1_data = static_cast<const Constant *>(op1)->data();
        luisa::vector<std::byte> result_data(type->size());
        for (uint32_t i = 0; i < dim; ++i) {
            auto elem0 = static_cast<const std::byte *>(op0_data) + i * op_elem_size;
            auto elem1 = op1_type->is_scalar()
                             ? static_cast<const std::byte *>(op1_data)
                             : static_cast<const std::byte *>(op1_data) + i * op_elem_size;
            bool v = false;
            if (op_elem_type->is_float32()) {
                auto a = *static_cast<const float *>(static_cast<const void *>(elem0));
                auto b = *static_cast<const float *>(static_cast<const void *>(elem1));
                switch (op) {
                    case ArithmeticOp::BINARY_LESS: v = a < b; break;
                    case ArithmeticOp::BINARY_GREATER: v = a > b; break;
                    case ArithmeticOp::BINARY_LESS_EQUAL: v = a <= b; break;
                    case ArithmeticOp::BINARY_GREATER_EQUAL: v = a >= b; break;
                    case ArithmeticOp::BINARY_EQUAL: v = a == b; break;
                    case ArithmeticOp::BINARY_NOT_EQUAL: v = a != b; break;
                    default: break;
                }
            } else if (op_elem_type->is_int32()) {
                auto a = *static_cast<const int32_t *>(static_cast<const void *>(elem0));
                auto b = *static_cast<const int32_t *>(static_cast<const void *>(elem1));
                switch (op) {
                    case ArithmeticOp::BINARY_LESS: v = a < b; break;
                    case ArithmeticOp::BINARY_GREATER: v = a > b; break;
                    case ArithmeticOp::BINARY_LESS_EQUAL: v = a <= b; break;
                    case ArithmeticOp::BINARY_GREATER_EQUAL: v = a >= b; break;
                    case ArithmeticOp::BINARY_EQUAL: v = a == b; break;
                    case ArithmeticOp::BINARY_NOT_EQUAL: v = a != b; break;
                    default: break;
                }
            } else if (op_elem_type->is_uint32()) {
                auto a = *static_cast<const uint32_t *>(static_cast<const void *>(elem0));
                auto b = *static_cast<const uint32_t *>(static_cast<const void *>(elem1));
                switch (op) {
                    case ArithmeticOp::BINARY_LESS: v = a < b; break;
                    case ArithmeticOp::BINARY_GREATER: v = a > b; break;
                    case ArithmeticOp::BINARY_LESS_EQUAL: v = a <= b; break;
                    case ArithmeticOp::BINARY_GREATER_EQUAL: v = a >= b; break;
                    case ArithmeticOp::BINARY_EQUAL: v = a == b; break;
                    case ArithmeticOp::BINARY_NOT_EQUAL: v = a != b; break;
                    default: break;
                }
            } else {
                return nullptr;
            }
            std::memcpy(result_data.data() + i, &v, 1);
        }
        return module->create_constant(type, result_data.data());
    }

    auto elem_size = elem_type->size();
    luisa::vector<std::byte> result_data(type->size());
    std::memset(result_data.data(), 0, type->size());

    // Get element data for a given operand index and component index
    // Scalars broadcast to all components
    auto get_elem = [&](size_t op_idx, uint32_t comp_idx) -> const void * {
        auto op = inst->operand(op_idx);
        auto op_type = op->type();
        auto op_data = static_cast<const Constant *>(op)->data();
        if (op_type->is_scalar()) {
            return op_data;// scalar broadcast
        }
        auto op_elem_size = op_type->element()->size();
        return static_cast<const std::byte *>(op_data) + comp_idx * op_elem_size;
    };

    for (uint32_t i = 0; i < dim; ++i) {
        auto dst = result_data.data() + i * elem_size;
        bool ok = eval_scalar_op(elem_type, op, dst,
                                 inst->operand_count() > 0 ? get_elem(0, i) : nullptr,
                                 inst->operand_count() > 1 ? get_elem(1, i) : nullptr,
                                 inst->operand_count() > 2 ? get_elem(2, i) : nullptr);
        if (!ok) { return nullptr; }
    }
    return module->create_constant(type, result_data.data());
}

// Try to fold arithmetic with all constant operands
[[nodiscard]] static Constant *try_fold_arithmetic(Module *module, const ArithmeticInst *inst) noexcept {
    auto type = inst->type();
    if (type == nullptr) { return nullptr; }

    // Don't fold aggregate/shuffle/extract/insert, reductions, matrix ops, cross/dot etc.
    switch (inst->op()) {
        case ArithmeticOp::AGGREGATE:
        case ArithmeticOp::SHUFFLE:
        case ArithmeticOp::INSERT:
        case ArithmeticOp::EXTRACT:
        case ArithmeticOp::ALL:
        case ArithmeticOp::ANY:
        case ArithmeticOp::REDUCE_SUM:
        case ArithmeticOp::REDUCE_PRODUCT:
        case ArithmeticOp::REDUCE_MIN:
        case ArithmeticOp::REDUCE_MAX:
        case ArithmeticOp::OUTER_PRODUCT:
        case ArithmeticOp::MATRIX_COMP_NEG:
        case ArithmeticOp::MATRIX_COMP_ADD:
        case ArithmeticOp::MATRIX_COMP_SUB:
        case ArithmeticOp::MATRIX_COMP_MUL:
        case ArithmeticOp::MATRIX_COMP_DIV:
        case ArithmeticOp::MATRIX_LINALG_MUL:
        case ArithmeticOp::MATRIX_DETERMINANT:
        case ArithmeticOp::MATRIX_TRANSPOSE:
        case ArithmeticOp::MATRIX_INVERSE:
        case ArithmeticOp::CROSS:
        case ArithmeticOp::DOT:
        case ArithmeticOp::LENGTH:
        case ArithmeticOp::LENGTH_SQUARED:
        case ArithmeticOp::NORMALIZE:
        case ArithmeticOp::FACEFORWARD:
        case ArithmeticOp::REFLECT:
        case ArithmeticOp::SELECT:
            return nullptr;
        default:
            break;
    }

    // No type means void or no result
    if (type == nullptr) { return nullptr; }

    if (type->is_scalar()) {
        return try_fold_scalar(module, inst);
    }
    if (type->is_vector()) {
        return try_fold_vector(module, inst);
    }
    return nullptr;
}

static void const_fold_pass_on_function(Function *function, ConstFoldInfo &info) noexcept {
    auto def = function->definition();
    if (def == nullptr) { return; }
    auto module = function->parent_module();

    luisa::vector<ArithmeticInst *> to_fold;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ArithmeticInst>()) {
            auto *ari = static_cast<ArithmeticInst *>(inst);
            if (ari->type() != nullptr) {
                // Check all operands are constants
                bool all_const = true;
                for (size_t i = 0; i < ari->operand_count(); ++i) {
                    if (!ari->operand(i)->isa<Constant>()) {
                        all_const = false;
                        break;
                    }
                }
                if (all_const && ari->operand_count() > 0) {
                    to_fold.push_back(ari);
                }
            }
        }
    });

    for (auto inst : to_fold) {
        auto folded = try_fold_arithmetic(module, inst);
        if (folded != nullptr) {
            inst->replace_all_uses_with(folded);
            inst->remove_self();
            info.folded_inst_count++;
        }
    }
}

}// namespace detail

ConstFoldInfo const_fold_pass_run_on_function(Function *function) noexcept {
    ConstFoldInfo info;
    detail::const_fold_pass_on_function(function, info);
    return info;
}

ConstFoldInfo const_fold_pass_run_on_module(Module *module) noexcept {
    ConstFoldInfo info;
    for (auto f : module->function_list()) {
        detail::const_fold_pass_on_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
