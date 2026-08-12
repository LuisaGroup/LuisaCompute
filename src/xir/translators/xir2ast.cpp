#include <algorithm>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/op.h>
#include <luisa/ast/constant_data.h>
#include <luisa/xir/module.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/assert.h>
#include <luisa/xir/instructions/assume.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/clock.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/print.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/thread_group.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_reg2mem.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/passes/gvn.h>
#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/lower_irreducible_cfg.h>
#include <luisa/xir/passes/lower_ray_query_loop_to_loop.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/post_dom_tree.h>
#include <luisa/xir/passes/promote_ref_arg.h>
#include <luisa/xir/passes/pass_verification.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/unused_callable_removal.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/translators/coro_xir2ast.h>
#include <luisa/xir/verifier.h>

#include <cstdlib>
#include <limits>
#include <type_traits>

#include "../pointer_containers.h"

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static CallOp xir2ast_make_vector_op(const Type *type) noexcept {
    LUISA_ASSERT(type->is_vector(), "Expected vector type, got {}.", type->description());
    auto elem = type->element();
    auto dim = type->dimension();
#define LUISA_XIR2AST_VEC_OP(T, PREFIX)               \
    if (elem == Type::of<T>()) {                      \
        switch (dim) {                                \
            case 2u: return CallOp::MAKE_##PREFIX##2; \
            case 3u: return CallOp::MAKE_##PREFIX##3; \
            case 4u: return CallOp::MAKE_##PREFIX##4; \
            default: break;                           \
        }                                             \
    }
    LUISA_XIR2AST_VEC_OP(bool, BOOL)
    LUISA_XIR2AST_VEC_OP(byte, BYTE)
    LUISA_XIR2AST_VEC_OP(ubyte, UBYTE)
    LUISA_XIR2AST_VEC_OP(short, SHORT)
    LUISA_XIR2AST_VEC_OP(ushort, USHORT)
    LUISA_XIR2AST_VEC_OP(int, INT)
    LUISA_XIR2AST_VEC_OP(uint, UINT)
    LUISA_XIR2AST_VEC_OP(slong, LONG)
    LUISA_XIR2AST_VEC_OP(ulong, ULONG)
    LUISA_XIR2AST_VEC_OP(half, HALF)
    LUISA_XIR2AST_VEC_OP(float, FLOAT)
    LUISA_XIR2AST_VEC_OP(double, DOUBLE)
#undef LUISA_XIR2AST_VEC_OP
    LUISA_ERROR_WITH_LOCATION("Unsupported vector maker type {}.", type->description());
}

[[nodiscard]] static CallOp xir2ast_make_matrix_op(const Type *type) noexcept {
    LUISA_ASSERT(type->is_matrix(), "Expected matrix type, got {}.", type->description());
    switch (type->dimension()) {
        case 2u: return CallOp::MAKE_FLOAT2X2;
        case 3u: return CallOp::MAKE_FLOAT3X3;
        case 4u: return CallOp::MAKE_FLOAT4X4;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported matrix maker type {}.", type->description());
}

[[nodiscard]] static CallOp xir2ast_arithmetic_call_op(ArithmeticOp op) noexcept {
    switch (op) {
        case ArithmeticOp::ALL: return CallOp::ALL;
        case ArithmeticOp::ANY: return CallOp::ANY;
        case ArithmeticOp::SELECT: return CallOp::SELECT;
        case ArithmeticOp::CLAMP: return CallOp::CLAMP;
        case ArithmeticOp::SATURATE: return CallOp::SATURATE;
        case ArithmeticOp::LERP: return CallOp::LERP;
        case ArithmeticOp::SMOOTHSTEP: return CallOp::SMOOTHSTEP;
        case ArithmeticOp::STEP: return CallOp::STEP;
        case ArithmeticOp::ABS: return CallOp::ABS;
        case ArithmeticOp::MIN: return CallOp::MIN;
        case ArithmeticOp::MAX: return CallOp::MAX;
        case ArithmeticOp::CLZ: return CallOp::CLZ;
        case ArithmeticOp::CTZ: return CallOp::CTZ;
        case ArithmeticOp::POPCOUNT: return CallOp::POPCOUNT;
        case ArithmeticOp::REVERSE: return CallOp::REVERSE;
        case ArithmeticOp::ISINF: return CallOp::ISINF;
        case ArithmeticOp::ISNAN: return CallOp::ISNAN;
        case ArithmeticOp::ACOS: return CallOp::ACOS;
        case ArithmeticOp::ACOSH: return CallOp::ACOSH;
        case ArithmeticOp::ASIN: return CallOp::ASIN;
        case ArithmeticOp::ASINH: return CallOp::ASINH;
        case ArithmeticOp::ATAN: return CallOp::ATAN;
        case ArithmeticOp::ATAN2: return CallOp::ATAN2;
        case ArithmeticOp::ATANH: return CallOp::ATANH;
        case ArithmeticOp::COS: return CallOp::COS;
        case ArithmeticOp::COSH: return CallOp::COSH;
        case ArithmeticOp::SIN: return CallOp::SIN;
        case ArithmeticOp::SINH: return CallOp::SINH;
        case ArithmeticOp::TAN: return CallOp::TAN;
        case ArithmeticOp::TANH: return CallOp::TANH;
        case ArithmeticOp::EXP: return CallOp::EXP;
        case ArithmeticOp::EXP2: return CallOp::EXP2;
        case ArithmeticOp::EXP10: return CallOp::EXP10;
        case ArithmeticOp::LOG: return CallOp::LOG;
        case ArithmeticOp::LOG2: return CallOp::LOG2;
        case ArithmeticOp::LOG10: return CallOp::LOG10;
        case ArithmeticOp::POW: [[fallthrough]];
        case ArithmeticOp::POW_INT: return CallOp::POW;
        case ArithmeticOp::SQRT: return CallOp::SQRT;
        case ArithmeticOp::RSQRT: return CallOp::RSQRT;
        case ArithmeticOp::CEIL: return CallOp::CEIL;
        case ArithmeticOp::FLOOR: return CallOp::FLOOR;
        case ArithmeticOp::FRACT: return CallOp::FRACT;
        case ArithmeticOp::TRUNC: return CallOp::TRUNC;
        case ArithmeticOp::ROUND: return CallOp::ROUND;
        case ArithmeticOp::RINT: return CallOp::RINT;
        case ArithmeticOp::FMA: return CallOp::FMA;
        case ArithmeticOp::COPYSIGN: return CallOp::COPYSIGN;
        case ArithmeticOp::CROSS: return CallOp::CROSS;
        case ArithmeticOp::DOT: return CallOp::DOT;
        case ArithmeticOp::LENGTH: return CallOp::LENGTH;
        case ArithmeticOp::LENGTH_SQUARED: return CallOp::LENGTH_SQUARED;
        case ArithmeticOp::NORMALIZE: return CallOp::NORMALIZE;
        case ArithmeticOp::FACEFORWARD: return CallOp::FACEFORWARD;
        case ArithmeticOp::REFLECT: return CallOp::REFLECT;
        case ArithmeticOp::REDUCE_SUM: return CallOp::REDUCE_SUM;
        case ArithmeticOp::REDUCE_PRODUCT: return CallOp::REDUCE_PRODUCT;
        case ArithmeticOp::REDUCE_MIN: return CallOp::REDUCE_MIN;
        case ArithmeticOp::REDUCE_MAX: return CallOp::REDUCE_MAX;
        case ArithmeticOp::OUTER_PRODUCT: return CallOp::OUTER_PRODUCT;
        case ArithmeticOp::MATRIX_COMP_MUL: return CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION;
        case ArithmeticOp::MATRIX_DETERMINANT: return CallOp::DETERMINANT;
        case ArithmeticOp::MATRIX_TRANSPOSE: return CallOp::TRANSPOSE;
        case ArithmeticOp::MATRIX_INVERSE: return CallOp::INVERSE;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported arithmetic call operation {}.", xir::to_string(op));
}

[[nodiscard]] static CallOp xir2ast_resource_query_op(ResourceQueryOp op) noexcept {
    switch (op) {
        case ResourceQueryOp::BUFFER_SIZE: return CallOp::BUFFER_SIZE;
        case ResourceQueryOp::BYTE_BUFFER_SIZE: return CallOp::BYTE_BUFFER_SIZE;
        case ResourceQueryOp::TEXTURE2D_SIZE: [[fallthrough]];
        case ResourceQueryOp::TEXTURE3D_SIZE: return CallOp::TEXTURE_SIZE;
        case ResourceQueryOp::BINDLESS_BUFFER_SIZE: return CallOp::BINDLESS_BUFFER_SIZE;
        case ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE: return CallOp::BINDLESS_BUFFER_SIZE;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE: return CallOp::BINDLESS_TEXTURE2D_SIZE;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE: return CallOp::BINDLESS_TEXTURE3D_SIZE;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: return CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: return CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL;
        case ResourceQueryOp::TEXTURE2D_SAMPLE: return CallOp::TEXTURE2D_SAMPLE;
        case ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL: return CallOp::TEXTURE2D_SAMPLE_LEVEL;
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD: return CallOp::TEXTURE2D_SAMPLE_GRAD;
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: return CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL;
        case ResourceQueryOp::TEXTURE3D_SAMPLE: return CallOp::TEXTURE3D_SAMPLE;
        case ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL: return CallOp::TEXTURE3D_SAMPLE_LEVEL;
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD: return CallOp::TEXTURE3D_SAMPLE_GRAD;
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: return CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: return CallOp::BINDLESS_TEXTURE2D_SAMPLE;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: return CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: return CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: return CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: return CallOp::BINDLESS_TEXTURE3D_SAMPLE;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: return CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: return CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: return CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: return CallOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: return CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: return CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: return CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: return CallOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: return CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: return CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: return CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
        case ResourceQueryOp::BUFFER_DEVICE_ADDRESS: return CallOp::BUFFER_ADDRESS;
        case ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: return CallOp::BINDLESS_BUFFER_ADDRESS;
        case ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM: return CallOp::RAY_TRACING_INSTANCE_TRANSFORM;
        case ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID: return CallOp::RAY_TRACING_INSTANCE_USER_ID;
        case ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: return CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK;
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: return CallOp::RAY_TRACING_TRACE_CLOSEST;
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY: return CallOp::RAY_TRACING_TRACE_ANY;
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL: return CallOp::RAY_TRACING_QUERY_ALL;
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY: return CallOp::RAY_TRACING_QUERY_ANY;
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: return CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX;
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT: return CallOp::RAY_TRACING_INSTANCE_MOTION_SRT;
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: return CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR;
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: return CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR;
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: return CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR;
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: return CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported resource query operation {}.", xir::to_string(op));
}

[[nodiscard]] static CallOp xir2ast_resource_query_op(
    ResourceQueryOp op, BindlessResourceAccess access) noexcept {
    auto base = xir2ast_resource_query_op(op);
    return specialize_bindless_resource_call(
        base, access.typed, access.uniform);
}

[[nodiscard]] static CallOp xir2ast_resource_read_op(ResourceReadOp op) noexcept {
    switch (op) {
        case ResourceReadOp::BUFFER_READ: return CallOp::BUFFER_READ;
        case ResourceReadOp::BUFFER_VOLATILE_READ: return CallOp::BUFFER_VOLATILE_READ;
        case ResourceReadOp::BYTE_BUFFER_READ: return CallOp::BYTE_BUFFER_READ;
        case ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: return CallOp::BYTE_BUFFER_VOLATILE_READ;
        case ResourceReadOp::TEXTURE2D_READ: [[fallthrough]];
        case ResourceReadOp::TEXTURE3D_READ: return CallOp::TEXTURE_READ;
        case ResourceReadOp::BINDLESS_BUFFER_READ: return CallOp::BINDLESS_BUFFER_READ;
        case ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: return CallOp::BINDLESS_BYTE_BUFFER_READ;
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ: return CallOp::BINDLESS_TEXTURE2D_READ;
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ: return CallOp::BINDLESS_TEXTURE3D_READ;
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL: return CallOp::BINDLESS_TEXTURE2D_READ_LEVEL;
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: return CallOp::BINDLESS_TEXTURE3D_READ_LEVEL;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported resource read operation {}.", xir::to_string(op));
}

[[nodiscard]] static CallOp xir2ast_resource_read_op(
    ResourceReadOp op, BindlessResourceAccess access) noexcept {
    auto base = xir2ast_resource_read_op(op);
    return specialize_bindless_resource_call(
        base, access.typed, access.uniform);
}

[[nodiscard]] static CallOp xir2ast_resource_write_op(ResourceWriteOp op) noexcept {
    switch (op) {
        case ResourceWriteOp::BUFFER_WRITE: return CallOp::BUFFER_WRITE;
        case ResourceWriteOp::BUFFER_VOLATILE_WRITE: return CallOp::BUFFER_VOLATILE_WRITE;
        case ResourceWriteOp::BYTE_BUFFER_WRITE: return CallOp::BYTE_BUFFER_WRITE;
        case ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: return CallOp::BYTE_BUFFER_VOLATILE_WRITE;
        case ResourceWriteOp::TEXTURE2D_WRITE: [[fallthrough]];
        case ResourceWriteOp::TEXTURE3D_WRITE: return CallOp::TEXTURE_WRITE;
        case ResourceWriteOp::BINDLESS_BUFFER_WRITE: return CallOp::BINDLESS_BUFFER_WRITE;
        case ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE:
            LUISA_ERROR_WITH_LOCATION(
                "XIR bindless byte-buffer writes cannot be represented in AST: "
                "the AST has no byte-addressed bindless write operation.");
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: return CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM;
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK: return CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY;
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY: return CallOp::RAY_TRACING_SET_INSTANCE_OPACITY;
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: return CallOp::RAY_TRACING_SET_INSTANCE_USER_ID;
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: return CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX;
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: return CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT;
        case ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL: return CallOp::INDIRECT_SET_DISPATCH_KERNEL;
        case ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: return CallOp::INDIRECT_SET_DISPATCH_COUNT;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported resource write operation {}.", xir::to_string(op));
}

[[nodiscard]] static CallOp xir2ast_resource_write_op(
    ResourceWriteOp op, BindlessResourceAccess access) noexcept {
    auto base = xir2ast_resource_write_op(op);
    return specialize_bindless_resource_call(
        base, access.typed, access.uniform);
}

[[nodiscard]] static CallOp xir2ast_ray_query_object_read_op(
    RayQueryObjectReadOp op) noexcept {
    switch (op) {
        case RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
            return CallOp::RAY_QUERY_WORLD_SPACE_RAY;
        case RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT:
            return CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT;
        case RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT:
            return CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT;
        case RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT:
            return CallOp::RAY_QUERY_COMMITTED_HIT;
        case RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE:
            return CallOp::RAY_QUERY_IS_TRIANGLE_CANDIDATE;
        case RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE:
            return CallOp::RAY_QUERY_IS_PROCEDURAL_CANDIDATE;
        case RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED:
            break;
    }
    LUISA_ERROR_WITH_LOCATION(
        "Ray-query termination state has no standalone AST operation; "
        "it must be paired with the immediately preceding proceed write.");
}

[[nodiscard]] static CallOp xir2ast_ray_query_object_write_op(
    RayQueryObjectWriteOp op) noexcept {
    switch (op) {
        case RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE:
            return CallOp::RAY_QUERY_COMMIT_TRIANGLE;
        case RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL:
            return CallOp::RAY_QUERY_COMMIT_PROCEDURAL;
        case RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE:
            return CallOp::RAY_QUERY_TERMINATE;
        case RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED:
            break;
    }
    LUISA_ERROR_WITH_LOCATION(
        "Ray-query proceed writes have no standalone AST operation; "
        "they must be paired with the immediately following termination read.");
}

[[nodiscard]] static CallOp xir2ast_atomic_op(AtomicOp op) noexcept {
    switch (op) {
        case AtomicOp::EXCHANGE: return CallOp::ATOMIC_EXCHANGE;
        case AtomicOp::COMPARE_EXCHANGE: return CallOp::ATOMIC_COMPARE_EXCHANGE;
        case AtomicOp::FETCH_ADD: return CallOp::ATOMIC_FETCH_ADD;
        case AtomicOp::FETCH_SUB: return CallOp::ATOMIC_FETCH_SUB;
        case AtomicOp::FETCH_AND: return CallOp::ATOMIC_FETCH_AND;
        case AtomicOp::FETCH_OR: return CallOp::ATOMIC_FETCH_OR;
        case AtomicOp::FETCH_XOR: return CallOp::ATOMIC_FETCH_XOR;
        case AtomicOp::FETCH_MIN: return CallOp::ATOMIC_FETCH_MIN;
        case AtomicOp::FETCH_MAX: return CallOp::ATOMIC_FETCH_MAX;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported atomic operation {}.", xir::to_string(op));
}

[[nodiscard]] static CallOp xir2ast_thread_group_op(ThreadGroupOp op) noexcept {
    switch (op) {
        case ThreadGroupOp::SHADER_EXECUTION_REORDER: return CallOp::SHADER_EXECUTION_REORDER;
        case ThreadGroupOp::RASTER_QUAD_DDX: return CallOp::DDX;
        case ThreadGroupOp::RASTER_QUAD_DDY: return CallOp::DDY;
        case ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: return CallOp::WARP_IS_FIRST_ACTIVE_LANE;
        case ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: return CallOp::WARP_FIRST_ACTIVE_LANE;
        case ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: return CallOp::WARP_ACTIVE_ALL_EQUAL;
        case ThreadGroupOp::WARP_ACTIVE_BIT_AND: return CallOp::WARP_ACTIVE_BIT_AND;
        case ThreadGroupOp::WARP_ACTIVE_BIT_OR: return CallOp::WARP_ACTIVE_BIT_OR;
        case ThreadGroupOp::WARP_ACTIVE_BIT_XOR: return CallOp::WARP_ACTIVE_BIT_XOR;
        case ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: return CallOp::WARP_ACTIVE_COUNT_BITS;
        case ThreadGroupOp::WARP_ACTIVE_MAX: return CallOp::WARP_ACTIVE_MAX;
        case ThreadGroupOp::WARP_ACTIVE_MIN: return CallOp::WARP_ACTIVE_MIN;
        case ThreadGroupOp::WARP_ACTIVE_PRODUCT: return CallOp::WARP_ACTIVE_PRODUCT;
        case ThreadGroupOp::WARP_ACTIVE_SUM: return CallOp::WARP_ACTIVE_SUM;
        case ThreadGroupOp::WARP_ACTIVE_ALL: return CallOp::WARP_ACTIVE_ALL;
        case ThreadGroupOp::WARP_ACTIVE_ANY: return CallOp::WARP_ACTIVE_ANY;
        case ThreadGroupOp::WARP_ACTIVE_BIT_MASK: return CallOp::WARP_ACTIVE_BIT_MASK;
        case ThreadGroupOp::WARP_PREFIX_COUNT_BITS: return CallOp::WARP_PREFIX_COUNT_BITS;
        case ThreadGroupOp::WARP_PREFIX_SUM: return CallOp::WARP_PREFIX_SUM;
        case ThreadGroupOp::WARP_PREFIX_PRODUCT: return CallOp::WARP_PREFIX_PRODUCT;
        case ThreadGroupOp::WARP_READ_LANE: return CallOp::WARP_READ_LANE;
        case ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: return CallOp::WARP_READ_FIRST_ACTIVE_LANE;
        case ThreadGroupOp::SYNCHRONIZE_BLOCK: return CallOp::SYNCHRONIZE_BLOCK;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported thread-group operation {}.", xir::to_string(op));
}

}// namespace detail

namespace {

void require_xir_for_ast(
    const XIRVerificationResult &verification,
    luisa::string_view stage) noexcept {
    if (!verification.succeeded()) {
        auto &first = verification.errors.front();
        auto function_name = first.function == nullptr ?
                                 luisa::string_view{"<none>"} :
                                 first.function->name().value_or("<unnamed>");
        auto instruction_name = first.instruction == nullptr ?
                                    luisa::string_view{"<none>"} :
                                    to_string(first.instruction->derived_instruction_tag());
        LUISA_ERROR_WITH_LOCATION(
            "Invalid XIR at XIR-to-AST {} in {} '{}' at {}: {} ({} error(s) total).",
            stage,
            first.function == nullptr ? luisa::string_view{"unknown"} :
                                        to_string(first.function->derived_function_tag()),
            function_name, instruction_name, first.message, verification.errors.size());
    }
}

void verify_xir_for_ast(const Function *function, luisa::string_view stage,
                        const XIRVerificationOptions &options = {}) noexcept {
    auto verification = xir_verify_function(function, options);
    require_xir_for_ast(verification, stage);
}

void verify_xir_for_ast(const Module *module, luisa::string_view stage,
                        const XIRVerificationOptions &options = {}) noexcept {
    auto verification = xir_verify_module(module, options);
    require_xir_for_ast(verification, stage);
}

}// namespace

class XIR2ASTContext {
private:
    // Bindings are short-lived translation state. No iterator or element
    // address survives insertion/erase, so node stability has no semantic
    // value here. Keep exact pointer identity while storing entries
    // contiguously: system-STL builds otherwise allocate one node and run the
    // general hash64 path for every materialized XIR value.
    using ValueMap = detail::DensePointerMap<
        const Value *, const Expression *>;
    struct ValueMapFrame {
        ValueMap map;
        luisa::vector<const Value *> insertion_log;
    };

    XIR2ASTConfig _config;
    const Module *_verified_module;
    luisa::shared_ptr<const ASTFunctionBuilder> _builder;
    luisa::unordered_map<const FunctionDefinition *, luisa::shared_ptr<const ASTFunctionBuilder>> _function_map;
    luisa::unordered_set<const FunctionDefinition *> _translating_functions;
    luisa::unordered_set<const BasicBlock *> _active_blocks;
    ValueMap _value_map;
    luisa::vector<const Value *> _value_map_insertion_log;
    luisa::vector<ValueMapFrame> _value_map_stack;
    struct LoopUpdateContext {
        const BasicBlock *prepare;
        const BasicBlock *update;
        bool emit_update_slice;
    };
    luisa::vector<LoopUpdateContext> _loop_update_stack;

private:
    void _verify_translation_input(
        const FunctionDefinition &function,
        luisa::string_view stage) noexcept {
        if (_verified_module != nullptr) {
            LUISA_ASSERT(
                function.parent_module() == _verified_module,
                "A whole-module-verified XIR-to-AST batch reached a function "
                "from a different module.");
            return;
        }
        if (auto statistics = _config.statistics) {
            statistics->function_verification_count++;
        }
        luisa::Clock verification_clock;
        verify_xir_for_ast(
            &function, stage,
            {.require_no_phi = true,
             .require_canonical_break_continue_targets = true});
        if (auto statistics = _config.statistics) {
            statistics->verification_milliseconds +=
                verification_clock.toc();
        }
    }

    void _bind_value(const Value *value,
                     const Expression *expression) noexcept {
        auto [iter, inserted] =
            _value_map.emplace(value, expression);
        LUISA_ASSERT(
            inserted || iter->second == expression,
            "XIR-to-AST attempted to change an existing value binding.");
        if (!inserted) { return; }
        _value_map_insertion_log.emplace_back(value);
        if (auto statistics = _config.statistics) {
            statistics->value_binding_insertions++;
            statistics->peak_value_map_size =
                std::max(statistics->peak_value_map_size,
                         _value_map.size());
        }
    }

    [[nodiscard]] ASTFunctionBuilder *_current_builder() const noexcept {
        return ASTFunctionBuilder::current();
    }

    [[nodiscard]] const Expression *_literal(const Constant *c) noexcept {
        auto b = _current_builder();
        auto type = c->type();
#define LUISA_XIR2AST_CONST(T) \
    if (type == Type::of<T>()) { return b->literal(type, c->as<T>()); }
#define LUISA_XIR2AST_CONST_VEC(T) \
    LUISA_XIR2AST_CONST(T)         \
    LUISA_XIR2AST_CONST(T##2)      \
    LUISA_XIR2AST_CONST(T##3)      \
    LUISA_XIR2AST_CONST(T##4)
        LUISA_XIR2AST_CONST_VEC(bool)
        LUISA_XIR2AST_CONST_VEC(byte)
        LUISA_XIR2AST_CONST_VEC(ubyte)
        LUISA_XIR2AST_CONST_VEC(short)
        LUISA_XIR2AST_CONST_VEC(ushort)
        LUISA_XIR2AST_CONST_VEC(int)
        LUISA_XIR2AST_CONST_VEC(uint)
        LUISA_XIR2AST_CONST_VEC(slong)
        LUISA_XIR2AST_CONST_VEC(ulong)
        LUISA_XIR2AST_CONST_VEC(half)
        LUISA_XIR2AST_CONST_VEC(float)
        LUISA_XIR2AST_CONST_VEC(double)
        LUISA_XIR2AST_CONST(float2x2)
        LUISA_XIR2AST_CONST(float3x3)
        LUISA_XIR2AST_CONST(float4x4)
#undef LUISA_XIR2AST_CONST_VEC
#undef LUISA_XIR2AST_CONST
        return b->constant(ConstantData::create(type, c->data(), type->size()));
    }

    [[nodiscard]] const Expression *_undefined(const Undefined *u) noexcept {
        auto type = u->type();
        LUISA_ASSERT(type != nullptr, "Cannot translate void undefined value to AST.");
        luisa::vector<std::byte> data(type->size());
        return _current_builder()->constant(ConstantData::create(type, data.data(), data.size()));
    }

    [[nodiscard]] const Expression *_special_register(const SpecialRegister *r) noexcept {
        auto b = _current_builder();
        switch (r->derived_special_register_tag()) {
            case DerivedSpecialRegisterTag::THREAD_ID: return b->thread_id();
            case DerivedSpecialRegisterTag::BLOCK_ID: return b->block_id();
            case DerivedSpecialRegisterTag::WARP_LANE_ID: return b->warp_lane_id();
            case DerivedSpecialRegisterTag::DISPATCH_ID: return b->dispatch_id();
            case DerivedSpecialRegisterTag::KERNEL_ID: return b->kernel_id();
            case DerivedSpecialRegisterTag::RASTER_OBJECT_ID: return b->raster_object_id();
            case DerivedSpecialRegisterTag::RASTER_BARYCENTRICS: return b->raster_barycentrics();
            case DerivedSpecialRegisterTag::BLOCK_SIZE: LUISA_ERROR_WITH_LOCATION("XIR-to-AST does not support block_size special register.");
            case DerivedSpecialRegisterTag::WARP_SIZE: return b->warp_lane_count();
            case DerivedSpecialRegisterTag::DISPATCH_SIZE: return b->dispatch_size();
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported special register {}.", xir::to_string(r->derived_special_register_tag()));
    }

    [[nodiscard]] luisa::vector<const Expression *> _operands(const User *user, size_t offset = 0u) noexcept {
        luisa::vector<const Expression *> args;
        args.reserve(user->operand_count() - offset);
        for (auto i = offset; i < user->operand_count(); i++) { args.emplace_back(_expr(user->operand(i))); }
        return args;
    }

    [[nodiscard]] static bool _is_canonical_ray_query_proceed_pair(
        const RayQueryObjectWriteInst *write,
        const RayQueryObjectReadInst *read) noexcept {
        return write != nullptr && read != nullptr &&
               write->op() ==
                   RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED &&
               read->op() ==
                   RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED &&
               write->operand_count() == 1u && read->operand_count() == 1u &&
               write->operand(0u) == read->operand(0u) &&
               write->next() == read && read->prev() == write;
    }

    void _emit_ray_query_object_write(
        const RayQueryObjectWriteInst *write) noexcept {
        if (write->op() ==
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED) {
            auto next = write->next();
            LUISA_ASSERT(
                next != nullptr && !next->is_sentinel() &&
                    next->isa<RayQueryObjectReadInst>() &&
                    _is_canonical_ray_query_proceed_pair(
                        write,
                        static_cast<const RayQueryObjectReadInst *>(next)),
                "XIR-to-AST requires each ray-query proceed write to be "
                "immediately followed by an is-terminated read of the same "
                "query object. The AST represents that pair as one stateful "
                "proceed expression.");
            // Emitted atomically when translating the paired read below. The
            // pair is adjacent, so moving the transition to the value-producing
            // half cannot cross another instruction or observable effect.
            return;
        }
        _current_builder()->call(
            detail::xir2ast_ray_query_object_write_op(write->op()),
            _operands(write));
    }

    [[nodiscard]] const Expression *_ray_query_object_read(
        const RayQueryObjectReadInst *read) noexcept {
        if (read->op() ==
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED) {
            auto prev = read->prev();
            LUISA_ASSERT(
                prev != nullptr && !prev->is_sentinel() &&
                    prev->isa<RayQueryObjectWriteInst>() &&
                    _is_canonical_ray_query_proceed_pair(
                        static_cast<const RayQueryObjectWriteInst *>(prev),
                        read),
                "XIR-to-AST cannot represent a standalone ray-query "
                "is-terminated read. It must immediately follow a proceed "
                "write of the same query object.");
            // AST proceed returns `active`, while normalized XIR exposes the
            // complementary `terminated` state. Materialize the stateful call
            // exactly once, then reconstruct the XIR value by negation.
            auto active = _materialize(
                Type::of<bool>(),
                _current_builder()->call(
                    Type::of<bool>(), CallOp::RAY_QUERY_PROCEED,
                    _operands(read)));
            return _current_builder()->unary(
                Type::of<bool>(), UnaryOp::NOT, active);
        }
        return _materialize(
            read->type(),
            _current_builder()->call(
                read->type(),
                detail::xir2ast_ray_query_object_read_op(read->op()),
                _operands(read)));
    }

    void _emit_ray_query_pipeline(
        const RayQueryPipelineInst *pipeline) noexcept {
        auto query = _expr(pipeline->query_object());
        LUISA_ASSERT(
            query->tag() == Expression::Tag::REF,
            "XIR ray-query pipeline object must translate to an AST "
            "reference expression, got expression tag {}.",
            luisa::to_underlying(query->tag()));
        auto query_ref = static_cast<const RefExpr *>(query);
        auto ray_query = _current_builder()->ray_query_(query_ref);

        auto emit_handler = [&](ScopeStmt *scope,
                                const Function *handler) noexcept {
            LUISA_ASSERT(
                handler != nullptr &&
                    handler->derived_function_tag() ==
                        DerivedFunctionTag::CALLABLE &&
                    handler->definition() != nullptr,
                "XIR ray-query pipeline handler must be a callable "
                "definition.");
            auto handler_builder = _translate_callable(
                *handler->definition());
            auto args = luisa::vector<const Expression *>{};
            args.reserve(pipeline->captured_argument_count() + 1u);
            args.emplace_back(query_ref);
            for (auto i = 0u;
                 i < pipeline->captured_argument_count(); ++i) {
                args.emplace_back(
                    _expr(pipeline->captured_argument(i)));
            }
            LUISA_ASSERT(
                args.size() == handler_builder->arguments().size(),
                "XIR ray-query pipeline handler argument mismatch: "
                "pipeline provides {}, callable expects {}.",
                args.size(), handler_builder->arguments().size());
            _current_builder()->with(scope, [&] {
                _current_builder()->call(
                    handler_builder->function(), args);
            });
        };
        emit_handler(ray_query->on_triangle_candidate(),
                     pipeline->on_surface_function());
        emit_handler(ray_query->on_procedural_candidate(),
                     pipeline->on_procedural_function());
    }

    [[nodiscard]] const Expression *_gep(const GEPInst *inst) noexcept {
        auto expr = _expr(inst->base());
        for (auto i = 0u; i < inst->index_count(); i++) {
            auto type = expr->type();
            if (type->is_structure()) {
                auto index_value = _constant_uint(inst->index(i));
                LUISA_ASSERT(index_value < type->members().size(), "Struct member index out of range.");
                expr = _current_builder()->member(type->members()[index_value], expr, index_value);
            } else if (type->is_array() || type->is_vector()) {
                expr = _current_builder()->access(type->element(), expr, _expr(inst->index(i)));
            } else if (type->is_matrix()) {
                auto column_type = Type::vector(type->element(), type->dimension());
                expr = _current_builder()->access(column_type, expr, _expr(inst->index(i)));
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid GEP base type {}.", type->description());
            }
        }
        LUISA_ASSERT(expr->type() == inst->type(), "GEP result type mismatch: {} vs {}.", expr->type()->description(), inst->type()->description());
        return expr;
    }

    [[nodiscard]] uint64_t _constant_uint(const Value *v) noexcept {
        uint64_t result = 0u;
        LUISA_ASSERT(try_decode_constant_nonnegative_integer(v, result),
                     "Expected a nonnegative integer constant index.");
        return result;
    }

    [[nodiscard]] const Expression *_arithmetic(const ArithmeticInst *inst) noexcept {
        auto b = _current_builder();
        auto args = _operands(inst);
        switch (inst->op()) {
            case ArithmeticOp::UNARY_MINUS: return b->unary(inst->type(), UnaryOp::MINUS, args[0]);
            case ArithmeticOp::UNARY_BIT_NOT: return b->unary(inst->type(), inst->type()->is_bool_or_bool_vector() ? UnaryOp::NOT : UnaryOp::BIT_NOT, args[0]);
            case ArithmeticOp::BINARY_ADD: return b->binary(inst->type(), BinaryOp::ADD, args[0], args[1]);
            case ArithmeticOp::BINARY_SUB: return b->binary(inst->type(), BinaryOp::SUB, args[0], args[1]);
            case ArithmeticOp::BINARY_MUL: return b->binary(inst->type(), BinaryOp::MUL, args[0], args[1]);
            case ArithmeticOp::BINARY_DIV: return b->binary(inst->type(), BinaryOp::DIV, args[0], args[1]);
            case ArithmeticOp::BINARY_MOD: return b->binary(inst->type(), BinaryOp::MOD, args[0], args[1]);
            case ArithmeticOp::BINARY_BIT_AND: return b->binary(inst->type(), inst->type()->is_bool_or_bool_vector() ? BinaryOp::AND : BinaryOp::BIT_AND, args[0], args[1]);
            case ArithmeticOp::BINARY_BIT_OR: return b->binary(inst->type(), inst->type()->is_bool_or_bool_vector() ? BinaryOp::OR : BinaryOp::BIT_OR, args[0], args[1]);
            case ArithmeticOp::BINARY_BIT_XOR: return b->binary(inst->type(), BinaryOp::BIT_XOR, args[0], args[1]);
            case ArithmeticOp::BINARY_SHIFT_LEFT: return b->binary(inst->type(), BinaryOp::SHL, args[0], args[1]);
            case ArithmeticOp::BINARY_SHIFT_RIGHT: return b->binary(inst->type(), BinaryOp::SHR, args[0], args[1]);
            case ArithmeticOp::BINARY_LESS: return b->binary(inst->type(), BinaryOp::LESS, args[0], args[1]);
            case ArithmeticOp::BINARY_GREATER: return b->binary(inst->type(), BinaryOp::GREATER, args[0], args[1]);
            case ArithmeticOp::BINARY_LESS_EQUAL: return b->binary(inst->type(), BinaryOp::LESS_EQUAL, args[0], args[1]);
            case ArithmeticOp::BINARY_GREATER_EQUAL: return b->binary(inst->type(), BinaryOp::GREATER_EQUAL, args[0], args[1]);
            case ArithmeticOp::BINARY_EQUAL: return b->binary(inst->type(), BinaryOp::EQUAL, args[0], args[1]);
            case ArithmeticOp::BINARY_NOT_EQUAL: return b->binary(inst->type(), BinaryOp::NOT_EQUAL, args[0], args[1]);
            // AST binary multiplication is linear algebra whenever either
            // operand has the matrix/vector shape recognized by AST-to-XIR.
            // Component-wise matrix multiplication remains the explicit
            // MATRIX_COMPONENT_WISE_MULTIPLICATION call below. Keeping these
            // cases at the BinaryExpr boundary makes the two translators exact
            // inverses for every matrix arithmetic opcode.
            case ArithmeticOp::MATRIX_COMP_NEG: return b->unary(inst->type(), UnaryOp::MINUS, args[0]);
            case ArithmeticOp::MATRIX_COMP_ADD: return b->binary(inst->type(), BinaryOp::ADD, args[0], args[1]);
            case ArithmeticOp::MATRIX_COMP_SUB: return b->binary(inst->type(), BinaryOp::SUB, args[0], args[1]);
            case ArithmeticOp::MATRIX_COMP_DIV: return b->binary(inst->type(), BinaryOp::DIV, args[0], args[1]);
            case ArithmeticOp::MATRIX_LINALG_MUL: return b->binary(inst->type(), BinaryOp::MUL, args[0], args[1]);
            case ArithmeticOp::AGGREGATE: {
                if (inst->type()->is_vector()) { return b->call(inst->type(), detail::xir2ast_make_vector_op(inst->type()), args); }
                if (inst->type()->is_matrix()) { return b->call(inst->type(), detail::xir2ast_make_matrix_op(inst->type()), args); }
                if (inst->type()->is_structure()) {
                    LUISA_ASSERT(args.size() == inst->type()->members().size(), "Struct aggregate member count mismatch.");
                    auto tmp = b->local(inst->type());
                    for (auto i = 0u; i < args.size(); i++) {
                        b->assign(b->member(inst->type()->members()[i], tmp, i), args[i]);
                    }
                    return tmp;
                }
                if (inst->type()->is_array()) {
                    LUISA_ASSERT(args.size() == inst->type()->dimension(), "Array aggregate element count mismatch.");
                    auto tmp = b->local(inst->type());
                    for (auto i = 0u; i < args.size(); i++) {
                        auto index = b->literal(Type::of<uint32_t>(), i);
                        b->assign(b->access(inst->type()->element(), tmp, index), args[i]);
                    }
                    return tmp;
                }
                LUISA_ERROR_WITH_LOCATION("Invalid aggregate type {}.", inst->type()->description());
            }
            case ArithmeticOp::SHUFFLE: {
                LUISA_ASSERT(args.size() >= 2u, "Shuffle requires a source and at least one index.");
                auto source = args.front();
                auto swizzle_size = args.size() - 1u;
                auto swizzle_code = 0ull;
                for (auto i = 0u; i < swizzle_size; i++) { swizzle_code |= _constant_uint(inst->operand(i + 1u)) << (i * 4u); }
                return b->swizzle(inst->type(), source, swizzle_size, swizzle_code);
            }
            case ArithmeticOp::EXTRACT: {
                LUISA_ASSERT(args.size() >= 2u, "Extract requires a source and at least one index.");
                auto expr = args.front();
                for (auto i = 1u; i < args.size(); i++) {
                    auto type = expr->type();
                    if (type->is_structure()) {
                        auto index = _constant_uint(inst->operand(i));
                        expr = b->member(type->members()[index], expr, index);
                    } else if (type->is_array() || type->is_vector()) {
                        expr = b->access(type->element(), expr, args[i]);
                    } else if (type->is_matrix()) {
                        auto column_type = Type::vector(type->element(), type->dimension());
                        expr = b->access(column_type, expr, args[i]);
                    } else {
                        LUISA_ERROR_WITH_LOCATION("Invalid extract base type {}.", type->description());
                    }
                }
                return expr;
            }
            case ArithmeticOp::INSERT: {
                LUISA_ASSERT(args.size() >= 3u, "Insert requires a base, a value, and at least one index.");
                auto tmp = b->local(inst->type());
                b->assign(tmp, args[0]);
                const Expression *lhs = tmp;
                for (auto i = 2u; i < args.size(); i++) {
                    auto type = lhs->type();
                    if (type->is_structure()) {
                        auto index = _constant_uint(inst->operand(i));
                        LUISA_ASSERT(index < type->members().size(), "Struct member index out of range.");
                        lhs = b->member(type->members()[index], lhs, index);
                    } else if (type->is_array() || type->is_vector()) {
                        lhs = b->access(type->element(), lhs, args[i]);
                    } else if (type->is_matrix()) {
                        auto column_type = Type::vector(type->element(), type->dimension());
                        lhs = b->access(column_type, lhs, args[i]);
                    } else {
                        LUISA_ERROR_WITH_LOCATION("Invalid insert base type {}.", type->description());
                    }
                }
                b->assign(lhs, args[1]);
                return tmp;
            }
            default: return b->call(inst->type(), detail::xir2ast_arithmetic_call_op(inst->op()), args);
        }
    }

    [[nodiscard]] const Expression *_expr(const Value *value) noexcept {
        if (auto iter = _value_map.find(value); iter != _value_map.end()) { return iter->second; }
        auto expr = [&]() noexcept -> const Expression * {
            switch (value->derived_value_tag()) {
                case DerivedValueTag::UNDEFINED: return _undefined(static_cast<const Undefined *>(value));
                case DerivedValueTag::CONSTANT: return _literal(static_cast<const Constant *>(value));
                case DerivedValueTag::SPECIAL_REGISTER: return _special_register(static_cast<const SpecialRegister *>(value));
                case DerivedValueTag::ARGUMENT: LUISA_ERROR_WITH_LOCATION("Unmapped argument.");
                case DerivedValueTag::INSTRUCTION: break;
                default: LUISA_ERROR_WITH_LOCATION("Unsupported XIR value tag {}.", xir::to_string(value->derived_value_tag()));
            }
            auto inst = static_cast<const Instruction *>(value);
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::ALLOCA: {
                    auto alloca = static_cast<const AllocaInst *>(inst);
                    return alloca->is_shared() ? _current_builder()->shared(alloca->type()) : _current_builder()->local(alloca->type());
                }
                case DerivedInstructionTag::LOAD: {
                    auto load = static_cast<const LoadInst *>(inst);
                    auto tmp = _current_builder()->local(load->type());
                    _current_builder()->assign(tmp, _expr(load->variable()));
                    return tmp;
                }
                case DerivedInstructionTag::GEP: return _gep(static_cast<const GEPInst *>(inst));
                case DerivedInstructionTag::ARITHMETIC: return _arithmetic(static_cast<const ArithmeticInst *>(inst));
                case DerivedInstructionTag::CAST: {
                    auto cast = static_cast<const CastInst *>(inst);
                    auto op = cast->op() == CastOp::STATIC_CAST ? compute::CastOp::STATIC : compute::CastOp::BITWISE;
                    return _current_builder()->cast(cast->type(), op, _expr(cast->value()));
                }
                case DerivedInstructionTag::CALL: {
                    auto call = static_cast<const CallInst *>(inst);
                    LUISA_ASSERT(call->type() != nullptr, "Void call cannot be used as an expression.");
                    auto callee = call->callee();
                    if (callee->derived_function_tag() != DerivedFunctionTag::CALLABLE) {
                        LUISA_ERROR_WITH_LOCATION("XIR-to-AST only supports calls to callable definitions.");
                    }
                    auto callee_def = static_cast<const FunctionDefinition *>(callee);
                    auto callee_builder = _translate_callable(*callee_def);
                    luisa::vector<const Expression *> args;
                    args.reserve(call->argument_count());
                    for (auto i = 0u; i < call->argument_count(); i++) { args.emplace_back(_expr(call->argument(i))); }
                    return _materialize(call->type(), _current_builder()->call(call->type(), callee_builder->function(), args));
                }
                case DerivedInstructionTag::RESOURCE_QUERY:
                    return _resource_query(
                        static_cast<const ResourceQueryInst *>(inst));
                case DerivedInstructionTag::RESOURCE_READ: {
                    auto read = static_cast<const ResourceReadInst *>(inst);
                    return _materialize(
                        inst->type(),
                        _current_builder()->call(
                            inst->type(),
                            detail::xir2ast_resource_read_op(
                                read->op(), read->bindless_access()),
                            _operands(inst)));
                }
                case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
                    return _ray_query_object_read(
                        static_cast<const RayQueryObjectReadInst *>(inst));
                case DerivedInstructionTag::ATOMIC: return _materialize(inst->type(), _atomic(static_cast<const AtomicInst *>(inst)));
                case DerivedInstructionTag::THREAD_GROUP: {
                    auto expr = _current_builder()->call(inst->type(), detail::xir2ast_thread_group_op(static_cast<const ThreadGroupInst *>(inst)->op()), _operands(inst));
                    return inst->type() == nullptr ? expr : _materialize(inst->type(), expr);
                }
                case DerivedInstructionTag::CLOCK: return _materialize(Type::of<uint64_t>(), _current_builder()->call(Type::of<uint64_t>(), CallOp::CLOCK, {}));
                case DerivedInstructionTag::ASSERT: return _assert_or_assume(static_cast<const AssertInst *>(inst));
                case DerivedInstructionTag::ASSUME: return _assert_or_assume(static_cast<const AssumeInst *>(inst));
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Unsupported expression instruction {}.", xir::to_string(inst->derived_instruction_tag()));
        }();
        _bind_value(value, expr);
        return expr;
    }

    [[nodiscard]] const Expression *_atomic(const AtomicInst *inst) noexcept {
        auto args = luisa::vector<const Expression *>{_expr(inst->base())};
        for (auto use : inst->index_uses()) { args.emplace_back(_expr(use->value())); }
        for (auto use : inst->value_uses()) { args.emplace_back(_expr(use->value())); }
        return _current_builder()->call(inst->type(), detail::xir2ast_atomic_op(inst->op()), args);
    }

    [[nodiscard]] const Expression *_materialize(const Type *type, const Expression *expr) noexcept {
        LUISA_ASSERT(type != nullptr && expr != nullptr, "Cannot materialize a void expression.");
        auto tmp = _current_builder()->local(type);
        _current_builder()->assign(tmp, expr);
        return tmp;
    }

    [[nodiscard]] const Expression *_resource_query(
        const ResourceQueryInst *inst) noexcept {
        auto args = _operands(inst);
        if (inst->op() == ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE) {
            // AST represents byte-buffer size with the element-size query and
            // an explicit one-byte stride. Preserve the XIR byte unit instead
            // of silently emitting a malformed two-argument AST call.
            args.emplace_back(
                _current_builder()->literal(Type::of<uint32_t>(), 1u));
        }
        return _current_builder()->call(
            inst->type(),
            detail::xir2ast_resource_query_op(
                inst->op(), inst->bindless_access()),
            args);
    }

    template<typename F>
    void _with_value_map_checkpoint(F &&f) noexcept {
        // Value bindings are monotone along one structured CFG path: a value
        // is inserted at most once and never rebound. A checkpoint therefore
        // needs only the insertion-log length. Restoring erases exactly the
        // branch-local suffix, so nested checkpoints compose and the work is
        // O(number of bindings created in the branch), independent of the
        // number of dominating bindings retained across it.
        auto checkpoint = _value_map_insertion_log.size();
        ValueMap oracle_snapshot;
        if (_config.verify_value_map_checkpoints) {
            oracle_snapshot = _value_map;
        }
        if (auto statistics = _config.statistics) {
            statistics->value_map_checkpoint_count++;
        }
        f();
        auto rollback_work =
            _value_map_insertion_log.size() - checkpoint;
        for (auto i = _value_map_insertion_log.size();
             i > checkpoint; --i) {
            auto *value = _value_map_insertion_log[i - 1u];
            LUISA_ASSERT(
                _value_map.erase(value) == 1u,
                "XIR-to-AST value checkpoint lost a logged binding.");
        }
        _value_map_insertion_log.resize(checkpoint);
        if (auto statistics = _config.statistics) {
            statistics->value_map_rollback_work += rollback_work;
        }
        if (_config.verify_value_map_checkpoints) {
            LUISA_ASSERT(
                _value_map.size() == oracle_snapshot.size(),
                "XIR-to-AST incremental checkpoint changed the retained value-map size ({} != {}).",
                _value_map.size(), oracle_snapshot.size());
            for (auto [value, expression] : oracle_snapshot) {
                auto iter = _value_map.find(value);
                LUISA_ASSERT(
                    iter != _value_map.end() &&
                        iter->second == expression,
                    "XIR-to-AST incremental checkpoint changed a retained value binding.");
            }
        }
    }

    void _emit_selection_path(
        const BasicBlock *target,
        const BasicBlock *stop) noexcept {
        if (target == nullptr || target == stop) { return; }
        // Restructuring represents a physical loop-boundary guard as a
        // structured selection without an OpSelectionMerge. One arm may point
        // directly at the enclosing loop's canonical update block instead of
        // owning a ContinueInst proxy. Recursing through that block as an
        // ordinary selection arm walks back to the loop prepare and invents an
        // unstructured cycle. Preserve the edge semantics explicitly: execute
        // the update slice once, then continue the AST loop.
        if (!_loop_update_stack.empty() &&
            target == _loop_update_stack.back().update) {
            auto context = _loop_update_stack.back();
            if (context.emit_update_slice) {
                _emit_block(context.update, context.prepare);
            }
            _current_builder()->continue_();
            return;
        }
        _emit_block(target, stop);
    }

    [[nodiscard]] luisa::optional<const BasicBlock *> _find_conditional_branch_merge(
        const ConditionalBranchInst *br, const BasicBlock *stop) noexcept {
        auto true_block = br->true_block();
        auto false_block = br->false_block();
        auto branch_target = [](const BasicBlock *block) noexcept -> const BasicBlock * {
            auto term = block->terminator();
            return term != nullptr && term->isa<BranchInst>() ?
                       static_cast<const BranchInst *>(term)->target_block() :
                       nullptr;
        };
        auto true_target = true_block == stop ? stop : branch_target(true_block);
        auto false_target = false_block == stop ? stop : branch_target(false_block);
        if (true_block == stop || false_block == stop) { return stop; }
        if (true_target != nullptr && true_target == false_block) { return false_block; }
        if (false_target != nullptr && false_target == true_block) { return true_block; }
        if (true_target == false_target) { return luisa::optional<const BasicBlock *>{true_target}; }
        auto *function = const_cast<Function *>(br->parent_function());
        auto pdom = compute_post_dom_tree(function);
        auto *parent = const_cast<BasicBlock *>(br->parent_block());
        auto *merge = pdom.immediate_post_dominator(parent);
        if (merge == parent) { return luisa::nullopt; }
        if (merge != nullptr) { return merge; }
        if (stop != nullptr && pdom.contains(const_cast<BasicBlock *>(stop)) &&
            pdom.post_dominates(const_cast<BasicBlock *>(stop), parent)) {
            return stop;
        }
        return luisa::nullopt;
    }

    template<typename T>
    [[nodiscard]] const Expression *_assert_or_assume(const T *inst) noexcept {
        auto b = _current_builder();
        auto args = luisa::vector<const Expression *>{_expr(inst->condition())};
        if (!inst->message().empty()) { args.emplace_back(b->string_id(luisa::string{inst->message()})); }
        auto op = std::is_same_v<T, AssertInst> ? CallOp::ASSERT : CallOp::ASSUME;
        return b->call(nullptr, op, args);
    }

    [[nodiscard]] const RefExpr *_declare_bound_argument(const Argument *arg, const compute::Function::Binding &binding) noexcept {
        auto b = _current_builder();
        return luisa::visit(
            [&]<typename B>(B &&bound) noexcept -> const RefExpr * {
                using T = std::remove_cvref_t<B>;
                if constexpr (std::is_same_v<T, compute::Function::BufferBinding>) {
                    return b->buffer_binding(arg->type(), bound.handle, bound.offset, bound.size);
                } else if constexpr (std::is_same_v<T, compute::Function::TextureBinding>) {
                    return b->texture_binding(arg->type(), bound.handle, bound.level);
                } else if constexpr (std::is_same_v<T, compute::Function::BindlessArrayBinding>) {
                    return b->bindless_array_binding(bound.handle);
                } else if constexpr (std::is_same_v<T, compute::Function::AccelBinding>) {
                    return b->accel_binding(bound.handle);
                } else {
                    LUISA_ERROR_WITH_LOCATION("Unexpected unbound argument binding.");
                }
            },
            binding);
    }

    void _declare_arguments(const FunctionDefinition &f) noexcept {
        auto b = _current_builder();
        auto index = 0u;
        for (auto arg : f.arguments()) {
            const RefExpr *expr = nullptr;
            if (index < _config.bound_arguments.size() &&
                !luisa::holds_alternative<luisa::monostate>(_config.bound_arguments[index])) {
                expr = _declare_bound_argument(arg, _config.bound_arguments[index]);
            } else if (arg->is_resource()) {
                if (arg->type()->is_buffer()) {
                    expr = b->buffer(arg->type());
                } else if (arg->type()->is_texture()) {
                    expr = b->texture(arg->type());
                } else if (arg->type()->is_bindless_array()) {
                    expr = b->bindless_array();
                } else if (arg->type()->is_accel()) {
                    expr = b->accel();
                } else {
                    LUISA_ERROR_WITH_LOCATION("Unsupported resource argument type {}.", arg->type()->description());
                }
            } else if (arg->is_reference()) {
                expr = b->reference(arg->type());
                b->mark_variable_usage(expr->variable().uid(), Usage::READ_WRITE);
            } else {
                expr = b->argument(arg->type());
            }
            _bind_value(arg, expr);
            index++;
        }
    }

    void _predeclare_allocas(const BasicBlock *block) noexcept {
        if (block == nullptr) { return; }
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::ALLOCA: static_cast<void>(_expr(inst)); break;
                case DerivedInstructionTag::IF: {
                    auto if_inst = static_cast<const IfInst *>(inst);
                    _predeclare_allocas(if_inst->true_block());
                    _predeclare_allocas(if_inst->false_block());
                    _predeclare_allocas(if_inst->merge_block());
                    break;
                }
                case DerivedInstructionTag::LOOP: {
                    auto loop = static_cast<const LoopInst *>(inst);
                    _predeclare_allocas(loop->prepare_block());
                    _predeclare_allocas(loop->body_block());
                    _predeclare_allocas(loop->update_block());
                    _predeclare_allocas(loop->merge_block());
                    break;
                }
                case DerivedInstructionTag::SIMPLE_LOOP: {
                    auto loop = static_cast<const SimpleLoopInst *>(inst);
                    _predeclare_allocas(loop->body_block());
                    _predeclare_allocas(loop->merge_block());
                    break;
                }
                case DerivedInstructionTag::SWITCH: {
                    auto sw = static_cast<const SwitchInst *>(inst);
                    for (auto i = 0u; i < sw->case_count(); i++) { _predeclare_allocas(sw->case_block(i)); }
                    _predeclare_allocas(sw->default_block());
                    _predeclare_allocas(sw->merge_block());
                    break;
                }
                default: break;
            }
        }
    }

    void _emit_loop_prepare_prefix(const BasicBlock *block) noexcept {
        for (auto inst : block->instructions()) {
            if (inst->is_terminator()) { return; }
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::ALLOCA: break;
                case DerivedInstructionTag::STORE: {
                    auto store = static_cast<const StoreInst *>(inst);
                    _current_builder()->assign(_expr(store->variable()), _expr(store->value()));
                    break;
                }
                case DerivedInstructionTag::PRINT: {
                    auto print = static_cast<const PrintInst *>(inst);
                    _current_builder()->print_(luisa::string{print->format()}, _operands(print));
                    break;
                }
                case DerivedInstructionTag::RESOURCE_WRITE: {
                    auto write = static_cast<const ResourceWriteInst *>(inst);
                    _current_builder()->call(
                        detail::xir2ast_resource_write_op(
                            write->op(), write->bindless_access()),
                        _operands(write));
                    break;
                }
                case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: {
                    _emit_ray_query_object_write(
                        static_cast<const RayQueryObjectWriteInst *>(inst));
                    break;
                }
                case DerivedInstructionTag::RAY_QUERY_PIPELINE: {
                    _emit_ray_query_pipeline(
                        static_cast<const RayQueryPipelineInst *>(inst));
                    break;
                }
                case DerivedInstructionTag::ASSERT: [[fallthrough]];
                case DerivedInstructionTag::ASSUME: [[fallthrough]];
                case DerivedInstructionTag::ATOMIC: [[fallthrough]];
                case DerivedInstructionTag::THREAD_GROUP: [[fallthrough]];
                case DerivedInstructionTag::RESOURCE_QUERY: [[fallthrough]];
                case DerivedInstructionTag::RESOURCE_READ: [[fallthrough]];
                case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: [[fallthrough]];
                case DerivedInstructionTag::LOAD: [[fallthrough]];
                case DerivedInstructionTag::GEP: [[fallthrough]];
                case DerivedInstructionTag::CLOCK: [[fallthrough]];
                case DerivedInstructionTag::ARITHMETIC: [[fallthrough]];
                case DerivedInstructionTag::CAST: {
                    static_cast<void>(_expr(inst));
                    break;
                }
                case DerivedInstructionTag::CALL: {
                    auto call = static_cast<const CallInst *>(inst);
                    if (call->type() == nullptr) {
                        auto callee = call->callee();
                        if (callee->derived_function_tag() != DerivedFunctionTag::CALLABLE) {
                            LUISA_ERROR_WITH_LOCATION("XIR-to-AST only supports calls to callable definitions.");
                        }
                        auto callee_def = static_cast<const FunctionDefinition *>(callee);
                        auto callee_builder = _translate_callable(*callee_def);
                        auto args = luisa::vector<const Expression *>{};
                        args.reserve(call->argument_count());
                        for (auto i = 0u; i < call->argument_count(); i++) { args.emplace_back(_expr(call->argument(i))); }
                        _current_builder()->call(callee_builder->function(), args);
                    } else {
                        static_cast<void>(_expr(inst));
                    }
                    break;
                }
                default: LUISA_ERROR_WITH_LOCATION("Unsupported loop prepare instruction {}.", xir::to_string(inst->derived_instruction_tag()));
            }
        }
    }

    struct ForLoopPattern {
        const Value *variable;
        const Value *condition;
        const Value *step;
    };

    [[nodiscard]] luisa::optional<ForLoopPattern> _match_for_loop(const LoopInst *loop) noexcept {
        auto prepare = loop->prepare_block();
        for (auto inst : prepare->instructions()) {
            if (!inst->is_terminator()) { return luisa::nullopt; }
        }
        auto prepare_term = prepare->terminator();
        if (prepare_term == nullptr || !prepare_term->isa<ConditionalBranchInst>()) { return luisa::nullopt; }
        auto cond_br = static_cast<const ConditionalBranchInst *>(prepare_term);
        if (cond_br->true_block() != loop->body_block() || cond_br->false_block() != loop->merge_block()) { return luisa::nullopt; }
        auto update = loop->update_block();
        auto update_term = update->terminator();
        if (update_term == nullptr || !update_term->isa<BranchInst>()) { return luisa::nullopt; }
        if (static_cast<const BranchInst *>(update_term)->target_block() != prepare) { return luisa::nullopt; }
        const StoreInst *store = nullptr;
        for (auto inst : update->instructions()) {
            if (inst->is_terminator()) { break; }
            if (inst->isa<StoreInst>()) {
                if (store != nullptr) { return luisa::nullopt; }
                store = static_cast<const StoreInst *>(inst);
            }
        }
        if (store == nullptr || !store->value()->isa<ArithmeticInst>()) { return luisa::nullopt; }
        auto add = static_cast<const ArithmeticInst *>(store->value());
        if (add->op() != ArithmeticOp::BINARY_ADD || add->operand_count() != 2u) { return luisa::nullopt; }
        auto match_step = [&](const Value *lhs, const Value *rhs) noexcept -> const Value * {
            if (lhs->isa<LoadInst>() && static_cast<const LoadInst *>(lhs)->variable() == store->variable()) { return rhs; }
            return nullptr;
        };
        auto step = match_step(add->operand(0u), add->operand(1u));
        if (step == nullptr) { step = match_step(add->operand(1u), add->operand(0u)); }
        if (step == nullptr) { return luisa::nullopt; }
        return ForLoopPattern{store->variable(), cond_br->condition(), step};
    }

    void _emit_block(const BasicBlock *block, const BasicBlock *stop = nullptr) noexcept {
        if (block == nullptr || block == stop) { return; }
        if (!_active_blocks.emplace(block).second) { return; }
        struct ActiveBlockGuard {
            luisa::unordered_set<const BasicBlock *> &active_blocks;
            const BasicBlock *block;
            ~ActiveBlockGuard() noexcept { active_blocks.erase(block); }
        } active_guard{_active_blocks, block};
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::ALLOCA: break;
                case DerivedInstructionTag::STORE: {
                    auto store = static_cast<const StoreInst *>(inst);
                    _current_builder()->assign(_expr(store->variable()), _expr(store->value()));
                    break;
                }
                case DerivedInstructionTag::PRINT: {
                    auto print = static_cast<const PrintInst *>(inst);
                    _current_builder()->print_(luisa::string{print->format()}, _operands(print));
                    break;
                }
                case DerivedInstructionTag::RESOURCE_WRITE: {
                    auto write = static_cast<const ResourceWriteInst *>(inst);
                    _current_builder()->call(
                        detail::xir2ast_resource_write_op(
                            write->op(), write->bindless_access()),
                        _operands(write));
                    break;
                }
                case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: {
                    _emit_ray_query_object_write(
                        static_cast<const RayQueryObjectWriteInst *>(inst));
                    break;
                }
                case DerivedInstructionTag::RAY_QUERY_PIPELINE: {
                    _emit_ray_query_pipeline(
                        static_cast<const RayQueryPipelineInst *>(inst));
                    break;
                }
                case DerivedInstructionTag::ASSERT: [[fallthrough]];
                case DerivedInstructionTag::ASSUME: [[fallthrough]];
                case DerivedInstructionTag::ATOMIC: [[fallthrough]];
                case DerivedInstructionTag::THREAD_GROUP: [[fallthrough]];
                case DerivedInstructionTag::RESOURCE_QUERY: [[fallthrough]];
                case DerivedInstructionTag::RESOURCE_READ: [[fallthrough]];
                case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: [[fallthrough]];
                case DerivedInstructionTag::LOAD: [[fallthrough]];
                case DerivedInstructionTag::GEP: [[fallthrough]];
                case DerivedInstructionTag::CLOCK: [[fallthrough]];
                case DerivedInstructionTag::ARITHMETIC: [[fallthrough]];
                case DerivedInstructionTag::CAST: {
                    static_cast<void>(_expr(inst));
                    break;
                }
                case DerivedInstructionTag::CALL: {
                    auto call = static_cast<const CallInst *>(inst);
                    if (call->type() == nullptr) {
                        auto callee = call->callee();
                        if (callee->derived_function_tag() != DerivedFunctionTag::CALLABLE) {
                            LUISA_ERROR_WITH_LOCATION("XIR-to-AST only supports calls to callable definitions.");
                        }
                        auto callee_def = static_cast<const FunctionDefinition *>(callee);
                        auto callee_builder = _translate_callable(*callee_def);
                        auto args = luisa::vector<const Expression *>{};
                        args.reserve(call->argument_count());
                        for (auto i = 0u; i < call->argument_count(); i++) { args.emplace_back(_expr(call->argument(i))); }
                        _current_builder()->call(callee_builder->function(), args);
                    } else {
                        static_cast<void>(_expr(inst));
                    }
                    break;
                }
                case DerivedInstructionTag::RETURN: {
                    auto ret = static_cast<const ReturnInst *>(inst);
                    _current_builder()->return_(ret->return_value() == nullptr ? nullptr : _expr(ret->return_value()));
                    return;
                }
                case DerivedInstructionTag::BREAK: {
                    _current_builder()->break_();
                    return;
                }
                case DerivedInstructionTag::CONTINUE: {
                    auto continue_inst = static_cast<const ContinueInst *>(inst);
                    if (continue_inst->target_block() == stop) { return; }
                    if (!_loop_update_stack.empty() && continue_inst->target_block() == _loop_update_stack.back().update) {
                        auto context = _loop_update_stack.back();
                        if (context.emit_update_slice) {
                            _emit_block(context.update, context.prepare);
                        }
                    }
                    _current_builder()->continue_();
                    return;
                }
                case DerivedInstructionTag::UNREACHABLE: {
                    auto un = static_cast<const UnreachableInst *>(inst);
                    auto args = luisa::vector<const Expression *>{};
                    if (!un->message().empty()) { args.emplace_back(_current_builder()->string_id(luisa::string{un->message()})); }
                    _current_builder()->call(CallOp::UNREACHABLE, args);
                    return;
                }
                case DerivedInstructionTag::BRANCH: {
                    auto br = static_cast<const BranchInst *>(inst);
                    if (br->target_block() != stop) { _emit_block(br->target_block(), stop); }
                    return;
                }
                case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                    auto br = static_cast<const ConditionalBranchInst *>(inst);
                    auto true_block = br->true_block();
                    auto false_block = br->false_block();
                    auto merge = _find_conditional_branch_merge(br, stop);
                    if (!merge) {
                        auto *function = br->parent_function();
                        auto block_index = [&](const BasicBlock *needle) noexcept {
                            auto index = size_t{0u};
                            if (function != nullptr) {
                                for (auto *candidate : function->basic_blocks()) {
                                    if (candidate == needle) { return index; }
                                    ++index;
                                }
                            }
                            return std::numeric_limits<size_t>::max();
                        };
                        auto name = function == nullptr ?
                                        luisa::optional<luisa::string_view>{} :
                                        function->name();
                        if (function != nullptr) {
                            auto trace = false;
                            if (auto *value = std::getenv(
                                    "LUISA_XIR_TRACE_PASSES")) {
                                trace = luisa::string_view{value} == "1";
                            }
                            auto block_count = size_t{0u};
                            auto instruction_count = size_t{0u};
                            for (auto *block : function->basic_blocks()) {
                                ++block_count;
                                for ([[maybe_unused]] auto *instruction :
                                     block->instructions()) {
                                    ++instruction_count;
                                }
                            }
                            if (trace && block_count <= 128u &&
                                instruction_count <= 4096u) {
                                luisa::string dump;
                                XIRDebugPrinter printer;
                                printer.emit_function(dump, function);
                                LUISA_VERBOSE_WITH_LOCATION(
                                    "XIR-to-AST unstructured function "
                                    "dump:\n{}",
                                    dump);
                            }
                        }
                        LUISA_ERROR_WITH_LOCATION(
                            "XIR-to-AST requires structured control flow in "
                            "function '{}' at conditional block {} "
                            "(true={}, false={}, stop={}).",
                            name ? *name : luisa::string_view{"<unnamed>"},
                            block_index(br->parent_block()),
                            block_index(true_block),
                            block_index(false_block),
                            block_index(stop));
                    }
                    auto merge_block = *merge;
                    auto ast_if = _current_builder()->if_(_expr(br->condition()));
                    if (true_block != merge_block) {
                        _current_builder()->with(ast_if->true_branch(), [&] {
                            _with_value_map_checkpoint([&] {
                                _emit_selection_path(
                                    true_block, merge_block);
                            });
                        });
                    }
                    if (false_block != merge_block) {
                        _current_builder()->with(ast_if->false_branch(), [&] {
                            _with_value_map_checkpoint([&] {
                                _emit_selection_path(
                                    false_block, merge_block);
                            });
                        });
                    }
                    if (merge_block != stop) { _emit_block(merge_block, stop); }
                    return;
                }
                case DerivedInstructionTag::IF: {
                    auto if_inst = static_cast<const IfInst *>(inst);
                    auto ast_if = _current_builder()->if_(_expr(if_inst->condition()));
                    _current_builder()->with(ast_if->true_branch(), [&] {
                        _with_value_map_checkpoint([&] {
                            _emit_selection_path(
                                if_inst->true_block(),
                                if_inst->merge_block());
                        });
                    });
                    _current_builder()->with(ast_if->false_branch(), [&] {
                        _with_value_map_checkpoint([&] {
                            _emit_selection_path(
                                if_inst->false_block(),
                                if_inst->merge_block());
                        });
                    });
                    _emit_block(if_inst->merge_block(), stop);
                    return;
                }
                case DerivedInstructionTag::SIMPLE_LOOP: {
                    auto loop = static_cast<const SimpleLoopInst *>(inst);
                    auto ast_loop = _current_builder()->loop_();
                    _loop_update_stack.emplace_back(
                        LoopUpdateContext{
                            loop->body_block(),
                            loop->body_block(),
                            false});
                    _current_builder()->with(ast_loop->body(), [&] {
                        _with_value_map_checkpoint([&] { _emit_block(loop->body_block(), loop->merge_block()); });
                    });
                    _loop_update_stack.pop_back();
                    _emit_block(loop->merge_block(), stop);
                    return;
                }
                case DerivedInstructionTag::SWITCH: {
                    auto sw = static_cast<const SwitchInst *>(inst);
                    auto ast_switch = _current_builder()->switch_(_expr(sw->value()));
                    _current_builder()->with(ast_switch->body(), [&] {
                        for (auto i = 0u; i < sw->case_count(); i++) {
                            auto selector_type = sw->value()->type();
                            auto case_value = sw->case_value(i);
                            auto case_expr = [&]() noexcept -> const LiteralExpr * {
                                switch (selector_type->tag()) {
                                    case Type::Tag::BOOL:
                                        return _current_builder()->literal(
                                            selector_type, static_cast<bool>(case_value));
                                    case Type::Tag::INT8:
                                        return _current_builder()->literal(
                                            selector_type, luisa::bit_cast<int8_t>(
                                                               static_cast<uint8_t>(case_value)));
                                    case Type::Tag::UINT8:
                                        return _current_builder()->literal(
                                            selector_type, static_cast<uint8_t>(case_value));
                                    case Type::Tag::INT16:
                                        return _current_builder()->literal(
                                            selector_type, luisa::bit_cast<int16_t>(
                                                               static_cast<uint16_t>(case_value)));
                                    case Type::Tag::UINT16:
                                        return _current_builder()->literal(
                                            selector_type, static_cast<uint16_t>(case_value));
                                    case Type::Tag::INT32:
                                        return _current_builder()->literal(
                                            selector_type, luisa::bit_cast<int32_t>(
                                                               static_cast<uint32_t>(case_value)));
                                    case Type::Tag::UINT32:
                                        return _current_builder()->literal(
                                            selector_type, static_cast<uint32_t>(case_value));
                                    case Type::Tag::INT64:
                                        return _current_builder()->literal(
                                            selector_type, luisa::bit_cast<int64_t>(case_value));
                                    case Type::Tag::UINT64:
                                        return _current_builder()->literal(selector_type, case_value);
                                    default:
                                        LUISA_ERROR_WITH_LOCATION(
                                            "Invalid XIR switch selector type {}.",
                                            selector_type->description());
                                }
                            }();
                            auto ast_case = _current_builder()->case_(case_expr);
                            _current_builder()->with(ast_case->body(), [&] {
                                _with_value_map_checkpoint([&] {
                                    _emit_selection_path(
                                        sw->case_block(i),
                                        sw->merge_block());
                                });
                            });
                        }
                        auto ast_default = _current_builder()->default_();
                        _current_builder()->with(ast_default->body(), [&] {
                            _with_value_map_checkpoint([&] {
                                _emit_selection_path(
                                    sw->default_block(),
                                    sw->merge_block());
                            });
                        });
                    });
                    _emit_block(sw->merge_block(), stop);
                    return;
                }
                case DerivedInstructionTag::LOOP: {
                    auto loop = static_cast<const LoopInst *>(inst);
                    if (auto for_loop = _match_for_loop(loop)) {
                        auto ast_for = _current_builder()->for_(_expr(for_loop->variable), _expr(for_loop->condition), _expr(for_loop->step));
                        _loop_update_stack.emplace_back(
                            LoopUpdateContext{
                                loop->prepare_block(),
                                loop->update_block(),
                                false});
                        _current_builder()->with(ast_for->body(), [&] {
                            _with_value_map_checkpoint([&] { _emit_block(loop->body_block(), loop->update_block()); });
                        });
                        _loop_update_stack.pop_back();
                        _emit_block(loop->merge_block(), stop);
                        return;
                    }
                    auto ast_loop = _current_builder()->loop_();
                    _loop_update_stack.emplace_back(
                        LoopUpdateContext{
                            loop->prepare_block(),
                            loop->update_block(),
                            true});
                    _current_builder()->with(ast_loop->body(), [&] {
                        _with_value_map_checkpoint([&] {
                            _emit_loop_prepare_prefix(loop->prepare_block());
                            auto term = loop->prepare_block()->terminator();
                            if (term != nullptr && term->isa<ConditionalBranchInst>()) {
                                auto cond_br = static_cast<const ConditionalBranchInst *>(term);
                                if (cond_br->true_block() != loop->body_block() || cond_br->false_block() != loop->merge_block()) {
                                    LUISA_ERROR_WITH_LOCATION("XIR-to-AST requires canonical LoopInst prepare cond_br targets.");
                                }
                                auto break_if = _current_builder()->if_(_current_builder()->unary(Type::of<bool>(), UnaryOp::NOT, _expr(cond_br->condition())));
                                _current_builder()->with(break_if->true_branch(), [&] { _current_builder()->break_(); });
                            }
                            _emit_block(loop->body_block(), loop->update_block());
                            _emit_block(loop->update_block(), loop->prepare_block());
                        });
                    });
                    _loop_update_stack.pop_back();
                    _emit_block(loop->merge_block(), stop);
                    return;
                }
                case DerivedInstructionTag::PHI: LUISA_ERROR_WITH_LOCATION("XIR-to-AST does not support PHI nodes. Run reg2mem first.");
                default: LUISA_ERROR_WITH_LOCATION("Unsupported statement instruction {}.", xir::to_string(inst->derived_instruction_tag()));
            }
        }
    }

    [[nodiscard]] luisa::shared_ptr<const ASTFunctionBuilder> _translate(const FunctionDefinition &f) noexcept {
        auto build = [&] {
            _declare_arguments(f);
            if (f.derived_function_tag() == DerivedFunctionTag::KERNEL) { _current_builder()->set_block_size(static_cast<const KernelFunction &>(f).block_size()); }
            _predeclare_allocas(f.body_block());
            _emit_block(f.body_block());
        };
        switch (f.derived_function_tag()) {
            case DerivedFunctionTag::KERNEL: return ASTFunctionBuilder::define_kernel(build);
            case DerivedFunctionTag::CALLABLE: return ASTFunctionBuilder::define_callable(build);
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Cannot translate external XIR function to AST.");
    }

    [[nodiscard]] luisa::shared_ptr<const ASTFunctionBuilder> _translate_callable(const FunctionDefinition &f) noexcept {
        LUISA_ASSERT(f.derived_function_tag() == DerivedFunctionTag::CALLABLE, "Expected callable function.");
        if (auto iter = _function_map.find(&f); iter != _function_map.end()) {
            if (auto statistics = _config.statistics) {
                statistics->function_cache_hits++;
            }
            return iter->second;
        }
        _verify_translation_input(
            f, "callable translation input");
        if (!_translating_functions.emplace(&f).second) {
            LUISA_ERROR_WITH_LOCATION("Recursive XIR callables are not supported by XIR-to-AST.");
        }
        _value_map_stack.emplace_back(ValueMapFrame{
            std::move(_value_map),
            std::move(_value_map_insertion_log)});
        _value_map = {};
        _value_map_insertion_log = {};
        if (auto statistics = _config.statistics) {
            statistics->function_translations++;
        }
        auto builder = _translate(f);
        _value_map = std::move(_value_map_stack.back().map);
        _value_map_insertion_log =
            std::move(_value_map_stack.back().insertion_log);
        _value_map_stack.pop_back();
        _function_map.emplace(&f, builder);
        _translating_functions.erase(&f);
        return builder;
    }

public:
    explicit XIR2ASTContext(
        const XIR2ASTConfig &config,
        const Module *verified_module = nullptr) noexcept
        : _config{config},
          _verified_module{verified_module} {}

    [[nodiscard]] luisa::shared_ptr<const ASTFunctionBuilder>
    translate_function(const FunctionDefinition &f) noexcept {
        // A context may translate several roots from one immutable module.
        // Per-root expression/CFG state must never cross that boundary; only
        // completed ordinary callable builders in _function_map are shared.
        LUISA_ASSERT(
            _value_map_stack.empty() && _active_blocks.empty() &&
                _loop_update_stack.empty() && _translating_functions.empty(),
            "XIR-to-AST root translation started with live per-root state.");
        _value_map.clear();
        _value_map_insertion_log.clear();
        if (auto iter = _function_map.find(&f);
            iter != _function_map.end()) {
            if (auto statistics = _config.statistics) {
                statistics->function_cache_hits++;
            }
            return iter->second;
        }
        _verify_translation_input(f, "translation input");
        if (!_translating_functions.emplace(&f).second) {
            LUISA_ERROR_WITH_LOCATION(
                "Recursive XIR root functions are not supported by XIR-to-AST.");
        }
        if (auto statistics = _config.statistics) {
            statistics->function_translations++;
        }
        auto builder = _translate(f);
        _translating_functions.erase(&f);
        auto [iter, inserted] = _function_map.emplace(&f, builder);
        LUISA_ASSERT(
            inserted || iter->second == builder,
            "XIR-to-AST root cache changed during translation.");
        LUISA_ASSERT(
            _value_map_stack.empty() && _active_blocks.empty() &&
                _loop_update_stack.empty() && _translating_functions.empty(),
            "XIR-to-AST root translation leaked per-root state.");
        return builder;
    }

    void add_function(const FunctionDefinition &f) noexcept {
        LUISA_ASSERT(_builder == nullptr, "XIR2ASTContext currently accepts one function.");
        _builder = translate_function(f);
    }

    [[nodiscard]] luisa::shared_ptr<const ASTFunctionBuilder> finalize() noexcept {
        return std::move(_builder);
    }
};

XIR2ASTContext *xir_to_ast_translate_begin(const XIR2ASTConfig &config) noexcept {
    return luisa::new_with_allocator<XIR2ASTContext>(config);
}

void xir_to_ast_translate_add_function(XIR2ASTContext *ctx, const FunctionDefinition &f) noexcept {
    ctx->add_function(f);
}

luisa::shared_ptr<const ASTFunctionBuilder> xir_to_ast_translate_finalize(XIR2ASTContext *ctx) noexcept {
    auto f = ctx->finalize();
    luisa::delete_with_allocator(ctx);
    return f;
}

void xir_to_ast_normalize_module(Module *module) noexcept {
    verify_xir_for_ast(module, "normalization input",
                       {.require_terminated_blocks = false});
    // Generic normalization exposes generated continuation callables but does
    // not replace/remove the source coroutine definition. Once continuations
    // exist, keep them even though they intentionally have no IR call users;
    // full source-coroutine ownership belongs to compile_coroutine_pipeline.
    auto preserve_generated_coro_continuations = false;
    PassPipeline pipeline;
    pipeline.add_sequence("phase-A", create_basic_optimization_pipeline());
    pipeline.add("inline-all", [](Module *m, PassReport &r) {
        auto i = inline_all_pass_run_on_module(m, &r);
        return i.changed();
    });
    pipeline.add_sequence("post-inline-cleanup", create_post_inline_cleanup_pipeline());
    pipeline.add("lower-ray-query-loop-to-loop", [](Module *m, PassReport &r) {
        auto i = lower_ray_query_loop_to_loop_pass_run_on_module(m, &r);
        if (!i.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "XIR-to-AST normalization rejected {} unsupported ray-query loop(s); "
                "the normalization pipeline cannot continue safely.",
                i.error_count);
        }
        return i.lowered_ray_query_loop_count > 0u;
    });
    pipeline.add("destructure-cfg", [](Module *m, PassReport &r) {
        auto i = destructure_cfg_pass_run_on_module(m, &r);
        if (i.error_count != 0u) {
            LUISA_ERROR_WITH_LOCATION(
                "XIR-to-AST destructuring failed (errors={}, leaked_blocks={}).",
                i.error_count, i.leaked_block_count);
        }
        return i.changed();
    });
    // Structured constant branches become ordinary CFG only after
    // destructuring. Canonicalize reachability at this exact boundary so the
    // coroutine scope/token domain is derived from executable suspend edges,
    // not from the original front-end statement set.
    pipeline.add("coro-pre-distill-simplify-cfg", [](Module *m, PassReport &r) {
        auto i = simplify_cfg_pass_run_on_module(m, &r);
        return i.changed();
    });
    pipeline.add("coro-pre-distill-dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.changed();
    });
    // Coroutine pipeline: runs only for definitions that still carry a
    // coroutine suspend/resume/terminate marker.
    // Must run after destructure_cfg (needs destructured CFG) but BEFORE
    // mem2reg/ssa-opt which may eliminate dead resume blocks.
    pipeline.add("coro-pipeline", [&preserve_generated_coro_continuations](Module *m, PassReport & /*r*/) {
        bool changed = false;
        for (auto *f : m->function_list()) {
            if (auto *def = f->definition()) {
                bool has_coro = false;
                def->traverse_instructions([&](Instruction *inst) noexcept {
                    switch (inst->derived_instruction_tag()) {
                        case DerivedInstructionTag::CORO_SUSPEND:
                        case DerivedInstructionTag::CORO_RESUME:
                        case DerivedInstructionTag::CORO_TERMINATE:
                            has_coro = true;
                            break;
                        default: break;
                    }
                });
                if (!has_coro) { continue; }
                auto cfg = coro_cfg_distill_pass_run_on_function(f);
                if (!cfg.succeeded()) {
                    LUISA_ERROR_WITH_LOCATION(
                        "XIR-to-AST coroutine CFG distillation rejected its input "
                        "(structured={}, invalid_input={}, invalid_cfg={}); "
                        "the pipeline cannot continue safely.",
                        cfg.structured_cfg_error_count,
                        cfg.invalid_input_error_count,
                        cfg.invalid_cfg_error_count);
                }
                auto split = coro_split_pass_run_on_module_with_cfg_and_frame_info(m, cfg, nullptr);
                if (!split.succeeded()) {
                    LUISA_ERROR_WITH_LOCATION(
                        "XIR-to-AST coroutine normalization rejected its input "
                        "(structured={}, invalid_cfg={}); the pipeline cannot continue safely.",
                        split.structured_cfg_error_count, split.invalid_cfg_error_count);
                }
                if (split.subroutines.empty()) {
                    LUISA_ERROR_WITH_LOCATION(
                        "XIR-to-AST coroutine normalization found a suspend but produced no "
                        "continuation scopes; the pipeline cannot continue safely.");
                }
                auto materialized = coro_materialize_pass_run_on_module_with_cfg(m, cfg, split);
                if (!materialized.succeeded()) {
                    LUISA_ERROR_WITH_LOCATION(
                        "XIR-to-AST coroutine materialization rejected its input "
                        "(structured={}, invalid_input={}); the pipeline cannot continue safely.",
                        materialized.structured_cfg_error_count,
                        materialized.invalid_input_error_count);
                }
                (void)coro_reg2mem_pass_run_on_split(split);
                preserve_generated_coro_continuations = true;
                changed |= !split.subroutines.empty();
            }
        }
        return changed;
    });
    pipeline.add("mem2reg", [](Module *m, PassReport &r) {
        auto i = mem2reg_pass_run_on_module(m, &r);
        return i.changed();
    });
    pipeline.add_sequence("ssa-opt", create_ssa_optimization_pipeline());
    pipeline.add("unused-callable-removal", [&preserve_generated_coro_continuations](Module *m, PassReport &r) {
        if (preserve_generated_coro_continuations) {
            r.set("skipped_for_coro_continuations", 1u);
            return false;
        }
        auto i = unused_callable_removal_pass_run_on_module(m, &r);
        return i.removed_callable_count > 0u;
    });
    pipeline.add("simplify-cfg", [](Module *m, PassReport &r) {
        auto i = simplify_cfg_pass_run_on_module(m, &r);
        return i.changed();
    });
    pipeline.add("reg2mem-pre", [](Module *m, PassReport &r) {
        auto i = reg2mem_pass_run_on_module(m, &r);
        return i.changed();
    });
    // Coroutine splitting can cut a reducible source loop at a suspend-scope
    // boundary, leaving the source coroutine and one or more generated
    // continuations with multi-entry cyclic SCCs. More generally, XIR-to-AST
    // normalization accepts raw CFG, so make reducibility an explicit pipeline
    // invariant before the strictly structural rewriter. The lowering is
    // semantics-preserving and linear: incoming edges select their original
    // target through one dispatcher instead of cloning any shader body.
    pipeline.add("lower-irreducible-cfg", [](Module *m, PassReport &r) {
        auto i = lower_irreducible_cfg_pass_run_on_module(m, &r);
        if (!i.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "XIR-to-AST irreducible-CFG lowering failed "
                "(remaining={}, errors={}).",
                i.remaining_irreducible_region_count,
                i.error_count);
        }
        return i.changed();
    });
    pipeline.add("restructure-cfg", [](Module *m, PassReport &r) {
        auto i = restructure_cfg_pass_run_on_module(m, &r);
        if (!i.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "XIR-to-AST restructuring failed (irreducible={}, unstructured={}, invalid={}, iteration_limit={}).",
                i.irreducible_region_count, i.unstructured_branch_count,
                i.invalid_construct_count, i.iteration_limit_count);
        }
        return i.changed();
    });
    pipeline.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.changed();
    });
    pipeline.add("reg2mem-mid", [](Module *m, PassReport &r) {
        auto i = reg2mem_pass_run_on_module(m, &r);
        return i.changed();
    });
    pipeline.add_sequence("post-restructure-cleanup", create_post_restructure_cleanup_pipeline());
    pipeline.add("reg2mem-post", [](Module *m, PassReport &r) {
        auto i = reg2mem_pass_run_on_module(m, &r);
        return i.changed();
    });
    pipeline.add("unused-callable-removal-final", [&preserve_generated_coro_continuations](Module *m, PassReport &r) {
        if (preserve_generated_coro_continuations) {
            r.set("skipped_for_coro_continuations", 1u);
            return false;
        }
        auto i = unused_callable_removal_pass_run_on_module(m, &r);
        return i.removed_callable_count > 0u;
    });
    auto stats = pipeline.run(module);
    verify_xir_for_ast(
        module, "normalization output",
        {.require_no_phi = true,
         .require_canonical_break_continue_targets = true});
    stats.log("xir_to_ast_normalize_module");
}

luisa::shared_ptr<const ASTFunctionBuilder> xir_to_ast_translate(const FunctionDefinition &function, const XIR2ASTConfig &config) noexcept {
    XIR2ASTContext ctx{config};
    ctx.add_function(function);
    return ctx.finalize();
}

[[nodiscard]] luisa::shared_ptr<const ASTFunctionBuilder>
xir_to_ast_translate_continuation(const FunctionDefinition &function,
                                  const XIR2ASTConfig &config) noexcept {
    LUISA_ASSERT(function.derived_function_tag() == DerivedFunctionTag::CALLABLE,
                 "xir_to_ast_translate_continuation requires a CallableFunction.");
    XIR2ASTContext ctx{config};
    ctx.add_function(function);
    return ctx.finalize();
}

luisa::vector<luisa::shared_ptr<const ASTFunctionBuilder>>
xir_to_ast_translate_continuations(
    luisa::span<const FunctionDefinition *const> functions,
    const XIR2ASTConfig &config) noexcept {
    luisa::vector<luisa::shared_ptr<const ASTFunctionBuilder>> builders;
    builders.reserve(functions.size());
    LUISA_ASSERT(
        !(config.verify_same_module_once &&
          config.verification_transaction != nullptr),
        "A continuation batch cannot independently verify a module while "
        "also closing an enclosing verification transaction.");
    if (functions.empty()) {
        LUISA_ASSERT(
            config.verification_transaction == nullptr,
            "An enclosing verification transaction cannot be closed by an "
            "empty continuation batch.");
        return builders;
    }

    const Module *verified_module = nullptr;
    for (auto *function : functions) {
        LUISA_ASSERT(
            function != nullptr &&
                function->derived_function_tag() ==
                    DerivedFunctionTag::CALLABLE,
            "xir_to_ast_translate_continuations requires CallableFunction definitions.");
    }
    if (config.verify_same_module_once ||
        config.verification_transaction != nullptr) {
        verified_module =
            config.verification_transaction == nullptr ?
                functions.front()->parent_module() :
                config.verification_transaction->module();
        LUISA_ASSERT(
            verified_module != nullptr,
            "Whole-module XIR-to-AST verification requires owned functions.");
        for (auto *function : functions) {
            LUISA_ASSERT(
                function->parent_module() == verified_module,
                "Whole-module XIR-to-AST verification requires all roots to "
                "belong to the same module.");
        }
        luisa::Clock verification_clock;
        constexpr XIRVerificationOptions verification_options{
            .require_no_phi = true,
            .require_canonical_break_continue_targets = true};
        if (config.verification_transaction == nullptr) {
            verify_xir_for_ast(
                verified_module, "continuation batch input",
                verification_options);
        } else {
            auto verification =
                config.verification_transaction->verify_output(
                    verification_options);
            require_xir_for_ast(
                verification,
                "continuation batch transaction output");
        }
        if (auto statistics = config.statistics) {
            statistics->whole_module_verification_count++;
            statistics->verification_milliseconds +=
                verification_clock.toc();
        }
    }

    // The verified-module capability never escapes this synchronous call.
    // Translation is read-only, and every root/dependency is checked against
    // the exact verified module before per-function verification is elided.
    XIR2ASTContext ctx{config, verified_module};
    for (auto *function : functions) {
        builders.emplace_back(ctx.translate_function(*function));
    }
    return builders;
}

}// namespace luisa::compute::xir
