#pragma once
// if msvc
#ifdef _MSC_VER
#pragma warning(disable : 4190)
#endif

#include <cstdint>
#include <luisa/core/dll_export.h>
#ifndef BINDGEN
#include <array>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#endif
namespace luisa::compute {
class Type;
}// namespace luisa::compute
namespace luisa::compute::ir_v2 {

struct Node;
class BasicBlock;
struct CallableModule;
struct Module;
struct KernelModule;
class Pool;

struct PhiIncoming {
    BasicBlock *block = nullptr;
    Node *value = nullptr;
};
struct SwitchCase {
    int32_t value = 0;
    BasicBlock *block = nullptr;
};
struct CpuExternFn {
    void *data = nullptr;
    void (*func)(void *data, void *args) = nullptr;
    void (*dtor)(void *data) = nullptr;
    const Type *arg_ty = nullptr;
};
struct FuncMetadata {
    bool has_side_effects = false;
};
const FuncMetadata *func_metadata();

struct Func;
struct FuncData;
enum class FuncTag : unsigned int {
    UNDEF,
    ZERO,
    ONE,
    ASSUME,
    UNREACHABLE,
    THREAD_ID,
    BLOCK_ID,
    WARP_SIZE,
    WARP_LANE_ID,
    DISPATCH_ID,
    DISPATCH_SIZE,
    PROPAGATE_GRADIENT,
    OUTPUT_GRADIENT,
    REQUIRES_GRADIENT,
    BACKWARD,
    GRADIENT,
    ACC_GRAD,
    DETACH,
    RAY_TRACING_INSTANCE_TRANSFORM,
    RAY_TRACING_INSTANCE_VISIBILITY_MASK,
    RAY_TRACING_INSTANCE_USER_ID,
    RAY_TRACING_SET_INSTANCE_TRANSFORM,
    RAY_TRACING_SET_INSTANCE_OPACITY,
    RAY_TRACING_SET_INSTANCE_VISIBILITY,
    RAY_TRACING_SET_INSTANCE_USER_ID,
    RAY_TRACING_TRACE_CLOSEST,
    RAY_TRACING_TRACE_ANY,
    RAY_TRACING_QUERY_ALL,
    RAY_TRACING_QUERY_ANY,
    RAY_QUERY_WORLD_SPACE_RAY,
    RAY_QUERY_PROCEDURAL_CANDIDATE_HIT,
    RAY_QUERY_TRIANGLE_CANDIDATE_HIT,
    RAY_QUERY_COMMITTED_HIT,
    RAY_QUERY_COMMIT_TRIANGLE,
    RAY_QUERY_COMMITD_PROCEDURAL,
    RAY_QUERY_TERMINATE,
    LOAD,
    CAST,
    BIT_CAST,
    ADD,
    SUB,
    MUL,
    DIV,
    REM,
    BIT_AND,
    BIT_OR,
    BIT_XOR,
    SHL,
    SHR,
    ROT_RIGHT,
    ROT_LEFT,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    MAT_COMP_MUL,
    NEG,
    NOT,
    BIT_NOT,
    ALL,
    ANY,
    SELECT,
    CLAMP,
    LERP,
    STEP,
    SATURATE,
    SMOOTH_STEP,
    ABS,
    MIN,
    MAX,
    REDUCE_SUM,
    REDUCE_PROD,
    REDUCE_MIN,
    REDUCE_MAX,
    CLZ,
    CTZ,
    POP_COUNT,
    REVERSE,
    IS_INF,
    IS_NAN,
    ACOS,
    ACOSH,
    ASIN,
    ASINH,
    ATAN,
    ATAN2,
    ATANH,
    COS,
    COSH,
    SIN,
    SINH,
    TAN,
    TANH,
    EXP,
    EXP2,
    EXP10,
    LOG,
    LOG2,
    LOG10,
    POWI,
    POWF,
    SQRT,
    RSQRT,
    CEIL,
    FLOOR,
    FRACT,
    TRUNC,
    ROUND,
    FMA,
    COPYSIGN,
    CROSS,
    DOT,
    OUTER_PRODUCT,
    LENGTH,
    LENGTH_SQUARED,
    NORMALIZE,
    FACEFORWARD,
    DISTANCE,
    REFLECT,
    DETERMINANT,
    TRANSPOSE,
    INVERSE,
    WARP_IS_FIRST_ACTIVE_LANE,
    WARP_FIRST_ACTIVE_LANE,
    WARP_ACTIVE_ALL_EQUAL,
    WARP_ACTIVE_BIT_AND,
    WARP_ACTIVE_BIT_OR,
    WARP_ACTIVE_BIT_XOR,
    WARP_ACTIVE_COUNT_BITS,
    WARP_ACTIVE_MAX,
    WARP_ACTIVE_MIN,
    WARP_ACTIVE_PRODUCT,
    WARP_ACTIVE_SUM,
    WARP_ACTIVE_ALL,
    WARP_ACTIVE_ANY,
    WARP_ACTIVE_BIT_MASK,
    WARP_PREFIX_COUNT_BITS,
    WARP_PREFIX_SUM,
    WARP_PREFIX_PRODUCT,
    WARP_READ_LANE_AT,
    WARP_READ_FIRST_LANE,
    SYNCHRONIZE_BLOCK,
    ATOMIC_EXCHANGE,
    ATOMIC_COMPARE_EXCHANGE,
    ATOMIC_FETCH_ADD,
    ATOMIC_FETCH_SUB,
    ATOMIC_FETCH_AND,
    ATOMIC_FETCH_OR,
    ATOMIC_FETCH_XOR,
    ATOMIC_FETCH_MIN,
    ATOMIC_FETCH_MAX,
    BUFFER_WRITE,
    BUFFER_READ,
    BUFFER_SIZE,
    BYTE_BUFFER_WRITE,
    BYTE_BUFFER_READ,
    BYTE_BUFFER_SIZE,
    TEXTURE2D_READ,
    TEXTURE2D_WRITE,
    TEXTURE2D_SIZE,
    TEXTURE3D_READ,
    TEXTURE3D_WRITE,
    TEXTURE3D_SIZE,
    BINDLESS_TEXTURE2D_SAMPLE,
    BINDLESS_TEXTURE2D_SAMPLE_LEVEL,
    BINDLESS_TEXTURE2D_SAMPLE_GRAD,
    BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL,
    BINDLESS_TEXTURE2D_READ,
    BINDLESS_TEXTURE2D_SIZE,
    BINDLESS_TEXTURE2D_SIZE_LEVEL,
    BINDLESS_TEXTURE3D_SAMPLE,
    BINDLESS_TEXTURE3D_SAMPLE_LEVEL,
    BINDLESS_TEXTURE3D_SAMPLE_GRAD,
    BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL,
    BINDLESS_TEXTURE3D_READ,
    BINDLESS_TEXTURE3D_SIZE,
    BINDLESS_TEXTURE3D_SIZE_LEVEL,
    BINDLESS_BUFFER_WRITE,
    BINDLESS_BUFFER_READ,
    BINDLESS_BUFFER_SIZE,
    BINDLESS_BYTE_BUFFER_WRITE,
    BINDLESS_BYTE_BUFFER_READ,
    BINDLESS_BYTE_BUFFER_SIZE,
    VEC,
    VEC2,
    VEC3,
    VEC4,
    PERMUTE,
    GET_ELEMENT_PTR,
    EXTRACT_ELEMENT,
    INSERT_ELEMENT,
    ARRAY,
    STRUCT,
    MAT_FULL,
    MAT2,
    MAT3,
    MAT4,
    BINDLESS_ATOMIC_EXCHANGE,
    BINDLESS_ATOMIC_COMPARE_EXCHANGE,
    BINDLESS_ATOMIC_FETCH_ADD,
    BINDLESS_ATOMIC_FETCH_SUB,
    BINDLESS_ATOMIC_FETCH_AND,
    BINDLESS_ATOMIC_FETCH_OR,
    BINDLESS_ATOMIC_FETCH_XOR,
    BINDLESS_ATOMIC_FETCH_MIN,
    BINDLESS_ATOMIC_FETCH_MAX,
    CALLABLE,
    CPU_EXT,
    SHADER_EXECUTION_REORDER,
};
struct LC_IR_API FuncData {
#ifndef BINDGEN
    virtual FuncTag tag() const noexcept = 0;
    virtual ~FuncData() = default;
#endif
};
struct AssumeFn;
struct UnreachableFn;
struct BindlessAtomicExchangeFn;
struct BindlessAtomicCompareExchangeFn;
struct BindlessAtomicFetchAddFn;
struct BindlessAtomicFetchSubFn;
struct BindlessAtomicFetchAndFn;
struct BindlessAtomicFetchOrFn;
struct BindlessAtomicFetchXorFn;
struct BindlessAtomicFetchMinFn;
struct BindlessAtomicFetchMaxFn;
struct CallableFn;
struct CpuExtFn;
struct Instruction;
struct InstructionData;
enum class InstructionTag : unsigned int {
    BUFFER,
    TEXTURE2D,
    TEXTURE3D,
    BINDLESS_ARRAY,
    ACCEL,
    SHARED,
    UNIFORM,
    ARGUMENT,
    CONSTANT,
    CALL,
    PHI,
    BASIC_BLOCK_SENTINEL,
    IF,
    GENERIC_LOOP,
    SWITCH,
    LOCAL,
    BREAK,
    CONTINUE,
    RETURN,
    PRINT,
    UPDATE,
    RAY_QUERY,
    REV_AUTODIFF,
    FWD_AUTODIFF,
};
struct LC_IR_API InstructionData {
#ifndef BINDGEN
    virtual InstructionTag tag() const noexcept = 0;
    virtual ~InstructionData() = default;
#endif
};
struct ArgumentInst;
struct ConstantInst;
struct CallInst;
struct PhiInst;
struct IfInst;
struct GenericLoopInst;
struct SwitchInst;
struct LocalInst;
struct ReturnInst;
struct PrintInst;
struct UpdateInst;
struct RayQueryInst;
struct RevAutodiffInst;
struct FwdAutodiffInst;
struct Binding;
struct BindingData;
enum class BindingTag : unsigned int {
    BUFFER_BINDING,
    TEXTURE_BINDING,
    BINDLESS_ARRAY_BINDING,
    ACCEL_BINDING,
};
struct LC_IR_API BindingData {
#ifndef BINDGEN
    virtual BindingTag tag() const noexcept = 0;
    virtual ~BindingData() = default;
#endif
};
struct BufferBinding;
struct TextureBinding;
struct BindlessArrayBinding;
struct AccelBinding;
}// namespace luisa::compute::ir_v2
