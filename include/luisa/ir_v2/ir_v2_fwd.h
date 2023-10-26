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
template<class T>
struct Slice;
struct CInstruction;
struct CFunc;
struct CBinding;

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
inline const char *tag_name(FuncTag tag) {
    switch (tag) {
        case FuncTag::UNDEF: return "UndefFn";
        case FuncTag::ZERO: return "ZeroFn";
        case FuncTag::ONE: return "OneFn";
        case FuncTag::ASSUME: return "AssumeFn";
        case FuncTag::UNREACHABLE: return "UnreachableFn";
        case FuncTag::THREAD_ID: return "ThreadIdFn";
        case FuncTag::BLOCK_ID: return "BlockIdFn";
        case FuncTag::WARP_SIZE: return "WarpSizeFn";
        case FuncTag::WARP_LANE_ID: return "WarpLaneIdFn";
        case FuncTag::DISPATCH_ID: return "DispatchIdFn";
        case FuncTag::DISPATCH_SIZE: return "DispatchSizeFn";
        case FuncTag::PROPAGATE_GRADIENT: return "PropagateGradientFn";
        case FuncTag::OUTPUT_GRADIENT: return "OutputGradientFn";
        case FuncTag::REQUIRES_GRADIENT: return "RequiresGradientFn";
        case FuncTag::BACKWARD: return "BackwardFn";
        case FuncTag::GRADIENT: return "GradientFn";
        case FuncTag::ACC_GRAD: return "AccGradFn";
        case FuncTag::DETACH: return "DetachFn";
        case FuncTag::RAY_TRACING_INSTANCE_TRANSFORM: return "RayTracingInstanceTransformFn";
        case FuncTag::RAY_TRACING_INSTANCE_VISIBILITY_MASK: return "RayTracingInstanceVisibilityMaskFn";
        case FuncTag::RAY_TRACING_INSTANCE_USER_ID: return "RayTracingInstanceUserIdFn";
        case FuncTag::RAY_TRACING_SET_INSTANCE_TRANSFORM: return "RayTracingSetInstanceTransformFn";
        case FuncTag::RAY_TRACING_SET_INSTANCE_OPACITY: return "RayTracingSetInstanceOpacityFn";
        case FuncTag::RAY_TRACING_SET_INSTANCE_VISIBILITY: return "RayTracingSetInstanceVisibilityFn";
        case FuncTag::RAY_TRACING_SET_INSTANCE_USER_ID: return "RayTracingSetInstanceUserIdFn";
        case FuncTag::RAY_TRACING_TRACE_CLOSEST: return "RayTracingTraceClosestFn";
        case FuncTag::RAY_TRACING_TRACE_ANY: return "RayTracingTraceAnyFn";
        case FuncTag::RAY_TRACING_QUERY_ALL: return "RayTracingQueryAllFn";
        case FuncTag::RAY_TRACING_QUERY_ANY: return "RayTracingQueryAnyFn";
        case FuncTag::RAY_QUERY_WORLD_SPACE_RAY: return "RayQueryWorldSpaceRayFn";
        case FuncTag::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT: return "RayQueryProceduralCandidateHitFn";
        case FuncTag::RAY_QUERY_TRIANGLE_CANDIDATE_HIT: return "RayQueryTriangleCandidateHitFn";
        case FuncTag::RAY_QUERY_COMMITTED_HIT: return "RayQueryCommittedHitFn";
        case FuncTag::RAY_QUERY_COMMIT_TRIANGLE: return "RayQueryCommitTriangleFn";
        case FuncTag::RAY_QUERY_COMMITD_PROCEDURAL: return "RayQueryCommitdProceduralFn";
        case FuncTag::RAY_QUERY_TERMINATE: return "RayQueryTerminateFn";
        case FuncTag::LOAD: return "LoadFn";
        case FuncTag::CAST: return "CastFn";
        case FuncTag::BIT_CAST: return "BitCastFn";
        case FuncTag::ADD: return "AddFn";
        case FuncTag::SUB: return "SubFn";
        case FuncTag::MUL: return "MulFn";
        case FuncTag::DIV: return "DivFn";
        case FuncTag::REM: return "RemFn";
        case FuncTag::BIT_AND: return "BitAndFn";
        case FuncTag::BIT_OR: return "BitOrFn";
        case FuncTag::BIT_XOR: return "BitXorFn";
        case FuncTag::SHL: return "ShlFn";
        case FuncTag::SHR: return "ShrFn";
        case FuncTag::ROT_RIGHT: return "RotRightFn";
        case FuncTag::ROT_LEFT: return "RotLeftFn";
        case FuncTag::EQ: return "EqFn";
        case FuncTag::NE: return "NeFn";
        case FuncTag::LT: return "LtFn";
        case FuncTag::LE: return "LeFn";
        case FuncTag::GT: return "GtFn";
        case FuncTag::GE: return "GeFn";
        case FuncTag::MAT_COMP_MUL: return "MatCompMulFn";
        case FuncTag::NEG: return "NegFn";
        case FuncTag::NOT: return "NotFn";
        case FuncTag::BIT_NOT: return "BitNotFn";
        case FuncTag::ALL: return "AllFn";
        case FuncTag::ANY: return "AnyFn";
        case FuncTag::SELECT: return "SelectFn";
        case FuncTag::CLAMP: return "ClampFn";
        case FuncTag::LERP: return "LerpFn";
        case FuncTag::STEP: return "StepFn";
        case FuncTag::SATURATE: return "SaturateFn";
        case FuncTag::SMOOTH_STEP: return "SmoothStepFn";
        case FuncTag::ABS: return "AbsFn";
        case FuncTag::MIN: return "MinFn";
        case FuncTag::MAX: return "MaxFn";
        case FuncTag::REDUCE_SUM: return "ReduceSumFn";
        case FuncTag::REDUCE_PROD: return "ReduceProdFn";
        case FuncTag::REDUCE_MIN: return "ReduceMinFn";
        case FuncTag::REDUCE_MAX: return "ReduceMaxFn";
        case FuncTag::CLZ: return "ClzFn";
        case FuncTag::CTZ: return "CtzFn";
        case FuncTag::POP_COUNT: return "PopCountFn";
        case FuncTag::REVERSE: return "ReverseFn";
        case FuncTag::IS_INF: return "IsInfFn";
        case FuncTag::IS_NAN: return "IsNanFn";
        case FuncTag::ACOS: return "AcosFn";
        case FuncTag::ACOSH: return "AcoshFn";
        case FuncTag::ASIN: return "AsinFn";
        case FuncTag::ASINH: return "AsinhFn";
        case FuncTag::ATAN: return "AtanFn";
        case FuncTag::ATAN2: return "Atan2Fn";
        case FuncTag::ATANH: return "AtanhFn";
        case FuncTag::COS: return "CosFn";
        case FuncTag::COSH: return "CoshFn";
        case FuncTag::SIN: return "SinFn";
        case FuncTag::SINH: return "SinhFn";
        case FuncTag::TAN: return "TanFn";
        case FuncTag::TANH: return "TanhFn";
        case FuncTag::EXP: return "ExpFn";
        case FuncTag::EXP2: return "Exp2Fn";
        case FuncTag::EXP10: return "Exp10Fn";
        case FuncTag::LOG: return "LogFn";
        case FuncTag::LOG2: return "Log2Fn";
        case FuncTag::LOG10: return "Log10Fn";
        case FuncTag::POWI: return "PowiFn";
        case FuncTag::POWF: return "PowfFn";
        case FuncTag::SQRT: return "SqrtFn";
        case FuncTag::RSQRT: return "RsqrtFn";
        case FuncTag::CEIL: return "CeilFn";
        case FuncTag::FLOOR: return "FloorFn";
        case FuncTag::FRACT: return "FractFn";
        case FuncTag::TRUNC: return "TruncFn";
        case FuncTag::ROUND: return "RoundFn";
        case FuncTag::FMA: return "FmaFn";
        case FuncTag::COPYSIGN: return "CopysignFn";
        case FuncTag::CROSS: return "CrossFn";
        case FuncTag::DOT: return "DotFn";
        case FuncTag::OUTER_PRODUCT: return "OuterProductFn";
        case FuncTag::LENGTH: return "LengthFn";
        case FuncTag::LENGTH_SQUARED: return "LengthSquaredFn";
        case FuncTag::NORMALIZE: return "NormalizeFn";
        case FuncTag::FACEFORWARD: return "FaceforwardFn";
        case FuncTag::DISTANCE: return "DistanceFn";
        case FuncTag::REFLECT: return "ReflectFn";
        case FuncTag::DETERMINANT: return "DeterminantFn";
        case FuncTag::TRANSPOSE: return "TransposeFn";
        case FuncTag::INVERSE: return "InverseFn";
        case FuncTag::WARP_IS_FIRST_ACTIVE_LANE: return "WarpIsFirstActiveLaneFn";
        case FuncTag::WARP_FIRST_ACTIVE_LANE: return "WarpFirstActiveLaneFn";
        case FuncTag::WARP_ACTIVE_ALL_EQUAL: return "WarpActiveAllEqualFn";
        case FuncTag::WARP_ACTIVE_BIT_AND: return "WarpActiveBitAndFn";
        case FuncTag::WARP_ACTIVE_BIT_OR: return "WarpActiveBitOrFn";
        case FuncTag::WARP_ACTIVE_BIT_XOR: return "WarpActiveBitXorFn";
        case FuncTag::WARP_ACTIVE_COUNT_BITS: return "WarpActiveCountBitsFn";
        case FuncTag::WARP_ACTIVE_MAX: return "WarpActiveMaxFn";
        case FuncTag::WARP_ACTIVE_MIN: return "WarpActiveMinFn";
        case FuncTag::WARP_ACTIVE_PRODUCT: return "WarpActiveProductFn";
        case FuncTag::WARP_ACTIVE_SUM: return "WarpActiveSumFn";
        case FuncTag::WARP_ACTIVE_ALL: return "WarpActiveAllFn";
        case FuncTag::WARP_ACTIVE_ANY: return "WarpActiveAnyFn";
        case FuncTag::WARP_ACTIVE_BIT_MASK: return "WarpActiveBitMaskFn";
        case FuncTag::WARP_PREFIX_COUNT_BITS: return "WarpPrefixCountBitsFn";
        case FuncTag::WARP_PREFIX_SUM: return "WarpPrefixSumFn";
        case FuncTag::WARP_PREFIX_PRODUCT: return "WarpPrefixProductFn";
        case FuncTag::WARP_READ_LANE_AT: return "WarpReadLaneAtFn";
        case FuncTag::WARP_READ_FIRST_LANE: return "WarpReadFirstLaneFn";
        case FuncTag::SYNCHRONIZE_BLOCK: return "SynchronizeBlockFn";
        case FuncTag::ATOMIC_EXCHANGE: return "AtomicExchangeFn";
        case FuncTag::ATOMIC_COMPARE_EXCHANGE: return "AtomicCompareExchangeFn";
        case FuncTag::ATOMIC_FETCH_ADD: return "AtomicFetchAddFn";
        case FuncTag::ATOMIC_FETCH_SUB: return "AtomicFetchSubFn";
        case FuncTag::ATOMIC_FETCH_AND: return "AtomicFetchAndFn";
        case FuncTag::ATOMIC_FETCH_OR: return "AtomicFetchOrFn";
        case FuncTag::ATOMIC_FETCH_XOR: return "AtomicFetchXorFn";
        case FuncTag::ATOMIC_FETCH_MIN: return "AtomicFetchMinFn";
        case FuncTag::ATOMIC_FETCH_MAX: return "AtomicFetchMaxFn";
        case FuncTag::BUFFER_WRITE: return "BufferWriteFn";
        case FuncTag::BUFFER_READ: return "BufferReadFn";
        case FuncTag::BUFFER_SIZE: return "BufferSizeFn";
        case FuncTag::BYTE_BUFFER_WRITE: return "ByteBufferWriteFn";
        case FuncTag::BYTE_BUFFER_READ: return "ByteBufferReadFn";
        case FuncTag::BYTE_BUFFER_SIZE: return "ByteBufferSizeFn";
        case FuncTag::TEXTURE2D_READ: return "Texture2dReadFn";
        case FuncTag::TEXTURE2D_WRITE: return "Texture2dWriteFn";
        case FuncTag::TEXTURE2D_SIZE: return "Texture2dSizeFn";
        case FuncTag::TEXTURE3D_READ: return "Texture3dReadFn";
        case FuncTag::TEXTURE3D_WRITE: return "Texture3dWriteFn";
        case FuncTag::TEXTURE3D_SIZE: return "Texture3dSizeFn";
        case FuncTag::BINDLESS_TEXTURE2D_SAMPLE: return "BindlessTexture2dSampleFn";
        case FuncTag::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: return "BindlessTexture2dSampleLevelFn";
        case FuncTag::BINDLESS_TEXTURE2D_SAMPLE_GRAD: return "BindlessTexture2dSampleGradFn";
        case FuncTag::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: return "BindlessTexture2dSampleGradLevelFn";
        case FuncTag::BINDLESS_TEXTURE2D_READ: return "BindlessTexture2dReadFn";
        case FuncTag::BINDLESS_TEXTURE2D_SIZE: return "BindlessTexture2dSizeFn";
        case FuncTag::BINDLESS_TEXTURE2D_SIZE_LEVEL: return "BindlessTexture2dSizeLevelFn";
        case FuncTag::BINDLESS_TEXTURE3D_SAMPLE: return "BindlessTexture3dSampleFn";
        case FuncTag::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: return "BindlessTexture3dSampleLevelFn";
        case FuncTag::BINDLESS_TEXTURE3D_SAMPLE_GRAD: return "BindlessTexture3dSampleGradFn";
        case FuncTag::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: return "BindlessTexture3dSampleGradLevelFn";
        case FuncTag::BINDLESS_TEXTURE3D_READ: return "BindlessTexture3dReadFn";
        case FuncTag::BINDLESS_TEXTURE3D_SIZE: return "BindlessTexture3dSizeFn";
        case FuncTag::BINDLESS_TEXTURE3D_SIZE_LEVEL: return "BindlessTexture3dSizeLevelFn";
        case FuncTag::BINDLESS_BUFFER_WRITE: return "BindlessBufferWriteFn";
        case FuncTag::BINDLESS_BUFFER_READ: return "BindlessBufferReadFn";
        case FuncTag::BINDLESS_BUFFER_SIZE: return "BindlessBufferSizeFn";
        case FuncTag::BINDLESS_BYTE_BUFFER_WRITE: return "BindlessByteBufferWriteFn";
        case FuncTag::BINDLESS_BYTE_BUFFER_READ: return "BindlessByteBufferReadFn";
        case FuncTag::BINDLESS_BYTE_BUFFER_SIZE: return "BindlessByteBufferSizeFn";
        case FuncTag::VEC: return "VecFn";
        case FuncTag::VEC2: return "Vec2Fn";
        case FuncTag::VEC3: return "Vec3Fn";
        case FuncTag::VEC4: return "Vec4Fn";
        case FuncTag::PERMUTE: return "PermuteFn";
        case FuncTag::GET_ELEMENT_PTR: return "GetElementPtrFn";
        case FuncTag::EXTRACT_ELEMENT: return "ExtractElementFn";
        case FuncTag::INSERT_ELEMENT: return "InsertElementFn";
        case FuncTag::ARRAY: return "ArrayFn";
        case FuncTag::STRUCT: return "StructFn";
        case FuncTag::MAT_FULL: return "MatFullFn";
        case FuncTag::MAT2: return "Mat2Fn";
        case FuncTag::MAT3: return "Mat3Fn";
        case FuncTag::MAT4: return "Mat4Fn";
        case FuncTag::BINDLESS_ATOMIC_EXCHANGE: return "BindlessAtomicExchangeFn";
        case FuncTag::BINDLESS_ATOMIC_COMPARE_EXCHANGE: return "BindlessAtomicCompareExchangeFn";
        case FuncTag::BINDLESS_ATOMIC_FETCH_ADD: return "BindlessAtomicFetchAddFn";
        case FuncTag::BINDLESS_ATOMIC_FETCH_SUB: return "BindlessAtomicFetchSubFn";
        case FuncTag::BINDLESS_ATOMIC_FETCH_AND: return "BindlessAtomicFetchAndFn";
        case FuncTag::BINDLESS_ATOMIC_FETCH_OR: return "BindlessAtomicFetchOrFn";
        case FuncTag::BINDLESS_ATOMIC_FETCH_XOR: return "BindlessAtomicFetchXorFn";
        case FuncTag::BINDLESS_ATOMIC_FETCH_MIN: return "BindlessAtomicFetchMinFn";
        case FuncTag::BINDLESS_ATOMIC_FETCH_MAX: return "BindlessAtomicFetchMaxFn";
        case FuncTag::CALLABLE: return "CallableFn";
        case FuncTag::CPU_EXT: return "CpuExtFn";
        case FuncTag::SHADER_EXECUTION_REORDER: return "ShaderExecutionReorderFn";
    }
    return "unknown";
}
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
inline const char *tag_name(InstructionTag tag) {
    switch (tag) {
        case InstructionTag::BUFFER: return "BufferInst";
        case InstructionTag::TEXTURE2D: return "Texture2dInst";
        case InstructionTag::TEXTURE3D: return "Texture3dInst";
        case InstructionTag::BINDLESS_ARRAY: return "BindlessArrayInst";
        case InstructionTag::ACCEL: return "AccelInst";
        case InstructionTag::SHARED: return "SharedInst";
        case InstructionTag::UNIFORM: return "UniformInst";
        case InstructionTag::ARGUMENT: return "ArgumentInst";
        case InstructionTag::CONSTANT: return "ConstantInst";
        case InstructionTag::CALL: return "CallInst";
        case InstructionTag::PHI: return "PhiInst";
        case InstructionTag::BASIC_BLOCK_SENTINEL: return "BasicBlockSentinelInst";
        case InstructionTag::IF: return "IfInst";
        case InstructionTag::GENERIC_LOOP: return "GenericLoopInst";
        case InstructionTag::SWITCH: return "SwitchInst";
        case InstructionTag::LOCAL: return "LocalInst";
        case InstructionTag::BREAK: return "BreakInst";
        case InstructionTag::CONTINUE: return "ContinueInst";
        case InstructionTag::RETURN: return "ReturnInst";
        case InstructionTag::PRINT: return "PrintInst";
        case InstructionTag::UPDATE: return "UpdateInst";
        case InstructionTag::RAY_QUERY: return "RayQueryInst";
        case InstructionTag::REV_AUTODIFF: return "RevAutodiffInst";
        case InstructionTag::FWD_AUTODIFF: return "FwdAutodiffInst";
    }
    return "unknown";
}
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
inline const char *tag_name(BindingTag tag) {
    switch (tag) {
        case BindingTag::BUFFER_BINDING: return "BufferBinding";
        case BindingTag::TEXTURE_BINDING: return "TextureBinding";
        case BindingTag::BINDLESS_ARRAY_BINDING: return "BindlessArrayBinding";
        case BindingTag::ACCEL_BINDING: return "AccelBinding";
    }
    return "unknown";
}
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
const Type *ir_v2_binding_type_extract(const Type *ty, uint32_t index);
size_t ir_v2_binding_type_size(const Type *ty);
size_t ir_v2_binding_type_alignment(const Type *ty);
bool ir_v2_binding_type_is_scalar(const Type *ty);
bool ir_v2_binding_type_is_bool(const Type *ty);
bool ir_v2_binding_type_is_int16(const Type *ty);
bool ir_v2_binding_type_is_int32(const Type *ty);
bool ir_v2_binding_type_is_int64(const Type *ty);
bool ir_v2_binding_type_is_uint16(const Type *ty);
bool ir_v2_binding_type_is_uint32(const Type *ty);
bool ir_v2_binding_type_is_uint64(const Type *ty);
bool ir_v2_binding_type_is_float16(const Type *ty);
bool ir_v2_binding_type_is_float32(const Type *ty);
bool ir_v2_binding_type_is_array(const Type *ty);
bool ir_v2_binding_type_is_vector(const Type *ty);
bool ir_v2_binding_type_is_struct(const Type *ty);
bool ir_v2_binding_type_is_custom(const Type *ty);
bool ir_v2_binding_type_is_matrix(const Type *ty);
const Type *ir_v2_binding_type_element(const Type *ty);
Slice<const char> ir_v2_binding_type_description(const Type *ty);
size_t ir_v2_binding_type_dimension(const Type *ty);
Slice<const Type *const> ir_v2_binding_type_members(const Type *ty);
const Type *ir_v2_binding_make_struct(size_t alignment, const Type **tys, uint32_t count);
const Type *ir_v2_binding_make_array(const Type *ty, uint32_t count);
const Type *ir_v2_binding_make_vector(const Type *ty, uint32_t count);
const Type *ir_v2_binding_make_matrix(uint32_t dim);
const Type *ir_v2_binding_make_custom(Slice<const char> name);
const Type *ir_v2_binding_from_desc(Slice<const char> desc);
const Type *ir_v2_binding_type_bool();
const Type *ir_v2_binding_type_int16();
const Type *ir_v2_binding_type_int32();
const Type *ir_v2_binding_type_int64();
const Type *ir_v2_binding_type_uint16();
const Type *ir_v2_binding_type_uint32();
const Type *ir_v2_binding_type_uint64();
const Type *ir_v2_binding_type_float16();
const Type *ir_v2_binding_type_float32();
const Node *ir_v2_binding_node_prev(const Node *node);
const Node *ir_v2_binding_node_next(const Node *node);
const CInstruction *ir_v2_binding_node_inst(const Node *node);
const Node *ir_v2_binding_basic_block_first(const BasicBlock *block);
const Node *ir_v2_binding_basic_block_last(const BasicBlock *block);
}// namespace luisa::compute::ir_v2
