#pragma once
// if msvc
#ifdef _MSC_VER
#pragma warning(disable : 4190)
#endif

#include <cstdint>
#include <array>
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
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

struct Func;
struct ZeroFn;
struct OneFn;
struct AssumeFn;
struct UnreachableFn;
struct ThreadIdFn;
struct BlockIdFn;
struct WarpSizeFn;
struct WarpLaneIdFn;
struct DispatchIdFn;
struct DispatchSizeFn;
struct PropagateGradientFn;
struct OutputGradientFn;
struct RequiresGradientFn;
struct BackwardFn;
struct GradientFn;
struct AccGradFn;
struct DetachFn;
struct RayTracingInstanceTransformFn;
struct RayTracingInstanceVisibilityMaskFn;
struct RayTracingInstanceUserIdFn;
struct RayTracingSetInstanceTransformFn;
struct RayTracingSetInstanceOpacityFn;
struct RayTracingSetInstanceVisibilityFn;
struct RayTracingSetInstanceUserIdFn;
struct RayTracingTraceClosestFn;
struct RayTracingTraceAnyFn;
struct RayTracingQueryAllFn;
struct RayTracingQueryAnyFn;
struct RayQueryWorldSpaceRayFn;
struct RayQueryProceduralCandidateHitFn;
struct RayQueryTriangleCandidateHitFn;
struct RayQueryCommittedHitFn;
struct RayQueryCommitTriangleFn;
struct RayQueryCommitdProceduralFn;
struct RayQueryTerminateFn;
struct LoadFn;
struct CastFn;
struct BitCastFn;
struct AddFn;
struct SubFn;
struct MulFn;
struct DivFn;
struct RemFn;
struct BitAndFn;
struct BitOrFn;
struct BitXorFn;
struct ShlFn;
struct ShrFn;
struct RotRightFn;
struct RotLeftFn;
struct EqFn;
struct NeFn;
struct LtFn;
struct LeFn;
struct GtFn;
struct GeFn;
struct MatCompMulFn;
struct NegFn;
struct NotFn;
struct BitNotFn;
struct AllFn;
struct AnyFn;
struct SelectFn;
struct ClampFn;
struct LerpFn;
struct StepFn;
struct SaturateFn;
struct SmoothStepFn;
struct AbsFn;
struct MinFn;
struct MaxFn;
struct ReduceSumFn;
struct ReduceProdFn;
struct ReduceMinFn;
struct ReduceMaxFn;
struct ClzFn;
struct CtzFn;
struct PopCountFn;
struct ReverseFn;
struct IsInfFn;
struct IsNanFn;
struct AcosFn;
struct AcoshFn;
struct AsinFn;
struct AsinhFn;
struct AtanFn;
struct Atan2Fn;
struct AtanhFn;
struct CosFn;
struct CoshFn;
struct SinFn;
struct SinhFn;
struct TanFn;
struct TanhFn;
struct ExpFn;
struct Exp2Fn;
struct Exp10Fn;
struct LogFn;
struct Log2Fn;
struct Log10Fn;
struct PowiFn;
struct PowfFn;
struct SqrtFn;
struct RsqrtFn;
struct CeilFn;
struct FloorFn;
struct FractFn;
struct TruncFn;
struct RoundFn;
struct FmaFn;
struct CopysignFn;
struct CrossFn;
struct DotFn;
struct OuterProductFn;
struct LengthFn;
struct LengthSquaredFn;
struct NormalizeFn;
struct FaceforwardFn;
struct DistanceFn;
struct ReflectFn;
struct DeterminantFn;
struct TransposeFn;
struct InverseFn;
struct WarpIsFirstActiveLaneFn;
struct WarpFirstActiveLaneFn;
struct WarpActiveAllEqualFn;
struct WarpActiveBitAndFn;
struct WarpActiveBitOrFn;
struct WarpActiveBitXorFn;
struct WarpActiveCountBitsFn;
struct WarpActiveMaxFn;
struct WarpActiveMinFn;
struct WarpActiveProductFn;
struct WarpActiveSumFn;
struct WarpActiveAllFn;
struct WarpActiveAnyFn;
struct WarpActiveBitMaskFn;
struct WarpPrefixCountBitsFn;
struct WarpPrefixSumFn;
struct WarpPrefixProductFn;
struct WarpReadLaneAtFn;
struct WarpReadFirstLaneFn;
struct SynchronizeBlockFn;
struct AtomicExchangeFn;
struct AtomicCompareExchangeFn;
struct AtomicFetchAddFn;
struct AtomicFetchSubFn;
struct AtomicFetchAndFn;
struct AtomicFetchOrFn;
struct AtomicFetchXorFn;
struct AtomicFetchMinFn;
struct AtomicFetchMaxFn;
struct BufferWriteFn;
struct BufferReadFn;
struct BufferSizeFn;
struct ByteBufferWriteFn;
struct ByteBufferReadFn;
struct ByteBufferSizeFn;
struct Texture2dReadFn;
struct Texture2dWriteFn;
struct Texture2dSizeFn;
struct Texture3dReadFn;
struct Texture3dWriteFn;
struct Texture3dSizeFn;
struct BindlessTexture2dSampleFn;
struct BindlessTexture2dSampleLevelFn;
struct BindlessTexture2dSampleGradFn;
struct BindlessTexture2dSampleGradLevelFn;
struct BindlessTexture2dReadFn;
struct BindlessTexture2dSizeFn;
struct BindlessTexture2dSizeLevelFn;
struct BindlessTexture3dSampleFn;
struct BindlessTexture3dSampleLevelFn;
struct BindlessTexture3dSampleGradFn;
struct BindlessTexture3dSampleGradLevelFn;
struct BindlessTexture3dReadFn;
struct BindlessTexture3dSizeFn;
struct BindlessTexture3dSizeLevelFn;
struct BindlessBufferWriteFn;
struct BindlessBufferReadFn;
struct BindlessBufferSizeFn;
struct BindlessByteBufferWriteFn;
struct BindlessByteBufferReadFn;
struct BindlessByteBufferSizeFn;
struct VecFn;
struct Vec2Fn;
struct Vec3Fn;
struct Vec4Fn;
struct PermuteFn;
struct GetElementPtrFn;
struct ExtractElementFn;
struct InsertElementFn;
struct ArrayFn;
struct StructFn;
struct MatFullFn;
struct Mat2Fn;
struct Mat3Fn;
struct Mat4Fn;
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
struct ShaderExecutionReorderFn;
enum class FuncTag : unsigned int {
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
struct Instruction;
struct BufferInst;
struct Texture2dInst;
struct Texture3dInst;
struct BindlessArrayInst;
struct AccelInst;
struct SharedInst;
struct UniformInst;
struct ArgumentInst;
struct ConstantInst;
struct CallInst;
struct PhiInst;
struct BasicBlockSentinelInst;
struct IfInst;
struct GenericLoopInst;
struct SwitchInst;
struct LocalInst;
struct BreakInst;
struct ContinueInst;
struct ReturnInst;
struct PrintInst;
struct UpdateInst;
struct RayQueryInst;
struct RevAutodiffInst;
struct FwdAutodiffInst;
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
struct Binding;
struct BufferBinding;
struct TextureBinding;
struct BindlessArrayBinding;
struct AccelBinding;
enum class BindingTag : unsigned int {
    BUFFER_BINDING,
    TEXTURE_BINDING,
    BINDLESS_ARRAY_BINDING,
    ACCEL_BINDING,
};
}// namespace luisa::compute::ir_v2
