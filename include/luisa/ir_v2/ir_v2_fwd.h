#pragma once
// if msvc
#ifdef _MSC_VER
#pragma warning(disable : 4190)
#endif

#include <cstdint>
namespace luisa::compute {
class Type;
}// namespace luisa::compute
namespace luisa::compute::ir_v2 {

struct Node;
class BasicBlock;
struct CallableModule;
struct Module;
struct KernelModule;

struct PhiIncoming {
    const BasicBlock *block = nullptr;
    const Node *value = nullptr;
};
struct SwitchCase {
    int32_t value = 0;
    const BasicBlock *block = nullptr;
};
struct CpuExternFn {
    void *data = nullptr;
    void (*func)(void *data, void *args) = nullptr;
    void (*dtor)(void *data) = nullptr;
    const Type *arg_ty = nullptr;
};

struct Func;
struct Zero;
struct One;
struct Assume;
struct Unreachable;
struct ThreadId;
struct BlockId;
struct WarpSize;
struct WarpLaneId;
struct DispatchId;
struct DispatchSize;
struct PropagateGradient;
struct OutputGradient;
struct RequiresGradient;
struct Backward;
struct Gradient;
struct AccGrad;
struct Detach;
struct RayTracingInstanceTransform;
struct RayTracingInstanceVisibilityMask;
struct RayTracingInstanceUserId;
struct RayTracingSetInstanceTransform;
struct RayTracingSetInstanceOpacity;
struct RayTracingSetInstanceVisibility;
struct RayTracingSetInstanceUserId;
struct RayTracingTraceClosest;
struct RayTracingTraceAny;
struct RayTracingQueryAll;
struct RayTracingQueryAny;
struct RayQueryWorldSpaceRay;
struct RayQueryProceduralCandidateHit;
struct RayQueryTriangleCandidateHit;
struct RayQueryCommittedHit;
struct RayQueryCommitTriangle;
struct RayQueryCommitdProcedural;
struct RayQueryTerminate;
struct Load;
struct Store;
struct Cast;
struct BitCast;
struct Add;
struct Sub;
struct Mul;
struct Div;
struct Rem;
struct BitAnd;
struct BitOr;
struct BitXor;
struct Shl;
struct Shr;
struct RotRight;
struct RotLeft;
struct Eq;
struct Ne;
struct Lt;
struct Le;
struct Gt;
struct Ge;
struct MatCompMul;
struct Neg;
struct Not;
struct BitNot;
struct All;
struct Any;
struct Select;
struct Clamp;
struct Lerp;
struct Step;
struct Saturate;
struct SmoothStep;
struct Abs;
struct Min;
struct Max;
struct ReduceSum;
struct ReduceProd;
struct ReduceMin;
struct ReduceMax;
struct Clz;
struct Ctz;
struct PopCount;
struct Reverse;
struct IsInf;
struct IsNan;
struct Acos;
struct Acosh;
struct Asin;
struct Asinh;
struct Atan;
struct Atan2;
struct Atanh;
struct Cos;
struct Cosh;
struct Sin;
struct Sinh;
struct Tan;
struct Tanh;
struct Exp;
struct Exp2;
struct Exp10;
struct Log;
struct Log2;
struct Log10;
struct Powi;
struct Powf;
struct Sqrt;
struct Rsqrt;
struct Ceil;
struct Floor;
struct Fract;
struct Trunc;
struct Round;
struct Fma;
struct Copysign;
struct Cross;
struct Dot;
struct OuterProduct;
struct Length;
struct LengthSquared;
struct Normalize;
struct Faceforward;
struct Distance;
struct Reflect;
struct Determinant;
struct Transpose;
struct Inverse;
struct WarpIsFirstActiveLane;
struct WarpFirstActiveLane;
struct WarpActiveAllEqual;
struct WarpActiveBitAnd;
struct WarpActiveBitOr;
struct WarpActiveBitXor;
struct WarpActiveCountBits;
struct WarpActiveMax;
struct WarpActiveMin;
struct WarpActiveProduct;
struct WarpActiveSum;
struct WarpActiveAll;
struct WarpActiveAny;
struct WarpActiveBitMask;
struct WarpPrefixCountBits;
struct WarpPrefixSum;
struct WarpPrefixProduct;
struct WarpReadLaneAt;
struct WarpReadFirstLane;
struct SynchronizeBlock;
struct AtomicExchange;
struct AtomicCompareExchange;
struct AtomicFetchAdd;
struct AtomicFetchSub;
struct AtomicFetchAnd;
struct AtomicFetchOr;
struct AtomicFetchXor;
struct AtomicFetchMin;
struct AtomicFetchMax;
struct BufferWrite;
struct BufferRead;
struct BufferSize;
struct ByteBufferWrite;
struct ByteBufferRead;
struct ByteBufferSize;
struct Texture2dRead;
struct Texture2dWrite;
struct Texture2dSize;
struct Texture3dRead;
struct Texture3dWrite;
struct Texture3dSize;
struct BindlessTexture2dSample;
struct BindlessTexture2dSampleLevel;
struct BindlessTexture2dSampleGrad;
struct BindlessTexture2dSampleGradLevel;
struct BindlessTexture2dRead;
struct BindlessTexture2dSize;
struct BindlessTexture2dSizeLevel;
struct BindlessTexture3dSample;
struct BindlessTexture3dSampleLevel;
struct BindlessTexture3dSampleGrad;
struct BindlessTexture3dSampleGradLevel;
struct BindlessTexture3dRead;
struct BindlessTexture3dSize;
struct BindlessTexture3dSizeLevel;
struct BindlessBufferWrite;
struct BindlessBufferRead;
struct BindlessBufferSize;
struct BindlessByteBufferWrite;
struct BindlessByteBufferRead;
struct BindlessByteBufferSize;
struct Vec;
struct Vec2;
struct Vec3;
struct Vec4;
struct Permute;
struct GetElementPtr;
struct ExtractElement;
struct InsertElement;
struct Array;
struct Struct;
struct MatFull;
struct Mat2;
struct Mat3;
struct Mat4;
struct BindlessAtomicExchange;
struct BindlessAtomicCompareExchange;
struct BindlessAtomicFetchAdd;
struct BindlessAtomicFetchSub;
struct BindlessAtomicFetchAnd;
struct BindlessAtomicFetchOr;
struct BindlessAtomicFetchXor;
struct BindlessAtomicFetchMin;
struct BindlessAtomicFetchMax;
struct Callable;
struct CpuExt;
struct ShaderExecutionReorder;
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
    STORE,
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
struct Buffer;
struct Texture2d;
struct Texture3d;
struct BindlessArray;
struct Accel;
struct Shared;
struct Uniform;
struct Argument;
struct Const;
struct Call;
struct Phi;
struct BasicBlockSentinel;
struct If;
struct GenericLoop;
struct Switch;
struct Local;
struct Break;
struct Continue;
struct Return;
struct Print;
enum class InstructionTag : unsigned int {
    BUFFER,
    TEXTURE2D,
    TEXTURE3D,
    BINDLESS_ARRAY,
    ACCEL,
    SHARED,
    UNIFORM,
    ARGUMENT,
    CONST,
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
