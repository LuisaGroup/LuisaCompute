pub mod convert;
use crate::ir::{Binding, KernelModule, Primitive};

use serde::{Deserialize, Serialize};
use half::f16;

#[derive(Clone, Serialize, Deserialize)]
pub struct SerializedKernelModule {
    pub blocks: Vec<SerializedBlock>,
    pub nodes: Vec<SerializedNode>,
    pub types: Vec<SerializedType>,
    pub entry: SerializedBlockRef,
    pub captures: Vec<SerializedCapture>,
    pub args: Vec<SerializedNodeRef>,
    pub shared: Vec<SerializedNodeRef>,
    pub block_size: [u32; 3],
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SerializedCallableModule {
    pub id: u64,
    pub name: String,
    pub entry: SerializedBlockRef,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct SerializedCallableModuleRef(pub u64);
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct SerializedNodeRef(pub u64);
#[derive(Clone, Serialize, Deserialize)]
pub struct SerializedNode {
    pub ty: SerializedTypeRef,
    pub inst: SerializedInstruction,
}
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct SerializedBlockRef(pub u64);

#[derive(Clone, Serialize, Deserialize)]
pub struct SerializedBlock {
    pub nodes: Vec<SerializedNode>,
}
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct SerializedTypeRef(pub u64);
#[derive(Clone, Serialize, Deserialize)]
pub enum SerializedType {
    Void,
    UserData,
    Primitive(Primitive),
    Vector(Primitive, u32),
    Matrix(Primitive, u32),
    Array(SerializedTypeRef, u32),
    Struct {
        fields: Vec<SerializedTypeRef>,
        align: u32,
        size: u32,
    },
    Opqaue(String),
}
#[derive(Serialize, Deserialize, Clone)]
#[repr(C)]
pub struct SerializedCapture {
    pub node: SerializedNodeRef,
    pub binding: Binding,
}
#[derive(Clone, Serialize, Deserialize)]
pub enum SerializedConst {
    Zero(SerializedTypeRef),
    One(SerializedTypeRef),
    Bool(bool),
    Int8(i8),
    Uint8(u8),
    Int16(i16),
    Uint16(u16),
    Int32(i32),
    Uint32(u32),
    Int64(i64),
    Uint64(u64),
    Float16(f16),
    Float32(f32),
    Float64(f64),
    Generic(Vec<u8>, SerializedTypeRef),
}
#[repr(C)]
#[derive(Clone, Serialize, Deserialize)]
pub struct SerializedSwitchCase {
    pub value: i32,
    pub block: SerializedBlockRef,
}
#[derive(Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct SerializedPhiIncoming {
    pub value: SerializedNodeRef,
    pub block: SerializedBlockRef,
}
#[derive(Clone, Serialize, Deserialize)]
pub enum SerializedInstruction {
    Buffer,
    Bindless,
    Texture2D,
    Texture3D,
    Accel,
    Shared,
    Uniform,
    Local {
        init: SerializedNodeRef,
    },
    Argument {
        by_value: bool,
    },
    UserData,
    Invalid,
    Const(SerializedConst),

    Update {
        var: SerializedNodeRef,
        value: SerializedNodeRef,
    },

    Call(SerializedFunc, Vec<SerializedNodeRef>),

    Phi(Vec<SerializedPhiIncoming>),
    Return(SerializedNodeRef),
    Loop {
        body: SerializedBlockRef,
        cond: SerializedNodeRef,
    },
    GenericLoop {
        prepare: SerializedBlockRef,
        cond: SerializedNodeRef,
        body: SerializedBlockRef,
        update: SerializedBlockRef,
    },
    Break,
    Continue,
    If {
        cond: SerializedNodeRef,
        true_branch: SerializedBlockRef,
        false_branch: SerializedBlockRef,
    },
    Switch {
        value: SerializedNodeRef,
        default: SerializedBlockRef,
        cases: Vec<SerializedSwitchCase>,
    },
    AdScope {
        body: SerializedBlockRef,
        forward:bool,
    },
    AdDetach(SerializedBlockRef),
    RayQuery {
        ray_query: SerializedNodeRef,
        on_triangle_hit: SerializedBlockRef,
        on_procedural_hit: SerializedBlockRef,
    },
    Comment(Vec<u8>),
    Assert(SerializedNodeRef, Vec<u8>),
}

#[derive(Clone, Serialize, Deserialize)]
pub enum SerializedFunc {

    ZeroInitializer,

    Assume,
    Unreachable(Vec<u8>),
    Assert(Vec<u8>),

    ThreadId,
    BlockId,
    WarpSize,
    WarpLaneId,
    DispatchId,
    DispatchSize,

    RequiresGradient,
    Backward,
    // marks the beginning of backward pass
    Gradient,
    GradientMarker,
    // marks a (node, gradient) tuple
    AccGrad,
    // grad (local), increment
    Detach,

    // (handle, instance_id) -> Mat4
    RayTracingInstanceTransform,
    RayTracingSetInstanceTransform,
    RayTracingSetInstanceOpacity,
    RayTracingSetInstanceVisibility,
    // (handle, Ray, mask) -> Hit
    // struct Ray alignas(16) { float origin[3], tmin; float direction[3], tmax; };
    // struct Hit alignas(16) { uint inst; uint prim; float u; float v; };
    RayTracingTraceClosest,
    // (handle, Ray, mask) -> bool
    RayTracingTraceAny,
    RayTracingQueryAll,
    // (ray, mask)-> rq
    RayTracingQueryAny, // (ray, mask)-> rq

    RayQueryWorldSpaceRay,
    // (rq) -> Ray
    RayQueryProceduralCandidateHit,
    // (rq) -> ProceduralHit
    RayQueryTriangleCandidateHit,
    // (rq) -> TriangleHit
    RayQueryCommittedHit,
    // (rq) -> CommitedHit
    RayQueryCommitTriangle,
    // (rq) -> ()
    RayQueryCommitProcedural,
    // (rq, f32) -> ()
    RayQueryTerminate, // (rq) -> ()

    RasterDiscard,

    IndirectDispatchSetCount,
    IndirectDispatchSetKernel,

    /// When referencing a Local in Call, it is always interpreted as a load
    /// However, there are cases you want to do this explicitly
    Load,

    Cast,
    Bitcast,

    Pack,
    Unpack,

    // Binary op
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    RotRight,
    RotLeft,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    MatCompMul,

    // Unary op
    Neg,
    Not,
    BitNot,

    All,
    Any,

    // select(p, a, b) => p ? a : b
    Select,
    Clamp,
    Lerp,
    Step,
    SmoothStep,
    Saturate,

    Abs,
    Min,
    Max,

    // reduction
    ReduceSum,
    ReduceProd,
    ReduceMin,
    ReduceMax,

    // bit manipulation
    Clz,
    Ctz,
    PopCount,
    Reverse,

    IsInf,
    IsNan,

    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atan2,
    Atanh,

    Cos,
    Cosh,
    Sin,
    Sinh,
    Tan,
    Tanh,

    Exp,
    Exp2,
    Exp10,
    Log,
    Log2,
    Log10,
    Powi,
    Powf,

    Sqrt,
    Rsqrt,

    Ceil,
    Floor,
    Fract,
    Trunc,
    Round,

    Fma,
    Copysign,

    // Vector operations
    Cross,
    Dot,
    // outer_product(a, b) => a * b^T
    OuterProduct,
    Length,
    LengthSquared,
    Normalize,
    Faceforward,
    // reflect(i, n) => i - 2 * dot(n, i) * n
    Reflect,

    // Matrix operations
    Determinant,
    Transpose,
    Inverse,

    WarpIsFirstActiveLane,
    WarpFirstActiveLane,
    WarpActiveAllEqual,
    WarpActiveBitAnd,
    WarpActiveBitOr,
    WarpActiveBitXor,
    WarpActiveCountBits,
    WarpActiveMax,
    WarpActiveMin,
    WarpActiveProduct,
    WarpActiveSum,
    WarpActiveAll,
    WarpActiveAny,
    WarpActiveBitMask,
    WarpPrefixCountBits,
    WarpPrefixSum,
    WarpPrefixProduct,
    WarpReadLaneAt,
    WarpReadFirstLane,

    SynchronizeBlock,

    /// (buffer/smem, indices..., desired) -> old: stores desired, returns old.
    AtomicExchange,
    /// (buffer/smem, indices..., expected, desired) -> old: stores (old == expected ? desired : old), returns old.
    AtomicCompareExchange,
    /// (buffer/smem, indices..., val) -> old: stores (old + val), returns old.
    AtomicFetchAdd,
    /// (buffer/smem, indices..., val) -> old: stores (old - val), returns old.
    AtomicFetchSub,
    /// (buffer/smem, indices..., val) -> old: stores (old & val), returns old.
    AtomicFetchAnd,
    /// (buffer/smem, indices..., val) -> old: stores (old | val), returns old.
    AtomicFetchOr,
    /// (buffer/smem, indices..., val) -> old: stores (old ^ val), returns old.
    AtomicFetchXor,
    /// (buffer/smem, indices..., val) -> old: stores min(old, val), returns old.
    AtomicFetchMin,
    /// (buffer/smem, indices..., val) -> old: stores max(old, val), returns old.
    AtomicFetchMax,
    // memory access
    /// (buffer, index) -> value: reads the index-th element in buffer
    BufferRead,
    /// (buffer, index, value) -> void: writes value into the indeex
    BufferWrite,
    /// buffer -> uint: returns buffer size in *elements*
    BufferSize,
    /// (texture, coord) -> value
    Texture2dRead,
    /// (texture, coord, value) -> void
    Texture2dWrite,
    /// (texture, coord) -> value
    Texture3dRead,
    /// (texture, coord, value) -> void
    Texture3dWrite,
    ///(bindless_array, index: uint, uv: float2) -> float4
    BindlessTexture2dSample,
    ///(bindless_array, index: uint, uv: float2, level: float) -> float4
    BindlessTexture2dSampleLevel,
    ///(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2) -> float4
    BindlessTexture2dSampleGrad,
    ///(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2, min_mip: float) -> float4
    BindlessTexture2dSampleGradLevel,
    ///(bindless_array, index: uint, uv: float3) -> float4
    BindlessTexture3dSample,
    ///(bindless_array, index: uint, uv: float3, level: float) -> float4
    BindlessTexture3dSampleLevel,
    ///(bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3) -> float4
    BindlessTexture3dSampleGrad,
    ///(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2, min_mip: float) -> float4
    BindlessTexture3dSampleGradLevel,
    ///(bindless_array, index: uint, coord: uint2) -> float4
    BindlessTexture2dRead,
    ///(bindless_array, index: uint, coord: uint3) -> float4
    BindlessTexture3dRead,
    ///(bindless_array, index: uint, coord: uint2, level: uint) -> float4
    BindlessTexture2dReadLevel,
    ///(bindless_array, index: uint, coord: uint3, level: uint) -> float4
    BindlessTexture3dReadLevel,
    ///(bindless_array, index: uint) -> uint2
    BindlessTexture2dSize,
    ///(bindless_array, index: uint) -> uint3
    BindlessTexture3dSize,
    ///(bindless_array, index: uint, level: uint) -> uint2
    BindlessTexture2dSizeLevel,
    ///(bindless_array, index: uint, level: uint) -> uint3
    BindlessTexture3dSizeLevel,
    /// (bindless_array, index: uint, element: uint) -> T
    BindlessBufferRead,
    /// (bindless_array, index: uint, stride: uint) -> uint: returns the size of the buffer in *elements*
    BindlessBufferSize,
    // (bindless_array, index: uint) -> u64: returns the type of the buffer
    BindlessBufferType,

    // scalar -> vector, the resulting type is stored in node
    Vec,
    // (scalar, scalar) -> vector
    Vec2,
    // (scalar, scalar, scalar) -> vector
    Vec3,
    // (scalar, scalar, scalar, scalar) -> vector
    Vec4,

    // (vector, indices,...) -> vector
    Permute,
    // (vector, scalar, index) -> vector
    InsertElement,
    // (vector, index) -> scalar
    ExtractElement,
    //(struct, index) -> value; the value can be passed to an Update instruction
    GetElementPtr,
    // (fields, ...) -> struct
    Struct,

    // (fields, ...) -> array
    Array,

    // scalar -> matrix, all elements are set to the scalar
    Mat,
    // vector x 2 -> matrix
    Mat2,
    // vector x 3 -> matrix
    Mat3,
    // vector x 4 -> matrix
    Mat4,

    Callable(SerializedCallableModuleRef),

    ShaderExecutionReorder, // (uint hint, uint hint_bits): void
}
pub fn serialize_kernel_module_to_json(m: &KernelModule) -> serde_json::Value {
    let v = convert::serialize_kernel_module(m);
    serde_json::to_value(v).unwrap()
}
pub fn serialize_kernel_module_to_json_str(m: &KernelModule) -> String {
    let json = serialize_kernel_module_to_json(m);
    serde_json::to_string(&json).unwrap()
}
