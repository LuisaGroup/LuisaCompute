use half::f16;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

use crate::ast2ir;
use crate::usage_detect::detect_usage;
use crate::*;
use bitflags::bitflags;
use std::any::{Any, TypeId};
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::hash::Hasher;
use std::ops::Deref;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
#[derive(Serialize, Deserialize)]
pub enum Primitive {
    Bool,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Float16,
    Float32,
    Float64,
}

impl std::fmt::Display for Primitive {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Bool => "bool",
                Self::Int8 => "i8",
                Self::Uint8 => "u8",
                Self::Int16 => "i16",
                Self::Uint16 => "u16",
                Self::Int32 => "i32",
                Self::Uint32 => "u32",
                Self::Int64 => "i64",
                Self::Uint64 => "u64",
                Self::Float16 => "f16",
                Self::Float32 => "f32",
                Self::Float64 => "f64",
            }
        )
    }
}

//cbindgen:derive-tagged-enum-destructor
//cbindgen:derive-tagged-enum-copy-constructor
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub enum VectorElementType {
    Scalar(Primitive),
    Vector(CArc<VectorType>),
}

impl VectorElementType {
    pub fn as_primitive(&self) -> Option<Primitive> {
        match self {
            VectorElementType::Scalar(p) => Some(*p),
            _ => None,
        }
    }
    pub fn is_float(&self) -> bool {
        match self {
            VectorElementType::Scalar(Primitive::Float16) => true,
            VectorElementType::Scalar(Primitive::Float32) => true,
            VectorElementType::Scalar(Primitive::Float64) => true,
            VectorElementType::Vector(v) => v.element.is_float(),
            _ => false,
        }
    }
    pub fn is_int(&self) -> bool {
        match self {
            VectorElementType::Scalar(Primitive::Int8) => true,
            VectorElementType::Scalar(Primitive::Uint8) => true,
            VectorElementType::Scalar(Primitive::Int16) => true,
            VectorElementType::Scalar(Primitive::Uint16) => true,
            VectorElementType::Scalar(Primitive::Int32) => true,
            VectorElementType::Scalar(Primitive::Uint32) => true,
            VectorElementType::Scalar(Primitive::Int64) => true,
            VectorElementType::Scalar(Primitive::Uint64) => true,
            VectorElementType::Vector(v) => v.element.is_int(),
            _ => false,
        }
    }
    pub fn is_bool(&self) -> bool {
        match self {
            VectorElementType::Scalar(Primitive::Bool) => true,
            VectorElementType::Vector(v) => v.element.is_bool(),
            _ => false,
        }
    }
    pub fn to_type(&self) -> CArc<Type> {
        match self {
            VectorElementType::Scalar(p) => context::register_type(Type::Primitive(*p)),
            VectorElementType::Vector(v) => context::register_type(Type::Vector(v.deref().clone())),
        }
    }
}

impl std::fmt::Display for VectorElementType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scalar(primitive) => std::fmt::Display::fmt(primitive, f),
            Self::Vector(vector) => std::fmt::Display::fmt(vector, f),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub struct VectorType {
    pub element: VectorElementType,
    pub length: u32,
}

impl std::fmt::Display for VectorType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec<{};{}>", self.element, self.length)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub struct MatrixType {
    pub element: VectorElementType,
    pub dimension: u32,
}

impl std::fmt::Display for MatrixType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mat<{};{}>", self.element, self.dimension)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub struct StructType {
    pub fields: CBoxedSlice<CArc<Type>>,
    pub alignment: usize,
    pub size: usize,
    // pub id: u64,
}

impl std::fmt::Display for StructType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Struct<")?;
        for field in self.fields.as_ref().iter() {
            write!(f, "{},", field)?;
        }
        write!(f, ">")?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub struct ArrayType {
    pub element: CArc<Type>,
    pub length: usize,
}

impl std::fmt::Display for ArrayType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Arr<{}; {}>", self.element, self.length)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub enum Type {
    Void,
    UserData,
    Primitive(Primitive),
    Vector(VectorType),
    Matrix(MatrixType),
    Struct(StructType),
    Array(ArrayType),
    Opaque(CBoxedSlice<u8>),
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UserData => write!(f, "userdata"),
            Self::Void => write!(f, "void"),
            Self::Primitive(primitive) => std::fmt::Display::fmt(primitive, f),
            Self::Vector(vector) => std::fmt::Display::fmt(vector, f),
            Self::Matrix(matrix) => std::fmt::Display::fmt(matrix, f),
            Self::Struct(struct_type) => std::fmt::Display::fmt(struct_type, f),
            Self::Array(arr) => std::fmt::Display::fmt(arr, f),
            Self::Opaque(name) => std::fmt::Display::fmt(&name.to_string(), f),
        }
    }
}

impl VectorElementType {
    pub fn size(&self) -> usize {
        match self {
            VectorElementType::Scalar(p) => p.size(),
            VectorElementType::Vector(v) => v.size(),
        }
    }
}

impl Primitive {
    pub fn size(&self) -> usize {
        match self {
            Primitive::Bool => 1,
            Primitive::Int8 => 1,
            Primitive::Uint8 => 1,
            Primitive::Int16 => 2,
            Primitive::Uint16 => 2,
            Primitive::Int32 => 4,
            Primitive::Uint32 => 4,
            Primitive::Int64 => 8,
            Primitive::Uint64 => 8,
            Primitive::Float16 => 2,
            Primitive::Float32 => 4,
            Primitive::Float64 => 8,
        }
    }
}

impl VectorType {
    pub fn size(&self) -> usize {
        let el_sz = self.element.size();
        let aligned_len = {
            let four = self.length / 4;
            let rem = self.length % 4;
            if rem <= 2 {
                four * 4 + rem
            } else {
                four * 4 + 4
            }
        };
        let len = match self.element {
            VectorElementType::Scalar(_) => aligned_len,
            VectorElementType::Vector(_) => self.length,
        };
        el_sz * len as usize
    }
    pub fn alignment(&self) -> usize {
        match self.element {
            VectorElementType::Scalar(prim) => {
                let elem_align = prim.size();
                let aligned_dim = if self.length == 3 {
                    4
                } else {
                    assert!(self.length >= 2 && self.length <= 4);
                    self.length
                };
                std::cmp::min(elem_align * aligned_dim as usize, 16)
            }
            VectorElementType::Vector(_) => todo!(),
        }
    }
    pub fn element(&self) -> Primitive {
        match self.element {
            VectorElementType::Scalar(s) => s,
            _ => unreachable!(),
        }
    }
}

impl MatrixType {
    pub fn size(&self) -> usize {
        let el_sz = self.element.size();
        let len = match self.element {
            VectorElementType::Scalar(s) => match s {
                Primitive::Float32 => {
                    (match self.dimension {
                        2 => 2u32,
                        3 => 4u32,
                        4 => 4u32,
                        _ => panic!("Invalid matrix dimension"),
                    }) * self.dimension
                }
                _ => panic!("Invalid matrix element type"),
            },
            VectorElementType::Vector(_) => todo!(),
        };
        el_sz * len as usize
    }
    pub fn column(&self) -> CArc<Type> {
        match &self.element {
            VectorElementType::Scalar(t) => Type::vector(t.clone(), self.dimension),
            VectorElementType::Vector(t) => Type::vector_vector(t.clone(), self.dimension),
        }
    }
    pub fn element(&self) -> Primitive {
        match self.element {
            VectorElementType::Scalar(s) => s,
            _ => unreachable!(),
        }
    }
}

impl Type {
    pub fn void() -> CArc<Type> {
        context::register_type(Type::Void)
    }
    pub fn userdata() -> CArc<Type> {
        context::register_type(Type::UserData)
    }
    pub fn extract(&self, i: usize) -> CArc<Type> {
        match self {
            Self::Void | Self::Primitive(_) | Self::UserData => unreachable!(),
            Self::Vector(_) => self.element(),
            Self::Matrix(mt) => mt.column(),
            Self::Array(arr) => arr.element.clone(),
            Self::Struct(s) => s.fields[i].clone(),
            Self::Opaque(_) => unreachable!(),
        }
    }
    pub fn size(&self) -> usize {
        match self {
            Type::Void | Type::UserData => 0,
            Type::Primitive(t) => t.size(),
            Type::Struct(t) => t.size,
            Type::Vector(t) => t.size(),
            Type::Matrix(t) => t.size(),
            Type::Array(t) => t.element.size() * t.length,
            Self::Opaque(_) => unreachable!(),
        }
    }
    pub fn element(&self) -> CArc<Type> {
        match self {
            Type::Void | Type::Primitive(_) | Type::UserData => {
                context::register_type(self.clone())
            }
            Type::Vector(vec_type) => vec_type.element.to_type(),
            Type::Matrix(mat_type) => mat_type.element.to_type(),
            Type::Struct(_) => CArc::null(),
            Type::Array(arr_type) => arr_type.element.clone(),
            Self::Opaque(_) => unreachable!(),
        }
    }
    pub fn dimension(&self) -> usize {
        match self {
            Type::Void | Type::UserData => 0,
            Type::Primitive(_) => 1,
            Type::Vector(vec_type) => vec_type.length as usize,
            Type::Matrix(mat_type) => mat_type.dimension as usize,
            Type::Struct(struct_type) => struct_type.fields.as_ref().len(),
            Type::Array(arr_type) => arr_type.length,
            Self::Opaque(_) => unreachable!(),
        }
    }
    pub fn alignment(&self) -> usize {
        match self {
            Type::Void | Type::UserData => 0,
            Type::Primitive(t) => t.size(),
            Type::Struct(t) => t.alignment,
            Type::Vector(t) => t.alignment(),
            Type::Matrix(t) => t.column().alignment(),
            Type::Array(t) => t.element.alignment(),
            Self::Opaque(_) => unreachable!(),
        }
    }
    pub fn vector_to_bool(from: &VectorType) -> CArc<VectorType> {
        match &from.element {
            VectorElementType::Scalar(_) => CArc::new(VectorType {
                element: VectorElementType::Scalar(Primitive::Bool),
                length: from.length,
            }),
            VectorElementType::Vector(v) => Type::vector_to_bool(v.deref()),
        }
    }
    pub fn bool(from: CArc<Type>) -> CArc<Type> {
        match from.deref() {
            Type::Primitive(_) => context::register_type(Type::Primitive(Primitive::Bool)),
            Type::Vector(vec_type) => match &vec_type.element {
                VectorElementType::Scalar(_) => Type::vector(Primitive::Bool, vec_type.length),
                VectorElementType::Vector(v) => {
                    Type::vector_vector(Type::vector_to_bool(v.deref()), vec_type.length)
                }
            },
            _ => panic!("Cannot convert to bool"),
        }
    }
    pub fn opaque(name: String) -> CArc<Type> {
        context::register_type(Type::Opaque(name.into()))
    }
    pub fn vector(element: Primitive, length: u32) -> CArc<Type> {
        context::register_type(Type::Vector(VectorType {
            element: VectorElementType::Scalar(element),
            length,
        }))
    }
    pub fn vector_vector(element: CArc<VectorType>, length: u32) -> CArc<Type> {
        context::register_type(Type::Vector(VectorType {
            element: VectorElementType::Vector(element),
            length,
        }))
    }
    pub fn vector_of(element: CArc<Type>, length: u32) -> CArc<Type> {
        match element.as_ref() {
            Type::Primitive(v) => Self::vector(*v, length),
            Type::Vector(v) => Self::vector_vector(CArc::new(v.clone()), length),
            _ => panic!("Cannot create vector of non-primitive type"),
        }
    }
    pub fn matrix(element: Primitive, dimension: u32) -> CArc<Type> {
        context::register_type(Type::Matrix(MatrixType {
            element: VectorElementType::Scalar(element),
            dimension,
        }))
    }
    pub fn matrix_vector(element: CArc<VectorType>, dimension: u32) -> CArc<Type> {
        context::register_type(Type::Matrix(MatrixType {
            element: VectorElementType::Vector(element),
            dimension,
        }))
    }
    pub fn matrix_of(element: CArc<Type>, dimension: u32) -> CArc<Type> {
        match element.as_ref() {
            Type::Primitive(v) => Self::matrix(*v, dimension),
            Type::Vector(v) => Self::matrix_vector(CArc::new(v.clone()), dimension),
            _ => panic!("Cannot create matrix of non-primitive type"),
        }
    }
    pub fn array_of(element: CArc<Type>, length: u32) -> CArc<Type> {
        context::register_type(Type::Array(ArrayType {
            element,
            length: length as usize,
        }))
    }
    pub fn struct_of(alignment: u32, members: Vec<CArc<Type>>) -> CArc<Type> {
        let mut size = 0;
        let mut align = 0;
        for member in members.iter() {
            let a = member.alignment();
            size = (size + a - 1) / a * a;
            size += member.size();
            align = std::cmp::max(align, a);
        }
        assert!(
            align <= alignment as usize,
            "Struct alignment must be at least as \
                large as the largest member alignment."
        );
        align = alignment as usize;
        size = (size + align - 1) / align * align;
        context::register_type(Type::Struct(StructType {
            fields: CBoxedSlice::new(members),
            alignment: align,
            size,
        }))
    }
    pub fn is_void(&self) -> bool {
        match self {
            Type::Void => true,
            _ => false,
        }
    }
    pub fn is_opaque(&self, name: &str) -> bool {
        match self {
            Type::Opaque(name_) => name.is_empty() || name_.to_string().as_str() == name,
            _ => false,
        }
    }
    pub fn is_primitive(&self) -> bool {
        match self {
            Type::Primitive(_) => true,
            _ => false,
        }
    }
    pub fn is_struct(&self) -> bool {
        match self {
            Type::Struct(_) => true,
            _ => false,
        }
    }
    pub fn is_float(&self) -> bool {
        match self {
            Type::Primitive(p) => match p {
                Primitive::Float16 | Primitive::Float32 | Primitive::Float64 => true,
                _ => false,
            },
            Type::Vector(v) => v.element.is_float(),
            Type::Matrix(m) => m.element.is_float(),
            _ => false,
        }
    }
    pub fn is_bool(&self) -> bool {
        match self {
            Type::Primitive(p) => match p {
                Primitive::Bool => true,
                _ => false,
            },
            Type::Vector(v) => v.element.is_bool(),
            Type::Matrix(m) => m.element.is_bool(),
            _ => false,
        }
    }
    pub fn is_int(&self) -> bool {
        match self {
            Type::Primitive(p) => match p {
                Primitive::Int8
                | Primitive::Uint8
                | Primitive::Int16
                | Primitive::Uint16
                | Primitive::Int32
                | Primitive::Uint32
                | Primitive::Int64
                | Primitive::Uint64 => true,
                _ => false,
            },
            Type::Vector(v) => v.element.is_int(),
            Type::Matrix(m) => m.element.is_int(),
            _ => false,
        }
    }
    pub fn is_unsigned(&self) -> bool {
        match self {
            Type::Primitive(p) => match p {
                Primitive::Uint8 | Primitive::Uint16 | Primitive::Uint32 | Primitive::Uint64 => {
                    true
                }
                _ => false,
            },
            Type::Vector(v) => v.element.to_type().is_unsigned(),
            Type::Matrix(m) => m.element.to_type().is_unsigned(),
            _ => false,
        }
    }
    pub fn is_signed(&self) -> bool {
        match self {
            Type::Primitive(p) => match p {
                Primitive::Int8
                | Primitive::Int16
                | Primitive::Int32
                | Primitive::Int64
                | Primitive::Float16
                | Primitive::Float32
                | Primitive::Float64 => true,
                _ => false,
            },
            Type::Vector(v) => v.element.to_type().is_signed(),
            Type::Matrix(m) => m.element.to_type().is_signed(),
            _ => false,
        }
    }
    pub fn is_matrix(&self) -> bool {
        match self {
            Type::Matrix(_) => true,
            _ => false,
        }
    }
    pub fn is_array(&self) -> bool {
        match self {
            Type::Array(_) => true,
            _ => false,
        }
    }
    pub fn is_vector(&self) -> bool {
        match self {
            Type::Vector(_) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[repr(C)]
pub struct Node {
    pub type_: CArc<Type>,
    pub next: NodeRef,
    pub prev: NodeRef,
    pub instruction: CArc<Instruction>,
}

pub const INVALID_REF: NodeRef = NodeRef(0);

impl Node {
    pub fn new(instruction: CArc<Instruction>, type_: CArc<Type>) -> Node {
        Node {
            instruction,
            type_,
            next: INVALID_REF,
            prev: INVALID_REF,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize)]
#[repr(C)]
pub enum Func {
    ZeroInitializer,

    Assume,
    Unreachable(CBoxedSlice<u8>),
    Assert(CBoxedSlice<u8>),

    ThreadId,
    BlockId,
    WarpSize,
    WarpLaneId,
    DispatchId,
    DispatchSize,

    // Forward AD
    /// (input, grads, ...) -> ()
    PropagateGrad,
    /// (var, idx) -> dvar/dinput_{idx}
    OutputGrad,

    // Reverse AD
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
    RayTracingInstanceUserId,
    RayTracingSetInstanceTransform,
    RayTracingSetInstanceOpacity,
    RayTracingSetInstanceVisibility,
    RayTracingSetInstanceUserId,
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
    Distance,
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
    /// (buffer/smem, indices...): do not appear in the final IR, but will be lowered to an Atomic* instruction
    AtomicRef,
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
    /// (buffer, index_bytes) -> value
    ByteBufferRead,
    /// (buffer, index_bytes, value) -> void
    ByteBufferWrite,
    /// buffer -> size in bytes
    ByteBufferSize,
    /// (texture, coord) -> value
    Texture2dRead,
    /// (texture, coord, value) -> void
    Texture2dWrite,
    /// (texture) -> uint2
    Texture2dSize,
    /// (texture, coord) -> value
    Texture3dRead,
    /// (texture, coord, value) -> void
    Texture3dWrite,
    /// (texture) -> uint3
    Texture3dSize,
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
    // (bindless_array, index: uint, element_bytes: uint) -> T
    BindlessByteBufferRead,

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
    // (vector, scalar, indices...) -> vector
    InsertElement,
    // (vector, indices...) -> scalar
    ExtractElement,
    //(struct, indices, ...) -> value; the value can be passed to an Update instruction
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

    Callable(CallableModuleRef),

    // ArgT -> ArgT
    CpuCustomOp(CArc<CpuCustomOp>),

    ShaderExecutionReorder, // (uint hint, uint hint_bits): void

    Unknown0,
    Unknown1,
}

#[derive(Clone, Debug, Serialize)]
#[repr(C)]
pub enum Const {
    Zero(CArc<Type>),
    One(CArc<Type>),
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
    Generic(CBoxedSlice<u8>, CArc<Type>),
}

impl std::fmt::Display for Const {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Const::Zero(t) => write!(f, "0_{}", t),
            Const::One(t) => write!(f, "1_{}", t),
            Const::Bool(b) => write!(f, "{}", b),
            Const::Int8(i) => write!(f, "{}", i),
            Const::Uint8(u) => write!(f, "{}", u),
            Const::Int16(i) => write!(f, "{}", i),
            Const::Uint16(u) => write!(f, "{}", u),
            Const::Int32(i) => write!(f, "{}", i),
            Const::Uint32(u) => write!(f, "{}", u),
            Const::Int64(i) => write!(f, "{}", i),
            Const::Uint64(u) => write!(f, "{}", u),
            Const::Float16(fl) => write!(f, "{}", fl),
            Const::Float32(fl) => write!(f, "{}", fl),
            Const::Float64(fl) => write!(f, "{}", fl),
            Const::Generic(data, t) => write!(f, "byte<{}>[{}]", t, data.as_ref().len()),
        }
    }
}

impl Const {
    pub fn get_i32(&self) -> i32 {
        match self {
            Const::Int8(v) => *v as i32,
            Const::Uint8(v) => *v as i32,
            Const::Int16(v) => *v as i32,
            Const::Uint16(v) => *v as i32,
            Const::Int32(v) => *v,
            Const::Uint32(v) => *v as i32,
            Const::Int64(v) => *v as i32,
            Const::Uint64(v) => *v as i32,
            Const::One(t) => {
                assert!(
                    t.is_primitive() && t.is_int(),
                    "cannot convert {:?} to i32",
                    t
                );
                1
            }
            Const::Zero(t) => {
                assert!(
                    t.is_primitive() && t.is_int(),
                    "cannot convert {:?} to i32",
                    t
                );
                0
            }
            Const::Generic(slice, t) => {
                assert!(
                    t.is_primitive() && t.is_int(),
                    "cannot convert {:?} to i32",
                    t
                );
                assert_eq!(slice.len(), 4, "invalid slice length for i32");
                let mut buf = [0u8; 4];
                buf.copy_from_slice(slice);
                i32::from_le_bytes(buf)
            }
            _ => panic!("cannot convert to i32"),
        }
    }
    pub fn type_(&self) -> CArc<Type> {
        match self {
            Const::Zero(ty) => ty.clone(),
            Const::One(ty) => ty.clone(),
            Const::Bool(_) => <bool as TypeOf>::type_(),
            Const::Int8(_) => <i8 as TypeOf>::type_(),
            Const::Uint8(_) => <u8 as TypeOf>::type_(),
            Const::Int16(_) => <i16 as TypeOf>::type_(),
            Const::Uint16(_) => <u16 as TypeOf>::type_(),
            Const::Int32(_) => <i32 as TypeOf>::type_(),
            Const::Uint32(_) => <u32 as TypeOf>::type_(),
            Const::Int64(_) => <i64 as TypeOf>::type_(),
            Const::Uint64(_) => <u64 as TypeOf>::type_(),
            Const::Float16(_) => <f16 as TypeOf>::type_(),
            Const::Float32(_) => <f32 as TypeOf>::type_(),
            Const::Float64(_) => <f64 as TypeOf>::type_(),
            Const::Generic(_, t) => t.clone(),
        }
    }
}

/// cbindgen:derive-eq
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, PartialOrd, Ord)]
#[repr(C)]
pub struct NodeRef(pub usize);

#[repr(C)]
#[derive(Debug)]
pub struct UserData {
    tag: u64,
    data: *const u8,
    eq: extern "C" fn(*const u8, *const u8) -> bool,
}

impl PartialEq for UserData {
    fn eq(&self, other: &Self) -> bool {
        (self.eq)(self.data, other.data)
    }
}

impl Eq for UserData {}

impl Serialize for UserData {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let state = serializer.serialize_struct("UserData", 1)?;
        state.end()
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[repr(C)]
pub struct PhiIncoming {
    pub value: NodeRef,
    pub block: Pooled<BasicBlock>,
}

#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct CpuCustomOp {
    pub data: *mut u8,
    /// func(data, args); func should modify args in place
    pub func: extern "C" fn(*mut u8, *mut u8),
    pub destructor: extern "C" fn(*mut u8),
    pub arg_type: CArc<Type>,
}

impl Serialize for CpuCustomOp {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let state = serializer.serialize_struct("CpuCustomOp", 1)?;
        state.end()
    }
}

impl Debug for CpuCustomOp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        f.debug_struct("CpuCustomOp").finish()
    }
}

#[repr(C)]
#[derive(Clone, Debug, Serialize)]
pub struct SwitchCase {
    pub value: i32,
    pub block: Pooled<BasicBlock>,
}

#[repr(C)]
#[derive(Clone, Debug, Serialize)]
pub enum Instruction {
    Buffer,
    Bindless,
    Texture2D,
    Texture3D,
    Accel,
    // Shared memory
    Shared,
    // Uniform kernel arguments
    Uniform,
    Local {
        init: NodeRef,
    },
    // Callable arguments
    Argument {
        by_value: bool,
    },
    UserData(CArc<UserData>),
    Invalid,
    Const(Const),

    // a variable that can be assigned to
    // similar to LLVM's alloca
    Update {
        var: NodeRef,
        value: NodeRef,
    },

    Call(Func, CBoxedSlice<NodeRef>),

    Phi(CBoxedSlice<PhiIncoming>),
    /* represent a loop in the form of
    loop {
        body();
        if (!cond) {
            break;
        }
    }
    */
    Return(NodeRef),
    Loop {
        body: Pooled<BasicBlock>,
        cond: NodeRef,
    },
    /* represent a loop in the form of
    loop {
        prepare;// typically the computation of the loop condition
        if cond {
            body;
            update; // continue goes here
        }
    }
    for (;; update) {
        prepare;
        if (!cond) {
            break;
        }
        body;
    }
    */
    GenericLoop {
        prepare: Pooled<BasicBlock>,
        cond: NodeRef,
        body: Pooled<BasicBlock>,
        update: Pooled<BasicBlock>,
    },
    Break,
    Continue,
    If {
        cond: NodeRef,
        true_branch: Pooled<BasicBlock>,
        false_branch: Pooled<BasicBlock>,
    },
    Switch {
        value: NodeRef,
        default: Pooled<BasicBlock>,
        cases: CBoxedSlice<SwitchCase>,
    },
    AdScope {
        body: Pooled<BasicBlock>,
        forward: bool,
        n_forward_grads: usize,
    },
    RayQuery {
        ray_query: NodeRef,
        on_triangle_hit: Pooled<BasicBlock>,
        on_procedural_hit: Pooled<BasicBlock>,
    },
    Print {
        fmt: CBoxedSlice<u8>,
        args: CBoxedSlice<NodeRef>,
    },
    AdDetach(Pooled<BasicBlock>),
    Comment(CBoxedSlice<u8>),
}

extern "C" fn eq_impl<T: UserNodeData>(a: *const u8, b: *const u8) -> bool {
    let a = unsafe { &*(a as *const T) };
    let b = unsafe { &*(b as *const T) };
    a.equal(b)
}

fn type_id_u64<T: UserNodeData>() -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    TypeId::of::<T>().hash(&mut hasher);
    hasher.finish()
}

pub fn new_user_node<T: UserNodeData>(pools: &CArc<ModulePools>, data: T) -> NodeRef {
    new_node(
        pools,
        Node::new(
            CArc::new(Instruction::UserData(CArc::new(UserData {
                tag: type_id_u64::<T>(),
                data: Box::into_raw(Box::new(data)) as *mut u8,
                eq: eq_impl::<T>,
            }))),
            Type::userdata(),
        ),
    )
}

impl Instruction {
    pub fn is_call(&self) -> bool {
        match self {
            Instruction::Call(_, _) => true,
            _ => false,
        }
    }
    pub fn is_const(&self) -> bool {
        match self {
            Instruction::Const(_) => true,
            _ => false,
        }
    }
    pub fn is_phi(&self) -> bool {
        match self {
            Instruction::Phi(_) => true,
            _ => false,
        }
    }
    pub fn has_value(&self) -> bool {
        self.is_call() || self.is_const() || self.is_phi()
    }
}

pub const INVALID_INST: Instruction = Instruction::Invalid;

pub fn new_node(pools: &CArc<ModulePools>, node: Node) -> NodeRef {
    let ptr = pools.node_pool.alloc(node);
    NodeRef(ptr.ptr as usize)
}

pub trait UserNodeData: Any + Debug {
    fn equal(&self, other: &dyn UserNodeData) -> bool;
    fn as_any(&self) -> &dyn Any;
}
macro_rules! impl_userdata {
    ($t:ty) => {
        impl UserNodeData for $t {
            fn equal(&self, other: &dyn UserNodeData) -> bool {
                let other = other.as_any().downcast_ref::<$t>().unwrap();
                self == other
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
    };
}
impl_userdata!(usize);
impl_userdata!(u32);
impl_userdata!(u64);
impl_userdata!(i32);
impl_userdata!(i64);
impl_userdata!(bool);

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BasicBlock {
    pub(crate) first: NodeRef,
    pub(crate) last: NodeRef,
}
impl BasicBlock {
    pub fn first(&self) -> NodeRef {
        self.first
    }
    pub fn last(&self) -> NodeRef {
        self.last
    }
}

#[derive(Serialize)]
struct NodeRefAndNode<'a> {
    id: NodeRef,
    data: &'a Node,
}

impl Serialize for BasicBlock {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("BasicBlock", 1)?;
        let nodes = self.nodes();
        let nodes = nodes
            .iter()
            .map(|n| NodeRefAndNode {
                id: *n,
                data: n.get(),
            })
            .collect::<Vec<_>>();
        state.serialize_field("nodes", &nodes)?;
        state.end()
    }
}

pub struct BasicBlockIter<'a> {
    cur: NodeRef,
    last: NodeRef,
    _block: &'a BasicBlock,
}

impl Iterator for BasicBlockIter<'_> {
    type Item = NodeRef;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.last {
            None
        } else {
            let ret = self.cur;
            self.cur = self.cur.get().next;
            Some(ret)
        }
    }
}

impl BasicBlock {
    pub fn iter(&self) -> BasicBlockIter {
        BasicBlockIter {
            cur: self.first.get().next,
            last: self.last,
            _block: self,
        }
    }
    pub fn phis(&self) -> Vec<NodeRef> {
        self.iter().filter(|n| n.is_phi()).collect()
    }
    pub fn nodes(&self) -> Vec<NodeRef> {
        self.iter().collect()
    }
    pub fn into_vec(self) -> Vec<NodeRef> {
        let mut vec = Vec::new();
        let mut cur = self.first.get().next;
        while cur != self.last {
            vec.push(cur.clone());
            let next = cur.get().next;
            cur.update(|node| {
                node.prev = INVALID_REF;
                node.next = INVALID_REF;
            });
            cur = next;
        }
        self.first.update(|node| node.next = self.last);
        self.last.update(|node| node.prev = self.first);
        for i in &vec {
            debug_assert!(!i.is_linked());
        }
        vec
    }
    pub fn new(pools: &CArc<ModulePools>) -> Self {
        let first = new_node(
            pools,
            Node::new(CArc::new(Instruction::Invalid), Type::void()),
        );
        let last = new_node(
            pools,
            Node::new(CArc::new(Instruction::Invalid), Type::void()),
        );
        first.update(|node| node.next = last);
        last.update(|node| node.prev = first);
        Self { first, last }
    }
    pub fn push(&self, node: NodeRef) {
        // node.insert_before(self.last);
        self.last.insert_before_self(node);
    }

    pub fn is_empty(&self) -> bool {
        !self.first.valid()
    }
    pub fn len(&self) -> usize {
        let mut len = 0;
        let mut cur = self.first.get().next;
        while cur != self.last {
            len += 1;
            cur = cur.get().next;
        }
        len
    }
    pub fn merge(&self, other: Pooled<BasicBlock>) {
        let nodes = other.into_vec();
        for node in nodes {
            self.push(node);
        }
    }
    /// split the block into two at @at
    /// @at is not transfered into the other block
    pub fn split(&self, at: NodeRef, pools: &CArc<ModulePools>) -> Pooled<BasicBlock> {
        #[cfg(debug_assertions)]
        {
            let nodes = self.nodes();
            assert!(nodes.contains(&at));
        }
        let new_bb_start = at.get().next;
        let second_last = self.last.get().prev;
        let new_bb = pools.bb_pool.alloc(BasicBlock::new(pools));
        if new_bb_start != self.last {
            new_bb.first.update(|first| {
                first.next = new_bb_start;
            });
            new_bb.last.update(|last| {
                last.prev = second_last;
            });
            second_last.update(|node| {
                node.next = new_bb.last;
            });
            new_bb_start.update(|node| {
                node.prev = new_bb.first;
            });
        }

        at.update(|at| {
            at.next = self.last;
        });
        self.last.update(|last| {
            last.prev = at;
        });
        new_bb
    }
}

impl NodeRef {
    pub fn get_i32(&self) -> i32 {
        match self.get().instruction.as_ref() {
            Instruction::Const(c) => c.get_i32(),
            _ => panic!("not i32 node; found: {:?}", self.get().instruction),
        }
    }
    pub fn is_unreachable(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::Call(Func::Unreachable(_), _) => true,
            _ => false,
        }
    }
    pub fn get_user_data(&self) -> &UserData {
        match self.get().instruction.as_ref() {
            Instruction::UserData(data) => data,
            _ => panic!("not user data node; found: {:?}", self.get().instruction),
        }
    }
    pub fn is_user_data(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::UserData(_) => true,
            _ => false,
        }
    }
    pub fn unwrap_user_data<T: UserNodeData>(&self) -> &T {
        let data = self.get_user_data();
        assert_eq!(data.tag, type_id_u64::<T>());
        let data = data.data as *const T;
        unsafe { &*data }
    }
    pub fn is_local(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::Local { .. } => true,
            _ => false,
        }
    }
    pub fn is_const(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::Const(_) => true,
            _ => false,
        }
    }
    pub fn is_uniform(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::Uniform => true,
            _ => false,
        }
    }
    pub fn is_atomic_ref(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::Call(Func::AtomicRef, _) => true,
            _ => false,
        }
    }
    pub fn is_argument(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::Argument { .. } => true,
            _ => false,
        }
    }
    pub fn is_value_argument(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::Argument { by_value } => *by_value,
            _ => false,
        }
    }
    pub fn is_refernece_argument(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::Argument { by_value } => !*by_value,
            _ => false,
        }
    }
    pub fn is_gep(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::Call(Func::GetElementPtr, _) => true,
            _ => false,
        }
    }
    pub fn is_phi(&self) -> bool {
        self.get().instruction.is_phi()
    }
    pub fn get<'a>(&'a self) -> &'a Node {
        assert!(self.valid());
        unsafe { &*(self.0 as *const Node) }
    }
    pub fn get_mut(&self) -> &mut Node {
        assert!(self.valid());
        unsafe { &mut *(self.0 as *mut Node) }
    }
    pub fn valid(&self) -> bool {
        self.0 != INVALID_REF.0
    }
    pub fn set(&self, node: Node) {
        *self.get_mut() = node;
    }
    pub fn replace_with(&self, node: &Node) {
        let instr = node.instruction.clone();
        let type_ = node.type_.clone();
        self.get_mut().instruction = instr;
        self.get_mut().type_ = type_;
    }
    pub fn update<T>(&self, f: impl FnOnce(&mut Node) -> T) -> T {
        f(self.get_mut())
    }
    pub fn access_chain(&self) -> Option<(NodeRef, Vec<(CArc<Type>, usize)>)> {
        match self.get().instruction.as_ref() {
            Instruction::Call(f, args) => {
                if *f == Func::GetElementPtr {
                    let var = args[0];
                    let idx = args[1].get_i32() as usize;
                    if let Some((parent, mut indices)) = var.access_chain() {
                        indices.push((self.type_().clone(), idx));
                        return Some((parent, indices));
                    }
                    Some((var, vec![(self.type_().clone(), idx)]))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
    pub fn type_(&self) -> &CArc<Type> {
        &self.get().type_
    }
    pub fn is_linked(&self) -> bool {
        assert!(self.valid());
        self.get().prev.valid() || self.get().next.valid()
    }
    pub fn remove(&self) {
        assert!(self.valid());
        let prev = self.get().prev;
        let next = self.get().next;
        prev.update(|node| node.next = next);
        next.update(|node| node.prev = prev);
        self.update(|node| {
            node.prev = INVALID_REF;
            node.next = INVALID_REF;
        });
    }
    pub fn insert_after_self(&self, node_ref: NodeRef) {
        assert!(self.valid());
        assert!(!node_ref.is_linked());
        let next = self.get().next;
        self.update(|node| node.next = node_ref);
        next.update(|node| node.prev = node_ref);
        node_ref.update(|node| {
            node.prev = *self;
            node.next = next;
        });
    }
    pub fn insert_before_self(&self, node_ref: NodeRef) {
        assert!(self.valid());
        assert!(!node_ref.is_linked());
        let prev = self.get().prev;
        self.update(|node| node.prev = node_ref);
        prev.update(|node| node.next = node_ref);
        node_ref.update(|node| {
            node.prev = prev;
            node.next = *self;
        });
    }
    pub fn is_lvalue(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::Local { .. } => true,
            Instruction::Argument { by_value } => !by_value,
            Instruction::Shared => true,
            Instruction::Call(f, _) => *f == Func::GetElementPtr,
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub enum ModuleKind {
    Block,
    Function,
    Kernel,
}
bitflags! {
    #[repr(C)]
    #[derive(Copy, Clone, Debug, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct ModuleFlags : u32 {
        const NONE = 0;
        const REQUIRES_REV_AD_TRANSFORM = 1;
        const REQUIRES_FWD_AD_TRANSFORM = 2;
    }
}
#[repr(C)]
#[derive(Debug, Serialize)]
pub struct Module {
    pub kind: ModuleKind,
    pub entry: Pooled<BasicBlock>,
    pub flags: ModuleFlags,
    #[serde(skip)]
    pub pools: CArc<ModulePools>,
}

#[repr(C)]
#[derive(Debug, Serialize, Clone)]
pub struct CallableModuleRef(pub CArc<CallableModule>);

#[repr(C)]
#[derive(Debug, Serialize)]
pub struct CallableModule {
    pub module: Module,
    pub ret_type: CArc<Type>,
    pub args: CBoxedSlice<NodeRef>,
    pub captures: CBoxedSlice<Capture>,
    pub cpu_custom_ops: CBoxedSlice<CArc<CpuCustomOp>>,
    #[serde(skip)]
    pub pools: CArc<ModulePools>,
}

impl PartialEq for CallableModuleRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}

impl Eq for CallableModuleRef {}

impl Hash for CallableModuleRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

// buffer binding
#[repr(C)]
#[derive(Debug, Serialize, Deserialize, Copy, Clone, Hash, PartialEq, Eq)]
pub struct BufferBinding {
    pub handle: u64,
    pub offset: u64,
    pub size: usize,
}

// texture binding
#[repr(C)]
#[derive(Debug, Serialize, Deserialize, Copy, Clone, Hash, PartialEq, Eq)]
pub struct TextureBinding {
    pub handle: u64,
    pub level: u32,
}

// bindless array binding
#[repr(C)]
#[derive(Debug, Serialize, Deserialize, Copy, Clone, Hash, PartialEq, Eq)]
pub struct BindlessArrayBinding {
    pub handle: u64,
}

// accel binding
#[repr(C)]
#[derive(Debug, Serialize, Deserialize, Copy, Clone, Hash, PartialEq, Eq)]
pub struct AccelBinding {
    pub handle: u64,
}

#[repr(C)]
#[derive(Debug, Serialize, Copy, Clone, Deserialize, Hash, PartialEq, Eq)]
pub enum Binding {
    Buffer(BufferBinding),
    Texture(TextureBinding),
    BindlessArray(BindlessArrayBinding),
    Accel(AccelBinding),
}

#[derive(Debug, Serialize, Copy, Clone, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct Capture {
    pub node: NodeRef,
    pub binding: Binding,
}

#[derive(Debug)]
pub struct ModulePools {
    pub node_pool: Pool<Node>,
    pub bb_pool: Pool<BasicBlock>,
}

impl ModulePools {
    pub fn new() -> Self {
        Self {
            node_pool: Pool::new(),
            bb_pool: Pool::new(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Serialize)]
pub struct KernelModule {
    pub module: Module,
    pub captures: CBoxedSlice<Capture>,
    pub args: CBoxedSlice<NodeRef>,
    pub shared: CBoxedSlice<NodeRef>,
    pub cpu_custom_ops: CBoxedSlice<CArc<CpuCustomOp>>,
    pub block_size: [u32; 3],
    #[serde(skip)]
    pub pools: CArc<ModulePools>,
}

unsafe impl Send for KernelModule {}

#[repr(C)]
#[derive(Debug, Serialize)]
pub struct BlockModule {
    pub module: Module,
}

unsafe impl Send for BlockModule {}

struct NodeCollector {
    nodes: Vec<NodeRef>,
    unique: HashSet<NodeRef>,
}

impl NodeCollector {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            unique: HashSet::new(),
        }
    }
    fn visit_block(&mut self, block: Pooled<BasicBlock>) {
        for node in block.iter() {
            self.visit_node(node);
        }
    }
    fn visit_node(&mut self, node_ref: NodeRef) {
        if self.unique.contains(&node_ref) {
            return;
        }
        self.unique.insert(node_ref);
        self.nodes.push(node_ref);
        let inst = node_ref.get().instruction.as_ref();
        match inst {
            Instruction::AdScope { body, .. } => {
                self.visit_block(*body);
            }
            Instruction::If {
                cond: _,
                true_branch,
                false_branch,
            } => {
                self.visit_block(*true_branch);
                self.visit_block(*false_branch);
            }
            Instruction::Loop { body, cond: _ } => {
                self.visit_block(*body);
            }
            Instruction::GenericLoop {
                prepare,
                cond: _,
                body,
                update,
            } => {
                self.visit_block(*prepare);
                self.visit_block(*body);
                self.visit_block(*update);
            }
            Instruction::Switch {
                value: _,
                default,
                cases,
            } => {
                self.visit_block(*default);
                for SwitchCase { value: _, block } in cases.as_ref().iter() {
                    self.visit_block(*block);
                }
            }
            _ => {}
        }
    }
}

impl Module {
    pub fn collect_nodes(&self) -> Vec<NodeRef> {
        let mut collector = NodeCollector::new();
        collector.visit_block(self.entry);
        collector.nodes
    }
}

struct ModuleDuplicatorCtx {
    nodes: HashMap<NodeRef, NodeRef>,
    blocks: HashMap<*const BasicBlock, Pooled<BasicBlock>>,
}

struct ModuleDuplicator {
    callables: HashMap<*const CallableModule, CArc<CallableModule>>,
    current: Option<ModuleDuplicatorCtx>,
}

impl ModuleDuplicator {
    fn new() -> Self {
        Self {
            callables: HashMap::new(),
            current: None,
        }
    }

    fn with_context<T, F: FnOnce(&mut Self) -> T>(&mut self, f: F) -> T {
        let ctx = ModuleDuplicatorCtx {
            nodes: HashMap::new(),
            blocks: HashMap::new(),
        };
        let old_ctx = self.current.replace(ctx);
        let ret = f(self);
        self.current = old_ctx;
        ret
    }

    fn duplicate_callable(&mut self, callable: &CArc<CallableModule>) -> CArc<CallableModule> {
        if let Some(copy) = self.callables.get(&callable.as_ptr()) {
            return copy.clone();
        }
        let dup_callable = self.with_context(|this| {
            let dup_args = this.duplicate_args(&callable.pools, &callable.args);
            let dup_captures = this.duplicate_captures(&callable.pools, &callable.captures);
            let dup_module = this.duplicate_module(&callable.module);
            CallableModule {
                module: dup_module,
                ret_type: callable.ret_type.clone(),
                args: dup_args,
                captures: dup_captures,
                cpu_custom_ops: callable.cpu_custom_ops.clone(),
                pools: callable.pools.clone(),
            }
        });
        let dup_callable = CArc::new(dup_callable);
        self.callables
            .insert(callable.as_ptr(), dup_callable.clone());
        dup_callable
    }

    fn duplicate_arg(&mut self, pools: &CArc<ModulePools>, node_ref: NodeRef) -> NodeRef {
        let node = node_ref.get();
        let instr = &node.instruction;
        let dup_instr = match instr.as_ref() {
            Instruction::Buffer => instr.clone(),
            Instruction::Bindless => instr.clone(),
            Instruction::Texture2D => instr.clone(),
            Instruction::Texture3D => instr.clone(),
            Instruction::Accel => instr.clone(),
            Instruction::Shared => instr.clone(),
            Instruction::Uniform => instr.clone(),
            Instruction::Argument { .. } => CArc::new(instr.as_ref().clone()),
            _ => unreachable!("invalid argument type"),
        };
        let dup_node = Node::new(dup_instr, node.type_.clone());
        let dup_node_ref = new_node(pools, dup_node);
        // add to node map
        self.current
            .as_mut()
            .unwrap()
            .nodes
            .insert(node_ref, dup_node_ref);
        dup_node_ref
    }

    fn duplicate_args(
        &mut self,
        pools: &CArc<ModulePools>,
        args: &CBoxedSlice<NodeRef>,
    ) -> CBoxedSlice<NodeRef> {
        let dup_args: Vec<_> = args
            .iter()
            .map(|arg| self.duplicate_arg(pools, arg.clone()))
            .collect();
        CBoxedSlice::new(dup_args)
    }

    fn duplicate_captures(
        &mut self,
        pools: &CArc<ModulePools>,
        captures: &CBoxedSlice<Capture>,
    ) -> CBoxedSlice<Capture> {
        let dup_captures: Vec<_> = captures
            .iter()
            .map(|capture| Capture {
                node: self.duplicate_arg(pools, capture.node.clone()),
                binding: capture.binding.clone(),
            })
            .collect();
        CBoxedSlice::new(dup_captures)
    }

    fn duplicate_shared(
        &mut self,
        pools: &CArc<ModulePools>,
        shared: &CBoxedSlice<NodeRef>,
    ) -> CBoxedSlice<NodeRef> {
        let dup_shared: Vec<_> = shared
            .iter()
            .map(|node| self.duplicate_arg(pools, node.clone()))
            .collect();
        CBoxedSlice::new(dup_shared)
    }

    fn duplicate_node(&mut self, builder: &mut IrBuilder, node_ref: NodeRef) -> NodeRef {
        if !node_ref.valid() {
            return INVALID_REF;
        }
        let node = node_ref.get();
        assert!(
            !self.current.as_ref().unwrap().nodes.contains_key(&node_ref),
            "Node {:?} has already been duplicated",
            node
        );
        let dup_node = match node.instruction.as_ref() {
            Instruction::Buffer => unreachable!("Buffer should be handled by duplicate_args"),
            Instruction::Bindless => unreachable!("Bindless should be handled by duplicate_args"),
            Instruction::Texture2D => unreachable!("Texture2D should be handled by duplicate_args"),
            Instruction::Texture3D => unreachable!("Texture3D should be handled by duplicate_args"),
            Instruction::Accel => unreachable!("Accel should be handled by duplicate_args"),
            Instruction::Shared => unreachable!("Shared should be handled by duplicate_shared"),
            Instruction::Uniform => unreachable!("Uniform should be handled by duplicate_args"),
            Instruction::Argument { .. } => {
                unreachable!("Argument should be handled by duplicate_args")
            }
            Instruction::Local { init } => {
                let dup_init = self.find_duplicated_node(*init);
                builder.local(dup_init)
            }
            Instruction::UserData(data) => builder.userdata(data.clone()),
            Instruction::Invalid => {
                unreachable!("Invalid node should not appear in non-sentinel nodes")
            }
            Instruction::Const(const_) => builder.const_(const_.clone()),
            Instruction::Update { var, value } => {
                let dup_var = self.find_duplicated_node(*var);
                let dup_value = self.find_duplicated_node(*value);
                builder.update(dup_var, dup_value)
            }
            Instruction::Call(func, args) => {
                let dup_func = match func {
                    Func::Callable(callable) => {
                        let dup_callable = self.duplicate_callable(&callable.0);
                        Func::Callable(CallableModuleRef(dup_callable))
                    }
                    _ => func.clone(),
                };
                let dup_args: Vec<_> = args
                    .iter()
                    .map(|arg| self.find_duplicated_node(*arg))
                    .collect();
                builder.call(dup_func, dup_args.as_slice(), node.type_.clone())
            }
            Instruction::Phi(incomings) => {
                let dup_incomings: Vec<_> = incomings
                    .iter()
                    .map(|incoming| {
                        let dup_block = self.find_duplicated_block(&incoming.block);
                        let dup_value = self.find_duplicated_node(incoming.value);
                        PhiIncoming {
                            value: dup_value,
                            block: dup_block,
                        }
                    })
                    .collect();
                builder.phi(dup_incomings.as_slice(), node.type_.clone())
            }
            Instruction::Return(value) => {
                let dup_value = self.find_duplicated_node(*value);
                builder.return_(dup_value)
            }
            Instruction::Loop { body, cond } => {
                let dup_body = self.duplicate_block(&builder.pools, body);
                let dup_cond = self.find_duplicated_node(*cond);
                builder.loop_(dup_body, dup_cond)
            }
            Instruction::GenericLoop {
                prepare,
                cond,
                body,
                update,
            } => {
                let dup_prepare = self.duplicate_block(&builder.pools, prepare);
                let dup_body = self.duplicate_block(&builder.pools, body);
                let dup_update = self.duplicate_block(&builder.pools, update);
                let dup_cond = self.find_duplicated_node(*cond);
                builder.generic_loop(dup_prepare, dup_cond, dup_body, dup_update)
            }
            Instruction::Break => builder.break_(),
            Instruction::Continue => builder.continue_(),
            Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                let dup_cond = self.find_duplicated_node(*cond);
                let dup_true_branch = self.duplicate_block(&builder.pools, true_branch);
                let dup_false_branch = self.duplicate_block(&builder.pools, false_branch);
                builder.if_(dup_cond, dup_true_branch, dup_false_branch)
            }
            Instruction::Switch {
                value,
                cases,
                default,
            } => {
                let dup_value = self.find_duplicated_node(*value);
                let dup_cases: Vec<_> = cases
                    .iter()
                    .map(|case| {
                        let dup_block = self.duplicate_block(&builder.pools, &case.block);
                        SwitchCase {
                            value: case.value,
                            block: dup_block,
                        }
                    })
                    .collect();
                let dup_default = self.duplicate_block(&builder.pools, default);
                builder.switch(dup_value, dup_cases.as_slice(), dup_default)
            }
            Instruction::AdScope {
                body,
                forward,
                n_forward_grads,
            } => {
                let dup_body = self.duplicate_block(&builder.pools, body);
                if *forward {
                    builder.fwd_ad_scope(dup_body, *n_forward_grads)
                } else {
                    builder.ad_scope(dup_body)
                }
            }
            Instruction::RayQuery {
                ray_query,
                on_triangle_hit,
                on_procedural_hit,
            } => {
                let dup_ray_query = self.find_duplicated_node(*ray_query);
                let dup_on_triangle_hit = self.duplicate_block(&builder.pools, on_triangle_hit);
                let dup_on_procedural_hit = self.duplicate_block(&builder.pools, on_procedural_hit);
                builder.ray_query(
                    dup_ray_query,
                    dup_on_triangle_hit,
                    dup_on_procedural_hit,
                    node.type_.clone(),
                )
            }
            Instruction::AdDetach(body) => {
                let dup_body = self.duplicate_block(&builder.pools, body);
                builder.ad_detach(dup_body)
            }
            Instruction::Comment(msg) => builder.comment(msg.clone()),
            Instruction::Print { fmt, args } => {
                let args = args
                    .iter()
                    .map(|x| self.find_duplicated_node(*x))
                    .collect::<Vec<_>>();
                builder.print(fmt.clone(), &args)
            }
        };
        // insert the duplicated node into the map
        self.current
            .as_mut()
            .unwrap()
            .nodes
            .insert(node_ref, dup_node);
        dup_node
    }

    fn find_duplicated_block(&self, bb: &Pooled<BasicBlock>) -> Pooled<BasicBlock> {
        let ctx = self.current.as_ref().unwrap();
        ctx.blocks.get(&bb.as_ptr()).unwrap().clone()
    }

    fn find_duplicated_node(&self, node: NodeRef) -> NodeRef {
        if !node.valid() {
            return INVALID_REF;
        }
        let ctx = self.current.as_ref().unwrap();
        ctx.nodes.get(&node).unwrap().clone()
    }

    fn duplicate_block(
        &mut self,
        pools: &CArc<ModulePools>,
        bb: &Pooled<BasicBlock>,
    ) -> Pooled<BasicBlock> {
        assert!(
            !self
                .current
                .as_ref()
                .unwrap()
                .blocks
                .contains_key(&bb.as_ptr()),
            "Basic block {:?} has already been duplicated",
            bb
        );
        let mut builder = IrBuilder::new(pools.clone());
        bb.iter().for_each(|node| {
            self.duplicate_node(&mut builder, node);
        });
        let dup_bb = builder.finish();
        // insert the duplicated block into the map
        self.current
            .as_mut()
            .unwrap()
            .blocks
            .insert(bb.as_ptr(), dup_bb.clone());
        dup_bb
    }

    fn duplicate_kernel(&mut self, kernel: &KernelModule) -> KernelModule {
        self.with_context(|this| {
            let dup_args = this.duplicate_args(&kernel.pools, &kernel.args);
            let dup_captures = this.duplicate_captures(&kernel.pools, &kernel.captures);
            let dup_shared = this.duplicate_shared(&kernel.pools, &kernel.shared);
            let dup_module = this.duplicate_module(&kernel.module);
            KernelModule {
                module: dup_module,
                captures: dup_captures,
                args: dup_args,
                shared: dup_shared,
                cpu_custom_ops: kernel.cpu_custom_ops.clone(),
                block_size: kernel.block_size,
                pools: kernel.pools.clone(),
            }
        })
    }

    fn duplicate_module(&mut self, module: &Module) -> Module {
        let dup_entry = self.duplicate_block(&module.pools, &module.entry);
        Module {
            kind: module.kind,
            entry: dup_entry,
            pools: module.pools.clone(),
            flags: module.flags,
        }
    }
}

pub fn duplicate_kernel(kernel: &KernelModule) -> KernelModule {
    let mut dup = ModuleDuplicator::new();
    dup.duplicate_kernel(kernel)
}

pub fn duplicate_callable(callable: &CArc<CallableModule>) -> CArc<CallableModule> {
    let mut dup = ModuleDuplicator::new();
    dup.duplicate_callable(callable)
}

#[repr(C)]
pub struct IrBuilder {
    bb: Pooled<BasicBlock>,
    pub(crate) pools: CArc<ModulePools>,
    pub(crate) insert_point: NodeRef,
}

impl IrBuilder {
    pub fn pools(&self) -> &CArc<ModulePools> {
        &self.pools
    }
    pub fn new_without_bb(pools: CArc<ModulePools>) -> Self {
        Self {
            bb: Pooled::null(),
            insert_point: INVALID_REF,
            pools,
        }
    }
    pub fn new(pools: CArc<ModulePools>) -> Self {
        let bb = pools.bb_pool.alloc(BasicBlock::new(&pools));
        let insert_point = bb.first;
        Self {
            bb,
            insert_point,
            pools,
        }
    }
    pub fn bb(&self) -> Pooled<BasicBlock> {
        self.bb
    }
    pub fn set_insert_point(&mut self, node: NodeRef) {
        assert!(node.valid());
        self.insert_point = node;
    }
    pub fn set_insert_point_to_end(&mut self) {
        assert!(!self.bb.as_ptr().is_null());
        while self.insert_point.get().next != self.bb.last {
            self.insert_point = self.insert_point.get().next;
        }
    }
    pub fn get_insert_point(&self) -> NodeRef {
        self.insert_point
    }
    pub fn append(&mut self, node: NodeRef) {
        self.insert_point.insert_after_self(node);
        self.insert_point = node;
    }
    pub fn append_block(&mut self, block: Pooled<BasicBlock>) {
        self.bb.merge(block);
        self.insert_point = self.bb.last.get().prev;
    }
    pub fn comment(&mut self, msg: CBoxedSlice<u8>) -> NodeRef {
        let new_node = new_node(
            &self.pools,
            Node::new(CArc::new(Instruction::Comment(msg)), Type::void()),
        );
        self.append(new_node);
        new_node
    }
    pub fn print(&mut self, fmt: CBoxedSlice<u8>, args: &[NodeRef]) -> NodeRef {
        let new_node = new_node(
            &self.pools,
            Node::new(
                CArc::new(Instruction::Print {
                    fmt,
                    args: CBoxedSlice::new(args.to_vec()),
                }),
                Type::void(),
            ),
        );
        self.append(new_node);
        new_node
    }
    pub fn userdata(&mut self, data: CArc<UserData>) -> NodeRef {
        let new_node = new_node(
            &self.pools,
            Node::new(CArc::new(Instruction::UserData(data)), Type::void()),
        );
        self.append(new_node);
        new_node
    }
    pub fn break_(&mut self) -> NodeRef {
        let new_node = new_node(
            &self.pools,
            Node::new(CArc::new(Instruction::Break), Type::void()),
        );
        self.append(new_node);
        new_node
    }
    pub fn continue_(&mut self) -> NodeRef {
        let new_node = new_node(
            &self.pools,
            Node::new(CArc::new(Instruction::Continue), Type::void()),
        );
        self.append(new_node);
        new_node
    }
    pub fn return_(&mut self, node: NodeRef) -> NodeRef {
        let new_node = new_node(
            &self.pools,
            Node::new(CArc::new(Instruction::Return(node)), Type::void()),
        );
        self.append(new_node);
        new_node
    }
    pub fn zero_initializer(&mut self, ty: CArc<Type>) -> NodeRef {
        self.call(Func::ZeroInitializer, &[], ty)
    }
    pub fn requires_gradient(&mut self, node: NodeRef) -> NodeRef {
        self.call(Func::RequiresGradient, &[node], Type::void())
    }
    pub fn gradient(&mut self, node: NodeRef) -> NodeRef {
        self.call(Func::Gradient, &[node], node.type_().clone())
    }
    pub fn clone_node(&mut self, node: NodeRef) -> NodeRef {
        let node = node.get();
        let new_node = new_node(
            &self.pools,
            Node::new(
                CArc::new(node.instruction.as_ref().clone()),
                node.type_.clone(),
            ),
        );
        self.append(new_node);
        new_node
    }
    pub fn load(&mut self, var: NodeRef) -> NodeRef {
        assert!(var.is_lvalue(), "{:?}", var.get().instruction.as_ref());
        let node = Node::new(
            CArc::new(Instruction::Call(Func::Load, CBoxedSlice::new(vec![var]))),
            var.type_().clone(),
        );
        let node = new_node(&self.pools, node);
        self.append(node.clone());
        node
    }
    pub fn const_(&mut self, const_: Const) -> NodeRef {
        let t = const_.type_();
        let node = Node::new(CArc::new(Instruction::Const(const_)), t);
        let node = new_node(&self.pools, node);
        self.append(node.clone());
        node
    }
    pub fn local_zero_init(&mut self, ty: CArc<Type>) -> NodeRef {
        let node = self.zero_initializer(ty);
        let local = self.local(node);
        local
    }
    pub fn local(&mut self, init: NodeRef) -> NodeRef {
        let t = init.type_();
        let node = Node::new(CArc::new(Instruction::Local { init }), t.clone());
        let node = new_node(&self.pools, node);
        self.append(node.clone());
        node
    }
    pub fn extract(&mut self, node: NodeRef, index: usize, ret_type: CArc<Type>) -> NodeRef {
        match node.type_().as_ref() {
            Type::Vector(vt) => assert_eq!(vt.element.to_type(), ret_type),
            Type::Matrix(mt) => assert_eq!(mt.column(), ret_type),
            Type::Struct(st) => assert_eq!(st.fields[index], ret_type),
            Type::Array(at) => assert_eq!(at.element, ret_type),
            Type::Opaque(_) => {}
            _ => panic!("Invalid type for extract"),
        }
        let c = self.const_(Const::Int32(index as i32));
        self.call(Func::ExtractElement, &[node, c], ret_type)
    }
    pub fn extract_dynamic(
        &mut self,
        node: NodeRef,
        index: NodeRef,
        ret_type: CArc<Type>,
    ) -> NodeRef {
        match node.type_().as_ref() {
            Type::Vector(vt) => assert_eq!(vt.element.to_type(), ret_type),
            Type::Matrix(mt) => assert_eq!(mt.column(), ret_type),
            Type::Array(at) => assert_eq!(at.element, ret_type),
            Type::Opaque(_) => {}
            _ => panic!("Invalid type for extract with dynamic index"),
        }
        self.call(Func::ExtractElement, &[node, index], ret_type)
    }
    pub fn call(&mut self, func: Func, args: &[NodeRef], ret_type: CArc<Type>) -> NodeRef {
        let node = Node::new(
            CArc::new(Instruction::Call(func, CBoxedSlice::new(args.to_vec()))),
            ret_type,
        );
        let node = new_node(&self.pools, node);
        self.append(node.clone());
        node
    }
    pub fn call_no_append(&mut self, func: Func, args: &[NodeRef], ret_type: CArc<Type>) -> NodeRef {
        let node = Node::new(
            CArc::new(Instruction::Call(func, CBoxedSlice::new(args.to_vec()))),
            ret_type,
        );
        let node = new_node(&self.pools, node);
        node
    }
    pub fn cast(&mut self, node: NodeRef, t: CArc<Type>) -> NodeRef {
        self.call(Func::Cast, &[node], t)
    }
    pub fn bitcast(&mut self, node: NodeRef, t: CArc<Type>) -> NodeRef {
        self.call(Func::Bitcast, &[node], t)
    }
    pub fn gep(&mut self, this: NodeRef, indices: &[NodeRef], t: CArc<Type>) -> NodeRef {
        assert!(this.is_lvalue());
        let mut e = Some(this.type_().clone());
        indices.iter().for_each(|i| {
            if let Some(ee) = &e {
                e = match ee.as_ref() {
                    Type::Vector(vt) => Some(vt.element.to_type()),
                    Type::Matrix(mt) => Some(mt.column()),
                    Type::Array(at) => Some(at.element.clone()),
                    Type::Struct(st) => Some(st.fields[i.get_i32() as usize].clone()),
                    Type::Opaque(_) => None,
                    _ => panic!(
                        "Invalid type {:?} for GEP with dynamic indices",
                        this.type_()
                    ),
                };
            };
        });
        if let Some(e) = e {
            assert_eq!(e, t);
        }
        self.call(Func::GetElementPtr, &[&[this], indices].concat(), t)
    }
    // append the indices to this if it's a gep; otherwise behave like gep
    pub fn gep_chained(&mut self, this: NodeRef, indices: &[NodeRef], t: CArc<Type>) -> NodeRef {
        match this.get().instruction.as_ref() {
            Instruction::Call(func, args) => match func {
                Func::GetElementPtr => {
                    let this = args[0];
                    let indices = [&args[1..], indices].concat();
                    self.gep(this, &indices, t)
                }
                _ => self.gep(this, indices, t),
            },
            _ => self.gep(this, indices, t),
        }
    }
    pub fn update(&mut self, var: NodeRef, value: NodeRef) -> NodeRef {
        assert!(
            context::is_type_equal(var.type_(), value.type_()),
            "type mismatch {} {}",
            var.type_(),
            value.type_()
        );
        match var.get().instruction.as_ref() {
            Instruction::Local { .. } => {}
            Instruction::Argument { by_value } => {
                assert!(!*by_value, "updating argument passed by value");
            }
            Instruction::Shared { .. } => {}
            Instruction::Call(func, _) => match func {
                Func::GetElementPtr => {}
                _ => panic!("not local or getelementptr"),
            },
            _ => panic!("not a var"),
        }
        let node = Node::new(CArc::new(Instruction::Update { var, value }), Type::void());
        let node = new_node(&self.pools, node);
        self.append(node);
        node
    }
    pub fn update_unchecked(&mut self, var: NodeRef, value: NodeRef) -> NodeRef {
        let node = Node::new(CArc::new(Instruction::Update { var, value }), Type::void());
        let node = new_node(&self.pools, node);
        self.append(node);
        node
    }
    pub fn phi(&mut self, incoming: &[PhiIncoming], t: CArc<Type>) -> NodeRef {
        if t == Type::userdata() {
            let userdata0 = incoming[0].value.get_user_data();
            for i in 1..incoming.len() {
                if incoming[i].value.is_unreachable() {
                    continue;
                }
                let userdata = incoming[i].value.get_user_data();
                assert_eq!(
                    userdata0.tag, userdata.tag,
                    "Different UserData node found!"
                );
                assert_eq!(userdata0.eq, userdata.eq, "Different UserData node found!");
                assert!(
                    (userdata0.eq)(userdata0.data, userdata.data),
                    "Different UserData node found!"
                );
            }
            return incoming[0].value;
        }
        let node = Node::new(
            CArc::new(Instruction::Phi(CBoxedSlice::new(incoming.to_vec()))),
            t,
        );
        let node = new_node(&self.pools, node);
        self.append(node.clone());
        node
    }
    pub fn switch(
        &mut self,
        value: NodeRef,
        cases: &[SwitchCase],
        default: Pooled<BasicBlock>,
    ) -> NodeRef {
        let node = Node::new(
            CArc::new(Instruction::Switch {
                value,
                default,
                cases: CBoxedSlice::new(cases.to_vec()),
            }),
            Type::void(),
        );
        let node = new_node(&self.pools, node);
        self.append(node);
        node
    }
    pub fn if_(
        &mut self,
        cond: NodeRef,
        true_branch: Pooled<BasicBlock>,
        false_branch: Pooled<BasicBlock>,
    ) -> NodeRef {
        let node = Node::new(
            CArc::new(Instruction::If {
                cond,
                true_branch,
                false_branch,
            }),
            Type::void(),
        );
        let node = new_node(&self.pools, node);
        self.append(node);
        node
    }
    pub fn ad_detach(&mut self, body: Pooled<BasicBlock>) -> NodeRef {
        let node = Node::new(CArc::new(Instruction::AdDetach(body)), Type::void());
        let node = new_node(&self.pools, node);
        self.append(node);
        node
    }
    pub fn ad_scope(&mut self, body: Pooled<BasicBlock>) -> NodeRef {
        let node = Node::new(
            CArc::new(Instruction::AdScope {
                body,
                forward: false,
                n_forward_grads: 0,
            }),
            Type::void(),
        );
        let node = new_node(&self.pools, node);
        self.append(node);
        node
    }
    pub fn fwd_ad_scope(&mut self, body: Pooled<BasicBlock>, n_grads: usize) -> NodeRef {
        let node = Node::new(
            CArc::new(Instruction::AdScope {
                body,
                forward: true,
                n_forward_grads: n_grads,
            }),
            Type::void(),
        );
        let node = new_node(&self.pools, node);
        self.append(node);
        node
    }
    pub fn ray_query(
        &mut self,
        ray_query: NodeRef,
        on_triangle_hit: Pooled<BasicBlock>,
        on_procedural_hit: Pooled<BasicBlock>,
        type_: CArc<Type>,
    ) -> NodeRef {
        let node = Node::new(
            CArc::new(Instruction::RayQuery {
                ray_query,
                on_triangle_hit,
                on_procedural_hit,
            }),
            type_,
        );
        let node = new_node(&self.pools, node);
        self.append(node);
        node
    }
    pub fn loop_(&mut self, body: Pooled<BasicBlock>, cond: NodeRef) -> NodeRef {
        let node = Node::new(CArc::new(Instruction::Loop { body, cond }), Type::void());
        let node = new_node(&self.pools, node);
        self.append(node);
        node
    }
    pub fn generic_loop(
        &mut self,
        prepare: Pooled<BasicBlock>,
        cond: NodeRef,
        body: Pooled<BasicBlock>,
        update: Pooled<BasicBlock>,
    ) -> NodeRef {
        let node = Node::new(
            CArc::new(Instruction::GenericLoop {
                prepare,
                cond,
                body,
                update,
            }),
            Type::void(),
        );
        let node = new_node(&self.pools, node);
        self.append(node);
        node
    }
    pub fn finish(self) -> Pooled<BasicBlock> {
        self.bb
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum Usage {
    NONE,
    READ,
    WRITE,
    READ_WRITE,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum UsageMark {
    READ,
    WRITE,
}

impl Usage {
    pub fn mark(&self, usage: UsageMark) -> Usage {
        match (self, usage) {
            (Usage::NONE, UsageMark::READ) => Usage::READ,
            (Usage::NONE, UsageMark::WRITE) => Usage::WRITE,
            (Usage::READ, UsageMark::READ) => Usage::READ,
            (Usage::READ, UsageMark::WRITE) => Usage::READ_WRITE,
            (Usage::WRITE, UsageMark::READ) => Usage::READ_WRITE,
            (Usage::WRITE, UsageMark::WRITE) => Usage::WRITE,
            (Usage::READ_WRITE, _) => Usage::READ_WRITE,
        }
    }
    pub fn to_u8(&self) -> u8 {
        match self {
            Usage::NONE => 0,
            Usage::READ => 1,
            Usage::WRITE => 2,
            Usage::READ_WRITE => 3,
        }
    }
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_node_usage(kernel: &KernelModule) -> CBoxedSlice<u8> {
    let mut usage_map = detect_usage(&kernel.module);
    let mut usage = Vec::new();
    for captured in kernel.captures.as_ref() {
        usage.push(
            usage_map
                .remove(&captured.node)
                .expect(
                    format!(
                        "Requested resource {} not exist in usage map",
                        captured.node.0
                    )
                    .as_str(),
                )
                .to_u8(),
        );
    }
    for argument in kernel.args.as_ref() {
        usage.push(
            usage_map
                .remove(argument)
                .expect(
                    format!("Requested argument {} not exist in usage map", argument.0).as_str(),
                )
                .to_u8(),
        );
    }
    CBoxedSlice::new(usage)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_type_size(ty: &CArc<Type>) -> usize {
    ty.size()
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_type_alignment(ty: &CArc<Type>) -> usize {
    ty.alignment()
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_node(pools: CArc<ModulePools>, node: Node) -> NodeRef {
    new_node(&pools, node)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_node_get(node_ref: NodeRef) -> *const Node {
    node_ref.get()
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_node_replace_with(node_ref: NodeRef, new_node: *const Node) {
    let new_node = unsafe { &*new_node };
    node_ref.replace_with(new_node);
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_node_insert_before_self(node_ref: NodeRef, new_node: NodeRef) {
    node_ref.insert_before_self(new_node);
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_node_insert_after_self(node_ref: NodeRef, new_node: NodeRef) {
    node_ref.insert_after_self(new_node);
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_node_remove(node_ref: NodeRef) {
    node_ref.remove();
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_append_node(builder: &mut IrBuilder, node_ref: NodeRef) {
    builder.append(node_ref)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_call(
    builder: &mut IrBuilder,
    func: Func,
    args: CSlice<NodeRef>,
    ret_type: CArc<Type>,
) -> NodeRef {
    let args = args.as_ref();
    builder.call(func, args, ret_type)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_const(builder: &mut IrBuilder, const_: Const) -> NodeRef {
    builder.const_(const_)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_update(
    builder: &mut IrBuilder,
    var: NodeRef,
    value: NodeRef,
) {
    builder.update(var, value);
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_local(builder: &mut IrBuilder, init: NodeRef) -> NodeRef {
    builder.local(init)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_if(
    builder: &mut IrBuilder,
    cond: NodeRef,
    true_branch: Pooled<BasicBlock>,
    false_branch: Pooled<BasicBlock>,
) -> NodeRef {
    builder.if_(cond, true_branch, false_branch)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_phi(
    builder: &mut IrBuilder,
    incoming: CSlice<PhiIncoming>,
    t: CArc<Type>,
) -> NodeRef {
    let incoming = incoming.as_ref();
    builder.phi(incoming, t)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_switch(
    builder: &mut IrBuilder,
    value: NodeRef,
    cases: CSlice<SwitchCase>,
    default: Pooled<BasicBlock>,
) -> NodeRef {
    let cases = cases.as_ref();
    builder.switch(value, cases, default)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_generic_loop(
    builder: &mut IrBuilder,
    prepare: Pooled<BasicBlock>,
    cond: NodeRef,
    body: Pooled<BasicBlock>,
    update: Pooled<BasicBlock>,
) -> NodeRef {
    builder.generic_loop(prepare, cond, body, update)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_loop(
    builder: &mut IrBuilder,
    body: Pooled<BasicBlock>,
    cond: NodeRef,
) -> NodeRef {
    builder.loop_(body, cond)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_local_zero_init(
    builder: &mut IrBuilder,
    ty: CArc<Type>,
) -> NodeRef {
    builder.local_zero_init(ty)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_module_pools() -> *mut CArcSharedBlock<ModulePools> {
    CArc::into_raw(CArc::new(ModulePools::new()))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_builder(pools: CArc<ModulePools>) -> IrBuilder {
    IrBuilder::new(pools.clone())
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_builder_set_insert_point(
    builder: &mut IrBuilder,
    node_ref: NodeRef,
) {
    builder.set_insert_point(node_ref);
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_finish(builder: IrBuilder) -> Pooled<BasicBlock> {
    builder.finish()
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_instruction(
    inst: Instruction,
) -> *mut CArcSharedBlock<Instruction> {
    CArc::into_raw(CArc::new(inst))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_callable_module(
    m: CallableModule,
) -> *mut CArcSharedBlock<CallableModule> {
    CArc::into_raw(CArc::new(m))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_kernel_module(
    m: KernelModule,
) -> *mut CArcSharedBlock<KernelModule> {
    CArc::into_raw(CArc::new(m))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_block_module(
    m: BlockModule,
) -> *mut CArcSharedBlock<BlockModule> {
    CArc::into_raw(CArc::new(m))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_register_type(ty: &Type) -> *mut CArcSharedBlock<Type> {
    CArc::into_raw(context::register_type(ty.clone()))
}

// #[no_mangle]
// pub extern "C"
pub mod debug {
    use crate::display::DisplayIR;
    use std::ffi::CString;

    use super::*;

    pub fn dump_ir_json(module: &Module) -> serde_json::Value {
        serde_json::to_value(&module).unwrap()
    }

    pub fn dump_ir_binary(module: &Module) -> Vec<u8> {
        bincode::serialize(module).unwrap()
    }

    pub fn dump_ir_human_readable(module: &Module) -> String {
        let mut d = DisplayIR::new();
        d.display_ir(module)
    }

    #[no_mangle]
    pub extern "C" fn luisa_compute_ir_dump_json(module: &Module) -> CBoxedSlice<u8> {
        let json = dump_ir_json(module);
        let s = serde_json::to_string(&json).unwrap();
        let cstring = CString::new(s).unwrap();
        CBoxedSlice::new(cstring.as_bytes().to_vec())
    }

    #[no_mangle]
    pub extern "C" fn luisa_compute_ir_dump_binary(module: &Module) -> CBoxedSlice<u8> {
        CBoxedSlice::new(dump_ir_binary(module))
    }

    #[no_mangle]
    pub extern "C" fn luisa_compute_ir_dump_human_readable(module: &Module) -> CBoxedSlice<u8> {
        let mut d = DisplayIR::new();
        let s = d.display_ir(module);
        let cstring = CString::new(s).unwrap();
        CBoxedSlice::new(cstring.as_bytes().to_vec())
    }

    #[no_mangle]
    pub extern "C" fn luisa_compute_ir_ast_json_to_ir_kernel(
        j: CBoxedSlice<u8>,
    ) -> *mut CArcSharedBlock<KernelModule> {
        let j = j.to_string();
        let kernel = ast2ir::convert_ast_to_ir_kernel(j);
        CArc::into_raw(kernel)
    }

    #[no_mangle]
    pub extern "C" fn luisa_compute_ir_ast_json_to_ir_callable(
        j: CBoxedSlice<u8>,
    ) -> *mut CArcSharedBlock<CallableModule> {
        let j = j.to_string();
        let callable = ast2ir::convert_ast_to_ir_callable(j);
        CArc::into_raw(callable)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_layout() {
        assert_eq!(std::mem::size_of::<super::NodeRef>(), 8);
    }
}
