use gc::{GcHeader, GcObject, Trace};
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use crate::context::with_context;
use crate::*;
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::binary_heap::Iter;
use std::collections::HashSet;
use std::ffi::CString;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
#[derive(Serialize)]
pub enum Primitive {
    Bool,
    Int32,
    Uint32,
    Int64,
    Uint64,
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
                Self::Int32 => "i32",
                Self::Uint32 => "u32",
                Self::Int64 => "i64",
                Self::Uint64 => "u64",
                Self::Float32 => "f32",
                Self::Float64 => "f64",
            }
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub enum VectorElementType {
    Scalar(Primitive),
    Vector(Gc<VectorType>),
}

impl VectorElementType {
    pub fn is_float(&self) -> bool {
        match self {
            VectorElementType::Scalar(Primitive::Float32) => true,
            VectorElementType::Scalar(Primitive::Float64) => true,
            VectorElementType::Vector(v) => v.element.is_float(),
            _ => false,
        }
    }
    pub fn is_int(&self) -> bool {
        match self {
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
    pub fn to_type(&self) -> Gc<Type> {
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
    pub fields: CBoxedSlice<Gc<Type>>,
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
    pub element: Gc<Type>,
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
        }
    }
}

impl Trace for Type {
    fn trace(&self) {
        match self {
            Type::Void => {}
            Type::UserData => {}
            Type::Primitive(_) => {}
            Type::Vector(v) => v.trace(),
            Type::Matrix(m) => m.trace(),
            Type::Struct(s) => s.trace(),
            Type::Array(a) => a.trace(),
        }
    }
}

impl Trace for VectorElementType {
    fn trace(&self) {
        match self {
            VectorElementType::Scalar(_) => {}
            VectorElementType::Vector(v) => v.trace(),
        }
    }
}

impl Trace for VectorType {
    fn trace(&self) {
        self.element.trace();
    }
}

impl Trace for MatrixType {
    fn trace(&self) {
        self.element.trace();
    }
}

impl Trace for StructType {
    fn trace(&self) {
        self.fields.trace();
    }
}

impl Trace for ArrayType {
    fn trace(&self) {
        self.element.trace();
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
            Primitive::Int32 => 4,
            Primitive::Uint32 => 4,
            Primitive::Int64 => 8,
            Primitive::Uint64 => 8,
            Primitive::Float32 => 4,
            Primitive::Float64 => 8,
        }
    }
}

impl VectorType {
    pub fn size(&self) -> usize {
        self.element.size() * self.length as usize
    }
}

impl MatrixType {
    pub fn size(&self) -> usize {
        self.element.size() * self.dimension as usize * self.dimension as usize
    }
    pub fn column(&self) -> Gc<Type> {
        match self.element {
            VectorElementType::Scalar(t) => Type::vector(t, self.dimension),
            VectorElementType::Vector(t) => Type::vector_vector(t, self.dimension),
        }
    }
}

impl Type {
    pub fn void() -> Gc<Type> {
        context::register_type(Type::Void)
    }
    pub fn userdata() -> Gc<Type> {
        context::register_type(Type::UserData)
    }
    pub fn size(&self) -> usize {
        match self {
            Type::Void | Type::UserData => 0,
            Type::Primitive(t) => t.size(),
            Type::Struct(t) => t.size,
            Type::Vector(t) => t.size(),
            Type::Matrix(t) => t.size(),
            Type::Array(t) => t.element.size() * t.length,
        }
    }
    pub fn element(&self) -> Gc<Type> {
        match self {
            Type::Void | Type::Primitive(_) | Type::UserData => {
                context::register_type(self.clone())
            }
            Type::Vector(vec_type) => vec_type.element.to_type(),
            Type::Matrix(mat_type) => mat_type.element.to_type(),
            Type::Struct(_) => Gc::null(),
            Type::Array(arr_type) => arr_type.element,
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
        }
    }
    pub fn alignment(&self) -> usize {
        match self {
            Type::Void | Type::UserData => 0,
            Type::Primitive(t) => t.size(),
            Type::Struct(t) => t.alignment,
            Type::Vector(t) => t.element.size(), // TODO
            Type::Matrix(t) => t.element.size(),
            Type::Array(t) => t.element.alignment(),
        }
    }
    pub fn vector_to_bool(from: &VectorType) -> Gc<VectorType> {
        match from.element {
            VectorElementType::Scalar(_) => Gc::new(VectorType {
                element: VectorElementType::Scalar(Primitive::Bool),
                length: from.length,
            }),
            VectorElementType::Vector(v) => Type::vector_to_bool(v.deref()),
        }
    }
    pub fn bool(from: Gc<Type>) -> Gc<Type> {
        match from.deref() {
            Type::Primitive(_) => context::register_type(Type::Primitive(Primitive::Bool)),
            Type::Vector(vec_type) => match vec_type.element {
                VectorElementType::Scalar(_) => Type::vector(Primitive::Bool, vec_type.length),
                VectorElementType::Vector(v) => {
                    Type::vector_vector(Type::vector_to_bool(v.deref()), vec_type.length)
                }
            },
            _ => panic!("Cannot convert to bool"),
        }
    }
    pub fn vector(element: Primitive, length: u32) -> Gc<Type> {
        context::register_type(Type::Vector(VectorType {
            element: VectorElementType::Scalar(element),
            length,
        }))
    }
    pub fn vector_vector(element: Gc<VectorType>, length: u32) -> Gc<Type> {
        context::register_type(Type::Vector(VectorType {
            element: VectorElementType::Vector(element),
            length,
        }))
    }
    pub fn matrix(element: Primitive, dimension: u32) -> Gc<Type> {
        context::register_type(Type::Matrix(MatrixType {
            element: VectorElementType::Scalar(element),
            dimension,
        }))
    }
    pub fn matrix_vector(element: Gc<VectorType>, dimension: u32) -> Gc<Type> {
        context::register_type(Type::Matrix(MatrixType {
            element: VectorElementType::Vector(element),
            dimension,
        }))
    }
    pub fn is_primitive(&self) -> bool {
        match self {
            Type::Primitive(_) => true,
            _ => false,
        }
    }
    pub fn is_float(&self) -> bool {
        match self {
            Type::Primitive(p) => match p {
                Primitive::Float32 | Primitive::Float64 => true,
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
                Primitive::Int32 | Primitive::Uint32 | Primitive::Int64 | Primitive::Uint64 => true,
                _ => false,
            },
            Type::Vector(v) => v.element.is_int(),
            Type::Matrix(m) => m.element.is_int(),
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

#[derive(Clone, Debug, Copy, Serialize)]
#[repr(C)]
pub struct Node {
    pub type_: Gc<Type>,
    pub next: NodeRef,
    pub prev: NodeRef,
    pub instruction: Gc<Instruction>,
}

impl Trace for Node {
    fn trace(&self) {
        self.type_.trace();
        self.instruction.trace();
        self.next.trace();
        self.prev.trace();
    }
}

impl Trace for NodeRef {
    fn trace(&self) {
        unsafe {
            let ptr: Gc<Node> = std::mem::transmute(self.0);
            ptr.trace();
        }
    }
}

impl Trace for Instruction {
    fn trace(&self) {
        match self {
            Instruction::Buffer => {}
            Instruction::Bindless => {}
            Instruction::Texture2D => {}
            Instruction::Texture3D => {}
            Instruction::Accel => {}
            Instruction::Shared => {}
            Instruction::Uniform => {}
            Instruction::Local { init } => init.trace(),
            Instruction::Argument { .. } => todo!(),
            Instruction::UserData(_) => {}
            Instruction::Invalid => {}
            Instruction::Const(c) => c.trace(),
            Instruction::Update { var, value } => {
                var.trace();
                value.trace();
            }
            Instruction::Call(f, args) => {
                f.trace();
                args.trace();
            }
            Instruction::Phi(incomings) => {
                incomings.trace();
            }
            Instruction::Return(v) => v.trace(),
            Instruction::Loop { body, cond } => {
                body.trace();
                cond.trace();
            }
            Instruction::GenericLoop { .. } => todo!(),
            Instruction::Break => {}
            Instruction::Continue => {}
            Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                cond.trace();
                true_branch.trace();
                false_branch.trace();
            }
            Instruction::Switch {
                value,
                default,
                cases,
            } => {
                value.trace();
                default.trace();
                cases.trace();
            }
            Instruction::AdScope {
                forward,
                backward,
                epilogue,
            } => {
                forward.trace();
                backward.trace();
                epilogue.trace();
            }
            Instruction::AdDetach(bb) => bb.trace(),
            Instruction::Comment(_) => todo!(),
            crate::ir::Instruction::Debug { .. } => {}
        }
    }
}

pub const INVALID_REF: NodeRef = NodeRef(0);

impl Node {
    pub fn new(instruction: Gc<Instruction>, type_: Gc<Type>) -> Node {
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
    Unreachable,
    Assert, // Assert(condition, message*) message is of Instruction::Debug

    ThreadId,
    BlockId,
    DispatchId,
    DispatchSize,

    RequiresGradient,
    Gradient,
    GradientMarker, // marks a (node, gradient) tuple
    AccGrad,        // grad (local), increment
    Detach,

    // (handle, instance_id) -> Mat4
    RayTracingInstanceTransform,
    RayTracingInstanceAabb,
    RayTracingInstanceVisibility,
    RayTracingInstanceOpacity,
    RayTracingSetInstanceAabb,
    RayTracingSetInstanceTransform,
    RayTracingSetInstanceOpactiy,
    RayTracingSetInstanceVisibility,
    // (handle, Ray) -> Hit
    // struct Ray alignas(16) { float origin[3], tmin; float direction[3], tmax; };
    // struct Hit alignas(16) { uint inst; uint prim; float u; float v; };
    RayTracingTraceClosest,
    RayTracingTraceAny,

    RayQueryProceed,
    RayQueryIsCandidateTriangle,
    RayQueryProceduralCandidateHit,
    RayQueryTriangleCandidateHit,
    RayQueryCommittedHit,
    RayQueryCommitTriangle,
    RayQueryCommitProcedural,

    /// When referencing a Local in Call, it is always interpreted as a load
    /// However, there are cases you want to do this explicitly
    Load,

    Cast,
    Bitcast,

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
    // (a, b) => a * b^T
    OuterProduct,
    Length,
    LengthSquared,
    Normalize,
    Faceforward,

    // Matrix operations
    Determinant,
    Transpose,
    Inverse,

    SynchronizeBlock,

    /// (buffer/smem, index, desired) -> old: stores desired, returns old.
    AtomicExchange,
    /// (buffer/smem, index, expected, desired) -> old: stores (old == expected ? desired : old), returns old.
    AtomicCompareExchange,
    /// (buffer/smem, index, val) -> old: stores (old + val), returns old.
    AtomicFetchAdd,
    /// (buffer/smem, index, val) -> old: stores (old - val), returns old.
    AtomicFetchSub,
    /// (buffer/smem, index, val) -> old: stores (old & val), returns old.
    AtomicFetchAnd,
    /// (buffer/smem, index, val) -> old: stores (old | val), returns old.
    AtomicFetchOr,
    /// (buffer/smem, index, val) -> old: stores (old ^ val), returns old.
    AtomicFetchXor,
    /// (buffer/smem, index, val) -> old: stores min(old, val), returns old.
    AtomicFetchMin,
    /// (buffer/smem, index, val) -> old: stores max(old, val), returns old.
    AtomicFetchMax,
    // memory access
    /// (buffer, index) -> value: reads the index-th element in bu
    BufferRead,
    /// (buffer, index, value) -> void: writes value into the inde
    BufferWrite,
    /// buffer -> uint: returns buffer size in *elements*
    BufferSize,
    /// (texture, coord) -> value
    TextureRead,
    /// (texture, coord, value) -> void
    TextureWrite,
    ///(bindless_array, index: uint, uv: float2) -> float4
    BindlessTexture2dSample,
    ///(bindless_array, index: uint, uv: float2, level: float) -> float4
    BindlessTexture2dSampleLevel,
    ///(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2) -> float4
    BindlessTexture2dSampleGrad,
    ///(bindless_array, index: uint, uv: float3) -> float4
    BindlessTexture3dSample,
    ///(bindless_array, index: uint, uv: float3, level: float) -> float
    BindlessTexture3dSampleLevel,
    ///(bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3) -> float4
    BindlessTexture3dSampleGrad,
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
    /// (bindless_array, index: uint) -> uint: returns the size of the buffer in *elements*
    BindlessBufferSize(Gc<Type>),
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

    // scalar -> matrix, all elements are set to the scalar
    Mat,
    // vector x 2 -> matrix
    Mat2,
    // vector x 3 -> matrix
    Mat3,
    // vector x 4 -> matrix
    Mat4,

    Callable(u64),

    // ArgT -> ArgT
    CpuCustomOp(CRc<CpuCustomOp>),
}

impl Trace for Func {
    fn trace(&self) {}
}

#[derive(Clone, Debug, Serialize)]
#[repr(C)]
pub enum Const {
    Zero(Gc<Type>),
    One(Gc<Type>),
    Bool(bool),
    Int32(i32),
    Uint32(u32),
    Int64(i64),
    Uint64(u64),
    Float32(f32),
    Float64(f64),
    Generic(CBoxedSlice<u8>, Gc<Type>),
}

impl std::fmt::Display for Const {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Const::Zero(t) => write!(f, "0_{}", t),
            Const::One(t) => write!(f, "1_{}", t),
            Const::Bool(b) => write!(f, "{}", b),
            Const::Int32(i) => write!(f, "{}", i),
            Const::Uint32(u) => write!(f, "{}", u),
            Const::Int64(i) => write!(f, "{}", i),
            Const::Uint64(u) => write!(f, "{}", u),
            Const::Float32(fl) => write!(f, "{}", fl),
            Const::Float64(fl) => write!(f, "{}", fl),
            Const::Generic(data, t) => write!(f, "byte<{}>[{}]", t, data.as_ref().len()),
        }
    }
}

impl Trace for Const {
    fn trace(&self) {
        match self {
            Const::Zero(ty) => ty.trace(),
            Const::Generic(_, ty) => ty.trace(),
            _ => {}
        }
    }
}

impl Const {
    pub fn get_i32(&self) -> i32 {
        match self {
            Const::Int32(v) => *v,
            Const::Uint32(v) => *v as i32,
            _ => panic!("cannot convert to i32"),
        }
    }
    pub fn type_(&self) -> Gc<Type> {
        match self {
            Const::Zero(ty) => ty.clone(),
            Const::One(ty) => ty.clone(),
            Const::Bool(_) => <bool as TypeOf>::type_(),
            Const::Int32(_) => <i32 as TypeOf>::type_(),
            Const::Uint32(_) => <u32 as TypeOf>::type_(),
            Const::Int64(_) => <i64 as TypeOf>::type_(),
            Const::Uint64(_) => <u64 as TypeOf>::type_(),
            Const::Float32(_) => <f32 as TypeOf>::type_(),
            Const::Float64(_) => <f64 as TypeOf>::type_(),
            Const::Generic(_, t) => *t,
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
    pub block: Gc<BasicBlock>,
}

impl Trace for PhiIncoming {
    fn trace(&self) {
        self.block.trace();
        self.value.trace();
    }
}

#[repr(C)]
pub struct CpuCustomOp {
    pub data: *mut u8,
    /// func(data, args); func should modify args in place
    pub func: extern "C" fn(*mut u8, *mut u8),
    pub destructor: extern "C" fn(*mut u8),
    pub arg_type: Gc<Type>,
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
    pub block: Gc<BasicBlock>,
}

impl Trace for SwitchCase {
    fn trace(&self) {
        self.value.trace();
        self.block.trace();
    }
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
    UserData(CRc<UserData>),
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
        body: Gc<BasicBlock>,
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
        prepare: Gc<BasicBlock>,
        cond: NodeRef,
        body: Gc<BasicBlock>,
        update: Gc<BasicBlock>,
    },
    Break,
    Continue,
    If {
        cond: NodeRef,
        true_branch: Gc<BasicBlock>,
        false_branch: Gc<BasicBlock>,
    },
    Switch {
        value: NodeRef,
        default: Gc<BasicBlock>,
        cases: CBoxedSlice<SwitchCase>,
    },
    AdScope {
        forward: Gc<BasicBlock>,
        backward: Gc<BasicBlock>,
        epilogue: Gc<BasicBlock>,
    },
    AdDetach(Gc<BasicBlock>),
    Comment(CBoxedSlice<u8>),
    Debug(CBoxedSlice<u8>), // for CPU only, would print the message if executed
}
extern "C" fn eq_impl<T: UserNodeData>(a: *const u8, b: *const u8) -> bool {
    let a = unsafe { &*(a as *const T) };
    let b = unsafe { &*(b as *const T) };
    a.equal(b)
}
extern "C" fn dtor_impl<T: UserNodeData>(a: *mut UserData) {
    unsafe {
        let data = Box::from_raw((*a).data as *mut T);
        drop(data);
        drop(Box::from_raw(a));
    };
}
fn type_id_u64<T: UserNodeData>() -> u64 {
    unsafe { std::mem::transmute(TypeId::of::<T>()) }
}
pub fn new_user_node<T: UserNodeData>(data: T) -> NodeRef {
    new_node(Node::new(
        Gc::new(Instruction::UserData(CRc::new_with_dtor(
            UserData {
                tag: type_id_u64::<T>(),
                data: Box::into_raw(Box::new(data)) as *mut u8,
                eq: eq_impl::<T>,
            },
            dtor_impl::<T>,
        ))),
        Type::userdata(),
    ))
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

pub fn new_node(node: Node) -> NodeRef {
    with_context(|ctx| ctx.alloc_node(node))
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

#[derive(Debug)]
#[repr(C)]
pub struct BasicBlock {
    pub(crate) first: NodeRef,
    pub(crate) last: NodeRef,
}

impl Trace for BasicBlock {
    fn trace(&self) {
        let mut cur = self.first;
        while cur != INVALID_REF {
            let node = cur.get();
            node.trace();
            cur = node.next;
        }
    }
}

#[derive(Serialize)]
struct NodeRefAndNode {
    id: NodeRef,
    data: Node,
}

impl Serialize for BasicBlock {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("BasicBlock", 1)?;
        let nodes = self.nodes();
        let nodes = nodes
            .iter()
            .map(|n| NodeRefAndNode {
                id: *n,
                data: *n.get(),
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
    pub fn into_vec(&self) -> Vec<NodeRef> {
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
    pub fn new() -> Self {
        let first = new_node(Node::new(Gc::new(Instruction::Invalid), Type::void()));
        let last = new_node(Node::new(Gc::new(Instruction::Invalid), Type::void()));
        first.update(|node| node.next = last);
        last.update(|node| node.prev = first);
        Self { first, last }
    }
    pub fn push(&self, node: NodeRef) {
        // node.insert_before(self.last);
        self.last.insert_before(node);
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
    pub fn merge(&self, other: Gc<BasicBlock>) {
        let nodes = other.into_vec();
        for node in nodes {
            self.push(node);
        }
    }
}

impl NodeRef {
    pub fn get_i32(&self) -> i32 {
        match self.get().instruction.as_ref() {
            Instruction::Const(c) => c.get_i32(),
            _ => panic!("not i32 node; found: {:?}", self.get().instruction),
        }
    }
    pub fn get_user_data(&self) -> &UserData {
        match self.get().instruction.as_ref() {
            Instruction::UserData(data) => data,
            _ => panic!("not user data node; found: {:?}", self.get().instruction),
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
    pub fn is_phi(&self) -> bool {
        match self.get().instruction.as_ref() {
            Instruction::Phi { .. } => true,
            _ => false,
        }
    }
    pub fn get_gc_node(&self) -> Gc<Node> {
        assert!(self.valid());
        unsafe { std::mem::transmute(self.0) }
    }
    pub fn get<'a>(&'a self) -> &'a Node {
        // assert!(self.valid());
        // with_node_pool(|pool| unsafe { std::mem::transmute(pool[self.0]) })
        let gc = self.get_gc_node();
        unsafe { std::mem::transmute(gc.as_ref()) }
    }
    pub fn valid(&self) -> bool {
        self.0 != INVALID_REF.0
    }
    pub fn set(&self, node: Node) {
        let gc = self.get_gc_node();
        unsafe {
            *Gc::get_mut(&gc) = node;
        }
    }
    pub fn update<T>(&self, f: impl FnOnce(&mut Node) -> T) -> T {
        let gc = self.get_gc_node();
        unsafe {
            let node = Gc::get_mut(&gc);
            f(node)
        }
    }
    pub fn type_(&self) -> Gc<Type> {
        self.get().type_
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
    pub fn insert_after(&self, node_ref: NodeRef) {
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
    pub fn insert_before(&self, node_ref: NodeRef) {
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

#[repr(C)]
#[derive(Debug, Serialize)]
pub struct Module {
    pub kind: ModuleKind,
    pub entry: Gc<BasicBlock>,
}

impl Trace for Module {
    fn trace(&self) {
        self.entry.trace();
    }
}

#[repr(C)]
#[derive(Debug, Serialize)]
pub struct CallableModule {
    pub module: Module,
    pub args: CBoxedSlice<NodeRef>,
}

impl Trace for CallableModule {
    fn trace(&self) {
        self.module.trace();
        self.args.trace();
    }
}

// buffer binding
#[repr(C)]
#[derive(Debug, Serialize, Copy, Clone)]
pub struct BufferBinding {
    pub handle: u64,
    pub offset: u64,
    pub size: usize,
}

// texture binding
#[repr(C)]
#[derive(Debug, Serialize, Copy, Clone)]
pub struct TextureBinding {
    pub handle: u64,
    pub level: u32,
}

// bindless array binding
#[repr(C)]
#[derive(Debug, Serialize, Copy, Clone)]
pub struct BindlessArrayBinding {
    pub handle: u64,
}

// accel binding
#[repr(C)]
#[derive(Debug, Serialize, Copy, Clone)]
pub struct AccelBinding {
    pub handle: u64,
}

#[repr(C)]
#[derive(Debug, Serialize, Copy, Clone)]
pub enum Binding {
    Buffer(BufferBinding),
    Texture(TextureBinding),
    BindlessArray(BindlessArrayBinding),
    Accel(AccelBinding),
}

impl Trace for Binding {
    fn trace(&self) {}
}

#[derive(Debug, Serialize, Copy, Clone)]
#[repr(C)]
pub struct Capture {
    pub node: NodeRef,
    pub binding: Binding,
}

impl Trace for Capture {
    fn trace(&self) {
        self.node.trace();
        self.binding.trace();
    }
}

#[repr(C)]
#[derive(Debug, Serialize)]
pub struct KernelModule {
    pub module: Module,
    pub captures: CBoxedSlice<Capture>,
    pub args: CBoxedSlice<NodeRef>,
    pub shared: CBoxedSlice<NodeRef>,
    pub cpu_custom_ops: CBoxedSlice<CRc<CpuCustomOp>>,
    pub block_size: [u32; 3],
}

impl Trace for KernelModule {
    fn trace(&self) {
        self.module.trace();
        self.captures.trace();
        self.args.trace();
        self.shared.trace();
    }
}

impl Module {
    pub fn from_fragment(entry: Gc<BasicBlock>) -> Self {
        Self {
            kind: ModuleKind::Block,
            entry,
        }
    }
}
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
    fn visit_block(&mut self, block: Gc<BasicBlock>) {
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
            Instruction::AdScope {
                forward,
                backward,
                epilogue,
            } => {
                self.visit_block(*forward);
                self.visit_block(*backward);
                self.visit_block(*epilogue);
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
struct ModuleCloner {
    node_map: HashMap<NodeRef, NodeRef>,
}

impl ModuleCloner {
    fn new() -> Self {
        Self {
            node_map: HashMap::new(),
        }
    }
    fn clone_node(&mut self, node: NodeRef, builder: &mut IrBuilder) -> NodeRef {
        if let Some(&node) = self.node_map.get(&node) {
            return node;
        }
        let new_node = match node.get().instruction.as_ref() {
            Instruction::Buffer => node,
            Instruction::Bindless => node,
            Instruction::Texture2D => node,
            Instruction::Texture3D => node,
            Instruction::Accel => node,
            Instruction::Shared => node,
            Instruction::Uniform => node,
            Instruction::Local { .. } => todo!(),
            Instruction::Argument { .. } => todo!(),
            Instruction::UserData(_) => node,
            Instruction::Invalid => node,
            Instruction::Const(_) => todo!(),
            Instruction::Update { var, value } => todo!(),
            Instruction::Call(_, _) => todo!(),
            Instruction::Phi(_) => todo!(),
            Instruction::Loop { body, cond } => todo!(),
            Instruction::GenericLoop { .. } => todo!(),
            Instruction::Break => builder.break_(),
            Instruction::Continue => builder.continue_(),
            Instruction::Return(_) => todo!(),
            Instruction::If { .. } => todo!(),
            Instruction::Switch { .. } => todo!(),
            Instruction::AdScope { .. } => todo!(),
            Instruction::AdDetach(_) => todo!(),
            Instruction::Comment(_) => builder.clone_node(node),
            crate::ir::Instruction::Debug { .. } => builder.clone_node(node),
        };
        self.node_map.insert(node, new_node);
        new_node
    }
    fn clone_block(&mut self, block: Gc<BasicBlock>, mut builder: IrBuilder) -> Gc<BasicBlock> {
        let mut cur = block.first.get().next;
        while cur != block.last {
            let _ = self.clone_node(cur, &mut builder);
            cur = cur.get().next;
        }
        builder.finish()
    }
    fn clone_module(&mut self, module: &Module) -> Module {
        Module {
            kind: module.kind,
            entry: self.clone_block(module.entry, IrBuilder::new()),
        }
    }
}

impl Clone for Module {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind,
            entry: self.entry,
        }
    }
}

#[repr(C)]
pub struct IrBuilder {
    bb: Gc<BasicBlock>,
    pub(crate) insert_point: NodeRef,
}

impl IrBuilder {
    pub fn new() -> Self {
        let bb = Gc::new(BasicBlock::new());
        let insert_point = bb.first;
        Self { bb, insert_point }
    }
    pub fn set_insert_point(&mut self, node: NodeRef) {
        self.insert_point = node;
    }
    pub fn append(&mut self, node: NodeRef) {
        self.insert_point.insert_after(node);
        self.insert_point = node;
    }
    pub fn append_block(&mut self, block: Gc<BasicBlock>) {
        self.bb.merge(block);
        self.insert_point = self.bb.last.get().prev;
    }
    pub fn break_(&mut self) -> NodeRef {
        let new_node = new_node(Node::new(Gc::new(Instruction::Break), Type::void()));
        self.append(new_node);
        new_node
    }
    pub fn continue_(&mut self) -> NodeRef {
        let new_node = new_node(Node::new(Gc::new(Instruction::Continue), Type::void()));
        self.append(new_node);
        new_node
    }
    pub fn return_(&mut self, node: NodeRef) {
        let new_node = new_node(Node::new(Gc::new(Instruction::Return(node)), Type::void()));
        self.append(new_node);
    }
    pub fn zero_initializer(&mut self, ty: Gc<Type>) -> NodeRef {
        self.call(Func::ZeroInitializer, &[], ty)
    }
    pub fn requires_gradient(&mut self, node: NodeRef) -> NodeRef {
        self.call(Func::RequiresGradient, &[node], Type::void())
    }
    pub fn gradient(&mut self, node: NodeRef) -> NodeRef {
        self.call(Func::Gradient, &[node], node.type_())
    }
    pub fn clone_node(&mut self, node: NodeRef) -> NodeRef {
        let node = node.get();
        let new_node = new_node(Node::new(node.instruction, node.type_));
        self.append(new_node);
        new_node
    }
    pub fn const_(&mut self, const_: Const) -> NodeRef {
        let t = const_.type_();
        let node = Node::new(Gc::new(Instruction::Const(const_)), t);
        let node = new_node(node);
        self.append(node.clone());
        node
    }
    pub fn local_zero_init(&mut self, ty: Gc<Type>) -> NodeRef {
        let node = self.zero_initializer(ty);
        let local = self.local(node);
        local
    }
    pub fn local(&mut self, init: NodeRef) -> NodeRef {
        let t = init.type_();
        let node = Node::new(Gc::new(Instruction::Local { init }), t);
        let node = new_node(node);
        self.append(node.clone());
        node
    }
    pub fn store(&mut self, var: NodeRef, value: NodeRef) {
        assert!(var.is_lvalue());
        let node = Node::new(Gc::new(Instruction::Update { var, value }), Type::void());
        let node = new_node(node);
        self.append(node);
    }
    pub fn call(&mut self, func: Func, args: &[NodeRef], ret_type: Gc<Type>) -> NodeRef {
        let node = Node::new(
            Gc::new(Instruction::Call(func, CBoxedSlice::new(args.to_vec()))),
            ret_type,
        );
        let node = new_node(node);
        self.append(node.clone());
        node
    }
    pub fn cast(&mut self, node: NodeRef, t: Gc<Type>) -> NodeRef {
        self.call(Func::Cast, &[node], t)
    }
    pub fn bitcast(&mut self, node: NodeRef, t: Gc<Type>) -> NodeRef {
        self.call(Func::Bitcast, &[node], t)
    }
    pub fn update(&mut self, var: NodeRef, value: NodeRef) {
        match var.get().instruction.as_ref() {
            Instruction::Local { .. } => {}
            Instruction::Call(func, _) => match func {
                Func::GetElementPtr => {}
                _ => panic!("not local or getelementptr"),
            },
            _ => panic!("not a var"),
        }
        let node = Node::new(Gc::new(Instruction::Update { var, value }), Type::void());
        let node = new_node(node);
        self.append(node);
    }
    pub fn phi(&mut self, incoming: &[PhiIncoming], t: Gc<Type>) -> NodeRef {
        if t == Type::userdata() {
            let userdata0 = incoming[0].value.get_user_data();
            for i in 1..incoming.len() {
                let userdata = incoming[i].value.get_user_data();
                assert_eq!(userdata0.tag, userdata.tag, "Different UserData node found!");
                assert_eq!(userdata0.eq, userdata.eq, "Different UserData node found!");
                assert!(
                    (userdata0.eq)(userdata0.data, userdata.data),
                    "Different UserData node found!"
                );
            }
            return incoming[0].value;
        }
        let node = Node::new(
            Gc::new(Instruction::Phi(CBoxedSlice::new(incoming.to_vec()))),
            t,
        );
        let node = new_node(node);
        self.append(node.clone());
        node
    }
    pub fn switch(&mut self, value: NodeRef, cases: &[SwitchCase], default: Gc<BasicBlock>) {
        let node = Node::new(
            Gc::new(Instruction::Switch {
                value,
                default,
                cases: CBoxedSlice::new(cases.to_vec()),
            }),
            Type::void(),
        );
        let node = new_node(node);
        self.append(node);
    }
    pub fn if_(
        &mut self,
        cond: NodeRef,
        true_branch: Gc<BasicBlock>,
        false_branch: Gc<BasicBlock>,
    ) -> NodeRef {
        let node = Node::new(
            Gc::new(Instruction::If {
                cond,
                true_branch,
                false_branch,
            }),
            Type::void(),
        );
        let node = new_node(node);
        self.append(node);
        node
    }
    pub fn loop_(&mut self, body: Gc<BasicBlock>, cond: NodeRef) -> NodeRef {
        let node = Node::new(Gc::new(Instruction::Loop { body, cond }), Type::void());
        let node = new_node(node);
        self.append(node);
        node
    }
    pub fn generic_loop(
        &mut self,
        prepare: Gc<BasicBlock>,
        cond: NodeRef,
        body: Gc<BasicBlock>,
        update: Gc<BasicBlock>,
    ) -> NodeRef {
        let node = Node::new(
            Gc::new(Instruction::GenericLoop {
                prepare,
                cond,
                body,
                update,
            }),
            Type::void(),
        );
        let node = new_node(node);
        self.append(node);
        node
    }
    pub fn finish(self) -> Gc<BasicBlock> {
        self.bb
    }
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_node(node: Node) -> NodeRef {
    new_node(node)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_node_get(node_ref: NodeRef) -> *const Node {
    node_ref.get()
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_node_set_root(node_ref: NodeRef, flag: bool) {
    if flag {
        Gc::set_root(node_ref.get_gc_node());
    } else {
        Gc::unset_root(node_ref.get_gc_node());
    }
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
    ret_type: Gc<Type>,
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
    builder.update(var, value)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_local(builder: &mut IrBuilder, init: NodeRef) -> NodeRef {
    builder.local(init)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_local_zero_init(
    builder: &mut IrBuilder,
    ty: Gc<Type>,
) -> NodeRef {
    builder.local_zero_init(ty)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_builder() -> IrBuilder {
    IrBuilder::new()
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_finish(builder: IrBuilder) -> Gc<BasicBlock> {
    builder.finish()
}

#[no_mangle]
pub extern "C" fn luisa_compute_gc_create_context() -> *mut gc::GcContext {
    gc::create_context()
}

#[no_mangle]
pub extern "C" fn luisa_compute_gc_set_context(ctx: *mut gc::GcContext) {
    assert_ne!(ctx, std::ptr::null_mut());
    if ctx == gc::context() {
        return;
    }
    gc::set_context(ctx)
}

#[no_mangle]
pub extern "C" fn luisa_compute_gc_context() -> *mut gc::GcContext {
    gc::context()
}

#[no_mangle]
pub unsafe extern "C" fn luisa_compute_gc_destroy_context() {
    gc::destroy_context()
}

#[no_mangle]
pub unsafe extern "C" fn luisa_compute_gc_collect() {
    gc::collect()
}

#[no_mangle]
pub extern "C" fn luisa_compute_gc_append_object(object: *mut GcHeader) {
    gc::gc_append_object(object)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_instruction(inst: Instruction) -> Gc<Instruction> {
    Gc::new(inst)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_callable_module(
    m: CallableModule,
) -> *mut GcObject<CallableModule> {
    Gc::into_raw(Gc::new(m))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_kernel_module(
    m: KernelModule,
) -> *mut GcObject<KernelModule> {
    Gc::into_raw(Gc::new(m))
}
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
}

#[cfg(test)]
mod test {
    #[test]
    fn test_layout() {
        assert_eq!(std::mem::size_of::<super::NodeRef>(), 8);
    }
}
