use bumpalo::Bump;
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use crate::context::with_context;
use crate::*;
use std::any::Any;
use std::cell::RefCell;
use std::ffi::CString;
use std::fmt::Debug;

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

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub enum VectorElementType {
    Scalar(Primitive),
    Vector(&'static VectorType),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub struct VectorType {
    pub element: VectorElementType,
    pub length: u32,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub struct MatrixType {
    pub element: VectorElementType,
    pub dimension: u32,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub struct StructType {
    pub fields: CSlice<'static, &'static Type>,
    pub alignment: usize,
    pub size: usize,
    // pub id: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(C)]
pub enum Type {
    Void,
    Primitive(Primitive),
    Vector(VectorType),
    Matrix(MatrixType),
    Struct(StructType),
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
}
impl Type {
    pub fn size(&self) -> usize {
        match self {
            Type::Void => 0,
            Type::Primitive(t) => t.size(),

            Type::Struct(t) => t.size,
            Type::Vector(t) => t.size(),
            Type::Matrix(t) => t.size(),
        }
    }
    pub fn alignment(&self) -> usize {
        match self {
            Type::Void => 0,
            Type::Primitive(t) => t.size(),

            Type::Struct(t) => t.alignment,
            Type::Vector(t) => t.element.size(), // TODO
            Type::Matrix(t) => t.element.size(),
        }
    }
}
#[derive(Clone, Debug, Copy, Serialize)]
#[repr(C)]
pub struct Node {
    pub type_: &'static Type,
    pub next: NodeRef,
    pub prev: NodeRef,
    pub instruction: &'static Instruction,
}

pub const INVALID_REF: NodeRef = NodeRef(usize::MAX);
impl Node {
    pub fn new(instruction: &'static Instruction, type_: &'static Type) -> Node {
        Node {
            instruction,
            type_,
            next: INVALID_REF,
            prev: INVALID_REF,
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize)]
#[repr(C)]
pub enum Func {
    ZeroInitializer,

    Assume,
    Unreachable,
    Assert,

    RequiresGradient,
    Gradient,
    GradientMarker, // marks a (node, gradient) tuple

    InstanceToWorldMatrix,
    TraceClosest,
    TraceAny,
    SetInstanceTransform,
    SetInstanceVisibility,

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

    // Unary op
    Neg,
    Not,
    BitNot,

    All,
    Any,

    Select,
    Clamp,
    Lerp,
    Step,

    Abs,
    Min,
    Max,

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
    Length,
    LenghtSquared,
    Normalize,
    Faceforward,

    // Matrix operations
    Determinant,
    Transpose,
    Inverse,

    SynchronizeBlock,

    AtomicExchange,
    /// [(atomic_ref, desired) -> old]: stores desired, returns old.
    AtomicCompareExchange,
    /// [(atomic_ref, expected, desired) -> old]: stores (old == expected ? desired : old), returns old.
    AtomicFetchAdd,
    /// [(atomic_ref, val) -> old]: stores (old + val), returns old.
    AtomicFetchSub,
    /// [(atomic_ref, val) -> old]: stores (old - val), returns old.
    AtomicFetchAnd,
    /// [(atomic_ref, val) -> old]: stores (old & val), returns old.
    AtomicFetchOr,
    /// [(atomic_ref, val) -> old]: stores (old | val), returns old.
    AtomicFetchXor,
    /// [(atomic_ref, val) -> old]: stores (old ^ val), returns old.
    AtomicFetchMin,
    /// [(atomic_ref, val) -> old]: stores min(old, val), returns old.
    AtomicFetchMax,
    /// [(atomic_ref, val) -> old]: stores max(old, val), returns old.
    // memory access
    BufferRead,
    /// [(buffer, index) -> value]: reads the index-th element in bu
    BufferWrite,
    /// [(buffer, index, value) -> void]: writes value into the inde
    TextureRead,
    /// [(texture, coord) -> value]
    TextureWrite,
    /// [(texture, coord, value) -> void]
    // bindless texture
    BindlessTexture2dSample,
    ///(bindless_array, index: uint, uv: float2): float4
    BindlessTexture2dSampleLevel,
    ///(bindless_array, index: uint, uv: float2, level: float): float
    BindlessTexture2dSampleGrad,
    ///(bindless_array, index: uint, uv: float2, ddx: float2, ddy: fl
    BindlessTexture3dSample,
    ///(bindless_array, index: uint, uv: float3): float4
    BindlessTexture3dSampleLevel,
    ///(bindless_array, index: uint, uv: float3, level: float): float
    BindlessTexture3dSampleGrad,
    ///(bindless_array, index: uint, uv: float3, ddx: float3, ddy: fl
    BindlessTexture2dRead,
    ///(bindless_array, index: uint, coord: uint2): float4
    BindlessTexture3dRead,
    ///(bindless_array, index: uint, coord: uint3): float4
    BindlessTexture2dReadLevel,
    ///(bindless_array, index: uint, coord: uint2, level: uint): floa
    BindlessTexture3dReadLevel,
    ///(bindless_array, index: uint, coord: uint3, level: uint): floa
    BindlessTexture2dSize,
    ///(bindless_array, index: uint): uint2
    BindlessTexture3dSize,
    ///(bindless_array, index: uint): uint3
    BindlessTexture2dSizeLevel,
    ///(bindless_array, index: uint, level: uint): uint2
    BindlessTexture3dSizeLevel,
    ///(bindless_array, index: uint, level: uint): uint3
    BindlessBufferRead,

    Vec,  // scalar -> vector, the resulting type is stored in node
    Vec2, // (scalar, scalar) -> vector
    Vec3, // (scalar, scalar, scalar) -> vector
    Vec4, // (scalar, scalar, scalar, scalar) -> vector

    Permute,        // (vector, indices,...) -> vector
    ExtractElement, // (vector, index) -> scalar
    InsertElement,  // (vector, scalar, index) -> vector

    ExtractValue,  //(struct, index) -> value
    InsertValue,   //(struct, value, index) -> struct
    GetElementPtr, //(struct, index) -> value; the value can be passed to an Update instruction

    Matrix,  // scalar -> matrix, the resulting type is stored in node
    Matrix2, // scalar x 4 -> matrix
    Matrix3, // scalar x 9 -> matrix
    Matrix4, // scalar x 16 -> matrix
}
#[derive(Clone, Debug, Serialize)]
#[repr(C)]
pub enum Const {
    Zero(&'static Type),
    Bool(bool),
    Int32(i32),
    Uint32(u32),
    Int64(i64),
    Uint64(u64),
    Float32(f32),
    Float64(f64),
    Generic(CBoxedSlice<u8>, &'static Type),
}
impl Const {
    pub fn get_i32(&self) -> i32 {
        match self {
            Const::Int32(v) => *v,
            _ => panic!("not an i32"),
        }
    }
    pub fn type_(&self) -> &'static Type {
        match self {
            Const::Zero(ty) => ty.clone(),
            Const::Bool(_) => <bool as TypeOf>::type_(),
            Const::Int32(_) => <i32 as TypeOf>::type_(),
            Const::Uint32(_) => <u32 as TypeOf>::type_(),
            Const::Int64(_) => <i64 as TypeOf>::type_(),
            Const::Uint64(_) => <u64 as TypeOf>::type_(),
            Const::Float32(_) => <f32 as TypeOf>::type_(),
            Const::Float64(_) => <f64 as TypeOf>::type_(),
            Const::Generic(_, t) => t,
        }
    }
}
/// cbindgen:derive-eq
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, PartialOrd, Ord)]
#[repr(C)]
pub struct NodeRef(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize)]
#[repr(C)]
pub struct UserNodeDataRef(pub usize);

#[derive(Clone, Copy, Debug, Serialize)]
#[repr(C)]
pub struct PhiIncoming {
    pub value: NodeRef,
    pub block: &'static BasicBlock,
}

#[repr(C)]
pub struct CpuCustomOp {
    pub name: CBoxedSlice<u8>,
    pub data: *mut u8,
    /// func(data: *mut u8, active:*const u8, arg:*mut u8, vector_length: u32)
    pub func: extern "C" fn(*mut u8, *const u8, *mut u8, u32),
    pub destructor: extern "C" fn(*mut u8),
}
impl Serialize for CpuCustomOp {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("CpuCustomOp", 1)?;
        state.serialize_field("name", &CString::from(self.name.clone()))?;
        state.end()
    }
}
impl Debug for CpuCustomOp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        f.debug_struct("CpuCustomOp")
            .field("name", &CString::from(self.name.clone()))
            .finish()
    }
}

#[repr(C)]
#[derive(Clone, Debug, Serialize)]
pub struct SwitchCase {
    pub value: NodeRef,
    pub block: &'static BasicBlock,
}
#[repr(C)]
#[derive(Clone, Debug, Serialize)]

pub enum Instruction {
    Buffer,
    Bindless(CSlice<'static, NodeRef>),
    Texture,
    Shared,
    Local {
        init: NodeRef,
    },
    UserData(UserNodeDataRef),
    Invalid,
    Const(Const),

    // a variable that can be assigned to
    // similar to LLVM's alloca
    Update {
        var: NodeRef,
        value: NodeRef,
    },

    Call(Func, CBoxedSlice<NodeRef>),
    CpuCustomOp(CRc<CpuCustomOp>, NodeRef),
    Phi(CBoxedSlice<PhiIncoming>),
    /* represent a loop if the form of
    loop {
        body();
        if (cond) {
            break;
        }
    }
    */
    Loop {
        body: &'static BasicBlock,
        cond: NodeRef,
    },
    Break,
    Continue,
    If {
        cond: NodeRef,
        true_branch: &'static BasicBlock,
        false_branch: &'static BasicBlock,
    },
    Switch {
        value: NodeRef,
        default: &'static BasicBlock,
        cases: CBoxedSlice<SwitchCase>,
    },
}

// pub fn new_user_node<T: UserNodeData>(data: T) -> NodeRef {
//     new_node(Node::new(
//         Instruction::UserData(Rc::new(data)),
//         false,
//         &VOID_TYPE,
//     ))
// }
const INVALID_INST: Instruction = Instruction::Invalid;
pub(crate) fn with_node_pool<T>(f: impl FnOnce(&mut [*mut Node]) -> T) -> T {
    with_context(|ctx| f(&mut ctx.nodes))
}
pub(crate) fn new_node(node: Node) -> NodeRef {
    with_context(|ctx| ctx.alloc_node(node))
}
pub trait UserNodeData: Any + Debug {
    fn equal(&self, other: &dyn UserNodeData) -> bool;
    fn as_any(&self) -> &dyn Any;
}
#[derive(Debug)]
#[repr(C)]
pub struct BasicBlock {
    pub(crate) first: NodeRef,
    pub(crate) last: NodeRef,
}
#[derive(Serialize)]
struct NodeRefAndNode {
    id: NodeRef,
    data: &'static Node,
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
impl BasicBlock {
    pub(crate) fn nodes(&self) -> Vec<NodeRef> {
        let mut vec = Vec::new();
        let mut cur = self.first.get().next;

        while cur != self.last {
            vec.push(cur);
            cur = cur.get().next;
        }
        vec
    }
    pub(crate) fn into_vec(self) -> Vec<NodeRef> {
        let mut vec = Vec::new();
        let mut cur = self.first.get().next;
        while cur != self.last {
            vec.push(cur.clone());
            cur = cur.get().next;
            cur.update(|node| {
                node.prev = INVALID_REF;
                node.next = INVALID_REF;
            });
        }
        vec
    }
    pub(crate) fn new() -> Self {
        let first = new_node(Node::new(&Instruction::Invalid, &VOID_TYPE));
        let last = new_node(Node::new(&Instruction::Invalid, &VOID_TYPE));
        first.update(|node| node.next = last);
        last.update(|node| node.prev = first);
        Self { first, last }
    }
    pub(crate) fn push(&mut self, node: NodeRef) {
        if self.last.valid() {
            self.last.update(|last| {
                last.next = node.clone();
            });
            node.update(|node| {
                node.prev = self.last;
            });
        } else {
            self.first = node;
        }
        self.last = node;
    }

    pub(crate) fn is_empty(&self) -> bool {
        !self.first.valid()
    }
    pub(crate) fn len(&self) -> usize {
        let mut len = 0;
        let mut cur = self.first.get().next;
        while cur != self.last {
            len += 1;
            cur = cur.get().next;
        }
        len
    }
}
pub static VOID_TYPE: Type = Type::Void;

impl NodeRef {
    pub fn get_i32(&self) -> i32 {
        match self.get().instruction {
            Instruction::Const(c) => c.get_i32(),
            _ => panic!("not i32 node"),
        }
    }
    pub fn get(&self) -> &'static Node {
        assert!(self.valid());
        with_node_pool(|pool| unsafe { std::mem::transmute(pool[self.0]) })
    }
    pub fn valid(&self) -> bool {
        self.0 != INVALID_REF.0
    }
    pub fn set(&self, node: Node) {
        assert!(self.valid());
        with_node_pool(|pool| unsafe { *pool[self.0] = node });
    }
    pub fn update<T>(&self, f: impl FnOnce(&mut Node) -> T) -> T {
        assert!(self.valid());
        with_node_pool(|pool| unsafe { f(&mut *pool[self.0]) })
    }
    pub fn type_(&self) -> &'static Type {
        assert!(self.valid());
        with_node_pool(|pool| unsafe { (*pool[self.0]).type_ })
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
    pub entry: &'static BasicBlock,
}
impl Module {
    pub fn from_fragment(entry: &'static BasicBlock) -> Self {
        Self {
            kind: ModuleKind::Block,
            entry,
        }
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
        let new_node = match node.get().instruction {
            Instruction::Buffer => node,
            Instruction::Bindless(_) => node,
            Instruction::Texture => node,
            Instruction::Shared => node,
            Instruction::Local { .. } => todo!(),
            Instruction::UserData(_) => node,
            Instruction::Invalid => node,
            Instruction::Const(_) => todo!(),
            Instruction::Update { var, value } => todo!(),
            Instruction::Call(_, _) => todo!(),
            Instruction::CpuCustomOp(_, _) => node,
            Instruction::Phi(_) => todo!(),
            Instruction::Loop { body, cond } => todo!(),
            Instruction::Break => builder.break_(),
            Instruction::Continue => builder.continue_(),
            Instruction::If { .. } => todo!(),
            Instruction::Switch { .. } => todo!(),
        };
        self.node_map.insert(node, new_node);
        new_node
    }
    fn clone_block(
        &mut self,
        block: &'static BasicBlock,
        mut builder: IrBuilder,
    ) -> &'static BasicBlock {
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
    bb: &'static mut BasicBlock,
    insert_point: NodeRef,
}
impl IrBuilder {
    pub fn new() -> Self {
        let bb = context::alloc_deferred_drop(BasicBlock::new());
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
    pub fn break_(&mut self) -> NodeRef {
        let new_node = new_node(Node::new(
            context::alloc_deferred_drop(Instruction::Break),
            &VOID_TYPE,
        ));
        self.append(new_node);
        new_node
    }
    pub fn continue_(&mut self) -> NodeRef {
        let new_node = new_node(Node::new(
            context::alloc_deferred_drop(Instruction::Continue),
            &VOID_TYPE,
        ));
        self.append(new_node);
        new_node
    }
    pub fn zero_initializer(&mut self, ty: &'static Type) -> NodeRef {
        self.call(Func::ZeroInitializer, &[], ty)
    }
    pub fn requires_gradient(&mut self, node: NodeRef) -> NodeRef {
        self.call(Func::RequiresGradient, &[node], &VOID_TYPE)
    }
    pub fn gradient(&mut self, node: NodeRef) -> NodeRef {
        self.call(Func::Gradient, &[node], node.type_())
    }
    pub fn clone_node(&mut self, node: NodeRef) -> NodeRef {
        let node = node.get();
        let new_node = new_node(Node::new(
            context::alloc_deferred_drop(node.instruction.clone()),
            node.type_,
        ));
        self.append(new_node);
        new_node
    }
    pub fn const_(&mut self, const_: Const) -> NodeRef {
        let t = const_.type_();
        let node = Node::new(context::alloc_deferred_drop(Instruction::Const(const_)), t);
        let node = new_node(node);
        self.append(node.clone());
        node
    }
    pub fn local_zero_init(&mut self, ty: &'static Type) -> NodeRef {
        let node = self.zero_initializer(ty);
        let local = self.local(node);
        local
    }
    pub fn local(&mut self, init: NodeRef) -> NodeRef {
        let t = init.type_();
        let node = Node::new(context::alloc_deferred_drop(Instruction::Local { init }), t);
        let node = new_node(node);
        self.append(node.clone());
        node
    }
    pub fn call(&mut self, func: Func, args: &[NodeRef], ret_type: &'static Type) -> NodeRef {
        let node = Node::new(
            context::alloc_deferred_drop(Instruction::Call(func, CBoxedSlice::new(args.to_vec()))),
            ret_type,
        );
        let node = new_node(node);
        self.append(node.clone());
        node
    }
    pub fn cast(&mut self, node: NodeRef, t: &'static Type) -> NodeRef {
        self.call(Func::Cast, &[node], t)
    }
    pub fn bitcast(&mut self, node: NodeRef, t: &'static Type) -> NodeRef {
        self.call(Func::Bitcast, &[node], t)
    }
    pub fn update(&mut self, var: NodeRef, value: NodeRef) {
        match var.get().instruction {
            Instruction::Local { .. } => {}
            Instruction::Call(func, _) => match func {
                Func::GetElementPtr => {}
                _ => panic!("not local or getelementptr"),
            },
            _ => panic!("not a var"),
        }
        let node = Node::new(
            context::alloc_deferred_drop(Instruction::Update { var, value }),
            &VOID_TYPE,
        );
        let node = new_node(node);
        self.append(node);
    }
    pub fn phi(&mut self, incoming: &[PhiIncoming], t: &'static Type) -> NodeRef {
        let node = Node::new(
            context::alloc_deferred_drop(Instruction::Phi(CBoxedSlice::new(incoming.to_vec()))),
            t,
        );
        let node = new_node(node);
        self.append(node.clone());
        node
    }
    pub fn if_(
        &mut self,
        cond: NodeRef,
        true_branch: &'static BasicBlock,
        false_branch: &'static BasicBlock,
    ) -> NodeRef {
        let node = Node::new(
            context::alloc_deferred_drop(Instruction::If {
                cond,
                true_branch,
                false_branch,
            }),
            &VOID_TYPE,
        );
        let node = new_node(node);
        self.append(node);
        node
    }
    pub fn loop_(&mut self, body: &'static BasicBlock, cond: NodeRef) -> NodeRef {
        let node = Node::new(
            context::alloc_deferred_drop(Instruction::Loop { body, cond }),
            &VOID_TYPE,
        );
        let node = new_node(node);
        self.append(node);
        node
    }
    pub fn finish(self) -> &'static BasicBlock {
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
pub extern "C" fn luisa_compute_ir_build_call(
    builder: &mut IrBuilder,
    func: Func,
    args: CSlice<NodeRef>,
    ret_type: &'static Type,
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
    ty: &'static Type,
) -> NodeRef {
    builder.local_zero_init(ty)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_builder() -> IrBuilder {
    IrBuilder::new()
}
#[no_mangle]
pub extern "C" fn luisa_compute_ir_build_finish(builder: IrBuilder) -> &'static BasicBlock {
    builder.finish()
}

pub mod debug {
    use std::ffi::CString;

    use super::*;

    pub fn dump_ir(module: &Module) -> serde_json::Value {
        serde_json::to_value(&module).unwrap()
    }
    #[no_mangle]
    pub extern "C" fn luisa_compute_ir_dump(module: &Module) -> CBoxedSlice<u8> {
        let json = dump_ir(module);
        let s = serde_json::to_string(&json).unwrap();
        let cstring = CString::new(s).unwrap();
        CBoxedSlice::new(cstring.as_bytes().to_vec())
    }
}
