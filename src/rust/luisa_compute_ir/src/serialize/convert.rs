use std::collections::HashMap;

use crate::{
    ir::{
        BasicBlock, Capture, Const, Func, Instruction, KernelModule, Node, NodeRef, PhiIncoming,
        SwitchCase, Type,
    },
    CArc, Pooled,
};

use super::{
    SerializedBlock, SerializedBlockRef, SerializedCapture, SerializedConst, SerializedFunc,
    SerializedInstruction, SerializedKernelModule, SerializedNode, SerializedNodeRef,
    SerializedPhiIncoming, SerializedSwitchCase, SerializedType, SerializedTypeRef,
};

struct KernelSerializer {
    type_to_id: HashMap<*const Type, SerializedTypeRef>,
    types: Vec<SerializedType>,
    block_to_id: HashMap<*const BasicBlock, SerializedBlockRef>,
    blocks: Vec<SerializedBlock>,
    node_to_id: HashMap<NodeRef, SerializedNodeRef>,
    nodes: Vec<SerializedNode>,
}

impl KernelSerializer {
    fn serialize_type_inner(&mut self, ty: &CArc<Type>) -> SerializedType {
        match ty.as_ref() {
            Type::Void => SerializedType::Void,
            Type::UserData => SerializedType::Void,
            Type::Primitive(p) => SerializedType::Primitive(*p),
            Type::Vector(v) => SerializedType::Vector(v.element.as_primitive().unwrap(), v.length),
            Type::Matrix(v) => {
                SerializedType::Vector(v.element.as_primitive().unwrap(), v.dimension)
            }
            Type::Struct(v) => {
                let mut fields = vec![];
                for field in v.fields.as_ref() {
                    fields.push(self.serialize_type(&field));
                }
                SerializedType::Struct {
                    fields,
                    align: v.alignment as u32,
                    size: v.size as u32,
                }
            }
            Type::Array(v) => {
                SerializedType::Array(self.serialize_type(&v.element), v.length as u32)
            }
            Type::Opaque(name) => SerializedType::Opqaue(name.to_string()),
        }
    }
    fn serialize_type(&mut self, ty: &CArc<Type>) -> SerializedTypeRef {
        let ptr = CArc::as_ptr(ty);
        if let Some(id) = self.type_to_id.get(&ptr) {
            *id
        } else {
            let id = self.types.len();
            self.type_to_id.insert(ptr, SerializedTypeRef(id as u64));
            let inner = self.serialize_type_inner(ty);
            self.types.push(inner);
            SerializedTypeRef(id as u64)
        }
    }
    fn serialize_block(&mut self, block: &Pooled<BasicBlock>) -> SerializedBlockRef {
        let ptr = block.ptr as *const BasicBlock;
        if let Some(id) = self.block_to_id.get(&ptr) {
            *id
        } else {
            let id = self.blocks.len();
            self.block_to_id.insert(ptr, SerializedBlockRef(id as u64));
            let inner = self.serialize_block_inner(block.get());
            self.blocks.push(inner);
            SerializedBlockRef(id as u64)
        }
    }
    fn serialize_block_inner(&mut self, block: &BasicBlock) -> SerializedBlock {
        let nodes = block
            .nodes()
            .iter()
            .map(|n| self.serialize_node(n.get()))
            .collect::<Vec<_>>();
        SerializedBlock { nodes }
    }
    fn serialize_node(&mut self, node: &Node) -> SerializedNode {
        let ty = self.serialize_type(&node.type_);
        let inst = self.serialize_instruction(&node.instruction);
        SerializedNode { ty, inst }
    }
    fn serialize_capture(&mut self, capture: &Capture) -> SerializedCapture {
        let node = self.serialize_noderef(capture.node);
        SerializedCapture {
            node,
            binding: capture.binding,
        }
    }
    fn serialize_noderef(&mut self, node: NodeRef) -> SerializedNodeRef {
        if let Some(id) = self.node_to_id.get(&node) {
            *id
        } else {
            let id = self.nodes.len();
            self.node_to_id.insert(node, SerializedNodeRef(id as u64));
            let inner = self.serialize_node(&node.get());
            self.nodes.push(inner);
            SerializedNodeRef(id as u64)
        }
    }
    fn serialize_const(&mut self, cst: &Const) -> SerializedConst {
        match cst {
            Const::Zero(t) => SerializedConst::Zero(self.serialize_type(t)),
            Const::One(t) => SerializedConst::One(self.serialize_type(t)),
            Const::Bool(v) => SerializedConst::Bool(*v),
            Const::Int8(v) => SerializedConst::Int8(*v),
            Const::Uint8(v) => SerializedConst::Uint8(*v),
            Const::Int16(v) => SerializedConst::Int16(*v),
            Const::Uint16(v) => SerializedConst::Uint16(*v),
            Const::Int32(v) => SerializedConst::Int32(*v),
            Const::Uint32(v) => SerializedConst::Uint32(*v),
            Const::Int64(v) => SerializedConst::Int64(*v),
            Const::Uint64(v) => SerializedConst::Uint64(*v),
            Const::Float16(v) => SerializedConst::Float16(*v),
            Const::Float32(v) => SerializedConst::Float32(*v),
            Const::Float64(v) => SerializedConst::Float64(*v),
            Const::Generic(v, t) => {
                let t = self.serialize_type(t);
                let v = v.as_ref().to_vec();
                SerializedConst::Generic(v, t)
            }
        }
    }
    fn serialize_instruction(&mut self, inst: &Instruction) -> SerializedInstruction {
        match inst {
            Instruction::Buffer => SerializedInstruction::Buffer,
            Instruction::Bindless => SerializedInstruction::Bindless,
            Instruction::Texture2D => SerializedInstruction::Texture2D,
            Instruction::Texture3D => SerializedInstruction::Texture3D,
            Instruction::Accel => SerializedInstruction::Accel,
            Instruction::Shared => SerializedInstruction::Shared,
            Instruction::Uniform => SerializedInstruction::Uniform,
            Instruction::Local { init } => SerializedInstruction::Local {
                init: self.serialize_noderef(*init),
            },
            Instruction::Argument { by_value } => SerializedInstruction::Argument {
                by_value: *by_value,
            },
            Instruction::UserData(_) => SerializedInstruction::UserData,
            Instruction::Invalid => SerializedInstruction::Invalid,
            Instruction::Const(c) => SerializedInstruction::Const(self.serialize_const(c)),
            Instruction::Update { var, value } => SerializedInstruction::Update {
                var: self.serialize_noderef(*var),
                value: self.serialize_noderef(*value),
            },
            Instruction::Call(f, args) => {
                let f = self.serialize_func(f);
                let args = args
                    .as_ref()
                    .iter()
                    .map(|arg| self.serialize_noderef(*arg))
                    .collect();
                SerializedInstruction::Call(f, args)
            }
            Instruction::Phi(incomings) => {
                let incomings = incomings
                    .as_ref()
                    .iter()
                    .map(|PhiIncoming { value, block }| SerializedPhiIncoming {
                        value: self.serialize_noderef(*value),
                        block: self.serialize_block(block),
                    })
                    .collect();
                SerializedInstruction::Phi(incomings)
            }
            Instruction::Return(v) => {
                let v = self.serialize_noderef(*v);
                SerializedInstruction::Return(v)
            }
            Instruction::Loop { body, cond } => {
                let body = self.serialize_block(body);
                let cond = self.serialize_noderef(*cond);
                SerializedInstruction::Loop { body, cond }
            }
            Instruction::GenericLoop {
                prepare,
                cond,
                body,
                update,
            } => {
                let prepare = self.serialize_block(prepare);
                let cond = self.serialize_noderef(*cond);
                let body = self.serialize_block(body);
                let update = self.serialize_block(update);
                SerializedInstruction::GenericLoop {
                    prepare,
                    cond,
                    body,
                    update,
                }
            }
            Instruction::Break => SerializedInstruction::Break,
            Instruction::Continue => SerializedInstruction::Continue,
            Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                let cond = self.serialize_noderef(*cond);
                let true_branch = self.serialize_block(true_branch);
                let false_branch = self.serialize_block(false_branch);
                SerializedInstruction::If {
                    cond,
                    true_branch,
                    false_branch,
                }
            }
            Instruction::Switch {
                value,
                default,
                cases,
            } => {
                let value = self.serialize_noderef(*value);
                let default = self.serialize_block(default);
                let cases = cases
                    .as_ref()
                    .iter()
                    .map(|SwitchCase { value, block }| SerializedSwitchCase {
                        value: *value,
                        block: self.serialize_block(block),
                    })
                    .collect();
                SerializedInstruction::Switch {
                    value,
                    default,
                    cases,
                }
            }
            Instruction::AdScope {
                body,
                forward,
                n_forward_grads: _,
            } => {
                let body = self.serialize_block(body);
                SerializedInstruction::AdScope {
                    body,
                    forward: *forward,
                }
            }
            Instruction::AdDetach(b) => {
                let b = self.serialize_block(b);
                SerializedInstruction::AdDetach(b)
            }
            Instruction::RayQuery {
                ray_query,
                on_triangle_hit,
                on_procedural_hit,
            } => {
                let ray_query = self.serialize_noderef(*ray_query);
                let on_triangle_hit = self.serialize_block(on_triangle_hit);
                let on_procedural_hit = self.serialize_block(on_procedural_hit);
                SerializedInstruction::RayQuery {
                    ray_query,
                    on_triangle_hit,
                    on_procedural_hit,
                }
            }
            Instruction::Comment(s) => {
                let s = s.as_ref().to_vec();
                SerializedInstruction::Comment(s)
            }
            _ => todo!(),
        }
    }
    fn serialize_func(&mut self, func: &Func) -> SerializedFunc {
        match func {
            Func::ZeroInitializer => SerializedFunc::ZeroInitializer,
            Func::Assume => SerializedFunc::Assume,
            Func::Unreachable(msg) => SerializedFunc::Unreachable(msg.to_vec()),
            Func::Assert(msg) => SerializedFunc::Assert(msg.to_vec()),
            Func::ThreadId => SerializedFunc::ThreadId,
            Func::BlockId => SerializedFunc::BlockId,
            Func::DispatchId => SerializedFunc::DispatchId,
            Func::DispatchSize => SerializedFunc::DispatchSize,
            Func::Backward => SerializedFunc::Backward,
            Func::RequiresGradient => SerializedFunc::RequiresGradient,
            Func::Gradient => SerializedFunc::Gradient,
            Func::GradientMarker => SerializedFunc::GradientMarker,
            Func::AccGrad => SerializedFunc::AccGrad,
            Func::Detach => SerializedFunc::Detach,
            Func::RayTracingInstanceTransform => SerializedFunc::RayTracingInstanceTransform,
            Func::RayTracingSetInstanceTransform => SerializedFunc::RayTracingSetInstanceTransform,
            Func::RayTracingSetInstanceOpacity => SerializedFunc::RayTracingSetInstanceOpacity,
            Func::RayTracingSetInstanceVisibility => {
                SerializedFunc::RayTracingSetInstanceVisibility
            }
            Func::RayTracingTraceClosest => SerializedFunc::RayTracingTraceClosest,
            Func::RayTracingTraceAny => SerializedFunc::RayTracingTraceAny,
            Func::RayTracingQueryAll => SerializedFunc::RayTracingQueryAll,
            Func::RayTracingQueryAny => SerializedFunc::RayTracingQueryAny,
            Func::RayQueryWorldSpaceRay => SerializedFunc::RayQueryWorldSpaceRay,
            Func::RayQueryProceduralCandidateHit => SerializedFunc::RayQueryProceduralCandidateHit,
            Func::RayQueryTriangleCandidateHit => SerializedFunc::RayQueryTriangleCandidateHit,
            Func::RayQueryCommittedHit => SerializedFunc::RayQueryCommittedHit,
            Func::RayQueryCommitTriangle => SerializedFunc::RayQueryCommitTriangle,
            Func::RayQueryCommitProcedural => SerializedFunc::RayQueryCommitProcedural,
            Func::RayQueryTerminate => SerializedFunc::RayQueryTerminate,
            Func::RasterDiscard => SerializedFunc::RasterDiscard,
            Func::IndirectDispatchSetCount => SerializedFunc::IndirectDispatchSetCount,
            Func::IndirectDispatchSetKernel => SerializedFunc::IndirectDispatchSetKernel,
            Func::Load => SerializedFunc::Load,
            Func::Cast => SerializedFunc::Cast,
            Func::Bitcast => SerializedFunc::Bitcast,
            Func::Pack => SerializedFunc::Pack,
            Func::Unpack => SerializedFunc::Unpack,
            Func::Add => SerializedFunc::Add,
            Func::Sub => SerializedFunc::Sub,
            Func::Mul => SerializedFunc::Mul,
            Func::Div => SerializedFunc::Div,
            Func::Rem => SerializedFunc::Rem,
            Func::BitAnd => SerializedFunc::BitAnd,
            Func::BitOr => SerializedFunc::BitOr,
            Func::BitXor => SerializedFunc::BitXor,
            Func::Shl => SerializedFunc::Shl,
            Func::Shr => SerializedFunc::Shr,
            Func::RotRight => SerializedFunc::RotRight,
            Func::RotLeft => SerializedFunc::RotLeft,
            Func::Eq => SerializedFunc::Eq,
            Func::Ne => SerializedFunc::Ne,
            Func::Lt => SerializedFunc::Lt,
            Func::Le => SerializedFunc::Le,
            Func::Gt => SerializedFunc::Gt,
            Func::Ge => SerializedFunc::Ge,
            Func::MatCompMul => SerializedFunc::MatCompMul,
            Func::Neg => SerializedFunc::Neg,
            Func::Not => SerializedFunc::Not,
            Func::BitNot => SerializedFunc::BitNot,
            Func::All => SerializedFunc::All,
            Func::Any => SerializedFunc::Any,
            Func::Select => SerializedFunc::Select,
            Func::Clamp => SerializedFunc::Clamp,
            Func::Lerp => SerializedFunc::Lerp,
            Func::Step => SerializedFunc::Step,
            Func::SmoothStep => SerializedFunc::SmoothStep,
            Func::Saturate => SerializedFunc::Saturate,
            Func::Abs => SerializedFunc::Abs,
            Func::Min => SerializedFunc::Min,
            Func::Max => SerializedFunc::Max,
            Func::ReduceSum => SerializedFunc::ReduceSum,
            Func::ReduceProd => SerializedFunc::ReduceProd,
            Func::ReduceMin => SerializedFunc::ReduceMin,
            Func::ReduceMax => SerializedFunc::ReduceMax,
            Func::Clz => SerializedFunc::Clz,
            Func::Ctz => SerializedFunc::Ctz,
            Func::PopCount => SerializedFunc::PopCount,
            Func::Reverse => SerializedFunc::Reverse,
            Func::IsInf => SerializedFunc::IsInf,
            Func::IsNan => SerializedFunc::IsNan,
            Func::Acos => SerializedFunc::Acos,
            Func::Acosh => SerializedFunc::Acosh,
            Func::Asin => SerializedFunc::Asin,
            Func::Asinh => SerializedFunc::Asinh,
            Func::Atan => SerializedFunc::Atan,
            Func::Atan2 => SerializedFunc::Atan2,
            Func::Atanh => SerializedFunc::Atanh,
            Func::Cos => SerializedFunc::Cos,
            Func::Cosh => SerializedFunc::Cosh,
            Func::Sin => SerializedFunc::Sin,
            Func::Sinh => SerializedFunc::Sinh,
            Func::Tan => SerializedFunc::Tan,
            Func::Tanh => SerializedFunc::Tanh,
            Func::Exp => SerializedFunc::Exp,
            Func::Exp2 => SerializedFunc::Exp2,
            Func::Exp10 => SerializedFunc::Exp10,
            Func::Log => SerializedFunc::Log,
            Func::Log2 => SerializedFunc::Log2,
            Func::Log10 => SerializedFunc::Log10,
            Func::Powi => SerializedFunc::Powi,
            Func::Powf => SerializedFunc::Powf,
            Func::Sqrt => SerializedFunc::Sqrt,
            Func::Rsqrt => SerializedFunc::Rsqrt,
            Func::Ceil => SerializedFunc::Ceil,
            Func::Floor => SerializedFunc::Floor,
            Func::Fract => SerializedFunc::Fract,
            Func::Trunc => SerializedFunc::Trunc,
            Func::Round => SerializedFunc::Round,
            Func::Fma => SerializedFunc::Fma,
            Func::Copysign => SerializedFunc::Copysign,
            Func::Cross => SerializedFunc::Cross,
            Func::Dot => SerializedFunc::Dot,
            Func::OuterProduct => SerializedFunc::OuterProduct,
            Func::Length => SerializedFunc::Length,
            Func::LengthSquared => SerializedFunc::LengthSquared,
            Func::Normalize => SerializedFunc::Normalize,
            Func::Faceforward => SerializedFunc::Faceforward,
            Func::Reflect => SerializedFunc::Reflect,
            Func::Determinant => SerializedFunc::Determinant,
            Func::Transpose => SerializedFunc::Transpose,
            Func::Inverse => SerializedFunc::Inverse,
            Func::SynchronizeBlock => SerializedFunc::SynchronizeBlock,
            Func::AtomicExchange => SerializedFunc::AtomicExchange,
            Func::AtomicCompareExchange => SerializedFunc::AtomicCompareExchange,
            Func::AtomicFetchAdd => SerializedFunc::AtomicFetchAdd,
            Func::AtomicFetchSub => SerializedFunc::AtomicFetchSub,
            Func::AtomicFetchAnd => SerializedFunc::AtomicFetchAnd,
            Func::AtomicFetchOr => SerializedFunc::AtomicFetchOr,
            Func::AtomicFetchXor => SerializedFunc::AtomicFetchXor,
            Func::AtomicFetchMin => SerializedFunc::AtomicFetchMin,
            Func::AtomicFetchMax => SerializedFunc::AtomicFetchMax,
            Func::BufferRead => SerializedFunc::BufferRead,
            Func::BufferWrite => SerializedFunc::BufferWrite,
            Func::BufferSize => SerializedFunc::BufferSize,
            Func::Texture2dRead => SerializedFunc::Texture2dRead,
            Func::Texture2dWrite => SerializedFunc::Texture2dWrite,
            Func::Texture3dRead => SerializedFunc::Texture3dRead,
            Func::Texture3dWrite => SerializedFunc::Texture3dWrite,
            Func::BindlessTexture2dSample => SerializedFunc::BindlessTexture2dSample,
            Func::BindlessTexture2dSampleLevel => SerializedFunc::BindlessTexture2dSampleLevel,
            Func::BindlessTexture2dSampleGrad => SerializedFunc::BindlessTexture2dSampleGrad,
            Func::BindlessTexture2dSampleGradLevel => {
                SerializedFunc::BindlessTexture2dSampleGradLevel
            }
            Func::BindlessTexture3dSample => SerializedFunc::BindlessTexture3dSample,
            Func::BindlessTexture3dSampleLevel => SerializedFunc::BindlessTexture3dSampleLevel,
            Func::BindlessTexture3dSampleGrad => SerializedFunc::BindlessTexture3dSampleGrad,
            Func::BindlessTexture3dSampleGradLevel => {
                SerializedFunc::BindlessTexture3dSampleGradLevel
            }
            Func::BindlessTexture2dRead => SerializedFunc::BindlessTexture2dRead,
            Func::BindlessTexture3dRead => SerializedFunc::BindlessTexture3dRead,
            Func::BindlessTexture2dReadLevel => SerializedFunc::BindlessTexture2dReadLevel,
            Func::BindlessTexture3dReadLevel => SerializedFunc::BindlessTexture3dReadLevel,
            Func::BindlessTexture2dSize => SerializedFunc::BindlessTexture2dSize,
            Func::BindlessTexture3dSize => SerializedFunc::BindlessTexture3dSize,
            Func::BindlessTexture2dSizeLevel => SerializedFunc::BindlessTexture2dSizeLevel,
            Func::BindlessTexture3dSizeLevel => SerializedFunc::BindlessTexture3dSizeLevel,
            Func::BindlessBufferRead => SerializedFunc::BindlessBufferRead,
            Func::BindlessBufferSize => SerializedFunc::BindlessBufferSize,
            Func::BindlessBufferType => SerializedFunc::BindlessBufferType,
            Func::Vec => SerializedFunc::Vec,
            Func::Vec2 => SerializedFunc::Vec2,
            Func::Vec3 => SerializedFunc::Vec3,
            Func::Vec4 => SerializedFunc::Vec4,
            Func::Permute => SerializedFunc::Permute,
            Func::InsertElement => SerializedFunc::InsertElement,
            Func::ExtractElement => SerializedFunc::ExtractElement,
            Func::GetElementPtr => SerializedFunc::GetElementPtr,
            Func::Struct => SerializedFunc::Struct,
            Func::Array => SerializedFunc::Array,
            Func::Mat => SerializedFunc::Mat,
            Func::Mat2 => SerializedFunc::Mat2,
            Func::Mat3 => SerializedFunc::Mat3,
            Func::Mat4 => SerializedFunc::Mat4,
            Func::Callable(_) => todo!(),
            Func::CpuCustomOp(_) => panic!("cpu custom op not serializable"),
            Func::WarpSize => SerializedFunc::WarpSize,
            Func::WarpLaneId => SerializedFunc::WarpLaneId,
            Func::WarpIsFirstActiveLane => SerializedFunc::WarpIsFirstActiveLane,
            Func::WarpFirstActiveLane => SerializedFunc::WarpFirstActiveLane,
            Func::WarpActiveAllEqual => SerializedFunc::WarpActiveAllEqual,
            Func::WarpActiveBitAnd => SerializedFunc::WarpActiveBitAnd,
            Func::WarpActiveBitOr => SerializedFunc::WarpActiveBitOr,
            Func::WarpActiveBitXor => SerializedFunc::WarpActiveBitXor,
            Func::WarpActiveCountBits => SerializedFunc::WarpActiveCountBits,
            Func::WarpActiveMax => SerializedFunc::WarpActiveMax,
            Func::WarpActiveMin => SerializedFunc::WarpActiveMin,
            Func::WarpActiveProduct => SerializedFunc::WarpActiveProduct,
            Func::WarpActiveSum => SerializedFunc::WarpActiveSum,
            Func::WarpActiveAll => SerializedFunc::WarpActiveAll,
            Func::WarpActiveAny => SerializedFunc::WarpActiveAny,
            Func::WarpActiveBitMask => SerializedFunc::WarpActiveBitMask,
            Func::WarpPrefixCountBits => SerializedFunc::WarpPrefixCountBits,
            Func::WarpPrefixSum => SerializedFunc::WarpPrefixSum,
            Func::WarpPrefixProduct => SerializedFunc::WarpPrefixProduct,
            Func::WarpReadLaneAt => SerializedFunc::WarpReadLaneAt,
            Func::WarpReadFirstLane => SerializedFunc::WarpReadFirstLane,
            Func::ShaderExecutionReorder => SerializedFunc::ShaderExecutionReorder,
            Func::Unknown0 => todo!(),
            Func::Unknown1 => todo!(),
            _ => todo!(),
        }
    }
    fn new() -> Self {
        Self {
            type_to_id: HashMap::new(),
            types: Vec::new(),
            block_to_id: HashMap::new(),
            blocks: Vec::new(),
            node_to_id: HashMap::new(),
            nodes: Vec::new(),
        }
    }
}

pub fn serialize_kernel_module(m: &KernelModule) -> SerializedKernelModule {
    let mut serializer = KernelSerializer::new();
    let args = m
        .args
        .as_ref()
        .iter()
        .map(|a| serializer.serialize_noderef(*a))
        .collect();
    let shared = m
        .shared
        .as_ref()
        .iter()
        .map(|a| serializer.serialize_noderef(*a))
        .collect();
    let captures = m
        .captures
        .as_ref()
        .iter()
        .map(|a| serializer.serialize_capture(a))
        .collect();
    let entry = serializer.serialize_block(&m.module.entry);

    SerializedKernelModule {
        entry,
        types: serializer.types,
        nodes: serializer.nodes,
        blocks: serializer.blocks,
        block_size: m.block_size,
        captures,
        args,
        shared,
    }
}
