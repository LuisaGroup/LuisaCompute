use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    hash::Hash,
};

use lazy_static::lazy_static;
use parking_lot::RwLock;

use crate::{
    context,
    ir::{
        BasicBlock, Func, IrBuilder, MatrixType, Module, ModuleKind, Node, NodeRef, StructType,
        Type, VectorElementType, VectorType, VOID_TYPE,
    },
    CSlice, TypeOf,
};

use super::Transform;
// Simple backward autodiff
// Loop is not supported since users would use path replay[https://rgl.epfl.ch/publications/Vicini2021PathReplay] anyway

lazy_static! {
    static ref GRAD_TYPES: RwLock<HashMap<&'static Type, Option<&'static Type>>> =
        RwLock::new(HashMap::new());
}
fn grad_type_of_ve(t: &'static VectorElementType) -> Option<VectorElementType> {
    match t {
        VectorElementType::Scalar(p) => {
            let p = grad_type_of(context::register_type(Type::Primitive(*p)))?;
            Some(VectorElementType::Scalar(match p {
                Type::Primitive(p) => *p,
                _ => unreachable!(),
            }))
        }
        VectorElementType::Vector(v) => {
            let v = grad_type_of(context::register_type(Type::Vector((**v).clone())))?;
            Some(VectorElementType::Vector(match v {
                Type::Vector(vt) => vt,
                _ => unreachable!(),
            }))
        }
    }
}
fn grad_type_of(type_: &'static Type) -> Option<&'static Type> {
    if let Some(t) = GRAD_TYPES.read().get(type_) {
        return *t;
    }
    let t = _grad_type_of(type_);
    GRAD_TYPES.write().insert(type_, t);
    t
}
fn _grad_type_of(type_: &'static Type) -> Option<&'static Type> {
    let ty = match type_ {
        Type::Void => None,
        Type::Primitive(p) => match p {
            crate::ir::Primitive::Bool => None,
            crate::ir::Primitive::Int32 => None,
            crate::ir::Primitive::Uint32 => None,
            crate::ir::Primitive::Int64 => None,
            crate::ir::Primitive::Uint64 => None,
            crate::ir::Primitive::Float32 => Some(Type::Primitive(crate::ir::Primitive::Float32)),
            crate::ir::Primitive::Float64 => Some(Type::Primitive(crate::ir::Primitive::Float64)),
        },
        Type::Vector(v) => {
            let ve = grad_type_of_ve(&v.element)?;
            Some(Type::Vector(VectorType {
                element: ve,
                length: v.length,
            }))
        }
        Type::Matrix(m) => {
            let ve = grad_type_of_ve(&m.element)?;
            Some(Type::Matrix(MatrixType {
                element: ve,
                dimension: m.dimension,
            }))
        }
        Type::Struct(st) => {
            let fields: Vec<&'static Type> = st
                .fields
                .as_ref()
                .iter()
                .map(|f| grad_type_of(f))
                .filter(|f| f.is_some())
                .map(|f| f.unwrap())
                .collect();
            let alignment = fields.iter().map(|f| f.alignment()).max().unwrap();
            let size = fields.iter().map(|f| f.size()).sum();
            let fields: &[_] = Vec::leak(fields.to_vec());
            Some(Type::Struct(StructType {
                fields: CSlice::from(fields),
                alignment,
                size,
            }))
        }
    };
    let ty = ty.map(|ty| context::register_type(ty));
    ty
}

struct StoreIntermediate<'a> {
    // map from node to its intermediate
    map: HashMap<NodeRef, NodeRef>,
    requires_grad: HashSet<NodeRef>,
    grads: HashMap<NodeRef, NodeRef>,
    final_grad: HashSet<NodeRef>,
    module: &'a Module,
    builder: IrBuilder,
}
impl<'a> StoreIntermediate<'a> {
    fn new(module: &'a Module) -> Self {
        let mut builder = IrBuilder::new();
        builder.set_insert_point(module.entry.first);
        Self {
            map: HashMap::new(),
            grads: HashMap::new(),
            module,
            requires_grad: HashSet::new(),
            builder,
            final_grad: HashSet::new(),
        }
    }
    fn visit_block(&mut self, block: &BasicBlock) {
        for node in block.nodes() {
            self.visit(node);
        }
    }
    fn create_intermediate(&mut self, node: NodeRef) {
        let local = self.builder.local(node);
        self.map.insert(node, local);
    }
    fn add_grad(&mut self, node: NodeRef) {
        if self.grads.contains_key(&node) {
            return;
        }
        let grad_type = grad_type_of(node.type_()).unwrap();
        let grad = self.builder.local_zero_init(grad_type);
        self.grads.insert(node, grad);
    }
    fn visit(&mut self, node: NodeRef) {
        if self.map.contains_key(&node) {
            return;
        }
        let instruction = node.get().instruction;
        let type_ = node.get().type_;
        let grad_type = grad_type_of(type_);
        match instruction {
            crate::ir::Instruction::Buffer => {}
            crate::ir::Instruction::Bindless(_) => {}
            crate::ir::Instruction::Texture => {}
            crate::ir::Instruction::Shared => {}
            crate::ir::Instruction::Local { .. } => {}
            crate::ir::Instruction::UserData(_) => {}
            crate::ir::Instruction::Invalid => {}
            crate::ir::Instruction::Const(_) => {
                if self.requires_grad.contains(&node) && grad_type.is_some() {
                    self.create_intermediate(node);
                }
            }
            crate::ir::Instruction::Update { .. } => {
                panic!("should not have update here, call ToSSA before autodiff");
            }
            crate::ir::Instruction::Call(func, args) => {
                if *func == Func::RequiresGradient {
                    self.requires_grad.insert(args.as_ref()[0]);
                    self.add_grad(args.as_ref()[0]);
                }
                if *func == Func::Gradient {
                    self.final_grad.insert(args.as_ref()[0]);
                    return;
                }
                if self.requires_grad.contains(&node) && grad_type.is_some() {
                    self.create_intermediate(node);
                    self.add_grad(node);
                    for arg in args.as_ref() {
                        self.requires_grad.insert(*arg);
                    }
                    for arg in args.as_ref() {
                        self.visit(*arg);
                    }
                }
            }
            crate::ir::Instruction::CpuCustomOp(_, _) => {
                panic!("cpu custom op not supported yet");
            }
            crate::ir::Instruction::Phi(_) => todo!(),
            crate::ir::Instruction::Loop { body, cond: _ } => {
                self.visit_block(body);
            }
            crate::ir::Instruction::Break => {}
            crate::ir::Instruction::Continue => {}
            crate::ir::Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                self.create_intermediate(*cond);
                self.visit_block(true_branch);
                self.visit_block(false_branch);
            }
        }
    }
}
struct Backward {
    grads: HashMap<NodeRef, NodeRef>,
    intermediate: HashMap<NodeRef, NodeRef>,
    final_grad: HashSet<NodeRef>,
}
impl Backward {
    fn grad(&mut self, node: NodeRef) -> Option<NodeRef> {
        self.grads.get(&node).copied()
    }
    fn add_grad(&mut self, node: NodeRef, grad: NodeRef, builder: &mut IrBuilder) {
        let grad_var = self.grad(node).unwrap();
        let new_grad = builder.call(Func::Add, &[grad_var, grad], grad_var.type_());
        builder.update(grad_var, new_grad);
    }
    fn backward(&mut self, node: NodeRef, builder: &mut IrBuilder) {
        let instruction = node.get().instruction;
        let type_ = node.get().type_;
        let grad_type = grad_type_of(type_);
        if grad_type.is_none() {
            return;
        }
        let grad_type = grad_type.unwrap();
        match instruction {
            crate::ir::Instruction::Buffer => {}
            crate::ir::Instruction::Bindless(_) => {}
            crate::ir::Instruction::Texture => {}
            crate::ir::Instruction::Shared => {}
            crate::ir::Instruction::Local { .. } => {}
            crate::ir::Instruction::UserData(_) => {}
            crate::ir::Instruction::Invalid => {}
            crate::ir::Instruction::Const(_) => {}
            crate::ir::Instruction::Update { .. } => todo!(),
            crate::ir::Instruction::Call(func, args) => {
                let args = args
                    .as_ref()
                    .iter()
                    .map(|a| self.intermediate.get(a).cloned().unwrap())
                    .collect::<Vec<_>>();
                let out_grad = self.grad(node).unwrap();
                match func {
                    Func::Add => {
                        self.add_grad(args[0], out_grad, builder);
                        self.add_grad(args[1], out_grad, builder);
                    }
                    Func::Sub => {
                        self.add_grad(args[0], out_grad, builder);
                        let neg_out_grad = builder.call(Func::Neg, &[out_grad], out_grad.type_());
                        self.add_grad(args[1], neg_out_grad, builder);
                    }
                    Func::Mul => {
                        let lhs_grad =
                            builder.call(Func::Mul, &[out_grad, args[1]], out_grad.type_());
                        self.add_grad(args[0], lhs_grad, builder);
                        let rhs_grad =
                            builder.call(Func::Mul, &[out_grad, args[0]], out_grad.type_());
                        self.add_grad(args[1], rhs_grad, builder);
                    }
                    _ => todo!(),
                }
            }
            crate::ir::Instruction::CpuCustomOp(_, _) => todo!(),
            crate::ir::Instruction::Phi(_) => todo!(),
            crate::ir::Instruction::Loop { .. } => todo!(),
            crate::ir::Instruction::Break => todo!(),
            crate::ir::Instruction::Continue => todo!(),
            crate::ir::Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => todo!(),
        }
    }
    fn backward_block(
        &mut self,
        block: &BasicBlock,
        mut builder: IrBuilder,
    ) -> &'static BasicBlock {
        for node in block.nodes().iter().rev() {
            self.backward(*node, &mut builder);
        }
        for v in &self.final_grad {
            let grad = self.grads.get(v).unwrap();
            builder.call(Func::GradientMarker, &[*v, *grad], &VOID_TYPE);
        }
        builder.finish()
    }
}
pub struct Autodiff;
impl Transform for Autodiff {
    fn transform(&self, module: crate::ir::Module) -> crate::ir::Module {
        assert!(
            module.kind == crate::ir::ModuleKind::Block,
            "autodiff should be applied to a block"
        );
        let mut store = StoreIntermediate::new(&module);
        store.visit_block(module.entry);
        let StoreIntermediate {
            grads,
            final_grad,
            map: intermediate,
            ..
        } = store;
        let mut backward = Backward {
            grads,
            final_grad,
            intermediate,
        };
        let bb = backward.backward_block(module.entry, IrBuilder::new());
        Module {
            kind: ModuleKind::Block,
            entry: bb,
        }
    }
}
