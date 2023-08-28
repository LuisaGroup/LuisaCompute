use indexmap::{IndexMap, IndexSet};

use std::ops::Deref;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

use crate::context::is_type_equal;
use crate::ir::{new_node, Const, Instruction, ModulePools, PhiIncoming, Primitive, SwitchCase};
use crate::transform::ssa::ToSSA;
use crate::{
    context,
    ir::{
        ArrayType, BasicBlock, Func, IrBuilder, MatrixType, Module, ModuleKind, Node, NodeRef,
        StructType, Type, VectorElementType, VectorType,
    },
    CBoxedSlice, TypeOf,
};
use crate::{CArc, Pooled};

use super::Transform;
// Simple backward autodiff
// Loop is not supported since users would use path replay[https://rgl.epfl.ch/publications/Vicini2021PathReplay] anyway
struct GradTypeRecord {
    grad_type: CArc<Type>,
    // save for later
    // maybe we want to use tapes?
    #[allow(dead_code)]
    primal_field_to_grad_field: HashMap<usize, usize>,
    #[allow(dead_code)]
    grad_field_to_primal_field: HashMap<usize, usize>,
}

thread_local! {
    static GRAD_TYPES: RefCell<HashMap<CArc<Type>, Option<GradTypeRecord>>> =
    RefCell::new(HashMap::new());
}
fn grad_type_of_ve(t: &VectorElementType) -> Option<VectorElementType> {
    match t {
        VectorElementType::Scalar(p) => {
            let p = grad_type_of(context::register_type(Type::Primitive(*p)))?;
            Some(VectorElementType::Scalar(match p.as_ref() {
                Type::Primitive(p) => *p,
                _ => unreachable!(),
            }))
        }
        VectorElementType::Vector(v) => {
            let v = grad_type_of(context::register_type(Type::Vector((**v).clone())))?;
            Some(VectorElementType::Vector(match v.as_ref() {
                Type::Vector(vt) => CArc::new(vt.clone()),
                _ => unreachable!(),
            }))
        }
    }
}

pub fn grad_type_of(type_: CArc<Type>) -> Option<CArc<Type>> {
    GRAD_TYPES.with(|grad_types| loop {
        if let Some(record) = grad_types.borrow().get(&type_) {
            if let Some(record) = record {
                return Some(record.grad_type.clone());
            } else {
                return None;
            }
        }
        let t = _grad_type_of(type_.clone());
        grad_types.borrow_mut().insert(type_.clone(), t);
    })
}

fn _grad_type_of(type_: CArc<Type>) -> Option<GradTypeRecord> {
    let record = match type_.as_ref() {
        Type::Void | Type::UserData => None,
        Type::Primitive(p) => match p {
            crate::ir::Primitive::Bool => None,
            crate::ir::Primitive::Int16 => None,
            crate::ir::Primitive::Uint16 => None,
            crate::ir::Primitive::Int32 => None,
            crate::ir::Primitive::Uint32 => None,
            crate::ir::Primitive::Int64 => None,
            crate::ir::Primitive::Uint64 => None,
            crate::ir::Primitive::Float16 => todo!(),
            crate::ir::Primitive::Float32 => Some(GradTypeRecord {
                grad_type: context::register_type(Type::Primitive(crate::ir::Primitive::Float32)),
                primal_field_to_grad_field: HashMap::new(),
                grad_field_to_primal_field: HashMap::new(),
            }),
            crate::ir::Primitive::Float64 => Some(GradTypeRecord {
                grad_type: context::register_type(Type::Primitive(crate::ir::Primitive::Float64)),
                primal_field_to_grad_field: HashMap::new(),
                grad_field_to_primal_field: HashMap::new(),
            }),
        },
        Type::Vector(v) => {
            let ve = grad_type_of_ve(&v.element)?;
            let mut map = HashMap::new();
            for i in 0..v.length {
                map.insert(i as usize, i as usize);
            }
            Some(GradTypeRecord {
                grad_type: context::register_type(Type::Vector(VectorType {
                    element: ve,
                    length: v.length,
                })),
                primal_field_to_grad_field: map.clone(),
                grad_field_to_primal_field: map.clone(),
            })
        }
        Type::Matrix(m) => {
            let ve = grad_type_of_ve(&m.element)?;
            let mut map = HashMap::new();
            for i in 0..m.dimension {
                map.insert(i as usize, i as usize);
            }
            Some(GradTypeRecord {
                grad_type: context::register_type(Type::Matrix(MatrixType {
                    element: ve,
                    dimension: m.dimension,
                })),
                primal_field_to_grad_field: map.clone(),
                grad_field_to_primal_field: map.clone(),
            })
        }
        Type::Struct(st) => {
            let fields: Vec<(usize, CArc<Type>)> = st
                .fields
                .as_ref()
                .iter()
                .map(|f| grad_type_of(f.clone()))
                .enumerate()
                .filter(|(_, f)| f.is_some())
                .map(|(i, f)| (i, f.unwrap()))
                .collect();
            let mut grad_field_to_primal_field = HashMap::new();
            let mut primal_field_to_grad_field = HashMap::new();
            for (i, (j, _)) in fields.iter().enumerate() {
                grad_field_to_primal_field.insert(i, *j);
                primal_field_to_grad_field.insert(*j, i);
            }
            if fields.len() == 0 {
                return None;
            }
            let fields = fields.into_iter().map(|(_, f)| f).collect::<Vec<_>>();
            let alignment = fields.iter().map(|f| f.alignment()).max().unwrap();
            let size = fields.iter().map(|f| f.size()).sum();
            let fields: &[_] = Vec::leak(fields.to_vec());
            let st = Type::Struct(StructType {
                fields: CBoxedSlice::from(fields),
                alignment,
                size,
            });
            Some(GradTypeRecord {
                grad_type: context::register_type(st),
                primal_field_to_grad_field,
                grad_field_to_primal_field,
            })
        }
        Type::Array(a) => {
            let element = grad_type_of(a.element.clone())?;
            let at = Type::Array(ArrayType {
                element,
                length: a.length,
            });
            let mut map = HashMap::new();
            for i in 0..a.length {
                map.insert(i as usize, i as usize);
            }
            Some(GradTypeRecord {
                grad_type: context::register_type(at),
                primal_field_to_grad_field: map.clone(),
                grad_field_to_primal_field: map.clone(),
            })
        }
        Type::Opaque(_) => None,
    };
    record
}

struct StoreIntermediate<'a> {
    // map from node to its intermediate
    intermediate: IndexMap<NodeRef, NodeRef>,
    intermediate_to_node: IndexMap<NodeRef, NodeRef>,
    forward_reachable: IndexSet<NodeRef>,
    backward_reachable: IndexSet<NodeRef>,
    grads: IndexMap<NodeRef, NodeRef>,
    final_grad: IndexMap<NodeRef, usize>,
    #[allow(dead_code)]
    module: &'a Module,
    builder: IrBuilder,
    locally_defined_nodes: HashSet<NodeRef>,
}

impl<'a> StoreIntermediate<'a> {
    fn new(module: &'a Module) -> Self {
        let mut builder = IrBuilder::new(module.pools.clone());
        builder.set_insert_point(module.entry.first);
        let locally_defined_nodes = HashSet::from_iter(module.collect_nodes());
        Self {
            intermediate: IndexMap::new(),
            intermediate_to_node: IndexMap::new(),
            grads: IndexMap::new(),
            module,
            forward_reachable: IndexSet::new(),
            backward_reachable: IndexSet::new(),
            builder,
            final_grad: IndexMap::new(),
            locally_defined_nodes,
        }
    }
    fn run(&mut self) {
        self.forward_sweep_block(self.module.entry);
        self.backward_sweep_block(self.module.entry);
        for n in &self.backward_reachable.clone() {
            self.create_intermediate(*n);
            self.add_grad(*n);
            match n.get().instruction.as_ref() {
                Instruction::Call(_, args) => {
                    for a in args.as_ref() {
                        if !is_type_equal(&a.type_(), &Type::void())
                            && a.get().instruction.has_value()
                        {
                            assert!(a.is_linked(), "{:?}", a.get().instruction);
                            self.create_intermediate(*a);
                        }
                    }
                }
                _ => {}
            }
        }
        self.builder
            .set_insert_point(self.module.entry.last.get().prev);
        for (n, local) in &mut self.intermediate {
            if local.is_local() {
                let l = self
                    .builder
                    .call(Func::Load, &[local.clone()], n.type_().clone());
                *local = l;
                self.intermediate_to_node.insert(l, *n);
            } else {
                self.intermediate_to_node.insert(*n, *n);
            }
        }
    }
    // fn visit_block(&mut self, block: &BasicBlock) {
    //     for node in block.iter() {
    //         self.visit(node);
    //     }
    // }
    fn add_grad(&mut self, node: NodeRef) {
        if self.grads.contains_key(&node) {
            return;
        }
        let grad = self.builder.local_zero_init(node.type_().clone());
        self.grads.insert(node, grad);
    }
    fn create_intermediate(&mut self, node: NodeRef) {
        if self.intermediate.contains_key(&node) {
            return;
        }
        // {
        //     println!("create intermediate");
        //     let debug = crate::ir::debug::luisa_compute_ir_dump_human_readable(&self.module);
        //     let debug = std::ffi::CString::new(debug.as_ref()).unwrap();
        //     println!("{}", debug.to_str().unwrap());
        // }
        if self.locally_defined_nodes.contains(&node) {
            let local = self.builder.local_zero_init(node.type_().clone());

            let st = Node::new(
                CArc::new(Instruction::Update {
                    var: local,
                    value: node,
                }),
                Type::void(),
            );
            let st = new_node(&self.module.pools, st);
            node.insert_after_self(st);
            self.intermediate.insert(node, local);
        } else {
            self.intermediate.insert(node, node);
        }
    }
    fn forward_sweep_block(&mut self, block: Pooled<BasicBlock>) {
        for node in block.iter() {
            self.forward_sweep(node);
        }
    }
    // forward sweep marks all nodes that (might) require gradient
    // in other words, nodes that are reachable from a RequiresGradient call
    fn forward_sweep(&mut self, node: NodeRef) {
        let instruction = &node.get().instruction;
        let type_ = &node.get().type_;

        let grad_type = grad_type_of(type_.clone());
        match instruction.as_ref() {
            Instruction::Call(func, args) => {
                if args
                    .as_ref()
                    .iter()
                    .any(|a| self.forward_reachable.contains(a))
                    && grad_type.is_some()
                    && *func != Func::Detach
                {
                    self.forward_reachable.insert(node);
                }
                if *func == Func::RequiresGradient {
                    self.forward_reachable.insert(args.as_ref()[0]);
                    if !self.final_grad.contains_key(&args.as_ref()[0]) {
                        let i = self.final_grad.len();
                        self.final_grad.insert(args.as_ref()[0], i);
                    }
                } else if *func == Func::GradientMarker {
                    assert!(!self.final_grad.contains_key(&args.as_ref()[0]));
                    self.grads.insert(args.as_ref()[0], args.as_ref()[1]);
                    // self.final_grad.insert(args.as_ref()[0]);
                }
            }
            Instruction::Switch {
                value,
                default,
                cases,
            } => {
                self.create_intermediate(*value);
                self.forward_sweep_block(*default);
                for SwitchCase { value: _, block } in cases.as_ref().iter() {
                    self.forward_sweep_block(*block);
                }
            }
            Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                self.create_intermediate(*cond);
                self.forward_sweep_block(*true_branch);
                self.forward_sweep_block(*false_branch);
            }
            Instruction::Phi(incomings) => {
                self.create_intermediate(node);
                if incomings
                    .as_ref()
                    .iter()
                    .any(|PhiIncoming { value, .. }| self.forward_reachable.contains(value))
                    && grad_type.is_some()
                {
                    self.forward_reachable.insert(node);
                }
            }
            _ => {}
        }
    }
    fn backward_sweep(&mut self, node: NodeRef) {
        let instruction = &node.get().instruction;
        let type_ = &node.get().type_;

        let grad_type = grad_type_of(type_.clone());
        match instruction.as_ref() {
            Instruction::Call(func, args) => {
                if *func == Func::GradientMarker {
                    self.backward_reachable.insert(args.as_ref()[0]);
                }
                if self.backward_reachable.contains(&node)
                    && grad_type.is_some()
                    && *func != Func::Detach
                {
                    for a in args.as_ref().iter() {
                        if self.forward_reachable.contains(a) {
                            self.backward_reachable.insert(*a);
                        }
                    }
                }
            }
            Instruction::If {
                cond: _,
                true_branch,
                false_branch,
            } => {
                self.backward_sweep_block(*true_branch);
                self.backward_sweep_block(*false_branch);
            }
            Instruction::Switch {
                value: _,
                default,
                cases,
            } => {
                self.backward_sweep_block(*default);
                for SwitchCase { value: _, block } in cases.as_ref().iter() {
                    self.backward_sweep_block(*block);
                }
            }
            Instruction::Phi(incomings) => {
                for PhiIncoming { value, .. } in incomings.as_ref().iter() {
                    if self.forward_reachable.contains(value) {
                        self.backward_reachable.insert(*value);
                    }
                }
            }
            _ => {}
        }
    }
    fn backward_sweep_block(&mut self, block: Pooled<BasicBlock>) {
        for node in block.nodes().iter().rev() {
            self.backward_sweep(*node);
        }
    }
}

struct Backward {
    pools: CArc<ModulePools>,
    grads: IndexMap<NodeRef, NodeRef>,
    intermediate: IndexMap<NodeRef, NodeRef>,
    intermediate_to_node: IndexMap<NodeRef, NodeRef>,
    final_grad: IndexMap<NodeRef, usize>,
}

impl Backward {
    fn grad(&mut self, node: NodeRef) -> Option<NodeRef> {
        self.grads.get(&node).copied()
    }

    fn accumulate_grad(&mut self, mut node: NodeRef, grad: NodeRef, builder: &mut IrBuilder) {
        if self.intermediate_to_node.contains_key(&node) {
            node = self.intermediate_to_node[&node];
        }
        if let Some(grad_var) = self.grad(node) {
            builder.call(Func::AccGrad, &[grad_var, grad], Type::void());
        }
        // } else {
        //     self.grads.insert(node, grad);
        // }
    }

    // out = -in
    // => δ(in) = -δ(out)
    fn backward_neg(
        &mut self,
        in_: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), in_.type_()));
        let neg_out_grad = builder.call(Func::Neg, &[out_grad], out_grad.type_().clone());
        return neg_out_grad;
    }

    // out = lhs + rhs
    // => δ(lhs) = δ(out)
    //    δ(rhs) = δ(out)
    fn backward_add(
        &mut self,
        lhs: NodeRef,
        rhs: NodeRef,
        out_grad: NodeRef,
        _: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(out_grad.type_(), lhs.type_()));
        assert!(is_type_equal(out_grad.type_(), rhs.type_()));
        return (out_grad, out_grad);
    }

    // out = lhs - rhs
    // => δ(lhs) = δ(out)
    //    δ(rhs) = -δ(out)
    fn backward_sub(
        &mut self,
        lhs: NodeRef,
        rhs: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(out_grad.type_(), lhs.type_()));
        assert!(is_type_equal(out_grad.type_(), rhs.type_()));
        return (out_grad, self.backward_neg(rhs, out_grad, builder));
    }

    // out = lhs * rhs
    // => δ(lhs) = δ(out) * rhs
    //    δ(rhs) = δ(out) * lhs
    fn backward_comp_mul(
        &mut self,
        lhs: NodeRef,
        rhs: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(out_grad.type_(), lhs.type_()));
        assert!(is_type_equal(out_grad.type_(), rhs.type_()));
        let func = if lhs.type_().is_matrix() {
            Func::MatCompMul
        } else {
            Func::Mul
        };
        let lhs_grad = builder.call(func.clone(), &[out_grad, rhs], out_grad.type_().clone());
        let rhs_grad = builder.call(func.clone(), &[out_grad, lhs], out_grad.type_().clone());
        return (lhs_grad, rhs_grad);
    }

    // out = lhs(matrix) * rhs(vector)
    // => δ(lhs) = δ(out) * rhs^T = outer_product(δ(out), rhs)
    //    δ(rhs) = lhs^T * δ(out)
    fn backward_mat_vec_mul(
        &mut self,
        lhs: NodeRef,
        rhs: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(out_grad.type_(), rhs.type_()));
        let lhs_grad = builder.call(Func::OuterProduct, &[out_grad, rhs], lhs.type_().clone());
        // let out_grad_t = builder.call(Func::Transpose, &[out_grad], out_grad.type_());
        let lhs_t = builder.call(Func::Transpose, &[lhs], lhs.type_().clone());
        let rhs_grad = builder.call(Func::Mul, &[lhs_t, out_grad], rhs.type_().clone());
        return (lhs_grad, rhs_grad);
    }

    // out = lhs(matrix) * rhs(matrix)
    // => δ(lhs) = δ(out) * rhs^T
    //    δ(rhs) = lhs^T * δ(out)
    fn backward_mat_mul(
        &mut self,
        lhs: NodeRef,
        rhs: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(out_grad.type_(), lhs.type_()));
        assert!(is_type_equal(out_grad.type_(), rhs.type_()));
        let rhs_t = builder.call(Func::Transpose, &[rhs], rhs.type_().clone());
        let lhs_grad = builder.call(Func::Mul, &[out_grad, rhs_t], lhs.type_().clone());
        let lhs_t = builder.call(Func::Transpose, &[lhs], lhs.type_().clone());
        let rhs_grad = builder.call(Func::Mul, &[lhs_t, out_grad], rhs.type_().clone());
        return (lhs_grad, rhs_grad);
    }

    // out = lhs * rhs
    // case matrix * vector:
    // => δ(lhs) = δ(out) * rhs^T = outer_product(δ(out), rhs)
    //    δ(rhs) = lhs^T * δ(out)
    // case matrix * matrix:
    // => δ(lhs) = δ(out) * rhs^T
    //    δ(rhs) = lhs^T * δ(out)
    // other:
    // => δ(lhs) = δ(out) * rhs
    //    δ(rhs) = δ(out) * lhs
    fn backward_mul(
        &mut self,
        lhs: NodeRef,
        rhs: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        match (lhs.type_().deref(), rhs.type_().deref()) {
            (Type::Matrix(matrix_type), Type::Vector(vector_type)) => {
                assert_eq!(matrix_type.dimension, vector_type.length);
                return self.backward_mat_vec_mul(lhs, rhs, out_grad, builder);
            }
            (Type::Matrix(_), Type::Matrix(_)) => {
                return self.backward_mat_mul(lhs, rhs, out_grad, builder);
            }
            _ => {
                return self.backward_comp_mul(lhs, rhs, out_grad, builder);
            }
        }
    }

    // out = lhs / rhs
    // => δ(lhs) = δ(out) / rhs
    //    δ(rhs) = δ(out) * -lhs / (rhs * rhs)
    fn backward_div(
        &mut self,
        lhs: NodeRef,
        rhs: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(out_grad.type_(), lhs.type_()));
        assert!(is_type_equal(out_grad.type_(), rhs.type_()));
        let lhs_grad = builder.call(Func::Div, &[out_grad, rhs], out_grad.type_().clone());
        let neg_lhs = builder.call(Func::Neg, &[lhs], lhs.type_().clone());
        let sqr_rhs = builder.call(Func::Mul, &[rhs, rhs], rhs.type_().clone());
        let out_rhs = builder.call(Func::Div, &[neg_lhs, sqr_rhs], rhs.type_().clone());
        let rhs_grad = builder.call(Func::Mul, &[out_grad, out_rhs], out_grad.type_().clone());
        return (lhs_grad, rhs_grad);
    }

    // out = p ? a : b
    // => δ(a) = p ? δ(out) : 0
    //    δ(b) = p ? 0 : δ(out)
    fn backward_select(
        &mut self,
        p: NodeRef,
        a: NodeRef,
        b: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(out_grad.type_(), a.type_()));
        assert!(is_type_equal(out_grad.type_(), b.type_()));
        let zero = builder.const_(Const::Zero(a.type_().clone()));
        let a_grad = builder.call(Func::Select, &[p, out_grad, zero], out_grad.type_().clone());
        let b_grad = builder.call(Func::Select, &[p, zero, out_grad], out_grad.type_().clone());
        return (a_grad, b_grad);
    }

    // out = min(a, b) = a < b ? a : b
    // => δ(a) = a < b ? δ(out) : 0
    //    δ(b) = a < b ? 0 : δ(out)
    fn backward_min(
        &mut self,
        a: NodeRef,
        b: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(out_grad.type_(), a.type_()));
        assert!(is_type_equal(out_grad.type_(), b.type_()));
        let a_lt_b = builder.call(Func::Lt, &[a, b], Type::bool(a.type_().clone()));
        let (a_grad, b_grad) = self.backward_select(a_lt_b, a, b, out_grad, builder);
        return (a_grad, b_grad);
    }

    // out = max(a, b) = a > b ? a : b
    // => δ(a) = a > b ? δ(out) : 0
    //    δ(b) = a > b ? 0 : δ(out)
    fn backward_max(
        &mut self,
        a: NodeRef,
        b: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(out_grad.type_(), a.type_()));
        assert!(is_type_equal(out_grad.type_(), b.type_()));
        let a_gt_b = builder.call(Func::Gt, &[a, b], Type::bool(a.type_().clone()));
        let (a_grad, b_grad) = self.backward_select(a_gt_b, a, b, out_grad, builder);
        return (a_grad, b_grad);
    }

    // out = sum(x_i)
    // => δ(x_i) = δ(out)
    fn backward_reduce_sum(
        &mut self,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(&out_grad.type_(), &x.type_().element()));
        if x.type_().is_vector() {
            return builder.call(Func::Vec, &[out_grad], x.type_().clone());
        } else if x.type_().is_matrix() {
            return builder.call(Func::Mat, &[out_grad], x.type_().clone());
        } else {
            return out_grad;
        }
    }

    // out = reduce_sum(a * b) = reduce_sum(ab)
    // => δab = broadcast(δ(out))
    // => δa = δab * b
    // => δb = δab * a
    fn backward_dot(
        &mut self,
        a: NodeRef,
        b: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(a.type_(), b.type_()));
        let ab = builder.call(Func::Mul, &[a, b], a.type_().clone());
        let d_ab = self.backward_reduce_sum(ab, out_grad, builder);
        let d_a = builder.call(Func::Mul, &[d_ab, b], a.type_().clone());
        let d_b = builder.call(Func::Mul, &[d_ab, a], b.type_().clone());
        (d_a, d_b)
    }
    // out = cross(a, b)
    // => δa = cross(b, δ(out))
    // => δb = cross(δ(out), a)
    fn backward_cross(
        &mut self,
        a: NodeRef,
        b: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(a.type_(), out_grad.type_()));
        assert!(is_type_equal(b.type_(), out_grad.type_()));
        let d_a = builder.call(Func::Cross, &[b, out_grad], a.type_().clone());
        let d_b = builder.call(Func::Cross, &[out_grad, a], b.type_().clone());
        (d_a, d_b)
    }
    // out = inv(m)
    // => δm = -m^T * δ(out) * m^T
    fn backward_inverse(
        &mut self,
        m: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(m.type_(), out_grad.type_()));
        let m_t = builder.call(Func::Transpose, &[m], m.type_().clone());
        let m_t_mul_out_grad = builder.call(Func::Mul, &[m_t, out_grad], m.type_().clone());
        let m_t_mul_out_grad_mul_m_t =
            builder.call(Func::Mul, &[m_t_mul_out_grad, m_t], m.type_().clone());
        let neg_m_t_mul_out_grad_mul_m_t =
            builder.call(Func::Neg, &[m_t_mul_out_grad_mul_m_t], m.type_().clone());
        neg_m_t_mul_out_grad_mul_m_t
    }
    // out = transpose(m)
    // => δm = δ(out)^T
    fn backward_transpose(
        &mut self,
        m: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(m.type_(), out_grad.type_()));
        builder.call(Func::Transpose, &[out_grad], m.type_().clone())
    }
    // out = det(m)
    // => δm = δ(out) * det(m) * m^(-1)^T
    fn backward_determinant(
        &mut self,
        out: NodeRef,
        m: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        let dout_mul_det = builder.call(Func::Mul, &[out_grad, out], out.type_().clone());
        let m_inv = builder.call(Func::Inverse, &[m], m.type_().clone());
        let m_inv_t = builder.call(Func::Transpose, &[m_inv], m.type_().clone());
        let dout_mul_det = builder.call(Func::Mat, &[dout_mul_det], m.type_().clone());
        let dout_mul_det_mul_m_inv_t = builder.call(
            Func::MatCompMul,
            &[dout_mul_det, m_inv_t],
            m.type_().clone(),
        );
        dout_mul_det_mul_m_inv_t
    }
    // out = acos(x)
    // => δ(x) = -1 / sqrt(1 - x^2)
    fn backward_acos(&mut self, x: NodeRef, out_grad: NodeRef, builder: &mut IrBuilder) -> NodeRef {
        assert!(is_type_equal(x.type_(), out_grad.type_()));
        let x2 = builder.call(Func::Mul, &[x, x], x.type_().clone());
        let one = builder.const_(Const::One(x.type_().clone()));
        let one_minus_x2 = builder.call(Func::Sub, &[one, x2], x.type_().clone());
        let sqrt = builder.call(Func::Sqrt, &[one_minus_x2], x.type_().clone());
        let one_over_sqrt = builder.call(Func::Div, &[one, sqrt], x.type_().clone());
        let neg_one_over_sqrt = builder.call(Func::Neg, &[one_over_sqrt], x.type_().clone());
        let x_grad = builder.call(Func::Mul, &[out_grad, neg_one_over_sqrt], x.type_().clone());
        return x_grad;
    }

    // out = acosh(x)
    // => δ(x) = 1 / sqrt(x^2 - 1)
    fn backward_acosh(
        &mut self,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(x.type_(), out_grad.type_()));
        let x2 = builder.call(Func::Mul, &[x, x], x.type_().clone());
        let one = builder.const_(Const::One(x.type_().clone()));
        let x2_minus_one = builder.call(Func::Sub, &[x2, one], x.type_().clone());
        let sqrt = builder.call(Func::Sqrt, &[x2_minus_one], x.type_().clone());
        let one_over_sqrt = builder.call(Func::Div, &[one, sqrt], x.type_().clone());
        let x_grad = builder.call(Func::Mul, &[out_grad, one_over_sqrt], x.type_().clone());
        return x_grad;
    }

    // out = asin(x)
    // => δ(x) = 1 / sqrt(1 - x^2)
    fn backward_asin(&mut self, x: NodeRef, out_grad: NodeRef, builder: &mut IrBuilder) -> NodeRef {
        assert!(is_type_equal(x.type_(), out_grad.type_()));
        let x2 = builder.call(Func::Mul, &[x, x], x.type_().clone());
        let one = builder.const_(Const::One(x.type_().clone()));
        let one_minus_x2 = builder.call(Func::Sub, &[one, x2], x.type_().clone());
        let sqrt = builder.call(Func::Sqrt, &[one_minus_x2], x.type_().clone());
        let one_over_sqrt = builder.call(Func::Div, &[one, sqrt], x.type_().clone());
        let x_grad = builder.call(Func::Mul, &[out_grad, one_over_sqrt], x.type_().clone());
        return x_grad;
    }

    // out = asinh(x)
    // => δ(x) = 1 / sqrt(1 + x^2)
    fn backward_asinh(
        &mut self,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(x.type_(), out_grad.type_()));
        let x2 = builder.call(Func::Mul, &[x, x], x.type_().clone());
        let one = builder.const_(Const::One(x.type_().clone()));
        let one_plus_x2 = builder.call(Func::Add, &[one, x2], x.type_().clone());
        let sqrt = builder.call(Func::Sqrt, &[one_plus_x2], x.type_().clone());
        let one_over_sqrt = builder.call(Func::Div, &[one, sqrt], x.type_().clone());
        let x_grad = builder.call(Func::Mul, &[out_grad, one_over_sqrt], x.type_().clone());
        return x_grad;
    }

    // out = length_squared(x) = dot(x, x)
    // => δ(x) = 2 * x * δ(out)
    fn backward_length_squared(
        &mut self,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), &x.type_().element()));
        let twice_x = builder.call(Func::Add, &[x, x], x.type_().clone());
        let out_grad = builder.call(Func::Vec, &[out_grad], x.type_().clone());
        let x_grad = builder.call(Func::Mul, &[twice_x, out_grad], x.type_().clone());
        return x_grad;
    }

    // out = sqrt(x)
    // => δ(x) = 1 / (2 * sqrt(x)) * δ(out)
    fn backward_sqrt(
        &mut self,
        out: NodeRef,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), x.type_()));
        let twice_sqrt_x = builder.call(Func::Add, &[out, out], x.type_().clone());
        let x_grad = builder.call(Func::Div, &[out_grad, twice_sqrt_x], x.type_().clone());
        return x_grad;
    }

    // out = rsqrt(x) = 1 / sqrt(x)
    // => δ(x) = -1 / (2 * x * sqrt(x)) * δ(out)
    fn backward_rsqrt(
        &mut self,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), x.type_()));
        let sqrt_x = builder.call(Func::Sqrt, &[x], x.type_().clone());
        let twice_x = builder.call(Func::Add, &[x, x], x.type_().clone());
        let twice_x_times_sqrt_x = builder.call(Func::Mul, &[twice_x, sqrt_x], x.type_().clone());
        let neg_out_grad = builder.call(Func::Neg, &[out_grad], x.type_().clone());
        let x_grad = builder.call(
            Func::Div,
            &[neg_out_grad, twice_x_times_sqrt_x],
            x.type_().clone(),
        );
        return x_grad;
    }

    // out = atan(x)
    // => δ(x) = 1 / (1 + x^2) * δ(out)
    fn backward_atan(&mut self, x: NodeRef, out_grad: NodeRef, builder: &mut IrBuilder) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), x.type_()));
        let xx = builder.call(Func::Mul, &[x, x], x.type_().clone());
        let one = builder.const_(Const::One(x.type_().clone()));
        let one_plus_xx = builder.call(Func::Add, &[one, xx], x.type_().clone());
        let x_grad = builder.call(Func::Div, &[out_grad, one_plus_xx], x.type_().clone());
        return x_grad;
    }

    // out = atan2(y, x)
    // => δ(y) = x / (x^2 + y^2) * δ(out)
    // => δ(x) = -y / (x^2 + y^2) * δ(out)
    fn backward_atan2(
        &mut self,
        y: NodeRef,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        let xx = builder.call(Func::Mul, &[x, x], x.type_().clone());
        let yy = builder.call(Func::Mul, &[y, y], y.type_().clone());
        let xx_plus_yy = builder.call(Func::Add, &[xx, yy], x.type_().clone());
        let x_over_xx_plus_yy = builder.call(Func::Div, &[x, xx_plus_yy], x.type_().clone());
        let neg_y = builder.call(Func::Neg, &[y], y.type_().clone());
        let neg_y_over_xx_plus_yy =
            builder.call(Func::Div, &[neg_y, xx_plus_yy], y.type_().clone());
        let y_grad = builder.call(Func::Mul, &[x_over_xx_plus_yy, out_grad], y.type_().clone());
        let x_grad = builder.call(
            Func::Mul,
            &[neg_y_over_xx_plus_yy, out_grad],
            x.type_().clone(),
        );
        return (y_grad, x_grad);
    }

    // out = atanh(x) = 0.5 * log((1 + x) / (1 - x))
    // => δ(x) = 1 / (1 - x^2) * δ(out)
    fn backward_atanh(
        &mut self,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), x.type_()));
        let xx = builder.call(Func::Mul, &[x, x], x.type_().clone());
        let one = builder.const_(Const::One(x.type_().clone()));
        let one_minus_xx = builder.call(Func::Sub, &[one, xx], x.type_().clone());
        let x_grad = builder.call(Func::Div, &[out_grad, one_minus_xx], x.type_().clone());
        return x_grad;
    }

    // out = log(x)
    // => δ(x) = 1 / x * δ(out)
    fn backward_log(&mut self, x: NodeRef, out_grad: NodeRef, builder: &mut IrBuilder) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), x.type_()));
        let x_grad = builder.call(Func::Div, &[out_grad, x], x.type_().clone());
        return x_grad;
    }

    fn fp_constant(&mut self, t: CArc<Type>, x: f64, builder: &mut IrBuilder) -> NodeRef {
        return match t.deref() {
            Type::Primitive(p) => match p {
                Primitive::Float16 => todo!(),
                Primitive::Float32 => builder.const_(Const::Float32(x as f32)),
                Primitive::Float64 => builder.const_(Const::Float64(x)),
                _ => panic!("fp_constant: invalid type: {:?}", t),
            },
            Type::Vector(_) => {
                let elem = self.fp_constant(t.element(), x, builder);
                builder.call(Func::Vec, &[elem], t)
            }
            Type::Matrix(_) => {
                let elem = self.fp_constant(t.element(), x, builder);
                builder.call(Func::Mat, &[elem], t)
            }
            _ => panic!("fp_constant: invalid type to broadcast: {:?}", t),
        };
    }

    // out = log(a, x) = log(x) / log(a)
    // => δ(x) = 1 / (x * log(a)) * δ(out)
    fn backward_log_base(
        &mut self,
        base: f64,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), x.type_()));
        let inv_log = self.fp_constant(out_grad.type_().clone(), 1.0 / base.ln(), builder);
        let out_grad_times_inv_log =
            builder.call(Func::Mul, &[out_grad, inv_log], out_grad.type_().clone());
        let x_grad = builder.call(Func::Div, &[out_grad_times_inv_log, x], x.type_().clone());
        return x_grad;
    }

    // out = exp(x)
    // => δ(x) = exp(x) * δ(out)
    fn backward_exp(
        &mut self,
        out: NodeRef,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), x.type_()));
        let x_grad = builder.call(Func::Mul, &[out, out_grad], x.type_().clone());
        return x_grad;
    }

    // out = exp(2, x) = 2^x
    // => δ(x) = log(2) * exp(2, x) * δ(out)
    fn backward_exp2(
        &mut self,
        out: NodeRef,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), x.type_()));
        let log2 = self.fp_constant(out_grad.type_().clone(), 2.0f64.ln(), builder);
        let exp2_x = out;
        let log2_times_exp2_x = builder.call(Func::Mul, &[log2, exp2_x], x.type_().clone());
        let x_grad = builder.call(Func::Mul, &[log2_times_exp2_x, out_grad], x.type_().clone());
        return x_grad;
    }

    // out = exp10(x) = 10^x
    // => δ(x) = log(10) * exp10(x) * δ(out)
    fn backward_exp10(
        &mut self,
        out: NodeRef,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), x.type_()));
        let log10 = self.fp_constant(out_grad.type_().clone(), 10.0f64.ln(), builder);
        let exp10_x = out;
        let log10_times_exp10_x = builder.call(Func::Mul, &[log10, exp10_x], x.type_().clone());
        let x_grad = builder.call(
            Func::Mul,
            &[log10_times_exp10_x, out_grad],
            x.type_().clone(),
        );
        return x_grad;
    }

    // out = pow(a, b) = a^b
    // => δ(a) = b * a^(b-1) * δ(out)
    // => δ(b) = a^b * log(a) * δ(out)
    fn backward_pow(
        &mut self,
        a: NodeRef,
        b: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert!(is_type_equal(out_grad.type_(), a.type_()));
        assert!(is_type_equal(out_grad.type_(), b.type_()));
        let one = builder.const_(Const::One(b.type_().clone()));
        let b_minus_one = builder.call(Func::Sub, &[b, one], b.type_().clone());
        let pow = builder.call(Func::Powf, &[a, b_minus_one], a.type_().clone());
        let b_times_pow = builder.call(Func::Mul, &[b, pow], a.type_().clone());
        let a_grad = builder.call(Func::Mul, &[b_times_pow, out_grad], a.type_().clone());
        let log_a = builder.call(Func::Log, &[a], a.type_().clone());
        let pow = builder.call(Func::Powf, &[a, b], a.type_().clone());
        let pow_times_log_a = builder.call(Func::Mul, &[pow, log_a], a.type_().clone());
        let b_grad = builder.call(Func::Mul, &[pow_times_log_a, out_grad], b.type_().clone());
        return (a_grad, b_grad);
    }

    // out = powi(a, n) = a^n
    // => δ(a) = n * a^(n-1) * δ(out)
    fn backward_powi(
        &mut self,
        a: NodeRef,
        n: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), a.type_()));
        let one = builder.const_(Const::One(n.type_().clone()));
        let n_minus_one = builder.call(Func::Sub, &[n, one], n.type_().clone());
        let pow = builder.call(Func::Powi, &[a, n_minus_one], a.type_().clone());
        let n = builder.call(Func::Cast, &[n], a.type_().clone());
        let n_times_pow = builder.call(Func::Mul, &[n, pow], a.type_().clone());
        let a_grad = builder.call(Func::Mul, &[n_times_pow, out_grad], a.type_().clone());
        return a_grad;
    }

    // out = length(x) = sqrt(dot(x, x))
    // => δ(x) = x / length(x) * δ(out) = normalize(x) * δ(out)
    fn backward_length(
        &mut self,
        x: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), &x.type_().element()));
        let n = builder.call(Func::Normalize, &[x], x.type_().clone());
        let out_grad = builder.call(Func::Vec, &[out_grad], x.type_().clone());
        let x_grad = builder.call(Func::Mul, &[n, out_grad], x.type_().clone());
        return x_grad;
    }

    // out = normalize(v) = v / sqrt(dot(v, v))
    // => δ(v) = (δ(out) - dot(n, δ(out)) * n) / length(v)
    fn backward_normalize(
        &mut self,
        v: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> NodeRef {
        assert!(is_type_equal(out_grad.type_(), v.type_()));
        let n = builder.call(Func::Normalize, &[v], v.type_().clone());
        let dot = builder.call(Func::Dot, &[n, out_grad], v.type_().element());
        let dot = builder.call(Func::Vec, &[dot], v.type_().clone());
        let dot_times_n = builder.call(Func::Mul, &[dot, n], v.type_().clone());
        let minus = builder.call(Func::Sub, &[out_grad, dot_times_n], v.type_().clone());
        let l = builder.call(Func::Length, &[v], v.type_().element());
        let l = builder.call(Func::Vec, &[l], v.type_().clone());
        let v_grad = builder.call(Func::Div, &[minus, l], v.type_().clone());
        return v_grad;
    }

    // out = outer_product(a, b) = a * b^T (a, b: n x 1)
    // => δ(a) = δ(out) * b
    // => δ(b) = δ(out)^T * a
    fn backward_outer_product(
        &mut self,
        a: NodeRef,
        b: NodeRef,
        out_grad: NodeRef,
        builder: &mut IrBuilder,
    ) -> (NodeRef, NodeRef) {
        assert_eq!(out_grad.type_().element(), a.type_().element());
        assert_eq!(out_grad.type_().element(), b.type_().element());
        assert_eq!(out_grad.type_().dimension(), a.type_().dimension());
        assert_eq!(out_grad.type_().dimension(), b.type_().dimension());
        let a_grad = builder.call(Func::Mul, &[out_grad, b], a.type_().clone());
        let out_grad_t = builder.call(Func::Transpose, &[out_grad], out_grad.type_().clone());
        let b_grad = builder.call(Func::Mul, &[out_grad_t, a], b.type_().clone());
        return (a_grad, b_grad);
    }
    fn get_intermediate(&self, node: NodeRef) -> NodeRef {
        self.intermediate
            .get(&node)
            .cloned()
            .unwrap_or_else(|| panic!("{:?}", node.get().instruction))
    }
    fn backward(&mut self, node: NodeRef, builder: &mut IrBuilder) {
        let instruction = &node.get().instruction;
        let type_ = &node.get().type_;
        let grad_type = grad_type_of(type_.clone());

        match instruction.as_ref() {
            crate::ir::Instruction::Buffer => {}
            crate::ir::Instruction::Bindless => {}
            crate::ir::Instruction::Texture2D => {}
            crate::ir::Instruction::Texture3D => {}
            crate::ir::Instruction::Accel => {}
            crate::ir::Instruction::Shared => {}
            crate::ir::Instruction::Uniform => {}
            crate::ir::Instruction::Local { .. } => {}
            crate::ir::Instruction::Argument { .. } => {}
            crate::ir::Instruction::UserData(_) => {}
            crate::ir::Instruction::Invalid => {}
            crate::ir::Instruction::Const(_) => {}
            crate::ir::Instruction::Update { .. } => {}
            crate::ir::Instruction::AdScope { .. } => {
                todo!()
            }
            crate::ir::Instruction::AdDetach(_) => {}
            Instruction::RayQuery { .. } => panic!("RayQuery is not supported yet"),
            crate::ir::Instruction::Call(func, args) => {
                if grad_type.is_none() {
                    return;
                }
                let out_grad = self.grad(node);
                if out_grad.is_none() {
                    return;
                }
                // dbg!(node);
                // dbg!(func);
                let original_args = args.as_ref().iter().cloned().collect::<Vec<_>>();
                let args = args
                    .as_ref()
                    .iter()
                    .map(|a| self.get_intermediate(*a))
                    .collect::<Vec<_>>();
                let node = self.get_intermediate(node);
                let out_grad = out_grad.unwrap();
                match func {
                    Func::Add => {
                        let (lhs_grad, rhs_grad) =
                            self.backward_add(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], lhs_grad, builder);
                        self.accumulate_grad(args[1], rhs_grad, builder);
                    }
                    Func::Sub => {
                        let (lhs_grad, rhs_grad) =
                            self.backward_sub(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], lhs_grad, builder);
                        self.accumulate_grad(args[1], rhs_grad, builder);
                    }
                    Func::Mul => {
                        let (lhs_grad, rhs_grad) =
                            self.backward_mul(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], lhs_grad, builder);
                        self.accumulate_grad(args[1], rhs_grad, builder);
                    }
                    Func::Div => {
                        let (lhs_grad, rhs_grad) =
                            self.backward_div(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], lhs_grad, builder);
                        self.accumulate_grad(args[1], rhs_grad, builder);
                    }
                    Func::MatCompMul => {
                        let (lhs_grad, rhs_grad) =
                            self.backward_comp_mul(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], lhs_grad, builder);
                        self.accumulate_grad(args[1], rhs_grad, builder);
                    }
                    Func::Neg => {
                        let x_grad = self.backward_neg(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Select => {
                        let (a_grad, b_grad) =
                            self.backward_select(args[0], args[1], args[2], out_grad, builder);
                        self.accumulate_grad(args[1], a_grad, builder);
                        self.accumulate_grad(args[2], b_grad, builder);
                    }
                    Func::Clamp => {
                        // clamp(x, a, b) = min(max(x, a), b)
                        let min_x_a = builder.call(Func::Min, &[args[0], args[1]], type_.clone());
                        let (max_grad, b_grad) =
                            self.backward_min(min_x_a, args[2], out_grad, builder);
                        let (x_grad, a_grad) =
                            self.backward_max(args[0], args[1], max_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                        self.accumulate_grad(args[1], a_grad, builder);
                        self.accumulate_grad(args[2], b_grad, builder);
                    }
                    Func::Saturate => {
                        let zero = builder.const_(Const::Zero(type_.clone()));
                        let one = builder.const_(Const::One(type_.clone()));
                        let gt_zero =
                            builder.call(Func::Gt, &[args[0], zero], Type::bool(type_.clone()));
                        let lt_one =
                            builder.call(Func::Lt, &[args[0], one], Type::bool(type_.clone()));
                        let between_zero_and_one = builder.call(
                            Func::BitAnd,
                            &[gt_zero, lt_one],
                            Type::bool(type_.clone()),
                        );
                        let grad = builder.call(
                            Func::Select,
                            &[between_zero_and_one, out_grad, zero],
                            type_.clone(),
                        );
                        self.accumulate_grad(args[0], grad, builder);
                    }
                    Func::Lerp => {
                        // lerp(a, b, t) = (b - a) * t + a
                        let b_minus_a = builder.call(Func::Sub, &[args[1], args[0]], type_.clone());
                        let (b_minus_a_grad, t_grad) =
                            self.backward_comp_mul(b_minus_a, args[2], out_grad, builder);
                        let (b_grad, a_grad) =
                            self.backward_sub(args[1], args[0], b_minus_a_grad, builder);
                        let a_grad = builder.call(Func::Add, &[a_grad, out_grad], type_.clone());
                        self.accumulate_grad(args[0], a_grad, builder);
                        self.accumulate_grad(args[1], b_grad, builder);
                        self.accumulate_grad(args[2], t_grad, builder);
                    }
                    Func::Abs => {
                        // abs(x) = x >= 0 ? x : -x
                        let zero = builder.const_(Const::Zero(type_.clone()));
                        let cond =
                            builder.call(Func::Ge, &[args[0], zero], Type::bool(type_.clone()));
                        let neg_out_grad = builder.call(Func::Neg, &[out_grad], type_.clone());
                        let x_grad = builder.call(
                            Func::Select,
                            &[cond, out_grad, neg_out_grad],
                            type_.clone(),
                        );
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Min => {
                        let (a_grad, b_grad) =
                            self.backward_min(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], a_grad, builder);
                        self.accumulate_grad(args[1], b_grad, builder);
                    }
                    Func::Max => {
                        let (a_grad, b_grad) =
                            self.backward_max(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], a_grad, builder);
                        self.accumulate_grad(args[1], b_grad, builder);
                    }
                    Func::ReduceSum => {
                        let x_grad = self.backward_reduce_sum(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::ReduceProd => match args[0].type_().as_ref() {
                        Type::Vector(vec_type) => {
                            let elem_type = vec_type.element.to_type();
                            let mut x_grad = builder.const_(Const::Zero(args[0].type_().clone()));
                            for i in 0..vec_type.length {
                                let mut elem_grad = out_grad;
                                for j in 0..vec_type.length {
                                    if i != j {
                                        let index = builder.const_(Const::Uint32(j));
                                        let other_elem = builder.call(
                                            Func::ExtractElement,
                                            &[args[0], index],
                                            elem_type.clone(),
                                        );
                                        elem_grad = builder.call(
                                            Func::Mul,
                                            &[elem_grad, other_elem],
                                            elem_type.clone(),
                                        );
                                    }
                                }
                                let index = builder.const_(Const::Uint32(i));
                                x_grad = builder.call(
                                    Func::InsertElement,
                                    &[x_grad, elem_grad, index],
                                    args[0].type_().clone(),
                                );
                            }
                            self.accumulate_grad(args[0], x_grad, builder);
                        }
                        _ => unreachable!(),
                    },
                    Func::ReduceMin => {
                        let min = builder.call(Func::ReduceMin, &[args[0]], type_.clone());
                        let is_min = builder.call(
                            Func::Eq,
                            &[min, args[0]],
                            Type::bool(args[0].type_().clone()),
                        );
                        let zero = builder.const_(Const::Zero(args[0].type_().clone()));
                        let out_grad = if args[0].type_().is_vector() {
                            builder.call(Func::Vec, &[out_grad], args[0].type_().clone())
                        } else {
                            assert!(args[0].type_().is_matrix());
                            builder.call(Func::Mat, &[out_grad], args[0].type_().clone())
                        };
                        let x_grad = builder.call(
                            Func::Select,
                            &[is_min, out_grad, zero],
                            args[0].type_().clone(),
                        );
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::ReduceMax => {
                        let max = builder.call(Func::ReduceMax, &[args[0]], type_.clone());
                        let is_max = builder.call(
                            Func::Eq,
                            &[max, args[0]],
                            Type::bool(args[0].type_().clone()),
                        );
                        let zero = builder.const_(Const::Zero(args[0].type_().clone()));
                        let out_grad = if args[0].type_().is_vector() {
                            builder.call(Func::Vec, &[out_grad], args[0].type_().clone())
                        } else {
                            assert!(args[0].type_().is_matrix());
                            builder.call(Func::Mat, &[out_grad], args[0].type_().clone())
                        };
                        let x_grad = builder.call(
                            Func::Select,
                            &[is_max, out_grad, zero],
                            args[0].type_().clone(),
                        );
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Acos => {
                        let x_grad = self.backward_acos(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Acosh => {
                        let x_grad = self.backward_acosh(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Asin => {
                        let x_grad = self.backward_asin(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Asinh => {
                        let x_grad = self.backward_asinh(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Atan => {
                        let x_grad = self.backward_atan(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Atan2 => {
                        let (y_grad, x_grad) =
                            self.backward_atan2(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], y_grad, builder);
                        self.accumulate_grad(args[1], x_grad, builder);
                    }
                    Func::Atanh => {
                        let x_grad = self.backward_atanh(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Cos => {
                        let sin_x = builder.call(Func::Sin, &[args[0]], type_.clone());
                        let neg_sin_x = builder.call(Func::Neg, &[sin_x], type_.clone());
                        let x_grad = builder.call(Func::Mul, &[neg_sin_x, out_grad], type_.clone());
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Cosh => {
                        let sinh_x = builder.call(Func::Sinh, &[args[0]], type_.clone());
                        let x_grad = builder.call(Func::Mul, &[sinh_x, out_grad], type_.clone());
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Sin => {
                        let cos_x = builder.call(Func::Cos, &[args[0]], type_.clone());
                        let x_grad = builder.call(Func::Mul, &[cos_x, out_grad], type_.clone());
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Sinh => {
                        let cosh_x = builder.call(Func::Cosh, &[args[0]], type_.clone());
                        let x_grad = builder.call(Func::Mul, &[cosh_x, out_grad], type_.clone());
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Tan => {
                        let cos_x = builder.call(Func::Cos, &[args[0]], type_.clone());
                        let sqr_cos_x = builder.call(Func::Mul, &[cos_x, cos_x], type_.clone());
                        let x_grad = builder.call(Func::Div, &[out_grad, sqr_cos_x], type_.clone());
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Tanh => {
                        let cosh_x = builder.call(Func::Cosh, &[args[0]], type_.clone());
                        let sqr_cosh_x = builder.call(Func::Mul, &[cosh_x, cosh_x], type_.clone());
                        let x_grad =
                            builder.call(Func::Div, &[out_grad, sqr_cosh_x], type_.clone());
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Exp => {
                        let x_grad = self.backward_exp(node, args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Exp2 => {
                        let x_grad = self.backward_exp2(node, args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Exp10 => {
                        let x_grad = self.backward_exp10(node, args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Log => {
                        let x_grad = self.backward_log(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Log2 => {
                        let x_grad = self.backward_log_base(2.0, args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Log10 => {
                        let x_grad = self.backward_log_base(10.0, args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Powi => {
                        let x_grad = self.backward_powi(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Powf => {
                        let (a_grad, b_grad) =
                            self.backward_pow(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], a_grad, builder);
                        self.accumulate_grad(args[1], b_grad, builder);
                    }
                    Func::Sqrt => {
                        let x_grad = self.backward_sqrt(node, args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Rsqrt => {
                        let x_grad = self.backward_rsqrt(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Fract => {
                        self.accumulate_grad(args[0], out_grad, builder);
                    }
                    Func::Fma => {
                        // a * b + c
                        let a_grad = builder.call(Func::Mul, &[args[1], out_grad], type_.clone());
                        let b_grad = builder.call(Func::Mul, &[args[0], out_grad], type_.clone());
                        let c_grad = out_grad;
                        self.accumulate_grad(args[0], a_grad, builder);
                        self.accumulate_grad(args[1], b_grad, builder);
                        self.accumulate_grad(args[2], c_grad, builder);
                    }

                    Func::Copysign => {
                        // copysign(x, y) = y >= 0 ? abs(x) : -abs(x) = (y < 0) == (x < 0) ? x : -x
                        let zero = builder.const_(Const::Zero(type_.clone()));
                        let x_negative =
                            builder.call(Func::Lt, &[args[0], zero], Type::bool(type_.clone()));
                        let y_negative =
                            builder.call(Func::Lt, &[args[1], zero], Type::bool(type_.clone()));
                        let same_sign = builder.call(
                            Func::Eq,
                            &[x_negative, y_negative],
                            Type::bool(type_.clone()),
                        );
                        let neg_out_grad = builder.call(Func::Neg, &[out_grad], type_.clone());
                        let x_grad = builder.call(
                            Func::Select,
                            &[same_sign, out_grad, neg_out_grad],
                            type_.clone(),
                        );
                        self.accumulate_grad(args[0], x_grad, builder);
                    }

                    // Vector operations
                    Func::Cross => {
                        let (a_grad, b_grad) =
                            self.backward_cross(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], a_grad, builder);
                        self.accumulate_grad(args[1], b_grad, builder);
                    }
                    Func::Dot => {
                        let (a_grad, b_grad) =
                            self.backward_dot(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], a_grad, builder);
                        self.accumulate_grad(args[1], b_grad, builder);
                    }
                    Func::OuterProduct => {
                        let (a_grad, b_grad) =
                            self.backward_outer_product(args[0], args[1], out_grad, builder);
                        self.accumulate_grad(args[0], a_grad, builder);
                        self.accumulate_grad(args[1], b_grad, builder);
                    }
                    Func::Length => {
                        let x_grad = self.backward_length(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::LengthSquared => {
                        let x_grad = self.backward_length_squared(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Normalize => {
                        let x_grad = self.backward_normalize(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Faceforward => {
                        // faceforward(N, I, Nref) = dot(Nref, I) < 0.0 ? N : -N
                        let n = args[0];
                        let i = args[1];
                        let nref = args[2];
                        let dot = builder.call(Func::Dot, &[nref, i], n.type_().element());
                        let zero = builder.const_(Const::Zero(n.type_().element()));
                        let cond = builder.call(Func::Lt, &[dot, zero], n.type_().element());
                        let cond = builder.call(Func::Vec, &[cond], n.type_().clone());
                        let neg_out_grad = builder.call(Func::Neg, &[out_grad], n.type_().clone());
                        let n_grad = builder.call(
                            Func::Select,
                            &[cond, out_grad, neg_out_grad],
                            n.type_().clone(),
                        );
                        self.accumulate_grad(n, n_grad, builder);
                    }
                    Func::Reflect => {
                        // o = v - 2 * dot(v, n) * n
                        let v = args[0];
                        let n = args[1];
                        // do/dv = [1, 1, 1] - 2 * n * n
                        // do/dn = -2 * v * n
                        let one = builder.const_(Const::One(v.type_().clone()));
                        let two = builder.const_(Const::Float32(2.0));
                        let two = builder.call(Func::Vec, &[two], v.type_().clone());
                        let nn = builder.call(Func::Mul, &[n, n], v.type_().clone());
                        let twice_nn = builder.call(Func::Mul, &[two, nn], v.type_().clone());
                        let do_dv = builder.call(Func::Sub, &[one, twice_nn], v.type_().clone());
                        let v_grad = builder.call(Func::Mul, &[do_dv, out_grad], v.type_().clone());
                        self.accumulate_grad(v, v_grad, builder);
                        let minus_two = builder.call(Func::Neg, &[two], v.type_().clone());
                        let vn = builder.call(Func::Mul, &[v, n], v.type_().clone());
                        let do_dn = builder.call(Func::Mul, &[minus_two, vn], v.type_().clone());
                        let n_grad = builder.call(Func::Mul, &[do_dn, out_grad], v.type_().clone());
                        self.accumulate_grad(n, n_grad, builder);
                    }
                    // Matrix operations
                    Func::Determinant => {
                        let x_grad = self.backward_determinant(node, args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Inverse => {
                        let x_grad = self.backward_inverse(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Transpose => {
                        let x_grad = self.backward_transpose(args[0], out_grad, builder);
                        self.accumulate_grad(args[0], x_grad, builder);
                    }
                    Func::Vec => {
                        // vec(s) => [s, s, ...]
                        let sum =
                            builder.call(Func::ReduceSum, &[out_grad], out_grad.type_().element());
                        self.accumulate_grad(args[0], sum, builder);
                    }
                    Func::Vec2 | Func::Vec3 | Func::Vec4 => {
                        assert_eq!(args.len(), out_grad.type_().dimension());
                        for i in 0..args.len() {
                            let idx = builder.const_(Const::Uint32(i as u32));
                            let elem_grad = builder.call(
                                Func::ExtractElement,
                                &[out_grad, idx],
                                out_grad.type_().element(),
                            );
                            self.accumulate_grad(args[i as usize], elem_grad, builder);
                        }
                    }
                    Func::Permute => {
                        // permute(v, i...) => [v[i0], v[i1], ...]
                        let mut v_grad = builder.const_(Const::Zero(args[0].type_().clone()));
                        for (out_idx, in_idx) in original_args[1..].iter().enumerate() {
                            let in_idx = *in_idx;
                            let out_idx = builder.const_(Const::Uint32(out_idx as u32));
                            let out_grad_elem = builder.call(
                                Func::ExtractElement,
                                &[out_grad, out_idx],
                                out_grad.type_().element(),
                            );
                            let v_grad_elem = builder.call(
                                Func::ExtractElement,
                                &[v_grad, in_idx],
                                v_grad.type_().element(),
                            );
                            let v_grad_elem = builder.call(
                                Func::Add,
                                &[v_grad_elem, out_grad_elem],
                                v_grad.type_().element(),
                            );
                            v_grad = builder.call(
                                Func::InsertElement,
                                &[v_grad, v_grad_elem, in_idx],
                                v_grad.type_().clone(),
                            );
                        }
                        self.accumulate_grad(args[0], v_grad, builder);
                    }
                    Func::ExtractElement => {
                        let v_grad = builder.const_(Const::Zero(args[0].type_().clone()));
                        let v_grad = builder.call(
                            Func::InsertElement,
                            &[v_grad, out_grad, original_args[1]],
                            v_grad.type_().clone(),
                        );
                        self.accumulate_grad(args[0], v_grad, builder);
                    }
                    Func::InsertElement => {
                        let zero = builder.const_(Const::Zero(args[1].type_().clone()));
                        let v_grad = builder.call(
                            Func::InsertElement,
                            &[out_grad, zero, original_args[2]],
                            args[0].type_().clone(),
                        );
                        let x_grad = builder.call(
                            Func::ExtractElement,
                            &[out_grad, original_args[2]],
                            args[1].type_().clone(),
                        );
                        self.accumulate_grad(args[0], v_grad, builder);
                        self.accumulate_grad(args[1], x_grad, builder);
                    }
                    Func::Mat => {
                        let col_type = match out_grad.type_().deref() {
                            Type::Matrix(t) => t.column(),
                            _ => unreachable!(),
                        };
                        let mut sum = builder.const_(Const::Zero(args[0].type_().clone()));
                        for i in 0..out_grad.type_().dimension() {
                            let idx = builder.const_(Const::Uint32(i as u32));
                            let col = builder.call(
                                Func::ExtractElement,
                                &[out_grad, idx],
                                col_type.clone(),
                            );
                            let col_sum = builder.call(Func::ReduceSum, &[col], col_type.element());
                            sum = builder.call(Func::Add, &[sum, col_sum], sum.type_().clone());
                        }
                        self.accumulate_grad(args[0], sum, builder);
                    }
                    Func::Mat2 | Func::Mat3 | Func::Mat4 => {
                        let n = out_grad.type_().dimension();
                        assert_eq!(args.len(), n);
                        let col_type = match out_grad.type_().deref() {
                            Type::Matrix(t) => t.column(),
                            _ => unreachable!(),
                        };
                        for i in 0..n {
                            let col = builder.const_(Const::Uint32(i as u32));
                            let col_grad = builder.call(
                                Func::ExtractElement,
                                &[out_grad, col],
                                col_type.clone(),
                            );
                            // for j in 0..n {
                            //     let row = builder.const_(Const::Uint32(j as u32));
                            //     let elem_grad = builder.call(
                            //         Func::ExtractElement,
                            //         &[col_grad, row],
                            //         col_type.element(),
                            //     );
                            //     self.accumulate_grad(args[i * n + j], elem_grad, builder);
                            // }
                            self.accumulate_grad(args[i], col_grad, builder)
                        }
                    }
                    Func::GetElementPtr => panic!("GetElementPtr should be lowered"),
                    Func::Struct => {
                        for (idx, member) in args.iter().enumerate() {
                            let idx = builder.const_(Const::Uint32(idx as u32));
                            let member_grad = builder.call(
                                Func::ExtractElement,
                                &[out_grad, idx],
                                member.type_().clone(),
                            );
                            self.accumulate_grad(*member, member_grad, builder);
                        }
                    }
                    _ => {}
                }
            }
            crate::ir::Instruction::Phi(incomings) => {
                if grad_type.is_none() {
                    return;
                }
                let out_grad = self.grad(node).unwrap();
                for PhiIncoming { value, .. } in incomings.as_ref() {
                    self.accumulate_grad(*value, out_grad, builder);
                }
            }
            crate::ir::Instruction::Loop { .. } => todo!(),
            crate::ir::Instruction::GenericLoop { .. } => todo!(),
            crate::ir::Instruction::Break => todo!(),
            crate::ir::Instruction::Continue => todo!(),
            crate::ir::Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                let cond = self.get_intermediate(*cond);
                let true_branch =
                    self.backward_block(true_branch, IrBuilder::new(self.pools.clone()));
                let false_branch =
                    self.backward_block(false_branch, IrBuilder::new(self.pools.clone()));
                builder.if_(cond, true_branch, false_branch);
            }
            crate::ir::Instruction::Switch {
                value,
                cases,
                default,
            } => {
                let value = self.get_intermediate(*value);
                let cases = cases
                    .as_ref()
                    .iter()
                    .map(|SwitchCase { value, block }| {
                        let block = self.backward_block(block, IrBuilder::new(self.pools.clone()));
                        SwitchCase {
                            value: *value,
                            block,
                        }
                    })
                    .collect::<Vec<_>>();
                let default = self.backward_block(default, IrBuilder::new(self.pools.clone()));
                builder.switch(value, &cases, default);
            }
            crate::ir::Instruction::Comment { .. } => {}
            crate::ir::Instruction::Return(_) => {
                panic!("should not have return in autodiff section")
            }
        }
    }
    fn backward_block(&mut self, block: &BasicBlock, mut builder: IrBuilder) -> Pooled<BasicBlock> {
        for node in block.nodes().iter().rev() {
            self.backward(*node, &mut builder);
        }

        builder.finish()
    }
    fn run(&mut self, block: &BasicBlock) -> Pooled<BasicBlock> {
        let mut builder = IrBuilder::new(self.pools.clone());
        for node in block.nodes().iter().rev() {
            self.backward(*node, &mut builder);
        }
        let mut final_grads = self
            .final_grad
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect::<Vec<_>>();
        final_grads.sort_by_key(|(_, k)| *k);
        for (n, _) in final_grads {
            let mut grad = self
                .grads
                .get(&n)
                .cloned()
                .unwrap_or_else(|| builder.zero_initializer(n.type_().clone()));
            if grad.is_local() {
                grad = builder.call(Func::Load, &[grad], grad.type_().clone());
            }
            builder.call(Func::GradientMarker, &[n, grad], Type::void());
        }
        builder.finish()
    }
}

pub struct Autodiff;
fn ad_transform_block(module: crate::ir::Module) -> crate::ir::Module {
    assert!(
        module.kind == crate::ir::ModuleKind::Block,
        "ad_transform_block should be applied to a block"
    );
    let mut store = StoreIntermediate::new(&module);
    store.run();
    let StoreIntermediate {
        grads,
        final_grad,
        intermediate,
        intermediate_to_node,
        // backward_reachable,
        // forward_reachable,
        ..
    } = store;
    // dbg!(backward_reachable);
    // dbg!(forward_reachable);
    let mut backward = Backward {
        grads,
        final_grad,
        intermediate,
        intermediate_to_node,
        pools: module.pools.clone(),
    };
    // dbg!(&backward.intermediate);
    // dbg!(&backward.grads);
    let fwd = module.entry;

    let bwd = backward.run(&fwd);
    fwd.merge(bwd);

    // let mut display = display::DisplayIR::new();
    // let fwd_src = display.display_ir(&module);
    // print!("{}\n", fwd_src);

    Module {
        kind: ModuleKind::Block,
        entry: fwd,
        pools: module.pools,
    }
}
fn ad_transform_recursive(block: Pooled<BasicBlock>, pools: &CArc<ModulePools>) {
    for node in block.iter() {
        match node.get().instruction.as_ref() {
            Instruction::AdScope { body } => {
                let ad_block = Module {
                    kind: ModuleKind::Block,
                    entry: body.clone(),
                    pools: pools.clone(),
                };
                let ad_block = ToSSA.transform(ad_block);
                let mut backward = None;
                for node in body.iter() {
                    match node.get().instruction.as_ref() {
                        Instruction::Call(f, _) => {
                            if *f == Func::Backward {
                                if backward == None {
                                    backward = Some(node);
                                } else {
                                    panic!("multiple backward calls inside AdScope!");
                                }
                            }
                        }
                        _ => {}
                    }
                }
                let backward = backward.unwrap_or_else(|| {
                    panic!("no backward call inside AdScope!");
                });
                let epilogue = body.split(backward, pools);
                backward.remove();
                let ad_block = ad_transform_block(ad_block);
                assert_eq!(ad_block.entry.ptr, body.ptr);
                body.merge(epilogue);
            }
            Instruction::If {
                cond: _,
                true_branch,
                false_branch,
            } => {
                ad_transform_recursive(*true_branch, pools);
                ad_transform_recursive(*false_branch, pools);
            }
            Instruction::GenericLoop {
                prepare,
                cond: _,
                body,
                update,
            } => {
                ad_transform_recursive(*prepare, pools);
                ad_transform_recursive(*body, pools);
                ad_transform_recursive(*update, pools);
            }
            Instruction::Loop { body, cond: _ } => {
                ad_transform_recursive(*body, pools);
            }
            Instruction::Switch {
                value: _,
                default,
                cases,
            } => {
                ad_transform_recursive(*default, pools);
                for SwitchCase { value: _, block } in cases.iter() {
                    ad_transform_recursive(*block, pools);
                }
            }
            _ => {}
        }
    }
}
impl Transform for Autodiff {
    fn transform(&self, module: crate::ir::Module) -> crate::ir::Module {
        ad_transform_recursive(module.entry, &module.pools);
        module
    }
}
