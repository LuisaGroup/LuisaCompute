use std::{collections::HashSet, ops::Deref};

use indexmap::IndexMap;
use smallvec::{smallvec, SmallVec};

use crate::{
    ir::{
        BasicBlock, Const, Func, Instruction, IrBuilder, Module, ModuleFlags, ModuleKind,
        ModulePools, Node, NodeRef, PhiIncoming, SwitchCase,
    },
    *,
};

use super::Transform;
type NodeVec = SmallVec<[NodeRef; 4]>;

#[derive(Clone, Debug)]
struct Dual {
    var: NodeRef,
    grad: NodeVec,
    loaded_grad: NodeVec,
}

// Inplace transform
struct ForwardAdTransform {
    n_grads: usize,
    duals: RefCell<IndexMap<NodeRef, Dual>>,
    pools: CArc<ModulePools>,
    locally_defined_nodes: HashSet<NodeRef>,
}
impl ForwardAdTransform {
    fn new(
        n_grads: usize,
        pools: &CArc<ModulePools>,
        locally_defined_nodes: HashSet<NodeRef>,
    ) -> Self {
        Self {
            n_grads,
            duals: RefCell::new(IndexMap::new()),
            pools: pools.clone(),
            locally_defined_nodes,
        }
    }
    fn grads(&self, node: NodeRef) -> NodeVec {
        assert!(!context::is_type_equal(node.type_(), &Type::void()));
        {
            let duals = self.duals.borrow();
            if !self.locally_defined_nodes.contains(&node) {
                if !duals.contains_key(&node) {
                    drop(duals);
                    let mut builder = IrBuilder::new(self.pools.clone());
                    builder.set_insert_point(node);
                    self.zero_grad(node, &mut builder);
                }
            }
        }

        let duals = self.duals.borrow();
        assert!(
            duals.contains_key(&node),
            "node {:?} {:?} has no grad",
            node,
            node.get().instruction.as_ref()
        );
        duals[&node].loaded_grad.clone()
    }
    fn grad_vars(&self, node: NodeRef) -> NodeVec {
        assert!(!context::is_type_equal(node.type_(), &Type::void()));
        let duals = self.duals.borrow();
        duals[&node].grad.clone()
    }
    fn zero_grad(&self, node: NodeRef, builder: &mut IrBuilder) {
        assert!(!context::is_type_equal(node.type_(), &Type::void()));
        let zero = builder.const_(Const::Zero(node.type_().clone()));
        let grads = (0..self.n_grads).map(|_| zero.clone()).collect::<NodeVec>();
        self.create_grad(node, &grads, builder);
    }
    fn create_grad(&self, node: NodeRef, out_grads: &[NodeRef], builder: &mut IrBuilder) {
        let mut duals = self.duals.borrow_mut();
        assert!(
            !duals.contains_key(&node),
            "node {:?} {:?} already has a grad",
            node,
            node.get().instruction.as_ref()
        );
        // println!(
        //     "adding grad for {:?} {:?}",
        //     node,
        //     node.get().instruction.as_ref()
        // );
        let grads = (0..self.n_grads)
            .map(|_| builder.local_zero_init(out_grads[0].type_().clone()))
            .collect::<NodeVec>();
        for (out_grad, grad) in out_grads.iter().zip(grads.iter()) {
            builder.update(grad.clone(), out_grad.clone());
        }
        let loaded_grads = out_grads.into();
        duals.insert(
            node,
            Dual {
                var: node,
                grad: grads,
                loaded_grad: loaded_grads,
            },
        );
    }
    fn create_call(
        &self,
        builder: &mut IrBuilder,
        f: Func,
        args: &[&[NodeRef]],
        type_: &CArc<Type>,
    ) -> NodeVec {
        let n = args.iter().map(|x| x.len()).max().unwrap();
        for a in args {
            assert!(a.len() == n || a.len() == 1);
        }
        let mut ret = NodeVec::new();
        for i in 0..n {
            let mut args_i = NodeVec::new();
            for j in 0..args.len() {
                let arg = if args[j].len() == 1 {
                    args[j][0]
                } else {
                    args[j][i]
                };
                args_i.push(arg);
            }
            ret.push(builder.call(f.clone(), &args_i, type_.clone()));
        }
        ret
    }
    fn transform_call(
        &mut self,
        out: NodeRef,
        f: &Func,
        args: &[NodeRef],
        builder: &mut IrBuilder,
    ) {
        // dbg!(out.get().instruction.as_ref());
        let type_ = out.type_();
        let n_grads = self.n_grads;
        match f {
            Func::PropagateGrad => {
                let var = args[0];
                let grads = &args[1..];
                assert_eq!(grads.len(), n_grads);
                self.create_grad(var, grads, builder);
                out.remove();
            }
            Func::OutputGrad => {
                let idx = args[1].get_i32();
                let var = args[0];
                let duals = self.grad_vars(var);
                let grad = duals[idx as usize];
                assert!(grad.is_lvalue());
                #[allow(unused_variables)]
                let f = ();
                #[allow(unused_variables)]
                let args = ();
                // this operation is unsafe as it invalidates f and args
                #[allow(unused_unsafe)]
                unsafe {
                    let inst = CArc::get_mut(&mut out.get_mut().instruction).unwrap();
                    *inst = Instruction::Call(Func::Load, CBoxedSlice::new(vec![grad]));
                }
            }
            Func::Add => {
                let grads = self.create_call(
                    builder,
                    Func::Add,
                    &[&self.grads(args[0]), &self.grads(args[1])],
                    type_,
                );
                self.create_grad(out, &grads, builder);
            }
            Func::Sub => {
                let grads = self.create_call(
                    builder,
                    Func::Sub,
                    &[&self.grads(args[0]), &self.grads(args[1])],
                    type_,
                );
                self.create_grad(out, &grads, builder);
            }
            Func::Mul => match (args[0].type_().deref(), args[1].type_().deref()) {
                (Type::Matrix(matrix_type), Type::Vector(vector_type)) => {
                    assert_eq!(matrix_type.dimension, vector_type.length);
                    todo!()
                }
                (Type::Matrix(m1), Type::Matrix(m2)) => {
                    assert_eq!(m1.dimension, m2.dimension);
                    todo!()
                }
                _ => {
                    let lhs = self.create_call(
                        builder,
                        Func::Mul,
                        &[&self.grads(args[0]), &[args[1]]],
                        type_,
                    );
                    let rhs = self.create_call(
                        builder,
                        Func::Mul,
                        &[&self.grads(args[1]), &[args[0]]],
                        type_,
                    );
                    let grads = self.create_call(builder, Func::Add, &[&lhs, &rhs], type_);
                    self.create_grad(out, &grads, builder);
                }
            },
            Func::Div => {
                let lhs = self.create_call(
                    builder,
                    Func::Mul,
                    &[&self.grads(args[0]), &[args[1]]],
                    type_,
                );
                let rhs = self.create_call(
                    builder,
                    Func::Mul,
                    &[&self.grads(args[1]), &[args[0]]],
                    type_,
                );
                let sqr = self.create_call(builder, Func::Mul, &[&[args[1]], &[args[1]]], type_);
                let numerator = self.create_call(builder, Func::Sub, &[&lhs, &rhs], type_);
                let grads = self.create_call(builder, Func::Div, &[&numerator, &sqr], type_);
                self.create_grad(out, &grads, builder);
            }
            Func::MatCompMul => {
                let lhs = self.create_call(
                    builder,
                    Func::MatCompMul,
                    &[&self.grads(args[0]), &[args[1]]],
                    type_,
                );
                let rhs = self.create_call(
                    builder,
                    Func::MatCompMul,
                    &[&self.grads(args[1]), &[args[0]]],
                    type_,
                );
                let grads = self.create_call(builder, Func::Add, &[&lhs, &rhs], type_);
                self.create_grad(out, &grads, builder);
            }
            Func::Acos => {
                // out = acos(x)
                // => δ(x) = -1 / sqrt(1 - x^2)
                let x = [args[0]];
                let grad_x = self.grads(args[0]);
                let x2 = self.create_call(builder, Func::Mul, &[&x, &x], type_);
                let one = [builder.const_(Const::One(type_.clone()))];
                let one_minus_x2 = self.create_call(builder, Func::Sub, &[&one, &x2], type_);
                let sqrt = self.create_call(builder, Func::Sqrt, &[&one_minus_x2], type_);
                let one_over_sqrt = self.create_call(builder, Func::Div, &[&one, &sqrt], type_);
                let neg_one_over_sqrt =
                    self.create_call(builder, Func::Neg, &[&one_over_sqrt], type_);
                let out_grad =
                    self.create_call(builder, Func::Mul, &[&grad_x, &neg_one_over_sqrt], type_);
                self.create_grad(out, &out_grad, builder);
            }
            Func::Acosh => {
                // out = acosh(x)
                // => δ(x) = 1 / sqrt(x^2 - 1)
                let x = [args[0]];
                let grad_x = self.grads(args[0]);
                let x2 = self.create_call(builder, Func::Mul, &[&x, &x], type_);
                let one = [builder.const_(Const::One(type_.clone()))];
                let one_minus_x2 = self.create_call(builder, Func::Sub, &[&x2, &one], type_);
                let sqrt = self.create_call(builder, Func::Sqrt, &[&one_minus_x2], type_);
                let one_over_sqrt = self.create_call(builder, Func::Div, &[&one, &sqrt], type_);
                let neg_one_over_sqrt =
                    self.create_call(builder, Func::Neg, &[&one_over_sqrt], type_);
                let out_grad =
                    self.create_call(builder, Func::Mul, &[&grad_x, &neg_one_over_sqrt], type_);
                self.create_grad(out, &out_grad, builder);
            }
            Func::Asin => {
                // out = asin(x)
                // => δ(x) = 1 / sqrt(1 - x^2)
                let x = [args[0]];
                let grad_x = self.grads(args[0]);
                let x2 = self.create_call(builder, Func::Mul, &[&x, &x], type_);
                let one = [builder.const_(Const::One(type_.clone()))];
                let one_minus_x2 = self.create_call(builder, Func::Sub, &[&x2, &one], type_);
                let sqrt = self.create_call(builder, Func::Sqrt, &[&one_minus_x2], type_);
                let one_over_sqrt = self.create_call(builder, Func::Div, &[&one, &sqrt], type_);
                let out_grad =
                    self.create_call(builder, Func::Mul, &[&grad_x, &one_over_sqrt], type_);
                self.create_grad(out, &out_grad, builder);
            }
            Func::Sin => {
                let cos = self.create_call(builder, Func::Cos, &[&[args[0]]], type_);
                let grads =
                    self.create_call(builder, Func::Mul, &[&cos, &self.grads(args[0])], type_);
                self.create_grad(out, &grads, builder);
            }
            Func::Cos => {
                let sin = self.create_call(builder, Func::Sin, &[&[args[0]]], type_);
                let neg_sin = self.create_call(builder, Func::Neg, &[&sin], type_);
                let grads =
                    self.create_call(builder, Func::Mul, &[&neg_sin, &self.grads(args[0])], type_);
                self.create_grad(out, &grads, builder);
            }
            Func::Tan => {
                let grad_x = self.grads(args[0]);
                let cos_x = self.create_call(builder, Func::Cos, &[&[args[0]]], type_);
                let sqr_cos_x = self.create_call(builder, Func::Mul, &[&cos_x, &cos_x], type_);
                let grad = self.create_call(builder, Func::Div, &[&grad_x, &sqr_cos_x], type_);
                self.create_grad(out, &grad, builder);
            }
            Func::ExtractElement => {
                let grads = self.create_call(
                    builder,
                    Func::ExtractElement,
                    &[&self.grads(args[0]), &self.grads(args[1]), &[args[2]]],
                    type_,
                );
                self.create_grad(out, &grads, builder);
            }
            Func::InsertElement => {
                let grads = self.create_call(
                    builder,
                    Func::InsertElement,
                    &[&self.grads(args[0]), &self.grads(args[1]), &[args[2]]],
                    type_,
                );
                self.create_grad(out, &grads, builder);
            }
            Func::GetElementPtr => {
                let grads = self.create_call(
                    builder,
                    Func::GetElementPtr,
                    &[&self.grads(args[0]), &self.grads(args[1]), &[args[2]]],
                    type_,
                );
                // GEP is special as it is technically not a value
                let mut duals = self.duals.borrow_mut();
                duals.insert(
                    out,
                    Dual {
                        var: out,
                        grad: grads.clone(),
                        loaded_grad: grads.clone(),
                    },
                );
            }
            Func::Vec => {
                let out_grads = self.create_call(builder, Func::Vec, &[&[args[0]]], type_);
                self.create_grad(out, &out_grads, builder);
            }
            Func::Vec2 => {
                let out_grads =
                    self.create_call(builder, Func::Vec2, &[&[args[0]], &[args[1]]], type_);
                self.create_grad(out, &out_grads, builder);
            }
            Func::Vec3 => {
                let out_grads = self.create_call(
                    builder,
                    Func::Vec3,
                    &[&[args[0]], &[args[1]], &[args[2]]],
                    type_,
                );
                self.create_grad(out, &out_grads, builder);
            }
            Func::Vec4 => {
                let out_grads = self.create_call(
                    builder,
                    Func::Vec4,
                    &[&[args[0]], &[args[1]], &[args[2]], &[args[3]]],
                    type_,
                );
                self.create_grad(out, &out_grads, builder);
            }
            Func::Mat => {
                let out_grads = self.create_call(builder, Func::Mat, &[&[args[0]]], type_);
                self.create_grad(out, &out_grads, builder);
            }
            Func::Mat2 => {
                let out_grads =
                    self.create_call(builder, Func::Mat2, &[&[args[0]], &[args[1]]], type_);
                self.create_grad(out, &out_grads, builder);
            }
            Func::Mat3 => {
                let out_grads = self.create_call(
                    builder,
                    Func::Mat3,
                    &[&[args[0]], &[args[1]], &[args[2]]],
                    type_,
                );
                self.create_grad(out, &out_grads, builder);
            }
            Func::Mat4 => {
                let out_grads = self.create_call(
                    builder,
                    Func::Mat4,
                    &[&[args[0]], &[args[1]], &[args[2]], &[args[3]]],
                    type_,
                );
                self.create_grad(out, &out_grads, builder);
            }
            Func::Dot => {
                let lhs = self.create_call(
                    builder,
                    Func::Dot,
                    &[&self.grads(args[0]), &[args[1]]],
                    type_,
                );
                let rhs = self.create_call(
                    builder,
                    Func::Dot,
                    &[&self.grads(args[1]), &[args[0]]],
                    type_,
                );
                let grads = self.create_call(builder, Func::Add, &[&lhs, &rhs], type_);
                self.create_grad(out, &grads, builder);
            }
            Func::Cross => {
                let lhs = self.create_call(
                    builder,
                    Func::Cross,
                    &[&self.grads(args[0]), &[args[1]]],
                    type_,
                );
                let rhs = self.create_call(
                    builder,
                    Func::Cross,
                    &[&self.grads(args[1]), &[args[0]]],
                    type_,
                );
                let grads = self.create_call(builder, Func::Add, &[&lhs, &rhs], type_);
                self.create_grad(out, &grads, builder);
            }
            Func::Cast => {
                let grads = self.create_call(builder, Func::Cast, &[&self.grads(args[0])], type_);
                self.create_grad(out, &grads, builder);
            }
            Func::Le
            | Func::Lt
            | Func::Ge
            | Func::Gt
            | Func::Eq
            | Func::Ne
            | Func::BufferRead
            | Func::BufferSize
            | Func::BufferWrite
            | Func::Shl
            | Func::Shr
            | Func::BitAnd
            | Func::BitOr
            | Func::BitNot
            | Func::BitXor
            | Func::Clz
            | Func::Ctz
            | Func::PopCount
            | Func::Reverse
            | Func::IsInf
            | Func::IsNan
            | Func::WarpIsFirstActiveLane
            | Func::WarpFirstActiveLane
            | Func::WarpActiveAllEqual
            | Func::WarpActiveBitAnd
            | Func::WarpActiveBitOr
            | Func::WarpActiveBitXor
            | Func::WarpActiveCountBits
            | Func::WarpActiveMax
            | Func::WarpActiveMin
            | Func::WarpActiveProduct
            | Func::WarpActiveSum
            | Func::WarpActiveAll
            | Func::WarpActiveAny
            | Func::WarpActiveBitMask
            | Func::WarpPrefixCountBits
            | Func::WarpPrefixSum
            | Func::WarpPrefixProduct
            | Func::WarpReadLaneAt
            | Func::WarpReadFirstLane
            | Func::AtomicCompareExchange
            | Func::AtomicExchange
            | Func::AtomicFetchAdd
            | Func::AtomicFetchSub
            | Func::AtomicFetchAnd
            | Func::AtomicFetchOr
            | Func::AtomicFetchXor
            | Func::AtomicFetchMin
            | Func::AtomicFetchMax
            | Func::ShaderExecutionReorder
            | Func::Assert(_)
            | Func::Assume
            | Func::Unreachable(_)
            | Func::Unpack
            | Func::Pack
            | Func::CpuCustomOp(_)
            | Func::Texture2dRead
            | Func::Texture2dWrite
            | Func::Texture2dSize
            | Func::Texture3dRead
            | Func::Texture3dWrite
            | Func::Texture3dSize
            | Func::BindlessTexture2dSample
            | Func::BindlessTexture2dSampleLevel
            | Func::BindlessTexture2dSampleGrad
            | Func::BindlessTexture2dSampleGradLevel
            | Func::BindlessTexture3dSample
            | Func::BindlessTexture3dSampleLevel
            | Func::BindlessTexture3dSampleGrad
            | Func::BindlessTexture3dSampleGradLevel
            | Func::BindlessTexture2dRead
            | Func::BindlessTexture3dRead
            | Func::BindlessTexture2dReadLevel
            | Func::BindlessTexture3dReadLevel
            | Func::BindlessTexture2dSize
            | Func::BindlessTexture3dSize
            | Func::BindlessTexture2dSizeLevel
            | Func::BindlessTexture3dSizeLevel
            | Func::BindlessBufferRead
            | Func::BindlessBufferSize
            | Func::BindlessBufferType
            | Func::BindlessByteBufferRead => {
                if !context::is_type_equal(out.type_(), &Type::void()) {
                    self.zero_grad(out, builder);
                }
            }
            _ => todo!("{:?}", f),
        }
    }
    fn transform_node(&mut self, node: NodeRef, builder: &mut IrBuilder) {
        builder.set_insert_point(node);
        let inst = node.get().instruction.as_ref();
        match inst {
            ir::Instruction::Buffer => {}
            ir::Instruction::Bindless => {}
            ir::Instruction::Texture2D => {}
            ir::Instruction::Texture3D => {}
            ir::Instruction::Accel => {}
            ir::Instruction::Shared => todo!(),
            ir::Instruction::Uniform => todo!(),
            ir::Instruction::Local { init } => todo!(),
            ir::Instruction::Argument { by_value } => todo!(),
            ir::Instruction::UserData(_) => todo!(),
            ir::Instruction::Invalid => todo!(),
            ir::Instruction::Const(_) => {
                self.zero_grad(node, builder);
            },
            ir::Instruction::Update { var, value } => todo!(),
            ir::Instruction::Call(f, args) => {
                self.transform_call(node, &f, args, builder);
            }
            ir::Instruction::Phi(incomings) => {
                let mut incoming_grads = vec![];
                for incoming in incomings.iter() {
                    let value = incoming.value;
                    let grads = self.grads(value);
                    incoming_grads.push(grads);
                }
                let mut phi_grads = NodeVec::new();

                // generate a phi for each grad
                for i in 0..self.n_grads {
                    let mut grad_incomings = vec![];
                    for j in 0..incomings.len() {
                        grad_incomings.push(PhiIncoming {
                            value:incoming_grads[j][i],
                            block:incomings[j].block,
                        });
                    }
                    let phi_grad = builder.phi(&grad_incomings, node.type_().clone());
                    phi_grads.push(phi_grad);
                }
                self.create_grad(node, &phi_grads, builder);
            },
            ir::Instruction::Return(_) => todo!(),
            ir::Instruction::Loop { body, cond } => {
                self.transform_block(body, builder);
                self.transform_node(*cond, builder);
            }
            ir::Instruction::GenericLoop {
                prepare,
                cond:_,
                body,
                update,
            } => {
                self.transform_block(prepare, builder);
                self.transform_block(body, builder);
                self.transform_block(update, builder);
            },
            ir::Instruction::Break => {},
            ir::Instruction::Continue => {},
            ir::Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                self.transform_block(true_branch, builder);
                self.transform_block(false_branch, builder);
            }
            ir::Instruction::Switch {
                value,
                default,
                cases,
            } => {
                self.transform_block(default, builder);
                for SwitchCase { value:_, block } in cases.iter() {
                    self.transform_block(block, builder);
                }
            },
            ir::Instruction::AdScope { .. } => {
                todo!("Nested AD scope is not supported");
            },
            ir::Instruction::RayQuery {
                ..
            } => panic!("RayQuery not supported in AD. Please recompute ray intersection after the RayQuery result is obtained"),
            ir::Instruction::AdDetach(block) => {
                todo!()
            },
            ir::Instruction::Comment(_) => {}
            ir::Instruction::Print{..}=>{}
        }
    }
    fn transform_block(&mut self, block: &Pooled<BasicBlock>, builder: &mut IrBuilder) {
        let nodes = block.nodes();
        for node in nodes {
            self.transform_node(node, builder);
        }
    }
}
pub(crate) struct FwdAutodiff;

fn ad_transform_block(module: crate::ir::Module, n_grads: usize) {
    let nodes = module.collect_nodes().into_iter().collect();
    let mut transform = ForwardAdTransform::new(n_grads, &module.pools, nodes);
    transform.transform_block(&module.entry, &mut IrBuilder::new(transform.pools.clone()));
}
fn ad_transform_recursive(block: Pooled<BasicBlock>, pools: &CArc<ModulePools>) {
    let nodes = block.nodes();
    let mut i = 0;
    while i < nodes.len() {
        let node = nodes[i];
        match node.get().instruction.as_ref() {
            Instruction::AdScope {
                body,
                forward,
                n_forward_grads,
            } => {
                if !*forward {
                    i += 1;
                    continue;
                }
                let ad_block = Module {
                    kind: ModuleKind::Block,
                    entry: body.clone(),
                    pools: pools.clone(),
                    flags: ModuleFlags::NONE,
                };
                ad_transform_block(ad_block, *n_forward_grads);
                node.remove();
                block.merge(*body);
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
            Instruction::Call(f, ..) => match f {
                Func::Callable(callable) => {
                    let callable = &callable.0;
                    if callable
                        .module
                        .flags
                        .contains(ModuleFlags::REQUIRES_FWD_AD_TRANSFORM)
                    {
                        ad_transform_recursive(callable.module.entry, pools);
                    }
                }
                _ => {}
            },
            _ => {}
        }
        i += 1;
    }
}

impl Transform for FwdAutodiff {
    fn transform(&self, mut module: crate::ir::Module) -> crate::ir::Module {
        log::debug!("FwdAutodiff transform");
        // {
        //     println!("Before AD:");
        //     let debug = crate::ir::debug::luisa_compute_ir_dump_human_readable(&module);
        //     let debug = std::ffi::CString::new(debug.as_ref()).unwrap();
        //     println!("{}", debug.to_str().unwrap());
        // }
        ad_transform_recursive(module.entry, &module.pools);
        // {
        //     let debug = crate::ir::debug::luisa_compute_ir_dump_human_readable(&module);
        //     let debug = std::ffi::CString::new(debug.as_ref()).unwrap();
        //     println!("After AD:\n{}", debug.to_str().unwrap());
        // }
        module.flags.remove(ModuleFlags::REQUIRES_FWD_AD_TRANSFORM);
        module
    }
}
