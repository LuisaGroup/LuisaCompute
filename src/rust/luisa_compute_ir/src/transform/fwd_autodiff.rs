use std::ops::Deref;

use indexmap::IndexMap;
use smallvec::{smallvec, SmallVec};

use crate::{
    ir::{BasicBlock, Const, Func, IrBuilder, ModulePools, Node, NodeRef},
    *,
};

use super::Transform;
type NodeVec = SmallVec<[NodeRef; 4]>;

#[derive(Clone, Debug)]
struct Dual {
    var: NodeRef,
    grad: NodeVec,
}

// Inplace transform
struct ForwardAdTransform {
    n_grads: usize,
    duals: IndexMap<NodeRef, Dual>,
    pools: CArc<ModulePools>,
}
impl ForwardAdTransform {
    fn grads(&self, node: NodeRef) -> &NodeVec {
        &self.duals[&node].grad
    }
    fn zero_grad(&mut self, node: NodeRef, builder: &mut IrBuilder) {
        assert!(!self.duals.contains_key(&node));
        let zero = builder.const_(Const::Zero(node.type_().clone()));
        let grads = (0..self.n_grads).map(|_| zero.clone()).collect::<NodeVec>();
        self.duals.insert(
            node,
            Dual {
                var: node,
                grad: grads,
            },
        );
    }
    fn create_grad(&mut self, node: NodeRef, out_grads: &[NodeRef], builder: &mut IrBuilder) {
        assert!(!self.duals.contains_key(&node));
        let grads = (0..self.n_grads)
            .map(|_| builder.local_zero_init(out_grads[0].type_().clone()))
            .collect::<NodeVec>();
        for (out_grad, grad) in out_grads.iter().zip(grads.iter()) {
            builder.call(Func::AccGrad, &[*grad, *out_grad], Type::void());
        }
        self.duals.insert(
            node,
            Dual {
                var: node,
                grad: grads,
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
        let type_ = out.type_();
        let n_grads = self.n_grads;
        match f {
            Func::Add => {
                let grads = self.create_call(
                    builder,
                    Func::Add,
                    &[self.grads(args[0]), self.grads(args[1])],
                    type_,
                );
                self.create_grad(out, &grads, builder);
            }
            Func::Sub => {
                let grads = self.create_call(
                    builder,
                    Func::Sub,
                    &[self.grads(args[0]), self.grads(args[1])],
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
                        &[self.grads(args[0]), &[args[1]]],
                        type_,
                    );
                    let rhs = self.create_call(
                        builder,
                        Func::Mul,
                        &[self.grads(args[1]), &[args[0]]],
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
                    &[self.grads(args[0]), &[args[1]]],
                    type_,
                );
                let rhs = self.create_call(
                    builder,
                    Func::Mul,
                    &[self.grads(args[1]), &[args[0]]],
                    type_,
                );
                let sqr = self.create_call(builder, Func::Mul, &[&[args[1]], &[args[1]]], type_);
                let numerator = self.create_call(builder, Func::Sub, &[&lhs, &rhs], type_);
                let denominator = self.create_call(builder, Func::Div, &[&sqr, &[args[1]]], type_);
                let grads =
                    self.create_call(builder, Func::Div, &[&numerator, &denominator], type_);
                self.create_grad(out, &grads, builder);
            }
            Func::MatCompMul => {
                let lhs = self.create_call(
                    builder,
                    Func::MatCompMul,
                    &[self.grads(args[0]), &[args[1]]],
                    type_,
                );
                let rhs = self.create_call(
                    builder,
                    Func::MatCompMul,
                    &[self.grads(args[1]), &[args[0]]],
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
                    self.create_call(builder, Func::Mul, &[grad_x, &neg_one_over_sqrt], type_);
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
                    self.create_call(builder, Func::Mul, &[grad_x, &neg_one_over_sqrt], type_);
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
                    self.create_call(builder, Func::Mul, &[grad_x, &one_over_sqrt], type_);
                self.create_grad(out, &out_grad, builder);
            }
            Func::Sin => {
                let cos = self.create_call(builder, Func::Cos, &[&[args[0]]], type_);
                let grads =
                    self.create_call(builder, Func::Mul, &[&cos, self.grads(args[0])], type_);
                self.create_grad(out, &grads, builder);
            }
            Func::Cos => {
                let sin = self.create_call(builder, Func::Sin, &[&[args[0]]], type_);
                let neg_sin = self.create_call(builder, Func::Neg, &[&sin], type_);
                let grads =
                    self.create_call(builder, Func::Mul, &[&neg_sin, self.grads(args[0])], type_);
                self.create_grad(out, &grads, builder);
            }
            Func::Tan => {
                let grad_x = self.grads(args[0]);
                let cos_x = self.create_call(builder, Func::Cos, &[&[args[0]]], type_);
                let sqr_cos_x = self.create_call(builder, Func::Mul, &[&cos_x, &cos_x], type_);
                let grad = self.create_call(builder, Func::Div, &[grad_x, &sqr_cos_x], type_);
                self.create_grad(out, &grad, builder);
            }
            Func::ExtractElement => {
                let grads = self.create_call(
                    builder,
                    Func::ExtractElement,
                    &[self.grads(args[0]), self.grads(args[1]), &[args[2]]],
                    type_,
                );
                self.create_grad(out, &grads, builder);
            }
            Func::InsertElement => {
                let grads = self.create_call(
                    builder,
                    Func::InsertElement,
                    &[self.grads(args[0]), self.grads(args[1]), &[args[2]]],
                    type_,
                );
                self.create_grad(out, &grads, builder);
            }
            _ => {
                self.zero_grad(out, builder);
            }
        }
    }
    fn transform_block(&mut self, block: &Pooled<BasicBlock>, mut builder: IrBuilder) {
        for node in block.iter() {
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
                ir::Instruction::Const(_) => todo!(),
                ir::Instruction::Update { var, value } => todo!(),
                ir::Instruction::Call(_, _) => todo!(),
                ir::Instruction::Phi(_) => todo!(),
                ir::Instruction::Return(_) => todo!(),
                ir::Instruction::Loop { body, cond } => todo!(),
                ir::Instruction::GenericLoop {
                    prepare,
                    cond,
                    body,
                    update,
                } => todo!(),
                ir::Instruction::Break => todo!(),
                ir::Instruction::Continue => todo!(),
                ir::Instruction::If {
                    cond,
                    true_branch,
                    false_branch,
                } => todo!(),
                ir::Instruction::Switch {
                    value,
                    default,
                    cases,
                } => todo!(),
                ir::Instruction::AdScope { body, forward } => {
                    todo!("Nested AD scope is not supported");
                },
                ir::Instruction::RayQuery {
                    ray_query,
                    on_triangle_hit,
                    on_procedural_hit,
                } => panic!("RayQuery not supported in AD. Please recompute ray intersection after the RayQuery result is obtained"),
                ir::Instruction::AdDetach(block) => {

                },
                ir::Instruction::Comment(_) => {}
            }
        }
    }
}
pub(crate) struct FwdAutodiff;
impl Transform for FwdAutodiff {
    fn transform(&self, mut module: crate::ir::Module) -> crate::ir::Module {
        todo!()
    }
}