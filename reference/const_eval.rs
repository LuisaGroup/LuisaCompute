// This file implements a simple constant evaluator which only takes care of constants
// or apparently constant expressions computed from constants.

use crate::ir::{Const, Func, Instruction, NodeRef};
use std::collections::HashMap;

pub(crate) struct ConstEval {
    known: HashMap<NodeRef, Option<Const>>,
}

impl ConstEval {
    fn try_eval(&mut self, node: NodeRef) -> Option<Const> {
        let value = match node.get().instruction.as_ref() {
            Instruction::Const(c) => Some(c.clone()),
            Instruction::Call(func, args) => {
                let const_args = args
                    .iter()
                    .filter_map(|arg| self.eval(*arg))
                    .collect::<Vec<_>>();
                if const_args.len() == args.len() {
                    match func {
                        Func::ZeroInitializer => Some(Const::Zero(node.get().type_.clone())),
                        Func::Cast => None,           // TODO
                        Func::Bitcast => None,        // TODO
                        Func::Pack => None,           // TODO
                        Func::Unpack => None,         // TODO
                        Func::Add => None,            // TODO
                        Func::Sub => None,            // TODO
                        Func::Mul => None,            // TODO
                        Func::Div => None,            // TODO
                        Func::Rem => None,            // TODO
                        Func::BitAnd => None,         // TODO
                        Func::BitOr => None,          // TODO
                        Func::BitXor => None,         // TODO
                        Func::Shl => None,            // TODO
                        Func::Shr => None,            // TODO
                        Func::RotRight => None,       // TODO
                        Func::RotLeft => None,        // TODO
                        Func::Eq => None,             // TODO
                        Func::Ne => None,             // TODO
                        Func::Lt => None,             // TODO
                        Func::Le => None,             // TODO
                        Func::Gt => None,             // TODO
                        Func::Ge => None,             // TODO
                        Func::MatCompMul => None,     // TODO
                        Func::Neg => None,            // TODO
                        Func::Not => None,            // TODO
                        Func::BitNot => None,         // TODO
                        Func::All => None,            // TODO
                        Func::Any => None,            // TODO
                        Func::Select => None,         // TODO
                        Func::Clamp => None,          // TODO
                        Func::Lerp => None,           // TODO
                        Func::Step => None,           // TODO
                        Func::SmoothStep => None,     // TODO
                        Func::Saturate => None,       // TODO
                        Func::Abs => None,            // TODO
                        Func::Min => None,            // TODO
                        Func::Max => None,            // TODO
                        Func::ReduceSum => None,      // TODO
                        Func::ReduceProd => None,     // TODO
                        Func::ReduceMin => None,      // TODO
                        Func::ReduceMax => None,      // TODO
                        Func::Clz => None,            // TODO
                        Func::Ctz => None,            // TODO
                        Func::PopCount => None,       // TODO
                        Func::Reverse => None,        // TODO
                        Func::IsInf => None,          // TODO
                        Func::IsNan => None,          // TODO
                        Func::Acos => None,           // TODO
                        Func::Acosh => None,          // TODO
                        Func::Asin => None,           // TODO
                        Func::Asinh => None,          // TODO
                        Func::Atan => None,           // TODO
                        Func::Atan2 => None,          // TODO
                        Func::Atanh => None,          // TODO
                        Func::Cos => None,            // TODO
                        Func::Cosh => None,           // TODO
                        Func::Sin => None,            // TODO
                        Func::Sinh => None,           // TODO
                        Func::Tan => None,            // TODO
                        Func::Tanh => None,           // TODO
                        Func::Exp => None,            // TODO
                        Func::Exp2 => None,           // TODO
                        Func::Exp10 => None,          // TODO
                        Func::Log => None,            // TODO
                        Func::Log2 => None,           // TODO
                        Func::Log10 => None,          // TODO
                        Func::Powi => None,           // TODO
                        Func::Powf => None,           // TODO
                        Func::Sqrt => None,           // TODO
                        Func::Rsqrt => None,          // TODO
                        Func::Ceil => None,           // TODO
                        Func::Floor => None,          // TODO
                        Func::Fract => None,          // TODO
                        Func::Trunc => None,          // TODO
                        Func::Round => None,          // TODO
                        Func::Fma => None,            // TODO
                        Func::Copysign => None,       // TODO
                        Func::Cross => None,          // TODO
                        Func::Dot => None,            // TODO
                        Func::OuterProduct => None,   // TODO
                        Func::Length => None,         // TODO
                        Func::LengthSquared => None,  // TODO
                        Func::Normalize => None,      // TODO
                        Func::Faceforward => None,    // TODO
                        Func::Distance => None,       // TODO
                        Func::Reflect => None,        // TODO
                        Func::Determinant => None,    // TODO
                        Func::Transpose => None,      // TODO
                        Func::Inverse => None,        // TODO
                        Func::Vec => None,            // TODO
                        Func::Vec2 => None,           // TODO
                        Func::Vec3 => None,           // TODO
                        Func::Vec4 => None,           // TODO
                        Func::Permute => None,        // TODO
                        Func::InsertElement => None,  // TODO
                        Func::ExtractElement => None, // TODO
                        Func::Struct => None,         // TODO
                        Func::Array => None,          // TODO
                        Func::Mat => None,            // TODO
                        Func::Mat2 => None,           // TODO
                        Func::Mat3 => None,           // TODO
                        Func::Mat4 => None,           // TODO
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        };
        self.known.insert(node, value.clone());
        value
    }
}

impl ConstEval {
    pub fn new() -> Self {
        Self {
            known: HashMap::new(),
        }
    }

    pub fn eval(&mut self, node: NodeRef) -> Option<Const> {
        if let Some(result) = self.known.get(&node).cloned() {
            result
        } else {
            self.try_eval(node)
        }
    }
}
