use std::{
    any::TypeId,
    borrow::BorrowMut,
    cell::RefCell,
    collections::{HashMap, HashSet},
    hash::Hash,
};

use super::autodiff::grad_type_of;
use crate::{context, ir::*, CBoxedSlice, CSlice, TypeOf};
use gc::Gc;
use lazy_static::lazy_static;
use parking_lot::RwLock;

use super::Transform;
struct Validator {}
fn check_types_equal(types: &[Gc<Type>]) -> bool {
    let mut iter = types.iter();
    let first = iter.next().unwrap();
    iter.all(|t| t == first && *t != Type::void())
}
fn _vector_compatible(a: &VectorType, b: &VectorType) -> bool {
    if a.length != b.length {
        return false;
    }
    match (&a.element, &b.element) {
        (VectorElementType::Scalar(_), VectorElementType::Scalar(_)) => true,
        (VectorElementType::Vector(a), VectorElementType::Vector(b)) => _vector_compatible(&a, &b),
        _ => false,
    }
}
fn _matrix_compatible(a: &MatrixType, b: &MatrixType) -> bool {
    if a.dimension != b.dimension {
        return false;
    }
    match (&a.element, &b.element) {
        (VectorElementType::Scalar(_), VectorElementType::Scalar(_)) => true,
        (VectorElementType::Vector(a), VectorElementType::Vector(b)) => _vector_compatible(&a, &b),
        _ => false,
    }
}
fn vector_compatible(a: Gc<Type>, b: Gc<Type>) -> bool {
    match (a.as_ref(), b.as_ref()) {
        (Type::Vector(a), Type::Vector(b)) => _vector_compatible(a, b),
        (Type::Matrix(a), Type::Matrix(b)) => _matrix_compatible(a, b),
        _ => false,
    }
}
impl Validator {
    fn new() -> Self {
        Self {}
    }
    fn check_block(&mut self, block: &BasicBlock) {
        let mut cur = block.first.get().next;
        while cur != block.last {
            self.check_node(cur);
            cur = cur.get().next;
        }
    }
    fn check_node(&mut self, node: NodeRef) {
        let node = node.get();
        let type_ = node.type_;
        let uvec3_ty = context::register_type(Type::Vector(VectorType {
            element: VectorElementType::Scalar(Primitive::Uint32),
            length: 3,
        }));
        match node.instruction.as_ref() {
            Instruction::Buffer => todo!(),
            Instruction::Bindless => todo!(),
            Instruction::Texture2D => todo!(),
            Instruction::Texture3D => todo!(),
            Instruction::Accel => todo!(),
            Instruction::Shared => todo!(),
            Instruction::Uniform => todo!(),
            Instruction::Local { init } => {
                assert!(check_types_equal(&[type_, init.type_()]));
            }
            Instruction::Argument { by_value: _ } => {
                assert!(check_types_equal(&[type_]));
            }
            Instruction::UserData(_) => {}
            Instruction::Invalid => {
                panic!("invalid instruction found");
            }
            Instruction::Const(c) => {
                let const_type = match c {
                    Const::Zero(t) => *t,
                    Const::Bool(_) => <bool as TypeOf>::type_(),
                    Const::Int32(_) => <i32 as TypeOf>::type_(),
                    Const::Uint32(_) => <u32 as TypeOf>::type_(),
                    Const::Int64(_) => <i64 as TypeOf>::type_(),
                    Const::Uint64(_) => <u64 as TypeOf>::type_(),
                    Const::Float32(_) => <f32 as TypeOf>::type_(),
                    Const::Float64(_) => <f64 as TypeOf>::type_(),
                    Const::Generic(_, t) => *t,
                };
                assert_eq!(type_, const_type);
            }
            Instruction::Update { var, value } => {
                assert_eq!(type_, Type::void());
                assert_eq!(value.type_(), var.type_());
            }
            Instruction::Call(func, args) => {
                let args = args.as_ref();
                macro_rules! check_binop_float_int {
                    () => {{
                        assert_eq!(args.len(), 2);
                        check_types_equal(&[type_, args[0].type_(), args[1].type_()]);
                        assert!(!type_.is_bool());
                    }};
                }
                macro_rules! check_binop_same {
                    () => {{
                        assert_eq!(args.len(), 2);
                        check_types_equal(&[type_, args[0].type_(), args[1].type_()]);
                    }};
                }
                macro_rules! check_binop_bitwise {
                    () => {{
                        assert_eq!(args.len(), 2);
                        assert!(args[0].type_().is_int());
                        assert!(args[1].type_().is_int());
                        check_types_equal(&[type_, args[0].type_(), args[1].type_()]);
                        assert!(!type_.is_bool());
                    }};
                }
                macro_rules! check_binop_bitwise_bool {
                    () => {{
                        assert_eq!(args.len(), 2);
                        assert!(args[0].type_().is_int() || args[0].type_().is_bool());
                        assert!(args[1].type_().is_int() || args[1].type_().is_bool());
                        check_types_equal(&[type_, args[0].type_(), args[1].type_()]);
                    }};
                }
                macro_rules! check_cmp {
                    () => {{
                        assert_eq!(args.len(), 2);
                        check_types_equal(&[args[0].type_(), args[1].type_()]);
                        assert!(type_.is_bool());
                    }};
                }
                macro_rules! check_float_func1 {
                    () => {{
                        assert_eq!(args.len(), 1);
                        assert!(args[0].type_().is_float());
                        assert_eq!(type_, args[0].type_());
                    }};
                }
                match func {
                    Func::ZeroInitializer => {
                        assert!(args.is_empty());
                    }
                    Func::Assume => {
                        assert_eq!(args.len(), 1);
                        assert_eq!(args[0].type_(), <bool as TypeOf>::type_());
                    }
                    Func::Unreachable => {
                        assert!(args.is_empty());
                    }
                    Func::Assert => {
                        assert_eq!(args.len(), 1);
                        assert_eq!(args[0].type_(), <bool as TypeOf>::type_());
                    }
                    Func::ThreadId => {
                        assert!(args.is_empty());
                        assert_eq!(type_, uvec3_ty);
                    }
                    Func::BlockId => {
                        assert!(args.is_empty());
                        assert_eq!(type_, uvec3_ty);
                    }
                    Func::DispatchId => {
                        assert!(args.is_empty());
                        assert_eq!(type_, uvec3_ty);
                    }
                    Func::DispatchSize => {
                        assert!(args.is_empty());
                        assert_eq!(type_, uvec3_ty);
                    }
                    Func::RequiresGradient => {
                        assert_eq!(args.len(), 1);
                        assert!(grad_type_of(args[0].type_()).is_some());
                    }
                    Func::Gradient => {
                        assert_eq!(args.len(), 1);
                        assert!(grad_type_of(args[0].type_()).is_some());
                    }
                    Func::GradientMarker => {
                        assert_eq!(args.len(), 1);
                        assert!(grad_type_of(args[0].type_()).is_some());
                    }
                    Func::InstanceToWorldMatrix => todo!(),
                    Func::TraceClosest => todo!(),
                    Func::TraceAny => todo!(),
                    Func::SetInstanceTransform => todo!(),
                    Func::SetInstanceVisibility => todo!(),
                    Func::Cast => {
                        assert_eq!(args.len(), 1);
                        match type_.as_ref() {
                            Type::Struct(_) => panic!("cannt cast to struct"),
                            _ => {}
                        }
                        match args[0].type_().as_ref() {
                            Type::Struct(_) => panic!("cannt cast from struct"),
                            _ => {}
                        }
                        assert!(vector_compatible(type_, args[0].type_()));
                    }
                    Func::Bitcast => {
                        assert_eq!(args.len(), 1);
                        assert_eq!(type_.size(), args[0].type_().size());
                        match type_.as_ref() {
                            Type::Struct(_) => panic!("cannt cast to struct"),
                            _ => {}
                        }
                        match args[0].type_().as_ref() {
                            Type::Struct(_) => panic!("cannt cast from struct"),
                            _ => {}
                        }
                        assert!(vector_compatible(type_, args[0].type_()));
                    }
                    Func::Add => {
                        check_binop_float_int!();
                    }
                    Func::Sub => {
                        check_binop_float_int!();
                    }
                    Func::Mul => {
                        check_binop_float_int!();
                    }
                    Func::Div => {
                        check_binop_float_int!();
                    }
                    Func::Rem => {
                        check_binop_float_int!();
                    }
                    Func::BitAnd => {
                        check_binop_bitwise_bool!();
                    }
                    Func::BitOr => {
                        check_binop_bitwise_bool!();
                    }
                    Func::BitXor => {
                        check_binop_bitwise_bool!();
                    }
                    Func::Shl => {
                        check_binop_bitwise!();
                    }
                    Func::Shr => {
                        check_binop_bitwise!();
                    }
                    Func::RotRight => {
                        check_binop_bitwise!();
                    }
                    Func::RotLeft => {
                        check_binop_bitwise!();
                    }
                    Func::Eq => {
                        check_cmp!();
                    }
                    Func::Ne => {
                        check_cmp!();
                    }
                    Func::Lt => {
                        check_cmp!();
                    }
                    Func::Le => {
                        check_cmp!();
                    }
                    Func::Gt => {
                        check_cmp!();
                    }
                    Func::Ge => {
                        check_cmp!();
                    }
                    Func::MatCompMul => {
                        check_binop_same!();
                    }
                    Func::MatCompDiv => {
                        check_binop_same!();
                    }
                    Func::Neg => {
                        assert_eq!(args.len(), 1);
                        check_types_equal(&[type_, args[0].type_()]);
                        assert!(type_.is_float() || type_.is_int());
                    }
                    Func::Not => {
                        assert_eq!(args.len(), 1);
                        check_types_equal(&[type_, args[0].type_()]);
                        assert!(type_.is_bool() || type_.is_int());
                    }
                    Func::BitNot => {
                        assert_eq!(args.len(), 1);
                        check_types_equal(&[type_, args[0].type_()]);
                        assert!(type_.is_bool() || type_.is_int());
                    }
                    Func::All => {
                        assert_eq!(args.len(), 1);
                        assert!(args[0].type_().is_bool());
                        assert!(type_.is_bool());
                    }
                    Func::Any => {
                        assert_eq!(args.len(), 1);
                        assert!(args[0].type_().is_bool());
                        assert!(type_.is_bool());
                    }
                    Func::Select => {
                        assert_eq!(args.len(), 3);
                        assert!(args[0].type_().is_bool());
                        check_types_equal(&[args[1].type_(), args[2].type_()]);
                        check_types_equal(&[type_, args[1].type_()]);
                        assert!(vector_compatible(type_, args[0].type_()));
                        assert!(vector_compatible(type_, args[1].type_()));
                    }
                    Func::Clamp => {
                        assert_eq!(args.len(), 3);
                        check_types_equal(&[
                            type_,
                            args[0].type_(),
                            args[1].type_(),
                            args[2].type_(),
                        ]);
                        assert!(type_.is_float() || type_.is_int());
                    }
                    Func::Lerp => {
                        assert_eq!(args.len(), 3);
                        check_types_equal(&[
                            type_,
                            args[0].type_(),
                            args[1].type_(),
                            args[2].type_(),
                        ]);
                        assert!(type_.is_float());
                    }
                    Func::Step => {
                        assert_eq!(args.len(), 2);
                        check_types_equal(&[type_, args[0].type_(), args[1].type_()]);
                        assert!(type_.is_float());
                    }
                    Func::Abs => {
                        assert_eq!(args.len(), 1);
                        check_types_equal(&[type_, args[0].type_()]);
                        assert!(type_.is_float() || type_.is_int());
                    }
                    Func::Min => {
                        assert_eq!(args.len(), 2);
                        check_types_equal(&[type_, args[0].type_(), args[1].type_()]);
                        assert!(type_.is_float() || type_.is_int());
                    }
                    Func::Max => {
                        assert_eq!(args.len(), 2);
                        check_types_equal(&[type_, args[0].type_(), args[1].type_()]);
                        assert!(type_.is_float() || type_.is_int());
                    }
                    Func::ReduceSum => {
                        assert_eq!(args.len(), 1);
                        assert!(args[0].type_().is_float() || args[0].type_().is_int());
                        assert!(type_.is_float() || type_.is_int());
                        todo!("check element");
                    }
                    Func::ReduceProd => {
                        assert_eq!(args.len(), 1);
                        assert!(args[0].type_().is_float() || args[0].type_().is_int());
                        assert!(type_.is_float() || type_.is_int());
                        todo!("check element");
                    }
                    Func::ReduceMin => {
                        assert_eq!(args.len(), 1);
                        assert!(args[0].type_().is_float() || args[0].type_().is_int());
                        assert!(type_.is_float() || type_.is_int());
                        todo!("check element");
                    }
                    Func::ReduceMax => {
                        assert_eq!(args.len(), 1);
                        assert!(args[0].type_().is_float() || args[0].type_().is_int());
                        assert!(type_.is_float() || type_.is_int());
                        todo!("check element");
                    }
                    Func::Clz => todo!(),
                    Func::Ctz => todo!(),
                    Func::PopCount => todo!(),
                    Func::Reverse => todo!(),
                    Func::IsInf => todo!(),
                    Func::IsNan => todo!(),
                    Func::Acos => check_float_func1!(),
                    Func::Acosh => check_float_func1!(),
                    Func::Asin => check_float_func1!(),
                    Func::Asinh => check_float_func1!(),
                    Func::Atan => check_float_func1!(),
                    Func::Atan2 => todo!(),
                    Func::Atanh => check_float_func1!(),
                    Func::Cos => check_float_func1!(),
                    Func::Cosh => check_float_func1!(),
                    Func::Sin => check_float_func1!(),
                    Func::Sinh => check_float_func1!(),
                    Func::Tan => check_float_func1!(),
                    Func::Tanh => check_float_func1!(),
                    Func::Exp => check_float_func1!(),
                    Func::Exp2 => check_float_func1!(),
                    Func::Exp10 => check_float_func1!(),
                    Func::Log => todo!(),
                    Func::Log2 => check_float_func1!(),
                    Func::Log10 => check_float_func1!(),
                    Func::Powi => todo!(),
                    Func::Powf => todo!(),
                    Func::Sqrt => check_float_func1!(),
                    Func::Rsqrt => check_float_func1!(),
                    Func::Ceil => check_float_func1!(),
                    Func::Floor => check_float_func1!(),
                    Func::Fract => check_float_func1!(),
                    Func::Trunc => check_float_func1!(),
                    Func::Round => check_float_func1!(),
                    Func::Fma => todo!(),
                    Func::Copysign => todo!(),
                    Func::Cross => todo!(),
                    Func::Dot => todo!(),
                    Func::Length => todo!(),
                    Func::LengthSquared => todo!(),
                    Func::Normalize => todo!(),
                    Func::Faceforward => todo!(),
                    Func::Determinant => todo!(),
                    Func::Transpose => todo!(),
                    Func::Inverse => todo!(),
                    Func::SynchronizeBlock => todo!(),
                    Func::AtomicExchange => todo!(),
                    Func::AtomicCompareExchange => todo!(),
                    Func::AtomicFetchAdd => todo!(),
                    Func::AtomicFetchSub => todo!(),
                    Func::AtomicFetchAnd => todo!(),
                    Func::AtomicFetchOr => todo!(),
                    Func::AtomicFetchXor => todo!(),
                    Func::AtomicFetchMin => todo!(),
                    Func::AtomicFetchMax => todo!(),
                    Func::BufferRead => todo!(),
                    Func::BufferWrite => todo!(),
                    Func::BufferSize => todo!(),
                    Func::TextureRead => todo!(),
                    Func::TextureWrite => todo!(),
                    Func::BindlessTexture2dSample => todo!(),
                    Func::BindlessTexture2dSampleLevel => todo!(),
                    Func::BindlessTexture2dSampleGrad => todo!(),
                    Func::BindlessTexture3dSample => todo!(),
                    Func::BindlessTexture3dSampleLevel => todo!(),
                    Func::BindlessTexture3dSampleGrad => todo!(),
                    Func::BindlessTexture2dRead => todo!(),
                    Func::BindlessTexture3dRead => todo!(),
                    Func::BindlessTexture2dReadLevel => todo!(),
                    Func::BindlessTexture3dReadLevel => todo!(),
                    Func::BindlessTexture2dSize => todo!(),
                    Func::BindlessTexture3dSize => todo!(),
                    Func::BindlessTexture2dSizeLevel => todo!(),
                    Func::BindlessTexture3dSizeLevel => todo!(),
                    Func::BindlessBufferRead => todo!(),
                    Func::BindlessBufferSize => todo!(),
                    Func::Vec => todo!(),
                    Func::Vec2 => todo!(),
                    Func::Vec3 => todo!(),
                    Func::Vec4 => todo!(),
                    Func::Permute => todo!(),
                    Func::ExtractElement => todo!(),
                    Func::InsertElement => todo!(),
                    Func::GetElementPtr => todo!(),
                    Func::Mat => todo!(),
                    Func::Matrix2 => todo!(),
                    Func::Matrix3 => todo!(),
                    Func::Matrix4 => todo!(),
                    Func::Callable(_) => todo!(),
                    Func::CpuCustomOp(_) => todo!(),
                }
            }
            Instruction::Phi(incomings) => {
                for incoming in incomings.as_ref() {
                    assert_eq!(incoming.value.type_(), type_);
                }
            }
            Instruction::Return(_) => todo!(),
            Instruction::Loop { body, cond } => {
                assert_eq!(type_, Type::void());
                assert_eq!(cond.type_(), <bool as TypeOf>::type_());
                self.check_block(body);
            }
            Instruction::GenericLoop {
                prepare,
                cond,
                body,
                update,
            } => {
                assert_eq!(type_, Type::void());
                self.check_block(prepare);
                assert_eq!(cond.type_(), <bool as TypeOf>::type_());
                self.check_block(body);
                self.check_block(update);
            }
            Instruction::Break => {
                assert_eq!(type_, Type::void());
            }
            Instruction::Continue => {
                assert_eq!(type_, Type::void());
            }
            Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                assert_eq!(type_, Type::void());
                assert_eq!(cond.type_(), <bool as TypeOf>::type_());
                self.check_block(true_branch);
                self.check_block(false_branch);
            }
            Instruction::Switch {
                value,
                default,
                cases,
            } => {
                assert_eq!(type_, Type::void());
                assert_eq!(value.type_(), <i32 as TypeOf>::type_());
                self.check_block(default);
                for case in cases.as_ref() {
                    assert_eq!(case.value.type_(), <i32 as TypeOf>::type_());
                    self.check_block(&case.block);
                }
            }
            Instruction::Comment(_) => {
                assert_eq!(type_, Type::void());
            }
        }
    }
    fn validate(&mut self, module: &Module) {
        self.check_block(&module.entry);
    }
}
pub struct Validate;
impl Transform for Validate {
    fn transform(&self, module: Module) -> Module {
        let mut validator = Validator::new();
        validator.validate(&module);
        module
    }
}
