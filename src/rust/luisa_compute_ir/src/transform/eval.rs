use crate::{ir::*, *};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub(crate) enum Value {
    Unit,
    Primitive(NodeRef),
    Vector(Vec<NodeRef>),
    Matrix(Vec<Vec<NodeRef>>),
    Array(Vec<Value>),
    Struct(Vec<Value>),
}
impl Value {
    fn is_matrix(&self) -> bool {
        match self {
            Self::Matrix(_) => true,
            _ => false,
        }
    }
    fn is_vector(&self) -> bool {
        match self {
            Self::Vector(_) => true,
            _ => false,
        }
    }
    fn is_primitive(&self) -> bool {
        match self {
            Self::Primitive(_) => true,
            _ => false,
        }
    }
    fn primitive(&self) -> NodeRef {
        match self {
            Self::Primitive(p) => *p,
            _ => panic!("not a primitive"),
        }
    }
    fn vector_length(&self) -> usize {
        match self {
            Self::Vector(v) => v.len(),
            _ => panic!("not a vector"),
        }
    }
    fn vector_elements(&self) -> Vec<NodeRef> {
        match self {
            Self::Vector(v) => v.clone(),
            _ => panic!("not a vector"),
        }
    }
    fn extract(&self, i: usize) -> Value {
        match self {
            Self::Unit | Self::Primitive(_) => panic!("cannot extract from {:?}", self),
            Self::Vector(v) => Value::Primitive(v[i]),
            Self::Matrix(m) => {
                let out = m[i].clone();
                Value::Vector(out)
            }
            Self::Struct(fields) => fields[i].clone(),
            Self::Array(fields) => fields[i].clone(),
        }
    }
    fn insert(&self, i: usize, v: Value, _builder: &mut IrBuilder) -> Value {
        match self {
            Self::Unit | Self::Primitive(_) => panic!("cannot insert into {:?}", self),
            Self::Vector(vs) => {
                let mut out = vs.clone();
                out[i] = v.primitive();
                Value::Vector(out)
            }
            Self::Matrix(m) => {
                let mut out = m.clone();
                out[i] = v.vector_elements();
                Value::Matrix(out)
            }
            Self::Struct(fields) => {
                let mut out = fields.clone();
                out[i] = v;
                Value::Struct(out)
            }
            Self::Array(fields) => {
                let mut out = fields.clone();
                out[i] = v;
                Value::Array(out)
            }
        }
    }
}

struct EvaluatorImpl {
    inline_callable: bool,
    trace: bool,
    original: Module,
    env: NestedHashMap<NodeRef, Value>,
    locally_defined: HashSet<NodeRef>,
    pools: CArc<ModulePools>,
}

impl EvaluatorImpl {
    fn run(
        m: Module,
        inline_callable: bool,
        trace: bool,
        env: HashMap<NodeRef, NodeRef>,
    ) -> Module {
        let locally_defined = HashSet::from_iter(m.collect_nodes());
        let pools = CArc::new(ModulePools::new());
        let mut eval = Self {
            inline_callable,
            trace,
            original: m,
            env: NestedHashMap::new(),
            locally_defined,
            pools: pools.clone(),
        };
        let mut main_builder = IrBuilder::new(pools.clone());
        for (k, v) in env {
            let destructed = eval.destruct(v, &mut main_builder);
            eval.env.insert(k, destructed);
        }
        Module {
            kind: ModuleKind::Block,
            entry: main_builder.finish(),
            pools: eval.pools,
        }
    }
    fn eval_node(&mut self, node: NodeRef, builder: &mut IrBuilder) -> Value {
        if self.env.contains_key(&node) {
            return self.env.get(&node).unwrap().clone();
        } else {
            let result = self._eval_node(node, builder);
            self.env.insert(node, result.clone());
            result
        }
    }
    fn try_eval_builtin(
        &mut self,
        node: NodeRef,
        f: Func,
        original_args: &[NodeRef],
        args: &[Value],
        builder: &mut IrBuilder,
    ) -> Option<Value> {
        let is_any_matrix = args.iter().any(|a| a.is_matrix());
        let can_do = is_any_matrix
            || match f {
                Func::Normalize
                | Func::OuterProduct
                | Func::Dot
                | Func::Cross
                | Func::ReduceMax
                | Func::ReduceMin
                | Func::ReduceProd
                | Func::ReduceSum
                | Func::Faceforward
                | Func::Transpose
                | Func::MatCompMul
                | Func::Inverse
                | Func::Determinant
                | Func::Length
                | Func::LengthSquared => true,
                _ => false,
            };
        if can_do {
            let nodes = args
                .iter()
                .enumerate()
                .map(|(i, a)| self.construct(a.clone(), original_args[i].type_().clone(), builder))
                .collect::<Vec<_>>();
            let node = builder.call(f, &nodes, node.type_().clone());
            Some(self.destruct(node, builder))
        } else {
            None
        }
    }
    fn try_eval_arith(
        &mut self,
        node: NodeRef,
        f: Func,
        original_args: &[NodeRef],
        args: &[Value],
        builder: &mut IrBuilder,
    ) -> Option<Value> {
        let is_binop = match f {
            Func::Add
            | Func::Sub
            | Func::Mul
            | Func::Div
            | Func::Rem
            | Func::BitAnd
            | Func::BitOr
            | Func::BitXor
            | Func::Shl
            | Func::Shr
            | Func::RotLeft
            | Func::RotRight
            | Func::Lt
            | Func::Le
            | Func::Gt
            | Func::Ge
            | Func::Eq
            | Func::Ne
            | Func::Powf
            | Func::Powi
            | Func::Atan2
            | Func::Copysign => true,
            _ => false,
        };
        let is_unary_op = match f {
            Func::Neg
            | Func::Sin
            | Func::Cos
            | Func::Tan
            | Func::Asin
            | Func::Acos
            | Func::Atan
            | Func::Sinh
            | Func::Cosh
            | Func::Tanh
            | Func::Asinh
            | Func::Acosh
            | Func::Atanh
            | Func::Exp
            | Func::Exp2
            | Func::Log
            | Func::Log2
            | Func::Log10
            | Func::Abs
            | Func::Floor
            | Func::Ceil
            | Func::Round
            | Func::Trunc
            | Func::Fract => true,
            _ => false,
        };
        if args.len() == 2 && is_binop {
            if original_args[0].type_().is_primitive() && original_args[1].type_().is_primitive() {
                let nodes = args
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        self.construct(a.clone(), original_args[i].type_().clone(), builder)
                    })
                    .collect::<Vec<_>>();
                let node = builder.call(f, &nodes, node.type_().clone());
                return Some(self.destruct(node, builder));
            }
            let is_any_vector = args.iter().any(|a| a.is_vector());
            if is_any_vector {
                let len = args[1].vector_length();
                let lhs = if args[0].is_primitive() {
                    Value::Vector(vec![args[0].primitive(); args[1].vector_length()])
                } else {
                    args[1].clone()
                };
                let rhs = if args[1].is_primitive() {
                    Value::Vector(vec![args[1].primitive(); args[0].vector_length()])
                } else {
                    args[0].clone()
                };
                let comps = (0..len)
                    .map(|i| {
                        let lhs = lhs.extract(i);
                        let rhs = rhs.extract(i);
                        let nodes = vec![lhs.primitive(), rhs.primitive()];
                        builder.call(f.clone(), &nodes, node.type_().extract(i))
                    })
                    .collect::<Vec<_>>();
                return Some(Value::Vector(comps));
            }
        }
        if args.len() == 1 && is_unary_op {
            if original_args[0].type_().is_primitive() {
                let node = builder.call(f, &[args[0].primitive()], node.type_().clone());
                return Some(self.destruct(node, builder));
            }
            assert!(args[0].is_vector());
            let len = args[0].vector_length();
            let comps = (0..len)
                .map(|i| {
                    builder.call(
                        f.clone(),
                        &[args[0].extract(i).primitive()],
                        node.type_().extract(i),
                    )
                })
                .collect::<Vec<_>>();
            return Some(Value::Vector(comps));
        }
        None
    }
    fn _eval_node(&mut self, node: NodeRef, builder: &mut IrBuilder) -> Value {
        let inst = node.get().instruction.as_ref();
        let _ty = node.type_();
        match inst {
            Instruction::Const(_) => self.destruct(node, builder),
            Instruction::Call(f, args) => {
                let evaled_args = args
                    .iter()
                    .map(|a| self.eval_node(*a, builder))
                    .collect::<Vec<_>>();
                if let Some(v) = self.try_eval_builtin(node, f.clone(), args, &evaled_args, builder)
                {
                    return v;
                }
                if let Some(v) = self.try_eval_arith(node, f.clone(), args, &evaled_args, builder) {
                    return v;
                }
                match f {
                    Func::ExtractElement => {
                        let v = evaled_args[0].clone();
                        let i = args[1].get_i32();
                        v.extract(i as usize)
                    }
                    Func::InsertElement => {
                        let v = evaled_args[0].clone();
                        let e = evaled_args[1].clone();
                        let i = args[2].get_i32();
                        v.insert(i as usize, e, builder)
                    }
                    _ => unreachable!("unimplemented function: {:?}", f),
                }
            }
            _ => todo!(),
        }
    }
    fn construct(&self, value: Value, ty: CArc<Type>, builder: &mut IrBuilder) -> NodeRef {
        match value {
            Value::Unit => unreachable!(),
            Value::Primitive(node) => node,
            Value::Vector(vs) => builder.call(Func::Vec, &vs, ty.element()),
            Value::Matrix(cols) => {
                let mut result = Vec::new();
                for (i, c) in cols.iter().enumerate() {
                    result.push(self.construct(Value::Vector(c.clone()), ty.extract(i), builder));
                }
                builder.call(Func::Mat, &result, ty)
            }
            Value::Array(vs) => {
                let mut result = Vec::new();
                for v in vs {
                    result.push(self.construct(v, ty.element(), builder));
                }
                builder.call(Func::Array, &result, ty)
            }
            Value::Struct(fields) => {
                let mut result = Vec::new();
                for (i, f) in fields.iter().enumerate() {
                    result.push(self.construct(f.clone(), ty.extract(i), builder));
                }
                builder.call(Func::Struct, &result, ty)
            }
        }
    }
    fn destruct(&self, node: NodeRef, builder: &mut IrBuilder) -> Value {
        let ty = node.type_().as_ref();
        match ty {
            Type::Primitive(_) => Value::Primitive(node),
            Type::Vector(vt) => {
                let el = context::register_type(Type::Primitive(vt.element()));
                let mut result = Vec::new();
                for i in 0..vt.length {
                    let e = builder.extract(node, i as usize, el.clone());
                    result.push(e);
                }
                Value::Vector(result)
            }
            Type::Matrix(mt) => {
                let col = mt.column();
                let mut cols = Vec::new();
                for i in 0..mt.dimension {
                    let e = builder.extract(node, i as usize, col.clone());
                    cols.push(self.destruct(e, builder));
                }
                Value::Matrix(
                    cols.into_iter()
                        .map(|v| match v {
                            Value::Vector(es) => es,
                            _ => unreachable!(),
                        })
                        .collect(),
                )
            }
            Type::Array(at) => {
                let el = at.element.clone();
                let mut result = Vec::new();
                for i in 0..at.length {
                    let e = builder.extract(node, i as usize, el.clone());
                    result.push(self.destruct(e, builder));
                }
                Value::Array(result)
            }
            Type::Struct(st) => {
                let mut result = Vec::new();
                for i in 0..st.fields.len() {
                    let e = builder.extract(node, i as usize, st.fields[i].clone());
                    result.push(self.destruct(e, builder));
                }
                Value::Struct(result)
            }
            Type::Void | Type::UserData | Type::Opaque(_) => Value::Unit,
        }
    }
}
