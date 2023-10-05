use crate::ir::*;
use crate::{CArc, CBoxedSlice, Pooled, TypeOf};
use base64ct::{Base64, Encoding};
use bitflags::Flags;
use half::f16;
use json::{parse as parse_json, JsonValue as JSON};
use std::cmp::max;
use std::collections::HashMap;
use std::iter::zip;

struct AST2IRCtx<'a> {
    j: &'a JSON,
    j_tag: &'a str,
    j_variables: &'a JSON,
    ret_type: CArc<Type>,
    builder: Option<IrBuilder>,
    arguments: HashMap<u32, NodeRef>,
    variables: HashMap<u32, NodeRef>,
    shared: Vec<NodeRef>,
    has_autodiff: bool,
}

struct AST2IR<'a: 'b, 'b> {
    j_functions: &'a JSON,
    j_constants: &'a JSON,
    j_types: &'a JSON,
    functions: HashMap<usize, FunctionModule>,
    constants: HashMap<usize, Const>,
    types: HashMap<usize, CArc<Type>>,
    ctx: Option<AST2IRCtx<'b>>,
    pools: CArc<ModulePools>,
}

#[derive(Clone)]
enum FunctionModule {
    Kernel(CArc<KernelModule>),
    Callable(CArc<CallableModule>),
}

impl<'a: 'b, 'b> AST2IR<'a, 'b> {
    fn convert_type(&mut self, i: usize) -> CArc<Type> {
        if let Some(t) = self.types.get(&i) {
            return t.clone();
        }
        let j = &self.j_types[i];
        let tag = if j.is_null() {
            "VOID"
        } else {
            j["tag"].as_str().unwrap()
        };
        let t = match tag {
            "VOID" => Type::void(),
            "BOOL" => <bool as TypeOf>::type_(),
            "INT16" => <i16 as TypeOf>::type_(),
            "UINT16" => <u16 as TypeOf>::type_(),
            "INT32" => <i32 as TypeOf>::type_(),
            "UINT32" => <u32 as TypeOf>::type_(),
            "INT64" => <i64 as TypeOf>::type_(),
            "UINT64" => <u64 as TypeOf>::type_(),
            "FLOAT16" => <f16 as TypeOf>::type_(),
            "FLOAT32" => <f32 as TypeOf>::type_(),
            "FLOAT64" => <f64 as TypeOf>::type_(),
            "VECTOR" => {
                let elem_index = j["element"].as_usize().unwrap();
                let elem_type = self.convert_type(elem_index);
                let dim = j["dimension"].as_u32().unwrap();
                Type::vector_of(elem_type, dim)
            }
            "MATRIX" => {
                if let Some(elem) = j["element"].as_usize() {
                    let elem = self.convert_type(elem);
                    assert!(
                        elem.is_float() && elem.is_primitive(),
                        "Matrix element type must be float scalars."
                    );
                }
                let elem = <f32 as TypeOf>::type_();
                let dim = j["dimension"].as_u32().unwrap();
                Type::matrix_of(elem, dim)
            }
            "ARRAY" => {
                let elem_index = j["element"].as_usize().unwrap();
                let elem_type = self.convert_type(elem_index);
                let dim = j["dimension"].as_u32().unwrap();
                Type::array_of(elem_type, dim)
            }
            "STRUCTURE" => {
                let members: Vec<_> = j["members"]
                    .members()
                    .map(|m| self.convert_type(m.as_usize().unwrap()))
                    .collect();
                let align = j["alignment"].as_u32().unwrap();
                Type::struct_of(align, members)
            }
            "BUFFER" => {
                let elem_index = j["element"].as_usize().unwrap();
                self.convert_type(elem_index)
            }
            "TEXTURE" => {
                let elem_index = j["element"].as_usize().unwrap();
                Type::vector_of(self.convert_type(elem_index), 4)
            }
            "BINDLESS_ARRAY" => Type::void(),
            "ACCEL" => Type::void(),
            "CUSTOM" => Type::opaque(j["id"].as_str().unwrap().into()),
            _ => panic!("Invalid type tag: {}", tag),
        };
        self.types.insert(i, t.clone());
        t
    }

    fn convert_constant(&mut self, i: usize) -> Const {
        if let Some(c) = self.constants.get(&i) {
            return c.clone();
        }
        let c = &self.j_constants[i];
        let t = self.convert_type(c["type"].as_usize().unwrap());
        let raw = Base64::decode_vec(c["raw"].as_str().unwrap()).unwrap();
        let c = Const::Generic(CBoxedSlice::new(raw), t);
        self.constants.insert(i, c.clone());
        c
    }

    fn _curr_ctx_mut(&mut self) -> &mut AST2IRCtx<'b> {
        self.ctx.as_mut().unwrap()
    }

    fn _curr_ctx(&self) -> &AST2IRCtx<'b> {
        self.ctx.as_ref().unwrap()
    }

    fn unwrap_ctx(
        &mut self,
    ) -> (
        &mut IrBuilder,
        &mut HashMap<u32, NodeRef>,
        &mut HashMap<u32, NodeRef>,
    ) {
        let ctx = self.ctx.as_mut().unwrap();
        let AST2IRCtx {
            builder,
            arguments,
            variables,
            ..
        } = ctx;
        (builder.as_mut().unwrap(), arguments, variables)
    }
    fn convert_variables(&mut self) {
        self._curr_ctx()
            .j_variables
            .members()
            .enumerate()
            .for_each(|(i, v)| {
                let v_tag = v["tag"].as_str().unwrap();
                let v_type = v["type"].as_usize().unwrap();
                let v = match v_tag {
                    "LOCAL" => {
                        let t = self.convert_type(v_type);
                        let (builder, _, _) = self.unwrap_ctx();
                        builder.local_zero_init(t)
                    }
                    "SHARED" => {
                        let t = self.convert_type(v_type);
                        let node = new_node(
                            &self.pools,
                            Node::new(CArc::new(Instruction::Shared), t.clone()),
                        );
                        self._curr_ctx_mut().shared.push(node.clone());
                        node
                    }
                    "ARGUMENT" => {
                        let t = self.convert_type(v_type);
                        let arg = if self._curr_ctx().j_tag == "KERNEL" {
                            Instruction::Uniform
                        } else {
                            Instruction::Argument { by_value: true }
                        };
                        let arg = new_node(&self.pools, Node::new(CArc::new(arg), t.clone()));
                        let (builder, arguments, _) = self.unwrap_ctx();
                        arguments.insert(i as u32, arg);
                        builder.local(arg)
                    }
                    "REFERENCE" => {
                        let t = self.convert_type(v_type);
                        assert_eq!(
                            self._curr_ctx().j_tag,
                            "CALLABLE",
                            "Only callable can have reference variables."
                        );
                        let arg = Instruction::Argument { by_value: false };
                        let arg = new_node(&self.pools, Node::new(CArc::new(arg), t.clone()));
                        let (_builder, arguments, _) = self.unwrap_ctx();
                        arguments.insert(i as u32, arg);
                        arg
                    }
                    "BUFFER" => {
                        let t = &self.j_types[v_type];
                        assert_eq!(t["tag"], "BUFFER", "Only buffer can be a buffer variable.");
                        let t = t["element"].as_usize().unwrap();
                        let t = self.convert_type(t);
                        let arg = new_node(
                            &self.pools,
                            Node::new(CArc::new(Instruction::Buffer), t.clone()),
                        );
                        self._curr_ctx_mut().arguments.insert(i as u32, arg);
                        arg
                    }
                    "TEXTURE" => {
                        let t = &self.j_types[v_type];
                        assert_eq!(
                            t["tag"], "TEXTURE",
                            "Only texture can be a texture variable."
                        );
                        let dim = t["dimension"].as_u32().unwrap();
                        let t = t["element"].as_usize().unwrap();
                        let t = Type::vector_of(self.convert_type(t), 4);
                        let instr = match dim {
                            2 => Instruction::Texture2D,
                            3 => Instruction::Texture3D,
                            _ => panic!("Invalid texture dimension: {}", dim),
                        };
                        let arg = new_node(&self.pools, Node::new(CArc::new(instr), t.clone()));
                        self._curr_ctx_mut().arguments.insert(i as u32, arg);
                        arg
                    }
                    "BINDLESS_ARRAY" => {
                        let t = &self.j_types[v_type];
                        assert_eq!(
                            t["tag"], "BINDLESS_ARRAY",
                            "Only bindless array can be a bindless array variable."
                        );
                        let arg = new_node(
                            &self.pools,
                            Node::new(CArc::new(Instruction::Bindless), Type::void()),
                        );
                        self._curr_ctx_mut().arguments.insert(i as u32, arg);
                        arg
                    }
                    "ACCEL" => {
                        let t = &self.j_types[v_type];
                        assert_eq!(t["tag"], "ACCEL", "Only accel can be a accel variable.");
                        let arg = new_node(
                            &self.pools,
                            Node::new(CArc::new(Instruction::Accel), Type::void()),
                        );
                        self._curr_ctx_mut().arguments.insert(i as u32, arg);
                        arg
                    }
                    "THREAD_ID" => {
                        let t = Type::vector_of(<u32 as TypeOf>::type_(), 3);
                        let (builder, _, _) = self.unwrap_ctx();
                        let v = builder.call(Func::ThreadId, &[], t);
                        builder.local(v)
                    }
                    "BLOCK_ID" => {
                        let t = Type::vector_of(<u32 as TypeOf>::type_(), 3);
                        let (builder, _, _) = self.unwrap_ctx();
                        let v = builder.call(Func::BlockId, &[], t);
                        builder.local(v)
                    }
                    "DISPATCH_ID" => {
                        let t = Type::vector_of(<u32 as TypeOf>::type_(), 3);
                        let (builder, _, _) = self.unwrap_ctx();
                        let v = builder.call(Func::DispatchId, &[], t);
                        builder.local(v)
                    }
                    "DISPATCH_SIZE" => {
                        let t = Type::vector_of(<u32 as TypeOf>::type_(), 3);
                        let (builder, _, _) = self.unwrap_ctx();
                        let v = builder.call(Func::DispatchSize, &[], t);
                        builder.local(v)
                    }
                    "KERNEL_ID" => todo!("KERNEL_ID"),
                    "WARP_LANE_COUNT" => {
                        let t = <u32 as TypeOf>::type_();
                        let (builder, _, _) = self.unwrap_ctx();
                        let v = builder.call(Func::WarpSize, &[], t);
                        builder.local(v)
                    }
                    "WARP_LANE_ID" => {
                        let t = <u32 as TypeOf>::type_();
                        let (builder, _, _) = self.unwrap_ctx();
                        let v = builder.call(Func::WarpLaneId, &[], t);
                        builder.local(v)
                    }
                    "OBJECT_ID" => todo!("OBJECT_ID"),
                    _ => panic!("Invalid variable tag: {}", v_tag),
                };
                self._curr_ctx_mut().variables.insert(i as u32, v);
            });
    }

    fn _cast(builder: &mut IrBuilder, dst: &CArc<Type>, node: NodeRef) -> NodeRef {
        if dst.is_void() {
            INVALID_REF
        } else if dst.as_ref() == node.type_().as_ref() {
            node
        } else {
            let src = node.type_();
            match (src.as_ref(), dst.as_ref()) {
                (Type::Primitive(_), Type::Primitive(_)) => builder.cast(node, dst.clone()),
                (Type::Vector(vsrc), Type::Vector(vdst)) => {
                    assert_eq!(vsrc.length, vdst.length);
                    builder.cast(node, dst.clone())
                }
                (Type::Primitive(_), Type::Vector(_)) => {
                    let node = Self::_cast(builder, &dst.element(), node);
                    builder.call(Func::Vec, &[node], dst.clone())
                }
                _ => panic!("Invalid cast."),
            }
        }
    }

    fn _convert_unary_expr(&mut self, t: &CArc<Type>, j: &JSON) -> NodeRef {
        let operand = self._convert_expression(&j["operand"], false);
        let op = j["op"].as_str().unwrap();
        let (builder, ..) = self.unwrap_ctx();
        match op {
            "PLUS" => {
                assert_eq!(
                    operand.type_().as_ref(),
                    t.as_ref(),
                    "Mismatched types in unary plus operator."
                );
                operand
            }
            "MINUS" => {
                assert_eq!(
                    operand.type_().as_ref(),
                    t.as_ref(),
                    "Mismatched types in unary minus operator."
                );
                builder.call(Func::Neg, &[operand], t.clone())
            }
            "NOT" => {
                assert!(t.is_bool(), "Invalid result type for unary not operator.");
                let operand = Self::_cast(builder, &t, operand);
                builder.call(Func::Not, &[operand], t.clone())
            }
            "BIT_NOT" => {
                assert_eq!(
                    operand.type_().as_ref(),
                    t.as_ref(),
                    "Mismatched types in unary bitwise not operator."
                );
                assert!(
                    t.is_int(),
                    "Only integral operands are allowed in unary bitwise not operator."
                );
                builder.call(Func::BitNot, &[operand], t.clone())
            }
            _ => panic!("Invalid unary operator: {}.", op),
        }
    }

    fn _promote_binary_op_types(
        op: &str,
        t_lhs: &CArc<Type>,
        t_rhs: &CArc<Type>,
    ) -> (
        CArc<Type>, // dest lhs type
        CArc<Type>, // dest rhs type
        CArc<Type>, // result type
    ) {
        assert!(
            t_lhs.is_primitive() || t_lhs.is_vector() || t_lhs.is_matrix(),
            "Invalid LHS operand type for binary operator."
        );
        assert!(
            t_rhs.is_primitive() || t_rhs.is_vector() || t_rhs.is_matrix(),
            "Invalid RHS operand type for binary operator."
        );
        let dim_lhs = t_lhs.dimension();
        let dim_rhs = t_rhs.dimension();
        assert!(
            dim_lhs == dim_rhs || dim_lhs == 1 || dim_rhs == 1,
            "Incompatible dimensions in binary operator."
        );
        let dim = max(dim_lhs, dim_rhs) as u32;

        // logical operator, convert both operands to boolean
        if op == "AND" || op == "OR" {
            assert!(
                (t_lhs.is_primitive() || t_lhs.is_vector())
                    && (t_rhs.is_primitive() || t_rhs.is_vector()),
                "Logical operators must have scalar or vector operands."
            );
            let b = if dim == 1 {
                <bool as TypeOf>::type_()
            } else {
                Type::vector(Primitive::Bool, dim)
            };
            return (b.clone(), b.clone(), b);
        }

        // scalar op scalar
        if t_lhs.is_primitive() && t_rhs.is_primitive() {
            let score_scalar = |t: &CArc<Type>| match t.as_ref() {
                Type::Primitive(s) => match s {
                    Primitive::Bool => 10,
                    Primitive::Int8 => 20,
                    Primitive::Uint8 => 30,
                    Primitive::Int16 => 40,
                    Primitive::Uint16 => 50,
                    Primitive::Int32 => 60,
                    Primitive::Uint32 => 70,
                    Primitive::Int64 => 80,
                    Primitive::Uint64 => 90,
                    Primitive::Float16 => 100,
                    Primitive::Float32 => 110,
                    Primitive::Float64 => 120,
                },
                _ => unreachable!("Invalid scalar type."),
            };
            let t = if score_scalar(t_lhs) > score_scalar(t_rhs) {
                t_lhs.clone()
            } else {
                t_rhs.clone()
            };
            let ret = match op {
                "LESS" | "GREATER" | "LESS_EQUAL" | "GREATER_EQUAL" | "EQUAL" | "NOT_EQUAL" => {
                    <bool as TypeOf>::type_()
                }
                _ => t.clone(),
            };
            return (t.clone(), t, ret);
        }

        // scalar op vector | vector op scalar | vector op vector
        if (t_lhs.is_primitive() && t_rhs.is_vector())
            || (t_lhs.is_vector() && t_rhs.is_primitive())
            || (t_lhs.is_vector() && t_rhs.is_vector())
        {
            let (prom_lhs, prom_rhs, prom_ret) =
                Self::_promote_binary_op_types(&op, &t_lhs.element(), &t_rhs.element());
            assert!(prom_lhs.is_primitive() && prom_rhs.is_primitive() && prom_ret.is_primitive());
            return (
                Type::vector_of(prom_lhs, dim),
                Type::vector_of(prom_rhs, dim),
                Type::vector_of(prom_ret, dim),
            );
        }

        // matrix op matrix
        if t_lhs.is_matrix() && t_rhs.is_matrix() {
            assert_eq!(t_lhs.as_ref(), t_rhs.as_ref());
            return (t_lhs.clone(), t_lhs.clone(), t_lhs.clone());
        }

        // matrix op scalar
        if t_lhs.is_matrix() && t_rhs.is_primitive() {
            return (t_lhs.clone(), t_lhs.element(), t_lhs.clone());
        }

        // scalar op matrix
        if t_lhs.is_primitive() && t_rhs.is_matrix() {
            return (t_rhs.element(), t_rhs.clone(), t_rhs.clone());
        }

        // otherwise, should be matrix op vector
        assert!(t_lhs.is_matrix() && t_rhs.is_vector());
        assert_eq!(t_lhs.element().as_ref(), t_rhs.element().as_ref());
        (t_lhs.clone(), t_rhs.clone(), t_rhs.clone())
    }

    fn _convert_binary_expr(&mut self, t: &CArc<Type>, j: &JSON) -> NodeRef {
        let lhs = self._convert_expression(&j["lhs"], false);
        let rhs = self._convert_expression(&j["rhs"], false);
        let op = j["op"].as_str().unwrap();
        let (t_lhs, t_rhs, t_ret) = Self::_promote_binary_op_types(op, lhs.type_(), rhs.type_());
        assert_eq!(
            t_ret.as_ref(),
            t.as_ref(),
            "Mismatched result types in binary operator."
        );
        let (builder, ..) = self.unwrap_ctx();
        let lhs = Self::_cast(builder, &t_lhs, lhs);
        let rhs = Self::_cast(builder, &t_rhs, rhs);
        // matrix-scalar operators requires special handling
        let mut scalar_to_mat = |scalar: NodeRef, t_mat: &CArc<Type>| {
            let t_scalar = scalar.type_();
            let scalar = if op == "DIV" {
                let one = builder.const_(Const::One(t_scalar.clone()));
                builder.call(Func::Div, &[one, scalar], t_scalar.clone())
            } else {
                scalar
            };
            builder.call(Func::Mat, &[scalar], t_mat.clone())
        };
        let (is_mat_scalar, lhs, rhs) = if t_lhs.is_primitive() && t_rhs.is_matrix() {
            // scalar op matrix
            (true, scalar_to_mat(lhs, &t_rhs), rhs)
        } else if t_lhs.is_matrix() && t_rhs.is_primitive() {
            // matrix op scalar
            (true, lhs, scalar_to_mat(rhs, &t_lhs))
        } else {
            (false, lhs, rhs)
        };
        // build the expression
        let op = match op {
            "ADD" => Func::Add,
            "SUB" => Func::Sub,
            "MUL" => {
                if is_mat_scalar {
                    Func::MatCompMul
                } else {
                    Func::Mul
                }
            }
            "DIV" => {
                if is_mat_scalar {
                    Func::MatCompMul
                } else {
                    Func::Div
                }
            }
            "MOD" => Func::Rem,
            "BIT_AND" | "AND" => Func::BitAnd,
            "BIT_OR" | "OR" => Func::BitOr,
            "BIT_XOR" => Func::BitXor,
            "SHL" => Func::Shl,
            "SHR" => Func::Shr,
            "LESS" => Func::Lt,
            "GREATER" => Func::Gt,
            "LESS_EQUAL" => Func::Le,
            "GREATER_EQUAL" => Func::Ge,
            "EQUAL" => Func::Eq,
            "NOT_EQUAL" => Func::Ne,
            _ => panic!("Invalid binary operator: {}.", op),
        };
        builder.call(op, &[lhs, rhs], t_ret.clone())
    }

    fn _convert_member_expr(&mut self, t: &CArc<Type>, j: &JSON, is_lval: bool) -> NodeRef {
        let v = self._convert_expression(&j["self"], is_lval);
        let t_v = v.type_();
        let (builder, ..) = self.unwrap_ctx();
        if let Some(swizzle) = j["swizzle"].as_str() {
            assert!(t_v.is_vector());
            assert!(
                swizzle.len() >= 1 && swizzle.len() <= 4,
                "Invalid swizzle length."
            );
            let mut indices = Vec::with_capacity(swizzle.len());
            for c in swizzle.chars() {
                let i = match c {
                    'x' | 'r' | '0' => 0,
                    'y' | 'g' | '1' => 1,
                    'z' | 'b' | '2' => 2,
                    'w' | 'a' | '3' => 3,
                    _ => panic!("Invalid swizzle character: {}.", c),
                };
                assert!(i < t_v.dimension(), "Invalid swizzle index.");
                indices.push(i as u32);
            }
            if indices.len() == 1 {
                assert_eq!(t.as_ref(), t_v.element().as_ref(), "Invalid swizzle type.");
                if is_lval {
                    let i = builder.const_(Const::Uint32(indices[0]));
                    builder.gep_chained(v, &[i], t.clone())
                } else {
                    builder.extract(v, indices[0] as usize, t.clone())
                }
            } else {
                assert!(!is_lval, "L-value cannot be a swizzle.");
                let indices: Vec<_> = indices
                    .iter()
                    .map(|i| builder.const_(Const::Uint32(*i)))
                    .collect();
                assert!(
                    indices.len() >= 1 && indices.len() <= 4,
                    "Invalid swizzle length."
                );
                let t_elem = t_v.element();
                let t_swizzle = Type::vector_of(t_elem.clone(), indices.len() as u32);
                assert_eq!(t.as_ref(), t_swizzle.as_ref(), "Invalid swizzle type.");
                let args = [&[v], indices.as_slice()].concat();
                builder.call(Func::Permute, args.as_slice(), t_swizzle)
            }
        } else {
            // member access
            let i = j["member"].as_usize().unwrap();
            let t_elem = match t_v.as_ref() {
                Type::Vector(vt) => {
                    assert!((i as u32) < vt.length);
                    vt.element.to_type()
                }
                Type::Struct(st) => {
                    assert!(i < st.fields.len());
                    st.fields[i].clone()
                }
                _ => panic!("Invalid member access."),
            };
            assert_eq!(t.as_ref(), t_elem.as_ref(), "Invalid member type.");
            if is_lval {
                let i = builder.const_(Const::Uint32(i as u32));
                builder.gep_chained(v, &[i], t.clone())
            } else {
                builder.extract(v, i, t.clone())
            }
        }
    }

    fn _convert_access_expr(&mut self, t: &CArc<Type>, j: &JSON, is_lval: bool) -> NodeRef {
        let range = self._convert_expression(&j["range"], is_lval);
        let index = self._convert_expression(&j["index"], false);
        assert!(index.type_().is_int(), "Index must be an integer.");
        let t_range = range.type_();
        assert!(t_range.is_array() || t_range.is_vector() || t_range.is_matrix());
        let elem = if t_range.is_matrix() {
            Type::vector_of(t_range.element(), t_range.dimension() as u32)
        } else {
            t_range.element()
        };
        assert_eq!(elem.as_ref(), t.as_ref(), "Invalid access type.");
        let (builder, ..) = self.unwrap_ctx();
        if is_lval {
            builder.gep_chained(range, &[index], elem)
        } else {
            builder.extract_dynamic(range, index, elem)
        }
    }

    fn _convert_literal_expr(&mut self, t: &CArc<Type>, j: &JSON) -> NodeRef {
        let v = Base64::decode_vec(j["value"].as_str().unwrap()).unwrap();
        let (builder, ..) = self.unwrap_ctx();
        match t.as_ref() {
            Type::Primitive(s) => match s {
                Primitive::Bool => {
                    assert_eq!(v.len(), 1, "Invalid bool literal.");
                    unsafe {
                        let b = std::mem::transmute(v[0]);
                        builder.const_(Const::Bool(b))
                    }
                }
                Primitive::Int8 => {
                    assert_eq!(v.len(), 1, "Invalid int8 literal");
                    builder.const_(Const::Int8(v[0] as i8))
                }
                Primitive::Uint8 => {
                    assert_eq!(v.len(), 1, "Invalid uint8 literal.");
                    builder.const_(Const::Uint8(v[0]))
                }
                Primitive::Int16 => {
                    assert_eq!(v.len(), 2, "Invalid int16 literal.");
                    unsafe {
                        let i = std::mem::transmute([v[0], v[1]]);
                        builder.const_(Const::Int16(i))
                    }
                }
                Primitive::Uint16 => {
                    assert_eq!(v.len(), 2, "Invalid uint16 literal.");
                    unsafe {
                        let i = std::mem::transmute([v[0], v[1]]);
                        builder.const_(Const::Uint16(i))
                    }
                }
                Primitive::Int32 => {
                    assert_eq!(v.len(), 4, "Invalid int32 literal.");
                    unsafe {
                        let i = std::mem::transmute([v[0], v[1], v[2], v[3]]);
                        builder.const_(Const::Int32(i))
                    }
                }
                Primitive::Uint32 => {
                    assert_eq!(v.len(), 4, "Invalid uint32 literal.");
                    unsafe {
                        let i = std::mem::transmute([v[0], v[1], v[2], v[3]]);
                        builder.const_(Const::Uint32(i))
                    }
                }
                Primitive::Int64 => {
                    assert_eq!(v.len(), 8, "Invalid int64 literal.");
                    unsafe {
                        let i =
                            std::mem::transmute([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]);
                        builder.const_(Const::Int64(i))
                    }
                }
                Primitive::Uint64 => {
                    assert_eq!(v.len(), 8, "Invalid uint64 literal.");
                    unsafe {
                        let i =
                            std::mem::transmute([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]);
                        builder.const_(Const::Uint64(i))
                    }
                }
                Primitive::Float16 => {
                    assert_eq!(v.len(), 2, "Invalid float16 literal.");
                    unsafe {
                        let f = std::mem::transmute([v[0], v[1]]);
                        builder.const_(Const::Float16(f))
                    }
                }
                Primitive::Float32 => {
                    assert_eq!(v.len(), 4, "Invalid float32 literal.");
                    unsafe {
                        let f = std::mem::transmute([v[0], v[1], v[2], v[3]]);
                        builder.const_(Const::Float32(f))
                    }
                }
                Primitive::Float64 => {
                    assert_eq!(v.len(), 8, "Invalid float64 literal.");
                    unsafe {
                        let f =
                            std::mem::transmute([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]);
                        builder.const_(Const::Float64(f))
                    }
                }
            },
            Type::Vector(_) | Type::Matrix(_) => {
                assert_eq!(t.size(), v.len(), "Invalid vector/matrix literal.");
                builder.const_(Const::Generic(CBoxedSlice::new(v), t.clone()))
            }
            _ => panic!("Invalid literal type."),
        }
    }

    fn _convert_ref_expr(&mut self, _: &CArc<Type>, j: &JSON, is_lval: bool) -> NodeRef {
        let v = self
            ._curr_ctx()
            .variables
            .get(&j["variable"].as_u32().unwrap())
            .unwrap()
            .clone();
        let loadable = match v.get().instruction.as_ref() {
            Instruction::Shared => true,
            Instruction::Local { .. } => true,
            Instruction::Argument { by_value } => {
                assert!(!by_value);
                true
            }
            Instruction::Buffer => false,
            Instruction::Bindless => false,
            Instruction::Texture2D => false,
            Instruction::Texture3D => false,
            Instruction::Accel => false,
            _ => unreachable!("Invalid variable instruction."),
        };
        if is_lval || !loadable {
            v
        } else {
            let (builder, ..) = self.unwrap_ctx();
            builder.load(v)
        }
    }

    fn _convert_constant_expr(&mut self, t: &CArc<Type>, j: &JSON) -> NodeRef {
        let c = self.convert_constant(j["data"].as_usize().unwrap());
        assert_eq!(c.type_().as_ref(), t.as_ref(), "Constant type mismatch.");
        let (builder, ..) = self.unwrap_ctx();
        builder.const_(c)
    }

    fn _convert_call_builtin(&mut self, t: &CArc<Type>, f: &str, args: &JSON) -> NodeRef {
        // zero and one are special cases
        if f == "ZERO" {
            let (builder, ..) = self.unwrap_ctx();
            return builder.const_(Const::Zero(t.clone()));
        }
        if f == "ONE" {
            let (builder, ..) = self.unwrap_ctx();
            return builder.const_(Const::One(t.clone()));
        }
        let decode_string_id_expr = |j: &JSON| {
            assert_eq!(j["tag"], "STRING_ID");
            let s = j["data"].as_str().unwrap();
            CBoxedSlice::from(s.as_bytes())
        };
        let func = match f {
            "ALL" => Func::All,
            "ANY" => Func::Any,
            "SELECT" => Func::Select,
            "CLAMP" => Func::Clamp,
            "SATURATE" => Func::Saturate,
            "LERP" => Func::Lerp,
            "SMOOTHSTEP" => Func::SmoothStep,
            "STEP" => Func::Step,
            "ABS" => Func::Abs,
            "MIN" => Func::Min,
            "MAX" => Func::Max,
            "CLZ" => Func::Clz,
            "CTZ" => Func::Ctz,
            "POPCOUNT" => Func::PopCount,
            "REVERSE" => Func::Reverse,
            "ISINF" => Func::IsInf,
            "ISNAN" => Func::IsNan,
            "ACOS" => Func::Acos,
            "ACOSH" => Func::Acosh,
            "ASIN" => Func::Asin,
            "ASINH" => Func::Asinh,
            "ATAN" => Func::Atan,
            "ATAN2" => Func::Atan2,
            "ATANH" => Func::Atanh,
            "COS" => Func::Cos,
            "COSH" => Func::Cosh,
            "SIN" => Func::Sin,
            "SINH" => Func::Sinh,
            "TAN" => Func::Tan,
            "TANH" => Func::Tanh,
            "EXP" => Func::Exp,
            "EXP2" => Func::Exp2,
            "EXP10" => Func::Exp10,
            "LOG" => Func::Log,
            "LOG2" => Func::Log2,
            "LOG10" => Func::Log10,
            "POW" => Func::Powf,
            "SQRT" => Func::Sqrt,
            "RSQRT" => Func::Rsqrt,
            "CEIL" => Func::Ceil,
            "FLOOR" => Func::Floor,
            "FRACT" => Func::Fract,
            "TRUNC" => Func::Trunc,
            "ROUND" => Func::Round,
            "FMA" => Func::Fma,
            "COPYSIGN" => Func::Copysign,
            "CROSS" => Func::Cross,
            "DOT" => Func::Dot,
            "LENGTH" => Func::Length,
            "LENGTH_SQUARED" => Func::LengthSquared,
            "NORMALIZE" => Func::Normalize,
            "FACEFORWARD" => Func::Faceforward,
            "REFLECT" => Func::Reflect,
            "REDUCE_SUM" => Func::ReduceSum,
            "REDUCE_PRODUCT" => Func::ReduceProd,
            "REDUCE_MIN" => Func::ReduceMin,
            "REDUCE_MAX" => Func::ReduceMax,
            "OUTER_PRODUCT" => Func::OuterProduct,
            "MATRIX_COMPONENT_WISE_MULTIPLICATION" => Func::MatCompMul,
            "DETERMINANT" => Func::Determinant,
            "TRANSPOSE" => Func::Transpose,
            "INVERSE" => Func::Inverse,
            "SYNCHRONIZE_BLOCK" => Func::SynchronizeBlock,
            "ATOMIC_EXCHANGE" => Func::AtomicExchange,
            "ATOMIC_COMPARE_EXCHANGE" => Func::AtomicCompareExchange,
            "ATOMIC_FETCH_ADD" => Func::AtomicFetchAdd,
            "ATOMIC_FETCH_SUB" => Func::AtomicFetchSub,
            "ATOMIC_FETCH_AND" => Func::AtomicFetchAnd,
            "ATOMIC_FETCH_OR" => Func::AtomicFetchOr,
            "ATOMIC_FETCH_XOR" => Func::AtomicFetchXor,
            "ATOMIC_FETCH_MIN" => Func::AtomicFetchMin,
            "ATOMIC_FETCH_MAX" => Func::AtomicFetchMax,
            "BUFFER_READ" => Func::BufferRead,
            "BUFFER_WRITE" => Func::BufferWrite,
            "BUFFER_SIZE" => Func::BufferSize,
            "BYTE_BUFFER_READ" => Func::ByteBufferRead,
            "BYTE_BUFFER_WRITE" => Func::ByteBufferWrite,
            "BYTE_BUFFER_SIZE" => Func::ByteBufferSize,
            "TEXTURE_READ" | "TEXTURE_WRITE" | "TEXTURE_SIZE" => {
                let tt = &self.j_types[args[0]["type"].as_usize().unwrap()];
                assert_eq!(tt["tag"], "TEXTURE");
                let dim = tt["dimension"].as_u32().unwrap();
                match (dim, f) {
                    (2, "TEXTURE_READ") => Func::Texture2dRead,
                    (2, "TEXTURE_WRITE") => Func::Texture2dWrite,
                    (2, "TEXTURE_SIZE") => Func::Texture2dSize,
                    (3, "TEXTURE_READ") => Func::Texture3dRead,
                    (3, "TEXTURE_WRITE") => Func::Texture3dWrite,
                    (3, "TEXTURE_SIZE") => Func::Texture3dSize,
                    _ => panic!("Invalid texture dimension: {}.", dim),
                }
            }
            "BINDLESS_TEXTURE2D_SAMPLE" => Func::BindlessTexture2dSample,
            "BINDLESS_TEXTURE2D_SAMPLE_LEVEL" => Func::BindlessTexture2dSampleLevel,
            "BINDLESS_TEXTURE2D_SAMPLE_GRAD" => Func::BindlessTexture2dSampleGrad,
            "BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL" => Func::BindlessTexture2dSampleGradLevel,
            "BINDLESS_TEXTURE3D_SAMPLE" => Func::BindlessTexture3dSample,
            "BINDLESS_TEXTURE3D_SAMPLE_LEVEL" => Func::BindlessTexture3dSampleLevel,
            "BINDLESS_TEXTURE3D_SAMPLE_GRAD" => Func::BindlessTexture3dSampleGrad,
            "BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL" => Func::BindlessTexture3dSampleGradLevel,
            "BINDLESS_TEXTURE2D_READ" => Func::BindlessTexture2dRead,
            "BINDLESS_TEXTURE3D_READ" => Func::BindlessTexture3dRead,
            "BINDLESS_TEXTURE2D_READ_LEVEL" => Func::BindlessTexture2dReadLevel,
            "BINDLESS_TEXTURE3D_READ_LEVEL" => Func::BindlessTexture3dReadLevel,
            "BINDLESS_TEXTURE2D_SIZE" => Func::BindlessTexture2dSize,
            "BINDLESS_TEXTURE3D_SIZE" => Func::BindlessTexture3dSize,
            "BINDLESS_TEXTURE2D_SIZE_LEVEL" => Func::BindlessTexture2dSizeLevel,
            "BINDLESS_TEXTURE3D_SIZE_LEVEL" => Func::BindlessTexture3dSizeLevel,
            "BINDLESS_BUFFER_READ" => Func::BindlessBufferRead,
            "BINDLESS_BYTE_BUFFER_READ" => Func::BindlessByteBufferRead,
            "BINDLESS_BUFFER_SIZE" => Func::BindlessBufferSize,
            "BINDLESS_BUFFER_TYPE" => Func::BindlessBufferType,
            "MAKE_BOOL2" | "MAKE_SHORT2" | "MAKE_USHORT2" | "MAKE_INT2" | "MAKE_UINT2"
            | "MAKE_LONG2" | "MAKE_ULONG2" | "MAKE_HALF2" | "MAKE_FLOAT2" | "MAKE_DOUBLE2" => {
                Func::Vec2
            }
            "MAKE_BOOL3" | "MAKE_SHORT3" | "MAKE_USHORT3" | "MAKE_INT3" | "MAKE_UINT3"
            | "MAKE_LONG3" | "MAKE_ULONG3" | "MAKE_HALF3" | "MAKE_FLOAT3" | "MAKE_DOUBLE3" => {
                Func::Vec3
            }
            "MAKE_BOOL4" | "MAKE_SHORT4" | "MAKE_USHORT4" | "MAKE_INT4" | "MAKE_UINT4"
            | "MAKE_LONG4" | "MAKE_ULONG4" | "MAKE_HALF4" | "MAKE_FLOAT4" | "MAKE_DOUBLE4" => {
                Func::Vec4
            }
            "MAKE_FLOAT2X2" => Func::Mat2,
            "MAKE_FLOAT3X3" => Func::Mat3,
            "MAKE_FLOAT4X4" => Func::Mat4,
            "ASSERT" => {
                let msg = if args.len() > 1 {
                    decode_string_id_expr(&args[1])
                } else {
                    CBoxedSlice::from("Assertion failed!".as_bytes())
                };
                Func::Assert(msg)
            }
            "ASSUME" => Func::Assume,
            "UNREACHABLE" => {
                let msg = if args.len() > 0 {
                    decode_string_id_expr(&args[1])
                } else {
                    CBoxedSlice::from("Unreachable code!".as_bytes())
                };
                Func::Unreachable(msg)
            }
            "ZERO" | "ONE" => unreachable!(),
            "PACK" => Func::Pack,
            "UNPACK" => Func::Unpack,
            "REQUIRES_GRADIENT" => Func::RequiresGradient,
            "GRADIENT" => Func::Gradient,
            "GRADIENT_MARKER" => Func::GradientMarker,
            "ACCUMULATE_GRADIENT" => Func::AccGrad,
            "BACKWARD" => Func::Backward,
            "DETACH" => Func::Detach,
            "RAY_TRACING_INSTANCE_TRANSFORM" => Func::RayTracingInstanceTransform,
            "RAY_TRACING_INSTANCE_USER_ID" => Func::RayTracingSetInstanceUserId,
            "RAY_TRACING_SET_INSTANCE_TRANSFORM" => Func::RayTracingSetInstanceTransform,
            "RAY_TRACING_SET_INSTANCE_VISIBILITY" => Func::RayTracingSetInstanceVisibility,
            "RAY_TRACING_SET_INSTANCE_OPACITY" => Func::RayTracingSetInstanceOpacity,
            "RAY_TRACING_SET_INSTANCE_USER_ID" => Func::RayTracingSetInstanceUserId,
            "RAY_TRACING_TRACE_CLOSEST" => Func::RayTracingTraceClosest,
            "RAY_TRACING_TRACE_ANY" => Func::RayTracingTraceAny,
            "RAY_TRACING_QUERY_ALL" => Func::RayTracingQueryAll,
            "RAY_TRACING_QUERY_ANY" => Func::RayTracingQueryAny,
            "RAY_QUERY_WORLD_SPACE_RAY" => Func::RayQueryWorldSpaceRay,
            "RAY_QUERY_PROCEDURAL_CANDIDATE_HIT" => Func::RayQueryProceduralCandidateHit,
            "RAY_QUERY_TRIANGLE_CANDIDATE_HIT" => Func::RayQueryTriangleCandidateHit,
            "RAY_QUERY_COMMITTED_HIT" => Func::RayQueryCommittedHit,
            "RAY_QUERY_COMMIT_TRIANGLE" => Func::RayQueryCommitTriangle,
            "RAY_QUERY_COMMIT_PROCEDURAL" => Func::RayQueryCommitProcedural,
            "RAY_QUERY_TERMINATE" => Func::RayQueryTerminate,
            "RASTER_DISCARD" => unimplemented!("Func::RasterDiscard"),
            "DDX" => unimplemented!("Func::Ddx"),
            "DDY" => unimplemented!("Func::Ddy"),
            "WARP_IS_FIRST_ACTIVE_LANE" => Func::WarpIsFirstActiveLane,
            "WARP_FIRST_ACTIVE_LANE" => Func::WarpFirstActiveLane,
            "WARP_ACTIVE_ALL_EQUAL" => Func::WarpActiveAllEqual,
            "WARP_ACTIVE_BIT_AND" => Func::WarpActiveBitAnd,
            "WARP_ACTIVE_BIT_OR" => Func::WarpActiveBitOr,
            "WARP_ACTIVE_BIT_XOR" => Func::WarpActiveBitXor,
            "WARP_ACTIVE_COUNT_BITS" => Func::WarpActiveCountBits,
            "WARP_ACTIVE_MAX" => Func::WarpActiveMax,
            "WARP_ACTIVE_MIN" => Func::WarpActiveMin,
            "WARP_ACTIVE_PRODUCT" => Func::WarpActiveProduct,
            "WARP_ACTIVE_SUM" => Func::WarpActiveSum,
            "WARP_ACTIVE_ALL" => Func::WarpActiveAll,
            "WARP_ACTIVE_ANY" => Func::WarpActiveAny,
            "WARP_ACTIVE_BIT_MASK" => Func::WarpActiveBitMask,
            "WARP_PREFIX_COUNT_BITS" => Func::WarpPrefixCountBits,
            "WARP_PREFIX_SUM" => Func::WarpPrefixSum,
            "WARP_PREFIX_PRODUCT" => Func::WarpPrefixProduct,
            "WARP_READ_LANE" => Func::WarpReadLaneAt,
            "WARP_READ_FIRST_ACTIVE_LANE" => Func::WarpReadFirstLane,
            "INDIRECT_SET_DISPATCH_KERNEL" => Func::IndirectDispatchSetKernel,
            "INDIRECT_SET_DISPATCH_COUNT" => Func::IndirectDispatchSetCount,
            "SHADER_EXECUTION_REORDER" => Func::ShaderExecutionReorder,
            _ => panic!("Invalid built-in function: {}.", f),
        };

        let mut convert_args = |is_lval: &[bool]| -> Vec<_> {
            assert_eq!(is_lval.len(), args.len());
            zip(args.members(), is_lval)
                .map(|(arg, is_lval)| self._convert_expression(arg, *is_lval))
                .collect()
        };
        let check_is_ray_query = |node: NodeRef| {
            let t = node.type_();
            assert!(
                t.is_opaque("LC_RayQueryAll") || t.is_opaque("LC_RayQueryAny"),
                "Invalid ray query type."
            );
        };
        let check_is_accel = |node: NodeRef| match node.get().instruction.as_ref() {
            Instruction::Accel => {}
            _ => panic!("Invalid accel type."),
        };
        let check_is_buffer = |node: NodeRef| match node.get().instruction.as_ref() {
            Instruction::Buffer => {}
            _ => panic!("Invalid buffer type."),
        };
        let check_is_texture = |node: NodeRef| match node.get().instruction.as_ref() {
            Instruction::Texture2D => {}
            Instruction::Texture3D => {}
            _ => panic!("Invalid texture type."),
        };
        let check_is_bindless = |node: NodeRef| match node.get().instruction.as_ref() {
            Instruction::Bindless => {}
            _ => panic!("Invalid bindless type."),
        };
        let check_is_index = |t: &CArc<Type>| {
            assert!(t.is_int() && t.is_primitive());
        };
        let check_is_tex_int_coord = |t: &CArc<Type>| {
            assert!(t.is_int() && t.is_vector() && (t.dimension() == 2 || t.dimension() == 3));
        };
        let check_is_tex_float_coord = |t: &CArc<Type>| {
            assert!(t.is_float() && t.is_vector() && (t.dimension() == 2 || t.dimension() == 3));
        };
        macro_rules! check_same_types {
            ($first: expr) => {};
            ($first: expr, $($rest: expr),+) => {
                {
                    let first = $first;
                    $(
                        assert_eq!(first.as_ref(), $rest.as_ref());
                    )+
                }
            };
        }
        let args = match f {
            "ALL" | "ANY" => {
                // (boolN) -> bool
                let args = convert_args(&[false]);
                assert!(args[0].type_().is_bool() && args[0].type_().is_vector());
                assert!(t.is_bool() && t.is_primitive());
                args
            }
            "SELECT" => {
                // (T, T, bool) -> T or (vecN, vecN, boolN) -> vecN
                let args = convert_args(&[false, false, false]);
                // IR and AST have different order of arguments for select
                let args = vec![args[2], args[1], args[0]];
                assert!(args[0].type_().is_bool());
                // Now: (bool, T, T) -> T or (boolN, vecN, vecN) -> vecN
                assert!(args[0].type_().is_bool());
                check_same_types!(t, args[1].type_(), args[2].type_());
                match args[0].type_().as_ref() {
                    Type::Primitive(_) => {}
                    Type::Vector(_) => {
                        assert_eq!(args[0].type_().dimension(), t.dimension());
                    }
                    _ => panic!("Invalid select type."),
                }
                args
            }
            "CLAMP" | "LERP" => {
                // (vecN, vecN, vecN) -> vecN
                let args = convert_args(&[false, false, false]);
                check_same_types!(t, args[0].type_(), args[1].type_(), args[2].type_());
                args
            }
            "SMOOTHSTEP" | "STEP" | "FMA" => {
                // (floatN, floatN, floatN) -> floatN
                let args = convert_args(&[false, false, false]);
                check_same_types!(t, args[0].type_(), args[1].type_(), args[2].type_());
                assert!(t.is_float());
                args
            }
            "ABS" => {
                // (vecN) -> vecN
                let args = convert_args(&[false]);
                check_same_types!(t, args[0].type_());
                args
            }
            "MIN" | "MAX" => {
                // (vecN, vecN) -> vecN
                let args = convert_args(&[false, false]);
                check_same_types!(t, args[0].type_());
                args
            }
            "CLZ" | "CTZ" | "POPCOUNT" | "REVERSE" => {
                // (uintN) -> uintN
                let args = convert_args(&[false]);
                let t_arg = args[0].type_();
                assert!(t_arg.is_int() && t_arg.is_unsigned());
                check_same_types!(t, t_arg);
                args
            }
            "ISINF" | "ISNAN" => {
                // (floatN) -> boolN
                let args = convert_args(&[false]);
                let t_arg = args[0].type_();
                assert!(t_arg.is_float() && t.is_bool());
                assert_eq!(t_arg.dimension(), t.dimension());
                args
            }
            "ACOS" | "ACOSH" | "ASIN" | "ASINH" | "ATAN" | "ATANH" | "COS" | "COSH" | "SIN"
            | "SINH" | "TAN" | "TANH" | "EXP" | "EXP2" | "EXP10" | "LOG" | "LOG2" | "LOG10"
            | "CEIL" | "FLOOR" | "FRACT" | "TRUNC" | "ROUND" | "SQRT" | "RSQRT" | "SATURATE"
            | "NORMALIZE" => {
                // (floatN) -> floatN
                let args = convert_args(&[false]);
                check_same_types!(t, args[0].type_());
                assert!(t.is_float());
                args
            }
            "POW" | "ATAN2" | "COPYSIGN" => {
                // (floatN, floatN) -> floatN
                let args = convert_args(&[false, false]);
                check_same_types!(t, args[0].type_(), args[1].type_());
                assert!(t.is_float());
                args
            }
            "CROSS" | "REFLECT" => {
                // (float3, float3) -> float3
                let args = convert_args(&[false, false]);
                check_same_types!(t, args[0].type_(), args[1].type_());
                assert!(t.is_float());
                assert_eq!(t.dimension(), 3);
                args
            }
            "DOT" => {
                // (floatN, floatN) -> float
                let args = convert_args(&[false, false]);
                check_same_types!(args[0].type_(), args[1].type_());
                let t_arg = args[0].type_();
                assert_eq!(t_arg.element().as_ref(), t.as_ref());
                assert!(t_arg.is_float() && t_arg.is_vector() && t.is_float());
                args
            }
            "LENGTH" | "LENGTH_SQUARED" => {
                // (floatN) -> float
                let args = convert_args(&[false]);
                let t_arg = args[0].type_();
                assert_eq!(t_arg.element().as_ref(), t.as_ref());
                assert!(t_arg.is_float() && t_arg.is_vector() && t.is_float());
                args
            }
            "FACEFORWARD" => {
                // (float3, float3, float3) -> float3
                let args = convert_args(&[false, false, false]);
                check_same_types!(t, args[0].type_(), args[1].type_(), args[2].type_());
                assert!(t.is_float());
                assert_eq!(t.dimension(), 3);
                args
            }
            "REDUCE_SUM" | "REDUCE_PRODUCT" | "REDUCE_MIN" | "REDUCE_MAX" => {
                // (vecN) -> scalar or (matNxN) -> scalar
                let args = convert_args(&[false]);
                let t_arg = args[0].type_();
                assert!(t_arg.is_vector() || t_arg.is_matrix());
                assert_eq!(t_arg.element().as_ref(), t.as_ref());
                args
            }
            "OUTER_PRODUCT" => {
                // (floatN, floatN) -> floatNxN or (floatNxN, floatNxN) -> floatNxN
                let args = convert_args(&[false, false]);
                let t_lhs = args[0].type_();
                let t_rhs = args[1].type_();
                check_same_types!(t.element(), t_lhs.element(), t_rhs.element());
                assert_eq!(t_lhs.dimension(), t_rhs.dimension());
                assert_eq!(t_lhs.dimension(), t.dimension());
                match (t_lhs.as_ref(), t_rhs.as_ref(), t.as_ref()) {
                    (Type::Vector(_), Type::Vector(_), Type::Matrix(_)) => {}
                    (Type::Matrix(_), Type::Matrix(_), Type::Matrix(_)) => {}
                    _ => panic!("Invalid outer product type."),
                }
                args
            }
            "MATRIX_COMPONENT_WISE_MULTIPLICATION" => {
                // (floatNxN, floatNxN) -> floatNxN
                let args = convert_args(&[false, false]);
                check_same_types!(t, args[0].type_(), args[1].type_());
                assert!(t.is_matrix());
                args
            }
            "DETERMINANT" => {
                // (floatNxN) -> float
                let args = convert_args(&[false]);
                let t_arg = args[0].type_();
                assert!(t_arg.is_matrix());
                assert_eq!(t_arg.element().as_ref(), t.as_ref());
                assert!(t.is_float());
                args
            }
            "TRANSPOSE" | "INVERSE" => {
                // (floatNxN) -> floatNxN
                let args = convert_args(&[false]);
                check_same_types!(t, args[0].type_());
                assert!(t.is_matrix());
                args
            }
            "SYNCHRONIZE_BLOCK" => {
                // () -> void
                assert!(t.is_void());
                convert_args(&[])
            }
            "ATOMIC_EXCHANGE" | "ATOMIC_FETCH_ADD" | "ATOMIC_FETCH_SUB" | "ATOMIC_FETCH_MIN"
            | "ATOMIC_FETCH_MAX" => {
                assert!(args.len() >= 3);
                let is_lval: Vec<_> = args.members().enumerate().map(|(i, _)| i == 0).collect();
                let args = convert_args(is_lval.as_slice());
                check_same_types!(args.last().unwrap().type_(), t);
                assert!(t.is_primitive() && (t.is_int() || t.is_float()));
                args
            }
            "ATOMIC_COMPARE_EXCHANGE" => {
                assert!(args.len() >= 4);
                let is_lval: Vec<_> = args.members().enumerate().map(|(i, _)| i == 0).collect();
                let args = convert_args(is_lval.as_slice());
                let n = args.len();
                check_same_types!(args[n - 2].type_(), args[n - 1].type_(), t);
                assert!(t.is_primitive() && (t.is_int() || t.is_float()));
                args
            }
            "ATOMIC_FETCH_AND" | "ATOMIC_FETCH_OR" | "ATOMIC_FETCH_XOR" => {
                // [(atomic_ref, val) -> old]: stores (old & val), returns old.
                assert!(args.len() >= 3);
                let is_lval: Vec<_> = args.members().enumerate().map(|(i, _)| i == 0).collect();
                let args = convert_args(is_lval.as_slice());
                check_same_types!(args.last().unwrap().type_(), t);
                assert!(t.is_primitive() && t.is_int());
                args
            }
            "BUFFER_READ" | "BYTE_BUFFER_READ" => {
                // [(buffer, index) -> value]: reads the index-th element in buffer
                let args = convert_args(&[false, false]);
                check_is_buffer(args[0]);
                check_is_index(args[1].type_());
                args
            }
            "BUFFER_WRITE" | "BYTE_BUFFER_WRITE" => {
                // [(buffer, index, value) -> void]: writes value into the index-th element of buffer
                let args = convert_args(&[false, false, false]);
                check_is_buffer(args[0]);
                check_is_index(args[1].type_());
                assert!(t.is_void());
                args
            }
            "BUFFER_SIZE" | "BYTE_BUFFER_SIZE" => {
                // [(buffer) -> size]
                let args = convert_args(&[false]);
                check_is_buffer(args[0]);
                assert!(t.is_int() && t.is_unsigned());
                args
            }
            "TEXTURE_READ" => {
                // [(texture, coord) -> value]
                let args = convert_args(&[false, false]);
                check_is_texture(args[0]);
                check_is_tex_int_coord(args[1].type_());
                args
            }
            "TEXTURE_WRITE" => {
                // [(texture, coord, value) -> void]
                let args = convert_args(&[false, false, false]);
                check_is_texture(args[0]);
                check_is_tex_int_coord(args[1].type_());
                assert!(t.is_void());
                args
            }
            "TEXTURE_SIZE" => {
                // [(texture) -> Vector<uint, dim>]
                let args = convert_args(&[false]);
                check_is_texture(args[0]);
                check_is_tex_int_coord(t);
                args
            }
            "BINDLESS_TEXTURE2D_SAMPLE" | "BINDLESS_TEXTURE3D_SAMPLE" => {
                let args = convert_args(&[false, false, false]);
                check_is_bindless(args[0]);
                check_is_index(args[1].type_());
                check_is_tex_float_coord(args[2].type_());
                args
            }
            "BINDLESS_TEXTURE2D_SAMPLE_LEVEL" | "BINDLESS_TEXTURE3D_SAMPLE_LEVEL" => {
                let args = convert_args(&[false, false, false, false]);
                check_is_bindless(args[0]);
                check_is_index(args[1].type_());
                check_is_tex_float_coord(args[2].type_());
                assert!(args[3].type_().is_float() && args[3].type_().is_primitive());
                args
            }
            "BINDLESS_TEXTURE2D_SAMPLE_GRAD" | "BINDLESS_TEXTURE3D_SAMPLE_GRAD" => {
                let args = convert_args(&[false, false, false, false, false]);
                check_is_bindless(args[0]);
                check_is_index(args[1].type_());
                check_is_tex_float_coord(args[2].type_());
                check_is_tex_float_coord(args[3].type_());
                check_is_tex_float_coord(args[4].type_());
                args
            }
            "BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL" | "BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL" => {
                let args = convert_args(&[false, false, false, false, false, false]);
                check_is_bindless(args[0]);
                check_is_index(args[1].type_());
                check_is_tex_float_coord(args[2].type_());
                check_is_tex_float_coord(args[3].type_());
                check_is_tex_float_coord(args[4].type_());
                assert!(args[5].type_().is_float() && args[5].type_().is_primitive());
                args
            }
            "BINDLESS_TEXTURE2D_READ" | "BINDLESS_TEXTURE3D_READ" => {
                let args = convert_args(&[false, false, false]);
                check_is_bindless(args[0]);
                check_is_index(args[1].type_());
                check_is_tex_int_coord(args[2].type_());
                args
            }
            "BINDLESS_TEXTURE2D_READ_LEVEL" | "BINDLESS_TEXTURE3D_READ_LEVEL" => {
                let args = convert_args(&[false, false, false, false]);
                check_is_bindless(args[0]);
                check_is_index(args[1].type_());
                check_is_tex_int_coord(args[2].type_());
                check_is_index(args[3].type_());
                args
            }
            "BINDLESS_TEXTURE2D_SIZE" | "BINDLESS_TEXTURE3D_SIZE" => {
                let args = convert_args(&[false, false]);
                check_is_bindless(args[0]);
                check_is_index(args[1].type_());
                check_is_tex_int_coord(t);
                args
            }
            "BINDLESS_TEXTURE2D_SIZE_LEVEL" | "BINDLESS_TEXTURE3D_SIZE_LEVEL" => {
                let args = convert_args(&[false, false, false]);
                check_is_bindless(args[0]);
                check_is_index(args[1].type_());
                check_is_index(args[2].type_());
                check_is_tex_int_coord(t);
                args
            }
            "BINDLESS_BUFFER_READ" | "BINDLESS_BYTE_BUFFER_READ" => {
                let args = convert_args(&[false, false, false]);
                check_is_bindless(args[0]);
                check_is_index(args[1].type_());
                check_is_index(args[2].type_());
                args
            }
            "BINDLESS_BUFFER_SIZE" => {
                // (bindless_array, index: uint, stride: uint) -> size
                let args = convert_args(&[false, false, false]);
                check_is_bindless(args[0]);
                check_is_index(args[1].type_());
                check_is_index(args[2].type_());
                check_is_index(t);
                args
            }
            "BINDLESS_BUFFER_TYPE" => {
                let args = convert_args(&[false, false]);
                check_is_bindless(args[0]);
                check_is_index(args[1].type_());
                check_is_index(t);
                args
            }
            "MAKE_BOOL2" | "MAKE_BOOL3" | "MAKE_BOOL4" | "MAKE_INT2" | "MAKE_INT3"
            | "MAKE_INT4" | "MAKE_UINT2" | "MAKE_UINT3" | "MAKE_UINT4" | "MAKE_FLOAT2"
            | "MAKE_FLOAT3" | "MAKE_FLOAT4" | "MAKE_SHORT2" | "MAKE_SHORT3" | "MAKE_SHORT4"
            | "MAKE_USHORT2" | "MAKE_USHORT3" | "MAKE_USHORT4" | "MAKE_LONG2" | "MAKE_LONG3"
            | "MAKE_LONG4" | "MAKE_ULONG2" | "MAKE_ULONG3" | "MAKE_ULONG4" | "MAKE_HALF2"
            | "MAKE_HALF3" | "MAKE_HALF4" | "MAKE_DOUBLE2" | "MAKE_DOUBLE3" | "MAKE_DOUBLE4" => {
                let n = f.chars().last().unwrap().to_digit(10).unwrap();
                let elem = &f[5..f.len() - 1];
                let elem = match elem {
                    "BOOL" => <bool as TypeOf>::type_(),
                    "INT" => <i32 as TypeOf>::type_(),
                    "UINT" => <u32 as TypeOf>::type_(),
                    "FLOAT" => <f32 as TypeOf>::type_(),
                    "SHORT" => <i16 as TypeOf>::type_(),
                    "USHORT" => <u16 as TypeOf>::type_(),
                    "LONG" => <i64 as TypeOf>::type_(),
                    "ULONG" => <u64 as TypeOf>::type_(),
                    "HALF" => <f16 as TypeOf>::type_(),
                    "DOUBLE" => <f64 as TypeOf>::type_(),
                    _ => panic!("Invalid vector element type: {}.", elem),
                };
                let ret = Type::vector_of(elem.clone(), n);
                check_same_types!(t, ret);
                let is_lval: Vec<_> = (0..args.len()).map(|_| false).collect();
                let args = convert_args(is_lval.as_slice());
                let (builder, ..) = self.unwrap_ctx();
                if args.len() == 1 {
                    if args[0].type_().is_primitive() {
                        let s = Self::_cast(builder, &elem, args[0]);
                        (0..n).map(|_| s.clone()).collect()
                    } else {
                        let v = args[0];
                        let v_elem = v.type_().element();
                        assert!(v.type_().dimension() >= n as usize);
                        (0..n)
                            .map(|i| {
                                let x = builder.extract(v, i as usize, v_elem.clone());
                                Self::_cast(builder, &elem, x)
                            })
                            .collect()
                    }
                } else {
                    let mut scalars = Vec::new();
                    for arg in args {
                        if arg.type_().is_primitive() {
                            assert_eq!(arg.type_().as_ref(), elem.as_ref());
                            scalars.push(arg);
                        } else {
                            assert_eq!(arg.type_().element().as_ref(), elem.as_ref());
                            assert!(arg.type_().is_vector());
                            for i in 0..arg.type_().dimension() {
                                scalars.push(builder.extract(arg, i, elem.clone()));
                            }
                        };
                    }
                    assert_eq!(scalars.len(), n as usize);
                    scalars
                }
            }
            "MAKE_FLOAT2X2" | "MAKE_FLOAT3X3" | "MAKE_FLOAT4X4" => {
                let n = f.chars().last().unwrap().to_digit(10).unwrap();
                let ret = Type::matrix(Primitive::Float32, n);
                check_same_types!(t, ret);
                let is_lval: Vec<_> = (0..n).map(|_| false).collect();
                let args = convert_args(is_lval.as_slice());
                let col = Type::vector(Primitive::Float32, n);
                args.iter().for_each(|arg| {
                    assert_eq!(arg.type_().as_ref(), col.as_ref());
                });
                args
            }
            "ASSERT" => {
                assert!(args.len() == 1 || args.len() == 2);
                let a = self._convert_expression(&args[0], false);
                assert!(a.type_().is_bool() && a.type_().is_primitive());
                assert!(t.is_void());
                vec![a]
            }
            "ASSUME" => {
                let args = convert_args(&[false]);
                assert!(args[0].type_().is_bool());
                assert!(t.is_void());
                args
            }
            "UNREACHABLE" => Vec::new(),
            "ZERO" => unreachable!(),
            "ONE" => unreachable!(),
            "PACK" => {
                let args = convert_args(&[false, false, false]);
                check_is_buffer(args[1]);
                check_is_index(args[2].type_());
                assert!(t.is_void());
                args
            }
            "UNPACK" => {
                let args = convert_args(&[false, false]);
                check_is_buffer(args[0]);
                check_is_index(args[1].type_());
                args
            }
            "REQUIRES_GRADIENT" => {
                let args = convert_args(&[true]);
                assert!(args[0].is_local());
                assert!(t.is_void());
                args
            }
            "GRADIENT" => {
                let args = convert_args(&[true]);
                assert!(args[0].is_local());
                check_same_types!(t, args[0].type_());
                args
            }
            "GRADIENT_MARKER" => {
                let args = convert_args(&[false, false]);
                check_same_types!(args[0].type_(), args[1].type_());
                assert!(t.is_void());
                args
            }
            "ACCUMULATE_GRADIENT" => {
                let args = convert_args(&[true, false]);
                check_same_types!(args[0].type_(), args[1].type_());
                assert!(t.is_void());
                args
            }
            "BACKWARD" => {
                let args = convert_args(&[false]);
                assert!(t.is_void());
                args
            }
            "DETACH" => {
                let args = convert_args(&[false]);
                check_same_types!(t, args[0].type_());
                args
            }
            "RAY_TRACING_INSTANCE_TRANSFORM" => {
                let args = convert_args(&[false, false]);
                check_is_accel(args[0]);
                check_is_index(args[1].type_());
                assert!(t.is_matrix() && t.is_float());
                assert_eq!(t.dimension(), 4);
                args
            }
            "RAY_TRACING_INSTANCE_USER_ID" => {
                let args = convert_args(&[false, false]);
                check_is_accel(args[0]);
                check_is_index(args[1].type_());
                assert!(t.is_int() && t.is_primitive());
                args
            }
            "RAY_TRACING_SET_INSTANCE_TRANSFORM" => {
                // (Accel, uint, float4x4)
                let args = convert_args(&[false, false, false]);
                check_is_accel(args[0]);
                check_is_index(args[1].type_());
                assert!(args[2].type_().is_matrix() && args[2].type_().is_float());
                assert_eq!(args[2].type_().dimension(), 4);
                assert!(t.is_void());
                args
            }
            "RAY_TRACING_SET_INSTANCE_VISIBILITY" => {
                // (Accel, uint, uint)
                let args = convert_args(&[false, false, false]);
                check_is_accel(args[0]);
                check_is_index(args[1].type_());
                assert!(args[2].type_().is_int() && args[2].type_().is_primitive());
                assert!(t.is_void());
                args
            }
            "RAY_TRACING_SET_INSTANCE_OPACITY" => {
                // (Accel, uint, bool)
                let args = convert_args(&[false, false, false]);
                check_is_accel(args[0]);
                check_is_index(args[1].type_());
                assert!(args[2].type_().is_bool() && args[2].type_().is_primitive());
                assert!(t.is_void());
                args
            }
            "RAY_TRACING_SET_INSTANCE_USER_ID" => {
                // (Accel, uint, uint)
                let args = convert_args(&[false, false, false]);
                check_is_accel(args[0]);
                check_is_index(args[1].type_());
                check_is_index(args[2].type_());
                assert!(t.is_void());
                args
            }
            "RAY_TRACING_TRACE_CLOSEST" => {
                // (Accel, ray, mask: uint): TriangleHit
                let args = convert_args(&[false, false, false]);
                check_is_accel(args[0]);
                assert!(args[2].type_().is_int() && args[2].type_().is_primitive());
                args
            }
            "RAY_TRACING_TRACE_ANY" => {
                // (Accel, ray, mask: uint): bool
                let args = convert_args(&[false, false, false]);
                check_is_accel(args[0]);
                assert!(args[2].type_().is_int() && args[2].type_().is_primitive());
                assert!(t.is_bool() && t.is_primitive());
                args
            }
            "RAY_TRACING_QUERY_ALL" | "RAY_TRACING_QUERY_ANY" => {
                // (Accel, ray, mask: uint): RayQuery
                let args = convert_args(&[false, false, false]);
                check_is_accel(args[0]);
                assert!(args[2].type_().is_int() && args[2].type_().is_primitive());
                args
            }
            "RAY_QUERY_WORLD_SPACE_RAY"
            | "RAY_QUERY_PROCEDURAL_CANDIDATE_HIT"
            | "RAY_QUERY_TRIANGLE_CANDIDATE_HIT"
            | "RAY_QUERY_COMMITTED_HIT" => {
                let args = convert_args(&[true]);
                check_is_ray_query(args[0]);
                args
            }
            "RAY_QUERY_COMMIT_TRIANGLE" => {
                assert!(t.is_void());
                let args = convert_args(&[true]);
                check_is_ray_query(args[0]);
                args
            }
            "RAY_QUERY_COMMIT_PROCEDURAL" => {
                assert!(t.is_void());
                let args = convert_args(&[true, false]);
                check_is_ray_query(args[0]);
                assert!(args[1].type_().is_float() && args[1].type_().is_primitive());
                args
            }
            "RAY_QUERY_TERMINATE" => {
                assert!(t.is_void());
                let args = convert_args(&[true]);
                check_is_ray_query(args[0]);
                args
            }
            "RASTER_DISCARD" => unimplemented!("Func::RasterDiscard"),
            "DDX" => unimplemented!("Func::Ddx"),
            "DDY" => unimplemented!("Func::Ddy"),
            "WARP_IS_FIRST_ACTIVE_LANE" => {
                let args = convert_args(&[]);
                assert!(t.is_bool() && t.is_primitive());
                args
            }
            "WARP_FIRST_ACTIVE_LANE" => {
                let args = convert_args(&[]);
                assert!(t.is_int() && t.is_primitive());
                args
            }
            "WARP_ACTIVE_ALL_EQUAL" => {
                // (scalar/vector): boolN
                let args = convert_args(&[false]);
                assert!(t.is_bool());
                assert_eq!(args[0].type_().dimension(), t.dimension());
                args
            }
            "WARP_ACTIVE_BIT_AND" | "WARP_ACTIVE_BIT_OR" | "WARP_ACTIVE_BIT_XOR" => {
                // (scalar/vector): uintN
                let args = convert_args(&[false]);
                check_same_types!(t, args[0].type_());
                assert!(t.is_int());
                args
            }
            "WARP_ACTIVE_COUNT_BITS" | "WARP_PREFIX_COUNT_BITS" => {
                // (bool): uint
                let args = convert_args(&[false]);
                assert!(t.is_int() && t.is_primitive());
                assert!(args[0].type_().is_bool() && args[0].type_().is_primitive());
                args
            }
            "WARP_ACTIVE_MAX"
            | "WARP_ACTIVE_MIN"
            | "WARP_ACTIVE_PRODUCT"
            | "WARP_ACTIVE_SUM"
            | "WARP_PREFIX_SUM"
            | "WARP_PREFIX_PRODUCT" => {
                let args = convert_args(&[false]);
                check_same_types!(t, args[0].type_());
                assert!(t.is_primitive() || t.is_vector());
                args
            }
            "WARP_ACTIVE_ALL" | "WARP_ACTIVE_ANY" => {
                // (bool): bool
                let args = convert_args(&[false]);
                check_same_types!(args[0].type_(), t);
                assert!(t.is_bool() && t.is_primitive());
                args
            }
            "WARP_ACTIVE_BIT_MASK" => {
                // (bool): uint4 (uint4 contained 128-bit)
                let args = convert_args(&[false]);
                assert!(t.is_int() && t.is_vector() && t.dimension() == 4);
                assert!(args[0].type_().is_bool() && args[0].type_().is_primitive());
                args
            }
            "WARP_READ_LANE" => {
                // (type: scalar/vector/matrix, index: uint): type (read this variable's value at this lane)
                let args = convert_args(&[false, false]);
                check_same_types!(args[0].type_(), t);
                assert!(t.is_primitive() || t.is_vector() || t.is_matrix());
                check_is_index(args[1].type_());
                args
            }
            "WARP_READ_FIRST_ACTIVE_LANE" => {
                // (type: scalar/vector/matrix): type (read this variable's value at the first lane)
                let args = convert_args(&[false]);
                check_same_types!(args[0].type_(), t);
                assert!(t.is_primitive() || t.is_vector() || t.is_matrix());
                args
            }
            "INDIRECT_SET_DISPATCH_KERNEL" => {
                // (Buffer, uint offset, uint3 block_size, uint3 dispatch_size, uint kernel_id)
                let args = convert_args(&[false, false, false, false, false]);
                check_same_types!(args[1].type_(), args[4].type_());
                check_same_types!(args[2].type_(), args[3].type_());
                check_is_index(args[1].type_());
                assert!(
                    args[3].type_().is_int()
                        && args[3].type_().is_vector()
                        && args[3].type_().dimension() == 3
                );
                args
            }
            "INDIRECT_SET_DISPATCH_COUNT" => {
                // (Buffer, uint count)
                let args = convert_args(&[false, false]);
                check_is_index(args[1].type_());
                assert!(t.is_void());
                args
            }
            "SHADER_EXECUTION_REORDER" => {
                // (uint hint, uint hint_bits): void
                let args = convert_args(&[false, false]);
                check_same_types!(args[0].type_(), args[1].type_());
                assert!(args[0].type_().is_int() && args[0].type_().is_primitive());
                assert!(t.is_void());
                args
            }
            _ => panic!("Invalid built-in function: {}.", f),
        };
        let (builder, ..) = self.unwrap_ctx();
        builder.call(func, args.as_slice(), t.clone())
    }

    fn _convert_call_custom(&mut self, t: &CArc<Type>, f: usize, args: &JSON) -> NodeRef {
        let f = match self.convert_function(f) {
            FunctionModule::Callable(callable) => callable,
            _ => panic!("Invalid custom function."),
        };
        self._curr_ctx_mut().has_autodiff |= f
            .module
            .flags
            .contains(ModuleFlags::REQUIRES_REV_AD_TRANSFORM);
        let args: Vec<_> = f
            .args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let by_value = match a.get().instruction.as_ref() {
                    Instruction::Argument { by_value } => *by_value,
                    _ => true,
                };
                let arg = self._convert_expression(&args[i], !by_value);
                if arg.type_().as_ref() != a.get().type_.as_ref() {
                    assert!(by_value, "Invalid argument type.");
                    let (builder, ..) = self.unwrap_ctx();
                    Self::_cast(builder, &a.get().type_, arg)
                } else {
                    arg
                }
            })
            .collect();
        let (builder, ..) = self.unwrap_ctx();
        assert_eq!(f.ret_type.as_ref(), t.as_ref(), "Invalid return type.");
        builder.call(
            Func::Callable(CallableModuleRef(f)),
            args.as_slice(),
            t.clone(),
        )
    }

    fn _convert_call_expr(&mut self, t: &CArc<Type>, j: &JSON) -> NodeRef {
        let op = j["op"].as_str().unwrap();
        match op {
            "CUSTOM" => {
                self._convert_call_custom(t, j["custom"].as_usize().unwrap(), &j["arguments"])
            }
            "EXTERNAL" => unimplemented!("External calls."),
            _ => {
                // built-in functions
                assert!(!j.contains("custom") && !j.contains("external"));
                self._convert_call_builtin(t, op, &j["arguments"])
            }
        }
    }

    fn _convert_cast_expr(&mut self, t: &CArc<Type>, j: &JSON) -> NodeRef {
        let expr = self._convert_expression(&j["expression"], false);
        let op = j["op"].as_str().unwrap();
        let (builder, ..) = self.unwrap_ctx();
        match op {
            "BITWISE" => builder.call(Func::Bitcast, &[expr], t.clone()),
            "STATIC" => Self::_cast(builder, t, expr),
            _ => panic!("Invalid cast operator: {}.", op),
        }
    }

    fn _convert_expression(&mut self, j: &JSON, is_lval: bool) -> NodeRef {
        if j.is_null() {
            assert!(!is_lval, "L-value cannot be null.");
            INVALID_REF
        } else {
            let tag = j["tag"].as_str().unwrap();
            let t = self.convert_type(j["type"].as_usize().unwrap());
            match tag {
                "UNARY" => {
                    assert!(!is_lval, "Unary expressions cannot be used as L-values.");
                    self._convert_unary_expr(&t, j)
                }
                "BINARY" => {
                    assert!(!is_lval, "Binary expressions cannot be used as L-values.");
                    self._convert_binary_expr(&t, j)
                }
                "MEMBER" => self._convert_member_expr(&t, j, is_lval),
                "ACCESS" => self._convert_access_expr(&t, j, is_lval),
                "LITERAL" => {
                    assert!(!is_lval, "Literals cannot be used as L-values.");
                    self._convert_literal_expr(&t, j)
                }
                "REF" => self._convert_ref_expr(&t, j, is_lval),
                "CONSTANT" => {
                    assert!(!is_lval, "Constants cannot be used as L-values.");
                    self._convert_constant_expr(&t, j)
                }
                "CALL" => {
                    assert!(!is_lval, "Call expressions cannot return L-values.");
                    self._convert_call_expr(&t, j)
                }
                "CAST" => {
                    assert!(!is_lval, "Cast expressions cannot be used as L-values.");
                    self._convert_cast_expr(&t, j)
                }
                "TYPE_ID" => unimplemented!("TypeID expressions."),
                "STRING_ID" => unimplemented!("StringID expressions."),
                _ => panic!("Invalid expression tag: {}", tag),
            }
        }
    }

    fn _with_builder<F: FnOnce(&mut Self)>(&mut self, f: F) -> Pooled<BasicBlock> {
        let builder = IrBuilder::new(self.pools.clone());
        let old_builder = self._curr_ctx_mut().builder.replace(builder);
        f(self);
        let builder = self._curr_ctx_mut().builder.take().unwrap();
        self._curr_ctx_mut().builder = old_builder;
        builder.finish()
    }

    fn _convert_scope(&mut self, j: &JSON, ignore_top_level_break: bool) -> Pooled<BasicBlock> {
        assert_eq!(j["tag"], "SCOPE", "Scope must be a scope.");
        self._with_builder(|this| {
            for s in j["statements"].members() {
                if ignore_top_level_break && s["tag"].as_str().unwrap() == "BREAK" {
                    break;
                }
                this._convert_statement(s);
            }
        })
    }

    fn _convert_statement(&mut self, j: &JSON) -> NodeRef {
        let tag = j["tag"].as_str().unwrap();
        match tag {
            "BREAK" => {
                let (builder, ..) = self.unwrap_ctx();
                builder.break_()
            }
            "CONTINUE" => {
                let (builder, ..) = self.unwrap_ctx();
                builder.continue_()
            }
            "RETURN" => {
                let v = self._convert_expression(&j["expression"], false);
                let ret_type = self._curr_ctx().ret_type.clone();
                let (builder, ..) = self.unwrap_ctx();
                let v = Self::_cast(builder, &ret_type, v);
                builder.return_(v)
            }
            "SCOPE" => unreachable!("Scope should be handled by _convert_scope."),
            "IF" => {
                let cond = self._convert_expression(&j["condition"], false);
                let true_ = self._convert_scope(&j["true_branch"], false);
                let false_ = self._convert_scope(&j["false_branch"], false);
                let (builder, ..) = self.unwrap_ctx();
                let cond = Self::_cast(builder, &<bool as TypeOf>::type_(), cond);
                builder.if_(cond, true_, false_)
            }
            "LOOP" => {
                let body = self._convert_scope(&j["body"], false);
                let (builder, ..) = self.unwrap_ctx();
                let cond = builder.const_(Const::Bool(true));
                builder.loop_(body, cond)
            }
            "EXPR" => {
                self._convert_expression(&j["expression"], false);
                INVALID_REF
            }
            "SWITCH" => {
                let v = self._convert_expression(&j["expression"], false);
                let body = &j["body"];
                assert_eq!(body["tag"], "SCOPE", "Switch body must be a scope.");
                let cases: Vec<_> = body["statements"]
                    .members()
                    .filter_map(|s| {
                        let s_tag = s["tag"].as_str().unwrap();
                        match s_tag {
                            "SWITCH_CASE" => Some(SwitchCase {
                                value: s["value"].as_i32().unwrap(),
                                block: self._convert_scope(&s["body"], true),
                            }),
                            "SWITCH_DEFAULT" => None,
                            "COMMENT" => None,
                            _ => panic!("Invalid switch case tag: {}", s_tag),
                        }
                    })
                    .collect();
                let default: Vec<_> = body["statements"]
                    .members()
                    .filter_map(|s| {
                        let s_tag = s["tag"].as_str().unwrap();
                        match s_tag {
                            "SWITCH_CASE" => None,
                            "SWITCH_DEFAULT" => Some(self._convert_scope(&s["body"], true)),
                            "COMMENT" => None,
                            _ => panic!("Invalid switch case tag: {}", s_tag),
                        }
                    })
                    .collect();
                let default = if default.is_empty() {
                    IrBuilder::new(self.pools.clone()).finish()
                } else {
                    assert_eq!(default.len(), 1, "Switch can only have one default case.");
                    default[0]
                };
                let (builder, ..) = self.unwrap_ctx();
                builder.switch(v, cases.as_slice(), default)
            }
            "SWITCH_CASE" => {
                unreachable!("Switch case should be handled by _convert_statement.")
            }
            "SWITCH_DEFAULT" => {
                unreachable!("Switch default should be handled by _convert_statement.")
            }
            "ASSIGN" => {
                let lhs = self._convert_expression(&j["lhs"], true);
                let rhs = self._convert_expression(&j["rhs"], false);
                let (builder, ..) = self.unwrap_ctx();
                let rhs = Self::_cast(builder, lhs.type_(), rhs);
                builder.update(lhs, rhs)
            }
            "FOR" => {
                let mut cond = INVALID_REF;
                let prepare = self._with_builder(|this| {
                    cond = this._convert_expression(&j["condition"], false);
                    let (builder, ..) = this.unwrap_ctx();
                    cond = Self::_cast(builder, &<bool as TypeOf>::type_(), cond);
                });
                let body = self._convert_scope(&j["body"], false);
                let update = self._with_builder(|this| {
                    let var = this._convert_expression(&j["variable"], true);
                    let step = this._convert_expression(&j["step"], false);
                    let (builder, ..) = this.unwrap_ctx();
                    let step = Self::_cast(builder, &var.type_(), step);
                    let next = builder.call(Func::Add, &[var, step], var.type_().clone());
                    builder.update(var, next);
                });
                let (builder, ..) = self.unwrap_ctx();
                builder.generic_loop(prepare, cond, body, update)
            }
            "COMMENT" => {
                let comment = j["comment"].as_str().unwrap();
                let (builder, ..) = self.unwrap_ctx();
                builder.comment(comment.to_string().into())
            }
            "RAY_QUERY" => {
                let rq = self._convert_expression(&j["query"], true);
                let on_triangle_candidate = &j["on_triangle_candidate"];
                assert_eq!(
                    on_triangle_candidate["tag"], "SCOPE",
                    "On triangle candidate must be a scope."
                );
                let on_triangle_candidate = self._convert_scope(on_triangle_candidate, false);
                let on_procedural_candidate = &j["on_procedural_candidate"];
                assert_eq!(
                    on_procedural_candidate["tag"], "SCOPE",
                    "On procedural candidate must be a scope."
                );
                let on_procedural_candidate = self._convert_scope(on_procedural_candidate, false);
                let (builder, ..) = self.unwrap_ctx();
                builder.ray_query(
                    rq,
                    on_triangle_candidate,
                    on_procedural_candidate,
                    Type::void(),
                )
            }
            "AUTO_DIFF" => {
                self._curr_ctx_mut().has_autodiff = true;
                let body = self._convert_scope(&j["body"], false);
                let (builder, ..) = self.unwrap_ctx();
                builder.ad_scope(body)
            }
            _ => panic!("Invalid statement tag: {}", tag),
        }
    }

    fn _convert_module(&mut self, kind: ModuleKind) -> Module {
        // push builder
        let builder = IrBuilder::new(self.pools.clone());
        let old_builder = self._curr_ctx_mut().builder.replace(builder);
        // convert variables
        self.convert_variables();
        // convert body
        self._curr_ctx().j["body"]["statements"]
            .members()
            .for_each(|s| {
                self._convert_statement(s);
            });
        // pop builder
        let builder = self._curr_ctx_mut().builder.take().unwrap();
        self._curr_ctx_mut().builder = old_builder;
        // finalize
        let entry = builder.finish();
        Module {
            kind,
            entry,
            pools: self.pools.clone(),
            flags: if self._curr_ctx().has_autodiff {
                ModuleFlags::REQUIRES_REV_AD_TRANSFORM
            } else {
                ModuleFlags::NONE
            },
        }
    }

    fn _do_convert_kernel(&mut self) -> KernelModule {
        let module = self._convert_module(ModuleKind::Kernel);
        let bound_args = &self._curr_ctx().j["bound_arguments"];
        let args: Vec<_> = self._curr_ctx().j["arguments"]
            .members()
            .map(|a| {
                let a = a.as_usize().unwrap();
                self._curr_ctx().arguments.get(&(a as u32)).unwrap().clone()
            })
            .collect();
        let captures: Vec<_> = bound_args
            .members()
            .enumerate()
            .map(|(i, a)| {
                let tag = a["tag"].as_str().unwrap();
                let handle = a["handle"].as_str().unwrap().parse().unwrap();
                let node = args[i].clone();
                let binding = match tag {
                    "BUFFER" => Binding::Buffer(BufferBinding {
                        handle,
                        offset: a["offset"].as_str().unwrap().parse().unwrap(),
                        size: a["size"].as_str().unwrap().parse().unwrap(),
                    }),
                    "TEXTURE" => Binding::Texture(TextureBinding {
                        handle,
                        level: a["level"].as_str().unwrap().parse().unwrap(),
                    }),
                    "BINDLESS_ARRAY" => Binding::BindlessArray(BindlessArrayBinding { handle }),
                    "ACCEL" => Binding::Accel(AccelBinding { handle }),
                    _ => panic!("Invalid capture tag: {}", tag),
                };
                Capture { node, binding }
            })
            .collect();
        let args = args[captures.len()..].to_vec();
        let shared = self._curr_ctx().shared.clone();
        let block_size = &self._curr_ctx().j["block_size"];
        let block_size = [
            block_size[0].as_u32().unwrap(),
            block_size[1].as_u32().unwrap(),
            block_size[2].as_u32().unwrap(),
        ];
        KernelModule {
            module,
            captures: CBoxedSlice::new(captures),
            args: CBoxedSlice::new(args),
            shared: CBoxedSlice::new(shared),
            cpu_custom_ops: CBoxedSlice::new(vec![]),
            block_size,
            pools: self.pools.clone(),
        }
    }

    fn _do_convert_callable(&mut self) -> CallableModule {
        let module = self._convert_module(ModuleKind::Function);
        let args: Vec<_> = self._curr_ctx().j["arguments"]
            .members()
            .map(|a| {
                let a = a.as_usize().unwrap();
                self._curr_ctx().arguments.get(&(a as u32)).unwrap().clone()
            })
            .collect();
        let ret_type = self.convert_type(self._curr_ctx().j["return_type"].as_usize().unwrap());
        CallableModule {
            module,
            ret_type,
            args: CBoxedSlice::new(args),
            captures: CBoxedSlice::new(Vec::new()),
            cpu_custom_ops: CBoxedSlice::new(Vec::new()),
            pools: self.pools.clone(),
        }
    }

    fn convert_function(&mut self, i: usize) -> FunctionModule {
        if let Some(f) = self.functions.get(&i) {
            return f.clone();
        }
        let j = &self.j_functions[i];
        let ctx = AST2IRCtx {
            j: &j,
            j_tag: j["tag"].as_str().unwrap(),
            j_variables: &j["variables"],
            builder: None,
            arguments: HashMap::new(),
            variables: HashMap::new(),
            shared: Vec::new(),
            ret_type: if let Some(ret) = j["return_type"].as_usize() {
                self.convert_type(ret)
            } else {
                Type::void()
            },
            has_autodiff: false,
        };
        // push current context
        let tag = ctx.j_tag;
        let old_ctx = self.ctx.replace(ctx);
        let module = match tag {
            "KERNEL" => FunctionModule::Kernel(CArc::new(self._do_convert_kernel())),
            "CALLABLE" => FunctionModule::Callable(CArc::new(self._do_convert_callable())),
            _ => panic!("Unsupported function tag: {}", tag),
        };
        // pop current context
        self.ctx = old_ctx;
        // insert into functions and return the module
        self.functions.insert(i, module.clone());
        module
    }

    fn convert(j: JSON) -> FunctionModule {
        let mut ast2ir = AST2IR {
            j_functions: &j["functions"],
            j_constants: &j["constants"],
            j_types: &j["types"],
            functions: HashMap::new(),
            constants: HashMap::new(),
            types: HashMap::new(),
            ctx: None,
            pools: CArc::new(ModulePools::new()),
        };
        for i in 0usize..j["functions"].len() {
            ast2ir.convert_function(i);
        }
        let entry = j["entry"].as_usize().unwrap();
        ast2ir.functions.get(&entry).unwrap().clone()
    }
}

pub fn convert_ast_to_ir_kernel(data: String) -> CArc<KernelModule> {
    let j: JSON = parse_json(data.as_str()).unwrap();
    match AST2IR::convert(j) {
        FunctionModule::Kernel(k) => k,
        _ => panic!("Expected kernel module."),
    }
}

pub fn convert_ast_to_ir_callable(data: String) -> CArc<CallableModule> {
    let j: JSON = parse_json(data.as_str()).unwrap();
    match AST2IR::convert(j) {
        FunctionModule::Callable(c) => c,
        _ => panic!("Expected callable module."),
    }
}
