use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
};

use indexmap::{IndexMap, IndexSet};

use crate::{
    context::is_type_equal,
    ir::{self, *},
    transform::autodiff::grad_type_of,
    CArc, CBox, CBoxedSlice, Pooled,
};

use super::{sha256, CodeGen};
use crate::ir::Instruction::Invalid;
use std::fmt::Write;

pub(crate) struct TypeGen {
    cache: HashMap<CArc<Type>, String>,
    struct_typedefs: String,
}

impl TypeGen {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            struct_typedefs: String::new(),
        }
    }
    fn to_c_type_(&mut self, t: &CArc<Type>) -> String {
        match t.as_ref() {
            Type::Primitive(t) => match t {
                ir::Primitive::Bool => "bool".to_string(),
                ir::Primitive::Int32 => "int32_t".to_string(),
                ir::Primitive::Uint32 => "uint32_t".to_string(),
                ir::Primitive::Int64 => "int64_t".to_string(),
                ir::Primitive::Uint64 => "uint64_t".to_string(),
                ir::Primitive::Float32 => "float".to_string(),
                ir::Primitive::Float64 => "double".to_string(),
                // crate::ir::Primitive::USize => format!("i{}", std::mem::size_of::<usize>() * 8),
            },
            Type::Void => "()".to_string(),
            Type::UserData => "lc_user_data_t".to_string(),
            Type::Struct(st) => {
                let field_types: Vec<String> = st
                    .fields
                    .as_ref()
                    .iter()
                    .map(|f| self.to_c_type(f))
                    .collect();
                let field_types_str = field_types.join(", ");
                let hash = sha256(&format!("{}_alignas({})", field_types_str, st.alignment));
                let hash = hash.replace("-", "x_");
                let name = format!("s_{}", hash);

                self.cache.insert(t.clone(), name.clone());
                let mut tmp = String::new();
                writeln!(tmp, "struct alignas({0}) {1} {{", st.alignment, name).unwrap();
                for (i, field) in st.fields.as_ref().iter().enumerate() {
                    let field_name = format!("f{}", i);
                    let field_type = self.to_c_type(field);
                    writeln!(tmp, "    {} {};", field_type, field_name).unwrap();
                }
                writeln!(tmp, "    __device__ constexpr static auto one() {{").unwrap();
                writeln!(tmp, "        return {0} {{", name).unwrap();
                for (_, field) in st.fields.as_ref().iter().enumerate() {
                    let field_type = self.to_c_type(field);
                    writeln!(tmp, "        lc_one<{}>(),", field_type).unwrap();
                }
                writeln!(tmp, "        }};").unwrap();
                writeln!(tmp, "    }}").unwrap();
                writeln!(tmp, "    __device__ constexpr static auto zero() {{").unwrap();
                writeln!(tmp, "        return {0} {{", name).unwrap();
                for (_, field) in st.fields.as_ref().iter().enumerate() {
                    let field_type = self.to_c_type(field);
                    writeln!(tmp, "        lc_zero<{}>(),", field_type).unwrap();
                }
                writeln!(tmp, "        }};").unwrap();
                writeln!(tmp, "    }}").unwrap();

                writeln!(tmp, "}};").unwrap();
                writeln!(
                    tmp,
                    "__device__ inline void lc_accumulate_grad({0} *dst, {0} grad) noexcept {{",
                    name
                )
                .unwrap();
                for (i, t) in st.fields.as_ref().iter().enumerate() {
                    if grad_type_of(t.clone()).is_none() {
                        continue;
                    }
                    let field_name = format!("f{}", i);
                    writeln!(
                        tmp,
                        "        lc_accumulate_grad(&dst->{}, grad.{});",
                        field_name, field_name
                    )
                    .unwrap();
                }
                writeln!(tmp, "    }}").unwrap();
                self.struct_typedefs.push_str(&tmp);
                name
            }
            Type::Vector(vt) => {
                let n = vt.length;
                match vt.element {
                    VectorElementType::Scalar(s) => match s {
                        Primitive::Bool => format!("lc_bool{}", n),
                        Primitive::Int32 => format!("lc_int{}", n),
                        Primitive::Uint32 => format!("lc_uint{}", n),
                        Primitive::Int64 => format!("lc_long{}", n),
                        Primitive::Uint64 => format!("lc_ulong{}", n),
                        Primitive::Float32 => format!("lc_float{}", n),
                        Primitive::Float64 => format!("lc_double{}", n),
                    },
                    _ => todo!(),
                }
            }
            Type::Matrix(mt) => {
                let n = mt.dimension;
                match mt.element {
                    VectorElementType::Scalar(s) => match s {
                        Primitive::Float32 => format!("lc_float{0}x{0}", n),
                        // Primitive::Float64 => format!("DMat{}", n),
                        _ => unreachable!(),
                    },
                    _ => todo!(),
                }
            }
            Type::Array(at) => {
                let element_type = self.to_c_type(&at.element);
                format!("lc_array<{}, {}>", element_type, at.length)
            }
        }
    }
    fn to_c_type(&mut self, t: &CArc<Type>) -> String {
        if let Some(t) = self.cache.get(t) {
            return t.clone();
        } else {
            let t_ = self.to_c_type_(t);
            self.cache.insert(t.clone(), t_.clone());
            return t_;
        }
    }
}

pub struct PhiCollector {
    phis: IndexSet<NodeRef>,
    phis_per_block: IndexMap<*const BasicBlock, Vec<NodeRef>>,
}

impl PhiCollector {
    pub fn new() -> Self {
        Self {
            phis: IndexSet::new(),
            phis_per_block: IndexMap::new(),
        }
    }
    pub fn visit_block(&mut self, block: Pooled<BasicBlock>) {
        for phi in block.phis() {
            self.phis.insert(phi);
            if let Instruction::Phi(incomings) = phi.get().instruction.as_ref() {
                for incoming in incomings.as_ref() {
                    let ptr = Pooled::into_raw(incoming.block) as *const _;
                    self.phis_per_block
                        .entry(ptr)
                        .or_insert_with(Vec::new)
                        .push(phi);
                }
            } else {
                unreachable!()
            }
        }
        for node in block.iter() {
            let inst = &node.get().instruction;
            match inst.as_ref() {
                Instruction::If {
                    cond: _,
                    true_branch,
                    false_branch,
                } => {
                    self.visit_block(*true_branch);
                    self.visit_block(*false_branch);
                }
                Instruction::Loop { body, cond: _ } => {
                    self.visit_block(*body);
                }
                Instruction::Switch {
                    value: _,
                    default,
                    cases,
                } => {
                    self.visit_block(*default);
                    for SwitchCase { value: _, block } in cases.as_ref() {
                        self.visit_block(*block);
                    }
                }
                Instruction::GenericLoop {
                    prepare,
                    cond: _,
                    body,
                    update,
                } => {
                    self.visit_block(*prepare);
                    self.visit_block(*body);
                    self.visit_block(*update);
                }
                _ => {}
            }
        }
    }
}

pub struct GenericCppCodeGen {
    type_gen: TypeGen,
    node_to_var: HashMap<NodeRef, String>,
    body: String,
    fwd_defs: String,
    captures: IndexMap<NodeRef, usize>,
    args: IndexMap<NodeRef, usize>,
    cpu_custom_ops: IndexMap<usize, usize>,
    phis: IndexSet<NodeRef>,
    phis_per_block: IndexMap<*const BasicBlock, Vec<NodeRef>>,
    generated_globals: HashSet<String>,
    indent: usize,
    visited: HashSet<NodeRef>,
    signature: Vec<String>,
    cpu_kernel_parameters: Vec<String>,
    cpu_kernel_unpack_parameters: Vec<String>,
}

impl GenericCppCodeGen {
    pub fn new() -> Self {
        Self {
            type_gen: TypeGen::new(),
            node_to_var: HashMap::new(),
            body: String::new(),
            fwd_defs: String::new(),
            captures: IndexMap::new(),
            args: IndexMap::new(),
            phis: IndexSet::new(),
            phis_per_block: IndexMap::new(),
            cpu_custom_ops: IndexMap::new(),
            generated_globals: HashSet::new(),
            indent: 1,
            visited: HashSet::new(),
            signature: Vec::new(),
            cpu_kernel_parameters: Vec::new(),
            cpu_kernel_unpack_parameters: Vec::new(),
        }
    }
    fn write_ident(&mut self) {
        for _ in 0..self.indent {
            write!(&mut self.body, "  ").unwrap();
        }
    }
    fn gen_node(&mut self, node: NodeRef) -> String {
        if let Some(var) = self.node_to_var.get(&node) {
            return var.clone();
        } else {
            let index = self.node_to_var.len();
            let var = match node.get().instruction.as_ref() {
                Instruction::Buffer => format!("b{}", index),
                Instruction::Bindless => format!("bl{}", index),
                Instruction::Texture2D => format!("t2d{}", index),
                Instruction::Texture3D => format!("t3d{}", index),
                Instruction::Accel => format!("a{}", index),
                Instruction::Shared => format!("s{}", index),
                Instruction::Uniform => format!("u{}", index),
                Instruction::Local { .. } => format!("v{}", index),
                Instruction::Argument { .. } => format!("arg{}", index),
                Instruction::UserData(_) => format!("_lc_user_data"),
                Instruction::Const(_) => format!("c{}", index),
                Instruction::Call(_, _) => {
                    if is_type_equal(&node.type_(), &Type::void()) {
                        "".to_string()
                    } else {
                        format!("f{}", index)
                    }
                }
                Instruction::Phi(_) => format!("phi{}", index),
                _ => unreachable!(),
            };
            self.node_to_var.insert(node, var.clone());
            return var;
        }
    }
    fn gep_field_name(node: NodeRef, i: i32) -> String {
        let node_ty = node.type_();
        match node_ty.as_ref() {
            Type::Struct(_) => {
                format!("f{}", i)
            }
            Type::Vector(_) => match i {
                0 => "x".to_string(),
                1 => "y".to_string(),
                2 => "z".to_string(),
                3 => "w".to_string(),
                _ => unreachable!(),
            },
            Type::Matrix(_) => {
                format!("cols[{}]", i)
            }
            Type::Void | Type::Primitive(_) => unreachable!(),
            _ => todo!(),
        }
    }
    fn gen_binop(
        &mut self,
        var: &String,
        node_ty_s: &String,
        f: &Func,
        args_v: &Vec<String>,
    ) -> bool {
        let binop = match f {
            Func::Add => Some("+"),
            Func::Sub => Some("-"),
            Func::Mul => Some("*"),
            Func::Div => Some("/"),
            Func::Rem => Some("%"),
            Func::BitAnd => Some("&"),
            Func::BitOr => Some("|"),
            Func::BitXor => Some("^"),
            Func::Shl => Some("<<"),
            Func::Shr => Some(">>"),
            Func::Eq => Some("=="),
            Func::Ne => Some("!="),
            Func::Lt => Some("<"),
            Func::Le => Some("<="),
            Func::Gt => Some(">"),
            Func::Ge => Some(">="),
            _ => None,
        };
        if let Some(binop) = binop {
            writeln!(
                &mut self.body,
                "const {} {} = {} {} {};",
                node_ty_s, var, args_v[0], binop, args_v[1]
            )
            .unwrap();
            true
        } else {
            false
        }
    }
    fn gen_call_op(
        &mut self,
        var: &String,
        node_ty_s: &String,
        f: &Func,
        args_v: &Vec<String>,
    ) -> bool {
        let func = match f {
            Func::Abs => Some("lc_abs"),
            Func::Acos => Some("lc_acos"),
            Func::Acosh => Some("lc_acosh"),
            Func::Asin => Some("lc_asin"),
            Func::Asinh => Some("lc_asinh"),
            Func::Atan => Some("lc_atan"),
            Func::Atan2 => Some("lc_atan2"),
            Func::Atanh => Some("lc_atanh"),
            Func::Cos => Some("lc_cos"),
            Func::Cosh => Some("lc_cosh"),
            Func::Sin => Some("lc_sin"),
            Func::Sinh => Some("lc_sinh"),
            Func::Tan => Some("lc_tan"),
            Func::Tanh => Some("lc_tanh"),

            Func::Exp => Some("lc_exp"),
            Func::Exp2 => Some("lc_exp2"),
            Func::Exp10 => Some("lc_exp10"),
            Func::Log => Some("lc_log"),
            Func::Log2 => Some("lc_log2"),
            Func::Log10 => Some("lc_log10"),
            Func::Powi => Some("lc_lc_powi"),
            Func::Powf => Some("lc_powf"),

            Func::Sqrt => Some("lc_sqrt"),
            Func::Rsqrt => Some("lc_rsqrt"),

            Func::Ceil => Some("lc_ceil"),
            Func::Floor => Some("lc_floor"),
            Func::Fract => Some("lc_fract"),
            Func::Trunc => Some("lc_trunc"),
            Func::Round => Some("lc_round"),

            Func::Fma => Some("lc_fma"),
            Func::Copysign => Some("lc_copysign"),
            Func::Cross => Some("lc_cross"),
            Func::Dot => Some("lc_dot"),
            Func::OuterProduct => Some("lc_outer_product"),
            Func::Length => Some("lc_length"),
            Func::LengthSquared => Some("lc_length_squared"),
            Func::Normalize => Some("lc_normalize"),
            Func::Faceforward => Some("lc_faceforward"),
            Func::Determinant => Some("lc_determinant"),
            Func::Transpose => Some("lc_transpose"),
            Func::Inverse => Some("lc_inverse"),
            Func::ReduceSum => Some("lc_reduce_sum"),
            Func::ReduceProd => Some("lc_reduce_prod"),
            Func::ReduceMin => Some("lc_reduce_min"),
            Func::ReduceMax => Some("lc_reduce_max"),

            Func::IsInf => Some("lc_isinf"),
            Func::IsNan => Some("lc_isnan"),
            Func::Any => Some("lc_any"),
            Func::All => Some("lc_all"),

            Func::PopCount => Some("lc_popcount"),
            Func::Clz => Some("lc_clz"),
            Func::Ctz => Some("lc_ctz"),
            Func::Reverse => Some("lc_reverse_bits"),
            Func::Min => Some("lc_min"),
            Func::Max => Some("lc_max"),
            Func::Clamp => Some("lc_clamp"),
            _ => None,
        };
        if let Some(func) = func {
            writeln!(
                &mut self.body,
                "const {} {} = {}({});",
                node_ty_s,
                var,
                func,
                args_v.join(", ")
            )
            .unwrap();
            true
        } else {
            false
        }
    }
    fn gen_buffer_op(
        &mut self,
        var: &String,
        node_ty_s: &String,
        f: &Func,
        args: &[NodeRef],
        args_v: &Vec<String>,
    ) -> bool {
        match f {
            Func::BufferRead => {
                self.gen_instr(args[0]);
                let buffer_ty = self.type_gen.to_c_type(args[0].type_());
                writeln!(
                    &mut self.body,
                    "const auto {1} = lc_buffer_read<{0}>({2}, {3});",
                    buffer_ty, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::BufferWrite => {
                self.gen_instr(args[0]);
                let buffer_ty = self.type_gen.to_c_type(args[0].type_());
                writeln!(
                    &mut self.body,
                    "lc_buffer_write<{}>({}, {}, {});",
                    buffer_ty, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::BufferSize => {
                self.gen_instr(args[0]);
                let buffer_ty = self.type_gen.to_c_type(args[0].type_());
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_buffer_size<{}>({});",
                    node_ty_s, var, buffer_ty, args_v[0]
                )
                .unwrap();
                true
            }
            Func::BindlessBufferRead => {
                self.gen_instr(args[0]);
                writeln!(
                    &mut self.body,
                    "const auto {1} = lc_bindless_buffer_read<{0}>({2}, {3}, {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::BindlessBufferSize(t) => {
                self.gen_instr(args[0]);
                let buffer_ty = self.type_gen.to_c_type(t);
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_bindless_buffer_size<{}>({}, {});",
                    node_ty_s, var, buffer_ty, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::BindlessBufferType => {
                self.gen_instr(args[0]);
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_bindless_buffer_type({}, {});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            _ => false,
        }
    }
    fn gen_misc(
        &mut self,
        var: &String,
        node: NodeRef,
        node_ty_s: &String,
        node_ty: &CArc<Type>,
        f: &Func,
        args: &[NodeRef],
        args_v: &Vec<String>,
    ) -> bool {
        match f {
            Func::RequiresGradient => true,
            Func::Assume => {
                writeln!(&mut self.body, "lc_assume({});", args_v.join(", ")).unwrap();
                true
            }
            Func::Assert => {
                writeln!(&mut self.body, "lc_assert({});", args_v.join(", ")).unwrap();
                true
            }
            Func::Unreachable => {
                if !is_type_equal(node_ty, &Type::void()) {
                    writeln!(&mut self.body, "{} {};", node_ty_s, var).unwrap();
                }
                writeln!(&mut self.body, "lc_unreachable({});", args_v.join(", ")).unwrap();
                true
            }
            Func::ExtractElement => {
                let i = args.as_ref()[1].get_i32();
                let field_name = Self::gep_field_name(args.as_ref()[0], i);
                writeln!(
                    self.body,
                    "const {} {} = {}.{};",
                    node_ty_s, var, args_v[0], field_name
                )
                .unwrap();
                true
            }
            Func::InsertElement => {
                let i = args.as_ref()[2].get_i32();
                let field_name = Self::gep_field_name(args.as_ref()[0], i);
                writeln!(
                    self.body,
                    "{0} _{1} = {2}; _{1}.{3} = {4}; const auto {1} = _{1};",
                    node_ty_s, var, args_v[0], field_name, args_v[1]
                )
                .unwrap();
                true
            }
            Func::GetElementPtr => {
                if args[0].type_().is_array() {
                    let const_ = if !args[0].is_local() { "const " } else { "" };
                    writeln!(
                        self.body,
                        "{}{}& {} = {}[{}];",
                        const_, node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap();
                } else {
                    let i = args.as_ref()[1].get_i32();
                    let field_name = Self::gep_field_name(args.as_ref()[0], i);
                    writeln!(
                        self.body,
                        "{} & {} = {}.{};",
                        node_ty_s, var, args_v[0], field_name
                    )
                    .unwrap();
                }
                true
            }
            Func::Struct => {
                writeln!(
                    &mut self.body,
                    "const {} {} = {{ {} }};",
                    node_ty_s,
                    var,
                    args_v.join(", ")
                )
                .unwrap();
                true
            }
            Func::DispatchId => {
                writeln!(self.body, "const {} {} = lc_dispatch_id();", node_ty_s, var).unwrap();
                true
            }
            Func::DispatchSize => {
                writeln!(
                    self.body,
                    "const {} {} = lc_dispatch_size();",
                    node_ty_s, var
                )
                .unwrap();
                true
            }
            Func::BlockId => {
                writeln!(self.body, "const {} {} = lc_block_id();", node_ty_s, var).unwrap();
                true
            }
            Func::ThreadId => {
                writeln!(self.body, "const {} {} = lc_thread_id();", node_ty_s, var).unwrap();
                true
            }
            Func::ZeroInitializer => {
                writeln!(
                    self.body,
                    "const {} {} = lc_zero<{}>();",
                    node_ty_s, var, node_ty_s
                )
                .unwrap();
                true
            }
            Func::Vec | Func::Vec2 | Func::Vec3 | Func::Vec4 => {
                writeln!(
                    self.body,
                    "const {} {} = {}({});",
                    node_ty_s,
                    var,
                    node_ty_s,
                    args_v.join(", ")
                )
                .unwrap();
                true
            }
            Func::Mat2 | Func::Mat3 | Func::Mat4 => {
                writeln!(
                    self.body,
                    "const {} {} = {}({});",
                    node_ty_s,
                    var,
                    node_ty_s,
                    args_v.join(", ")
                )
                .unwrap();
                true
            }
            Func::Mat => {
                writeln!(
                    self.body,
                    "const {} {} = {}::full({});",
                    node_ty_s,
                    var,
                    node_ty_s,
                    args_v.join(", ")
                )
                .unwrap();
                true
            }
            Func::MatCompMul => {
                writeln!(
                    self.body,
                    "const {} {} = {}.comp_mul({});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::Select => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_select({4},{3},{2});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::Load => {
                writeln!(self.body, "const {} {} = {};", node_ty_s, var, args_v[0]).unwrap();
                true
            }
            Func::GradientMarker => {
                let ty = self.type_gen.to_c_type(args.as_ref()[1].type_());
                writeln!(
                    self.body,
                    "const {} {}_grad = {};",
                    ty, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::Detach => {
                writeln!(self.body, "const {} {} = {};", node_ty_s, var, args_v[0]).unwrap();
                true
            }
            Func::Gradient => {
                writeln!(
                    self.body,
                    "const {0} {1} = {2}_grad;",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            Func::AccGrad => {
                writeln!(
                    self.body,
                    "lc_accumulate_grad(&{0}, {1});",
                    args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::BitNot => {
                if node_ty.is_bool() {
                    writeln!(self.body, "const {} {} = !{};", node_ty_s, var, args_v[0]).unwrap();
                } else {
                    writeln!(self.body, "const {} {} = ~{};", node_ty_s, var, args_v[0]).unwrap();
                }
                true
            }
            Func::Neg => {
                writeln!(self.body, "const {} {} = -{};", node_ty_s, var, args_v[0]).unwrap();
                true
            }
            Func::Not => {
                writeln!(self.body, "const {} {} = !{};", node_ty_s, var, args_v[0]).unwrap();
                true
            }
            Func::Bitcast => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bitcast<{1}>({2});",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            Func::Cast => {
                if node_ty.is_bool() && node_ty.is_primitive() {
                    writeln!(
                        self.body,
                        "const {} {} = {} != 0;",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap();
                } else if (node_ty.is_float() || node_ty.is_int()) && node_ty.is_primitive() {
                    writeln!(
                        self.body,
                        "const {} {} = static_cast<{}>({});",
                        node_ty_s, var, node_ty_s, args_v[0]
                    )
                    .unwrap();
                } else if node_ty.is_vector() {
                    let vt_s = node_ty_s[3..].to_string();
                    writeln!(
                        self.body,
                        "const {} {} = lc_make_{}({});",
                        node_ty_s, var, vt_s, args_v[0]
                    )
                    .unwrap();
                } else {
                    unreachable!()
                }
                true
            }
            Func::Permute => {
                let indices: Vec<_> = args[1..].iter().map(|a| a.get_i32()).collect();
                let indices_s: Vec<_> = indices
                    .iter()
                    .map(|i| format!("{}.{}", args_v[0], Self::gep_field_name(node, *i)))
                    .collect();
                writeln!(
                    self.body,
                    "const {} {} = {}({});",
                    node_ty_s,
                    var,
                    node_ty_s,
                    indices_s.join(", ")
                )
                .unwrap();
                true
            }
            Func::AtomicExchange => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_atomic_exchange(lc_buffer_ref<{0}>({2}, {3}), {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::AtomicCompareExchange => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_atomic_compare_exchange(lc_buffer_ref<{0}>({2}, {3}), {4}, {5});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2], args_v[3]
                ).unwrap();
                true
            }
            Func::AtomicFetchAdd => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_atomic_fetch_add(lc_buffer_ref<{0}>({2}, {3}), {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::AtomicFetchSub => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_atomic_fetch_sub(lc_buffer_ref<{0}>({2}, {3}), {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::AtomicFetchMin => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_atomic_fetch_min(lc_buffer_ref<{0}>({2}, {3}), {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::AtomicFetchMax => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_atomic_fetch_max(lc_buffer_ref<{0}>({2}, {3}), {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::AtomicFetchAnd => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_atomic_fetch_and(lc_buffer_ref<{0}>({2}, {3}), {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::AtomicFetchOr => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_atomic_fetch_or(lc_buffer_ref<{0}>({2}, {3}), {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::AtomicFetchXor => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_atomic_fetch_xor(lc_buffer_ref<{0}>({2}, {3}), {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::CpuCustomOp(op) => {
                let i = *self
                    .cpu_custom_ops
                    .get(&(CArc::as_ptr(op) as usize))
                    .unwrap();
                writeln!(
                    self.body,
                    "const {0} {1} = lc_cpu_custom_op(k_args, {2}, {3});",
                    node_ty_s, var, i, args_v[0],
                )
                .unwrap();
                true
            }
            Func::RayTracingTraceAny => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_trace_any({2}, lc_bit_cast<Ray>({3}));",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::RayTracingTraceClosest => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bit_cast<{0}>(lc_trace_closest({2}, lc_bit_cast<Ray>({3})));",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::RayTracingInstanceTransform => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_accel_instance_transform({2}, {3});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::RayTracingSetInstanceTransform => {
                writeln!(
                    self.body,
                    "lc_set_instance_transform({0}, {1});",
                    args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::RayTracingSetInstanceVisibility => {
                writeln!(
                    self.body,
                    "lc_set_instance_visibility({0}, {1});",
                    args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            _ => false,
        }
    }
    fn gen_const(&mut self, var: &String, node_ty_s: &String, cst: &Const) {
        match cst {
            Const::Zero(_) => {
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_zero<{}>();",
                    node_ty_s, var, node_ty_s
                )
                .unwrap();
            }
            Const::One(_) => {
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_one<{}>();",
                    node_ty_s, var, node_ty_s
                )
                .unwrap();
            }
            Const::Bool(v) => {
                writeln!(&mut self.body, "const bool {} = {};", var, *v).unwrap();
            }
            Const::Int32(v) => {
                writeln!(&mut self.body, "const int32_t {} = {};", var, *v).unwrap();
            }
            Const::Uint32(v) => {
                writeln!(&mut self.body, "const uint32_t {} = {};", var, *v).unwrap();
            }
            Const::Int64(v) => {
                writeln!(&mut self.body, "const int64_t {} = {};", var, *v).unwrap();
            }
            Const::Uint64(v) => {
                writeln!(&mut self.body, "const uint64_t {} = {};", var, *v).unwrap();
            }
            Const::Float32(v) => {
                writeln!(
                    &mut self.body,
                    "const float {} = lc_bit_cast<float>(uint32_t({})); // {}",
                    var,
                    v.to_bits(),
                    v
                )
                .unwrap();
            }
            Const::Float64(v) => {
                writeln!(
                    &mut self.body,
                    "const double {} = lc_bit_cast<double>(uint64_t({})); // {}",
                    var,
                    v.to_bits(),
                    v
                )
                .unwrap();
            }
            Const::Generic(bytes, t) => {
                let gen_def = |dst: &mut String, qualifier| {
                    writeln!(
                        dst,
                        "{0} uint8_t {2}_bytes[{1}] = {{ {3} }};",
                        qualifier,
                        t.size(),
                        var,
                        bytes
                            .as_ref()
                            .iter()
                            .map(|b| format!("{}", b))
                            .collect::<Vec<String>>()
                            .join(", ")
                    )
                    .unwrap();
                };
                match t.as_ref() {
                    Type::Array(_) => {
                        if !self.generated_globals.contains(var) {
                            gen_def(&mut self.fwd_defs, "__constant__ constexpr const");
                            self.generated_globals.insert(var.clone());
                        }
                    }
                    _ => gen_def(&mut self.body, "constexpr const"),
                }
                self.write_ident();
                writeln!(
                    &mut self.body,
                    "const {0} {1} = *reinterpret_cast<const {0}*>({1}_bytes);",
                    node_ty_s, var
                )
                .unwrap();
            }
        }
    }
    fn gen_instr(&mut self, node: NodeRef) {
        if self.visited.contains(&node) {
            return;
        }
        self.visited.insert(node);
        let inst = &node.get().instruction;
        let node_ty = node.type_();
        let node_ty_s = self.type_gen.to_c_type(node_ty);
        match inst.as_ref() {
            Instruction::Buffer => {}
            Instruction::Bindless => {}
            Instruction::Texture2D => {}
            Instruction::Texture3D => {}
            Instruction::Accel => {}
            Instruction::Shared => todo!(),
            Instruction::Uniform => todo!(),
            Instruction::Local { init } => {
                self.write_ident();
                let var = self.gen_node(node);
                let init_v = self.gen_node(*init);
                writeln!(&mut self.body, "{0} {1} = {2};", node_ty_s, var, init_v).unwrap();
            }
            Instruction::Argument { by_value } => todo!(),
            Instruction::UserData(_) => {}
            Instruction::Invalid => todo!(),
            Instruction::Const(cst) => {
                self.write_ident();
                let var = self.gen_node(node);
                self.gen_const(&var, &node_ty_s, cst);
            }
            Instruction::Update { var, value } => {
                self.write_ident();
                let value_v = self.gen_node(*value);
                let var_v = self.gen_node(*var);
                writeln!(&mut self.body, "{} = {};", var_v, value_v).unwrap();
            }
            Instruction::Call(f, args) => {
                self.write_ident();
                let args_v = args
                    .as_ref()
                    .iter()
                    .map(|arg| self.gen_node(*arg))
                    .collect::<Vec<_>>();
                let var = self.gen_node(node);
                let mut done = self.gen_binop(&var, &node_ty_s, f, &args_v);
                if !done {
                    done = self.gen_call_op(&var, &node_ty_s, f, &args_v);
                }
                if !done {
                    done = self.gen_buffer_op(&var, &node_ty_s, f, args.as_ref(), &args_v);
                }
                if !done {
                    done = self.gen_misc(
                        &var,
                        node,
                        &node_ty_s,
                        node.type_(),
                        f,
                        args.as_ref(),
                        &args_v,
                    );
                }
                assert!(done, "{:?} is not implemented", f);
            }
            Instruction::Phi(_) => {
                self.write_ident();
                let var = self.gen_node(node);
                writeln!(&mut self.fwd_defs, "{0} {1} = {0}{{}};", node_ty_s, var).unwrap();
            }
            Instruction::Return(v) => {
                self.write_ident();
                if v.valid() {
                    let v_v = self.gen_node(*v);
                    writeln!(&mut self.body, "return {};", v_v).unwrap();
                } else {
                    writeln!(&mut self.body, "return;").unwrap();
                }
            }
            Instruction::Loop { body, cond } => {
                self.write_ident();
                writeln!(&mut self.body, "do {{").unwrap();
                self.gen_block(*body);
                let cond_v = self.gen_node(*cond);
                self.write_ident();
                writeln!(&mut self.body, "}} while({});", cond_v).unwrap();
            }
            Instruction::GenericLoop {
                prepare,
                cond,
                body,
                update,
            } => {
                /* template:
                while(true) {
                    bool loop_break = false;
                    prepare();
                    if (!cond()) break;
                    do {
                        // break => { loop_break = true; break; }
                        // continue => { break; }
                    } while(false);
                    if (loop_break) break;
                    update();
                }
                */
                self.write_ident();
                writeln!(&mut self.body, "while(true) {{").unwrap();
                self.write_ident();
                writeln!(&mut self.body, "bool loop_break = false;").unwrap();
                self.write_ident();
                writeln!(&mut self.body, "{{").unwrap();
                self.indent += 1;
                self.gen_block_(*prepare);
                let cond_v = self.gen_node(*cond);
                self.write_ident();
                writeln!(&mut self.body, "if (!{}) break;", cond_v).unwrap();
                self.write_ident();
                self.indent -= 1;
                writeln!(&mut self.body, "}}").unwrap();
                self.write_ident();
                writeln!(&mut self.body, "do").unwrap();
                self.write_ident();
                self.gen_block(*body);
                self.write_ident();
                writeln!(&mut self.body, "while(false);").unwrap();
                self.write_ident();
                writeln!(&mut self.body, "if (loop_break) break;").unwrap();
                self.gen_block(*update);
                self.write_ident();
                writeln!(&mut self.body, "}}").unwrap();
            }
            Instruction::Break => {
                self.write_ident();
                writeln!(&mut self.body, "loop_break = true;").unwrap();
                writeln!(&mut self.body, "break;").unwrap();
            }
            Instruction::Continue => {
                self.write_ident();
                writeln!(&mut self.body, "break;").unwrap();
            }
            Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                self.write_ident();
                let cond_v = self.gen_node(*cond);
                writeln!(&mut self.body, "if ({})", cond_v).unwrap();
                self.gen_block(*true_branch);
                self.write_ident();
                writeln!(&mut self.body, "else").unwrap();
                self.gen_block(*false_branch);
            }
            Instruction::Switch {
                value,
                default,
                cases,
            } => {
                self.write_ident();
                let value_v = self.gen_node(*value);
                writeln!(&mut self.body, "switch ({}) {{", value_v).unwrap();
                for SwitchCase { value, block } in cases.as_ref() {
                    self.write_ident();
                    writeln!(&mut self.body, "case {}:", *value).unwrap();
                    self.gen_block(*block);
                    self.write_ident();
                    writeln!(&mut self.body, "break;").unwrap();
                }
                self.write_ident();
                writeln!(&mut self.body, "default:").unwrap();
                self.gen_block(*default);
                self.write_ident();
                writeln!(&mut self.body, "break;").unwrap();
                self.write_ident();
                writeln!(&mut self.body, "}}").unwrap();
            }
            Instruction::AdScope {
                forward,
                backward,
                epilogue,
            } => {
                writeln!(&mut self.body, "/* AdScope Forward */").unwrap();
                self.gen_block(*forward);
                writeln!(&mut self.body, "/* AdScope Forward End */").unwrap();
                writeln!(&mut self.body, "/* AdScope Backward */").unwrap();
                self.gen_block(*backward);
                writeln!(&mut self.body, "/* AdScope Backward End */").unwrap();
                writeln!(&mut self.body, "/* AdScope Epilogue */").unwrap();
                self.gen_block(*epilogue);
                writeln!(&mut self.body, "/* AdScope Epilogue End */").unwrap();
            }
            Instruction::AdDetach(bb) => {
                self.write_ident();
                writeln!(&mut self.body, "/* AdDetach */").unwrap();
                self.gen_block(*bb);
                self.write_ident();
                writeln!(&mut self.body, "/* AdDetach End */").unwrap();
            }
            Instruction::Comment(comment) => {
                self.write_ident();
                let comment = CString::new(comment.as_ref()).unwrap();
                writeln!(&mut self.body, "/* {} */", comment.to_string_lossy()).unwrap();
            }
            Instruction::Debug(_) => todo!(),
        }
    }
    fn gen_block_(&mut self, block: Pooled<ir::BasicBlock>) {
        for n in block.iter() {
            self.gen_instr(n);
        }
        let phis = self
            .phis_per_block
            .get(&(Pooled::into_raw(block) as *const _))
            .cloned()
            .unwrap_or(vec![]);
        for phi in &phis {
            let phi_v = self.gen_node(*phi);
            if let Instruction::Phi(incomings) = phi.get().instruction.as_ref() {
                let value = incomings
                    .as_ref()
                    .iter()
                    .find(|incoming| {
                        Pooled::into_raw(incoming.block) as *const _
                            == Pooled::into_raw(block) as *const _
                    })
                    .unwrap()
                    .value;
                let value = self.gen_node(value);
                self.write_ident();
                writeln!(&mut self.body, "{} = {};", phi_v, value).unwrap();
            } else {
                unreachable!()
            }
        }
    }
    fn gen_block(&mut self, block: Pooled<ir::BasicBlock>) {
        self.write_ident();
        writeln!(&mut self.body, "{{").unwrap();
        self.indent += 1;
        self.gen_block_(block);
        self.indent -= 1;
        self.write_ident();
        writeln!(&mut self.body, "}}").unwrap();
    }
    fn gen_arg(&mut self, node: NodeRef, index: usize, is_capture: bool) {
        let arg_name = self.gen_node(node);
        let arg_array = if is_capture {
            "k_args->captured"
        } else {
            "k_args->args"
        };
        match node.get().instruction.as_ref() {
            Instruction::Accel => {
                self.signature.push(format!("const Accel& {}", arg_name));
                self.cpu_kernel_unpack_parameters.push(format!(
                    "const Accel& {} = {}[{}].accel.body._0;",
                    arg_name, arg_array, index
                ));
                self.cpu_kernel_parameters.push(arg_name);
            }
            Instruction::Bindless => {
                self.signature
                    .push(format!("const BindlessArray& {}", arg_name));
                self.cpu_kernel_unpack_parameters.push(format!(
                    "const BindlessArray& {} = {}[{}].bindless_array._0;",
                    arg_name, arg_array, index
                ));
                self.cpu_kernel_parameters.push(arg_name);
            }
            Instruction::Buffer => {
                self.signature
                    .push(format!("const BufferView& {}", arg_name));
                self.cpu_kernel_unpack_parameters.push(format!(
                    "const BufferView& {} = {}[{}].buffer._0;",
                    arg_name, arg_array, index
                ));
                self.cpu_kernel_parameters.push(arg_name);
            }
            Instruction::Texture2D => {
                self.signature
                    .push(format!("const Texture2DView& {}", arg_name));
                self.cpu_kernel_unpack_parameters.push(format!(
                    "const Texture2DView& {} = {}[{}].texture2d._0;",
                    arg_name, arg_array, index
                ));
                self.cpu_kernel_parameters.push(arg_name);
            }
            Instruction::Texture3D => {
                self.signature
                    .push(format!("const Texture3DView& {}", arg_name));
                self.cpu_kernel_unpack_parameters.push(format!(
                    "const Texture3DView& {} = {}[{}].texture3d._0;",
                    arg_name, arg_array, index
                ));
                self.cpu_kernel_parameters.push(arg_name);
            }
            _ => unreachable!(),
        }
        if is_capture {
            assert!(!self.captures.contains_key(&node));
            self.captures.insert(node, index);
        } else {
            assert!(!self.args.contains_key(&node));
            self.args.insert(node, index);
        }
    }
    fn gen_module(&mut self, module: &ir::KernelModule) {
        let mut phi_collector = PhiCollector::new();
        phi_collector.visit_block(module.module.entry);
        let PhiCollector {
            phis_per_block,
            phis,
        } = phi_collector;
        self.phis_per_block = phis_per_block;
        self.phis = phis;
        for (i, capture) in module.captures.as_ref().iter().enumerate() {
            self.gen_arg(capture.node, i, true);
        }
        for (i, arg) in module.args.as_ref().iter().enumerate() {
            self.gen_arg(*arg, i, false);
        }
        for (i, op) in module.cpu_custom_ops.as_ref().iter().enumerate() {
            self.cpu_custom_ops.insert(CArc::as_ptr(op) as usize, i);
        }
        self.gen_block(module.module.entry);
    }
}

pub struct CpuCodeGen;

impl CodeGen for CpuCodeGen {
    fn run(module: &ir::KernelModule) -> String {
        let mut codegen = GenericCppCodeGen::new();
        codegen.gen_module(module);
        let kernel_fn_decl = r#"lc_kernel void kernel_fn(const KernelFnArgs* k_args) {"#;
        let kernel_fn = format!(
            "{}\n{}\nkernel_(k_args, {});\n}}\n",
            kernel_fn_decl,
            codegen.cpu_kernel_unpack_parameters.join("\n"),
            codegen.cpu_kernel_parameters.join(", "),
        );
        let kernel_wrapper_decl = format!(
            "void kernel_(const KernelFnArgs* k_args, {}) {{",
            codegen.signature.join(", ")
        );
        let includes = r#"#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
using namespace std;"#;
        format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            includes,
            CPU_KERNEL_DEFS,
            CPU_PRELUDE,
            DEVICE_MATH_SRC,
            CPU_RESOURCE,
            CPU_TEXTURE,
            codegen.type_gen.struct_typedefs,
            kernel_wrapper_decl,
            codegen.fwd_defs,
            codegen.body,
            "}",
            kernel_fn
        )
    }
}

pub const CPU_PRELUDE: &str = include_str!("cpu_prelude.h");
pub const CPU_RESOURCE: &str = include_str!("cpu_resource.h");
pub const DEVICE_MATH_SRC: &str = include_str!("device_math.h");
pub const CPU_KERNEL_DEFS: &str =
    include_str!("../../../luisa_compute_cpu_kernel_defs/cpu_kernel_defs.h");
pub const CPU_TEXTURE: &str = include_str!("cpu_texture.h");
#[no_mangle]
pub extern "C" fn luisa_compute_codegen_cpp(module: KernelModule) -> CBoxedSlice<u8> {
    let src = CpuCodeGen::run(&module);
    let c_string = CString::new(src).unwrap();
    CBoxedSlice::new(c_string.as_bytes().to_vec())
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_codegen_cuda(module: KernelModule) -> CBoxedSlice<u8> {
    todo!();
    // let src = "";
    // let c_string = CString::new(src).unwrap();
    // CBoxedSlice::new(c_string.as_bytes().to_vec())
}
