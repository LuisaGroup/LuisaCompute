use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    ffi::CString,
};

use indexmap::{IndexMap, IndexSet};

use luisa_compute_ir::{
    context::is_type_equal,
    ir::{self, *},
    transform::autodiff::grad_type_of,
    CArc, Pooled,
};

use super::sha256_short;

use super::decode_const_data;
use std::fmt::Write;

pub(crate) struct TypeGenInner {
    cache: HashMap<CArc<Type>, String>,
    struct_typedefs: String,
}

impl TypeGenInner {
    pub(crate) fn new() -> Self {
        Self {
            cache: HashMap::new(),
            struct_typedefs: String::new(),
        }
    }
    fn to_c_type_(&mut self, t: &CArc<Type>) -> String {
        match t.as_ref() {
            Type::Primitive(t) => match t {
                ir::Primitive::Bool => "bool".to_string(),
                ir::Primitive::Int8 => "int8_t".to_string(),
                ir::Primitive::Uint8 => "uint8_t".to_string(),
                ir::Primitive::Int16 => "int16_t".to_string(),
                ir::Primitive::Uint16 => "uint16_t".to_string(),
                ir::Primitive::Int32 => "int32_t".to_string(),
                ir::Primitive::Uint32 => "uint32_t".to_string(),
                ir::Primitive::Int64 => "int64_t".to_string(),
                ir::Primitive::Uint64 => "uint64_t".to_string(),
                ir::Primitive::Float16 => "half".to_string(),
                ir::Primitive::Float32 => "float".to_string(),
                ir::Primitive::Float64 => "double".to_string(),
                // crate::ir::Primitive::USize => format!("i{}", std::mem::size_of::<usize>() * 8),
            },
            Type::Void => "void".to_string(),
            Type::UserData => "lc_user_data_t".to_string(),
            Type::Struct(st) => {
                let field_types: Vec<String> = st
                    .fields
                    .as_ref()
                    .iter()
                    .map(|f| self.to_c_type(f))
                    .collect();
                let field_types_str = field_types.join(", ");
                let hash = sha256_short(&format!("{}_alignas({})", field_types_str, st.alignment));
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
                        Primitive::Int8 => format!("lc_char{}", n),
                        Primitive::Uint8 => format!("lc_uchar{}", n),
                        Primitive::Int16 => format!("lc_short{}", n),
                        Primitive::Uint16 => format!("lc_ushort{}", n),
                        Primitive::Int32 => format!("lc_int{}", n),
                        Primitive::Uint32 => format!("lc_uint{}", n),
                        Primitive::Int64 => format!("lc_long{}", n),
                        Primitive::Uint64 => format!("lc_ulong{}", n),
                        Primitive::Float16 => format!("lc_half{}", n),
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
            Type::Opaque(name) => {
                let name = name.to_string();
                let builtin = ["LC_RayQueryAny", "LC_RayQueryAll"];
                if !builtin.contains(&name.as_str()) {
                    self.struct_typedefs
                        .push_str(&format!("struct {};", name.to_string()));
                }
                name
            }
        }
    }
    pub(crate) fn to_c_type(&mut self, t: &CArc<Type>) -> String {
        if let Some(t) = self.cache.get(t) {
            return t.clone();
        } else {
            let t_ = self.to_c_type_(t);
            self.cache.insert(t.clone(), t_.clone());
            return t_;
        }
    }
}

struct TypeGen {
    inner: RefCell<TypeGenInner>,
}

impl TypeGen {
    fn new() -> Self {
        Self {
            inner: RefCell::new(TypeGenInner::new()),
        }
    }
    fn gen_c_type(&self, t: &CArc<Type>) -> String {
        self.inner.borrow_mut().to_c_type(t)
    }
    fn generated(&self) -> String {
        self.inner.borrow().struct_typedefs.clone()
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
                Instruction::RayQuery {
                    ray_query: _,
                    on_triangle_hit,
                    on_procedural_hit,
                } => {
                    self.visit_block(*on_triangle_hit);
                    self.visit_block(*on_procedural_hit);
                }
                Instruction::AdDetach(detach) => {
                    self.visit_block(*detach);
                }
                Instruction::AdScope { body, .. } => {
                    self.visit_block(*body);
                }
                _ => {}
            }
        }
    }
}

struct GlobalEmitter {
    message: Vec<String>,
    global_vars: HashMap<NodeRef, String>,
    generated_callables: HashMap<u64, String>,
    generated_callable_sources: HashMap<String, String>,
    callable_def: String,
    captures: IndexMap<NodeRef, usize>,
    args: IndexMap<NodeRef, usize>,
    cpu_custom_ops: IndexMap<usize, usize>,
}

struct FunctionEmitter<'a> {
    type_gen: &'a TypeGen,
    node_to_var: HashMap<NodeRef, String>,
    body: String,
    fwd_defs: String,
    phis: IndexSet<NodeRef>,
    phis_per_block: IndexMap<*const BasicBlock, Vec<NodeRef>>,
    indent: usize,
    visited: HashSet<NodeRef>,
    globals: &'a mut GlobalEmitter,
    inside_generic_loop: bool,
    // message: Vec<String>,
}

impl<'a> FunctionEmitter<'a> {
    fn new(globals: &'a mut GlobalEmitter, type_gen: &'a TypeGen) -> Self {
        Self {
            type_gen,
            node_to_var: HashMap::new(),
            body: String::new(),
            fwd_defs: String::new(),
            phis: IndexSet::new(),
            phis_per_block: IndexMap::new(),
            indent: 1,
            visited: HashSet::new(),
            globals,
            inside_generic_loop: false,
        }
    }
    fn write_ident(&mut self) {
        for _ in 0..self.indent {
            write!(&mut self.body, "    ").unwrap();
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
    fn access_chain(&mut self, mut var: String, node: NodeRef, indices: &[NodeRef]) -> String {
        let mut ty = node.type_().clone();
        for (i, index) in indices.iter().enumerate() {
            if ty.is_vector() || ty.is_matrix() {
                var = format!("{}[{}]", var, self.gen_node(*index));
                assert_eq!(i, indices.len() - 1);
                break;
            } else if ty.is_array() {
                var = format!("{}[{}]", var, self.gen_node(*index));
                ty = ty.extract(0)
            } else {
                assert!(ty.is_struct());
                let idx = index.get_i32() as usize;
                var = format!("{}.f{}", var, idx);
                ty = ty.extract(idx);
            }
        }
        var
    }
    fn atomic_chain_op(
        &mut self,
        var: &str,
        node_ty_s: &String,
        args: &[NodeRef],
        args_v: &[String],
        op: &str,
        noperand: usize,
    ) {
        let n = args.len();
        let buffer_ty = self.type_gen.gen_c_type(args[0].type_());
        let indices = &args[2..n - noperand];
        let buffer_ref = format!(
            "(*lc_buffer_ref<{0}>(k_args, {1}, {2}))",
            buffer_ty, args_v[0], args_v[1]
        );
        let access_chain = self.access_chain(buffer_ref, args[0], indices);
        writeln!(
            self.body,
            "const {} {} = {}(&{}, {});",
            node_ty_s,
            var,
            op,
            access_chain,
            args_v[n - noperand..].join(", ")
        )
        .unwrap();
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
            _ => unreachable!(),
        }
    }
    fn gen_callable(
        &mut self,
        var: &String,
        node_ty_s: &String,
        f: &Func,
        args_v: &Vec<String>,
    ) -> bool {
        let f = match f {
            Func::Callable(f) => f,
            _ => return false,
        };
        let fid = CArc::as_ptr(&f.0) as u64;
        if !self.globals.generated_callables.contains_key(&fid) {
            let mut callable_emitter = FunctionEmitter::new(&mut self.globals, &self.type_gen);

            let mut params = vec![];
            for (i, arg) in f.0.args.iter().enumerate() {
                let mut param = String::new();
                let var = format!("ca_{}", i);
                match arg.get().instruction.as_ref() {
                    Instruction::Accel => {
                        write!(&mut param, "const Accel& {}", var).unwrap();
                    }
                    Instruction::Bindless => {
                        write!(&mut param, "const BindlessArray& {}", var).unwrap();
                    }
                    Instruction::Buffer => {
                        write!(&mut param, "const BufferView& {}", var).unwrap();
                    }
                    Instruction::Texture2D => {
                        write!(&mut param, "const Texture2D& {}", var).unwrap();
                    }
                    Instruction::Texture3D => {
                        write!(&mut param, "const Texture3D& {}", var).unwrap();
                    }
                    Instruction::Argument { by_value } => {
                        let ty = self.type_gen.gen_c_type(arg.type_());
                        if *by_value {
                            write!(&mut param, "const {}& {}", ty, var).unwrap();
                        } else {
                            write!(&mut param, "{}* {}", ty, var).unwrap();
                        }
                    }
                    _ => {
                        unreachable!()
                    }
                }
                callable_emitter
                    .node_to_var
                    .insert(arg.clone(), var.clone());
                params.push(param);
            }
            callable_emitter.gen_callable_module(&f.0);
            callable_emitter.indent += self.indent;

            let ret_type = self.type_gen.gen_c_type(&f.0.ret_type);

            let fname = format!(
                "callable_{}",
                callable_emitter.globals.generated_callables.len()
            );
            let source = format!(
                "[=]({}) -> {} {{\n{}\n{};}}",
                params.join(","),
                ret_type,
                callable_emitter.fwd_defs,
                callable_emitter.body
            );
            if let Some(fname) = self.globals.generated_callable_sources.get(&source) {
                self.globals.generated_callables.insert(fid, fname.clone());
            } else {
                self.globals.generated_callables.insert(fid, fname.clone());
                self.globals
                    .generated_callable_sources
                    .insert(source.clone(), fname.clone());
                writeln!(
                    &mut self.globals.callable_def,
                    "const auto {} = {};\n",
                    fname, source
                )
                .unwrap();
                self.write_ident();
            }
        }
        let fname = &self.globals.generated_callables[&fid];
        if var != "" {
            writeln!(
                &mut self.body,
                "const {} {} = {}({});",
                node_ty_s,
                var,
                fname,
                args_v.join(", ")
            )
            .unwrap();
        } else {
            writeln!(&mut self.body, "{}({});", fname, args_v.join(", ")).unwrap();
        }
        true
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
            Func::Powf => Some("lc_pow"),

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
            Func::Distance => Some("lc_distance"),
            Func::Faceforward => Some("lc_faceforward"),
            Func::Reflect => Some("lc_reflect"),
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
            Func::Saturate => Some("lc_saturate"),
            Func::Lerp => Some("lc_lerp"),
            Func::Step => Some("lc_step"),
            Func::SmoothStep => Some("lc_smoothstep"),
            Func::SynchronizeBlock => Some("lc_synchronize_block"),
            Func::WarpIsFirstActiveLane => Some("lc_warp_is_first_active_lane"),
            Func::WarpFirstActiveLane => Some("lc_warp_first_active_lane"),
            Func::WarpActiveAllEqual => Some("lc_warp_active_all_equal"),
            Func::WarpActiveBitAnd => Some("lc_warp_active_bit_and"),
            Func::WarpActiveBitOr => Some("lc_warp_active_bit_or"),
            Func::WarpActiveBitXor => Some("lc_warp_active_bit_xor"),
            Func::WarpActiveCountBits => Some("lc_warp_active_count_bits"),
            Func::WarpActiveMax => Some("lc_warp_active_max"),
            Func::WarpActiveMin => Some("lc_warp_active_min"),
            Func::WarpActiveProduct => Some("lc_warp_active_product"),
            Func::WarpActiveSum => Some("lc_warp_active_sum"),
            Func::WarpActiveAll => Some("lc_warp_active_all"),
            Func::WarpActiveAny => Some("lc_warp_active_any"),
            Func::WarpActiveBitMask => Some("lc_warp_active_bit_mask"),
            Func::WarpPrefixSum => Some("lc_warp_prefix_sum"),
            Func::WarpPrefixProduct => Some("lc_warp_prefix_product"),
            Func::WarpReadLaneAt => Some("lc_warp_read_lane_at"),
            Func::WarpReadFirstLane => Some("lc_warp_read_first_lane"),
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
            Func::ByteBufferRead => {
                writeln!(
                    &mut self.body,
                    "const auto {1} = lc_byte_buffer_read<{0}>(k_args, {2}, {3});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::ByteBufferWrite => {
                let v_ty = self.type_gen.gen_c_type(args[2].type_());
                writeln!(
                    &mut self.body,
                    "lc_byte_buffer_write<{}>(k_args, {}, {}, {});",
                    v_ty, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::ByteBufferSize => {
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_buffer_size<uint8_t>(k_args, {});",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            Func::BufferRead => {
                let buffer_ty = self.type_gen.gen_c_type(args[0].type_());
                writeln!(
                    &mut self.body,
                    "const auto {1} = lc_buffer_read<{0}>(k_args, {2}, {3});",
                    buffer_ty, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::BufferWrite => {
                let buffer_ty = self.type_gen.gen_c_type(args[0].type_());
                writeln!(
                    &mut self.body,
                    "lc_buffer_write<{}>(k_args, {}, {}, {});",
                    buffer_ty, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::BufferSize => {
                let buffer_ty = self.type_gen.gen_c_type(args[0].type_());
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_buffer_size<{}>(k_args, {});",
                    node_ty_s, var, buffer_ty, args_v[0]
                )
                .unwrap();
                true
            }
            Func::BindlessByteBufferRead => {
                writeln!(
                    &mut self.body,
                    "const auto {1} = lc_bindless_byte_buffer_read<{0}>(k_args, {2}, {3}, {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::BindlessBufferRead => {
                writeln!(
                    &mut self.body,
                    "const auto {1} = lc_bindless_buffer_read<{0}>(k_args, {2}, {3}, {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::BindlessBufferSize => {
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_bindless_buffer_size(k_args, {}, {}, {});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::BindlessBufferType => {
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_bindless_buffer_type(k_args, {}, {});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::BindlessTexture2dRead => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture2d_read(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::BindlessTexture3dRead => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture3d_read(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::BindlessTexture2dReadLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture2d_read_level(k_args, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3]
                )
                    .unwrap();
                true
            }
            Func::BindlessTexture3dReadLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture3d_read_level(k_args, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3]
                )
                    .unwrap();
                true
            }
            Func::BindlessTexture2dSample => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture2d_sample(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2],
                )
                .unwrap();
                true
            }
            Func::BindlessTexture3dSample => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture3d_sample(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::BindlessTexture2dSampleLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture2d_sample_level(k_args, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3]
                )
                    .unwrap();
                true
            }
            Func::BindlessTexture3dSampleLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture3d_sample_level(k_args, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3]
                )
                    .unwrap();
                true
            }
            Func::BindlessTexture2dSampleGrad => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture2d_sample_grad(k_args, {}, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3], args_v[4]
                )
                    .unwrap();
                true
            }
            Func::BindlessTexture3dSampleGrad => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture3d_sample_grad(k_args, {}, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3], args_v[4]
                )
                    .unwrap();
                true
            }
            Func::BindlessTexture2dSize => {
                writeln!(
                    &mut self.body,
                    "const lc_uint2 {} = lc_bindless_texture2d_size(k_args, {}, {});",
                    var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::BindlessTexture3dSize => {
                writeln!(
                    &mut self.body,
                    "const lc_uint3 {} = lc_bindless_texture3d_size(k_args, {}, {});",
                    var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::BindlessTexture2dSizeLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_uint2 {} = lc_bindless_texture2d_size_level(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::BindlessTexture3dSizeLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_uint3 {} = lc_bindless_texture3d_size_level(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::Texture2dSize=> {
                writeln!(
                    &mut self.body,
                    "const lc_uint2 {} = lc_texture2d_size(k_args, {});",
                    var, args_v[0]
                )
                .unwrap();
                true
            }
            Func::Texture3dSize=> {
                writeln!(
                    &mut self.body,
                    "const lc_uint3 {} = lc_texture3d_size(k_args, {});",
                    var, args_v[0]
                )
                .unwrap();
                true
            }
            Func::Texture2dRead => {
                writeln!(
                    &mut self.body,
                    "const {0} {1} = lc_texture2d_read<{0}>(k_args, {2}, {3});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::Texture3dRead => {
                writeln!(
                    &mut self.body,
                    "const {0} {1} = lc_texture3d_read<{0}>(k_args, {2}, {3});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::Texture2dWrite => {
                writeln!(
                    &mut self.body,
                    "lc_texture2d_write(k_args, {0}, {1}, {2});",
                    args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::Texture3dWrite => {
                writeln!(
                    &mut self.body,
                    "lc_texture3d_write(k_args, {0}, {1}, {2});",
                    args_v[0], args_v[1], args_v[2]
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
            Func::PropagateGrad => true,
            Func::RequiresGradient => true,
            Func::AtomicRef => panic!("AtomicRef should have been lowered"),
            Func::Assume => {
                writeln!(&mut self.body, "lc_assume({});", args_v.join(", ")).unwrap();
                true
            }
            Func::Assert(msg) => {
                let msg = msg.to_string();
                let id = self.globals.message.len();
                self.globals.message.push(msg);
                writeln!(&mut self.body, "lc_assert({}, {});", args_v.join(", "), id).unwrap();
                true
            }
            Func::ShaderExecutionReorder => {
                writeln!(
                    &mut self.body,
                    "lc_shader_execution_reorder({});",
                    args_v.join(", ")
                )
                .unwrap();
                true
            }
            Func::Unreachable(msg) => {
                let msg = msg.to_string();
                let id = self.globals.message.len();
                self.globals.message.push(msg);
                if !is_type_equal(node_ty, &Type::void()) {
                    writeln!(&mut self.body, "{} {};", node_ty_s, var).unwrap();
                    self.write_ident();
                }
                writeln!(&mut self.body, "lc_unreachable({});", id).unwrap();
                true
            }
            Func::ExtractElement => {
                let indices = &args[1..];
                let access_chain = self.access_chain(args_v[0].clone(), args[0], indices);
                writeln!(
                    self.body,
                    "const {} {} = {};",
                    node_ty_s, var, access_chain
                )
                .unwrap();
                true
            }
            Func::InsertElement => {
                let indices = &args[2..];
                let access_chain = self.access_chain(format!("_{}", var), args[0], indices);
                writeln!(
                    self.body,
                    "{0} _{1} = {2}; {3} = {4}; const auto& {1} = _{1};",
                    node_ty_s, var, args_v[0], access_chain, args_v[1]
                )
                .unwrap();
                true
            }
            Func::GetElementPtr => {
                let indices = &args[1..];
                let access_chain = self.access_chain(format!("(*{})", args_v[0]), args[0], indices);
                writeln!(self.body, "{} * {} = &{};", node_ty_s, var, access_chain).unwrap();
                true
            }
            Func::Struct | Func::Array => {
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
                writeln!(self.body, "const {} {} = *{};", node_ty_s, var, args_v[0]).unwrap();
                true
            }
            Func::GradientMarker => {
                let ty = self.type_gen.gen_c_type(args[1].type_());
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
                    "const {0}& {1} = {2}_grad;",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            Func::AccGrad => {
                writeln!(
                    self.body,
                    "lc_accumulate_grad({0}, {1});",
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
                    "const {0}& {1} = lc_bit_cast<{0}>({2});",
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
            Func::Pack => {
                writeln!(
                    self.body,
                    "lc_pack_to({}, {}, {});",
                    args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::Unpack => {
                writeln!(
                    self.body,
                    "const {} {} = lc_unpack_from({}, {});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
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
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_exchange", 1);
                true
            }
            Func::AtomicCompareExchange => {
                self.atomic_chain_op(
                    var,
                    node_ty_s,
                    args,
                    args_v,
                    "lc_atomic_compare_exchange",
                    2,
                );
                true
            }
            Func::AtomicFetchAdd => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_add", 1);
                true
            }
            Func::AtomicFetchSub => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_sub", 1);
                true
            }
            Func::AtomicFetchMin => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_min", 1);
                true
            }
            Func::AtomicFetchMax => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_max", 1);
                true
            }
            Func::AtomicFetchAnd => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_and", 1);
                true
            }
            Func::AtomicFetchOr => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_or", 1);
                true
            }
            Func::AtomicFetchXor => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_xor", 1);
                true
            }
            Func::CpuCustomOp(op) => {
                let i = *self
                    .globals
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
                    "const {0} {1} = lc_trace_any({2}, lc_bit_cast<Ray>({3}), {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::RayTracingTraceClosest => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bit_cast<{0}>(lc_trace_closest({2}, lc_bit_cast<Ray>({3}), {4}));",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
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
                    "lc_set_instance_transform({0}, {1}, {2});",
                    args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::RayTracingSetInstanceVisibility => {
                writeln!(
                    self.body,
                    "lc_set_instance_visibility({0}, {1}, {2});",
                    args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            Func::RayTracingQueryAll => {
                writeln!(
                    self.body,
                    "LC_RayQueryAll _{0} = lc_ray_query_all({1}, lc_bit_cast<Ray>({2}), {3});auto& {0} = _{0};",
                    var, args_v[0], args_v[1], args_v[2]
                )
                    .unwrap();
                true
            }
            Func::RayTracingQueryAny => {
                writeln!(
                    self.body,
                    "LC_RayQueryAny _{0} = lc_ray_query_any({1}, lc_bit_cast<Ray>({2}), {3});auto& {0} = _{0};",
                    var, args_v[0], args_v[1], args_v[2]
                )
                    .unwrap();
                true
            }
            Func::RayQueryWorldSpaceRay => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bit_cast<{0}>(lc_ray_query_world_space_ray({2}));",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            Func::RayQueryProceduralCandidateHit => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bit_cast<{0}>(lc_ray_query_procedural_candidate_hit({2}));",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            Func::RayQueryTriangleCandidateHit => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bit_cast<{0}>(lc_ray_query_triangle_candidate_hit({2}));",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            Func::RayQueryCommittedHit => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bit_cast<{0}>(lc_ray_query_committed_hit({2}));",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            Func::RayQueryCommitTriangle => {
                writeln!(self.body, "lc_ray_query_commit_triangle({0});", args_v[0]).unwrap();
                true
            }
            Func::RayQueryCommitProcedural => {
                writeln!(
                    self.body,
                    "lc_ray_query_commit_procedural({0}, {1});",
                    args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            Func::RayQueryTerminate => {
                writeln!(self.body, "lc_ray_query_terminate({0});", args_v[0]).unwrap();
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
            Const::Int8(v) => {
                writeln!(&mut self.body, "const int8_t {} = {};", var, *v).unwrap();
            }
            Const::Uint8(v) => {
                writeln!(&mut self.body, "const uint8_t {} = {};", var, *v).unwrap();
            }
            Const::Int16(v) => {
                writeln!(&mut self.body, "const int16_t {} = {};", var, *v).unwrap();
            }
            Const::Uint16(v) => {
                writeln!(&mut self.body, "const uint16_t {} = {};", var, *v).unwrap();
            }
            Const::Int32(v) => {
                writeln!(&mut self.body, "const int32_t {} = {};", var, *v).unwrap();
            }
            Const::Uint32(v) => {
                writeln!(&mut self.body, "const uint32_t {} = {};", var, *v).unwrap();
            }
            Const::Int64(v) => {
                writeln!(&mut self.body, "const int64_t {} = {}ll;", var, *v).unwrap();
            }
            Const::Uint64(v) => {
                writeln!(&mut self.body, "const uint64_t {} = {}ull;", var, *v).unwrap();
            }
            Const::Float16(v) => {
                let bits = v.to_bits();
                writeln!(
                    &mut self.body,
                    "const lc_half {} = lc_bit_cast<half>(uint16_t(0x{:04x})); // {}",
                    var, bits, v
                )
                .unwrap();
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
                writeln!(
                    &mut self.body,
                    "const {0} {1} = {2};",
                    node_ty_s,
                    var,
                    decode_const_data(bytes.as_ref(), t)
                )
                .unwrap();
                // let gen_def = |dst: &mut String, qualifier| {
                //     writeln!(
                //         dst,
                //         "    {0} uint8_t {2}_bytes[{1}] = {{ {3} }};",
                //         qualifier,
                //         t.size(),
                //         var,
                //         bytes
                //             .as_ref()
                //             .iter()
                //             .map(|b| format!("{}", b))
                //             .collect::<Vec<String>>()
                //             .join(", ")
                //     )
                //         .unwrap();
                // };
                // match t.as_ref() {
                //     Type::Array(_) => {
                //         if !self.generated_globals.contains(var) {
                //             gen_def(&mut self.fwd_defs, "__constant__ constexpr const");
                //             self.generated_globals.insert(var.clone());
                //         }
                //     }
                //     _ => gen_def(&mut self.body, "constexpr const"),
                // }
                // self.write_ident();
                // writeln!(
                //     &mut self.body,
                //     "const {0} {1} = *reinterpret_cast<const {0}*>({1}_bytes);",
                //     node_ty_s, var
                // )
                //     .unwrap();
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
        let node_ty_s = self.type_gen.gen_c_type(node_ty);
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
                writeln!(
                    &mut self.body,
                    "{0} _{1} = {2};{0} * {1} = &_{1};",
                    node_ty_s, var, init_v
                )
                .unwrap();
            }
            Instruction::Argument { by_value: _ } => todo!(),
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
                writeln!(&mut self.body, "*{} = {};", var_v, value_v).unwrap();
            }
            Instruction::Call(f, args) => {
                // println!("call: {:?}({:?})", f, args);
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
                    done = self.gen_buffer_op(&var, &node_ty_s, f, args, &args_v);
                }
                if !done {
                    done = self.gen_misc(&var, node, &node_ty_s, node.type_(), f, args, &args_v);
                }
                if !done {
                    done = self.gen_callable(&var, &node_ty_s, f, &args_v);
                }
                assert!(done, "{:?} is not implemented", f);
            }
            Instruction::Phi(_) => {
                self.write_ident();
                let var = self.gen_node(node);
                writeln!(&mut self.fwd_defs, "    {0} {1} = {0}{{}};", node_ty_s, var).unwrap();
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
                let old_inside_generic_loop = self.inside_generic_loop;
                self.inside_generic_loop = false;
                {
                    self.write_ident();
                    writeln!(&mut self.body, "do {{").unwrap();
                    self.gen_block(*body);
                    let cond_v = self.gen_node(*cond);
                    self.write_ident();
                    writeln!(&mut self.body, "}} while({});", cond_v).unwrap();
                }
                self.inside_generic_loop = old_inside_generic_loop;
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
                let old_inside_generic_loop = self.inside_generic_loop;
                self.inside_generic_loop = true;
                {
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
                self.inside_generic_loop = old_inside_generic_loop;
            }
            Instruction::Break => {
                self.write_ident();
                if self.inside_generic_loop {
                    writeln!(&mut self.body, "loop_break = true;").unwrap();
                }
                writeln!(&mut self.body, "break;").unwrap();
            }
            Instruction::Continue => {
                self.write_ident();
                if self.inside_generic_loop {
                    writeln!(&mut self.body, "break;").unwrap();
                } else {
                    writeln!(&mut self.body, "continue;").unwrap();
                }
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
            Instruction::AdScope { body, .. } => {
                writeln!(&mut self.body, "/* AdScope */").unwrap();
                self.gen_block(*body);
                self.write_ident();
                writeln!(&mut self.body, "/* AdScope End */").unwrap();
            }
            Instruction::AdDetach(bb) => {
                self.write_ident();
                writeln!(&mut self.body, "/* AdDetach */").unwrap();
                self.gen_block(*bb);
                self.write_ident();
                writeln!(&mut self.body, "/* AdDetach End */").unwrap();
            }
            Instruction::RayQuery {
                ray_query,
                on_triangle_hit,
                on_procedural_hit,
            } => {
                writeln!(&mut self.body, "/* RayQuery */").unwrap();
                let rq_v = self.gen_node(*ray_query);
                self.write_ident();
                writeln!(
                    &mut self.body,
                    "lc_ray_query({}, [&](const TriangleHit& hit) {{",
                    rq_v
                )
                .unwrap();
                self.gen_block(*on_triangle_hit);
                self.write_ident();
                writeln!(&mut self.body, "}}, [&](const ProceduralHit& hit) {{").unwrap();
                self.gen_block(*on_procedural_hit);
                self.write_ident();
                writeln!(&mut self.body, "}});").unwrap();
                self.write_ident();
                writeln!(&mut self.body, "/* RayQuery End*/").unwrap();
            }
            Instruction::Comment(comment) => {
                self.write_ident();
                writeln!(&mut self.body, "/* {} */", comment.to_string()).unwrap();
            }
            Instruction::Print { fmt, args } => {
                todo!()
            }
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
                writeln!(
                    &mut self.fwd_defs,
                    "    const Accel& {} = {}[{}].accel._0;",
                    arg_name, arg_array, index
                )
                .unwrap();
            }
            Instruction::Bindless => {
                writeln!(
                    &mut self.fwd_defs,
                    "    const BindlessArray& {} = {}[{}].bindless_array._0;",
                    arg_name, arg_array, index
                )
                .unwrap();
            }
            Instruction::Buffer => {
                writeln!(
                    &mut self.fwd_defs,
                    "    const BufferView& {} = {}[{}].buffer._0;",
                    arg_name, arg_array, index
                )
                .unwrap();
            }
            Instruction::Texture2D => {
                writeln!(
                    &mut self.fwd_defs,
                    "    const Texture2D& {} = {}[{}].texture;",
                    arg_name, arg_array, index
                )
                .unwrap();
            }
            Instruction::Texture3D => {
                writeln!(
                    &mut self.fwd_defs,
                    "    const Texture3D& {} = {}[{}].texture;",
                    arg_name, arg_array, index
                )
                .unwrap();
            }
            Instruction::Uniform => {
                let ty = self.type_gen.gen_c_type(node.type_());
                writeln!(
                    &mut self.fwd_defs,
                    "    const {0}& {1} = *reinterpret_cast<const {0}*>({2}[{3}].uniform._0);",
                    ty, arg_name, arg_array, index
                )
                .unwrap();
            }
            _ => unreachable!(),
        }
        if is_capture {
            assert!(!self.globals.captures.contains_key(&node));
            self.globals.captures.insert(node, index);
        } else {
            assert!(!self.globals.args.contains_key(&node));
            self.globals.args.insert(node, index);
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
        for (i, arg) in module.args.iter().enumerate() {
            self.gen_arg(*arg, i, false);
        }
        assert!(self.globals.global_vars.is_empty());
        self.globals.global_vars = self.node_to_var.clone();
        for (i, op) in module.cpu_custom_ops.as_ref().iter().enumerate() {
            self.globals
                .cpu_custom_ops
                .insert(CArc::as_ptr(op) as usize, i);
        }
        self.gen_block(module.module.entry);
    }
    fn gen_callable_module(&mut self, module: &ir::CallableModule) {
        let mut phi_collector = PhiCollector::new();
        phi_collector.visit_block(module.module.entry);
        let PhiCollector {
            phis_per_block,
            phis,
        } = phi_collector;
        self.phis_per_block = phis_per_block;
        self.phis = phis;
        // self.node_to_var = self.globals.global_vars.clone();
        for (k, v) in &self.globals.global_vars {
            assert!(!self.node_to_var.contains_key(k));
            self.node_to_var.insert(k.clone(), v.clone());
        }
        for (_, capture) in module.captures.as_ref().iter().enumerate() {
            assert!(self.globals.captures.contains_key(&capture.node));
        }
        self.gen_block(module.module.entry);
    }
}

pub struct CpuCodeGen;

pub struct Generated {
    pub source: String,
    pub messages: Vec<String>,
}

impl CpuCodeGen {
    pub(crate) fn run(module: &ir::KernelModule) -> Generated {
        let mut globals = GlobalEmitter {
            message: vec![],
            generated_callables: HashMap::new(),
            generated_callable_sources: HashMap::new(),
            global_vars: HashMap::new(),
            captures: IndexMap::new(),
            args: IndexMap::new(),
            cpu_custom_ops: IndexMap::new(),
            callable_def: String::new(),
        };
        let type_gen = TypeGen::new();
        let mut codegen = FunctionEmitter::new(&mut globals, &type_gen);
        codegen.gen_module(module);
        let defs = r#"using uint8_t = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
using int8_t = signed char;
using int16_t = signed short;
using int32_t = signed int;
using int64_t = signed long long;
using size_t = unsigned long long;
struct Accel;"#;
        let kernel_fn_decl = r#"lc_kernel void ##kernel_fn##(const KernelFnArgs* k_args) {"#;
        Generated {
            source: format!(
                "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
                defs,
                CPU_LIBM_DEF,
                CPU_KERNEL_DEFS,
                CPU_PRELUDE,
                DEVICE_MATH_SRC,
                CPU_RESOURCE,
                CPU_TEXTURE,
                type_gen.generated(),
                kernel_fn_decl,
                codegen.fwd_defs,
                codegen.globals.callable_def,
                codegen.body,
                "}",
            ),
            messages: globals.message,
        }
    }
}

pub const CPU_PRELUDE: &str = include_str!("cpu_prelude.h");
pub const CPU_RESOURCE: &str = include_str!("cpu_resource.h");
pub const DEVICE_MATH_SRC: &str = include_str!("device_math.h");
pub const CPU_LIBM_DEF: &str = include_str!("cpu_libm_def.h");
pub const CPU_KERNEL_DEFS: &str =
    include_str!("../../../../luisa_compute_cpu_kernel_defs/cpu_kernel_defs.h");
pub const CPU_TEXTURE: &str = include_str!("cpu_texture.h");
