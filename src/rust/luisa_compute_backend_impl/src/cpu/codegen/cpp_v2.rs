use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

use super::{decode_const_data, decode_const_data_v2, sha256_short};
use indexmap::{IndexMap, IndexSet};
use luisa_compute_ir_v2::*;
use std::fmt::Write;
pub(crate) struct TypeGenInner {
    cache: HashMap<TypeRef, String>,
    struct_typedefs: String,
}

impl TypeGenInner {
    pub(crate) fn new() -> Self {
        Self {
            cache: HashMap::new(),
            struct_typedefs: String::new(),
        }
    }
    fn to_c_type_(&mut self, t: TypeRef) -> String {
        if t.is_null() {
            return "void".to_string();
        }
        match t.tag() {
            TypeTag::Bool => return "bool".to_string(),
            TypeTag::Float16 => return "half".to_string(),
            TypeTag::Float32 => return "float".to_string(),
            TypeTag::Float64 => return "double".to_string(),
            TypeTag::Int16 => return "int16_t".to_string(),
            TypeTag::Int32 => return "int32_t".to_string(),
            TypeTag::Int64 => return "int64_t".to_string(),
            TypeTag::Uint16 => return "uint16_t".to_string(),
            TypeTag::Uint32 => return "uint32_t".to_string(),
            TypeTag::Uint64 => return "uint64_t".to_string(),
            TypeTag::Vector => {
                let dim = t.dimension();
                let el = t.element();
                match el.tag() {
                    TypeTag::Bool => return format!("lc_bool{}", dim),
                    TypeTag::Float16 => return format!("lc_half{}", dim),
                    TypeTag::Float32 => return format!("lc_float{}", dim),
                    TypeTag::Float64 => return format!("lc_double{}", dim),
                    TypeTag::Int16 => return format!("lc_short{}", dim),
                    TypeTag::Int32 => return format!("lc_int{}", dim),
                    TypeTag::Int64 => return format!("lc_long{}", dim),
                    TypeTag::Uint16 => return format!("lc_ushort{}", dim),
                    TypeTag::Uint32 => return format!("lc_uint{}", dim),
                    TypeTag::Uint64 => return format!("lc_ulong{}", dim),
                    _ => unreachable!(),
                }
            }
            TypeTag::Matrix => {
                let dim = t.dimension();
                format!("lc_float{0}x{0}", dim)
            }
            TypeTag::Array => {
                let dim = t.dimension();
                let el = t.element();
                let el = self.to_c_type(el);
                format!("lc_array<{}, {}>", el, dim)
            }
            TypeTag::Struct => {
                let fields = t.members();
                let alignment = t.alignment();
                let field_types: Vec<String> = fields.iter().map(|f| self.to_c_type(*f)).collect();
                let field_types_str = field_types.join(", ");
                let hash = sha256_short(&format!("{}_alignas({})", field_types_str, alignment));
                let hash = hash.replace("-", "x_");
                let name = format!("s_{}", hash);

                self.cache.insert(t.clone(), name.clone());
                let mut tmp = String::new();
                writeln!(tmp, "struct alignas({0}) {1} {{", alignment, name).unwrap();
                for (i, field) in fields.iter().enumerate() {
                    let field_name = format!("f{}", i);
                    let field_type = self.to_c_type(*field);
                    writeln!(tmp, "    {} {};", field_type, field_name).unwrap();
                }
                writeln!(tmp, "    __device__ constexpr static auto one() {{").unwrap();
                writeln!(tmp, "        return {0} {{", name).unwrap();
                for (_, field) in fields.as_ref().iter().enumerate() {
                    let field_type = self.to_c_type(*field);
                    writeln!(tmp, "        lc_one<{}>(),", field_type).unwrap();
                }
                writeln!(tmp, "        }};").unwrap();
                writeln!(tmp, "    }}").unwrap();
                writeln!(tmp, "    __device__ constexpr static auto zero() {{").unwrap();
                writeln!(tmp, "        return {0} {{", name).unwrap();
                for (_, field) in fields.as_ref().iter().enumerate() {
                    let field_type = self.to_c_type(*field);
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
                for (i, _t) in fields.iter().enumerate() {
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
            TypeTag::Custom => {
                let name = t.description();
                let builtin = ["LC_RayQueryAny", "LC_RayQueryAll"];
                if !builtin.contains(&name.as_str()) {
                    self.struct_typedefs
                        .push_str(&format!("struct {};", name.to_string()));
                }
                name
            }
            _ => unreachable!(),
        }
    }
    pub(crate) fn to_c_type(&mut self, t: TypeRef) -> String {
        if let Some(t) = self.cache.get(&t) {
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
    fn gen_c_type(&self, t: TypeRef) -> String {
        self.inner.borrow_mut().to_c_type(t)
    }
    fn generated(&self) -> String {
        self.inner.borrow().struct_typedefs.clone()
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
    phis_per_block: IndexMap<BasicBlockRef, Vec<NodeRef>>,
    indent: usize,
    visited: HashSet<NodeRef>,
    globals: &'a mut GlobalEmitter,
    inside_generic_loop: bool,
    // message: Vec<String>,
}

pub struct PhiCollector {
    phis: IndexSet<NodeRef>,
    phis_per_block: IndexMap<BasicBlockRef, Vec<NodeRef>>,
}

impl PhiCollector {
    pub fn new() -> Self {
        Self {
            phis: IndexSet::new(),
            phis_per_block: IndexMap::new(),
        }
    }
    pub unsafe fn visit_block(&mut self, block: BasicBlockRef) {
        for phi in block.phis() {
            self.phis.insert(phi);
            let phi_node = phi;
            let phi = phi.inst().as_phi();
            if !phi.is_null() {
                let incomings = phi.incomings();
                for incoming in incomings.as_ref() {
                    let ptr = incoming.block;
                    self.phis_per_block
                        .entry(ptr)
                        .or_insert_with(Vec::new)
                        .push(phi_node);
                }
            } else {
                unreachable!()
            }
        }
        for node in block.iter() {
            let inst = node.inst();
            let tag = inst.tag();
            match tag {
                InstructionTag::If => {
                    let if_ = inst.as_if();
                    let true_branch = if_.true_branch();
                    let false_branch = if_.false_branch();
                    self.visit_block(true_branch);
                    self.visit_block(false_branch);
                }
                InstructionTag::Switch => {
                    let sw = inst.as_switch();
                    let default = sw.default();
                    let cases = sw.cases();
                    self.visit_block(default);
                    for SwitchCase { value: _, block } in cases.as_ref() {
                        self.visit_block(*block);
                    }
                }
                InstructionTag::GenericLoop => {
                    let generic_loop = inst.as_generic_loop();
                    let prepare = generic_loop.prepare();
                    let body = generic_loop.body();
                    let update = generic_loop.update();
                    self.visit_block(prepare);
                    self.visit_block(body);
                    self.visit_block(update);
                }
                InstructionTag::RayQuery => {
                    let rq = inst.as_ray_query();
                    let on_triangle_hit = rq.on_triangle_hit();
                    let on_procedural_hit = rq.on_procedural_hit();
                    self.visit_block(on_triangle_hit);
                    self.visit_block(on_procedural_hit);
                }
                InstructionTag::FwdAutodiff => {
                    let fwd = inst.as_fwd_autodiff();
                    let body = fwd.body();
                    self.visit_block(body);
                }
                InstructionTag::RevAutodiff => {
                    let rev = inst.as_rev_autodiff();
                    let body = rev.body();
                    self.visit_block(body);
                }
                _ => {}
            }
        }
    }
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
            let var = match node.inst().tag() {
                InstructionTag::Buffer => format!("b{}", index),
                InstructionTag::BindlessArray => format!("bl{}", index),
                InstructionTag::Texture2d => format!("t2d{}", index),
                InstructionTag::Texture3d => format!("t3d{}", index),
                InstructionTag::Accel => format!("a{}", index),
                InstructionTag::Shared => format!("s{}", index),
                InstructionTag::Uniform => format!("u{}", index),
                InstructionTag::Local => format!("v{}", index),
                InstructionTag::Argument => format!("arg{}", index),
                InstructionTag::Constant => format!("c{}", index),
                InstructionTag::Call => {
                    if node.type_().is_void() {
                        "".to_string()
                    } else {
                        format!("f{}", index)
                    }
                }
                InstructionTag::Phi => format!("phi{}", index),
                _ => unreachable!(),
            };
            self.node_to_var.insert(node, var.clone());
            return var;
        }
    }
    fn access_chain(&mut self, mut var: String, node: NodeRef, indices: &[NodeRef]) -> String {
        let mut ty = node.type_();
        for (i, index) in indices.iter().enumerate() {
            if ty.is_matrix() || ty.is_vector() {
                var = format!("{}[{}]", var, self.gen_node(*index));
                assert_eq!(i, indices.len() - 1);
                break;
            } else if ty.is_array() {
                var = format!("{}[{}]", var, self.gen_node(*index));
                ty = ty.element();
            } else {
                assert!(ty.is_struct());
                let idx = node.get_index() as u32;
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
        let ty = node.type_();
        if ty.is_struct() {
            format!("f{}", i)
        } else if ty.is_vector() {
            match i {
                0 => "x".to_string(),
                1 => "y".to_string(),
                2 => "z".to_string(),
                3 => "w".to_string(),
                _ => unreachable!(),
            }
        } else if ty.is_matrix() {
            format!("cols[{}]", i)
        } else {
            unreachable!()
        }
    }
    fn gen_binop(
        &mut self,
        var: &String,
        node_ty_s: &String,
        f: FuncRef,
        args_v: &Vec<String>,
    ) -> bool {
        let binop = match f.tag() {
            FuncTag::Add => Some("+"),
            FuncTag::Sub => Some("-"),
            FuncTag::Mul => Some("*"),
            FuncTag::Div => Some("/"),
            FuncTag::Rem => Some("%"),
            FuncTag::BitAnd => Some("&"),
            FuncTag::BitOr => Some("|"),
            FuncTag::BitXor => Some("^"),
            FuncTag::Shl => Some("<<"),
            FuncTag::Shr => Some(">>"),
            FuncTag::Eq => Some("=="),
            FuncTag::Ne => Some("!="),
            FuncTag::Lt => Some("<"),
            FuncTag::Le => Some("<="),
            FuncTag::Gt => Some(">"),
            FuncTag::Ge => Some(">="),
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
        f: FuncRef,
        args_v: &Vec<String>,
    ) -> bool {
        let func = match f.tag() {
            FuncTag::Abs => Some("lc_abs"),
            FuncTag::Acos => Some("lc_acos"),
            FuncTag::Acosh => Some("lc_acosh"),
            FuncTag::Asin => Some("lc_asin"),
            FuncTag::Asinh => Some("lc_asinh"),
            FuncTag::Atan => Some("lc_atan"),
            FuncTag::Atan2 => Some("lc_atan2"),
            FuncTag::Atanh => Some("lc_atanh"),
            FuncTag::Cos => Some("lc_cos"),
            FuncTag::Cosh => Some("lc_cosh"),
            FuncTag::Sin => Some("lc_sin"),
            FuncTag::Sinh => Some("lc_sinh"),
            FuncTag::Tan => Some("lc_tan"),
            FuncTag::Tanh => Some("lc_tanh"),

            FuncTag::Exp => Some("lc_exp"),
            FuncTag::Exp2 => Some("lc_exp2"),
            FuncTag::Exp10 => Some("lc_exp10"),
            FuncTag::Log => Some("lc_log"),
            FuncTag::Log2 => Some("lc_log2"),
            FuncTag::Log10 => Some("lc_log10"),
            FuncTag::Powi => Some("lc_powi"), // TODO: powi
            FuncTag::Powf => Some("lc_pow"),

            FuncTag::Sqrt => Some("lc_sqrt"),
            FuncTag::Rsqrt => Some("lc_rsqrt"),

            FuncTag::Ceil => Some("lc_ceil"),
            FuncTag::Floor => Some("lc_floor"),
            FuncTag::Fract => Some("lc_fract"),
            FuncTag::Trunc => Some("lc_trunc"),
            FuncTag::Round => Some("lc_round"),

            FuncTag::Fma => Some("lc_fma"),
            FuncTag::Copysign => Some("lc_copysign"),
            FuncTag::Cross => Some("lc_cross"),
            FuncTag::Dot => Some("lc_dot"),
            FuncTag::OuterProduct => Some("lc_outer_product"),
            FuncTag::Length => Some("lc_length"),
            FuncTag::LengthSquared => Some("lc_length_squared"),
            FuncTag::Normalize => Some("lc_normalize"),
            FuncTag::Distance => Some("lc_distance"),
            FuncTag::Faceforward => Some("lc_faceforward"),
            FuncTag::Reflect => Some("lc_reflect"),
            FuncTag::Determinant => Some("lc_determinant"),
            FuncTag::Transpose => Some("lc_transpose"),
            FuncTag::Inverse => Some("lc_inverse"),
            FuncTag::ReduceSum => Some("lc_reduce_sum"),
            FuncTag::ReduceProd => Some("lc_reduce_prod"),
            FuncTag::ReduceMin => Some("lc_reduce_min"),
            FuncTag::ReduceMax => Some("lc_reduce_max"),

            FuncTag::IsInf => Some("lc_isinf"),
            FuncTag::IsNan => Some("lc_isnan"),
            FuncTag::Any => Some("lc_any"),
            FuncTag::All => Some("lc_all"),

            FuncTag::PopCount => Some("lc_popcount"),
            FuncTag::Clz => Some("lc_clz"),
            FuncTag::Ctz => Some("lc_ctz"),
            FuncTag::Reverse => Some("lc_reverse"),
            FuncTag::Min => Some("lc_min"),
            FuncTag::Max => Some("lc_max"),
            FuncTag::Clamp => Some("lc_clamp"),
            FuncTag::Saturate => Some("lc_saturate"),
            FuncTag::Lerp => Some("lc_lerp"),
            FuncTag::Step => Some("lc_step"),
            FuncTag::SmoothStep => Some("lc_smoothstep"),
            FuncTag::SynchronizeBlock => Some("lc_synchronize_block"),
            FuncTag::WarpIsFirstActiveLane => Some("lc_warp_is_first_active_lane"),
            FuncTag::WarpFirstActiveLane => Some("lc_warp_first_active_lane"),
            FuncTag::WarpActiveAllEqual => Some("lc_warp_active_all_equal"),
            FuncTag::WarpActiveBitAnd => Some("lc_warp_active_bit_and"),
            FuncTag::WarpActiveBitOr => Some("lc_warp_active_bit_or"),
            FuncTag::WarpActiveBitXor => Some("lc_warp_active_bit_xor"),
            FuncTag::WarpActiveCountBits => Some("lc_warp_active_count_bits"),
            FuncTag::WarpActiveMax => Some("lc_warp_active_max"),
            FuncTag::WarpActiveMin => Some("lc_warp_active_min"),
            FuncTag::WarpActiveProduct => Some("lc_warp_active_product"),
            FuncTag::WarpActiveSum => Some("lc_warp_active_sum"),
            FuncTag::WarpActiveAll => Some("lc_warp_active_all"),
            FuncTag::WarpActiveAny => Some("lc_warp_active_any"),
            FuncTag::WarpActiveBitMask => Some("lc_warp_active_bit_mask"),
            FuncTag::WarpPrefixSum => Some("lc_warp_prefix_sum"),
            FuncTag::WarpPrefixProduct => Some("lc_warp_prefix_product"),
            FuncTag::WarpReadLaneAt => Some("lc_warp_read_lane_at"),
            FuncTag::WarpReadFirstLane => Some("lc_warp_read_first_lane"),
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
        f: FuncRef,
        args: &[NodeRef],
        args_v: &Vec<String>,
    ) -> bool {
        match f.tag() {
            FuncTag::ByteBufferRead => {
                writeln!(
                    &mut self.body,
                    "const auto {1} = lc_byte_buffer_read<{0}>(k_args, {2}, {3});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::ByteBufferWrite => {
                let v_ty = self.type_gen.gen_c_type(args[2].type_());
                writeln!(
                    &mut self.body,
                    "lc_byte_buffer_write<{}>(k_args, {}, {}, {});",
                    v_ty, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::ByteBufferSize => {
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_buffer_size<uint8_t>(k_args, {});",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            FuncTag::BufferRead => {
                let buffer_ty = self.type_gen.gen_c_type(args[0].type_());
                writeln!(
                    &mut self.body,
                    "const auto {1} = lc_buffer_read<{0}>(k_args, {2}, {3});",
                    buffer_ty, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::BufferWrite => {
                let buffer_ty = self.type_gen.gen_c_type(args[0].type_());
                writeln!(
                    &mut self.body,
                    "lc_buffer_write<{}>(k_args, {}, {}, {});",
                    buffer_ty, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::BufferSize => {
                let buffer_ty = self.type_gen.gen_c_type(args[0].type_());
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_buffer_size<{}>(k_args, {});",
                    node_ty_s, var, buffer_ty, args_v[0]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessByteBufferRead => {
                writeln!(
                    &mut self.body,
                    "const auto {1} = lc_bindless_byte_buffer_read<{0}>(k_args, {2}, {3}, {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessBufferWrite => {
                let v_ty = self.type_gen.gen_c_type(args[3].type_());
                writeln!(
                    &mut self.body,
                    "lc_bindless_buffer_write<{}>(k_args, {}, {}, {}, {});",
                    v_ty, args_v[0], args_v[1], args_v[2], args_v[3]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessBufferRead => {
                writeln!(
                    &mut self.body,
                    "const auto {1} = lc_bindless_buffer_read<{0}>(k_args, {2}, {3}, {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessBufferSize => {
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_bindless_buffer_size(k_args, {}, {}, {});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessBufferType => {
                writeln!(
                    &mut self.body,
                    "const {} {} = lc_bindless_buffer_type(k_args, {}, {});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessTexture2dRead => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture2d_read(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessTexture3dRead => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture3d_read(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessTexture2dReadLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture2d_read_level(k_args, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3]
                )
                    .unwrap();
                true
            }
            FuncTag::BindlessTexture3dReadLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture3d_read_level(k_args, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3]
                )
                    .unwrap();
                true
            }
            FuncTag::BindlessTexture2dSample => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture2d_sample(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2],
                )
                .unwrap();
                true
            }
            FuncTag::BindlessTexture3dSample => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture3d_sample(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessTexture2dSampleLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture2d_sample_level(k_args, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3]
                )
                    .unwrap();
                true
            }
            FuncTag::BindlessTexture3dSampleLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture3d_sample_level(k_args, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3]
                )
                    .unwrap();
                true
            }
            FuncTag::BindlessTexture2dSampleGrad => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture2d_sample_grad(k_args, {}, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3], args_v[4]
                )
                    .unwrap();
                true
            }
            FuncTag::BindlessTexture3dSampleGrad => {
                writeln!(
                    &mut self.body,
                    "const lc_float4 {} = lc_bindless_texture3d_sample_grad(k_args, {}, {}, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2], args_v[3], args_v[4]
                )
                    .unwrap();
                true
            }
            FuncTag::BindlessTexture2dSize => {
                writeln!(
                    &mut self.body,
                    "const lc_uint2 {} = lc_bindless_texture2d_size(k_args, {}, {});",
                    var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessTexture3dSize => {
                writeln!(
                    &mut self.body,
                    "const lc_uint3 {} = lc_bindless_texture3d_size(k_args, {}, {});",
                    var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessTexture2dSizeLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_uint2 {} = lc_bindless_texture2d_size_level(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::BindlessTexture3dSizeLevel => {
                writeln!(
                    &mut self.body,
                    "const lc_uint3 {} = lc_bindless_texture3d_size_level(k_args, {}, {}, {});",
                    var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::Texture2dSize => {
                writeln!(
                    &mut self.body,
                    "const lc_uint2 {} = lc_texture2d_size(k_args, {});",
                    var, args_v[0]
                )
                .unwrap();
                true
            }
            FuncTag::Texture3dSize => {
                writeln!(
                    &mut self.body,
                    "const lc_uint3 {} = lc_texture3d_size(k_args, {});",
                    var, args_v[0]
                )
                .unwrap();
                true
            }
            FuncTag::Texture2dRead => {
                writeln!(
                    &mut self.body,
                    "const {0} {1} = lc_texture2d_read<{0}>(k_args, {2}, {3});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::Texture3dRead => {
                writeln!(
                    &mut self.body,
                    "const {0} {1} = lc_texture3d_read<{0}>(k_args, {2}, {3});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::Texture2dWrite => {
                writeln!(
                    &mut self.body,
                    "lc_texture2d_write(k_args, {0}, {1}, {2});",
                    args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::Texture3dWrite => {
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
        node_ty: TypeRef,
        f: FuncRef,
        args: &[NodeRef],
        args_v: &Vec<String>,
    ) -> bool {
        match f.tag() {
            FuncTag::PropagateGradient => true,
            FuncTag::RequiresGradient => true,
            FuncTag::Assume => {
                writeln!(&mut self.body, "lc_assume({});", args_v.join(", ")).unwrap();
                true
            }
            FuncTag::Assert => {
                let assert = f.as_assert();
                let msg = assert.message();
                let id = self.globals.message.len();
                self.globals.message.push(msg);
                writeln!(&mut self.body, "lc_assert({}, {});", args_v.join(", "), id).unwrap();
                true
            }
            FuncTag::ShaderExecutionReorder => {
                writeln!(
                    &mut self.body,
                    "lc_shader_execution_reorder({});",
                    args_v.join(", ")
                )
                .unwrap();
                true
            }
            FuncTag::Unreachable => {
                let u = f.as_unreachable();
                let msg = u.message();
                let id = self.globals.message.len();
                self.globals.message.push(msg);
                if !node_ty.is_void() {
                    writeln!(&mut self.body, "{} {};", node_ty_s, var).unwrap();
                    self.write_ident();
                }
                writeln!(&mut self.body, "lc_unreachable({});", id).unwrap();
                true
            }
            FuncTag::ExtractElement => {
                let indices = &args[1..];
                let access_chain = self.access_chain(args_v[0].clone(), args[0], indices);
                writeln!(self.body, "const {} {} = {};", node_ty_s, var, access_chain).unwrap();
                true
            }
            FuncTag::InsertElement => {
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
            FuncTag::GetElementPtr => {
                let indices = &args[1..];
                let access_chain = self.access_chain(format!("(*{})", args_v[0]), args[0], indices);
                writeln!(self.body, "{} * {} = &{};", node_ty_s, var, access_chain).unwrap();
                true
            }
            FuncTag::Struct | FuncTag::Array => {
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
            FuncTag::DispatchId => {
                writeln!(self.body, "const {} {} = lc_dispatch_id();", node_ty_s, var).unwrap();
                true
            }
            FuncTag::DispatchSize => {
                writeln!(
                    self.body,
                    "const {} {} = lc_dispatch_size();",
                    node_ty_s, var
                )
                .unwrap();
                true
            }
            FuncTag::BlockId => {
                writeln!(self.body, "const {} {} = lc_block_id();", node_ty_s, var).unwrap();
                true
            }
            FuncTag::ThreadId => {
                writeln!(self.body, "const {} {} = lc_thread_id();", node_ty_s, var).unwrap();
                true
            }
            FuncTag::Zero => {
                writeln!(
                    self.body,
                    "const {} {} = lc_zero<{}>();",
                    node_ty_s, var, node_ty_s
                )
                .unwrap();
                true
            }
            FuncTag::Vec | FuncTag::Vec2 | FuncTag::Vec3 | FuncTag::Vec4 => {
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
            FuncTag::Mat2 | FuncTag::Mat3 | FuncTag::Mat4 => {
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
            FuncTag::MatFull => {
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
            FuncTag::MatCompMul => {
                writeln!(
                    self.body,
                    "const {} {} = {}.comp_mul({});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::Select => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_select({4},{3},{2});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::Load => {
                writeln!(self.body, "const {} {} = *{};", node_ty_s, var, args_v[0]).unwrap();
                true
            }
            FuncTag::Detach => {
                writeln!(self.body, "const {} {} = {};", node_ty_s, var, args_v[0]).unwrap();
                true
            }
            FuncTag::Gradient => {
                writeln!(
                    self.body,
                    "const {0}& {1} = {2}_grad;",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            FuncTag::AccGrad => {
                writeln!(
                    self.body,
                    "lc_accumulate_grad({0}, {1});",
                    args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::BitNot => {
                if node_ty.is_bool() {
                    writeln!(self.body, "const {} {} = !{};", node_ty_s, var, args_v[0]).unwrap();
                } else {
                    writeln!(self.body, "const {} {} = ~{};", node_ty_s, var, args_v[0]).unwrap();
                }
                true
            }
            FuncTag::Neg => {
                writeln!(self.body, "const {} {} = -{};", node_ty_s, var, args_v[0]).unwrap();
                true
            }
            FuncTag::Not => {
                writeln!(self.body, "const {} {} = !{};", node_ty_s, var, args_v[0]).unwrap();
                true
            }
            FuncTag::BitCast => {
                writeln!(
                    self.body,
                    "const {0}& {1} = lc_bit_cast<{0}>({2});",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            FuncTag::Cast => {
                if node_ty.is_bool() && node_ty.is_scalar() {
                    writeln!(
                        self.body,
                        "const {} {} = {} != 0;",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap();
                } else if (node_ty.is_float() || node_ty.is_int()) && node_ty.is_scalar() {
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
            FuncTag::Permute => {
                let indices: Vec<_> = args[1..].iter().map(|a| a.get_index()).collect();
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
            FuncTag::AtomicExchange => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_exchange", 1);
                true
            }
            FuncTag::AtomicCompareExchange => {
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
            FuncTag::AtomicFetchAdd => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_add", 1);
                true
            }
            FuncTag::AtomicFetchSub => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_sub", 1);
                true
            }
            FuncTag::AtomicFetchMin => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_min", 1);
                true
            }
            FuncTag::AtomicFetchMax => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_max", 1);
                true
            }
            FuncTag::AtomicFetchAnd => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_and", 1);
                true
            }
            FuncTag::AtomicFetchOr => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_or", 1);
                true
            }
            FuncTag::AtomicFetchXor => {
                self.atomic_chain_op(var, node_ty_s, args, args_v, "lc_atomic_fetch_xor", 1);
                true
            }
            FuncTag::CpuExt => {
                let cpu_ext = f.as_cpu_ext();
                let f = cpu_ext.func();
                let i = *self
                    .globals
                    .cpu_custom_ops
                    .get(&(f.as_ptr() as usize))
                    .unwrap();
                writeln!(
                    self.body,
                    "const {0} {1} = lc_cpu_custom_op(k_args, {2}, {3});",
                    node_ty_s, var, i, args_v[0],
                )
                .unwrap();
                true
            }
            FuncTag::RayTracingTraceAny => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_trace_any({2}, lc_bit_cast<Ray>({3}), {4});",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::RayTracingTraceClosest => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bit_cast<{0}>(lc_trace_closest({2}, lc_bit_cast<Ray>({3}), {4}));",
                    node_ty_s, var, args_v[0], args_v[1], args_v[2]
                )
                    .unwrap();
                true
            }
            FuncTag::RayTracingInstanceTransform => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_accel_instance_transform({2}, {3});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::RayTracingInstanceVisibilityMask => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_accel_instance_visibility_mask({2}, {3});",
                    node_ty_s, var, args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::RayTracingSetInstanceUserId => {
                writeln!(
                    self.body,
                    "lc_set_instance_user_id({0}, {1}, {2});",
                    args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::RayTracingSetInstanceTransform => {
                writeln!(
                    self.body,
                    "lc_set_instance_transform({0}, {1}, {2});",
                    args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::RayTracingSetInstanceVisibility => {
                writeln!(
                    self.body,
                    "lc_set_instance_visibility({0}, {1}, {2});",
                    args_v[0], args_v[1], args_v[2]
                )
                .unwrap();
                true
            }
            FuncTag::RayTracingQueryAll => {
                writeln!(
                    self.body,
                    "LC_RayQueryAll _{0} = lc_ray_query_all({1}, lc_bit_cast<Ray>({2}), {3});auto& {0} = _{0};",
                    var, args_v[0], args_v[1], args_v[2]
                )
                    .unwrap();
                true
            }
            FuncTag::RayTracingQueryAny => {
                writeln!(
                    self.body,
                    "LC_RayQueryAny _{0} = lc_ray_query_any({1}, lc_bit_cast<Ray>({2}), {3});auto& {0} = _{0};",
                    var, args_v[0], args_v[1], args_v[2]
                )
                    .unwrap();
                true
            }
            FuncTag::RayQueryWorldSpaceRay => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bit_cast<{0}>(lc_ray_query_world_space_ray({2}));",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            FuncTag::RayQueryProceduralCandidateHit => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bit_cast<{0}>(lc_ray_query_procedural_candidate_hit({2}));",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            FuncTag::RayQueryTriangleCandidateHit => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bit_cast<{0}>(lc_ray_query_triangle_candidate_hit({2}));",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            FuncTag::RayQueryCommittedHit => {
                writeln!(
                    self.body,
                    "const {0} {1} = lc_bit_cast<{0}>(lc_ray_query_committed_hit({2}));",
                    node_ty_s, var, args_v[0]
                )
                .unwrap();
                true
            }
            FuncTag::RayQueryCommitTriangle => {
                writeln!(self.body, "lc_ray_query_commit_triangle({0});", args_v[0]).unwrap();
                true
            }
            FuncTag::RayQueryCommitProcedural => {
                writeln!(
                    self.body,
                    "lc_ray_query_commit_procedural({0}, {1});",
                    args_v[0], args_v[1]
                )
                .unwrap();
                true
            }
            FuncTag::RayQueryTerminate => {
                writeln!(self.body, "lc_ray_query_terminate({0});", args_v[0]).unwrap();
                true
            }
            _ => false,
        }
    }
    fn gen_const(&mut self, var: &String, node_ty_s: &String, cst: ConstantInstRef) {
        let bytes = cst.data();

        writeln!(
            &mut self.body,
            "const {0} {1} = {2};",
            node_ty_s,
            var,
            decode_const_data_v2(bytes.as_ref(), cst.ty())
        )
        .unwrap();
    }
    fn gen_instr(&mut self, node: NodeRef) {
        if self.visited.contains(&node) {
            return;
        }
        self.visited.insert(node);
        let inst = node.inst();
        let node_ty = node.type_();
        let node_ty_s = self.type_gen.gen_c_type(node_ty);
        match inst.tag() {
            InstructionTag::BasicBlockSentinel => unreachable!(),
            InstructionTag::Buffer => {}
            InstructionTag::BindlessArray => {}
            InstructionTag::Texture2d => {}
            InstructionTag::Texture3d => {}
            InstructionTag::Accel => {}
            InstructionTag::Shared => todo!(),
            InstructionTag::Uniform => todo!(),
            InstructionTag::Local => {
                self.write_ident();
                let local = inst.as_local();
                let var = self.gen_node(node);
                let init_v = self.gen_node(local.init());
                writeln!(
                    &mut self.body,
                    "{0} _{1} = {2};{0} * {1} = &_{1};",
                    node_ty_s, var, init_v
                )
                .unwrap();
            }
            InstructionTag::Argument => todo!(),
            InstructionTag::Constant => {
                let cst = inst.as_constant();
                self.write_ident();
                let var = self.gen_node(node);
                self.gen_const(&var, &node_ty_s, cst);
            }
            InstructionTag::Update => {
                let update = inst.as_update();
                self.write_ident();
                let value_v = self.gen_node(update.value());
                let var_v = self.gen_node(update.var());
                writeln!(&mut self.body, "*{} = {};", var_v, value_v).unwrap();
            }
            InstructionTag::Call => {
                let call = inst.as_call();
                let f = call.func();
                let args = call.args();
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
            InstructionTag::Phi => {
                self.write_ident();
                let var = self.gen_node(node);
                writeln!(&mut self.fwd_defs, "    {0} {1} = {0}{{}};", node_ty_s, var).unwrap();
            }
            InstructionTag::Return => {
                let r = inst.as_return();
                let v = r.value();
                self.write_ident();
                if !v.is_null() {
                    let v_v = self.gen_node(v);
                    writeln!(&mut self.body, "return {};", v_v).unwrap();
                } else {
                    writeln!(&mut self.body, "return;").unwrap();
                }
            }
            InstructionTag::GenericLoop => {
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
                let generic_loop = inst.as_generic_loop();
                let prepare = generic_loop.prepare();
                let cond = generic_loop.cond();
                let update = generic_loop.update();
                let body = generic_loop.body();
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
                    self.gen_block_(prepare);
                    let cond_v = self.gen_node(cond);
                    self.write_ident();
                    writeln!(&mut self.body, "if (!{}) break;", cond_v).unwrap();
                    self.write_ident();
                    self.indent -= 1;
                    writeln!(&mut self.body, "}}").unwrap();
                    self.write_ident();
                    writeln!(&mut self.body, "do").unwrap();
                    self.write_ident();
                    self.gen_block(body);
                    self.write_ident();
                    writeln!(&mut self.body, "while(false);").unwrap();
                    self.write_ident();
                    writeln!(&mut self.body, "if (loop_break) break;").unwrap();
                    self.gen_block(update);
                    self.write_ident();
                    writeln!(&mut self.body, "}}").unwrap();
                }
                self.inside_generic_loop = old_inside_generic_loop;
            }
            InstructionTag::Break => {
                self.write_ident();
                if self.inside_generic_loop {
                    writeln!(&mut self.body, "loop_break = true;").unwrap();
                }
                writeln!(&mut self.body, "break;").unwrap();
            }
            InstructionTag::Continue => {
                self.write_ident();
                if self.inside_generic_loop {
                    writeln!(&mut self.body, "break;").unwrap();
                } else {
                    writeln!(&mut self.body, "continue;").unwrap();
                }
            }
            InstructionTag::If => {
                let if_ = inst.as_if();
                let cond = if_.cond();
                let true_branch = if_.true_branch();
                let false_branch = if_.false_branch();
                self.write_ident();
                let cond_v = self.gen_node(cond);
                writeln!(&mut self.body, "if ({})", cond_v).unwrap();
                self.gen_block(true_branch);
                self.write_ident();
                writeln!(&mut self.body, "else").unwrap();
                self.gen_block(false_branch);
            }
            InstructionTag::Switch => {
                let sw = inst.as_switch();
                let value = sw.value();
                let cases = sw.cases();
                let default = sw.default();
                self.write_ident();
                let value_v = self.gen_node(value);
                writeln!(&mut self.body, "switch ({}) {{", value_v).unwrap();
                for SwitchCase { value, block } in cases {
                    self.write_ident();
                    writeln!(&mut self.body, "case {}:", *value).unwrap();
                    self.gen_block(*block);
                    self.write_ident();
                    writeln!(&mut self.body, "break;").unwrap();
                }
                self.write_ident();
                writeln!(&mut self.body, "default:").unwrap();
                self.gen_block(default);
                self.write_ident();
                writeln!(&mut self.body, "break;").unwrap();
                self.write_ident();
                writeln!(&mut self.body, "}}").unwrap();
            }
            InstructionTag::FwdAutodiff => {
                let fwd_autodiff = inst.as_fwd_autodiff();
                let body = fwd_autodiff.body();
                writeln!(&mut self.body, "/* Forward AdScope */").unwrap();
                self.gen_block(body);
                self.write_ident();
                writeln!(&mut self.body, "/* Forward AdScope End */").unwrap();
            }
            InstructionTag::RevAutodiff => {
                let fwd_autodiff = inst.as_fwd_autodiff();
                let body = fwd_autodiff.body();
                writeln!(&mut self.body, "/* Forward AdScope */").unwrap();
                self.gen_block(body);
                self.write_ident();
                writeln!(&mut self.body, "/* Forward AdScope End */").unwrap();
            }
            InstructionTag::RayQuery => {
                let rq = inst.as_ray_query();
                let ray_query = rq.ray_query();
                let on_triangle_hit = rq.on_triangle_hit();
                let on_procedural_hit = rq.on_procedural_hit();
                writeln!(&mut self.body, "/* RayQuery */").unwrap();
                let rq_v = self.gen_node(ray_query);
                self.write_ident();
                writeln!(
                    &mut self.body,
                    "lc_ray_query({}, [&](const TriangleHit& hit) {{",
                    rq_v
                )
                .unwrap();
                self.gen_block(on_triangle_hit);
                self.write_ident();
                writeln!(&mut self.body, "}}, [&](const ProceduralHit& hit) {{").unwrap();
                self.gen_block(on_procedural_hit);
                self.write_ident();
                writeln!(&mut self.body, "}});").unwrap();
                self.write_ident();
                writeln!(&mut self.body, "/* RayQuery End*/").unwrap();
            }
            InstructionTag::Comment => {
                self.write_ident();
                writeln!(&mut self.body, "/* {} */", inst.as_comment().comment()).unwrap();
            }
            InstructionTag::Print => {
                todo!()
                // self.print_impl(fmt.to_string(), args.as_ref());
            }
        }
    }
    fn gen_block_(&mut self, block: BasicBlockRef) {
        for n in block.iter() {
            self.gen_instr(n);
        }
        let phis = self.phis_per_block.get(&block).cloned().unwrap_or(vec![]);
        for phi in &phis {
            let phi_v = self.gen_node(*phi);
            let phi = phi.inst().as_phi();
            let value = phi
                .incomings()
                .iter()
                .find(|incoming| incoming.block == block)
                .unwrap()
                .value;
            let value = self.gen_node(value);
            self.write_ident();
            writeln!(&mut self.body, "{} = {};", phi_v, value).unwrap();
        }
    }
    fn gen_block(&mut self, block: BasicBlockRef) {
        self.write_ident();
        writeln!(&mut self.body, "{{").unwrap();
        self.indent += 1;
        self.gen_block_(block);
        self.indent -= 1;
        self.write_ident();
        writeln!(&mut self.body, "}}").unwrap();
    }
    fn gen_callable(
        &mut self,
        var: &String,
        node_ty_s: &String,
        f: FuncRef,
        args_v: &Vec<String>,
    ) -> bool {
        // let f =f.as_callable();
        // let fid = CArc::as_ptr(&f.0) as u64;
        // if !self.globals.generated_callables.contains_key(&fid) {
        //     let mut callable_emitter = FunctionEmitter::new(&mut self.globals, &self.type_gen);
        //     callable_emitter.indent = 2;
        //     let mut params = vec![];
        //     for (i, arg) in f.0.args.iter().enumerate() {
        //         let mut param = String::new();
        //         let var = format!("ca_{}", i);
        //         match arg.inst().tag(){
        //             InstructionTag::Accel => {
        //                 write!(&mut param, "const Accel& {}", var).unwrap();
        //             }
        //             InstructionTag::BindlessArray => {
        //                 write!(&mut param, "const BindlessArray& {}", var).unwrap();
        //             }
        //             InstructionTag::Buffer => {
        //                 write!(&mut param, "const BufferView& {}", var).unwrap();
        //             }
        //             InstructionTag::Texture2d => {
        //                 write!(&mut param, "const Texture2D& {}", var).unwrap();
        //             }
        //             InstructionTag::Texture3d => {
        //                 write!(&mut param, "const Texture3D& {}", var).unwrap();
        //             }
        //             InstructionTag::Argument => {
        //                 let by_value = arg.inst
        //                 let ty = self.type_gen.gen_c_type(arg.type_());
        //                 if *by_value {
        //                     write!(&mut param, "const {}& {}", ty, var).unwrap();
        //                 } else {
        //                     write!(&mut param, "{}* {}", ty, var).unwrap();
        //                 }
        //             }
        //             _ => {
        //                 unreachable!("{:?}", arg.get().instruction.as_ref());
        //             }
        //         }
        //         callable_emitter
        //             .node_to_var
        //             .insert(arg.clone(), var.clone());
        //         params.push(param);
        //     }
        //     callable_emitter.gen_callable_module(&f.0);
        //     callable_emitter.indent += self.indent;

        //     let ret_type = self.type_gen.gen_c_type(&f.0.ret_type);

        //     let fname = format!(
        //         "callable_{}",
        //         callable_emitter.globals.generated_callables.len()
        //     );
        //     let source = format!(
        //         "[=]({}) -> {} {{\n{}\n{}    }}",
        //         params.join(","),
        //         ret_type,
        //         callable_emitter.fwd_defs,
        //         callable_emitter.body
        //     );
        //     if let Some(fname) = self.globals.generated_callable_sources.get(&source) {
        //         self.globals.generated_callables.insert(fid, fname.clone());
        //     } else {
        //         self.globals.generated_callables.insert(fid, fname.clone());
        //         self.globals
        //             .generated_callable_sources
        //             .insert(source.clone(), fname.clone());
        //         writeln!(
        //             &mut self.globals.callable_def,
        //             "    const auto {} = {};\n",
        //             fname, source
        //         )
        //         .unwrap();
        //     }
        // }
        // let fname = &self.globals.generated_callables[&fid];
        // if var != "" {
        //     writeln!(
        //         &mut self.body,
        //         "const {} {} = {}({});",
        //         node_ty_s,
        //         var,
        //         fname,
        //         args_v.join(", ")
        //     )
        //     .unwrap();
        // } else {
        //     writeln!(&mut self.body, "{}({});", fname, args_v.join(", ")).unwrap();
        // }
        // true
        todo!()
    }
}
