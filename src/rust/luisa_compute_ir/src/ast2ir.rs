use crate::context::register_type;
use crate::ir::*;
use crate::{CArc, CBoxedSlice, Pooled, TypeOf};
use base64ct::{Base64, Encoding};
use half::f16;
use json::{iterators, parse as parse_json, JsonValue as JSON, Result};
use std::collections::HashMap;

struct AST2IRCtx<'a> {
    j: &'a JSON,
    j_tag: &'a str,
    j_variables: &'a JSON,
    builder: Option<IrBuilder>,
    arguments: HashMap<u32, NodeRef>,
    variables: HashMap<u32, NodeRef>,
    inside_generic_loop: bool,
}

struct AST2IR<'a: 'b, 'b> {
    j: &'a JSON,
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

impl<'a, 'b> AST2IR<'a, 'b> {
    fn convert_type(&mut self, i: usize) -> CArc<Type> {
        if let Some(t) = self.types.get(&i) {
            return t.clone();
        }
        let j = &self.j_types[i];
        let tag = j["tag"].as_str().unwrap();
        let t = match tag {
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
            "BUFFER" => panic!("Buffer's are not treated as types in IR."),
            "TEXTURE" => panic!("Texture's are not treated as types in IR."),
            "BINDLESS_ARRAY" => panic!("BindlessArray's are not treated as types in IR."),
            "ACCEL" => panic!("Accel's are not treated as types in IR."),
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
                        new_node(
                            &self.pools,
                            Node::new(CArc::new(Instruction::Shared), t.clone()),
                        )
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
            });
    }

    fn _do_convert_kernel(&mut self) -> KernelModule {
        todo!()
    }

    fn _do_convert_callable(&mut self) -> CallableModule {
        todo!()
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
            inside_generic_loop: false,
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

    fn convert(j: JSON) -> CArc<KernelModule> {
        let mut ast2ir = AST2IR {
            j: &j,
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
        let kernels: Vec<_> = ast2ir
            .functions
            .iter()
            .filter_map(|(_, f)| {
                if let FunctionModule::Kernel(k) = f {
                    Some(k)
                } else {
                    None
                }
            })
            .collect();
        // TODO: support converting multiple kernels
        assert_eq!(kernels.len(), 1, "There should be only one kernel function");
        kernels[0].clone()
    }
}

pub fn convert_ast_to_ir_kernel(data: String) -> CArc<KernelModule> {
    let j: JSON = parse_json(data.as_str()).unwrap();
    AST2IR::convert(j)
}
