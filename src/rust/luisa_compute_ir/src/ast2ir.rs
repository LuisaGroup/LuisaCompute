use crate::context::register_type;
use crate::ir::*;
use crate::{CArc, CBox, CBoxedSlice, Pooled, TypeOf};
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
    shared: Vec<NodeRef>,
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

    fn _convert_expression(&mut self, j: &JSON, is_lval: bool) -> NodeRef {
        if j.is_null() {
            INVALID_REF
        } else {
            todo!()
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
                let (builder, ..) = self.unwrap_ctx();
                builder.return_(v)
            }
            "SCOPE" => unreachable!("Scope should be handled by _convert_scope."),
            "IF" => {
                let cond = self._convert_expression(&j["condition"], false);
                let true_ = self._convert_scope(&j["true_branch"], false);
                let false_ = self._convert_scope(&j["false_branch"], false);
                let (builder, ..) = self.unwrap_ctx();
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
                builder.update(lhs, rhs)
            }
            "FOR" => {
                let mut cond = INVALID_REF;
                let prepare = self._with_builder(|this| {
                    cond = this._convert_expression(&j["condition"], false);
                    let ty_cond = cond.type_();
                    if !ty_cond.is_bool() {
                        let (builder, ..) = this.unwrap_ctx();
                        cond = builder.cast(cond, <bool as TypeOf>::type_());
                    }
                });
                let body = self._convert_scope(&j["body"], false);
                let update = self._with_builder(|this| {
                    let var = this._convert_expression(&j["variable"], true);
                    let step = this._convert_expression(&j["step"], false);
                    let (builder, ..) = this.unwrap_ctx();
                    let step = if step.type_().as_ref() != var.type_().as_ref() {
                        builder.cast(step, var.type_().clone())
                    } else {
                        step
                    };
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
        self.j["body"]["statements"].members().for_each(|s| {
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
        }
    }

    fn _do_convert_kernel(&mut self) -> KernelModule {
        let module = self._convert_module(ModuleKind::Kernel);
        let args: Vec<_> = self._curr_ctx().j["arguments"]
            .members()
            .map(|a| {
                let a = a.as_usize().unwrap();
                self._curr_ctx().arguments.get(&(a as u32)).unwrap().clone()
            })
            .collect();
        let captures: Vec<_> = self._curr_ctx().j["bound_arguments"]
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
        let shared = self._curr_ctx().shared.clone();
        let block_size = &self.j["block_size"];
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
