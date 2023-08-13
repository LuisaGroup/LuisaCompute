use std::collections::HashMap;
use std::mem::replace;
use std::ptr::null;
use crate::{CArc, CBox, CBoxedSlice, Pooled};
use crate::context::register_type;
use crate::ir::{BasicBlock, CallableModule, CallableModuleRef, Const, Func, Instruction, INVALID_REF, IrBuilder, KernelModule, Module, ModulePools, new_node, Node, NodeRef, StructType, Type};
use crate::transform::Transform;

pub struct Ref2Ret;

struct Ref2RetCtx {
    pools: CArc<ModulePools>,
    ret_type: CArc<Type>,
    ref_args: Vec<NodeRef>,
    ref_arg_indices: Vec<usize>,
}

struct Ref2RetMetadata {
    module: CArc<CallableModule>,
    ref_arg_indices: Option<Vec<usize>>,
}

struct Ref2RetImpl {
    processed: HashMap<*const CallableModule, Ref2RetMetadata>,
    current: Option<Ref2RetCtx>,
}

impl Ref2RetImpl {
    fn new() -> Self {
        Self {
            processed: HashMap::new(),
            current: None,
        }
    }

    fn is_transform_needed(module: &CallableModule) -> bool {
        for arg in module.args.iter() {
            match arg.get().instruction.as_ref() {
                Instruction::Argument { by_value, .. } => {
                    if !by_value {
                        return true;
                    }
                }
                _ => {}
            }
        }
        return false;
    }

    fn prepare_context(module: &mut CallableModule) -> Ref2RetCtx {
        let mut ret_types = Vec::new();
        let mut ref_args = Vec::new();
        let mut ref_arg_indices = Vec::new();
        // collect reference arguments
        for (i, node) in module.args.iter().enumerate() {
            match node.get().instruction.get_mut().unwrap() {
                Instruction::Argument { by_value, .. } => {
                    if !*by_value {
                        *by_value = true;
                        ref_args.push(node.clone());
                        ref_arg_indices.push(i);
                        ret_types.push(node.get().type_.clone());
                    }
                }
                _ => {}
            }
        }
        // add the original return type if any
        if !module.ret_type.is_void() {
            ret_types.push(module.ret_type.clone());
        }
        // create the return type
        let mut ret_size = 0usize;
        let mut ret_align = 0usize;
        for t in ret_types.iter() {
            let size = t.size();
            let align = t.alignment();
            ret_align = ret_align.max(align);
            ret_size = (ret_size + align - 1) / align * align + size;
        }
        ret_size = (ret_size + ret_align - 1) / ret_align * ret_align;
        let ret_struct = register_type(Type::Struct(StructType {
            fields: CBoxedSlice::from(ret_types.as_slice()),
            size: ret_size,
            alignment: ret_align,
        }));
        // update the return type of the module
        module.ret_type = ret_struct.clone();

        Ref2RetCtx {
            pools: module.pools.clone(),
            ret_type: ret_struct,
            ref_args,
            ref_arg_indices,
        }
    }

    fn transform_callable(&mut self, module: CArc<CallableModule>) {
        if self.processed.contains_key(&module.as_ptr()) {
            return;
        }
        if Self::is_transform_needed(module.as_ref()) {
            let copy = module.clone();// FIXME: clone the module
            let ctx = Self::prepare_context(copy.get_mut().unwrap());
            let old_ctx = replace(&mut self.current, Some(ctx));
            self.transform_block(&copy.module.entry);
            let ctx = replace(&mut self.current, old_ctx).unwrap();
            self.processed.insert(module.as_ptr(), Ref2RetMetadata {
                module: copy,
                ref_arg_indices: Some(ctx.ref_arg_indices),
            });
        } else {
            let old_ctx = replace(&mut self.current, None);
            self.transform_block(&module.module.entry);
            self.current = old_ctx;
            self.processed.insert(module.as_ptr(), Ref2RetMetadata {
                module,
                ref_arg_indices: None,
            });
        }
    }

    fn transform_block(&mut self, bb: &Pooled<BasicBlock>) {
        for node in bb.nodes() {
            match node.get().instruction.as_ref() {
                Instruction::Buffer => {}
                Instruction::Bindless => {}
                Instruction::Texture2D => {}
                Instruction::Texture3D => {}
                Instruction::Accel => {}
                Instruction::Shared => {}
                Instruction::Uniform => {}
                Instruction::Local { .. } => {}
                Instruction::Argument { .. } => {}
                Instruction::UserData(_) => {}
                Instruction::Invalid => {}
                Instruction::Const(_) => {}
                Instruction::Update { .. } => {}
                Instruction::Call(_, _) => self.transform_call(node),
                Instruction::Phi(_) => {}
                Instruction::Return(_) => self.transform_return(node),
                Instruction::Loop { body, .. } => {
                    self.transform_block(body)
                }
                Instruction::GenericLoop { prepare, body, update, .. } => {
                    self.transform_block(prepare);
                    self.transform_block(body);
                    self.transform_block(update);
                }
                Instruction::Break => {}
                Instruction::Continue => {}
                Instruction::If { true_branch, false_branch, .. } => {
                    self.transform_block(true_branch);
                    self.transform_block(false_branch);
                }
                Instruction::Switch { cases, default, .. } => {
                    for case in cases.as_ref() {
                        self.transform_block(&case.block);
                    }
                    self.transform_block(default);
                }
                Instruction::AdScope { body } => {
                    self.transform_block(body)
                }
                Instruction::RayQuery { on_triangle_hit, on_procedural_hit, .. } => {
                    self.transform_block(on_triangle_hit);
                    self.transform_block(on_procedural_hit);
                }
                Instruction::AdDetach(_) => {}
                Instruction::Comment(_) => {}
            }
        }
    }

    fn transform_call(&mut self, node: NodeRef) {
        match node.get().instruction.as_ref() {
            Instruction::Call(func, args) => {
                match func {
                    Func::Callable(callable) => {
                        let callable = &callable.0;
                        self.transform_callable(callable.clone());
                        let metadata = self.processed.get(&callable.as_ptr()).unwrap();
                        // the ref args are now passed by value and we need to copy the return value to the ref args
                        if let Some(ref_args) = metadata.ref_arg_indices.as_ref() {
                            // insert before the call node
                            let mut builder = IrBuilder::new(metadata.module.pools.clone());
                            builder.set_insert_point(node.get().prev);
                            // get the packed return values
                            let transformed_callable = &metadata.module;
                            let transformed_ret_type = &transformed_callable.ret_type;
                            let packed = builder.call(
                                Func::Callable(CallableModuleRef(transformed_callable.clone())),// the transformed callable
                                args.to_vec().as_slice(),
                                transformed_ret_type.clone());
                            // unpack the return values
                            let ret_struct = match transformed_ret_type.as_ref() {
                                Type::Struct(s) => s,
                                _ => unreachable!(),
                            };
                            for i in ref_args.iter() {
                                let type_ = ret_struct.fields.get(*i).unwrap().clone();
                                let index = builder.const_(Const::Uint32(*i as u32));
                                let value = builder.call(
                                    Func::ExtractElement, &[packed.clone(), index], type_);
                                builder.update(args[*i].clone(), value);
                            }
                            // update the original return value if any
                            if node.type_().is_void() {
                                assert_eq!(ret_struct.size, ref_args.len());
                                node.remove();// remove the original call node if no return
                            } else {
                                // extract the old return value
                                assert_eq!(ret_struct.size, ref_args.len() + 1);
                                let index = builder.const_(Const::Uint32(ref_args.len() as u32));
                                let type_ = ret_struct.fields.last().unwrap().clone();
                                let args = CBoxedSlice::new(vec![packed.clone(), index]);
                                let instr = Instruction::Call(Func::ExtractElement, args);
                                let ret = Node::new(CArc::new(instr), type_);
                                // update the old return value
                                node.replace_with(&ret);
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => unreachable!(),
        }
    }

    fn transform_return(&mut self, node: NodeRef) {
        if let Some(ctx) = &self.current {
            match node.get().instruction.as_ref() {
                Instruction::Return(value) => {
                    // insert a new return node that packs all ref args
                    let mut builder = IrBuilder::new(ctx.pools.clone());
                    builder.set_insert_point(node.get().prev);
                    let mut args = ctx.ref_args.clone();
                    // if the node returns a valid value, add it to the return args
                    if value.valid() {
                        args.push(value.clone());
                    }
                    let value = builder.call(
                        Func::Struct,
                        args.as_slice(),
                        ctx.ret_type.clone());
                    builder.return_(value);
                    // remove the original return node
                    node.remove();
                }
                _ => unreachable!(),
            }
        }
    }
}

impl Transform for Ref2Ret {
    fn transform(&self, module: Module) -> Module {
        let mut transform = Ref2RetImpl::new();
        transform.transform_block(&module.entry);
        module
    }
}
