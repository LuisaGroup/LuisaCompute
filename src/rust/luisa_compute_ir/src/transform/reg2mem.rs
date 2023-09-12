/*
 * This file implements the Reg2Mem pass, which demotes SSA phi-nodes to local variables.
 * This resembles LLVM's Reg2Mem pass, roughly done in the following steps:
 *   1. Traverse the CFG and collect the phi-nodes to construct a phi-to-blocks map.
 *   2. For each phi-node, create a local variable and insert a store instruction at the end of each incoming block.
 *   3. Replace each phi-node with a load instruction so that all references to it are replaced with the new local variable.
 *   4. Move all local variables to the beginning of the function to make the following passes (e.g., lower control flow) easier.
 */

use crate::ir::{
    new_node, BasicBlock, Const, Func, Instruction, IrBuilder, Module, ModulePools, Node, NodeRef,
};
use crate::transform::Transform;
use crate::{CArc, CBoxedSlice, Pooled};
use std::collections::HashSet;

struct Reg2MemCtx {
    pools: CArc<ModulePools>,
    entry: Pooled<BasicBlock>,
    locals: Vec<NodeRef>,
    phis: Vec<NodeRef>,
}

struct Reg2MemImpl {
    ctx: Option<Reg2MemCtx>,
    transformed: HashSet<*const BasicBlock>,
}

impl Reg2MemImpl {
    fn new() -> Self {
        Self {
            ctx: None,
            transformed: HashSet::new(),
        }
    }

    fn transform_recursive(&mut self, block: &Pooled<BasicBlock>) {
        for node_ref in block.iter() {
            let node = node_ref.get();
            match node.instruction.as_ref() {
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
                Instruction::Call(func, _) => match func {
                    Func::Callable(callable) => {
                        self.transform_module(&callable.0.module);
                    }
                    _ => {}
                },
                Instruction::Phi(_) => {}
                Instruction::Return(_) => {}
                Instruction::Loop { body, cond: _ } => {
                    self.transform_recursive(body);
                }
                Instruction::GenericLoop {
                    prepare,
                    body,
                    update,
                    cond: _,
                } => {
                    self.transform_recursive(prepare);
                    self.transform_recursive(body);
                    self.transform_recursive(update);
                }
                Instruction::Break => {}
                Instruction::Continue => {}
                Instruction::If {
                    cond: _,
                    true_branch,
                    false_branch,
                } => {
                    self.transform_recursive(true_branch);
                    self.transform_recursive(false_branch);
                }
                Instruction::Switch {
                    value: _,
                    cases,
                    default,
                } => {
                    for case in cases.iter() {
                        self.transform_recursive(&case.block);
                    }
                    self.transform_recursive(default);
                }
                Instruction::AdScope { body, .. } => {
                    self.transform_recursive(body);
                }
                Instruction::RayQuery {
                    ray_query: _,
                    on_triangle_hit,
                    on_procedural_hit,
                } => {
                    self.transform_recursive(on_triangle_hit);
                    self.transform_recursive(on_procedural_hit);
                }
                Instruction::AdDetach(body) => {
                    self.transform_recursive(body);
                }
                Instruction::Comment(_) => {}
                Instruction::Print { .. } => {todo!()}
            }
        }
    }

    fn collect_phi_and_local_nodes(&mut self, block: &Pooled<BasicBlock>) {
        for node in block.iter() {
            match node.get().instruction.as_ref() {
                Instruction::Buffer => {}
                Instruction::Bindless => {}
                Instruction::Texture2D => {}
                Instruction::Texture3D => {}
                Instruction::Accel => {}
                Instruction::Shared => {}
                Instruction::Uniform => {}
                Instruction::Local { .. } => {
                    self.ctx.as_mut().unwrap().locals.push(node.clone());
                }
                Instruction::Argument { .. } => {}
                Instruction::UserData(_) => {}
                Instruction::Invalid => {}
                Instruction::Const(_) => {}
                Instruction::Update { .. } => {}
                Instruction::Call(_, _) => {}
                Instruction::Phi(_) => {
                    self.ctx.as_mut().unwrap().phis.push(node.clone());
                }
                Instruction::Return(_) => {}
                Instruction::Loop { body, cond: _ } => {
                    self.collect_phi_and_local_nodes(body);
                }
                Instruction::GenericLoop {
                    prepare,
                    body,
                    update,
                    cond: _,
                } => {
                    self.collect_phi_and_local_nodes(prepare);
                    self.collect_phi_and_local_nodes(body);
                    self.collect_phi_and_local_nodes(update);
                }
                Instruction::Break => {}
                Instruction::Continue => {}
                Instruction::If {
                    cond: _,
                    true_branch,
                    false_branch,
                } => {
                    self.collect_phi_and_local_nodes(true_branch);
                    self.collect_phi_and_local_nodes(false_branch);
                }
                Instruction::Switch {
                    value: _,
                    cases,
                    default,
                } => {
                    for case in cases.iter() {
                        self.collect_phi_and_local_nodes(&case.block);
                    }
                    self.collect_phi_and_local_nodes(default);
                }
                Instruction::AdScope { body, .. } => {
                    self.collect_phi_and_local_nodes(body);
                }
                Instruction::RayQuery {
                    ray_query: _,
                    on_triangle_hit,
                    on_procedural_hit,
                } => {
                    self.collect_phi_and_local_nodes(on_triangle_hit);
                    self.collect_phi_and_local_nodes(on_procedural_hit);
                }
                Instruction::AdDetach(body) => {
                    self.collect_phi_and_local_nodes(body);
                }
                Instruction::Comment(_) => {}
                Instruction::Print { .. } => {
                    todo!()
                }
            }
        }
    }

    fn hoist_locals(&self) {
        let pools = &self.ctx.as_ref().unwrap().pools;
        let mut builder = IrBuilder::new(pools.clone());
        builder.set_insert_point(self.ctx.as_ref().unwrap().entry.first.clone());
        // process the local variables
        for node_ref in self.ctx.as_ref().unwrap().locals.iter() {
            // move the local variable to the beginning of the function,
            // and replace initializations with assignments
            let node = node_ref.get();
            match node.instruction.as_ref() {
                Instruction::Local { init } => {
                    let is_zero_init = match init.get().instruction.as_ref() {
                        Instruction::Const(Const::Zero(_)) => true,
                        _ => false,
                    };
                    if !is_zero_init {
                        // backup the current insert point
                        let decl_point = builder.insert_point.clone();
                        // construct an assignment
                        builder.set_insert_point(node_ref.clone());
                        builder.update_unchecked(node_ref.clone(), init.clone());
                        // restore the insert point
                        builder.set_insert_point(decl_point);
                    }
                    // replace with a zero-initialized local variable
                    let zero = new_node(
                        &pools,
                        Node::new(
                            CArc::new(Instruction::Const(Const::Zero(node.type_.clone()))),
                            node.type_.clone(),
                        ),
                    );
                    let local = Node::new(
                        CArc::new(Instruction::Local { init: zero }),
                        node.type_.clone(),
                    );
                    node_ref.replace_with(&local);
                    // move to the beginning of the function
                    if builder.insert_point != node_ref.clone() {
                        node_ref.remove();
                        builder.append(node_ref.clone());
                    }
                }
                _ => unreachable!(),
            }
        }
        // process the phi-nodes
        for node_ref in self.ctx.as_ref().unwrap().phis.iter() {
            let node = node_ref.get();
            match node.instruction.as_ref() {
                Instruction::Phi(incomings) => {
                    // create a local variable
                    let local = builder.local_zero_init(node.type_.clone());
                    // backup the current insert point
                    let decl_point = builder.insert_point.clone();
                    // insert the store instructions to the end of each incoming block
                    for incoming in incomings.iter() {
                        builder.set_insert_point(incoming.block.get().last.get().prev);
                        builder.update(local.clone(), incoming.value.clone());
                    }
                    // restore the insert point
                    builder.set_insert_point(decl_point);
                    // replace the phi node with a load instruction
                    let load = Node::new(
                        CArc::new(Instruction::Call(Func::Load, CBoxedSlice::new(vec![local]))),
                        node.type_.clone(),
                    );
                    node_ref.replace_with(&load);
                }
                _ => unreachable!(),
            }
        }
    }

    fn transform_module(&mut self, module: &Module) {
        if self.transformed.insert(module.entry.as_ptr()) {
            let ctx = Reg2MemCtx {
                pools: module.pools.clone(),
                entry: module.entry.clone(),
                locals: Vec::new(),
                phis: Vec::new(),
            };
            self.ctx.replace(ctx);
            self.collect_phi_and_local_nodes(&module.entry);
            self.hoist_locals();
            self.ctx = None;
            // recursively transform the callees
            self.transform_recursive(&module.entry);
        }
    }
}

pub struct Reg2Mem;

impl Transform for Reg2Mem {
    fn transform(&self, module: Module) -> Module {
        let mut reg2mem = Reg2MemImpl::new();
        reg2mem.transform_module(&module);
        module
    }
}
