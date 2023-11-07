use std::collections::{HashMap, HashSet};

use crate::ir::{Func, Module, NodeRef};

pub struct UseDef {
    pub uses: HashMap<NodeRef, HashSet<NodeRef>>,
    pub root: HashSet<NodeRef>,
}

impl UseDef {
    pub fn compute(module: &Module) -> Self {
        let mut init = Self {
            uses: HashMap::new(),
            root: HashSet::new(),
        };
        init.run(module);
        init
    }
    pub fn reachable(&self, node: NodeRef) -> bool {
        if self.root.contains(&node) {
            return true;
        }
        let mut visited = HashSet::new();
        let mut stack = vec![node];
        while let Some(node) = stack.pop() {
            if visited.contains(&node) {
                continue;
            }
            visited.insert(node);
            if self.root.contains(&node) {
                return true;
            }
            if let Some(uses) = self.uses.get(&node) {
                for use_ in uses {
                    stack.push(*use_);
                }
            }
        }
        false
    }
    fn run(&mut self, module: &Module) {
        let nodes = module.collect_nodes();
        // init uses
        for node in &nodes {
            self.uses.insert(*node, HashSet::new());
        }

        for node in &nodes {
            macro_rules! uses {
                ($v:expr) => {
                    assert!($v != node);
                    self.uses.entry(*$v).or_insert(HashSet::new()).insert(*node);
                };
            }
            let inst = node.get().instruction.as_ref();
            match inst {
                crate::ir::Instruction::Buffer => {}
                crate::ir::Instruction::Bindless => {}
                crate::ir::Instruction::Texture2D => {}
                crate::ir::Instruction::Texture3D => {}
                crate::ir::Instruction::Accel => {}
                crate::ir::Instruction::Shared => {}
                crate::ir::Instruction::Uniform => {}
                crate::ir::Instruction::Local { init } => {
                    uses!(init);
                }
                crate::ir::Instruction::Argument { .. } => {}
                crate::ir::Instruction::UserData(_) => {}
                crate::ir::Instruction::Invalid => {}
                crate::ir::Instruction::Const(_) => {}
                crate::ir::Instruction::Update { var, value } => {
                    let dst = var.access_chain().0;
                    self.uses.entry(*var).or_insert(HashSet::new()).insert(dst);

                    self.uses
                        .entry(*value)
                        .or_insert(HashSet::new())
                        .insert(dst);
                    
                    self.uses.entry(*node).or_insert(HashSet::new()).insert(dst);

                    self.uses
                        .entry(*value)
                        .or_insert(HashSet::new())
                        .insert(*node);
                }
                crate::ir::Instruction::Call(f, args) => {
                    for arg in args.iter() {
                        uses!(arg);
                    }
                    match f {
                        Func::AtomicExchange
                        | Func::AtomicCompareExchange
                        | Func::AtomicFetchAdd
                        | Func::AtomicFetchSub
                        | Func::AtomicFetchAnd
                        | Func::AtomicFetchOr
                        | Func::AtomicFetchXor
                        | Func::AtomicFetchMin
                        | Func::AtomicFetchMax
                        | Func::SynchronizeBlock
                        | Func::BindlessBufferWrite
                        | Func::BufferWrite
                        | Func::ByteBufferWrite
                        | Func::Texture2dWrite
                        | Func::Texture3dWrite
                        | Func::RayQueryCommitProcedural
                        | Func::RayQueryCommitTriangle
                        | Func::RayQueryCommittedHit
                        | Func::Callable(_)
                        | Func::CpuCustomOp(_)
                        | Func::Assert(_)
                        | Func::Unreachable(_)
                        | Func::AccGrad
                        | Func::GradientMarker
                        | Func::Backward
                        | Func::PropagateGrad
                        | Func::RequiresGradient => {
                            self.root.insert(*node);
                        }
                        _ => {}
                    }
                }
                crate::ir::Instruction::Phi(incomings) => {
                    for incoming in incomings.iter() {
                        uses!(&incoming.value);
                    }
                }
                crate::ir::Instruction::Return(v) => {
                    self.root.insert(*node);
                    uses!(v);
                }
                crate::ir::Instruction::Loop { body: _, cond } => {
                    self.root.insert(*node);
                    self.root.insert(*cond);
                }
                crate::ir::Instruction::GenericLoop { cond, .. } => {
                    self.root.insert(*cond);
                    self.root.insert(*node);
                }
                crate::ir::Instruction::Break => {
                    self.root.insert(*node);
                }
                crate::ir::Instruction::Continue => {
                    self.root.insert(*node);
                }
                crate::ir::Instruction::If { cond, .. } => {
                    self.root.insert(*node);
                    self.root.insert(*cond);
                }
                crate::ir::Instruction::Switch { value, .. } => {
                    self.root.insert(*node);
                    self.root.insert(*value);
                }
                crate::ir::Instruction::AdScope { .. } => {
                    self.root.insert(*node);
                }
                crate::ir::Instruction::RayQuery { ray_query, .. } => {
                    self.root.insert(*node);
                    self.root.insert(*ray_query);
                }
                crate::ir::Instruction::Print { .. } => {
                    self.root.insert(*node);
                }
                crate::ir::Instruction::AdDetach(_) => {
                    self.root.insert(*node);
                }
                crate::ir::Instruction::Comment(_) => {
                    self.root.insert(*node);
                }
            }
        }
    }
}
