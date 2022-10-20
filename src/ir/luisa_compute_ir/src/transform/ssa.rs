use std::collections::{BTreeSet, HashSet};

use super::Transform;

use crate::*;
use ir::*;

/*
Remove all Instruction::Update nodes

*/
pub struct ToSSA;
struct SSABlockRecord {
    defined: NestedHashSet<NodeRef>,
    stored: NestedHashMap<NodeRef, NodeRef>,
    phis: BTreeSet<NodeRef>,
}
impl SSABlockRecord {
    fn from_parent(parent: &Self) -> Self {
        Self {
            defined: NestedHashSet::from_parent(&parent.defined),
            stored: NestedHashMap::from_parent(&parent.stored),
            phis: BTreeSet::new(),
        }
    }
    fn new() -> Self {
        Self {
            defined: NestedHashSet::new(),
            stored: NestedHashMap::new(),
            phis: BTreeSet::new(),
        }
    }
}
fn promote(v: NodeRef, stored: &NestedHashMap<NodeRef, NodeRef>) -> NodeRef {
    if let Some(r) = stored.get(&v) {
        *r
    } else {
        v
    }
}
impl ToSSA {
    pub fn new() -> Self {
        Self {}
    }
    fn promote(
        &self,
        node: NodeRef,
        builder: &mut IrBuilder,
        record: &mut SSABlockRecord,
    ) -> NodeRef {
        // assert!(record.stored.get(&node).is_none(), "promote called on already promoted node {:?}", node.get());
        if let Some(v) = record.stored.get(&node) {
            return *v;
        }
        let instruction = node.get().instruction;
        let type_ = node.get().type_;
        match instruction {
            Instruction::Buffer => return node,
            Instruction::Bindless => return node,
            Instruction::Texture2D => return node,
            Instruction::Texture3D => return node,
            Instruction::Accel => return node,
            Instruction::Shared => return node,
            Instruction::Uniform => return node,
            Instruction::Local { init } => {
                let var = builder.local(*init);
                record.defined.insert(node);
                record.stored.insert(node, *init);
                return var;
            }
            Instruction::UserData(_) => return node,
            Instruction::Invalid => return node,
            Instruction::Const(c) => return builder.const_(c.clone()),
            Instruction::Update { var, value } => {
                let value = self.promote(*value, builder, record);
                record.phis.insert(*var);
                record.stored.insert(*var, value);
                // no need to return the value as it is already stored in `value`
                return INVALID_REF;
            }
            Instruction::Call(func, args) => {
                let promoted_args = args
                    .as_ref()
                    .iter()
                    .map(|a| promote(*a, &record.stored))
                    .collect::<Vec<_>>();
                let v = builder.call(func.clone(), &promoted_args, type_);
                record.stored.insert(node, v);
                return v;
            }
            Instruction::Phi(_) => todo!(),
            Instruction::Break => {
                let v = builder.break_();
                record.stored.insert(node, v);
                return v;
            }
            Instruction::Continue => {
                let v = builder.continue_();
                record.stored.insert(node, v);
                return v;
            }
            Instruction::Return(_) => todo!(),
            Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                let cond = self.promote(*cond, builder, record);
                let mut true_record = SSABlockRecord::from_parent(record);
                let true_branch = self.promote_bb(*true_branch, IrBuilder::new(), &mut true_record);
                let mut false_record = SSABlockRecord::from_parent(record);
                let false_branch =
                    self.promote_bb(*false_branch, IrBuilder::new(), &mut false_record);
                let phis = true_record
                    .phis
                    .union(&false_record.phis)
                    .cloned()
                    .collect::<Vec<_>>();
                let mut new_phis = vec![];
                for phi in &phis {
                    let incomings = [
                        PhiIncoming {
                            value: *true_record.stored.get(phi).unwrap(),
                            block: true_branch,
                        },
                        PhiIncoming {
                            value: *false_record.stored.get(phi).unwrap(),
                            block: false_branch,
                        },
                    ];
                    if incomings[0].value == incomings[1].value {
                        continue;
                    }
                    let new_phi = builder.phi(&incomings, phi.get().type_);
                    new_phis.push(new_phi);
                    if record.defined.contains(phi) {
                        record.stored.insert(*phi, new_phi);
                        record.phis.insert(*phi);
                    }
                }
                for phi in new_phis {
                    record.defined.insert(phi);
                }
                builder.if_(cond, true_branch, false_branch)
            }
            Instruction::Switch { .. } => todo!(),
            Instruction::Loop { body, cond } => {
                let mut body_record = SSABlockRecord::from_parent(record);
                let body = self.promote_bb(*body, IrBuilder::new(), &mut body_record);
                let cond = self.promote(*cond, builder, record);
                builder.loop_(body, cond)
            },
            Instruction::Comment(_) => return node
        }
    }
    fn promote_bb(
        &self,
        bb: &'static BasicBlock,
        mut builder: IrBuilder,
        record: &mut SSABlockRecord,
    ) -> &'static BasicBlock {
        for node in bb.nodes().iter() {
            self.promote(*node, &mut builder, record);
        }
        builder.finish()
    }
}
impl Transform for ToSSA {
    fn transform(&self, module: Module) -> Module {
        let new_bb = self.promote_bb(module.entry, IrBuilder::new(), &mut SSABlockRecord::new());
        Module {
            kind: module.kind,
            entry: new_bb,
        }
    }
}
