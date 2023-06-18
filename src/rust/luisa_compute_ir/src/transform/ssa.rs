use std::collections::{BTreeSet, HashSet};

use super::Transform;

use crate::*;
use indexmap::IndexSet;
use ir::*;

/*
Remove all Instruction::Update nodes

*/
pub struct ToSSA;
struct ToSSAImpl {
    map_blocks: HashMap<*mut BasicBlock, *mut BasicBlock>,
    local_defs: HashSet<NodeRef>,
    map_immutables: HashMap<NodeRef, NodeRef>,
}
struct SSABlockRecord {
    defined: NestedHashSet<NodeRef>,
    stored: NestedHashMap<NodeRef, NodeRef>,
    phis: IndexSet<NodeRef>,
}

impl SSABlockRecord {
    fn from_parent(parent: &Self) -> Self {
        Self {
            defined: NestedHashSet::from_parent(&parent.defined),
            stored: NestedHashMap::from_parent(&parent.stored),
            phis: IndexSet::new(),
        }
    }
    fn new() -> Self {
        Self {
            defined: NestedHashSet::new(),
            stored: NestedHashMap::new(),
            phis: IndexSet::new(),
        }
    }
}

impl ToSSAImpl {
    fn new(model: &Module) -> Self {
        Self {
            map_blocks: HashMap::new(),
            local_defs: model.collect_nodes().into_iter().collect(),
            map_immutables: HashMap::new(),
        }
    }
    fn load(
        &mut self,
        node: NodeRef,
        builder: &mut IrBuilder,
        record: &mut SSABlockRecord,
    ) -> NodeRef {
        if !self.local_defs.contains(&node) {
            return builder.call(Func::Load, &[node], node.type_().clone());
        }
        if let Some((var, indices)) = node.access_chain() {
            let mut cur = self.promote(var, builder, record);
            for (t, i) in &indices {
                let i = builder.const_(Const::Int32(*i as i32));
                let el = builder.call(Func::ExtractElement, &[cur, i], t.clone());
                cur = el;
            }
            cur
        } else {
            self.promote(node, builder, record)
        }
    }
    fn update(
        &mut self,
        var: NodeRef,
        value: NodeRef,
        builder: &mut IrBuilder,
        record: &mut SSABlockRecord,
    ) {
        if var.is_local() {
            let value = self.promote(value, builder, record);
            record.phis.insert(var);
            record.stored.insert(var, value);
        } else {
            // the hardpart
            let (var, indices) = var.access_chain().unwrap();
            let var = self.promote(var, builder, record);
            let mut st = vec![var];
            let mut cur = var;
            for (t, i) in &indices[..indices.len() - 1] {
                let i = builder.const_(Const::Int32(*i as i32));
                let el = builder.call(Func::ExtractElement, &[cur, i], t.clone());
                st.push(el);
                cur = el;
            }
            let mut value = value;
            for (t, i) in indices.iter().rev() {
                let i = builder.const_(Const::Int32(*i as i32));
                let el = builder.call(Func::InsertElement, &[cur, value, i], t.clone());
                value = el;
                cur = st.pop().unwrap();
            }
            record.phis.insert(var);
            record.stored.insert(var, cur);
        }
        if !self.local_defs.contains(&var) {
            builder.update(var, value);
        }
    }
    fn promote(
        &mut self,
        node: NodeRef,
        builder: &mut IrBuilder,
        record: &mut SSABlockRecord,
    ) -> NodeRef {
        // assert!(record.stored.get(&node).is_none(), "promote called on already promoted node {:?}", node.get());
        if let Some(v) = record.stored.get(&node) {
            return *v;
        }
        if !node.is_lvalue() && !self.local_defs.contains(&node) {
            return node;
        }
        if !node.is_lvalue() && self.map_immutables.contains_key(&node) {
            return self.map_immutables[&node];
        }
        let instruction = &node.get().instruction;
        let type_ = &node.get().type_;
        match instruction.as_ref() {
            Instruction::Buffer => return node,
            Instruction::Bindless => return node,
            Instruction::Texture2D => return node,
            Instruction::Texture3D => return node,
            Instruction::Accel => return node,
            Instruction::Shared => return node,
            Instruction::Uniform => return node,
            Instruction::Local { init } => {
                if !self.local_defs.contains(&node) {
                    return node;
                }
                let init = self.promote(*init, builder, record);
                let var = builder.local(init);
                record.defined.insert(node);
                record.stored.insert(node, init);
                return var;
            }
            Instruction::Argument { .. } => todo!(),
            Instruction::UserData(_) => return node,
            Instruction::Invalid => return node,
            Instruction::Const(c) => {
                let c = builder.const_(c.clone());
                self.map_immutables.insert(node, c);
                return c;
            }
            Instruction::Update { var, value } => {
                self.update(*var, *value, builder, record);
                // no need to return the value as it is already stored in `value`
                return INVALID_REF;
            }
            Instruction::Call(func, args) => {
                if *func == Func::Load {
                    return self.load(args[0], builder, record);
                }
                let promoted_args = args
                    .as_ref()
                    .iter()
                    .map(|a| self.promote(*a, builder, record))
                    .collect::<Vec<_>>();
                let v = builder.call(func.clone(), &promoted_args, type_.clone());
                self.map_immutables.insert(node, v);
                return v;
            }
            Instruction::Phi(incomings) => {
                let mut new_incomings = vec![];
                for incoming in incomings.iter() {
                    let value = self.promote(incoming.value, builder, record);
                    new_incomings.push(PhiIncoming {
                        value,
                        block: Pooled {
                            ptr: self.map_blocks[&incoming.block.ptr],
                        },
                    });
                }
                let v = builder.phi(&new_incomings, type_.clone());
                self.map_immutables.insert(node, v);
                return v;
            }
            Instruction::Break => {
                let v = builder.break_();
                return v;
            }
            Instruction::Continue => {
                let v = builder.continue_();
                return v;
            }
            Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                let cond = self.promote(*cond, builder, record);
                let mut true_record = SSABlockRecord::from_parent(record);
                let true_branch = self.promote_bb(
                    *true_branch,
                    IrBuilder::new(builder.pools.clone()),
                    &mut true_record,
                );
                let mut false_record = SSABlockRecord::from_parent(record);
                let false_branch = self.promote_bb(
                    *false_branch,
                    IrBuilder::new(builder.pools.clone()),
                    &mut false_record,
                );
                let phis = true_record
                    .phis
                    .union(&false_record.phis)
                    .cloned()
                    .collect::<Vec<_>>();
                builder.if_(cond, true_branch, false_branch);
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
                    let new_phi = builder.phi(&incomings, phi.get().type_.clone());
                    new_phis.push(new_phi);
                    // if record.defined.contains(phi) {
                    self.map_immutables.insert(*phi, new_phi);
                    record.phis.insert(*phi);
                    // }
                }
                for phi in new_phis {
                    record.defined.insert(phi);
                }
                return INVALID_REF;
            }
            Instruction::Switch {
                value,
                default,
                cases,
            } => {
                let value = self.promote(*value, builder, record);
                let mut default_record = SSABlockRecord::from_parent(record);
                let default_branch = self.promote_bb(
                    *default,
                    IrBuilder::new(builder.pools.clone()),
                    &mut default_record,
                );
                let mut processed_cases = vec![];
                for case in cases.iter() {
                    let mut record = SSABlockRecord::from_parent(record);
                    let bb = self.promote_bb(
                        case.block,
                        IrBuilder::new(builder.pools.clone()),
                        &mut record,
                    );
                    processed_cases.push((bb, record));
                }
                let mut phis = default_record.phis.clone();
                for case in &processed_cases {
                    phis = phis.union(&case.1.phis).cloned().collect();
                }
                builder.switch(
                    value,
                    &processed_cases
                        .iter()
                        .enumerate()
                        .map(|(i, x)| SwitchCase {
                            value: cases[i].value,
                            block: x.0,
                        })
                        .collect::<Vec<_>>(),
                    default_branch,
                );
                let mut new_phis = vec![];
                for phi in &phis {
                    let mut incomings = vec![];
                    incomings.push(PhiIncoming {
                        value: *default_record.stored.get(phi).unwrap(),
                        block: default_branch,
                    });
                    for case in &processed_cases {
                        incomings.push(PhiIncoming {
                            value: *case.1.stored.get(phi).unwrap(),
                            block: case.0,
                        });
                    }
                    let new_phi = builder.phi(&incomings, phi.get().type_.clone());
                    new_phis.push(new_phi);
                    // if record.defined.contains(phi) {
                    self.map_immutables.insert(*phi, new_phi);
                    record.phis.insert(*phi);
                    // }
                }
                for phi in new_phis {
                    record.defined.insert(phi);
                }
                return INVALID_REF;
            }
            Instruction::Loop { body, cond } => {
                let mut body_record = SSABlockRecord::from_parent(record);
                let body = self.promote_bb(
                    *body,
                    IrBuilder::new(builder.pools.clone()),
                    &mut body_record,
                );
                let cond = self.promote(*cond, builder, record);
                builder.loop_(body, cond)
            }
            Instruction::GenericLoop { .. } => todo!(),
            Instruction::AdScope { .. } => todo!(),
            Instruction::AdDetach(_) => todo!(),
            Instruction::Comment(_) => return node,
            Instruction::Return(_) => {
                panic!("call LowerControlFlow before ToSSA");
            }
        }
    }
    fn promote_bb(
        &mut self,
        bb: Pooled<BasicBlock>,
        mut builder: IrBuilder,
        record: &mut SSABlockRecord,
    ) -> Pooled<BasicBlock> {
        for node in bb.nodes().iter() {
            self.promote(*node, &mut builder, record);
        }
        let out = builder.finish();
        assert!(self.map_blocks.insert(bb.ptr, out.ptr).is_none());
        out
    }
}

impl Transform for ToSSA {
    fn transform(&self, module: Module) -> Module {
        let mut imp = ToSSAImpl::new(&module);
        let new_bb = imp.promote_bb(
            module.entry,
            IrBuilder::new(module.pools.clone()),
            &mut SSABlockRecord::new(),
        );
        let mut entry = module.entry;
        *entry.get_mut() = *new_bb;
        Module {
            kind: module.kind,
            entry,
            pools: module.pools,
        }
    }
}
