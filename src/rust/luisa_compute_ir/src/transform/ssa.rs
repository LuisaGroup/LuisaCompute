use std::collections::HashSet;

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
        // if !self.local_defs.contains(&node) {
        //     return builder.call(Func::Load, &[node], node.type_().clone());
        // }
        if node.is_gep() {
            let inst = node.get().instruction.as_ref();
            let args = match inst {
                Instruction::Call(f, args) => match f {
                    Func::GetElementPtr => args,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            let var = args[0];
            let indices = &args[1..];
            let indices = indices
                .iter()
                .map(|x| self.promote(*x, builder, record))
                .collect::<Vec<_>>();
            let promoted_var = self.promote(var, builder, record);
            let args = vec![promoted_var]
                .into_iter()
                .chain(indices.into_iter())
                .collect::<Vec<_>>();
            let value = builder.call(Func::ExtractElement, &args, node.type_().clone());
            value
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
        let value = self.promote(value, builder, record);
        if var.is_local() || var.is_refernece_argument() {
            record.phis.insert(var);
            record.stored.insert(var, value);
            if !self.local_defs.contains(&var) {
                builder.update(var, value);
            }
        } else {
            // the hardpart
            let inst = var.get().instruction.as_ref();
            let args = match inst {
                Instruction::Call(f, args) => match f {
                    Func::GetElementPtr => args,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            let var = args[0];
            let indices = &args[1..];
            let indices = indices
                .iter()
                .map(|x| self.promote(*x, builder, record))
                .collect::<Vec<_>>();
            let promoted_var = self.promote(var, builder, record);
            let args = vec![promoted_var, value]
                .into_iter()
                .chain(indices.into_iter())
                .collect::<Vec<_>>();
            let value = builder.call(Func::InsertElement, &args, promoted_var.type_().clone());
            // let (var, indices) = var.access_chain().unwrap();
            // // dbg!(var.type_(), &indices);
            // let unpromoted_var = var;
            // let var = self.promote(var, builder, record);
            // let mut st = vec![var];
            // let mut cur = var;
            // for (t, i) in &indices[..indices.len() - 1] {
            //     let i = builder.const_(Const::Int32(*i as i32));
            //     let el = builder.call(Func::ExtractElement, &[cur, i], t.clone());
            //     st.push(el);
            //     cur = el;
            // }
            // let mut value = value;
            // for (_, i) in indices.iter().rev() {
            //     let i = builder.const_(Const::Int32(*i as i32));
            //     let el = builder.call(Func::InsertElement, &[cur, value, i], cur.type_().clone());
            //     value = el;
            //     cur = st.pop().unwrap();
            // }
            // assert!(
            //     context::is_type_equal(unpromoted_var.type_(), value.type_()),
            //     "Type mismatch: {} vs {}",
            //     unpromoted_var.type_(),
            //     value.type_()
            // );
            record.phis.insert(var);
            record.stored.insert(var, value);
            assert_eq!(self.promote(var, builder, record), value);
            if !self.local_defs.contains(&var) {
                builder.update(var, value);
            }
        }
    }
    fn promote_branches(
        &mut self,
        branches: &[Pooled<BasicBlock>],
        builder: &mut IrBuilder,
        record: &mut SSABlockRecord,
    ) -> (Vec<SSABlockRecord>, Vec<Pooled<BasicBlock>>, Vec<NodeRef>) {
        let mut new_branches = vec![];
        let mut new_phis = IndexSet::new();
        let mut records = vec![];
        for branch in branches {
            let mut new_record = SSABlockRecord::from_parent(record);
            let new_branch = self.promote_bb(
                *branch,
                IrBuilder::new(builder.pools.clone()),
                &mut new_record,
            );

            new_branches.push(new_branch);
            new_phis = new_phis.union(&new_record.phis).cloned().collect();
            records.push(new_record);
        }
        (records, new_branches, new_phis.into_iter().collect())
    }

    fn merge_incomings(
        &mut self,
        incoming_records: &[SSABlockRecord],
        incoming_branches: &[Pooled<BasicBlock>],
        phis: &[NodeRef],
        builder: &mut IrBuilder,
        record: &mut SSABlockRecord,
    ) {
        let mut new_phis = vec![];
        for phi in phis {
            let incomings = incoming_records
                .iter()
                .enumerate()
                .map(|(i, x)| PhiIncoming {
                    value: *x.stored.get(phi).unwrap(),
                    block: incoming_branches[i],
                })
                .collect::<Vec<_>>();
            if incomings.iter().all(|x| x.value == incomings[0].value) && incomings.len() > 1 {
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
                if !self.local_defs.contains(&node) && !record.stored.contains_key(&node) {
                    let val = builder.load(node);
                    record.stored.insert(node, val);
                    return val;
                }
                let init = self.promote(*init, builder, record);
                let var = builder.local(init);
                record.defined.insert(node);
                record.stored.insert(node, init);
                return init;
            }
            Instruction::Argument { by_value } => {
                if *by_value {
                    return node;
                }
                assert!(!self.local_defs.contains(&node));
                if !record.stored.contains_key(&node) {
                    let val = builder.load(node);
                    record.stored.insert(node, val);
                    return node;
                }
                unreachable!();
            }
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
                    let v =  self.load(args[0], builder, record);
                    self.map_immutables.insert(node, v);
                    return v;
                }
                if *func == Func::GetElementPtr {
                    return INVALID_REF;
                    // let v = self.load(args[0], builder, record);
                    // // let idx = self.promote(args[1], builder, record);
                    // let indices = args[1..]
                    //     .iter()
                    //     .map(|x| self.promote(*x, builder, record))
                    //     .collect::<Vec<_>>();
                    // let args = vec![v]
                    //     .into_iter()
                    //     .chain(indices.into_iter())
                    //     .collect::<Vec<_>>();
                    // return builder.call(Func::ExtractElement, &args, type_.clone());
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
            Instruction::RayQuery { .. } => panic!("ray query not supported"),
            Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                let cond = self.promote(*cond, builder, record);
                let (records, branches, phis) =
                    self.promote_branches(&[*true_branch, *false_branch], builder, record);
                builder.if_(cond, branches[0], branches[1]);
                self.merge_incomings(&records, &branches, &phis, builder, record);
                return INVALID_REF;
            }
            Instruction::Switch {
                value,
                default,
                cases,
            } => {
                let value = self.promote(*value, builder, record);
                let branches = cases
                    .iter()
                    .map(|x| x.block)
                    .chain(std::iter::once(*default))
                    .collect::<Vec<_>>();
                let (incoming_records, incoming_branches, phis) =
                    self.promote_branches(&branches, builder, record);
                builder.switch(
                    value,
                    &incoming_branches[0..incoming_branches.len() - 1]
                        .iter()
                        .enumerate()
                        .map(|(i, x)| SwitchCase {
                            value: cases[i].value,
                            block: *x,
                        })
                        .collect::<Vec<_>>(),
                    *incoming_branches.last().unwrap(),
                );
                self.merge_incomings(
                    &incoming_records,
                    &incoming_branches,
                    &phis,
                    builder,
                    record,
                );
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
            Instruction::Print { fmt, args } => {
                let args = args
                    .iter()
                    .map(|node| self.promote(*node, builder, record))
                    .collect::<Vec<_>>();
                builder.print(fmt.clone(), &args);
                return INVALID_REF;
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
            flags: module.flags,
        }
    }
}
