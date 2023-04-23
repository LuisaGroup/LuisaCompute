use crate::{ir::*, *};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub(crate) enum Value {
    Atom(NodeRef),
    Compound(Vec<NodeRef>),
}

pub struct Evaluator {
    inline_callable: bool,
    trace: bool,
    original: Module,
    env: NestedHashMap<NodeRef, Value>,
    locally_defined: HashSet<NodeRef>,
    transformed: IrBuilder,
    pools: CArc<ModulePools>,
}

impl Evaluator {
    pub fn new(
        m: Module,
        inline_callable: bool,
        trace: bool,
        env: HashMap<NodeRef, NodeRef>,
    ) -> Self {
        let mut env_ = NestedHashMap::new();
        for (k, v) in env {
            env_.insert(k, Value::Atom(v));
        }
        let locally_defined = HashSet::from_iter(m.collect_nodes());
        let pools = CArc::new(ModulePools::new());
        Self {
            inline_callable,
            trace,
            original: m,
            env: env_,
            locally_defined,
            pools: pools.clone(),
            transformed: IrBuilder::new(pools.clone()),
        }
    }
    fn eval_node(&mut self, node: NodeRef, builder:&mut IrBuilder) -> Value {
        if self.env.contains_key(&node) {
            return self.env.get(&node).unwrap().clone();
        } else {
            let result = self._eval_node(node, builder);
            self.env.insert(node, result.clone());
            result
        }
    }
    fn _eval_node(&mut self, node: NodeRef, builder:&mut IrBuilder) -> Value {
        let inst = node.get().instruction.as_ref();
        let ty = node.type_();
        match inst {
            Instruction::Const(c)=>{
                if !ty.is_primitive() {
                    self.destruct(node, builder)
                } else {
                    Value::Atom(builder.const_(c.clone()))
                }
            }
            _=>todo!()
        }
    }
    fn destruct(&self, node:NodeRef, builder:&mut IrBuilder)->Value {
        // let ty = node.type_();
        // let mut result = Vec::new();
        // // match ty.as_ref() {
        // //     Type::Vector()
        // // }
        // //
        // Value::Compound(result)
        todo!()
    }
}
