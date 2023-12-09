use crate::{ir::*, CArc, CBoxedSlice};

pub fn inline_callable(caller: &Module, call: NodeRef, recursive: bool) {
    let inst = call.get().instruction.as_ref();

    let (f, args) = match inst {
        Instruction::Call(f, args) => (f, args),
        _ => unreachable!(),
    };
    let f = match f {
        Func::Callable(f) => f.0.clone(),
        _ => unreachable!(),
    };

    {
        let nodes = f.module.collect_nodes();
        for (i, n) in nodes.iter().enumerate() {
            let is_return = match n.get().instruction.as_ref() {
                Instruction::Return(_) => true,
                _ => false,
            };
            if is_return {
                assert_eq!(
                    i,
                    nodes.len() - 1,
                    "cannot have early return in inlined function"
                );
            }
        }
    }
    let mut dup = ModuleDuplicator::new();
    let inlined_block = dup.with_context(|this| {
        for i in 0..args.len() {
            let ctx = this.current.as_mut().unwrap();
            ctx.nodes.insert(f.args[i], args[i]);
        }
        this.duplicate_block(&caller.pools, &f.module.entry)
    });
    if recursive {
        let inlined = Module {
            curve_basis_set: caller.curve_basis_set,
            pools: caller.pools.clone(),
            kind: ModuleKind::Block,
            entry: inlined_block,
            flags: ModuleFlags::empty(),
        };
        let nodes = inlined.collect_nodes();
        for n in nodes {
            let inst = n.get().instruction.as_ref();
            match inst {
                Instruction::Call(f, _) => match f {
                    Func::Callable(_) => inline_callable(&inlined, n, true),
                    _ => {}
                },
                _ => {}
            }
        }
    }
    if call.type_().is_void() {
        let next = call.get().next;
        let n = inlined_block.len();
        for (i, node) in inlined_block.iter().enumerate() {
            node.remove();
            match node.get().instruction.as_ref() {
                Instruction::Return(v) => {
                    assert_eq!(i + 1, n);
                    assert!(!v.valid());
                }
                _ => next.insert_before_self(node),
            }
        }
        call.remove();
    } else {
        let next = call;
        for node in inlined_block.iter() {
            node.remove();
            next.insert_before_self(node);
        }
        // replace call.inst with the return value
        let return_node = next.get().prev;
        let return_v = match return_node.get().instruction.as_ref() {
            Instruction::Return(v) => *v,
            _ => unreachable!(),
        };
        let mut b = IrBuilder::new_without_bb(caller.pools.clone());
        b.set_insert_point(return_node);
        let v = b.local(return_v);
        call.update(|call| {
            call.instruction = CArc::new(Instruction::Call(Func::Load, CBoxedSlice::new(vec![v])));
        });
        return_node.remove();
    }
}
