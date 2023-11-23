use crate::{ir::*, CArc, CBoxedSlice};

pub fn inline_callable(caller: &Module, call: NodeRef, recursive: bool) {
    let inst = call.get().instruction.as_ref();

    let (f, args) = match inst {
        Instruction::Call(f, args) => (f, args),
        _ => unreachable!(),
    };
    let f = match f {
        Func::Callable(f) => &f.0,
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
                assert!(
                    i == nodes.len() - 1,
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
        let prev = call.get().prev;
        let next = call.get().next;
        prev.update(|prev| {
            prev.next = inlined_block.first().get().next;
        });
        inlined_block.first().get().next.update(|next| {
            next.prev = prev;
        });
        next.update(|next| {
            next.prev = inlined_block.last().get().prev;
        });
        inlined_block.last().get().prev.update(|prev| {
            prev.next = next;
        });
        call.update(|call| {
            call.prev = INVALID_REF;
            call.next = INVALID_REF;
        });
    } else {
        let last_node_in_block = inlined_block.last().get().prev;
        let prev = call.get().prev;
        let next = call;
        prev.update(|prev| {
            prev.next = inlined_block.first().get().next;
        });
        inlined_block.first().get().next.update(|next| {
            next.prev = prev;
        });
        next.update(|next| {
            next.prev = last_node_in_block;
        });
        last_node_in_block.update(|prev| {
            prev.next = next;
        });

        // replace call.inst with the return value
        let return_node = last_node_in_block;

        let return_v = match return_node.get().instruction.as_ref() {
            Instruction::Return(v) => *v,
            _ => unreachable!(),
        };
        let mut b = IrBuilder::new_without_bb(caller.pools.clone());
        b.set_insert_point(last_node_in_block);
        let v = b.local(return_v);
        call.update(|call| {
            call.instruction = CArc::new(Instruction::Call(Func::Load, CBoxedSlice::new(vec![v])));
        });
        return_node.remove();
    }
}
