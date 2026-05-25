use crate::ir::{BasicBlock, CallableModule, Func, Instruction, Module, NodeRef, Usage};
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct UsageTreeNodeRef(pub usize);

impl UsageTreeNodeRef {
    fn invalid() -> Self {
        Self(std::usize::MAX)
    }

    fn is_valid(&self) -> bool {
        self.0 != std::usize::MAX
    }
}

#[derive(Debug, Clone)]
pub(crate) struct UsageTreeNode {
    pub children: HashMap<usize, UsageTreeNodeRef>, // For aggregate types, records the usage of each field for precise usage tracking.
}

#[derive(Debug, Clone)]
pub(crate) struct UsageTree {
    pub read_map: HashMap<NodeRef, UsageTreeNodeRef>,
    pub write_map: HashMap<NodeRef, UsageTreeNodeRef>,
    pub nodes: Vec<UsageTreeNode>,
}

impl UsageTree {
    fn add_node(&mut self) -> UsageTreeNodeRef {
        let index = self.nodes.len();
        self.nodes.push(UsageTreeNode {
            children: HashMap::new(),
        });
        UsageTreeNodeRef(index)
    }
}

#[derive(Debug, Clone)]
pub struct ArgumentUsages {
    tree: UsageTree,
    args: Vec<NodeRef>,
}

impl ArgumentUsages {
    pub fn any_possible_usage(&self, arg_index: usize, access_chain: &[usize]) -> Usage {
        let has_usage = |map: &HashMap<NodeRef, UsageTreeNodeRef>| -> bool {
            if let Some(node) = self.args.get(arg_index).and_then(|i| map.get(i)) {
                let mut node_ref = node;
                for i in access_chain.iter() {
                    let node = &self.tree.nodes[node_ref.0];
                    if node.children.is_empty() {
                        return true;
                    } else if let Some(child) = node.children.get(i) {
                        node_ref = child;
                    } else {
                        return false;
                    }
                }
                return true;
            }
            false
        };
        let has_read = has_usage(&self.tree.read_map);
        let has_write = has_usage(&self.tree.write_map);
        match (has_read, has_write) {
            (true, true) => Usage::READ_WRITE,
            (true, false) => Usage::READ,
            (false, true) => Usage::WRITE,
            (false, false) => Usage::NONE,
        }
    }
}

pub struct CallableArgumentUsageAnalysis {
    analyzed: HashMap<*const BasicBlock, Rc<ArgumentUsages>>,
}

// internal implementation
impl CallableArgumentUsageAnalysis {
    fn probe_access_chain(
        tree: &UsageTree,
        node: NodeRef,
        chain: &mut Vec<NodeRef>,
    ) -> Option<NodeRef> {
        match node.get().instruction.as_ref() {
            Instruction::Buffer
            | Instruction::Bindless
            | Instruction::Texture2D
            | Instruction::Texture3D
            | Instruction::Accel
            | Instruction::Argument { .. } => Some(node),
            Instruction::Call(func, args) => match func {
                Func::GetElementPtr | Func::ExtractElement | Func::Load => {
                    if let Some(root) = Self::probe_access_chain(tree, args[0], chain) {
                        chain.extend(args.iter().skip(1).cloned());
                        Some(root)
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn evaluate_constant_indices(indices: &[NodeRef]) -> Vec<usize> {
        let mut const_indices = Vec::new();
        for index in indices.iter() {
            match index.get().instruction.as_ref() {
                Instruction::Const(c) => const_indices.push(c.get_i32() as usize),
                Instruction::Call(func, _) => match func {
                    Func::ZeroInitializer => {
                        const_indices.push(0);
                    }
                    _ => {
                        return const_indices;
                    }
                },
                _ => {
                    return const_indices;
                }
            }
        }
        const_indices
    }

    fn mark_usage(tree: &mut UsageTree, node: NodeRef, usage: Usage) {
        match node.get().instruction.as_ref() {
            Instruction::Buffer
            | Instruction::Bindless
            | Instruction::Texture2D
            | Instruction::Texture3D
            | Instruction::Accel
            | Instruction::Argument { .. } => {
                Self::mark_access_chain_usage(tree, node, &[], usage);
            }
            Instruction::Call(func, args) => match func {
                Func::GetElementPtr | Func::ExtractElement | Func::Load => {
                    let mut chain = Vec::new();
                    if let Some(root) = Self::probe_access_chain(tree, args[0], &mut chain) {
                        chain.extend(args.iter().skip(1).cloned());
                        let chain = Self::evaluate_constant_indices(&chain);
                        Self::mark_access_chain_usage(tree, root, &chain, usage);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn mark_access_chain_usage(tree: &mut UsageTree, root: NodeRef, chain: &[usize], usage: Usage) {
        let usage = match root.get().instruction.as_ref() {
            Instruction::Argument { by_value } => {
                if *by_value {
                    match usage {
                        Usage::READ | Usage::READ_WRITE => Usage::READ,
                        Usage::WRITE | Usage::NONE => Usage::NONE,
                    }
                } else {
                    usage
                }
            }
            _ => usage,
        };
        match root.get().instruction.as_ref() {
            Instruction::Buffer
            | Instruction::Bindless
            | Instruction::Texture2D
            | Instruction::Texture3D
            | Instruction::Accel
            | Instruction::Argument { .. } => {
                let p_tree = tree as *mut UsageTree;
                let maps = match usage {
                    Usage::NONE => vec![],
                    Usage::READ => vec![&mut tree.read_map],
                    Usage::WRITE => vec![&mut tree.write_map],
                    Usage::READ_WRITE => vec![&mut tree.read_map, &mut tree.write_map],
                };
                unsafe {
                    // Safe! Safe! Safe!
                    let tree = &mut *p_tree;
                    for map in maps {
                        let mut parent_is_new = false;
                        let mut node_ref = map
                            .entry(root)
                            .or_insert_with(|| {
                                parent_is_new = true;
                                tree.add_node()
                            })
                            .clone();
                        for &index in chain {
                            if !parent_is_new && tree.nodes[node_ref.0].children.is_empty() {
                                // parent already marked, don't need to mark the children
                                break;
                            }
                            if let Some(child) = tree.nodes[node_ref.0].children.get(&index) {
                                parent_is_new = false;
                                node_ref = *child;
                            } else {
                                parent_is_new = true;
                                let child = tree.add_node();
                                tree.nodes[node_ref.0].children.insert(index, child);
                                node_ref = child;
                            }
                        }
                        tree.nodes[node_ref.0].children.clear();
                    }
                }
            }
            _ => unreachable!("root node must be a valid callable argument"),
        }
    }

    fn propagate_callee_usage_recursively(
        caller_root: NodeRef,
        caller_access_chain: &mut Vec<usize>,
        caller_tree: &mut UsageTree,
        callee_tree_node: UsageTreeNodeRef,
        callee_tree: &UsageTree,
        usage: Usage,
    ) {
        assert!(callee_tree_node.is_valid());
        let callee_tree_node = &callee_tree.nodes[callee_tree_node.0];
        if callee_tree_node.children.is_empty() {
            // leaf node
            Self::mark_access_chain_usage(caller_tree, caller_root, caller_access_chain, usage);
        } else {
            // non-leaf node
            for (&i, &child) in callee_tree_node.children.iter() {
                caller_access_chain.push(i);
                Self::propagate_callee_usage_recursively(
                    caller_root,
                    caller_access_chain,
                    caller_tree,
                    child,
                    callee_tree,
                    usage,
                );
                caller_access_chain.pop();
            }
        }
    }

    fn propagate_callee_usages(
        &mut self,
        caller_nodes: &[NodeRef],
        callee_usages: &ArgumentUsages,
        tree: &mut UsageTree,
    ) {
        for (i, &arg) in caller_nodes.iter().enumerate() {
            // do not put effort on meaningless nodes
            let mut chain = Vec::new();
            if let Some(caller_root) =
                Self::probe_access_chain(&callee_usages.tree, arg.clone(), &mut chain)
            {
                let mut constant_chain = Self::evaluate_constant_indices(&chain);
                if chain.is_empty() || constant_chain.len() < chain.len() {
                    // dynamic indices in the chain, be conservative and mark the known chain
                    let callee_usage = callee_usages.any_possible_usage(i, &[]);
                    Self::mark_access_chain_usage(tree, caller_root, &constant_chain, callee_usage);
                } else {
                    // we may do it precisely
                    for (callee_map, usage) in [
                        (&callee_usages.tree.read_map, Usage::READ),
                        (&callee_usages.tree.write_map, Usage::WRITE),
                    ] {
                        // traverse the leaves and mark the usages
                        if let Some(callee_tree_node) = callee_map.get(&arg) {
                            Self::propagate_callee_usage_recursively(
                                caller_root,
                                &mut constant_chain,
                                tree,
                                *callee_tree_node,
                                &callee_usages.tree,
                                usage,
                            );
                        }
                    }
                }
            }
        }
    }

    fn analyze_block(&mut self, block: &BasicBlock, tree: &mut UsageTree) {
        for node in block.iter() {
            match node.get().instruction.as_ref() {
                Instruction::Local { init, .. } => {
                    Self::mark_usage(tree, init.clone(), Usage::READ);
                }
                Instruction::Update { var, value } => {
                    Self::mark_usage(tree, var.clone(), Usage::WRITE);
                    Self::mark_usage(tree, value.clone(), Usage::READ);
                }
                Instruction::Call(func, args) => match func {
                    Func::PropagateGrad => todo!(),
                    Func::AccGrad => {
                        Self::mark_usage(tree, args[0].clone(), Usage::READ_WRITE);
                        Self::mark_usage(tree, args[1].clone(), Usage::READ);
                    }
                    Func::RayTracingSetInstanceTransform
                    | Func::RayTracingSetInstanceOpacity
                    | Func::RayTracingSetInstanceVisibility
                    | Func::RayTracingSetInstanceUserId
                    | Func::RayQueryCommitTriangle
                    | Func::RayQueryCommitProcedural
                    | Func::RayQueryTerminate
                    | Func::IndirectDispatchSetCount
                    | Func::IndirectDispatchSetKernel => {
                        Self::mark_usage(tree, args[0].clone(), Usage::READ_WRITE);
                        for arg in args.iter().skip(1) {
                            Self::mark_usage(tree, arg.clone(), Usage::READ);
                        }
                    }
                    Func::AddressOf => {
                        // be conservative
                        Self::mark_usage(tree, args[0].clone(), Usage::READ_WRITE);
                    }
                    Func::AtomicExchange
                    | Func::AtomicCompareExchange
                    | Func::AtomicFetchAdd
                    | Func::AtomicFetchSub
                    | Func::AtomicFetchAnd
                    | Func::AtomicFetchOr
                    | Func::AtomicFetchXor
                    | Func::AtomicFetchMin
                    | Func::AtomicFetchMax
                    | Func::BufferWrite
                    | Func::BufferAddress
                    | Func::ByteBufferWrite
                    | Func::Texture2dWrite
                    | Func::Texture3dWrite
                    | Func::BindlessBufferWrite
                    | Func::BindlessBufferAddress => {
                        Self::mark_usage(tree, args[0].clone(), Usage::READ_WRITE);
                        for arg in args.iter().skip(1) {
                            Self::mark_usage(tree, arg.clone(), Usage::READ);
                        }
                    }
                    Func::GetElementPtr | Func::ExtractElement | Func::Load => {
                        /* analysis will be deferred until users are analyzed */
                        for node in args.iter().skip(1) {
                            Self::mark_usage(tree, node.clone(), Usage::READ);
                        }
                    }
                    Func::Callable(callable_ref) => {
                        let usages = self.analyze_callable(callable_ref.0.as_ref());
                        self.propagate_callee_usages(args, &usages, tree);
                    }
                    Func::CpuCustomOp(_) => todo!(),
                    _ => {
                        for arg in args.iter() {
                            Self::mark_usage(tree, arg.clone(), Usage::READ);
                        }
                    }
                },
                Instruction::Phi(incomings) => {
                    for incoming in incomings.iter() {
                        Self::mark_usage(tree, incoming.value.clone(), Usage::READ);
                    }
                }
                Instruction::Return(value) => {
                    if value.valid() {
                        Self::mark_usage(tree, value.clone(), Usage::READ);
                    }
                }
                Instruction::Loop { body, cond } => {
                    self.analyze_block(body, tree);
                    Self::mark_usage(tree, cond.clone(), Usage::READ);
                }
                Instruction::GenericLoop {
                    prepare,
                    body,
                    update,
                    cond,
                } => {
                    self.analyze_block(prepare, tree);
                    self.analyze_block(body, tree);
                    self.analyze_block(update, tree);
                    Self::mark_usage(tree, cond.clone(), Usage::READ);
                }
                Instruction::If {
                    cond,
                    true_branch,
                    false_branch,
                } => {
                    Self::mark_usage(tree, cond.clone(), Usage::READ);
                    self.analyze_block(true_branch.as_ref(), tree);
                    self.analyze_block(false_branch.as_ref(), tree);
                }
                Instruction::Switch {
                    value,
                    cases,
                    default,
                } => {
                    Self::mark_usage(tree, value.clone(), Usage::READ);
                    for case in cases.iter() {
                        self.analyze_block(case.block.as_ref(), tree);
                    }
                    self.analyze_block(default.as_ref(), tree);
                }
                Instruction::AdScope { body, .. } => {
                    self.analyze_block(body, tree);
                }
                Instruction::RayQuery {
                    ray_query,
                    on_triangle_hit,
                    on_procedural_hit,
                } => {
                    Self::mark_usage(tree, ray_query.clone(), Usage::READ);
                    self.analyze_block(on_triangle_hit.as_ref(), tree);
                    self.analyze_block(on_procedural_hit.as_ref(), tree);
                }
                Instruction::Print { args, .. } => {
                    for arg in args.iter() {
                        Self::mark_usage(tree, arg.clone(), Usage::READ);
                    }
                }
                Instruction::AdDetach(body) => {
                    self.analyze_block(body, tree);
                }
                Instruction::CoroRegister { value, .. } => {
                    Self::mark_usage(tree, value.clone(), Usage::READ);
                }
                _ => {}
            }
        }
    }

    // replayable values analysis needs to know the usage of each argument
    pub(crate) fn analyze_module(&mut self, module: &Module) -> UsageTree {
        let mut tree = UsageTree {
            read_map: HashMap::new(),
            write_map: HashMap::new(),
            nodes: Vec::new(),
        };
        self.analyze_block(module.entry.as_ref(), &mut tree);
        tree
    }

    fn analyze_callable(&mut self, callable: &CallableModule) -> Rc<ArgumentUsages> {
        if !self.analyzed.contains_key(&callable.module.entry.as_ptr()) {
            let mut tree = self.analyze_module(&callable.module);
            for arg in callable.args.iter() {
                match arg.get().instruction.as_ref() {
                    Instruction::Argument { by_value } => {
                        if *by_value {
                            // by-value argument is always read
                            tree.write_map.remove(arg);
                        }
                    }
                    _ => {}
                }
            }
            let usages = ArgumentUsages {
                tree,
                args: callable.args.to_vec(),
            };
            self.analyzed
                .insert(callable.module.entry.as_ptr(), Rc::new(usages));
        };
        self.analyzed
            .get(&callable.module.entry.as_ptr())
            .unwrap()
            .clone()
    }
}

// public interfaces
impl CallableArgumentUsageAnalysis {
    pub fn new() -> Self {
        Self {
            analyzed: HashMap::new(),
        }
    }

    pub fn analyze(&mut self, module: &CallableModule) -> Rc<ArgumentUsages> {
        self.analyze_callable(module)
    }

    pub fn get_possible_usage(
        &mut self,
        module: &CallableModule,
        arg_index: usize,
        access_chain: &[usize],
    ) -> Usage {
        let usages = self.analyze(module);
        usages.any_possible_usage(arg_index, access_chain)
    }
}
