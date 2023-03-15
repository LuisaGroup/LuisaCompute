use std::collections::HashMap;

use crate::ir::{BasicBlock, KernelModule, Module, NodeRef, SwitchCase, Usage, UsageMark};

struct UsageDetector {
    map: HashMap<NodeRef, Usage>,
}

impl UsageDetector {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn mark(&mut self, node_ref: NodeRef, flag: UsageMark) {
        self.map.get_mut(&node_ref).map(|item| {
            *item = item.mark(flag);
        });
    }

    fn detect_block(&mut self, block: &BasicBlock) {
        let mut current_node = block.first.get().next;
        while current_node != block.last {
            self.detect_node(current_node);
            current_node = current_node.get().next;
        }
    }

    fn detect_node(&mut self, node_ref: NodeRef) {
        let node = node_ref.get();
        let type_ = node.type_.clone();
        match node.instruction.as_ref() {
            crate::ir::Instruction::Buffer => {
                self.map.insert(node_ref, Usage::NONE);
            }
            crate::ir::Instruction::Bindless => {
                self.map.insert(node_ref, Usage::NONE);
            }
            crate::ir::Instruction::Texture2D => {
                self.map.insert(node_ref, Usage::NONE);
            }
            crate::ir::Instruction::Texture3D => {
                self.map.insert(node_ref, Usage::NONE);
            }
            crate::ir::Instruction::Accel => {
                self.map.insert(node_ref, Usage::NONE);
            }
            crate::ir::Instruction::Shared => {}
            crate::ir::Instruction::Uniform => {
                self.map.insert(node_ref, Usage::NONE);
            }
            crate::ir::Instruction::Local { init } => {
                self.detect_node(*init);
            }
            crate::ir::Instruction::Argument { by_value: _ } => {
                self.map.insert(node_ref, Usage::NONE);
            }
            crate::ir::Instruction::UserData(_) => {}
            crate::ir::Instruction::Invalid => {}
            crate::ir::Instruction::Const(_) => {}
            crate::ir::Instruction::Update { var, value } => {
                self.mark(*var, UsageMark::WRITE);
                self.mark(*value, UsageMark::READ);
            }
            crate::ir::Instruction::Call(func, args) => {
                match func {
                    // args[0] READ & WRITE, 其余 READ
                    crate::ir::Func::AtomicExchange
                    | crate::ir::Func::AtomicCompareExchange
                    | crate::ir::Func::AtomicFetchAdd
                    | crate::ir::Func::AtomicFetchSub
                    | crate::ir::Func::AtomicFetchAnd
                    | crate::ir::Func::AtomicFetchOr
                    | crate::ir::Func::AtomicFetchXor
                    | crate::ir::Func::AtomicFetchMin
                    | crate::ir::Func::AtomicFetchMax => {
                        for (index, arg) in args.as_ref().iter().enumerate() {
                            self.mark(*arg, UsageMark::READ);
                            if index == 0 {
                                self.mark(*arg, UsageMark::WRITE);
                            }
                        }
                    }
                    // args[0] WRITE, 其余 READ
                    crate::ir::Func::RayTracingSetInstanceTransform
                    | crate::ir::Func::RayTracingSetInstanceOpacity
                    | crate::ir::Func::RayTracingSetInstanceVisibility
                    | crate::ir::Func::BufferWrite
                    | crate::ir::Func::Texture2dWrite
                    | crate::ir::Func::Texture3dWrite => {
                        for (index, arg) in args.as_ref().iter().enumerate() {
                            if index == 0 {
                                self.mark(*arg, UsageMark::WRITE);
                            } else {
                                self.mark(*arg, UsageMark::READ);
                            }
                        }
                    }
                    // 全都是 READ
                    _ => {
                        for arg in args.as_ref() {
                            self.mark(*arg, UsageMark::READ);
                        }
                    }
                }
            }
            crate::ir::Instruction::Phi(phis) => {
                for phi in phis.as_ref() {
                    self.mark(phi.value, UsageMark::READ);
                }
            }
            crate::ir::Instruction::Return(return_) => {
                self.mark(*return_, UsageMark::READ);
            }
            crate::ir::Instruction::Loop { body, cond } => {
                self.mark(*cond, UsageMark::READ);
                self.detect_block(body);
            }
            crate::ir::Instruction::GenericLoop {
                prepare,
                cond,
                body,
                update,
            } => {
                self.detect_block(prepare);
                self.mark(*cond, UsageMark::READ);
                self.detect_block(body);
                self.detect_block(update);
            }
            crate::ir::Instruction::Break => {}
            crate::ir::Instruction::Continue => {}
            crate::ir::Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                self.mark(*cond, UsageMark::READ);
                self.detect_block(true_branch);
                self.detect_block(false_branch);
            }
            crate::ir::Instruction::Switch {
                value,
                default,
                cases,
            } => {
                self.mark(*value, UsageMark::READ);
                for SwitchCase { value: _, block } in cases.as_ref() {
                    self.detect_block(block);
                }
                self.detect_block(default);
            }
            crate::ir::Instruction::AdScope {
                forward,
                backward,
                epilogue,
            } => {
                self.detect_block(forward);
                self.detect_block(backward);
                self.detect_block(epilogue);
            }
            crate::ir::Instruction::AdDetach(block) => {
                self.detect_block(block);
            }
            crate::ir::Instruction::Comment(_) => {}
            crate::ir::Instruction::Debug(_) => {}
        }
    }

    fn detect_module(mut self, module: &Module) -> HashMap<NodeRef, Usage> {
        self.detect_block(&module.entry);
        self.map
    }
}

pub fn detect_usage(module: &Module) -> HashMap<NodeRef, Usage> {
    let usage_detector = UsageDetector::new();
    usage_detector.detect_module(&module)
}
