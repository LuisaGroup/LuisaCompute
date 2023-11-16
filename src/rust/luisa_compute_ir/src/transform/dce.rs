use crate::{
    analysis::usedef::UseDef,
    ir::{Func, Instruction},
};

use super::Transform;

/// Very conservative dead code elimination.
/// Only removes certain nodes that are obviously not used.
pub struct Dce;

impl Transform for Dce {
    fn transform(&self, module: crate::ir::Module) -> crate::ir::Module {
        let use_def = UseDef::compute(&module);
        let nodes = module.collect_nodes();
        for node in nodes {
            if !use_def.reachable(node) {
                let inst = node.get().instruction.as_ref();
                let mut need_remove = false;
                match inst {
                    Instruction::Const(_) => {
                        need_remove = true;
                    }
                    Instruction::Call(f, _) => {
                        // math functions
                        if f.discriminant() >= Func::Add.discriminant()
                            && f.discriminant() <= Func::Inverse.discriminant()
                        {
                            need_remove = true;
                        }
                        // aggregate functions
                        if f.discriminant() >= Func::Vec.discriminant()
                            && f.discriminant() <= Func::Mat4.discriminant()
                        {
                            need_remove = true;
                        }
                        match f {
                            Func::Load
                            | Func::Cast
                            | Func::Bitcast
                            | Func::BufferRead
                            | Func::BufferSize
                            | Func::BindlessBufferRead
                            | Func::BindlessBufferSize
                            | Func::BindlessBufferType
                            | Func::ByteBufferRead
                            | Func::ByteBufferSize
                            | Func::Texture2dRead
                            | Func::Texture2dSize
                            | Func::Texture3dRead
                            | Func::Texture3dSize
                            | Func::BindlessTexture2dRead
                            | Func::BindlessTexture2dSize
                            | Func::BindlessTexture2dReadLevel
                            | Func::BindlessTexture2dSample
                            | Func::BindlessTexture2dSampleLevel
                            | Func::BindlessTexture2dSampleGrad
                            | Func::BindlessTexture2dSampleGradLevel
                            | Func::BindlessTexture3dRead
                            | Func::BindlessTexture3dSize
                            | Func::BindlessTexture3dReadLevel
                            | Func::BindlessTexture3dSample
                            | Func::BindlessTexture3dSampleLevel
                            | Func::BindlessTexture3dSampleGrad
                            | Func::BindlessTexture3dSampleGradLevel => {
                                need_remove = true;
                            }

                            _ => {}
                        }
                    }
                    _ => {}
                }
                if need_remove {
                    node.remove();
                }
            }
        }
        module
    }
}
