pub mod autodiff;
pub mod canonicalize_control_flow;
pub mod ssa;
// pub mod validate;
pub mod vectorize;
// pub mod eval;
pub mod fwd_autodiff;
pub mod ref2ret;
pub mod reg2mem;

use crate::ir::{self, ModuleFlags};
use bitflags::Flags;

pub trait Transform {
    fn transform(&self, module: ir::Module) -> ir::Module;
}

pub struct TransformPipeline {
    transforms: Vec<Box<dyn Transform>>,
}
impl TransformPipeline {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }
    pub fn add_transform(&mut self, transform: Box<dyn Transform>) {
        self.transforms.push(transform);
    }
}
impl Transform for TransformPipeline {
    fn transform(&self, module: ir::Module) -> ir::Module {
        let mut module = module;
        for transform in &self.transforms {
            module = transform.transform(module);
        }
        module
    }
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_transform_pipeline_new() -> *mut TransformPipeline {
    Box::into_raw(Box::new(TransformPipeline::new()))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_transform_pipeline_add_transform(
    pipeline: *mut TransformPipeline,
    name: *const std::os::raw::c_char,
) {
    let name = unsafe { std::ffi::CStr::from_ptr(name) }
        .to_str()
        .unwrap()
        .to_string();
    match name.as_str() {
        "ssa" => {
            let transform = ssa::ToSSA;
            unsafe { (*pipeline).add_transform(Box::new(transform)) };
        }
        "canonicalize_control_flow" => {
            let transform = canonicalize_control_flow::CanonicalizeControlFlow;
            unsafe { (*pipeline).add_transform(Box::new(transform)) };
        }
        // "vectorize"=>{
        //     let transform = vectorize::Vectorize::new();
        //     unsafe { (*pipeline).add_transform(Box::new(transform)) };
        // }
        "autodiff" => {
            let transform = autodiff::Autodiff;
            unsafe { (*pipeline).add_transform(Box::new(transform)) };
        }
        "ref2ret" => {
            let transform = ref2ret::Ref2Ret;
            unsafe { (*pipeline).add_transform(Box::new(transform)) };
        }
        "reg2mem" => {
            let transform = reg2mem::Reg2Mem;
            unsafe { (*pipeline).add_transform(Box::new(transform)) };
        }
        _ => panic!("unknown transform {}", name),
    }
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_transform_pipeline_transform(
    pipeline: *mut TransformPipeline,
    module: ir::Module,
) -> ir::Module {
    unsafe { (*pipeline).transform(module) }
}
#[no_mangle]
pub extern "C" fn luisa_compute_ir_transform_pipeline_destroy(pipeline: *mut TransformPipeline) {
    unsafe {
        std::mem::drop(Box::from_raw(pipeline));
    }
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_transform_auto(module: ir::Module) -> ir::Module {
    let flags = module.flags;
    // dbg!(flags);
    let mut pipeline = TransformPipeline::new();
    if flags.contains(ModuleFlags::REQUIRES_REV_AD_TRANSFORM) {
        pipeline.add_transform(Box::new(autodiff::Autodiff));
    }
    if flags.contains(ModuleFlags::REQUIRES_FWD_AD_TRANSFORM) {
        pipeline.add_transform(Box::new(fwd_autodiff::FwdAutodiff));
    }
    pipeline.transform(module)
}
