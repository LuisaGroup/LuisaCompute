pub mod autodiff;
pub mod lower_control_flow;
pub mod ssa;
pub mod validate;
pub mod vectorize;
use crate::ir;

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
        // "lower_control_flow"=>{
        //     let transform = lower_control_flow::LowerControlFlow::new();
        //     unsafe { (*pipeline).add_transform(Box::new(transform)) };
        // }
        // "vectorize"=>{
        //     let transform = vectorize::Vectorize::new();
        //     unsafe { (*pipeline).add_transform(Box::new(transform)) };
        // }
        "autodiff" => {
            let transform = autodiff::Autodiff;
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
