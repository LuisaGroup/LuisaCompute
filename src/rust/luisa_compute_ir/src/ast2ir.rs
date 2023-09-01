use std::collections::HashMap;
use crate::{CArc, Pooled};
use serde_json::{Value as JSON};
use base64ct::Base64;
use crate::ir::*;

struct AST2IRCtx<'a> {
    j: &'a JSON,
    variables: HashMap<u32, CArc<NodeRef>>,
}

struct AST2IR<'a: 'b, 'b> {
    j: &'a JSON,
    kernels: HashMap<u32, CArc<KernelModule>>,
    callables: HashMap<u32, CArc<CallableModule>>,
    constants: HashMap<u32, CArc<Const>>,
    ctx: Option<AST2IRCtx<'b>>,
}

enum FunctionModule {
    Kernel(CArc<KernelModule>),
    Callable(CArc<CallableModule>),
}

impl<'a, 'b> AST2IR<'a, 'b> {

    fn convert_function(&mut self, j: &'a JSON) -> FunctionModule {
        let ctx = AST2IRCtx {
            j: &j,
            variables: HashMap::new(),
        };
        let old_ctx = self.ctx.replace(ctx);
        // TODO
        self.ctx = old_ctx;
        todo!()
    }

    fn convert(j: JSON) -> CArc<KernelModule> {
        let mut ast2ir = AST2IR {
            j: &j,
            kernels: HashMap::new(),
            callables: HashMap::new(),
            constants: HashMap::new(),
            ctx: None,
        };
        j.pointer("functions").unwrap().as_array().unwrap().iter().for_each(|f| {
            ast2ir.convert_function(&f);
        });
        assert_eq!(ast2ir.kernels.len(), 1, "There should be exactly one kernel in the AST");
        ast2ir.kernels.remove(&0).unwrap()
    }
}

pub fn convert_ast_to_ir_kernel(data: String) -> CArc<KernelModule> {
    let j: JSON = serde_json::from_str(data.as_str()).unwrap();
    AST2IR::convert(j)
}
