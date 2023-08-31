use crate::ir;
use json;
use json::JsonValue;

struct AST2IR {
}

impl AST2IR {
    fn convert(j: JsonValue) {

    }
}

pub fn convert_ast_to_ir(data: String) -> Vec<ir::KernelModule> {
    let j = json::parse(data.as_str()).unwrap();
    todo!()
}
