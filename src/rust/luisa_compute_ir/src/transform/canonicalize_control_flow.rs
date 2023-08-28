/*
 * This file implements the control flow canonicalization transform.
 * This transform removes all break/continue/early-return statements, in the following steps:
 *
 */

use crate::ir::Module;
use crate::transform::Transform;

pub struct CanonicalizeControlFlow;

impl Transform for CanonicalizeControlFlow {
    fn transform(&self, _module: Module) -> Module {
        todo!()
    }
}
