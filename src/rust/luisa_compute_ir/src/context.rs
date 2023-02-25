use std::collections::{HashMap, HashSet};

use lazy_static::lazy_static;
use parking_lot::RwLock;

use crate::{
    ir::{CallableModule, Node, NodeRef, Type},
    CArc, CArcSharedBlock,
};

pub struct Context {
    pub(crate) types: RwLock<HashSet<CArc<Type>>>,
    pub(crate) type_equivalences: RwLock<HashMap<(*const Type, *const Type), bool>>,
}
unsafe impl Sync for Context {}
unsafe impl Send for Context {}

impl Context {
    pub fn register_type(&self, type_: Type) -> CArc<Type> {
        let type_ = CArc::new(type_);
        let types = self.types.read();
        if let Some(type_) = types.get(&type_) {
            type_.clone()
        } else {
            drop(types);
            let mut types = self.types.write();
            types.insert(type_.clone());
            type_
        }
    }
    pub fn is_type_equal(&self, a: &CArc<Type>, b: &CArc<Type>) -> bool {
        if CArc::as_ptr(a) == CArc::as_ptr(b) {
            return true;
        }
        let type_equivalences = self.type_equivalences.read();
        if let Some(&equal) = type_equivalences.get(&(CArc::as_ptr(a), CArc::as_ptr(b))) {
            return equal;
        }
        drop(type_equivalences);
        let mut type_equivalences = self.type_equivalences.write();
        let equal = *a == *b;
        type_equivalences.insert((CArc::as_ptr(a), CArc::as_ptr(b)), equal);
        equal
    }
    pub fn new() -> Self {
        Self {
            types: RwLock::new(HashSet::new()),
            type_equivalences: RwLock::new(HashMap::new()),
        }
    }
}

lazy_static! {
    static ref CONTEXT: Context = Context::new();
}

pub fn with_context<T>(f: impl FnOnce(&Context) -> T) -> T {
    f(&CONTEXT)
}
pub fn reset_context() {
    with_context(|context| {
        context.types.write().clear();
        context.type_equivalences.write().clear();
    });
}
pub fn register_type(type_: Type) -> CArc<Type> {
    with_context(|context| context.register_type(type_))
}
pub fn is_type_equal(a: &CArc<Type>, b: &CArc<Type>) -> bool {
    if CArc::as_ptr(a) == CArc::as_ptr(b) {
        return true;
    }
    with_context(|context| context.is_type_equal(a, b))
}
