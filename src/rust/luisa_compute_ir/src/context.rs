use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
};

use lazy_static::lazy_static;
use parking_lot::RwLock;

use crate::{ir::Type, CArc};

pub struct Context {
    pub(crate) types: RwLock<HashSet<CArc<Type>>>,
    pub(crate) type_equivalences: RwLock<HashMap<(*const Type, *const Type), bool>>,
    pub(crate) type_hash: RwLock<HashMap<*const Type, u64>>,
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
    pub fn register_arc_type(&self, type_: &CArc<Type>) {
        let types = self.types.read();
        if let Some(_type_) = types.get(&type_) {
            return;
        } else {
            drop(types);
            let mut types = self.types.write();
            types.insert(type_.clone());
        }
    }
    pub fn is_type_equal(&self, a: &CArc<Type>, b: &CArc<Type>) -> bool {
        self.register_arc_type(a);
        self.register_arc_type(b);
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
    pub fn type_hash(&self, type_: &CArc<Type>) -> u64 {
        self.register_arc_type(type_);
        let type_hash = self.type_hash.read();
        if let Some(&hash) = type_hash.get(&CArc::as_ptr(type_)) {
            return hash;
        }
        drop(type_hash);
        let mut type_hash = self.type_hash.write();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        type_.hash(&mut hasher);
        let hash = hasher.finish();
        type_hash.insert(CArc::as_ptr(type_), hash);
        hash
    }
    pub fn new() -> Self {
        Self {
            types: RwLock::new(HashSet::new()),
            type_equivalences: RwLock::new(HashMap::new()),
            type_hash: RwLock::new(HashMap::new()),
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
pub fn type_hash(type_: &CArc<Type>) -> u64 {
    with_context(|context| context.type_hash(type_))
}