pub mod ffi;
pub mod ir;
pub use ffi::*;
use std::{hash::Hash, collections::HashMap, rc::Rc};
pub mod context;
pub mod transform;
mod output;
mod display;

pub use gc::Gc;
use ir::{Primitive, Type};

pub trait TypeOf {
    fn type_() -> Gc<Type>;
}

impl TypeOf for bool {
    fn type_() -> Gc<Type> {
        context::register_type(Type::Primitive(Primitive::Bool))
    }
}
impl TypeOf for u32 {
    fn type_() -> Gc<Type> {
        context::register_type(Type::Primitive(Primitive::Uint32))
    }
}
impl TypeOf for i32 {
    fn type_() -> Gc<Type> {
        context::register_type(Type::Primitive(Primitive::Int32))
    }
}
impl TypeOf for f32 {
    fn type_() -> Gc<Type> {
        context::register_type(Type::Primitive(Primitive::Float32))
    }
}
impl TypeOf for f64 {
    fn type_() -> Gc<Type> {
        context::register_type(Type::Primitive(Primitive::Float64))
    }
}
impl TypeOf for i64 {
    fn type_() -> Gc<Type> {
        context::register_type(Type::Primitive(Primitive::Int64))
    }
}
impl TypeOf for u64 {
    fn type_() -> Gc<Type> {
        context::register_type(Type::Primitive(Primitive::Uint64))
    }
}

struct NestedHashMapInner<K: Hash, V> {
    map: HashMap<K, V>,
    parent: Option<Rc<NestedHashMapInner<K, V>>>,
}
impl<K: Hash + Eq, V> NestedHashMapInner<K, V> {
    pub(crate) fn get(&self, k: &K) -> Option<&V> {
        if let Some(v) = self.map.get(k) {
            Some(v)
        } else {
            if let Some(p) = &self.parent {
                p.get(k)
            } else {
                None
            }
        }
    }
    pub(crate) fn new() -> Self {
        Self {
            map: HashMap::new(),
            parent: None,
        }
    }
    pub(crate) fn from_parent(parent: Rc<NestedHashMapInner<K, V>>) -> Self {
        Self {
            map: HashMap::new(),
            parent: Some(parent),
        }
    }
}
#[allow(dead_code)]
pub(crate) struct NestedHashMap<K: Hash + Eq, V> {
    inner: Rc<NestedHashMapInner<K, V>>,
}
#[allow(dead_code)]
impl<K: Hash + Eq, V> NestedHashMap<K, V> {
    pub(crate) fn insert(&mut self, k: K, v: V) {
        unsafe {
            let inner = self.inner.as_ref() as *const _ as *mut NestedHashMapInner<K, V>;
            let inner = &mut *inner;
            inner.map.insert(k, v);
        }
    }
    pub(crate) fn get(&self, k: &K) -> Option<&V> {
        self.inner.get(k)
    }
    pub(crate) fn new() -> Self {
        Self {
            inner: Rc::new(NestedHashMapInner::new()),
        }
    }
    pub(crate) fn from_parent(parent: &NestedHashMap<K, V>) -> Self {
        Self {
            inner: Rc::new(NestedHashMapInner::from_parent(parent.inner.clone())),
        }
    }
}
#[allow(dead_code)]
pub(crate) struct NestedHashSet<K: Hash + Eq> {
    inner: Rc<NestedHashMapInner<K, ()>>,
}
#[allow(dead_code)]
impl<K: Hash + Eq> NestedHashSet<K> {
    pub(crate) fn insert(&mut self, k: K) {
        let inner = Rc::get_mut(&mut self.inner).unwrap();
        inner.map.insert(k, ());
    }
    pub(crate) fn contains(&self, k: &K) -> bool {
        self.inner.get(k).is_some()
    }
    pub(crate) fn new() -> Self {
        Self {
            inner: Rc::new(NestedHashMapInner::new()),
        }
    }
    pub(crate) fn from_parent(parent: &NestedHashSet<K>) -> Self {
        Self {
            inner: Rc::new(NestedHashMapInner::from_parent(parent.inner.clone())),
        }
    }
}
