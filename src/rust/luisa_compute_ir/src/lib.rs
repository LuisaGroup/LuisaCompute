pub mod ffi;
pub mod ir;
pub use ffi::*;
use serde::Serialize;
pub mod codegen;
use std::{cell::RefCell, collections::HashMap, hash::Hash, rc::Rc};
pub mod context;
mod display;
pub mod transform;

use ir::{ArrayType, Primitive, Type};

pub trait TypeOf {
    fn type_() -> CArc<Type>;
}

impl TypeOf for bool {
    fn type_() -> CArc<Type> {
        context::register_type(Type::Primitive(Primitive::Bool))
    }
}
impl TypeOf for u32 {
    fn type_() -> CArc<Type> {
        context::register_type(Type::Primitive(Primitive::Uint32))
    }
}
impl TypeOf for i32 {
    fn type_() -> CArc<Type> {
        context::register_type(Type::Primitive(Primitive::Int32))
    }
}
impl TypeOf for f32 {
    fn type_() -> CArc<Type> {
        context::register_type(Type::Primitive(Primitive::Float32))
    }
}
impl TypeOf for f64 {
    fn type_() -> CArc<Type> {
        context::register_type(Type::Primitive(Primitive::Float64))
    }
}
impl TypeOf for i64 {
    fn type_() -> CArc<Type> {
        context::register_type(Type::Primitive(Primitive::Int64))
    }
}
impl TypeOf for u64 {
    fn type_() -> CArc<Type> {
        context::register_type(Type::Primitive(Primitive::Uint64))
    }
}
impl<T: TypeOf, const N: usize> TypeOf for [T; N] {
    fn type_() -> CArc<Type> {
        context::register_type(Type::Array(ArrayType {
            element: T::type_(),
            length: N,
        }))
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
#[derive(Debug)]
pub struct Pool<T> {
    chunks: RefCell<Vec<(*mut T, usize, usize)>>,
}
#[derive(Debug)]
#[repr(C)]
pub struct Pooled<T> {
    pub(crate) ptr: *mut T,
}
impl<T: Serialize> Serialize for Pooled<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.get().serialize(serializer)
    }
}
impl<T> Copy for Pooled<T> {}
impl<T> Clone for Pooled<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Pooled<T> {
    pub fn get(&self) -> &T {
        unsafe { &*self.ptr }
    }
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr }
    }
    pub fn into_raw(self) -> *mut T {
        self.ptr
    }
}
impl<T> AsRef<T> for Pooled<T> {
    fn as_ref(&self) -> &T {
        self.get()
    }
}
impl<T> AsMut<T> for Pooled<T> {
    fn as_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}
impl<T> std::ops::Deref for Pooled<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.get()
    }
}
impl<T> Pool<T> {
    fn alloc_chunk(&self, size: usize) {
        let mut chunks = self.chunks.borrow_mut();
        let chunk =
            unsafe { std::alloc::alloc(std::alloc::Layout::array::<T>(size).unwrap()) as *mut T };
        chunks.push((chunk, 0, size));
    }
    pub fn alloc(&self, node: T) -> Pooled<T> {
        loop {
            let mut chunks = self.chunks.borrow_mut();
            if !chunks.is_empty() {
                let (chunk, offset, size) = chunks.last_mut().unwrap();
                if *offset + 1 < *size {
                    unsafe {
                        let ptr = chunk.add(*offset);
                        std::ptr::write(ptr, node);
                        *offset += 1;
                        return Pooled { ptr };
                    }
                }
            }
            drop(chunks);
            self.alloc_chunk(1024);
        }
    }
    pub fn new() -> Self {
        let pool = Self {
            chunks: RefCell::new(Vec::new()),
        };
        pool.alloc_chunk(1024);
        pool
    }
}
unsafe impl<T: Send> Send for Pool<T> {}
impl<T> Drop for Pool<T> {
    fn drop(&mut self) {
        let chunks = self.chunks.borrow();
        for (chunk, count, size) in chunks.iter() {
            for i in 0..*count {
                unsafe {
                    let ptr = chunk.add(i);
                    std::ptr::drop_in_place(ptr);
                }
            }
            unsafe {
                let layout = std::alloc::Layout::array::<T>(*size).unwrap();
                std::alloc::dealloc(*chunk as *mut u8, layout);
            }
        }
    }
}
