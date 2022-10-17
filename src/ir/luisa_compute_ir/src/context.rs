use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

use bumpalo::Bump;

use crate::ir::{Module, Node, NodeRef, Type};

pub struct Context {
    arena: Bump,
    destructors: Vec<Box<dyn FnOnce()>>,
    pub(crate) nodes: Vec<*mut Node>,
    pub(crate) types: HashSet<Box<Type>>,
    pub(crate) symbols: HashMap<u64, Module>,
}
impl Context {
    pub fn add_symbol(&mut self, id: u64, symbol: Module) {
        self.symbols.insert(id, symbol);
    }
    pub fn get_symbol(&self, id: u64) -> Option<&Module> {
        self.symbols.get(&id)
    }
    pub fn register_type(&mut self, type_: Type) -> &'static Type {
        let type_ = Box::new(type_);
        if let Some(type_) = self.types.get(&type_) {
            unsafe { std::mem::transmute(&**type_) }
        } else {
            let r = unsafe { std::mem::transmute(&*type_) };
            self.types.insert(type_);
            r
        }
    }
    pub fn alloc<T: Copy + 'static>(&mut self, value: T) -> &'static mut T {
        unsafe {
            let ptr = self.arena.alloc(value);
            std::mem::transmute(&mut *ptr)
        }
    }
    pub fn alloc_deferred_drop<T: 'static>(&mut self, value: T) -> &'static mut T {
        let r = self.arena.alloc(value);
        let ptr = r as *mut T;
        self.destructors.push(Box::new(move || unsafe {
            std::ptr::drop_in_place(ptr);
        }));
        unsafe { std::mem::transmute(r) }
    }
    pub fn alloc_node(&mut self, node: Node) -> NodeRef {
        let id = self.nodes.len();
        let r = self.alloc(node);
        self.nodes.push(r);
        NodeRef(id)
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        for destructor in self.destructors.drain(..).rev() {
            destructor();
        }
    }
}

thread_local! {
    static CONTEXT: RefCell<Option<Context>> = RefCell::new(None);
}

pub fn with_context<T>(f: impl FnOnce(&mut Context) -> T) -> T {
    CONTEXT.with(|context| {
        let mut context = context.borrow_mut();
        if context.is_none() {
            *context = Some(Context {
                arena: Bump::new(),
                destructors: Vec::new(),
                nodes: Vec::new(),
                types: HashSet::new(),
                symbols: HashMap::new(),
            });
        }
        f(context.as_mut().unwrap())
    })
}
pub unsafe fn reset_context() {
    CONTEXT.with(|context| {
        *context.borrow_mut() = None;
    });
}

pub fn register_type(type_: Type) -> &'static Type {
    with_context(|context| context.register_type(type_))
}
pub fn alloc<T: Copy + 'static>(value: T) -> &'static mut T {
    with_context(|context| context.alloc(value))
}
pub fn alloc_deferred_drop<T: 'static>(value: T) -> &'static mut T {
    with_context(|context| context.alloc_deferred_drop(value))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_add_symbol(id: u64, m: Module) {
    with_context(|context| context.add_symbol(id, m));
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_get_symbol(id: u64) -> *const Module {
    with_context(|context| context.get_symbol(id).map(|m| m as *const Module).unwrap_or(std::ptr::null()))
}
