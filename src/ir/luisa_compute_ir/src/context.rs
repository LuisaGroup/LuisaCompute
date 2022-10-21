use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

use bumpalo::Bump;
use gc::Gc;

use crate::ir::{CallableModule, Node, NodeRef, Type};

pub struct Context {
    arena: Bump,
    // pub(crate) nodes: Vec<Gc<Node>>,
    pub(crate) types: HashSet<Gc<Type>>,
    pub(crate) symbols: HashMap<u64, CallableModule>,
}

impl Context {
    pub fn add_symbol(&mut self, id: u64, symbol: CallableModule) {
        self.symbols.insert(id, symbol);
    }
    pub fn get_symbol(&self, id: u64) -> Option<&CallableModule> {
        self.symbols.get(&id)
    }
    pub fn register_type(&mut self, type_: Type) -> Gc<Type> {
        let type_ = Gc::new(type_);
        if let Some(type_) = self.types.get(&type_) {
            *type_
        } else {
            let r = unsafe { std::mem::transmute(&*type_) };
            self.types.insert(type_);
            r
        }
    }

    pub fn alloc_node(&mut self, node: Node) -> NodeRef {
        let node = Gc::new(node);
        unsafe{ NodeRef(std::mem::transmute(node)) }
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

pub fn register_type(type_: Type) -> Gc<Type> {
    with_context(|context| context.register_type(type_))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_register_type(type_: Type) -> Gc<Type> {
    register_type(type_)
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_add_symbol(id: u64, m: CallableModule) {
    with_context(|context| context.add_symbol(id, m));
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_get_symbol(id: u64) -> *const CallableModule {
    with_context(|context| {
        context
            .get_symbol(id)
            .map(|m| m as *const CallableModule)
            .unwrap_or(std::ptr::null())
    })
}
