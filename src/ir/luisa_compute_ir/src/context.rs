use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
};

use gc::{Gc, GcObject, Trace};
use parking_lot::RwLock;

use crate::ir::{CallableModule, Node, NodeRef, Type};

pub struct Context {
    pub(crate) types: RwLock<HashSet<Gc<Type>>>,
    pub(crate) symbols: RwLock<HashMap<u64, Gc<CallableModule>>>,
}

impl Context {
    pub fn add_symbol(&self, id: u64, symbol: Gc<CallableModule>) {
        self.symbols.write().insert(id, symbol);
    }
    pub fn get_symbol(&self, id: u64) -> Option<Gc<CallableModule>> {
        self.symbols.read().get(&id).cloned()
    }
    pub fn register_type(&self, type_: Type) -> Gc<Type> {
        let type_ = Gc::new(type_);
        let types = self.types.read();
        if let Some(type_) = types.get(&type_) {
            *type_
        } else {
            drop(types);
            let mut types = self.types.write();
            let r = unsafe { std::mem::transmute(&*type_) };
            types.insert(type_);
            r
        }
    }

    pub fn alloc_node(&self, node: Node) -> NodeRef {
        let node = Gc::new(node);
        unsafe { NodeRef(std::mem::transmute(node)) }
    }
}
impl Trace for Context {
    fn trace(&self) {
        for t in self.types.read().iter() {
            t.trace();
        }
        for c in self.symbols.read().values() {
            c.trace();
        }
    }
}
static CONTEXT: RwLock<Option<Gc<Context>>> = RwLock::new(None);

pub fn with_context<T>(f: impl FnOnce(&Context) -> T) -> T {
    let ctx = CONTEXT.read();
    f(ctx.as_ref().unwrap())
}

pub unsafe fn reset_context() {
    let mut ctx = CONTEXT.write();
    *ctx = None;
}

pub fn register_type(type_: Type) -> Gc<Type> {
    with_context(|context| context.register_type(type_))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_register_type(type_: Type) -> *mut GcObject<Type> {
    Gc::into_raw(register_type(type_))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_add_symbol(id: u64, m: Gc<CallableModule>) {
    with_context(|context| context.add_symbol(id, m));
}

#[no_mangle]
pub unsafe extern "C" fn luisa_compute_ir_get_symbol(id: u64) -> *mut GcObject<CallableModule> {
    Gc::into_raw(with_context(|context| {
        context.get_symbol(id).unwrap_or(Gc::null())
    }))
}

#[no_mangle]
pub extern "C" fn luisa_compute_ir_context() -> *mut c_void {
    Gc::into_raw(CONTEXT.read().unwrap()) as *mut c_void
}
#[no_mangle]
pub extern "C" fn luisa_compute_ir_new_context() -> *mut c_void {
    Gc::into_raw(Gc::new(Context {
        types: RwLock::new(HashSet::new()),
        symbols: RwLock::new(HashMap::new()),
    })) as *mut c_void
}
#[no_mangle]
pub extern "C" fn luisa_compute_ir_set_context(ctx: *mut c_void) {
    let ctx = ctx as *mut GcObject<Context>;
    let ctx = unsafe { Gc::from_raw(ctx) };
    let mut context = CONTEXT.write();
    assert!(context.is_none());
    Gc::set_root(ctx);
    *context = Some(ctx)
}
