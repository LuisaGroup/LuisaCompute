#![allow(unused_unsafe)]

#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
pub mod binding;
use core::panic;
use std::{ffi::CStr, sync::Once};

pub use binding::*;
use libloading::Library;

pub mod convert;
pub type TypeTag = RustyTypeTag;
pub type InstructionTag = RustyInstructionTag;
pub type FuncTag = RustyFuncTag;
pub type BindingTag = RustyBindingTag;

#[allow(dead_code)]
struct Lib {
    binding: IrV2BindingTable,
    lib: Library,
}
static mut LIB: Option<Lib> = None;
static LIB_INIT: Once = Once::new();
pub fn init<P: AsRef<std::path::Path>>(path: P) {
    assert!(!LIB_INIT.is_completed());
    LIB_INIT.call_once(|| unsafe {
        let lib = Library::new(path.as_ref()).unwrap();
        let load_binding = lib
            .get::<extern "C" fn() -> IrV2BindingTable>(b"lc_ir_v2_binding_table\0")
            .unwrap();
        let binding = (load_binding)();
        LIB = Some(Lib { binding, lib });
    });
}
pub fn binding_table() -> &'static IrV2BindingTable {
    unsafe { &LIB.as_ref().unwrap().binding }
}
#[macro_export]
macro_rules! call {
    ($name:ident, $($args:expr),*) => {
        unsafe {
            (binding_table().$name.unwrap())($($args),*)
        }
    };
}
impl std::fmt::Debug for Slice<::std::os::raw::c_char> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = unsafe {
            let slice = std::slice::from_raw_parts(self.data as *const u8, self.len);
            std::ffi::CStr::from_bytes_with_nul(slice).unwrap()
        };
        write!(f, "{:?}", s)
    }
}

impl<T> From<Slice<T>> for &[T] {
    fn from(s: Slice<T>) -> Self {
        unsafe { std::slice::from_raw_parts(s.data, s.len) }
    }
}
impl std::ops::Deref for BasicBlockRefMut {
    type Target = BasicBlockRef;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const BasicBlockRefMut as *const BasicBlockRef) }
    }
}
impl std::ops::Deref for NodeRefMut {
    type Target = NodeRef;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const NodeRefMut as *const NodeRef) }
    }
}
impl std::ops::Deref for InstructionRefMut {
    type Target = InstructionRef;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const InstructionRefMut as *const InstructionRef) }
    }
}
impl std::ops::Deref for FuncRefMut {
    type Target = FuncRef;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const FuncRefMut as *const FuncRef) }
    }
}
impl std::ops::Deref for BindingRefMut {
    type Target = BindingRef;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const BindingRefMut as *const BindingRef) }
    }
}
impl std::ops::Deref for ModuleRefMut {
    type Target = ModuleRef;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const ModuleRefMut as *const ModuleRef) }
    }
}
impl std::ops::Deref for CallableModuleRefMut {
    type Target = CallableModuleRef;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const CallableModuleRefMut as *const CallableModuleRef) }
    }
}
impl std::ops::Deref for KernelModuleRefMut {
    type Target = KernelModuleRef;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const KernelModuleRefMut as *const KernelModuleRef) }
    }
}
impl BasicBlockRef {
    #[inline]
    pub fn as_mut(&self) -> BasicBlockRefMut {
        BasicBlockRefMut(self.0 as *mut _)
    }
    #[inline]
    pub fn iter(&self) -> BasicBlockIter {
        BasicBlockIter::new(*self)
    }
    #[inline]
    pub fn first(&self) -> NodeRef {
        NodeRef(call!(basic_block_first, self.0))
    }
    #[inline]
    pub fn last(&self) -> NodeRef {
        NodeRef(call!(basic_block_last, self.0))
    }
    #[inline]
    pub fn phis(&self) -> Vec<NodeRef> {
        self.iter()
            .filter(|node| {
                let inst = node.inst();
                let tag = inst.tag();
                tag == InstructionTag::Phi
            })
            .collect::<Vec<_>>()
    }
}
#[allow(dead_code)]
pub struct BasicBlockIter {
    bb: BasicBlockRef,
    first: NodeRef,
    last: NodeRef,
    current: NodeRef,
}
impl BasicBlockIter {
    #[inline]
    pub fn new(bb: BasicBlockRef) -> Self {
        let first = bb.first();
        let last = bb.last();
        Self {
            bb,
            first,
            last,
            current: first,
        }
    }
}
impl Iterator for BasicBlockIter {
    type Item = NodeRef;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.last {
            None
        } else {
            let ret = self.current;
            self.current = self.current.next();
            Some(ret)
        }
    }
}
impl NodeRef {
    #[inline]
    pub fn as_mut(&self) -> NodeRefMut {
        NodeRefMut(self.0 as *mut _)
    }
    #[inline]
    pub fn prev(&self) -> NodeRef {
        NodeRef(call!(node_prev, self.0))
    }
    #[inline]
    pub fn next(&self) -> NodeRef {
        NodeRef(call!(node_next, self.0))
    }
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn inst(&self) -> InstructionRef {
        InstructionRef(call!(node_inst, self.0))
    }
    #[inline]
    pub fn type_(self) -> TypeRef {
        TypeRef(call!(node_type, self.0))
    }
    #[inline]
    pub fn get_index(self) -> i32 {
        call!(node_get_index, self.0)
    }
}

impl InstructionRef {
    #[inline]
    pub fn tag(&self) -> InstructionTag {
        call!(Instruction_tag, self.0)
    }
    #[inline]
    pub fn as_mut(&self) -> InstructionRefMut {
        InstructionRefMut(self.0 as *mut _)
    }
    #[inline]
    pub fn as_phi(&self) -> PhiInstRef {
        assert_eq!(self.tag(), InstructionTag::Phi);
        PhiInstRef(call!(Instruction_as_PhiInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_if(&self) -> IfInstRef {
        assert_eq!(self.tag(), InstructionTag::If);
        IfInstRef(call!(Instruction_as_IfInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_call(&self) -> CallInstRef {
        assert_eq!(self.tag(), InstructionTag::Call);
        CallInstRef(call!(Instruction_as_CallInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_return(&self) -> ReturnInstRef {
        assert_eq!(self.tag(), InstructionTag::Return);
        ReturnInstRef(call!(Instruction_as_ReturnInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_generic_loop(&self) -> GenericLoopInstRef {
        assert_eq!(self.tag(), InstructionTag::GenericLoop);
        GenericLoopInstRef(call!(Instruction_as_GenericLoopInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_local(&self) -> LocalInstRef {
        assert_eq!(self.tag(), InstructionTag::Local);
        LocalInstRef(call!(Instruction_as_LocalInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_switch(&self) -> SwitchInstRef {
        assert_eq!(self.tag(), InstructionTag::Switch);
        SwitchInstRef(call!(Instruction_as_SwitchInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_ray_query(&self) -> RayQueryInstRef {
        assert_eq!(self.tag(), InstructionTag::RayQuery);
        RayQueryInstRef(call!(Instruction_as_RayQueryInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_fwd_autodiff(&self) -> FwdAutodiffInstRef {
        assert_eq!(self.tag(), InstructionTag::FwdAutodiff);
        FwdAutodiffInstRef(call!(Instruction_as_FwdAutodiffInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_rev_autodiff(&self) -> RevAutodiffInstRef {
        assert_eq!(self.tag(), InstructionTag::RevAutodiff);
        RevAutodiffInstRef(call!(Instruction_as_RevAutodiffInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_constant(&self) -> ConstantInstRef {
        assert_eq!(self.tag(), InstructionTag::Constant);
        ConstantInstRef(call!(Instruction_as_ConstantInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_update(&self) -> UpdateInstRef {
        assert_eq!(self.tag(), InstructionTag::Update);
        UpdateInstRef(call!(Instruction_as_UpdateInst, self.0 as *mut _))
    }
    #[inline]
    pub fn as_comment(&self) -> CommentInstRef {
        assert_eq!(self.tag(), InstructionTag::Comment);
        CommentInstRef(call!(Instruction_as_CommentInst, self.0 as *mut _))
    }
}

impl PhiInstRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn incomings(&self) -> &[PhiIncoming] {
        let incomings = call!(PhiInst_incomings, self.0 as *mut _);
        unsafe { std::slice::from_raw_parts(incomings.data as *const PhiIncoming, incomings.len) }
    }
}
impl IfInstRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn cond(&self) -> NodeRef {
        NodeRef(call!(IfInst_cond, self.0 as *mut _))
    }
    #[inline]
    pub fn true_branch(&self) -> BasicBlockRef {
        BasicBlockRef(call!(IfInst_true_branch, self.0 as *mut _))
    }
    #[inline]
    pub fn false_branch(&self) -> BasicBlockRef {
        BasicBlockRef(call!(IfInst_false_branch, self.0 as *mut _))
    }
}
impl CallInstRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn func(&self) -> FuncRef {
        FuncRef(call!(CallInst_func, self.0 as *mut _))
    }
    #[inline]
    pub fn args(&self) -> &[NodeRef] {
        let arg = call!(CallInst_args, self.0 as *mut _);
        unsafe { std::slice::from_raw_parts(arg.data as *const NodeRef, arg.len) }
    }
}
impl ReturnInstRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn value(&self) -> NodeRef {
        NodeRef(call!(ReturnInst_value, self.0 as *mut _))
    }
}
impl GenericLoopInstRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn prepare(&self) -> BasicBlockRef {
        BasicBlockRef(call!(GenericLoopInst_prepare, self.0 as *mut _))
    }
    #[inline]
    pub fn cond(&self) -> NodeRef {
        NodeRef(call!(GenericLoopInst_cond, self.0 as *mut _))
    }
    #[inline]
    pub fn body(&self) -> BasicBlockRef {
        BasicBlockRef(call!(GenericLoopInst_body, self.0 as *mut _))
    }
    #[inline]
    pub fn update(&self) -> BasicBlockRef {
        BasicBlockRef(call!(GenericLoopInst_update, self.0 as *mut _))
    }
}
impl LocalInstRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn init(&self) -> NodeRef {
        NodeRef(call!(LocalInst_init, self.0 as *mut _))
    }
}
impl UpdateInstRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn value(&self) -> NodeRef {
        NodeRef(call!(UpdateInst_value, self.0 as *mut _))
    }
    #[inline]
    pub fn var(&self) -> NodeRef {
        NodeRef(call!(UpdateInst_var, self.0 as *mut _))
    }
}
impl SwitchInstRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn value(&self) -> NodeRef {
        NodeRef(call!(SwitchInst_value, self.0 as *mut _))
    }
    #[inline]
    pub fn cases(&self) -> &[SwitchCase] {
        let cases = call!(SwitchInst_cases, self.0 as *mut _);
        unsafe { std::slice::from_raw_parts(cases.data as *const SwitchCase, cases.len) }
    }
    #[inline]
    pub fn default(&self) -> BasicBlockRef {
        BasicBlockRef(call!(SwitchInst_default_, self.0 as *mut _))
    }
}
impl RayQueryInstRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn on_triangle_hit(&self) -> BasicBlockRef {
        BasicBlockRef(call!(RayQueryInst_on_triangle_hit, self.0 as *mut _))
    }
    #[inline]
    pub fn on_procedural_hit(&self) -> BasicBlockRef {
        BasicBlockRef(call!(RayQueryInst_on_procedural_hit, self.0 as *mut _))
    }
    #[inline]
    pub fn ray_query(&self) -> NodeRef {
        NodeRef(call!(RayQueryInst_query, self.0 as *mut _))
    }
}
impl FwdAutodiffInstRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn body(&self) -> BasicBlockRef {
        BasicBlockRef(call!(FwdAutodiffInst_body, self.0 as *mut _))
    }
}
impl RevAutodiffInstRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn body(&self) -> BasicBlockRef {
        BasicBlockRef(call!(RevAutodiffInst_body, self.0 as *mut _))
    }
}
impl TypeRef {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    #[inline]
    pub fn tag(&self) -> TypeTag {
        call!(type_tag, self.0)
    }
    #[inline]
    pub fn is_bool(&self) -> bool {
        self.tag() == TypeTag::Bool
    }
    #[inline]
    pub fn is_f16(&self) -> bool {
        self.tag() == TypeTag::Float16
    }
    #[inline]
    pub fn is_f32(&self) -> bool {
        self.tag() == TypeTag::Float32
    }
    #[inline]
    pub fn is_f64(&self) -> bool {
        self.tag() == TypeTag::Float64
    }
    #[inline]
    pub fn is_i16(&self) -> bool {
        self.tag() == TypeTag::Int16
    }
    #[inline]
    pub fn is_i32(&self) -> bool {
        self.tag() == TypeTag::Int32
    }
    #[inline]
    pub fn is_i64(&self) -> bool {
        self.tag() == TypeTag::Int64
    }
    #[inline]
    pub fn is_u16(&self) -> bool {
        self.tag() == TypeTag::Uint16
    }
    #[inline]
    pub fn is_u32(&self) -> bool {
        self.tag() == TypeTag::Uint32
    }
    #[inline]
    pub fn is_u64(&self) -> bool {
        self.tag() == TypeTag::Uint64
    }
    #[inline]
    pub fn is_vector(&self) -> bool {
        self.tag() == TypeTag::Vector
    }
    #[inline]
    pub fn is_matrix(&self) -> bool {
        self.tag() == TypeTag::Matrix
    }
    #[inline]
    pub fn is_array(&self) -> bool {
        self.tag() == TypeTag::Array
    }
    #[inline]
    pub fn is_struct(&self) -> bool {
        self.tag() == TypeTag::Struct
    }
    #[inline]
    pub fn is_custom(&self) -> bool {
        self.tag() == TypeTag::Custom
    }
    #[inline]
    pub fn is_void(&self) -> bool {
        self.is_null()
    }
    #[inline]
    pub fn is_scalar(&self) -> bool {
        call!(type_is_scalar, self.0)
    }
    #[inline]
    pub fn is_float(&self) -> bool {
        self.is_f16() || self.is_f32() || self.is_f64()
    }
    #[inline]
    pub fn is_int(&self) -> bool {
        self.is_i16()
            || self.is_i32()
            || self.is_i64()
            || self.is_u16()
            || self.is_u32()
            || self.is_u64()
    }
    #[inline]
    pub fn description(&self) -> String {
        assert!(!self.is_null());
        unsafe {
            let desc = call!(type_description, self.0);
            CStr::from_bytes_until_nul(std::slice::from_raw_parts(desc.data as *mut _, desc.len))
                .unwrap()
                .to_str()
                .unwrap()
                .to_string()
        }
    }
    #[inline]
    pub fn element(&self) -> TypeRef {
        assert!(self.is_vector() || self.is_matrix() || self.is_array());
        TypeRef(call!(type_element, self.0))
    }
    #[inline]
    pub fn members(&self) -> &[TypeRef] {
        assert!(self.is_struct());
        let members = call!(type_members, self.0);
        unsafe { std::slice::from_raw_parts(members.data as *const TypeRef, members.len) }
    }
    #[inline]
    pub fn dimension(&self) -> u32 {
        assert!(self.is_vector() || self.is_matrix() || self.is_array());
        call!(type_dimension, self.0) as u32
    }
    #[inline]
    pub fn alignment(&self) -> usize {
        assert!(self.is_struct());
        call!(type_alignment, self.0)
    }
    #[inline]
    pub fn extract(&self, index: u32) -> TypeRef {
        assert!(self.is_vector() || self.is_matrix() || self.is_array());
        TypeRef(call!(type_extract, self.0, index))
    }
    #[inline]
    pub fn vector(el: TypeRef, len: u32) -> TypeRef {
        TypeRef(call!(make_vector, el.0, len))
    }
    #[inline]
    pub fn array(el: TypeRef, len: u32) -> TypeRef {
        TypeRef(call!(make_array, el.0, len))
    }
    #[inline]
    pub fn matrix(dim: u32) -> TypeRef {
        TypeRef(call!(make_matrix, dim))
    }
    #[inline]
    pub fn struct_(align: usize, members: &[TypeRef]) -> TypeRef {
        let members = Slice {
            data: members.as_ptr() as *mut _,
            len: members.len(),
            _phantom_0: std::marker::PhantomData,
        };
        TypeRef(call!(make_struct, align, members.data, members.len as u32))
    }
}
impl FuncRef {
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    pub fn tag(&self) -> FuncTag {
        call!(Func_tag, self.0)
    }
    pub fn as_assert(&self) -> AssertFnRef {
        assert_eq!(self.tag(), FuncTag::Assert);
        AssertFnRef(call!(Func_as_AssertFn, self.0 as *mut _) as *const _)
    }
    pub fn as_unreachable(&self) -> UnreachableFnRef {
        assert_eq!(self.tag(), FuncTag::Unreachable);
        UnreachableFnRef(call!(Func_as_UnreachableFn, self.0 as *mut _) as *const _)
    }
    pub fn as_cpu_ext(&self) -> CpuExtFnRef {
        assert_eq!(self.tag(), FuncTag::CpuExt);
        CpuExtFnRef(call!(Func_as_CpuExtFn, self.0 as *mut _) as *const _)
    }
    pub fn as_callable(&self) -> CallableFnRef {
        assert_eq!(self.tag(), FuncTag::Callable);
        CallableFnRef(call!(Func_as_CallableFn, self.0 as *mut _) as *const _)
    }
}
impl AssertFnRef {
    pub fn message(&self) -> String {
        let msg = call!(AssertFn_msg, self.0 as *mut _);
        slice_i8_to_string(msg)
    }
}
impl UnreachableFnRef {
    pub fn message(&self) -> String {
        let msg = call!(UnreachableFn_msg, self.0 as *mut _);
        slice_i8_to_string(msg)
    }
}
#[repr(transparent)]
pub struct CpuExternFnRef(*const CpuExternFn);
impl CpuExternFnRef {
    pub fn new(data: CpuExternFnData) -> Self {
        Self(call!(cpu_ext_fn_new, data))
    }
    pub unsafe fn from_raw(ptr: *const CpuExternFn) -> Self {
        Self(call!(cpu_ext_fn_clone, ptr))
    }
    pub fn as_ptr(&self) -> *const CpuExternFn {
        self.0
    }
}

impl std::ops::Deref for CpuExternFnRef {
    type Target = CpuExternFnData;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { call!(cpu_ext_fn_data, self.0).as_ref().unwrap() }
    }
}
impl CpuExtFnRef {
    pub fn func(&self) -> CpuExternFnRef {
        unsafe { CpuExternFnRef::from_raw(call!(CpuExtFn_f, self.0 as *mut _)) }
    }
}
impl Drop for CpuExternFnRef {
    fn drop(&mut self) {
        call!(cpu_ext_fn_drop, self.0);
    }
}
impl Clone for CpuExternFnRef {
    fn clone(&self) -> Self {
        Self(call!(cpu_ext_fn_clone, self.0))
    }
}
impl std::ops::Deref for PoolRefMut {
    type Target = PoolRef;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const PoolRefMut as *const PoolRef) }
    }
}
impl Clone for PoolRefMut {
    fn clone(&self) -> Self {
        Self(call!(pool_clone, self.0))
    }
}
impl Drop for PoolRefMut {
    fn drop(&mut self) {
        panic!("don't call drop(), call release() instead");
    }
}
impl Drop for ModuleRefMut {
    fn drop(&mut self) {
        panic!("don't call drop(), call release() instead");
    }
}

impl PoolRefMut {
    /// the only unsafe method since it drops the **entire** pool
    /// If there is any segfault, it happens here
    pub unsafe fn release(self) {
        call!(pool_drop, self.0);
    }
}
impl PoolRef {
    /// the only unsafe method since it drops the **entire** pool
    /// If there is any segfault, it happens here
    pub unsafe fn release(self) {
        call!(pool_drop, self.0 as *mut _);
    }
}

impl IrBuilderRefMut {
    #[inline]
    pub fn finish(self) -> BasicBlockRef {
        let bb = call!(ir_builder_finish, self.0);
        call!(ir_builder_drop, self.0);
        BasicBlockRef(bb)
    }
    #[inline]
    pub fn call(&self, tag: FuncTag, args: &[NodeRef], ty: TypeRef) -> NodeRef {
        let args = Slice {
            data: args.as_ptr() as *mut _,
            len: args.len(),
            _phantom_0: std::marker::PhantomData,
        };
        NodeRef(call!(ir_build_call_tag, self.0, tag, args, ty.0))
    }
    #[inline]
    pub fn call_ex(&self, f: CFunc, args: &[NodeRef], ty: TypeRef) -> NodeRef {
        let args = Slice {
            data: args.as_ptr() as *mut _,
            len: args.len(),
            _phantom_0: std::marker::PhantomData,
        };
        let node = NodeRef(call!(
            ir_build_call,
            self.0,
            &f as *const _ as *mut _,
            args,
            ty.0
        ));
        std::mem::forget(f);
        node
    }
}
impl Drop for IrBuilderRefMut {
    fn drop(&mut self) {
        panic!("Don't drop IrBuilderRefMut");
    }
}
impl Drop for CFunc {
    fn drop(&mut self) {
        panic!("Don't drop");
    }
}
impl Drop for CInstruction {
    fn drop(&mut self) {
        panic!("Don't drop");
    }
}
impl Drop for CBinding {
    fn drop(&mut self) {
        panic!("Don't drop");
    }
}

impl ConstantInstRef {
    pub fn data(&self) -> &[u8] {
        let data = call!(ConstantInst_value, self.0 as *mut _);
        unsafe { std::slice::from_raw_parts(data.data as *const u8, data.len) }
    }
    pub fn ty(&self) -> TypeRef {
        TypeRef(call!(ConstantInst_ty, self.0 as *mut _))
    }
}
impl CommentInstRef {
    pub fn comment(&self) -> String {
        let comment = call!(CommentInst_comment, self.0 as *mut _);
        slice_i8_to_string(comment)
    }
}

fn slice_i8_to_string(slice: Slice<i8>) -> String {
    unsafe {
        CStr::from_bytes_with_nul(std::slice::from_raw_parts(slice.data as *mut _, slice.len))
            .unwrap()
            .to_str()
            .unwrap()
            .to_string()
    }
}
