use std::ffi::CString;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Index;
use std::sync::atomic::{AtomicI64};

use serde::Serialize;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CSlice<'a, T> {
    ptr: *const T,
    len: usize,
    phantom: PhantomData<&'a T>,
}
impl<'a, T: Serialize> Serialize for CSlice<'a, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let slice: &[T] = self.as_ref();
        slice.serialize(serializer)
    }
}
unsafe impl<'a, T: Send> Send for CSlice<'a, T> {}
unsafe impl<'a, T: Sync> Sync for CSlice<'a, T> {}
impl<'a, T: PartialEq> PartialEq for CSlice<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        let a: &[T] = self.as_ref();
        let b: &[T] = other.as_ref();
        a == b
    }
}
impl<'a, T: Eq> Eq for CSlice<'a, T> {}
impl<'a, T: Hash> Hash for CSlice<'a, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let a: &[T] = self.as_ref();
        a.hash(state);
    }
}
impl<'a, T: Debug> std::fmt::Debug for CSlice<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slice: &[T] = self.as_ref();
        slice.fmt(f)
    }
}
impl<'a, T> AsRef<[T]> for CSlice<'a, T> {
    fn as_ref(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl<'a, T> Index<usize> for CSlice<'a, T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_ref()[index]
    }
}
impl<'a, T> From<&'a [T]> for CSlice<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            phantom: PhantomData {},
        }
    }
}

#[repr(C)]
pub struct CSliceMut<'a, T> {
    ptr: *mut T,
    len: usize,
    phantom: PhantomData<&'a T>,
}
impl<'a, T> AsRef<[T]> for CSliceMut<'a, T> {
    fn as_ref(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl<'a, T> AsMut<[T]> for CSliceMut<'a, T> {
    fn as_mut(&mut self) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl<'a, T: Debug> std::fmt::Debug for CSliceMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slice: &[T] = self.as_ref();
        slice.fmt(f)
    }
}
#[repr(C)]
pub struct CArcSharedBlock<T> {
    ptr: *mut T,
    ref_count: AtomicI64,
    destructor: extern "C" fn(*mut CArcSharedBlock<T>),
}
impl<T> CArcSharedBlock<T> {
    pub fn retain(&self) {
        let old = self.ref_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        assert!(old > 0);
    }
    pub fn release(&self) {
        let old = self
            .ref_count
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        assert!(old > 0, "old: {}", old);
        let cur = self.ref_count.load(std::sync::atomic::Ordering::SeqCst);
        if cur == 0 {
            (self.destructor)(self as *const _ as *mut _);
        }
    }
}
#[repr(C)]
pub struct CArc<T> {
    inner: *mut CArcSharedBlock<T>,
}
unsafe impl<T: Send> Send for CArc<T> {}
unsafe impl<T: Sync> Sync for CArc<T> {}
impl<T: PartialEq> PartialEq for CArc<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.is_null() && other.is_null() {
            return true;
        }
        if self.is_null() || other.is_null() {
            return false;
        }
        let lhs: &T = self.as_ref();
        let rhs: &T = other.as_ref();
        lhs == rhs
    }
}
impl<T: Eq> Eq for CArc<T> {}
impl<T: std::fmt::Display> std::fmt::Display for CArc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T: Hash> Hash for CArc<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        assert!(!self.is_null());
        let data: &T = self.as_ref();
        data.hash(state);
    }
}
extern "C" fn default_destructor<T>(ptr: *mut T) {
    unsafe {
        std::mem::drop(Box::from_raw(ptr));
    }
}
extern "C" fn default_destructor_carc<T>(ptr: *mut CArcSharedBlock<T>) {
    unsafe {
        std::mem::drop(Box::from_raw((*ptr).ptr));
        std::mem::drop(Box::from_raw(ptr));
    }
}
impl<T: Debug> Debug for CArc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}
impl<T: Serialize> Serialize for CArc<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.as_ref().serialize(serializer)
    }
}
impl<T> CArc<T> {
    pub unsafe fn from_raw(inner: *mut CArcSharedBlock<T>) -> Self {
        Self { inner }
    }
    pub fn into_raw(ptr: CArc<T>) -> *mut CArcSharedBlock<T> {
        let inner = ptr.inner;
        std::mem::forget(ptr);
        inner
    }
    pub fn null() -> Self {
        Self {
            inner: std::ptr::null_mut(),
        }
    }
    pub fn is_null(&self) -> bool {
        self.inner.is_null()
    }
    pub fn new(value: T) -> Self {
        Self::new_with_dtor(Box::into_raw(Box::new(value)), default_destructor_carc::<T>)
    }
    pub fn new_with_dtor(value: *mut T, dtor: extern "C" fn(*mut CArcSharedBlock<T>)) -> Self {
        let inner = Box::into_raw(Box::new(CArcSharedBlock {
            ptr: value,
            ref_count: AtomicI64::new(1),
            destructor: dtor,
        }));
        Self { inner }
    }
}
impl<T> Clone for CArc<T> {
    fn clone(&self) -> Self {
        if self.is_null() {
            return Self::null();
        }
        unsafe {
            (*self.inner).retain();
        }
        Self { inner: self.inner }
    }
}
impl<T> Drop for CArc<T> {
    fn drop(&mut self) {
        if self.is_null() {
            return;
        }
        unsafe {
            (*self.inner).release();
        }
    }
}
impl<T> CArc<T> {
    pub fn as_ptr(&self) -> *const T {
        unsafe { (*self.inner).ptr }
    }
}
impl<T> std::ops::Deref for CArc<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        assert!(!self.is_null());
        unsafe { &*(*self.inner).ptr }
    }
}
impl<T> AsRef<T> for CArc<T> {
    fn as_ref(&self) -> &T {
        assert!(!self.is_null());
        unsafe { &mut *(*self.inner).ptr }
    }
}
#[repr(C)]
pub struct CBox<T> {
    ptr: *mut T,
    destructor: unsafe extern "C" fn(*mut T),
}
impl<T> CBox<T> {
    pub fn new(value: T) -> Self {
        let ptr = Box::into_raw(Box::new(value));
        Self {
            ptr,
            destructor: default_destructor::<T>,
        }
    }
}
impl<T> AsRef<T> for CBox<T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.ptr }
    }
}

impl<T> Drop for CBox<T> {
    fn drop(&mut self) {
        unsafe { (self.destructor)(self.ptr) }
    }
}
#[repr(C)]
pub struct CBoxedSlice<T> {
    ptr: *mut T,
    len: usize,
    destructor: unsafe extern "C" fn(*mut T, usize),
}
impl<T: Clone> From<&[T]> for CBoxedSlice<T> {
    fn from(slice: &[T]) -> Self {
        let v = slice.to_vec();
        Self::new(v)
    }
}
impl<T: Eq> Eq for CBoxedSlice<T> {}
impl<T: PartialEq> PartialEq for CBoxedSlice<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl<T: Hash> Hash for CBoxedSlice<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state);
    }
}

impl<T: Serialize> Serialize for CBoxedSlice<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let slice: &[T] = self.as_ref();
        slice.serialize(serializer)
    }
}
impl<T: Clone> Clone for CBoxedSlice<T> {
    fn clone(&self) -> Self {
        let v: Vec<T> = self.as_ref().to_vec();
        Self::new(v)
    }
}
extern "C" fn default_destructor_slice<T>(ptr: *mut T, len: usize) {
    unsafe {
        let layout = std::alloc::Layout::array::<T>(len).unwrap();
        for i in 0..len {
            std::ptr::drop_in_place(ptr.add(i));
        }
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}
impl<T> CBoxedSlice<T> {
    pub fn new(value: Vec<T>) -> Self {
        let mut value = value;
        let len = value.len();
        unsafe {
            let layout = std::alloc::Layout::array::<T>(len).unwrap();
            let ptr = std::alloc::alloc(layout) as *mut T;
            for i in 0..len {
                std::ptr::write(ptr.add(i), std::ptr::read(&value[i]));
            }
            value.set_len(0);
            Self {
                ptr,
                len,
                destructor: default_destructor_slice::<T>,
            }
        }
    }
}
unsafe impl<T: Send> Send for CBoxedSlice<T> {}
unsafe impl<T: Sync> Sync for CBoxedSlice<T> {}
impl<T> AsRef<[T]> for CBoxedSlice<T> {
    fn as_ref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl<T> Drop for CBoxedSlice<T> {
    fn drop(&mut self) {
        unsafe { (self.destructor)(self.ptr, self.len) }
    }
}
impl<T: Debug> Debug for CBoxedSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slice: &[T] = self.as_ref();
        slice.fmt(f)
    }
}
impl From<CString> for CBoxedSlice<u8> {
    fn from(s: CString) -> Self {
        let bytes = s.into_bytes_with_nul();
        Self::new(bytes)
    }
}
impl From<CBoxedSlice<u8>> for CString {
    fn from(s: CBoxedSlice<u8>) -> Self {
        let bytes = s.as_ref().to_vec();
        CString::new(bytes).unwrap()
    }
}

struct _TestSize(CArcSharedBlock<i32>);
