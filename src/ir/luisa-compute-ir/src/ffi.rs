use std::ffi::CString;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Index;

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
pub struct CRcSharedBlock<T> {
    ptr: T,
    ref_count: usize,
    destructor: extern "C" fn(*mut T),
}
#[repr(C)]
pub struct CRc<T> {
    inner: *mut CRcSharedBlock<T>,
}
extern "C" fn default_destructor<T>(ptr: *mut T) {
    unsafe {
        std::mem::drop(Box::from_raw(ptr));
    }
}
impl<T: Debug> Debug for CRc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}
impl<T: Serialize> Serialize for CRc<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.as_ref().serialize(serializer)
    }
}
impl<T> CRc<T> {
    pub fn new(value: T) -> Self {
        let inner = Box::into_raw(Box::new(CRcSharedBlock {
            ptr: value,
            ref_count: 1,
            destructor: default_destructor::<T>,
        }));
        Self { inner }
    }
}
impl<T> Clone for CRc<T> {
    fn clone(&self) -> Self {
        unsafe {
            (*self.inner).ref_count += 1;
        }
        Self { inner: self.inner }
    }
}
impl<T> Drop for CRc<T> {
    fn drop(&mut self) {
        unsafe {
            (*self.inner).ref_count -= 1;
            if (*self.inner).ref_count == 0 {
                std::mem::drop(Box::from_raw(self.inner));
            }
        }
    }
}
impl<T> std::ops::Deref for CRc<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &(*self.inner).ptr }
    }
}
impl<T> AsRef<T> for CRc<T> {
    fn as_ref(&self) -> &T {
        unsafe { &(*self.inner).ptr }
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
        let len = value.len();
        unsafe {
            let layout = std::alloc::Layout::array::<T>(len).unwrap();
            let ptr = std::alloc::alloc(layout) as *mut T;
            for i in 0..len {
                std::ptr::write(ptr.add(i), std::ptr::read(&value[i]));
            }
            Self {
                ptr,
                len,
                destructor: default_destructor_slice::<T>,
            }
        }
    }
}
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
