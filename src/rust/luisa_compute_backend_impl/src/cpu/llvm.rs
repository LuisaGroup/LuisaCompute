#![allow(dead_code)]

use crate::panic_abort;
use lazy_static::lazy_static;
use libloading::Symbol;
use serde::{Deserialize, Serialize};
use std::env::{current_dir, current_exe, var};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::ops::Deref;
use std::process::{abort, exit};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::yield_now;
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{CStr, CString},
    path::{Path, PathBuf},
};

enum LLVMContext {}

enum LLVMModule {}

enum LLVMExecutionEngine {}

enum LLVMMemoryBuffer {}

enum LLVMOrcOpaqueThreadSafeModule {}

enum LLVMOrcOpaqueThreadSafeContext {}

enum LLVMOrcOpaqueLLJIT {}

enum LLVMOrcOpaqueLLJITBuilder {}

enum LLVMOrcOpaqueJITDylib {}

enum LLVMOpaqueError {}

enum LLVMOrcOpaqueDumpObjects {}

enum LLVMPassManager {}

enum LLVMOrcOpaqueObjectTransformLayer {}

enum LLVMOpaquePassBuilderOptions {}

enum LLVMOpaqueTargetMachine {}

enum LLVMTarget {}

type LLVMBool = i32;
type LLVMTargetRef = *mut LLVMTarget;
type LLVMMemoryBufferRef = *mut LLVMMemoryBuffer;
type LLVMContextRef = *mut LLVMContext;
type LLVMModuleRef = *mut LLVMModule;
type LLVMExecutionEngineRef = *mut LLVMExecutionEngine;
type LLVMOrcThreadSafeContextRef = *mut LLVMOrcOpaqueThreadSafeContext;
type LLVMOrcThreadSafeModuleRef = *mut LLVMOrcOpaqueThreadSafeModule;
type LLVMErrorRef = *mut LLVMOpaqueError;
type LLVMOrcLLJITRef = *mut LLVMOrcOpaqueLLJIT;
type LLVMOrcLLJITBuilderRef = *mut LLVMOrcOpaqueLLJITBuilder;
type LLVMOrcJITDylibRef = *mut LLVMOrcOpaqueJITDylib;
type LLVMOrcExecutorAddress = u64;
type LLVMOrcDumpObjectsRef = *mut LLVMOrcOpaqueDumpObjects;
type LLVMOrcObjectTransformLayerRef = *mut LLVMOrcOpaqueObjectTransformLayer;
type LLVMOrcObjectTransformLayerTransformFunction =
    extern "C" fn(Ctx: *mut c_void, ObjInOut: *mut LLVMMemoryBufferRef) -> LLVMErrorRef;
type LLVMPassManagerRef = *mut LLVMPassManager;
type LLVMPassBuilderOptionsRef = *mut LLVMOpaquePassBuilderOptions;
type LLVMTargetMachineRef = *mut LLVMOpaqueTargetMachine;

pub enum LLVMOrcOpaqueSymbolStringPoolEntry {}

pub type LLVMOrcSymbolStringPoolEntryRef = *mut LLVMOrcOpaqueSymbolStringPoolEntry;

#[repr(C)]
pub struct LLVMJITSymbolFlags {
    pub GenericFlags: u8,
    pub TargetFlags: u8,
}

#[repr(C)]
pub struct LLVMOrcCSymbolMapPair {
    pub Name: LLVMOrcSymbolStringPoolEntryRef,
    pub Sym: LLVMJITEvaluatedSymbol,
}

#[repr(C)]
pub struct LLVMJITEvaluatedSymbol {
    pub Address: LLVMOrcExecutorAddress,
    pub Flags: LLVMJITSymbolFlags,
}

pub type LLVMOrcCSymbolMapPairs = *mut LLVMOrcCSymbolMapPair;

pub enum LLVMOrcOpaqueMaterializationUnit {}

pub type LLVMOrcMaterializationUnitRef = *mut LLVMOrcOpaqueMaterializationUnit;

use crate::cpu::shader::ShaderImpl;
use crate::cpu::stream::ShaderDispatchContext;
use libc::{c_char, c_void, size_t};
use parking_lot::{Mutex, ReentrantMutex};

use super::shader::KernelFn;

#[repr(C)]
#[allow(non_camel_case_types)]
pub enum LLVMCodeGenOptLevel {
    LLVMCodeGenLevelNone,
    LLVMCodeGenLevelLess,
    LLVMCodeGenLevelDefault,
    LLVMCodeGenLevelAggressive,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub enum LLVMRelocMode {
    LLVMRelocDefault,
    LLVMRelocStatic,
    LLVMRelocPIC,
    LLVMRelocDynamicNoPic,
    LLVMRelocROPI,
    LLVMRelocRWPI,
    LLVMRelocROPI_RWPI,
}

#[repr(C)]
pub enum LLVMCodeModel {
    LLVMCodeModelDefault,
    LLVMCodeModelJITDefault,
    LLVMCodeModelTiny,
    LLVMCodeModelSmall,
    LLVMCodeModelKernel,
    LLVMCodeModelMedium,
    LLVMCodeModelLarge,
}

#[allow(non_snake_case)]
struct LibLLVM {
    lib: libloading::Library,
    LLVMOrcJITDylibDefine: Symbol<
        'static,
        unsafe extern "C" fn(
            JD: LLVMOrcJITDylibRef,
            MU: LLVMOrcMaterializationUnitRef,
        ) -> LLVMErrorRef,
    >,
    LLVMOrcLLJITMangleAndIntern: Symbol<
        'static,
        unsafe extern "C" fn(
            J: LLVMOrcLLJITRef,
            UnmangledName: *const c_char,
        ) -> LLVMOrcSymbolStringPoolEntryRef,
    >,
    LLVMOrcAbsoluteSymbols: Symbol<
        'static,
        unsafe extern "C" fn(
            Syms: LLVMOrcCSymbolMapPairs,
            NumPairs: usize,
        ) -> LLVMOrcMaterializationUnitRef,
    >,
    LLVMContextCreate: Symbol<'static, unsafe extern "C" fn() -> LLVMContextRef>,
    LLVMCreateMemoryBufferWithMemoryRange: Symbol<
        'static,
        unsafe extern "C" fn(
            InputData: *const c_char,
            InputDataLength: size_t,
            BufferName: *const c_char,
            RequiresNullTerminator: LLVMBool,
        ) -> LLVMMemoryBufferRef,
    >,
    LLVMParseIRInContext: Symbol<
        'static,
        unsafe extern "C" fn(
            ContextRef: LLVMContextRef,
            MemBuf: LLVMMemoryBufferRef,
            OutM: *mut LLVMModuleRef,
            OutMessage: *mut *mut i8,
        ) -> LLVMBool,
    >,
    LLVMParseBitcodeInContext2: Symbol<
        'static,
        unsafe extern "C" fn(
            ContextRef: LLVMContextRef,
            MemBuf: LLVMMemoryBufferRef,
            OutM: *mut LLVMModuleRef,
        ) -> LLVMBool,
    >,
    LLVMDumpModule: Symbol<'static, unsafe extern "C" fn(M: LLVMModuleRef)>,
    // LLVMCreateMCJITCompilerForModule: Symbol<
    //     'static,
    //     unsafe extern "C" fn(
    //         OutJIT: *mut LLVMExecutionEngineRef,
    //         M: LLVMModuleRef,
    //         Options: *mut LLVMMCJITCompilerOptions,
    //         SizeOfOptions: size_t,
    //         OutError: *mut *mut c_char,
    //     ) -> LLVMBool,
    // >,
    LLVMLinkInMCJIT: Symbol<'static, unsafe extern "C" fn()>,
    LLVMInitializeNativeTarget: Symbol<'static, unsafe extern "C" fn()>,
    LLVMInitializeNativeTargetMC: Symbol<'static, unsafe extern "C" fn()>,
    LLVMInitializeNativeTargetMCA: Symbol<'static, unsafe extern "C" fn()>,
    LLVMInitializeNativeTargetInfo: Symbol<'static, unsafe extern "C" fn()>,
    LLVMInitializeNativeAsmPrinter: Symbol<'static, unsafe extern "C" fn()>,
    LLVMContextDispose: Symbol<'static, unsafe extern "C" fn(C: LLVMContextRef)>,
    LLVMDisposeModule: Symbol<'static, unsafe extern "C" fn(M: LLVMModuleRef)>,
    LLVMDisposeMemoryBuffer: Symbol<'static, unsafe extern "C" fn(MemBuf: LLVMMemoryBufferRef)>,
    LLVMOrcCreateNewThreadSafeContext:
        Symbol<'static, unsafe extern "C" fn() -> LLVMOrcThreadSafeContextRef>,
    LLVMOrcThreadSafeContextGetContext:
        Symbol<'static, unsafe extern "C" fn(TSCtx: LLVMOrcThreadSafeContextRef) -> LLVMContextRef>,
    LLVMOrcCreateNewThreadSafeModule: Symbol<
        'static,
        unsafe extern "C" fn(
            M: LLVMModuleRef,
            TSCtx: LLVMOrcThreadSafeContextRef,
        ) -> LLVMOrcThreadSafeModuleRef,
    >,
    LLVMOrcDisposeThreadSafeModule:
        Symbol<'static, unsafe extern "C" fn(TSM: LLVMOrcThreadSafeModuleRef)>,
    LLVMOrcDisposeThreadSafeContext:
        Symbol<'static, unsafe extern "C" fn(TSCtx: LLVMOrcThreadSafeContextRef)>,
    LLVMOrcCreateLLJIT: Symbol<
        'static,
        unsafe extern "C" fn(
            Result: *mut LLVMOrcLLJITRef,
            Builder: LLVMOrcLLJITBuilderRef,
        ) -> LLVMErrorRef,
    >,
    LLVMOrcDisposeLLJIT: Symbol<'static, unsafe extern "C" fn(J: LLVMOrcLLJITRef) -> LLVMErrorRef>,
    LLVMOrcLLJITGetMainJITDylib:
        Symbol<'static, unsafe extern "C" fn(J: LLVMOrcLLJITRef) -> LLVMOrcJITDylibRef>,
    LLVMOrcLLJITAddLLVMIRModule: Symbol<
        'static,
        unsafe extern "C" fn(
            J: LLVMOrcLLJITRef,
            JD: LLVMOrcJITDylibRef,
            TSM: LLVMOrcThreadSafeModuleRef,
        ) -> LLVMErrorRef,
    >,
    LLVMOrcLLJITLookup: Symbol<
        'static,
        unsafe extern "C" fn(
            J: LLVMOrcLLJITRef,
            Result: *mut LLVMOrcExecutorAddress,
            Name: *const c_char,
        ) -> LLVMErrorRef,
    >,
    LLVMGetErrorMessage: Symbol<'static, unsafe extern "C" fn(E: LLVMErrorRef) -> *mut c_char>,
    LLVMDisposeErrorMessage: Symbol<'static, unsafe extern "C" fn(ErrMsg: *mut c_char)>,
    LLVMOrcCreateDumpObjects: Symbol<
        'static,
        unsafe extern "C" fn(
            DumpDir: *const c_char,
            IdentifierOverride: *const c_char,
        ) -> LLVMOrcDumpObjectsRef,
    >,
    LLVMOrcObjectTransformLayerSetTransform: Symbol<
        'static,
        unsafe extern "C" fn(
            ObjTransformLayer: LLVMOrcObjectTransformLayerRef,
            TransformFunction: LLVMOrcObjectTransformLayerTransformFunction,
            Ctx: *mut c_void,
        ),
    >,
    LLVMOrcLLJITGetObjTransformLayer:
        Symbol<'static, unsafe extern "C" fn(J: LLVMOrcLLJITRef) -> LLVMOrcObjectTransformLayerRef>,
    LLVMOrcDumpObjects_CallOperator: Symbol<
        'static,
        unsafe extern "C" fn(
            DumpObjects: LLVMOrcDumpObjectsRef,
            ObjBuffer: *mut LLVMMemoryBufferRef,
        ) -> LLVMErrorRef,
    >,

    LLVMGetTargetFromName:
        Symbol<'static, unsafe extern "C" fn(Name: *const c_char) -> LLVMTargetRef>,
    LLVMCreateTargetMachine: Symbol<
        'static,
        unsafe extern "C" fn(
            T: LLVMTargetRef,
            Triple: *const c_char,
            CPU: *const c_char,
            Features: *const c_char,
            Level: LLVMCodeGenOptLevel,
            Reloc: LLVMRelocMode,
            CodeModel: LLVMCodeModel,
        ) -> LLVMTargetMachineRef,
    >,
    LLVMRunPasses: Symbol<
        'static,
        unsafe extern "C" fn(
            M: LLVMModuleRef,
            Passes: *const c_char,
            TM: LLVMTargetMachineRef,
            Options: LLVMPassBuilderOptionsRef,
        ) -> LLVMErrorRef,
    >,
    LLVMCreatePassBuilderOptions:
        Symbol<'static, unsafe extern "C" fn() -> LLVMPassBuilderOptionsRef>,
    LLVMDisposePassBuilderOptions:
        Symbol<'static, unsafe extern "C" fn(Options: LLVMPassBuilderOptionsRef)>,
    // LLVMCreatePassBuilderOptions:
    //     Symbol<'static, unsafe extern "C" fn() -> LLVMPassBuilderOptionsRef>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LLVMPaths {
    pub clang: String,
    pub llvm: String,
}

const PATH_CACHE_FILE: &str = "llvm_paths.json";

impl LLVMPaths {
    fn override_from_env(&mut self) {
        match var("LUISA_LLVM_PATH") {
            Ok(s) => {
                self.llvm = s;
            }
            Err(_) => {}
        }
        match var("LUISA_CLANG_PATH") {
            Ok(s) => {
                self.clang = s;
            }
            Err(_) => {}
        }
    }
    pub fn get() -> LLVMPaths {
        let cur = current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .join(PATH_CACHE_FILE);
        let paths = if cur.exists() {
            let file = File::open(&cur).unwrap();
            let reader = BufReader::new(file);
            let mut paths: LLVMPaths = serde_json::from_reader(reader).unwrap();
            paths.override_from_env();
            paths
        } else {
            let mut paths = LLVMPaths {
                clang: find_clang().map(|s| {
                    log::info!("Found clang: {}", s);
                    s
                }).unwrap_or_else(|| {
                    match var("LUISA_CLANG_PATH") {
                        Ok(s) => s,
                        Err(_) => {
                            panic_abort!("Could not find clang. Please set LUISA_CLANG_PATH to the path of clang++")
                        }
                    }
                }),
                llvm: find_llvm().map(|s| {
                    log::info!("Found LLVM: {}", s);
                    s
                }).unwrap_or_else(|| {
                    match var("LUISA_LLVM_PATH") {
                        Ok(s) => s,
                        Err(_) => {
                            panic_abort!("Could not find LLVM. Please set LUISA_LLVM_PATH to the path of LLVM")
                        }
                    }
                }),
            };
            paths.override_from_env();
            paths
        };
        let file = File::create(cur).unwrap();
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &paths).unwrap();
        paths
    }
}
lazy_static! {
    pub(crate) static ref LLVM_PATH: LLVMPaths = LLVMPaths::get();
}
fn try_exists(s: &str) -> Option<String> {
    let path = Path::new(s);
    if path.exists() {
        Some(s.to_string())
    } else {
        None
    }
}

const MAX_LLVM_VERSION: u32 = 17;

fn find_clang() -> Option<String> {
    if cfg!(target_os = "windows") {
        try_exists(r"C:\Program Files\LLVM\bin\clang++.exe")
    } else if cfg!(target_os = "linux") {
        for version in 14..=MAX_LLVM_VERSION {
            let path = format!("/usr/bin/clang++-{}", version);
            if let Some(path) = try_exists(&path) {
                return Some(path);
            }
        }
        None
    } else {
        None
    }
}

fn find_llvm() -> Option<String> {
    if cfg!(target_os = "windows") {
        try_exists(r"C:\Program Files\LLVM\bin\LLVM-C.dll")
    } else if cfg!(target_os = "linux") {
        for version in 14..=MAX_LLVM_VERSION {
            let path = format!("/usr/lib/llvm-{}/lib/libLLVM.so", version);
            if let Some(path) = try_exists(&path) {
                return Some(path);
            }
        }
        None
    } else {
        None
    }
}

fn lift<'a, T>(t: Symbol<'a, T>) -> Symbol<'static, T> {
    unsafe { std::mem::transmute(t) }
}

fn llvm_lib_path() -> &'static str {
    &LLVM_PATH.llvm
}

#[allow(non_snake_case)]
impl LibLLVM {
    fn new() -> Self {
        if cfg!(all(
            not(target_arch = "x86_64"),
            not(target_arch = "aarch64")
        )) {
            panic_abort!("only x86_64 and aarch64 are supported");
        }
        unsafe {
            let lib = libloading::Library::new(&llvm_lib_path()).unwrap();
            let LLVMContextCreate = lift(lib.get(b"LLVMContextCreate").unwrap());
            let LLVMParseIRInContext = lift(lib.get(b"LLVMParseIRInContext").unwrap());
            let LLVMCreateMemoryBufferWithMemoryRange =
                lift(lib.get(b"LLVMCreateMemoryBufferWithMemoryRange").unwrap());
            let LLVMParseBitcodeInContext2 = lift(lib.get(b"LLVMParseBitcodeInContext2").unwrap());
            let LLVMDumpModule = lift(lib.get(b"LLVMDumpModule").unwrap());
            let LLVMLinkInMCJIT = lift(lib.get(b"LLVMLinkInMCJIT").unwrap());

            let LLVMInitializeNativeTarget = lift(
                lib.get(if cfg!(target_arch = "x86_64") {
                    b"LLVMInitializeX86Target"
                } else if cfg!(target_arch = "aarch64") {
                    b"LLVMInitializeAArch64Target"
                } else {
                    unreachable!()
                })
                .unwrap(),
            );

            let LLVMInitializeNativeTargetInfo = lift(
                lib.get(if cfg!(target_arch = "x86_64") {
                    b"LLVMInitializeX86TargetInfo"
                } else if cfg!(target_arch = "aarch64") {
                    b"LLVMInitializeAArch64TargetInfo"
                } else {
                    unreachable!()
                })
                .unwrap(),
            );

            let LLVMInitializeNativeTargetMC = lift(
                lib.get(if cfg!(target_arch = "x86_64") {
                    b"LLVMInitializeX86TargetMC"
                } else if cfg!(target_arch = "aarch64") {
                    b"LLVMInitializeAArch64TargetMC"
                } else {
                    unreachable!()
                })
                .unwrap(),
            );

            let LLVMInitializeNativeTargetMCA = lift(
                lib.get(if cfg!(target_arch = "x86_64") {
                    b"LLVMInitializeX86TargetMCA"
                } else if cfg!(target_arch = "aarch64") {
                    b"LLVMInitializeAArch64TargetMC"
                } else {
                    unreachable!()
                })
                .unwrap(),
            );

            let LLVMInitializeNativeAsmPrinter = lift(
                lib.get(if cfg!(target_arch = "x86_64") {
                    b"LLVMInitializeX86AsmPrinter"
                } else if cfg!(target_arch = "aarch64") {
                    b"LLVMInitializeAArch64AsmPrinter"
                } else {
                    unreachable!()
                })
                .unwrap(),
            );

            let LLVMContextDispose = lift(lib.get(b"LLVMContextDispose").unwrap());
            let LLVMDisposeModule = lift(lib.get(b"LLVMDisposeModule").unwrap());
            let LLVMDisposeMemoryBuffer = lift(lib.get(b"LLVMDisposeMemoryBuffer").unwrap());
            let LLVMOrcCreateNewThreadSafeContext =
                lift(lib.get(b"LLVMOrcCreateNewThreadSafeContext").unwrap());
            let LLVMOrcThreadSafeContextGetContext =
                lift(lib.get(b"LLVMOrcThreadSafeContextGetContext").unwrap());
            let LLVMOrcCreateNewThreadSafeModule =
                lift(lib.get(b"LLVMOrcCreateNewThreadSafeModule").unwrap());
            let LLVMOrcDisposeThreadSafeModule =
                lift(lib.get(b"LLVMOrcDisposeThreadSafeModule").unwrap());
            let LLVMOrcDisposeThreadSafeContext =
                lift(lib.get(b"LLVMOrcDisposeThreadSafeContext").unwrap());
            let LLVMOrcCreateLLJIT = lift(lib.get(b"LLVMOrcCreateLLJIT").unwrap());
            let LLVMOrcDisposeLLJIT = lift(lib.get(b"LLVMOrcDisposeLLJIT").unwrap());
            let LLVMOrcLLJITGetMainJITDylib =
                lift(lib.get(b"LLVMOrcLLJITGetMainJITDylib").unwrap());

            let LLVMOrcLLJITAddLLVMIRModule =
                lift(lib.get(b"LLVMOrcLLJITAddLLVMIRModule").unwrap());
            let LLVMOrcLLJITLookup = lift(lib.get(b"LLVMOrcLLJITLookup").unwrap());
            let LLVMGetErrorMessage = lift(lib.get(b"LLVMGetErrorMessage").unwrap());
            let LLVMDisposeErrorMessage = lift(lib.get(b"LLVMDisposeErrorMessage").unwrap());
            let LLVMOrcCreateDumpObjects = lift(lib.get(b"LLVMOrcCreateDumpObjects").unwrap());
            let LLVMOrcObjectTransformLayerSetTransform =
                lift(lib.get(b"LLVMOrcObjectTransformLayerSetTransform").unwrap());
            let LLVMOrcLLJITGetObjTransformLayer =
                lift(lib.get(b"LLVMOrcLLJITGetObjTransformLayer").unwrap());
            let LLVMOrcDumpObjects_CallOperator =
                lift(lib.get(b"LLVMOrcDumpObjects_CallOperator").unwrap());
            let LLVMGetTargetFromName = lift(lib.get(b"LLVMGetTargetFromName").unwrap());
            let LLVMCreateTargetMachine = lift(lib.get(b"LLVMCreateTargetMachine").unwrap());
            let LLVMCreatePassBuilderOptions =
                lift(lib.get(b"LLVMCreatePassBuilderOptions").unwrap());
            let LLVMDisposePassBuilderOptions =
                lift(lib.get(b"LLVMDisposePassBuilderOptions").unwrap());
            let LLVMRunPasses = lift(lib.get(b"LLVMRunPasses").unwrap());
            let LLVMOrcAbsoluteSymbols = lift(lib.get(b"LLVMOrcAbsoluteSymbols").unwrap());
            let LLVMOrcLLJITMangleAndIntern =
                lift(lib.get(b"LLVMOrcLLJITMangleAndIntern").unwrap());
            let LLVMOrcJITDylibDefine = lift(lib.get(b"LLVMOrcJITDylibDefine").unwrap());
            LibLLVM {
                lib,
                LLVMOrcJITDylibDefine,
                LLVMOrcLLJITMangleAndIntern,
                LLVMOrcAbsoluteSymbols,
                LLVMContextCreate,
                LLVMCreateMemoryBufferWithMemoryRange,
                LLVMParseBitcodeInContext2,
                LLVMDumpModule,
                LLVMLinkInMCJIT,
                LLVMInitializeNativeTarget,
                LLVMInitializeNativeTargetInfo,
                LLVMInitializeNativeTargetMC,
                LLVMInitializeNativeTargetMCA,
                LLVMInitializeNativeAsmPrinter,
                LLVMContextDispose,
                LLVMDisposeModule,
                LLVMDisposeMemoryBuffer,
                LLVMOrcCreateNewThreadSafeContext,
                LLVMOrcThreadSafeContextGetContext,
                LLVMOrcCreateNewThreadSafeModule,
                LLVMOrcDisposeThreadSafeModule,
                LLVMOrcDisposeThreadSafeContext,
                LLVMOrcCreateLLJIT,
                LLVMOrcDisposeLLJIT,
                LLVMOrcLLJITGetMainJITDylib,
                LLVMOrcLLJITAddLLVMIRModule,
                LLVMOrcLLJITLookup,
                LLVMGetErrorMessage,
                LLVMDisposeErrorMessage,
                LLVMOrcCreateDumpObjects,
                LLVMOrcObjectTransformLayerSetTransform,
                LLVMOrcLLJITGetObjTransformLayer,
                LLVMOrcDumpObjects_CallOperator,
                LLVMGetTargetFromName,
                LLVMCreateTargetMachine,
                LLVMCreatePassBuilderOptions,
                LLVMDisposePassBuilderOptions,
                LLVMRunPasses,
                LLVMParseIRInContext,
            }
        }
    }
    fn handle_error(&self, error: LLVMErrorRef) {
        let error_message = unsafe {
            let error_message = (self.LLVMGetErrorMessage)(error);
            let msg = CStr::from_ptr(error_message);
            // self.LLVMDisposeErrorMessage(error);
            let msg = msg.to_str().unwrap().to_string();
            (self.LLVMDisposeErrorMessage)(error_message);
            msg
        };
        eprintln!("LLVMError: {}", error_message);
        eprintln!("Pleas check clang++ version matches llvm version");
        let paths = LLVM_PATH.deref();
        eprintln!("clang++: {}, llvm: {}", paths.clang, paths.llvm);
    }
}

pub(crate) fn compile_llvm_ir(name: &String, path_: &String) -> KernelFn {
    init_llvm();
    unsafe {
        let c = CONTEXT.lock();
        {
            let mut c = c.borrow_mut();
            let c = c.as_mut().unwrap();
            if let Some(record) = c.cached_functions.get(path_) {
                return *record;
            }
        }
        let record = {
            let c = c.borrow();
            let c = c.as_ref().unwrap();
            let lib = &c.lib;
            let tsctx = (lib.LLVMOrcCreateNewThreadSafeContext)();
            let ctx = (lib.LLVMOrcThreadSafeContextGetContext)(tsctx);
            let name = CString::new(name.clone()).unwrap();
            // let path = CString::new(path_.clone()).unwrap();
            let bitcode = {
                let mut bc_file = std::fs::File::open(path_).unwrap();
                let mut buf = vec![];
                use std::io::Read;
                bc_file.read_to_end(&mut buf).unwrap();
                buf
            };
            let bc_buffer = (lib.LLVMCreateMemoryBufferWithMemoryRange)(
                bitcode.as_ptr() as *const i8,
                bitcode.len(),
                name.as_ptr() as *const i8,
                0,
            );
            // let ll_src = std::fs::read_to_string(path_).unwrap();
            // let ll_src = CString::new(ll_src).unwrap();
            // let ll_buffer = (lib.LLVMCreateMemoryBufferWithMemoryRange)(
            //     ll_src.as_ptr() as *const i8,
            //     ll_src.as_bytes().len(),
            //     name.as_ptr() as *const i8,
            //     0,
            // );
            let mut module: LLVMModuleRef = std::ptr::null_mut();
            if (lib.LLVMParseBitcodeInContext2)(ctx, bc_buffer, &mut module as *mut LLVMModuleRef)
                != 0
            {
                panic_abort!("LLVMParseBitcodeInContext2 failed");
            }
            // let mut msg: *mut i8 = std::ptr::null_mut();
            // if (lib.LLVMParseIRInContext)(
            //     ctx,
            //     ll_buffer,
            //     &mut module as *mut LLVMModuleRef,
            //     &mut msg as *mut *mut i8,
            // ) != 0
            // {
            //     panic_abort!("LLVMParseIRInContext failed");
            // }
            // let pass = CString::new("default<O3>").unwrap();
            // let pass_builder_options = (lib.LLVMCreatePassBuilderOptions)();
            // let err = (lib.LLVMRunPasses)(
            //     module,
            //     pass.as_ptr(),
            //     c.target_machine,
            //     pass_builder_options,
            // );
            // if !err.is_null() {
            //     lib.handle_error(err);
            // }
            let tsm = (lib.LLVMOrcCreateNewThreadSafeModule)(module, tsctx);

            // (lib.LLVMDisposePassBuilderOptions)(pass_builder_options);
            let main_jd = (lib.LLVMOrcLLJITGetMainJITDylib)(c.jit);
            let err = (lib.LLVMOrcLLJITAddLLVMIRModule)(c.jit, main_jd, tsm);
            if !err.is_null() {
                lib.handle_error(err);
            }
            let mut addr: LLVMOrcExecutorAddress = 0;
            let err = (lib.LLVMOrcLLJITLookup)(c.jit, &mut addr, name.as_ptr());
            if !err.is_null() {
                lib.handle_error(err);
            }
            (lib.LLVMOrcDisposeThreadSafeContext)(tsctx);
            let function = std::mem::transmute(addr as *mut u8);
            function
        };
        {
            let mut c = c.borrow_mut();
            let c = c.as_mut().unwrap();
            c.cached_functions.insert(path_.clone(), record);
        }
        record
    }
}

static CONTEXT: ReentrantMutex<RefCell<Option<Context>>> = ReentrantMutex::new(RefCell::new(None));

fn init_llvm() {
    let c = CONTEXT.lock();
    let inited = c.borrow().is_some();
    if !inited {
        *c.borrow_mut() = Some(Context::new());
    }
}

struct Context {
    lib: LibLLVM,
    context: LLVMContextRef,
    cached_functions: HashMap<String, KernelFn>,
    jit: LLVMOrcLLJITRef,
    dump: LLVMOrcDumpObjectsRef,
    target: LLVMTargetRef,
    target_machine: LLVMTargetMachineRef,
}

unsafe impl Send for Context {}

unsafe impl Sync for Context {}

static ABORT_MUTEX: Mutex<()> = Mutex::new(());

impl Context {
    fn new() -> Self {
        let lib = LibLLVM::new();
        let context = unsafe { (lib.LLVMContextCreate)() };

        let mut jit: LLVMOrcLLJITRef = std::ptr::null_mut();

        let target: LLVMTargetRef;
        let target_machine: LLVMTargetMachineRef;
        unsafe {
            (lib.LLVMLinkInMCJIT)();
            (lib.LLVMInitializeNativeTargetInfo)();
            (lib.LLVMInitializeNativeTarget)();
            (lib.LLVMInitializeNativeTargetMC)();
            (lib.LLVMInitializeNativeTargetMCA)();
            (lib.LLVMInitializeNativeAsmPrinter)();

            let err = (lib.LLVMOrcCreateLLJIT)(&mut jit, std::ptr::null_mut());
            if !err.is_null() {
                lib.handle_error(err);
            }
            let target_name = target_name();
            let target_name = CString::new(target_name).unwrap();
            let features = CString::new(cpu_features().join(",")).unwrap();
            let target_triple = CString::new(target_triple()).unwrap();
            target = (lib.LLVMGetTargetFromName)(target_name.as_ptr());
            assert!(!target.is_null());
            target_machine = (lib.LLVMCreateTargetMachine)(
                target,
                target_triple.as_ptr(),
                target_name.as_ptr(),
                features.as_ptr(),
                LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
                LLVMRelocMode::LLVMRelocDefault,
                LLVMCodeModel::LLVMCodeModelDefault,
            );
            assert!(!target_machine.is_null());

            let jd = (lib.LLVMOrcLLJITGetMainJITDylib)(jit);
            macro_rules! add_symbol {
                ($s:ident, $v:expr) => {{
                    let name = stringify!($s);
                    let name = CString::new(name).unwrap();
                    let addr = $v as *mut u8 as LLVMOrcExecutorAddress;
                    let symbol = LLVMJITEvaluatedSymbol {
                        Address: addr,
                        Flags: LLVMJITSymbolFlags {
                            GenericFlags: 0,
                            TargetFlags: 0,
                        },
                    };
                    let name = (lib.LLVMOrcLLJITMangleAndIntern)(jit, name.as_ptr());
                    let mut pair = LLVMOrcCSymbolMapPair {
                        Name: name,
                        Sym: symbol,
                    };
                    let syms = (lib.LLVMOrcAbsoluteSymbols)(&mut pair, 1);
                    (lib.LLVMOrcJITDylibDefine)(jd, syms);
                }};
            }
            add_symbol!(memcpy, libc::memcpy);
            add_symbol!(memset, libc::memset);
            unsafe extern "C" fn lc_abort(ctx: *const c_void, msg: i32) {
                let _lk = ABORT_MUTEX.lock();
                {
                    let ctx = ctx as *const ShaderDispatchContext;
                    let ctx = &*ctx;
                    let shader = ctx.shader as *const ShaderImpl;
                    let shader = &*shader;
                    let mut err = (&*ctx.error).lock();
                    if err.is_none() {
                        *err = Some(shader.messages[msg as usize].clone());
                    }
                }
                panic!("##lc_kernel##");
            }
            add_symbol!(lc_abort, lc_abort);
            add_symbol!(__stack_chk_fail, libc::abort);
            if cfg!(target_os = "windows") {
                let kernel32_dll =
                    libloading::Library::new("C:\\Windows\\system32\\kernel32.dll").unwrap();
                let __chkstk = *kernel32_dll.get::<u64>(b"__chkstk\0").unwrap();
                add_symbol!(__chkstk, __chkstk);
            }
            unsafe extern "C" fn lc_abort_and_print_sll(
                ctx: *const c_void,
                msg: *const c_char,
                i: u32,
                j: u32,
            ) {
                let _lk = ABORT_MUTEX.lock();
                {
                    let ctx = ctx as *const ShaderDispatchContext;
                    let ctx = &*ctx;
                    let msg = CStr::from_ptr(msg).to_str().unwrap().to_string();
                    let idx = msg.find("{}").unwrap();
                    let mut display = String::new();
                    display.push_str(&msg[..idx]);
                    display.push_str(&format!("{}", i));
                    let idx2 = msg[idx + 2..].find("{}").unwrap();
                    display.push_str(&msg[idx + 2..idx + 2 + idx2]);
                    display.push_str(&format!("{}", j));
                    display.push_str(&msg[idx + 2 + idx2 + 2..]);
                    let mut err = (&*ctx.error).lock();
                    if err.is_none() {
                        *err = Some(display);
                    }
                }
                panic!("##lc_kernel##");
            }
            add_symbol!(lc_abort_and_print_sll, lc_abort_and_print_sll);
            // min/max/abs/acos/asin/asinh/acosh/atan/atanh/atan2/
            //cos/cosh/sin/sinh/tan/tanh/exp/exp2/exp10/log/log2/
            //log10/sqrt/rsqrt/ceil/floor/trunc/round/fma/copysignf/
            //isinf/isnan
            macro_rules! add_libm_symbol {
                ($x:ident) => {
                    add_symbol!($x, libm::$x);
                };
                ($x:ident, $($y:ident),*) => {
                    add_symbol!($x,libm::$x);
                    add_libm_symbol!($($y),*);
                };
            }
            add_libm_symbol!(
                fminf, fmaxf, sinf, fabsf, acosf, asinf, atanf, acoshf, asinhf, atanhf, atan2f,
                cosf, coshf, sinf, sinhf, tanf, tanhf, expf, exp2f, exp10f, logf, log2f, log10f,
                sqrtf, ceilf, floorf, truncf, roundf, fmaf, copysignf, powf
            );
            extern "C" fn rsqrtf(x: f32) -> f32 {
                1.0 / x.sqrt()
            }
            extern "C" fn sincos_(x: f32, s: &mut f32, c: &mut f32) {
                let (a, b) = x.sin_cos();
                *s = a;
                *c = b;
            }
            add_symbol!(rsqrtf, rsqrtf);
            add_symbol!(sincosf, sincos_);
        }
        let work_dir = CString::new("").unwrap();
        let ident = CString::new("").unwrap();
        let dump = unsafe { (lib.LLVMOrcCreateDumpObjects)(work_dir.as_ptr(), ident.as_ptr()) };
        unsafe {
            match std::env::var("LUISA_DUMP_OBJECTS") {
                Ok(val) => {
                    if val == "1" {
                        (lib.LLVMOrcObjectTransformLayerSetTransform)(
                            (lib.LLVMOrcLLJITGetObjTransformLayer)(jit),
                            dump_objects,
                            std::ptr::null_mut(),
                        );
                    }
                }
                Err(_) => {}
            }
        }
        Self {
            target,
            target_machine,
            lib,
            context,
            cached_functions: HashMap::new(),
            jit,
            dump,
        }
    }
}

fn target_name() -> String {
    if cfg!(target_arch = "x86_64") {
        "x86-64".to_string()
    } else if cfg!(target_arch = "aarch64") {
        "aarch64".to_string()
    } else {
        panic_abort!("unsupported target")
    }
}

fn cpu_features() -> Vec<String> {
    if cfg!(target_arch = "x86_64") {
        // "+avx,+avx2,+fma,+popcnt,+sse4.1,+sse4.2,+sse4a".to_string()
        vec![
            "avx".into(),
            "avx2".into(),
            "fma".into(),
            "popcnt".into(),
            "sse4.1".into(),
            "sse4.2".into(),
            "sse4a".into(),
        ]
    } else if cfg!(target_arch = "aarch64") {
        vec!["neon".into()]
    } else {
        panic_abort!("unsupported target")
    }
}

fn target_triple() -> String {
    if cfg!(target_os = "windows") {
        "x86_64-pc-windows-msvc".to_string()
    } else if cfg!(target_os = "linux") {
        "x86_64-unknown-linux-gnu".to_string()
    } else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "x86_64") {
            "x86_64-apple-darwin".to_string()
        } else if cfg!(target_arch = "aarch64") {
            "aarch64-apple-darwin".to_string()
        } else {
            panic_abort!("unsupported target")
        }
    } else {
        panic_abort!("unsupported target")
    }
}

extern "C" fn dump_objects(_: *mut c_void, obj_in_out: *mut LLVMMemoryBufferRef) -> LLVMErrorRef {
    let c = CONTEXT.lock();
    let c = c.borrow();
    let c = c.as_ref().unwrap();
    let dump = c.dump;
    unsafe { (c.lib.LLVMOrcDumpObjects_CallOperator)(dump, obj_in_out) }
}
