#![allow(dead_code)]

use crate::panic_abort;
use lazy_static::lazy_static;
use libloading::Symbol;
use serde::{Deserialize, Serialize};
use std::env::{current_exe, var};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::ops::Deref;
use std::ptr::null;
use std::sync::atomic::Ordering;
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{CStr, CString},
    path::Path,
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
                if !Path::new(&s).exists() {
                    panic!(
                        "LUISA_LLVM_PATH is set to {}, but the path does not exist",
                        s
                    );
                }
                if Path::new(&s).is_dir() {
                    panic!("LUISA_LLVM_PATH is set to {}, but the path is a directory. Should be path to library", s);
                }
                self.llvm = s;
            }
            Err(_) => {}
        }
        match var("LUISA_CLANG_PATH") {
            Ok(s) => {
                if !Path::new(&s).exists() {
                    panic!(
                        "LUISA_CLANG_PATH is set to {}, but the path does not exist",
                        s
                    );
                }
                if Path::new(&s).is_dir() {
                    panic!("LUISA_CLANG_PATH is set to {}, but the path is a directory. Should be path to executable", s);
                }
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
                            panic!("Could not find clang. Please set LUISA_CLANG_PATH to the path of clang++ executable")
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
                            let libllvm = if !cfg!(target_os = "windows") {
                                "libLLVM.so"
                            } else {
                                "LLVM-C.dll"
                            };
                            panic!("Could not find LLVM. Please set LUISA_LLVM_PATH to the path of {}", libllvm);
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
    } else if cfg!(target_os = "macos") {
        try_exists("/opt/homebrew/opt/llvm/bin/clang")
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
    } else if cfg!(target_os = "macos") {
        try_exists("/opt/homebrew/opt/llvm/lib/libLLVM-C.dylib")
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
            panic!("only x86_64 and aarch64 are supported");
        }
        unsafe {
            let path = llvm_lib_path();
            let lib = libloading::Library::new(&path).unwrap_or_else(|e| {
                panic!("Failed to load LLVM: could not load {}, error: {}", path, e);
            });
            log::info!("Loading LLVM functions from {}", path);
            macro_rules! load {
                ($name:expr) => {
                    lift(lib.get($name).unwrap_or_else(|e| {
                        panic!(
                            "Failed to load LLVM function {}: could not load {}, error: {}",
                            std::str::from_utf8($name).unwrap(),
                            path,
                            e
                        );
                    }))
                };
            }
            let LLVMContextCreate = load!(b"LLVMContextCreate");
            let LLVMParseIRInContext = load!(b"LLVMParseIRInContext");
            let LLVMCreateMemoryBufferWithMemoryRange =
                load!(b"LLVMCreateMemoryBufferWithMemoryRange");
            let LLVMParseBitcodeInContext2 = load!(b"LLVMParseBitcodeInContext2");
            let LLVMDumpModule = load!(b"LLVMDumpModule");
            let LLVMLinkInMCJIT = load!(b"LLVMLinkInMCJIT");

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

            let LLVMContextDispose = load!(b"LLVMContextDispose");
            let LLVMDisposeModule = load!(b"LLVMDisposeModule");
            let LLVMDisposeMemoryBuffer = load!(b"LLVMDisposeMemoryBuffer");
            let LLVMOrcCreateNewThreadSafeContext = load!(b"LLVMOrcCreateNewThreadSafeContext");
            let LLVMOrcThreadSafeContextGetContext = load!(b"LLVMOrcThreadSafeContextGetContext");
            let LLVMOrcCreateNewThreadSafeModule = load!(b"LLVMOrcCreateNewThreadSafeModule");
            let LLVMOrcDisposeThreadSafeModule = load!(b"LLVMOrcDisposeThreadSafeModule");
            let LLVMOrcDisposeThreadSafeContext = load!(b"LLVMOrcDisposeThreadSafeContext");
            let LLVMOrcCreateLLJIT = load!(b"LLVMOrcCreateLLJIT");
            let LLVMOrcDisposeLLJIT = load!(b"LLVMOrcDisposeLLJIT");
            let LLVMOrcLLJITGetMainJITDylib = load!(b"LLVMOrcLLJITGetMainJITDylib");

            let LLVMOrcLLJITAddLLVMIRModule = load!(b"LLVMOrcLLJITAddLLVMIRModule");
            let LLVMOrcLLJITLookup = load!(b"LLVMOrcLLJITLookup");
            let LLVMGetErrorMessage = load!(b"LLVMGetErrorMessage");
            let LLVMDisposeErrorMessage = load!(b"LLVMDisposeErrorMessage");
            let LLVMOrcCreateDumpObjects = load!(b"LLVMOrcCreateDumpObjects");
            let LLVMOrcObjectTransformLayerSetTransform =
                load!(b"LLVMOrcObjectTransformLayerSetTransform");
            let LLVMOrcLLJITGetObjTransformLayer = load!(b"LLVMOrcLLJITGetObjTransformLayer");
            let LLVMOrcDumpObjects_CallOperator = load!(b"LLVMOrcDumpObjects_CallOperator");
            let LLVMGetTargetFromName = load!(b"LLVMGetTargetFromName");
            let LLVMCreateTargetMachine = load!(b"LLVMCreateTargetMachine");
            let LLVMCreatePassBuilderOptions = load!(b"LLVMCreatePassBuilderOptions");
            let LLVMDisposePassBuilderOptions = load!(b"LLVMDisposePassBuilderOptions");
            let LLVMRunPasses = load!(b"LLVMRunPasses");
            let LLVMOrcAbsoluteSymbols = load!(b"LLVMOrcAbsoluteSymbols");
            let LLVMOrcLLJITMangleAndIntern = load!(b"LLVMOrcLLJITMangleAndIntern");
            let LLVMOrcJITDylibDefine = load!(b"LLVMOrcJITDylibDefine");
            log::info!("LLVM functions loaded from {}", path);
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

#[repr(C)]
struct LLVMExecutorAddr {
    addr: u64,
}

#[repr(C)]
struct LLVMExecutorAddrRange {
    start: LLVMExecutorAddr,
    end: LLVMExecutorAddr,
}

#[repr(C)]
struct LLVMError {
    payload: *const c_void,
}

#[no_mangle]
unsafe extern "C" fn llvm_orc_registerEHFrameSectionWrapper(_: LLVMExecutorAddrRange) -> LLVMError {
    LLVMError { payload: null() }
}

#[no_mangle]
unsafe extern "C" fn llvm_orc_deregisterEHFrameSectionWrapper(
    _: LLVMExecutorAddrRange,
) -> LLVMError {
    LLVMError { payload: null() }
}

pub(crate) fn compile_llvm_ir(name: &String, path_: &String) -> Option<KernelFn> {
    init_llvm();
    unsafe {
        let c = CONTEXT.lock();
        {
            let mut c = c.borrow_mut();
            let c = c.as_mut().unwrap();
            if let Some(record) = c.cached_functions.get(path_) {
                return Some(*record);
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
                log::error!("LLVMParseBitcodeInContext2 failed");
                return None;
            }
            // let mut msg: *mut i8 = std::ptr::null_mut();
            // if (lib.LLVMParseIRInContext)(
            //     ctx,
            //     ll_buffer,
            //     &mut module as *mut LLVMModuleRef,
            //     &mut msg as *mut *mut i8,
            // ) != 0
            // {
            //     panic!("LLVMParseIRInContext failed");
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
                return None;
            }
            let mut addr: LLVMOrcExecutorAddress = 0;
            let err = (lib.LLVMOrcLLJITLookup)(c.jit, &mut addr, name.as_ptr());
            if !err.is_null() {
                lib.handle_error(err);
                return None;
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
        Some(record)
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
                    if ctx.terminated.load(Ordering::SeqCst) {
                        return;
                    }
                    loop {
                        let current = ctx.terminated.load(Ordering::SeqCst);
                        if current {
                            return;
                        }
                        match ctx.terminated.compare_exchange(
                            current,
                            true,
                            Ordering::SeqCst,
                            Ordering::Acquire,
                        ) {
                            Ok(false) => break,
                            _ => return,
                        }
                    }
                    let shader = ctx.shader as *const ShaderImpl;
                    let shader = &*shader;

                    eprintln!("{}", shader.messages[msg as usize]);
                    use std::io::Write;
                    let mut file = std::fs::File::create("luisa-compute-abort.txt").unwrap();
                    writeln!(
                        file,
                        "LuisaCompute CPU backend kernel aborted:\n{}",
                        shader.messages[msg as usize]
                    )
                    .unwrap();
                }

                panic_abort!("kernel execution aborted. see `luisa-compute-abort.txt` for details");
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
                    if ctx.terminated.load(Ordering::SeqCst) {
                        return;
                    }
                    loop {
                        let current = ctx.terminated.load(Ordering::SeqCst);
                        if current {
                            return;
                        }
                        match ctx.terminated.compare_exchange(
                            current,
                            true,
                            Ordering::SeqCst,
                            Ordering::Acquire,
                        ) {
                            Ok(false) => break,
                            _ => return,
                        }
                    }
                    let msg = CStr::from_ptr(msg).to_str().unwrap().to_string();
                    dbg!(msg.len());
                    let idx = msg.find("{}").unwrap();
                    let mut display = String::new();
                    display.push_str(&msg[..idx]);
                    display.push_str(&format!("{}", i));
                    let idx2 = msg[idx + 2..].find("{}").unwrap();
                    display.push_str(&msg[idx + 2..idx + 2 + idx2]);
                    display.push_str(&format!("{}", j));
                    display.push_str(&msg[idx + 2 + idx2 + 2..]);
                    eprintln!("{}", display);
                    use std::io::Write;
                    let mut file = std::fs::File::create("luisa-compute-abort.txt").unwrap();
                    writeln!(
                        file,
                        "LuisaCompute CPU backend kernel aborted:\n{}",
                        display
                    )
                    .unwrap();
                }
                panic_abort!("kernel execution aborted. see `luisa-compute-abort.txt` for details");
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
                sqrtf, ceilf, floorf, truncf, roundf, fmaf, copysignf, powf, fmodf
            );
            extern "C" fn rsqrtf(x: f32) -> f32 {
                1.0 / x.sqrt()
            }
            extern "C" fn sincos_(x: f32, s: &mut f32, c: &mut f32) {
                let (a, b) = x.sin_cos();
                *s = a;
                *c = b;
            }
            #[repr(C)]
            struct F32x2 {
                x: f32,
                y: f32,
            }
            extern "C" fn sincos_stret(x: f32) -> F32x2 {
                let (x, y) = x.sin_cos();
                F32x2 { x, y }
            }
            add_symbol!(rsqrtf, rsqrtf);
            add_symbol!(sincosf, sincos_);
            add_symbol!(__sincosf_stret, sincos_stret);
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
        "arm64".to_string()
    } else {
        panic!("unsupported target")
    }
}

#[cfg(target_arch = "aarch64")]
fn cpu_features() -> Vec<String> {
    vec!["neon".into()]
}

#[rustfmt::skip]
#[cfg(target_arch = "x86_64")]
fn cpu_features() -> Vec<String> {
    let mut features = vec![];
    if is_x86_feature_detected!("aes") { features.push("aes"); }
    // if is_x86_feature_detected!("pclmulqdq") { features.push("pclmulqdq"); }
    // if is_x86_feature_detected!("rdrand") { features.push("rdrand"); }
    // if is_x86_feature_detected!("rdseed") { features.push("rdseed"); }
    // if is_x86_feature_detected!("tsc") { features.push("tsc"); }
    if is_x86_feature_detected!("mmx") { features.push("mmx"); }
    if is_x86_feature_detected!("sse") { features.push("sse"); }
    if is_x86_feature_detected!("sse2") { features.push("sse2"); }
    if is_x86_feature_detected!("sse3") { features.push("sse3"); }
    if is_x86_feature_detected!("ssse3") { features.push("ssse3"); }
    if is_x86_feature_detected!("sse4.1") { features.push("sse4.1"); }
    if is_x86_feature_detected!("sse4.2") { features.push("sse4.2"); }
    if is_x86_feature_detected!("sse4a") { features.push("sse4a"); }
    if is_x86_feature_detected!("sha") { features.push("sha"); }
    if is_x86_feature_detected!("avx") { features.push("avx"); }
    if is_x86_feature_detected!("avx2") { features.push("avx2"); }
    if is_x86_feature_detected!("avx512f") { features.push("avx512f"); }
    if is_x86_feature_detected!("avx512cd") { features.push("avx512cd"); }
    if is_x86_feature_detected!("avx512er") { features.push("avx512er"); }
    if is_x86_feature_detected!("avx512pf") { features.push("avx512pf"); }
    if is_x86_feature_detected!("avx512bw") { features.push("avx512bw"); }
    if is_x86_feature_detected!("avx512dq") { features.push("avx512dq"); }
    if is_x86_feature_detected!("avx512vl") { features.push("avx512vl"); }
    if is_x86_feature_detected!("avx512ifma") { features.push("avx512ifma"); }
    if is_x86_feature_detected!("avx512vbmi") { features.push("avx512vbmi"); }
    if is_x86_feature_detected!("avx512vpopcntdq") { features.push("avx512vpopcntdq"); }
    if is_x86_feature_detected!("avx512vbmi2") { features.push("avx512vbmi2"); }
    if is_x86_feature_detected!("gfni") { features.push("gfni"); }
    if is_x86_feature_detected!("vaes") { features.push("vaes"); }
    if is_x86_feature_detected!("vpclmulqdq") { features.push("vpclmulqdq"); }
    if is_x86_feature_detected!("avx512vnni") { features.push("avx512vnni"); }
    if is_x86_feature_detected!("avx512bitalg") { features.push("avx512bitalg"); }
    if is_x86_feature_detected!("avx512bf16") { features.push("avx512bf16"); }
    if is_x86_feature_detected!("avx512vp2intersect") { features.push("avx512vp2intersect"); }
    if is_x86_feature_detected!("f16c") { features.push("f16c"); }
    if is_x86_feature_detected!("fma") { features.push("fma"); }
    // if is_x86_feature_detected!("bmi1") { features.push("bmi1"); }
    if is_x86_feature_detected!("bmi2") { features.push("bmi2"); }
    // if is_x86_feature_detected!("abm") { features.push("abm"); }
    if is_x86_feature_detected!("lzcnt") { features.push("lzcnt"); }
    if is_x86_feature_detected!("tbm") { features.push("tbm"); }
    if is_x86_feature_detected!("popcnt") { features.push("popcnt"); }
    if is_x86_feature_detected!("fxsr") { features.push("fxsr"); }
    if is_x86_feature_detected!("xsave") { features.push("xsave"); }
    if is_x86_feature_detected!("xsaveopt") { features.push("xsaveopt"); }
    if is_x86_feature_detected!("xsaves") { features.push("xsaves"); }
    if is_x86_feature_detected!("xsavec") { features.push("xsavec"); }
    // if is_x86_feature_detected!("cmpxchg16b") { features.push("cmpxchg16b"); }
    if is_x86_feature_detected!("adx") { features.push("adx"); }
    if is_x86_feature_detected!("rtm") { features.push("rtm"); }
    // this breaks msvc shipped with vs2019
    // if is_x86_feature_detected!("movbe") { features.push("movbe"); }
    // if is_x86_feature_detected!("ermsb") { features.push("ermsb"); }
    features.into_iter().map(|s| s.to_string()).collect()
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
            "arm64-apple-darwin".to_string()
        } else {
            panic!("unsupported target")
        }
    } else {
        panic!("unsupported target")
    }
}

extern "C" fn dump_objects(_: *mut c_void, obj_in_out: *mut LLVMMemoryBufferRef) -> LLVMErrorRef {
    let c = CONTEXT.lock();
    let c = c.borrow();
    let c = c.as_ref().unwrap();
    let dump = c.dump;
    unsafe { (c.lib.LLVMOrcDumpObjects_CallOperator)(dump, obj_in_out) }
}
