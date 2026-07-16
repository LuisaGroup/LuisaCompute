---
name: llvm-spirv
description: LLVM SPIR-V codegen backend in src/backends/common/spirv_llvm/ — AST→LLVM IR→SPIR-V binary via spirv64 TargetMachine.
---

# LLVM SPIR-V Codegen

**Location**: `src/backends/common/spirv_llvm/` (4 C++ files, 4 headers, xmake.lua)

Pipeline: `Function` (Luisa AST) → `LLVMStateVisitor` builds `llvm::IRBuilder<>` IR → `llvm::Module` → aggregate legalization passes → `legacy::PassManager` emits SPIR-V binary via `spirv64-unknown-vulkan1.2` TargetMachine → post-process (strip `Addresses`/`Linkage` capabilities, validate with SPIR-V Tools).

## Files

| File | Role |
|------|------|
| `src/backends/common/spirv_llvm/llvm_codegen_result.h` | `LLVMCodegenResult` — SPIR-V binary + metadata |
| `src/backends/common/spirv_llvm/llvm_codegen_stack_data.h` / `.cpp` | `LLVMCodegenStackData` — per-codegen mutable state, object pool |
| `src/backends/common/spirv_llvm/llvm_codegen_utility.h` / `.cpp` | `LLVMCodegenUtility` — type mapping, constants, function codegen, SPIR-V emission |
| `src/backends/common/spirv_llvm/llvm_state_visitor.h` / `.cpp` | `LLVMStateVisitor` — AST visitor (StmtVisitor + ExprVisitor) |
| `src/backends/common/spirv_llvm/spirv_llvm.cpp` | LLVM SPIR-V target initialization, `InitializeLLVMSPIRVTarget()` |
| `src/backends/common/spirv_llvm/xmake.lua` | Static lib `lc-spirv-llvm`, links `LLVM*.lib` + `spirv-tools` |

## LLVMCodegenResult

`src/backends/common/spirv_llvm/llvm_codegen_result.h` lines 17–32:

```cpp
struct LLVMCodegenResult {
    using Properties = vstd::vector<hlsl::Property>;
    luisa::vector<uint32_t> spv_bin;                                       // SPIR-V binary words
    Properties properties;
    vstd::vector<std::pair<vstd::string, Type const *>> printers;
    bool useTex2DBindless{false};
    bool useTex3DBindless{false};
    bool useBufferBindless{false};
    vstd::MD5 typeMD5;
    luisa::vector<std::byte> constant_ubo_data;
};
```

## LLVMCodegenStackData

`src/backends/common/spirv_llvm/llvm_codegen_stack_data.h` lines 30–113 (abridged):

```cpp
struct LLVMCodegenStackData : public vstd::IOperatorNewBase {
    LLVMCodegenUtility *util{nullptr};

    uint64_t struct_count{0}, const_count{0}, func_count{0}, temp_count{0};

    luisa::unordered_map<Type const *, uint64_t> type_counts;
    luisa::unordered_map<Type const *, llvm::StructType *> struct_types;
    luisa::unordered_map<uint64_t, uint64_t> const_types;
    luisa::unordered_map<uint64_t, llvm::Function *> func_types;

    luisa::unordered_map<uint32_t, llvm::Value *> variables;  // uid → alloca/param
    luisa::unordered_map<uint32_t, llvm::Value *> arguments;
    luisa::unordered_set<uint32_t> shared_variable_uids;

    struct LoopContext { llvm::BasicBlock *break_target; llvm::BasicBlock *continue_target; };
    luisa::vector<LoopContext> loop_stack;
    luisa::vector<llvm::BasicBlock *> break_stack;          // also used by switch

    llvm::SwitchInst *current_switch{nullptr};
    llvm::BasicBlock *switch_merge_block{nullptr};
    size_t switch_case_counter{0};
    luisa::unordered_map<uint64_t, llvm::Function *> atomic_funcs;

    // --- Helper state ---
    uint arg_offset{0};
    int64_t scope_count{-1};

    // --- Bindless tracking ---
    bool useTex2DBindless{false}, useTex3DBindless{false}, useBufferBindless{false};
    vstd::vector<std::pair<vstd::string, Type const *>> printers;
    luisa::vector<std::byte> constant_ubo_data;
    bool has_constant_ubo{false};

    static vstd::unique_ptr<LLVMCodegenStackData> Allocate(LLVMCodegenUtility *util);
    static void DeAllocate(vstd::unique_ptr<LLVMCodegenStackData> &&v);
    void Clear();
    uint64_t GetTypeCount(Type const *t);
    std::pair<uint64_t, bool> GetConstCount(uint64_t data_hash);
    llvm::Function *GetFunc(llvm::Function *f, uint64_t hash);
};
```

`Allocate`/`DeAllocate` use a global mutex-protected pool (`detail::LLVMCodegenGlobalPool` in `llvm_codegen_stack_data.cpp` lines 60–83).

## LLVMCodegenUtility

`src/backends/common/spirv_llvm/llvm_codegen_utility.h` lines 50–144 (abridged):

```cpp
class LLVMCodegenUtility {
public:
    vstd::unique_ptr<LLVMCodegenStackData> opt{};

    /// Main entry point for VK backend: codegen Function → SPIR-V result.
    [[nodiscard]] static LLVMCodegenResult CompileSPIRV(
        Function kernel,
        const ShaderOption &option);

private:
    std::unique_ptr<llvm::LLVMContext> _context;
    std::unique_ptr<llvm::Module> _module;
    std::unique_ptr<llvm::IRBuilder<>> _builder;
    llvm::Function *_current_function{nullptr};
    std::unique_ptr<llvm::TargetMachine> _target_machine;

public:
    LLVMCodegenUtility();
    ~LLVMCodegenUtility();

    // Accessors
    llvm::LLVMContext &context();
    llvm::Module &module();
    llvm::IRBuilder<> &builder();
    llvm::Function *current_function();
    void set_current_function(llvm::Function *f);

    // Type mapping
    [[nodiscard]] llvm::Type *ToLLVMType(Type const &type);
    [[nodiscard]] llvm::StructType *RegistStructType(Type const *type);
    void GetTypeName(Type const &type, vstd::StringBuilder &str);

    // Naming
    void GetVariableName(Function func, Variable const &v, vstd::StringBuilder &str);
    void GetVariableName(Function func, Variable::Tag tag, uint32_t id, vstd::StringBuilder &str);
    void GetFunctionName(Function callable, vstd::StringBuilder &result);
    void GetFunctionName(CallExpr const *expr, vstd::StringBuilder &result, LLVMStateVisitor &visitor);

    // Constants
    [[nodiscard]] llvm::Constant *CreateConstant(ConstantData const &data, llvm::Type *type);
    [[nodiscard]] llvm::GlobalVariable *CreateConstantGlobal(ConstantData const &data, llvm::Type *type);

    // Function codegen
    [[nodiscard]] llvm::Function *CodegenFunction(Function func);
    [[nodiscard]] llvm::Function *GetOrDeclareFunction(Function func);
    [[nodiscard]] llvm::Function *CodegenKernelEntry(Function kernel);

    // Temp variable name
    vstd::StringBuilder GetNewTempVarName();

    // Module
    [[nodiscard]] luisa::string ToString() const;
    void WriteBitcodeToFile(luisa::string_view path) const;
    void ResetModule();

    // SPIR-V
    void InitializeSPIRVModule();
    [[nodiscard]] luisa::vector<uint32_t> EmitSPIRV();
    void GenerateProperties(Function kernel,
                           LLVMCodegenResult::Properties &properties);
};
```

### CompileSPIRV (main entry)

`src/backends/common/spirv_llvm/llvm_codegen_utility.cpp` lines 1273–1312:

```cpp
LLVMCodegenResult LLVMCodegenUtility::CompileSPIRV(
    Function kernel,
    const ShaderOption &option) {

    LLVMCodegenResult result;
    // 1. Create utility and initialize SPIR-V module
    LLVMCodegenUtility util;
    util.InitializeSPIRVModule();

    // 2. Codegen the kernel function into LLVM IR
    util.CodegenFunction(kernel);

    // 3. Generate binding properties from kernel arguments
    util.GenerateProperties(kernel, result.properties);

    // 4. Collect bindless usage flags from stack data
    result.useTex2DBindless = util.opt->useTex2DBindless;
    result.useTex3DBindless = util.opt->useTex3DBindless;
    result.useBufferBindless = util.opt->useBufferBindless;

    // 5. Collect printer info
    result.printers = std::move(util.opt->printers);

    // 6. Collect constant UBO data
    result.constant_ubo_data = std::move(util.opt->constant_ubo_data);

    // 7. Emit SPIR-V binary via LLVM SPIRV target
    result.spv_bin = util.EmitSPIRV();

    // 8. Strip Addresses/Linkage capabilities and convert PtrAccessChain
    strip_addresses_capability(result.spv_bin);

    // 9. Validate and optimize the SPIR-V binary (mirrors XIR path post-processing)
    luisa_spirv_validate_post_llvm(result.spv_bin, "post-llvm");

    // 10. Compute type MD5 for caching
    result.typeMD5 = hlsl::CodegenUtility::GetTypeMD5(kernel);

    return result;
}
```

## Type Mapping

`src/backends/common/spirv_llvm/llvm_codegen_utility.cpp` lines 59–142:

| Luisa Type::Tag | LLVM Type |
|---|---|
| `BOOL` | `getInt1Ty()` |
| `INT8`/`UINT8` | `getInt8Ty()` |
| `INT16`/`UINT16` | `getInt16Ty()` |
| `INT32`/`UINT32` | `getInt32Ty()` |
| `INT64`/`UINT64` | `getInt64Ty()` |
| `FLOAT16` | `getHalfTy()` |
| `FLOAT32` | `getFloatTy()` |
| `FLOAT64` | `getDoubleTy()` |
| `FLOAT8_E4M3`/`FLOAT8_E5M2` | `getInt8Ty()` |
| `VECTOR` | `FixedVectorType::get(elem_type, dim)` |
| `MATRIX` | `ArrayType::get(FixedVectorType::get(float, dim), dim)` |
| `ARRAY` | `ArrayType::get(elem_type, dim)` |
| `STRUCTURE` | `RegistStructType()` → named `StructType::create()` |
| `BUFFER`/`TEXTURE`/`BINDLESS_ARRAY`/`ACCEL` | `getPtrTy(0)` (opaque pointer) |
| `COOPERATIVE_VECTOR`/`COOPERATIVE_VECTOR_REF`/`COOPERATIVE_MATRIX_REF` | `getInt32Ty()` |
| `CUSTOM` | `getPtrTy(0)` |

### RegistStructType

`src/backends/common/spirv_llvm/llvm_codegen_utility.cpp` lines 144–171. Creates a named struct type, caches in `opt->struct_types`, sets body from type members.

## Variable Naming

`src/backends/common/spirv_llvm/llvm_codegen_utility.cpp` lines 225–277:

| Variable::Tag | Name |
|---|---|
| `LOCAL` | `_V{uid}` |
| `SHARED` | `_S{uid}` |
| `REFERENCE` | `_R{uid}` |
| `BUFFER` | `_B{uid}` |
| `TEXTURE` | `_T{uid}` |
| `BINDLESS_ARRAY` | `_BA{uid}` |
| `ACCEL` | `_A{uid}` |
| `THREAD_ID` | `_thread_id` |
| `BLOCK_ID` | `_block_id` |
| `DISPATCH_ID` | `_dispatch_id` |
| `DISPATCH_SIZE` | `_dispatch_size` |
| `KERNEL_ID` | `_kernel_id` |
| `WARP_LANE_COUNT` | `_warp_lane_count` |
| `WARP_LANE_ID` | `_warp_lane_id` |

## Function Codegen

`src/backends/common/spirv_llvm/llvm_codegen_utility.cpp` lines 409–653:

1. Dedup via `opt->func_types` hash map
2. Save/restore `_builder` IP, `_current_function`, and `opt->variables` (enables recursive callable codegen)
3. Build `llvm::FunctionType` from return type + arguments. For kernels, arguments are **not** function parameters (Vulkan requires entry points with no parameters); they are emitted as global variables.
4. Create `llvm::Function` — `ExternalLinkage` for kernels, `InternalLinkage` for callables. Kernel entry is named `main` and gets:
   - `addFnAttr("hlsl.shader", "compute")`
   - `addFnAttr("hlsl.numthreads", "{x},{y},{z}")` from `func.block_size()`
5. Create entry BB, allocate locals + shared vars via `CreateAlloca`
6. **Kernel-specific argument lowering** (lines 512–623):
   - Resource arguments (`BUFFER`, `TEXTURE`, `BINDLESS_ARRAY`, `ACCEL`) and `REFERENCE` → `GlobalVariable` in addrspace(1)
   - Non-resource value arguments → packed into a struct `_Global` in addrspace(1), loaded into allocas
   - Builtin variables (`DISPATCH_ID`, `THREAD_ID`, `BLOCK_ID`) → `@llvm.spv.thread.id` / `@llvm.spv.thread.id.in.group` / `@llvm.spv.group.id` intrinsics; other builtins get undef allocas
7. Create `LLVMStateVisitor` → `VisitFunction(func)` to walk the AST body
8. If missing terminator: `RetVoid` or `Ret(UndefValue)`
9. `verifyFunction()` (warns on failure)
10. Restore saved state

## LLVMStateVisitor

`src/backends/common/spirv_llvm/llvm_state_visitor.h` lines 41–126 (abridged). Inherits `StmtVisitor` + `ExprVisitor`. Expression visitors set `_last_value`; statement visitors emit IR directly.

```cpp
class LLVMStateVisitor final : public StmtVisitor, public ExprVisitor {
public:
    Function f;
private:
    LLVMCodegenUtility *_util;
    llvm::LLVMContext &_ctx;
    llvm::Module &_module;
    llvm::IRBuilder<> &_builder;
    llvm::Value *_last_value{nullptr};
    llvm::BasicBlock *_entry_block{nullptr};
    llvm::SwitchInst *_current_switch{nullptr};
    llvm::BasicBlock *_switch_merge_block{nullptr};
public:
    // Expression visitors (set _last_value)
    void visit(const LiteralExpr *) override;
    void visit(const RefExpr *) override;
    void visit(const UnaryExpr *) override;
    void visit(const BinaryExpr *) override;
    void visit(const MemberExpr *) override;
    void visit(const AccessExpr *) override;
    void visit(const CastExpr *) override;
    void visit(const ConstantExpr *) override;
    void visit(const CallExpr *) override;
    void visit(const TypeIDExpr *) override;
    void visit(const StringIDExpr *) override;
    void visit(const FuncRefExpr *) override { LUISA_NOT_IMPLEMENTED(); }
    void visit(const CpuCustomOpExpr *) override { LUISA_NOT_IMPLEMENTED(); }
    void visit(const GpuCustomOpExpr *) override { LUISA_NOT_IMPLEMENTED(); }

    // Statement visitors
    void visit(const BreakStmt *) override;
    void visit(const ContinueStmt *) override;
    void visit(const ReturnStmt *) override;
    void visit(const ScopeStmt *) override;
    void visit(const IfStmt *) override;
    void visit(const LoopStmt *) override;
    void visit(const ForStmt *) override;
    void visit(const ExprStmt *) override;
    void visit(const SwitchStmt *) override;
    void visit(const SwitchCaseStmt *) override;
    void visit(const SwitchDefaultStmt *) override;
    void visit(const AssignStmt *) override;
    void visit(const CommentStmt *) override;
    void visit(const RayQueryStmt *) override;
    void visit(const AutoDiffStmt *) override;
    void visit(const PrintStmt *) override;
    void visit(const DebugBreakStmt *) override;

    // Helpers
    [[nodiscard]] llvm::Value *EvalExpr(Expression const *expr);
    [[nodiscard]] llvm::Type *ToLLVMType(Type const &type);
    [[nodiscard]] llvm::Value *GetVariable(uint32_t uid, Type const *type);
    void StoreVariable(uint32_t uid, llvm::Value *value);

private:
    void _push_loop(llvm::BasicBlock *break_target, llvm::BasicBlock *continue_target);
    void _pop_loop();
    void _codegen_builtin_call(CallOp op, const CallExpr *expr);
    // Math helpers
    llvm::Value *_emit_abs(llvm::Value *v, Type const &type);
    llvm::Value *_emit_min(llvm::Value *a, llvm::Value *b, Type const &type);
    llvm::Value *_emit_max(llvm::Value *a, llvm::Value *b, Type const &type);
    llvm::Value *_emit_clamp(llvm::Value *v, llvm::Value *lo, llvm::Value *hi, Type const &type);
    llvm::Value *_emit_lerp(llvm::Value *a, llvm::Value *b, llvm::Value *t, Type const &type);
    llvm::Value *_emit_length(llvm::Value *v);
    llvm::Value *_emit_normalize(llvm::Value *v);
    llvm::Value *_emit_dot(llvm::Value *a, llvm::Value *b);
    llvm::Value *_emit_cross(llvm::Value *a, llvm::Value *b);
    llvm::Value *_emit_all(llvm::Value *v);
    llvm::Value *_emit_any(llvm::Value *v);
};
```

### Variable Access

`src/backends/common/spirv_llvm/llvm_state_visitor.cpp` lines 48–73 (abridged):

```cpp
// GetVariable: loads from AllocaInst, returns direct Value for resources
llvm::Value *LLVMStateVisitor::GetVariable(uint32_t uid, Type const *type) {
    auto it = _util->opt->variables.find(uid);
    if (it != _util->opt->variables.end()) {
        auto *alloca = it->second;
        if (llvm::isa<llvm::AllocaInst>(alloca))
            return _builder.CreateLoad(ToLLVMType(*type), alloca);
        return alloca;  // resource — direct value
    }
    LUISA_ERROR_WITH_LOCATION("Variable {} not found.", uid);
}

// StoreVariable: stores to AllocaInst, errors otherwise
void LLVMStateVisitor::StoreVariable(uint32_t uid, llvm::Value *value) {
    auto it = _util->opt->variables.find(uid);
    if (it != _util->opt->variables.end()) {
        if (llvm::isa<llvm::AllocaInst>(it->second))
            _builder.CreateStore(value, it->second);
        return;
    }
    LUISA_ERROR_WITH_LOCATION("Cannot store to variable {}.", uid);
}
```

### BinaryExpr Scalar Broadcast

`src/backends/common/spirv_llvm/llvm_state_visitor.cpp` lines 192–210: when one operand is scalar and the other is vector, the scalar is broadcast via `insertelement` chain before the operation. The helper `broadcast_scalar` builds a `UndefValue` vector and inserts the scalar into every element.

### Builtin Call Codegen

`src/backends/common/spirv_llvm/llvm_state_visitor.cpp` lines 509–1367 (`_codegen_builtin_call`):

| Category | CallOp examples | IR strategy |
|---|---|---|
| Math intrinsics | `SQRT`, `RSQRT`, `SIN`, `COS`, `EXP`, `EXP2`, `LOG`, `LOG2`, `POW`, `FMA`, `COPYSIGN`, `FLOOR`, `CEIL`, `TRUNC`, `ROUND`, `FRACT` | `llvm::Intrinsic::getDeclarationIfExists()` (RSQRT = `1/sqrt`) |
| Bit intrinsics | `CLZ`, `CTZ`, `POPCOUNT`, `REVERSE` | `llvm::Intrinsic::getDeclarationIfExists()` |
| Manual math | `ABS` | `fabs` intrinsic or `select(neg, v, v<0)` |
| | `MIN`/`MAX` | `minnum`/`maxnum` intrinsic, or `select` for ints |
| | `CLAMP` | `max(min(v,hi), lo)` |
| | `SATURATE` | `clamp(v, 0, 1)` |
| | `LERP` | `a + t*(b-a)` |
| | `STEP` | `x>=edge ? 1.0 : 0.0` |
| | `SMOOTHSTEP` | `t*t*(3-2*t)` with `t=clamp((x-e0)/(e1-e0),0,1)` |
| | `DOT` | element-wise `fmul` + `fadd` chain |
| | `CROSS` | float3 formula |
| | `LENGTH` | `sqrt(dot(v,v))` |
| | `LENGTH_SQUARED` | `dot(v,v)` |
| | `NORMALIZE` | `v / sqrt(dot(v,v))` |
| | `REFLECT` | `i - 2*dot(n,i)*n` |
| | `ALL`/`ANY` | `and`/`or` reduce chain |
| Float tests | `ISINF`, `ISNAN` | `FCmpOEQ(abs(v), inf)` / `FCmpUNO(v, v)` |
| Linear algebra | `DETERMINANT` (2x2 only), `TRANSPOSE`, `INVERSE`, `OUTER_PRODUCT`, `MATRIX_COMPONENT_WISE_MULTIPLICATION`, `FACEFORWARD` | Implemented or stubbed |
| Vector make | `MAKE_FLOAT2..4`, `MAKE_INT2..4`, `MAKE_UINT2..4`, `MAKE_BOOL2..4`, `MAKE_SHORT2..4`, `MAKE_USHORT2..4`, `MAKE_LONG2..4`, `MAKE_ULONG2..4`, `MAKE_HALF2..4`, `MAKE_DOUBLE2..4`, `MAKE_BYTE2..4`, `MAKE_UBYTE2..4` | `insertelement` chain, supports sub-vector args and single-scalar broadcast |
| Matrix make | `MAKE_FLOAT2X2`, `3X3`, `4X4` | `insertvalue` chain |
| Reduce | `REDUCE_SUM` | element-wise `fadd` reduce |
| Select | `SELECT` | `CreateSelect(cond, true_val, false_val)` (AST order: false, true, cond) |
| Atomic | `ATOMIC_EXCHANGE`, `ATOMIC_FETCH_{ADD\|SUB\|AND\|OR\|XOR\|MIN\|MAX}` | `CreateAtomicRMW()` |
| | `ATOMIC_COMPARE_EXCHANGE` | `CreateAtomicCmpXchg()` |
| Buffer | `BUFFER_{READ\|WRITE}` (incl. `*_VOLATILE_*`) | `InBoundsGEP` + `Load`/`Store` |
| | `BYTE_BUFFER_{READ\|WRITE}` (incl. `*_VOLATILE_*`) | Stub: zero/null for read, no-op for write (SPIR-V logical addressing) |
| | `BUFFER_SIZE` | Stub: returns `0` |
| Bindless | `BINDLESS_BUFFER_READ`/`WRITE`, `TEXTURE_READ`/`WRITE` | Minimal stub; sets bindless flag, evaluates args, returns null |
| Sync | `SYNCHRONIZE_BLOCK` | `llvm.nvvm.barrier0` call |
| Control flow hints | `UNREACHABLE`, `ASSUME` | `CreateUnreachable()` / no-op |
| Constants | `ZERO`, `ONE` | `Constant::getNullValue` / `ConstantFP::get(ty, 1.0)` |
| Stubs | `WARP_*`, `DDX`, `DDY`, `CLOCK`, `ACOS`, `ASIN`, `ATAN`, `ATAN2`, `TAN`, `COSH`, `SINH`, `TANH`, `TRANSPOSE`, `INVERSE`, `RAY_QUERY_*`, `PRINT`, `DEBUG_BREAK` | `LUISA_NOT_IMPLEMENTED()` (returns null/undef) |

### Adding a new builtin

Typical steps in `llvm_state_visitor.cpp` `_codegen_builtin_call`:

1. Add a `case CallOp::YOUR_OP:` block.
2. Evaluate arguments with `EvalExpr(args[i])`.
3. Emit LLVM IR via `_builder` or `llvm::Intrinsic::getDeclarationIfExists(&_module, llvm::Intrinsic::..., {types})`.
4. Set `_last_value` to the result (or `nullptr` for void ops).
5. If the op affects binding properties (e.g. bindless), set the corresponding `opt->use*Bindless` flag.
6. For unimplemented ops, use `LUISA_NOT_IMPLEMENTED()` and return a safe null/undef value so compilation tests can still pass.

Example pattern for a unary float intrinsic:

```cpp
case CallOp::YOUR_OP: {
    auto *v = EvalExpr(args[0]);
    auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
        &_module, llvm::Intrinsic::your_llvm_intrinsic, {v->getType()});
    _last_value = _builder.CreateCall(intrinsic, {v});
    break;
}
```

### Statement Visitors

`src/backends/common/spirv_llvm/llvm_state_visitor.cpp`:

| Statement | Lines | Strategy |
|---|---|---|
| `BreakStmt` | 1525–1530 | `CreateBr(break_stack.back())` |
| `ContinueStmt` | 1532–1537 | `CreateBr(loop_stack.back().continue_target)` |
| `ReturnStmt` | 1539–1546 | `CreateRet(val)` or `CreateRetVoid()` |
| `ScopeStmt` | 1548–1556 | Iterate children, stop if block has terminator |
| `ExprStmt` | 1558–1560 | `EvalExpr(expr)` and discard result |
| `AssignStmt` | 1562–1679 | Resolves RefExpr/MemberExpr/AccessExpr LHS, issues `CreateStore` |
| `IfStmt` | 1681–1721 | `then_bb` + optional `else_bb` + `merge_bb`, `CreateCondBr`; normalizes cond to i1 |
| `LoopStmt` | 1723–1744 | `header`/`body`/`exit` BBs, `_push_loop`/`_pop_loop` |
| `ForStmt` | 1746–1790 | `for_cond`/`for_body`/`for_step`/`for_exit`, evaluates step into loop var |
| `SwitchStmt` | 1792–1860 | Collects `CaseInfo` from body, `CreateSwitch(expr, default_bb)`, pushes merge to `break_stack` |
| `SwitchCaseStmt` | 1862–1867 | Iterates case body, stops at terminator |
| `SwitchDefaultStmt` | 1869–1874 | Iterates default body, stops at terminator |

## SPIR-V Emission

### Target Initialization

`src/backends/common/spirv_llvm/spirv_llvm.cpp` lines 8–20:

```cpp
extern void LLVMInitializeSPIRVTarget();
extern void LLVMInitializeSPIRVTargetInfo();
extern void LLVMInitializeSPIRVTargetMC();
extern void LLVMInitializeSPIRVAsmPrinter();

void InitializeLLVMSPIRVTarget() {
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTargetMC();
    LLVMInitializeSPIRVAsmPrinter();
}
```

These symbols (from `LLVMSPIRVCodeGen`, `LLVMSPIRVTargetInfo`, `LLVMSPIRVTargetMC`, `LLVMSPIRVAsmPrinter` libraries) are explicitly called to prevent linker dead-stripping on Windows.

### InitializeSPIRVModule

`src/backends/common/spirv_llvm/llvm_codegen_utility.cpp` lines 695–729: Sets `spirv64-unknown-vulkan1.2` triple, looks up target via `TargetRegistry` ("spirv64"), creates `TargetMachine` with `Reloc::PIC_`, `CodeModel::Small`, `CodeGenOptLevel::Default`. Sets data layout from target machine. `spirv64` is used because `spirv32` crashes in `SPIRVLegalizePointerCast`.

### EmitSPIRV

`src/backends/common/spirv_llvm/llvm_codegen_utility.cpp` lines 948–1018:

1. `ScalarizeAggregateMemOps(module)` — LLVM SPIR-V backend cannot legalize aggregate loads/stores
2. `LowerAggregateReturns(module)` — LLVM's `SPIRVPrepareFunctions` pass mutates aggregate returns to i32, breaking IR; proactively convert to void + out-param
3. `ScalarizeAggregateMemOps(module)` — re-run (step 2 may introduce new aggregate ops)
4. `legacy::PassManager` + `addPassesToEmitFile(ObjectFile)` → run
5. Check for ELF magic (`.spv` section extraction not implemented — raw buffer returned)

The emitted binary is then post-processed by `CompileSPIRV` (see below), not inside `EmitSPIRV`.

### SPIR-V Post-Processing

`src/backends/common/spirv_llvm/llvm_codegen_utility.cpp` lines 1145–1271:

After `EmitSPIRV`, `CompileSPIRV` runs two post-processing steps before returning the result:

1. **`strip_addresses_capability(spv_bin)`** (lines 1149–1239)
   - Removes `OpCapability Addresses` and `OpCapability Linkage`
   - Strips `OpDecorate LinkageAttributes`
   - Converts `OpPtrAccessChain`/`OpInBoundsPtrAccessChain` to `OpAccessChain`/`OpInBoundsAccessChain` by dropping the Element operand
   - This fixes the physical addressing emitted by the LLVM `spirv64` target so the binary conforms to Vulkan logical addressing.

2. **`luisa_spirv_validate_post_llvm(spv_bin, "post-llvm")`** (lines 1243–1271)
   - Uses `spvtools::SpirvTools` with `SPV_ENV_VULKAN_1_2`
   - Mirrors the XIR path validation; fails with a detailed message if the binary is invalid

### ScalarizeAggregateMemOps

`src/backends/common/spirv_llvm/llvm_codegen_utility.cpp` lines 784–815. Iterates all `LoadInst`/`StoreInst` with aggregate types, replaces with recursive `BuildAggregateLoad`/`StoreAggregateValue` (lines 733–779).

### LowerAggregateReturns

`src/backends/common/spirv_llvm/llvm_codegen_utility.cpp` lines 821–946. Three-pass transform:
1. Create new functions with added `ptr` out-parameter
2. `CloneFunctionInto` body, replace `ret {agg}` → `store {agg}, ret_ptr; ret void`
3. Update call sites: pass existing alloca or allocate temp + load

## Binding Properties

`src/backends/common/spirv_llvm/llvm_codegen_utility.cpp` lines 1020–1139:

Walks kernel arguments, emits `hlsl::Property` entries:
- Non-resource arguments → `_Global` structured buffer at `register_index=0`
- `BUFFER` (writable → `RWStructuredBuffer`, readable → `StructuredBuffer`)
- `TEXTURE` (writable → `UAVTextureHeap`, readable → `SRVTextureHeap`)
- `BINDLESS_ARRAY` → `StructuredBuffer`
- `ACCEL` → `SPIRVAccel`

Bindless flags are detected from `propagated_builtin_callables` using a broad set of `CallOp` values (e.g. `BINDLESS_BUFFER_READ`/`WRITE`, `UNIFORM_BINDLESS_BUFFER_*`, `TYPED_BINDLESS_BUFFER_*`, `BINDLESS_TEXTURE2D_*`, `BINDLESS_TEXTURE3D_*`, and their `UNIFORM_`/`TYPED_` variants).

## Build

`src/backends/common/spirv_llvm/xmake.lua`: static lib `lc-spirv-llvm`, depends on `lc-vstl`, `lc-runtime`, and `spirv-tools`. Links all `LLVM*.lib` from LLVM build dir (configured via `lc_llvm_path`), excluding `LLVM-C`. Platform syslinks:
- Windows: `Version`, `advapi32`, `Shcore`, `user32`, `shell32`, `Ole32`, `Ws2_32`, `ntdll`
- Linux: `uuid`
- macOS: `CoreFoundation`

## Debugging

- The backend always writes `llvm_ir_debug.ll` in the working directory before running the SPIR-V emission passes (lines 976–981). Read this file to inspect the LLVM IR handed to the LLVM SPIR-V backend.
- `verifyFunction()` is called after each function is codegen'd; `verifyModule()` is called before the emission pass manager runs and errors out on failure.
- After emission, `luisa_spirv_validate_post_llvm()` validates the binary with SPIR-V Tools and prints a detailed message on failure.

### Common debugging workflow

1. Run the failing kernel compile; the backend writes `llvm_ir_debug.ll` next to the executable.
2. Open `llvm_ir_debug.ll` and search for the kernel name (`main`) or the last `_V{uid}` variable before the crash.
3. Run `llvm-as llvm_ir_debug.ll` to sanity-check the IR if you have LLVM tools available.
4. If the crash happens inside the LLVM SPIR-V backend, look for:
   - Aggregate loads/stores that were not scalarized.
   - Functions returning structs/arrays that were not lowered.
   - `ptr` operands in contexts where logical addressing is required.
5. If the crash is a SPIR-V validation error after emission, the validator message points to the offending instruction and rule (e.g. missing `Addresses` capability stripping).

## Pitfalls

1. **Aggregate load/store**: SPIR-V Vulkan backend crashes on aggregate memory ops — always run `ScalarizeAggregateMemOps` before emission.
2. **Aggregate returns**: `SPIRVPrepareFunctions` breaks IR by mutating return type to i32 — run `LowerAggregateReturns` first, then scalarize again.
3. **spirv32**: Crashes in `SPIRVLegalizePointerCast` — always use `spirv64`.
4. **BYTE_BUFFER ops**: Not supported in logical addressing — reads return zero/null, writes are no-op.
5. **Intrinsics**: Use `getDeclarationIfExists()`, not `getDeclaration()` — some may be absent; the code does not null-check before calling, so missing intrinsics will assert.
6. **Recursive callables**: `CodegenFunction` saves/restores `_builder` IP, `_current_function`, and `opt->variables`.
7. **Kernel entry points**: Vulkan requires entry functions to have no parameters. The backend lowers resources/value args to addrspace(1) globals and builtins to SPIR-V intrinsics; do not add kernel args as `llvm::Function` parameters.
8. **Post-processing is mandatory**: Raw output from the LLVM `spirv64` target uses physical addressing (`OpCapability Addresses`). `CompileSPIRV` must run `strip_addresses_capability()` and `luisa_spirv_validate_post_llvm()`; calling `EmitSPIRV()` alone does not produce valid Vulkan SPIR-V.
9. **Bindless detection**: Bindless flags are set both during property generation (`GenerateProperties`) and at codegen sites (`BUFFER_READ`/`WRITE` on a `BINDLESS_ARRAY`, explicit `BINDLESS_BUFFER_*` calls). Keep both in sync when adding new bindless ops.


## LLVM API Cheatsheet

APIs actually called by the spirv_llvm code, organized by LLVM header.

### `<llvm/IR/IRBuilder.h>` — `llvm::IRBuilder<>`

The main instruction factory. All methods return the created instruction.

| Category | Method | Used for |
|---|---|---|
| **Insertion point** | `SetInsertPoint(BasicBlock*)` | Set where new instrs go |
| | `GetInsertBlock()` | Current block |
| | `saveIP()` / `restoreIP(InsertPoint)` | Save/restore position for recursive codegen |
| **Terminators** | `CreateRetVoid()` | `return;` |
| | `CreateRet(Value*)` | `return val;` |
| | `CreateBr(BasicBlock*)` | Unconditional branch |
| | `CreateCondBr(Value*, BasicBlock*, BasicBlock*)` | `if (cond) then else` |
| | `CreateSwitch(Value*, BasicBlock*, unsigned NumCases)` | Switch; add cases via `SwitchInst::addCase()` |
| | `CreateUnreachable()` | `unreachable` |
| **Integer arithmetic** | `CreateAdd/Sub/Mul/UDiv/SDiv/URem/SRem(LHS, RHS)` | Integer ops |
| | `CreateNeg(Value*)` | `-x` |
| | `CreateAnd/Or/Xor(LHS, RHS)` | Bitwise |
| | `CreateShl/LShr/AShr(LHS, RHS)` | Shift |
| | `CreateNot(Value*)` | `~x` |
| **Float arithmetic** | `CreateFAdd/FSub/FMul/FDiv/FRem(L, R)` | Float ops |
| | `CreateFNeg(Value*)` | `-x` |
| **Casts** | `CreateTrunc/ZExt/SExt(V, Type*)` | Int width change |
| | `CreateFPToUI/FPToSI/UIToFP/SIToFP(V, Type*)` | Int↔float |
| | `CreateFPTrunc/FPExt(V, Type*)` | Float width change |
| | `CreateBitCast(V, Type*)` | Reinterpret bits |
| **Compare** | `CreateICmpEQ/NE/UGT/UGE/ULT/ULE/SGT/SGE(L,R)` | Integer cmp → i1 |
| | `CreateFCmpOEQ/ONE/OGT/OGE/OLT/OLE(L,R)` | Ordered float cmp → i1 |
| | `CreateFCmpUNO(L,R)` | `isnan` (unordered) |
| **Memory** | `CreateAlloca(Type*, Value* ArraySize, Name)` | Stack allocation → `AllocaInst*` |
| | `CreateLoad(Type*, Value* Ptr)` | Load from pointer |
| | `CreateStore(Value* Val, Value* Ptr)` | Store to pointer |
| | `CreateGEP(Type*, Value* Ptr, {indices})` | Pointer arithmetic |
| | `CreateInBoundsGEP(Type*, Value* Ptr, {indices})` | Bounds-checked GEP |
| | `CreateStructGEP(Type*, Value* Ptr, unsigned Idx)` | Struct field GEP |
| **Aggregate** | `CreateExtractValue(Value* Agg, {indices})` | Extract struct/array member |
| | `CreateInsertValue(Value* Agg, Value* Elt, {indices})` | Insert into struct/array |
| | `CreateExtractElement(Value* Vec, Value* Idx)` | Extract vector element |
| | `CreateInsertElement(Value* Vec, Value* Elt, Value* Idx)` | Insert vector element |
| **Other** | `CreateSelect(Value* Cond, Value* T, Value* F)` | `cond ? T : F` |
| | `CreateCall(FunctionType*, Value* Callee, Args)` | Function call |
| | `CreateAtomicRMW(Op, Ptr, Val, MaybeAlign, Ordering)` | Atomic read-modify-write |
| | `CreateAtomicCmpXchg(Ptr, Expected, Desired, MaybeAlign, Succ, Fail)` | Atomic CAS |
| **Type getters** | `getInt1Ty()`, `getInt8Ty()`, `getInt16Ty()`, `getInt32Ty()`, `getInt64Ty()` | Integer types |
| | `getHalfTy()`, `getFloatTy()`, `getDoubleTy()`, `getVoidTy()` | FP/void types |
| | `getPtrTy(unsigned AddrSpace=0)` | Opaque pointer |
| **Intrinsics** | `CreateIntrinsic(Intrinsic::ID, Types, Args)` | Generic intrinsic call |

### `<llvm/IR/Constants.h>` — `llvm::Constant*` hierarchy

```cpp
llvm::ConstantInt::get(IntegerType*, uint64_t Val);        // integer constant
llvm::ConstantInt::getTrue/getFalse(LLVMContext&);         // bool constants
llvm::ConstantFP::get(Type*, double Val);                  // float constant
llvm::ConstantFP::get(Type*, const APFloat&);              // float via APFloat
llvm::ConstantFP::getInfinity(Type*);                      // +inf
llvm::Constant::getNullValue(Type*);                       // zero/null
llvm::UndefValue::get(Type*);                              // undef
llvm::ConstantVector::get(ArrayRef<Constant*>);            // vector constant
llvm::ConstantArray::get(ArrayType*, ArrayRef<Constant*>); // array constant
llvm::ConstantStruct::get(StructType*, ArrayRef<Constant*>);// struct constant
llvm::PoisonValue::get(Type*);                             // poison
```

### `<llvm/IR/DerivedTypes.h>` — Compound types

```cpp
llvm::StructType::create(LLVMContext&, StringRef Name);    // opaque struct
void structTy->setBody(ArrayRef<Type*>);                   // define members
llvm::FixedVectorType::get(Type* Elem, unsigned NumElts);  // <N x T>
llvm::ArrayType::get(Type* Elem, uint64_t NumElts);        // [N x T]
llvm::FunctionType::get(Type* Ret, ArrayRef<Type*> Params, bool VarArg);
```

### `<llvm/IR/Function.h>` — `llvm::Function`

```cpp
llvm::Function::Create(FunctionType*, Linkage, StringRef, Module*);
llvm::Function::Create(FunctionType*, Linkage, unsigned AddressSpace, StringRef, Module*); // with addr space
void func->addFnAttr(StringRef Key, StringRef Value);      // e.g. "hlsl.shader", "compute"
Argument* func->getArg(unsigned N);                        // get N-th argument
```

### `<llvm/IR/BasicBlock.h>` — `llvm::BasicBlock`

```cpp
llvm::BasicBlock::Create(LLVMContext&, StringRef Name, Function* Parent);
Instruction* bb->getTerminatorOrNull();                    // null if not terminated
```

### `<llvm/IR/Intrinsics.h>` — Intrinsic declarations

```cpp
// Always use getDeclarationIfExists — returns nullptr if unavailable
llvm::Function *f = llvm::Intrinsic::getDeclarationIfExists(Module*, Intrinsic::ID, {Type*...});
// Examples: Intrinsic::sqrt, sin, cos, exp, exp2, log, log2, pow, fma,
//           copysign, floor, ceil, trunc, round, fabs, minnum, maxnum,
//           ctlz, cttz, ctpop, bitreverse
```

### `<llvm/IR/Module.h>` — `llvm::Module`

```cpp
llvm::Function *m->getFunction(StringRef Name);            // lookup by name
void m->setTargetTriple(llvm::Triple);                     // e.g. "spirv64-unknown-vulkan1.2"
void m->setDataLayout(DataLayout);
void m->print(raw_ostream&, ...);                          // dump IR text
```

### `<llvm/Target/TargetMachine.h>` — `llvm::TargetMachine`

```cpp
auto *t = llvm::TargetRegistry::lookupTarget("spirv64", ErrorStr);
auto *tm = t->createTargetMachine(Triple, CPU, Features, TargetOptions,
                                   std::optional<Reloc::Model>(Reloc::PIC_),
                                   std::optional<CodeModel::Model>(CodeModel::Small),
                                   CodeGenOptLevel::Default, false);
DataLayout dl = tm->createDataLayout();
bool failed = tm->addPassesToEmitFile(PassManager&, raw_ostream&, nullptr, CodeGenFileType);
```

### `<llvm/IR/Verifier.h>` — Verification

```cpp
bool llvm::verifyFunction(Function&, raw_ostream*);        // returns true if broken
bool llvm::verifyModule(Module&, raw_ostream*);
```

### `<llvm/Transforms/Utils/Cloning.h>` — Cloning

```cpp
void llvm::CloneFunctionInto(Function *NewFunc, Function *OldFunc,
                             ValueToValueMapTy &VMap, CloneFunctionChangeType,
                             SmallVectorImpl<ReturnInst*> &Returns);
```

## Luisa AST Reference

### `include/luisa/ast/type.h` — `Type`

```cpp
enum struct Type::Tag { BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32,
    INT64, UINT64, FLOAT16, FLOAT32, FLOAT64, FLOAT8_E4M3, FLOAT8_E5M2,
    VECTOR, MATRIX, ARRAY, STRUCTURE,
    BUFFER, TEXTURE, BINDLESS_ARRAY, ACCEL,
    COOPERATIVE_VECTOR, COOPERATIVE_VECTOR_REF, COOPERATIVE_MATRIX_REF, CUSTOM };

Type::Tag tag();                     const Type *element();        // inner type
size_t dimension();                  luisa::span<const Type*> members();   // struct members
size_t size();                       size_t alignment();
uint64_t hash();                     bool is_float/int/vector/matrix/array/structure/buffer/texture/...();
```

### `include/luisa/ast/variable.h` — `Variable`

```cpp
enum struct Variable::Tag { LOCAL, SHARED, REFERENCE,
    BUFFER, TEXTURE, BINDLESS_ARRAY, ACCEL,
    THREAD_ID, BLOCK_ID, DISPATCH_ID, DISPATCH_SIZE, KERNEL_ID,
    WARP_LANE_COUNT, WARP_LANE_ID, RASTER_OBJECT_ID, RASTER_BARYCENTRICS };

uint32_t uid();              const Type *type();          Tag tag();
bool is_local/shared/reference/resource/builtin();
```

### `include/luisa/ast/function.h` — `Function`

```cpp
enum struct Function::Tag { KERNEL, CALLABLE, RASTER_STAGE };

Tag tag();                            luisa::span<const Variable> arguments();
luisa::span<const Variable> builtin_variables();
luisa::span<const Variable> local_variables();
luisa::span<const Variable> shared_variables();
const ScopeStmt *body();              const Type *return_type();
uint64_t hash();                      luisa::string_view name();
CallOpSet propagated_builtin_callables();
Usage variable_usage(uint32_t uid);   bool requires_atomic();
```

### `include/luisa/ast/expression.h` — Expression hierarchy

| Class | Key members |
|---|---|
| `UnaryExpr` | `op()`: `UnaryOp::PLUS\|MINUS\|NOT\|BIT_NOT`, `operand()` |
| `BinaryExpr` | `op()`: `BinaryOp::ADD\|SUB\|MUL\|DIV\|MOD\|BIT_AND\|BIT_OR\|BIT_XOR\|SHL\|SHR\|AND\|OR\|LESS\|GREATER\|LESS_EQUAL\|GREATER_EQUAL\|EQUAL\|NOT_EQUAL`, `lhs()`, `rhs()` |
| `MemberExpr` | `self()`, `is_swizzle()`, `swizzle_index(uint)`, `member_index()` |
| `AccessExpr` | `range()`, `index()` |
| `LiteralExpr` | `value()`: variant of `bool/float/double/half/int/uint/vector/matrix` |
| `RefExpr` | `variable()`: returns `Variable` (uid + type + tag) |
| `ConstantExpr` | `data()`: `ConstantData` (type + raw bytes) |
| `CallExpr` | `op()`: `CallOp`, `arguments()`, `is_builtin()`, `is_custom()`, `custom()` |
| `CastExpr` | `op()`: `CastOp::STATIC\|BITWISE`, `expression()` |
| `TypeIDExpr` | `data_type()`: returns `uint64_t` hash |
| `StringIDExpr` | `data()`: returns `uint64_t` hash |

### `include/luisa/ast/statement.h` — Statement hierarchy

| Class | Key members |
|---|---|
| `BreakStmt` | none |
| `ContinueStmt` | none |
| `ReturnStmt` | `expression()`: nullable |
| `ScopeStmt` | `statements()`: span of `Statement*`, `append()`, `pop()` |
| `IfStmt` | `condition()`, `true_branch()`, `false_branch()` |
| `LoopStmt` | `body()` |
| `ForStmt` | `variable()`, `condition()`, `step()`, `body()` |
| `AssignStmt` | `lhs()`, `rhs()` |
| `ExprStmt` | `expression()` |
| `SwitchStmt` | `expression()`, `body()` (contains SwitchCase/SwitchDefault) |
| `SwitchCaseStmt` | `expression()`, `body()` |
| `SwitchDefaultStmt` | `body()` |
| `CommentStmt` | `comment()`: `string_view` |

### `include/luisa/ast/op.h` — Operators

`UnaryOp`: `PLUS`, `MINUS`, `NOT`, `BIT_NOT`

`BinaryOp`: `ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `BIT_AND`, `BIT_OR`, `BIT_XOR`, `SHL`, `SHR`, `AND`, `OR`, `LESS`, `GREATER`, `LESS_EQUAL`, `GREATER_EQUAL`, `EQUAL`, `NOT_EQUAL`

`CastOp`: `STATIC`, `BITWISE`

`CallOp`: ~430+ values (see `include/luisa/ast/op.h` lines 79–507). Key groups used/mentioned by this backend:
- **Math**: `ABS`, `MIN`, `MAX`, `CLAMP`, `SATURATE`, `LERP`, `STEP`, `SMOOTHSTEP`
- **Trig/Exp**: `SQRT`, `RSQRT`, `SIN`, `COS`, `EXP`, `EXP2`, `EXP10`, `LOG`, `LOG2`, `LOG10`, `POW`, `FMA`, `COPYSIGN`, `FLOOR`, `CEIL`, `TRUNC`, `ROUND`, `FRACT`
- **Float tests**: `ISINF`, `ISNAN`
- **Bit**: `CLZ`, `CTZ`, `POPCOUNT`, `REVERSE`
- **Vector**: `DOT`, `CROSS`, `LENGTH`, `LENGTH_SQUARED`, `NORMALIZE`, `ALL`, `ANY`, `REFLECT`, `FACEFORWARD`, `REDUCE_SUM`
- **Linear algebra**: `DETERMINANT`, `TRANSPOSE`, `INVERSE`, `OUTER_PRODUCT`, `MATRIX_COMPONENT_WISE_MULTIPLICATION`
- **Make**: `MAKE_FLOAT2..4`, `MAKE_INT2..4`, `MAKE_UINT2..4`, `MAKE_BOOL2..4`, `MAKE_SHORT2..4`, `MAKE_USHORT2..4`, `MAKE_LONG2..4`, `MAKE_ULONG2..4`, `MAKE_HALF2..4`, `MAKE_DOUBLE2..4`, `MAKE_BYTE2..4`, `MAKE_UBYTE2..4`, `MAKE_FLOAT2X2/3X3/4X4`
- **Select**: `SELECT`
- **Atomic**: `ATOMIC_EXCHANGE`, `ATOMIC_FETCH_{ADD|SUB|AND|OR|XOR|MIN|MAX}`, `ATOMIC_COMPARE_EXCHANGE`
- **Buffer**: `BUFFER_{READ|WRITE}` (and `BUFFER_VOLATILE_*`), `BYTE_BUFFER_{READ|WRITE}` (and `BYTE_BUFFER_VOLATILE_*`), `BUFFER_SIZE`
- **Bindless/Texture**: `BINDLESS_BUFFER_READ`/`WRITE`, `TEXTURE_READ`/`WRITE`; many `BINDLESS_TEXTURE2D_*`, `BINDLESS_TEXTURE3D_*`, `UNIFORM_BINDLESS_*`, `TYPED_BINDLESS_*`, `TYPED_UNIFORM_BINDLESS_*` variants used for property detection
- **Sync**: `SYNCHRONIZE_BLOCK`
- **Hints/Utility**: `UNREACHABLE`, `ASSUME`, `ZERO`, `ONE`
- **Stubs**: `WARP_*`, `DDX`, `DDY`, `CLOCK`, `ACOS`, `ACOSH`, `ASIN`, `ASINH`, `ATAN`, `ATAN2`, `ATANH`, `TAN`, `COSH`, `SINH`, `TANH`, `TRANSPOSE`, `INVERSE`, `RAY_QUERY_*`, `PRINT`, `DEBUG_BREAK`

`CallOpSet`: bitset tracking which `CallOp` values are used. `test(CallOp)`, `mark(CallOp)`, `uses_atomic()`, `uses_raytracing()`, etc.

### `include/luisa/ast/constant_data.h` — `ConstantData`

```cpp
const Type *type();            const std::byte *raw();         uint64_t hash();
static ConstantData create(const Type*, const void *data, size_t size);
```

### `include/luisa/ast/usage.h` — `Usage`

```cpp
enum struct Usage : uint32_t { NONE=0, READ=0x01, WRITE=0x02, READ_WRITE=READ|WRITE };
```

### Visitor pattern

```cpp
// Expression visitors: all return void, set _last_value
void visit(const UnaryExpr *expr) override;   // etc.
// Statement visitors: emit IR directly
void visit(const BreakStmt *stmt) override;   // etc.
// Entry
void VisitFunction(Function func);
// Eval helper
llvm::Value *EvalExpr(Expression const *expr);  // expr->accept(*this); return _last_value;
```

## Related Skills

- [`glslang`](glslang/SKILL.md) — SPIR-V builder API; useful for understanding SPIR-V instruction-level details and decorations.
- [`hlsl`](hlsl/SKILL.md) — HLSL string backend that this LLVM backend mirrors (`CodegenUtility`, `CodegenStackData`, `Property` reuse).
- [`spv-opt`](spv-opt/SKILL.md) — SPIRV-Tools optimizer passes; the validation/post-processing path here is conceptually related.
- [`backend_architecture`](backend_architecture/SKILL.md) — Device interface, backend plugin loading, and command encoding; explains where `lc-spirv-llvm` fits in the backend stack.
- [`xmake`](xmake/SKILL.md) — Build options and target patterns for LuisaCompute.
