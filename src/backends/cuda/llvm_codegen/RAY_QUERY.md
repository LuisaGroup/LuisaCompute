# Ray Query Support for CUDA LLVM Codegen

## Status: ⚠️ PARTIALLY WORKING - NEEDS FIX

**Last Updated:** 2026-02-28
**Current Implementer:** Needs handoff to Gemini

## Test Results

| Test | Status | Details |
|------|--------|---------|
| `test_ray_query_simple` | ✅ **PASS** | Both Test 1 & Test 2 work correctly |
| `test_procedural` | ⚠️ **PARTIAL** | Only ~28,000 non-black pixels vs 251,000 in AST codegen |
| `test_procedural_callable` | ⚠️ **UNTESTED** | Likely same issues as test_procedural |

## What's Working

1. **Basic Ray Query Infrastructure**
   - Handler detection via intrinsic scanning
   - Ray query loop extraction from kernel
   - Handler call removal from kernel (prevents illegal inlining)
   - OptiX entry point generation (`__anyhit__`, `__intersection__`)
   - Inline pass that only inlines into entry points

2. **Simple Ray Queries**
   - Triangle intersection works
   - Basic committed hit data retrieval
   - Test 1 & 2 in test_ray_query_simple pass

3. **Handler Cloning Architecture**
   - Creates two versions of each handler:
     - Surface version: state constant = SURFACE_CANDIDATE (1)
     - Procedural version: state constant = PROCEDURAL_CANDIDATE (2)
   - LLVM constant propagation should fold switch statements and eliminate dead code

## What's Broken

### 1. Handler Cloning Argument Remapping

**Problem:** `_clone_and_lower_handler()` creates new function versions but argument references aren't properly remapped.

**Error:** `Referring to an argument in another function!`

**Location:** `cuda_codegen_llvm_impl.cpp`, `_clone_and_lower_handler()` function

**Current Code (Broken):**
```cpp
// Clone blocks with vmap
llvm::ValueToValueMapTy vmap;
for (auto &block : *handler) {
    auto new_block = llvm::BasicBlock::Create(_llvm_context, block.getName(), new_handler);
    vmap[&block] = new_block;
}

// Clone instructions
for (auto &block : *handler) {
    auto new_block = llvm::cast<llvm::BasicBlock>(vmap[&block]);
    for (auto &inst : block) {
        auto new_inst = inst.clone();
        // ... insert instruction
        vmap[&inst] = new_inst;
    }
}

// Remap operands
for (auto &block : *new_handler) {
    for (auto &inst : block) {
        llvm::RemapInstruction(&inst, vmap, llvm::RF_None);
    }
}
```

**What's Missing:**
- Original handler's arguments aren't mapped to new_handler's arguments in vmap
- When blocks are moved to final_handler, argument references still point to old function
- Need to map arguments BEFORE cloning instructions

**Attempted Fix (Still Broken):**
```cpp
// Map original handler's arguments to new_handler's arguments
auto orig_arg = handler->arg_begin();
auto new_arg = new_handler->arg_begin();
for (; orig_arg != handler->arg_end(); ++orig_arg, ++new_arg) {
    vmap[&*orig_arg] = &*new_arg;
}
```

### 2. State Constant Not Triggering Dead Code Elimination

**Problem:** Even with separate handler versions, the switch on `state()` doesn't get folded.

**Expected:** LLVM should see constant state value and eliminate unreachable branches.

**Actual:** Both code paths remain, causing intersection program to have surface candidate code (with illegal `optixGetTriangleBarycentrics`).

**Test:** Check generated LLVM IR for dead code elimination:
```bash
cat debug_after_inline.ll | grep -A5 "switch.*state"
```

### 3. Result Struct Sharing

**Problem:** Result pointer passing between entry point and handler may be incorrect.

**Current Flow:**
1. Entry point allocates `result_alloca`
2. Entry point calls handler with `result_alloca` as last argument
3. Handler writes to `result_ptr` (last argument)
4. Entry point reads from `result_alloca`

**Verification Needed:** Ensure both sides use the same memory location.

## Architecture

The correct architecture is implemented but buggy:

```
Original Handler
    ↓ Clone
Surface Handler (state = SURFACE_CANDIDATE) → __anyhit__ray_query
    ↓ Clone  
Procedural Handler (state = PROCEDURAL_CANDIDATE) → __intersection__ray_query

LLVM should fold:
  switch (state) {          →  switch (1) {
    case 1: surface();          case 1: surface();
    case 2: procedural();       case 2: procedural(); ← Dead code, eliminated
  }                          }
```

## Critical Files

- `cuda_codegen_llvm_impl.cpp` - Main implementation, contains bugs
- `cuda_codegen_llvm_impl.h` - Declarations
- `cuda_codegen_llvm_impl_type.cpp` - Type definitions
- `cuda_codegen_llvm_impl_resource.cpp` - OptiX inline ASM helpers

## Key Functions to Fix

1. **`_clone_and_lower_handler()`** - Lines ~1216-1500
   - Must properly remap ALL value references
   - Arguments, blocks, instructions all need vmap entries
   - Use `llvm::CloneFunction()` instead of manual cloning?

2. **`_generate_intersection_program()`** - Lines ~974-1100
   - Verify only procedural handlers are called
   - Check that surface code is eliminated

3. **`_generate_anyhit_program()`** - Lines ~1119-1214
   - Verify only surface handlers are called
   - Check that procedural code is eliminated

## Testing

Run tests with:
```bash
cd cmake-build-debug

# Test simple (should pass)
LUISA_EXPERIMENTAL_LLVM_CODEGEN=1 ./bin/test_ray_query_simple cuda

# Test procedural (currently broken)
LUISA_EXPERIMENTAL_LLVM_CODEGEN=1 ./bin/test_procedural cuda

# Compare with AST codegen
./bin/test_procedural cuda  # AST version
mv test_procedural.png test_procedural_ast.png
LUISA_EXPERIMENTAL_LLVM_CODEGEN=1 ./bin/test_procedural cuda  # LLVM version

# Check pixel counts
python3 -c "
from PIL import Image
ast = Image.open('test_procedural_ast.png')
llvm = Image.open('test_procedural.png')
ast_px = list(ast.getdata())
llvm_px = list(llvm.getdata())
ast_non_black = sum(1 for p in ast_px if p[0] > 10 or p[1] > 10 or p[2] > 10)
llvm_non_black = sum(1 for p in llvm_px if p[0] > 10 or p[1] > 10 or p[2] > 10)
print(f'AST: {ast_non_black}, LLVM: {llvm_non_black}')
"
```

## What Needs to Be Done

1. **Fix Handler Cloning**
   - Rewrite `_clone_and_lower_handler()` to properly clone functions
   - Ensure all value references are remapped
   - Consider using `llvm::CloneFunction()` utility
   - Verify no "argument in another function" errors

2. **Verify Dead Code Elimination**
   - Check that LLVM actually eliminates unreachable branches
   - May need to run specific optimization passes
   - Verify intersection program has no surface candidate code

3. **Test Mixed Handlers**
   - Handlers with both surface and procedural candidates
   - Should produce correct results matching AST codegen

4. **Performance**
   - Current implementation is slow (handler cloning)
   - May need optimization

## Lessons Learned

1. **Don't manually clone LLVM functions** - Use `llvm::CloneFunction()`
2. **ValueToValueMap needs ALL mappings** - Arguments, blocks, instructions
3. **RemapInstruction must be called after ALL mappings are set**
4. **Test constantly** - Run pixel comparison after every change
5. **Debug with `debug_after_*.ll` files** - Check generated IR

## References

- `RAY_QUERY.md` (this file) - Design document
- `test_procedural.cpp` - Test case with mixed hit types
- AST codegen in `cuda_codegen_xir.cpp` - Reference implementation
- OptiX Programming Guide - Entry point semantics

## Handoff Notes for Gemini

The architecture is correct but implementation has bugs in function cloning. The key insight is that we need TWO versions of each handler with different state constants, and LLVM should fold the switch statements.

Focus on fixing `_clone_and_lower_handler()` - it's the root cause of all current issues. Once cloning works correctly, everything else should fall into place.

Good luck.
