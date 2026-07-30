---
name: llvm-spirv
description: Experimental Vulkan AST-to-LLVM-to-SPIR-V backend, its fail-closed runtime-interface boundary, LLVM build integration, and validation path.
---

# LLVM SPIR-V Codegen

The experimental backend lives in
`src/backends/common/spirv_llvm/`. It lowers a Luisa AST `Function` to LLVM IR
and asks LLVM's native SPIR-V target to emit a
`spirv64-unknown-vulkan1.2` module.

This is not the native XIR-to-SPIR-V path. The two Vulkan codegen selections
are mutually exclusive:

- CMake: `LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV=ON` and
  `LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV=OFF`;
- XMake: `lc_vk_backend_use_ast_llvm_spirv=true`, with `lc_llvm_path`
  pointing to an LLVM installation/build prefix or its `llvm-config`.

The public entry is deliberately LLVM-header-free:

```cpp
#include <spirv_llvm/spirv_llvm.h>

auto result = lc::llvm_codegen::compile_spirv(kernel, option);
```

Do not expose an LLVM type through `spirv_llvm.h` or
`llvm_codegen_result.h`; consumers of the static facade must not inherit LLVM
include directories or unrelated LLVM preprocessor definitions.

## Current support boundary

The Vulkan descriptor interface is preflighted before AST lowering by
`validate_llvm_vulkan_resource_model`. At present it rejects:

- every kernel argument, including ordinary value arguments and direct
  buffers;
- textures, bindless arrays, acceleration structures, custom/indirect
  resources, and their global heaps;
- shader printing.

Consequently the current end-to-end Vulkan contract is intentionally narrow:
no-argument, no-print compute kernels whose operations are implemented by the
visitor. The successful interface contains only the fixed 16-entry immutable
sampler property. Do not describe the visitor's partial resource stubs as
runtime support, and do not bypass this preflight merely because LLVM happens
to emit some SPIR-V for an address-space global.

Unsupported AST operations call `LUISA_NOT_IMPLEMENTED` or otherwise fail
closed. Returning zero, `undef`, or a no-op is not an acceptable way to claim
support. When expanding the backend, implement the LLVM IR lowering, extend
the property preflight, validate the Vulkan runtime binding shape, and add an
end-to-end SPIR-V validator test together.

## Source map

| Files | Responsibility |
|---|---|
| `spirv_llvm.h/.cpp` | LLVM-free facade and once-only SPIR-V target registration |
| `llvm_codegen_result.h` | SPIR-V words, properties, printer records, and global bindless flags returned to Vulkan |
| `llvm_codegen_stack_data.h/.cpp` | Per-compilation state and mutex-protected reusable state pool |
| `llvm_codegen_utility.h/.cpp` | Type/constant/function lowering, module legalization, target-machine emission, post-processing, and validation |
| `llvm_state_visitor.h/.cpp` | AST expression and statement lowering through `llvm::IRBuilder<>` |
| `llvm_compat.h` | API-detection boundary for LLVM-version differences such as nullable terminator lookup |
| `vulkan_binding_properties.h` | Pure property planner and fail-closed resource-model support check |
| `CMakeLists.txt` / `xmake.lua` | Component-aware LLVM discovery and static-facade linkage |

## Compilation pipeline

`LLVMCodegenUtility::CompileSPIRV` owns the complete path:

1. Create a fresh LLVM context/module/builder and register LLVM's SPIR-V target
   exactly once with `std::call_once`.
2. Initialize a `spirv64-unknown-vulkan1.2` target machine and its data layout.
3. Detect prospective bindless/property usage from the AST, validate the
   current Vulkan resource-model boundary, and freeze the property plan before
   visiting the AST.
4. Lower the kernel and reachable callables to LLVM IR. A compute entry is
   named `main`, has no function parameters, and carries `hlsl.shader=compute`
   plus `hlsl.numthreads` attributes.
5. Before target emission, recursively scalarize aggregate loads/stores,
   lower aggregate returns to `void` plus an out pointer, scalarize again, and
   verify the LLVM module.
6. Emit object bytes through LLVM's legacy pass manager. The expected output
   is raw SPIR-V words; an ELF result is not silently accepted as valid SPIR-V
   and will fail the final validator (there is no ELF section extractor yet).
7. Strip LLVM's `Addresses`/`Linkage` capabilities and linkage decorations,
   and convert `OpPtrAccessChain` forms to their logical-addressing
   counterparts.
8. Validate the returned module with SPIRV-Tools under
   `SPV_ENV_VULKAN_1_2`.

Unlike the native path, this backend currently has no SPIRV-Tools optimizer
stage and does not produce the native exact per-argument role plan. Vulkan
serializes it as `LLVM_SPIRV`, embeds constants directly in the module, and
uses the backend's conservative SPIR-V artifact feature requirements.

## Function and control-flow rules

- Vulkan entry points have no LLVM function parameters. The current code has
  provisional global-variable lowering for arguments, but the property
  preflight rejects those kernels until a real descriptor ABI exists.
- Callable arguments remain LLVM function parameters.
- Save and restore the builder insertion point, current function, and variable
  map around recursive callable generation.
- Probe incomplete blocks through `llvm_compat.h::terminator_or_null`; do not
  guess the API from an LLVM major version.
- Verify the full module before invoking LLVM target passes. Per-function
  verification warnings are useful during construction but do not replace the
  module check.
- Aggregate legalization is mandatory because the LLVM SPIR-V target cannot
  reliably legalize the aggregate memory/return forms produced here.

## Target initialization

`InitializeLLVMSPIRVTarget` registers global LLVM state in dependency order:

```cpp
LLVMInitializeSPIRVTargetInfo();
LLVMInitializeSPIRVTarget();
LLVMInitializeSPIRVTargetMC();
LLVMInitializeSPIRVAsmPrinter();
```

Use the declarations from `<llvm/Support/TargetSelect.h>`. Hand-written
declarations can acquire the wrong language linkage. Keep the `std::call_once`
guard because shader compilation may be concurrent.

The target triple is deliberately `spirv64`: the supported LLVM revision's
`spirv32` path fails in pointer-cast legalization. Vulkan logical addressing is
restored by the checked post-processing step and then enforced by validation;
raw `EmitSPIRV()` output is not the public contract.

## CMake integration

The target is `luisa-compute-spirv-llvm`. It is created only when Vulkan and
`LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV` are enabled. The Vulkan plugin also
links `luisa-compute-spirv`, because the shared artifact codec still needs the
native/common SPIR-V validation and feature-reconciliation utilities.

`src/backends/common/spirv_llvm/CMakeLists.txt` must:

- `find_package(LLVM CONFIG REQUIRED)` without allowing `LLVMConfig.cmake` to
  overwrite the project's `CMAKE_MSVC_RUNTIME_LIBRARY` default;
- require an exact `SPIRV` token in `LLVM_TARGETS_TO_BUILD` and locate
  `llvm/IR/IntrinsicsSPIRV.h` in LLVM's reported include directories;
- keep LLVM includes and ordinary definitions private;
- compile-check `_GLIBCXX_USE_CXX11_ABI` and propagate only that ABI macro
  through the C++ facade when LLVM reports it;
- request `core`, `support`, `bitwriter`, `transformutils`, `analysis`,
  `codegen`, `target`, `mc`, `spirvcodegen`, `spirvdesc`, `spirvinfo`, and
  `spirvanalysis` through LLVM's component mapper;
- use `llvm_config(... USE_SHARED ...)` when a monolithic `LLVM` target exists
  so target-specific archives omitted by the dylib remain linked;
- reject incompatible Windows CRT families or Debug iterator modes, and stage
  and install a monolithic LLVM DLL when that is the selected import.

The facade is static, so implementation link dependencies must still reach the
final Vulkan plugin/test link. Do not make LLVM headers public as a workaround
for a link failure.

## XMake integration

The target is `lc-spirv-llvm`. Configuration requires an explicit
`lc_llvm_path`. Keep `lc_vk_backend_use_xir_spirv=false` when selecting it:
the top-level configuration rejects an explicit conflict, while the option
hook forces `lc_enable_xir` on and normalizes the native option off. The
SPIR-V targets and tests are created only when the Vulkan backend itself is
enabled.

Treat the selected `llvm-config` as the source of truth:

- accept an executable path, install prefix, build prefix, or common
  source-tree build layout;
- parse `--quote-paths` output with `os.argv`, including all `-I` paths from
  `--cppflags` so generated intrinsics headers in development trees are found;
- require an exact `SPIRV` token from `--targets-built`;
- query `--shared-mode` for the complete component set before selecting
  `--link-shared` or `--link-static`;
- use static LLVM components on Windows because `llvm-config` reports DLL
  filenames rather than MSVC import libraries for shared mode;
- propagate shared-library rpaths on Unix and the required component system
  libraries on every platform;
- verify the libstdc++ ABI and, on Windows, the CRT family and Debug/non-Debug
  mode before compiling the facade.

## Vulkan artifact boundary

The LLVM result reuses `hlsl::Property` and the common Vulkan artifact codec,
but it is a distinct `LLVM_SPIRV` dialect:

- do not apply native XIR capability reconciliation or exact accel-role rules
  to LLVM artifacts;
- constants are embedded in SPIR-V, so there is no constant-UBO payload;
- saved arguments use the legacy/unspecified resource-role sentinel;
- loaded modules are still integrity-checked and Vulkan-validated before
  pipeline creation.

## Tests

`test_spirv_llvm_facade` is registered only when the LLVM facade target exists.
It compiles a no-argument kernel through the public header, checks the fixed
sampler property, and independently validates/disassembles the returned module
for Vulkan 1.2. `test_vk_shader_binary_contract` separately covers the common
artifact boundary under the `LUISA_AST_LLVM_TO_SPIRV` dialect define.

When adding support, include at least:

1. a pure property-preflight rejection/acceptance test;
2. a public-facade compile test;
3. independent `SPV_ENV_VULKAN_1_2` validation;
4. a Vulkan artifact round trip when properties or feature contracts change;
5. a runtime Vulkan test before claiming descriptor or dispatch support.

## Diagnostics and pitfalls

- `EmitSPIRV()` writes `llvm_ir_debug.ll` in the process working directory.
- Missing SPIR-V intrinsics or target components are configuration errors, not
  reasons to guess library filenames or add every LLVM archive.
- `getDeclarationIfExists()` may return null. Handle that at the operation's
  semantic boundary; never call through a null intrinsic declaration.
- Do not call `EmitSPIRV()` directly from Vulkan. Only the public facade runs
  required post-processing and Vulkan validation.
- Do not report an AST opcode as supported merely because a visitor case
  exists; it must survive resource preflight, LLVM verification, SPIR-V
  emission, Vulkan validation, artifact loading, and runtime dispatch.
