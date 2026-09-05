# SSA formation after late HIP callable inlining

## Root cause and rewrite

The HIP post-IPO ABI pipeline can find a generated callable with exactly one
direct use outside its own body whose legalized arguments or return exceed AMDGPU's
32-VGPR callable window. Replacing that call with its body is a zero-duplication
CFG splice: the callee is deleted, so there is no remaining consumer with which
to share it. This avoids introducing a private ABI record solely for that
boundary. Ordinary multi-use calls keep LLVM's profitability decision; explicit
`noinline` and unsupported uses are not rewritten.

However, the splice runs **after** the ordinary optimization pipeline. Before
inlining, a caller-owned alloca passed by reference escapes to the retained
callee. After `InlineFunction` substitutes the actual pointer for the formal,
all accesses may be visible in the caller, but the utility does not run SROA.
Even scalar counters therefore remained as private loads and stores in the
final Luisa LLVM module. An address-space cast in those accesses was a symptom,
not the missing alias proof: ordinary LLVM SROA handles the resulting accesses.

The postcondition is now restored with one CFG-preserving SROA pass, restricted
to surviving functions changed by this late inlining fixed point. If a modified
caller is itself inlined and deleted, it is removed from the cleanup set. The
implementation uses LLVM's promotion legality checks; it does not add alias or
initialization assumptions and does not change the CFG, SVM dispatch, or the
DSL's default scalar/vector initialization semantics.

## Permanent reductions

`src/tests/unit/core/test_hip_late_inline.cpp`, linked into
`test_hip_callable_abi`, covers:

- Private scalar and vector storage passed through generic pointer formals,
  including a conditional update and preservation of the incoming RGB zero.
- Escaped and volatile storage, which must remain addressable.
- An unrelated function, which must not receive the new cleanup pass.
- A nested one-use chain, including deletion of an intermediate caller.
- The independent return-value limit, and shared, self-recursive or
  address-taken call boundaries.

Before cleanup, the two promotion tests failed seven structural assertions;
the escape/volatile controls already passed. Existing ABI tests also exercise
the exact 32-location boundary, oversized one-use arguments, explicit
`noinline`, and preservation of shared callable argument packing.

## Whole-program check, 2026-09-05

The original Psycles Lone Monk scene was recompiled with shader caching
disabled, then rendered on HIP at 1440x1080, 256 fixed spp, frame 4, seed 0.
This was not an extracted sibling shader. The full final surface LLVM module
changed from 136283 to 135612 lines and from 18 to 12 allocas. The eliminated
objects were the closure count/budget, shader flags, emission/extinction, and a
temporary local view of the coroutine frame. The dynamically addressed SVM
stack and closure pool remain.

The persistent coroutine frame did **not** change: 448 B AoS / 444 B actual SoA
per slot. This must not be confused with private memory objects in one kernel.
The surface kernel still used 256 VGPRs and 5072 B private scratch after HIPRTC.
One profiled run took 19.5275 s, with 11.581827 s in surface shading, versus the
preceding 20.0004 s / 12.021888 s run. These unpaired observations do not
establish a speedup. Combined relative RMSE against Cycles HIP was 1.113156%,
and all eight compared passes contained zero nonfinite pixels.

Validation completed with 32-way builds:

- ABI regression: 39 tests / 584 assertions, both the active build and the
  isolated `origin/next` candidate.
- Psycles: 148 HIP tests and 150 fallback tests passed.
- HIP runtime: callable boundaries (95 assertions), RT initialization/scratch
  reuse (356 assertions), and packed-pointer effects (1280 assertions) passed.
- Five native Vulkan canaries passed: sampled-sun PDF, native NEE setup and
  evaluation, wireframe, and bump state. The invocation set
  `LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
  `LUISA_VULKAN_DISABLE_DXC=1`; the verbose log records SPIR-V generation.

The active workspace still contains unrelated coroutine and renderer changes.
Only the late-callable ABI transform, its SROA postcondition and regressions
are part of this candidate; no parent-project submodule update is included.

Local evidence directories:

- `/var/tmp/psycles-surface-kernel-audit-SDPenr`: original LLVM module,
  minimal red/green regression logs, builds and backend suites.
- `/var/tmp/psycles-surface-late-sroa-aTH1UV`: forced full-module recompilation,
  LLVM dumps, profiler database, rendered image and Cycles comparison.
- `/var/tmp/psycles-hip-late-inline-clean-DOQtne`: isolated candidate based on
  `origin/next`, with a clean-source ABI test build. This host test links the
  existing core library and dependencies; it is not a claim of a clean build
  of every Luisa backend.
