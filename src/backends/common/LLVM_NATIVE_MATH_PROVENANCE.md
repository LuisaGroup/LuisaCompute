# LLVM fixed-vector native-math provenance

This file records the source and validation boundary of
`llvm_native_math*.cpp`. The implementation emits target-independent LLVM
fixed-vector IR. It does not link a vector math library and does not contain
x86, Arm, or other target-specific intrinsics.

## Precise tier

The precise f32 algorithms and the 416-word Payne-Hanek reduction table are
adapted from SLEEF's single-precision SIMD math implementation, copyright
Naoki Shibata and contributors 2010-2025. SLEEF is distributed under the
Boost Software License 1.0; the retained license is
[`LICENSE.SLEEF.txt`](LICENSE.SLEEF.txt), and the upstream project is
<https://github.com/shibatch/sleef>.

The adaptation replaces SLEEF's vector abstraction with generic LLVM
fixed-vector arithmetic, comparison, selection, bit operations, and masked
table gathers. The precise expression order was preserved when the original
provider was split into:

- `llvm_native_math_range_reduction.cpp`;
- `llvm_native_math_precise_trig.cpp`;
- `llvm_native_math_precise_inverse_trig.cpp`;
- `llvm_native_math_precise_exp_log.cpp`.

Precise code deliberately does not allow FP contraction where its compensated
arithmetic depends on separate multiply and add rounding.

## Fast tier

The fast control flow, range partitions, transforms, and rational-series
formulas were derived and audited locally. No ISPC source or coefficients
were copied. ISPC's standard library was consulted only as evidence that a
separate vector fast-math tier is a viable design; it is not a source
dependency or provenance ancestor of these files.

The fast formulas are:

- `sin`/`cos`/`tan`: for `abs(x) < 128`, round to the nearest quadrant and
  subtract a three-term f32 split of pi or pi/2. The coarse leading term makes
  its product exact over this range; two residual terms retain accuracy near
  a tangent pole. A packet containing any non-common lane branches to the
  precise SLEEF-derived reduction body (which uses Payne-Hanek for its large
  range), then selects the precise result only for those lanes. `sin` and
  `cos` reuse the already licensed and audited degree-9 precise polynomial.
  `tan` reuses the lower five precise coefficients and omits its highest-order
  term; the resulting degree-11 polynomial has its own fast-tier error audit.
- `asin`/`acos`: use `asin(x) = pi/2 - 2 asin(sqrt((1-x)/2))` outside the
  half interval and the exact Maclaurin coefficients `1/6`, `3/40`, and
  `5/112` inside it.
- `atan`: use the `tan(pi/8)` and `tan(3pi/8)` partitions with reciprocal and
  pi/4 transforms. The reduced interval uses the alternating series through
  `z^9`, whose coefficients are the exact rationals `-1/3`, `1/5`, `-1/7`,
  and `1/9`.
- `exp`: reduce by the nearest integer multiple of `ln(2)` and evaluate the
  degree-4 Maclaurin polynomial with exact rational coefficients `1/2`,
  `1/6`, and `1/24`. Exponent construction uses f32 bit operations.
- `log`: normalize the f32 mantissa around one, form
  `z = (m - 1) / (m + 1)`, and evaluate
  `2 * (z + z^3/3 + z^5/5)` before adding the exponent contribution.

The fast implementation resides in:

- `llvm_native_math_fast_trig.cpp`;
- `llvm_native_math_fast_inverse_trig.cpp`;
- `llvm_native_math_fast_exp_log.cpp`.

These bodies set only LLVM's contraction permission. NaN, infinity, signed
zero, subnormal, and domain repair are explicit IR operations. The exact
behavior and numerical envelopes are normative in
[`../simd/SIMD_NATIVE_EXECUTION_CONTRACT.md`](../simd/SIMD_NATIVE_EXECUTION_CONTRACT.md).

## Validation record

`test_simd_llvm_native_math` builds precise and fast providers together for
W2/W3/W4/W8/W16. For each operation and width it checks fixed special values
and boundaries, 8,192 deterministic raw f32 bit patterns, 8,192 values focused
on the mathematical domain, and 4,096 values focused on range-reduction and
overflow transitions. It also checks canonical special-value bits, generic IR
shape, inactive tails, scalar uniformity, and optimized assembly symbols.
`test_fallback_llvm_native_math` independently checks that fallback
float2/float3/float4 lowering selects the requested tier.

`benchmark_llvm_native_math` measures 4,096 packets per call, grows the repeat
count until the precise sample lasts at least 20 ms, interleaves precise and
fast order, and reports the median of nine samples. It covers fallback
float2/float3/float4 and SIMD W4/W8/W16, reports nanoseconds per element and
static entry instruction counts, rejects scalar libm symbols, and requires at
least 1.05x aggregate throughput improvement at every width.

On the 2026-08-11 audit host (AMD Ryzen 9 9950X3D, x86-64, LLVM 22.1.8,
Release build), all twelve individual operations were faster in every width.
Aggregate speedups were 1.389x (W2), 1.376x (W3), 1.396x (W4), 1.370x (W8),
and 1.383x (W16); the slowest individual result was 1.113x. These numbers are
a reproducibility record, not a cross-machine performance guarantee.
