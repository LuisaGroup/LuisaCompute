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

The binary `atan2` provider in `llvm_native_math_atan2.cpp` separately adapts
SLEEF's `xatan2f`/`atan2kf` signed quadrant reduction and polynomial to generic
fixed-vector LLVM IR. Its operand sanitization and explicit NaN, infinity, and
signed-zero repair are local to this provider.

The direct `exp2`, `exp10`, `log2`, and `log10` bodies were separately
adapted and audited from SLEEF's `xexp2f`, `xexp10f`, `xlog2f_u35`, and
`xlog10f` formulas. Their reductions and destination-base polynomials remain
independent provider bodies; they do not scale an emitted `exp` or `log`
result. The exponential bodies reuse the already audited precise exponential
polynomial after their own split range reductions.

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
- `atan2`: divide the smaller operand magnitude by the larger, evaluate a
  locally Remez-derived degree-11 odd minimax polynomial on `[-1, 1]`, and
  reconstruct the quadrant from operand signs. None of its six coefficients
  were copied from SLEEF or ISPC. A dense double audit measured maximum
  approximation error `1.663e-6`; simulated f32 Horner evaluation measured
  `1.756e-6` before quadrant and special-value repair.
- `exp`: reduce by the nearest integer multiple of `ln(2)` and evaluate the
  degree-4 Maclaurin polynomial with exact rational coefficients `1/2`,
  `1/6`, and `1/24`. Exponent construction uses f32 bit operations.
- `exp2`: round the input directly to the nearest integer exponent and apply
  the same degree-4 polynomial to the residual multiplied by `ln(2)`.
- `exp10`: round `x * log2(10)` to the nearest integer exponent, reduce in
  base ten, and apply the degree-4 polynomial after conversion by `ln(10)`.
- `log`: normalize the f32 mantissa around one, form
  `z = (m - 1) / (m + 1)`, and evaluate
  `2 * (z + z^3/3 + z^5/5)` before adding the exponent contribution.
- `log2` and `log10`: reuse that normalized mantissa series but combine the
  mantissa and extracted exponent directly in the destination base, without
  first rounding a base-e logarithm result.

The fast implementation resides in:

- `llvm_native_math_fast_trig.cpp`;
- `llvm_native_math_fast_inverse_trig.cpp`;
- `llvm_native_math_fast_exp_log.cpp`;
- `llvm_native_math_atan2.cpp`.

These bodies set only LLVM's contraction permission. NaN, infinity, signed
zero, subnormal, and domain repair are explicit IR operations. The exact
behavior and numerical envelopes are normative in
[`../simd/SIMD_NATIVE_EXECUTION_CONTRACT.md`](../simd/SIMD_NATIVE_EXECUTION_CONTRACT.md).

## Validation record

`test_simd_llvm_native_math` builds precise and fast providers together for
W2/W3/W4/W8/W16. For each operation and width it checks fixed special values
and boundaries, 8,192 deterministic raw f32 bit patterns, 8,192 values focused
on the mathematical domain, and 4,096 values focused on range-reduction and
overflow transitions. The four independent `exp2`/`exp10`/`log2`/`log10`
providers raise those counts to 65,536, 65,536, and 16,384 respectively and
sample both precise and fast reduction partitions. Binary `atan2` uses paired
corpora of the same expanded sizes, including a permanent two-ULP precise
counterexample, dense ratios, all quadrants, axes, infinities, and varied
magnitudes. The test also checks canonical special-value bits, generic IR
shape, inactive tails, scalar
uniformity, and optimized assembly symbols. Schedule execution gives the four
independent results distinct weights so a swapped operation-to-provider
mapping cannot disappear inside a commutative sum.
`test_fallback_llvm_native_math` independently checks that fallback
float2/float3/float4 lowering selects the requested tier.

`benchmark_llvm_native_math` measures 4,096 packets per call, grows the repeat
count until the precise sample lasts at least 20 ms, interleaves precise and
fast order, and reports the median of nine samples. It covers fallback
float2/float3/float4 and SIMD W4/W8/W16, reports nanoseconds per element and
static entry instruction counts, rejects scalar libm symbols, and requires at
least 1.05x aggregate throughput improvement at every width.

On the 2026-08-11 audit host (AMD Ryzen 9 9950X3D, x86-64, LLVM 22.1.8,
Release build), three consecutive runs of the independent-provider checkpoint
kept all thirteen individual operations faster in every width. Median
aggregate speedups were 1.355x (W2), 1.352x (W3), 1.362x (W4), 1.355x (W8),
and 1.317x (W16); the slowest individual result across the three runs was
1.089x. The `atan2` medians ranged from 1.297x to 1.346x. Every
reported row had `scalar_libm=no`. These numbers are a reproducibility record,
not a cross-machine performance guarantee.
