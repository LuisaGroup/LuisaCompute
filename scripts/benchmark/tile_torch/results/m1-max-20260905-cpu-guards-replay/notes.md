# Ragged CPU input views: fixed-geometry, frozen-binary A/B

The two ragged GEMMs improve in **all four pairs**, with paired median
speedups of **1.279× and 1.818×** over the preceding lowering. This measures
the combined Boolean-proof repair and full-vector guard specialization;
both binaries request input forwarding. It is **not a speedup over Torch**.

| M×N×K | Previous µs | Candidate µs | Paired speedup median [range] | Candidate slower |
|---|---:|---:|---:|---:|
| 32³ | 5.411 | 5.168 | 1.033 [0.966, 1.205] | 2/4 |
| 128³ | 16.461 | 15.209 | 0.974 [0.900, 1.150] | 2/4 |
| 512³ | 689.156 | 709.835 | 0.980 [0.928, 1.201] | 2/4 |
| 1024³ | 5973.946 | 5938.617 | 1.034 [0.899, 1.142] | 1/4 |
| 256×1024×128 | 215.764 | 231.670 | 0.932 [0.714, 1.311] | 3/4 |
| 1024×128×256 | 168.559 | 187.285 | 0.911 [0.829, 1.185] | 3/4 |
| 127×193×61 | 42.471 | 33.433 | 1.279 [1.160, 1.427] | 0/4 |
| 513×257×129 | 516.224 | 282.626 | 1.818 [1.354, 2.601] | 0/4 |

Times are medians of per-round p50s; ratios are paired round medians, not
quotients of those displayed times. Ranges are observations, not confidence
intervals. No slow or failed round is omitted.

## Actual code changes and retained no-op controls

For all six regular shapes, every paired LLVM comparison is identical apart
from quoted allocation-identity TBAA strings. Raw archived files are not
rewritten. Their timing increases/decreases are **not evidence of changed
instructions** or of this transformation's benefit. In particular, the tall
case's 0.911× paired ratio is retained, not hidden by selecting its best run.

For every ragged pair the previous lowering has 256-float A and 1024-float B
staging arrays; neither remains in the candidate. The candidate retains
vector FMA calls (four static sites for the smaller case, six for the larger),
with the original guarded path for non-full packs. Site counts are not
dynamic instruction counts. Example raw sources:

- 127×193×61: [previous](sources/bd19c0870952446f1c25036adbd36818e90a41c9497b0610e5202e84251b37bc.ll),
  [candidate](sources/fe5c45e0d8855edb120a67d29c3204bd8a5d7c6b1f4f302d1308cd9eb2f481ea.ll).
- 513×257×129: [previous](sources/5111db31006e47af9e17772008ea3c24200c164eb3c658d1379ccfeb4088f719.ll),
  [candidate](sources/d85549788f529af22ab724de1c7b3b9db1974c739b96b07c62e6731aa3736dbb.ll).

The intermediate [failed experiment](../m1-max-20260904-cpu-guard-plan/notes.md)
forwarded these arrays but scalarized all input loads/FMAs. It motivates the
full-vector repair; its single-pass timing is not part of this paired result.
No generated LLVM is patched and no BLAS call is substituted into Tile code.

## Controls and verification

- A frozen pre-repair executable plus adjacent Luisa libraries is compared
  with the rebuilt candidate, against the same native TVM libraries. The two
  frozen plans are identical: 4×16×32 worker tiles, window 2, automatic CPU
  vectorization, 8192-byte stack budget, 64 logical pack lanes, and immutable
  input views. The pilot report supplies parameters only, never timing scores.
- Four rotating shape rounds balance A/B and Tile/Torch order. Seven samples
  target 30 ms each, after 200 ms warmup; eight CPU threads requested. Warm
  synchronized host-wall timing includes dispatch/internal scratch handling;
  JIT, allocations and transfers are separate. No concurrent build, test or
  profiler; ordinary OS/user activity is not controlled.
- **128/128 complete outputs valid**: 64 Tile and 64 Torch outputs, 30,043,136
  checked elements, maximum absolute error zero against the FP64 oracle on
  deterministic dyadic inputs. Non-dyadic inputs, changed inputs, nonzero
  accumulators, transposes, ordered K, alias/mutation rejection and both
  pipeline windows are additionally covered by C++ tests.
- Generic non-MMA vector fixtures cover negative origins, nonzero minima,
  scalar tails, interior mask holes, dynamic/zero-trip recurrences, and lazy
  branches that would overflow or divide by zero if improperly speculated.
  GEMM tests assert both actual staging removal and vector FMA emission.
- Both complete builds succeeded. Final selected CPU/Metal CTest cohorts are
  **23/25 in each**, not green: [patched](ctest-patched-final.log) and
  [original](ctest-original-final.log). Only the existing cooperative/memory
  Metal fence assertions fail (`mem_flags(3)` expected, separate worktree
  change emits 2). Those assertions and that separate change were not altered.
- All four changed C++ translation units pass the source checker. The Python
  benchmark-contract suite passes 54/54. All **40** recorded executable/library
  paths were stable and independently rehashed after timing; all **64** raw
  LLVM records were independently rehashed as well.
- Apple M1 Max, macOS 26.6.2, Torch 2.14.0. The run started September 5 local
  time (September 4 UTC). Exact binary identities, native commands, schedules,
  ordering, errors, and samples are in [results.json](results.json).

The remaining library gap must be measured separately. Immutable forwarding
and automatic packing remain opt-in; these results do not establish a
universal materialization policy or completion of the performance goal.
