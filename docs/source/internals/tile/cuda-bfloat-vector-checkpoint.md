# CUDA BF16 vectorization: shutdown checkpoint

This branch is WIP. Do not merge it into `next` before completing the checks below.

The published measurement report is commit `9d66ed62c` on `next`, section 16
of `cuda-workflow-b.md`, with all 70 samples. The first global-eager prototype
improved L4 by 23.35% against the same-lane old control, but only 2.21% against
the strongest old L1 control, and remained 3.07% slower than fresh Torch.
Its L1 regression of 1.75% is retained in the report.

The current prototype restores the original lazy BF16 lowering. It makes a
separate independent vector-phase copy, binds an already evaluated integer
word with existing TIRx Let, and uses an eager Select only after a restricted
integer-expression proof. Scalar pointer fallback and partial packs keep the
original expressions. Contribution staging is unchanged. L8 remains bounded
by the target's lack of Bool8 support.

This prototype passed syntax checking and a full MSVC LLVM 22 build through
`cmake --build`. Its final host tests, LLVM 23 build, runtime admission and
performance comparison have NOT been completed. The existing host tests still
contain expectations for the earlier global-eager lowering; update those
expectations while retaining the real L2/L4 wide-access requirements. Remove
the temporary `canonicalize_vector_integer_values_for_test` export and test
through the existing compilation entry points. Keep phase-memory counters
explicitly scoped to the vector copy, rather than claiming they describe the
original lazy fallback.

Local continuation packets (retained under the ignored `.deps` directory):

- `tirx-vector-only-eager-prep-v1`: implementation and unfinished host-test work.
- `tirx-bfloat-vectorcopy-admission-prep-v1`: reviewed 17-case diagnostic runner;
  it has not been executed. L1 must match the original source byte for byte.
- `tirx-bfloat-vectorcopy-pairs-prep-v1`: prepared old/new comparison, pending
  final implementation and admission. Reuses the built DLL-observer harness.
- `oct04-tirx-vector-only-eager-full-build-v1.log`: successful LLVM 22 build.
- `oct04-tirx-vector-only-eager-reduction-syntax-v2.log`: zero syntax errors.
- `oct04-tirx-bfloat-select-pairs-summary-v1/checkpoint.json`: completed audit
  of the previous experiment, with 10 processes, 900 graph nodes and all
  1,441,792 saved values checked against the original FP64 bounds.

After final edits, repeat the full LLVM 22/23 builds before testing. Keep the
original numerical bounds, independent alignment/tail cases and fresh Torch
comparison. All previous failed and successful experiment packets are kept.
The dirty TVM submodule contains already packaged patches and is deliberately
excluded from this checkpoint commit.
