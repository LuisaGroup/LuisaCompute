# Logical continuation selection and opt-in resume batching

## Contract and cause

A before-resume suffix belongs to its target continuation for scheduling, but
its typed binding descriptor belongs to its original suspend boundary. These
are distinct identities. Comparing physical entry queues independently can
select a seven-path rival over a logical continuation with two four-path
entries. Sorting and resuming those entries independently also cannot supply
one jointly sorted eight-path launch.

The first repair sums queue populations by logical continuation. Ordinary
semantic Extensions retain independent owners. The winner consumes all its
physical members from one immutable membership snapshot. Incremental mode
must gather that snapshot before any member resumes: a self-edge can enqueue
new members which belong only to the next scheduling decision. Auxiliary
producer admission uses the aggregate population, while counter publication
still consumes each original source queue exactly once.

The second repair permits actual queue union only through a nonempty matching
`WavefrontCoroSchedulerExtensionHandler::batching_identity()`. The default
empty identity preserves independent Handler invocations. An opt-in identity
certifies that one representative instance, resources and descriptor can
process the disjoint union; it cannot depend on predecessor identity or the
number of separate invocations. Identity alone is insufficient. The complete
ordered before-resume chains must also match in target, normalized metadata,
read-only typed physical projections, and resident/reconstruction certificates.
Different field/access-chain/bit projections do not merge. Guarded bindings
with no exposed flat projection fail closed. Independent semantic prefixes
remain intact; only source routes and prefix-to-suffix routes are redirected.
All prepared Handlers remain alive until scheduler destruction.

No application schema, material identifier, sorting algorithm, profiling data,
or renderer-specific policy is encoded in the scheduler.

## Permanent witnesses

- `test_coro_wavefront_resume_annotation`: fifteen runtime routes first create
  populations `target=4, producer=8, rival=3`; the producer then creates
  `target=(4+4), rival=7`. Plain and annotated controls must select the same
  logical winner. Mixed bare/annotated entries, self-edge snapshot epochs,
  distinct physical bindings, AoS/SoA, publication modes and compaction are
  covered without assuming order among physical members of one batch.
- `test_coro_wavefront_auxiliary_admission`: one old auxiliary item plus a
  split four-path producer population requires draining before either member;
  drain observations are `{1, 4, 2}`.
- `test_coro_wavefront_resume_batching`: two predecessors supply interleaved
  even/odd keys. A tiny test-only GPU rank Handler and the target's physical
  lane outputs prove one complete eight-path permutation and one resume.
  Default/empty/distinct identities do not union those entries. Real compiled
  projections provide host metadata controls for attributes, physical fields,
  ordered suffixes and resident/reconstruction sets. Semantic prefixes and
  self-edges conserve state and membership. The runtime matrix covers AoS/SoA,
  full gather, separate incremental publication with compaction, and fused
  publication. There is no CPU coroutine or reference renderer.

The new target has a host-only metadata CTest and enabled HIP/fallback/Vulkan
runtime CTests. Native Vulkan registrations require XIR-to-SPIR-V and disable
DXC explicitly.

## Frozen baseline and validation scope

Evidence: `/var/tmp/psycles-holdout-CXlJR7/barber-sort-audit.md` and adjacent
`barber-queue-*` / `barber-batching-*` logs. Source base is exactly
`6e58928d84604bf0d976f0d6b00f44006b21ceb5`, in isolated worktree
`/var/tmp/luisa-resume-queue-coalescing-HuxMUl`. Builds used all 32 threads.
Standalone test translation units used the candidate headers and existing
`/home/mike/Projects/Psycles-surface-svm/build/bin` libraries. This is not a
fresh full SDK build, not validation of newer upstream `next` changes, and not
a renderer performance measurement. Later integration needs its own complete
SDK/backend and whole-renderer validation.

| Check | Result |
|---|---|
| Original logical-cardinality HIP witness | Red: 28 expected rank failures; binding control green |
| Pre-batching joint permutation, HIP | Red: 20 expected batch/launch/permutation failures |
| Pre-batching self-edge, HIP | Red: 2 expected batch/launch failures |
| Pre-batching metadata / identity / prefix controls | Green: 18 / 66 / 19 assertions |
| Candidate complete three-binary HIP run | Green: 14 tests, 6,977 assertions; no skips |
| Candidate complete three-binary fallback run | Green: 14 tests, 6,977 assertions; no skips |
| Candidate complete three-binary native Vulkan run | Green: 14 tests, 6,977 assertions; no skips |

All candidate runtime checks enable `LUISA_CORO_WAVEFRONT_VERIFY_QUEUES=1`.
The old red binaries and matching source are retained separately. HIP logs
show compatible entry/self queues aliasing the representative and one jointly
sorted target launch; default handlers and semantic prefix writes remain
independent. Loader-traced native Vulkan runs additionally set
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`. The three complete logs contain 293 successful
SPIR-V compilations (50 batching, 189 priority, 54 auxiliary) and no DXC/DXIL
library loads or references. Fallback and Vulkan loader traces confirm the
frozen coroutine/compiler/backend library paths, not newer upstream binaries.

Key SHA-256 identities (full evidence also records every test and log hash):

```text
640d98dfb50fff7d2965fa62df3720048f29b0f782d479aef1ffd890ada5f166  preserved pre-batching executable
083ef2c10fa2c11ccb089029a8954e8923c1b037ecedc27fa5f7756fd67a67ff  preserved pre-batching source
26a91e41cde406e5458bb3c6fda432ca588da94667727eb420cf3238a2b73610  candidate wavefront.h
0d8e4c49add5ab00f6e91a49da64231cf6f0fb621b0372505353eabbd57fa51a  candidate compatibility predicate
77ac93cbb552943f008d2d517207a07d986d195086e388898a2dd545ac690048  candidate batching executable
50493a131f2d515432e921762f291e5188e2ff02c6202fb5be6a55359e4ac225  frozen libluisa-coro.so
60e91e7a0adf433040dbdd5a561ffd46bc84727092f832e13b2169cca1a8b5d6  frozen libluisa-xir.so
ca904d40c0af65b8940f3c85ed860c6839991103739b1418eef046bd13076693  frozen libluisa-dsl.so
359b4c50fb0993abb637d058f36d1cabd07a96e77ad1cb094dcb4c42acb62030  frozen libluisa-backend-fallback.so
6c6e69e0a5e5770608f63833e18e51fc0faaa3b956e5c62119541d3fdaac4bed  frozen libluisa-backend-vk.so
bbf767982507a1548bda4b01a7004d3fd3f84c90c7d59d8338c89859a578456d  complete batching fallback log
5886762c9f1b5aaa31d3bcb137946002822ac92a890dcae0bbb4a32e6b701d02  complete batching native Vulkan log
```
