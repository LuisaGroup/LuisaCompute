# Embree 4 Out-of-Tree Static Triangle BLAS Archive Architecture

## 1. Zero Out-of-Tree Patch Requirement

Static triangle BLAS archiving for LuisaCompute's `fallback` and `simd` backends runs **100% out-of-tree inside LuisaCompute** against unmodified Embree 4 binaries on x86-64 and AArch64. No Embree source forks, custom DLLs, `LUISA_EXTRA_EMBREE_BIN` overrides, or CMake dependency changes are required.

When initialized (`EmbreeAccelArchiveExt`), Luisa queries public `RTCDevice` properties (`RTC_DEVICE_PROPERTY_VERSION` in `4.x`, 64-bit pointers, `RTC_DEVICE_PROPERTY_COMPACT_POLYS_ENABLED == 0`), discovers and cross-verifies exact Embree 4 runtime offsets across distinct compact and non-compact probe scenes guarded by OS readable-memory checks, runs a live `Triangle4i` vertex-slot swap test, and executes a full export -> dummy-commit import -> transformed TLAS `rtcIntersect1`/`rtcOccluded1` bit-exact parity self-test before marking archiving supported (`layout_info = "embree4_out_of_tree_bvh4_bvh8_static_tri_v1"`).

If `LUISA_EMBREE_ACCEL_ARCHIVE=0` is set, or if any ABI check or live ray self-test fails on a future/custom Embree binary, initialization fails closed (`supported = false`, `AccelArchiveStatus::UNSUPPORTED_FEATURE`), allowing callers to fall back cleanly to standard Embree BVH rebuild.

## 2. Supported Scope and Layout

Only static single-mesh triangle BLAS (`RTCScene` with one static, non-motion-blur triangle geometry at `geomID == 0`, stride `>= 12` and `% 4 == 0`, `!allow_update`) using Embree aligned-AABB nodes (`tyAABBNode`) and 4-wide triangle leaf blocks is supported:

| Acceleration structure | Leaf block type | Usage in Luisa (`ROBUST` enabled) | Payload element sizes (x64) |
|---|---|---|---|
| `BVH4` | `Triangle4` (`FAST`) / `Triangle4v` (`ROBUST`) / `Triangle4i` | Compact BLAS (`allow_compaction = true`) selects `BVH4<Triangle4i>` | `BVH4::AABBNode` = 128 B, `Triangle4i` = 80 B |
| `BVH8` | `Triangle4` (`FAST`) / `Triangle4v` (`ROBUST`) / `Triangle4i` | Non-compact BLAS on AVX CPUs (`allow_compaction = false`) selects `BVH8<Triangle4v>` | `BVH8::AABBNode` = 256 B, `Triangle4v` = 176 B, `Triangle4` = 176 B |

Dynamic scenes (`allow_update = true`), motion blur, quantized/OBB/motion nodes, barriers, and non-triangle primitives are rejected cleanly.

AArch64 uses NEON. Embree calls its 4-wide NEON intersectors `sse2::` and its 2×NEON intersectors `avx2::`; these names do not mean ARM needs x86 CPU features. A compatible build can use BVH4 or BVH8, but each layout must pass the same live self-test before use. ARM64 and x86-64 archives are intentionally not interchangeable.

## 3. Binary Encoding, External Slab Materialization, and Lifetime Safety

No `RTCScene`, `FastAllocator` metadata, vtables, or function pointers are serialized. Instead:

1. **Single-pass BVH extraction and parallel payload emission (`src/backends/common/embree_accel_archive.cpp`)**:
   - Export traverses reachable `BVHN<N>::AABBNode` internal nodes and leaf primitive blocks from `bvh->root`, assigning contiguous dense indices and emitting a self-contained 136-byte `EBVH` payload header + node table + leaf table in parallel across worker threads.
   - `NodeRef` slots (`emptyNode = 8`, checked before `tyLeaf = 8`, `maxLeafBlocks = 7`) are converted to portable relocatable node/leaf descriptors.
   - Payload integrity uses Luisa's parallel CPU/GPU-identical `FastArchiveChecksum128` (`src/runtime/rtx/accel_archive.cpp`), preserving strong `luisa::hash128` for headers, geometry keys, and compatibility metadata.

2. **Pre-allocation validation and stock-Embree dummy commit on import**:
   - Outer envelope headers, `CompatBlob` (`embree_version_number`, `build_config_flags`, `required_isa_mask`), and leaf blocks (`geomID == 0`, `primID < triangle_count`, aligned `Triangle4i` vertex offsets) are validated in parallel before allocating or mutating any scene state.
   - `FallbackMesh::import_from_archive` and `SIMDMesh::import_from_archive` bind a tiny static 1-triangle dummy mesh and invoke stock `rtcCommitScene`, letting Embree allocate the genuine native `BVH4`/`BVH8`, select native SIMD `Accel::Intersectors`, and initialize `scene->vertices[0]`.
   - `embree_import_static_triangle_bvh_oot` allocates one 64-byte-aligned external slab (`EmbreeImportedBVHHandle`), copies leaf blocks and reconstructs relocated 64-bit `NodeRef` pointers in parallel, and swaps `bvh->root`, `bvh->bounds`, `scene->bounds`, and `scene->vertices[0]` (for `Triangle4i`) onto the imported BVH and rebound caller vertex buffer.

3. **Deterministic cleanup (`EmbreeImportedBVHHandle`)**:
   - Original Embree dummy pointers and bounds are saved inside `EmbreeImportedBVHHandle` and restored before rebuilding, re-importing, freeing the external slab, or calling `rtcReleaseScene` in `~FallbackMesh()` / `~SIMDMesh()`, so stock Embree never attempts to free Luisa's external slab or rebound user buffers.
