# Embree 4.4.1 Static Triangle BLAS Archive Prototype (Opt-In)

## 1. Pinned Source Revision and Build Configuration

- **Pinned revision**: Embree `v4.4.1` (`f590db83ef6559387df7f6d8725c34fb7acf851d`)
- **Stock runtime behavior**: The bundled stock `embree4.dll` (`deps/embree-4.4.1.x64.windows/bin/embree4.dll`) is left unmodified. Both `fallback` and `simd` resolve the opt-in C entry points (`rtcExQueryStaticTriangleBVHArchiveConfig`, `rtcExExportStaticTriangleBVH`, `rtcExImportStaticTriangleBVH`) dynamically via `GetModuleHandleW(L"embree4.dll")` + `GetProcAddress`. When running against stock `embree4.dll` (or when `LUISA_EMBREE_ACCEL_ARCHIVE=0`), `AccelArchiveExt::capabilities().supported` is `false` (`layout_info = "embree4_optin_static_triangle_bvh_unavailable"`), and `validate_mesh_archive` / `export_mesh_archive` / `import_mesh_archive` return `AccelArchiveStatus::UNSUPPORTED_FEATURE` so callers fall back cleanly to normal BVH rebuild.
- **Matching CMake configuration** (aligned with `deps/embree-4.4.1.x64.windows/lib/cmake/embree-4.4.1/embree-config.cmake`):
  - `EMBREE_MAX_ISA=AVX512` (`SSE2`, `SSE42`, `AVX`, `AVX2`, `AVX512`)
  - `EMBREE_TASKING_SYSTEM=TBB` (linked against `oneapi-tbb-2023.1.0`)
  - `EMBREE_RAY_MASK=ON`
  - `EMBREE_FILTER_FUNCTION=ON`
  - `EMBREE_MIN_WIDTH=OFF`
  - `EMBREE_RAY_PACKETS=ON`
  - `EMBREE_MAX_INSTANCE_LEVEL_COUNT=1`
  - `EMBREE_COMPACT_POLYS=OFF` (default)

## 2. Supported Scope and Layout

Only static single-mesh triangle BLAS (`RTCScene` containing one static, non-motion-blur `TriangleMesh` at `geomID == 0`, with `numTimeSteps == 1`, `vertex_stride >= 12`, `vertex_stride % 4 == 0`, and `!(scene_flags & RTC_SCENE_FLAG_DYNAMIC)`) using Embree aligned-AABB nodes (`tyAABBNode`) and 4-wide triangle leaf blocks is supported:

| Acceleration structure | Leaf block type | Usage in Luisa (`ROBUST` enabled) | Payload element sizes (x64) |
|---|---|---|---|
| `BVH4` | `Triangle4` (`FAST`) / `Triangle4v` (`ROBUST`) / `Triangle4i` | Compact BLAS (`allow_compaction = true`) selects `BVH4<Triangle4i>` | `BVH4::AABBNode` = 128 B, `Triangle4i` = 80 B |
| `BVH8` | `Triangle4` (`FAST`) / `Triangle4v` (`ROBUST`) / `Triangle4i` | Non-compact BLAS on AVX CPUs (`allow_compaction = false`) selects `BVH8<Triangle4v>` | `BVH8::AABBNode` = 256 B, `Triangle4v` = 176 B, `Triangle4` = 176 B |

The following are rejected cleanly with `UNSUPPORTED_LAYOUT` / `UNSUPPORTED_FEATURE`:
- Dynamic scenes (`allow_update = true` / `RTC_SCENE_FLAG_DYNAMIC`), motion blur (`numTimeSteps > 1`), curves, quads, grids, subdivs, user geometries, instances / instance arrays.
- Quantized (`tyQuantizedNode`), OBB (`tyOBBNode`), or motion-blur (`tyAABBNodeMB`) nodes, or `NodeRef` barriers.

## 3. Binary Encoding, Fixups, and Validation (No Raw Heap Dumps)

No `RTCScene`, `FastAllocator` slab metadata, `vptr` / vtables, or function pointers are serialized. Instead:

1. **Pre-order DFS node/leaf serialization (`kernels/bvh/bvh_accel_archive.cpp`)**:
   - Export uses a fast pass-1 stack count walk (`node_count`, `leaf_block_count`) followed by direct streaming into the caller output buffer (eliminating intermediate node/leaf vector copies).
   - Internal `BVHN<N>::AABBNode` records (child bounds + N child slots) are flattened into an explicit node table (`node_count * sizeof(AABBNode)`).
   - Leaf primitive blocks (`Triangle4`, `Triangle4v`, `Triangle4i`) are copied sequentially into a contiguous leaf-block array (`leaf_block_count * sizeof(Prim)`).
   - Every 64-bit `NodeRef` slot is rewritten from a process-local virtual pointer into a relocatable 64-bit descriptor:
     - `0`: `BVHN<N>::emptyNode` (checked before `isLeaf()` because `emptyNode == tyLeaf`)
     - bit 63 == `0`, value `!= 0`: internal node index `(ref - 1)` in `[0, node_count - 1]`
     - bit 63 == `1`: leaf block slice `(num_blocks << 56) | leaf_block_index` with `1 <= num_blocks <= BVH::maxLeafBlocks` (`1..7`)
2. **Pre-allocation validation on import**:
   - Outer RHI envelope (`XXH3_128` over header, compat blob, and payload) + `EmbreeAccelArchiveExt` compat check verify `embree_version_number == 40401`, `build_config_flags`, and `enabled_cpu_features` before Embree is invoked. (Callers using `import_mesh_archive_with_options` or `import_mesh_archive_file` may opt out of per-payload `XXH3_128` verification via `verify_payload_checksum = false`.)
   - `rtcExImportStaticTriangleBVH` verifies payload magic (`"EBVH"`), ABI version, scene flags (`RTC_SCENE_FLAG_*`), quality flags, `branching_factor`, `(prim_kind, intersect_variant)` validity, exact `sizeof(AABBNode)` / `sizeof(Prim)`, exact payload size (`checked_mul_add`), and rebound `TriangleMesh` vertex count, triangle count, and stride.
   - Before mutating `RTCScene`, the importer scans every node reference sequentially (`child_idx > parent_idx && child_idx < node_count`, disjoint/in-bounds leaf slices) and validates every referenced leaf primitive slot (`geomID == 0`, `primID < triangle_count`, valid aligned `Triangle4i` vertex offsets) in parallel via `embree::parallel_for`.
3. **Single-slab materialization and geometry rebinding (`requires_geometry_rebind = true`)**:
   - Fresh `BVH4` / `BVH8` allocates one contiguous 64-byte-aligned block via `bvh->alloc.init(total_slab, total_slab, total_slab)` + `bvh->alloc.specialAlloc(total_slab)` (`nodes_bytes_aligned + leaf_bytes`), freeing cleanly through standard `~BVHN()` on `rtcReleaseScene` without per-node `FastAllocator` lock contention.
   - Internal `AABBNode` reconstruction (`BVHN<N>::encodeNode` / `BVHN<N>::encodeLeaf` fixups) and leaf-block payload copies execute in parallel across worker threads via `embree::parallel_for` once above small-mesh thresholds (`>= 2048` nodes / `>= 8192` leaf blocks).
   - Native SIMD `Accel::Intersectors` (`Intersector1` / `4` / `8` / `16` filter + nofilter) are freshly populated via `BVH4Factory::BVH4TriangleIntersectorsByPrimKind` / `BVH8Factory::BVH8TriangleIntersectorsByPrimKind` and selected via `scene->accels_select(scene->hasFilterFunction())`.
   - `TriangleMesh` buffers are rebound from caller-owned host buffers via `rtcSetSharedGeometryBuffer` + `rtcCommitGeometry`, and `scene->vertices[0]` is rebound to `mesh->getCompactVertexArray()`.
   - Luisa's direct-triangle TLAS optimization (`FallbackAccel` / `SIMDAccel`) is disabled whenever `mesh->is_imported_blas()` is true so tracing preserves the imported BLAS `RTCScene` via `RTC_GEOMETRY_TYPE_INSTANCE`.

## 4. Known Blockers vs. Upstream Stock Embree

Upstream Embree 4.4.1 exposes no public API to query, serialize, deserialize, swap, or attach `BVH4` / `BVH8` nodes or select `BVH4Factory` / `BVH8Factory` intersectors without rebuilding the BVH from primitives. Safely restoring native Embree traversal without writing a separate custom software BVH therefore requires this minimal, opt-in internal patch (`docs/patches/embree-4.4.1-accel-archive.patch`) on Embree 4.4.1 (`embree4.dll` replacement loaded via `LUISA_EXTRA_EMBREE_BIN`).
