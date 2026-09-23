// Bottom-level build of the fallback: the triangle AABBs (plus the geometry
// copy into the region) and the shared tree build.

#include "fallback_rtx_blas.h"

namespace lc::fallback_rtx {

FallbackBlasBuilder::FallbackBlasBuilder(FallbackRtxStorage &storage) noexcept
    : _storage{&storage},
      _prim_kernel{storage.device().compile(Kernel1D{
          // AABB of the triangles of one BLAS, plus the copy of the geometry the
          // region has to carry.  The two views are raw *uint* words: the vertex
          // stride is the caller's, and only a word-addressed view can walk it
          // without assuming the DSL's own `float3` layout.
          [](BufferVar<uint> vertices, BufferVar<uint> indices, BufferVar<uint4> accel,
             BufferVar<FallbackRtxPrim> prims, BufferVar<uint> reduce, UInt count,
             UInt prim_base, UInt index_base, UInt vertex_base,
             UInt vertex_stride_words, UInt reduce_offset) noexcept {
              set_block_size(sort_block_size);
              // one vertex: `vertex_stride_words` words apart, x/y/z in the
              // first three of them
              auto read_vertex = [](const BufferVar<uint> &words, UInt index,
                                    UInt stride) noexcept {
                  auto first = index * stride;
                  return make_float3(words.read(first + 0u).bitcast<float>(),
                                     words.read(first + 1u).bitcast<float>(),
                                     words.read(first + 2u).bitcast<float>());
              };
              // what the region stores per geometry element: (x, y, z, 0)
              auto pack_vertex = [](Float3 v) noexcept {
                  return make_uint4(v.x.bitcast<uint>(), v.y.bitcast<uint>(),
                                    v.z.bitcast<uint>(), 0u);
              };
              UInt i = dispatch_id().x;
              $if (i < count) {
                  auto i0 = indices.read(3u * i + 0u);
                  auto i1 = indices.read(3u * i + 1u);
                  auto i2 = indices.read(3u * i + 2u);
                  auto v0 = read_vertex(vertices, i0, vertex_stride_words);
                  auto v1 = read_vertex(vertices, i1, vertex_stride_words);
                  auto v2 = read_vertex(vertices, i2, vertex_stride_words);
                  // The geometry travels with the tree: a traversal holds one
                  // descriptor that starts at the region and reads the vertices
                  // and the indices through it.  Several triangles of one mesh
                  // may share a vertex; they write the same value.
                  accel.write(index_base + i, make_uint4(i0, i1, i2, 0u));
                  accel.write(vertex_base + i0, pack_vertex(v0));
                  accel.write(vertex_base + i1, pack_vertex(v1));
                  accel.write(vertex_base + i2, pack_vertex(v2));
                  auto lo = min(min(v0, v1), v2);
                  auto hi = max(max(v0, v1), v2);
                  Var<FallbackRtxPrim> prim;
                  // the local triangle index the leaf node carries
                  prim.id = i;
                  prim.lo = lo;
                  prim.hi = hi;
                  prims.write(prim_base + i, prim);
                  // The scene bounds of this mesh are not known on the host
                  // either, so the build reduces them here (orderable keys, see
                  // fallback_rtx_storage.h) and the Morton kernel reads them
                  // back; the union of the triangle AABBs is the union of the
                  // vertices the triangles use.
                  reduce.atomic(reduce_offset + reduce_min_key + 0u).fetch_min(orderable_key(lo.x));
                  reduce.atomic(reduce_offset + reduce_min_key + 1u).fetch_min(orderable_key(lo.y));
                  reduce.atomic(reduce_offset + reduce_min_key + 2u).fetch_min(orderable_key(lo.z));
                  reduce.atomic(reduce_offset + reduce_max_key + 0u).fetch_max(orderable_key(hi.x));
                  reduce.atomic(reduce_offset + reduce_max_key + 1u).fetch_max(orderable_key(hi.y));
                  reduce.atomic(reduce_offset + reduce_max_key + 2u).fetch_max(orderable_key(hi.z));
              };
          }})} {}

void FallbackBlasBuilder::build(CommandList &commands, FallbackBlas &blas,
                                const FallbackRtxDevice::MeshGeometry &geometry) noexcept {
    // ---- what the build needs from the geometry -----------------------------
    // The fallback reads the *raw* buffers, so it has to state the shape it can
    // read instead of guessing: three uint per triangle, and a vertex whose
    // x/y/z are the first three 32-bit words of a stride that is a multiple of 4.
    if (geometry.vertex_stride == 0u || geometry.vertex_stride % sizeof(uint) != 0u) {
        LUISA_ERROR("The fallback RTX BLAS build needs a vertex stride that is a positive "
                    "multiple of 4 bytes, got {}.",
                    geometry.vertex_stride);
    }
    if (geometry.vertex_buffer_offset % sizeof(uint) != 0u ||
        geometry.triangle_buffer_offset % sizeof(uint) != 0u) {
        LUISA_ERROR("The fallback RTX BLAS build needs vertex/triangle buffer offsets "
                    "aligned to 4 bytes, got {} / {}.",
                    geometry.vertex_buffer_offset, geometry.triangle_buffer_offset);
    }
    if (geometry.triangle_buffer_size == 0u ||
        geometry.triangle_buffer_size % (3u * sizeof(uint)) != 0u) {
        LUISA_ERROR("The fallback RTX BLAS build needs three uint per triangle, but the "
                    "index buffer holds {} bytes.",
                    geometry.triangle_buffer_size);
    }
    if (geometry.vertex_buffer_size == 0u ||
        geometry.vertex_buffer_size % geometry.vertex_stride != 0u) {
        LUISA_ERROR("The fallback RTX BLAS build got {} vertex bytes for a stride of {}.",
                    geometry.vertex_buffer_size, geometry.vertex_stride);
    }
    auto triangle_count = static_cast<uint>(geometry.triangle_buffer_size /
                                            (3u * sizeof(uint)));
    auto vertex_count = static_cast<uint>(geometry.vertex_buffer_size /
                                          geometry.vertex_stride);
    // The views carry the caller's byte offset and cover exactly the view it
    // built the mesh from, so an index is an index into *that* view (the same
    // convention the hardware backends use).  The element stride passed here is
    // what a `uint` is: the kernels address the buffers word by word.
    auto vertices = BufferView<uint>{nullptr, geometry.vertex_buffer, sizeof(uint),
                                     geometry.vertex_buffer_offset,
                                     geometry.vertex_buffer_size / sizeof(uint),
                                     geometry.vertex_buffer_size / sizeof(uint)};
    auto indices = BufferView<uint>{nullptr, geometry.triangle_buffer, sizeof(uint),
                                    geometry.triangle_buffer_offset,
                                    geometry.triangle_buffer_size / sizeof(uint),
                                    geometry.triangle_buffer_size / sizeof(uint)};

    // ---- the region ---------------------------------------------------------
    auto region = _storage->plan_blas(commands, triangle_count, vertex_count);
    _storage->write_region_header(commands, region, 0u /* a BLAS has no table */);
    _storage->reset_reduction(commands, region);
    commands << _prim_kernel(vertices, indices, _storage->accel(), _storage->prims(),
                             _storage->reduce(), triangle_count, region.prim_offset,
                             region.index_base, region.vertex_base,
                             static_cast<uint>(geometry.vertex_stride / sizeof(uint)),
                             region.reduce_offset)
                    .dispatch(triangle_count);
    _storage->build_tree(commands, region);
    // The directory record is what an instance's `blas_index` resolves to: it
    // carries every offset a traversal needs to descend into this region.
    blas.directory_entry = _storage->append_blas_directory(
        commands,
        make_uint4(region.base, region.node_base, region.index_base, region.vertex_base),
        make_uint4(triangle_count, 0u, 0u, 0u));
    blas.region = region;
    blas.triangle_count = triangle_count;
    blas.vertex_count = vertex_count;
    blas.built = true;
}

}// namespace lc::fallback_rtx
