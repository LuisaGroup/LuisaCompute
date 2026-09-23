// The software two-level LBVH facade: the compiled two-level traversal kernel
// plus the thin wrappers that forward to the shared storage and the two
// builders, following the hardware backends' estimate -> create -> pre_build ->
// build order.

#include "software_lbvh.h"

namespace luisa::example::lbvh {

namespace {

// One thread per ray; the traversal itself lives in tlas.cpp, where the top
// level drives the bottom level.  One thread handles the ray
// `ray_offset + i * ray_stride`, so a caller can keep every single submission
// short by walking a strided slice of the rays (striding keeps every slice
// representative of the whole ray set, a contiguous prefix is not: in a camera
// frustum the first rows can miss the whole scene).
[[nodiscard]] auto make_trace_kernel() noexcept {
    return Kernel1D{[](BufferVar<LbvhNode> nodes, BufferVar<LbvhBlas> blas_table,
                       BufferVar<LbvhInstance> instances, BufferVar<float3> vertices,
                       BufferVar<Triangle> triangles, BufferVar<LbvhRay> rays,
                       BufferVar<LbvhHit> hits, UInt tlas_node_offset, UInt ray_offset,
                       UInt ray_stride, UInt count) noexcept {
        set_block_size(64u);
        UInt i = dispatch_id().x;
        $if (i < count) {
            auto index = ray_offset + i * ray_stride;
            auto ray = rays.read(index);
            auto hit = tlas_traversal(ray, tlas_node_offset, nodes, blas_table,
                                      instances, vertices, triangles);
            hits.write(index, hit);
        };
    }};
}

}// namespace

SoftwareLbvh::SoftwareLbvh(Device &device, const Sizes &sizes) noexcept
    : _storage{device, sizes},
      _blas_builder{device},
      _tlas_builder{device},
      _trace_kernel{device.compile(make_trace_kernel())} {}

Blas SoftwareLbvh::create_blas(const AccelOption &option, uint triangle_offset,
                               uint triangle_count, float3 object_min,
                               float3 object_max) noexcept {
    auto blas = _blas_builder.create(option, triangle_offset, triangle_count,
                                     object_min, object_max);
    _blas_count++;
    return blas;
}

Tlas SoftwareLbvh::create_accel(const AccelOption &option, uint instance_count) noexcept {
    return _tlas_builder.create(option, instance_count);
}

size_t SoftwareLbvh::pre_build_blas(Blas &blas) noexcept {
    return _blas_builder.pre_build(_storage, blas);
}

size_t SoftwareLbvh::pre_build_accel(Stream &stream, Tlas &tlas,
                                     luisa::span<const Blas> blases,
                                     luisa::span<const InstanceDesc> instances) noexcept {
    return _tlas_builder.pre_build(stream, _storage, tlas, blases, instances);
}

void SoftwareLbvh::build_blas(Stream &stream, const Blas &blas,
                              const Buffer<float3> &vertices,
                              const Buffer<Triangle> &triangles,
                              AccelBuildRequest request,
                              LbvhBuildTimings *timings) noexcept {
    _blas_builder.build(stream, _storage, blas, vertices, triangles, request, timings);
}

void SoftwareLbvh::build_accel(Stream &stream, const Tlas &tlas,
                               AccelBuildRequest request,
                               LbvhBuildTimings *timings) noexcept {
    _tlas_builder.build(stream, _storage, tlas, request, timings);
}

void SoftwareLbvh::trace_software(Stream &stream, const Buffer<float3> &vertices,
                                  const Buffer<Triangle> &triangles,
                                  const Buffer<LbvhRay> &rays, const Buffer<LbvhHit> &hits,
                                  const Tlas &tlas, uint ray_count, uint ray_offset,
                                  uint ray_stride) noexcept {
    LUISA_ASSERT(tlas.is_pre_built(), "trace_software() on a TLAS that was not built.");
    LUISA_ASSERT(ray_stride > 0u, "a strided traversal needs a positive stride.");
    // The kernel walks the ray indices `ray_offset, ray_offset + ray_stride, ...`
    // `ray_count` times; the caller owns that (offset, stride, count) triple, so
    // this is where a slice that would leave the ray/hit buffers is rejected
    // instead of reading and writing one element past them - the kernels address
    // buffers by index and a device bounds check only exists in a debug build.
    if (ray_count > 0u) {
        auto last = static_cast<size_t>(ray_offset) +
                    static_cast<size_t>(ray_count - 1u) * ray_stride;
        LUISA_ASSERT(last < static_cast<size_t>(rays.size()) &&
                         last < static_cast<size_t>(hits.size()),
                     "strided traversal [{} + {} * {}] runs past the "
                     "{} rays / {} hits it was given.",
                     ray_offset, ray_count, ray_stride, rays.size(), hits.size());
    }
    stream << _trace_kernel(_storage.nodes(), _storage.blas_table(), _storage.instances(),
                            vertices, triangles, rays, hits, tlas.node_offset(), ray_offset,
                            ray_stride, ray_count)
                  .dispatch(ray_count);
}

}// namespace luisa::example::lbvh
