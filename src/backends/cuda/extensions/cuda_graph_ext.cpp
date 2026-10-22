#include "cuda_graph_ext.h"
#include "../cuda_device.h"
#include "../cuda_stream.h"
#include "../cuda_buffer.h"
#include "../cuda_bindless_array.h"
#include "../cuda_shader.h"
#include "../cuda_shader_tile.h"
#include "../cuda_texture.h"
#include "../cuda_error.h"
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/core/stl/format.h>
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <luisa/core/stl/algorithm.h>

namespace luisa::compute::cuda {

namespace {
namespace cuda_graph_ext_detail {

[[nodiscard]] CUstream to_cu_stream(uint64_t stream_handle) noexcept {
    return reinterpret_cast<CUDAStream *>(stream_handle)->handle();
}

[[nodiscard]] CUcontext current_context() noexcept {
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    return ctx;
}

[[nodiscard]] unsigned long long to_instantiate_flags(CudaGraphExt::InstantiateFlag flags) noexcept {
    unsigned long long f = 0u;
    auto v = static_cast<uint32_t>(flags);
    if (v & static_cast<uint32_t>(CudaGraphExt::InstantiateFlag::AUTO_FREE_ON_LAUNCH)) {
        f |= CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    }
    if (v & static_cast<uint32_t>(CudaGraphExt::InstantiateFlag::DEVICE_LAUNCH)) {
        f |= CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH;
    }
    return f;
}

void CUDA_CB cuda_graph_host_copy_callback(void *user_data) noexcept {
    auto *data = static_cast<CudaGraphHostCopyData *>(user_data);
    std::memcpy(data->dst, data->src, data->size);
}

void cuda_graph_dag_fail(const char *what, CUresult err) noexcept {
    const char *err_name = nullptr;
    cuGetErrorName(err, &err_name);
    LUISA_WARNING_WITH_LOCATION("CudaGraphExt: {} failed: {}",
                                what, err_name ? err_name : "unknown");
}

void cuda_graph_refuse_command(const char *what) noexcept {
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: {} cannot be captured into a CUDA graph. Refusing the "
        "capture - a graph built from this command list could not reproduce "
        "the semantics of the ordinary submission path.",
        what);
}

// -- G5 (SA-6): graph memory helpers ---------------------------------------

/// `alignment` is a power of two (and non-zero). The CUDA memory node API
/// takes no explicit alignment, so the only way to express it is the requested
/// size (see `round_up_to_alignment`), which is why a non-power-of-two value
/// has to be refused instead of silently normalized.
[[nodiscard]] bool is_power_of_two(size_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

/// Round `size` up to the next multiple of `alignment`.
///
/// `cuGraphAddMemAllocNode` has no alignment parameter: the node's `bytesize`
/// is the only lever this API has on the address the driver hands out, so the
/// extension expresses `alignment` by asking for a size rounded up to it (a
/// pool allocation is served per requested size). Only a power of two is
/// accepted, so the rounding really is an "at least this aligned" request.
[[nodiscard]] size_t round_up_to_alignment(size_t size, size_t alignment) noexcept {
    return (size + alignment - 1u) / alignment * alignment;
}

/// Add `count` edges to `graph`, from `from[i]` to `to[i]`.
///
/// `cuGraphAddDependencies` gained a per-edge annotation in CUDA 12.4
/// (`cuGraphAddDependencies_v2`, taking one `CUgraphEdgeData` per edge). On
/// 13.x headers cuda.h remaps the unversioned name onto the `_v2` entry point
/// (`#if defined(cuGraphAddDependencies)`); on 12.1-12.8 the plain 4-arg form
/// is a plain default edge, which is exactly what this helper wants. A
/// zero-initialized annotation means "full completion of the upstream node
/// with memory visibility", i.e. the semantics of the pre-12.4 call, so both
/// branches create plain edges and this stays correct on every toolkit.
[[nodiscard]] CUresult add_graph_dependencies(CUgraph graph,
                                              const CUgraphNode *from,
                                              const CUgraphNode *to,
                                              size_t count) noexcept {
#if defined(cuGraphAddDependencies)
    luisa::vector<CUgraphEdgeData> edges(count);
    for (auto &edge : edges) { edge = CUgraphEdgeData{}; }
    return cuGraphAddDependencies(graph, from, to, edges.data(), count);
#else
    return cuGraphAddDependencies(graph, from, to, count);
#endif
}

#if CUDA_VERSION >= 12040
// -- SA-3: annotated (typed) edges -----------------------------------------
//
// The annotated entry points `cuGraphAddDependencies_v2` /
// `cuGraphRemoveDependencies_v2` exist from CUDA 12.4 on (declared directly,
// and - on 13.x - also reachable through the remapped unversioned names; the
// `_v2` spelling is used explicitly so the same code compiles on 12.4-12.8,
// where the unversioned names are NOT remapped). On an older toolkit there is
// no way to express an edge annotation at all, and the caller refuses loudly
// instead of creating an ordinary edge that would silently drop the PDL
// contract.

/// The annotation of an edge. A zero-initialized `CUgraphEdgeData` means "full
/// completion of the upstream node with memory visibility to the downstream
/// node", i.e. exactly the pre-12.4 semantics (this is also what the driver
/// assumes for a null annotation). The programmatic annotation instead ties the
/// downstream's launch to the upstream kernel's *programmatic* completion port:
/// the upstream signals it explicitly with
/// `trigger_programmatic_launch_completion()` (or implicitly by terminating),
/// and the downstream may then be scheduled early but must call
/// `grid_dependency_synchronize()` before it observes the upstream's writes -
/// the early launch carries no memory visibility of its own.
///
/// The programmatic port is an OUTGOING port of the upstream kernel; the
/// downstream side of the edge keeps the default port (no node type defines a
/// non-zero `to_port`).
[[nodiscard]] CUgraphEdgeData make_edge_data(CudaGraphExt::DependencyType type) noexcept {
    CUgraphEdgeData edge_data{};
    if (type == CudaGraphExt::DependencyType::PROGRAMMATIC) {
        edge_data.from_port = static_cast<unsigned char>(CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC);
        edge_data.to_port = static_cast<unsigned char>(CU_GRAPH_KERNEL_NODE_PORT_DEFAULT);
        edge_data.type = static_cast<unsigned char>(CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC);
    }
    return edge_data;
}

/// Add the single edge `from -> to` with an explicit annotation.
[[nodiscard]] CUresult add_graph_dependency_typed(CUgraph graph,
                                                  const CUgraphNode *from,
                                                  const CUgraphNode *to,
                                                  const CUgraphEdgeData &edge_data) noexcept {
    return cuGraphAddDependencies_v2(graph, from, to, &edge_data, 1u);
}

/// Remove the single edge `from -> to` annotated with `edge_data`.
///
/// A pair that carries no edge at all is rejected (CUDA_ERROR_INVALID_VALUE on
/// the driver this was verified against), so a successful removal proves that
/// the edge existed - that is what makes retyping an edge possible at all,
/// since there is no API that mutates the annotation of an existing edge in
/// place. The annotation is passed along because the annotated API takes it and
/// because a removal of an annotated edge conceptually names the annotation it
/// removes; the driver (13.4) currently matches on the node pair only, so the
/// caller always passing the annotation it believes the edge carries stays
/// correct either way.
[[nodiscard]] CUresult remove_graph_dependency_typed(CUgraph graph,
                                                     const CUgraphNode *from,
                                                     const CUgraphNode *to,
                                                     const CUgraphEdgeData &edge_data) noexcept {
    return cuGraphRemoveDependencies_v2(graph, from, to, &edge_data, 1u);
}
#endif// CUDA_VERSION >= 12040

#if CUDA_VERSION >= 12030
/// Create a node from its `CUgraphNodeParams` description (CUDA 12.0+, needed
/// for conditional nodes from 12.3 on).
///
/// `cuGraphAddNode` gained a per-edge annotation in CUDA 12.4: from then on
/// cuda.h remaps the unversioned name onto the annotated entry point, which
/// takes one `CUgraphEdgeData` per dependency. A null annotation means exactly
/// the pre-12.4 "default edge" semantics, which is what every caller here
/// wants, so the `defined()` test selects the matching signature without
/// changing behavior (same pattern as `add_graph_dependencies`).
[[nodiscard]] CUresult add_graph_node(CUgraph graph, const CUgraphNode *deps,
                                      size_t num_deps, CUgraphNodeParams *params,
                                      CUgraphNode *node_out) noexcept {
#if defined(cuGraphAddNode)
    return cuGraphAddNode(node_out, graph, deps, nullptr, num_deps, params);
#else
    return cuGraphAddNode(node_out, graph, deps, num_deps, params);
#endif
}
#endif// CUDA_VERSION >= 12030

/// Precise command -> command dependency tracking for CUDA graph DAG
/// construction, mirroring the interval bookkeeping of CommandReorderVisitor
/// but recording the exact set of earlier commands a new command must wait
/// on instead of assigning a totally-ordered layer:
/// - a READ depends on the writers of the ranges it touches (RAW);
/// - a WRITE depends on every command that wrote or read the range since
///   the last write (WAW + WAR).
/// Commands with no recorded hazard share no dependency entry, so the
/// resulting graph has no edge between them and the launcher may schedule
/// them concurrently.
struct DagInterval {
    uint64_t min;
    uint64_t max;
    // Node-command indices that wrote / read this interval. `writers` stays
    // empty for regions that were only ever read (e.g. reads of data a
    // command list never wrote).
    luisa::vector<uint32_t> writers;
    luisa::vector<uint32_t> readers;
};

struct DagResource {
    // Disjoint intervals sorted by `min`.
    luisa::vector<DagInterval> views;

    [[nodiscard]] static size_t lower_bound(const luisa::vector<DagInterval> &views,
                                            uint64_t min, uint64_t max) noexcept {
        size_t lo = 0u, hi = views.size();
        while (lo < hi) {
            auto mid = lo + (hi - lo) / 2u;
            if (views[mid].max <= min) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    // Records one access over [min, max) and appends every earlier command
    // it must wait on to `deps`. The caller must drop entries equal to
    // `self` (one command may touch the same resource through several
    // arguments) and deduplicate.
    void apply_access(uint64_t min, uint64_t max, bool is_write, uint32_t self,
                      luisa::vector<uint32_t> &deps) {
        auto lo = lower_bound(views, min, max);
        auto hi = lo;
        while (hi < views.size() && views[hi].min < max) { ++hi; }
        DagInterval mid{min, max, {}, {}};
        for (auto i = lo; i < hi; ++i) {
            auto &v = views[i];
            if (is_write) {
                deps.insert(deps.end(), v.writers.begin(), v.writers.end());
                deps.insert(deps.end(), v.readers.begin(), v.readers.end());
            } else {
                deps.insert(deps.end(), v.writers.begin(), v.writers.end());
                mid.writers.insert(mid.writers.end(), v.writers.begin(), v.writers.end());
                mid.readers.insert(mid.readers.end(), v.readers.begin(), v.readers.end());
            }
        }
        if (is_write) {
            mid.writers.push_back(self);
        } else {
            mid.readers.push_back(self);
        }
        // Fast path: no colliding interval, insert the new one in sorted
        // position without building a replacement window.
        if (hi == lo) {
            views.insert(views.begin() + lo, std::move(mid));
            return;
        }
        // Replace the colliding window with at most three intervals: the
        // pieces outside [min, max) keep their exact state, the covered
        // piece carries the merged state of this access.
        luisa::vector<DagInterval> replacement;
        if (hi > lo && views[lo].min < min) {
            replacement.push_back(DagInterval{views[lo].min, min,
                                              views[lo].writers, views[lo].readers});
        }
        replacement.push_back(std::move(mid));
        if (hi > lo && max < views[hi - 1u].max) {
            replacement.push_back(DagInterval{max, views[hi - 1u].max,
                                              views[hi - 1u].writers,
                                              views[hi - 1u].readers});
        }
        views.erase(views.begin() + lo, views.begin() + hi);
        views.insert(views.begin() + lo, replacement.begin(), replacement.end());
    }
};

/// Byte geometry of one (sub-)region of a texture level. This is exactly the
/// arithmetic `CUDACommandEncoder::visit(BufferToTextureCopyCommand*)`,
/// `visit(TextureUploadCommand*)`, `visit(TextureDownloadCommand*)`,
/// `visit(TextureCopyCommand*)` and `visit(TextureToBufferCopyCommand*)`
/// perform, so a graph memcpy node addresses the same bytes a stream-ordered
/// copy of the same command would.
struct TextureCopyGeometry {
    size_t row_bytes{0u};  // CUDA_MEMCPY3D::WidthInBytes / row pitch
    size_t height{0u};     // CUDA_MEMCPY3D::Height (rows per slice)
    size_t depth{0u};      // CUDA_MEMCPY3D::Depth (slices)
    size_t total_bytes{0u};// tightly packed footprint of the region
    size_t element_size{0u};
};

[[nodiscard]] TextureCopyGeometry texture_copy_geometry(PixelStorage storage, uint3 size) noexcept {
    auto row_bytes = pixel_storage_size(storage, make_uint3(size.x, 1u, 1u));
    auto rows_per_slice = pixel_storage_size(storage, make_uint3(size.xy(), 1u)) / row_bytes;
    return TextureCopyGeometry{
        .row_bytes = row_bytes,
        .height = rows_per_slice,
        .depth = size.z,
        .total_bytes = pixel_storage_size(storage, size),
        .element_size = pixel_storage_size(storage, make_uint3(1u, 1u, 1u)),
    };
}

/// Prepares a command list for graph construction.
///
///  - Every host -> device upload (buffer or texture) is re-pointed at a
///    pinned host staging buffer that the memcpy node can read from (a CUDA
///    graph node may not read arbitrary pageable host memory).
///  - Every device -> host download (buffer or texture) gets a pinned staging
///    buffer the memcpy node writes into, plus the payload copy that forwards
///    it to the user's host buffer. `CudaGraphExtImpl::launch` runs those
///    copies with `cuLaunchHostFunc`, exactly like the previous
///    implementation.
///  - Content the DAG builder cannot represent is refused loudly instead of
///    being dropped from the graph.
struct UploadStagingVisitor final : MutableCommandVisitor {

    luisa::vector<luisa::shared_ptr<PinnedHostBlock>> &host_allocs;
    luisa::vector<CudaGraphHostCopyData> &host_copies;
    bool ok{true};

    UploadStagingVisitor(luisa::vector<luisa::shared_ptr<PinnedHostBlock>> &host_allocs,
                         luisa::vector<CudaGraphHostCopyData> &host_copies) noexcept
        : host_allocs{host_allocs}, host_copies{host_copies} {}

    [[nodiscard]] void *allocate_pinned(size_t size) noexcept {
        void *host_mem = nullptr;
        if (cuMemAllocHost(&host_mem, size) != CUDA_SUCCESS) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: failed to allocate {} bytes of pinned host "
                "staging memory for a graph upload/download. Refusing the capture.",
                size);
            ok = false;
            return nullptr;
        }
        host_allocs.emplace_back(luisa::make_shared<PinnedHostBlock>(host_mem));
        return host_mem;
    }

    void visit(BufferUploadCommand *upload) noexcept override {
        if (!ok) { return; }
        auto host_mem = allocate_pinned(upload->size());
        if (host_mem == nullptr) { return; }
        std::memcpy(host_mem, upload->data(), upload->size());
        upload->set_data(host_mem);
    }

    void visit(BufferDownloadCommand *download) noexcept override {
        if (!ok) { return; }
        auto host_mem = allocate_pinned(download->size());
        if (host_mem == nullptr) { return; }
        host_copies.push_back(CudaGraphHostCopyData{
            .dst = download->data(),
            .src = host_mem,
            .size = download->size(),
        });
        download->set_data(host_mem);
    }

    void visit(TextureUploadCommand *upload) noexcept override {
        if (!ok) { return; }
        auto size_bytes = pixel_storage_size(upload->storage(), upload->size());
        auto host_mem = allocate_pinned(size_bytes);
        if (host_mem == nullptr) { return; }
        std::memcpy(host_mem, upload->data(), size_bytes);
        upload->set_data(host_mem);
    }

    void visit(TextureDownloadCommand *download) noexcept override {
        if (!ok) { return; }
        auto size_bytes = pixel_storage_size(download->storage(), download->size());
        auto host_mem = allocate_pinned(size_bytes);
        if (host_mem == nullptr) { return; }
        host_copies.push_back(CudaGraphHostCopyData{
            .dst = download->data(),
            .src = host_mem,
            .size = size_bytes,
        });
        download->set_data(host_mem);
    }

    void visit(BufferCopyCommand *) noexcept override {}
    void visit(BufferToTextureCopyCommand *) noexcept override {}
    void visit(ShaderDispatchCommand *) noexcept override {}
    void visit(TextureCopyCommand *) noexcept override {}
    void visit(TextureToBufferCopyCommand *) noexcept override {}

    // Ray-tracing builds must not be baked into a graph: they are driven by
    // OptiX/fallback host code that has no graph-node equivalent here, and
    // silently turning them into a missing node would make the graph quietly
    // different from the command list it was created from.
    void visit(AccelBuildCommand *) noexcept override { cuda_graph_refuse_command("AccelBuildCommand"); ok = false; }
    void visit(MeshBuildCommand *) noexcept override { cuda_graph_refuse_command("MeshBuildCommand"); ok = false; }
    void visit(CurveBuildCommand *) noexcept override { cuda_graph_refuse_command("CurveBuildCommand"); ok = false; }
    void visit(ProceduralPrimitiveBuildCommand *) noexcept override { cuda_graph_refuse_command("ProceduralPrimitiveBuildCommand"); ok = false; }
    void visit(MotionInstanceBuildCommand *) noexcept override { cuda_graph_refuse_command("MotionInstanceBuildCommand"); ok = false; }
    // The descriptor update runs a host-side compaction pass; it has no graph
    // node and must not be dropped.
    void visit(BindlessArrayUpdateCommand *) noexcept override { cuda_graph_refuse_command("BindlessArrayUpdateCommand"); ok = false; }
    void visit(CustomCommand *) noexcept override { cuda_graph_refuse_command("CustomCommand"); ok = false; }
};

/// Builds the explicit dependency DAG of a `CommandList` into a `CUgraph`.
///
/// The graph is built node by node (never by stream capture, which would
/// linearize every command onto one stream and replay as a serial chain):
/// every argument is tracked by its declared usage - like the command reorder
/// pass - and the edges of the resulting graph are only the real RAW/WAW/WAR
/// command -> command dependencies. Hazard-free commands share no edge, so a
/// replay of an independent batch overlaps on the device; a hazard chain gets
/// exactly the edges that preserve the sequential semantics of the original
/// command list.
///
/// The visitor writes into a caller-owned node list whose position *is* the
/// `node_index` every `CudaGraphExt` node API uses: one record per
/// node-producing command/entry, in creation order. It owns its hazard state,
/// so `reset()` re-targets the same visitor at another (empty) graph - this is
/// how a later phase builds conditional body graphs with the very same DAG
/// analysis as the top-level graph.
struct GraphDagVisitor final : MutableCommandVisitor {

    CUgraph graph{nullptr};
    CUcontext ctx{nullptr};
    luisa::vector<CudaGraphExtImpl::GraphNodeRecord> &nodes;
    luisa::unordered_map<uint64_t, DagResource> buffer_states;
    luisa::unordered_map<uint64_t, DagResource> texture_states;
    luisa::unordered_map<uint64_t, DagResource> bindless_states;
    luisa::vector<uint32_t> deps;
    luisa::vector<CUgraphNode> dep_nodes;
    luisa::vector<uint32_t> tile_nodes;
    bool ok{true};

    GraphDagVisitor(CUgraph graph, CUcontext ctx,
                    luisa::vector<CudaGraphExtImpl::GraphNodeRecord> &nodes) noexcept
        : graph{graph}, ctx{ctx}, nodes{nodes} {}

    /// Re-target the visitor at another graph. The caller passes a fresh,
    /// empty node list (the visitor appends to it); every hazard/state map is
    /// dropped so the new graph starts from a clean slate. `nodes` itself is
    /// caller-owned and is deliberately left alone.
    void reset(CUgraph new_graph, CUcontext new_ctx) noexcept {
        graph = new_graph;
        ctx = new_ctx;
        buffer_states.clear();
        texture_states.clear();
        bindless_states.clear();
        deps.clear();
        dep_nodes.clear();
        tile_nodes.clear();
        ok = true;
    }

    // Deduplicate the accumulated dependency indices, drop self
    // edges and map the survivors to their CUDA graph nodes.
    [[nodiscard]] luisa::span<const CUgraphNode> finish_deps(uint32_t self) noexcept {
        luisa::sort(deps.begin(), deps.end());
        deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
        dep_nodes.clear();
        for (auto d : deps) {
            if (d != self) { dep_nodes.push_back(nodes[d].node); }
        }
        return {dep_nodes.data(), dep_nodes.size()};
    }

    void push_node(uint32_t self, CUgraphNode node,
                   CudaGraphExtImpl::GraphNodeKind kind) noexcept {
        LUISA_ASSERT(static_cast<size_t>(self) == nodes.size(),
                     "CUDA graph nodes must be appended in creation order.");
        nodes.push_back(CudaGraphExtImpl::GraphNodeRecord{.node = node, .kind = kind});
    }

    void add_memcpy_node(const CUDA_MEMCPY3D &params, uint32_t self,
                         const char *what) noexcept {
        CUgraphNode node = nullptr;
        auto dependencies = finish_deps(self);
        if (auto err = cuGraphAddMemcpyNode(&node, graph,
                                            dependencies.data(), dependencies.size(),
                                            &params, ctx);
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail(what, err);
            ok = false;
            return;
        }
        push_node(self, node, CudaGraphExtImpl::GraphNodeKind::MEMCPY);
    }

    void visit(BufferUploadCommand *upload) noexcept override {
        if (!ok) { return; }
        auto *buffer = reinterpret_cast<const CUDABuffer *>(upload->handle());
        auto self = static_cast<uint32_t>(nodes.size());
        deps.clear();
        buffer_states[upload->handle()].apply_access(
            upload->offset(), upload->offset() + upload->size(),
            /*is_write=*/true, self, deps);
        CUDA_MEMCPY3D params{};
        params.srcMemoryType = CU_MEMORYTYPE_HOST;
        params.srcHost = upload->data();
        params.srcPitch = upload->size();
        params.srcHeight = 1;
        params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        params.dstDevice = static_cast<CUdeviceptr>(buffer->device_address() + upload->offset());
        params.dstPitch = upload->size();
        params.dstHeight = 1;
        params.WidthInBytes = upload->size();
        params.Height = 1;
        params.Depth = 1;
        add_memcpy_node(params, self, "cuGraphAddMemcpyNode (buffer upload)");
    }

    void visit(BufferDownloadCommand *download) noexcept override {
        if (!ok) { return; }
        auto *buffer = reinterpret_cast<const CUDABuffer *>(download->handle());
        auto self = static_cast<uint32_t>(nodes.size());
        deps.clear();
        buffer_states[download->handle()].apply_access(
            download->offset(), download->offset() + download->size(),
            /*is_write=*/false, self, deps);
        CUDA_MEMCPY3D params{};
        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        params.srcDevice = static_cast<CUdeviceptr>(buffer->device_address() + download->offset());
        params.srcPitch = download->size();
        params.srcHeight = 1;
        params.dstMemoryType = CU_MEMORYTYPE_HOST;
        // Pinned staging buffer; launch() forwards the payload to the
        // user-provided host buffer with a stream-ordered callback.
        params.dstHost = download->data();
        params.dstPitch = download->size();
        params.dstHeight = 1;
        params.WidthInBytes = download->size();
        params.Height = 1;
        params.Depth = 1;
        add_memcpy_node(params, self, "cuGraphAddMemcpyNode (buffer download)");
    }

    void visit(BufferCopyCommand *copy) noexcept override {
        if (!ok) { return; }
        auto *src = reinterpret_cast<const CUDABuffer *>(copy->src_handle());
        auto *dst = reinterpret_cast<const CUDABuffer *>(copy->dst_handle());
        auto self = static_cast<uint32_t>(nodes.size());
        deps.clear();
        buffer_states[copy->src_handle()].apply_access(
            copy->src_offset(), copy->src_offset() + copy->size(),
            /*is_write=*/false, self, deps);
        buffer_states[copy->dst_handle()].apply_access(
            copy->dst_offset(), copy->dst_offset() + copy->size(),
            /*is_write=*/true, self, deps);
        CUDA_MEMCPY3D params{};
        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        params.srcDevice = static_cast<CUdeviceptr>(src->device_address() + copy->src_offset());
        params.srcPitch = copy->size();
        params.srcHeight = 1;
        params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        params.dstDevice = static_cast<CUdeviceptr>(dst->device_address() + copy->dst_offset());
        params.dstPitch = copy->size();
        params.dstHeight = 1;
        params.WidthInBytes = copy->size();
        params.Height = 1;
        params.Depth = 1;
        add_memcpy_node(params, self, "cuGraphAddMemcpyNode (buffer copy)");
    }

    // ---- texture family ------------------------------------------------
    // Every texture command becomes a CUDA_MEMCPY3D array memcpy node whose
    // fields mirror `CUDACommandEncoder::visit` for the same command, with
    // the texture side addressed through `CUDATexture::level(mip)`.

    void visit(BufferToTextureCopyCommand *copy) noexcept override {
        if (!ok) { return; }
        auto *texture = reinterpret_cast<const CUDATexture *>(copy->texture());
        auto *buffer = reinterpret_cast<const CUDABuffer *>(copy->buffer());
        auto geometry = cuda_graph_ext_detail::texture_copy_geometry(copy->storage(), copy->size());
        if (copy->buffer_offset() >= buffer->size_bytes() ||
            buffer->size_bytes() - copy->buffer_offset() < geometry.total_bytes) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: BufferToTextureCopyCommand source range "
                "[{}, {}) does not fit the {}-byte buffer. Refusing the capture.",
                copy->buffer_offset(), copy->buffer_offset() + geometry.total_bytes,
                buffer->size_bytes());
            ok = false;
            return;
        }
        // `dstXInBytes` is a byte offset inside a row; block-compressed
        // formats would need block-granular arithmetic, so only x == 0 is
        // allowed for them (same restriction as the ordinary path).
        if (is_block_compressed(copy->storage()) && copy->texture_offset().x != 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: sub-region direct-storage copies into "
                "block-compressed textures are not supported on CUDA. "
                "Refusing the capture.");
            ok = false;
            return;
        }
        auto self = static_cast<uint32_t>(nodes.size());
        deps.clear();
        buffer_states[copy->buffer()].apply_access(
            copy->buffer_offset(), copy->buffer_offset() + geometry.total_bytes,
            /*is_write=*/false, self, deps);
        // An upload of one mip is a write of that single level.
        texture_states[copy->texture()].apply_access(
            copy->level(), static_cast<uint64_t>(copy->level()) + 1u,
            /*is_write=*/true, self, deps);
        CUDA_MEMCPY3D params{};
        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        params.srcDevice = static_cast<CUdeviceptr>(buffer->device_address() + copy->buffer_offset());
        params.srcPitch = geometry.row_bytes;
        params.srcHeight = geometry.height;
        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray = texture->level(copy->level());
        params.dstXInBytes = static_cast<size_t>(copy->texture_offset().x) * geometry.element_size;
        params.dstY = copy->texture_offset().y;
        params.dstZ = copy->texture_offset().z;
        params.WidthInBytes = geometry.row_bytes;
        params.Height = geometry.height;
        params.Depth = geometry.depth;
        add_memcpy_node(params, self, "cuGraphAddMemcpyNode (buffer -> texture)");
    }

    void visit(TextureUploadCommand *upload) noexcept override {
        if (!ok) { return; }
        auto *texture = reinterpret_cast<const CUDATexture *>(upload->handle());
        auto geometry = cuda_graph_ext_detail::texture_copy_geometry(upload->storage(), upload->size());
        auto self = static_cast<uint32_t>(nodes.size());
        deps.clear();
        texture_states[upload->handle()].apply_access(
            upload->level(), static_cast<uint64_t>(upload->level()) + 1u,
            /*is_write=*/true, self, deps);
        CUDA_MEMCPY3D params{};
        params.srcMemoryType = CU_MEMORYTYPE_HOST;
        // Pinned staging buffer filled by the staging visitor.
        params.srcHost = upload->data();
        params.srcPitch = geometry.row_bytes;
        params.srcHeight = geometry.height;
        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray = texture->level(upload->level());
        params.WidthInBytes = geometry.row_bytes;
        params.Height = geometry.height;
        params.Depth = geometry.depth;
        add_memcpy_node(params, self, "cuGraphAddMemcpyNode (texture upload)");
    }

    void visit(TextureDownloadCommand *download) noexcept override {
        if (!ok) { return; }
        auto *texture = reinterpret_cast<const CUDATexture *>(download->handle());
        auto geometry = cuda_graph_ext_detail::texture_copy_geometry(download->storage(), download->size());
        auto self = static_cast<uint32_t>(nodes.size());
        deps.clear();
        texture_states[download->handle()].apply_access(
            download->level(), static_cast<uint64_t>(download->level()) + 1u,
            /*is_write=*/false, self, deps);
        CUDA_MEMCPY3D params{};
        params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        params.srcArray = texture->level(download->level());
        params.dstMemoryType = CU_MEMORYTYPE_HOST;
        // Pinned staging buffer; launch() forwards the payload to the
        // user-provided host buffer with a stream-ordered callback.
        params.dstHost = download->data();
        params.dstPitch = geometry.row_bytes;
        params.dstHeight = geometry.height;
        params.WidthInBytes = geometry.row_bytes;
        params.Height = geometry.height;
        params.Depth = geometry.depth;
        add_memcpy_node(params, self, "cuGraphAddMemcpyNode (texture download)");
    }

    void visit(TextureCopyCommand *copy) noexcept override {
        if (!ok) { return; }
        auto *src_texture = reinterpret_cast<const CUDATexture *>(copy->src_handle());
        auto *dst_texture = reinterpret_cast<const CUDATexture *>(copy->dst_handle());
        auto geometry = cuda_graph_ext_detail::texture_copy_geometry(copy->storage(), copy->size());
        auto self = static_cast<uint32_t>(nodes.size());
        deps.clear();
        // One level is copied from `src_level` to `dst_level` (the ordinary
        // path does the same: it only ever touches `level(level)`), so the
        // hazard covers exactly those two single levels.
        texture_states[copy->src_handle()].apply_access(
            copy->src_level(), static_cast<uint64_t>(copy->src_level()) + 1u,
            /*is_write=*/false, self, deps);
        texture_states[copy->dst_handle()].apply_access(
            copy->dst_level(), static_cast<uint64_t>(copy->dst_level()) + 1u,
            /*is_write=*/true, self, deps);
        CUDA_MEMCPY3D params{};
        params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        params.srcArray = src_texture->level(copy->src_level());
        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray = dst_texture->level(copy->dst_level());
        params.WidthInBytes = geometry.row_bytes;
        params.Height = geometry.height;
        params.Depth = geometry.depth;
        add_memcpy_node(params, self, "cuGraphAddMemcpyNode (texture copy)");
    }

    void visit(TextureToBufferCopyCommand *copy) noexcept override {
        if (!ok) { return; }
        auto *texture = reinterpret_cast<const CUDATexture *>(copy->texture());
        auto *buffer = reinterpret_cast<const CUDABuffer *>(copy->buffer());
        auto geometry = cuda_graph_ext_detail::texture_copy_geometry(copy->storage(), copy->size());
        if (copy->buffer_offset() >= buffer->size_bytes() ||
            buffer->size_bytes() - copy->buffer_offset() < geometry.total_bytes) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: TextureToBufferCopyCommand destination range "
                "[{}, {}) does not fit the {}-byte buffer. Refusing the capture.",
                copy->buffer_offset(), copy->buffer_offset() + geometry.total_bytes,
                buffer->size_bytes());
            ok = false;
            return;
        }
        auto self = static_cast<uint32_t>(nodes.size());
        deps.clear();
        texture_states[copy->texture()].apply_access(
            copy->level(), static_cast<uint64_t>(copy->level()) + 1u,
            /*is_write=*/false, self, deps);
        buffer_states[copy->buffer()].apply_access(
            copy->buffer_offset(), copy->buffer_offset() + geometry.total_bytes,
            /*is_write=*/true, self, deps);
        CUDA_MEMCPY3D params{};
        params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        params.srcArray = texture->level(copy->level());
        params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        params.dstDevice = static_cast<CUdeviceptr>(buffer->device_address() + copy->buffer_offset());
        params.dstPitch = geometry.row_bytes;
        params.dstHeight = geometry.height;
        params.WidthInBytes = geometry.row_bytes;
        params.Height = geometry.height;
        params.Depth = geometry.depth;
        add_memcpy_node(params, self, "cuGraphAddMemcpyNode (texture -> buffer)");
    }

    // Refused by the staging visitor before this visitor ever runs; kept here
    // as a loud refusal so the DAG builder can never silently drop them.
    void visit(AccelBuildCommand *) noexcept override { cuda_graph_refuse_command("AccelBuildCommand"); ok = false; }
    void visit(MeshBuildCommand *) noexcept override { cuda_graph_refuse_command("MeshBuildCommand"); ok = false; }
    void visit(CurveBuildCommand *) noexcept override { cuda_graph_refuse_command("CurveBuildCommand"); ok = false; }
    void visit(ProceduralPrimitiveBuildCommand *) noexcept override { cuda_graph_refuse_command("ProceduralPrimitiveBuildCommand"); ok = false; }
    void visit(MotionInstanceBuildCommand *) noexcept override { cuda_graph_refuse_command("MotionInstanceBuildCommand"); ok = false; }
    void visit(BindlessArrayUpdateCommand *) noexcept override { cuda_graph_refuse_command("BindlessArrayUpdateCommand"); ok = false; }
    void visit(CustomCommand *) noexcept override { cuda_graph_refuse_command("CustomCommand"); ok = false; }

    /// Rebuild the by-value `Params` struct argument exactly like
    /// `CUDAShaderNative::_launch` does (16-byte-aligned slots, buffer
    /// arguments stored as their `CUDABuffer::Binding`, uniforms copied
    /// verbatim, trailing launch-size-and-kernel-id slot), so the DAG
    /// kernel node reads byte-identical arguments to a live dispatch.
    /// Any dispatch the normal path would treat specially (printing,
    /// bound arguments, indirect or multiple dispatch, a non-native
    /// shader, Accel arguments) is refused instead of approximated,
    /// so a captured graph is never subtly different from the
    /// ordinary submission path.
    [[nodiscard]] bool pack_kernel_arguments(const ShaderDispatchCommand *dispatch,
                                             const CUDAShader *shader,
                                             luisa::vector<std::byte> &argument_buffer) const noexcept {
        auto dispatch_size = dispatch->dispatch_size();
        static constexpr auto cuda_shader_native_alignment = 16u;
        auto argument_buffer_offset = size_t{0u};
        auto allocate_argument = [&argument_buffer_offset](size_t bytes) noexcept {
            auto offset = (argument_buffer_offset + cuda_shader_native_alignment - 1u) / cuda_shader_native_alignment * cuda_shader_native_alignment;
            argument_buffer_offset = offset + bytes;
            return offset;
        };
        auto store = [&](size_t offset, const void *src, size_t bytes) noexcept {
            if (argument_buffer.size() < offset + bytes) {
                argument_buffer.resize(offset + bytes);
            }
            std::memcpy(argument_buffer.data() + offset, src, bytes);
        };
        for (auto &&arg : dispatch->arguments()) {
            using Tag = ShaderDispatchCommand::Argument::Tag;
            switch (arg.tag) {
                case Tag::BUFFER: {
                    auto offset = allocate_argument(sizeof(CUDABuffer::Binding));
                    auto *buffer = reinterpret_cast<const CUDABuffer *>(arg.buffer.handle);
                    auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                    store(offset, &binding, sizeof(binding));
                    break;
                }
                case Tag::TEXTURE: {
                    auto offset = allocate_argument(sizeof(CUDATexture::Binding));
                    auto *texture = reinterpret_cast<const CUDATexture *>(arg.texture.handle);
                    auto binding = texture->binding(arg.texture.level);
                    store(offset, &binding, sizeof(binding));
                    break;
                }
                case Tag::UNIFORM: {
                    auto uniform = dispatch->uniform(arg.uniform);
                    auto offset = allocate_argument(uniform.size_bytes());
                    store(offset, uniform.data(), uniform.size_bytes());
                    break;
                }
                case Tag::BINDLESS_ARRAY: {
                    auto offset = allocate_argument(sizeof(CUDABindlessArray::Binding));
                    auto *array = reinterpret_cast<const CUDABindlessArray *>(arg.bindless_array.handle);
                    auto binding = array->binding();
                    store(offset, &binding, sizeof(binding));
                    break;
                }
                case Tag::ACCEL: {
                    // OptiX traversable handles must not be baked into
                    // a CUDA graph (they may be rebuilt between launches).
                    LUISA_WARNING_WITH_LOCATION(
                        "CudaGraphExt: capturing a dispatch with an Accel "
                        "argument is refused (traversable handles must not "
                        "be baked into a CUDA graph).");
                    return false;
                }
            }
        }
        // The trailing launch-size-and-kernel-id slot (uint4), exactly
        // like _launch: kernel id 0 for a single dispatch.
        auto launch_size_offset = allocate_argument(sizeof(uint4));
        auto launch_size_and_kernel_id = make_uint4(dispatch_size, 0u);
        store(launch_size_offset, &launch_size_and_kernel_id, sizeof(launch_size_and_kernel_id));
        return true;
    }

    void visit(ShaderDispatchCommand *dispatch) noexcept override {
        if (!ok) { return; }
        auto *shader = reinterpret_cast<const CUDAShader *>(dispatch->handle());
        if (shader == nullptr ||
            (!shader->is_native() && !shader->is_tile()) ||
            !shader->is_graph_compatible() ||
            shader->requires_printing() ||
            shader->bound_argument_count() != 0u ||
            dispatch->is_indirect() ||
            dispatch->is_multiple_dispatch()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: capturing a dispatch of a shader that is not "
                "a plain single-launch native CUDA kernel (printing, bound "
                "arguments, indirect or multiple dispatch). Refusing the "
                "capture - a graph created from it could not reproduce the "
                "ordinary launch semantics.");
            ok = false;
            return;
        }
        auto *tile_shader = shader->is_tile() ? static_cast<const CUDAShaderTile *>(shader) : nullptr;
        std::array<CUdeviceptr, 31u> tile_pointers{};
        std::array<void *, 31u> tile_parameters{};
        if (tile_shader != nullptr) {
            auto count = tile_shader->parameter_count();
            if (count > tile_pointers.size() ||
                !tile_shader->encode_buffer_pointers(dispatch->arguments(), {tile_pointers.data(), count})) {
                LUISA_WARNING_WITH_LOCATION("CudaGraphExt: invalid direct-buffer Tile arguments.");
                ok = false;
                return;
            }
            for (auto i = size_t{0u}; i < count; i++) { tile_parameters[i] = &tile_pointers[i]; }
        }
        auto dispatch_size = dispatch->dispatch_size();
        if (any(dispatch_size == make_uint3(0u))) [[unlikely]] {
            // The normal path ignores empty launches; silently turning
            // one into a missing graph node would change what the
            // command list captures, so refuse instead.
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: empty (zero-sized) dispatch cannot be "
                "captured into a graph. Refusing the capture.");
            ok = false;
            return;
        }

        // Track every argument by its declared usage (the same
        // contract as the command reorder pass): a declared READ is
        // trusted read-only (concurrent reads merge, no edge), a
        // declared WRITE is exclusive over its range (RAW/WAW/WAR
        // edges to the exact earlier commands only).
        auto self = static_cast<uint32_t>(nodes.size());
        deps.clear();
        size_t arg_idx = 0u;
        auto max_level = std::numeric_limits<uint64_t>::max();
        for (auto &&arg : dispatch->arguments()) {
            using Tag = ShaderDispatchCommand::Argument::Tag;
            switch (arg.tag) {
                case Tag::BUFFER: {
                    auto is_write = (static_cast<uint>(shader->argument_usage(arg_idx)) &
                                     static_cast<uint>(Usage::WRITE)) != 0u;
                    buffer_states[arg.buffer.handle].apply_access(
                        arg.buffer.offset,
                        arg.buffer.offset + arg.buffer.size,
                        is_write, self, deps);
                } break;
                case Tag::TEXTURE: {
                    auto usage = shader->argument_usage(arg_idx);
                    auto bits = static_cast<uint>(usage);
                    auto writes = (bits & static_cast<uint>(Usage::WRITE)) != 0u;
                    auto reads = (bits & static_cast<uint>(Usage::READ)) != 0u ||
                                 usage == Usage::NONE;
                    LUISA_ASSERT(reads || writes,
                                 "Texture argument has an invalid empty usage mask.");
                    // Sampling may select any mip from the bound base
                    // level onward; storage access targets only the
                    // explicitly bound base mip.
                    if (reads) {
                        texture_states[arg.texture.handle].apply_access(
                            arg.texture.level, max_level, false, self, deps);
                    }
                    if (writes) {
                        texture_states[arg.texture.handle].apply_access(
                            arg.texture.level, arg.texture.level + 1u, true, self, deps);
                    }
                } break;
                case Tag::UNIFORM: break;
                case Tag::BINDLESS_ARRAY: {
                    // Track the descriptor array object itself; the
                    // snapshot resources it points to are not
                    // traversed here (same coverage as the previous
                    // capture implementation).
                    auto is_write = (static_cast<uint>(shader->argument_usage(arg_idx)) &
                                     static_cast<uint>(Usage::WRITE)) != 0u;
                    bindless_states[arg.bindless_array.handle].apply_access(
                        0u, max_level, is_write, self, deps);
                } break;
                case Tag::ACCEL: break;// refused by pack_kernel_arguments below
            }
            ++arg_idx;
        }

        luisa::vector<std::byte> argument_buffer;
        if (tile_shader == nullptr && !pack_kernel_arguments(dispatch, shader, argument_buffer)) {
            ok = false;
            return;
        }
        auto func = static_cast<CUfunction>(shader->handle());
        auto block_size = shader->block_size();
        // The launch configuration _launch computes for this dispatch.
        auto blocks = (dispatch_size + block_size - 1u) / block_size;
        if (tile_shader != nullptr) {
            auto launch = tile_shader->select_launch({tile_pointers.data(), tile_shader->parameter_count()});
            func = launch.function;
            block_size = launch.block;
            blocks = make_uint3(launch.grid[0], launch.grid[1], launch.grid[2]);
        }
        // The kernel takes the packed Params struct by value, so the
        // single kernelParams entry must point AT the packed buffer
        // (exactly like the ordinary launch path's `&arguments`,
        // whose first array element is the buffer address).
        CUDA_KERNEL_NODE_PARAMS params{};
        params.func = func;
        params.gridDimX = blocks.x;
        params.gridDimY = blocks.y;
        params.gridDimZ = blocks.z;
        params.blockDimX = block_size.x;
        params.blockDimY = block_size.y;
        params.blockDimZ = block_size.z;
        params.sharedMemBytes = 0u;
        void *kernel_params[1] = {static_cast<void *>(argument_buffer.data())};
        params.kernelParams = tile_shader == nullptr ? kernel_params : tile_parameters.data();
        params.extra = nullptr;
        CUgraphNode node = nullptr;
        auto dependencies = finish_deps(self);
        if (auto err = cuGraphAddKernelNode(&node, graph,
                                            dependencies.data(), dependencies.size(),
                                            &params);
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphAddKernelNode", err);
            ok = false;
            return;
        }
        if (tile_shader != nullptr) { tile_nodes.emplace_back(self); }
        push_node(self, node, CudaGraphExtImpl::GraphNodeKind::KERNEL);
    }
};

/// Map a driver update result onto the extension's diagnostic enum. The
/// result codes are matched by their (ABI-frozen) numeric value rather than by
/// enumerator name so that this file keeps compiling on toolkits that predate
/// the newest of them.
[[nodiscard]] CudaGraphExt::GraphUpdateErrorType
to_graph_update_error_type(CUgraphExecUpdateResult result) noexcept {
    switch (static_cast<uint32_t>(result)) {
        case 0x0u: return CudaGraphExt::GraphUpdateErrorType::NONE;             // CU_GRAPH_EXEC_UPDATE_SUCCESS
        case 0x2u: return CudaGraphExt::GraphUpdateErrorType::TOPOLOGY_CHANGED; // CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED
        case 0x3u: return CudaGraphExt::GraphUpdateErrorType::NODE_TYPE_CHANGED;// CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED
        // CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED: a kernel node's
        // function changed, i.e. the node is a different node.
        case 0x4u: return CudaGraphExt::GraphUpdateErrorType::NODE_TYPE_CHANGED;
        case 0x5u: return CudaGraphExt::GraphUpdateErrorType::PARAMETERS_CHANGED;// CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED
        case 0x6u: return CudaGraphExt::GraphUpdateErrorType::UNSUPPORTED;        // CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED
        case 0x7u: return CudaGraphExt::GraphUpdateErrorType::NODE_TYPE_CHANGED;  // CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE
        case 0x8u: return CudaGraphExt::GraphUpdateErrorType::NODE_ATTR_CHANGED;  // CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED
        default: return CudaGraphExt::GraphUpdateErrorType::INTERNAL;             // CU_GRAPH_EXEC_UPDATE_ERROR
    }
}

[[nodiscard]] const char *to_graph_update_result_name(CUgraphExecUpdateResult result) noexcept {
    switch (static_cast<uint32_t>(result)) {
        case 0x0u: return "success";
        case 0x1u: return "CU_GRAPH_EXEC_UPDATE_ERROR";
        case 0x2u: return "CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED";
        case 0x3u: return "CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED";
        case 0x4u: return "CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED";
        case 0x5u: return "CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED";
        case 0x6u: return "CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED";
        case 0x7u: return "CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE";
        case 0x8u: return "CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED";
        default: return "unknown CUgraphExecUpdateResult";
    }
}

}  // namespace cuda_graph_ext_detail
}// namespace

CudaGraphExtImpl::CudaGraphExtImpl(CUDADevice *device) noexcept
    : _device{device} {}

CudaGraphExtImpl::~CudaGraphExtImpl() noexcept {
    // Executable graphs are independent copies and are released first; the
    // templates (including conditional body graphs, which are ordinary graphs
    // in `_graph_data`) follow. Child graph templates are referenced by the
    // parent by handle only - no ownership, the caller guarantees the child
    // outlives the parent.
    for (auto &[exec_handle, data] : _exec_data) {
        cuGraphExecDestroy(reinterpret_cast<CUgraphExec>(exec_handle));
    }
    for (auto &[graph_handle, data] : _graph_data) {
        cuGraphDestroy(reinterpret_cast<CUgraph>(graph_handle));
    }
}

CUgraphNode CudaGraphExtImpl::_get_node(GraphExecHandle exec, size_t node_index) const noexcept {
    auto it = _exec_data.find(exec);
    if (it == _exec_data.end()) { return nullptr; }
    auto git = _graph_data.find(it->second.graph_handle);
    if (git == _graph_data.end() || node_index >= git->second.nodes.size()) { return nullptr; }
    return git->second.nodes[node_index].node;
}

bool CudaGraphExtImpl::_device_supports_graph_memory() noexcept {
#if CUDA_VERSION >= 11040
    if (_memory_pools_supported < 0) {
        // A device that does not support the memory-pool family cannot satisfy
        // `cuGraphAddMemAllocNode` either: graph memory nodes allocate from a
        // pool. Probe once and keep the answer (a driver failure is not cached,
        // so a transient context problem can be retried).
        int supported = 0;
        auto err = cuDeviceGetAttribute(&supported,
                                        CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                                        _device->handle().device());
        if (err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail(
                "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)", err);
            return false;
        }
        _memory_pools_supported = supported != 0 ? 1 : 0;
    }
    return _memory_pools_supported != 0;
#else
    // Graph memory nodes only exist from CUDA 11.4 on.
    return false;
#endif
}

size_t CudaGraphExtImpl::_live_exec_count(GraphHandle graph) const noexcept {
    auto count = size_t{0u};
    for (auto &&[exec_handle, data] : _exec_data) {
        if (data.graph_handle == graph) { count++; }
    }
    return count;
}

ResourceCreationInfo CudaGraphExtImpl::_create_graph(CommandList &&cmdlist) noexcept {
    if (cmdlist.empty()) { return {invalid_handle, nullptr}; }

    return _device->with_handle([&]() -> ResourceCreationInfo {
        CUgraph graph = nullptr;
        if (auto err = cuGraphCreate(&graph, 0u); err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphCreate", err);
            return {invalid_handle, nullptr};
        }

        // Both the top-level graph and every conditional body graph are built
        // by the same pipeline (staging + usage-based DAG analysis).
        GraphData data;
        if (!_build_graph_from_commands(graph, std::move(cmdlist), data)) {
            cuGraphDestroy(graph);
            return {invalid_handle, nullptr};
        }

        auto graph_handle = reinterpret_cast<uint64_t>(graph);
        {
            std::scoped_lock lock{_mutex};
            _graph_data.emplace(graph_handle, std::move(data));
        }

        return {graph_handle, graph};
    });
}

bool CudaGraphExtImpl::_build_graph_from_commands(CUgraph graph, CommandList &&cmdlist,
                                                  GraphData &data) noexcept {
    auto commands = cmdlist.steal_commands();
    auto user_callbacks = cmdlist.steal_callbacks();

    // ---- phase 1: staging + refusal of unsupported content ----
    // The pinned blocks are moved into `data` before the DAG is built so that
    // a partially built graph never points at freed host memory.
    {
        luisa::vector<luisa::shared_ptr<PinnedHostBlock>> host_allocs;
        luisa::vector<CudaGraphHostCopyData> host_copies;
        cuda_graph_ext_detail::UploadStagingVisitor staging_visitor{host_allocs, host_copies};
        for (auto &cmd : commands) {
            cmd->accept(staging_visitor);
            if (!staging_visitor.ok) { break; }
        }
        if (!staging_visitor.ok) { return false; }
        data.host_allocations.insert(data.host_allocations.end(),
                                     std::make_move_iterator(host_allocs.begin()),
                                     std::make_move_iterator(host_allocs.end()));
        data.host_copies.insert(data.host_copies.end(),
                                std::make_move_iterator(host_copies.begin()),
                                std::make_move_iterator(host_copies.end()));
    }

    // ---- phase 2: dependency analysis + explicit DAG construction ----
    // Instead of stream capture, which linearizes every command onto one
    // stream and replays as a serial chain, `GraphDagVisitor` tracks every
    // argument by its declared usage (like the command reorder pass) and
    // builds an explicit CUDA graph whose edges are only the real RAW/WAW/WAR
    // command -> command dependencies. The node list is the creation order of
    // the node-producing commands, which is deterministic and matches the
    // order the update_* node APIs expect.
    cuda_graph_ext_detail::GraphDagVisitor visitor{
        graph, cuda_graph_ext_detail::current_context(), data.nodes};
    for (auto &cmd : commands) {
        cmd->accept(visitor);
        if (!visitor.ok) { break; }
    }
    if (!visitor.ok) {
        // Keep the records of the nodes that were created; the caller either
        // destroys the whole graph (`_create_graph`) or rolls the body nodes
        // back (`build_body_graph`).
        return false;
    }

    for (auto &cb : user_callbacks) { cb(); }

    data.tile_nodes.insert(data.tile_nodes.end(),
                           visitor.tile_nodes.begin(), visitor.tile_nodes.end());
    return true;
}

bool CudaGraphExtImpl::_find_conditional_body(GraphHandle body, GraphHandle *parent,
                                              size_t *conditional_index,
                                              size_t *body_slot) const noexcept {
    if (body == invalid_handle) { return false; }
    for (auto &&[graph_handle, data] : _graph_data) {
        for (auto i = size_t{0u}; i < data.conditional_nodes.size(); i++) {
            auto &record = data.conditional_nodes[i];
            for (auto slot = size_t{0u}; slot < record.bodies.size(); slot++) {
                if (record.bodies[slot] == body) {
                    if (parent != nullptr) { *parent = graph_handle; }
                    if (conditional_index != nullptr) { *conditional_index = i; }
                    if (body_slot != nullptr) { *body_slot = slot; }
                    return true;
                }
            }
        }
    }
    return false;
}

void CudaGraphExtImpl::destroy_graph(GraphHandle graph) noexcept {
    if (graph == invalid_handle) { return; }
    _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        // Conditional body graphs are owned by (and released with) the
        // conditional node that created them: destroying one directly would
        // corrupt the parent graph, so refuse loudly per the contract.
        if (_find_conditional_body(graph, nullptr, nullptr, nullptr)) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: destroy_graph was called on a conditional body "
                "graph. Body graphs are owned by their conditional node and "
                "must not be destroyed directly; destroy the parent graph "
                "instead. The request is ignored.");
            return;
        }
        _graph_data.erase(graph);
        cuGraphDestroy(reinterpret_cast<CUgraph>(graph));
    });
}

ResourceCreationInfo CudaGraphExtImpl::_instantiate(GraphHandle graph, InstantiateFlag flags) noexcept {
    if (graph == invalid_handle) { return {invalid_handle, nullptr}; }
    return _device->with_handle([&]() -> ResourceCreationInfo {
        auto flag_bits = static_cast<uint32_t>(flags);
        // Composition rules (research doc 3.3/3.4), enforced before the driver
        // call so the caller gets the real reason instead of a generic driver
        // error.
        if ((flag_bits & static_cast<uint32_t>(InstantiateFlag::DEVICE_LAUNCH)) != 0u &&
            (flag_bits & static_cast<uint32_t>(InstantiateFlag::AUTO_FREE_ON_LAUNCH)) != 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: InstantiateFlag::DEVICE_LAUNCH and "
                "InstantiateFlag::AUTO_FREE_ON_LAUNCH are mutually exclusive "
                "(the driver rejects the combination); no executable graph was "
                "created.");
            return {invalid_handle, nullptr};
        }
        {
            std::scoped_lock lock{_mutex};
            if (auto git = _graph_data.find(graph); git != _graph_data.end()) {
                if (git->second.has_device_updatable && git->second.device_updatable_instantiated) {
                    // CUDA: a graph containing one or more device-updatable
                    // nodes does not allow multiple instantiation - the device
                    // node handles are bound to the (single) executable graph.
                    LUISA_WARNING_WITH_LOCATION(
                        "CudaGraphExt: the graph contains device-updatable "
                        "kernel nodes and has already been instantiated; CUDA "
                        "does not allow multiple instantiation of such a "
                        "graph. No second executable graph was created.");
                    return {invalid_handle, nullptr};
                }
                if (git->second.has_mem_nodes && _live_exec_count(graph) != 0u) {
                    // CUDA: "Only one instantiation of the graph may exist at
                    // any point in time" for a graph that contains allocation
                    // and/or free nodes. Unlike the device-updatable case this
                    // restriction is *not* latched: destroying the live exec
                    // releases it again (the address of an allocation is fixed
                    // across instantiations, so the next exec sees the same
                    // one).
                    LUISA_WARNING_WITH_LOCATION(
                        "CudaGraphExt: the graph contains graph memory "
                        "(alloc/free) nodes and already has a live executable "
                        "graph; CUDA allows only one instantiation of such a "
                        "graph at a time. Destroy the live exec first. No "
                        "second executable graph was created.");
                    return {invalid_handle, nullptr};
                }
            }
        }
        CUgraphExec exec = nullptr;
        auto ret = cuGraphInstantiateWithFlags(&exec, reinterpret_cast<CUgraph>(graph),
                                               cuda_graph_ext_detail::to_instantiate_flags(flags));
        if (ret == CUDA_ERROR_UNKNOWN) {
            // Driver robustness: on some stacks (observed on WSL2 + Blackwell,
            // driver 617.42, with conditional-node graphs) the first
            // instantiate of an otherwise-valid graph intermittently fails
            // with CUDA_ERROR_UNKNOWN; the identical call then succeeds.
            // Nothing leaks on the failed attempt (exec stays null), so retry
            // exactly once before giving up.
            LUISA_WARNING_WITH_LOCATION(
                "cuGraphInstantiateWithFlags returned CUDA_ERROR_UNKNOWN; retrying once.");
            if (exec != nullptr) { cuGraphExecDestroy(exec); exec = nullptr; }
            ret = cuGraphInstantiateWithFlags(&exec, reinterpret_cast<CUgraph>(graph),
                                              cuda_graph_ext_detail::to_instantiate_flags(flags));
        }
        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuGraphInstantiateWithFlags failed: {}", err_name ? err_name : "unknown");
            return {invalid_handle, nullptr};
        }
        auto exec_handle = reinterpret_cast<uint64_t>(exec);
        {
            std::scoped_lock lock{_mutex};
            // Seed the executable with the graph's staging state: the pinned
            // blocks are shared (a graph may spawn several executable
            // graphs), the host-copy records are plain structs pointing into
            // them. update() replaces both with the new graph's state after
            // a successful cuGraphExecUpdate. The instantiation flags travel
            // with the exec: `get_device_launch_token` only hands out a token
            // for an exec that was instantiated for device launch.
            auto git = _graph_data.find(graph);
            ExecData exec_data{graph, {}, {}, {}};
            if (git != _graph_data.end()) {
                exec_data.host_allocations = git->second.host_allocations;
                exec_data.host_copies = git->second.host_copies;
                exec_data.tile_nodes = git->second.tile_nodes;
                // Latch the frozen-topology bookkeeping: from here on the
                // template is instantiated (no more device-updatable opt-ins)
                // and, when it carries device-updatable nodes, it may never be
                // instantiated again.
                git->second.instantiated = true;
                if (git->second.has_device_updatable) {
                    git->second.device_updatable_instantiated = true;
                }
            }
            exec_data.instantiate_flags = flags;
            _exec_data.emplace(exec_handle, std::move(exec_data));
        }
        return {exec_handle, exec};
    });
}

void CudaGraphExtImpl::destroy_exec(GraphExecHandle exec) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        {
            std::scoped_lock lock{_mutex};
            // The `device_updatable_instantiated` marker of the owning graph is
            // deliberately *not* cleared here: destroying the exec invalidates
            // the device node handles that were handed out from it, so the
            // fail-closed behaviour (the template keeps refusing another
            // instantiation) is the safe reading of CUDA's "no multiple
            // instantiation" rule for a graph with device-updatable nodes.
            _exec_data.erase(exec);
        }
        cuGraphExecDestroy(reinterpret_cast<CUgraphExec>(exec));
    });
}

void CudaGraphExtImpl::launch(GraphExecHandle exec, uint64_t stream_handle) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        auto stream = cuda_graph_ext_detail::to_cu_stream(stream_handle);
        auto ret = cuGraphLaunch(reinterpret_cast<CUgraphExec>(exec), stream);
        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuGraphLaunch failed: {}", err_name ? err_name : "unknown");
            return;
        }
        // Download nodes write into pinned staging buffers; forward the
        // payloads to the user-provided host buffers with stream-ordered
        // host callbacks, so they run after the graph completes on this
        // stream. (Explicit host nodes added with `add_host_node` are
        // independent of this forwarding and remain additive.)
        std::scoped_lock lock{_mutex};
        if (auto it = _exec_data.find(exec); it != _exec_data.end()) {
            for (auto &host_copy : it->second.host_copies) {
                if (auto err = cuLaunchHostFunc(stream, cuda_graph_ext_detail::cuda_graph_host_copy_callback, &host_copy);
                    err != CUDA_SUCCESS) {
                    const char *err_name = nullptr;
                    cuGetErrorName(err, &err_name);
                    LUISA_WARNING_WITH_LOCATION(
                        "CudaGraphExt: cuLaunchHostFunc for a graph download failed: {}",
                        err_name ? err_name : "unknown");
                }
            }
        }
    });
}

void CudaGraphExtImpl::upload(GraphExecHandle exec, uint64_t stream_handle) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        // `cuGraphUpload` is legal for every instantiated exec, and it is not
        // only the device-launch path that needs it: CUDA also requires a graph
        // that carries device-updatable kernel nodes to be uploaded before it
        // is launched, because that is what materialises the device-resident
        // parameter block a device-side update patches.
        auto ret = cuGraphUpload(reinterpret_cast<CUgraphExec>(exec), cuda_graph_ext_detail::to_cu_stream(stream_handle));
        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuGraphUpload failed: {}", err_name ? err_name : "unknown");
        }
    });
}

CudaGraphExt::Capabilities CudaGraphExtImpl::capabilities() const noexcept {
    Capabilities caps{};
    caps.cuda_version = CUDA_VERSION;
    // The installed driver may be older than the toolkit the code was
    // compiled with; a feature is only usable when both sides provide it.
    auto runtime_version = static_cast<uint32_t>(CUDA_VERSION);
    int driver_version = 0;
    if (cuDriverGetVersion(&driver_version) == CUDA_SUCCESS && driver_version > 0) {
        runtime_version = std::min(runtime_version, static_cast<uint32_t>(driver_version));
    }
    // The annotated-edge API additionally needs 12.4 headers (compile-time
    // `#if defined(cuGraphAddDependencies)` gate in set_dependency_type); the
    // runtime check keeps the capability honest for mixed toolkit/driver pairs.
    caps.programmatic_edges = runtime_version >= 12040u;
    // Device graph launch and device-updatable kernel nodes additionally
    // require Hopper or newer (compute capability >= 9.0).
    auto cc = _device->handle().compute_capability();
    auto hopper_or_newer = cc >= 90u;
    caps.device_launch = runtime_version >= 12000u && hopper_or_newer;
    caps.conditional_nodes = runtime_version >= 12030u;
    caps.device_updatable_nodes = runtime_version >= 12040u && hopper_or_newer;
    caps.graph_memory = runtime_version >= 11040u;
    caps.node_priority = runtime_version >= 12080u;
    caps.switch_conditional = runtime_version >= 12080u;
    return caps;
}

bool CudaGraphExtImpl::update(GraphExecHandle exec, CommandList &&cmdlist,
                              GraphUpdateError *error) noexcept {
    auto fail = [error](GraphUpdateErrorType type, uint64_t error_node,
                        luisa::string detail) noexcept {
        if (error != nullptr) {
            error->type = type;
            error->error_node = error_node;
            error->detail = std::move(detail);
        }
        return false;
    };
    if (error != nullptr) { *error = GraphUpdateError{}; }
    if (exec == invalid_handle) {
        return fail(GraphUpdateErrorType::INTERNAL, ~0ull,
                    "invalid executable graph handle");
    }
    if (cmdlist.empty()) {
        return fail(GraphUpdateErrorType::TOPOLOGY_CHANGED, ~0ull,
                    "the new command list is empty (node count 0)");
    }
    return _device->with_handle([&]() -> bool {
        {
            // Frozen-topology pre-check: CUDA freezes the topology of graphs
            // containing conditional nodes at instantiation, and a graph with
            // device-updatable nodes cannot be updated as a whole either.
            // Refuse before calling the driver so the caller gets the real
            // reason instead of a generic driver error.
            std::scoped_lock lock{_mutex};
            if (auto exec_it = _exec_data.find(exec); exec_it != _exec_data.end()) {
                if (auto graph_it = _graph_data.find(exec_it->second.graph_handle);
                    graph_it != _graph_data.end()) {
                    if (graph_it->second.has_conditional) {
                        return fail(GraphUpdateErrorType::UNSUPPORTED, ~0ull,
                                    "the executable graph was built from a graph "
                                    "containing conditional nodes; its topology is "
                                    "frozen at instantiation");
                    }
                    if (graph_it->second.has_device_updatable) {
                        return fail(GraphUpdateErrorType::UNSUPPORTED, ~0ull,
                                    "the executable graph contains device-updatable "
                                    "kernel nodes; whole-graph update is disabled");
                    }
                }
            }
        }

        auto new_graph_info = _create_graph(std::move(cmdlist));
        if (new_graph_info.handle == invalid_handle) {
            return fail(GraphUpdateErrorType::INTERNAL, ~0ull,
                        "the new command list could not be captured into a graph "
                        "(unsupported content or a driver failure; see the warnings)");
        }

        CUgraphExecUpdateResultInfo result_info{};
        auto ret = cuGraphExecUpdate(reinterpret_cast<CUgraphExec>(exec),
                                     reinterpret_cast<CUgraph>(new_graph_info.handle),
                                     &result_info);

        auto updated = ret == CUDA_SUCCESS;
        if (!updated && error != nullptr) {
            // The driver reports a rejected update by returning
            // CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE *and* filling
            // `result_info.result` with the specific reason (verified
            // empirically against driver 13.4: result=TOPOLOGY_CHANGED on a
            // node-count mismatch). Only when `result_info` carries no reason
            // (defensive; e.g. the call failed before an update was attempted)
            // do we fall back to the generic legacy code / the raw CUresult.
            auto result = result_info.result;
            if (result == CU_GRAPH_EXEC_UPDATE_SUCCESS && ret != CUDA_SUCCESS) {
                result = CU_GRAPH_EXEC_UPDATE_ERROR;
            }
            error->type = cuda_graph_ext_detail::to_graph_update_error_type(result);
            // The driver reports the offending node of the *new* graph; map it
            // back to its creation index so the caller can act on it.
            auto error_node = result_info.errorNode != nullptr
                                  ? result_info.errorNode
                                  : result_info.errorFromNode;
            error->error_node = ~0ull;
            if (error_node != nullptr) {
                std::scoped_lock lock{_mutex};
                if (auto it = _graph_data.find(new_graph_info.handle);
                    it != _graph_data.end()) {
                    auto &nodes = it->second.nodes;
                    for (auto i = size_t{0u}; i < nodes.size(); i++) {
                        if (nodes[i].node == error_node) {
                            error->error_node = i;
                            break;
                        }
                    }
                }
            }
            const char *err_name = nullptr;
            if (ret != CUDA_SUCCESS) { cuGetErrorName(ret, &err_name); }
            error->detail = luisa::format(
                "cuGraphExecUpdate failed: {}{}{}",
                cuda_graph_ext_detail::to_graph_update_result_name(result),
                ret == CUDA_SUCCESS ? "" : " (driver error: ",
                ret == CUDA_SUCCESS ? "" : (err_name != nullptr ? err_name : "unknown"));
            if (ret != CUDA_SUCCESS) { error->detail.append(")"); }
        }

        {
            std::scoped_lock lock{_mutex};
            auto new_graph_it = _graph_data.find(new_graph_info.handle);
            if (new_graph_it != _graph_data.end()) {
                if (updated) {
                    auto exec_it = _exec_data.find(exec);
                    if (exec_it != _exec_data.end()) {
                        // The stale staging state is released here (shared
                        // pointers); the new graph's state takes over.
                        exec_it->second.host_allocations = std::move(new_graph_it->second.host_allocations);
                        exec_it->second.host_copies = std::move(new_graph_it->second.host_copies);
                        exec_it->second.tile_nodes = std::move(new_graph_it->second.tile_nodes);
                    } else {
                        updated = false;
                    }
                }
                _graph_data.erase(new_graph_it);
            }
        }
        cuGraphDestroy(reinterpret_cast<CUgraph>(new_graph_info.handle));
        return updated;
    });
}

  void CudaGraphExtImpl::set_node_priority(GraphExecHandle exec, size_t node_index,
                                           int priority) noexcept {
      // The `CU_KERNEL_NODE_ATTRIBUTE_PRIORITY` spelling exists on every
      // supported toolkit, so a `defined()` test would be a no-op; gate on the
      // version that made priority valid for graph kernel nodes instead.
  #if CUDA_VERSION >= 12080
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (node == nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: set_node_priority on unknown node index {}.",
                node_index);
            return;
        }
        CUgraphNodeType node_type{};
        if (cuGraphNodeGetType(node, &node_type) == CUDA_SUCCESS &&
            node_type != CU_GRAPH_NODE_TYPE_KERNEL) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: set_node_priority requires a kernel node, but "
                "node index {} is of type {}.",
                node_index, static_cast<int>(node_type));
            return;
        }
        // The driver exposes the priority as a kernel-node attribute of the
        // graph template node (there is no exec-scoped setter: the API is
        // `cuGraphKernelNodeSetAttribute`, not `cuGraphExecNodeSetPriority`).
        // The attribute is baked into an executable graph at instantiation,
        // so an already-instantiated exec picks the new priority up on its
        // next instantiation.
        CUkernelNodeAttrValue value{};
        value.priority = priority;
        if (auto ret = cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &value);
            ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: cuGraphKernelNodeSetAttribute(priority) failed: {}",
                err_name ? err_name : "unknown");
        }
    });
#else
    // No node-priority attribute before CUDA 12.8 (and no driver API at all:
    // priority is exposed as `CU_LAUNCH_ATTRIBUTE_PRIORITY`, valid for graph
    // nodes only from 12.8 on).
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: set_node_priority requires CUDA >= 12.8 (toolkit is "
        "{}). The request is ignored.",
        CUDA_VERSION);
    (void)exec;
    (void)node_index;
    (void)priority;
#endif
}

bool CudaGraphExtImpl::update_kernel_node(GraphExecHandle exec, size_t node_index,
                                          uint3 dispatch_size,
                                          luisa::span<const Argument> arguments) noexcept {
    if (exec == invalid_handle) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        if (auto it = _exec_data.find(exec); it != _exec_data.end() &&
            std::find(it->second.tile_nodes.begin(), it->second.tile_nodes.end(), node_index) != it->second.tile_nodes.end()) {
            // This legacy API passes opaque Arguments rather than a typed Tile
            // invocation. Rebuild via update(CommandList) to retain ABI order,
            // static launch dimensions, views, and dependency validation.
            return false;
        }
        auto *node = _get_node(exec, node_index);
        if (!node) { return false; }

        CUDA_KERNEL_NODE_PARAMS params{};
        if (cuGraphKernelNodeGetParams(node, &params) != CUDA_SUCCESS) { return false; }

        luisa::vector<void *> kernel_args;
        for (const auto &arg : arguments) {
            kernel_args.push_back(const_cast<void *>(static_cast<const void *>(&arg)));
        }

        params.gridDimX = dispatch_size.x;
        params.gridDimY = dispatch_size.y;
        params.gridDimZ = dispatch_size.z;
        params.kernelParams = kernel_args.data();

        return cuGraphExecKernelNodeSetParams(
                   reinterpret_cast<CUgraphExec>(exec), node, &params) == CUDA_SUCCESS;
    });
}

bool CudaGraphExtImpl::update_upload_node(GraphExecHandle exec, size_t node_index,
                                          const void *data, size_t size) noexcept {
    if (exec == invalid_handle || !data || size == 0) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return false; }

        CUDA_MEMCPY3D params{};
        if (cuGraphMemcpyNodeGetParams(node, &params) != CUDA_SUCCESS) { return false; }

        params.srcMemoryType = CU_MEMORYTYPE_HOST;
        params.srcHost = data;
        params.srcPitch = size;
        params.srcHeight = 1;
        params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        // dstDevice is preserved from the original node params
        params.dstPitch = size;
        params.dstHeight = 1;
        params.WidthInBytes = size;
        params.Height = 1;
        params.Depth = 1;

        return cuGraphExecMemcpyNodeSetParams(
                   reinterpret_cast<CUgraphExec>(exec), node, &params, nullptr) == CUDA_SUCCESS;
    });
}

bool CudaGraphExtImpl::update_download_node(GraphExecHandle exec, size_t node_index,
                                            void *data, size_t size) noexcept {
    if (exec == invalid_handle || !data || size == 0) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return false; }

        CUDA_MEMCPY3D params{};
        if (cuGraphMemcpyNodeGetParams(node, &params) != CUDA_SUCCESS) { return false; }

        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        // srcDevice is preserved from the original node params
        params.srcPitch = size;
        params.srcHeight = 1;
        params.dstMemoryType = CU_MEMORYTYPE_HOST;
        params.dstHost = data;
        params.dstPitch = size;
        params.dstHeight = 1;
        params.WidthInBytes = size;
        params.Height = 1;
        params.Depth = 1;

        return cuGraphExecMemcpyNodeSetParams(
                   reinterpret_cast<CUgraphExec>(exec), node, &params, nullptr) == CUDA_SUCCESS;
    });
}

bool CudaGraphExtImpl::update_buffer_copy_node(GraphExecHandle exec, size_t node_index,
                                               uint64_t src_handle, size_t src_offset,
                                               uint64_t dst_handle, size_t dst_offset,
                                               size_t size) noexcept {
    if (exec == invalid_handle) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return false; }

        auto src_addr = reinterpret_cast<const CUDABuffer *>(src_handle)->device_address() + src_offset;
        auto dst_addr = reinterpret_cast<const CUDABuffer *>(dst_handle)->device_address() + dst_offset;

        CUDA_MEMCPY3D params{};
        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        params.srcDevice = static_cast<CUdeviceptr>(src_addr);
        params.srcPitch = size;
        params.srcHeight = 1;
        params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        params.dstDevice = static_cast<CUdeviceptr>(dst_addr);
        params.dstPitch = size;
        params.dstHeight = 1;
        params.WidthInBytes = size;
        params.Height = 1;
        params.Depth = 1;

        return cuGraphExecMemcpyNodeSetParams(
                   reinterpret_cast<CUgraphExec>(exec), node, &params, nullptr) == CUDA_SUCCESS;
    });
}

void CudaGraphExtImpl::set_node_enabled(GraphExecHandle exec, size_t node_index, bool enabled) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = _get_node(exec, node_index);
        if (!node) { return; }
        auto ret = cuGraphNodeSetEnabled(reinterpret_cast<CUgraphExec>(exec), node,
                                         enabled ? 1u : 0u);
        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuGraphNodeSetEnabled failed: {}", err_name ? err_name : "unknown");
        }
    });
}

bool CudaGraphExtImpl::is_node_enabled(GraphExecHandle exec, size_t node_index) const noexcept {
    if (exec == invalid_handle) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto *node = const_cast<CudaGraphExtImpl *>(this)->_get_node(exec, node_index);
        if (!node) { return false; }
        unsigned int is_enabled = 0u;
        auto ret = cuGraphNodeGetEnabled(reinterpret_cast<CUgraphExec>(exec), node, &is_enabled);
        return ret == CUDA_SUCCESS && is_enabled != 0u;
    });
}

// ---------------------------------------------------------------------------
// G6: additional node types (explicit graph construction)
//
// Each of these appends exactly one node to the graph's node list, so the
// node's creation index stays the index every node API takes. No dependency
// edge is attached: wire the node up with `add_dependency`.
// ---------------------------------------------------------------------------

bool CudaGraphExtImpl::add_memset_node(GraphHandle graph, uint64_t buffer_handle,
                                       size_t offset, size_t size, uint32_t value,
                                       size_t *node_index) noexcept {
    if (graph == invalid_handle || buffer_handle == 0u || size == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_memset_node needs a valid graph, a valid buffer handle and a non-zero size; no memset node was added.");
        return false;
    }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto graph_it = _graph_data.find(graph);
        if (graph_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION("CudaGraphExt: add_memset_node on an unknown graph.");
            return false;
        }
        auto *buffer = reinterpret_cast<const CUDABuffer *>(buffer_handle);
        auto buffer_size = buffer->size_bytes();
        if (offset > buffer_size || buffer_size - offset < size) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: add_memset_node range [{}, {}) exceeds the "
                "{}-byte buffer.",
                offset, offset + size, buffer_size);
            return false;
        }
        CUDA_MEMSET_NODE_PARAMS params{};
        params.dst = static_cast<CUdeviceptr>(buffer->device_address() + offset);
        params.pitch = size;
        params.value = value;
        params.elementSize = 1u;// byte-wise fill
        params.width = size;
        params.height = 1u;
        CUgraphNode node = nullptr;
        if (auto err = cuGraphAddMemsetNode(&node, reinterpret_cast<CUgraph>(graph),
                                            nullptr, 0u, &params,
                                            cuda_graph_ext_detail::current_context());
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphAddMemsetNode", err);
            return false;
        }
        auto index = graph_it->second.nodes.size();
        graph_it->second.nodes.push_back(GraphNodeRecord{.node = node, .kind = GraphNodeKind::MEMSET});
        if (node_index != nullptr) { *node_index = index; }
        return true;
    });
}

bool CudaGraphExtImpl::add_host_node(GraphHandle graph, HostNodeCallback callback,
                                     void *user_data, size_t *node_index) noexcept {
    if (graph == invalid_handle || callback == nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_host_node needs a valid graph and a non-null callback; no host node was added.");
        return false;
    }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto graph_it = _graph_data.find(graph);
        if (graph_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION("CudaGraphExt: add_host_node on an unknown graph.");
            return false;
        }
        CUDA_HOST_NODE_PARAMS params{};
        // `HostNodeCallback` is a plain function pointer; the driver's
        // `CUhostFn` is the same signature with `CUDA_CB`, which is empty on
        // the platforms this backend supports (x86-64 SysV / Win64).
        params.fn = reinterpret_cast<CUhostFn>(callback);
        params.userData = user_data;
        CUgraphNode node = nullptr;
        if (auto err = cuGraphAddHostNode(&node, reinterpret_cast<CUgraph>(graph),
                                          nullptr, 0u, &params);
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphAddHostNode", err);
            return false;
        }
        auto index = graph_it->second.nodes.size();
        graph_it->second.nodes.push_back(GraphNodeRecord{.node = node, .kind = GraphNodeKind::HOST});
        if (node_index != nullptr) { *node_index = index; }
        return true;
    });
}

// Event record/wait nodes take a raw driver `CUevent` (created with
// `cuEventCreate`, or obtained from another CUDA component) reinterpreted as
// the uint64 handle the extension API uses:
//
//     CUevent event = ...;
//     ext->add_event_record_node(graph, reinterpret_cast<uint64_t>(event), &index);
//
// Note on the LuisaCompute event model: an `Event` created with
// `Device::create_event()` on this backend is *not* a `CUevent` - it is a
// Vulkan timeline semaphore exported as a `CUexternalSemaphore` (see
// `src/backends/cuda/cuda_event.h`), and a timeline value cannot be expressed
// by `cuGraphAddEventRecordNode`'s parameter list. Recording/waiting an LC
// `Event` inside a graph therefore needs the external-semaphore node types and
// an explicit value, which is a separate API surface.
bool CudaGraphExtImpl::add_event_record_node(GraphHandle graph, uint64_t event_handle,
                                             size_t *node_index) noexcept {
    if (graph == invalid_handle || event_handle == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_event_record_node needs a valid graph and a non-null CUevent handle; no event-record node was added.");
        return false;
    }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto graph_it = _graph_data.find(graph);
        if (graph_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION("CudaGraphExt: add_event_record_node on an unknown graph.");
            return false;
        }
        auto event = reinterpret_cast<CUevent>(event_handle);
        CUgraphNode node = nullptr;
        if (auto err = cuGraphAddEventRecordNode(&node, reinterpret_cast<CUgraph>(graph),
                                                 nullptr, 0u, event);
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphAddEventRecordNode", err);
            return false;
        }
        auto index = graph_it->second.nodes.size();
        graph_it->second.nodes.push_back(GraphNodeRecord{.node = node, .kind = GraphNodeKind::EVENT_RECORD});
        if (node_index != nullptr) { *node_index = index; }
        return true;
    });
}

bool CudaGraphExtImpl::add_event_wait_node(GraphHandle graph, uint64_t event_handle,
                                           size_t *node_index) noexcept {
    if (graph == invalid_handle || event_handle == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_event_wait_node needs a valid graph and a non-null CUevent handle; no event-wait node was added.");
        return false;
    }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto graph_it = _graph_data.find(graph);
        if (graph_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION("CudaGraphExt: add_event_wait_node on an unknown graph.");
            return false;
        }
        auto event = reinterpret_cast<CUevent>(event_handle);
        CUgraphNode node = nullptr;
        if (auto err = cuGraphAddEventWaitNode(&node, reinterpret_cast<CUgraph>(graph),
                                               nullptr, 0u, event);
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphAddEventWaitNode", err);
            return false;
        }
        auto index = graph_it->second.nodes.size();
        graph_it->second.nodes.push_back(GraphNodeRecord{.node = node, .kind = GraphNodeKind::EVENT_WAIT});
        if (node_index != nullptr) { *node_index = index; }
        return true;
    });
}

bool CudaGraphExtImpl::add_child_graph_node(GraphHandle graph, GraphHandle child_graph,
                                            size_t *node_index) noexcept {
    if (graph == invalid_handle || child_graph == invalid_handle || graph == child_graph) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_child_graph_node needs two distinct valid graph handles; no child-graph node was added.");
        return false;
    }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto graph_it = _graph_data.find(graph);
        if (graph_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION("CudaGraphExt: add_child_graph_node on an unknown graph.");
            return false;
        }
        if (_graph_data.find(child_graph) == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: add_child_graph_node with an unknown child graph; "
                "the child must be a live graph of this device and must outlive "
                "the parent.");
            return false;
        }
        CUgraphNode node = nullptr;
        if (auto err = cuGraphAddChildGraphNode(&node, reinterpret_cast<CUgraph>(graph),
                                                nullptr, 0u,
                                                reinterpret_cast<CUgraph>(child_graph));
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphAddChildGraphNode", err);
            return false;
        }
        auto index = graph_it->second.nodes.size();
        graph_it->second.nodes.push_back(GraphNodeRecord{.node = node, .kind = GraphNodeKind::CHILD_GRAPH});
        // Reference only (see GraphData::child_graphs): the child must outlive
        // the parent, so the parent owns nothing.
        graph_it->second.child_graphs.emplace_back(child_graph);
        if (node_index != nullptr) { *node_index = index; }
        return true;
    });
}

bool CudaGraphExtImpl::add_dependency(GraphHandle graph, size_t from_index,
                                      size_t to_index) noexcept {
    if (graph == invalid_handle || from_index == to_index) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_dependency needs a valid graph and two distinct node indices; no dependency was added.");
        return false;
    }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        auto graph_it = _graph_data.find(graph);
        if (graph_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION("CudaGraphExt: add_dependency on an unknown graph.");
            return false;
        }
        auto &nodes = graph_it->second.nodes;
        if (from_index >= nodes.size() || to_index >= nodes.size()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: add_dependency indices out of range "
                "({} -> {} of {} nodes).",
                from_index, to_index, nodes.size());
            return false;
        }
        auto from = nodes[from_index].node;
        auto to = nodes[to_index].node;
        if (auto err = cuda_graph_ext_detail::add_graph_dependencies(
                reinterpret_cast<CUgraph>(graph), &from, &to, 1u);
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphAddDependencies", err);
            return false;
        }
        return true;
    });
}

bool CudaGraphExtImpl::set_dependency_type(GraphHandle graph, size_t from_index,
                                           size_t to_index, DependencyType type) noexcept {
#if CUDA_VERSION >= 12040
    // A typed edge needs the annotated driver API: it is the only way to attach
    // a `CUgraphEdgeData`, and the annotated `cuGraphRemoveDependencies` is what
    // verifies that the edge being retyped exists at all. The annotated `_v2`
    // entry points exist from CUDA 12.4 on (whether or not the unversioned
    // names are remapped - they are only remapped on 13.x), so this gate is the
    // version check; older toolkits take the graceful failure path at the
    // bottom.
    if (graph == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: set_dependency_type with an invalid graph handle; "
            "the edge was not retyped.");
        return false;
    }
    if (from_index == to_index) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: set_dependency_type cannot retype the self edge "
            "{} -> {}; a node never depends on itself.",
            from_index, to_index);
        return false;
    }
    if (type != DependencyType::DEFAULT && type != DependencyType::PROGRAMMATIC) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: set_dependency_type was given the unknown dependency "
            "type {}; the edge was not retyped.",
            static_cast<uint32_t>(type));
        return false;
    }
    return _device->with_handle([&]() -> bool {
        std::scoped_lock lock{_mutex};
        auto graph_it = _graph_data.find(graph);
        if (graph_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: set_dependency_type on an unknown graph handle "
                "{}; the edge was not retyped.",
                graph);
            return false;
        }
        auto &data = graph_it->second;
        if (from_index >= data.nodes.size() || to_index >= data.nodes.size()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: set_dependency_type indices out of range "
                "({} -> {} of {} nodes); the edge was not retyped.",
                from_index, to_index, data.nodes.size());
            return false;
        }
        auto recorded = std::find(data.programmatic_edges.cbegin(),
                                  data.programmatic_edges.cend(),
                                  std::pair{from_index, to_index});
        if (type == DependencyType::DEFAULT && recorded == data.programmatic_edges.cend()) {
            // Every edge the DAG analysis and `add_dependency` create is
            // already a default (fully serializing) edge, so there is nothing
            // to restore and nothing to do. This is a no-op, deliberately not
            // an edge creation: use `add_dependency` to wire a missing edge.
            return true;
        }
        if (type == DependencyType::PROGRAMMATIC &&
            (data.nodes[from_index].kind != GraphNodeKind::KERNEL ||
             data.nodes[to_index].kind != GraphNodeKind::KERNEL)) {
            // The programmatic port is defined for kernel nodes only, so an
            // edge that touches any other node type can never carry the
            // annotation. Refuse before touching the graph.
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: a PROGRAMMATIC (PDL) edge is only defined "
                "between two kernel nodes, but nodes {} -> {} are of kinds {} "
                "and {}; the edge was not retyped.",
                from_index, to_index,
                static_cast<uint32_t>(data.nodes[from_index].kind),
                static_cast<uint32_t>(data.nodes[to_index].kind));
            return false;
        }
        auto from = data.nodes[from_index].node;
        auto to = data.nodes[to_index].node;
        // Retyping is a remove + re-add: there is no API that mutates the
        // annotation of an existing edge in place. The removal is also the
        // existence check - a pair that the DAG analysis / add_dependency never
        // wired is rejected by the driver instead of being silently created
        // here - and it is passed the annotation the edge is currently believed
        // to carry so the round trip stays exact.
        auto is_programmatic = recorded != data.programmatic_edges.cend();
        auto current = cuda_graph_ext_detail::make_edge_data(
            is_programmatic ? DependencyType::PROGRAMMATIC : DependencyType::DEFAULT);
        if (auto err = cuda_graph_ext_detail::remove_graph_dependency_typed(
                reinterpret_cast<CUgraph>(graph), &from, &to, current);
            err != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(err, &err_name);
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: cannot retype the edge {} -> {}: the graph has "
                "no such edge ({}). Typed edges only retype an edge that the "
                "DAG analysis or add_dependency already created.",
                from_index, to_index, err_name ? err_name : "unknown");
            return false;
        }
        auto desired = cuda_graph_ext_detail::make_edge_data(type);
        if (auto err = cuda_graph_ext_detail::add_graph_dependency_typed(
                reinterpret_cast<CUgraph>(graph), &from, &to, desired);
            err != CUDA_SUCCESS) {
            // Never leave the graph quietly missing a dependency: put the
            // original edge back before reporting the failure.
            const char *err_name = nullptr;
            cuGetErrorName(err, &err_name);
            if (auto restore = cuda_graph_ext_detail::add_graph_dependency_typed(
                    reinterpret_cast<CUgraph>(graph), &from, &to, current);
                restore != CUDA_SUCCESS) {
                const char *restore_name = nullptr;
                cuGetErrorName(restore, &restore_name);
                LUISA_WARNING_WITH_LOCATION(
                    "CudaGraphExt: failed to restore the original annotation "
                    "of the edge {} -> {} after a failed retype ({}); the "
                    "graph may be missing that dependency.",
                    from_index, to_index, restore_name ? restore_name : "unknown");
            }
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: failed to re-add the edge {} -> {} with the "
                "requested annotation ({}); the edge was not retyped.",
                from_index, to_index, err_name ? err_name : "unknown");
            return false;
        }
        // Diagnostic bookkeeping of the typed edges of this graph. The
        // annotation itself lives in the graph topology from here on (it is
        // baked into an executable graph at instantiation).
        if (type == DependencyType::PROGRAMMATIC) {
            if (!is_programmatic) { data.programmatic_edges.emplace_back(from_index, to_index); }
        } else {
            data.programmatic_edges.erase(recorded);
        }
        return true;
    });
#else
    // No annotated edge API before CUDA 12.4 (there is no `CUgraphEdgeData` and
    // no `CUgraphDependencyType`, hence no programmatic port to name). Refuse
    // loudly instead of leaving an ordinary edge that would drop the
    // grid-dependency-synchronize contract.
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: set_dependency_type requires the CUDA >= 12.4 annotated "
        "graph edge API (toolkit is {}); the edge was not retyped.",
        CUDA_VERSION);
    (void)graph;
    (void)from_index;
    (void)to_index;
    (void)type;
    return false;
#endif
}

// ---------------------------------------------------------------------------
// G1 (SA-2): conditional nodes (IF / WHILE / SWITCH)
//
// A conditional node is a container that runs one of the body graphs CUDA
// creates with it. The condition lives in a handle created ahead of time for
// the node's graph and is written from device code via `set_graph_conditional`
// (or seeded from a launch-time default). Handles are tied to their graph and
// are never destroyed; the body graphs are owned by the node and are only
// populated through `build_body_graph`.
// ---------------------------------------------------------------------------

CudaGraphExt::ConditionalHandle CudaGraphExtImpl::create_conditional_handle(
    GraphHandle graph, uint32_t default_value, bool assign_default) noexcept {
#if CUDA_VERSION >= 12030
    if (graph == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: create_conditional_handle with an invalid graph handle; "
            "no conditional handle was created.");
        return 0u;// invalid
    }
    return _device->with_handle([&]() -> ConditionalHandle {
        std::scoped_lock lock{_mutex};
        auto graph_it = _graph_data.find(graph);
        if (graph_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: create_conditional_handle on an unknown graph; no "
                "conditional handle was created.");
            return 0u;// invalid
        }
        CUgraphConditionalHandle handle = 0;
        // CU_GRAPH_COND_ASSIGN_DEFAULT gives do-while semantics: the condition
        // starts as `default_value` on every launch of the graph. Without it a
        // kernel must write the handle before the conditional node evaluates
        // it (a true while-loop that may run zero times).
        auto flags = assign_default ? CU_GRAPH_COND_ASSIGN_DEFAULT : 0u;
        if (auto err = cuGraphConditionalHandleCreate(
                &handle, reinterpret_cast<CUgraph>(graph),
                cuda_graph_ext_detail::current_context(), default_value, flags);
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphConditionalHandleCreate", err);
            return 0u;// invalid
        }
        // Handles are tied to their graph and never destroyed; remember which
        // ones belong here so add_conditional_node can reject a foreign handle.
        graph_it->second.conditional_handles.emplace_back(static_cast<uint64_t>(handle));
        return static_cast<uint64_t>(handle);
    });
#else
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: conditional handles require CUDA >= 12.3 (toolkit is "
        "{}); no conditional handle was created.",
        CUDA_VERSION);
    return 0u;// invalid
#endif
}

bool CudaGraphExtImpl::add_conditional_node(GraphHandle graph, ConditionalType type,
                                            ConditionalHandle handle, size_t num_bodies,
                                            luisa::span<GraphHandle> body_graphs_out,
                                            size_t *node_index) noexcept {
#if CUDA_VERSION >= 12030
    if (graph == invalid_handle || handle == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_conditional_node with an invalid graph or "
            "conditional handle (0 is invalid); no conditional node was added.");
        return false;
    }
#if CUDA_VERSION < 12080
    if (type == ConditionalType::SWITCH) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: SWITCH conditional nodes require CUDA >= 12.8 "
            "(toolkit is {}); no conditional node was added.",
            CUDA_VERSION);
        return false;
    }
#endif
    // Body counts per the frozen contract: WHILE = 1, IF = 1 or 2 (body[1] is
    // the ELSE branch), SWITCH >= 1.
    auto valid_body_count = false;
    switch (type) {
        case ConditionalType::IF:
            valid_body_count = num_bodies == 1u || num_bodies == 2u;
            break;
        case ConditionalType::WHILE:
            valid_body_count = num_bodies == 1u;
            break;
        case ConditionalType::SWITCH:
            valid_body_count = num_bodies >= 1u;
            break;
    }
    if (!valid_body_count || body_graphs_out.size() < num_bodies) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_conditional_node with an invalid body count ({} "
            "bodies for type {}); IF needs 1 or 2, WHILE exactly 1, SWITCH at "
            "least 1, and body_graphs_out must be large enough. No conditional "
            "node was added.",
            num_bodies, static_cast<uint32_t>(type));
        return false;
    }
    return _device->with_handle([&]() -> bool {
        std::scoped_lock lock{_mutex};
        auto graph_it = _graph_data.find(graph);
        if (graph_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: add_conditional_node on an unknown graph; no "
                "conditional node was added.");
            return false;
        }
        auto &data = graph_it->second;
        if (std::find(data.conditional_handles.begin(), data.conditional_handles.end(),
                      handle) == data.conditional_handles.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: add_conditional_node with a handle that was not "
                "created for this graph; handles are tied to the graph that "
                "created them. No conditional node was added.");
            return false;
        }
      // CUDA creates the empty body graphs and hands them back through
      // `phGraph_out`; they are owned by the conditional node and must not be
      // registered in `_graph_data` or destroyed on their own. NOTE (verified
      // against driver 13.4): the driver REPLACES `params.conditional.phGraph_out`
      // with a pointer to its own array - the caller-provided buffer is left
      // untouched. The body handles must therefore be read back from the struct
      // field AFTER the add call.
      luisa::vector<CUgraph> bodies(num_bodies, nullptr);
      CUgraphNodeParams params{};
      params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
      params.conditional.handle = handle;
      params.conditional.type = static_cast<CUgraphConditionalNodeType>(type);
      params.conditional.size = static_cast<unsigned int>(num_bodies);
      params.conditional.phGraph_out = bodies.data();
      params.conditional.ctx = cuda_graph_ext_detail::current_context();
      CUgraphNode node = nullptr;
      // Explicit add_* APIs attach no dependency edges; the caller wires the
      // node up with add_dependency.
      if (auto err = cuda_graph_ext_detail::add_graph_node(
              reinterpret_cast<CUgraph>(graph), nullptr, 0u, &params, &node);
          err != CUDA_SUCCESS) {
          cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphAddNode (conditional)", err);
          return false;
      }
      auto driver_bodies = params.conditional.phGraph_out;
      auto bodies_ok = driver_bodies != nullptr;
      if (bodies_ok) {
          for (auto i = size_t{0u}; i < num_bodies; i++) {
              if (driver_bodies[i] == nullptr) { bodies_ok = false; break; }
          }
      }
      if (!bodies_ok) {
          LUISA_WARNING_WITH_LOCATION(
              "CudaGraphExt: the driver did not report {} conditional body "
              "graph(s); the conditional node was removed.",
              num_bodies);
          cuGraphDestroyNode(node);
          return false;
      }
      auto index = data.nodes.size();
      data.nodes.push_back(GraphNodeRecord{.node = node, .kind = GraphNodeKind::CONDITIONAL});
      // Frozen-topology marker: whole-graph update() must now fail with
      // GraphUpdateErrorType::UNSUPPORTED.
      data.has_conditional = true;
      GraphData::ConditionalRecord record;
      record.node_index = index;
      record.handle = handle;
      record.type = type;
      record.bodies.reserve(num_bodies);
      record.body_nodes.resize(num_bodies);
      for (auto i = size_t{0u}; i < num_bodies; i++) {
          auto body = reinterpret_cast<uint64_t>(driver_bodies[i]);
          if (body_graphs_out.size() > i) { body_graphs_out[i] = body; }
          record.bodies.emplace_back(body);
      }
        data.conditional_nodes.emplace_back(std::move(record));
        if (node_index != nullptr) { *node_index = index; }
        return true;
    });
#else
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: conditional nodes require CUDA >= 12.3 (toolkit is "
        "{}); no conditional node was added.",
        CUDA_VERSION);
    return false;
#endif
}

bool CudaGraphExtImpl::build_body_graph(GraphHandle body, CommandList &&cmdlist) noexcept {
#if CUDA_VERSION >= 12030
    if (body == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: build_body_graph with an invalid body graph handle; "
            "the body was left empty.");
        return false;
    }
    if (cmdlist.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: build_body_graph refuses an empty command list; the "
            "body was left empty.");
        return false;
    }
    // Conditional-body requirements (research doc 2.3): the body may contain
    // kernel, empty, memcpy, memset, child-graph and conditional nodes only, and
    // no memcpy node may touch a CUDA array. Every texture command here lowers
    // to an array memcpy node, so refuse them *before* creating any node: a body
    // graph cannot be thrown away like a freshly created top-level graph.
    for (auto &cmd : cmdlist.commands()) {
        switch (cmd->tag()) {
            case Command::Tag::EBufferToTextureCopyCommand:
            case Command::Tag::ETextureUploadCommand:
            case Command::Tag::ETextureDownloadCommand:
            case Command::Tag::ETextureCopyCommand:
            case Command::Tag::ETextureToBufferCopyCommand:
                LUISA_WARNING_WITH_LOCATION(
                    "CudaGraphExt: a texture command cannot be captured into a "
                    "conditional body graph (CUDA arrays are forbidden in "
                    "conditional bodies); the body was left empty.");
                return false;
            default: break;
        }
    }
    return _device->with_handle([&]() -> bool {
        std::scoped_lock lock{_mutex};
        GraphHandle parent_graph = invalid_handle;
        auto conditional_index = size_t{0u};
        auto body_slot = size_t{0u};
        if (!_find_conditional_body(body, &parent_graph, &conditional_index, &body_slot)) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: build_body_graph target is not a conditional body "
                "graph handed out by add_conditional_node; the body was left "
                "empty.");
            return false;
        }
        auto parent_it = _graph_data.find(parent_graph);
        if (parent_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: build_body_graph lost the parent graph of the "
                "body; the body was left empty.");
            return false;
        }
        auto &parent = parent_it->second;
        // Roll back everything this call added to the body. `cuGraphDestroyNode`
        // removes a node and its edges, so a failed build leaves the body empty
        // and reusable instead of half populated.
        auto rollback = [](luisa::vector<GraphNodeRecord> &records) noexcept {
            for (auto &record : records) {
                if (cuGraphDestroyNode(record.node) != CUDA_SUCCESS) {
                    LUISA_WARNING_WITH_LOCATION(
                        "CudaGraphExt: failed to roll back a partially built "
                        "conditional body graph node.");
                    break;
                }
            }
        };
        // A body is its own node-index space: its records live in a scratch
        // GraphData and never enter the parent's `nodes`, so the parent's
        // node_index space does not shift.
        GraphData body_data;
        if (!_build_graph_from_commands(reinterpret_cast<CUgraph>(body),
                                        std::move(cmdlist), body_data)) {
            rollback(body_data.nodes);
            return false;
        }
        // Belt and braces for the array ban: the pre-scan above already refused
        // every texture command, but check the actual node parameters so a
        // future command type that lowers to an array memcpy cannot slip
        // through unnoticed.
        for (auto &record : body_data.nodes) {
            if (record.kind != GraphNodeKind::MEMCPY) { continue; }
            CUDA_MEMCPY3D params{};
            if (cuGraphMemcpyNodeGetParams(record.node, &params) == CUDA_SUCCESS &&
                (params.srcMemoryType == CU_MEMORYTYPE_ARRAY ||
                 params.dstMemoryType == CU_MEMORYTYPE_ARRAY)) {
                LUISA_WARNING_WITH_LOCATION(
                    "CudaGraphExt: a CUDA-array memcpy node was created inside a "
                    "conditional body (CUDA arrays are forbidden there); the "
                    "body was rolled back.");
                rollback(body_data.nodes);
                return false;
            }
        }
        // The parent graph owns the conditional node and therefore the body: the
        // pinned staging blocks and tile-node indices of the body move into the
        // parent so they outlive the body nodes.
        parent.host_allocations.insert(parent.host_allocations.end(),
                                       std::make_move_iterator(body_data.host_allocations.begin()),
                                       std::make_move_iterator(body_data.host_allocations.end()));
        parent.host_copies.insert(parent.host_copies.end(),
                                  std::make_move_iterator(body_data.host_copies.begin()),
                                  std::make_move_iterator(body_data.host_copies.end()));
        parent.tile_nodes.insert(parent.tile_nodes.end(),
                                 body_data.tile_nodes.begin(), body_data.tile_nodes.end());
        auto &records = parent.conditional_nodes[conditional_index].body_nodes[body_slot];
        records.insert(records.end(),
                       std::make_move_iterator(body_data.nodes.begin()),
                       std::make_move_iterator(body_data.nodes.end()));
        return true;
    });
#else
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: conditional body graphs require CUDA >= 12.3 (toolkit "
        "is {}); the body graph was left empty.",
        CUDA_VERSION);
    return false;
#endif
}

// ---------------------------------------------------------------------------
// G3 (SA-4): device-side launch
//
// A CUgraphExec instantiated with CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH is
// the only thing the host has to prepare: the flag makes the driver build the
// device-resident work descriptors that the device runtime entry point
// (`cudaGraphLaunch(exec, cudaStreamGraph*)`, lowered from the DSL
// `graph_device_launch` builtin) needs to launch the graph from inside another
// graph execution. The token handed out here is therefore the executable graph
// handle itself - exactly the value the device runtime expects - and the
// caller passes it to a kernel as a plain `uint64` argument.
//
// Contract (research doc 3.2), enforced by the driver and documented here
// because nothing on the host side can check it:
//   * the graph is launchable from the device only after `upload()`: an exec
//     that was never uploaded has no device-resident image to launch; the
//     caller uploads on the stream that drives the launch;
//   * a device launch may only happen from within a graph execution (a plain
//     dispatch has no graph environment);
//   * never launch the same exec from the host and the device concurrently;
//   * updates stay host-only, i.e. a DEVICE_LAUNCH exec is updated with the
//     host-side update APIs, never while a device launch may be in flight;
//   * DEVICE_LAUNCH cannot be combined with AUTO_FREE_ON_LAUNCH (refused in
//     `_instantiate`).
// ---------------------------------------------------------------------------

uint64_t CudaGraphExtImpl::get_device_launch_token(GraphExecHandle exec) noexcept {
#if CUDA_VERSION >= 12000
    if (exec == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: get_device_launch_token on an invalid executable "
            "graph handle; no device launch token is available.");
        return 0u;
    }
    return _device->with_handle([&]() -> uint64_t {
        std::scoped_lock lock{_mutex};
        auto it = _exec_data.find(exec);
        if (it == _exec_data.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: get_device_launch_token on an unknown "
                "executable graph; no device launch token is available.");
            return 0u;
        }
        if ((static_cast<uint32_t>(it->second.instantiate_flags) &
             static_cast<uint32_t>(InstantiateFlag::DEVICE_LAUNCH)) == 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: device-side graph launch requires the "
                "executable graph to be instantiated with "
                "InstantiateFlag::DEVICE_LAUNCH; this exec is host-launch "
                "only and gets no device launch token.");
            return 0u;
        }
        // The token is the executable graph handle: that is the value the
        // device runtime expects for the exec argument of a device-side graph
        // launch, and it stays valid for the whole lifetime of the exec.
        return exec;
    });
#else
    (void)exec;
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: device-side graph launch requires CUDA >= 12.0 "
        "(toolkit is {}); no device launch token is available.",
        CUDA_VERSION);
    return 0u;
#endif
}

// ---------------------------------------------------------------------------
// G4 (SA-4): device-updatable kernel nodes
//
// Opting a kernel node into device updatability is a graph-template operation:
// the node attribute is baked into the executable graph at instantiation, which
// is what makes the returned CUgraphDeviceNode usable by device code
// (`cudaGraphKernelNodeSetParam`/`SetEnabled`/`SetGridDim`, lowered from the
// DSL `graph_node_*` builtins). The restrictions that follow from that are
// enforced here and in `_instantiate`:
//   * a node that opted in can never opt out, so the graph keeps
//     `has_device_updatable` forever and whole-graph `update()` refuses with
//     GraphUpdateErrorType::UNSUPPORTED (neither the graph nor its exec may be
//     passed to cuGraphExecUpdate);
//   * a graph containing device-updatable nodes cannot be instantiated more
//     than once (enforced in `_instantiate`);
//   * the opt-in is only valid before the first instantiation of the template;
//   * only kernel nodes qualify, and not the direct-buffer Tile kernel nodes,
//     whose launch is an entirely different mechanism (a Tile node is not a
//     `kernel_main` node, so its parameter block carries Tile pointers rather
//     than the plain CUDA kernel parameters a device update patches).
// ---------------------------------------------------------------------------

bool CudaGraphExtImpl::mark_node_device_updatable(GraphHandle graph, size_t node_index) noexcept {
#if CUDA_VERSION >= 12040
    if (graph == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: mark_node_device_updatable on an invalid graph "
            "handle; the node was left host-updatable.");
        return false;
    }
    return _device->with_handle([&]() -> bool {
        std::scoped_lock lock{_mutex};
        auto it = _graph_data.find(graph);
        if (it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: mark_node_device_updatable on an unknown "
                "graph; the node was left host-updatable.");
            return false;
        }
        auto &data = it->second;
        if (data.instantiated) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: mark_node_device_updatable requires a graph "
                "template that has not been instantiated yet: the device "
                "node handles are bound to the executable graph at "
                "instantiation. Node {} was left host-updatable.",
                node_index);
            return false;
        }
        if (node_index >= data.nodes.size()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: mark_node_device_updatable on out-of-range "
                "node index {} (the graph has {} node(s)).",
                node_index, data.nodes.size());
            return false;
        }
        auto &record = data.nodes[node_index];
        if (record.kind != GraphNodeKind::KERNEL) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: mark_node_device_updatable requires a kernel "
                "node, but node index {} is of kind {}; the node was left "
                "host-updatable.",
                node_index, static_cast<uint32_t>(record.kind));
            return false;
        }
        if (std::find(data.tile_nodes.begin(), data.tile_nodes.end(), node_index) !=
            data.tile_nodes.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: mark_node_device_updatable is not supported "
                "for direct-buffer Tile kernel nodes (node index {}): their "
                "launch and parameter block are a different mechanism. The "
                "node was left host-updatable.",
                node_index);
            return false;
        }
        if (record.device_updatable) {
            // The attribute cannot be cleared once set, so a repeated request
            // for the same node is a success (idempotent).
            return true;
        }
        CUkernelNodeAttrValue value{};
        value.deviceUpdatableKernelNode.deviceUpdatable = 1;
        value.deviceUpdatableKernelNode.devNode = nullptr;
        if (auto ret = cuGraphKernelNodeSetAttribute(
                record.node,
                CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE,
                &value);
            ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: cuGraphKernelNodeSetAttribute("
                "DEVICE_UPDATABLE_KERNEL_NODE) on node {} failed: {}; the "
                "node was left host-updatable.",
                node_index, err_name ? err_name : "unknown");
            return false;
        }
        record.device_updatable = true;
        // Latch the frozen-topology marker: whole-graph update() is now
        // permanently disabled for every exec of this graph, and the graph may
        // only be instantiated once.
        data.has_device_updatable = true;
        return true;
    });
#else
    (void)graph;
    (void)node_index;
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: device-updatable kernel nodes require CUDA >= 12.4 "
        "(toolkit is {}); the node was left host-updatable.",
        CUDA_VERSION);
    return false;
#endif
}

uint64_t CudaGraphExtImpl::get_device_node_handle(GraphExecHandle exec, size_t node_index) noexcept {
#if CUDA_VERSION >= 12040
    if (exec == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: get_device_node_handle on an invalid executable "
            "graph handle; no device node handle is available.");
        return 0u;
    }
    return _device->with_handle([&]() -> uint64_t {
        std::scoped_lock lock{_mutex};
        auto it = _exec_data.find(exec);
        if (it == _exec_data.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: get_device_node_handle on an unknown "
                "executable graph; no device node handle is available.");
            return 0u;
        }
        auto git = _graph_data.find(it->second.graph_handle);
        if (git == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: get_device_node_handle: the executable graph "
                "has no live graph template; no device node handle is "
                "available.");
            return 0u;
        }
        auto &data = git->second;
        if (!data.has_device_updatable) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: the graph of this executable contains no "
                "device-updatable kernel nodes; no device node handle is "
                "available.");
            return 0u;
        }
        if (node_index >= data.nodes.size()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: get_device_node_handle on out-of-range node "
                "index {} (the graph has {} node(s)).",
                node_index, data.nodes.size());
            return 0u;
        }
        auto &record = data.nodes[node_index];
        if (record.kind != GraphNodeKind::KERNEL || !record.device_updatable) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: node index {} is not a device-updatable "
                "kernel node (kind {}, device-updatable {}); no device node "
                "handle is available.",
                node_index, static_cast<uint32_t>(record.kind),
                record.device_updatable);
            return 0u;
        }
        // Retrieval. cuda.h exposes exactly one host-side accessor for a
        // CUgraphDeviceNode - the deviceUpdatableKernelNode field of the
        // kernel-node attribute - and no graph-exec-level getter (the type
        // appears nowhere else in the header), so the handle is read back from
        // the graph-template node the attribute was set on. Because a graph
        // with device-updatable nodes allows a single instantiation only
        // (enforced in `_instantiate`), that handle is unambiguous: it belongs
        // to the one executable graph the template spawned.
        CUkernelNodeAttrValue value{};
        if (auto ret = cuGraphKernelNodeGetAttribute(
                record.node,
                CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE,
                &value);
            ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: cuGraphKernelNodeGetAttribute("
                "DEVICE_UPDATABLE_KERNEL_NODE) on node {} failed: {}; no "
                "device node handle is available.",
                node_index, err_name ? err_name : "unknown");
            return 0u;
        }
        auto dev_node = value.deviceUpdatableKernelNode.devNode;
        if (dev_node == nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: the driver did not hand out a device node "
                "handle for the device-updatable node {} (this "
                "toolkit/driver combination does not support device-side "
                "node updates); no device node handle is available.",
                node_index);
            return 0u;
        }
        return reinterpret_cast<uint64_t>(dev_node);
    });
#else
    (void)exec;
    (void)node_index;
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: device-updatable kernel nodes require CUDA >= 12.4 "
        "(toolkit is {}); no device node handle is available.",
        CUDA_VERSION);
    return 0u;
#endif
}

// ---------------------------------------------------------------------------
// G5 (SA-6): graph memory nodes (cuGraphAddMemAllocNode / cuGraphAddMemFreeNode)
//
// A graph memory allocation is an ordinary device allocation taken from a CUDA
// memory pool, but it is created by a *graph node*: the driver reserves the
// address when the node is created, documents that "the allocation's address
// remains fixed across instantiations and launches", and ties the allocation's
// lifetime to the executable graph (a free node or
// CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH releases it earlier). The
// address therefore belongs to the graph *template* node, which is the very
// same node `_get_node` hands out for the node-parameter APIs; there is no
// per-exec address to track.
//
// Rules the driver enforces on a graph that contains mem alloc/free nodes (and
// which this implementation keeps fail-closed):
//  - only ONE instantiation of the graph may exist at any point in time
//    (`_instantiate` counts this template's live execs and refuses a second
//    one; destroying the exec releases the restriction again, unlike the
//    device-updatable case);
//  - nodes and edges of the graph cannot be deleted;
//  - a free node may only name an address produced by an alloc node of the
//    *same* graph, and may free it only once (`add_mem_free_node` validates
//    both against the recorded bookkeeping - the driver would reject them as
//    CUDA_ERROR_INVALID_VALUE, but a precise diagnostic is worth more);
//  - the alloc/free node parameters cannot be modified by cuGraphExecUpdate,
//    so nothing here feeds them into any auto-update path.
//
// Lifetime: a mem-free node must live in the graph that owns the allocation
// (freeing it elsewhere is a driver error, not an API this extension can
// express), and an executable graph that still references the allocation must
// be destroyed before the template - which is exactly the destruction order of
// `~CudaGraphExtImpl` (execs first, then templates).
// ---------------------------------------------------------------------------

bool CudaGraphExtImpl::add_mem_alloc_node(GraphHandle graph, size_t size, size_t alignment,
                                          size_t *node_index) noexcept {
#if CUDA_VERSION >= 11040
    if (graph == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_mem_alloc_node on an invalid graph handle; no "
            "allocation node was added.");
        return false;
    }
    if (size == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_mem_alloc_node with a zero-byte size; no "
            "allocation node was added.");
        return false;
    }
    if (!cuda_graph_ext_detail::is_power_of_two(alignment)) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_mem_alloc_node with alignment {} (not a power "
            "of two); no allocation node was added. The driver API has no "
            "alignment parameter - it is expressed by rounding the requested "
            "size up - so only a power of two is meaningful.",
            alignment);
        return false;
    }
    if (size > (std::numeric_limits<size_t>::max)() - (alignment - 1u)) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_mem_alloc_node size {} with alignment {} "
            "overflows when rounded up to the alignment; no allocation node "
            "was added.",
            size, alignment);
        return false;
    }
    return _device->with_handle([&]() -> bool {
        std::scoped_lock lock{_mutex};
        auto graph_it = _graph_data.find(graph);
        if (graph_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: add_mem_alloc_node on an unknown graph; no "
                "allocation node was added.");
            return false;
        }
        if (!_device_supports_graph_memory()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: graph memory nodes allocate from a CUDA memory "
                "pool, but this device does not report "
                "CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED; no allocation "
                "node was added.");
            return false;
        }
        auto &data = graph_it->second;
        CUDA_MEM_ALLOC_NODE_PARAMS params{};
        // The pool of a graph memory allocation is described entirely by
        // `poolProps`: pinned is the only legal pool allocation type, no
        // handle/export types (the driver requires CU_MEM_HANDLE_TYPE_NONE for
        // graph nodes, IPC is not supported), located on this device. The
        // remaining fields (win32SecurityAttributes/maxSize/usage) stay
        // zero-initialized, i.e. driver defaults.
        params.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        params.poolProps.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
        params.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        params.poolProps.location.id = static_cast<int>(_device->handle().index());
        params.accessDescs = nullptr;// no peer-GPU access descriptors
        params.accessDescCount = 0u;
        // `cuGraphAddMemAllocNode` has no alignment parameter: the node's
        // bytesize is the only knob, so the requested alignment is expressed by
        // rounding the allocation size up to it (pool allocations are served
        // per requested size). The address the driver then reports is the
        // allocation's address; a power-of-two alignment - enforced above - is
        // what makes the rounded request a truthful "at least this aligned".
        params.bytesize = cuda_graph_ext_detail::round_up_to_alignment(size, alignment);
        params.dptr = 0u;// out: the driver returns the allocation's address here
        CUgraphNode node = nullptr;
        // Explicit add_* APIs attach no dependency edges; the caller wires the
        // node up with add_dependency.
        if (auto err = cuGraphAddMemAllocNode(&node, reinterpret_cast<CUgraph>(graph),
                                              nullptr, 0u, &params);
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphAddMemAllocNode", err);
            return false;
        }
        auto index = data.nodes.size();
        data.nodes.push_back(GraphNodeRecord{.node = node, .kind = GraphNodeKind::MEM_ALLOC_NODE});
        data.has_mem_nodes = true;
        // `dptr` is an out parameter of the *add* call and the address is fixed
        // for the graph's whole life, so the bookkeeping can be filled right
        // away - that is what lets add_mem_free_node validate a caller-supplied
        // address for a graph that has never been instantiated. A null address
        // would mean the driver did not follow that contract; the record is
        // then simply left empty and get_mem_alloc_node_address() picks the
        // address up later.
        if (params.dptr != 0u) {
            data.mem_alloc_addresses.emplace_back(static_cast<uint64_t>(params.dptr));
        }
        if (node_index != nullptr) { *node_index = index; }
        return true;
    });
#else
    (void)graph;
    (void)size;
    (void)alignment;
    (void)node_index;
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: graph memory allocation nodes require CUDA >= 11.4 "
        "(toolkit is {}); no allocation node was added.",
        CUDA_VERSION);
    return false;
#endif
}

bool CudaGraphExtImpl::add_mem_free_node(GraphHandle graph, uint64_t device_address,
                                         size_t *node_index) noexcept {
#if CUDA_VERSION >= 11040
    if (graph == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_mem_free_node on an invalid graph handle; no "
            "free node was added.");
        return false;
    }
    if (device_address == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: add_mem_free_node with a null device address (0 is "
            "never a valid graph memory allocation); no free node was added.");
        return false;
    }
    return _device->with_handle([&]() -> bool {
        std::scoped_lock lock{_mutex};
        auto graph_it = _graph_data.find(graph);
        if (graph_it == _graph_data.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: add_mem_free_node on an unknown graph; no free "
                "node was added.");
            return false;
        }
        if (!_device_supports_graph_memory()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: graph memory nodes allocate from a CUDA memory "
                "pool, but this device does not report "
                "CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED; no free node was "
                "added.");
            return false;
        }
        auto &data = graph_it->second;
        // Fail-closed per the extension contract: the driver only accepts an
        // address returned by an alloc node of this very graph, and only once
        // per allocation. Both conditions are checked against the recorded
        // bookkeeping so the caller gets the real reason instead of the
        // driver's generic CUDA_ERROR_INVALID_VALUE.
        if (std::find(data.mem_freed_addresses.begin(), data.mem_freed_addresses.end(),
                      device_address) != data.mem_freed_addresses.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: add_mem_free_node for {:#x}, which this graph "
                "already has a free node for; a graph memory allocation can be "
                "freed only once in its owning graph. No free node was added.",
                device_address);
            return false;
        }
        if (std::find(data.mem_alloc_addresses.begin(), data.mem_alloc_addresses.end(),
                      device_address) == data.mem_alloc_addresses.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: add_mem_free_node for {:#x}, which is not an "
                "allocation address of this graph; a free node may only name "
                "an address produced by a mem-alloc node of the same graph "
                "(the driver enforces this too). The address is recorded at "
                "add_mem_alloc_node() time and can be read with "
                "get_mem_alloc_node_address(graph, ...) before instantiating. "
                "No free node was added.",
                device_address);
            return false;
        }
        // Driver rule (verified against driver 13.4): a mem-free node cannot be
        // added to a graph that has already been instantiated -
        // cuGraphAddMemFreeNode fails with CUDA_ERROR_INVALID_VALUE. The
        // allocation address is known at add_mem_alloc_node() time, so the
        // supported pattern is: alloc node -> free node -> dependency wiring ->
        // instantiate, all on the template.
        if (data.instantiated) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: add_mem_free_node on a graph template that has "
                "already been instantiated; the driver rejects mem-free nodes "
                "once a graph with memory nodes has been instantiated. Add the "
                "free node before the first instantiate() - the allocation "
                "address is available from get_mem_alloc_node_address(graph, ...) "
                "without instantiating. No free node was added.");
            return false;
        }
        CUgraphNode node = nullptr;
        // As everywhere else in this file: no implicit dependency edges.
        if (auto err = cuGraphAddMemFreeNode(&node, reinterpret_cast<CUgraph>(graph),
                                             nullptr, 0u,
                                             static_cast<CUdeviceptr>(device_address));
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphAddMemFreeNode", err);
            return false;
        }
        auto index = data.nodes.size();
        data.nodes.push_back(GraphNodeRecord{.node = node, .kind = GraphNodeKind::MEM_FREE_NODE});
        data.has_mem_nodes = true;
        data.mem_freed_addresses.emplace_back(device_address);
        if (node_index != nullptr) { *node_index = index; }
        return true;
    });
#else
    (void)graph;
    (void)device_address;
    (void)node_index;
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: graph memory free nodes require CUDA >= 11.4 (toolkit "
        "is {}); no free node was added.",
        CUDA_VERSION);
    return false;
#endif
}

  bool CudaGraphExtImpl::get_graph_mem_alloc_address(GraphHandle graph, size_t node_index,
                                                     uint64_t *device_address) noexcept {
#if CUDA_VERSION >= 11040
      if (graph == invalid_handle) {
          LUISA_WARNING_WITH_LOCATION(
              "CudaGraphExt: get_graph_mem_alloc_address on an invalid graph "
              "handle; no allocation address is available.");
          return false;
      }
      if (device_address == nullptr) {
          LUISA_WARNING_WITH_LOCATION(
              "CudaGraphExt: get_graph_mem_alloc_address without an output "
              "parameter; no allocation address is available.");
          return false;
      }
      return _device->with_handle([&]() -> bool {
          std::scoped_lock lock{_mutex};
          auto graph_it = _graph_data.find(graph);
          if (graph_it == _graph_data.end()) {
              LUISA_WARNING_WITH_LOCATION(
                  "CudaGraphExt: get_graph_mem_alloc_address on an unknown "
                  "graph; no allocation address is available.");
              return false;
          }
          auto &data = graph_it->second;
          if (node_index >= data.nodes.size()) {
              LUISA_WARNING_WITH_LOCATION(
                  "CudaGraphExt: get_mem_alloc_node_address on out-of-range node "
                  "index {} (the graph has {} node(s)); no allocation address is "
                  "available.",
                  node_index, data.nodes.size());
              return false;
          }
        auto &record = data.nodes[node_index];
        if (record.kind != GraphNodeKind::MEM_ALLOC_NODE) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: get_mem_alloc_node_address requires a graph "
                "memory allocation node, but node index {} is of kind {}; no "
                "allocation address is available.",
                node_index, static_cast<uint32_t>(record.kind));
            return false;
        }
        if (!_device_supports_graph_memory()) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: graph memory nodes allocate from a CUDA memory "
                "pool, but this device does not report "
                "CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED; no allocation "
                "address is available.");
            return false;
        }
        // Read the address back from the template node (the exec's node table
        // is a copy of it, and the two are the same node for every node API of
        // this extension).
        CUDA_MEM_ALLOC_NODE_PARAMS params{};
        if (auto err = cuGraphMemAllocNodeGetParams(record.node, &params);
            err != CUDA_SUCCESS) {
            cuda_graph_ext_detail::cuda_graph_dag_fail("cuGraphMemAllocNodeGetParams", err);
            return false;
        }
        if (params.dptr == 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "CudaGraphExt: the driver reported a null address for graph "
                "memory allocation node {}; the node has no usable allocation "
                "address.",
                node_index);
            return false;
        }
        auto address = static_cast<uint64_t>(params.dptr);
        *device_address = address;
        // Record the address (idempotent): it is the one the add call returned
        // and it stays fixed across instantiations, so this only completes the
        // bookkeeping for runtimes/drivers that leave `dptr` unset until the
        // graph is instantiated.
        if (std::find(data.mem_alloc_addresses.begin(), data.mem_alloc_addresses.end(),
                      address) == data.mem_alloc_addresses.end()) {
            data.mem_alloc_addresses.emplace_back(address);
        }
        return true;
    });
#else
    (void)graph;
    (void)node_index;
    (void)device_address;
    LUISA_WARNING_WITH_LOCATION(
        "CudaGraphExt: graph memory allocation nodes require CUDA >= 11.4 "
        "(toolkit is {}); no allocation address is available.",
        CUDA_VERSION);
    return false;
#endif
}

bool CudaGraphExtImpl::get_mem_alloc_node_address(GraphExecHandle exec, size_t node_index,
                                                  uint64_t *device_address) noexcept {
    if (exec == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: get_mem_alloc_node_address on an invalid executable "
            "graph handle; no allocation address is available.");
        return false;
    }
    // Resolve the exec's template under the lock, then run the (locking) graph
    // query without holding it (the bookkeeping mutex is not recursive).
    auto graph_handle = invalid_handle;
    {
        std::scoped_lock lock{_mutex};
        if (auto exec_it = _exec_data.find(exec); exec_it != _exec_data.end()) {
            graph_handle = exec_it->second.graph_handle;
        }
    }
    if (graph_handle == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "CudaGraphExt: get_mem_alloc_node_address on an unknown executable "
            "graph; no allocation address is available.");
        return false;
    }
    return get_graph_mem_alloc_address(graph_handle, node_index, device_address);
}

}// namespace luisa::compute::cuda
