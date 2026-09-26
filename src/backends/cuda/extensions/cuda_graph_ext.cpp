#include "cuda_graph_ext.h"
#include "../cuda_device.h"
#include "../cuda_stream.h"
#include "../cuda_buffer.h"
#include "../cuda_bindless_array.h"
#include "../cuda_shader.h"
#include "../cuda_error.h"
#include <cuda.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <luisa/core/stl/algorithm.h>

namespace luisa::compute::cuda {

namespace {

[[nodiscard]] CUstream to_cu_stream(uint64_t stream_handle) noexcept {
    return reinterpret_cast<CUDAStream *>(stream_handle)->handle();
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

}// namespace

CudaGraphExtImpl::CudaGraphExtImpl(CUDADevice *device) noexcept
    : _device{device} {}

CudaGraphExtImpl::~CudaGraphExtImpl() noexcept {
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
    return git->second.nodes[node_index];
}

ResourceCreationInfo CudaGraphExtImpl::_create_graph(CommandList &&cmdlist) noexcept {
    if (cmdlist.empty()) { return {invalid_handle, nullptr}; }

    return _device->with_handle([&]() -> ResourceCreationInfo {
        luisa::vector<luisa::shared_ptr<PinnedHostBlock>> host_allocs;
        luisa::vector<CudaGraphHostCopyData> host_copies;
        auto commands = cmdlist.steal_commands();
        auto user_callbacks = cmdlist.steal_callbacks();

        struct UploadStagingVisitor final : MutableCommandVisitor {
            luisa::vector<luisa::shared_ptr<PinnedHostBlock>> &host_allocs;
            luisa::vector<CudaGraphHostCopyData> &host_copies;
            bool ok{true};

            UploadStagingVisitor(luisa::vector<luisa::shared_ptr<PinnedHostBlock>> &host_allocs,
                                 luisa::vector<CudaGraphHostCopyData> &host_copies) noexcept
                : host_allocs{host_allocs}, host_copies{host_copies} {}

            void visit(BufferUploadCommand *upload) noexcept override {
                if (!ok) { return; }
                void *host_mem = nullptr;
                if (cuMemAllocHost(&host_mem, upload->size()) != CUDA_SUCCESS) {
                    ok = false;
                    return;
                }
                std::memcpy(host_mem, upload->data(), upload->size());
                upload->set_data(host_mem);
                host_allocs.emplace_back(luisa::make_shared<PinnedHostBlock>(host_mem));
            }

            void visit(BufferDownloadCommand *download) noexcept override {
                if (!ok) { return; }
                void *host_mem = nullptr;
                if (cuMemAllocHost(&host_mem, download->size()) != CUDA_SUCCESS) {
                    ok = false;
                    return;
                }
                host_copies.push_back(CudaGraphHostCopyData{
                    .dst = download->data(),
                    .src = host_mem,
                    .size = download->size(),
                });
                download->set_data(host_mem);
                host_allocs.emplace_back(luisa::make_shared<PinnedHostBlock>(host_mem));
            }
            void visit(BufferCopyCommand *) noexcept override {}
            void visit(BufferToTextureCopyCommand *) noexcept override {}
            void visit(ShaderDispatchCommand *) noexcept override {}
            void visit(TextureUploadCommand *) noexcept override {}
            void visit(TextureDownloadCommand *) noexcept override {}
            void visit(TextureCopyCommand *) noexcept override {}
            void visit(TextureToBufferCopyCommand *) noexcept override {}
            void visit(AccelBuildCommand *) noexcept override {}
            void visit(MeshBuildCommand *) noexcept override {}
            void visit(CurveBuildCommand *) noexcept override {}
            void visit(ProceduralPrimitiveBuildCommand *) noexcept override {}
            void visit(MotionInstanceBuildCommand *) noexcept override {}
            void visit(BindlessArrayUpdateCommand *) noexcept override {}
            void visit(CustomCommand *) noexcept override {}
        };

        UploadStagingVisitor staging_visitor{host_allocs, host_copies};
        for (auto &cmd : commands) {
            cmd->accept(staging_visitor);
            if (!staging_visitor.ok) { break; }
        }
        if (!staging_visitor.ok) { return {invalid_handle, nullptr}; }

        // ---- phase 2: dependency analysis + explicit DAG construction ----
        // Instead of stream capture, which linearizes every command onto one
        // stream and replays as a serial chain, track every argument by its
        // declared usage (like the command reorder pass) and build an
        // explicit CUDA graph whose edges are only the real RAW/WAW/WAR
        // command -> command dependencies. Hazard-free commands share no
        // edge, so a replay of an independent batch overlaps on the device;
        // a hazard chain gets exactly the edges that preserve the sequential
        // semantics of the original command list.
        luisa::vector<CUgraphNode> nodes;// primary node of every node-producing command, in command order
        nodes.reserve(commands.size());
        CUgraph graph = nullptr;
        if (auto err = cuGraphCreate(&graph, 0u); err != CUDA_SUCCESS) {
            cuda_graph_dag_fail("cuGraphCreate", err);
            return {invalid_handle, nullptr};
        }

        CUcontext ctx = nullptr;
        cuCtxGetCurrent(&ctx);

        struct GraphDagVisitor final : MutableCommandVisitor {
            CUgraph graph;
            CUcontext ctx;
            luisa::vector<CUgraphNode> &nodes;
            luisa::unordered_map<uint64_t, DagResource> buffer_states;
            luisa::unordered_map<uint64_t, DagResource> texture_states;
            luisa::unordered_map<uint64_t, DagResource> bindless_states;
            luisa::vector<uint32_t> deps;
            luisa::vector<CUgraphNode> dep_nodes;
            bool ok{true};

            GraphDagVisitor(CUgraph graph, CUcontext ctx,
                            luisa::vector<CUgraphNode> &nodes) noexcept
                : graph{graph}, ctx{ctx}, nodes{nodes} {}

            // Deduplicate the accumulated dependency indices, drop self
            // edges and map the survivors to their CUDA graph nodes.
            [[nodiscard]] luisa::span<const CUgraphNode> finish_deps(uint32_t self) noexcept {
                luisa::sort(deps.begin(), deps.end());
                deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
                dep_nodes.clear();
                for (auto d : deps) {
                    if (d != self) { dep_nodes.push_back(nodes[d]); }
                }
                return {dep_nodes.data(), dep_nodes.size()};
            }

            void add_memcpy_node(const CUDA_MEMCPY3D &params, uint32_t self,
                                 const char *what) noexcept {
                CUgraphNode node = nullptr;
                auto dependencies = finish_deps(self);
                if (auto err = cuGraphAddMemcpyNode(&node, graph,
                                                    dependencies.data(), dependencies.size(),
                                                    &params, ctx);
                    err != CUDA_SUCCESS) {
                    cuda_graph_dag_fail(what, err);
                    ok = false;
                    return;
                }
                nodes.push_back(node);
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

            void visit(BufferToTextureCopyCommand *) noexcept override {}
            void visit(TextureUploadCommand *) noexcept override {}
            void visit(TextureDownloadCommand *) noexcept override {}
            void visit(TextureCopyCommand *) noexcept override {}
            void visit(TextureToBufferCopyCommand *) noexcept override {}
            void visit(AccelBuildCommand *) noexcept override {}
            void visit(MeshBuildCommand *) noexcept override {}
            void visit(CurveBuildCommand *) noexcept override {}
            void visit(ProceduralPrimitiveBuildCommand *) noexcept override {}
            void visit(MotionInstanceBuildCommand *) noexcept override {}
            void visit(BindlessArrayUpdateCommand *) noexcept override {}
            void visit(CustomCommand *) noexcept override {}

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
                    !shader->is_native() ||
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
                if (!pack_kernel_arguments(dispatch, shader, argument_buffer)) {
                    ok = false;
                    return;
                }
                auto func = static_cast<CUfunction>(shader->handle());
                auto block_size = shader->block_size();
                // The launch configuration _launch computes for this dispatch.
                auto blocks = (dispatch_size + block_size - 1u) / block_size;
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
                params.kernelParams = kernel_params;
                params.extra = nullptr;
                CUgraphNode node = nullptr;
                auto dependencies = finish_deps(self);
                if (auto err = cuGraphAddKernelNode(&node, graph,
                                                    dependencies.data(), dependencies.size(),
                                                    &params);
                    err != CUDA_SUCCESS) {
                    cuda_graph_dag_fail("cuGraphAddKernelNode", err);
                    ok = false;
                    return;
                }
                nodes.push_back(node);
            }
        };

        GraphDagVisitor visitor{graph, ctx, nodes};
        for (auto &cmd : commands) {
            cmd->accept(visitor);
            if (!visitor.ok) { break; }
        }

        if (!visitor.ok) {
            cuGraphDestroy(graph);
            return {invalid_handle, nullptr};
        }

        for (auto &cb : user_callbacks) { cb(); }

        auto graph_handle = reinterpret_cast<uint64_t>(graph);
        {
            std::scoped_lock lock{_mutex};
            // The node list is the creation order of the node-producing
            // commands, which is deterministic and matches the order the
            // update_* node APIs expect.
            _graph_data[graph_handle] = GraphData{std::move(nodes), std::move(host_allocs), std::move(host_copies)};
        }

        return {graph_handle, graph};
    });
}

void CudaGraphExtImpl::destroy_graph(GraphHandle graph) noexcept {
    if (graph == invalid_handle) { return; }
    _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
        _graph_data.erase(graph);
        cuGraphDestroy(reinterpret_cast<CUgraph>(graph));
    });
}

ResourceCreationInfo CudaGraphExtImpl::_instantiate(GraphHandle graph, InstantiateFlag flags) noexcept {
    if (graph == invalid_handle) { return {invalid_handle, nullptr}; }
    return _device->with_handle([&]() -> ResourceCreationInfo {
        CUgraphExec exec = nullptr;
        auto ret = cuGraphInstantiateWithFlags(&exec, reinterpret_cast<CUgraph>(graph),
                                               to_instantiate_flags(flags));
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
            // a successful cuGraphExecUpdate.
            auto git = _graph_data.find(graph);
            if (git != _graph_data.end()) {
                _exec_data.emplace(exec_handle,
                                   ExecData{graph,
                                            git->second.host_allocations,
                                            git->second.host_copies});
            } else {
                _exec_data.emplace(exec_handle, ExecData{graph, {}, {}});
            }
        }
        return {exec_handle, exec};
    });
}

void CudaGraphExtImpl::destroy_exec(GraphExecHandle exec) noexcept {
    if (exec == invalid_handle) { return; }
    _device->with_handle([&] {
        {
            std::scoped_lock lock{_mutex};
            _exec_data.erase(exec);
        }
        cuGraphExecDestroy(reinterpret_cast<CUgraphExec>(exec));
    });
}

  void CudaGraphExtImpl::launch(GraphExecHandle exec, uint64_t stream_handle) noexcept {
      if (exec == invalid_handle) { return; }
      _device->with_handle([&] {
          auto stream = to_cu_stream(stream_handle);
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
          // stream.
          std::scoped_lock lock{_mutex};
          if (auto it = _exec_data.find(exec); it != _exec_data.end()) {
              for (auto &host_copy : it->second.host_copies) {
                  if (auto err = cuLaunchHostFunc(stream, cuda_graph_host_copy_callback, &host_copy);
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
        auto ret = cuGraphUpload(reinterpret_cast<CUgraphExec>(exec), to_cu_stream(stream_handle));
        if (ret != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            cuGetErrorName(ret, &err_name);
            LUISA_WARNING_WITH_LOCATION("cuGraphUpload failed: {}", err_name ? err_name : "unknown");
        }
    });
}

bool CudaGraphExtImpl::update(GraphExecHandle exec, CommandList &&cmdlist) noexcept {
    if (exec == invalid_handle || cmdlist.empty()) { return false; }
    return _device->with_handle([&] {
        auto new_graph_info = _create_graph(std::move(cmdlist));
        if (new_graph_info.handle == invalid_handle) { return false; }

        CUgraphExecUpdateResultInfo result_info{};
        auto ret = cuGraphExecUpdate(reinterpret_cast<CUgraphExec>(exec),
                                     reinterpret_cast<CUgraph>(new_graph_info.handle),
                                     &result_info);

        auto updated = ret == CUDA_SUCCESS;
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

bool CudaGraphExtImpl::update_kernel_node(GraphExecHandle exec, size_t node_index,
                                           uint3 dispatch_size,
                                           luisa::span<const Argument> arguments) noexcept {
    if (exec == invalid_handle) { return false; }
    return _device->with_handle([&] {
        std::scoped_lock lock{_mutex};
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

}// namespace luisa::compute::cuda
