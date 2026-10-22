#pragma once

#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/spin_mutex.h>
#include <cuda.h>
#include <utility>

namespace luisa::compute::cuda {

class CUDADevice;

struct CudaGraphHostCopyData {
    void *dst;
    const void *src;
    size_t size;
};

/// Owning wrapper for the pinned host memory staged by graph upload/download
/// commands. Shared (via shared_ptr) between a graph and the executable
/// graphs built from it, so the staging buffers - and the payload copies
/// that point into them - outlive whichever side is destroyed first.
struct PinnedHostBlock {
    void *ptr{};
    explicit PinnedHostBlock(void *p) noexcept : ptr{p} {}
    PinnedHostBlock(PinnedHostBlock const &) = delete;
    PinnedHostBlock &operator=(PinnedHostBlock const &) = delete;
    ~PinnedHostBlock() {
        if (ptr != nullptr) { cuMemFreeHost(ptr); }
    }
};

class CudaGraphExtImpl final : public CudaGraphExt {

public:
    /// Which node-producing entry a graph node came from. Together with the
    /// creation index (the position in `GraphData::nodes`) this is the node's
    /// stable identity: the node APIs index nodes by creation order, and the
    /// per-node type lets them reject a node of the wrong kind.
    enum struct GraphNodeKind : uint32_t {
        KERNEL,
        MEMCPY,
        MEMSET,
        HOST,
        EVENT_RECORD,
        EVENT_WAIT,
        CHILD_GRAPH,
        CONDITIONAL,
        /// G5 (SA-6): a graph memory allocation node (cuGraphAddMemAllocNode).
        /// The `_NODE` suffix avoids the clash with the Windows memory-pool
        /// macros of winnt.h: the short spellings of these two enum names must
        /// never be used in this file.
        MEM_ALLOC_NODE,
        /// G5 (SA-6): a graph memory free node (cuGraphAddMemFreeNode).
        MEM_FREE_NODE,
    };

    /// One primary node of a graph, in creation order. Later phases (SA-2
    /// conditional nodes, SA-3 typed edges, SA-4 device-updatable nodes, SA-6
    /// graph memory) hang their per-node metadata off this record instead of
    /// adding more parallel vectors.
    struct GraphNodeRecord {
        CUgraphNode node{nullptr};
        GraphNodeKind kind{GraphNodeKind::KERNEL};
        /// SA-4: the node carries
        /// CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE (device code
        /// may patch its parameters through `cudaGraphKernelNodeSetParam`).
        /// Once set it can never be cleared (CUDA requirement), which is why
        /// `mark_node_device_updatable` also latches `has_device_updatable`.
        bool device_updatable{false};
    };

private:
    CUDADevice *_device;

    struct GraphData {
        /// Creation order: the CommandList order for `create_graph`, then the
        /// order of the explicit `add_*` calls. The position in this vector is
        /// the `node_index` every node API takes.
        luisa::vector<GraphNodeRecord> nodes;
        luisa::vector<luisa::shared_ptr<PinnedHostBlock>> host_allocations;
        luisa::vector<CudaGraphHostCopyData> host_copies;
        luisa::vector<uint32_t> tile_nodes;
        /// The typed (PDL) edges of the graph, as (from_index, to_index) pairs
        /// in the creation order `set_dependency_type` was called with. Only
        /// diagnostic/round-trip bookkeeping: the annotation itself is part of
        /// the graph topology and is baked into each executable graph at
        /// instantiation, so it is deliberately not mirrored into ExecData
        /// (there is nothing an exec could do with it). Note that the driver
        /// counts an annotation difference as a topology change: whole-graph
        /// `update()` from a command list (whose edges are always default)
        /// therefore fails with TOPOLOGY_CHANGED for such an exec, which is
        /// exactly the documented fall-back (destroy_exec + instantiate).
        luisa::vector<std::pair<size_t, size_t>> programmatic_edges;
        /// Child graph templates referenced by CHILD_GRAPH nodes. Referenced
        /// by handle only, never owned: the contract requires the child graph
        /// to outlive the parent, so destruction order is the caller's.
        luisa::vector<uint64_t> child_graphs;
        /// Frozen-topology markers. A graph carrying either of them can never
        /// be updated as a whole: CUDA freezes conditional topology at
        /// instantiation, and device-updatable nodes make the exec's kernel
        /// list immutable as well.
        bool has_conditional{false};
        bool has_device_updatable{false};

        // -- G3/G4 (SA-4): device launch + device-updatable bookkeeping -----
        //
        // `instantiated` is set by the first successful `_instantiate` of this
        // template and never cleared: CUDA bakes the node table (and the
        // device-updatable device handles) into the executable graph at
        // instantiation, so the opt-in attribute may only be set on a template
        // that has not been instantiated yet.
        bool instantiated{false};
        /// Set by a successful `_instantiate` of a graph that already carries
        /// device-updatable nodes. CUDA forbids instantiating such a graph more
        /// than once, so the second `_instantiate` is refused loudly.
        bool device_updatable_instantiated{false};

        // -- G1/SA-2: conditional nodes -----------------------------------
        //
        // A conditional handle is created by the driver for exactly one graph
        // and is never destroyed (CUDA requirement); the graph only remembers
        // which handles belong to it so `add_conditional_node` can reject a
        // handle that was created for another graph.
        luisa::vector<uint64_t> conditional_handles;
        /// One entry per conditional node, in creation order. `bodies` holds
        /// the body graph handles CUDA created for that node: they are owned
        /// by the node (released with the parent graph), never by
        /// `_graph_data`, so they must never be destroyed or registered as
        /// standalone graphs. `body_nodes` keeps the node records of each
        /// body - body node indices are local to their body and never enter
        /// `nodes`, so the parent's `node_index` space does not shift.
        struct ConditionalRecord {
            size_t node_index{0u};///< creation index of the conditional node
            uint64_t handle{0u};  ///< the handle the node was built with
            CudaGraphExt::ConditionalType type{CudaGraphExt::ConditionalType::IF};
            luisa::vector<uint64_t> bodies;
            luisa::vector<luisa::vector<GraphNodeRecord>> body_nodes;
        };
        luisa::vector<ConditionalRecord> conditional_nodes;

        // -- G5 (SA-6): graph memory nodes ---------------------------------
        //
        // A mem-alloc node is the only producer of a raw device address in
        // this API, and a mem-free node may only name an address that one of
        // *this* graph's alloc nodes produced (the driver rejects anything
        // else with CUDA_ERROR_INVALID_VALUE). The driver returns the address
        // from `cuGraphAddMemAllocNode` itself and documents that "the
        // allocation's address remains fixed across instantiations and
        // launches", so the set can be filled at add time and is only
        // re-confirmed (and, for execs built by an older call, recovered) by
        // `get_mem_alloc_node_address`.
        luisa::vector<uint64_t> mem_alloc_addresses;
        /// Addresses that already carry a mem-free node in this graph. The
        /// driver frees an allocation "only once in the owning graph", so a
        /// repeat request is refused here - with the real reason - instead of
        /// being forwarded to the driver just to fail there.
        luisa::vector<uint64_t> mem_freed_addresses;
        /// Latched by the first mem-alloc/mem-free node: such a graph may only
        /// have ONE live instantiation at a time (CUDA requirement), which
        /// `_instantiate` enforces by counting this template's live execs.
        bool has_mem_nodes{false};
    };

    struct ExecData {
        uint64_t graph_handle;
        luisa::vector<luisa::shared_ptr<PinnedHostBlock>> host_allocations;
        luisa::vector<CudaGraphHostCopyData> host_copies;
        luisa::vector<uint32_t> tile_nodes;
        /// The flags this exec was created with. `get_device_launch_token`
        /// hands out a token only for an exec carrying DEVICE_LAUNCH: without
        /// the flag the driver-relevant half of the device launch contract
        /// (device-resident work descriptors) was never set up, no matter what
        /// the caller passes to the device builtin.
        InstantiateFlag instantiate_flags{InstantiateFlag::INSTANTIATE_DEFAULT};
    };

    /// Lazily probed CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: -1 = not
    /// probed yet, 0 = the device has no memory pools, 1 = supported. The
    /// attribute is a per-device constant, so it is cached for the lifetime of
    /// the extension (and only read/written while `_mutex` is held).
    mutable int _memory_pools_supported{-1};

    mutable spin_mutex _mutex;
    luisa::unordered_map<uint64_t, GraphData> _graph_data;
    luisa::unordered_map<uint64_t, ExecData> _exec_data;// exec_handle -> executable data

    [[nodiscard]] CUgraphNode _get_node(GraphExecHandle exec, size_t node_index) const noexcept;

    /// The shared construction pipeline of `_create_graph` and
    /// `build_body_graph`: stage host uploads/downloads in `data`, run the
    /// usage-based dependency analysis of the command list and append its
    /// nodes (creation order) to `graph` / `data.nodes`. The pinned staging
    /// blocks are appended to `data` so they outlive the nodes that point at
    /// them. On failure no graph is destroyed and the partially created node
    /// records (if any) remain in `data.nodes` so callers can roll back.
    [[nodiscard]] bool _build_graph_from_commands(CUgraph graph, CommandList &&cmdlist,
                                                  GraphData &data) noexcept;

    /// Resolve a conditional body graph handle to the parent graph that owns
    /// it. Returns false when `body` is not a body graph of any live graph.
    /// Any of the out parameters may be null.
    [[nodiscard]] bool _find_conditional_body(GraphHandle body, GraphHandle *parent,
                                              size_t *conditional_index,
                                              size_t *body_slot) const noexcept;

    /// Lazily probe (and cache) CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED for
    /// this device, which is what graph memory nodes allocate from. Requires a
    /// current CUDA context (call it from inside `_device->with_handle`).
    [[nodiscard]] bool _device_supports_graph_memory() noexcept;

    /// Number of live executable graphs built from `graph`. CUDA allows only
    /// one instantiation at a time for a graph that contains memory nodes, so
    /// `_instantiate` consults this instead of latching a flag: destroying the
    /// exec releases the restriction again. Caller must hold `_mutex`.
    [[nodiscard]] size_t _live_exec_count(GraphHandle graph) const noexcept;

public:
    explicit CudaGraphExtImpl(CUDADevice *device) noexcept;
    ~CudaGraphExtImpl() noexcept;

    ResourceCreationInfo _create_graph(CommandList &&cmdlist) noexcept override;
    void destroy_graph(GraphHandle graph) noexcept override;
    ResourceCreationInfo _instantiate(GraphHandle graph, InstantiateFlag flags) noexcept override;
    void destroy_exec(GraphExecHandle exec) noexcept override;
    void launch(GraphExecHandle exec, uint64_t stream_handle) noexcept override;
    void upload(GraphExecHandle exec, uint64_t stream_handle) noexcept override;
    bool update(GraphExecHandle exec, CommandList &&cmdlist, GraphUpdateError *error) noexcept override;
    bool update_kernel_node(GraphExecHandle exec, size_t node_index,
                            uint3 dispatch_size, luisa::span<const Argument> arguments) noexcept override;
    bool update_upload_node(GraphExecHandle exec, size_t node_index,
                            const void *data, size_t size) noexcept override;
    bool update_download_node(GraphExecHandle exec, size_t node_index,
                              void *data, size_t size) noexcept override;
    bool update_buffer_copy_node(GraphExecHandle exec, size_t node_index,
                                 uint64_t src_handle, size_t src_offset,
                                 uint64_t dst_handle, size_t dst_offset,
                                 size_t size) noexcept override;
    void set_node_enabled(GraphExecHandle exec, size_t node_index, bool enabled) noexcept override;
    bool is_node_enabled(GraphExecHandle exec, size_t node_index) const noexcept override;

    // -- Capabilities --
    [[nodiscard]] Capabilities capabilities() const noexcept override;

    // -- G7: diagnostics + priority --
    void set_node_priority(GraphExecHandle exec, size_t node_index, int priority) noexcept override;

    // -- G6: additional node types --
    bool add_memset_node(GraphHandle graph, uint64_t buffer_handle,
                         size_t offset, size_t size, uint32_t value,
                         size_t *node_index) noexcept override;
    bool add_host_node(GraphHandle graph, HostNodeCallback callback,
                       void *user_data, size_t *node_index) noexcept override;
    bool add_event_record_node(GraphHandle graph, uint64_t event_handle,
                               size_t *node_index) noexcept override;
    bool add_event_wait_node(GraphHandle graph, uint64_t event_handle,
                             size_t *node_index) noexcept override;
    bool add_child_graph_node(GraphHandle graph, GraphHandle child_graph,
                              size_t *node_index) noexcept override;
    bool add_dependency(GraphHandle graph, size_t from_index, size_t to_index) noexcept override;

    // -- SA-3: typed (PDL) edges --
    bool set_dependency_type(GraphHandle graph, size_t from_index,
                             size_t to_index, DependencyType type) noexcept override;

    // -- SA-2: conditional nodes --
    //
    // The body graphs are created by CUDA when the conditional node is added
    // and handed back through `body_graphs_out`; they are owned by that node
    // and must NOT be released with destroy_graph.
    ConditionalHandle create_conditional_handle(GraphHandle graph, uint32_t default_value,
                                                bool assign_default) noexcept override;
    bool add_conditional_node(GraphHandle graph, ConditionalType type, ConditionalHandle handle,
                              size_t num_bodies, luisa::span<GraphHandle> body_graphs_out,
                              size_t *node_index) noexcept override;
    bool build_body_graph(GraphHandle body, CommandList &&cmdlist) noexcept override;

    // -- SA-4: device-side launch --
    uint64_t get_device_launch_token(GraphExecHandle exec) noexcept override;

    // -- SA-4: device-updatable kernel nodes --
    bool mark_node_device_updatable(GraphHandle graph, size_t node_index) noexcept override;
    uint64_t get_device_node_handle(GraphExecHandle exec, size_t node_index) noexcept override;

    // -- SA-6: graph memory --
    //
    // There is deliberately no per-graph pool binding API (the CUDA driver has
    // none: only per-device graph memory attributes and the per-alloc-node
    // `poolProps` below), so the pool of a mem-alloc node is expressed entirely
    // by its `poolProps`/`location`.
    bool add_mem_alloc_node(GraphHandle graph, size_t size, size_t alignment,
                            size_t *node_index) noexcept override;
    bool add_mem_free_node(GraphHandle graph, uint64_t device_address,
                           size_t *node_index) noexcept override;
    bool get_graph_mem_alloc_address(GraphHandle graph, size_t node_index,
                                     uint64_t *device_address) noexcept override;
    bool get_mem_alloc_node_address(GraphExecHandle exec, size_t node_index,
                                    uint64_t *device_address) noexcept override;
};

}// namespace luisa::compute::cuda
