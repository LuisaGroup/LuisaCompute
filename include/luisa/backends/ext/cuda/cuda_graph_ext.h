#pragma once

#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/rhi/argument.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/string.h>

namespace luisa::compute {

/// CUDA Graph extension: record CommandList once, instantiate, launch repeatedly
/// with low CPU overhead. Workflow: create_graph → instantiate → (update → launch)*
///
/// ----------------------------------------------------------------------------
/// API CONTRACT (orchestrator-frozen; all phases implement against this surface)
///
/// Node indices. Everywhere a `node_index` appears it is the creation index of
/// a node-producing command/entry in its graph, i.e. the position in the order
/// entries were added to the graph (CommandList order for create_graph, then
/// the order of the explicit add_* calls). It is stable for the graph's whole
/// lifetime and is the same index space the update_* and set_node_enabled APIs
/// already use.
///
/// Refusal policy. Unsupported content must be refused loudly (warning +
/// failure return), never silently dropped. A graph must never be subtly
/// different from the semantics of the command list it was built from.
///
/// Version gates. Driver APIs newer than the CMake floor (CUDAToolkit 12.1)
/// are wrapped in `#if CUDA_VERSION >= 12xxx` compile guards plus runtime
/// capability checks (see CudaGraphCapabilities). On older toolkits the
/// corresponding APIs fail gracefully with a warning and `false`/invalid.
///
/// Composability rules (from CUDA_Graphs_Deep_Research.md section 3.4):
///  - CDP / device graph launch are forbidden inside conditional body graphs.
///  - Device graphs (DEVICE_LAUNCH) are host-update-only: no update() while a
///    device launch may be in flight; concurrent host+device launch of the
///    same exec is forbidden.
///  - Graphs containing device-updatable kernel nodes cannot use whole-graph
///    update() and cannot be instantiated more than once.
///  - Conditional node topology is frozen at instantiation; whole-graph
///    update() on a graph/exec containing conditional nodes must fail with
///    GraphUpdateErrorType::UNSUPPORTED.
///  - CUDA arrays are forbidden in memcpy nodes inside conditional bodies.
/// ----------------------------------------------------------------------------
struct CudaGraphLaunch {
    uint64_t handle;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
};
struct CudaGraphUpload {
    uint64_t handle;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
};
class CudaGraphExt;
class CudaGraphInstance {
    CudaGraphExt *_ext;
    ResourceCreationInfo _resource_info;
public:
    explicit CudaGraphInstance(CudaGraphExt *ext, ResourceCreationInfo const &resource_info) noexcept
        : _ext{ext}, _resource_info{resource_info} {}
    ~CudaGraphInstance() noexcept;
    CudaGraphInstance(CudaGraphInstance &&another) noexcept
        : _ext{another._ext}, _resource_info{another._resource_info} {
        another._resource_info.invalidate();
    }
    CudaGraphInstance &operator=(CudaGraphInstance &&rhs) noexcept {
        if (this != &rhs) {
            this->~CudaGraphInstance();
            _ext = rhs._ext;
            _resource_info = rhs._resource_info;
            rhs._resource_info.invalidate();
        }
        return *this;
    }
    CudaGraphInstance(CudaGraphInstance const &) noexcept = delete;
    CudaGraphInstance &operator=(CudaGraphInstance const &) noexcept = delete;
    [[nodiscard]] ResourceCreationInfo release() noexcept;
    [[nodiscard]] ResourceCreationInfo const &handle() const noexcept { return _resource_info; }
};

class CudaGraphExecInstance {
    CudaGraphExt *_ext;
    ResourceCreationInfo _resource_info;
public:
    explicit CudaGraphExecInstance(CudaGraphExt *ext, ResourceCreationInfo const &resource_info) noexcept
        : _ext{ext}, _resource_info{resource_info} {}
    ~CudaGraphExecInstance() noexcept;
    CudaGraphExecInstance(CudaGraphExecInstance &&another) noexcept
        : _ext{another._ext}, _resource_info{another._resource_info} {
        another._resource_info.invalidate();
    }
    CudaGraphExecInstance &operator=(CudaGraphExecInstance &&rhs) noexcept {
        if (this != &rhs) {
            this->~CudaGraphExecInstance();
            _ext = rhs._ext;
            _resource_info = rhs._resource_info;
            rhs._resource_info.invalidate();
        }
        return *this;
    }
    CudaGraphExecInstance(CudaGraphExecInstance const &) noexcept = delete;
    CudaGraphExecInstance &operator=(CudaGraphExecInstance const &) noexcept = delete;
    [[nodiscard]] ResourceCreationInfo release() noexcept;
    [[nodiscard]] ResourceCreationInfo const &handle() const noexcept { return _resource_info; }
};

class CudaGraphExt : public DeviceExtension {

public:
    static constexpr luisa::string_view name = "CudaGraphExt";

    using GraphHandle = uint64_t;    // graph template (pre-instantiation)
    using GraphExecHandle = uint64_t;// executable graph

    static constexpr auto invalid_handle = ~0ull;

    enum struct InstantiateFlag : uint32_t {
        INSTANTIATE_DEFAULT = 0,
        AUTO_FREE_ON_LAUNCH = 1u << 0u,// free graph allocs on each launch
        DEVICE_LAUNCH = 1u << 1u,      // allow device-side launch
    };

    // -- Lifecycle --

    /// Build a dependency DAG from a CommandList. Plain CUDA compute and
    /// direct-buffer Tile kernels preserve their respective launch ABIs.
    [[nodiscard]] virtual ResourceCreationInfo _create_graph(CommandList &&cmdlist) noexcept = 0;
    [[nodiscard]] CudaGraphInstance create_graph(CommandList &&cmdlist) {
        return CudaGraphInstance{this, _create_graph(std::move(cmdlist))};
    }

    virtual void destroy_graph(GraphHandle graph) noexcept = 0;

    /// @param flags OR of InstantiateFlag.
    [[nodiscard]] virtual ResourceCreationInfo _instantiate(GraphHandle graph, InstantiateFlag flags) noexcept = 0;

    [[nodiscard]] ResourceCreationInfo _instantiate(GraphHandle graph) noexcept {
        return _instantiate(graph, InstantiateFlag::INSTANTIATE_DEFAULT);
    }

    [[nodiscard]] CudaGraphExecInstance instantiate(GraphHandle graph, InstantiateFlag flags) {
        return CudaGraphExecInstance{this, _instantiate(graph, flags)};
    }
    [[nodiscard]] CudaGraphExecInstance instantiate(GraphHandle graph) {
        return instantiate(graph, InstantiateFlag::INSTANTIATE_DEFAULT);
    }

    /// Caller must sync the stream before destroying.
    virtual void destroy_exec(GraphExecHandle exec) noexcept = 0;

    // -- Execution --

    virtual void launch(GraphExecHandle exec, uint64_t stream_handle) noexcept = 0;

    /// Upload for device launch; no-op if not instantiated with DEVICE_LAUNCH.
    virtual void upload(GraphExecHandle exec, uint64_t stream_handle) noexcept = 0;

    /// Create a deferred upload command for batched submission.
    [[nodiscard]] static CudaGraphUpload upload(GraphExecHandle exec) noexcept {
        return CudaGraphUpload{exec};
    }

    /// Create a deferred launch command for batched submission.
    [[nodiscard]] static CudaGraphLaunch launch(GraphExecHandle exec) noexcept {
        return CudaGraphLaunch{exec};
    }

    // -- Whole-graph update --

    /// Update all params from a structurally-identical CommandList.
    /// @return false on topology mismatch → fall back to destroy_exec + instantiate.
    /// See the 3-arg overload for failure diagnostics.
    [[nodiscard]] bool update(GraphExecHandle exec, CommandList &&cmdlist) noexcept {
        return update(exec, std::move(cmdlist), nullptr);
    }

    // -- Individual node update (faster when few nodes change) --
    // Nodes indexed by position in the original CommandList.
    // Tile kernel nodes require update(CommandList); the raw kernel-node
    // update has no Tile buffer mapping and returns false for those nodes.

    [[nodiscard]] virtual bool update_kernel_node(
        GraphExecHandle exec, size_t node_index,
        uint3 dispatch_size, luisa::span<const Argument> arguments) noexcept = 0;

    [[nodiscard]] virtual bool update_upload_node(
        GraphExecHandle exec, size_t node_index,
        const void *data, size_t size) noexcept = 0;

    [[nodiscard]] virtual bool update_download_node(
        GraphExecHandle exec, size_t node_index,
        void *data, size_t size) noexcept = 0;

    [[nodiscard]] virtual bool update_buffer_copy_node(
        GraphExecHandle exec, size_t node_index,
        uint64_t src_handle, size_t src_offset,
        uint64_t dst_handle, size_t dst_offset,
        size_t size) noexcept = 0;

    // -- Node enable/disable --

    /// Disabled nodes become no-ops; params preserved across toggles.
    virtual void set_node_enabled(GraphExecHandle exec, size_t node_index, bool enabled) noexcept = 0;

    [[nodiscard]] virtual bool is_node_enabled(GraphExecHandle exec, size_t node_index) const noexcept = 0;

    // -- Capabilities (runtime + compile-time feature detection; G1-G7) --

    struct Capabilities {
        uint32_t cuda_version = 0u;           ///< Toolkit CUDA_VERSION (e.g. 12040)
        bool programmatic_edges = false;      ///< CUDA >= 12.4 headers (G2/PDL)
        bool device_launch = false;           ///< CUDA >= 12.0 + sm_90+ (G3)
        bool conditional_nodes = false;       ///< CUDA >= 12.3 (G1)
        bool device_updatable_nodes = false;  ///< CUDA >= 12.4 + sm_90+ (G4)
        bool graph_memory = false;            ///< CUDA >= 11.4 pool support (G5)
        bool node_priority = false;           ///< CUDA >= 12.8 (G7)
        bool switch_conditional = false;      ///< CUDA >= 12.8 (G1)
    };

    [[nodiscard]] virtual Capabilities capabilities() const noexcept = 0;

    // -- G7: whole-graph update diagnostics --

    enum struct GraphUpdateErrorType : uint32_t {
        NONE = 0,
        TOPOLOGY_CHANGED,   ///< node count / edge set changed
        NODE_TYPE_CHANGED,  ///< a node's type or function changed
        NODE_ATTR_CHANGED,  ///< attributes (e.g. body graph list) changed
        PARAMETERS_CHANGED, ///< params not updateable in place
        UNSUPPORTED,        ///< frozen-topology feature (conditional/device-updatable) or version gate
        INTERNAL,           ///< CUDA driver call failed
    };

    struct GraphUpdateError {
        GraphUpdateErrorType type = GraphUpdateErrorType::NONE;
        uint64_t error_node = ~0ull;///< node index in the new graph, ~0ull if unknown
        luisa::string detail;       ///< human-readable description (empty on success)
    };

    /// Like update(), but surfaces why an update failed. `error` may be null.
    [[nodiscard]] virtual bool update(GraphExecHandle exec, CommandList &&cmdlist,
                                      GraphUpdateError *error) noexcept = 0;

    /// CUDA 12.8+: adjust a kernel node's scheduling priority without
    /// re-instantiation. No-op warning on older toolkits.
    virtual void set_node_priority(GraphExecHandle exec, size_t node_index, int priority) noexcept = 0;

    // -- G6: additional node types (explicit graph construction) --
    // All return the new node's index in `*node_index` (may be null) and
    // attach no dependency edges; use add_dependency to wire them.

    using HostNodeCallback = void (*)(void *user_data) noexcept;

    /// 1-D device buffer fill: `value` written byte-wise over [offset, offset+size).
    [[nodiscard]] virtual bool add_memset_node(GraphHandle graph, uint64_t buffer_handle,
                                               size_t offset, size_t size, uint32_t value,
                                               size_t *node_index) noexcept = 0;
    /// Host callback node; runs on a driver thread during graph launch.
    /// `user_data` is stored by value and is the user's to keep alive.
    [[nodiscard]] virtual bool add_host_node(GraphHandle graph, HostNodeCallback callback,
                                             void *user_data, size_t *node_index) noexcept = 0;
    /// Record/wait on an LC event handle (CUDAEvent).
    [[nodiscard]] virtual bool add_event_record_node(GraphHandle graph, uint64_t event_handle,
                                                     size_t *node_index) noexcept = 0;
    [[nodiscard]] virtual bool add_event_wait_node(GraphHandle graph, uint64_t event_handle,
                                                   size_t *node_index) noexcept = 0;
    /// Inline a child graph template. The child must outlive the parent graph.
    [[nodiscard]] virtual bool add_child_graph_node(GraphHandle graph, GraphHandle child_graph,
                                                    size_t *node_index) noexcept = 0;

    /// Wire an explicit dependency between two existing nodes (default type).
    [[nodiscard]] virtual bool add_dependency(GraphHandle graph, size_t from_index,
                                              size_t to_index) noexcept = 0;

    // -- G2/SA-3: typed (PDL) edges --

    enum struct DependencyType : uint32_t {
        DEFAULT = 0,      ///< CU_GRAPH_DEPENDENCY_TYPE_DEFAULT
        PROGRAMMATIC = 1, ///< CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC; the downstream
                          ///< kernel must call grid_dependency_synchronize() before
                          ///< consuming the upstream kernel's output, the upstream
                          ///< may call trigger_programmatic_launch_completion().
    };

    /// Retype an existing edge (created by the DAG analysis or add_dependency):
    /// removes and re-adds it with CUgraphEdgeData. PROGRAMMATIC retypes the
    /// the upstream kernel's outgoing port to CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC;
    /// the downstream kernel must call grid_dependency_synchronize() before
    /// consuming the upstream kernel's output. Requires CUDA >= 12.4 headers
    /// (the annotated cuGraphAddDependencies); refused below that.
    [[nodiscard]] virtual bool set_dependency_type(GraphHandle graph, size_t from_index,
                                                   size_t to_index, DependencyType type) noexcept = 0;

    // -- G1/SA-2: conditional nodes --

    using ConditionalHandle = uint64_t;///< 0 = invalid

    enum struct ConditionalType : uint32_t {
        IF = 0,     ///< CU_GRAPH_COND_TYPE_IF: body[0] runs once when value != 0;
                    ///< with 2 bodies, body[1] is the ELSE branch (value == 0)
        WHILE = 1,  ///< CU_GRAPH_COND_TYPE_WHILE: body loops while value != 0
        SWITCH = 2, ///< CU_GRAPH_COND_TYPE_SWITCH (CUDA 12.8+): runs the value-th
                    ///< body graph (out-of-range value runs none)
    };

    /// Create a condition handle owned by `graph`. Handles are never destroyed
    /// (CUDA requirement) and do not reliably persist across launches unless a
    /// default is assigned. `assign_default` gives do-while semantics: the
    /// condition starts as `default_value` on every launch. Every conditional
    /// node should either set this or be preceded by a kernel that writes the
    /// handle via set_graph_conditional(). Requires CUDA >= 12.3.
    [[nodiscard]] virtual ConditionalHandle create_conditional_handle(
        GraphHandle graph, uint32_t default_value, bool assign_default) noexcept = 0;

    /// Add a conditional node to `graph`. CUDA creates `num_bodies` empty body
    /// graphs owned by the node; their handles are returned in
    /// `body_graphs_out` (they are NOT destroyable via destroy_graph and live
    /// as long as the node). Populate them with build_body_graph().
    /// num_bodies: WHILE = 1; IF = 1 or 2 (2 = with ELSE); SWITCH >= 1.
    /// Requires CUDA >= 12.3 (SWITCH: >= 12.8).
    [[nodiscard]] virtual bool add_conditional_node(
        GraphHandle graph, ConditionalType type, ConditionalHandle handle,
        size_t num_bodies, luisa::span<GraphHandle> body_graphs_out,
        size_t *node_index) noexcept = 0;

    /// Populate a conditional body graph (handle obtained from
    /// add_conditional_node) from a command list. Same node rules as
    /// create_graph plus the conditional-body restrictions from research doc
    /// section 2.3: no CUDA-array memcpys (refused here), device memory only.
    /// CALLER OBLIGATION (not enforceable from shader metadata today): CUDA
    /// forbids CDP and device-side graph launch inside conditional bodies, so
    /// the command list must not dispatch kernels that call
    /// graph_device_launch(); violating this is rejected by the driver at
    /// instantiation time. Requires CUDA >= 12.3.
    [[nodiscard]] virtual bool build_body_graph(GraphHandle body, CommandList &&cmdlist) noexcept = 0;

    // -- G3/SA-4: device-side launch --

    /// Returns the exec as a device token (uint64) to pass to kernels calling
    /// graph_launch(); 0 if the exec was not instantiated with DEVICE_LAUNCH.
    /// Device launch rules (research doc 3.2): launch only from within a graph,
    /// never host+device concurrently on the same exec, updates are host-only.
    [[nodiscard]] virtual uint64_t get_device_launch_token(GraphExecHandle exec) noexcept = 0;

    // -- G4/SA-4: device-updatable kernel nodes --

    /// Opt a kernel node (created from a CommandList dispatch) into device
    /// updatability. Must be called on the graph template before instantiate.
    /// The node can then never leave the graph and the whole-graph update() is
    /// permanently disabled for any exec instantiated from this graph.
    [[nodiscard]] virtual bool mark_node_device_updatable(GraphHandle graph, size_t node_index) noexcept = 0;

    /// Returns the cudaGraphDeviceNode_t for a device-updatable node of an
    /// instantiated exec as a uint64 to pass to kernels; 0 on failure.
    [[nodiscard]] virtual uint64_t get_device_node_handle(GraphExecHandle exec, size_t node_index) noexcept = 0;

    // -- G5/SA-6: graph memory nodes --

    /// Add a mem-alloc node. The driver assigns the device address at add time
    /// (it is fixed for the graph's whole life); query it with
    /// get_mem_alloc_node_address(). With AUTO_FREE_ON_LAUNCH the allocation is
    /// freed back to the pool on each launch.
    [[nodiscard]] virtual bool add_mem_alloc_node(GraphHandle graph, size_t size, size_t alignment,
                                                  size_t *node_index) noexcept = 0;
    /// Add a mem-free node for an address produced by a mem-alloc node of the
    /// same graph. DRIVER RULE (verified on 13.4): mem-free nodes must be added
    /// to the template BEFORE the first instantiation - the driver rejects them
    /// afterwards. The address is known at add_mem_alloc_node() time, so the
    /// supported pattern is alloc -> free -> add_dependency(alloc, free) ->
    /// instantiate, with no instantiation in between.
    [[nodiscard]] virtual bool add_mem_free_node(GraphHandle graph, uint64_t device_address,
                                                 size_t *node_index) noexcept = 0;
    /// Resolve the address of a mem-alloc node of a graph template (the address
    /// is assigned at add_mem_alloc_node() time; this works pre-instantiation).
    /// Distinct name because GraphHandle and GraphExecHandle are both uint64_t
    /// and cannot be overloaded.
    [[nodiscard]] virtual bool get_graph_mem_alloc_address(GraphHandle graph, size_t node_index,
                                                           uint64_t *device_address) noexcept = 0;
    /// Resolve the address of a mem-alloc node through an instantiated exec
    /// (same address as the template query; kept for the exec-first flow).
    [[nodiscard]] virtual bool get_mem_alloc_node_address(GraphExecHandle exec, size_t node_index,
                                                          uint64_t *device_address) noexcept = 0;
    /// NOTE: there is intentionally no per-graph pool-binding API: the CUDA
    /// driver only supports device-level graph memory attributes
    /// (cuDeviceGetGraphMemAttribute/cuDeviceSetGraphMemAttribute) and
    /// per-alloc-node poolProps (location/allocation type). Pools therefore
    /// stay per-device; use the default-pool mem nodes above.

protected:
    ~CudaGraphExt() noexcept = default;
};

inline void CudaGraphUpload::operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept {
    auto *ext = static_cast<CudaGraphExt *>(device->extension(CudaGraphExt::name));
    LUISA_ASSERT(ext, "CudaGraphExt not available on this device.");
    ext->upload(handle, stream_handle);
}

inline void CudaGraphLaunch::operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept {
    auto *ext = static_cast<CudaGraphExt *>(device->extension(CudaGraphExt::name));
    LUISA_ASSERT(ext, "CudaGraphExt not available on this device.");
    ext->launch(handle, stream_handle);
}

inline CudaGraphInstance::~CudaGraphInstance() noexcept {
    if (_resource_info.handle != CudaGraphExt::invalid_handle) {
        _ext->destroy_graph(_resource_info.handle);
    }
}

inline ResourceCreationInfo CudaGraphInstance::release() noexcept {
    auto info = _resource_info;
    _resource_info = {CudaGraphExt::invalid_handle, nullptr};
    return info;
}

inline CudaGraphExecInstance::~CudaGraphExecInstance() noexcept {
    if (_resource_info.handle != CudaGraphExt::invalid_handle) {
        _ext->destroy_exec(_resource_info.handle);
    }
}

inline ResourceCreationInfo CudaGraphExecInstance::release() noexcept {
    auto info = _resource_info;
    _resource_info = {CudaGraphExt::invalid_handle, nullptr};
    return info;
}

}// namespace luisa::compute
