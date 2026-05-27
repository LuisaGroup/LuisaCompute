#pragma once

#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/rhi/argument.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

/// CUDA Graph extension: record CommandList once, instantiate, launch repeatedly
/// with low CPU overhead. Workflow: create_graph → instantiate → (update → launch)*
struct CudaGraphLaunch {
    uint64_t handle;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
};
struct CudaGraphUpload {
    uint64_t handle;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
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

    /// Capture a CommandList into a CUDA graph via stream capture.
    [[nodiscard]] virtual ResourceCreationInfo create_graph(CommandList &&cmdlist) noexcept = 0;

    virtual void destroy_graph(GraphHandle graph) noexcept = 0;

    /// @param flags OR of InstantiateFlag.
    [[nodiscard]] virtual ResourceCreationInfo instantiate(GraphHandle graph, InstantiateFlag flags) noexcept = 0;

    [[nodiscard]] ResourceCreationInfo instantiate(GraphHandle graph) noexcept {
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
    [[nodiscard]] virtual bool update(GraphExecHandle exec, CommandList &&cmdlist) noexcept = 0;

    // -- Individual node update (faster when few nodes change) --
    // Nodes indexed by position in the original CommandList.

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

}// namespace luisa::compute
