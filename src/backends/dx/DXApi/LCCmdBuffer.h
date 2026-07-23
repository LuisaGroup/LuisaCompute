#pragma once
#include <luisa/vstl/common.h>
#include <luisa/runtime/device.h>
#include <DXRuntime/Device.h>
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandBuffer.h>
#include <luisa/runtime/command_list.h>
#include <DXRuntime/EnhancedBarrierTrackerImpl.h>
#include <DXRuntime/EnhancedBarrierTrackerBackup.h>
#include "../../common/command_reorder_visitor.h"
#include <Resource/BindlessArray.h>
#include <Shader/ComputeShader.h>
#include <Resource/BottomAccel.h>
#include <luisa/runtime/buffer.h>
#include <DXApi/CmdQueueBase.h>
namespace lc::dx {
using namespace luisa::compute;
class RenderTexture;
class LCSwapChain;
class BottomAccel;
struct ButtomCompactCmd {
    vstd::variant<BottomAccel *, TopAccel *> accel;
    size_t offset;
    size_t size;
};
struct ReorderFuncTable {
    [[nodiscard]] uint64_t canonical_buffer_handle(
        uint64_t handle) const noexcept {
        LUISA_ASSERT(handle != 0u,
                     "Cannot canonicalize a null DirectX buffer handle.");
        auto resource = reinterpret_cast<Resource const *>(handle);
        auto native_resource = resource->GetResource();
        LUISA_ASSERT(native_resource != nullptr,
                     "DirectX buffer has no underlying ID3D12Resource.");
        return static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(native_resource));
    }
    [[nodiscard]] uint64_t canonical_texture_handle(
        uint64_t handle) const noexcept {
        LUISA_ASSERT(handle != 0u,
                     "Cannot canonicalize a null DirectX texture handle.");
        auto resource = reinterpret_cast<Resource const *>(handle);
        auto native_resource = resource->GetResource();
        LUISA_ASSERT(native_resource != nullptr,
                     "DirectX texture has no underlying ID3D12Resource.");
        return static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(native_resource));
    }
    void traverse_bindless_resources(
        uint64_t bindless_handle,
        ReorderBindlessResourceVisitor visitor) const noexcept {
        auto bindless = reinterpret_cast<BindlessArray *>(bindless_handle);
        bindless->Lock();
        auto unlocker = vstd::scope_exit(
            [bindless]() noexcept { bindless->Unlock(); });
        bindless->TraversePendingResources(
            [&](uint64_t resource_handle) noexcept {
                auto resource = reinterpret_cast<Resource const *>(
                    resource_handle);
                auto tag = resource->get_tag();
                auto is_buffer =
                    tag == Resource::Tag::UploadBuffer ||
                    tag == Resource::Tag::ReadbackBuffer ||
                    tag == Resource::Tag::DefaultBuffer ||
                    tag == Resource::Tag::SparseBuffer ||
                    tag == Resource::Tag::ExternalBuffer;
                auto is_texture =
                    tag == Resource::Tag::RenderTexture ||
                    tag == Resource::Tag::SparseTexture ||
                    tag == Resource::Tag::DepthBuffer ||
                    tag == Resource::Tag::ExternalTexture ||
                    tag == Resource::Tag::ExternalDepth;
                LUISA_ASSERT(
                    is_buffer || is_texture,
                    "DirectX bindless reorder snapshot contains an invalid resource.");
                visitor(resource_handle, is_buffer);
            });
    }
    Usage get_usage(uint64_t shader_handle, size_t argument_index) const noexcept {
        auto shader = reinterpret_cast<Shader const *>(shader_handle);
        auto arguments = shader->args();
        LUISA_ASSERT(
            argument_index < arguments.size(),
            "DirectX command reordering requested shader argument {} from "
            "a saved table containing only {} entries.",
            argument_index, arguments.size());
        return arguments[argument_index].var_usage;
    }
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Modification> modifications) const noexcept {
        reinterpret_cast<BindlessArray *>(handle)->Bind(modifications);
    }
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::BufferModification> modifications) const noexcept {
        reinterpret_cast<BindlessArray *>(handle)->Bind(modifications);
    }
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture2DModification> modifications) const noexcept {
        reinterpret_cast<BindlessArray *>(handle)->Bind(modifications);
    }
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture3DModification> modifications) const noexcept {
        reinterpret_cast<BindlessArray *>(handle)->Bind(modifications);
    }
    luisa::span<const Argument> shader_bindings(uint64_t handle) const noexcept {
        return reinterpret_cast<ComputeShader const *>(handle)->arg_bindings();
    }
    luisa::span<const Argument> raster_shader_bindings(uint64_t) const noexcept {
        // The DirectX raster shader ABI currently has no captured-binding list.
        return {};
    }
    template<typename Func>
    void traverse_arguments(
        CustomDispatchCommand const *cmd,
        Func &&func) const {
        // TODO
        cmd->traverse_arguments(func);
    }
};
class LCCmdBuffer final : public CmdQueueBase {
protected:
    // ResourceStateTracker tracker;
    luisa::unique_ptr<EnhancedBarrierTracker> tracker;
    ReorderFuncTable reorderFuncTable;
    CommandReorderVisitor<ReorderFuncTable, true> reorder;
    vstd::vector<BindProperty> bindProps;
    vstd::vector<ButtomCompactCmd> updateAccel;
    vstd::vector<D3D12_VERTEX_BUFFER_VIEW> vbv;
    luisa::spin_mutex mtx;

    vstd::vector<std::pair<size_t, size_t>> argVecs;
    vstd::vector<uint8_t> argBuffer;
    vstd::vector<BottomAccelData> bottomAccelDatas;
    vstd::fixed_vector<std::pair<size_t, size_t>, 4> accelOffset;

public:
    CommandQueue queue;
    LCCmdBuffer(
        Device *device,
        GpuAllocator *resourceAllocator,
        D3D12_COMMAND_LIST_TYPE type);
    void Execute(
        vstd::span<const luisa::unique_ptr<Command>> commands,
        luisa::vector<luisa::move_only_function<void()>> &&funcs,
        vstd::span<const SwapchainPresent> presents,
        size_t maxAlloc);
    void Sync();
    void Present(
        LCSwapChain *swapchain,
        TextureBase *img,
        uint mip,
        size_t maxAlloc);
    void CompressBC(
        TextureBase *rt,
        uint level,
        luisa::compute::BufferView<uint> const &result,
        bool isHDR,
        float alphaImportance,
        GpuAllocator *allocator,
        size_t maxAlloc);
};

}// namespace lc::dx
