#pragma once
#include <cstdio>
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
    bool is_res_in_bindless(uint64_t bindless_handle, uint64_t resource_handle) const noexcept {
        return reinterpret_cast<BindlessArray *>(bindless_handle)->IsPtrInBindless(resource_handle);
    }
    Usage get_usage(uint64_t shader_handle, size_t argument_index) const noexcept {
        if (shader_handle == 0) {
            // Raster draw commands may not carry a shader handle through the
            // reorder visitor. Return READ_WRITE as conservative default.
            return Usage::READ_WRITE;
        }
        auto shader = reinterpret_cast<Shader *>(shader_handle);
        return shader->args()[argument_index].var_usage;
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
        reinterpret_cast<BindlessArray *>(handle)->Bind(luisa::span{
            reinterpret_cast<const BindlessArrayUpdateCommand::Texture2DModification *>(modifications.data()),
            modifications.size()});
    }
    luisa::span<const Argument> shader_bindings(uint64_t handle) const noexcept {
        return reinterpret_cast<ComputeShader const *>(handle)->arg_bindings();
    }
    void lock_bindless(uint64_t bindless_handle) const noexcept {
        reinterpret_cast<BindlessArray *>(bindless_handle)->Lock();
    }
    void unlock_bindless(uint64_t bindless_handle) const noexcept {
        reinterpret_cast<BindlessArray *>(bindless_handle)->Unlock();
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
