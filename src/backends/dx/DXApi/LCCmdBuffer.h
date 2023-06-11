#pragma once
#include <luisa/vstl/common.h>
#include <luisa/runtime/device.h>
#include <DXRuntime/Device.h>
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandBuffer.h>
#include <luisa/runtime/command_list.h>
#include <DXRuntime/ResourceStateTracker.h>
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
        auto cs = reinterpret_cast<ComputeShader *>(shader_handle);
        return cs->Args()[argument_index].varUsage;
    }
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Modification> modifications) const noexcept {
        reinterpret_cast<BindlessArray *>(handle)->Bind(modifications);
    }
    luisa::span<const Argument> shader_bindings(uint64_t handle) const noexcept {
        return reinterpret_cast<ComputeShader const *>(handle)->ArgBindings();
    }
    void lock_bindless(uint64_t bindless_handle) const noexcept {
        reinterpret_cast<BindlessArray *>(bindless_handle)->Lock();
    }
    void unlock_bindless(uint64_t bindless_handle) const noexcept {
        reinterpret_cast<BindlessArray *>(bindless_handle)->Unlock();
    }
};
class LCCmdBuffer final : public CmdQueueBase{
protected:
    ResourceStateTracker tracker;
    ReorderFuncTable reorderFuncTable;
    CommandReorderVisitor<ReorderFuncTable, false> reorder;
    vstd::vector<BindProperty> bindProps;
    vstd::vector<ButtomCompactCmd> updateAccel;
    vstd::vector<D3D12_VERTEX_BUFFER_VIEW> vbv;
    std::mutex mtx;

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
        CommandList &&cmdList,
        size_t maxAlloc);
    void Sync();
    void Present(
        LCSwapChain *swapchain,
        TextureBase *rt,
        size_t maxAlloc);
    void CompressBC(
        TextureBase *rt,
        luisa::compute::BufferView<uint> const &result,
        bool isHDR,
        float alphaImportance,
        GpuAllocator *allocator,
        size_t maxAlloc);
};

}// namespace lc::dx
