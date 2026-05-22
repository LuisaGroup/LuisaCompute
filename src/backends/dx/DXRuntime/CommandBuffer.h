#pragma once
#include <Resource/BindProperty.h>
#include <Resource/TextureBase.h>
#include <DXRuntime/DxPtr.h>
#include <DXRuntime/EnhancedBarrierTracker.h>
namespace lc::dx {
class CommandAllocator;
class Resource;
class SparseTexture;
class ComputeShader;
class DescriptorHeap;
class Shader;
class RTShader;
class RasterShader;
class CommandBuffer;
class CommandQueue;
class WorkGraphProgram;
class CommandBufferBuilder {
    friend class CommandBuffer;

private:
    CommandBuffer const *_cb;
    CommandBufferBuilder(CommandBuffer const *_cb);
    CommandBufferBuilder(CommandBufferBuilder const &) = delete;
    CommandBufferBuilder(CommandBufferBuilder &&);
    void set_compute_resources(
        Shader const *s,
        vstd::span<const BindProperty> resources);
    void set_raster_resources(
        Shader const *s,
        vstd::span<const BindProperty> resources);

public:
    [[nodiscard]] CommandBuffer const *get_cb() const { return _cb; }

    void dispatch_compute(
        ComputeShader const *cs,
        uint3 dispatchId,
        vstd::span<const BindProperty> resources);
    void dispatch_compute(
        ComputeShader const *cs,
        vstd::span<const uint3> dispatchSizes,
        uint constBindPos,
        vstd::span<const BindProperty> resources);
    void set_raster_shader(
        RasterShader const *s,
        ID3D12PipelineState *state,
        vstd::span<const BindProperty> resources);
    void dispatch_compute_indirect(
        ComputeShader const *cs,
        Buffer const &indirectBuffer,
        uint32_t indirectOffset,
        uint32_t maxIndirectCount,
        vstd::span<const BindProperty> resources);
    void dispatch_work_graph(
        WorkGraphProgram const *program,
        D3D12_DISPATCH_GRAPH_DESC const &dispatchDesc,
        vstd::span<const BindProperty> resources);
    /*void DispatchRT(
        RTShader const *rt,
        uint3 dispatchId,
        vstd::span<const BindProperty> resources);
    void DispatchRT(
        RTShader const *rt,
        uint3 dispatchId,
        std::initializer_list<BindProperty> resources) {
        DispatchRT(
            rt,
            dispatchId,
            vstd::span<const BindProperty>{resources.begin(), resources.size()});
    }*/
    void copy_buffer(
        Buffer const *src,
        Buffer const *dst,
        uint64 srcOffset,
        uint64 dstOffset,
        uint64 byteSize);
    void copy_texture(
        TextureBase const *source, uint sourceSlice, uint sourceMipLevel,
        TextureBase const *dest, uint destSlice, uint destMipLevel);
    void upload(BufferView const &buffer, void const *src);
    void readback(BufferView const &buffer, void *dst);
    BufferView get_temp_buffer(size_t size, size_t align = 0);
    enum class BufferTextureCopy {
        BufferToTexture,
        TextureToBuffer,
    };
    void copy_buffer_texture(
        BufferView const &buffer,
        TextureBase *texture,
        uint3 startCoord,
        uint3 size,
        uint targetMip,
        BufferTextureCopy ope,
        bool checkAlign);
    struct CopyInfo {
        size_t bufferSize;
        size_t alignedBufferSize;
        size_t stepSize;
        size_t copySize;
    };
    static CopyInfo get_copy_texture_buffer_size(
        TextureBase *texture,
        uint3 size);
    ~CommandBufferBuilder();
};
class CommandBuffer : public vstd::IOperatorNewBase {
    friend class CommandQueue;
    friend class CommandBufferBuilder;
    friend class CommandAllocator;
    mutable std::atomic_bool _is_opened;
    void _reset() const;
    void _close() const;
    DxPtr<ID3D12GraphicsCommandList4> _cmd_list;
    CommandAllocator *_alloc;

public:
    void update_command_buffer(Device *device);
    ID3D12GraphicsCommandList4 *cmd_list() const { return _cmd_list.Get(); }
    ComPtr<ID3D12GraphicsCommandList7> next_cmd_list() const {
        ComPtr<ID3D12GraphicsCommandList7> cmdlist;
        if (_cmd_list->QueryInterface(IID_PPV_ARGS(&cmdlist)) != S_OK) {
            return nullptr;
        }
        return cmdlist;
    }
    bool contained_cmd_list() const { return _cmd_list.Contained(); }
    CommandBuffer(
        Device *device,
        CommandAllocator *alloc);
    CommandAllocator *get_alloc() const;
    ~CommandBuffer();
    CommandBuffer(CommandBuffer &&v);
    CommandBufferBuilder build() const { return CommandBufferBuilder(this); }
    KILL_COPY_CONSTRUCT(CommandBuffer)
};
struct GraphicsCmdlistBarrierCallback : public BarrierCallback {
    ID3D12GraphicsCommandList *cmdlist;
    GraphicsCmdlistBarrierCallback(CommandBufferBuilder &builder) : cmdlist(builder.get_cb()->cmd_list()) {}
    void Barrier(
        UINT32 NumBarrierGroups,
        const D3D12_BARRIER_GROUP *pBarrierGroups) {
        ComPtr<ID3D12GraphicsCommandList7> new_cmdlist;
        if (cmdlist->QueryInterface(IID_PPV_ARGS(&new_cmdlist)) != S_OK) {
            LUISA_ERROR("Bad command-list");
        }
        new_cmdlist->Barrier(NumBarrierGroups, pBarrierGroups);
    }
    void ResourceBarrier(
        UINT NumBarriers,
        D3D12_RESOURCE_BARRIER *pBarriers) {
        cmdlist->ResourceBarrier(NumBarriers, pBarriers);
    }
};

}// namespace lc::dx
