#pragma once
#include <Resource/BindProperty.h>
#include <Resource/TextureBase.h>
#include <DXRuntime/DxPtr.h>
namespace toolhub::directx {
class CommandAllocator;
class CommandAllocatorBase;
class Resource;
class ComputeShader;
class DescriptorHeap;
class Shader;
class RTShader;
class RasterShader;
class CommandBuffer;
class CommandQueue;
class CommandBufferBuilder {
    friend class CommandBuffer;

private:
    CommandBuffer const *cb;
    CommandBufferBuilder(CommandBuffer const *cb);
    CommandBufferBuilder(CommandBufferBuilder const &) = delete;
    CommandBufferBuilder(CommandBufferBuilder &&);
    void SetComputeResources(
        Shader const *s,
        vstd::span<const BindProperty> resources);
    void SetRasterResources(
        Shader const *s,
        vstd::span<const BindProperty> resources);
    DescriptorHeap const *currentDesc = nullptr;

public:
    CommandBuffer const *GetCB() const { return cb; }

    void DispatchCompute(
        ComputeShader const *cs,
        uint3 dispatchId,
        vstd::span<const BindProperty> resources);
    void SetRasterShader(
        RasterShader const *s,
        vstd::span<const BindProperty> resources);
    void DispatchComputeIndirect(
        ComputeShader const *cs,
        Buffer const &indirectBuffer,
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
    void CopyBuffer(
        Buffer const *src,
        Buffer const *dst,
        uint64 srcOffset,
        uint64 dstOffset,
        uint64 byteSize);
    void CopyTexture(
        TextureBase const *source, uint sourceSlice, uint sourceMipLevel,
        TextureBase const *dest, uint destSlice, uint destMipLevel);
    void Upload(BufferView const &buffer, void const *src);
    void Readback(BufferView const &buffer, void *dst);
    BufferView GetTempBuffer(size_t size, size_t align = 0);
    enum class BufferTextureCopy {
        BufferToTexture,
        TextureToBuffer,
    };
    void CopyBufferTexture(
        BufferView const &buffer,
        TextureBase *texture,
        uint targetMip,
        BufferTextureCopy ope);
    struct CopyInfo {
        size_t bufferSize;
        size_t alignedBufferSize;
        size_t stepSize;
        size_t copySize;
    };
    static CopyInfo GetCopyTextureBufferSize(
        TextureBase *texture,
        uint targetMip);
    ~CommandBufferBuilder();
};
class CommandBuffer : public vstd::IOperatorNewBase {
    friend class CommandQueue;
    friend class CommandBufferBuilder;
    friend class CommandAllocator;
    friend class CommandAllocatorBase;
    mutable std::atomic_bool isOpened;
    void Reset() const;
    void Close() const;
    DxPtr<ID3D12GraphicsCommandList4> cmdList;
    CommandAllocatorBase *alloc;

public:
    ID3D12GraphicsCommandList4 *CmdList() const { return cmdList.Get(); }
    bool ContainedCmdList() const { return cmdList.Contained(); }
    CommandBuffer(
        Device *device,
        CommandAllocatorBase *alloc);
    CommandAllocator *GetAlloc() const;
    ~CommandBuffer();
    CommandBuffer(CommandBuffer &&v);
    CommandBufferBuilder Build() const { return CommandBufferBuilder(this); }
    KILL_COPY_CONSTRUCT(CommandBuffer)
};

}// namespace toolhub::directx