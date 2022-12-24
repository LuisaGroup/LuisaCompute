
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/Device.h>
#include <Resource/Buffer.h>
#include <Shader/ComputeShader.h>
#include <Shader/RasterShader.h>
namespace toolhub::directx {
ID3D12GraphicsCommandList4 *CommandBufferBuilder::CmdList() const { return cb->cmdList.Get(); }
CommandBuffer::CommandBuffer(CommandBuffer &&v)
    : cmdList(std::move(v.cmdList)),
      alloc(v.alloc) {
    v.alloc = nullptr;
}
CommandAllocator *CommandBuffer::GetAlloc() const { return static_cast<CommandAllocator *>(alloc); }

CommandBuffer::CommandBuffer(
    Device *device,
    CommandAllocatorBase *alloc)
    : alloc(alloc) {
    ThrowIfFailed(device->device->CreateCommandList(
        0,
        alloc->Type(),
        alloc->Allocator(),// Associated command allocator
        nullptr,           // Initial PipelineStateObject
        IID_PPV_ARGS(&cmdList)));
    ThrowIfFailed(cmdList->Close());
    isOpened = false;
}
void CommandBufferBuilder::SetComputeResources(
    Shader const *s,
    vstd::span<const BindProperty> resources) {
    assert(resources.size() == s->Properties().size());
    for (auto i : vstd::range(resources.size())) {
        resources[i].visit(
            [&](auto &&b) {
                s->SetComputeResource(
                    i,
                    this,
                    b);
            });
    }
}
void CommandBufferBuilder::SetRasterResources(
    Shader const *s,
    vstd::span<const BindProperty> resources) {
    assert(resources.size() == s->Properties().size());
    for (auto i : vstd::range(resources.size())) {
        resources[i].visit(
            [&](auto &&b) {
                s->SetRasterResource(
                    i,
                    this,
                    b);
            });
    }
}
void CommandBufferBuilder::DispatchCompute(
    ComputeShader const *cs,
    uint3 dispatchId,
    vstd::span<const BindProperty> resources) {
    auto calc = [](uint disp, uint thd) {
        return (disp + thd - 1) / thd;
    };
    uint3 blk = cs->BlockSize();
    uint3 dispId = {
        calc(dispatchId.x, blk.x),
        calc(dispatchId.y, blk.y),
        calc(dispatchId.z, blk.z)};
    auto c = cb->cmdList.Get();
    c->SetComputeRootSignature(cs->RootSig());
    SetComputeResources(cs, resources);
    c->SetPipelineState(cs->Pso());
    c->Dispatch(dispId.x, dispId.y, dispId.z);
}
void CommandBufferBuilder::SetRasterShader(
    RasterShader const *s,
    vstd::span<const BindProperty> resources) {
    auto c = CmdList();
    c->SetPipelineState(s->Pso());
    c->SetGraphicsRootSignature(s->RootSig());
    SetRasterResources(s, resources);
}
void CommandBufferBuilder::DispatchComputeIndirect(
    ComputeShader const *cs,
    Buffer const &indirectBuffer,
    vstd::span<const BindProperty> resources) {
    auto c = cb->cmdList.Get();
    auto res = indirectBuffer.GetResource();
    size_t byteSize = indirectBuffer.GetByteSize();
    size_t cmdSize = (byteSize - 4) / 28;
    assert(cmdSize >= 1);
    c->SetComputeRootSignature(cs->RootSig());
    SetComputeResources(cs, resources);
    c->SetPipelineState(cs->Pso());
    c->ExecuteIndirect(
        cs->CmdSig(),
        cmdSize,
        res,
        sizeof(uint),
        res,
        0);
}
/*void CommandBufferBuilder::DispatchRT(
    RTShader const *rt,
    uint3 dispatchId,
    vstd::span<const BindProperty> resources) {
    auto c = cb->cmdList.Get();
    c->SetComputeRootSignature(rt->RootSig());
    SetResources(rt, resources);
    rt->DispatchRays(
        *this,
        dispatchId.x,
        dispatchId.y,
        dispatchId.z);
}*/
void CommandBufferBuilder::CopyBuffer(
    Buffer const *src,
    Buffer const *dst,
    uint64 srcOffset,
    uint64 dstOffset,
    uint64 byteSize) {
    auto c = cb->cmdList.Get();
    c->CopyBufferRegion(
        dst->GetResource(),
        dstOffset,
        src->GetResource(),
        srcOffset,
        byteSize);
}
CommandBufferBuilder::CopyInfo CommandBufferBuilder::GetCopyTextureBufferSize(
    TextureBase *texture,
    uint targetMip) {
    uint width = texture->Width();
    uint height = texture->Height();
    uint depth = texture->Depth();
    if (Resource::IsBCtex(texture->Format())) {
        width /= 4;
        height /= 4;
    }
    auto GetValue = [&](uint &v) {
        v = std::max<uint>(1, v >> targetMip);
    };
    GetValue(width);
    GetValue(height);
    GetValue(depth);
    auto pureLineSize = width * Resource::GetTexturePixelSize(texture->Format());
    auto lineSize = CalcConstantBufferByteSize(pureLineSize);
    return {
        size_t(pureLineSize * height * depth),
        size_t(lineSize * height * depth),
        size_t(lineSize),
        size_t(pureLineSize)};
}
void CommandBufferBuilder::CopyBufferTexture(
    BufferView const &buffer,
    TextureBase *texture,
    uint targetMip,
    BufferTextureCopy ope) {
    uint width = texture->Width();
    uint height = texture->Height();
    uint depth = texture->Depth();

    auto GetValue = [&](uint &v) {
        v = std::max<uint>(1, v >> targetMip);
    };
    GetValue(width);
    GetValue(height);
    GetValue(depth);
    auto c = cb->cmdList.Get();
    D3D12_TEXTURE_COPY_LOCATION sourceLocation;
    sourceLocation.pResource = buffer.buffer->GetResource();
    sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    sourceLocation.PlacedFootprint.Offset = buffer.offset;
    sourceLocation.PlacedFootprint.Footprint =
        {
            (DXGI_FORMAT)texture->Format(),//DXGI_FORMAT Format;
            width,                         //uint Width;
            height,                        //uint Height;
            depth,                         //uint Depth;
            static_cast<uint>(
                CalcConstantBufferByteSize(
                    width / (Resource::IsBCtex(texture->Format()) ? 4 : 1) * Resource::GetTexturePixelSize(texture->Format())))//uint RowPitch;
        };
    D3D12_TEXTURE_COPY_LOCATION destLocation;
    destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destLocation.SubresourceIndex = targetMip;
    destLocation.pResource = texture->GetResource();
    if (ope == BufferTextureCopy::BufferToTexture) {
        c->CopyTextureRegion(
            &destLocation,
            0, 0, 0,
            &sourceLocation,
            nullptr);
    } else {
        c->CopyTextureRegion(
            &sourceLocation,
            0, 0, 0,
            &destLocation,
            nullptr);
    }
}
void CommandBufferBuilder::Upload(BufferView const &buffer, void const *src) {
    auto uBuffer = cb->GetAlloc()->GetTempUploadBuffer(buffer.byteSize);
    static_cast<UploadBuffer const *>(uBuffer.buffer)
        ->CopyData(
            uBuffer.offset,
            {reinterpret_cast<uint8_t const *>(src), size_t(uBuffer.byteSize)});
    CopyBuffer(
        uBuffer.buffer,
        buffer.buffer,
        uBuffer.offset,
        buffer.offset,
        buffer.byteSize);
}
void CommandBufferBuilder::CopyTexture(
    TextureBase const *source, uint sourceSlice, uint sourceMipLevel,
    TextureBase const *dest, uint destSlice, uint destMipLevel) {
    if (source->Dimension() == TextureDimension::Tex2D) sourceSlice = 0;
    if (dest->Dimension() == TextureDimension::Tex2D) destSlice = 0;
    D3D12_TEXTURE_COPY_LOCATION sourceLocation;
    sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    sourceLocation.SubresourceIndex = sourceSlice * source->Mip() + sourceMipLevel;
    D3D12_TEXTURE_COPY_LOCATION destLocation;
    destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destLocation.SubresourceIndex = destSlice * dest->Mip() + destMipLevel;
    sourceLocation.pResource = source->GetResource();
    destLocation.pResource = dest->GetResource();
    cb->cmdList->CopyTextureRegion(
        &destLocation,
        0, 0, 0,
        &sourceLocation,
        nullptr);
}
BufferView CommandBufferBuilder::GetTempBuffer(size_t size, size_t align) {
    return cb->GetAlloc()->GetTempDefaultBuffer(size, align);
}
void CommandBufferBuilder::Readback(BufferView const &buffer, void *dst) {
    auto rBuffer = cb->GetAlloc()->GetTempReadbackBuffer(buffer.byteSize);
    CopyBuffer(
        buffer.buffer,
        rBuffer.buffer,
        buffer.offset,
        rBuffer.offset,
        buffer.byteSize);
    cb->GetAlloc()->ExecuteAfterComplete(
        [rBuffer, dst] {
            static_cast<ReadbackBuffer const *>(rBuffer.buffer)
                ->CopyData(
                    rBuffer.offset,
                    {reinterpret_cast<uint8_t *>(dst), size_t(rBuffer.byteSize)});
        });
}
void CommandBuffer::Reset() const {
    if (isOpened.exchange(true)) return;
    ThrowIfFailed(cmdList->Reset(alloc->Allocator(), nullptr));
}
void CommandBuffer::Close() const {
    if (!isOpened.exchange(false)) return;
    ThrowIfFailed(cmdList->Close());
}
CommandBufferBuilder::CommandBufferBuilder(CommandBuffer const *cb)
    : cb(cb) {
    cb->Reset();
}
CommandBufferBuilder::~CommandBufferBuilder() {
    if (cb)
        cb->Close();
}
CommandBufferBuilder::CommandBufferBuilder(CommandBufferBuilder &&v)
    : cb(v.cb) {
    v.cb = nullptr;
}

CommandBuffer::~CommandBuffer() {
    Close();
}

}// namespace toolhub::directx