#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/Device.h>
#include <Resource/Buffer.h>
#include <Shader/ComputeShader.h>
#include <Shader/RasterShader.h>
#include <Resource/SparseTexture.h>
#include <luisa/core/logging.h>
namespace lc::dx {
CommandBuffer::CommandBuffer(CommandBuffer &&v)
    : cmdList(std::move(v.cmdList)),
      alloc(v.alloc) {
    v.alloc = nullptr;
}
CommandAllocator *CommandBuffer::GetAlloc() const { return static_cast<CommandAllocator *>(alloc); }
void CommandBuffer::UpdateCommandBuffer(Device *device) {
    if (!device->deviceSettings) return;
    auto newCmdList = static_cast<ID3D12GraphicsCommandList4 *>(device->deviceSettings->BorrowCommandList(alloc->Type()));
    if (newCmdList) {
        cmdList = {newCmdList, false};
    }
}
CommandBuffer::CommandBuffer(
    Device *device,
    CommandAllocator *alloc)
    : alloc(alloc) {
    if (device->deviceSettings) {
        cmdList = {static_cast<ID3D12GraphicsCommandList4 *>(device->deviceSettings->BorrowCommandList(alloc->Type())), false};
    }
    if (!cmdList) {
        ThrowIfFailed(device->device->CreateCommandList(
            0,
            alloc->Type(),
            alloc->Allocator(),// Associated command allocator
            nullptr,           // Initial PipelineStateObject
            IID_PPV_ARGS(cmdList.GetAddressOf())));
    }
    if (cmdList.Contained())
        ThrowIfFailed(cmdList->Close());
    isOpened = false;
}
void CommandBufferBuilder::SetComputeResources(
    Shader const *s,
    vstd::span<const BindProperty> resources) {
    LUISA_ASSUME(resources.size() == s->Properties().size());
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
    LUISA_ASSUME(resources.size() == s->Properties().size());
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
void CommandBufferBuilder::DispatchCompute(
    ComputeShader const *cs,
    vstd::span<const uint3> dispatchSizes,
    uint constBindPos,
    vstd::span<const BindProperty> resources) {
    auto c = cb->cmdList.Get();
    c->SetComputeRootSignature(cs->RootSig());
    SetComputeResources(cs, resources);
    c->SetPipelineState(cs->Pso());
    auto calc = [](uint disp, uint thd) {
        return (disp + thd - 1) / thd;
    };
    uint3 blk = cs->BlockSize();
    uint kernelId = 0;
    for (auto dispatchId : dispatchSizes) {
        uint3 dispId = {
            calc(dispatchId.x, blk.x),
            calc(dispatchId.y, blk.y),
            calc(dispatchId.z, blk.z)};
        uint4 constValue{dispatchId.x, dispatchId.y, dispatchId.z, kernelId};
        c->SetComputeRoot32BitConstants(constBindPos, 4, &constValue, 0);
        ++kernelId;
        c->Dispatch(dispId.x, dispId.y, dispId.z);
    }
}
void CommandBufferBuilder::SetRasterShader(
    RasterShader const *s,
    ID3D12PipelineState *state,
    vstd::span<const BindProperty> resources) {
    auto c = cb->CmdList();
    c->SetGraphicsRootSignature(s->RootSig());
    c->SetPipelineState(state);
    SetRasterResources(s, resources);
}
void CommandBufferBuilder::DispatchComputeIndirect(
    ComputeShader const *cs,
    Buffer const &indirectBuffer,
    uint32_t indirectOffset,
    uint32_t maxIndirectCount,
    vstd::span<const BindProperty> resources) {
    auto c = cb->cmdList.Get();
    auto res = indirectBuffer.GetResource();
    size_t byteSize = indirectBuffer.GetByteSize();
    size_t cmdSize = (byteSize - 4) / ComputeShader::DispatchIndirectStride;
    LUISA_ASSUME(cmdSize >= 1);
    c->SetComputeRootSignature(cs->RootSig());
    SetComputeResources(cs, resources);
    c->SetPipelineState(cs->Pso());
    maxIndirectCount = std::min<uint>(maxIndirectCount, cmdSize - indirectOffset);
    // TODO
    c->ExecuteIndirect(
        cs->CmdSig(),
        maxIndirectCount,
        res,
        sizeof(uint) + static_cast<uint64_t>(indirectOffset) * ComputeShader::DispatchIndirectStride,
        res, 0);
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
    uint3 size) {
    if (Resource::IsBCtex(texture->Format())) {
        size.x /= 4;
        size.y /= 4;
    }
    auto pureLineSize = size.x * Resource::GetTexturePixelSize(texture->Format());
    auto lineSize = CalcConstantBufferByteSize(pureLineSize);
    return {
        size_t(pureLineSize * size.y * size.z),
        size_t(lineSize * size.y * size.z),
        size_t(lineSize),
        size_t(pureLineSize)};
}
void CommandBufferBuilder::CopyBufferTexture(
    BufferView const &buffer,
    TextureBase *texture,
    uint3 startCoord,
    uint3 size,
    uint targetMip,
    BufferTextureCopy ope,
    bool checkAlign) {
    auto c = cb->cmdList.Get();
    D3D12_TEXTURE_COPY_LOCATION sourceLocation;
    sourceLocation.pResource = buffer.buffer->GetResource();
    sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    sourceLocation.PlacedFootprint.Offset = buffer.offset;
    auto rowPitch = size.x / (Resource::IsBCtex(texture->Format()) ? 4ull : 1ull) * Resource::GetTexturePixelSize(texture->Format());
    if (checkAlign) {
        if ((rowPitch & (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) != 0) [[unlikely]] {
            LUISA_ERROR("Texture's row must be aligned as {}, current value row-size({}) x pixel-size({}) = {}.", D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, size.x / (Resource::IsBCtex(texture->Format()) ? 4ull : 1ull), Resource::GetTexturePixelSize(texture->Format()), rowPitch);
        }
        if((buffer.offset & (D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1)) != 0) [[unlikely]] {
            LUISA_ERROR("Buffer offset must be aligned as {}, current value is {}", D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, buffer.offset);
        }
    } else {
        rowPitch = CalcAlign(rowPitch, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    }
    sourceLocation.PlacedFootprint.Footprint =
        {
            (DXGI_FORMAT)texture->Format(),//DXGI_FORMAT Format;
            size.x,                        //uint Width;
            size.y,                        //uint Height;
            size.z,                        //uint Depth;
            static_cast<uint>(rowPitch)};

    D3D12_TEXTURE_COPY_LOCATION destLocation;
    destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destLocation.SubresourceIndex = targetMip;
    destLocation.pResource = texture->GetResource();
    if (ope == BufferTextureCopy::BufferToTexture) {
        c->CopyTextureRegion(
            &destLocation,
            startCoord.x, startCoord.y, startCoord.z,
            &sourceLocation,
            nullptr);
    } else {
        c->CopyTextureRegion(
            &sourceLocation,
            startCoord.x, startCoord.y, startCoord.z,
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
            LUISA_ASSUME(rBuffer.buffer->GetTag() == Resource::Tag::ReadbackBuffer);
            static_cast<ReadbackBuffer const *>(rBuffer.buffer)
                ->CopyData(
                    rBuffer.offset,
                    {reinterpret_cast<uint8_t *>(dst), size_t(rBuffer.byteSize)});
        });
}
void CommandBuffer::Reset() const {
    if (isOpened.exchange(true)) return;
    if (cmdList.Contained())
        ThrowIfFailed(cmdList->Reset(alloc->Allocator(), nullptr));
}
void CommandBuffer::Close() const {
    if (!isOpened.exchange(false)) return;
    if (cmdList.Contained())
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

}// namespace lc::dx
