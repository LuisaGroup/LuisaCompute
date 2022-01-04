#pragma vengine_package vengine_directx
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/Device.h>
#include <Resource/Buffer.h>
#include <Shader/ComputeShader.h>
#include <Shader/RTShader.h>
namespace toolhub::directx {
void CommandBufferBuilder::SetDescHeap(DescriptorHeap const *heap) {
    if (currentDesc == heap) return;
    currentDesc = heap;
    ID3D12DescriptorHeap *h = heap->GetHeap();
    cb->cmdList->SetDescriptorHeaps(1, &h);
}
ID3D12GraphicsCommandList4 *CommandBufferBuilder::CmdList() const { return cb->cmdList.Get(); }

CommandBuffer::CommandBuffer(
    Device *device,
    CommandAllocator *alloc)
    : alloc(alloc) {
    ThrowIfFailed(device->device->CreateCommandList(
        0,
        alloc->Type(),
        alloc->Allocator(),// Associated command allocator
        nullptr,           // Initial PipelineStateObject
        IID_PPV_ARGS(&cmdList)));
    ThrowIfFailed(cmdList->Close());
}
void CommandBufferBuilder::SetResources(
    Shader *s,
    vstd::span<const BindProperty> resources) {
    for (auto &&r : resources) {
        r.prop.visit(
            [&](auto &&b) {
                s->SetComputeResource(
                    r.name,
                    this,
                    b);
            });
    }
}
void CommandBufferBuilder::DispatchCompute(
    ComputeShader *cs,
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
    SetResources(cs, resources);
    c->SetPipelineState(cs->Pso());
    c->Dispatch(dispId.x, dispId.y, dispId.z);
}
void CommandBufferBuilder::DispatchRT(
    RTShader *rt,
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
}
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
void CommandBufferBuilder::CopyBufferToTexture(
    BufferView const &sourceBuffer,
    TextureBase *texture,
    uint targetMip,
    uint width,
    uint height,
    uint depth) {
    auto c = cb->cmdList.Get();
    D3D12_TEXTURE_COPY_LOCATION sourceLocation;
    sourceLocation.pResource = sourceBuffer.buffer->GetResource();
    sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    sourceLocation.PlacedFootprint.Offset = sourceBuffer.offset;
    sourceLocation.PlacedFootprint.Footprint =
        {
            (DXGI_FORMAT)texture->Format(),//DXGI_FORMAT Format;
            width,                         //uint Width;
            height,                        //uint Height;
            depth,                         //uint Depth;
            static_cast<uint>(
                CalcConstantBufferByteSize(
                    texture->Width() * Resource::GetTexturePixelSize(texture->Format())))//uint RowPitch;
        };
    D3D12_TEXTURE_COPY_LOCATION destLocation;
    destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destLocation.SubresourceIndex = targetMip;
    destLocation.pResource = texture->GetResource();
    c->CopyTextureRegion(
        &destLocation,
        0, 0, 0,
        &sourceLocation,
        nullptr);
}
void CommandBufferBuilder::Upload(BufferView const &buffer, void const *src) {
    auto uBuffer = cb->alloc->GetTempUploadBuffer(buffer.byteSize);
    static_cast<UploadBuffer const *>(uBuffer.buffer)
        ->CopyData(
            uBuffer.offset,
            {reinterpret_cast<vbyte const *>(src), size_t(uBuffer.byteSize)});
    CopyBuffer(
        uBuffer.buffer,
        buffer.buffer,
        uBuffer.offset,
        buffer.offset,
        buffer.byteSize);
}
BufferView CommandBufferBuilder::GetTempBuffer(size_t size) {
    return cb->alloc->GetTempDefaultBuffer(size);
}

void CommandBufferBuilder::Readback(BufferView const &buffer, void *dst) {
    auto rBuffer = cb->alloc->GetTempReadbackBuffer(buffer.byteSize);
    CopyBuffer(
        buffer.buffer,
        rBuffer.buffer,
        buffer.offset,
        rBuffer.offset,
        buffer.byteSize);
    cb->alloc->ExecuteAfterComplete(
        [rBuffer, dst] {
            static_cast<ReadbackBuffer const *>(rBuffer.buffer)
                ->CopyData(
                    rBuffer.offset,
                    {reinterpret_cast<vbyte *>(dst), size_t(rBuffer.byteSize)});
        });
}
void CommandBuffer::Reset() const {
    ThrowIfFailed(cmdList->Reset(alloc->Allocator(), nullptr));
}
void CommandBuffer::Close() const {
    ThrowIfFailed(cmdList->Close());
}

void CommandBuffer::Dispose() {
    alloc->CollectBuffer(this);
}
CommandBufferBuilder::CommandBufferBuilder(CommandBuffer const *cb)
    : cb(cb) {
    cb->Reset();
}
CommandBufferBuilder::~CommandBufferBuilder() {
    cb->Close();
}
CommandBuffer::~CommandBuffer() {
}

}// namespace toolhub::directx