#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/Device.h>
#include <Resource/Buffer.h>
#include <Shader/ComputeShader.h>
#include <Shader/RasterShader.h>
#include <Shader/WorkGraphProgram.h>
#include <Resource/SparseTexture.h>
#include <luisa/core/logging.h>
namespace lc::dx {
CommandBuffer::CommandBuffer(CommandBuffer &&v)
    : _cmd_list(std::move(v._cmd_list)),
      _alloc(v._alloc) {
    v._alloc = nullptr;
}
CommandAllocator *CommandBuffer::get_alloc() const { return static_cast<CommandAllocator *>(_alloc); }
void CommandBuffer::update_command_buffer(Device *device) {
    if (!device->device_settings) return;
    auto newCmdList = static_cast<ID3D12GraphicsCommandList4 *>(device->device_settings->BorrowCommandList(_alloc->type()));
    if (newCmdList) {
        _cmd_list = {newCmdList, false};
    }
}
CommandBuffer::CommandBuffer(
    Device *device,
    CommandAllocator *alloc)
    : _alloc(alloc) {
    if (device->device_settings) {
        _cmd_list = {static_cast<ID3D12GraphicsCommandList4 *>(device->device_settings->BorrowCommandList(_alloc->type())), false};
    }
    if (!_cmd_list) {
        ThrowIfFailed(device->device->CreateCommandList(
            0,
            _alloc->type(),
            _alloc->allocator(),// Associated command allocator
            nullptr,           // Initial PipelineStateObject
            IID_PPV_ARGS(_cmd_list.GetAddressOf())));
    }
    if (_cmd_list.Contained())
        ThrowIfFailed(_cmd_list->Close());
    _is_opened = false;
}
void CommandBufferBuilder::set_compute_resources(
    Shader const *s,
    vstd::span<const BindProperty> resources) {
    LUISA_ASSUME(resources.size() == s->properties().size());
    for (auto i : vstd::range(static_cast<int64>(resources.size()))) {
        resources[i].visit(
            [&](auto &&b) {
                s->set_compute_resource(
                    i,
                    this,
                    b);
            });
    }
}
void CommandBufferBuilder::set_raster_resources(
    Shader const *s,
    vstd::span<const BindProperty> resources) {
    LUISA_ASSUME(resources.size() == s->properties().size());
    for (auto i : vstd::range(static_cast<int64>(resources.size()))) {
        resources[i].visit(
            [&](auto &&b) {
                s->set_raster_resource(
                    i,
                    this,
                    b);
            });
    }
}
void CommandBufferBuilder::dispatch_compute(
    ComputeShader const *cs,
    uint3 dispatchId,
    vstd::span<const BindProperty> resources) {
    auto calc = [](uint disp, uint thd) {
        return (disp + thd - 1) / thd;
    };
    uint3 blk = cs->block_size();
    uint3 dispId = {
        calc(dispatchId.x, blk.x),
        calc(dispatchId.y, blk.y),
        calc(dispatchId.z, blk.z)};
    if (dispId.x > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) [[unlikely]] {
        LUISA_ERROR("Dispatch size X {} out of range {}", dispId.x, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
    }
    if (dispId.y > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) [[unlikely]] {
        LUISA_ERROR("Dispatch size Y {} out of range {}", dispId.y, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
    }
    if (dispId.z > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) [[unlikely]] {
        LUISA_ERROR("Dispatch size Z {} out of range {}", dispId.z, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
    }
    auto c = _cb->_cmd_list.Get();
    c->SetComputeRootSignature(cs->root_sig());
    set_compute_resources(cs, resources);
    c->SetPipelineState(cs->pso());
    c->Dispatch(dispId.x, dispId.y, dispId.z);
}
void CommandBufferBuilder::dispatch_compute(
    ComputeShader const *cs,
    vstd::span<const uint3> dispatchSizes,
    uint constBindPos,
    vstd::span<const BindProperty> resources) {
    auto c = _cb->_cmd_list.Get();
    c->SetComputeRootSignature(cs->root_sig());
    set_compute_resources(cs, resources);
    c->SetPipelineState(cs->pso());
    auto calc = [](uint disp, uint thd) {
        return (disp + thd - 1) / thd;
    };
    uint3 blk = cs->block_size();
    uint kernelId = 0;
    for (auto dispatchId : dispatchSizes) {
        uint3 dispId = {
            calc(dispatchId.x, blk.x),
            calc(dispatchId.y, blk.y),
            calc(dispatchId.z, blk.z)};
        if (dispId.x > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) [[unlikely]] {
            LUISA_ERROR("Dispatch size X {} out of range {}", dispId.x, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
        }
        if (dispId.y > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) [[unlikely]] {
            LUISA_ERROR("Dispatch size Y {} out of range {}", dispId.y, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
        }
        if (dispId.z > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) [[unlikely]] {
            LUISA_ERROR("Dispatch size Z {} out of range {}", dispId.z, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
        }
        uint4 constValue{dispatchId.x, dispatchId.y, dispatchId.z, kernelId};
        c->SetComputeRoot32BitConstants(constBindPos, 4, &constValue, 0);
        ++kernelId;
        c->Dispatch(dispId.x, dispId.y, dispId.z);
    }
}
void CommandBufferBuilder::set_raster_shader(
    RasterShader const *s,
    ID3D12PipelineState *state,
    vstd::span<const BindProperty> resources) {
    auto c = _cb->cmd_list();
    c->SetGraphicsRootSignature(s->root_sig());
    c->SetPipelineState(state);
    set_raster_resources(s, resources);
}
void CommandBufferBuilder::dispatch_compute_indirect(
    ComputeShader const *cs,
    Buffer const &indirectBuffer,
    uint32_t indirectOffset,
    uint32_t maxIndirectCount,
    vstd::span<const BindProperty> resources) {
    auto c = _cb->_cmd_list.Get();
    auto res = indirectBuffer.GetResource();
    size_t byteSize = indirectBuffer.GetByteSize();
    size_t cmdSize = (byteSize - 4) / ComputeShader::kDispatchIndirectStride;
    LUISA_ASSUME(cmdSize >= 1);
    c->SetComputeRootSignature(cs->root_sig());
    set_compute_resources(cs, resources);
    c->SetPipelineState(cs->pso());
    maxIndirectCount = std::min<uint32_t>(maxIndirectCount, static_cast<uint32_t>(cmdSize - indirectOffset));
    // TODO
    c->ExecuteIndirect(
        cs->cmd_sig(),
        maxIndirectCount,
        res,
        sizeof(uint) + static_cast<uint64_t>(indirectOffset) * ComputeShader::kDispatchIndirectStride,
        res, 0);
}
/*void CommandBufferBuilder::DispatchRT(
    RTShader const *rt,
    uint3 dispatchId,
    vstd::span<const BindProperty> resources) {
    auto c = _cb->_cmd_list.Get();
    c->SetComputeRootSignature(rt->root_sig());
    SetResources(rt, resources);
    rt->DispatchRays(
        *this,
        dispatchId.x,
        dispatchId.y,
        dispatchId.z);
}*/
void CommandBufferBuilder::copy_buffer(
    Buffer const *src,
    Buffer const *dst,
    uint64 srcOffset,
    uint64 dstOffset,
    uint64 byteSize) {
    auto c = _cb->_cmd_list.Get();
    c->CopyBufferRegion(
        dst->GetResource(),
        dstOffset,
        src->GetResource(),
        srcOffset,
        byteSize);
}
CommandBufferBuilder::CopyInfo CommandBufferBuilder::get_copy_texture_buffer_size(
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
void CommandBufferBuilder::copy_buffer_texture(
    BufferView const &buffer,
    TextureBase *texture,
    uint3 startCoord,
    uint3 size,
    uint targetMip,
    BufferTextureCopy ope,
    bool checkAlign) {
    auto c = _cb->_cmd_list.Get();
    D3D12_TEXTURE_COPY_LOCATION sourceLocation;
    sourceLocation.pResource = buffer.buffer->GetResource();
    sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    sourceLocation.PlacedFootprint.Offset = buffer.offset;
    auto rowPitch = size.x / (Resource::IsBCtex(texture->Format()) ? 4u : 1u) * Resource::GetTexturePixelSize(texture->Format());
    if (checkAlign) {
        if ((rowPitch & (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) != 0) [[unlikely]] {
            LUISA_ERROR("Texture's row must be aligned as {}, current value row-size({}) x pixel-size({}) = {}.", D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, size.x / (Resource::IsBCtex(texture->Format()) ? 4u : 1u), Resource::GetTexturePixelSize(texture->Format()), rowPitch);
        }
        if ((buffer.offset & (D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1)) != 0) [[unlikely]] {
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
void CommandBufferBuilder::upload(BufferView const &buffer, void const *src) {
    auto uBuffer = _cb->get_alloc()->get_temp_upload_buffer(buffer.byteSize);
    static_cast<UploadBuffer const *>(uBuffer.buffer)
        ->CopyData(
            uBuffer.offset,
            {reinterpret_cast<uint8_t const *>(src), size_t(uBuffer.byteSize)});
    copy_buffer(
        uBuffer.buffer,
        buffer.buffer,
        uBuffer.offset,
        buffer.offset,
        buffer.byteSize);
}
void CommandBufferBuilder::copy_texture(
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
    _cb->_cmd_list->CopyTextureRegion(
        &destLocation,
        0, 0, 0,
        &sourceLocation,
        nullptr);
}
BufferView CommandBufferBuilder::get_temp_buffer(size_t size, size_t align) {
    return _cb->get_alloc()->get_temp_default_buffer(size, align);
}
void CommandBufferBuilder::readback(BufferView const &buffer, void *dst) {
    auto rBuffer = _cb->get_alloc()->get_temp_readback_buffer(buffer.byteSize);
    copy_buffer(
        buffer.buffer,
        rBuffer.buffer,
        buffer.offset,
        rBuffer.offset,
        buffer.byteSize);
    _cb->get_alloc()->execute_after_complete(
        [rBuffer, dst] {
            LUISA_ASSUME(rBuffer.buffer->get_tag() == Resource::Tag::ReadbackBuffer);
            static_cast<ReadbackBuffer const *>(rBuffer.buffer)
                ->CopyData(
                    rBuffer.offset,
                    {reinterpret_cast<uint8_t *>(dst), size_t(rBuffer.byteSize)});
        });
}
void CommandBuffer::_reset() const {
    if (_is_opened.exchange(true)) return;
    if (_cmd_list.Contained())
        ThrowIfFailed(_cmd_list->Reset(_alloc->allocator(), nullptr));
}
void CommandBuffer::_close() const {
    if (!_is_opened.exchange(false)) return;
    if (_cmd_list.Contained())
        ThrowIfFailed(_cmd_list->Close());
}
CommandBufferBuilder::CommandBufferBuilder(CommandBuffer const *_cb)
    : _cb(_cb) {
    _cb->_reset();
}
CommandBufferBuilder::~CommandBufferBuilder() {
    if (_cb)
        _cb->_close();
}
CommandBufferBuilder::CommandBufferBuilder(CommandBufferBuilder &&v)
    : _cb(v._cb) {
    v._cb = nullptr;
}

CommandBuffer::~CommandBuffer() {
    _close();
}

void CommandBufferBuilder::dispatch_work_graph(
    WorkGraphProgram const *program,
    D3D12_DISPATCH_GRAPH_DESC const &dispatchDesc,
    vstd::span<const BindProperty> resources) {
    ComPtr<ID3D12GraphicsCommandList10> cmdList10;
    ThrowIfFailed(_cb->cmd_list()->QueryInterface(IID_PPV_ARGS(&cmdList10)));

    cmdList10->SetComputeRootSignature(program->root_sig());
    set_compute_resources(program, resources);

    D3D12_SET_PROGRAM_DESC setProgramDesc{};
    setProgramDesc.Type = D3D12_PROGRAM_TYPE_WORK_GRAPH;
    setProgramDesc.WorkGraph.ProgramIdentifier = program->program_id();
    setProgramDesc.WorkGraph.Flags = D3D12_SET_WORK_GRAPH_FLAG_INITIALIZE;
    if (program->backing_memory()) {
        setProgramDesc.WorkGraph.BackingMemory.StartAddress = program->backing_memory()->GetGPUVirtualAddress();
        setProgramDesc.WorkGraph.BackingMemory.SizeInBytes = program->backing_memory_size();
    }
    setProgramDesc.WorkGraph.NodeLocalRootArgumentsTable = {};
    cmdList10->SetProgram(&setProgramDesc);
    cmdList10->DispatchGraph(&dispatchDesc);
}

}// namespace lc::dx
