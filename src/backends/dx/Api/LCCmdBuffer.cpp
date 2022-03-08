#pragma vengine_package vengine_directx
#include <Api/LCCmdBuffer.h>
#include <Api/LCDevice.h>
#include <runtime/command.h>
#include <runtime/command_buffer.h>
#include <Codegen/DxCodegen.h>
#include <Shader/ComputeShader.h>
#include <Shader/RTShader.h>
#include <Resource/RenderTexture.h>
#include <Resource/BottomAccel.h>
#include <Resource/TopAccel.h>
#include <Resource/BindlessArray.h>
namespace toolhub::directx {
class LCPreProcessVisitor : public CommandVisitor {
public:
    CommandBufferBuilder *bd;
    ResourceStateTracker *stateTracker;
    vstd::vector<Resource const *> backState;
    vstd::vector<std::pair<size_t, size_t>> argVecs;
    vstd::vector<vbyte> argBuffer;
    void UniformAlign(size_t align) {
        argBuffer.resize(CalcAlign(argBuffer.size(), align));
    }
    template<typename T>
    void EmplaceData(T const &data) {
        size_t sz = argBuffer.size();
        argBuffer.resize(sz + sizeof(T));
        using PlaceHolder = std::aligned_storage_t<sizeof(T), 1>;
        *reinterpret_cast<PlaceHolder *>(argBuffer.data() + sz) =
            *reinterpret_cast<PlaceHolder const *>(&data);
    }
    template<typename T>
    void EmplaceData(T const *data, size_t size) {
        size_t sz = argBuffer.size();
        auto byteSize = size * sizeof(T);
        argBuffer.resize(sz + byteSize);
        memcpy(argBuffer.data() + sz, data, byteSize);
    }
    struct Visitor {
        LCPreProcessVisitor *self;
        Function f;
        Variable const *arg;

        void operator()(uint uid, ShaderDispatchCommand::BufferArgument const &bf) {
            auto res = reinterpret_cast<Buffer const *>(bf.handle);
            if (((uint)f.variable_usage(arg->uid()) & (uint)Usage::WRITE) != 0) {
                self->stateTracker->RecordState(
                    res,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            } else {
                self->stateTracker->RecordState(
                    res,
                    VEngineShaderResourceState);
            }
            ++arg;
        }
        void operator()(uint uid, ShaderDispatchCommand::TextureArgument const &bf) {
            vstd::string name;
            CodegenUtility::GetVariableName(
                *arg,
                name);
            RenderTexture *rt = reinterpret_cast<RenderTexture *>(bf.handle);
            //UAV
            if (((uint)f.variable_usage(arg->uid()) & (uint)Usage::WRITE) != 0) {
                self->stateTracker->RecordState(
                    rt,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            // SRV
            else {
                self->stateTracker->RecordState(
                    rt,
                    VEngineShaderResourceRTState);
            }
            ++arg;
        }
        void operator()(uint uid, ShaderDispatchCommand::BindlessArrayArgument const &bf) {
            auto arr = reinterpret_cast<BindlessArray *>(bf.handle);
            for (auto &&i : self->stateTracker->WriteStateMap()) {
                if (arr->IsPtrInBindless(reinterpret_cast<size_t>(i.first))) {
                    self->backState.emplace_back(i.first);
                }
            }
            for (auto &&i : self->backState) {
                self->stateTracker->RecordState(i);
            }
            self->backState.clear();
            ++arg;
        }
        void operator()(uint uid, vstd::span<std::byte const> bf) {
            if (bf.size() < 4) {
                bool v = (bool)bf[0];
                uint value = v ? std::numeric_limits<uint>::max() : 0;
                self->EmplaceData(value);

            } else {
                auto type = arg->type();
                if (type->is_vector() && type->dimension() == 3)
                    self->EmplaceData(bf.data(), 12);
                else
                    self->EmplaceData(bf.data(), bf.size());
            }
            ++arg;
        }
        void operator()(uint uid, ShaderDispatchCommand::AccelArgument const &bf) {
            auto accel = reinterpret_cast<TopAccel *>(bf.handle);
            if (accel->GetInstBuffer()) {
                if (((uint)f.variable_usage(arg->uid()) & (uint)Usage::WRITE) != 0) {
                    self->stateTracker->RecordState(
                        accel->GetInstBuffer(),
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                } else {
                    self->stateTracker->RecordState(
                        accel->GetInstBuffer(),
                        VEngineShaderResourceState);
                }
            }
            ++arg;
        }
    };
    void visit(const BufferUploadCommand *cmd) noexcept override {
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        stateTracker->RecordState(bf.buffer, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const BufferDownloadCommand *cmd) noexcept override {
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        stateTracker->RecordState(bf.buffer, VEngineShaderResourceState);
    }
    void visit(const BufferCopyCommand *cmd) noexcept override {
        auto srcBf = reinterpret_cast<Buffer const *>(cmd->src_handle());
        auto dstBf = reinterpret_cast<Buffer const *>(cmd->dst_handle());
        stateTracker->RecordState(srcBf, VEngineShaderResourceState);
        stateTracker->RecordState(dstBf, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const BufferToTextureCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        stateTracker->RecordState(
            rt,
            D3D12_RESOURCE_STATE_COPY_DEST);

        stateTracker->RecordState(
            bf,
            VEngineShaderResourceState);
    }
    void visit(const TextureUploadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->handle());
        stateTracker->RecordState(
            rt,
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const TextureDownloadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->handle());
        stateTracker->RecordState(
            rt,
            VEngineShaderResourceRTState);
    }
    void visit(const TextureCopyCommand *cmd) noexcept override {
        auto src = reinterpret_cast<RenderTexture *>(cmd->src_handle());
        auto dst = reinterpret_cast<RenderTexture *>(cmd->dst_handle());
        stateTracker->RecordState(
            src,
            VEngineShaderResourceRTState);
        stateTracker->RecordState(
            dst,
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const TextureToBufferCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        stateTracker->RecordState(
            rt,
            VEngineShaderResourceRTState);
        stateTracker->RecordState(
            bf,
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const ShaderDispatchCommand *cmd) noexcept override {
        size_t beforeSize = argBuffer.size();
        EmplaceData((vbyte const *)vstd::get_rvalue_ptr(cmd->dispatch_size()), 12);
        cmd->decode(Visitor{this, cmd->kernel(), cmd->kernel().arguments().data()});
        UniformAlign(16);
        size_t afterSize = argBuffer.size();
        argVecs.emplace_back(beforeSize, afterSize - beforeSize);
    }
    void visit(const AccelUpdateCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        accel->PreProcess(
            *stateTracker,
            *bd);
    }
    void visit(const AccelBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        accel->PreProcess(
            *stateTracker,
            *bd);
    }
    void visit(const MeshUpdateCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        accel->PreProcessStates(
            *bd,
            *stateTracker);
    }
    void visit(const MeshBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        accel->PreProcessStates(
            *bd,
            *stateTracker);
    }
    void visit(const BindlessArrayUpdateCommand *cmd) noexcept override {
        auto arr = reinterpret_cast<BindlessArray *>(cmd->handle());
        arr->PreProcessStates(
            *bd,
            *stateTracker);
    };
};
class LCCmdVisitor : public CommandVisitor {
public:
    Device *device;
    CommandBufferBuilder *bd;
    ResourceStateTracker *stateTracker;
    BufferView argBuffer;
    std::pair<size_t, size_t> *bufferVec;
    vstd::vector<BindProperty> bindProps;
    void visit(const BufferUploadCommand *cmd) noexcept override {
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        bd->Upload(bf, cmd->data());
        stateTracker->RecordState(
            bf.buffer,
            VEngineShaderResourceState);
    }
    void visit(const BufferDownloadCommand *cmd) noexcept override {
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        bd->Readback(
            bf,
            cmd->data());
    }
    void visit(const BufferCopyCommand *cmd) noexcept override {
        auto srcBf = reinterpret_cast<Buffer const *>(cmd->src_handle());
        auto dstBf = reinterpret_cast<Buffer const *>(cmd->dst_handle());
        bd->CopyBuffer(
            srcBf,
            dstBf,
            cmd->src_offset(),
            cmd->dst_offset(),
            cmd->size());
        stateTracker->RecordState(
            dstBf,
            VEngineShaderResourceState);
    }
    void visit(const BufferToTextureCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        bd->CopyBufferTexture(
            BufferView{bf},
            rt,
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::BufferToTexture);
        stateTracker->RecordState(
            rt,
            VEngineShaderResourceRTState);
    }
    struct Visitor {
        LCCmdVisitor *self;
        Function f;
        Variable const *arg;

        void operator()(uint uid, ShaderDispatchCommand::BufferArgument const &bf) {
            vstd::string name;
            CodegenUtility::GetVariableName(
                *arg,
                name);
            auto res = reinterpret_cast<Buffer const *>(bf.handle);

            self->bindProps.emplace_back(
                std::move(name),
                BufferView(res, bf.offset));
            ++arg;
        }
        void operator()(uint uid, ShaderDispatchCommand::TextureArgument const &bf) {
            vstd::string name;
            CodegenUtility::GetVariableName(
                *arg,
                name);
            RenderTexture *rt = reinterpret_cast<RenderTexture *>(bf.handle);
            //UAV
            if (((uint)f.variable_usage(arg->uid()) & (uint)Usage::WRITE) != 0) {
                self->bindProps.emplace_back(
                    std::move(name),
                    DescriptorHeapView(
                        self->device->globalHeap.get(),
                        rt->GetGlobalUAVIndex(bf.level)));
            }
            // SRV
            else {
                self->bindProps.emplace_back(
                    std::move(name),
                    DescriptorHeapView(
                        self->device->globalHeap.get(),
                        rt->GetGlobalSRVIndex(bf.level)));
            }
            ++arg;
        }
        void operator()(uint uid, ShaderDispatchCommand::BindlessArrayArgument const &bf) {
            auto arr = reinterpret_cast<BindlessArray *>(bf.handle);
            auto res = arr->Buffer();

            vstd::string name;
            CodegenUtility::GetVariableName(
                *arg,
                name);
            self->bindProps.emplace_back(
                std::move(name),
                BufferView(res, 0));
            ++arg;
        }
        void operator()(uint uid, ShaderDispatchCommand::AccelArgument const &bf) {
            auto accel = reinterpret_cast<TopAccel *>(bf.handle);
            vstd::string name;
            vstd::string instName;
            CodegenUtility::GetVariableName(
                *arg,
                name);
            instName << name << "Inst"sv;
            if ((static_cast<uint>(f.variable_usage(arg->uid())) & static_cast<uint>(Usage::WRITE)) == 0) {
                self->bindProps.emplace_back(
                    std::move(name),
                    accel);
            }
            self->bindProps.emplace_back(
                std::move(instName),
                BufferView(accel->GetInstBuffer()));
            ++arg;
        }
        void operator()(uint uid, vstd::span<std::byte const> bf) {
            ++arg;
        }
    };
    void visit(const ShaderDispatchCommand *cmd) noexcept override {
        bindProps.clear();
        auto shader = reinterpret_cast<Shader const *>(cmd->handle());
        cmd->decode(Visitor{this, cmd->kernel(), cmd->kernel().arguments().data()});
        auto &&tempBuffer = *bufferVec;
        bufferVec++;
        bindProps.emplace_back("_Global"sv, BufferView(argBuffer.buffer, argBuffer.offset + tempBuffer.first, tempBuffer.second));
        DescriptorHeapView globalHeapView(DescriptorHeapView(device->globalHeap.get()));
        bindProps.emplace_back("_BindlessTex"sv, globalHeapView);
        bindProps.emplace_back("_BindlessTex3D"sv, globalHeapView);
        for (auto i : vstd::range(shader->BindlessCount())) {
            bindProps.emplace_back(std::move(vstd::string("bdls") + vstd::to_string(i)), globalHeapView);
        }
        bindProps.emplace_back("samplers"sv, DescriptorHeapView(device->samplerHeap.get()));
        switch (shader->GetTag()) {
            case Shader::Tag::ComputeShader: {
                auto cs = static_cast<ComputeShader const *>(shader);
                bd->DispatchCompute(
                    cs,
                    cmd->dispatch_size(),
                    bindProps);
            } break;
            case Shader::Tag::RayTracingShader: {
                auto rts = static_cast<RTShader const *>(shader);
                bd->DispatchRT(
                    rts,
                    cmd->dispatch_size(),
                    bindProps);
            } break;
        }
    }
    void visit(const TextureUploadCommand *cmd) noexcept override {

        auto rt = reinterpret_cast<RenderTexture *>(cmd->handle());
        auto copyInfo = CommandBufferBuilder::GetCopyTextureBufferSize(
            rt,
            cmd->level());
        auto bfView = bd->GetCB()->GetAlloc()->GetTempUploadBuffer(copyInfo.alignedBufferSize, 512);
        auto uploadBuffer = static_cast<UploadBuffer const *>(bfView.buffer);
        if (copyInfo.bufferSize == copyInfo.alignedBufferSize) {
            uploadBuffer->CopyData(
                bfView.offset,
                {reinterpret_cast<vbyte const *>(cmd->data()),
                 bfView.byteSize});
        } else {
            size_t bufferOffset = bfView.offset;
            size_t leftedSize = copyInfo.bufferSize;
            auto dataPtr = reinterpret_cast<vbyte const *>(cmd->data());
            while (leftedSize > 0) {
                uploadBuffer->CopyData(
                    bufferOffset,
                    {dataPtr, copyInfo.copySize});
                dataPtr += copyInfo.copySize;
                leftedSize -= copyInfo.copySize;
                bufferOffset += copyInfo.stepSize;
            }
        }
        bd->CopyBufferTexture(
            bfView,
            rt,
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::BufferToTexture);
        stateTracker->RecordState(
            rt,
            VEngineShaderResourceRTState);
    }
    void visit(const TextureDownloadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->handle());
        auto copyInfo = CommandBufferBuilder::GetCopyTextureBufferSize(
            rt,
            cmd->level());
        auto bfView = bd->GetCB()->GetAlloc()->GetTempReadbackBuffer(copyInfo.alignedBufferSize, 512);

        if (copyInfo.alignedBufferSize == copyInfo.bufferSize) {
            bd->GetCB()->GetAlloc()->ExecuteAfterComplete(
                [bfView,
                 ptr = cmd->data()] {
                    auto rbBuffer = static_cast<ReadbackBuffer const *>(bfView.buffer);
                    size_t bufferOffset = bfView.offset;
                    rbBuffer->CopyData(
                        bufferOffset,
                        {reinterpret_cast<vbyte *>(ptr), bfView.byteSize});
                });
        } else {
            auto rbBuffer = static_cast<ReadbackBuffer const *>(bfView.buffer);
            size_t bufferOffset = bfView.offset;
            bd->GetCB()->GetAlloc()->ExecuteAfterComplete(
                [rbBuffer,
                 bufferOffset,
                 dataPtr = reinterpret_cast<vbyte *>(cmd->data()),
                 copyInfo]() mutable {
                    while (copyInfo.bufferSize > 0) {

                        rbBuffer->CopyData(
                            bufferOffset,
                            {dataPtr, copyInfo.copySize});
                        dataPtr += copyInfo.copySize;
                        copyInfo.bufferSize -= copyInfo.copySize;
                        bufferOffset += copyInfo.stepSize;
                    }
                });
        }
        bd->CopyBufferTexture(
            bfView,
            rt,
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::TextureToBuffer);
    }
    void visit(const TextureCopyCommand *cmd) noexcept override {
        auto src = reinterpret_cast<RenderTexture *>(cmd->src_handle());
        auto dst = reinterpret_cast<RenderTexture *>(cmd->dst_handle());
        bd->CopyTexture(
            src,
            0,
            cmd->src_level(),
            dst,
            0,
            cmd->dst_level());
        stateTracker->RecordState(
            dst,
            VEngineShaderResourceRTState);
    }
    void visit(const TextureToBufferCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        bd->CopyBufferTexture(
            BufferView{bf},
            rt,
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::TextureToBuffer);
        stateTracker->RecordState(
            bf,
            VEngineShaderResourceState);
    }
    void visit(const AccelUpdateCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        accel->Build(
            *stateTracker,
            *bd, true);
    }
    void visit(const AccelBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        accel->Build(
            *stateTracker,
            *bd, false);
    }
    void visit(const MeshUpdateCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        accel->UpdateStates(
            *bd,
            *stateTracker,
            //TODO: driver's bug, do not support update mesh's accel
            false);
    }
    void visit(const MeshBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        accel->UpdateStates(
            *bd,
            *stateTracker,
            false);
    }
    void visit(const BindlessArrayUpdateCommand *cmd) noexcept override {
        auto arr = reinterpret_cast<BindlessArray *>(cmd->handle());
        arr->UpdateStates(
            *bd,
            *stateTracker);
    }
};

LCCmdBuffer::LCCmdBuffer(
    Device *device,
    IGpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : queue(
          device,
          resourceAllocator,
          type),
      device(device) {
}
void LCCmdBuffer::Execute(vstd::span<CommandList const> const &c, size_t maxAlloc) {
    auto allocator = queue.CreateAllocator(maxAlloc);
    {
        LCPreProcessVisitor ppVisitor;
        ppVisitor.stateTracker = &tracker;
        LCCmdVisitor visitor;
        visitor.device = device;
        visitor.stateTracker = &tracker;
        auto cmdBuffer = allocator->GetBuffer();
        auto cmdBuilder = cmdBuffer->Build();
        visitor.bd = &cmdBuilder;
        ppVisitor.bd = &cmdBuilder;
        ID3D12DescriptorHeap *h[2] = {
            device->globalHeap->GetHeap(),
            device->samplerHeap->GetHeap()};
        cmdBuilder.CmdList()->SetDescriptorHeaps(vstd::array_count(h), h);

        for (auto &&lst : c) {
            // Clear caches
            ppVisitor.argVecs.clear();
            ppVisitor.argBuffer.clear();
            // Preprocess: record resources' states
            for (auto &&i : lst)
                i->accept(ppVisitor);
            // Upload CBuffers
            auto uploadBuffer = allocator->GetTempDefaultBuffer(ppVisitor.argBuffer.size(), 16);
            ppVisitor.stateTracker->RecordState(
                uploadBuffer.buffer,
                D3D12_RESOURCE_STATE_COPY_DEST);
            // Update recorded states
            ppVisitor.stateTracker->UpdateState(
                cmdBuilder);
            cmdBuilder.Upload(
                uploadBuffer,
                ppVisitor.argBuffer.data());
            ppVisitor.stateTracker->RecordState(
                uploadBuffer.buffer,
                VEngineShaderResourceState);
            ppVisitor.stateTracker->UpdateState(
                cmdBuilder);
            visitor.bufferVec = ppVisitor.argVecs.data();
            visitor.argBuffer = uploadBuffer;
            // Execute commands
            for (auto &&i : lst)
                i->accept(visitor);
        }
        tracker.RestoreState(cmdBuilder);
    }
    lastFence = queue.Execute(std::move(allocator));
}
void LCCmdBuffer::Sync() {
    queue.Complete(lastFence);
}
}// namespace toolhub::directx