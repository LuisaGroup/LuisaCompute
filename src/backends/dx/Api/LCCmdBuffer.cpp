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
    ResourceStateTracker stateTracker;
    vstd::vector<std::pair<vstd::vector<vbyte>, BufferView>> argVecs;
    vstd::vector<std::pair<Resource const *, D3D12_RESOURCE_STATES>> writeArgs;
    struct Visitor {
        LCPreProcessVisitor *self;
        Function f;
        Variable const *arg;
        vstd::vector<vbyte> *argVec;
        void UniformAlign(size_t align) {
            argVec->resize(CalcAlign(argVec->size(), align));
        }
        template<typename T>
        void EmplaceData(T const &data) {
            size_t sz = argVec->size();
            argVec->resize(sz + sizeof(T));
            *reinterpret_cast<T *>(argVec->data() + sz) = data;
        }
        template<typename T>
        void EmplaceData(T const *data, size_t size) {
            size_t sz = argVec->size();
            auto byteSize = size * sizeof(T);
            argVec->resize(sz + byteSize);
            memcpy(argVec->data() + sz, data, byteSize);
        }
        void operator()(uint uid, ShaderDispatchCommand::BufferArgument const &bf) {
            auto res = reinterpret_cast<Buffer const *>(bf.handle);
            if (((uint)f.variable_usage(arg->uid()) | (uint)Usage::WRITE) != 0) {
                self->stateTracker.RecordState(
                    res,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                self->writeArgs.emplace_back(res, VEngineShaderResourceState);
            } else {
                self->stateTracker.RecordState(
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
            if (((uint)f.variable_usage(arg->uid()) | (uint)Usage::WRITE) != 0) {
                self->stateTracker.RecordState(
                    rt,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                self->writeArgs.emplace_back(rt, VEngineShaderResourceRTState);
            }
            // SRV
            else {
                self->stateTracker.RecordState(
                    rt,
                    VEngineShaderResourceRTState);
            }
            ++arg;
        }
        void operator()(uint uid, ShaderDispatchCommand::BindlessArrayArgument const &bf) {
            ++arg;
        }
        void operator()(uint uid, vstd::span<std::byte const> bf) {
            auto PushArray = [&](size_t sz) {
                bf = {bf.data() + sz, bf.size() - sz};
            };
            auto AddArg = [&](auto &AddArg, Type const *type) -> void {
                switch (type->tag()) {
                    case Type::Tag::BOOL: {

                        bool v = ((vbyte)bf[0] != 0);
                        if (v) {
                            EmplaceData(std::numeric_limits<uint>::max());
                        } else {
                            EmplaceData<uint>(0);
                        }
                        PushArray(1);
                    } break;
                    case Type::Tag::UINT:
                    case Type::Tag::INT:
                    case Type::Tag::FLOAT:
                        EmplaceData<uint>(*(uint const *)bf.data());
                        PushArray(4);
                        break;
                    case Type::Tag::VECTOR: {
                        size_t align = 1;
                        switch (type->dimension()) {
                            case 1:
                                align = 4;
                                break;
                            case 2:
                                align = 8;
                                break;
                            case 3:
                            case 4:
                                align = 16;
                                break;
                        }
                        UniformAlign(align);
                        EmplaceData<uint>((uint const *)bf.data(), type->dimension());
                        PushArray(align);
                    } break;
                    case Type::Tag::MATRIX:
                        switch (type->dimension()) {
                            case 2:
                                UniformAlign(8);
                                EmplaceData(*(float2x2 const *)bf.data());
                                PushArray(sizeof(float2x2));
                                break;
                            case 3:
                                UniformAlign(16);
                                EmplaceData(*(float3x3 const *)bf.data());
                                PushArray(sizeof(float3x3));
                                break;
                            case 4:
                                UniformAlign(16);
                                EmplaceData(*(float4x4 const *)bf.data());
                                PushArray(sizeof(float4x4));
                        }
                        break;
                    case Type::Tag::ARRAY:
                        for (auto i : vstd::range(type->dimension())) {
                            UniformAlign(16);
                            AddArg(AddArg, type->element());
                        }
                        break;
                    case Type::Tag::STRUCTURE:
                        UniformAlign(16);
                        argVec->push_back_all(
                            (vbyte const *)bf.data(),
                            bf.size());
                        break;
                }
            };
            AddArg(AddArg, arg->type());
            ++arg;
        }
        void operator()(uint uid, ShaderDispatchCommand::AccelArgument const &bf) {
            auto accel = reinterpret_cast<TopAccel *>(bf.handle);
            self->stateTracker.RecordState(
                accel->GetInstBuffer(),
                VEngineShaderResourceState);
            ++arg;
        }
    };
    void visit(const BufferUploadCommand *cmd) noexcept override {
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        stateTracker.RecordState(bf.buffer, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const BufferDownloadCommand *cmd) noexcept override {
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        stateTracker.RecordState(bf.buffer, VEngineShaderResourceState);
    }
    void visit(const BufferCopyCommand *cmd) noexcept override {
        auto srcBf = reinterpret_cast<Buffer const *>(cmd->src_handle());
        auto dstBf = reinterpret_cast<Buffer const *>(cmd->dst_handle());
        stateTracker.RecordState(srcBf, VEngineShaderResourceState);
        stateTracker.RecordState(dstBf, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const BufferToTextureCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        stateTracker.RecordState(
            rt,
            D3D12_RESOURCE_STATE_COPY_DEST);
        stateTracker.RecordState(
            bf,
            VEngineShaderResourceState);
    }
    void visit(const TextureUploadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->handle());
        stateTracker.RecordState(
            rt,
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const TextureDownloadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->handle());
        stateTracker.RecordState(
            rt,
            VEngineShaderResourceRTState);
    }
    void visit(const TextureCopyCommand *cmd) noexcept override {
        auto src = reinterpret_cast<RenderTexture *>(cmd->src_handle());
        auto dst = reinterpret_cast<RenderTexture *>(cmd->dst_handle());
        stateTracker.RecordState(
            src,
            VEngineShaderResourceRTState);
        stateTracker.RecordState(
            dst,
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const TextureToBufferCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        stateTracker.RecordState(
            rt,
            VEngineShaderResourceRTState);
        stateTracker.RecordState(
            bf,
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const ShaderDispatchCommand *cmd) noexcept override {
        auto &&lastArgVec = argVecs.emplace_back();
        auto &&argVec = lastArgVec.first;

        argVec.resize(12);
        memcpy(argVec.data(), vstd::get_rvalue_ptr(cmd->dispatch_size()), 12);
        cmd->decode(Visitor{this, cmd->kernel(), cmd->kernel().arguments().data(), &argVec});
        lastArgVec.second = bd->GetTempBuffer(argVec.size(), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        stateTracker.RecordState(
            lastArgVec.second.buffer,
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const AccelUpdateCommand *cmd) noexcept override {
    }
    void visit(const AccelBuildCommand *cmd) noexcept override {
    }
    void visit(const MeshUpdateCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        accel->PreProcessStates(
            *bd,
            stateTracker);
    }
    void visit(const MeshBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        accel->PreProcessStates(
            *bd,
            stateTracker);
    }
    void visit(const BindlessArrayUpdateCommand *cmd) noexcept override {
        auto arr = reinterpret_cast<BindlessArray *>(cmd->handle());
        arr->PreProcessStates(
            *bd,
            stateTracker);
    };
};
class LCCmdVisitor : public CommandVisitor {
public:
    Device *device;
    CommandBufferBuilder *bd;
    ResourceStateTracker *accelStateTracker;
    std::pair<vstd::vector<vbyte>, BufferView> *bufferVec;
    vstd::vector<BindProperty> bindProps;
    void visit(const BufferUploadCommand *cmd) noexcept override {
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        bd->Upload(bf, cmd->data());
        accelStateTracker->RecordState(
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
        accelStateTracker->RecordState(
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
        accelStateTracker->RecordState(
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
            if (((uint)f.variable_usage(arg->uid()) | (uint)Usage::WRITE) != 0) {
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
            self->bindProps.emplace_back(
                std::move(name),
                accel);
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
        bindProps.emplace_back("_Global"sv, tempBuffer.second);
        bindProps.emplace_back("_BindlessTex", DescriptorHeapView(device->globalHeap.get()));
        bindProps.emplace_back("samplers", DescriptorHeapView(device->samplerHeap.get()));
        accelStateTracker->UpdateState(*bd);
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
        bd->CopyBufferTexture(
            bfView,
            rt,
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::BufferToTexture);
        accelStateTracker->RecordState(
            rt,
            VEngineShaderResourceRTState);
    }
    void visit(const TextureDownloadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<RenderTexture *>(cmd->handle());
        auto copyInfo = CommandBufferBuilder::GetCopyTextureBufferSize(
            rt,
            cmd->level());
        auto bfView = bd->GetCB()->GetAlloc()->GetTempReadbackBuffer(copyInfo.alignedBufferSize, 512);
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
        accelStateTracker->RecordState(
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
        accelStateTracker->RecordState(
            bf,
            VEngineShaderResourceState);
    }
    void visit(const AccelUpdateCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        accel->Build(
            *accelStateTracker,
            *bd);
    }
    void visit(const AccelBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        accel->Build(
            *accelStateTracker,
            *bd);
    }
    void visit(const MeshUpdateCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        accel->UpdateStates(
            *bd,
            *accelStateTracker);
    }
    void visit(const MeshBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        accel->UpdateStates(
            *bd,
            *accelStateTracker);
    }
    void visit(const BindlessArrayUpdateCommand *cmd) noexcept override {
        auto arr = reinterpret_cast<BindlessArray *>(cmd->handle());
        arr->UpdateStates(
            *bd,
            *accelStateTracker);
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
void LCCmdBuffer::Execute(vstd::span<CommandList const> const &c) {
    auto allocator = queue.CreateAllocator();
    vstd::unique_ptr<CommandBuffer> cmdBuffer;
    {
        LCPreProcessVisitor ppVisitor;
        LCCmdVisitor visitor;
        visitor.device = device;
        visitor.accelStateTracker = &ppVisitor.stateTracker;
        cmdBuffer = allocator->GetBuffer();
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
            ppVisitor.writeArgs.clear();
            // Preprocess: record resources' states
            for (auto &&i : lst)
                i->accept(ppVisitor);
            // Update recorded states
            ppVisitor.stateTracker.UpdateState(
                cmdBuilder);
            // Upload CBuffers
            for (auto &&i : ppVisitor.argVecs) {
                cmdBuilder.Upload(
                    i.second,
                    i.first.data());
            }
            // Update CBuffers' state to read
            for (auto &&i : ppVisitor.argVecs) {
                ppVisitor.stateTracker.RecordState(
                    i.second.buffer,
                    VEngineShaderResourceState);
            }
            ppVisitor.stateTracker.UpdateState(
                cmdBuilder);
            visitor.bufferVec = ppVisitor.argVecs.data();
            // Execute commands
            for (auto &&i : lst)
                i->accept(visitor);
            // Fallback writable resources to read
            for (auto &&i : ppVisitor.writeArgs) {
                ppVisitor.stateTracker.RecordState(
                    i.first,
                    i.second);
            }
        }
        ppVisitor.stateTracker.RestoreState(cmdBuilder);
    }
    lastFence = queue.Execute(std::move(allocator));
}
void LCCmdBuffer::Sync() {
    queue.Complete(lastFence);
}
}// namespace toolhub::directx