#pragma vengine_package vengine_directx
#include <Api/LCCmdBuffer.h>
#include <Api/LCDevice.h>
#include <runtime/command.h>
#include <runtime/command_buffer.h>
#include <Codegen/DxCodegen.h>
#include <Shader/ComputeShader.h>
#include <Resource/RenderTexture.h>
namespace toolhub::directx {
class LCCmdVisitor : public CommandVisitor {
public:
    Device *device;
    CommandBufferBuilder *bd;
    ResourceStateTracker stateTracker;
    using ArgVec = vstd::vector<vbyte, VEngine_AllocType::VEngine, 32>;
    ArgVec argVec;
    vstd::vector<BindProperty> bindProps;
    void visit(const BufferUploadCommand *cmd) noexcept override {
        BufferView bf(
            reinterpret_cast<DefaultBuffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        stateTracker.RecordState(bf.buffer, D3D12_RESOURCE_STATE_COPY_DEST);
        bd->Upload(bf, cmd->data());
    }
    void visit(const BufferDownloadCommand *cmd) noexcept override {
        BufferView bf(
            reinterpret_cast<DefaultBuffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        stateTracker.RecordState(bf.buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
        bd->Readback(
            bf,
            cmd->data());
    }
    void visit(const BufferCopyCommand *cmd) noexcept override {
        auto srcBf = reinterpret_cast<DefaultBuffer const *>(cmd->src_handle());
        auto dstBf = reinterpret_cast<DefaultBuffer const *>(cmd->dst_handle());
        stateTracker.RecordState(srcBf, D3D12_RESOURCE_STATE_COPY_SOURCE);
        stateTracker.RecordState(dstBf, D3D12_RESOURCE_STATE_COPY_DEST);
        bd->CopyBuffer(
            srcBf,
            dstBf,
            cmd->src_offset(),
            cmd->dst_offset(),
            cmd->size());
    }
    void visit(const BufferToTextureCopyCommand *cmd) noexcept override {}
    struct Visitor {
        LCCmdVisitor *self;
        Function f;
        Variable const *arg;
        ComputeShader const *cs;
        void UniformAlign(size_t align) {
            self->argVec.resize(CalcAlign(self->argVec.size(), align));
        }
        template<typename T>
        void EmplaceData(T const &data) {
            size_t sz = self->argVec.size();
            self->argVec.resize(sz + sizeof(T));
            *reinterpret_cast<T *>(self->argVec.data() + sz) = data;
        }
        void operator()(uint uid, ShaderDispatchCommand::BufferArgument const &bf) {
            vstd::string name;
            CodegenUtility::GetVariableName(
                *arg,
                name);
            auto res = reinterpret_cast<DefaultBuffer const *>(bf.handle);
            if (((uint)f.variable_usage(arg->uid()) | (uint)Usage::WRITE) != 0) {
                self->stateTracker.RecordState(
                    res,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            } else {
                self->stateTracker.RecordState(
                    res,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
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
                self->stateTracker.RecordState(
                    rt,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                self->bindProps.emplace_back(
                    std::move(name),
                    DescriptorHeapView(
                        self->device->globalHeap.get(),
                        rt->GetGlobalUAVIndex(bf.level)));
            }
            // SRV
            else {
                self->stateTracker.RecordState(
                    rt,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                self->bindProps.emplace_back(
                    std::move(name),
                    DescriptorHeapView(
                        self->device->globalHeap.get(),
                        rt->GetGlobalSRVIndex()));
            }
            ++arg;
        }
        void operator()(uint uid, ShaderDispatchCommand::BindlessArrayArgument const &bf) {}
        void operator()(uint uid, ShaderDispatchCommand::AccelArgument const &bf) {}
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
                        AddArg(AddArg, type->element());
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
                        self->argVec.push_back_all(
                            (vbyte const *)bf.data(),
                            bf.size());
                        break;
                }
            };
            AddArg(AddArg, arg->type());
            ++arg;
        }
    };
    void visit(const ShaderDispatchCommand *cmd) noexcept override {
        argVec.clear();
        bindProps.clear();
        argVec.resize(sizeof(uint3));
        *reinterpret_cast<uint3 *>(argVec.data()) = cmd->dispatch_size();
        auto cs = reinterpret_cast<ComputeShader const *>(cmd->handle());
        cmd->decode(Visitor{this, cmd->kernel(), cmd->kernel().arguments().data(), cs});
        auto tempBuffer = bd->GetTempBuffer(argVec.size());
        stateTracker.RecordState(
            tempBuffer.buffer,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        bd->Upload(
            tempBuffer,
            argVec.data());
        bindProps.emplace_back("_Global"sv, tempBuffer);
        bindProps.emplace_back("_BindlessTex", DescriptorHeapView(device->globalHeap.get()));
        bd->DispatchCompute(
            cs,
            cmd->dispatch_size(),
            bindProps);
    }
    void visit(const TextureUploadCommand *cmd) noexcept override {}
    void visit(const TextureDownloadCommand *cmd) noexcept override {}
    void visit(const TextureCopyCommand *cmd) noexcept override {}
    void visit(const TextureToBufferCopyCommand *cmd) noexcept override {}
    void visit(const AccelUpdateCommand *cmd) noexcept override {}
    void visit(const AccelBuildCommand *cmd) noexcept override {}
    void visit(const MeshUpdateCommand *cmd) noexcept override {}
    void visit(const MeshBuildCommand *cmd) noexcept override {}
    void visit(const BindlessArrayUpdateCommand *cmd) noexcept override {}
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
    {
        LCCmdVisitor visitor;
        visitor.device = device;
        for (auto &&lst : c) {
            auto stateBuffer = allocator->GetBuffer();
            auto cmdBuffer = allocator->GetBuffer();
            {
                auto cmdBuilder = cmdBuffer->Build();
                visitor.bd = &cmdBuilder;
                for (auto &&i : lst)
                    i->accept(visitor);
            }
            auto stateBuilder = stateBuffer->Build();
            visitor.stateTracker.UpdateState(stateBuilder);
        }
        auto finalBuffer = allocator->GetBuffer();
        auto builder = finalBuffer->Build();
        visitor.stateTracker.RestoreState(builder);
    }
    lastFence = queue.Execute(std::move(allocator));
}
void LCCmdBuffer::Sync() {
    queue.Complete(lastFence);
}

}// namespace toolhub::directx