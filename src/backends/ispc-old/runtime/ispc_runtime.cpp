#pragma vengine_package ispc_vsproject

#include "ispc_runtime.h"
#include "ispc_codegen.h"
#include "ispc_accel.h"
#include "ispc_mesh.h"
#include "ispc_bindless_array.h"
#include <backends/ispc/ISPCTest/Types.h>

namespace lc::ispc {
void CommandExecutor::visit(BufferUploadCommand const *cmd) noexcept {
    AddTask(*cmd);
}
void CommandExecutor::visit(BufferDownloadCommand const *cmd) noexcept {
    AddTask(*cmd);
}
void CommandExecutor::visit(BufferCopyCommand const *cmd) noexcept {
    AddTask(*cmd);
}

struct TextureView {
    Texture2D* tex;
    uint level;
};
struct ShaderDispatcher {
    Function func;
    Shader::ArgVector &vec;
    Shader *sd;
    void operator()(uint, ShaderDispatchCommand::BufferArgument const &arg) {
        Shader::PackArg<float *>(vec, reinterpret_cast<float *>(arg.handle));
    }
    void operator()(uint, ShaderDispatchCommand::TextureArgument const &arg) {
        Shader::PackArg<TextureView>(vec, TextureView{reinterpret_cast<Texture2D*>(arg.handle), arg.level});
    }
    void operator()(uint var_id, luisa::span<std::byte const> arg) {
        Shader::PackArr(vec, arg.data(), arg.size(), CodegenUtility::GetTypeAlign(*func.arguments()[sd->GetArgIndex(var_id)].type()));
    }
    void operator()(uint, ShaderDispatchCommand::BindlessArrayArgument const &arg) {
        Shader::PackArg<ISPCBindlessArray::DeviceData>(vec, reinterpret_cast<ISPCBindlessArray*>(arg.handle)->getDeviceData());
    }
    void operator()(uint, ShaderDispatchCommand::AccelArgument const &arg) {
        Shader::PackArg<RTCScene>(vec, reinterpret_cast<ISPCAccel*>(arg.handle)->getScene());
    }
};
void CommandExecutor::visit(ShaderDispatchCommand const *cmd) noexcept {
    Shader::ArgVector vec;
    auto sd = reinterpret_cast<Shader *>(cmd->handle());
    ShaderDispatcher disp{cmd->kernel(), vec, sd};
    cmd->decode(disp);
    auto handle = sd->dispatch(
        tPool,
        cmd->dispatch_size(),
        std::move(vec));
    AddTask(std::move(handle));
}
void CommandExecutor::visit(AccelUpdateCommand const *cmd) noexcept {
    auto accel = reinterpret_cast<ISPCAccel*>(cmd->handle());
    auto handle = tPool->GetTask([accel](){ accel->update(); }, true);
    AddTask(std::move(handle));
}
void CommandExecutor::visit(AccelBuildCommand const *cmd) noexcept {
    auto accel = reinterpret_cast<ISPCAccel*>(cmd->handle());
    auto handle = tPool->GetTask([accel](){ accel->build(); }, true);
    AddTask(std::move(handle));
}
void CommandExecutor::visit(MeshUpdateCommand const *cmd) noexcept {
    auto mesh = reinterpret_cast<ISPCMesh*>(cmd->handle());
    auto handle = tPool->GetTask([mesh](){ mesh->update(); }, true);
    AddTask(std::move(handle));
}
void CommandExecutor::visit(MeshBuildCommand const *cmd) noexcept {
    auto mesh = reinterpret_cast<ISPCMesh*>(cmd->handle());
    auto handle = tPool->GetTask([mesh](){ mesh->build(); }, true);
    AddTask(std::move(handle));
}
void CommandExecutor::visit(TextureUploadCommand const *cmd) noexcept { AddTask(*cmd); }
void CommandExecutor::visit(TextureDownloadCommand const *cmd) noexcept { AddTask(*cmd); }
void CommandExecutor::visit(TextureCopyCommand const *cmd) noexcept { AddTask(*cmd); }
void CommandExecutor::visit(TextureToBufferCopyCommand const *cmd) noexcept { AddTask(*cmd); }
void CommandExecutor::visit(BufferToTextureCopyCommand const *cmd) noexcept { AddTask(*cmd); }
void CommandExecutor::visit(BindlessArrayUpdateCommand const *cmd) noexcept {}
CommandExecutor::CommandExecutor(ThreadPool *tPool)
    : tPool(tPool),
      dispatchThread([&] {
          while (enabled)
              ThreadExecute();
      }) {}
void CommandExecutor::ThreadExecute() {
    auto check_texture_boundary = [](Texture2D* tex, uint level, uint3 size, uint3 offset)
    {
        if (offset.z != 0 || size.z!=1)
            throw "TextureDownloadCommand: unimplemented";
        // debug: check boundary
        if (level >= tex->lodLevel)
            throw "TextureDownloadCommand: lod out of bound";
        if (size.x + offset.x > max(tex->width>>level,1u))
            throw "TextureDownloadCommand: out of bound";
        if (size.y + offset.y > max(tex->height>>level,1u))
            throw "TextureDownloadCommand: out of bound";
    };
    while (auto job = syncTasks.Pop()) {
        job->multi_visit(
            [&](ThreadTaskHandle const &handle) { handle.Complete(); },
            [&](BufferUploadCommand const &cmd) {
                uint8_t *ptr = reinterpret_cast<uint8_t *>(cmd.handle());
                memcpy(ptr + cmd.offset(), cmd.data(), cmd.size());
            },
            [&](BufferDownloadCommand const &cmd) {
                uint8_t const *ptr = reinterpret_cast<uint8_t const *>(cmd.handle());
                memcpy(cmd.data(), ptr + cmd.offset(), cmd.size());
            },
            [&](BufferCopyCommand const &cmd) {
                uint8_t const *src = reinterpret_cast<uint8_t const *>(cmd.src_handle());
                uint8_t *dst = reinterpret_cast<uint8_t *>(cmd.dst_handle());
                memcpy(dst + cmd.dst_offset(), src + cmd.src_offset(), cmd.size());
            },
            [&](TextureUploadCommand const &cmd) {
                Texture2D* tex = reinterpret_cast<Texture2D*>(cmd.handle());
                check_texture_boundary(tex, cmd.level(), cmd.size(), cmd.offset());
                // copy data
                // data is void*; tex->lods is float* for now
                int target_stride = cmd.size().x * 4*sizeof(float); // TODO support for other data type
                int tex_stride = max(tex->width>>cmd.level(), 1u) * 4; // TODO support for other data type
                for (int i=0; i<cmd.size().y; ++i)
                    memcpy(tex->lods[cmd.level()] + (i+cmd.offset().y) * tex_stride + cmd.offset().x*4, (unsigned char*)cmd.data() + i*target_stride, cmd.size().x * 4*sizeof(float)); 
            },
            [&](TextureDownloadCommand const &cmd) {
                Texture2D* tex = reinterpret_cast<Texture2D*>(cmd.handle());
                check_texture_boundary(tex, cmd.level(), cmd.size(), cmd.offset());
                // copy data
                // data is void*; tex->lods is float* for now
                int target_stride = cmd.size().x * 4*sizeof(float); // TODO support for other data type
                int tex_stride = max(tex->width>>cmd.level(), 1u) * 4; // TODO support for other data type
                for (int i=0; i<cmd.size().y; ++i)
                    memcpy((unsigned char*)cmd.data() + i*target_stride, tex->lods[cmd.level()] + (i+cmd.offset().y) * tex_stride + cmd.offset().x*4, cmd.size().x * 4*sizeof(float)); 
            },
            [&](TextureCopyCommand const &cmd) {
                Texture2D* src_tex = reinterpret_cast<Texture2D*>(cmd.src_handle());
                Texture2D* dst_tex = reinterpret_cast<Texture2D*>(cmd.dst_handle());
                check_texture_boundary(src_tex, cmd.src_level(), cmd.size(), cmd.src_offset());
                check_texture_boundary(dst_tex, cmd.dst_level(), cmd.size(), cmd.dst_offset());
                // copy data
                int src_stride = max(src_tex->width>>cmd.src_level(), 1u) * 4; // TODO support for other data type
                int dst_stride = max(dst_tex->width>>cmd.dst_level(), 1u) * 4; // TODO support for other data type
                for (int i=0; i<cmd.size().y; ++i) {
                    memcpy(dst_tex->lods[cmd.dst_level()] + dst_stride * (i+cmd.dst_offset().y) + cmd.dst_offset().x*4,
                           src_tex->lods[cmd.src_level()] + src_stride * (i+cmd.src_offset().y) + cmd.src_offset().x*4,
                           cmd.size().x * 4*sizeof(float));
                }
            },
            [&](TextureToBufferCopyCommand const &cmd) {
                Texture2D* tex = reinterpret_cast<Texture2D*>(cmd.texture());
                check_texture_boundary(tex, cmd.level(), cmd.size(), cmd.offset());
                // copy data
                // data is void*; tex->lods is float* for now
                int target_stride = cmd.size().x * 4*sizeof(float); // TODO support for other data type
                int tex_stride = max(tex->width>>cmd.level(), 1u) * 4; // TODO support for other data type
                for (int i=0; i<cmd.size().y; ++i)
                    memcpy((unsigned char*)cmd.buffer() + cmd.buffer_offset() + i*target_stride, tex->lods[cmd.level()] + (i+cmd.offset().y) * tex_stride + cmd.offset().x*4, cmd.size().x * 4*sizeof(float)); 
            },
            [&](BufferToTextureCopyCommand const &cmd) {
                Texture2D* tex = reinterpret_cast<Texture2D*>(cmd.texture());
                check_texture_boundary(tex, cmd.level(), cmd.size(), cmd.offset());
                // copy data
                // data is void*; tex->lods is float* for now
                int target_stride = cmd.size().x * 4*sizeof(float); // TODO support for other data type
                int tex_stride = max(tex->width>>cmd.level(), 1u) * 4; // TODO support for other data type
                for (int i=0; i<cmd.size().y; ++i)
                    memcpy(tex->lods[cmd.level()] + (i+cmd.offset().y) * tex_stride + cmd.offset().x*4, (unsigned char*)cmd.buffer() + cmd.buffer_offset() + i*target_stride, cmd.size().x * 4*sizeof(float)); 

            },
            [&](Signal const &cmd) {
                std::lock_guard lck(cmd.evt->mtx);
                cmd.evt->targetFence--;
                cmd.evt->cv.notify_all();
            },
            [&](Wait const &cmd) {
                cmd.evt->Sync();
            });
        executedTask++;
        if (executedTask >= taskCount)
            break;
    }
    std::unique_lock lck(dispMtx);
    while (executedTask >= taskCount) {
        mainThdCv.notify_all();
        dispThdCv.wait(lck);
    }
}
void CommandExecutor::WaitThread() {
    std::unique_lock lck(dispMtx);
    if (executedTask < taskCount) {
        mainThdCv.wait(lck);
    }
}
void CommandExecutor::ExecuteDispatch() {
    {
        std::unique_lock lck(dispMtx);
        taskCount += outsideTaskCount;
        dispThdCv.notify_all();
    }
    outsideTaskCount = 0;
}

CommandExecutor::~CommandExecutor() {
    {
        std::lock_guard lck(dispMtx);
        enabled = false;
        executedTask = 0;
        taskCount = 1;
        dispThdCv.notify_all();
    }
    dispatchThread.join();
}
}// namespace lc::ispc