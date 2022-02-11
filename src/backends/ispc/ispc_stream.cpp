//
// Created by Mike Smith on 2022/2/7.
//

#include <backends/ispc/ispc_mesh.h>
#include <backends/ispc/ispc_accel.h>
#include <backends/ispc/ispc_event.h>
#include <backends/ispc/ispc_stream.h>
#include <backends/ispc/ispc_texture.h>
#include <backends/ispc/ispc_bindless_array.h>

namespace luisa::compute::ispc {

void ISPCStream::dispatch(const CommandList &cmd_list) noexcept {
    for (auto cmd : cmd_list) { cmd->accept(*this); }
    _pool.barrier();
}

void ISPCStream::signal(ISPCEvent *event) noexcept {
    event->signal(_pool.async([] {}));
    _pool.barrier();
}

void ISPCStream::wait(ISPCEvent *event) noexcept {
    _pool.async([event] { event->wait(); });
    _pool.barrier();
}

void ISPCStream::visit(const BufferUploadCommand *command) noexcept {
    luisa::vector<std::byte> temp_buffer(command->size());
    std::memcpy(temp_buffer.data(), command->data(), command->size());
    _pool.async([src = std::move(temp_buffer),
                 buffer = command->handle(), offset = command->offset()] {
        auto dst = reinterpret_cast<void *>(buffer + offset);
        std::memcpy(dst, src.data(), src.size());
    });
}

void ISPCStream::visit(const BufferDownloadCommand *command) noexcept {
    _pool.async([cmd = *command] {
        auto src = reinterpret_cast<const void *>(cmd.handle() + cmd.offset());
        std::memcpy(cmd.data(), src, cmd.size());
    });
}

void ISPCStream::visit(const BufferCopyCommand *command) noexcept {
    _pool.async([cmd = *command] {
        auto src = reinterpret_cast<const void *>(cmd.src_handle() + cmd.src_offset());
        auto dst = reinterpret_cast<void *>(cmd.dst_handle() + cmd.dst_offset());
        std::memcpy(dst, src, cmd.size());
    });
}

void ISPCStream::visit(const BufferToTextureCopyCommand *command) noexcept {
}

void ISPCStream::visit(const ShaderDispatchCommand *command) noexcept {
    auto shader = reinterpret_cast<const ISPCShader *>(command->handle());
    luisa::vector<std::byte> argument_buffer(shader->argument_buffer_size());
    command->decode([&](auto vid, auto argument) noexcept {
        auto ptr = argument_buffer.data() + shader->argument_offset(vid);
        using T = decltype(argument);
        if constexpr (std::is_same_v<T, ShaderDispatchCommand::BufferArgument>) {
            auto buffer = reinterpret_cast<void *>(argument.handle + argument.offset);
            std::memcpy(ptr, &buffer, sizeof(buffer));
        } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::TextureArgument>) {
            auto texture = reinterpret_cast<const ISPCTexture *>(argument.handle);
            auto handle = texture->handle();
            std::memcpy(ptr, &handle, sizeof(handle));
        } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::BindlessArrayArgument>) {
            auto array = reinterpret_cast<const ISPCBindlessArray *>(argument.handle);
            auto handle = array->handle();
            std::memcpy(ptr, &handle, sizeof(handle));
        } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::AccelArgument>) {
            auto accel = reinterpret_cast<const ISPCAccel *>(argument.handle);
            auto handle = accel->handle();
            std::memcpy(ptr, &handle, sizeof(handle));
        } else {// uniform
            static_assert(std::same_as<T, luisa::span<const std::byte>>);
            std::memcpy(ptr, argument.data(), argument.size_bytes());
        }
    });
    auto shared_buffer = luisa::make_shared<luisa::vector<std::byte>>(std::move(argument_buffer));
    auto dispatch_size = command->dispatch_size();
    auto block_size = command->kernel().block_size();
    auto grid_size = (dispatch_size + block_size - 1u) / block_size;
    _pool.parallel(
        grid_size.x, grid_size.y, grid_size.z,
        [shared_buffer, dispatch_size, module = shader->module()](auto bx, auto by, auto bz) noexcept {
            module->invoke(shared_buffer->data(), make_uint3(bx, by, bz), dispatch_size);
        });
}

void ISPCStream::visit(const TextureUploadCommand *command) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void ISPCStream::visit(const TextureDownloadCommand *command) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void ISPCStream::visit(const TextureCopyCommand *command) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void ISPCStream::visit(const TextureToBufferCopyCommand *command) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void ISPCStream::visit(const AccelUpdateCommand *command) noexcept {
    reinterpret_cast<ISPCAccel *>(command->handle())->update(_pool);
}

void ISPCStream::visit(const AccelBuildCommand *command) noexcept {
    reinterpret_cast<ISPCAccel *>(command->handle())->build(_pool);
}

void ISPCStream::visit(const MeshUpdateCommand *command) noexcept {
    _pool.async([mesh = reinterpret_cast<ISPCMesh *>(command->handle())] {
        mesh->commit();
    });
}

void ISPCStream::visit(const MeshBuildCommand *command) noexcept {
    _pool.async([mesh = reinterpret_cast<ISPCMesh *>(command->handle())] {
        mesh->commit();
    });
}

void ISPCStream::visit(const BindlessArrayUpdateCommand *command) noexcept {
    reinterpret_cast<ISPCBindlessArray *>(command->handle())->update(_pool);
}

}// namespace luisa::compute::ispc
