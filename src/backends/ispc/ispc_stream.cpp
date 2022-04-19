//
// Created by Mike Smith on 2022/2/7.
//

#include <algorithm>
#include <backends/ispc/ispc_mesh.h>
#include <backends/ispc/ispc_accel.h>
#include <backends/ispc/ispc_event.h>
#include <backends/ispc/ispc_stream.h>
#include <backends/ispc/ispc_shader.h>
#include <backends/ispc/ispc_texture.h>
#include <backends/ispc/ispc_bindless_array.h>

namespace luisa::compute::ispc {

using std::max;

void check_texture_boundary(ISPCTexture *tex, uint level, uint3 size) {
    // debug: check boundary
    if (level >= tex->lodLevel)
        LUISA_ERROR_WITH_LOCATION("check texture boundary: lod={} out of bound", level);
    if (size.x > max(tex->size[0] >> level, 1u))
        LUISA_ERROR_WITH_LOCATION("check texture boundary: out of bound");
    if (size.y > max(tex->size[1] >> level, 1u))
        LUISA_ERROR_WITH_LOCATION("check texture boundary: out of bound");
    if (size.z > max(tex->size[2] >> level, 1u))
        LUISA_ERROR_WITH_LOCATION("check texture boundary: out of bound");
};

void ISPCStream::dispatch(const CommandList &cmd_list) noexcept {
    for (auto cmd : cmd_list) {
        while (_pool.task_count() > _pool.size() * 4u) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(1ms);
        }
        cmd->accept(*this);
    }
    _pool.barrier();
}

void ISPCStream::signal(ISPCEvent *event) noexcept {
    event->signal(_pool.async([] {}));
}

void ISPCStream::wait(ISPCEvent *event) noexcept {
    _pool.async([future = event->future()] { future.wait(); });
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
    _pool.async([cmd = *command] {
        auto tex = reinterpret_cast<ISPCTexture *>(cmd.texture());
        check_texture_boundary(tex, cmd.level(), cmd.size());
        // calc stride
        auto pxsize = pixel_storage_size(tex->storage);
        auto texsizex = max(tex->size[0] >> cmd.level(), 1u);
        auto texsizey = max(tex->size[1] >> cmd.level(), 1u);
        auto tex_r_stride = texsizex * pxsize;
        auto tex_p_stride = texsizex * texsizey * pxsize;
        auto target_r_stride = cmd.size().x * pxsize;
        auto target_p_stride = cmd.size().x * cmd.size().y * pxsize;
        // copy data
        for (int dz = 0; dz < cmd.size().z; ++dz)
            for (int dy = 0; dy < cmd.size().y; ++dy)
                memcpy((unsigned char *)tex->lods[cmd.level()] + dz * tex_p_stride + dy * tex_r_stride,
                       (unsigned char *)cmd.buffer() + cmd.buffer_offset() + dz * target_p_stride + dy * target_r_stride,
                       cmd.size().x * pxsize);
    });
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
            auto handle = ISPCTexture::TextureView{texture->handle().ptr, argument.level, 0};
            std::memcpy(ptr, &handle, sizeof(handle));
        } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::BindlessArrayArgument>) {
            auto array = reinterpret_cast<const ISPCBindlessArray *>(argument.handle);
            auto handle = array->handle();
            std::memcpy(ptr, &handle, sizeof(handle));
        } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::AccelArgument>) {
            auto accel = reinterpret_cast<ISPCAccel *>(argument.handle);
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
        [shared_buffer, dispatch_size, module = shader->shared_module()](auto bx, auto by, auto bz) noexcept {
            module->invoke(shared_buffer->data(), luisa::make_uint3(bx, by, bz), dispatch_size);
        });
}

void ISPCStream::visit(const TextureUploadCommand *command) noexcept {
    _pool.async([cmd = *command] {
        auto tex = reinterpret_cast<ISPCTexture *>(cmd.handle());
        check_texture_boundary(tex, cmd.level(), cmd.size());
        // calc stride
        auto pxsize = pixel_storage_size(tex->storage);
        auto target_r_stride = cmd.size().x * pxsize;
        auto target_p_stride = cmd.size().x * cmd.size().y * pxsize;
        auto texsizex = max(tex->size[0] >> cmd.level(), 1u);
        auto texsizey = max(tex->size[1] >> cmd.level(), 1u);
        auto tex_r_stride = texsizex * pxsize;
        auto tex_p_stride = texsizex * texsizey * pxsize;
        // copy data
        for (int dz = 0; dz < cmd.size().z; ++dz)
            for (int dy = 0; dy < cmd.size().y; ++dy)
                memcpy((unsigned char *)tex->lods[cmd.level()] + dz * tex_p_stride + dy * tex_r_stride,
                       (unsigned char *)cmd.data() + dz * target_p_stride + dy * target_r_stride,
                       cmd.size().x * pxsize);
    });
}

void ISPCStream::visit(const TextureDownloadCommand *command) noexcept {
    _pool.async([cmd = *command] {
        auto tex = reinterpret_cast<ISPCTexture *>(cmd.handle());
        check_texture_boundary(tex, cmd.level(), cmd.size());
        // calc stride
        auto pxsize = pixel_storage_size(tex->storage);
        auto target_r_stride = cmd.size().x * pxsize;
        auto target_p_stride = cmd.size().x * cmd.size().y * pxsize;
        auto texsizex = max(tex->size[0] >> cmd.level(), 1u);
        auto texsizey = max(tex->size[1] >> cmd.level(), 1u);
        auto tex_r_stride = texsizex * pxsize;
        auto tex_p_stride = texsizex * texsizey * pxsize;
        // copy data
        for (int dz = 0; dz < cmd.size().z; ++dz)
            for (int dy = 0; dy < cmd.size().y; ++dy)
                memcpy((unsigned char *)cmd.data() + dz * target_p_stride + dy * target_r_stride,
                       (unsigned char *)tex->lods[cmd.level()] + dz * tex_p_stride + dy * tex_r_stride,
                       cmd.size().x * pxsize);
    });
}

void ISPCStream::visit(const TextureCopyCommand *command) noexcept {
    _pool.async([cmd = *command] {
        auto src_tex = reinterpret_cast<ISPCTexture *>(cmd.src_handle());
        auto dst_tex = reinterpret_cast<ISPCTexture *>(cmd.dst_handle());
        check_texture_boundary(src_tex, cmd.src_level(), cmd.size());
        check_texture_boundary(dst_tex, cmd.dst_level(), cmd.size());
        // calc stride
        auto pxsize = pixel_storage_size(src_tex->storage);
        auto src_sizex = max(src_tex->size[0] >> cmd.src_level(), 1u);
        auto dst_sizex = max(dst_tex->size[0] >> cmd.dst_level(), 1u);
        auto src_sizey = max(src_tex->size[1] >> cmd.src_level(), 1u);
        auto dst_sizey = max(dst_tex->size[1] >> cmd.dst_level(), 1u);
        auto src_r_stride = src_sizex * pxsize;
        auto src_p_stride = src_sizex * src_sizey * pxsize;
        auto dst_r_stride = dst_sizex * pxsize;
        auto dst_p_stride = dst_sizex * dst_sizey * pxsize;
        // copy data
        for (int dz = 0; dz < cmd.size().z; ++dz)
            for (int dy = 0; dy < cmd.size().y; ++dy)
                memcpy((unsigned char *)dst_tex->lods[cmd.dst_level()] + dz * dst_p_stride + dy * dst_r_stride,
                       (unsigned char *)src_tex->lods[cmd.src_level()] + dz * src_p_stride + dy * src_r_stride,
                       cmd.size().x * pxsize);
    });
}

void ISPCStream::visit(const TextureToBufferCopyCommand *command) noexcept {
    _pool.async([cmd = *command] {
        auto tex = reinterpret_cast<ISPCTexture *>(cmd.texture());
        check_texture_boundary(tex, cmd.level(), cmd.size());
        // calc stride
        auto pxsize = pixel_storage_size(tex->storage);
        auto texsizex = max(tex->size[0] >> cmd.level(), 1u);
        auto texsizey = max(tex->size[1] >> cmd.level(), 1u);
        auto tex_r_stride = texsizex * pxsize;
        auto tex_p_stride = texsizex * texsizey * pxsize;
        auto target_r_stride = cmd.size().x * pxsize;
        auto target_p_stride = cmd.size().x * cmd.size().y * pxsize;
        // copy data
        for (int dz = 0; dz < cmd.size().z; ++dz)
            for (int dy = 0; dy < cmd.size().y; ++dy)
                memcpy((unsigned char *)cmd.buffer() + cmd.buffer_offset() + dz * target_p_stride + dy * target_r_stride,
                       (unsigned char *)tex->lods[cmd.level()] + dz * tex_p_stride + dy * tex_r_stride,
                       cmd.size().x * pxsize);
    });
}

void ISPCStream::visit(const AccelBuildCommand *command) noexcept {
    reinterpret_cast<ISPCAccel *>(command->handle())->build(
        _pool, command->instance_count(), command->modifications());
}

void ISPCStream::visit(const MeshBuildCommand *command) noexcept {
    _pool.async([mesh = reinterpret_cast<ISPCMesh *>(command->handle())] {
        mesh->commit();
    });
}

void ISPCStream::visit(const BindlessArrayUpdateCommand *command) noexcept {
    reinterpret_cast<ISPCBindlessArray *>(command->handle())->update(_pool);
}

void ISPCStream::dispatch(luisa::move_only_function<void()> &&f) noexcept {
    auto ptr = new_with_allocator<luisa::move_only_function<void()>>(std::move(f));
    _pool.async([ptr] {
        (*ptr)();
        delete_with_allocator(ptr);
    });
    _pool.barrier();
}

}// namespace luisa::compute::ispc
