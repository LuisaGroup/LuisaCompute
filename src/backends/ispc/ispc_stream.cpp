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

void check_texture_boundary(ISPCTexture* tex, uint level, uint3 size, uint3 offset)
{
    if (offset.z != 0 || size.z!=1)
        LUISA_ERROR_WITH_LOCATION("TextureDownloadCommand: unimplemented");
    // debug: check boundary
    if (level >= tex->lodLevel)
        LUISA_ERROR_WITH_LOCATION("TextureDownloadCommand: lod={} out of bound", level);
    if (size.x + offset.x > max(tex->width>>level,1u))
        LUISA_ERROR_WITH_LOCATION("TextureDownloadCommand: out of bound");
    if (size.y + offset.y > max(tex->height>>level,1u))
        LUISA_ERROR_WITH_LOCATION("TextureDownloadCommand: out of bound");
};

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
    _pool.async([cmd = *command] {
        ISPCTexture* tex = reinterpret_cast<ISPCTexture*>(cmd.texture());
        check_texture_boundary(tex, cmd.level(), cmd.size(), cmd.offset());
        // copy data
        // data is void*; tex->lods is float* for now
        int target_stride = cmd.size().x * 4*sizeof(float); // TODO support for other data type
        int tex_stride = max(tex->width>>cmd.level(), 1u) * 4; // TODO support for other data type
        for (int i=0; i<cmd.size().y; ++i)
            memcpy(tex->lods[cmd.level()] + (i+cmd.offset().y) * tex_stride + cmd.offset().x*4,
                (unsigned char*)cmd.buffer() + cmd.buffer_offset() + i*target_stride,
                cmd.size().x * 4*sizeof(float)); 
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
//    LUISA_INFO(
//        "Dispatching ISPC kernel "
//        "with ({}, {}, {}) blocks, "
//        "each with ({}, {}, {}) threads.",
//        grid_size.x, grid_size.y, grid_size.z,
//        block_size.x, block_size.y, block_size.z);
    _pool.parallel(
        grid_size.x, grid_size.y, grid_size.z,
        [shared_buffer, dispatch_size, module = shader->shared_module()](auto bx, auto by, auto bz) noexcept {
            module->invoke(shared_buffer->data(), luisa::make_uint3(bx, by, bz), dispatch_size);
        });
}

void ISPCStream::visit(const TextureUploadCommand *command) noexcept {
    _pool.async([cmd = *command] {
        ISPCTexture* tex = reinterpret_cast<ISPCTexture*>(cmd.handle());
        check_texture_boundary(tex, cmd.level(), cmd.size(), cmd.offset());
        // copy data
        // data is void*; tex->lods is float* for now
        int target_stride = cmd.size().x * 4*sizeof(float); // TODO support for other data type
        int tex_stride = max(tex->width>>cmd.level(), 1u) * 4; // TODO support for other data type
        for (int i=0; i<cmd.size().y; ++i)
            memcpy(tex->lods[cmd.level()] + (i+cmd.offset().y) * tex_stride + cmd.offset().x*4, (unsigned char*)cmd.data() + i*target_stride, cmd.size().x * 4*sizeof(float)); 
    });
}

void ISPCStream::visit(const TextureDownloadCommand *command) noexcept {
    _pool.async([cmd = *command] {
        ISPCTexture* tex = reinterpret_cast<ISPCTexture*>(cmd.handle());
        check_texture_boundary(tex, cmd.level(), cmd.size(), cmd.offset());
        // copy data
        // data is void*; tex->lods is float* for now
        int target_stride = cmd.size().x * 4*sizeof(float); // TODO support for other data type
        int tex_stride = max(tex->width>>cmd.level(), 1u) * 4; // TODO support for other data type
        for (int i=0; i<cmd.size().y; ++i)
            memcpy((unsigned char*)cmd.data() + i*target_stride, tex->lods[cmd.level()] + (i+cmd.offset().y) * tex_stride + cmd.offset().x*4, cmd.size().x * 4*sizeof(float)); 
    });
}

void ISPCStream::visit(const TextureCopyCommand *command) noexcept {
    _pool.async([cmd = *command] {
        ISPCTexture* src_tex = reinterpret_cast<ISPCTexture*>(cmd.src_handle());
        ISPCTexture* dst_tex = reinterpret_cast<ISPCTexture*>(cmd.dst_handle());
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
    });
}

void ISPCStream::visit(const TextureToBufferCopyCommand *command) noexcept {
    _pool.async([cmd = *command] {
        ISPCTexture* tex = reinterpret_cast<ISPCTexture*>(cmd.texture());
        check_texture_boundary(tex, cmd.level(), cmd.size(), cmd.offset());
        // copy data
        // data is void*; tex->lods is float* for now
        int target_stride = cmd.size().x * 4*sizeof(float); // TODO support for other data type
        int tex_stride = max(tex->width>>cmd.level(), 1u) * 4; // TODO support for other data type
        for (int i=0; i<cmd.size().y; ++i)
            memcpy((unsigned char*)cmd.buffer() + cmd.buffer_offset() + i*target_stride,
                tex->lods[cmd.level()] + (i+cmd.offset().y) * tex_stride + cmd.offset().x*4,
                cmd.size().x * 4*sizeof(float)); 
    });
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
