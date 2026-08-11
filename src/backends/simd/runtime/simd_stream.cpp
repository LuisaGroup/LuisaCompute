#include "simd_stream.h"

#include <cstring>

#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/command.h>

#include "simd_bindless_array.h"
#include "simd_buffer.h"
#include "simd_shader.h"
#include "simd_texture.h"

namespace luisa::compute::simd {

void SIMDStream::dispatch(CommandList &&list) noexcept {
    auto commands = list.steal_commands();
    for (auto &&command : commands) {
        switch (command->tag()) {
            case Command::Tag::EBufferUploadCommand: {
                auto *upload = static_cast<BufferUploadCommand *>(
                    command.get());
                auto destination = reinterpret_cast<SIMDBuffer *>(
                                       upload->handle())
                                       ->view(upload->offset(), upload->size());
                std::memcpy(
                    destination.data, upload->data(),
                    destination.size_bytes);
                break;
            }
            case Command::Tag::EBufferDownloadCommand: {
                auto *download = static_cast<BufferDownloadCommand *>(
                    command.get());
                auto source = reinterpret_cast<SIMDBuffer *>(
                                  download->handle())
                                  ->view(download->offset(), download->size());
                std::memcpy(
                    download->data(), source.data, source.size_bytes);
                break;
            }
            case Command::Tag::EBufferCopyCommand: {
                auto *copy = static_cast<BufferCopyCommand *>(command.get());
                auto source = reinterpret_cast<SIMDBuffer *>(
                                  copy->src_handle())
                                  ->view(copy->src_offset(), copy->size());
                auto destination = reinterpret_cast<SIMDBuffer *>(
                                       copy->dst_handle())
                                       ->view(copy->dst_offset(), copy->size());
                std::memmove(
                    destination.data, source.data, source.size_bytes);
                break;
            }
            case Command::Tag::EShaderDispatchCommand: {
                auto *raw = static_cast<ShaderDispatchCommand *>(
                    command.release());
                auto shader = reinterpret_cast<SIMDShader *>(raw->handle());
                shader->dispatch(
                    *_thread_pool,
                    luisa::unique_ptr<ShaderDispatchCommand>{raw});
                break;
            }
            case Command::Tag::EBufferToTextureCopyCommand: {
                auto *copy = static_cast<BufferToTextureCopyCommand *>(
                    command.get());
                auto source = reinterpret_cast<SIMDBuffer *>(copy->buffer())
                                  ->view_with_offset(copy->buffer_offset());
                auto destination =
                    reinterpret_cast<SIMDTexture *>(copy->texture())
                        ->view(copy->level());
                LUISA_ASSERT(
                    all(copy->texture_offset() == 0u) &&
                        all(copy->size() == destination.size3d()),
                    "SIMD texture copies currently require a full mip view.");
                destination.copy_from(source.data);
                break;
            }
            case Command::Tag::ETextureUploadCommand: {
                auto *upload = static_cast<TextureUploadCommand *>(
                    command.get());
                auto destination =
                    reinterpret_cast<SIMDTexture *>(upload->handle())
                        ->view(upload->level());
                LUISA_ASSERT(
                    all(upload->offset() == 0u) &&
                        all(upload->size() == destination.size3d()),
                    "SIMD texture uploads currently require a full mip view.");
                destination.copy_from(upload->data());
                break;
            }
            case Command::Tag::ETextureDownloadCommand: {
                auto *download = static_cast<TextureDownloadCommand *>(
                    command.get());
                auto source =
                    reinterpret_cast<SIMDTexture *>(download->handle())
                        ->view(download->level());
                LUISA_ASSERT(
                    all(download->offset() == 0u) &&
                        all(download->size() == source.size3d()),
                    "SIMD texture downloads currently require a full mip view.");
                source.copy_to(download->data());
                break;
            }
            case Command::Tag::ETextureCopyCommand: {
                auto *copy = static_cast<TextureCopyCommand *>(
                    command.get());
                auto source =
                    reinterpret_cast<SIMDTexture *>(copy->src_handle())
                        ->view(copy->src_level());
                auto destination =
                    reinterpret_cast<SIMDTexture *>(copy->dst_handle())
                        ->view(copy->dst_level());
                LUISA_ASSERT(
                    all(copy->src_offset() == 0u) &&
                        all(copy->dst_offset() == 0u) &&
                        all(copy->size() == source.size3d()) &&
                        all(copy->size() == destination.size3d()),
                    "SIMD texture copies currently require full mip views.");
                LUISA_ASSERT(
                    source.size_bytes() == destination.size_bytes(),
                    "SIMD texture copy size mismatch.");
                source.copy_to(
                    const_cast<std::byte *>(destination.data()));
                break;
            }
            case Command::Tag::ETextureToBufferCopyCommand: {
                auto *copy = static_cast<TextureToBufferCopyCommand *>(
                    command.get());
                auto source =
                    reinterpret_cast<SIMDTexture *>(copy->texture())
                        ->view(copy->level());
                auto destination =
                    reinterpret_cast<SIMDBuffer *>(copy->buffer())
                        ->view_with_offset(copy->buffer_offset());
                LUISA_ASSERT(
                    all(copy->texture_offset() == 0u) &&
                        all(copy->size() == source.size3d()),
                    "SIMD texture copies currently require a full mip view.");
                source.copy_to(destination.data);
                break;
            }
            case Command::Tag::EAccelBuildCommand:
            case Command::Tag::EMeshBuildCommand:
            case Command::Tag::ECurveBuildCommand:
            case Command::Tag::EProceduralPrimitiveBuildCommand:
            case Command::Tag::EMotionInstanceBuildCommand:
                LUISA_ERROR_WITH_LOCATION(
                    "Ray-tracing commands are not implemented by the SIMD "
                    "runtime checkpoint yet.");
            case Command::Tag::EBindlessArrayUpdateCommand: {
                auto *update = static_cast<BindlessArrayUpdateCommand *>(
                    command.get());
                auto *array = reinterpret_cast<SIMDBindlessArray *>(
                    update->handle());
                array->update(*update);
                break;
            }
            case Command::Tag::ECustomCommand:
                LUISA_ERROR_WITH_LOCATION(
                    "Custom commands are not implemented by the SIMD runtime "
                    "checkpoint yet.");
        }
    }
    for (auto &&callback : list.steal_callbacks()) { callback(); }
}

}// namespace luisa::compute::simd
