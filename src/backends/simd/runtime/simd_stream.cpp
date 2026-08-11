#include "simd_stream.h"

#include <cstring>

#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/command.h>

#include "simd_buffer.h"
#include "simd_shader.h"

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
                    luisa::unique_ptr<ShaderDispatchCommand>{raw});
                break;
            }
            case Command::Tag::EBufferToTextureCopyCommand:
            case Command::Tag::ETextureUploadCommand:
            case Command::Tag::ETextureDownloadCommand:
            case Command::Tag::ETextureCopyCommand:
            case Command::Tag::ETextureToBufferCopyCommand:
                LUISA_ERROR_WITH_LOCATION(
                    "Texture commands are not implemented by the SIMD "
                    "runtime checkpoint yet.");
            case Command::Tag::EAccelBuildCommand:
            case Command::Tag::EMeshBuildCommand:
            case Command::Tag::ECurveBuildCommand:
            case Command::Tag::EProceduralPrimitiveBuildCommand:
            case Command::Tag::EMotionInstanceBuildCommand:
                LUISA_ERROR_WITH_LOCATION(
                    "Ray-tracing commands are not implemented by the SIMD "
                    "runtime checkpoint yet.");
            case Command::Tag::EBindlessArrayUpdateCommand:
                LUISA_ERROR_WITH_LOCATION(
                    "Bindless arrays are not implemented by the SIMD runtime "
                    "checkpoint yet.");
            case Command::Tag::ECustomCommand:
                LUISA_ERROR_WITH_LOCATION(
                    "Custom commands are not implemented by the SIMD runtime "
                    "checkpoint yet.");
        }
    }
    for (auto &&callback : list.steal_callbacks()) { callback(); }
}

}// namespace luisa::compute::simd
