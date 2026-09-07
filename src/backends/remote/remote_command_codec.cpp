#include "remote_command_codec.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <unordered_map>

#include <luisa/core/mathematics.h>
#include <luisa/core/stl/optional.h>
#include <luisa/runtime/rhi/pixel.h>

#include "../common/indirect_dispatch_layout.h"

namespace luisa::compute::remote {

namespace {

void write_uint3(Writer &writer, uint3 value) noexcept {
    writer.write_u32(value.x);
    writer.write_u32(value.y);
    writer.write_u32(value.z);
}

[[nodiscard]] uint3 read_uint3(Reader &reader) noexcept {
    return uint3{reader.read_u32(), reader.read_u32(), reader.read_u32()};
}

[[nodiscard]] bool valid_range(
    uint64_t offset, uint64_t size, size_t total) noexcept {
    return offset <= total && size <= static_cast<uint64_t>(total) - offset;
}

[[nodiscard]] bool valid_storage(uint32_t value) noexcept {
    return value < pixel_storage_count;
}

[[nodiscard]] bool texture_byte_size(
    PixelStorage storage, uint3 size, size_t &bytes) noexcept {
    auto result = checked_pixel_storage_size(storage, size);
    if (!result) { return false; }
    bytes = result.size;
    return true;
}

[[nodiscard]] bool valid_texture_region(
    const ResolvedTexture &texture, PixelStorage storage,
    uint32_t level, uint3 size, uint3 offset) noexcept {
    if (storage != texture.storage || level >= texture.mip_levels ||
        any(size == 0u)) {
        return false;
    }
    auto mip_size = luisa::max(texture.size >> level, 1u);
    return offset.x <= mip_size.x && size.x <= mip_size.x - offset.x &&
           offset.y <= mip_size.y && size.y <= mip_size.y - offset.y &&
           offset.z <= mip_size.z && size.z <= mip_size.z - offset.z;
}

}// namespace

uint32_t UploadBlobPlan::find(
    size_t command_index) const noexcept {
    for (auto &&reference : references) {
        if (reference.command_index == command_index) {
            return reference.index;
        }
    }
    return invalid_blob_index;
}

UploadBlobPlan plan_upload_blobs(
    luisa::span<const luisa::unique_ptr<Command>> commands,
    uint64_t minimum_blob_size,
    uint64_t maximum_blob_size,
    const ProtocolLimits &limits) noexcept {
    UploadBlobPlan result;
    if (maximum_blob_size == 0u ||
        minimum_blob_size > maximum_blob_size) {
        return result;
    }
    std::unordered_map<BlobKey, uint32_t, BlobKeyHash> indices;
    for (size_t command_index = 0u;
         command_index < commands.size(); command_index++) {
        auto &&command_ptr = commands[command_index];
        if (command_ptr == nullptr) {
            result.error = "Remote command list contains a null command.";
            return result;
        }
        const void *data = nullptr;
        size_t size = 0u;
        switch (command_ptr->tag()) {
            case Command::Tag::EBufferUploadCommand: {
                auto command = static_cast<const BufferUploadCommand *>(
                    command_ptr.get());
                data = command->data();
                size = command->size();
                break;
            }
            case Command::Tag::ETextureUploadCommand: {
                auto command = static_cast<const TextureUploadCommand *>(
                    command_ptr.get());
                if (!texture_byte_size(
                        command->storage(), command->size(), size)) {
                    result.error =
                        "Remote texture upload has an invalid storage or extent.";
                    return result;
                }
                data = command->data();
                break;
            }
            default: continue;
        }
        if (size != 0u && data == nullptr) {
            result.error = "Remote upload has a null source.";
            return result;
        }
        if (size == 0u || size < minimum_blob_size ||
            size > maximum_blob_size) {
            continue;
        }
        auto bytes = luisa::span{
            static_cast<const std::byte *>(data), size};
        auto key = compute_blob_key(bytes);
        uint32_t index{};
        if (auto iterator = indices.find(key);
            iterator != indices.end()) {
            index = iterator->second;
            auto existing = result.blobs[index].bytes;
            if (existing.size() != bytes.size() ||
                (!bytes.empty() && std::memcmp(
                                       existing.data(), bytes.data(),
                                       bytes.size()) != 0)) {
                result.error =
                    "Remote upload blob digest collision detected locally.";
                return result;
            }
        } else {
            if (result.blobs.size() >= limits.max_array_size ||
                result.blobs.size() >= invalid_blob_index) {
                result.error =
                    "Remote upload blob count exceeds the configured limit.";
                return result;
            }
            index = static_cast<uint32_t>(result.blobs.size());
            indices.emplace(key, index);
            result.blobs.emplace_back(UploadBlob{key, bytes});
        }
        result.references.emplace_back(
            UploadBlobReference{command_index, index});
    }
    constexpr uint64_t prepare_header_size = 16u;
    constexpr uint64_t descriptor_size =
        sizeof(uint64_t) + blob_digest_size;
    if (result.blobs.size() >
        (limits.max_frame_payload -
         std::min(prepare_header_size, limits.max_frame_payload)) /
            descriptor_size) {
        result.error =
            "Remote upload blob descriptors exceed the frame limit.";
    }
    return result;
}

EncodedSubmission encode_submission(
    uint64_t stream_handle, uint64_t submission_id,
    luisa::span<const luisa::unique_ptr<Command>> commands,
    const ProtocolLimits &limits,
    const UploadBlobPlan *blob_plan) noexcept {
    EncodedSubmission result;
    if (commands.size() > limits.max_array_size) {
        result.error = "Remote command count exceeds the configured limit.";
        return result;
    }
    Writer writer;
    writer.write_u64(stream_handle);
    writer.write_u64(submission_id);
    writer.write_u64(commands.size());
    for (size_t command_index = 0u;
         command_index < commands.size(); command_index++) {
        auto &&command_ptr = commands[command_index];
        if (command_ptr == nullptr) {
            result.error = "Remote command list contains a null command.";
            return result;
        }
        auto command = command_ptr.get();
        switch (command->tag()) {
            case Command::Tag::EBufferUploadCommand: {
                auto c = static_cast<const BufferUploadCommand *>(command);
                if (c->size() != 0u && c->data() == nullptr) {
                    result.error = "Remote buffer upload has a null source.";
                    return result;
                }
                auto blob_index = blob_plan == nullptr ?
                                      invalid_blob_index :
                                      blob_plan->find(command_index);
                writer.write_u16(static_cast<uint16_t>(
                    blob_index == invalid_blob_index ?
                        WireCommand::BUFFER_UPLOAD :
                        WireCommand::BUFFER_UPLOAD_CACHED));
                writer.write_u64(c->handle());
                writer.write_u64(c->offset());
                writer.write_u64(c->size());
                if (blob_index == invalid_blob_index) {
                    writer.write_bytes({static_cast<const std::byte *>(c->data()),
                                        c->size()});
                } else {
                    writer.write_u32(blob_index);
                }
                break;
            }
            case Command::Tag::EBufferDownloadCommand: {
                auto c = static_cast<const BufferDownloadCommand *>(command);
                if (c->size() != 0u && c->data() == nullptr) {
                    result.error = "Remote buffer download has a null destination.";
                    return result;
                }
                if (result.downloads.size() > std::numeric_limits<uint32_t>::max()) {
                    result.error = "Remote download count exceeds the wire limit.";
                    return result;
                }
                auto index = static_cast<uint32_t>(result.downloads.size());
                result.downloads.emplace_back(DownloadTarget{c->data(), c->size()});
                writer.write_u16(static_cast<uint16_t>(WireCommand::BUFFER_DOWNLOAD));
                writer.write_u32(index);
                writer.write_u64(c->handle());
                writer.write_u64(c->offset());
                writer.write_u64(c->size());
                break;
            }
            case Command::Tag::EBufferCopyCommand: {
                auto c = static_cast<const BufferCopyCommand *>(command);
                writer.write_u16(static_cast<uint16_t>(WireCommand::BUFFER_COPY));
                writer.write_u64(c->src_handle());
                writer.write_u64(c->dst_handle());
                writer.write_u64(c->src_offset());
                writer.write_u64(c->dst_offset());
                writer.write_u64(c->size());
                break;
            }
            case Command::Tag::EBufferToTextureCopyCommand: {
                auto c = static_cast<const BufferToTextureCopyCommand *>(command);
                writer.write_u16(static_cast<uint16_t>(WireCommand::BUFFER_TO_TEXTURE_COPY));
                writer.write_u64(c->buffer());
                writer.write_u64(c->buffer_offset());
                writer.write_u64(c->texture());
                writer.write_u32(static_cast<uint32_t>(c->storage()));
                writer.write_u32(c->level());
                write_uint3(writer, c->size());
                write_uint3(writer, c->texture_offset());
                break;
            }
            case Command::Tag::ETextureToBufferCopyCommand: {
                auto c = static_cast<const TextureToBufferCopyCommand *>(command);
                writer.write_u16(static_cast<uint16_t>(WireCommand::TEXTURE_TO_BUFFER_COPY));
                writer.write_u64(c->buffer());
                writer.write_u64(c->buffer_offset());
                writer.write_u64(c->texture());
                writer.write_u32(static_cast<uint32_t>(c->storage()));
                writer.write_u32(c->level());
                write_uint3(writer, c->size());
                write_uint3(writer, c->texture_offset());
                break;
            }
            case Command::Tag::ETextureUploadCommand: {
                auto c = static_cast<const TextureUploadCommand *>(command);
                size_t size_bytes{};
                if (!texture_byte_size(c->storage(), c->size(), size_bytes)) {
                    result.error = "Remote texture upload has an invalid storage or extent.";
                    return result;
                }
                if (size_bytes != 0u && c->data() == nullptr) {
                    result.error = "Remote texture upload has a null source.";
                    return result;
                }
                auto blob_index = blob_plan == nullptr ?
                                      invalid_blob_index :
                                      blob_plan->find(command_index);
                writer.write_u16(static_cast<uint16_t>(
                    blob_index == invalid_blob_index ?
                        WireCommand::TEXTURE_UPLOAD :
                        WireCommand::TEXTURE_UPLOAD_CACHED));
                writer.write_u64(c->handle());
                writer.write_u32(static_cast<uint32_t>(c->storage()));
                writer.write_u32(c->level());
                write_uint3(writer, c->size());
                write_uint3(writer, c->offset());
                writer.write_u64(size_bytes);
                if (blob_index == invalid_blob_index) {
                    writer.write_bytes({static_cast<const std::byte *>(c->data()),
                                        size_bytes});
                } else {
                    writer.write_u32(blob_index);
                }
                break;
            }
            case Command::Tag::ETextureDownloadCommand: {
                auto c = static_cast<const TextureDownloadCommand *>(command);
                size_t size_bytes{};
                if (!texture_byte_size(c->storage(), c->size(), size_bytes)) {
                    result.error = "Remote texture download has an invalid storage or extent.";
                    return result;
                }
                if (size_bytes != 0u && c->data() == nullptr) {
                    result.error = "Remote texture download has a null destination.";
                    return result;
                }
                if (result.downloads.size() > std::numeric_limits<uint32_t>::max()) {
                    result.error = "Remote download count exceeds the wire limit.";
                    return result;
                }
                auto index = static_cast<uint32_t>(result.downloads.size());
                result.downloads.emplace_back(DownloadTarget{c->data(), size_bytes});
                writer.write_u16(static_cast<uint16_t>(WireCommand::TEXTURE_DOWNLOAD));
                writer.write_u32(index);
                writer.write_u64(c->handle());
                writer.write_u32(static_cast<uint32_t>(c->storage()));
                writer.write_u32(c->level());
                write_uint3(writer, c->size());
                write_uint3(writer, c->offset());
                writer.write_u64(size_bytes);
                break;
            }
            case Command::Tag::ETextureCopyCommand: {
                auto c = static_cast<const TextureCopyCommand *>(command);
                writer.write_u16(static_cast<uint16_t>(WireCommand::TEXTURE_COPY));
                writer.write_u32(static_cast<uint32_t>(c->storage()));
                writer.write_u64(c->src_handle());
                writer.write_u64(c->dst_handle());
                writer.write_u32(c->src_level());
                writer.write_u32(c->dst_level());
                write_uint3(writer, c->size());
                write_uint3(writer, c->src_offset());
                write_uint3(writer, c->dst_offset());
                break;
            }
            case Command::Tag::EShaderDispatchCommand: {
                auto c = static_cast<const ShaderDispatchCommand *>(command);
                writer.write_u16(static_cast<uint16_t>(WireCommand::SHADER_DISPATCH));
                writer.write_u64(c->handle());
                writer.write_u64(c->arguments().size());
                for (auto &&argument : c->arguments()) {
                    writer.write_u8(static_cast<uint8_t>(argument.tag));
                    switch (argument.tag) {
                        case Argument::Tag::BUFFER:
                            writer.write_u64(argument.buffer.handle);
                            writer.write_u64(argument.buffer.offset);
                            writer.write_u64(argument.buffer.size);
                            break;
                        case Argument::Tag::TEXTURE:
                            writer.write_u64(argument.texture.handle);
                            writer.write_u32(argument.texture.level);
                            break;
                        case Argument::Tag::UNIFORM:
                            writer.write_u64(argument.uniform.alignment);
                            writer.write_blob(c->uniform(argument.uniform));
                            break;
                        case Argument::Tag::BINDLESS_ARRAY:
                            writer.write_u64(argument.bindless_array.handle);
                            break;
                        case Argument::Tag::ACCEL:
                            writer.write_u64(argument.accel.handle);
                            break;
                    }
                }
                if (c->is_indirect()) {
                    writer.write_u8(1u);
                    auto indirect = c->indirect_dispatch();
                    writer.write_u64(indirect.handle);
                    writer.write_u32(indirect.offset);
                    writer.write_u32(indirect.max_dispatch_size);
                } else if (c->is_multiple_dispatch()) {
                    writer.write_u8(2u);
                    auto sizes = c->dispatch_sizes();
                    writer.write_u64(sizes.size());
                    for (auto size : sizes) { write_uint3(writer, size); }
                } else {
                    writer.write_u8(0u);
                    write_uint3(writer, c->dispatch_size());
                }
                break;
            }
            case Command::Tag::EBindlessArrayUpdateCommand: {
                auto c = static_cast<const BindlessArrayUpdateCommand *>(command);
                writer.write_u16(static_cast<uint16_t>(WireCommand::BINDLESS_ARRAY_UPDATE));
                writer.write_u64(c->handle());
                auto write_buffer = [&](const BindlessArrayUpdateCommand::ModifiedBuffer &buffer) noexcept {
                    writer.write_u8(static_cast<uint8_t>(buffer.op));
                    if (buffer.op == BindlessArrayUpdateCommand::Operation::EMPLACE) {
                        writer.write_u64(buffer.handle);
                        writer.write_u64(buffer.offset_bytes);
                        writer.write_u64(buffer.size_bytes);
                    }
                };
                auto write_texture = [&](const BindlessArrayUpdateCommand::ModifiedTexture &texture) noexcept {
                    writer.write_u8(static_cast<uint8_t>(texture.op));
                    if (texture.op == BindlessArrayUpdateCommand::Operation::EMPLACE) {
                        writer.write_u64(texture.handle);
                        writer.write_u32(texture.sampler.code());
                    }
                };
                c->visit_modifications([&](auto const &modifications) noexcept {
                    using Modification = typename std::remove_cvref_t<decltype(modifications)>::value_type;
                    if constexpr (std::is_same_v<Modification, BindlessArrayUpdateCommand::Modification>) {
                        writer.write_u8(0u);
                    } else if constexpr (std::is_same_v<Modification, BindlessArrayUpdateCommand::BufferModification>) {
                        writer.write_u8(1u);
                    } else if constexpr (std::is_same_v<Modification, BindlessArrayUpdateCommand::Texture2DModification>) {
                        writer.write_u8(2u);
                    } else {
                        writer.write_u8(3u);
                    }
                    writer.write_u64(modifications.size());
                    for (auto &&modification : modifications) {
                        writer.write_u64(modification.slot);
                        if constexpr (std::is_same_v<Modification, BindlessArrayUpdateCommand::Modification>) {
                            write_buffer(modification.buffer);
                            write_texture(modification.tex2d);
                            write_texture(modification.tex3d);
                        } else if constexpr (std::is_same_v<Modification, BindlessArrayUpdateCommand::BufferModification>) {
                            write_buffer(modification.buffer);
                        } else if constexpr (std::is_same_v<Modification, BindlessArrayUpdateCommand::Texture2DModification>) {
                            write_texture(modification.tex2d);
                        } else {
                            write_texture(modification.tex3d);
                        }
                    }
                });
                break;
            }
            case Command::Tag::EMeshBuildCommand: {
                auto c = static_cast<const MeshBuildCommand *>(command);
                writer.write_u16(static_cast<uint16_t>(WireCommand::MESH_BUILD));
                writer.write_u64(c->handle());
                writer.write_u32(static_cast<uint32_t>(c->request()));
                writer.write_u64(c->vertex_buffer());
                writer.write_u64(c->vertex_buffer_offset());
                writer.write_u64(c->vertex_buffer_size());
                writer.write_u64(c->vertex_stride());
                writer.write_u64(c->triangle_buffer());
                writer.write_u64(c->triangle_buffer_offset());
                writer.write_u64(c->triangle_buffer_size());
                break;
            }
            case Command::Tag::ECurveBuildCommand: {
                auto c = static_cast<const CurveBuildCommand *>(command);
                writer.write_u16(static_cast<uint16_t>(WireCommand::CURVE_BUILD));
                writer.write_u64(c->handle());
                writer.write_u32(static_cast<uint32_t>(c->request()));
                writer.write_u32(static_cast<uint32_t>(c->basis()));
                writer.write_u64(c->cp_count());
                writer.write_u64(c->seg_count());
                writer.write_u64(c->cp_buffer());
                writer.write_u64(c->cp_buffer_offset());
                writer.write_u64(c->cp_stride());
                writer.write_u64(c->seg_buffer());
                writer.write_u64(c->seg_buffer_offset());
                break;
            }
            case Command::Tag::EProceduralPrimitiveBuildCommand: {
                auto c = static_cast<const ProceduralPrimitiveBuildCommand *>(command);
                writer.write_u16(static_cast<uint16_t>(WireCommand::PROCEDURAL_PRIMITIVE_BUILD));
                writer.write_u64(c->handle());
                writer.write_u32(static_cast<uint32_t>(c->request()));
                writer.write_u64(c->aabb_buffer());
                writer.write_u64(c->aabb_buffer_offset());
                writer.write_u64(c->aabb_buffer_size());
                break;
            }
            case Command::Tag::EMotionInstanceBuildCommand: {
                auto c = static_cast<const MotionInstanceBuildCommand *>(command);
                writer.write_u16(static_cast<uint16_t>(WireCommand::MOTION_INSTANCE_BUILD));
                writer.write_u64(c->handle());
                writer.write_u64(c->child());
                writer.write_u64(c->keyframes().size());
                for (auto &&keyframe : c->keyframes()) {
                    for (auto value : keyframe.data) { writer.write_f32(value); }
                }
                break;
            }
            case Command::Tag::EAccelBuildCommand: {
                auto c = static_cast<const AccelBuildCommand *>(command);
                writer.write_u16(static_cast<uint16_t>(WireCommand::ACCEL_BUILD));
                writer.write_u64(c->handle());
                writer.write_u32(c->instance_count());
                writer.write_u32(static_cast<uint32_t>(c->request()));
                writer.write_bool(c->update_instance_buffer_only());
                writer.write_u64(c->modifications().size());
                for (auto &&modification : c->modifications()) {
                    writer.write_u32(modification.index);
                    writer.write_u32(modification.user_id);
                    writer.write_u32(modification.flags);
                    writer.write_u32(modification.vis_mask);
                    for (auto value : modification.affine) { writer.write_f32(value); }
                    if ((modification.flags &
                         AccelBuildCommand::Modification::flag_primitive) != 0u) {
                        writer.write_u64(modification.primitive);
                    }
                }
                break;
            }
            default:
                result.error = "Remote backend does not support this command type in protocol v1.";
                return result;
        }
        if (writer.bytes().size() > limits.max_frame_payload) {
            result.error = "Encoded remote submission exceeds the configured frame limit.";
            return result;
        }
    }
    result.payload = std::move(writer).take();
    return result;
}

DecodedSubmission decode_submission(
    luisa::span<const std::byte> payload,
    const CommandHandleResolver &resolver,
    const ProtocolLimits &limits) noexcept {
    DecodedSubmission result;
    Reader reader{payload, limits};
    result.stream_handle = reader.read_u64();
    result.submission_id = reader.read_u64();
    auto command_count = reader.read_u64();
    auto fail = [&](luisa::string error) noexcept -> DecodedSubmission {
        result.command_list.clear();
        result.downloads.clear();
        result.upload_blobs.clear();
        result.upload_blob_indices.clear();
        result.error = std::move(error);
        return std::move(result);
    };
    if (!reader.ok()) { return fail(luisa::string{reader.error()}); }
    if (result.submission_id == 0u) {
        return fail("Remote submission ID must be nonzero.");
    }
    if (command_count > limits.max_array_size ||
        command_count > std::numeric_limits<size_t>::max()) {
        return fail("Remote command count exceeds the configured limit.");
    }
    uint64_t completion_size = 26u;// id + status + empty message + count
    auto reserve_download = [&](uint64_t size) noexcept {
        constexpr uint64_t per_download_overhead = 12u;// index + blob length
        if (completion_size > limits.max_frame_payload ||
            size > limits.max_frame_payload - completion_size ||
            per_download_overhead >
                limits.max_frame_payload - completion_size - size) {
            return false;
        }
        completion_size += per_download_overhead + size;
        return true;
    };
    result.command_list.reserve(static_cast<size_t>(command_count), 1u);
    for (uint64_t command_index = 0u; command_index < command_count; command_index++) {
        auto wire_command = static_cast<WireCommand>(reader.read_u16());
        if (!reader.ok()) { return fail(luisa::string{reader.error()}); }
        switch (wire_command) {
            case WireCommand::BUFFER_UPLOAD:
            case WireCommand::BUFFER_UPLOAD_CACHED: {
                auto remote = reader.read_u64();
                auto offset = reader.read_u64();
                auto size = reader.read_u64();
                uint64_t native{};
                size_t total{};
                luisa::string error;
                if (!resolver.resolve_buffer(remote, native, total, error)) {
                    return fail(std::move(error));
                }
                if (!valid_range(offset, size, total) ||
                    size > std::numeric_limits<size_t>::max()) {
                    return fail("Remote buffer upload range is out of bounds.");
                }
                const std::byte *data{};
                if (wire_command == WireCommand::BUFFER_UPLOAD_CACHED) {
                    auto blob_index = reader.read_u32();
                    BlobCache::BlobPtr blob;
                    if (!reader.ok()) {
                        return fail(luisa::string{reader.error()});
                    }
                    if (!resolver.resolve_upload_blob(
                            result.submission_id, blob_index,
                            static_cast<size_t>(size), blob, error)) {
                        return fail(std::move(error));
                    }
                    data = blob->data();
                    result.upload_blobs.emplace_back(std::move(blob));
                    result.upload_blob_indices.emplace_back(blob_index);
                } else {
                    auto inline_data = reader.read_bytes(
                        static_cast<size_t>(size));
                    if (!reader.ok()) {
                        return fail(luisa::string{reader.error()});
                    }
                    data = inline_data.data();
                }
                result.command_list.append(luisa::make_unique<BufferUploadCommand>(
                    native, static_cast<size_t>(offset),
                    static_cast<size_t>(size), data));
                break;
            }
            case WireCommand::BUFFER_DOWNLOAD: {
                auto download_index = reader.read_u32();
                auto remote = reader.read_u64();
                auto offset = reader.read_u64();
                auto size = reader.read_u64();
                uint64_t native{};
                size_t total{};
                luisa::string error;
                if (!resolver.resolve_buffer(remote, native, total, error)) {
                    return fail(std::move(error));
                }
                if (!valid_range(offset, size, total) ||
                    size > std::numeric_limits<size_t>::max()) {
                    return fail("Remote buffer download range is out of bounds.");
                }
                if (download_index != result.downloads.size()) {
                    return fail("Remote download indices must be contiguous and ordered.");
                }
                if (!reserve_download(size)) {
                    return fail("Remote downloads exceed the completion-frame limit.");
                }
                auto storage = luisa::make_shared<luisa::vector<std::byte>>();
                storage->resize(static_cast<size_t>(size));
                result.command_list.append(luisa::make_unique<BufferDownloadCommand>(
                    native, static_cast<size_t>(offset), static_cast<size_t>(size), storage->data()));
                result.downloads.emplace_back(ServerDownload{download_index, std::move(storage)});
                break;
            }
            case WireCommand::BUFFER_COPY: {
                auto remote_src = reader.read_u64();
                auto remote_dst = reader.read_u64();
                auto src_offset = reader.read_u64();
                auto dst_offset = reader.read_u64();
                auto size = reader.read_u64();
                uint64_t src{};
                uint64_t dst{};
                size_t src_total{};
                size_t dst_total{};
                luisa::string error;
                if (!resolver.resolve_buffer(remote_src, src, src_total, error) ||
                    !resolver.resolve_buffer(remote_dst, dst, dst_total, error)) {
                    return fail(std::move(error));
                }
                if (!valid_range(src_offset, size, src_total) ||
                    !valid_range(dst_offset, size, dst_total) ||
                    size > std::numeric_limits<size_t>::max()) {
                    return fail("Remote buffer copy range is out of bounds.");
                }
                result.command_list.append(luisa::make_unique<BufferCopyCommand>(
                    src, dst, static_cast<size_t>(src_offset),
                    static_cast<size_t>(dst_offset), static_cast<size_t>(size)));
                break;
            }
            case WireCommand::BUFFER_TO_TEXTURE_COPY:
            case WireCommand::TEXTURE_TO_BUFFER_COPY: {
                auto remote_buffer = reader.read_u64();
                auto buffer_offset = reader.read_u64();
                auto remote_texture = reader.read_u64();
                auto storage_value = reader.read_u32();
                auto level = reader.read_u32();
                auto size = read_uint3(reader);
                auto texture_offset = read_uint3(reader);
                uint64_t buffer{};
                ResolvedTexture texture;
                size_t buffer_total{};
                size_t size_bytes{};
                luisa::string error;
                if (!valid_storage(storage_value) ||
                    !texture_byte_size(static_cast<PixelStorage>(storage_value), size, size_bytes)) {
                    return fail("Remote texture copy has an invalid storage or extent.");
                }
                if (!resolver.resolve_buffer(remote_buffer, buffer, buffer_total, error) ||
                    !resolver.resolve_texture_descriptor(
                        remote_texture, texture, error)) {
                    return fail(std::move(error));
                }
                auto storage = static_cast<PixelStorage>(storage_value);
                if (!valid_range(buffer_offset, size_bytes, buffer_total) ||
                    !valid_texture_region(
                        texture, storage, level, size, texture_offset)) {
                    return fail("Remote texture copy range or storage is invalid.");
                }
                if (wire_command == WireCommand::BUFFER_TO_TEXTURE_COPY) {
                    result.command_list.append(luisa::make_unique<BufferToTextureCopyCommand>(
                        buffer, static_cast<size_t>(buffer_offset), texture.handle,
                        storage, level, size, texture_offset));
                } else {
                    result.command_list.append(luisa::make_unique<TextureToBufferCopyCommand>(
                        buffer, static_cast<size_t>(buffer_offset), texture.handle,
                        storage, level, size, texture_offset));
                }
                break;
            }
            case WireCommand::TEXTURE_UPLOAD:
            case WireCommand::TEXTURE_UPLOAD_CACHED: {
                auto remote = reader.read_u64();
                auto storage_value = reader.read_u32();
                auto level = reader.read_u32();
                auto size = read_uint3(reader);
                auto offset = read_uint3(reader);
                auto declared_size = reader.read_u64();
                ResolvedTexture texture;
                size_t expected_size{};
                luisa::string error;
                if (!valid_storage(storage_value) ||
                    !texture_byte_size(static_cast<PixelStorage>(storage_value), size, expected_size) ||
                    declared_size != expected_size) {
                    return fail("Remote texture upload byte size is invalid.");
                }
                if (!resolver.resolve_texture_descriptor(remote, texture, error)) {
                    return fail(std::move(error));
                }
                auto transfer_storage = static_cast<PixelStorage>(storage_value);
                if (!valid_texture_region(
                        texture, transfer_storage, level, size, offset)) {
                    return fail("Remote texture upload range or storage is invalid.");
                }
                const std::byte *data{};
                if (wire_command == WireCommand::TEXTURE_UPLOAD_CACHED) {
                    auto blob_index = reader.read_u32();
                    BlobCache::BlobPtr blob;
                    if (!reader.ok()) {
                        return fail(luisa::string{reader.error()});
                    }
                    if (!resolver.resolve_upload_blob(
                            result.submission_id, blob_index,
                            expected_size, blob, error)) {
                        return fail(std::move(error));
                    }
                    data = blob->data();
                    result.upload_blobs.emplace_back(std::move(blob));
                    result.upload_blob_indices.emplace_back(blob_index);
                } else {
                    auto inline_data = reader.read_bytes(expected_size);
                    if (!reader.ok()) {
                        return fail(luisa::string{reader.error()});
                    }
                    data = inline_data.data();
                }
                result.command_list.append(luisa::make_unique<TextureUploadCommand>(
                    texture.handle, transfer_storage, level,
                    size, data, offset));
                break;
            }
            case WireCommand::TEXTURE_DOWNLOAD: {
                auto download_index = reader.read_u32();
                auto remote = reader.read_u64();
                auto storage_value = reader.read_u32();
                auto level = reader.read_u32();
                auto size = read_uint3(reader);
                auto offset = read_uint3(reader);
                auto declared_size = reader.read_u64();
                ResolvedTexture texture;
                size_t expected_size{};
                luisa::string error;
                if (!valid_storage(storage_value) ||
                    !texture_byte_size(static_cast<PixelStorage>(storage_value), size, expected_size) ||
                    declared_size != expected_size) {
                    return fail("Remote texture download byte size is invalid.");
                }
                if (!resolver.resolve_texture_descriptor(remote, texture, error)) {
                    return fail(std::move(error));
                }
                auto transfer_storage = static_cast<PixelStorage>(storage_value);
                if (!valid_texture_region(
                        texture, transfer_storage, level, size, offset)) {
                    return fail("Remote texture download range or storage is invalid.");
                }
                if (download_index != result.downloads.size()) {
                    return fail("Remote download indices must be contiguous and ordered.");
                }
                if (!reserve_download(expected_size)) {
                    return fail("Remote downloads exceed the completion-frame limit.");
                }
                auto staging = luisa::make_shared<luisa::vector<std::byte>>();
                staging->resize(expected_size);
                result.command_list.append(luisa::make_unique<TextureDownloadCommand>(
                    texture.handle, transfer_storage, level,
                    size, staging->data(), offset));
                result.downloads.emplace_back(ServerDownload{download_index, std::move(staging)});
                break;
            }
            case WireCommand::TEXTURE_COPY: {
                auto storage_value = reader.read_u32();
                auto remote_src = reader.read_u64();
                auto remote_dst = reader.read_u64();
                auto src_level = reader.read_u32();
                auto dst_level = reader.read_u32();
                auto size = read_uint3(reader);
                auto src_offset = read_uint3(reader);
                auto dst_offset = read_uint3(reader);
                ResolvedTexture src;
                ResolvedTexture dst;
                size_t unused{};
                luisa::string error;
                if (!valid_storage(storage_value) ||
                    !texture_byte_size(static_cast<PixelStorage>(storage_value), size, unused)) {
                    return fail("Remote texture copy has an invalid storage or extent.");
                }
                if (!resolver.resolve_texture_descriptor(remote_src, src, error) ||
                    !resolver.resolve_texture_descriptor(remote_dst, dst, error)) {
                    return fail(std::move(error));
                }
                auto storage = static_cast<PixelStorage>(storage_value);
                if (!valid_texture_region(
                        src, storage, src_level, size, src_offset) ||
                    !valid_texture_region(
                        dst, storage, dst_level, size, dst_offset)) {
                    return fail("Remote texture copy range or storage is invalid.");
                }
                result.command_list.append(luisa::make_unique<TextureCopyCommand>(
                    storage, src.handle, dst.handle,
                    src_level, dst_level, size, src_offset, dst_offset));
                break;
            }
            case WireCommand::SHADER_DISPATCH: {
                auto remote_shader = reader.read_u64();
                auto argument_count = reader.read_u64();
                ResolvedShader resolved_shader;
                luisa::string error;
                if (!resolver.resolve_shader(
                        remote_shader, resolved_shader, error)) {
                    return fail(std::move(error));
                }
                if (argument_count > limits.max_array_size ||
                    argument_count >
                        std::numeric_limits<size_t>::max() / sizeof(Argument) ||
                    argument_count != resolved_shader.arguments.size()) {
                    return fail("Remote shader argument count does not match the shader signature.");
                }
                auto argument_count_size = static_cast<size_t>(argument_count);
                luisa::vector<std::byte> argument_buffer;
                argument_buffer.resize(argument_count_size * sizeof(Argument));
                for (size_t i = 0u; i < argument_count_size; i++) {
                    Argument argument{};
                    argument.tag = static_cast<Argument::Tag>(reader.read_u8());
                    auto expected = resolved_shader.arguments[i];
                    if (argument.tag != expected.tag) {
                        return fail("Remote shader argument tag does not match the shader signature.");
                    }
                    switch (argument.tag) {
                        case Argument::Tag::BUFFER: {
                            auto remote = reader.read_u64();
                            auto offset = reader.read_u64();
                            auto size = reader.read_u64();
                            ResolvedBuffer buffer;
                            if (!resolver.resolve_buffer_descriptor(
                                    remote, buffer, error)) {
                                return fail(std::move(error));
                            }
                            if (buffer.is_indirect_dispatch !=
                                expected.indirect_dispatch_buffer) {
                                return fail("Remote shader buffer kind does not match the shader signature.");
                            }
                            auto total = buffer.is_indirect_dispatch ?
                                             buffer.indirect_dispatch_capacity :
                                             buffer.size_bytes;
                            if (!valid_range(offset, size, total) ||
                                offset > std::numeric_limits<size_t>::max() ||
                                size > std::numeric_limits<size_t>::max()) {
                                return fail("Remote shader buffer argument is out of bounds.");
                            }
                            argument.buffer.handle = buffer.handle;
                            argument.buffer.offset = static_cast<size_t>(offset);
                            argument.buffer.size = static_cast<size_t>(size);
                            break;
                        }
                        case Argument::Tag::TEXTURE: {
                            ResolvedTexture texture;
                            if (!resolver.resolve_texture_descriptor(
                                    reader.read_u64(), texture, error)) {
                                return fail(std::move(error));
                            }
                            argument.texture.level = reader.read_u32();
                            if (argument.texture.level >= texture.mip_levels) {
                                return fail("Remote shader texture level is out of bounds.");
                            }
                            argument.texture.handle = texture.handle;
                            break;
                        }
                        case Argument::Tag::UNIFORM: {
                            auto alignment = reader.read_u64();
                            auto value = reader.read_blob();
                            if (alignment == 0u || alignment > 16u ||
                                (alignment & (alignment - 1u)) != 0u ||
                                alignment != expected.uniform_alignment ||
                                value.size() != expected.uniform_size) {
                                return fail("Remote shader uniform layout does not match the shader signature.");
                            }
                            auto mask = static_cast<size_t>(alignment - 1u);
                            auto aligned_offset =
                                (argument_buffer.size() + mask) & ~mask;
                            argument_buffer.resize(aligned_offset);
                            argument.uniform.offset = argument_buffer.size();
                            argument.uniform.size = value.size();
                            argument.uniform.alignment = static_cast<size_t>(alignment);
                            argument_buffer.insert(
                                argument_buffer.end(), value.begin(), value.end());
                            break;
                        }
                        case Argument::Tag::BINDLESS_ARRAY:
                            if (!resolver.resolve_bindless_array(
                                    reader.read_u64(), argument.bindless_array.handle, error)) {
                                return fail(std::move(error));
                            }
                            break;
                        case Argument::Tag::ACCEL:
                            if (!resolver.resolve_accel(
                                    reader.read_u64(), argument.accel.handle, error)) {
                                return fail(std::move(error));
                            }
                            break;
                        default:
                            return fail("Remote shader argument tag is invalid.");
                    }
                    if (!reader.ok()) { return fail(luisa::string{reader.error()}); }
                    std::memcpy(argument_buffer.data() + i * sizeof(Argument),
                                &argument, sizeof(Argument));
                }
                auto dispatch_kind = reader.read_u8();
                if (dispatch_kind == 0u) {
                    result.command_list.append(luisa::make_unique<ShaderDispatchCommand>(
                        resolved_shader.handle, std::move(argument_buffer), argument_count_size,
                        read_uint3(reader)));
                } else if (dispatch_kind == 1u) {
                    auto remote_indirect = reader.read_u64();
                    IndirectDispatchArg indirect{};
                    ResolvedBuffer buffer;
                    if (!resolver.resolve_buffer_descriptor(
                            remote_indirect, buffer, error)) {
                        return fail(std::move(error));
                    }
                    indirect.offset = reader.read_u32();
                    indirect.max_dispatch_size = reader.read_u32();
                    auto plan = lc::plan_indirect_dispatch(
                        buffer.indirect_dispatch_capacity,
                        indirect.offset, indirect.max_dispatch_size);
                    if (!buffer.is_indirect_dispatch || !plan) {
                        return fail("Remote indirect dispatch requires a valid indirect-dispatch buffer range.");
                    }
                    indirect.handle = buffer.handle;
                    result.command_list.append(luisa::make_unique<ShaderDispatchCommand>(
                        resolved_shader.handle, std::move(argument_buffer),
                        argument_count_size, indirect));
                } else if (dispatch_kind == 2u) {
                    auto dispatch_count = reader.read_u64();
                    if (dispatch_count > limits.max_array_size ||
                        dispatch_count > std::numeric_limits<size_t>::max()) {
                        return fail("Remote dispatch count exceeds the configured limit.");
                    }
                    luisa::vector<uint3> dispatches;
                    dispatches.reserve(static_cast<size_t>(dispatch_count));
                    for (uint64_t i = 0u; i < dispatch_count; i++) {
                        dispatches.emplace_back(read_uint3(reader));
                    }
                    result.command_list.append(luisa::make_unique<ShaderDispatchCommand>(
                        resolved_shader.handle, std::move(argument_buffer), argument_count_size,
                        std::move(dispatches)));
                } else {
                    return fail("Remote shader dispatch kind is invalid.");
                }
                break;
            }
            case WireCommand::BINDLESS_ARRAY_UPDATE: {
                auto remote_array = reader.read_u64();
                auto variant = reader.read_u8();
                auto modification_count = reader.read_u64();
                uint64_t native_array{};
                size_t slot_count{};
                BindlessSlotType slot_type{};
                luisa::string error;
                if (!resolver.resolve_bindless_array_for_update(
                        remote_array, native_array, slot_count,
                        slot_type, error)) {
                    return fail(std::move(error));
                }
                if (variant > 3u ||
                    variant != static_cast<uint8_t>(slot_type) ||
                    modification_count > limits.max_array_size ||
                    modification_count > std::numeric_limits<size_t>::max()) {
                    return fail("Remote bindless update variant or count is invalid.");
                }
                auto read_buffer = [&](BindlessArrayUpdateCommand::ModifiedBuffer &buffer) noexcept {
                    auto operation = static_cast<BindlessArrayUpdateCommand::Operation>(reader.read_u8());
                    if (operation == BindlessArrayUpdateCommand::Operation::NONE) {
                        buffer = {};
                        return true;
                    }
                    if (operation == BindlessArrayUpdateCommand::Operation::REMOVE) {
                        buffer = BindlessArrayUpdateCommand::ModifiedBuffer::remove();
                        return true;
                    }
                    if (operation != BindlessArrayUpdateCommand::Operation::EMPLACE) {
                        error = "Remote bindless buffer operation is invalid.";
                        return false;
                    }
                    auto remote = reader.read_u64();
                    auto offset = reader.read_u64();
                    auto size = reader.read_u64();
                    uint64_t native{};
                    size_t total{};
                    if (!resolver.resolve_buffer(remote, native, total, error)) {
                        return false;
                    }
                    auto whole = BindlessArrayUpdateCommand::ModifiedBuffer::whole_buffer_size;
                    if (offset > std::numeric_limits<size_t>::max() ||
                        size > std::numeric_limits<size_t>::max() ||
                        (size == whole ? offset > total :
                                         !valid_range(offset, size, total))) {
                        error = "Remote bindless buffer range is out of bounds.";
                        return false;
                    }
                    buffer = BindlessArrayUpdateCommand::ModifiedBuffer::emplace(
                        native, static_cast<size_t>(offset), static_cast<size_t>(size));
                    return true;
                };
                auto read_texture = [&](BindlessArrayUpdateCommand::ModifiedTexture &texture) noexcept {
                    auto operation = static_cast<BindlessArrayUpdateCommand::Operation>(reader.read_u8());
                    if (operation == BindlessArrayUpdateCommand::Operation::NONE) {
                        texture = {};
                        return true;
                    }
                    if (operation == BindlessArrayUpdateCommand::Operation::REMOVE) {
                        texture = BindlessArrayUpdateCommand::ModifiedTexture::remove();
                        return true;
                    }
                    if (operation != BindlessArrayUpdateCommand::Operation::EMPLACE) {
                        error = "Remote bindless texture operation is invalid.";
                        return false;
                    }
                    uint64_t native{};
                    if (!resolver.resolve_texture(reader.read_u64(), native, error)) {
                        return false;
                    }
                    auto sampler_code = reader.read_u32();
                    if (sampler_code > 15u) {
                        error = "Remote bindless sampler code is invalid.";
                        return false;
                    }
                    texture = BindlessArrayUpdateCommand::ModifiedTexture::emplace(
                        native, Sampler::decode(sampler_code));
                    return true;
                };
                auto read_slot = [&]() noexcept -> luisa::optional<size_t> {
                    auto slot = reader.read_u64();
                    if (slot >= slot_count || slot > std::numeric_limits<size_t>::max()) {
                        error = "Remote bindless slot is out of bounds.";
                        return luisa::nullopt;
                    }
                    return static_cast<size_t>(slot);
                };
                auto count = static_cast<size_t>(modification_count);
                if (variant == 0u) {
                    luisa::vector<BindlessArrayUpdateCommand::Modification> modifications;
                    modifications.reserve(count);
                    for (size_t i = 0u; i < count; i++) {
                        auto slot = read_slot();
                        BindlessArrayUpdateCommand::ModifiedBuffer buffer;
                        BindlessArrayUpdateCommand::ModifiedTexture tex2d;
                        BindlessArrayUpdateCommand::ModifiedTexture tex3d;
                        if (!slot || !read_buffer(buffer) ||
                            !read_texture(tex2d) || !read_texture(tex3d)) {
                            return fail(error.empty() ? luisa::string{reader.error()} : std::move(error));
                        }
                        modifications.emplace_back(*slot, buffer, tex2d, tex3d);
                    }
                    result.command_list.append(luisa::make_unique<BindlessArrayUpdateCommand>(
                        native_array, std::move(modifications)));
                } else if (variant == 1u) {
                    luisa::vector<BindlessArrayUpdateCommand::BufferModification> modifications;
                    modifications.reserve(count);
                    for (size_t i = 0u; i < count; i++) {
                        auto slot = read_slot();
                        BindlessArrayUpdateCommand::ModifiedBuffer buffer;
                        if (!slot || !read_buffer(buffer)) {
                            return fail(error.empty() ? luisa::string{reader.error()} : std::move(error));
                        }
                        modifications.emplace_back(*slot, buffer);
                    }
                    result.command_list.append(luisa::make_unique<BindlessArrayUpdateCommand>(
                        native_array, std::move(modifications)));
                } else if (variant == 2u) {
                    luisa::vector<BindlessArrayUpdateCommand::Texture2DModification> modifications;
                    modifications.reserve(count);
                    for (size_t i = 0u; i < count; i++) {
                        auto slot = read_slot();
                        BindlessArrayUpdateCommand::ModifiedTexture texture;
                        if (!slot || !read_texture(texture)) {
                            return fail(error.empty() ? luisa::string{reader.error()} : std::move(error));
                        }
                        modifications.emplace_back(*slot, texture);
                    }
                    result.command_list.append(luisa::make_unique<BindlessArrayUpdateCommand>(
                        native_array, std::move(modifications)));
                } else {
                    luisa::vector<BindlessArrayUpdateCommand::Texture3DModification> modifications;
                    modifications.reserve(count);
                    for (size_t i = 0u; i < count; i++) {
                        auto slot = read_slot();
                        BindlessArrayUpdateCommand::ModifiedTexture texture;
                        if (!slot || !read_texture(texture)) {
                            return fail(error.empty() ? luisa::string{reader.error()} : std::move(error));
                        }
                        modifications.emplace_back(*slot, texture);
                    }
                    result.command_list.append(luisa::make_unique<BindlessArrayUpdateCommand>(
                        native_array, std::move(modifications)));
                }
                break;
            }
            case WireCommand::MESH_BUILD: {
                auto remote_mesh = reader.read_u64();
                auto request_value = reader.read_u32();
                auto remote_vertex_buffer = reader.read_u64();
                auto vertex_offset = reader.read_u64();
                auto vertex_size = reader.read_u64();
                auto vertex_stride = reader.read_u64();
                auto remote_triangle_buffer = reader.read_u64();
                auto triangle_offset = reader.read_u64();
                auto triangle_size = reader.read_u64();
                uint64_t mesh{};
                uint64_t vertex_buffer{};
                uint64_t triangle_buffer{};
                size_t vertex_total{};
                size_t triangle_total{};
                luisa::string error;
                if (request_value > static_cast<uint32_t>(AccelBuildRequest::FORCE_BUILD) ||
                    vertex_stride == 0u ||
                    vertex_stride > std::numeric_limits<size_t>::max() ||
                    !resolver.resolve_mesh(remote_mesh, mesh, error) ||
                    !resolver.resolve_buffer(
                        remote_vertex_buffer, vertex_buffer, vertex_total, error) ||
                    !resolver.resolve_buffer(
                        remote_triangle_buffer, triangle_buffer, triangle_total, error)) {
                    return fail(error.empty() ? "Remote mesh-build descriptor is invalid." : std::move(error));
                }
                if (!valid_range(vertex_offset, vertex_size, vertex_total) ||
                    !valid_range(triangle_offset, triangle_size, triangle_total) ||
                    vertex_offset > std::numeric_limits<size_t>::max() ||
                    vertex_size > std::numeric_limits<size_t>::max() ||
                    triangle_offset > std::numeric_limits<size_t>::max() ||
                    triangle_size > std::numeric_limits<size_t>::max()) {
                    return fail("Remote mesh-build buffer range is out of bounds.");
                }
                result.command_list.append(luisa::make_unique<MeshBuildCommand>(
                    mesh, static_cast<AccelBuildRequest>(request_value),
                    vertex_buffer, static_cast<size_t>(vertex_offset),
                    static_cast<size_t>(vertex_size), static_cast<size_t>(vertex_stride),
                    triangle_buffer, static_cast<size_t>(triangle_offset),
                    static_cast<size_t>(triangle_size)));
                break;
            }
            case WireCommand::CURVE_BUILD: {
                auto remote_curve = reader.read_u64();
                auto request_value = reader.read_u32();
                auto basis_value = reader.read_u32();
                auto cp_count = reader.read_u64();
                auto seg_count = reader.read_u64();
                auto remote_cp_buffer = reader.read_u64();
                auto cp_offset = reader.read_u64();
                auto cp_stride = reader.read_u64();
                auto remote_seg_buffer = reader.read_u64();
                auto seg_offset = reader.read_u64();
                uint64_t curve{};
                uint64_t cp_buffer{};
                uint64_t seg_buffer{};
                size_t cp_total{};
                size_t seg_total{};
                luisa::string error;
                if (request_value > static_cast<uint32_t>(AccelBuildRequest::FORCE_BUILD) ||
                    basis_value >= curve_basis_count || cp_stride == 0u ||
                    cp_count > limits.max_array_size ||
                    seg_count > limits.max_array_size ||
                    cp_count > std::numeric_limits<size_t>::max() ||
                    seg_count > std::numeric_limits<size_t>::max() ||
                    cp_stride > std::numeric_limits<size_t>::max() ||
                    !resolver.resolve_curve(remote_curve, curve, error) ||
                    !resolver.resolve_buffer(remote_cp_buffer, cp_buffer, cp_total, error) ||
                    !resolver.resolve_buffer(remote_seg_buffer, seg_buffer, seg_total, error)) {
                    return fail(error.empty() ? "Remote curve-build descriptor is invalid." : std::move(error));
                }
                if (cp_count > std::numeric_limits<uint64_t>::max() / cp_stride ||
                    seg_count > std::numeric_limits<uint64_t>::max() / sizeof(uint)) {
                    return fail("Remote curve-build byte-size overflow.");
                }
                auto cp_size = cp_count * cp_stride;
                auto seg_size = seg_count * sizeof(uint);
                if (!valid_range(cp_offset, cp_size, cp_total) ||
                    !valid_range(seg_offset, seg_size, seg_total) ||
                    cp_offset > std::numeric_limits<size_t>::max() ||
                    seg_offset > std::numeric_limits<size_t>::max()) {
                    return fail("Remote curve-build buffer range is out of bounds.");
                }
                result.command_list.append(luisa::make_unique<CurveBuildCommand>(
                    curve, static_cast<AccelBuildRequest>(request_value),
                    static_cast<CurveBasis>(basis_value),
                    static_cast<size_t>(cp_count), static_cast<size_t>(seg_count),
                    cp_buffer, static_cast<size_t>(cp_offset),
                    static_cast<size_t>(cp_stride), seg_buffer,
                    static_cast<size_t>(seg_offset)));
                break;
            }
            case WireCommand::PROCEDURAL_PRIMITIVE_BUILD: {
                auto remote_primitive = reader.read_u64();
                auto request_value = reader.read_u32();
                auto remote_buffer = reader.read_u64();
                auto offset = reader.read_u64();
                auto size = reader.read_u64();
                uint64_t primitive{};
                uint64_t buffer{};
                size_t total{};
                luisa::string error;
                if (request_value > static_cast<uint32_t>(AccelBuildRequest::FORCE_BUILD) ||
                    !resolver.resolve_procedural_primitive(
                        remote_primitive, primitive, error) ||
                    !resolver.resolve_buffer(remote_buffer, buffer, total, error)) {
                    return fail(error.empty() ? "Remote procedural-build descriptor is invalid." : std::move(error));
                }
                if (!valid_range(offset, size, total) ||
                    offset > std::numeric_limits<size_t>::max() ||
                    size > std::numeric_limits<size_t>::max()) {
                    return fail("Remote procedural-build buffer range is out of bounds.");
                }
                result.command_list.append(
                    luisa::make_unique<ProceduralPrimitiveBuildCommand>(
                        primitive, static_cast<AccelBuildRequest>(request_value),
                        buffer, static_cast<size_t>(offset), static_cast<size_t>(size)));
                break;
            }
            case WireCommand::MOTION_INSTANCE_BUILD: {
                auto remote_instance = reader.read_u64();
                auto remote_child = reader.read_u64();
                auto keyframe_count = reader.read_u64();
                uint64_t instance{};
                uint64_t child{};
                size_t expected_keyframes{};
                luisa::string error;
                if (!resolver.resolve_motion_instance(
                        remote_instance, instance, expected_keyframes, error) ||
                    !resolver.resolve_primitive(remote_child, child, error)) {
                    return fail(std::move(error));
                }
                if (keyframe_count != expected_keyframes ||
                    keyframe_count > limits.max_array_size) {
                    return fail("Remote motion-instance keyframe count is invalid.");
                }
                luisa::vector<MotionInstanceTransform> keyframes;
                keyframes.resize(expected_keyframes);
                for (auto &keyframe : keyframes) {
                    for (auto &value : keyframe.data) {
                        value = reader.read_f32();
                        if (!std::isfinite(value)) {
                            return fail("Remote motion-instance transform is not finite.");
                        }
                    }
                }
                result.command_list.append(
                    luisa::make_unique<MotionInstanceBuildCommand>(
                        instance, child, std::move(keyframes)));
                break;
            }
            case WireCommand::ACCEL_BUILD: {
                auto remote_accel = reader.read_u64();
                auto instance_count = reader.read_u32();
                auto request_value = reader.read_u32();
                auto update_instance_buffer_only = reader.read_bool();
                auto modification_count = reader.read_u64();
                uint64_t accel{};
                luisa::string error;
                if (request_value > static_cast<uint32_t>(AccelBuildRequest::FORCE_BUILD) ||
                    instance_count > limits.max_array_size ||
                    modification_count > limits.max_array_size ||
                    modification_count > std::numeric_limits<size_t>::max() ||
                    !resolver.resolve_accel_resource(remote_accel, accel, error)) {
                    return fail(error.empty() ? "Remote accel-build descriptor is invalid." : std::move(error));
                }
                constexpr auto allowed_flags =
                    AccelBuildCommand::Modification::flag_primitive |
                    AccelBuildCommand::Modification::flag_transform |
                    AccelBuildCommand::Modification::flag_opaque |
                    AccelBuildCommand::Modification::flag_visibility |
                    AccelBuildCommand::Modification::flag_user_id;
                luisa::vector<AccelBuildCommand::Modification> modifications;
                modifications.reserve(static_cast<size_t>(modification_count));
                for (uint64_t i = 0u; i < modification_count; i++) {
                    AccelBuildCommand::Modification modification;
                    modification.index = reader.read_u32();
                    modification.user_id = reader.read_u32();
                    modification.flags = reader.read_u32();
                    modification.vis_mask = reader.read_u32();
                    for (auto &value : modification.affine) {
                        value = reader.read_f32();
                        if (!std::isfinite(value)) {
                            return fail("Remote accel-build transform is not finite.");
                        }
                    }
                    if ((modification.flags & ~allowed_flags) != 0u ||
                        (modification.flags & AccelBuildCommand::Modification::flag_opaque) ==
                            AccelBuildCommand::Modification::flag_opaque ||
                        modification.vis_mask > 0xffu ||
                        modification.index >= instance_count) {
                        return fail("Remote accel-build modification is invalid.");
                    }
                    if ((modification.flags &
                         AccelBuildCommand::Modification::flag_primitive) != 0u) {
                        if (!resolver.resolve_primitive(
                                reader.read_u64(), modification.primitive, error)) {
                            return fail(std::move(error));
                        }
                    }
                    modifications.emplace_back(modification);
                }
                result.command_list.append(luisa::make_unique<AccelBuildCommand>(
                    accel, instance_count,
                    static_cast<AccelBuildRequest>(request_value),
                    std::move(modifications), update_instance_buffer_only));
                break;
            }
            default:
                return fail("Remote command kind is unsupported.");
        }
        if (!reader.ok()) { return fail(luisa::string{reader.error()}); }
    }
    if (!reader.finish()) { return fail(luisa::string{reader.error()}); }
    auto committed = std::move(result.command_list.commit()).command_list();
    return DecodedSubmission{
        .stream_handle = result.stream_handle,
        .submission_id = result.submission_id,
        .command_list = std::move(committed),
        .downloads = std::move(result.downloads),
        .upload_blobs = std::move(result.upload_blobs),
        .upload_blob_indices = std::move(result.upload_blob_indices)};
}

}// namespace luisa::compute::remote
