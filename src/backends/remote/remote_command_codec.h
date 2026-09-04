#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/rhi/device_interface.h>

#include "remote_blob_cache.h"
#include "remote_protocol.h"

namespace luisa::compute::remote {

enum class WireCommand : uint16_t {
    BUFFER_UPLOAD = 1u,
    BUFFER_DOWNLOAD = 2u,
    BUFFER_COPY = 3u,
    BUFFER_TO_TEXTURE_COPY = 4u,
    TEXTURE_TO_BUFFER_COPY = 5u,
    TEXTURE_UPLOAD = 6u,
    TEXTURE_DOWNLOAD = 7u,
    TEXTURE_COPY = 8u,
    SHADER_DISPATCH = 9u,
    BINDLESS_ARRAY_UPDATE = 10u,
    MESH_BUILD = 11u,
    CURVE_BUILD = 12u,
    PROCEDURAL_PRIMITIVE_BUILD = 13u,
    MOTION_INSTANCE_BUILD = 14u,
    ACCEL_BUILD = 15u,
    BUFFER_UPLOAD_CACHED = 16u,
    TEXTURE_UPLOAD_CACHED = 17u,
};

struct DownloadTarget {
    void *data{};
    size_t size{};
};

struct ResolvedTexture {
    uint64_t handle{};
    PixelStorage storage{};
    uint3 size{};
    uint32_t mip_levels{};
};

struct ResolvedBuffer {
    uint64_t handle{};
    size_t size_bytes{};
    bool is_indirect_dispatch{};
    size_t indirect_dispatch_capacity{};
};

struct ShaderArgumentDesc {
    Argument::Tag tag{};
    size_t uniform_size{};
    size_t uniform_alignment{};
    bool indirect_dispatch_buffer{};
};

constexpr uint32_t invalid_blob_index = ~0u;

struct UploadBlob {
    BlobKey key;
    luisa::span<const std::byte> bytes;
};

struct UploadBlobReference {
    size_t command_index{};
    uint32_t index{invalid_blob_index};
};

struct UploadBlobPlan {
    luisa::vector<UploadBlob> blobs;
    luisa::vector<UploadBlobReference> references;
    luisa::string error;

    [[nodiscard]] explicit operator bool() const noexcept {
        return error.empty();
    }

    [[nodiscard]] uint32_t find(
        size_t command_index) const noexcept;
};

struct ResolvedShader {
    uint64_t handle{};
    luisa::span<const ShaderArgumentDesc> arguments;
};

struct EncodedSubmission {
    luisa::vector<std::byte> payload;
    luisa::vector<DownloadTarget> downloads;
    luisa::string error;

    [[nodiscard]] explicit operator bool() const noexcept {
        return error.empty();
    }
};

class CommandHandleResolver {

public:
    virtual ~CommandHandleResolver() noexcept = default;

    [[nodiscard]] virtual bool resolve_buffer(
        uint64_t remote_handle, uint64_t &native_handle,
        size_t &size_bytes, luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_buffer_descriptor(
        uint64_t remote_handle, ResolvedBuffer &buffer,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_texture(
        uint64_t remote_handle, uint64_t &native_handle,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_texture_descriptor(
        uint64_t remote_handle, ResolvedTexture &texture,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_bindless_array(
        uint64_t remote_handle, uint64_t &native_handle,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_bindless_array_for_update(
        uint64_t remote_handle, uint64_t &native_handle,
        size_t &slot_count, BindlessSlotType &slot_type,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_accel(
        uint64_t remote_handle, uint64_t &native_handle,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_shader(
        uint64_t remote_handle, ResolvedShader &shader,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_mesh(
        uint64_t remote_handle, uint64_t &native_handle,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_curve(
        uint64_t remote_handle, uint64_t &native_handle,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_procedural_primitive(
        uint64_t remote_handle, uint64_t &native_handle,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_motion_instance(
        uint64_t remote_handle, uint64_t &native_handle,
        size_t &keyframe_count,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_accel_resource(
        uint64_t remote_handle, uint64_t &native_handle,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_primitive(
        uint64_t remote_handle, uint64_t &native_handle,
        luisa::string &error) const noexcept = 0;
    [[nodiscard]] virtual bool resolve_upload_blob(
        uint64_t submission_id, uint32_t blob_index,
        size_t expected_size, BlobCache::BlobPtr &blob,
        luisa::string &error) const noexcept = 0;
};

struct ServerDownload {
    uint32_t index{};
    luisa::shared_ptr<luisa::vector<std::byte>> storage;
};

struct DecodedSubmission {
    uint64_t stream_handle{};
    uint64_t submission_id{};
    CommandList command_list;
    luisa::vector<ServerDownload> downloads;
    luisa::vector<BlobCache::BlobPtr> upload_blobs;
    luisa::vector<uint32_t> upload_blob_indices;
    luisa::string error;

    [[nodiscard]] explicit operator bool() const noexcept {
        return error.empty();
    }
};

[[nodiscard]] UploadBlobPlan plan_upload_blobs(
    luisa::span<const luisa::unique_ptr<Command>> commands,
    uint64_t minimum_blob_size,
    uint64_t maximum_blob_size,
    const ProtocolLimits &limits = {}) noexcept;

[[nodiscard]] EncodedSubmission encode_submission(
    uint64_t stream_handle, uint64_t submission_id,
    luisa::span<const luisa::unique_ptr<Command>> commands,
    const ProtocolLimits &limits = {},
    const UploadBlobPlan *blob_plan = nullptr) noexcept;

[[nodiscard]] DecodedSubmission decode_submission(
    luisa::span<const std::byte> payload,
    const CommandHandleResolver &resolver,
    const ProtocolLimits &limits = {}) noexcept;

}// namespace luisa::compute::remote
