#include "accel_archive_ext_impl.h"
#include "buffer.h"
#include "device.h"
#include "mesh.h"
#include "rw_resource.h"
#include "stream.h"

namespace lc::validation {

namespace {

[[nodiscard]] Stream *try_get_compute_or_graphics_stream(uint64_t handle) noexcept {
    auto *res = RWResource::try_get<RWResource>(handle);
    if (res == nullptr || res->tag() != Resource::Tag::STREAM) {
        return nullptr;
    }
    auto *stream = static_cast<Stream *>(res);
    if (stream->tag_value() != StreamTag::COMPUTE &&
        stream->tag_value() != StreamTag::GRAPHICS) {
        return nullptr;
    }
    return stream;
}

[[nodiscard]] Mesh *try_get_mesh(uint64_t handle) noexcept {
    auto *res = RWResource::try_get<RWResource>(handle);
    if (res == nullptr || res->tag() != Resource::Tag::MESH) {
        return nullptr;
    }
    return static_cast<Mesh *>(res);
}

[[nodiscard]] Buffer *try_get_buffer(uint64_t handle) noexcept {
    if (handle == invalid_resource_handle) {
        return nullptr;
    }
    auto *res = RWResource::try_get<RWResource>(handle);
    if (res == nullptr || res->tag() != Resource::Tag::BUFFER) {
        return nullptr;
    }
    return static_cast<Buffer *>(res);
}

}// namespace

AccelArchiveResult AccelArchiveExtImpl::export_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &desc,
    luisa::vector<std::byte> &out_archive) noexcept {
    auto *stream = try_get_compute_or_graphics_stream(stream_handle);
    if (stream == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "export_mesh_archive requires a valid COMPUTE or GRAPHICS stream.");
    }
    auto *mesh = try_get_mesh(mesh_handle);
    if (mesh == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "export_mesh_archive requires a valid Mesh handle.");
    }
    auto result = _native->export_mesh_archive(stream_handle, mesh_handle, desc, out_archive);
    if (result) {
        stream->sync();
    }
    return result;
}

AccelArchiveResult AccelArchiveExtImpl::import_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &geometry_binding,
    luisa::span<const std::byte> archive_bytes) noexcept {
    auto *stream = try_get_compute_or_graphics_stream(stream_handle);
    if (stream == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "import_mesh_archive requires a valid COMPUTE or GRAPHICS stream.");
    }
    auto *mesh = try_get_mesh(mesh_handle);
    if (mesh == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "import_mesh_archive requires a valid Mesh handle.");
    }
    const auto caps = _native->capabilities();
    Buffer *rebound_vert = nullptr;
    Buffer *rebound_tri = nullptr;
    if (caps.requires_geometry_rebind) {
        rebound_vert = try_get_buffer(geometry_binding.vertex_buffer);
        rebound_tri = try_get_buffer(geometry_binding.triangle_buffer);
        if (rebound_vert == nullptr || rebound_tri == nullptr) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::INVALID_ARGUMENT,
                "import_mesh_archive requires valid vertex and triangle buffers for geometry rebind.");
        }
    }
    auto result = _native->import_mesh_archive(
        stream_handle, mesh_handle, expected_desc, geometry_binding, archive_bytes);
    if (!result) {
        return result;
    }
    mesh->imported = true;
    if (caps.requires_geometry_rebind && rebound_vert != nullptr && rebound_tri != nullptr) {
        mesh->vert = rebound_vert;
        mesh->index = rebound_tri;
        mesh->vert_range = Range{
            geometry_binding.vertex_buffer_offset,
            geometry_binding.vertex_buffer_size};
        mesh->index_range = Range{
            geometry_binding.triangle_buffer_offset,
            geometry_binding.triangle_buffer_size};
    } else {
        mesh->vert = nullptr;
        mesh->index = nullptr;
        mesh->vert_range = Range{};
        mesh->index_range = Range{};
    }
    if (result.readiness == AccelArchiveReadiness::READY_ON_STREAM) {
        stream->dispatch();
        RWResource::set_usage(stream, mesh, Usage::WRITE, Range{});
        if (caps.requires_geometry_rebind && rebound_vert != nullptr && rebound_tri != nullptr) {
            RWResource::set_usage(stream, rebound_vert, Usage::READ, mesh->vert_range);
            RWResource::set_usage(stream, rebound_tri, Usage::READ, mesh->index_range);
        }
        stream->check_compete();
    } else if (result.readiness == AccelArchiveReadiness::READY_IMMEDIATELY) {
        stream->sync();
    }
    return result;
}

}// namespace lc::validation
