#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/stream.h>
#include <luisa/backends/ext/accel_archive_ext_interface.h>

namespace luisa::compute {

// Caller-owned TLAS instance descriptor referencing a logical BLAS index
// rather than process-local Luisa handles.
struct AccelArchiveInstanceDesc {
    uint32_t blas_index{0u};
    float4x4 transform{make_float4x4(1.f)};
    uint8_t visibility_mask{0xffu};
    bool opaque{true};
    uint32_t user_id{0u};
};

[[nodiscard]] inline AccelArchiveMeshDesc make_accel_archive_mesh_desc(
    const Mesh &mesh,
    const AccelOption &option,
    luisa::Hash128 geometry_content_hash) noexcept {
    AccelArchiveMeshDesc desc{};
    desc.is_cluster = mesh.is_cluster_mode();
    desc.has_cutout_mask = mesh.has_cutout_mask();
    desc.option = option;
    desc.geometry_content_hash = geometry_content_hash;
    if (!desc.is_cluster) {
        desc.vertex_count = mesh.vertex_count_per_motion_keyframe();
        desc.triangle_count = mesh.triangle_count();
        desc.vertex_stride = mesh.vertex_stride();
    }
    return desc;
}

[[nodiscard]] inline AccelArchiveGeometryBinding make_accel_archive_geometry_binding(
    const Mesh &mesh) noexcept {
    AccelArchiveGeometryBinding binding{};
    if (!mesh.is_cluster_mode()) {
        binding.vertex_buffer = mesh.vertex_buffer_handle();
        binding.vertex_buffer_offset = mesh.vertex_buffer_offset_bytes();
        binding.vertex_buffer_size = mesh.vertex_buffer_size_bytes();
        binding.vertex_stride = mesh.vertex_stride();
        binding.triangle_buffer = mesh.triangle_buffer_handle();
        binding.triangle_buffer_offset = mesh.triangle_buffer_offset_bytes();
        binding.triangle_buffer_size = mesh.triangle_buffer_size_bytes();
    }
    return binding;
}

[[nodiscard]] inline AccelArchiveResult export_mesh_archive(
    AccelArchiveExt *ext,
    Stream &stream,
    const Mesh &mesh,
    const AccelOption &option,
    luisa::Hash128 geometry_content_hash,
    luisa::vector<std::byte> &out_archive) noexcept {
    if (ext == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "AccelArchiveExt is not available on this device.");
    }
    auto desc = make_accel_archive_mesh_desc(mesh, option, geometry_content_hash);
    return ext->export_mesh_archive(stream.handle(), mesh.handle(), desc, out_archive);
}

[[nodiscard]] inline AccelArchiveResult validate_mesh_archive(
    const AccelArchiveExt *ext,
    luisa::span<const std::byte> archive_bytes,
    const Mesh &mesh,
    const AccelOption &option,
    luisa::Hash128 geometry_content_hash,
    AccelArchiveMetadata *out_meta = nullptr) noexcept {
    if (ext == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "AccelArchiveExt is not available on this device.");
    }
    auto desc = make_accel_archive_mesh_desc(mesh, option, geometry_content_hash);
    return ext->validate_mesh_archive(archive_bytes, desc, out_meta);
}

[[nodiscard]] inline AccelArchiveResult import_mesh_archive(
    AccelArchiveExt *ext,
    Stream &stream,
    const Mesh &mesh,
    const AccelOption &option,
    luisa::Hash128 geometry_content_hash,
    luisa::span<const std::byte> archive_bytes) noexcept {
    if (ext == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "AccelArchiveExt is not available on this device.");
    }
    auto desc = make_accel_archive_mesh_desc(mesh, option, geometry_content_hash);
    auto binding = make_accel_archive_geometry_binding(mesh);
    return ext->import_mesh_archive(stream.handle(), mesh.handle(), desc, binding, archive_bytes);
}

inline void populate_tlas_from_descriptors(
    Accel &accel,
    luisa::span<const Mesh *const> blases,
    luisa::span<const AccelArchiveInstanceDesc> instances) noexcept {
    while (accel.size() > 0u) {
        accel.pop_back();
    }
    for (const auto &inst : instances) {
        LUISA_ASSERT(inst.blas_index < blases.size() && blases[inst.blas_index] != nullptr,
                     "Invalid blas_index {} in AccelArchiveInstanceDesc.", inst.blas_index);
        accel.emplace_back(
            *blases[inst.blas_index],
            inst.transform,
            inst.visibility_mask,
            inst.opaque,
            inst.user_id);
    }
}

}// namespace luisa::compute
