#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include <volk.h>
#include <luisa/backends/ext/accel_archive_ext_interface.h>

namespace lc::vk {

class Device;

class VkAccelArchiveExt final : public luisa::compute::AccelArchiveExt {
public:
    struct StagingSlot {
        VkBuffer buffer{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
        void *mapped{nullptr};
        VkDeviceAddress device_address{0u};
        size_t capacity{0u};
        uint32_t mem_type_index{UINT32_MAX};
        bool host_coherent{false};
        std::atomic<uint32_t> gpu_refs{0u};
        bool caller_held{false};
    };

private:
    Device *_device{nullptr};
    mutable std::mutex _pool_mtx;
    luisa::vector<std::shared_ptr<StagingSlot>> _rebar_pool;
    luisa::vector<std::shared_ptr<StagingSlot>> _pinned_pool;
    luisa::vector<std::shared_ptr<StagingSlot>> _vram_pool;
    VkQueryPool _size_query_pool{VK_NULL_HANDLE};
    std::mutex _query_mtx;

    [[nodiscard]] std::shared_ptr<StagingSlot> _acquire_pool_slot(
        luisa::vector<std::shared_ptr<StagingSlot>> &pool,
        size_t min_bytes,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags required_flags,
        VkMemoryPropertyFlags preferred_flags,
        bool map_host,
        bool need_device_address,
        bool mark_caller_held,
        bool mark_gpu_ref = false) noexcept;

    [[nodiscard]] std::shared_ptr<StagingSlot> _find_caller_pinned_slot(
        const void *ptr,
        size_t bytes,
        size_t &out_offset) const noexcept;

    void _destroy_all_pools() noexcept;

public:
    explicit VkAccelArchiveExt(Device *device) noexcept;
    ~VkAccelArchiveExt() noexcept;

    [[nodiscard]] luisa::compute::AccelArchiveCapabilities capabilities() const noexcept override;

    [[nodiscard]] luisa::compute::AccelArchiveResult validate_mesh_archive(
        luisa::span<const std::byte> archive_bytes,
        const luisa::compute::AccelArchiveMeshDesc &expected_desc,
        luisa::compute::AccelArchiveMetadata *out_meta = nullptr) const noexcept override;

    [[nodiscard]] luisa::compute::AccelArchiveResult export_mesh_archive(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const luisa::compute::AccelArchiveMeshDesc &desc,
        luisa::vector<std::byte> &out_archive) noexcept override;

    [[nodiscard]] luisa::compute::AccelArchiveResult import_mesh_archive(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const luisa::compute::AccelArchiveMeshDesc &expected_desc,
        const luisa::compute::AccelArchiveGeometryBinding &geometry_binding,
        luisa::span<const std::byte> archive_bytes) noexcept override;

    [[nodiscard]] luisa::compute::AccelArchiveStagingBuffer acquire_staging_buffer(
        size_t min_bytes) noexcept override;

    void release_staging_buffer(
        luisa::compute::AccelArchiveStagingBuffer &&buffer) noexcept override;

    [[nodiscard]] luisa::compute::AccelArchiveResult import_mesh_archive_with_options(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const luisa::compute::AccelArchiveMeshDesc &expected_desc,
        const luisa::compute::AccelArchiveGeometryBinding &geometry_binding,
        luisa::span<const std::byte> archive_bytes,
        const luisa::compute::AccelArchiveImportOptions &options) noexcept override;
};

}// namespace lc::vk
