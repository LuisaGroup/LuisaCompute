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

    std::mutex _csum_init_mtx;
    std::mutex _csum_mtx;
    bool _csum_attempted{false};
    bool _csum_ready{false};
    VkDescriptorSetLayout _csum_dsl{VK_NULL_HANDLE};
    VkPipelineLayout _csum_pipe_layout{VK_NULL_HANDLE};
    VkShaderModule _csum_shader{VK_NULL_HANDLE};
    VkPipeline _csum_pipeline{VK_NULL_HANDLE};
    VkDescriptorPool _csum_desc_pool{VK_NULL_HANDLE};
    VkDescriptorSet _csum_desc_set{VK_NULL_HANDLE};
    VkBuffer _csum_dev_buf{VK_NULL_HANDLE};
    VkDeviceMemory _csum_dev_mem{VK_NULL_HANDLE};
    VkBuffer _csum_rb_buf{VK_NULL_HANDLE};
    VkDeviceMemory _csum_rb_mem{VK_NULL_HANDLE};
    void *_csum_rb_mapped{nullptr};
    bool _csum_rb_host_coherent{true};

    [[nodiscard]] bool _ensure_checksum_pipeline() noexcept;
    void _record_checksum_dispatch(
        VkCommandBuffer cmd,
        VkBuffer src_buf,
        uint32_t base_byte_offset,
        uint64_t byte_size,
        uint32_t start_word) noexcept;
    void _read_checksum_lanes(uint32_t (&out_lanes)[4]) noexcept;

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
