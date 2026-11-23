// Implementation of the device-scoped Vulkan pipeline cache (plan P1.3).
// See pipeline_cache.h for the ownership, validation and synchronization
// contract; this file only performs it.
#include "pipeline_cache.h"

#include <array>
#include <cstring>

#include <luisa/core/binary_io.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>

#include "device.h"
#include "shader_binary_contract.h"
#include "submit_profile.h"

namespace lc::vk::detail {

namespace {

// Domain separation for the artifact name: a shader blob name must never collide
// with the device artifact name, and a future contract bump must produce a
// different name instead of silently reading an older, differently-keyed file.
constexpr std::array<uint8_t, 8u> kDeviceCacheNameDomain{
    'L', 'C', 'V', 'K', 'P', 'S', 'O', 'G'};
constexpr uint32_t kDeviceCacheNameVersion = 1u;

[[nodiscard]] constexpr bool valid_cache_header_size(uint32_t header_size) noexcept {
    // Vulkan fixes the header size of VkPipelineCacheHeaderVersionOne. A blob
    // that claims anything else cannot be a version-one cache.
    return header_size == sizeof(VkPipelineCacheHeaderVersionOne);
}

}// namespace

luisa::string device_pipeline_cache_artifact_name(
    const VkPipelineCacheHeaderVersionOne &header) noexcept {
    // Hash semantic fields explicitly: never a raw struct, so no padding or
    // uninitialised byte can split two names for one device (the same rule the
    // shader semantic headers follow).
    std::array<uint8_t, kDeviceCacheNameDomain.size() + 5u * sizeof(uint32_t) +
                            VK_UUID_SIZE>
        bytes{};
    size_t offset = 0u;
    for (auto byte : kDeviceCacheNameDomain) { bytes[offset++] = byte; }
    auto append_u32 = [&bytes, &offset](uint32_t value) noexcept {
        for (auto shift = 0u; shift < 32u; shift += 8u) {
            bytes[offset++] = static_cast<uint8_t>(value >> shift);
        }
    };
    append_u32(kDeviceCacheNameVersion);
    append_u32(header.headerSize);
    append_u32(header.headerVersion);
    append_u32(header.vendorID);
    append_u32(header.deviceID);
    for (auto byte : header.pipelineCacheUUID) { bytes[offset++] = byte; }
    LUISA_ASSERT(offset == bytes.size(),
                 "Vulkan device pipeline-cache name encoding wrote {} of {} bytes.",
                 offset, bytes.size());
    auto md5 = vstd::MD5{vstd::span<const uint8_t>{bytes.data(), bytes.size()}};
    return luisa::format("luisa-vk-device-pso-{}.bin", md5.to_string(false));
}

PsoArtifact read_pso_artifact(Device *device, luisa::string_view name) noexcept {
    PsoArtifact result{};
    auto bin_io = device->binary_io();
    if (bin_io == nullptr || name.empty()) { return result; }
    auto stream = bin_io->read_shader_cache(name);
    if (!stream) { return result; }// kAbsent / "absent"
    result.status = PsoArtifactStatus::kRejected;
    result.reason = "truncated-header";
    auto stream_size = stream->length();
    if (stream_size < sizeof(PipelineCacheArtifactHeader)) { return result; }
    PipelineCacheArtifactHeader header{};
    stream->read({reinterpret_cast<std::byte *>(&header), sizeof(header)});
    result.reason = "bad-framing";
    if (!valid_pipeline_cache_artifact_framing(header, stream_size)) {
        return result;
    }
    // Bound the allocation *before* enlarging anything: the framing check above
    // already proves `payload_size <= max_pipeline_cache_byte_size` and that the
    // stream really carries that many bytes.
    result.reason = "short-payload";
    if (header.payload_size < sizeof(VkPipelineCacheHeaderVersionOne)) {
        return result;
    }
    luisa::enlarge_by(result.payload, static_cast<size_t>(header.payload_size));
    stream->read(result.payload);
    result.reason = "trailing-bytes";
    if (stream->pos() != stream_size) { return result; }
    result.reason = "bad-payload-md5";
    if (!valid_pipeline_cache_artifact_payload(header, result.payload)) {
        return result;
    }
    VkPipelineCacheHeaderVersionOne cache_header{};
    std::memcpy(&cache_header, result.payload.data(), sizeof(cache_header));
    result.reason = "bad-cache-header";
    if (!valid_cache_header_size(cache_header.headerSize) ||
        cache_header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE) {
        return result;
    }
    result.reason = "foreign-device";
    if (!device->is_pso_same(cache_header)) { return result; }
    result.status = PsoArtifactStatus::kAccepted;
    result.reason = "accepted";
    return result;
}

void DevicePipelineCache::create(Device *device) noexcept {
    auto vk_device = device->logic_device();
    _artifact_name = device_pipeline_cache_artifact_name(device->pso_header());
    if (vk_device == VK_NULL_HANDLE) {
        _seed_status = SeedStatus::kUnavailable;
        _seed_reason = "no-device";
        LUISA_WARNING(
            "Vulkan device pipeline cache: no logical device, pipeline caching "
            "is disabled for this Device.");
        return;
    }
    auto seed = read_pso_artifact(device, _artifact_name);
    _seed_bytes = seed.payload.size();
    VkPipelineCacheCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        // Vulkan 1.4's VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT is
        // not legal on this backend's 1.3 instance; see pipeline_cache.h.
        .flags = 0u};
    if (seed.status == PsoArtifactStatus::kAccepted) {
        create_info.initialDataSize = seed.payload.size();
        create_info.pInitialData = seed.payload.data();
    }
    auto create = [&]() noexcept {
        VkPipelineCache handle{VK_NULL_HANDLE};
        auto result = vkCreatePipelineCache(
            vk_device, &create_info, Device::alloc_callbacks(), &handle);
        if (result == VK_SUCCESS) {
            // Only a successful creation may publish a handle: a failed call
            // leaves the driver's out-parameter unspecified, and nothing may
            // store that value where a later pipeline creation would use it.
            _handle.store(handle, std::memory_order_relaxed);
            submit_profile::record_pipeline_cache_create();
        }
        return result;
    };
    auto result = create();
    auto seed_was_used = result == VK_SUCCESS &&
                         seed.status == PsoArtifactStatus::kAccepted;
    if (result != VK_SUCCESS) {
        // The driver refused the seeded bytes (or the create itself failed).
        // Fail closed to an empty but valid cache: losing PSO reuse is
        // recoverable, losing SPIR-V caching or aborting on device creation is
        // not.
        LUISA_WARNING(
            "Vulkan device pipeline cache: vkCreatePipelineCache failed with "
            "VkResult {} for the seeded artifact '{}' ({}); retrying empty.",
            static_cast<int>(result), _artifact_name, seed.reason);
        _handle.store(VK_NULL_HANDLE, std::memory_order_relaxed);
        create_info.initialDataSize = 0u;
        create_info.pInitialData = nullptr;
        result = create();
    }
    if (result != VK_SUCCESS) {
        _handle.store(VK_NULL_HANDLE, std::memory_order_relaxed);
        _seed_status = SeedStatus::kUnavailable;
        _seed_reason = "create-failed";
        LUISA_WARNING(
            "Vulkan device pipeline cache: an empty pipeline cache could not be "
            "created (VkResult {}); pipelines are created without a cache.",
            static_cast<int>(result));
        _seed_bytes = 0u;
        return;
    }
    switch (seed.status) {
        case PsoArtifactStatus::kAccepted:
            if (seed_was_used) {
                _seed_status = SeedStatus::kLoaded;
                _seed_reason = "loaded";
                // The validated on-disk payload already has this identity. Avoid
                // rewriting it at teardown when the driver returns the same bytes.
                _written_payload_md5 = pipeline_cache_payload_md5(seed.payload);
                _has_written = true;
                LUISA_INFO(
                    "Vulkan device pipeline cache: seeded {} bytes from '{}'.",
                    _seed_bytes, _artifact_name);
            } else {
                // The seeded vkCreatePipelineCache failed and the empty retry
                // succeeded. Reporting this as a warm load would be false.
                _seed_status = SeedStatus::kRejected;
                _seed_reason = "create-failed";
                _seed_bytes = 0u;
                LUISA_WARNING(
                    "Vulkan device pipeline cache: driver rejected the seeded "
                    "artifact '{}'; using the empty retry cache.",
                    _artifact_name);
            }
            break;
        case PsoArtifactStatus::kAbsent:
            _seed_status = SeedStatus::kAbsent;
            _seed_reason = "absent";
            _seed_bytes = 0u;
            break;
        case PsoArtifactStatus::kRejected:
            _seed_status = SeedStatus::kRejected;
            _seed_reason = seed.reason;
            _seed_bytes = 0u;
            LUISA_WARNING(
                "Vulkan device pipeline cache: ignoring unusable artifact '{}' "
                "({}); starting from an empty cache.",
                _artifact_name, seed.reason);
            break;
    }
}

void DevicePipelineCache::release(Device *device) noexcept {
    if (!available()) { return; }
    if (device->logic_device() != VK_NULL_HANDLE) {
        static_cast<void>(write_artifact(device));
    }
    {
        // Destroy under the same lock that guards every other cache access, so a
        // concurrent pipeline creation cannot race the teardown.
        std::lock_guard lock{_mtx};
        auto handle = _handle.exchange(VK_NULL_HANDLE, std::memory_order_relaxed);
        if (handle != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(device->logic_device(), handle,
                                   Device::alloc_callbacks());
        }
    }
}

DevicePipelineCache::Lease DevicePipelineCache::acquire() noexcept {
    std::unique_lock lock{_mtx};
    return Lease{std::move(lock), _handle.load(std::memory_order_relaxed)};
}

bool DevicePipelineCache::merge_seed(
    Device *device, luisa::span<const std::byte> payload) noexcept {
    if (payload.size() <= sizeof(VkPipelineCacheHeaderVersionOne)) { return false; }
    VkPipelineCacheHeaderVersionOne cache_header{};
    std::memcpy(&cache_header, payload.data(), sizeof(cache_header));
    if (!valid_cache_header_size(cache_header.headerSize) ||
        cache_header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
        !device->is_pso_same(cache_header)) {
        _rejected_seeds.fetch_add(1u, std::memory_order_relaxed);
        return false;
    }
    auto vk_device = device->logic_device();
    if (vk_device == VK_NULL_HANDLE) { return false; }
    VkPipelineCache source{VK_NULL_HANDLE};
    {
        std::lock_guard lock{_mtx};
        auto handle = _handle.load(std::memory_order_relaxed);
        if (handle == VK_NULL_HANDLE) { return false; }
        VkPipelineCacheCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .flags = 0u,
            .initialDataSize = payload.size(),
            .pInitialData = payload.data()};
        auto create_result = vkCreatePipelineCache(
            vk_device, &create_info, Device::alloc_callbacks(), &source);
        if (create_result != VK_SUCCESS || source == VK_NULL_HANDLE) {
            _rejected_seeds.fetch_add(1u, std::memory_order_relaxed);
            LUISA_WARNING(
                "Vulkan device pipeline cache: a per-shader PSO seed of {} bytes "
                "could not be wrapped in a source cache (VkResult {}); the seed "
                "is skipped.",
                payload.size(), static_cast<int>(create_result));
            return false;
        }
        // Both the source-cache creation and the merge are real creations/uses,
        // counted separately so a report can tell the device cache apart from
        // the short-lived merge source.
        submit_profile::record_pipeline_cache_create();
        submit_profile::record_pipeline_cache_merge_source_create();
        auto merge_result = vkMergePipelineCaches(vk_device, handle, 1u, &source);
        vkDestroyPipelineCache(vk_device, source, Device::alloc_callbacks());
        if (merge_result != VK_SUCCESS) {
            _rejected_seeds.fetch_add(1u, std::memory_order_relaxed);
            LUISA_WARNING(
                "Vulkan device pipeline cache: vkMergePipelineCaches rejected a "
                "per-shader PSO seed (VkResult {}); the device cache is "
                "unchanged.",
                static_cast<int>(merge_result));
            return false;
        }
        submit_profile::record_pipeline_cache_merge();
        _merged_seeds.fetch_add(1u, std::memory_order_relaxed);
    }
    return true;
}

bool DevicePipelineCache::write_artifact(Device *device) noexcept {
    auto vk_device = device->logic_device();
    auto bin_io = device->binary_io();
    if (vk_device == VK_NULL_HANDLE || bin_io == nullptr || _artifact_name.empty()) {
        return false;
    }
    luisa::vector<std::byte> payload;
    {
        std::lock_guard lock{_mtx};
        auto handle = _handle.load(std::memory_order_relaxed);
        if (handle == VK_NULL_HANDLE) { return false; }
        size_t size = 0u;
        auto query_result = vkGetPipelineCacheData(vk_device, handle, &size, nullptr);
        if (query_result != VK_SUCCESS) {
            LUISA_WARNING(
                "Vulkan device pipeline cache: vkGetPipelineCacheData failed with "
                "VkResult {}; no artifact written.",
                static_cast<int>(query_result));
            return false;
        }
        // A cache that holds nothing reports exactly the header; writing it would
        // only churn the cache directory.
        if (size <= sizeof(VkPipelineCacheHeaderVersionOne)) { return false; }
        if (size > max_pipeline_cache_byte_size) {
            LUISA_WARNING(
                "Vulkan device pipeline cache: refusing to persist an implausibly "
                "large cache ({} bytes).",
                size);
            return false;
        }
        payload.resize(size);
        auto read_result = vkGetPipelineCacheData(vk_device, handle, &size, payload.data());
        if (read_result != VK_SUCCESS) {
            LUISA_WARNING(
                "Vulkan device pipeline cache: vkGetPipelineCacheData failed with "
                "VkResult {} on the second (data) call; no artifact written.",
                static_cast<int>(read_result));
            return false;
        }
        payload.resize(size);
        if (payload.size() <= sizeof(VkPipelineCacheHeaderVersionOne)) { return false; }
        auto payload_md5 = pipeline_cache_payload_md5(payload);
        if (_has_written && payload_md5 == _written_payload_md5) {
            // Identical bytes are already on disk: rewriting them would only
            // touch the cache directory (and, with LMDB, dirty a page) for
            // nothing.
            return false;
        }
        auto artifact = encode_pipeline_cache_artifact(payload);
        static_cast<void>(bin_io->write_shader_cache(_artifact_name, artifact));
        _written_payload_md5 = payload_md5;
        _has_written = true;
    }
    LUISA_INFO(
        "Vulkan device pipeline cache: wrote {} bytes to '{}'.",
        payload.size(), _artifact_name);
    return true;
}

}// namespace lc::vk::detail