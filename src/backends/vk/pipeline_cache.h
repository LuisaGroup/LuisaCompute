// Device-scoped Vulkan pipeline cache (plan P1.3).
//
// Before P1.3 every DSL shader created its own `VkPipelineCache`, seeded it from
// that shader's own blob and destroyed it in its destructor (plan §4 F10). A
// `VkPipelineCache` is a *device-wide* driver facility: splitting it per shader
// duplicated the driver's bookkeeping, made every blob describe exactly one
// pipeline, and meant the number of cache objects grew with the number of
// compiled kernels for no benefit.
//
// This class owns the one `VkPipelineCache` of a Vulkan `Device`:
//
//  * `create()` seeds it exactly once from a single versioned artifact. The
//    artifact uses the same `PipelineCacheArtifactHeader` framing, payload MD5
//    and `VkPipelineCacheHeaderVersionOne` identity check the per-shader blobs
//    already used (shader_binary_contract.h, Device::is_pso_same), with the
//    payload size bounded before allocation. Absent, truncated, malformed or
//    foreign-driver input fails closed to an *empty but valid* cache, so shader
//    bytecode caching (the SPIR-V `.spv` artifact) is never lost because of a
//    bad PSO blob.
//  * the pre-P1.3 per-shader blobs remain useful inputs: `merge_seed()` folds
//    one into the device cache through a short-lived source cache and
//    `vkMergePipelineCaches`. Nothing rewrites or deletes those files.
//  * the cache is written back to that one artifact a single time, in
//    `release()` at device teardown, and only when its bytes changed. No shader
//    ever writes a cache file, so the cache directory does not grow one file per
//    compiled kernel.
//  * only `Device` destroys the handle. A `Shader` borrows it through
//    `acquire()` and a `PipelineRef` never touches it, so a late shader
//    destructor cannot destroy the device cache underneath a live device.
//
// Ownership/synchronization contract:
//
//  * Vulkan requires host access to a `VkPipelineCache` to be externally
//    synchronized for `vkCreate*Pipelines`, `vkGetPipelineCacheData`,
//    `vkMergePipelineCaches` and `vkDestroyPipelineCache`. Every one of those
//    calls in this backend runs while holding `_mtx`; `acquire()` hands that
//    lock out as a `Lease` for exactly the duration of one creation call.
//  * A lease never covers HLSL/DXC/SPIR-V compilation: callers compile first and
//    lock only around the driver call. A slow compile therefore cannot serialize
//    another thread's pipeline creation, which is what a future parallel compile
//    (P5.1) plus the current serial creation model need.
//  * `VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT` is a Vulkan 1.4
//    core flag (the EXT alias only exists with VK_EXT_... enabled). This backend
//    creates its instance at `VK_API_VERSION_1_3` (device.cpp `create_instance`),
//    so the effective logical-device version is at most 1.3 and the flag is not
//    legal here: `flags = 0` is the only legal value, and the mutex above is the
//    synchronization Vulkan demands on those versions. It is also what keeps the
//    Vulkan 1.2 floor working (plan §6.3).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include <volk.h>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/vstl/md5.h>

namespace lc::vk {

class Device;

namespace detail {

// Result class of reading one PSO artifact (the device-global cache blob and the
// pre-P1.3 per-shader blobs share the same on-disk contract).
enum class PsoArtifactStatus : uint8_t {
    kAbsent = 0u,  // no such entry in the BinaryIO
    kAccepted = 1u,// fully validated payload
    kRejected = 2u // present but unusable; `reason` says why
};

struct PsoArtifact {
    PsoArtifactStatus status{PsoArtifactStatus::kAbsent};
    // Raw `VkPipelineCache` payload (header + driver data), empty unless the
    // status is `kAccepted`.
    luisa::vector<std::byte> payload;
    // Fixed diagnostic token; a literal, never an allocation.
    luisa::string_view reason{"absent"};
};

// Reads and validates a PSO artifact through `BinaryIO`:
//  * the `PipelineCacheArtifactHeader` framing (magic, version, reserved,
//    `payload_size` == total - header, `payload_size <=
//    max_pipeline_cache_byte_size`), so a corrupt header can never drive an
//    unbounded allocation;
//  * the payload size against `sizeof(VkPipelineCacheHeaderVersionOne)` both
//    before the buffer is enlarged and after the payload was read;
//  * the absence of trailing bytes (`stream->pos() == stream->length()`);
//  * the payload MD5;
//  * the `VkPipelineCacheHeaderVersionOne` identity through
//    `Device::is_pso_same`, i.e. vendor, device and pipeline-cache UUID.
// Every failure is reported as `kRejected` plus a reason; nothing aborts, so a
// caller can fail closed.
[[nodiscard]] PsoArtifact read_pso_artifact(Device *device,
                                            luisa::string_view name) noexcept;

// Versioned identity of the device-global cache artifact name. The name is
// derived from the *device*, never from a shader: two drivers/GPUs must not read
// each other's blob, and one device must have exactly one file.
[[nodiscard]] luisa::string
device_pipeline_cache_artifact_name(
    const VkPipelineCacheHeaderVersionOne &header) noexcept;

class DevicePipelineCache {
public:
    // What happened when the cache was seeded at device creation. Reported
    // through `Device::query("vk_pipeline_cache_seed")` so a test (or a bug
    // report) can tell "cold", "warm" and "rejected blob" apart.
    enum class SeedStatus : uint8_t {
        kUnavailable = 0u,// no cache handle at all (creation failed)
        kAbsent = 1u,     // created empty, no artifact in the BinaryIO
        kLoaded = 2u,     // created from a validated artifact
        kRejected = 3u    // artifact existed but was rejected; created empty
    };

    // Holds the device cache for the duration of exactly one Vulkan call that
    // needs it (a pipeline creation). Obtained from `acquire()`.
    class Lease {
    public:
        Lease() noexcept = default;
        Lease(std::unique_lock<std::mutex> lock, VkPipelineCache handle) noexcept
            : _lock{std::move(lock)}, _handle{handle} {}
        [[nodiscard]] VkPipelineCache handle() const noexcept { return _handle; }

    private:
        std::unique_lock<std::mutex> _lock{};
        VkPipelineCache _handle{VK_NULL_HANDLE};
    };

    DevicePipelineCache() noexcept = default;
    DevicePipelineCache(const DevicePipelineCache &) = delete;
    DevicePipelineCache(DevicePipelineCache &&) = delete;
    DevicePipelineCache &operator=(const DevicePipelineCache &) = delete;
    DevicePipelineCache &operator=(DevicePipelineCache &&) = delete;
    ~DevicePipelineCache() noexcept = default;

    // One-shot initialization from `Device`'s constructor, after the logical
    // device and `_pso_header` exist. Reads and validates the artifact, then
    // creates the cache (seeded when the artifact is good, empty otherwise).
    // Never aborts: a driver that refuses even an empty cache leaves the handle
    // null, which Vulkan accepts for pipeline creation.
    void create(Device *device) noexcept;

    // Teardown path for `~Device`, while the VkDevice and the BinaryIO are still
    // alive: writes the artifact (only if the cache bytes changed) and then
    // destroys the handle. Idempotent; a second call is a no-op.
    void release(Device *device) noexcept;

    // Borrows the cache handle for one externally synchronized Vulkan call.
    // The lock is held until the returned `Lease` dies.
    [[nodiscard]] Lease acquire() noexcept;

    // Folds one already-framed PSO payload (a pre-P1.3 per-shader blob) into the
    // device cache via a short-lived source cache and vkMergePipelineCaches.
    // Returns false and leaves the cache untouched on any validation or driver
    // failure; never aborts.
    [[nodiscard]] bool merge_seed(Device *device,
                                  luisa::span<const std::byte> payload) noexcept;

    // Records a legacy seed that was rejected before it could even be wrapped in
    // a source cache (bad framing, bad payload MD5, foreign device identity).
    // Counted in the same `rejected_seeds` diagnostic as a failed merge, so the
    // number always means "legacy seeds this device threw away".
    void note_rejected_seed() noexcept {
        _rejected_seeds.fetch_add(1u, std::memory_order_relaxed);
    }

    // Serializes the cache to the single artifact. Also used by `release()`.
    // Returns true only when a new artifact was written.
    [[nodiscard]] bool write_artifact(Device *device) noexcept;

    // Diagnostics. `query` may run on any thread while another thread compiles,
    // so the handle and the counters are atomic; the seed record is written once
    // during `create()` before any shader exists.
    [[nodiscard]] VkPipelineCache handle() const noexcept {
        return _handle.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool available() const noexcept {
        return handle() != VK_NULL_HANDLE;
    }
    [[nodiscard]] SeedStatus seed_status() const noexcept { return _seed_status; }
    [[nodiscard]] uint64_t seed_bytes() const noexcept { return _seed_bytes; }
    [[nodiscard]] luisa::string_view seed_reason() const noexcept { return _seed_reason; }
    [[nodiscard]] luisa::string_view artifact_name() const noexcept { return _artifact_name; }
    [[nodiscard]] uint64_t merged_seeds() const noexcept {
        return _merged_seeds.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t rejected_seeds() const noexcept {
        return _rejected_seeds.load(std::memory_order_relaxed);
    }

private:
    std::atomic<VkPipelineCache> _handle{VK_NULL_HANDLE};
    std::mutex _mtx;
    luisa::string _artifact_name;
    SeedStatus _seed_status{SeedStatus::kUnavailable};
    uint64_t _seed_bytes{0u};
    luisa::string_view _seed_reason{"unavailable"};
    std::atomic<uint64_t> _merged_seeds{0u};
    std::atomic<uint64_t> _rejected_seeds{0u};
    // Payload identity of the artifact last written by this process, so a device
    // that compiled nothing does not rewrite an identical file.
    vstd::MD5 _written_payload_md5{};
    bool _has_written{false};
};

}// namespace detail
}// namespace lc::vk