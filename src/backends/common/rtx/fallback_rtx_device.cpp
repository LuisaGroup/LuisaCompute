// The backend-facing implementation of the fallback RTX module.
//
// `FallbackRtxDevice` is a pimpl; this file holds the pimpl and everything the
// library publishes: resource creation, the two builds (each returning the
// command list a backend splices into its own stream) and the host-synchronising
// validators.
//
// It is backend-independent: it only ever talks to the `DeviceInterface` it was
// given, through an aliased `Device` - `Device::Handle` is
// `luisa::shared_ptr<DeviceInterface>` and the constructor is public, so the
// ordinary Device / Buffer / Shader API can be used on a backend's own
// interface.  The alias must never outlive the interface, which is exactly what
// the header promises the caller.

#include "fallback_rtx.h"

#include "fallback_rtx_blas.h"
#include "fallback_rtx_storage.h"
#include "fallback_rtx_tlas.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cstdint>
#include <mutex>
#include <utility>

namespace lc::fallback_rtx {

namespace {

// The fallback allocates its own handles, because a backend hands the handle
// `create_blas()` returns straight to its user as the handle of its `Mesh`, and
// `owns()` later tells the backend which side that handle belongs to.  A tag in
// the top two bits keeps the fallback's handles apart from the backend's own
// (pointer-based or counted) handle space, and the live tables decide the rest.
constexpr uint64_t handle_tag_mask = 0xC000000000000000ull;
constexpr uint64_t handle_blas_tag = 0x4000000000000000ull;
constexpr uint64_t handle_accel_tag = 0x8000000000000000ull;

[[nodiscard]] uint64_t make_handle(uint64_t tag, uint64_t index) noexcept {
    // 2^62 handles of each kind; a device cannot exhaust that.
    return tag | (index & 0x3FFFFFFFFFFFFFFFull);
}

}// namespace

struct FallbackRtxDevice::Impl {

    Impl(DeviceInterface *device_interface) noexcept
        : interface{device_interface},
          // The alias: a Device over the backend's interface, with a deleter that
          // does nothing.  The interface outlives this object (see the header).
          device{Device::Handle{device_interface, [](DeviceInterface *) noexcept {}}},
          storage{device},
          blas_builder{storage},
          tlas_builder{storage} {}

    DeviceInterface *interface{nullptr};
    Device device;
    FallbackRtxStorage storage;
    FallbackBlasBuilder blas_builder;
    FallbackTlasBuilder tlas_builder;
    // The live handle spaces.  Both are only ever appended to on the host side
    // (the device memory they describe is append-only as well).
    luisa::unordered_map<uint64_t, FallbackBlas> blases;
    luisa::unordered_map<uint64_t, FallbackTlas> tlases;
    // Guards the two tables and the region bookkeeping, which the backends touch
    // from their resource-creation paths (the header's thread-safety note).
    mutable std::mutex mutex;
    uint64_t next_blas_index{0u};
    uint64_t next_accel_index{0u};
    size_t blas_builds{0u};
    size_t accel_builds{0u};
};

FallbackRtxDevice::FallbackRtxDevice(DeviceInterface *device) noexcept
    : _impl{luisa::make_unique<Impl>(device)} {
    LUISA_ASSERT(device != nullptr,
                 "A fallback RTX device needs the DeviceInterface it belongs to.");
}

FallbackRtxDevice::~FallbackRtxDevice() noexcept = default;

DeviceInterface *FallbackRtxDevice::device() const noexcept { return _impl->interface; }

// ---------------------------------------------------------------------------
// Resource creation
// ---------------------------------------------------------------------------

uint64_t FallbackRtxDevice::create_blas(const AccelOption &option) noexcept {
    // Fail closed on what the fallback cannot build; never silently ignore it.
    if (option.motion.is_enabled()) {
        LUISA_ERROR("The fallback RTX has no primitive motion blur, and silently ignoring the "
                    "motion option of a mesh would trace a different scene than the hardware "
                    "path.");
    }
    if (option.allow_compaction) {
        LUISA_WARNING("The fallback RTX does not compact its trees; the `allow_compaction` "
                      "option of this mesh is ignored (it is a hint, not a semantic change).");
    }
    auto &impl = *_impl;
    std::lock_guard lock{impl.mutex};
    auto handle = make_handle(handle_blas_tag, impl.next_blas_index++);
    FallbackBlas blas;
    blas.option = option;
    impl.blases.emplace(handle, std::move(blas));
    return handle;
}

void FallbackRtxDevice::destroy_blas(uint64_t blas) noexcept {
    auto &impl = *_impl;
    std::lock_guard lock{impl.mutex};
    impl.blases.erase(blas);
}

uint64_t FallbackRtxDevice::create_accel(const AccelOption &option) noexcept {
    if (option.motion.is_enabled()) {
        LUISA_ERROR("The fallback RTX has no instance motion blur, and silently ignoring the "
                    "motion option of an acceleration structure would trace a different scene "
                    "than the hardware path.");
    }
    if (option.allow_compaction) {
        LUISA_WARNING("The fallback RTX does not compact its trees; the `allow_compaction` "
                      "option of this acceleration structure is ignored.");
    }
    auto &impl = *_impl;
    std::lock_guard lock{impl.mutex};
    auto handle = make_handle(handle_accel_tag, impl.next_accel_index++);
    FallbackTlas tlas;
    tlas.option = option;
    impl.tlases.emplace(handle, std::move(tlas));
    return handle;
}

void FallbackRtxDevice::destroy_accel(uint64_t accel) noexcept {
    auto &impl = *_impl;
    std::lock_guard lock{impl.mutex};
    impl.tlases.erase(accel);
}

bool FallbackRtxDevice::owns_blas(uint64_t handle) const noexcept {
    if ((handle & handle_tag_mask) != handle_blas_tag) { return false; }
    auto &impl = *_impl;
    std::lock_guard lock{impl.mutex};
    return impl.blases.find(handle) != impl.blases.end();
}

bool FallbackRtxDevice::owns_accel(uint64_t handle) const noexcept {
    if ((handle & handle_tag_mask) != handle_accel_tag) { return false; }
    auto &impl = *_impl;
    std::lock_guard lock{impl.mutex};
    return impl.tlases.find(handle) != impl.tlases.end();
}

bool FallbackRtxDevice::owns(uint64_t handle) const noexcept {
    return owns_blas(handle) || owns_accel(handle);
}

FallbackAccelBinding FallbackRtxDevice::binding(uint64_t accel) noexcept {
    auto &impl = *_impl;
    std::lock_guard lock{impl.mutex};
    auto it = impl.tlases.find(accel);
    // Before the first build there is no heap and no region to point a descriptor
    // at, which is exactly what `valid()` reports.
    if (it == impl.tlases.end() || !it->second.built) { return {}; }
    FallbackAccelBinding binding;
    binding.accel_heap = it->second.accel_heap.handle();
    binding.accel_slot = heap_tlas_slot;
    binding.accel_buffer = impl.storage.accel().handle();
    binding.accel_offset_bytes = static_cast<size_t>(it->second.region.base) * 16u;
    binding.instance_buffer = impl.storage.instances().handle();
    binding.instance_offset_bytes =
        static_cast<size_t>(it->second.region.instance_offset) * 16u;
    return binding;
}

// ---------------------------------------------------------------------------
// Builds
// ---------------------------------------------------------------------------

CommandList FallbackRtxDevice::build_blas(uint64_t blas, const MeshGeometry &geometry) noexcept {
    auto &impl = *_impl;
    std::lock_guard lock{impl.mutex};
    auto it = impl.blases.find(blas);
    if (it == impl.blases.end()) {
        LUISA_ERROR("The fallback RTX was asked to build the BLAS of the handle {}, which it "
                    "does not own.",
                    blas);
    }
    CommandList commands;
    impl.blas_builder.build(commands, it->second, geometry);
    impl.blas_builds++;
    return commands;
}

CommandList FallbackRtxDevice::build_accel(
    uint64_t accel, uint32_t instance_count,
    luisa::span<const AccelBuildCommand::Modification> modifications,
    bool update_instance_buffer_only) noexcept {
    auto &impl = *_impl;
    std::lock_guard lock{impl.mutex};
    auto it = impl.tlases.find(accel);
    if (it == impl.tlases.end()) {
        LUISA_ERROR("The fallback RTX was asked to build the TLAS of the handle {}, which it "
                    "does not own.",
                    accel);
    }
    // The mesh handle of a fallback device *is* the handle of its fallback BLAS
    // (`create_blas` hands out what `create_mesh` returns), so a modification's
    // primitive is a BLAS handle and resolves here - with a clear error when it
    // is not one, instead of a table row that points nowhere.  The resolution
    // carries everything the TLAS build needs about the BLAS: its blas-directory
    // entry, and the region *view* it registers in the TLAS' bindless heap.
    luisa::vector<FallbackTlasBlas> resolved(modifications.size());
    for (auto i = 0u; i < modifications.size(); i++) {
        auto &&m = modifications[i];
        if ((m.flags & AccelBuildCommand::Modification::flag_primitive) == 0u) { continue; }
        auto blas = impl.blases.find(m.primitive);
        if (blas == impl.blases.end()) {
            LUISA_ERROR("Instance {} of the fallback RTX TLAS refers to the mesh handle {}, "
                        "which is not a mesh of this device.",
                        m.index, m.primitive);
        }
        if (!blas->second.built) {
            LUISA_ERROR("Instance {} of the fallback RTX TLAS refers to mesh {}, which has not "
                        "been built yet.",
                        m.index, m.primitive);
        }
        resolved[i].directory_entry = blas->second.directory_entry;
        resolved[i].region_base = blas->second.region.base;
        resolved[i].region_u4 = blas->second.region.region_u4();
    }
    CommandList commands;
    impl.tlas_builder.build(commands, it->second, instance_count, modifications,
                            luisa::span{resolved},
                            update_instance_buffer_only);
    impl.accel_builds++;
    return commands;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

FallbackRtxDevice::Stats FallbackRtxDevice::stats() const noexcept {
    auto &impl = *_impl;
    std::lock_guard lock{impl.mutex};
    Stats stats;
    stats.blas_count = impl.blases.size();
    stats.accel_count = impl.tlases.size();
    stats.accel_buffer_bytes = impl.storage.accel_buffer_bytes();
    stats.instance_buffer_bytes = impl.storage.instance_buffer_bytes();
    stats.blas_builds = impl.blas_builds;
    stats.accel_builds = impl.accel_builds;
    return stats;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

namespace {

// Download the whole live prefix of the shared storage and hand it to the
// structural check.  This synchronises: it is a debug helper, documented as
// such in fallback_rtx.h.  The stream is owned by the call and destroyed with
// it, so no command outlives the downloaded copies.
[[nodiscard]] size_t validate_region(FallbackRtxStorage &storage, Device &device,
                                     uint header_base, uint instance_base_u4,
                                     bool expect_tlas) noexcept {
    auto accel_u4 = storage.accel_used_u4();
    auto instance_u4 = storage.instance_used_u4();
    luisa::vector<uint4> accel(accel_u4);
    luisa::vector<uint4> instances(instance_u4);
    auto stream = device.create_stream();
    if (accel_u4 > 0u) {
        stream << storage.accel().view(0u, accel_u4).copy_to(luisa::span{accel});
    }
    if (instance_u4 > 0u) {
        stream << storage.instances().view(0u, instance_u4).copy_to(luisa::span{instances});
    }
    stream << synchronize();
    FallbackRtxStorage::HostView view;
    view.accel = luisa::span<const uint4>{accel.data(), accel.size()};
    view.instances = luisa::span<const uint4>{instances.data(), instances.size()};
    view.instance_base_u4 = instance_base_u4;
    view.blas_directory_entries = storage.blas_directory_entries();
    return FallbackRtxStorage::validate_tree(view, header_base, expect_tlas);
}

}// namespace

size_t FallbackRtxDevice::validate_blas(uint64_t blas) noexcept {
    auto &impl = *_impl;
    uint header_base = 0u;
    {
        std::lock_guard lock{impl.mutex};
        auto it = impl.blases.find(blas);
        if (it == impl.blases.end()) {
            LUISA_ERROR("validate_blas() was given the handle {}, which is not a BLAS of this "
                        "device.",
                        blas);
        }
        if (!it->second.built) {
            LUISA_ERROR("validate_blas() was given the handle {} of a BLAS that was never "
                        "built.",
                        blas);
        }
        header_base = it->second.region.base;
    }
    return validate_region(impl.storage, impl.device, header_base, 0u, false);
}

size_t FallbackRtxDevice::validate_accel(uint64_t accel) noexcept {
    auto &impl = *_impl;
    uint header_base = 0u;
    uint instance_base_u4 = 0u;
    {
        std::lock_guard lock{impl.mutex};
        auto it = impl.tlases.find(accel);
        if (it == impl.tlases.end()) {
            LUISA_ERROR("validate_accel() was given the handle {}, which is not an acceleration "
                        "structure of this device.",
                        accel);
        }
        if (!it->second.built) {
            LUISA_ERROR("validate_accel() was given the handle {} of an acceleration structure "
                        "that was never built.",
                        accel);
        }
        header_base = it->second.region.base;
        instance_base_u4 = it->second.region.instance_offset;
    }
    return validate_region(impl.storage, impl.device, header_base, instance_base_u4, true);
}

}// namespace lc::fallback_rtx
