#pragma once
// OCR engine -- grow-only aligned activation arenas (Phase 2).
//
// DeviceWorkspace owns ONE device byte buffer per arena and hands out
// checked, aligned, non-overlapping regions. Growth is grow-only: when a
// request needs more bytes than the current capacity, a NEW larger buffer is
// allocated and the old one is retired. Retirement NEVER invalidates
// in-flight work: the old buffer is kept alive until the caller proves the
// owning stream has drained (retire_after_sync / release_retired), because
// LuisaCompute defers resource destruction but a buffer must still outlive
// every command that references it.
//
// Reset semantics: reset() rewinds the allocation cursor to 0 WITHOUT
// freeing or touching memory. It is only legal when no in-flight command
// references any region (the caller synchronizes first). Every reset bumps
// reset_epoch(), and each allocation carries (generation, epoch): two live
// allocations at the same byte offset are distinguishable, so callers/tests
// can detect illegal reuse of a pre-reset region (same offset, older epoch).
//
// Host planning (WorkspacePlan) computes the full region layout with checked
// 64-bit offsets BEFORE any device allocation, so memory_budget can preflight
// the exact byte requirement. No per-region retained activations live here
// beyond what the caller explicitly keeps (e.g. final embeddings).
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>

#include "ocr/runtime/dispatch.h"
#include "ocr/runtime/tensor.h"
#include "ocr/status.h"

namespace luisa::ocr {

inline constexpr uint64_t kWorkspaceAlignment = 256u;

// One planned region: aligned byte offset + size + logical tensor metadata.
struct WorkspaceRegion {
    luisa::string name;
    uint64_t offset{0};
    uint64_t bytes{0};
    TensorDesc tensor;
};

// Host-side layout plan: a list of regions with checked, aligned offsets and
// the total capacity. Build it first, preflight it against the memory budget,
// then materialize it into a DeviceWorkspace.
class WorkspacePlan {
public:
    // Append a region of `bytes` bytes (aligned up to kWorkspaceAlignment).
    [[nodiscard]] Status add_bytes(luisa::string name, uint64_t bytes);
    // Append a region for a logical tensor (byte size = elements * dtype).
    [[nodiscard]] Status add_tensor(luisa::string name, TensorDesc tensor);

    [[nodiscard]] uint64_t total_bytes() const noexcept { return _total; }
    [[nodiscard]] luisa::span<const WorkspaceRegion> regions() const noexcept { return _regions; }
    [[nodiscard]] const WorkspaceRegion *find(luisa::string_view name) const noexcept;

private:
    luisa::vector<WorkspaceRegion> _regions;
    uint64_t _total{0};
};

// A live region handed out by DeviceWorkspace: byte range + owning buffer
// generation. Typed views are created by the caller with as<T>() so the
// element type is always explicit.
struct WorkspaceAllocation {
    uint64_t offset{0};
    uint64_t bytes{0};
    uint64_t generation{0};// arena buffer generation (bumped on growth)
    uint64_t epoch{0};     // reset epoch (bumped on every reset)

    // True when two allocations may reference the same live memory: same
    // buffer generation AND same reset epoch AND intersecting byte ranges.
    [[nodiscard]] bool aliases(const WorkspaceAllocation &other) const noexcept {
        if (generation != other.generation || epoch != other.epoch) { return false; }
        if (bytes == 0u || other.bytes == 0u) { return false; }
        return offset < other.offset + other.bytes && other.offset < offset + bytes;
    }

    [[nodiscard]] compute::ByteBufferView byte_view(const compute::ByteBuffer &buffer) const noexcept {
        return buffer.view(static_cast<size_t>(offset), static_cast<size_t>(bytes));
    }
    template<typename T>
    [[nodiscard]] compute::BufferView<T> view(const compute::ByteBuffer &buffer) const noexcept {
        return byte_view(buffer).as<T>();
    }
};

class DeviceWorkspace {
public:
    DeviceWorkspace() noexcept = default;
    explicit DeviceWorkspace(compute::Device &device) noexcept : _device(&device) {}

    // Grow the arena so that at least `capacity_bytes` are available from
    // offset 0. Grow-only: never shrinks. If growth is needed the previous
    // buffer is retired (kept alive) and the cursor resets to 0 -- the caller
    // MUST have drained in-flight work that used the old buffer before
    // calling this (documented contract; retire_after_sync proves it).
    [[nodiscard]] Status reserve(uint64_t capacity_bytes);

    // Allocate the next aligned region of `bytes`. Fails (out_of_memory) when
    // the arena is exhausted; it never silently grows mid-request (growth is
    // an explicit reserve() between requests).
    [[nodiscard]] Result<WorkspaceAllocation> allocate(uint64_t bytes);
    // Allocate a region for a logical tensor.
    [[nodiscard]] Result<WorkspaceAllocation> allocate_tensor(const TensorDesc &tensor);

    // Rewind the cursor to 0 and bump the reset epoch. Legal ONLY when no
    // in-flight command references any region of the current generation
    // (caller synchronizes first). Allocations after the reset carry the new
    // epoch, so a stale pre-reset allocation never `aliases()` a fresh one.
    void reset() noexcept;

    // Prove the owning stream drained: drops retired buffers and bumps the
    // epoch. Call after stream.synchronize() (or an equivalent completion).
    void retire_after_sync() noexcept;

    [[nodiscard]] uint64_t capacity_bytes() const noexcept { return _capacity; }
    [[nodiscard]] uint64_t used_bytes() const noexcept { return _cursor; }
    [[nodiscard]] uint64_t generation() const noexcept { return _generation; }
    [[nodiscard]] uint64_t reset_epoch() const noexcept { return _epoch; }
    [[nodiscard]] uint64_t retired_count() const noexcept { return _retired.size(); }
    // Actual resident bytes = live capacity + retired buffers still alive.
    [[nodiscard]] uint64_t resident_bytes() const noexcept;
    [[nodiscard]] bool has_buffer() const noexcept { return static_cast<bool>(_buffer); }
    [[nodiscard]] const compute::ByteBuffer &buffer() const noexcept { return _buffer; }

private:
    compute::Device *_device{nullptr};
    compute::ByteBuffer _buffer;
    uint64_t _capacity{0};
    uint64_t _cursor{0};
    uint64_t _generation{0};
    uint64_t _epoch{0};
    luisa::vector<compute::ByteBuffer> _retired;
};

}// namespace luisa::ocr