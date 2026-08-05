#pragma once

#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/buffer.h>

namespace luisa::compute {

/// A fixed-capacity device-side suballocation arena.
/// Kernels reserve byte ranges with dynamic_buffer_allocate without a host
/// round trip. Requests that do not fit set the overflow buffer and return
/// dynamic_buffer_invalid_offset.
class LUISA_RUNTIME_API DynamicBuffer final {
private:
    ByteBuffer _storage;
    Buffer<uint> _counter;
    Buffer<uint> _overflow;
    size_t _capacity_bytes{};

private:
    friend class Device;
    DynamicBuffer(ByteBuffer storage, Buffer<uint> counter,
                  Buffer<uint> overflow, size_t capacity_bytes) noexcept
        : _storage{std::move(storage)}, _counter{std::move(counter)},
          _overflow{std::move(overflow)}, _capacity_bytes{capacity_bytes} {}

public:
    DynamicBuffer() noexcept = default;
    ~DynamicBuffer() noexcept = default;
    DynamicBuffer(DynamicBuffer &&) noexcept = default;
    DynamicBuffer(DynamicBuffer const &) noexcept = delete;
    DynamicBuffer &operator=(DynamicBuffer &&) noexcept = default;
    DynamicBuffer &operator=(DynamicBuffer const &) noexcept = delete;
    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(_storage); }
    [[nodiscard]] auto capacity_bytes() const noexcept { return _capacity_bytes; }
    [[nodiscard]] auto const &storage() const noexcept { return _storage; }
    [[nodiscard]] auto const &counter() const noexcept { return _counter; }
    [[nodiscard]] auto const &overflow() const noexcept { return _overflow; }
    /// Upload commands that reset the device-side allocator state.
    [[nodiscard]] auto reset_counter() const noexcept {
        static constexpr uint zero = 0u;
        return _counter.copy_from(&zero);
    }
    [[nodiscard]] auto reset_overflow() const noexcept {
        static constexpr uint zero = 0u;
        return _overflow.copy_from(&zero);
    }
};

constexpr uint dynamic_buffer_invalid_offset = ~0u;

}// namespace luisa::compute
