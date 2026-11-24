#pragma once

#include <cstdint>
#include <utility>

#include <luisa/ast/usage.h>
#include <luisa/backends/ext/registry.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/pixel.h>

namespace luisa::compute {

struct BufferToTextureCopyRegion {
    uint64_t buffer_offset{0ull};
    uint32_t buffer_row_length{0u};
    uint32_t buffer_image_height{0u};
    uint32_t mip_level{0u};
    uint3 texture_offset{0u, 0u, 0u};
    uint3 texture_extent{0u, 0u, 1u};
};

// Batched buffer-to-texture transfer issued as one multi-region native copy per
// target mip level. Supports concurrent/copy-queue execution and subresource tracking.
class BatchBufferToTextureCopyCommand final : public CustomDispatchCommand,
                                            public vstd::IOperatorNewBase {

private:
    uint64_t _buffer_handle{};
    uint64_t _buffer_range_offset{};
    uint64_t _buffer_range_size{};
    uint64_t _texture_handle{};
    PixelStorage _storage{};
    uint32_t _min_mip_level{0u};
    uint32_t _max_mip_level{0u};
    luisa::vector<BufferToTextureCopyRegion> _regions;

public:
    BatchBufferToTextureCopyCommand(
        uint64_t buffer_handle,
        uint64_t buffer_range_offset,
        uint64_t buffer_range_size,
        uint64_t texture_handle,
        PixelStorage storage,
        luisa::vector<BufferToTextureCopyRegion> regions) noexcept
        : _buffer_handle{buffer_handle},
          _buffer_range_offset{buffer_range_offset},
          _buffer_range_size{buffer_range_size},
          _texture_handle{texture_handle},
          _storage{storage},
          _regions{std::move(regions)} {
        if (!_regions.empty()) {
            _min_mip_level = _regions.front().mip_level;
            _max_mip_level = _regions.front().mip_level;
            for (auto const &r : _regions) {
                if (r.mip_level < _min_mip_level) { _min_mip_level = r.mip_level; }
                if (r.mip_level > _max_mip_level) { _max_mip_level = r.mip_level; }
            }
        }
    }

    ~BatchBufferToTextureCopyCommand() noexcept override = default;

    [[nodiscard]] StreamTag stream_tag() const noexcept override { return StreamTag::COPY; }
    [[nodiscard]] uint64_t custom_cmd_uuid() const noexcept override {
        return luisa::to_underlying(CustomCommandUUID::BATCH_BUFFER_TO_TEXTURE_COPY);
    }
    [[nodiscard]] uint3 max_dispatch_size() const noexcept override {
        return uint3{1u, 1u, 1u};
    }

    [[nodiscard]] uint64_t buffer() const noexcept { return _buffer_handle; }
    [[nodiscard]] uint64_t buffer_range_offset() const noexcept { return _buffer_range_offset; }
    [[nodiscard]] uint64_t buffer_range_size() const noexcept { return _buffer_range_size; }
    [[nodiscard]] uint64_t texture() const noexcept { return _texture_handle; }
    [[nodiscard]] PixelStorage storage() const noexcept { return _storage; }
    [[nodiscard]] uint32_t min_mip_level() const noexcept { return _min_mip_level; }
    [[nodiscard]] uint32_t max_mip_level() const noexcept { return _max_mip_level; }
    [[nodiscard]] luisa::span<const BufferToTextureCopyRegion> regions() const noexcept {
        return luisa::span{_regions};
    }

    void traverse_arguments(MutableArgumentVisitor &visitor) noexcept override {
        if (_regions.empty()) { return; }
        Argument::Buffer buf{_buffer_handle, _buffer_range_offset, _buffer_range_size};
        visitor.visit(buf, Usage::READ);
        for (uint32_t mip = _min_mip_level; mip <= _max_mip_level; ++mip) {
            Argument::Texture tex{_texture_handle, mip};
            visitor.visit(tex, Usage::WRITE);
        }
    }

    void traverse_arguments(ArgumentVisitor &visitor) const noexcept override {
        if (_regions.empty()) { return; }
        Argument::Buffer buf{_buffer_handle, _buffer_range_offset, _buffer_range_size};
        visitor.visit(buf, Usage::READ);
        for (uint32_t mip = _min_mip_level; mip <= _max_mip_level; ++mip) {
            Argument::Texture tex{_texture_handle, mip};
            visitor.visit(tex, Usage::WRITE);
        }
    }
};

// Signals an Event timeline fence atomically at the end of the CommandList's
// native queue submission, avoiding a separate empty vkQueueSubmit round-trip.
class StreamSignalEventCommand final : public CustomDispatchCommand,
                                      public vstd::IOperatorNewBase {

private:
    uint64_t _event_handle{};
    uint64_t _fence_value{};

public:
    StreamSignalEventCommand(uint64_t event_handle, uint64_t fence_value) noexcept
        : _event_handle{event_handle},
          _fence_value{fence_value} {}

    ~StreamSignalEventCommand() noexcept override = default;

    [[nodiscard]] StreamTag stream_tag() const noexcept override { return StreamTag::COPY; }
    [[nodiscard]] uint64_t custom_cmd_uuid() const noexcept override {
        return luisa::to_underlying(CustomCommandUUID::STREAM_SIGNAL_EVENT);
    }
    [[nodiscard]] uint3 max_dispatch_size() const noexcept override {
        return uint3{1u, 1u, 1u};
    }

    [[nodiscard]] uint64_t event_handle() const noexcept { return _event_handle; }
    [[nodiscard]] uint64_t fence_value() const noexcept { return _fence_value; }

    void traverse_arguments(MutableArgumentVisitor &) noexcept override {}
    void traverse_arguments(ArgumentVisitor &) const noexcept override {}
};

}// namespace luisa::compute
