#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/pixel.h>

namespace luisa::compute {

struct HostByteRange {
    uintptr_t begin{0u};
    uintptr_t end{0u};

    [[nodiscard]] constexpr bool empty() const noexcept {
        return begin >= end;
    }

    [[nodiscard]] constexpr bool overlaps(const HostByteRange &other) const noexcept {
        return !empty() && !other.empty() && begin < other.end && other.begin < end;
    }
};

[[nodiscard]] inline HostByteRange make_host_byte_range(
    const void *ptr, size_t size) noexcept {
    if (ptr == nullptr || size == 0u) {
        return {};
    }
    const auto begin = reinterpret_cast<uintptr_t>(ptr);
    const auto max_addr = std::numeric_limits<uintptr_t>::max();
    const auto clamped = size <= max_addr - begin ? size : max_addr - begin;
    return HostByteRange{begin, begin + clamped};
}

[[nodiscard]] inline HostByteRange command_host_upload_range(
    const Command *cmd) noexcept {
    if (cmd == nullptr) {
        return {};
    }
    switch (cmd->tag()) {
        case Command::Tag::EBufferUploadCommand: {
            const auto *c = static_cast<const BufferUploadCommand *>(cmd);
            return make_host_byte_range(c->data(), c->size());
        }
        case Command::Tag::ETextureUploadCommand: {
            const auto *c = static_cast<const TextureUploadCommand *>(cmd);
            const auto checked = checked_pixel_storage_size(c->storage(), c->size());
            if (!checked) {
                return {};
            }
            return make_host_byte_range(c->data(), checked.size);
        }
        default:
            return {};
    }
}

[[nodiscard]] inline HostByteRange command_host_download_range(
    const Command *cmd) noexcept {
    if (cmd == nullptr) {
        return {};
    }
    switch (cmd->tag()) {
        case Command::Tag::EBufferDownloadCommand: {
            const auto *c = static_cast<const BufferDownloadCommand *>(cmd);
            return make_host_byte_range(c->data(), c->size());
        }
        case Command::Tag::ETextureDownloadCommand: {
            const auto *c = static_cast<const TextureDownloadCommand *>(cmd);
            const auto checked = checked_pixel_storage_size(c->storage(), c->size());
            if (!checked) {
                return {};
            }
            return make_host_byte_range(c->data(), checked.size);
        }
        default:
            return {};
    }
}

[[nodiscard]] inline size_t find_next_host_transfer_split(
    luisa::span<const luisa::unique_ptr<Command>> cmds,
    size_t start_index) noexcept {
    luisa::fixed_vector<HostByteRange, 16u> seen_downloads;
    for (size_t i = start_index; i < cmds.size(); ++i) {
        const auto *cmd = cmds[i].get();
        if (cmd == nullptr) {
            continue;
        }
        if (const auto dl = command_host_download_range(cmd); !dl.empty()) {
            seen_downloads.push_back(dl);
        } else if (!seen_downloads.empty()) {
            if (const auto ul = command_host_upload_range(cmd); !ul.empty()) {
                for (const auto &prev : seen_downloads) {
                    if (prev.overlaps(ul)) {
                        return i;
                    }
                }
            }
        }
    }
    return cmds.size();
}

template<typename Vector, typename Pred>
inline void erase_host_transfer_if(Vector &vec, Pred &&pred) noexcept {
    vec.erase(
        std::remove_if(vec.begin(), vec.end(), std::forward<Pred>(pred)),
        vec.end());
}

}// namespace luisa::compute
