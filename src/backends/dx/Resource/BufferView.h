#pragma once
#include <vstl/Common.h>
namespace toolhub::directx {
class Buffer;
struct BufferView {
    Buffer const *buffer = nullptr;
    uint64 offset = 0;
    uint64 byteSize = 0;
    BufferView() {}
    BufferView(Buffer const *buffer);
    BufferView(
        Buffer const *buffer,
        uint64 offset,
        uint64 byteSize);
    BufferView(
        Buffer const *buffer,
        uint64 offset);
    size_t get_bin_value() const {
        return reinterpret_cast<size_t>(buffer) + offset + byteSize;
    }
    bool operator==(BufferView const &a) const {
        return a.get_bin_value() == get_bin_value();
    }
    bool operator!=(BufferView const &a) const {
        return !operator==(a);
    }
    bool operator>(BufferView const &a) const {
        return a.get_bin_value() > get_bin_value();
    }
    bool operator<(BufferView const &a) const {
        return a.get_bin_value() < get_bin_value();
    }
};
}// namespace toolhub::directx