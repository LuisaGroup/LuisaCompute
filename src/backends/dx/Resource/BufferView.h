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
    VSTD_TRIVIAL_COMPARABLE(BufferView)
};
}// namespace toolhub::directx