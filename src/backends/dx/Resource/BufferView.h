#pragma once
#include <vstl/common.h>
namespace toolhub::directx {
class Buffer;
class TextureBase;
struct BufferView {
    Buffer const *buffer = nullptr;
    uint64 offset = 0;
    uint64 byteSize = 0;
    BufferView() {}
    BufferView(Buffer const *buffer);
    BufferView(
        Buffer const *buffer,
        uint64 offset,
        uint64 byteSize)
        : buffer(buffer),
          offset(offset),
          byteSize(byteSize) {}
    BufferView(
        Buffer const *buffer,
        uint64 offset);
    VSTD_TRIVIAL_COMPARABLE(BufferView)
};
struct TexView {
    TextureBase const *tex = nullptr;
    uint64 mipStart = 0;
    uint64 mipCount = 0;
    TexView() {}
    TexView(TextureBase const *tex);
    TexView(
        TextureBase const *tex,
        uint64 mipStart,
        uint64 mipCount);
    TexView(
        TextureBase const *tex,
        uint64 mipStart);
    VSTD_TRIVIAL_COMPARABLE(TexView)
};
}// namespace toolhub::directx