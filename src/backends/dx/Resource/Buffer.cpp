
#include <Resource/Buffer.h>
namespace toolhub::directx {
BufferView::BufferView(Buffer const *buffer)
    : buffer(buffer),
      byteSize(buffer ? buffer->GetByteSize() : 0),
      offset(0) {
}
BufferView::BufferView(
    Buffer const *buffer,
    uint64 offset,
    uint64 byteSize)
    : buffer(buffer),
      offset(offset),
      byteSize(byteSize) {}
BufferView::BufferView(
    Buffer const *buffer,
    uint64 offset)
    : buffer(buffer),
      offset(offset) {
    byteSize = buffer->GetByteSize() - offset;
}
Buffer::Buffer(
    Device *device)
    : Resource(device) {
}
Buffer::~Buffer() {
}
}// namespace toolhub::directx