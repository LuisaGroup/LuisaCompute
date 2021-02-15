//
// Created by Mike Smith on 2021/2/15.
//

#include <core/data_types.h>
#include <runtime/buffer.h>

namespace luisa::compute {

class FakeBuffer : public Buffer {

protected:
    void upload(Stream *stream, const void *data, size_t offset, size_t size) override {}
    void download(Stream *stream, void *data, size_t offset, size_t size) const override {}

public:
    using Buffer::Buffer;
};

}// namespace luisa::compute

int main() {
    
    using namespace luisa;
    using namespace luisa::compute;
    
    FakeBuffer buffer{1024u};
    auto view = buffer.view<float4>();
    
    const auto &const_buffer = buffer;
    auto const_view = const_buffer.view<float4>();
    
    const_view.download(nullptr)(nullptr);
    view.upload(nullptr)(nullptr);
    
    auto subview = view.subview(1, 16);
    ConstBufferView cbv = subview;
    BufferView bv = view;

    static_assert(std::is_same_v<decltype(view.const_view()), decltype(const_view)>);
}
