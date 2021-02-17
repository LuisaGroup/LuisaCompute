//
// Created by Mike Smith on 2021/2/15.
//

#include <core/data_types.h>
#include <runtime/buffer.h>

namespace luisa::compute {

struct Index {};

namespace detail {

template<>
struct BufferAccess<Index> {
    
    template<typename T>
    [[nodiscard]] auto operator()(BufferView<T>, Index) const noexcept {
        return 0ul;
    }
    
    template<typename T>
    [[nodiscard]] auto operator()(ConstBufferView<T>, Index) const noexcept {
        return 1.0f;
    }
};

}

class FakeDevice : public Device {
    
    void _dispose_buffer(uint64_t handle) noexcept override {}
    
    uint64_t _create_buffer(size_t byte_size) noexcept override {
        return 0;
    }
    
    uint64_t _create_buffer_with_data(size_t size_bytes, const void *data) noexcept override {
        return 0;
    }
};

}// namespace luisa::compute

int main() {
    
    using namespace luisa;
    using namespace luisa::compute;
    
    FakeDevice device;
    
    Buffer<float4> buffer{&device, 1024u};
    auto a = buffer[Index{}];
    BufferView av = buffer;
    ConstBufferView acv = buffer;
    
    auto view = buffer.view();
    auto x = view[Index{}];
    
    const auto &const_buffer = buffer;
    auto const_view = const_buffer.view();
    auto y = const_view[Index{}];
    
    auto subview = view.subview(1, 16);
    ConstBufferView cbv = subview;
    BufferView bv = view;
    
    auto v = bv.as<float2>();
    
    std::vector<float2> vector;
    std::span s{vector};
    Buffer another_buffer{&device, vector};
    static_assert(std::is_same_v<decltype(another_buffer), Buffer<float2>>);
    static_assert(std::is_same_v<decltype(view.const_view()), decltype(const_view)>);
}
