#include <runtime/buffer.h>
#include <runtime/device.h>

namespace luisa::compute {

class BufferArena {

private:
    std::mutex _mutex;
    Device &_device;
    luisa::vector<luisa::unique_ptr<Resource>> _buffers;
    luisa::optional<BufferView<float4>> _current_buffer;
    size_t _capacity;

public:
    explicit BufferArena(Device &device, size_t capacity = 4_mb) noexcept
        : _device{device}, _capacity{std::max(next_pow2(capacity), 64_kb) / sizeof(float4)} {}

    template<typename T>
    [[nodiscard]] BufferView<T> allocate(size_t n) noexcept {
        static_assert(alignof(T) <= 16u);
        std::scoped_lock lock{_mutex};
        auto size = n * sizeof(T);
        auto n_elem = (size + sizeof(float4) - 1u) / sizeof(float4);
        if (n_elem > _capacity) {// too big, will not use the arena
            auto buffer = luisa::make_unique<Buffer<T>>(_device.create_buffer<T>(n));
            auto view = buffer->view();
            _buffers.emplace_back(std::move(buffer));
            return view;
        }
        if (!_current_buffer || n_elem > _current_buffer->size()) {
            auto buffer = luisa::make_unique<Buffer<float4>>(
                _device.create_buffer<float4>(_capacity));
            _current_buffer = buffer->view();
            _buffers.emplace_back(std::move(buffer));
        }
        auto view = _current_buffer->subview(0u, n_elem);
        _current_buffer = _current_buffer->subview(
            n_elem, _current_buffer->size() - n_elem);
        return view.template as<T>().subview(0u, n);
    }
};

}// namespace luisa::compute
