//
// Created by Mike on 5/25/2023.
//

#include <luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

class AtomicQueueCounter {

private:
    Buffer<uint> _buffer;

public:
};

template<typename T>
class AtomicQueue {

private:
    Buffer<T> _buffer;
    Buffer<uint> _counter;
    Shader1D<> _reset;

public:
    AtomicQueue(Device &device, size_t capacity) noexcept
        : _buffer{device.create_buffer<T>(capacity)},
          _counter{device.create_buffer<uint>(1u)} {
        _reset = device.compile<1>([this] { _counter->write(0u, 0u); });
    }

    void push(Expr<T> value) noexcept {
        auto index = _counter->atomic(0u).fetch_add(1u);
        _buffer->write(index, value);
    }

    void reset(CommandList &list) noexcept {
        list << _reset().dispatch(1u);
    }
};

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    auto make_sampler = device.compile<1>([&tea](BufferUInt seed_buffer, UInt seed) noexcept {
        auto state = tea(dispatch_x(), seed);
        seed_buffer.write(dispatch_x(), state);
    });

    static constexpr auto queue_size = 4_M;
    AtomicQueue<float> q1{device, queue_size};
    AtomicQueue<float> q2{device, queue_size};

    Callable lcg = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    auto test_single = device.compile<1>([&](BufferUInt seed_buffer) noexcept {
        auto x = dispatch_x();
        auto seed = seed_buffer.read(x);
        auto r = lcg(seed);
        seed_buffer.write(x, seed);
        q1.push(r);
    });

    auto test_double = device.compile<1>([&](BufferUInt seed_buffer) noexcept {
        auto x = dispatch_x();
        auto seed = seed_buffer.read(x);
        auto r = lcg(seed);
        seed_buffer.write(x, seed);
        q1.push(r);
        q2.push(r);
    });

    auto test_select = device.compile<1>([&](BufferUInt seed_buffer) noexcept {
        auto x = dispatch_x();
        auto seed = seed_buffer.read(x);
        auto r = lcg(seed);
        seed_buffer.write(x, seed);
        $if(r < .5f) {
            q1.push(r);
        }
        $else {
            q2.push(r);
        };
    });

    auto sampler_state_buffer = device.create_buffer<uint>(queue_size);
    auto stream = device.create_stream();

    auto do_test = [&](auto &&shader, auto name, auto iterations) noexcept {
        stream << make_sampler(sampler_state_buffer, 0x19980810u).dispatch(queue_size)
               << synchronize();

        Clock clk;
        for (auto i = 0u; i < iterations; i++) {
            CommandList list;
            q1.reset(list);
            q2.reset(list);
            list << shader(sampler_state_buffer).dispatch(queue_size);
            stream << list.commit();
        }
        stream << synchronize();
        if (auto n = luisa::string_view{name}; !n.empty()) {
            LUISA_INFO("{}: {} ms", n, clk.toc());
        }
    };

    // warm up
    do_test(test_single, "", 64u);
    do_test(test_double, "", 64u);
    do_test(test_select, "", 64u);

    // test
    do_test(test_single, "single", 1000u);
    do_test(test_double, "double", 1000u);
    do_test(test_select, "select", 1000u);
}
