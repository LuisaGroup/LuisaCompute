#include <random>
#include <iostream>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

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

    void push_if(Expr<bool> pred, Expr<T> value) noexcept {
        Shared<uint> index{1};
        $if(thread_x() == 0u) { index.write(0u, 0u); };
        sync_block();
        auto local_index = def(0u);
        $if(pred) { local_index = index.atomic(0).fetch_add(1u); };
        sync_block();
        $if(thread_x() == 0u) {
            auto local_count = index.read(0u);
            auto global_offset = _counter->atomic(0u).fetch_add(local_count);
            index.write(0u, global_offset);
        };
        sync_block();
        $if(pred) {
            auto global_index = index.read(0u) + local_index;
            _buffer->write(global_index, value);
        };
    }

    void push(Expr<T> value) noexcept { push_if(true, value); }

    void reset(CommandList &list) noexcept {
        list << _reset().dispatch(1u);
    }
};

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    static constexpr auto queue_size = 16_M;
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
        auto pred = r < .5f;
        q1.push_if(pred, r);
        q2.push_if(!pred, r);
    });

    auto stream = device.create_stream();
    auto sampler_state_buffer = device.create_buffer<uint>(queue_size);

    luisa::vector<uint> sampler_seeds(queue_size);
    std::generate(sampler_seeds.begin(), sampler_seeds.end(),
                  std::mt19937{std::random_device{}()});

    auto do_test = [&](auto &&shader, auto name_in, auto iterations) noexcept {
        auto name = luisa::string_view{name_in};

        shader.set_name(name);
        stream << sampler_state_buffer.copy_from(sampler_seeds.data())
               << synchronize();

        Clock clk;
        for (auto i = 0u; i < iterations; i++) {
            CommandList list;
            list.reserve(3u, 0u);
            q1.reset(list);
            q2.reset(list);
            list << shader(sampler_state_buffer).dispatch(queue_size);
            stream << list.commit();
        }
        stream << synchronize();
        if (!name.empty()) {
            LUISA_INFO("{}: {} ms", name, clk.toc());
        }
    };

    // warm up
    do_test(test_single, "", 64u);
    do_test(test_double, "", 64u);
    do_test(test_select, "", 64u);

    // test
    do_test(test_single, "single", 1024u);
    do_test(test_double, "double", 1024u);
    do_test(test_select, "select", 1024u);
}

