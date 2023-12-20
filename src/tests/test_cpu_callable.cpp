#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <random>
#include <algorithm>
#include <iostream>

using namespace luisa;
using namespace luisa::compute;

struct SortArgs {
    uint64_t buffer_ptr;
    uint64_t begin;
    uint64_t end;
};

LUISA_STRUCT(SortArgs, buffer_ptr, begin, end) {};

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    Device device = context.create_device("cpu");
    constexpr size_t batch = 16;
    constexpr size_t count = 1024;
    Stream stream = device.create_stream();
    Buffer<uint> buffer = device.create_buffer<uint>(batch * count);
    Buffer<uint> sorted_locals = device.create_buffer<uint>(batch * count);
    Buffer<uint> sorted_arr_vars = device.create_buffer<uint>(batch * count);
    {
        auto dist = std::uniform_int_distribution<uint>{0, 100};
        std::mt19937 gen{std::random_device{}()};
        std::vector<uint> host_buffer(batch * count);
        for (auto i = 0u; i < batch * count; i++) {
            host_buffer[i] = dist(gen);
        }
        stream << buffer.copy_from(host_buffer.data()) << synchronize();
    }
    Kernel1D sort_kernel = [&]() noexcept {
        auto tid = dispatch_id().x;
        auto buffer_ptr = buffer->device_address();
        Var<SortArgs> args;
        args.buffer_ptr = buffer_ptr;
        args.begin = cast<uint64_t>(tid) * batch;
        args.end = args.begin + batch;
        auto sort = CpuCallable<SortArgs>([](SortArgs &args) {
            auto buffer = reinterpret_cast<uint *>(args.buffer_ptr);
            std::sort(buffer + args.begin, buffer + args.end);
        });
        Local<uint> arr{batch};
        $for (i, batch) {
            arr[i] = batch - i + cast<uint64_t>(tid);
        };
        auto _unused = sort(args);
        args.buffer_ptr = arr.address();
        args.begin = 0;
        args.end = batch;
        _unused = sort(args);
        $for (i, batch) {
            sorted_locals->write(cast<uint64_t>(tid) * batch + i, arr.read(i));
        };
        ArrayVar<uint, batch> arr2;
        $for (i, batch) {
            arr2[i] = batch - i + cast<uint64_t>(tid) + 1000ull;
        };
        args.buffer_ptr = arr2.address();
        _unused = sort(args);
        $for (i, batch) {
            sorted_arr_vars->write(cast<uint64_t>(tid) * batch + i, arr2[i]);
        };
    };
    auto sort = device.compile(sort_kernel);
    stream << sort().dispatch(count) << synchronize();
    std::vector<uint> host_buffer(batch * count);
    stream << buffer.copy_to(host_buffer.data()) << synchronize();
    // print first 3 batches
    for (auto i = 0u; i < 3u; i++) {
        for (auto j = 0u; j < batch; j++) {
            std::cout << host_buffer[i * batch + j] << ' ';
        }
        std::cout << '\n';
    }
    stream << sorted_locals.copy_to(host_buffer.data()) << synchronize();
    for (auto i = 0u; i < 3u; i++) {
        for (auto j = 0u; j < batch; j++) {
            std::cout << host_buffer[i * batch + j] << ' ';
        }
        std::cout << '\n';
    }
    stream << sorted_arr_vars.copy_to(host_buffer.data()) << synchronize();
    for (auto i = 0u; i < 3u; i++) {
        for (auto j = 0u; j < batch; j++) {
            std::cout << host_buffer[i * batch + j] << ' ';
        }
        std::cout << '\n';
    }
}
