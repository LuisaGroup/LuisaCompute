//
// Created by Mike Smith on 2021/2/27.
//

#include <numeric>

#include <core/dynamic_module.h>
#include <runtime/device.h>
#include <runtime/context.h>
#include <runtime/buffer.h>
#include <runtime/stream.h>
#include <dsl/buffer_view.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    luisa::log_level_verbose();

    auto runtime_dir = std::filesystem::canonical(argv[0]).parent_path();
    auto working_dir = std::filesystem::current_path();
    Context context{runtime_dir, working_dir};

#ifdef __APPLE__
    auto device = context.create_device("metal");
#else
    auto device = context.create_device("dx");
#endif
    auto buffer = device->create_buffer<float>(16384u);
    std::vector<float> data(16384u);
    std::vector<float> results(16384u);
    std::iota(data.begin(), data.end(), 1.0f);

    auto stream = device->create_stream();
    stream << buffer.upload(data.data())
           << buffer.download(results.data())
           << synchronize();

    LUISA_INFO("Results: {}, {}, {}, {}, ..., {}, {}",
               results[0], results[1], results[2], results[3],
               results[16382], results[16383]);
}
