//
// Created by Mike Smith on 2021/4/6.
//

#include <opencv2/opencv.hpp>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};

#if defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = FakeDevice::create(context);
#endif

    Kernel2D fill_image = [](ImageVar<float> image) noexcept {
        Var<uint2> coord{dispatch_id().x, dispatch_id().y};
        Var rg = Var<float2>{coord} / Var<float2>{launch_size().x, launch_size().y};
        image.write(coord, Var<float4>{rg, 1.0f, 1.0f});
    };
    
    device.compile(fill_image);
    auto device_image = device.create_image<float>(PixelStorage::BYTE4, 1024u, 1024u);
    cv::Mat host_image{1024u, 1024u, CV_8UC4, cv::Scalar::all(0)};

    auto event = device.create_event();
    auto stream = device.create_stream();
    auto copy_stream = device.create_stream();

    stream << fill_image(device_image).launch(1024u, 1024u)
           << event.signal();

    copy_stream << event.wait()
                << device_image.copy_to(host_image.data)
                << event.signal();

    event.synchronize();
    cv::cvtColor(host_image, host_image, cv::COLOR_RGBA2BGR);
    cv::imwrite("result.png", host_image);
}
