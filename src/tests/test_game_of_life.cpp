//
// Created by Mike Smith on 2021/6/25.
//

#include <iostream>
#include <random>

#include <opencv2/opencv.hpp>
#include <stb/stb_image_write.h>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>

using namespace luisa;
using namespace luisa::compute;

struct ImagePair {
    Image<float> prev;
    Image<float> curr;
    ImagePair(Device &device, PixelStorage storage, uint width, uint height) noexcept
        : prev{device.create_image<float>(storage, width, height)},
          curr{device.create_image<float>(storage, width, height)} {}
    void swap() noexcept { std::swap(prev, curr); }
};

LUISA_BINDING_GROUP(ImagePair, prev, curr)

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};

#if defined(LUISA_BACKEND_CUDA_ENABLED)
    auto device = context.create_device("cuda");
#elif defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = FakeDevice::create(context);
#endif

    Callable read_state = [](Var<ImagePair> pair, UInt2 uv) noexcept {
        return pair.prev.read(uv).x == 1.0f;
    };

    Kernel2D kernel = [&](Var<ImagePair> pair) noexcept {
        Var count = 0u;
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();
        Var state = read_state(pair, uv);
        Var p = make_int2(uv);
        for (auto dy = -1; dy <= 1; dy++) {
            for (auto dx = -1; dx <= 1; dx++) {
                if (dx != 0 || dy != 0) {
                    Var q = p + make_int2(dx, dy) + make_int2(size);
                    Var neighbor = pair.prev.read(make_uint2(q) % size).x;
                    count += neighbor;
                }
            }
        }
        Var c0 = count == 2u;
        Var c1 = count == 3u;
        pair.curr.write(uv, make_float4(ite((state & c0) | c1, 1.0f, 0.0f)));
    };
    auto shader = device.compile(kernel);

//#define GAME_OF_LIFE_RANDOMIZE
#ifdef GAME_OF_LIFE_RANDOMIZE
    static constexpr auto width = 512u;
    static constexpr auto height = 512u;
    cv::Mat host_image{height, width, CV_8U, cv::Scalar::all(0)};
    std::default_random_engine random{std::random_device{}()};
    for (auto i = 0u; i < width * height; i++) {
        host_image.data[i] = (random() & 1u) * 255u;
    }
#else
    auto host_image = cv::imread("src/tests/logo.png", cv::IMREAD_REDUCED_GRAYSCALE_2);
    static constexpr auto threshold = 100u;
    for (auto i = 0; i < host_image.rows * host_image.cols; i++) {
        auto &&p = host_image.data[i];
        p = p < threshold ? 0u : 255u;
    }
    auto bbox = cv::boundingRect(host_image);
    host_image(bbox).copyTo(host_image);
    auto width = static_cast<uint>(host_image.cols);
    auto height = static_cast<uint>(host_image.rows);
#endif
    cv::imshow("Display", host_image);
    cv::waitKey();

    ImagePair image_pair{device, PixelStorage::BYTE1, width, height};
    auto stream = device.create_stream();
    stream << image_pair.prev.copy_from(host_image.data);
    while (cv::waitKey(1) != 'q') {
        stream << shader(image_pair).dispatch(width, height)
               << image_pair.curr.copy_to(host_image.data)
               << synchronize();
        image_pair.swap();
        cv::imshow("Display", host_image);
    }
}
