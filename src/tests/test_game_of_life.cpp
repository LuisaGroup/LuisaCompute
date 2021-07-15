//
// Created by Mike Smith on 2021/6/25.
//

#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tests/stb_image_write.h>

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

    Kernel2D kernel = [](ImageVar<float> prev, ImageVar<float> curr) noexcept {
        Var count = 0u;
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();
        Var state = prev.read(uv).x == 1.0f;
        Var p = make_int2(uv);
        for (auto dy = -1; dy <= 1; dy++) {
            for (auto dx = -1; dx <= 1; dx++) {
                if (dx != 0 || dy != 0) {
                    Var q = p + make_int2(dx, dy) + make_int2(size);
                    Var neighbor = prev.read(make_uint2(q) % size).x;
                    count += neighbor;
                }
            }
        }
        Var c0 = count == 2u;
        Var c1 = count == 3u;
        curr.write(uv, make_float4(ite((state && c0) || c1, 1.0f, 0.0f)));
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

    auto prev = device.create_image<float>(PixelStorage::BYTE1, width, height);
    auto curr = device.create_image<float>(PixelStorage::BYTE1, width, height);
    auto stream = device.create_stream();
    stream << prev.copy_from(host_image.data);
    while (cv::waitKey(1) != 'q') {
        stream << shader(prev, curr).dispatch(width, height)
               << curr.copy_to(host_image.data);
        stream.synchronize();
        std::swap(curr, prev);
        cv::imshow("Display", host_image);
    }
}
