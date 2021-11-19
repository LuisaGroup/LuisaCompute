//
// Created by Mike Smith on 2021/9/17.
//

#include <array>
#include <opencv2/opencv.hpp>

#include <network/render_config.h>
#include <network/render_client.h>

using namespace luisa;
using namespace luisa::compute;

int main() {

    auto client = RenderClient::create("127.0.0.1", 23456u);
    client->set_display_handler([](const RenderConfig &config, size_t frame_count, std::span<float4> pixels) noexcept {
              LUISA_INFO("Received frame (spp = {}).", frame_count);
              cv::Mat image{
                  static_cast<int>(config.resolution().y),
                  static_cast<int>(config.resolution().x),
                  CV_32FC4, pixels.data()};
              cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
              auto mean = std::max(cv::mean(cv::mean(image))[0], 1e-3);
              cv::sqrt(image * (0.24 / mean), image);
              cv::imshow("Display", image);
              cv::waitKey(1);
          })
        .set_config(RenderConfig{
            0u, "scene", make_uint2(1280u, 720u), 0u,
            make_uint2(256u, 256u), 4u, 8u})
        .run();
}
