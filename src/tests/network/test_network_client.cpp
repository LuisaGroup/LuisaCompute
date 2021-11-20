//
// Created by Mike Smith on 2021/9/17.
//

#include <array>

#include <stb/stb_image.h>
#include <opencv2/opencv.hpp>

#include <network/render_config.h>
#include <network/render_client.h>

using namespace luisa;
using namespace luisa::compute;

int main() {
    auto client = RenderClient::create("127.0.0.1", 23456u);
    client->set_display_handler([](const RenderConfig &config, size_t frame_count, std::span<const std::byte> data) noexcept {
              LUISA_INFO("Frame size: {} bytes.", data.size_bytes());
              cv::_InputArray array{
                  reinterpret_cast<const uint8_t *>(data.data()),
                  static_cast<int>(data.size())};
              auto image = cv::imdecode(array, cv::IMREAD_COLOR);
              cv::imshow("Display", image);
              cv::waitKey(1);
          })
        .set_config(RenderConfig{
            0u, "scene", make_uint2(1280u, 720u), 0u,
            make_uint2(256u, 256u), 4u, 8u})
        .run();
}
