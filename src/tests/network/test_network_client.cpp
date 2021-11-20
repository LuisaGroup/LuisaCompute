//
// Created by Mike Smith on 2021/9/17.
//

#include <array>

#include <stb/stb_image.h>
#include <opencv2/opencv.hpp>

#include <core/clock.h>
#include <network/render_config.h>
#include <network/render_client.h>

using namespace luisa;
using namespace luisa::compute;

int main() {
    auto client = RenderClient::create("166.111.69.34", 23456u);
    client->set_display_handler([](const RenderConfig &config, size_t frame_count, std::span<const std::byte> data) noexcept {
              LUISA_INFO("Received frame (spp = {}, size = {}).", frame_count, data.size_bytes());
              int width, height, channels;
              Clock clock;
              std::unique_ptr<uint8_t, void (*)(void *)> pixels{
                  stbi_load_from_memory(
                      reinterpret_cast<const uint8_t *>(data.data()), data.size_bytes(),
                      &width, &height, &channels, 3),
                  stbi_image_free};
              LUISA_INFO("Decode: {} ms.", clock.toc());
              cv::Mat image{height, width, CV_8UC3, pixels.get()};
              cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
              cv::imshow("Display", image);
              cv::waitKey(1);
          })
        .set_config(RenderConfig{
            0u, "scene", make_uint2(1280u, 720u), 0u,
            make_uint2(256u, 256u), 32u, 8u})
        .run();
}
