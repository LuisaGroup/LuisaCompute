//
// Created by Mike Smith on 2021/9/17.
//

#include <opencv2/opencv.hpp>

#include <core/clock.h>
#include <network/thread_pool.h>
#include <network/binary_buffer.h>
#include <network/render_config.h>
#include <network/render_server.h>

using namespace luisa;
using namespace luisa::compute;

int main() {
    ThreadPool thread_pool;
    auto server = RenderServer::create(12345u, 23456u);
    server->set_encode_handler([&](BinaryBuffer &buffer, const RenderConfig &config, std::span<const float4> accum_buffer) noexcept {
              static constexpr auto pow = [](float3 v, float p) noexcept {
                  return make_float3(
                      std::pow(v.x, p),
                      std::pow(v.y, p),
                      std::pow(v.z, p));
              };
              static constexpr auto aces = [](float3 x) noexcept {
                  static constexpr auto a = 2.51f;
                  static constexpr auto b = 0.03f;
                  static constexpr auto c = 2.43f;
                  static constexpr auto d = 0.59f;
                  static constexpr auto e = 0.14f;
                  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
              };
              static constexpr auto linear_to_srgb = [](float4 v) noexcept {
                  auto x = aces(make_float3(v));
                  x = clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                                   12.92f * x,
                                   x <= 0.00031308f) *
                                255.0f,
                            0.0f, 255.0f);
                  return std::array<uint8_t, 3u>{
                      static_cast<uint8_t>(x.z),
                      static_cast<uint8_t>(x.y),
                      static_cast<uint8_t>(x.x)};
              };
              Clock clock;
              cv::Mat ldr{
                  static_cast<int>(config.resolution().y),
                  static_cast<int>(config.resolution().x),
                  CV_8UC3, cv::Scalar::all(0)};
              auto pixel_count = config.resolution().x * config.resolution().y;
              thread_pool.dispatch_1d(
                  pixel_count,
                  [hdr = accum_buffer.data(), ldr = reinterpret_cast<std::array<uint8_t, 3u> *>(ldr.data)](uint index) noexcept {
                      ldr[index] = linear_to_srgb(hdr[index]);
                  });
              auto t1 = clock.toc();
              static thread_local std::vector<uint8_t> jpeg;
              jpeg.clear();
              jpeg.reserve(config.resolution().x * config.resolution().y * 3u);
              cv::imencode(".jpg", ldr, jpeg, {cv::IMWRITE_JPEG_OPTIMIZE, cv::IMWRITE_JPEG_PROGRESSIVE, cv::IMWRITE_JPEG_QUALITY, 75});
              buffer.write(jpeg.data(), jpeg.size());
              auto t2 = clock.toc();
              LUISA_INFO("Updated sending buffer in {} ms (sRGB: {} ms, JPEG: {} ms).", t2, t1, t2 - t1);
          })
        .run();
}
