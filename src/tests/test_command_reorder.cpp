//
// Created by ChenXin on 2021/12/9.
//

#include <vector>

#include <runtime/context.h>
#include <gui/window.h>
#include <gui/framerate.h>
#include <runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};

#if defined(LUISA_BACKEND_CUDA_ENABLED)
    auto device = context.create_device("cuda");
#elif defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#else
    auto device = context.create_device("ispc");
#endif

    int width = 1920, height = 1080;
    std::vector<std::array<uint8_t, 4u>> host_image(width * height);

    Window window{"Display", make_uint2(width, height)};
    window.set_key_callback([&](int key, int action) noexcept {
        if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
            window.set_should_close();
        }
    });

    Clock clock;
    Framerate framerate{32};
    window.run([&] {
        framerate.record();
        auto time = static_cast<float>(clock.toc() * 1e-3);
        stream << shader(device_image, time).dispatch(width, height)
               << device_image.copy_to(host_image.data())
               << synchronize();
        window.set_background(host_image.data(), make_uint2(width, height));

        ImGui::Begin("Console", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("FPS: %.1f", framerate.report());
        ImGui::End();
    });

    return 0;
}