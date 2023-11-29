#include "GLFW/glfw3.h"

#include <imgui.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }

    auto device = context.create_device(argv[1]);
    static constexpr auto width = 1024u;
    static constexpr auto height = 1024u;
    static constexpr auto resolution = make_uint2(width, height);

    auto draw = device.compile<2>([](ImageFloat image, Float time) noexcept {
        auto p = dispatch_id().xy();
        auto uv = make_float2(p) / make_float2(resolution) * 2.0f - 1.0f;
        auto color = def(make_float4());
        Constant scales{pi, luisa::exp(1.f), luisa::sqrt(2.f)};
        for (auto i = 0u; i < 3u; i++) {
            color[i] = cos(time * scales[i] + uv.y * 11.f +
                           sin(-time * scales[2u - i] + uv.x * 7.f) * 4.f) *
                           .5f +
                       .5f;
        }
        color[3] = 1.0f;
        image.write(p, color);
    });

    auto stream = device.create_stream(StreamTag::GRAPHICS);

    auto demo_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    stream << draw(demo_image, 10.0f).dispatch(resolution);

    bool show_demo_window = true;
    bool show_another_window = false;
    float clear_color[4] = {0.45f, 0.55f, 0.60f, 1.00f};

    Clock clk;
    ImGuiWindow window{device, stream, "Display"};

    // We support multiple ImGui contexts, so if you would like to tweak ImGui settings for
    // a specific window, you can use window.with_context to get a context guard.
    window.with_context([&] {
        ImGui::StyleColorsClassic();
        auto &style = ImGui::GetStyle();
        style.TabRounding = 0.f;
    });

    while (!window.should_close()) {

        window.prepare_frame();

        // draw the background
        // ***** Note: the framebuffer might become invalid, e.g., when the window is minimized, so check before use it! *****
        if (window.framebuffer()) {
            stream << draw(window.framebuffer(), static_cast<float>(clk.toc() * 1e-3))
                          .dispatch(window.framebuffer().size());
        }

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Picture", nullptr,
                         ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoScrollWithMouse);
            {
                static auto scale = 1.;
                if (ImGui::IsWindowHovered()) {
                    auto scroll = ImGui::GetIO().MouseWheel;
                    scale = clamp(scale * pow(2, scroll / 10.), 0.1, 10.);
                }
                auto work_size = ImGui::GetContentRegionMax();
                auto padding = ImGui::GetStyle().WindowPadding;
                auto size = std::max(work_size.x - padding.x, work_size.y - padding.y);

                // You can register an image to ImGuiWindow, and get an id for it to use in ImGui::Image
                // Note: it's safe to register the same image multiple times, the same id will be returned
                auto demo_image_id = window.register_texture(demo_image, Sampler::linear_linear_zero());
                ImGui::Image(reinterpret_cast<ImTextureID>(demo_image_id), ImVec2{size, size},
                             {}, ImVec2(1.f / scale, 1.f / scale));
            }
            ImGui::End();

            ImGui::Begin("Hello, world!");// Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");         // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window);// Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);  // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", clear_color);// Edit 3 floats representing a color

            if (ImGui::Button("Button"))// Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);
            auto &io = ImGui::GetIO();
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window) {
            ImGui::Begin("Another Window", &show_another_window);// Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        window.render_frame();
    }
}
