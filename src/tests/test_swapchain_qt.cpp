//
// Created by Mike on 3/31/2023.
//

#include <QApplication>
#include <QPushButton>
#include <QFrame>
#include <QDialog>

#include <core/clock.h>
#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <gui/window.h>
#include <gui/framerate.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    static constexpr auto width = 1280u;
    static constexpr auto height = 720u;
    static constexpr auto resolution = make_uint2(width, height);

    auto draw = device.compile<2>([](ImageFloat image, Float time) noexcept {
        auto p = dispatch_id().xy();
        auto uv = make_float2(p) / make_float2(resolution) * 2.0f - 1.0f;
        auto color = def(make_float4());
        Constant<float> scales{pi, luisa::exp(1.f), luisa::sqrt(2.f)};
        for (auto i = 0u; i < 3u; i++) {
            color[i] = cos(time * scales[i] + uv.y * 11.f +
                           sin(-time * scales[2u - i] + uv.x * 7.f) * 4.f) *
                           .5f +
                       .5f;
        }
        color[3] = 1.0f;
        image.write(p, color);
    });

    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    auto image = device.create_image<float>(PixelStorage::BYTE4, resolution);

    QApplication app{argc, argv};
    QDialog dialog;
    dialog.setFixedSize(width, height);
    dialog.setWindowTitle("Display");

    QWidget canvas{&dialog};
    canvas.setFixedSize(dialog.contentsRect().size());
    canvas.move(dialog.contentsRect().topLeft());

    QWidget overlay{&dialog};
    overlay.setStyleSheet("background-color: rgba(64, 96, 128);");
    overlay.setFixedSize(dialog.contentsRect().size() / 2);
    overlay.move(dialog.contentsRect().center() - overlay.rect().center());

    QPushButton button{"Quit", &overlay};
    button.move(overlay.contentsRect().center() - button.rect().center());
    QObject::connect(&button, &QPushButton::clicked, [&] {
        dialog.setVisible(false);
    });

    auto swapchain = device.create_swapchain(
        canvas.winId(), stream,
        resolution, false, true, 3);

    dialog.show();

    Clock clk;
    Framerate framerate;
    while (dialog.isVisible()) {
        QApplication::processEvents();
        auto time = static_cast<float>(clk.toc() * 1e-3);
        stream << draw(image, time).dispatch(resolution)
               << swapchain.present(image);
        framerate.record();
        LUISA_INFO("FPS: {}", framerate.report());
    }

    stream << synchronize();
    QApplication::quit();
}
