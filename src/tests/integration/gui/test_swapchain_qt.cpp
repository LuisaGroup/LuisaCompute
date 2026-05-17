#include <QApplication>
#include <QPushButton>
#include <QWindow>
#include <QScreen>
#include <QMainWindow>
#include <qpa/qplatformnativeinterface.h>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/syntax.h>
#include <luisa/gui/window.h>
#include <luisa/gui/framerate.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
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
    QMainWindow window;
    window.setAttribute(Qt::WA_NativeWindow);
    window.setFixedSize(width, height);
    window.setWindowTitle("Display");
    window.setAutoFillBackground(true);

    QWindow canvas;
    auto canvas_widget = QWidget::createWindowContainer(&canvas);
    canvas_widget->setAttribute(Qt::WA_NativeWindow);
    window.setCentralWidget(canvas_widget);

    QWidget overlay{&window};
    overlay.setAttribute(Qt::WA_NativeWindow);// Note: this is required for overlay to be displayed above the canvas
    overlay.setFixedSize(window.contentsRect().size() / 2);
    overlay.move(window.contentsRect().center() - overlay.rect().center());
    overlay.setAutoFillBackground(true);

    QPushButton button{"Quit", &overlay};
    button.move(overlay.contentsRect().center() - button.rect().center());
    QObject::connect(&button, &QPushButton::clicked, [&] {
        window.setVisible(false);
    });

    window.show();

    auto native_app = QApplication::platformNativeInterface();
    auto native_display = native_app->nativeResourceForIntegration(QByteArrayLiteral("display"));
    auto native_window = reinterpret_cast<uint64_t>(native_app->nativeResourceForWindow(QByteArrayLiteral("surface"), &canvas));
    if (native_window == 0u) {
        native_window = canvas.winId();
    }

    auto swapchain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = reinterpret_cast<uint64_t>(native_display),
            .window = native_window,
            .size = make_uint2(width, height),
            .wants_hdr = false,
            .wants_vsync = true,
            .back_buffer_count = 2,
        });

    Clock clk;
    Framerate framerate;
    while (window.isVisible()) {
        QApplication::processEvents();
        auto time = static_cast<float>(clk.toc() * 1e-3);
        stream << draw(image, time).dispatch(resolution)
               << swapchain.present(image);
        framerate.record();
        auto title = luisa::format("Display - {:.2f} fps", framerate.report());
        window.setWindowTitle(title.c_str());
    }

    stream << synchronize();
    QApplication::quit();
}
