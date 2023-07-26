#include <random>
#include <wx/wx.h>

#include <stb/stb_image.h>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/gui/window.h>
#include <luisa/gui/framerate.h>

#if defined(__WXGTK__)
#include <gtk/gtk.h>
#include <gdk/gdkx.h>
#include <glib-object.h>
#endif

using namespace luisa;
using namespace luisa::compute;

class Renderer : public wxWindow {

private:
    Device &_device;
    Stream &_stream;
    luisa::unique_ptr<Swapchain> _swapchain;
    luisa::unique_ptr<Image<float>> _image;
    Framerate _framerate;

public:
    explicit Renderer(wxWindow *parent, Device &device, Stream &stream) noexcept
        : wxWindow{parent, wxID_ANY}, _device{device}, _stream{stream} {}

    void initialize() noexcept {

        auto width = 0;
        auto height = 0;
        auto channels = 0;
        auto pixels = stbi_load("src/tests/logo.png", &width, &height, &channels, 4);
        auto resolution = make_uint2(width, height);

        _image = luisa::make_unique<Image<float>>(
            _device.create_image<float>(PixelStorage::BYTE4, resolution));
        _stream << _image->copy_from(pixels) << synchronize();
        stbi_image_free(pixels);

        auto handle = GetHandle();
        LUISA_ASSERT(handle != nullptr, "Window handle is null.");

#ifdef __WXGTK__
        auto window = gtk_widget_get_window(handle);
        LUISA_ASSERT(window != nullptr, "Window is null.");
        auto window_handle = 0ull;
        if (GDK_IS_X11_WINDOW(window)) {
            window_handle = gdk_x11_window_get_xid(window);
        } else {
            LUISA_ERROR_WITH_LOCATION(
                "Unknown window type: {}",
                G_OBJECT_CLASS_NAME(window));
        }
#else
        auto window_handle = reinterpret_cast<uint64_t>(handle);
#endif

        _swapchain = luisa::make_unique<Swapchain>(
            _device.create_swapchain(
                window_handle, _stream, resolution, false, false, 3));

        SetSize(GetParent()->GetClientSize());
        Center();
    }

    void render(wxIdleEvent &event) noexcept {
        LUISA_INFO("FPS: {}", _framerate.report());
        if (_swapchain == nullptr) { return; }
        _stream << _swapchain->present(*_image);
        _framerate.record(1u);
        event.RequestMore();
    }
};

class Frame : public wxFrame {

public:
    explicit Frame(wxSize size) noexcept
        : wxFrame{nullptr, wxID_ANY, wxT("Display"),
                  wxDefaultPosition, size} {}
    void close(wxCommandEvent &) noexcept { Close(); }
};

class App : public wxApp {

private:
    luisa::unique_ptr<Context> _context;
    luisa::unique_ptr<Device> _device;
    luisa::unique_ptr<Stream> _stream;

public:
    bool OnInit() override {

        if (argc <= 1) {
            LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal",
                       GetAppDisplayName().c_str().AsChar());
            return false;
        }

        _context = luisa::make_unique<Context>(argv[0]);
        _device = luisa::make_unique<Device>(_context->create_device(argv[1].c_str().AsChar()));
        _stream = luisa::make_unique<Stream>(_device->create_stream(StreamTag::GRAPHICS));

        auto frame = new Frame{wxSize{1280, 720}};

        auto renderer = new Renderer{frame, *_device, *_stream};
        // FIXME: initializing in the ctor or on create event
        //  doesn't work on Windows, so doing it here manually.
        renderer->initialize();

        auto overlay = new wxWindow{renderer, wxID_ANY};
        overlay->SetClientSize(renderer->GetClientSize() / 2);
        overlay->SetBackgroundColour(wxColour{128, 64, 96, 128});
        overlay->Center();

        auto button = new wxButton{overlay, wxID_EXIT, wxT("Quit")};
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(Frame::close), frame);
        button->Center();

        Bind(wxEVT_IDLE, wxIdleEventHandler(Renderer::render), renderer);

        frame->Show();

        return true;
    }

    int OnExit() override {
        _stream->synchronize();
        return wxAppBase::OnExit();
    }
};

IMPLEMENT_APP_CONSOLE(App)

