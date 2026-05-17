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
#if LUISA_ENABLE_WAYLAND
#include <gdk/gdkwayland.h>
#endif
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

        auto display_handle = static_cast<uint64_t>(0u);
#ifdef __WXGTK__
        auto window = gtk_widget_get_window(GTK_WIDGET(handle));
        LUISA_ASSERT(window != nullptr, "Window is null.");
        auto window_handle = static_cast<uint64_t>(0u);
        if (GDK_IS_X11_WINDOW(window)) {
            window_handle = gdk_x11_window_get_xid(window);
            display_handle = reinterpret_cast<uint64_t>(gdk_x11_get_default_xdisplay());
        }
#if LUISA_ENABLE_WAYLAND
        else if (GDK_IS_WAYLAND_WINDOW(window)) {
            auto surface = gdk_wayland_window_get_wl_surface(window);
            auto display = gdk_wayland_display_get_wl_display(gdk_display_get_default());
            window_handle = reinterpret_cast<uint64_t>(surface);
            display_handle = reinterpret_cast<uint64_t>(display);
        }
#endif
        else {
            LUISA_ERROR_WITH_LOCATION(
                "Unknown window type: {}",
                G_OBJECT_CLASS_NAME(window));
        }
#else
        auto window_handle = reinterpret_cast<uint64_t>(handle);
#endif

        _swapchain = luisa::make_unique<Swapchain>(
            _device.create_swapchain(
                _stream,
                SwapchainOption{
                    .display = display_handle,
                    .window = window_handle,
                    .size = make_uint2(resolution),
                    .wants_hdr = false,
                    .wants_vsync = false,
                    .back_buffer_count = 3,
                }));
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

        wxApp::OnInit();

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
        using command_handler = void (wxEvtHandler::*)(wxCommandEvent &);
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, reinterpret_cast<command_handler>(&Frame::close), frame);
        button->Center();

        using idle_handler = void (wxEvtHandler::*)(wxIdleEvent &);
        Bind(wxEVT_IDLE, reinterpret_cast<idle_handler>(&Renderer::render), renderer);

        frame->Show();

        return true;
    }

    int OnExit() override {
        _stream->synchronize();
        return wxAppBase::OnExit();
    }
};

IMPLEMENT_APP_CONSOLE(App)
