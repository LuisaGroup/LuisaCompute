//
// Created by Mike on 3/31/2023.
//

#include <random>
#include <wx/wx.h>

#include <stb/stb_image.h>

#include <core/clock.h>
#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <gui/window.h>
#include <gui/framerate.h>

#if defined(__WXGTK__)
#include <gtk/gtkx.h>
#include <gdk/gdkx.h>
#endif

using namespace luisa;
using namespace luisa::compute;

class Renderer : public wxWindow {

private:
    Device &_device;
    Stream &_stream;
    luisa::unique_ptr<SwapChain> _swapchain;
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

#ifdef __WXGTK__
        auto window_handle =  gdk_x11_window_get_xid(gtk_widget_get_window(GetHandle()));
#else
        auto window_handle = reinterpret_cast<uint64_t>(GetHandle());
#endif

        _swapchain = luisa::make_unique<SwapChain>(
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
            LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal",
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

        auto panel = new wxPanel{renderer};
        panel->SetClientSize(renderer->GetClientSize() / 2);
        panel->SetBackgroundColour(wxColour{128, 64, 96, 128});
        panel->Center();

        auto button = new wxButton{panel, wxID_EXIT, wxT("Quit")};
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(Frame::close), frame);
        button->Center();

        Bind(wxEVT_IDLE, wxIdleEventHandler(Renderer::render), renderer);

        frame->Show();

        return true;
    }
};

IMPLEMENT_APP_CONSOLE(App)
