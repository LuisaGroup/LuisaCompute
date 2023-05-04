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
        : wxWindow{parent, wxID_ANY, wxDefaultPosition, parent->GetClientSize()},
          _device{device}, _stream{stream} {

        CentreOnParent();

        auto width = 0;
        auto height = 0;
        auto channels = 0;
        auto pixels = stbi_load("src/tests/logo.png", &width, &height, &channels, 4);
        auto resolution = make_uint2(width, height);

        _image = luisa::make_unique<Image<float>>(
            _device.create_image<float>(PixelStorage::BYTE4, resolution));
        _stream << _image->copy_from(pixels) << synchronize();
        stbi_image_free(pixels);

        _swapchain = luisa::make_unique<SwapChain>(
            _device.create_swapchain(
                reinterpret_cast<uint64_t>(GetHandle()),
                _stream, resolution, false, false, 3));
    }

    void render() noexcept {
        if (_swapchain == nullptr) { return; }
        _stream << _swapchain->present(*_image);
        _framerate.record(1u);
        LUISA_INFO("FPS: {}", _framerate.report());
    }
};

class Frame : public wxFrame {

public:
    Frame() noexcept : wxFrame{nullptr, wxID_ANY, wxT("Display")} { Centre(); }
};

class Panel : public wxPanel {

private:
    std::mt19937 _rng{std::random_device{}()};

public:
    explicit Panel(wxWindow *parent, wxSize size, const wxColour &color) noexcept
        : wxPanel{parent, wxID_ANY, wxDefaultPosition, size} {

        SetOwnBackgroundColour(color);
        CenterOnParent();
    }
};


class App : public wxApp {

private:
    luisa::unique_ptr<Context> _context;
    luisa::unique_ptr<Device> _device;
    luisa::unique_ptr<Stream> _stream;
    Frame *_frame{nullptr};
    Renderer *_renderer{nullptr};

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

        _frame = new Frame;
        _renderer = new Renderer{_frame, *_device, *_stream};

        auto panel = new Panel{_frame, _renderer->GetClientSize() / 2, wxColour{128, 64, 96}};

        auto button = new wxButton{panel, wxID_EXIT, wxT("Quit")};
        Connect(wxID_EXIT, wxEVT_COMMAND_BUTTON_CLICKED,
                wxCommandEventHandler(App::close));
        button->CenterOnParent();
        button->SetFocus();

        _frame->Show(true);

        Connect(wxID_ANY, wxEVT_IDLE, wxIdleEventHandler(App::idle));

        return true;
    }

    void idle(wxIdleEvent &event) noexcept {
        if (_renderer == nullptr) { return; }
        _renderer->render();
        event.RequestMore();
    }

    void close(wxCommandEvent &) noexcept {
        if (_frame == nullptr) { return; }
        _frame->Close();
    }
};

IMPLEMENT_APP(App)
