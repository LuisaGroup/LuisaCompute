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
                _stream, resolution, false, true, 8));

        Connect(wxEVT_IDLE, wxIdleEventHandler(Renderer::idle));
    }

    void idle(wxIdleEvent &) noexcept {
        if (_swapchain == nullptr) { return; }
        _stream << _swapchain->present(*_image);
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

        // draw a line
        Connect(wxEVT_PAINT, wxPaintEventHandler(Panel::paint));
        Connect(wxEVT_IDLE, wxIdleEventHandler(Panel::idle));
    }

    void idle(wxIdleEvent &) noexcept {
        Refresh();
    }

    void paint(wxPaintEvent &) noexcept {
        wxPaintDC dc{this};
        auto rgb_dist = std::uniform_int_distribution<uint8_t>{0, 255};
        auto rand_rgb = wxColour{rgb_dist(_rng), rgb_dist(_rng), rgb_dist(_rng)};
        dc.SetPen(wxPen{rand_rgb});
        auto size = dc.GetSize();
        auto x_dist = std::uniform_int_distribution<int>{0, size.x};
        auto y_dist = std::uniform_int_distribution<int>{0, size.y};
        dc.DrawLine(x_dist(_rng), y_dist(_rng), x_dist(_rng), y_dist(_rng));
    }
};


class App : public wxApp {

private:
    luisa::unique_ptr<Context> _context;
    luisa::unique_ptr<Device> _device;
    luisa::unique_ptr<Stream> _stream;
    Frame *_frame{nullptr};

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
        auto renderer = new Renderer{_frame, *_device, *_stream};

        auto panel = new Panel{_frame, renderer->GetClientSize() / 2, wxColour{128, 64, 96}};

        auto button = new wxButton{panel, wxID_EXIT, wxT("Quit")};
        Connect(wxID_EXIT, wxEVT_COMMAND_BUTTON_CLICKED,
                wxCommandEventHandler(App::close));
        button->CenterOnParent();
        button->SetFocus();

        _frame->Show(true);
        return true;
    }

    void close(wxCommandEvent &) noexcept {
        _frame->Close();
    }
};

IMPLEMENT_APP(App)
