#include <core/clock.h>
#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <dsl/sugar.h>
#include <runtime/rtx/accel.h>
#include <gui/window.h>
#include <backends/dx/dx_custom_cmd.h>
// Make sure FSR2 is under this dir
#include <core/magic_enum.h>
#include <ffx_fsr2.h>
// #include <ffx_fsr2_interface.h>
using namespace luisa;
using namespace luisa::compute;
class FSRCommand : public DXCustomCmd {
public:
    FSRCommand() {}
    StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory4 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept override {
    }
};
void fsr_assert(FfxErrorCode code) {
    if (code != FFX_OK) [[unlikely]] {
#define FSR2_LOG(x) LUISA_ERROR("FSR error code: {}", #x)
        switch (code) {
            case FFX_ERROR_INVALID_POINTER: FSR2_LOG(FFX_ERROR_INVALID_POINTER); break;
            case FFX_ERROR_INVALID_ALIGNMENT: FSR2_LOG(FFX_ERROR_INVALID_ALIGNMENT); break;
            case FFX_ERROR_INVALID_SIZE: FSR2_LOG(FFX_ERROR_INVALID_SIZE); break;
            case FFX_EOF: FSR2_LOG(FFX_EOF); break;
            case FFX_ERROR_INVALID_PATH: FSR2_LOG(FFX_ERROR_INVALID_PATH); break;
            case FFX_ERROR_EOF: FSR2_LOG(FFX_ERROR_EOF); break;
            case FFX_ERROR_MALFORMED_DATA: FSR2_LOG(FFX_ERROR_MALFORMED_DATA); break;
            case FFX_ERROR_OUT_OF_MEMORY: FSR2_LOG(FFX_ERROR_OUT_OF_MEMORY); break;
            case FFX_ERROR_INCOMPLETE_INTERFACE: FSR2_LOG(FFX_ERROR_INCOMPLETE_INTERFACE); break;
            case FFX_ERROR_INVALID_ENUM: FSR2_LOG(FFX_ERROR_INVALID_ENUM); break;
            case FFX_ERROR_INVALID_ARGUMENT: FSR2_LOG(FFX_ERROR_INVALID_ARGUMENT); break;
            case FFX_ERROR_OUT_OF_RANGE: FSR2_LOG(FFX_ERROR_OUT_OF_RANGE); break;
            case FFX_ERROR_NULL_DEVICE: FSR2_LOG(FFX_ERROR_NULL_DEVICE); break;
            case FFX_ERROR_BACKEND_API_ERROR: FSR2_LOG(FFX_ERROR_BACKEND_API_ERROR); break;
            case FFX_ERROR_INSUFFICIENT_MEMORY: FSR2_LOG(FFX_ERROR_INSUFFICIENT_MEMORY); break;
        }
#undef FSR2_LOG
    }
}
void fsr2_message(FfxFsr2MsgType type, const wchar_t *message) {
    luisa::wstring_view ws{message};
    luisa::vector<char> s;
    s.reserve(ws.size());
    for (auto &&i : ws) {
        s.push_back(i);
    }
    s.push_back(0);
    switch (type) {
        case FFX_FSR2_MESSAGE_TYPE_WARNING:
            LUISA_WARNING("FSR: {}", luisa::string_view{s.data(), s.size()});
            break;
        case FFX_FSR2_MESSAGE_TYPE_ERROR:
            LUISA_ERROR("FSR: {}", luisa::string_view{s.data(), s.size()});
            break;
    }
}
int main(int argc, char *argv[]) {
    log_level_info();

    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream();
    constexpr uint32_t width = 1920, height = 1080;
    FfxFsr2Context fsr2_context;
    FfxFsr2ContextDescription fsr2_desc{
        .flags = FFX_FSR2_ENABLE_DEPTH_INVERTED,
        .maxRenderSize = FfxDimensions2D{width, height},
        .displaySize = FfxDimensions2D{width, height},
        .device = device.impl()->native_handle(),
        .fpMessage = fsr2_message};
    fsr_assert(ffxFsr2ContextCreate(&fsr2_context, &fsr2_desc));
}