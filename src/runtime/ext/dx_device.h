#pragma once
#include <runtime/device.h>
namespace luisa::compute {
class DxDevice : public Device::Interface {
public:
    explicit DxDevice(Context ctx) noexcept : Device::Interface{std::move(ctx)} {}
    // Pipeline State Object SerDe
    [[nodiscard]] virtual uint64_t create_psolib(eastl::span<uint64_t> shaders) noexcept = 0;
    virtual void destroy_psolib(uint64_t lib_handle) noexcept = 0;
    virtual bool deser_psolib(uint64_t lib_handle, eastl::span<std::byte const> data) noexcept = 0;
    virtual size_t ser_psolib(uint64_t lib_handle, eastl::vector<std::byte> &result) noexcept = 0;
    [[nodiscard]] virtual uint64_t create_shader(Function kernel, std::string_view meta_options, uint64_t psolib) noexcept = 0;
};
}// namespace luisa::compute