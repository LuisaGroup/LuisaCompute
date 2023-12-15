#include <luisa/runtime/remote/server_interface.h>
namespace luisa::compute {
ServerInterface::ServerInterface(luisa::shared_ptr<DeviceInterface> device_impl) noexcept {}
void ServerInterface::execute(luisa::span<const std::byte> data) noexcept {}
}// namespace luisa::compute